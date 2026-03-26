package scene

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"

	"github.com/Carmen-Shannon/automation/tools/worker"
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/cogentcore/webgpu/wgpu"
)

// boneParticleUpdateGroup tracks the GPU resources needed to transform a kinematic
// body's particles through the skeletal animator's bone matrices each frame. Each
// kinematic-animated body gets its own group with a BGP that binds the shared
// particle/body buffers from physics and the scratch_matrices buffer from the
// owning Animator.
type boneParticleUpdateGroup struct {
	bgp           bind_group_provider.BindGroupProvider
	particleStart uint32
	particleCount uint32
	boneCount     uint32
	instanceIndex uint32
}

type scene struct {
	mu *sync.RWMutex

	name   string
	active bool

	animatorPool map[model.Model][]animator.Animator
	registry     map[uint64]game_object.GameObject // non-ephemeral objects by ID
	nextID       uint64

	physicsHandler  physics.Physics
	physicsGPUReady bool // true once initPhysicsGPU has run

	// Per-animator sync dispatch state. Each unique Animator that has physics
	// bodies gets its own group with a dedicated sync_map buffer and AnimationData
	// reference. This allows the sync shader to write each body's transform to the
	// correct Animator's AnimationData slot without cross-group interference
	// (bodies not belonging to a group are masked by a sentinel).
	physicsSyncGroup   map[int]bind_group_provider.BindGroupProvider
	physicsSyncAnimMap map[animator.Animator]int         // animator -> group ID in physicsSyncGroup
	physicsSyncWrites  []bind_group_provider.BufferWrite // staged per-group sync_map writes
	physicsAnimBinding int                               // cached AnimationData binding in compute shader (-1 = not resolved)

	// Per-kinematic-body bone particle update state. Each kinematic body with
	// skeletal animation gets a boneParticleUpdateGroup that dispatches the
	// bone_particle_update shader after animator compute to transform particles
	// through the current bone matrices.
	boneParticleUpdateGroups []*boneParticleUpdateGroup
	boneUpdateGPUReady       bool // true once the bone_update pipeline is registered

	cam camera.Camera
	r   renderer.Renderer

	screenWidth  int
	screenHeight int

	cullingDisabled bool // when true, skips frustum plane distribution to animators

	// Lighting subsystem — manages lights, shadow mapping, and Forward+ culling state.
	lightHandler light.LightingHandler
	lightObjects []game_object.GameObject // objects with attached lights (ephemeral and non-ephemeral)

	// Per-frame spot/point shadow state, populated by PrepareShadows.
	lightShadowEntries []light.GPULightShadowEntry // rebuilt each frame
	lightShadowMap     map[light.Light]uint32      // light → entry index

	tileBufferCapacity int // number of tiles the tile GPU buffers were sized for

	// Pre-allocated slices reused each frame to avoid per-frame allocations.
	writePool              []bind_group_provider.BufferWrite             // reusable coalesced buffer write slice
	drawBindGroupsPool     []bind_group_provider.BindGroupProvider       // reusable bind group slice for DrawCalls
	drawDeclsPool          []shader.Annotation                           // reusable annotations slice for DrawCalls
	drawGroupProvidersPool map[int]bind_group_provider.BindGroupProvider // reusable group-providers map for DrawCalls

	// computePool manages a bounded set of reusable goroutines for the parallel
	// CPU prep phase of PrepareCompute. Workers persist across frames, avoiding
	// per-frame goroutine spawn/teardown overhead.
	computePool    worker.DynamicWorkerPool
	computeWorkers int // stored so we can log/inspect the configured count
	maxBonesGPU    uint64

	// instanceLookup provides O(1) reverse lookup from (Animator, instanceSlot) → objID.
	// Maintained by Add/Remove so the swap-remove fixup in Remove avoids an O(N) registry scan.
	instanceLookup map[animator.Animator]map[uint32]uint64

	injections map[string]string

	postProcessingInitialized bool
}

// shadowPipelineKey resolves the shadow depth pipeline key for the given model
// type and cull mode from the lighting handler's pipeline key map.
func (s *scene) shadowPipelineKey(skinned bool, mode model.ShadowCullMode) string {
	prefix := "shadow_static_"
	if skinned && s.lightHandler.ShadowHandler().PipelineKey("shadow_skinned_back") != "" {
		prefix = "shadow_skinned_"
	}
	tag := "back"
	switch mode {
	case model.ShadowCullModeFront:
		tag = "front"
	case model.ShadowCullModeNone:
		tag = "none"
	}
	return s.lightHandler.ShadowHandler().PipelineKey(prefix + tag)
}

// buildInjectionMap builds the injection map for WGSL shader pre-processing,
// including dynamic values from the light handler.
func (s *scene) buildInjectionMap() {
	m := map[string]string{
		"max_bones":              fmt.Sprintf("%du", s.maxBonesGPU),
		"empty_sentinel":         fmt.Sprintf("0x%Xu", uint32(0xFFFFFFFF)),
		"flag_active":            fmt.Sprintf("%du", physics.PhysicsStateActive),
		"flag_static":            fmt.Sprintf("%du", physics.PhysicsStateStatic),
		"flag_kinematic":         fmt.Sprintf("%du", physics.PhysicsStateKinematic),
		"light_type_directional": fmt.Sprintf("%du", light.LightTypeDirectional),
		"light_type_point":       fmt.Sprintf("%du", light.LightTypePoint),
		"light_type_spot":        fmt.Sprintf("%du", light.LightTypeSpot),
	}
	if s.lightHandler != nil {
		ts := s.lightHandler.TileSize()
		m["tile_size"] = fmt.Sprintf("%du", ts)
		m["luminance_workgroup_size"] = fmt.Sprintf("%du", s.lightHandler.CompositionHandler().LuminanceWorkgroupSize())
		m["max_lights_per_tile"] = fmt.Sprintf("%du", s.lightHandler.MaxLightsPerTile())
		m["num_threads"] = fmt.Sprintf("%du", ts*ts)
		m["max_ssao_samples"] = fmt.Sprintf("%du", s.lightHandler.SSAOHandler().MaxSamples())
		m["pcf_samples"] = fmt.Sprintf("%du", s.lightHandler.ShadowHandler().PCFSamples())
	}
	if s.physicsHandler != nil {
		m["slots_per_cell"] = fmt.Sprintf("%du", s.physicsHandler.SlotsPerCell())
		m["body_idx_mask"] = fmt.Sprintf("0x%Xu", s.physicsHandler.BodyIdxMask())
	}
	s.injections = m
}

// generateSSAOKernel generates a hemisphere sample kernel of the given size
// (clamped to [1, 32]) as a flat byte buffer of array<vec4<f32>, 32> (512 bytes).
// Samples are distributed in a unit hemisphere with an accelerating distribution
// that biases samples closer to the origin.
func (s *scene) generateSSAOKernel(sampleCount int) []byte {
	if sampleCount < 1 {
		sampleCount = 1
	}
	if sampleCount > s.lightHandler.SSAOHandler().MaxSamples() {
		sampleCount = s.lightHandler.SSAOHandler().MaxSamples()
	}

	buf := make([]byte, s.lightHandler.SSAOHandler().MaxSamples()*16) // MaxSamples × vec4<f32>
	off := 0
	for i := 0; i < s.lightHandler.SSAOHandler().MaxSamples(); i++ {
		var x, y, z float32
		if i < sampleCount {
			// Random point in a hemisphere (z >= 0).
			x = rand.Float32()*2.0 - 1.0
			y = rand.Float32()*2.0 - 1.0
			z = rand.Float32() // [0, 1] — hemisphere
			length := float32(math.Sqrt(float64(x*x + y*y + z*z)))
			if length > 0.0001 {
				x /= length
				y /= length
				z /= length
			}
			// Accelerating distribution: samples near the center of the hemisphere
			// are denser, tapering off toward the edges.
			scale := float32(i) / float32(sampleCount)
			scale = 0.1 + scale*scale*0.9
			x *= scale
			y *= scale
			z *= scale
		}
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(x))
		binary.LittleEndian.PutUint32(buf[off+4:off+8], math.Float32bits(y))
		binary.LittleEndian.PutUint32(buf[off+8:off+12], math.Float32bits(z))
		binary.LittleEndian.PutUint32(buf[off+12:off+16], 0) // w = 0
		off += 16
	}
	return buf
}

// initSSAO initializes the SSAO subsystem: creates screen-sized occlusion textures,
// registers the SSAO compute and bilateral blur pipelines, and pre-creates all bind
// group providers with correctly-sized GPU buffers. The G-Buffer must be initialized
// before calling this method.
func (s *scene) initSSAO() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.lightHandler.SSAOHandler() == nil || s.lightHandler.GBufferHandler() == nil {
		return
	}
	if !s.lightHandler.GBufferHandler().Enabled() {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	// When half-resolution is enabled, allocate SSAO textures at half the
	// screen dimensions in each axis (quarter pixel count).
	ssaoW := w
	ssaoH := h
	if s.lightHandler.SSAOHandler().HalfResolution() {
		ssaoW = max(w/2, 1)
		ssaoH = max(h/2, 1)
	}

	// 1. Create SSAO textures (raw, blurred, scratch at SSAO res).
	rawView, rawTex, blurView, blurTex, scratchView, scratchTex, err := s.r.CreateSSAOTextures(ssaoW, ssaoH)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create SSAO textures: %v", err))
	}
	s.lightHandler.SSAOHandler().SetRawTexture(rawTex)
	s.lightHandler.SSAOHandler().SetRawTextureView(rawView)
	s.lightHandler.SSAOHandler().SetBlurredTexture(blurTex)
	s.lightHandler.SSAOHandler().SetBlurredTextureView(blurView)
	s.lightHandler.SSAOHandler().SetScratchTexture(scratchTex)
	s.lightHandler.SSAOHandler().SetScratchTextureView(scratchView)

	// 2. Create or reuse linear sampler for the blurred SSAO texture in the lit shader.
	linearSamp, err := s.r.CreateLinearSampler()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create SSAO linear sampler: %v", err))
	}
	s.lightHandler.SSAOHandler().SetLinearSampler(linearSamp)

	// 3. Register SSAO compute pipeline.
	ssaoCompShader := shader.NewShader("_ssao_compute", shader.ShaderTypeCompute, "engine/light/assets/ssao-compute.wgsl", shader.WithInjections(s.injections))
	ssaoCompKey := "ssao_compute"
	ssaoCompPipe := pipeline.NewPipeline(ssaoCompKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(ssaoCompShader),
	)
	if err := s.r.RegisterPipelines(ssaoCompPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register SSAO compute pipeline: %v", err))
	}
	s.lightHandler.SSAOHandler().SetPipelineKey("ssao_compute", ssaoCompKey)

	// 4. Register bilateral blur compute pipeline.
	blurCompShader := shader.NewShader("_ssao_blur_compute", shader.ShaderTypeCompute, "engine/light/assets/ssao-blur-compute.wgsl", shader.WithInjections(s.injections))
	blurCompKey := "ssao_blur_compute"
	blurCompPipe := pipeline.NewPipeline(blurCompKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(blurCompShader),
	)
	if err := s.r.RegisterPipelines(blurCompPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register SSAO blur compute pipeline: %v", err))
	}
	s.lightHandler.SSAOHandler().SetPipelineKey("ssao_blur", blurCompKey)

	// 5. Create SSAO compute bind group provider.
	ssaoDesc := ssaoCompShader.BindGroupLayoutDescriptor(0)
	ssaoSizeOverrides := map[int]uint64{
		4: uint64((&light.GPUSSAOParams{}).Size()), // ssao_params uniform
		5: 32 * 16,                                 // ssao_kernel: array<vec4<f32>, 32> = 512 bytes
	}
	ssaoBGP := s.lightHandler.SSAOHandler().Bgp("ssao_compute")
	ssaoBGP.SetTextureView(0, s.lightHandler.GBufferHandler().DepthTextureView())
	ssaoBGP.SetTextureView(1, s.lightHandler.GBufferHandler().NormalTextureView())
	ssaoBGP.SetTextureView(3, rawView)
	if err := s.r.InitBindGroup(ssaoBGP, ssaoDesc, nil, ssaoSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSAO compute bind group: %v", err))
	}

	// 6. Create blur bind group providers (bilateral blur, depth-aware).
	blurDesc := blurCompShader.BindGroupLayoutDescriptor(0)
	blurSizeOverrides := map[int]uint64{
		2: uint64((&light.GPUBlurParams{}).Size()),
	}

	// Horizontal: raw → scratch, depth from G-Buffer hardware depth texture.
	blurHBGP := s.lightHandler.SSAOHandler().Bgp("ssao_blur_h")
	blurHBGP.SetTextureView(0, rawView)
	blurHBGP.SetTextureView(1, scratchView)
	blurHBGP.SetTextureView(3, s.lightHandler.GBufferHandler().DepthTextureView())
	if err := s.r.InitBindGroup(blurHBGP, blurDesc, nil, blurSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSAO blur horizontal bind group: %v", err))
	}

	// Vertical: scratch → blurred, depth from G-Buffer hardware depth texture.
	blurVBGP := s.lightHandler.SSAOHandler().Bgp("ssao_blur_v")
	blurVBGP.SetTextureView(0, scratchView)
	blurVBGP.SetTextureView(1, blurView)
	blurVBGP.SetTextureView(3, s.lightHandler.GBufferHandler().DepthTextureView())
	if err := s.r.InitBindGroup(blurVBGP, blurDesc, nil, blurSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSAO blur vertical bind group: %v", err))
	}

	// 7. Generate hemisphere sample kernel and write to the SSAO compute BGP buffer.
	kernelData := s.generateSSAOKernel(s.lightHandler.SSAOHandler().SampleCount())
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: ssaoBGP, Binding: 5, Offset: 0, Data: kernelData},
	})

	s.lightHandler.SSAOHandler().Resize(ssaoW, ssaoH)
	s.lightHandler.SSAOHandler().SetEnabled(true)
}

// initSSAOLitBindGroup creates the SSAO bind group provider used by the lit
// fragment shader at @group(6). When the SSAO subsystem is enabled, the real
// blurred occlusion texture and linear sampler are bound. When SSAO is disabled
// or absent, a 1×1 white fallback texture is created so the shader reads ao=1.0
// (no darkening), keeping the bind group layout valid without any conditional
// branching in the shader.
func (s *scene) initSSAOLitBindGroup(litFragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if litFragmentShader == nil {
		return
	}

	// Resolve the SSAO bind group index from the lit fragment shader's annotations.
	ssaoGroup := -1
	for _, decl := range litFragmentShader.Declarations() {
		if decl.Type == shader.AnnotationTypeProvider && decl.Group != nil && decl.Args[0] == shader.AnnotationArgSSAO {
			ssaoGroup = *decl.Group
			break
		}
	}
	if ssaoGroup < 0 {
		return
	}

	bgp := s.lightHandler.Bgp("ssao_lit")
	desc := litFragmentShader.BindGroupLayoutDescriptor(ssaoGroup)

	// Determine whether SSAO is enabled and the blurred texture is available.
	ssaoReady := s.lightHandler.SSAOHandler().Enabled() &&
		s.lightHandler.SSAOHandler().BlurredTextureView() != nil && s.lightHandler.SSAOHandler().LinearSampler() != nil

	if ssaoReady {
		// Bind the real blurred SSAO texture and linear sampler.
		for _, entry := range desc.Entries {
			binding := int(entry.Binding)
			if entry.Texture.SampleType != wgpu.TextureSampleTypeUndefined {
				bgp.SetTextureView(binding, s.lightHandler.SSAOHandler().BlurredTextureView())
			}
			if entry.Sampler.Type != wgpu.SamplerBindingTypeUndefined {
				bgp.SetSampler(binding, s.lightHandler.SSAOHandler().LinearSampler())
			}
		}
	} else {
		// Create a 1×1 white fallback texture (ao=1.0, no darkening).
		for _, entry := range desc.Entries {
			binding := int(entry.Binding)
			if entry.Texture.SampleType != wgpu.TextureSampleTypeUndefined {
				fallback := common.TextureStagingData{
					Pixels: []byte{255, 255, 255, 255},
					Width:  1,
					Height: 1,
					Linear: true,
				}
				if err := s.r.InitTextureView(bgp, binding, fallback); err != nil {
					panic(fmt.Sprintf("scene: failed to init SSAO fallback texture: %v", err))
				}
			}
			if entry.Sampler.Type != wgpu.SamplerBindingTypeUndefined {
				fallbackSampler := common.SamplerStagingData{
					AddressModeU:  wgpu.AddressModeClampToEdge,
					AddressModeV:  wgpu.AddressModeClampToEdge,
					AddressModeW:  wgpu.AddressModeClampToEdge,
					MagFilter:     wgpu.FilterModeLinear,
					MinFilter:     wgpu.FilterModeLinear,
					MipmapFilter:  wgpu.MipmapFilterModeLinear,
					LodMinClamp:   0,
					LodMaxClamp:   1,
					MaxAnisotropy: 1,
				}
				if err := s.r.InitSampler(bgp, binding, fallbackSampler); err != nil {
					panic(fmt.Sprintf("scene: failed to init SSAO fallback sampler: %v", err))
				}
			}
		}
	}

	if err := s.r.InitBindGroup(bgp, desc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSAO lit bind group: %v", err))
	}
}

// initGBuffer initializes the G-Buffer MRT textures and registers the static
// and skinned G-Buffer render pipelines. The G-Buffer pre-pass writes per-pixel
// position, normal, and albedo data into screen-sized textures consumed by
// downstream screen-space effects (SSAO, SSR).
func (s *scene) initGBuffer() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.lightHandler.GBufferHandler() == nil {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	// Create G-Buffer MRT textures (normal + albedo + depth; position is
	// reconstructed from depth at read time by compute shaders).
	normView, normTex, albView, albTex, depthView, depthTex, err := s.r.CreateGBufferTextures(w, h)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create G-Buffer textures: %v", err))
	}
	s.lightHandler.GBufferHandler().SetNormalTexture(normTex)
	s.lightHandler.GBufferHandler().SetNormalTextureView(normView)
	s.lightHandler.GBufferHandler().SetAlbedoTexture(albTex)
	s.lightHandler.GBufferHandler().SetAlbedoTextureView(albView)
	s.lightHandler.GBufferHandler().SetDepthTexture(depthTex)
	s.lightHandler.GBufferHandler().SetDepthTextureView(depthView)

	// Load shaders for the G-Buffer pass.
	gbufferFrag := shader.NewShader("_gbuffer_frag", shader.ShaderTypeFragment, "engine/light/assets/gbuffer-frag.wgsl", shader.WithInjections(s.injections))
	staticVert := shader.NewShader("_gbuffer_static_vert", shader.ShaderTypeVertex, "engine/light/assets/lit-vert.wgsl", shader.WithInjections(s.injections))
	skinnedVert := shader.NewShader("_gbuffer_skinned_vert", shader.ShaderTypeVertex, "engine/light/assets/lit-skinned-vert.wgsl", shader.WithInjections(s.injections))

	// Register static G-Buffer pipeline.
	staticKey := "gbuffer_static"
	staticPipe := pipeline.NewPipeline(staticKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(staticVert),
		pipeline.WithFragmentShader(gbufferFrag),
	)
	if err := s.r.RegisterGBufferPipeline(staticPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register static G-Buffer pipeline: %v", err))
	}
	s.lightHandler.GBufferHandler().SetPipelineKey("static", staticKey)

	// Register skinned G-Buffer pipeline.
	skinnedKey := "gbuffer_skinned"
	skinnedPipe := pipeline.NewPipeline(skinnedKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(skinnedVert),
		pipeline.WithFragmentShader(gbufferFrag),
	)
	if err := s.r.RegisterGBufferPipeline(skinnedPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register skinned G-Buffer pipeline: %v", err))
	}
	s.lightHandler.GBufferHandler().SetPipelineKey("skinned", skinnedKey)

	s.lightHandler.GBufferHandler().Resize(w, h)
	s.lightHandler.GBufferHandler().SetEnabled(true)
}

// initContactShadows initializes the contact shadow subsystem: creates a
// screen-sized R32Float output texture, registers the contact shadow compute
// pipeline, and pre-creates the bind group provider with correctly-sized GPU
// buffers. The G-Buffer must be initialized before calling this method.
func (s *scene) initContactShadows() {
	s.mu.Lock()
	defer s.mu.Unlock()

	csHandler := s.lightHandler.ContactShadowHandler()
	if csHandler == nil || s.lightHandler.GBufferHandler() == nil {
		return
	}
	if !s.lightHandler.GBufferHandler().Enabled() {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	// 1. Create contact shadow output texture (full screen resolution).
	csView, csTex, err := s.r.CreateContactShadowTextures(w, h)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create contact shadow textures: %v", err))
	}
	csHandler.SetTexture(csTex)
	csHandler.SetTextureView(csView)

	// 2. Create linear sampler for the lit shader to sample the contact shadow texture.
	csLinearSamp, err := s.r.CreateLinearSampler()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create contact shadow linear sampler: %v", err))
	}
	csHandler.SetLinearSampler(csLinearSamp)

	// 3. Register contact shadow compute pipeline.
	csCompShader := shader.NewShader("_contact_shadow_compute", shader.ShaderTypeCompute, "engine/light/assets/contact-shadow-compute.wgsl", shader.WithInjections(s.injections))
	csCompKey := "contact_shadow_compute"
	csCompPipe := pipeline.NewPipeline(csCompKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(csCompShader),
	)
	if err := s.r.RegisterPipelines(csCompPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register contact shadow compute pipeline: %v", err))
	}
	csHandler.SetPipelineKey("contact_shadow_compute", csCompKey)

	// 4. Create contact shadow compute bind group provider.
	csDesc := csCompShader.BindGroupLayoutDescriptor(0)
	csSizeOverrides := map[int]uint64{
		2: uint64((&light.GPUContactShadowParams{}).Size()),
	}
	csBGP := csHandler.Bgp("contact_shadow_compute")
	csBGP.SetTextureView(0, s.lightHandler.GBufferHandler().DepthTextureView())
	csBGP.SetTextureView(1, csView)
	if err := s.r.InitBindGroup(csBGP, csDesc, nil, csSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init contact shadow compute bind group: %v", err))
	}

	csHandler.SetEnabled(true)
}

// initLightBindGroup initializes the GPU resources for the light storage buffer
// using the layout descriptor from the given fragment shader's light group.
func (s *scene) initLightBindGroup(fragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if fragmentShader == nil {
		return
	}

	// Resolve the lights bind group index from the shader's pre-processor
	// declarations by matching the LightHeader struct type annotation.
	lightGroup := -1
	for _, decl := range fragmentShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgLightHeader {
			lightGroup = *decl.Group
			break
		}
	}
	if lightGroup < 0 {
		return
	}

	bgp := s.lightHandler.Bgp("lights")

	// Build buffer size overrides: the light storage buffer (binding 1) must hold
	// MaxGPULights entries so it can accommodate dynamic light counts each frame.
	descriptor := fragmentShader.BindGroupLayoutDescriptor(lightGroup)
	sizeOverrides := make(map[int]uint64)
	for _, entry := range descriptor.Entries {
		binding := int(entry.Binding)
		if entry.Buffer.Type == wgpu.BufferBindingTypeReadOnlyStorage || entry.Buffer.Type == wgpu.BufferBindingTypeStorage {
			// Storage buffer: size it for max lights (header is in a separate uniform binding).
			sizeOverrides[binding] = uint64(s.lightHandler.MaxGPULights()) * uint64((&light.GPULight{}).Size())
		}
	}

	if err := s.r.InitBindGroup(bgp, descriptor, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init light bind group: %v", err))
	}
}

// initShadowMap initializes the CSM atlas depth texture, per-cascade bind group
// providers, and shadow depth render pipelines for PCF shadow mapping.
func (s *scene) initShadowMap(shadowVertShader, shadowSkinnedVertShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if shadowVertShader == nil {
		return
	}

	sh := s.lightHandler.ShadowHandler()
	res := sh.ShadowMapResolution()
	cascadeCount := sh.CascadeCount()
	atlasW := cascadeCount * res

	maxDim := int(s.r.MaxTextureDimension2D())
	if maxDim > 0 && atlasW > maxDim {
		panic(fmt.Sprintf(
			"scene: CSM atlas width %d (%d cascades × %d resolution) exceeds device MaxTextureDimension2D (%d). "+
				"Reduce WithShadowMapResolution.",
			atlasW, cascadeCount, res, maxDim,
		))
	}

	// Create CSM atlas depth texture: (atlasW × res).
	depthView, depthTex, err := s.r.CreateShadowDepthTexture(atlasW, res)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create CSM shadow depth texture: %v", err))
	}
	sh.SetCSMAtlasTexture(depthTex)
	sh.SetCSMAtlasTextureView(depthView)

	// Create comparison sampler for PCF shadow sampling in the lit fragment shader.
	compSampler, err := s.r.CreateComparisonSampler()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create CSM comparison sampler: %v", err))
	}
	sh.SetComparisonSampler(compSampler)

	// Create one "csm_data_N" BGP per cascade — each holds a GPUShadowUniform buffer
	// for the per-cascade shadow depth pass vertex shader (group 0, binding 0).
	shadowGroup := 0
	for _, decl := range shadowVertShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgShadowUniform {
			shadowGroup = *decl.Group
			break
		}
	}
	desc := shadowVertShader.BindGroupLayoutDescriptor(shadowGroup)
	sizeOverrides := make(map[int]uint64)
	for _, entry := range desc.Entries {
		if entry.Buffer.Type == wgpu.BufferBindingTypeUniform {
			sizeOverrides[int(entry.Binding)] = uint64((&light.GPUShadowUniform{}).Size())
		}
	}
	for i := 0; i < cascadeCount; i++ {
		bgpKey := fmt.Sprintf("csm_data_%d", i)
		sh.SetBgp(bgpKey, bind_group_provider.NewBindGroupProvider(bgpKey))
		bgp := sh.Bgp(bgpKey)
		if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init CSM data bind group %d: %v", i, err))
		}
	}

	// Compute 2D grid atlas dimensions.
	// Worst case: every GPU light is a point light needing 6 atlas slots.
	totalSlots := int(s.lightHandler.MaxGPULights()) * 6
	tileSize := sh.LightShadowTileSize()

	// Cap grid dimensions at GPU texture limits.
	// The WebGPU spec guarantees at least 8192 for maxTextureDimension2D.
	// Some wgpu backends report uint32-max instead of the real limit,
	// so we clamp to the spec-guaranteed minimum as a safe upper bound.
	const safeMaxTextureDim = 8192
	effectiveMaxDim := maxDim
	if effectiveMaxDim <= 0 || effectiveMaxDim > safeMaxTextureDim {
		effectiveMaxDim = safeMaxTextureDim
	}
	maxTilesPerAxis := effectiveMaxDim / tileSize
	if maxTilesPerAxis < 1 {
		maxTilesPerAxis = 1
	}

	cols := min(int(math.Ceil(math.Sqrt(float64(totalSlots)))), maxTilesPerAxis)
	rows := int(math.Ceil(float64(totalSlots) / float64(cols)))
	if rows > maxTilesPerAxis {
		rows = maxTilesPerAxis
	}
	// Physical atlas capacity.
	atlasCapacity := cols * rows
	sh.SetLightShadowAtlasSlots(atlasCapacity)
	sh.SetLightShadowAtlasCols(cols)

	spotAtlasW := cols * tileSize
	spotAtlasH := rows * tileSize
	spotView, spotTex, err := s.r.CreateShadowDepthTexture(spotAtlasW, spotAtlasH)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create spot/point shadow depth texture: %v", err))
	}
	sh.SetLightShadowAtlas(spotTex)
	sh.SetLightShadowAtlasView(spotView)

	// Create per-slot BGPs for spot/point shadow depth passes.
	for i := range atlasCapacity {
		bgpKey := fmt.Sprintf("spot_shadow_%d", i)
		sh.SetBgp(bgpKey, bind_group_provider.NewBindGroupProvider(bgpKey))
		bgp := sh.Bgp(bgpKey)
		if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init spot shadow bind group %d: %v", i, err))
		}
	}

	// Register shadow depth render pipelines for each ShadowCullMode variant.
	cullModes := []struct {
		mode model.ShadowCullMode
		wgpu wgpu.CullMode
		tag  string
	}{
		{model.ShadowCullModeBack, wgpu.CullModeBack, "back"},
		{model.ShadowCullModeFront, wgpu.CullModeFront, "front"},
		{model.ShadowCullModeNone, wgpu.CullModeNone, "none"},
	}

	for _, cm := range cullModes {
		key := "shadow_depth_static_" + cm.tag
		sp := pipeline.NewPipeline(key, pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(shadowVertShader),
			pipeline.WithCullMode(cm.wgpu),
		)
		if err := s.r.RegisterShadowDepthPipeline(sp); err != nil {
			panic(fmt.Sprintf("scene: failed to register shadow depth static pipeline (%s): %v", cm.tag, err))
		}
		sh.SetPipelineKey("shadow_static_"+cm.tag, key)
	}

	if shadowSkinnedVertShader != nil {
		for _, cm := range cullModes {
			key := "shadow_depth_skinned_" + cm.tag
			ssp := pipeline.NewPipeline(key, pipeline.PipelineTypeRender,
				pipeline.WithVertexShader(shadowSkinnedVertShader),
				pipeline.WithCullMode(cm.wgpu),
			)
			if err := s.r.RegisterShadowDepthPipeline(ssp); err != nil {
				panic(fmt.Sprintf("scene: failed to register shadow depth skinned pipeline (%s): %v", cm.tag, err))
			}
			sh.SetPipelineKey("shadow_skinned_"+cm.tag, key)
		}
	}
}

// initCSMShadowLitBindGroup initializes the "csm_shadow_lit" bind group provider
// that lit fragment shaders use to sample the CSM atlas texture. Binds:
//   - @binding(0): CSM atlas texture view (Depth32Float shadow depth atlas)
//   - @binding(1): comparison sampler
//   - @binding(2): GPUCSMData uniform buffer (192 bytes: 32 + 2*80)
func (s *scene) initCSMShadowLitBindGroup(litFragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if litFragmentShader == nil {
		return
	}

	sh := s.lightHandler.ShadowHandler()
	if sh.CSMAtlasTextureView() == nil || sh.ComparisonSampler() == nil {
		return // initShadowMap must be called first
	}

	// Resolve the shadow bind group index from the lit fragment shader's
	// provider annotation matching AnnotationArgShadow.
	shadowGroup := -1
	for _, decl := range litFragmentShader.Declarations() {
		if decl.Type == shader.AnnotationTypeProvider && decl.Group != nil && decl.Args[0] == shader.AnnotationArgShadow {
			shadowGroup = *decl.Group
			break
		}
	}
	if shadowGroup < 0 {
		return
	}

	sh.SetBgp("csm_shadow_lit", bind_group_provider.NewBindGroupProvider("csm_shadow_lit"))
	bgp := sh.Bgp("csm_shadow_lit")
	desc := litFragmentShader.BindGroupLayoutDescriptor(shadowGroup)

	// Pre-set texture views and samplers for explicit bindings.
	bgp.SetTextureView(0, sh.CSMAtlasTextureView())
	bgp.SetSampler(1, sh.ComparisonSampler())
	if sh.LightShadowAtlasView() != nil {
		bgp.SetTextureView(3, sh.LightShadowAtlasView())
	}

	// Override buffer sizes: CSM uniform + light shadow entries storage.
	// Contact shadow texture + sampler (bindings 5 and 6).
	csHandler := s.lightHandler.ContactShadowHandler()
	if csHandler != nil && csHandler.Enabled() && csHandler.TextureView() != nil && csHandler.LinearSampler() != nil {
		bgp.SetTextureView(5, csHandler.TextureView())
		bgp.SetSampler(6, csHandler.LinearSampler())
	} else {
		// Fallback: 1×1 white texture (shadow = 1.0, fully lit).
		fallback := common.TextureStagingData{
			Pixels: []byte{255, 255, 255, 255},
			Width:  1,
			Height: 1,
			Linear: true,
		}
		if err := s.r.InitTextureView(bgp, 5, fallback); err != nil {
			panic(fmt.Sprintf("scene: failed to init contact shadow fallback texture: %v", err))
		}
		fallbackSampler := common.SamplerStagingData{
			AddressModeU:  wgpu.AddressModeClampToEdge,
			AddressModeV:  wgpu.AddressModeClampToEdge,
			AddressModeW:  wgpu.AddressModeClampToEdge,
			MagFilter:     wgpu.FilterModeLinear,
			MinFilter:     wgpu.FilterModeLinear,
			MipmapFilter:  wgpu.MipmapFilterModeLinear,
			LodMinClamp:   0,
			LodMaxClamp:   1,
			MaxAnisotropy: 1,
		}
		if err := s.r.InitSampler(bgp, 6, fallbackSampler); err != nil {
			panic(fmt.Sprintf("scene: failed to init contact shadow fallback sampler: %v", err))
		}
	}

	sizeOverrides := make(map[int]uint64)
	for _, decl := range litFragmentShader.Declarations() {
		if decl.Type != shader.AnnotationTypeBindingGroup || decl.Group == nil || *decl.Group != shadowGroup || decl.Binding == nil {
			continue
		}
		typeArg := string(decl.Args[2])
		if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
			typeArg = strings.TrimSuffix(stripped, ">")
		}
		switch shader.AnnotationArg(typeArg) {
		case shader.AnnotationArgCSMData:
			sizeOverrides[*decl.Binding] = uint64((&light.GPUCSMData{}).Size())
		case shader.AnnotationArgLightShadowEntry:
			sizeOverrides[*decl.Binding] = uint64(sh.LightShadowAtlasSlots()) * uint64((&light.GPULightShadowEntry{}).Size())
		}
	}

	if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init CSM shadow lit bind group: %v", err))
	}
}

// initLightCullResources initializes the Forward+ light culling pipeline and buffer resources.
func (s *scene) initLightCullResources(cullComputeShader, litFragmentShader shader.Shader, screenWidth, screenHeight int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLightCullResourcesLocked(cullComputeShader, litFragmentShader, screenWidth, screenHeight)
}

// initLightCullResourcesLocked is the lock-free inner body of initLightCullResources.
// Caller must hold s.mu.
func (s *scene) initLightCullResourcesLocked(cullComputeShader, litFragmentShader shader.Shader, screenWidth, screenHeight int) {
	if cullComputeShader == nil || litFragmentShader == nil {
		return
	}
	lightsBGP := s.lightHandler.Bgp("lights")
	if lightsBGP.Buffer(1) == nil {
		return // initLightBindGroup must be called first
	}

	s.lightHandler.Resize(screenWidth, screenHeight)
	tileCountX := s.lightHandler.TileCountX()
	tileCountY := s.lightHandler.TileCountY()

	numTiles := uint64(tileCountX) * uint64(tileCountY)

	// ── 1. Create compute BGP (cull shader's @group(0)) ────────────────
	// binding 0: cull_uniforms (uniform, 160 bytes)
	// binding 1: cull_lights (storage, read) — shared from lightsBGP binding 1
	// binding 2: tile_light_counts (storage, rw) — new buffer
	// binding 3: tile_light_indices (storage, rw) — new buffer
	cullBGP := s.lightHandler.Bgp("light_cull")

	// Pre-set the lights buffer from lightsBGP so InitBindGroup reuses it.
	if lightsBuffer := lightsBGP.Buffer(1); lightsBuffer != nil {
		cullBGP.SetBuffer(1, lightsBuffer)
	}

	cullDesc := cullComputeShader.BindGroupLayoutDescriptor(0)
	sizeOverrides := map[int]uint64{
		0: uint64((&light.GPULightCullUniforms{}).Size()),           // LightCullUniforms
		2: numTiles * 4,                                             // tile_light_counts: one u32 per tile
		3: numTiles * uint64(s.lightHandler.MaxLightsPerTile()) * 4, // tile_light_indices
	}

	if err := s.r.InitBindGroup(cullBGP, cullDesc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init light cull bind group: %v", err))
	}

	// ── 2. Register the cull compute pipeline ──────────────────────────
	pipeKey := "light_cull_compute"
	cp := pipeline.NewPipeline(pipeKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(cullComputeShader),
	)
	if err := s.r.RegisterPipelines(cp); err != nil {
		panic(fmt.Sprintf("scene: failed to register light cull compute pipeline: %v", err))
	}
	s.lightHandler.SetPipelineKey("light_cull", pipeKey)

	// ── 3. Create fragment tile BGP (lit frag shader's @group(5)) ──────
	// binding 0: tile_uniforms (uniform, 8 bytes)
	// binding 1: tile_light_counts (storage, read) — shared from cullBGP binding 2
	// binding 2: tile_light_indices (storage, read) — shared from cullBGP binding 3
	tileBGP := s.lightHandler.Bgp("tile_lit")

	if countsBuf := cullBGP.Buffer(2); countsBuf != nil {
		tileBGP.SetBuffer(1, countsBuf)
	}
	if indicesBuf := cullBGP.Buffer(3); indicesBuf != nil {
		tileBGP.SetBuffer(2, indicesBuf)
	}

	// Resolve the tile bind group index from the shader's pre-processor
	// declarations by matching the TileUniforms struct type annotation.
	tileGroup := -1
	for _, decl := range litFragmentShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgTileUniforms {
			tileGroup = *decl.Group
			break
		}
	}
	if tileGroup < 0 {
		panic("scene: lit fragment shader has no tile bind group")
	}

	tileDesc := litFragmentShader.BindGroupLayoutDescriptor(tileGroup)
	tileSizeOverrides := map[int]uint64{
		0: uint64((&light.GPUTileUniforms{}).Size()), // TileUniforms
	}
	if err := s.r.InitBindGroup(tileBGP, tileDesc, nil, tileSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init tile lit bind group: %v", err))
	}

	// ── 4. Write initial tile uniforms ─────────────────────────────────
	tileUniforms := light.GPUTileUniforms{
		TileCountX:       uint32(tileCountX),
		MaxLightsPerTile: uint32(s.lightHandler.MaxLightsPerTile()),
		ScreenWidth:      uint32(s.screenWidth),
		ScreenHeight:     uint32(s.screenHeight),
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: tileBGP, Binding: 0, Offset: 0, Data: tileUniforms.Marshal()},
	})

	s.tileBufferCapacity = tileCountX * tileCountY
}

// initComposition initializes the composition and tone mapping subsystem: creates
// offscreen HDR render targets (with optional MSAA textures), a linear sampler,
// registers the fullscreen composition pipeline, and creates the composition bind
// group provider with pre-set texture views. The G-Buffer must be initialized
// before calling this method so that SSR textures are available for binding.
func (s *scene) initComposition() {
	s.mu.Lock()
	defer s.mu.Unlock()

	ch := s.lightHandler.CompositionHandler()
	if ch == nil {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	sampleCount := s.r.SampleCount()

	// 1. Create HDR + optional MSAA + depth textures.
	hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex, err := s.r.CreateCompositionTextures(w, h, sampleCount)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create composition textures: %v", err))
	}
	ch.SetHDRTexture(hdrTex)
	ch.SetHDRTextureView(hdrView)
	ch.SetMSAATexture(msaaTex)
	ch.SetMSAATextureView(msaaView)
	ch.SetDepthTexture(depthTex)
	ch.SetDepthTextureView(depthView)

	// Override the render pipeline color target format to RGBA16Float so that
	// all subsequently registered pipelines target the offscreen HDR texture
	// instead of the swapchain surface format.
	s.r.SetRenderTargetFormat(wgpu.TextureFormatRGBA16Float)

	// 2. Create linear sampler for HDR and SSR texture sampling.
	linearSamp, err := s.r.CreateLinearSampler()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create composition linear sampler: %v", err))
	}
	ch.SetLinearSampler(linearSamp)

	// 3. Load composition shaders and register the fullscreen pipeline.
	compVert := shader.NewShader("_composition_vert", shader.ShaderTypeVertex, "engine/light/assets/composition-vert.wgsl", shader.WithInjections(s.injections))
	compFrag := shader.NewShader("_composition_frag", shader.ShaderTypeFragment, "engine/light/assets/composition-frag.wgsl", shader.WithInjections(s.injections))

	compKey := "composition"
	compPipe := pipeline.NewPipeline(compKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(compVert),
		pipeline.WithFragmentShader(compFrag),
	)
	if err := s.r.RegisterCompositionPipeline(compPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register composition pipeline: %v", err))
	}
	ch.SetPipelineKey("composition", compKey)

	// 4. Initialize the composition bind group (group 0): hdr_texture, hdr_sampler,
	// ssr_texture, ssr_sampler, composition_params uniform.
	compBGP := ch.Bgp("composition")
	compDesc := compFrag.BindGroupLayoutDescriptor(0)

	// Pre-set the HDR texture view and sampler.
	compBGP.SetTextureView(0, hdrView)
	compBGP.SetSampler(1, linearSamp)

	// Bind the SSR texture if available, otherwise create a 1×1 black fallback.
	ssrHandler := s.lightHandler.SSRHandler()
	if ssrHandler.SSRTextureView() != nil {
		compBGP.SetTextureView(2, ssrHandler.SSRTextureView())
	} else {
		fallback := common.TextureStagingData{
			Pixels: []byte{0, 0, 0, 0},
			Width:  1,
			Height: 1,
			Linear: true,
		}
		if err := s.r.InitTextureView(compBGP, 2, fallback); err != nil {
			panic(fmt.Sprintf("scene: failed to init SSR fallback texture for composition: %v", err))
		}
	}
	compBGP.SetSampler(3, linearSamp)

	s.initLuminance(ch, compBGP)

	s.initBloom(ch, compBGP, w, h)

	sizeOverrides := map[int]uint64{
		4: uint64((&light.GPUCompositionParams{}).Size()),
		5: 4, // exposure_buffer: single f32
	}
	if err := s.r.InitBindGroup(compBGP, compDesc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init composition bind group: %v", err))
	}

	ch.Resize(w, h)
	ch.SetEnabled(true)
}

// initLuminance creates the luminance compute pipeline and exposure storage buffer
// used by the auto-exposure system. It registers the luminance pipeline, creates a
// persistent 4-byte exposure buffer (initialized to the composition handler's default
// exposure), wires the luminance BGP, and sets the exposure buffer at binding 5 of
// compBGP so it is included when the composition bind group is finalized.
//
// Must be called after the HDR texture view is set on ch and after bindings 0–3 are
// set on compBGP, but before s.r.InitBindGroup(compBGP, ...) is called.
func (s *scene) initLuminance(ch light.CompositionHandler, compBGP bind_group_provider.BindGroupProvider) {
	expBuf, err := s.r.CreateBuffer("luminance_exposure", 4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create luminance exposure buffer: %v", err))
	}

	initData := make([]byte, 4)
	binary.LittleEndian.PutUint32(initData, math.Float32bits(ch.Exposure()))
	s.r.WriteRawBuffer(expBuf, 0, initData)
	ch.SetExposureBuffer(expBuf)

	lumShader := shader.NewShader("_luminance_compute", shader.ShaderTypeCompute, "engine/light/assets/luminance-compute.wgsl", shader.WithInjections(s.injections))
	lumPipe := pipeline.NewPipeline("luminance_compute", pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(lumShader))
	if err := s.r.RegisterPipelines(lumPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register luminance compute pipeline: %v", err))
	}

	lumBGP := ch.Bgp("luminance_compute")
	lumDesc := lumShader.BindGroupLayoutDescriptor(0)
	lumBGP.SetTextureView(0, ch.HDRTextureView())
	lumBGP.SetBuffer(2, expBuf)
	lumSizeOverrides := map[int]uint64{
		1: (&light.GPULuminanceParams{}).Size(),
	}
	if err := s.r.InitBindGroup(lumBGP, lumDesc, nil, lumSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init luminance compute bind group: %v", err))
	}

	compBGP.SetBuffer(5, expBuf)
}

// initBloom creates bloom mip chain textures, registers bloom compute pipelines,
// and creates per-mip downsample and upsample bind group providers. When bloom is
// disabled, a 1×1 black fallback texture is bound at composition binding 6.
func (s *scene) initBloom(ch light.CompositionHandler, compBGP bind_group_provider.BindGroupProvider, width, height int) {
	if !ch.BloomEnabled() {
		fallback := common.TextureStagingData{
			Pixels: []byte{0, 0, 0, 0},
			Width:  1,
			Height: 1,
			Linear: true,
		}
		if err := s.r.InitTextureView(compBGP, 6, fallback); err != nil {
			panic(fmt.Sprintf("scene: failed to init bloom fallback texture: %v", err))
		}
		return
	}

	halfW := width / 2
	halfH := height / 2
	if halfW <= 0 || halfH <= 0 {
		return
	}

	downTex, downReadViews, downStorageViews, upTex, upReadViews, upStorageViews, upMip0View, mipCount, err :=
		s.r.CreateBloomTextures(halfW, halfH)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create bloom textures: %v", err))
	}
	ch.SetBloomDownTexture(downTex)
	ch.SetBloomDownReadViews(downReadViews)
	ch.SetBloomDownStorageViews(downStorageViews)
	ch.SetBloomUpTexture(upTex)
	ch.SetBloomUpReadViews(upReadViews)
	ch.SetBloomUpStorageViews(upStorageViews)
	ch.SetBloomUpMip0View(upMip0View)
	ch.SetBloomMipCount(mipCount)

	downShader := shader.NewShader("_bloom_downsample", shader.ShaderTypeCompute, "engine/light/assets/bloom-downsample.wgsl", shader.WithInjections(s.injections))
	downPipe := pipeline.NewPipeline("bloom_downsample", pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(downShader))
	if err := s.r.RegisterPipelines(downPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register bloom downsample pipeline: %v", err))
	}
	ch.SetPipelineKey("bloom_downsample", "bloom_downsample")

	upShader := shader.NewShader("_bloom_upsample", shader.ShaderTypeCompute, "engine/light/assets/bloom-upsample.wgsl", shader.WithInjections(s.injections))
	upPipe := pipeline.NewPipeline("bloom_upsample", pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(upShader))
	if err := s.r.RegisterPipelines(upPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register bloom upsample pipeline: %v", err))
	}
	ch.SetPipelineKey("bloom_upsample", "bloom_upsample")

	linearSamp := ch.LinearSampler()

	downDesc := downShader.BindGroupLayoutDescriptor(0)
	bloomParamSize := uint64((&light.GPUBloomParams{}).Size())
	for i := 0; i < mipCount; i++ {
		bgpName := fmt.Sprintf("bloom_down_%d", i)
		bgp := bind_group_provider.NewBindGroupProvider(bgpName)

		if i == 0 {
			bgp.SetTextureView(0, ch.HDRTextureView())
		} else {
			bgp.SetTextureView(0, downReadViews[i-1])
		}
		bgp.SetSampler(1, linearSamp)
		bgp.SetTextureView(2, downStorageViews[i])

		sizeOverrides := map[int]uint64{3: bloomParamSize}
		if err := s.r.InitBindGroup(bgp, downDesc, nil, sizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init bloom downsample bind group mip %d: %v", i, err))
		}
		ch.SetBgp(bgpName, bgp)
	}

	upDesc := upShader.BindGroupLayoutDescriptor(0)
	for i := mipCount - 2; i >= 0; i-- {
		bgpName := fmt.Sprintf("bloom_up_%d", i)
		bgp := bind_group_provider.NewBindGroupProvider(bgpName)

		if i == mipCount-2 {
			bgp.SetTextureView(0, downReadViews[mipCount-1])
		} else {
			bgp.SetTextureView(0, upReadViews[i+1])
		}
		bgp.SetSampler(1, linearSamp)
		bgp.SetTextureView(2, downReadViews[i])
		bgp.SetTextureView(3, upStorageViews[i])

		if err := s.r.InitBindGroup(bgp, upDesc, nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init bloom upsample bind group mip %d: %v", i, err))
		}
		ch.SetBgp(bgpName, bgp)
	}

	compBGP.SetTextureView(6, upMip0View)
}

// prepareLuminance dispatches the luminance compute shader to update the adapted
// exposure storage buffer based on the current HDR frame. No-ops if auto-exposure
// is disabled or the composition handler is not initialized.
func (s *scene) prepareLuminance(dt float32) {
	if s.lightHandler == nil {
		return
	}
	ch := s.lightHandler.CompositionHandler()
	if ch == nil || !ch.Enabled() || !ch.AutoExposureEnabled() {
		return
	}

	params := light.GPULuminanceParams{
		ScreenWidth:         uint32(ch.ScreenWidth()),
		ScreenHeight:        uint32(ch.ScreenHeight()),
		AdaptSpeed:          ch.AdaptSpeed(),
		DeltaTime:           dt,
		MinExposure:         ch.MinExposure(),
		MaxExposure:         ch.MaxExposure(),
		KeyValue:            0.18,
		AutoExposureEnabled: 1,
	}

	lumBGP := ch.Bgp("luminance_compute")
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: lumBGP, Binding: 1, Offset: 0, Data: params.Marshal()},
	})
	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{PipelineKey: "luminance_compute", Provider: lumBGP, WorkGroupCount: [3]uint32{1, 1, 1}},
	})
}

// initSSR initializes the SSR (screen-space reflections) subsystem: creates the
// half-resolution SSR output texture, the Hi-Z depth pyramid with per-mip views
// and bind groups, registers all compute pipelines (Hi-Z init, Hi-Z downsample,
// SSR compute), and wires up the SSR compute bind group with G-Buffer, HDR, and
// Hi-Z texture views. The G-Buffer and composition subsystems must be initialized
// before calling this method.
func (s *scene) initSSR() {
	s.mu.Lock()
	defer s.mu.Unlock()

	ssrHandler := s.lightHandler.SSRHandler()
	gbHandler := s.lightHandler.GBufferHandler()
	compHandler := s.lightHandler.CompositionHandler()
	if ssrHandler == nil || gbHandler == nil || compHandler == nil {
		return
	}
	if !gbHandler.Enabled() || !compHandler.Enabled() {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	// SSR runs at half-resolution.
	halfW := w / 2
	halfH := h / 2
	if halfW <= 0 {
		halfW = 1
	}
	if halfH <= 0 {
		halfH = 1
	}

	// 1. Create SSR output texture (RGBA16Float, storage + texture binding).
	ssrView, ssrTex, err := s.r.CreateSSRTextures(halfW, halfH)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create SSR textures: %v", err))
	}
	ssrHandler.SetSSRTexture(ssrTex)
	ssrHandler.SetSSRTextureView(ssrView)

	// 2. Create linear sampler for composition shader to sample SSR result.
	linearSamp, err := s.r.CreateLinearSampler()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create SSR linear sampler: %v", err))
	}
	ssrHandler.SetLinearSampler(linearSamp)

	// 3. Create Hi-Z depth pyramid texture with full mip chain and per-mip views.
	hizView, hizTex, mipReadViews, mipStorageViews, mipCount, err := s.r.CreateHiZTextures(w, h)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create Hi-Z textures: %v", err))
	}
	ssrHandler.SetHiZTexture(hizTex)
	ssrHandler.SetHiZTextureView(hizView)
	ssrHandler.SetHiZMipCount(mipCount)
	ssrHandler.SetHiZMipReadViews(mipReadViews)
	ssrHandler.SetHiZStorageViews(mipStorageViews)

	// 4. Register Hi-Z init compute pipeline (copies depth → Hi-Z mip 0).
	hizInitShader := shader.NewShader("_hiz_init", shader.ShaderTypeCompute, "engine/light/assets/hiz-init-compute.wgsl", shader.WithInjections(s.injections))
	hizInitKey := "hiz_init"
	hizInitPipe := pipeline.NewPipeline(hizInitKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(hizInitShader),
	)
	if err := s.r.RegisterPipelines(hizInitPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register Hi-Z init pipeline: %v", err))
	}
	ssrHandler.SetPipelineKey("hiz_init", hizInitKey)

	// 5. Create Hi-Z init bind group: binding 0 = gbuffer_depth, binding 1 = hiz mip 0 storage.
	hizInitBGP := bind_group_provider.NewBindGroupProvider("hiz_init")
	hizInitBGP.SetTextureView(0, gbHandler.DepthTextureView())
	hizInitBGP.SetTextureView(1, mipStorageViews[0])
	hizInitDesc := hizInitShader.BindGroupLayoutDescriptor(0)
	if err := s.r.InitBindGroup(hizInitBGP, hizInitDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init Hi-Z init bind group: %v", err))
	}
	ssrHandler.SetBgp("hiz_init", hizInitBGP)

	// 6. Register Hi-Z downsample compute pipeline (min of 2×2 from prev mip).
	hizDownShader := shader.NewShader("_hiz_downsample", shader.ShaderTypeCompute, "engine/light/assets/hiz-downsample-compute.wgsl", shader.WithInjections(s.injections))
	hizDownKey := "hiz_downsample"
	hizDownPipe := pipeline.NewPipeline(hizDownKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(hizDownShader),
	)
	if err := s.r.RegisterPipelines(hizDownPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register Hi-Z downsample pipeline: %v", err))
	}
	ssrHandler.SetPipelineKey("hiz_downsample", hizDownKey)

	// 7. Create per-mip downsample bind groups: for each mip level 1..N-1,
	//    binding 0 = read view of mip N-1, binding 1 = storage view of mip N.
	hizDownDesc := hizDownShader.BindGroupLayoutDescriptor(0)
	for i := 1; i < mipCount; i++ {
		bgpName := fmt.Sprintf("hiz_down_%d", i)
		bgp := bind_group_provider.NewBindGroupProvider(bgpName)
		bgp.SetTextureView(0, mipReadViews[i-1])
		bgp.SetTextureView(1, mipStorageViews[i])
		if err := s.r.InitBindGroup(bgp, hizDownDesc, nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init Hi-Z downsample bind group mip %d: %v", i, err))
		}
		ssrHandler.SetBgp(bgpName, bgp)
	}

	// 8. Load SSR compute shader and register compute pipeline.
	ssrCompShader := shader.NewShader("_ssr_compute", shader.ShaderTypeCompute, "engine/light/assets/ssr-compute.wgsl", shader.WithInjections(s.injections))
	ssrCompKey := "ssr_compute"
	ssrCompPipe := pipeline.NewPipeline(ssrCompKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(ssrCompShader),
	)
	if err := s.r.RegisterPipelines(ssrCompPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register SSR compute pipeline: %v", err))
	}
	ssrHandler.SetPipelineKey("ssr_compute", ssrCompKey)

	// 9. Initialize SSR compute bind group (group 0):
	//   binding 0: ssr_params uniform
	//   binding 1: gbuffer_normal texture
	//   binding 2: gbuffer_depth texture
	//   binding 3: hdr_texture
	//   binding 4: ssr_output storage texture
	//   binding 5: hiz_texture (full mip chain)
	ssrBGP := ssrHandler.Bgp("ssr_compute")
	ssrDesc := ssrCompShader.BindGroupLayoutDescriptor(0)

	ssrBGP.SetTextureView(1, gbHandler.NormalTextureView())
	ssrBGP.SetTextureView(2, gbHandler.DepthTextureView())
	ssrBGP.SetTextureView(3, compHandler.HDRTextureView())
	ssrBGP.SetTextureView(4, ssrView)
	ssrBGP.SetTextureView(5, hizView)

	ssrSizeOverrides := map[int]uint64{
		0: uint64((&light.GPUSSRParams{}).Size()),
	}
	if err := s.r.InitBindGroup(ssrBGP, ssrDesc, nil, ssrSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSR compute bind group: %v", err))
	}

	ssrHandler.Resize(w, h)
	ssrHandler.SetEnabled(true)
}

// initLighting initializes the entire lighting pipeline in the correct order:
// light storage buffer, shadow map resources, shadow lit bind group, and Forward+
// light culling. All lighting shaders are loaded internally from the engine's
// standard light shader assets.
func (s *scene) initLighting(screenWidth, screenHeight int) {
	litFragShader := shader.NewShader("_lit_frag_csm", shader.ShaderTypeFragment, "engine/light/assets/lit-frag-csm.wgsl", shader.WithInjections(s.injections))
	cullComputeShader := shader.NewShader("_light_cull_compute", shader.ShaderTypeCompute, "engine/light/assets/light-cull-compute.wgsl", shader.WithInjections(s.injections))

	// 1. Light storage buffer (must be first — other steps share this buffer).
	s.initLightBindGroup(litFragShader)

	// 2. Shadow resources — creates depth-only atlas texture, comparison sampler,
	// and PCF shadow render pipelines.
	shadowVertShader := shader.NewShader("_shadow_depth_vert", shader.ShaderTypeVertex, "engine/light/assets/shadow-depth-vert.wgsl", shader.WithInjections(s.injections))
	shadowSkinnedVertShader := shader.NewShader("_shadow_depth_skinned_vert", shader.ShaderTypeVertex, "engine/light/assets/shadow-depth-skinned-vert.wgsl", shader.WithInjections(s.injections))
	s.initShadowMap(shadowVertShader, shadowSkinnedVertShader)

	// 3. Shadow lit BGP (fragment-side shadow sampling — references shadow resources from step 2).
	s.initCSMShadowLitBindGroup(litFragShader)

	// 4. Forward+ tile culling pipeline and shared tile buffers (references lights buffer from step 1).
	s.initLightCullResources(cullComputeShader, litFragShader, screenWidth, screenHeight)

	// 5. Re-create the camera bind group with merged VERTEX|FRAGMENT visibility.
	//
	// The camera's bind group was originally created in NewScene from the vertex
	// shader alone (visibility = VERTEX). The lit fragment shader also declares the
	// camera group (visibility = FRAGMENT). The render pipeline merges these into
	// VERTEX|FRAGMENT. WebGPU requires exact bind group layout equivalence, so the
	// camera BGL must be recreated with the combined visibility to pass validation.
	s.reinitCameraBGPForLitPipeline(litFragShader)

	// 6. G-Buffer MRT pre-pass (required by SSAO and SSR).
	s.initGBuffer()

	// 7. SSAO — hemisphere sampling + bilateral blur (requires G-Buffer).
	s.initSSAO()

	// 7b. Contact shadows — screen-space ray march (requires G-Buffer).
	s.initContactShadows()

	// 8. SSAO lit bind group — binds blurred SSAO texture at @group(6) for the lit
	// fragment shader. When SSAO is disabled, a 1×1 white fallback is used (ao=1.0).
	s.initSSAOLitBindGroup(litFragShader)

	// 9. Composition — offscreen HDR render target + fullscreen tone-mapping pass.
	// Must come before SSR so the HDR texture exists for SSR to read.
	s.initComposition()

	// 10. SSR — screen-space reflections compute pass (requires G-Buffer + composition HDR texture).
	s.initSSR()

	// Re-bind the SSR texture on the composition BGP now that it exists.
	if s.lightHandler.SSRHandler().Enabled() && s.lightHandler.CompositionHandler().Enabled() {
		compBGP := s.lightHandler.CompositionHandler().Bgp("composition")
		if compBGP != nil && s.lightHandler.SSRHandler().SSRTextureView() != nil {
			compBGP.SetTextureView(2, s.lightHandler.SSRHandler().SSRTextureView())
			// Rebuild the composition bind group to pick up the real SSR texture.
			compFrag := shader.NewShader("_composition_frag_rebind", shader.ShaderTypeFragment, "engine/light/assets/composition-frag.wgsl", shader.WithInjections(s.injections))
			compDesc := compFrag.BindGroupLayoutDescriptor(0)
			sizeOverrides := map[int]uint64{
				4: uint64((&light.GPUCompositionParams{}).Size()),
			}
			if err := s.r.InitBindGroup(compBGP, compDesc, nil, sizeOverrides); err != nil {
				panic(fmt.Sprintf("scene: failed to re-init composition bind group with SSR texture: %v", err))
			}
		}
	}

	// 11. Mark the lighting subsystem as GPU-initialized.
	s.lightHandler.SetEnabled(true)
	s.postProcessingInitialized = true
}

// initPhysics creates GPU buffers, bind groups, and compute pipelines for the
// physics simulation. Called once when the first rigid body is added to the scene.
// Follows the same InitBindGroup + shared buffer + pipeline registration pattern
// used by createAnimator. Caller must hold s.mu write lock.
func (s *scene) initPhysics() {
	s.buildInjectionMap()
	ph := s.physicsHandler

	type stageEntry struct {
		name string
		path string
	}

	stages := []stageEntry{
		{"particle_values", "engine/physics/assets/particle-values.wgsl"},
		{"aabb_reduce", "engine/physics/assets/aabb-reduce.wgsl"},
		{"grid_build_params", "engine/physics/assets/grid-build-params.wgsl"},
		{"grid_clear", "engine/physics/assets/grid-clear.wgsl"},
		{"grid_insert", "engine/physics/assets/grid-insert.wgsl"},
		{"collision", "engine/physics/assets/collision-reaction.wgsl"},
		{"momenta", "engine/physics/assets/compute-momenta.wgsl"},
		{"integrate", "engine/physics/assets/integrate.wgsl"},
		{"sync", "engine/physics/assets/physics-sync.wgsl"},
	}

	shaders := make(map[string]shader.Shader, len(stages))
	for _, st := range stages {
		shaders[st.name] = shader.NewShader("physics_"+st.name, shader.ShaderTypeCompute, st.path, shader.WithInjections(s.injections))
	}

	// Canonical buffer indices on the buffers BGP. These are the contract between
	// physics.go staged writes and the GPU buffer layout.
	annotatedBufferIndex := map[shader.AnnotationArg]int{
		shader.AnnotationArgPhysicsBody:       0,
		shader.AnnotationArgPhysicsParticle:   1,
		shader.AnnotationArgPhysicsGrid:       2,
		shader.AnnotationArgPhysicsGlobals:    3,
		shader.AnnotationArgPhysicsGridParams: 4,
	}

	// Manually declared WGSL bindings (atomic types not expressible via annotations)
	// are matched by their variable name from the parsed shader source.
	manualVarBufferIndex := map[string]int{
		"aabb":     5, // AABB atomics (6 × atomic<u32>)
		"grid":     2, // atomic view of the grid cell buffer
		"sync_map": 7, // sync mapping (body index → animator instance ID)
	}

	// Derive per-shader binding→canonical buffer index maps from declarations and
	// var names. Simultaneously collect layout entries for the unified buffers BGP.
	bufferMaps := make(map[string]map[int]int, len(stages))
	collected := make(map[int]wgpu.BindGroupLayoutEntry)

	for _, st := range stages {
		sh := shaders[st.name]
		desc := sh.BindGroupLayoutDescriptor(0)
		bmap := make(map[int]int, len(desc.Entries))

		// Resolve annotated bindings via the declaration list
		for _, decl := range sh.Declarations() {
			typeArg := string(decl.Args[2])
			if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
				typeArg = strings.TrimSuffix(stripped, ">")
			}
			if bufIdx, ok := annotatedBufferIndex[shader.AnnotationArg(typeArg)]; ok {
				bmap[*decl.Binding] = bufIdx
			}
		}

		// Resolve remaining (manually-declared) bindings by WGSL variable name
		for _, entry := range desc.Entries {
			b := int(entry.Binding)
			if _, done := bmap[b]; done {
				continue
			}
			if bufIdx, ok := manualVarBufferIndex[sh.BindGroupVarName(0, b)]; ok {
				bmap[b] = bufIdx
			}
		}

		// Collect layout entries at canonical indices for the buffers BGP descriptor.
		// Prefer entries with larger MinBindingSize so the struct-based entry (e.g.
		// GridCell=16) wins over the atomic element size (u32=4).
		for _, entry := range desc.Entries {
			if canonIdx, ok := bmap[int(entry.Binding)]; ok {
				if existing, exists := collected[canonIdx]; !exists || entry.Buffer.MinBindingSize > existing.Buffer.MinBindingSize {
					e := entry
					e.Binding = uint32(canonIdx)
					collected[canonIdx] = e
				}
			}
		}

		bufferMaps[st.name] = bmap
	}

	// The sync mapping buffer is referenced by the sync shader at binding 1 (manual
	// var) and populated via staged writes during RegisterBody. Add it manually to the
	// buffers BGP descriptor since its canonical index (7) is not discovered from other
	// annotated stages.
	collected[7] = wgpu.BindGroupLayoutEntry{
		Binding: 7, Visibility: wgpu.ShaderStageCompute,
		Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4},
	}

	// Assemble the buffers BGP descriptor from collected entries, sorted by binding.
	buffersEntries := make([]wgpu.BindGroupLayoutEntry, 0, len(collected))
	for i := range 8 {
		if e, ok := collected[i]; ok {
			buffersEntries = append(buffersEntries, e)
		}
	}

	maxBodies := uint64(ph.MaxBodies())
	maxParticles := uint64(ph.MaxParticles())
	maxGridCells := uint64(ph.MaxGridCells())

	buffersDesc := wgpu.BindGroupLayoutDescriptor{
		Label:   "physics_buffers",
		Entries: buffersEntries,
	}

	buffersSizeOverrides := map[int]uint64{
		0: maxBodies * uint64((&physics.GPUBody{}).Size()),
		1: maxParticles * uint64((&physics.GPUParticle{}).Size()),
		2: maxGridCells * uint64((&physics.GPUGridCell{}).Size()),
		3: uint64((&physics.GPUPhysicsGlobals{}).Size()),
		4: uint64((&physics.GPUGridParams{}).Size()),
		5: 24,            // aabbAtomics: 6 × u32 (no struct type)
		7: maxBodies * 4, // syncMapping: u32 per body
	}

	buffersUsageOverrides := map[int]wgpu.BufferUsage{
		0: wgpu.BufferUsageCopySrc, // allow copy-to-staging for readback
	}

	if err := s.r.InitBindGroup(ph.Buffers(), buffersDesc, buffersUsageOverrides, buffersSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init physics buffers BGP: %v", err))
	}

	// Wire shared physical buffers from the buffers BGP into each per-shader BGP,
	// then create bind groups. InitBindGroup skips buffer creation for bindings
	// that already have a buffer set via SetBuffer.
	for _, st := range stages {
		bgp := ph.Bgp(st.name)
		for shaderBinding, canonIdx := range bufferMaps[st.name] {
			bgp.SetBuffer(shaderBinding, ph.Buffers().Buffer(canonIdx))
		}
		// Sync InitBindGroup is deferred to Add() because binding 2
		// (AnimationData) comes from the Animator, not the physics buffers.
		if st.name == "sync" {
			continue
		}
		desc := shaders[st.name].BindGroupLayoutDescriptor(0)
		if err := s.r.InitBindGroup(bgp, desc, nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init physics BGP %q: %v", st.name, err))
		}
	}

	// Register compute pipelines and store their keys on the physics handler.
	for _, st := range stages {
		sh := shaders[st.name]
		p := pipeline.NewPipeline(sh.Key(), pipeline.PipelineTypeCompute, pipeline.WithComputeShader(sh))
		if err := s.r.RegisterPipelines(p); err != nil {
			panic(fmt.Sprintf("scene: failed to register physics pipeline %q: %v", st.name, err))
		}
		ph.SetPipelineKey(st.name, p.PipelineKey())
	}

	// Register the bone_particle_update pipeline separately. Its BGP is created
	// per-group in createBoneParticleUpdateGroup because it requires buffers from
	// both the physics handler and a specific Animator.
	{
		boneUpdateShader := shader.NewShader("physics_bone_update", shader.ShaderTypeCompute,
			"engine/physics/assets/bone-particle-update.wgsl", shader.WithInjections(s.injections))
		boneUpdatePipe := pipeline.NewPipeline(boneUpdateShader.Key(), pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(boneUpdateShader))
		if err := s.r.RegisterPipelines(boneUpdatePipe); err != nil {
			panic(fmt.Sprintf("scene: failed to register bone_update pipeline: %v", err))
		}
		ph.SetPipelineKey("bone_update", boneUpdatePipe.PipelineKey())
	}

	// Create a staging buffer for GPU→CPU readback of body positions and quaternions.
	// Sized for the full body buffer so any number of bodies up to maxBodies can be read back.
	stagingSize := maxBodies * uint64((&physics.GPUBody{}).Size())
	stagingBuf, err := s.r.CreateBuffer("physics_staging", stagingSize, wgpu.BufferUsageMapRead|wgpu.BufferUsageCopyDst)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create physics staging buffer: %v", err))
	}
	ph.SetStagingBuffer(stagingBuf)
}

// initPhysicsSyncGroup creates a per-animator sync bind group provider for the
// physics sync shader dispatch. Each group has its own sync_map buffer (initialized
// to the 0xFFFFFFFF sentinel so the shader skips non-member bodies) and references
// the animator's AnimationData buffer at binding 2. The bodies and globals buffers
// are shared from the physics handler. Caller must hold s.mu write lock.
//
// Parameters:
//   - anim: the Animator that owns the bodies in this sync group
//   - computeShader: the compute shader used by the Animator (for AnimationData binding discovery)
//
// Returns:
//   - int: the ID of the new sync group in s.physicsSyncGroup
func (s *scene) initPhysicsSyncGroup(anim animator.Animator) int {
	ph := s.physicsHandler

	// Discover the AnimationData binding index from the standard simple compute shader (cached).
	if s.physicsAnimBinding < 0 {
		syncComputeShader := shader.NewShader("_sync_compute_init", shader.ShaderTypeCompute, "engine/renderer/animator/assets/simple-compute.wgsl", shader.WithInjections(s.injections))
		for _, decl := range syncComputeShader.Declarations() {
			typeArg := string(decl.Args[2])
			if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
				typeArg = strings.TrimSuffix(stripped, ">")
			}
			if shader.AnnotationArg(typeArg) == shader.AnnotationArgAnimationData {
				s.physicsAnimBinding = *decl.Binding
				break
			}
		}
	}

	if s.physicsSyncGroup == nil {
		s.physicsSyncGroup = make(map[int]bind_group_provider.BindGroupProvider)
	}
	groupID := uint32(len(s.physicsSyncGroup))
	bgpLabel := fmt.Sprintf("physics_sync_group_%d", groupID)
	bgp := bind_group_provider.NewBindGroupProvider(bgpLabel)

	// Wire shared physics buffers (bodies=0, globals=3) from the central buffers BGP.
	bgp.SetBuffer(0, ph.Buffers().Buffer(0))
	bgp.SetBuffer(3, ph.Buffers().Buffer(3))

	// Wire this animator's AnimationData buffer at binding 2.
	animDataBuf := anim.ComputeBindGroupProvider().Buffer(s.physicsAnimBinding)
	bgp.SetBuffer(2, animDataBuf)

	// Binding 1 (sync_map) is left unset so InitBindGroup creates a new per-group buffer.
	syncShader := s.r.Pipeline(ph.PipelineKey("sync")).Shader(shader.ShaderTypeCompute)
	syncDesc := syncShader.BindGroupLayoutDescriptor(0)

	sizeOverrides := map[int]uint64{
		1: uint64(ph.MaxBodies()) * 4,
	}
	if err := s.r.InitBindGroup(bgp, syncDesc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init physics sync BGP for group %d: %v", groupID, err))
	}

	// Initialize the sync_map buffer to all 0xFFFFFFFF (sentinel). The shader
	// checks this value and skips bodies not belonging to the group.
	sentinelData := make([]byte, ph.MaxBodies()*4)
	for i := 0; i < len(sentinelData); i += 4 {
		binary.LittleEndian.PutUint32(sentinelData[i:i+4], 0xFFFFFFFF)
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: bgp, Binding: 1, Offset: 0, Data: sentinelData},
	})

	s.physicsSyncGroup[int(groupID)] = bgp
	s.physicsSyncAnimMap[anim] = int(groupID)

	return int(groupID)
}

// reinitCameraBGPForLitPipeline recreates the camera's bind group with merged
// VERTEX|FRAGMENT visibility so it matches the lit render pipeline's layout.
//
// The camera BGL was originally created from the vertex shader alone (VERTEX).
// When the lit fragment shader also declares the same camera group, the render
// pipeline merges the layout entries with VERTEX|FRAGMENT visibility. WebGPU
// requires exact bind group layout equivalence, so the camera BGL must be
// recreated with the combined visibility to avoid SetBindGroup validation errors.
//
// The existing camera uniform buffer is preserved — only the layout and bind
// group objects are recreated.
//
// Parameters:
//   - litFragShader: the lit fragment shader that may declare a camera group
func (s *scene) reinitCameraBGPForLitPipeline(litFragShader shader.Shader) {
	if litFragShader == nil {
		return
	}

	// Resolve the camera group index from the shader's pre-processor
	// declarations by matching the Camera struct type annotation.
	cameraGroup := -1
	for _, decl := range litFragShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgCamera {
			cameraGroup = *decl.Group
			break
		}
	}
	if cameraGroup < 0 {
		return // Fragment shader doesn't declare a camera group; no re-init needed.
	}

	bgp := s.cam.BindGroupProvider()
	if bgp == nil {
		return
	}

	// Grab the fragment shader's descriptor and add VERTEX visibility to every
	// entry so the resulting layout matches both shader stages.
	fragDesc := litFragShader.BindGroupLayoutDescriptor(cameraGroup)
	entries := make([]wgpu.BindGroupLayoutEntry, len(fragDesc.Entries))
	copy(entries, fragDesc.Entries)
	for i := range entries {
		entries[i].Visibility |= wgpu.ShaderStageVertex
	}
	mergedDesc := wgpu.BindGroupLayoutDescriptor{
		Label:   fragDesc.Label,
		Entries: entries,
	}

	// Clear the old layout so InitBindGroup creates a new one from mergedDesc.
	bgp.SetBindGroupLayout(nil)
	if err := s.r.InitBindGroup(bgp, mergedDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to reinit camera bind group for lit pipeline: %v", err))
	}
}

// patchSyncMapEntry stages a write to the per-group sync_map buffer that maps the
// given object's physics body slot to a new Animator instance ID. Pass 0xFFFFFFFF
// as instanceID to sentinel the entry (disabling sync for that body). Caller must
// hold s.mu write lock. No-op if the object has no physics body or its Animator has
// no sync group.
func (s *scene) patchSyncMapEntry(anim animator.Animator, objID uint64, instanceID uint32) {
	if s.physicsHandler == nil || anim == nil {
		return
	}
	bodyIdx, ok := s.physicsHandler.BodyIndex(objID)
	if !ok {
		return
	}
	sgIdx, exists := s.physicsSyncAnimMap[anim]
	if !exists {
		return
	}
	data := make([]byte, 4)
	binary.LittleEndian.PutUint32(data, instanceID)
	s.physicsSyncWrites = append(s.physicsSyncWrites, bind_group_provider.BufferWrite{
		Provider: s.physicsSyncGroup[sgIdx],
		Binding:  1,
		Offset:   uint64(bodyIdx) * 4,
		Data:     data,
	})
}

// createBoneParticleUpdateGroup sets up a per-kinematic-body bind group provider
// for the bone_particle_update compute shader. The BGP wires the shared particle and
// body buffers from the physics handler with the scratch_matrices buffer from the
// Animator's compute bind group (binding 5 in skeletal-compute.wgsl). A small uniform
// at binding 3 carries the particle range, bone count, and animator instance index.
// Caller must hold s.mu write lock.
//
// Parameters:
//   - anim: the skeletal Animator owning the bone matrices
//   - bodyIndex: the GPU body slot returned by RegisterBody
//   - mdl: the Model with skeleton data (for bone count)
//   - instanceIndex: the Animator instance slot for scratch_matrices indexing
func (s *scene) createBoneParticleUpdateGroup(anim animator.Animator, bodyIndex int, mdl model.Model, instanceIndex uint32) {
	ph := s.physicsHandler

	particleStart, particleCount := ph.BodyParticleInfo(bodyIndex)
	if particleCount == 0 {
		return
	}

	boneCount := uint32(len(mdl.Skeleton().Bones))

	// scratch_matrices lives at binding 5 in the skeletal-compute shader's BGP.
	// This is a manually-declared WGSL binding (no @oxy:group annotation).
	const scratchBinding = 5

	bgpLabel := fmt.Sprintf("bone_particle_update_%d", len(s.boneParticleUpdateGroups))
	bgp := bind_group_provider.NewBindGroupProvider(bgpLabel)

	// Wire shared physics buffers and the animator's scratch_matrices buffer.
	// model_data lives at binding 6 in the skeletal-compute shader's BGP.
	const modelDataBinding = 6

	bgp.SetBuffer(0, ph.Buffers().Buffer(1))                                 // particles
	bgp.SetBuffer(1, ph.Buffers().Buffer(0))                                 // bodies
	bgp.SetBuffer(2, anim.ComputeBindGroupProvider().Buffer(scratchBinding)) // scratch_matrices
	// Binding 3 (params uniform) is left unset so InitBindGroup creates a new buffer.
	bgp.SetBuffer(4, anim.ComputeBindGroupProvider().Buffer(modelDataBinding)) // model_data

	// Use the bone_update shader's layout descriptor to initialize the bind group.
	boneUpdateKey := ph.PipelineKey("bone_update")
	boneUpdatePipe := s.r.Pipeline(boneUpdateKey)
	if boneUpdatePipe == nil {
		panic("scene: bone_update pipeline not registered")
	}
	boneUpdateShader := boneUpdatePipe.Shader(shader.ShaderTypeCompute)
	boneUpdateDesc := boneUpdateShader.BindGroupLayoutDescriptor(0)

	if err := s.r.InitBindGroup(bgp, boneUpdateDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init bone particle update BGP: %v", err))
	}

	// Upload the constant params uniform (does not change frame-to-frame).
	paramsData := make([]byte, 16)
	binary.LittleEndian.PutUint32(paramsData[0:4], particleStart)
	binary.LittleEndian.PutUint32(paramsData[4:8], particleCount)
	binary.LittleEndian.PutUint32(paramsData[8:12], boneCount)
	binary.LittleEndian.PutUint32(paramsData[12:16], instanceIndex)
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: bgp, Binding: 3, Offset: 0, Data: paramsData},
	})

	s.boneParticleUpdateGroups = append(s.boneParticleUpdateGroups, &boneParticleUpdateGroup{
		bgp:           bgp,
		particleStart: particleStart,
		particleCount: particleCount,
		boneCount:     boneCount,
		instanceIndex: instanceIndex,
	})
}

// initMaterialGPU creates GPU resources (textures, samplers, bind group) for a single Material
// by inspecting the fragment shader's pre-processed Declarations for @oxy:provider annotations
// with the "material" identity. Multiple material groups are supported: each group with an
// @oxy:provider material annotation gets its own BindGroupProvider, enabling a single material
// to own resources across several bind groups (e.g. textures at group 2, effect uniforms at group 3).
// Per-binding roles (diffuse_texture, normal_texture, etc.) are resolved from the declaration Args,
// eliminating the need for variable-name string matching.
//
// Parameters:
//   - mat: the Material to initialize GPU resources for
//   - fragmentShader: the fragment shader whose @oxy:provider material annotations define the layout
//   - providerName: a unique name prefix for the created BindGroupProviders
//
// Returns:
//   - error: an error if GPU resource creation fails
func (s *scene) initMaterialGPU(mat material.Material, fragmentShader shader.Shader, providerName string) error {
	// Phase 1: Collect all groups annotated with @oxy:provider material and their binding roles.
	type groupInfo struct {
		bindingRoles map[int]shader.AnnotationArg
	}
	materialGroups := make(map[int]*groupInfo)

	for _, decl := range fragmentShader.Declarations() {
		if decl.Type != shader.AnnotationTypeProvider || decl.Group == nil {
			continue
		}
		if decl.Args[0] != shader.AnnotationArgMaterial {
			continue
		}
		g := *decl.Group
		if _, exists := materialGroups[g]; !exists {
			materialGroups[g] = &groupInfo{bindingRoles: make(map[int]shader.AnnotationArg)}
		}
		if len(decl.Args) > 1 && decl.Binding != nil {
			materialGroups[g].bindingRoles[*decl.Binding] = decl.Args[1]
		}
	}

	if len(materialGroups) == 0 {
		return nil
	}

	// Sort group indices so the lowest group (typically the texture group) becomes the
	// primary BindGroupProvider for backward-compatible access via mat.BindGroupProvider().
	groupIndices := make([]int, 0, len(materialGroups))
	for g := range materialGroups {
		groupIndices = append(groupIndices, g)
	}
	sort.Ints(groupIndices)

	// Texture role → material texture lookup
	type textureBinding struct {
		tex  *common.ImportedTexture
		role shader.AnnotationArg
	}
	roleToTexture := map[shader.AnnotationArg]textureBinding{
		shader.AnnotationArgDiffuseTexture:           {tex: mat.DiffuseTexture(), role: shader.AnnotationArgDiffuseTexture},
		shader.AnnotationArgNormalTexture:            {tex: mat.NormalTexture(), role: shader.AnnotationArgNormalTexture},
		shader.AnnotationArgMetallicRoughnessTexture: {tex: mat.MetallicRoughnessTexture(), role: shader.AnnotationArgMetallicRoughnessTexture},
	}
	textureSamplerPairs := map[shader.AnnotationArg]shader.AnnotationArg{
		shader.AnnotationArgDiffuseTexture:           shader.AnnotationArgDiffuseSampler,
		shader.AnnotationArgNormalTexture:            shader.AnnotationArgNormalSampler,
		shader.AnnotationArgMetallicRoughnessTexture: shader.AnnotationArgMetallicRoughnessSampler,
	}

	// Phase 2: For each material group, create a BGP, process textures/samplers, init bind group.
	firstGroup := true
	for _, groupIdx := range groupIndices {
		gi := materialGroups[groupIdx]
		provName := fmt.Sprintf("%s_g%d", providerName, groupIdx)
		provider := bind_group_provider.NewBindGroupProvider(provName)

		// Build binding→role reverse map for this group
		roleToBinding := make(map[shader.AnnotationArg]int)
		for binding, role := range gi.bindingRoles {
			roleToBinding[role] = binding
		}

		// Process user-supplied textures + their paired samplers
		for texRole, tb := range roleToTexture {
			if tb.tex == nil {
				continue
			}
			texBindingIdx, hasTexBinding := roleToBinding[texRole]
			if !hasTexBinding {
				continue
			}

			samplerRole := textureSamplerPairs[texRole]
			samplerBindingIdx, hasSamplerBinding := roleToBinding[samplerRole]

			pixels, width, height, err := tb.tex.Decode()
			if err != nil {
				return fmt.Errorf("failed to decode %s texture: %w", texRole, err)
			}
			isLinear := texRole == shader.AnnotationArgNormalTexture || texRole == shader.AnnotationArgMetallicRoughnessTexture
			stagingData := common.TextureStagingData{
				Pixels: pixels,
				Width:  width,
				Height: height,
				Linear: isLinear,
			}
			if err := s.r.InitTextureView(provider, texBindingIdx, stagingData); err != nil {
				return fmt.Errorf("failed to init %s texture view: %w", texRole, err)
			}
			if hasSamplerBinding {
				samplerData := common.SamplerStagingData{
					AddressModeU:  wgpu.AddressModeRepeat,
					AddressModeV:  wgpu.AddressModeRepeat,
					AddressModeW:  wgpu.AddressModeRepeat,
					MagFilter:     wgpu.FilterModeLinear,
					MinFilter:     wgpu.FilterModeLinear,
					MipmapFilter:  wgpu.MipmapFilterModeLinear,
					LodMinClamp:   0,
					LodMaxClamp:   32,
					MaxAnisotropy: 1,
				}
				if tb.tex.SamplerData != nil {
					samplerData = *tb.tex.SamplerData
				}
				if err := s.r.InitSampler(provider, samplerBindingIdx, samplerData); err != nil {
					return fmt.Errorf("failed to init %s sampler: %w", samplerRole, err)
				}
			}
		}

		// Fill in fallback textures/samplers for any texture or sampler bindings without data.
		descriptor := fragmentShader.BindGroupLayoutDescriptor(groupIdx)
		for _, entry := range descriptor.Entries {
			binding := int(entry.Binding)
			isTexture := entry.Texture.SampleType != wgpu.TextureSampleTypeUndefined
			isSampler := entry.Sampler.Type != wgpu.SamplerBindingTypeUndefined

			if isTexture && provider.TextureView(binding) == nil {
				role := gi.bindingRoles[binding]
				var pixel [4]byte
				switch role {
				case shader.AnnotationArgNormalTexture:
					pixel = [4]byte{128, 128, 255, 255}
				case shader.AnnotationArgMetallicRoughnessTexture:
					// Encode the material's scalar roughness (G) and metallic (B) into the
					// fallback 1×1 texture so shaders that read these values from the texture
					// (e.g. the G-Buffer pass for SSR) get the correct material properties.
					roughByte := byte(mat.Roughness() * 255)
					metalByte := byte(mat.Metallic() * 255)
					pixel = [4]byte{0, roughByte, metalByte, 255}
				default:
					pixel = [4]byte{255, 255, 255, 255}
				}
				isLinear := role == shader.AnnotationArgNormalTexture || role == shader.AnnotationArgMetallicRoughnessTexture
				fallback := common.TextureStagingData{
					Pixels: pixel[:],
					Width:  1,
					Height: 1,
					Linear: isLinear,
				}
				if err := s.r.InitTextureView(provider, binding, fallback); err != nil {
					return fmt.Errorf("failed to init fallback texture at binding %d: %w", binding, err)
				}
			}

			if isSampler && provider.Sampler(binding) == nil {
				fallbackSampler := common.SamplerStagingData{
					AddressModeU:  wgpu.AddressModeRepeat,
					AddressModeV:  wgpu.AddressModeRepeat,
					AddressModeW:  wgpu.AddressModeRepeat,
					MagFilter:     wgpu.FilterModeLinear,
					MinFilter:     wgpu.FilterModeLinear,
					MipmapFilter:  wgpu.MipmapFilterModeLinear,
					LodMinClamp:   0,
					LodMaxClamp:   32,
					MaxAnisotropy: 1,
				}
				if err := s.r.InitSampler(provider, binding, fallbackSampler); err != nil {
					return fmt.Errorf("failed to init fallback sampler at binding %d: %w", binding, err)
				}
			}
		}

		// Check for a material_params uniform binding in this group.
		paramsBinding := -1
		for binding, role := range gi.bindingRoles {
			if role == shader.AnnotationArgMaterialParams {
				paramsBinding = binding
				break
			}
		}
		var sizeOverrides map[int]uint64
		if paramsBinding >= 0 {
			sizeOverrides = map[int]uint64{
				paramsBinding: uint64((&material.GPUMaterialParams{}).Size()),
			}
		}

		if err := s.r.InitBindGroup(provider, descriptor, nil, sizeOverrides); err != nil {
			return fmt.Errorf("failed to init material bind group for group %d: %w", groupIdx, err)
		}

		if paramsBinding >= 0 {
			params := material.GPUMaterialParams{AlphaCutoff: mat.AlphaCutoff()}
			s.r.WriteBuffers([]bind_group_provider.BufferWrite{
				{Provider: provider, Binding: paramsBinding, Offset: 0, Data: params.Marshal()},
			})
		}

		// First material group becomes the primary BindGroupProvider for backward compat.
		if firstGroup {
			mat.SetBindGroupProvider(provider)
			firstGroup = false
		}
		mat.SetProvider(groupIdx, provider)
	}

	return nil
}

// createAnimator creates a new Animator for the given Model, registers its compute
// and render pipelines on the renderer, initializes GPU resources for the animator's
// bind group providers, and returns the configured Animator. Caller must hold s.mu write lock.
//
// Parameters:
//   - mdl: the Model to create an Animator for
//   - computeShader: the compute shader for the animator's compute pipeline
//   - vertexShader: the vertex shader for the render pipeline
//   - fragmentShader: the fragment shader for the render pipeline
//   - pipelineOpts: optional pipeline builder options for the render pipeline
//
// Returns:
//   - animator.Animator: the fully initialized Animator
func (s *scene) createAnimator(mdl model.Model, computeShader, vertexShader, fragmentShader shader.Shader, pipelineOpts ...pipeline.PipelineBuilderOption) animator.Animator {
	// Pick backend type based on whether the model uses skeletal animation
	backendType := animator.BackendTypeSimple
	if mdl.Skinned() {
		backendType = animator.BackendTypeSkeletal
	}

	// Discover binding indices for the skeletal animator's bone and packed animation buffers.
	// boneBinding targets the BoneInfo declaration (receives bone data via SetBone/Flush).
	// packedBinding targets the raw "anim_packed" buffer (receives clip/channel/keyframe data via AddClip).
	// For simple animators these default to 0 and are unused.
	boneBinding := 0
	packedBinding := 0
	if backendType == animator.BackendTypeSkeletal {
		for _, decl := range computeShader.Declarations() {
			if decl.Binding == nil {
				continue
			}
			switch decl.Type {
			case shader.AnnotationTypeBindingGroup:
				typeArg := string(decl.Args[2])
				if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
					typeArg = strings.TrimSuffix(stripped, ">")
				}
				if shader.AnnotationArg(typeArg) == shader.AnnotationArgBoneInfo {
					boneBinding = *decl.Binding
				}
			case shader.AnnotationTypeProvider:
				if decl.Args[0] == shader.AnnotationArgAnimatorPacked {
					packedBinding = *decl.Binding
				}
			}
		}
	}

	anim := animator.NewAnimator(backendType, animator.WithModel(mdl, boneBinding, packedBinding))
	anim.SetBoundingRadius(mdl.BoundingRadius())

	// Init mesh provider GPU resources if not already done (e.g. hand-built models
	// skip this, while loader-produced models will already have VertexBuffer set).
	if meshBGP := mdl.MeshProvider(); meshBGP != nil && meshBGP.VertexBuffer() == nil {
		if err := s.r.InitMeshBuffers(meshBGP, mdl.VertexData(), mdl.IndexData(), mdl.IndexCount()); err != nil {
			panic(fmt.Sprintf("scene: failed to init mesh BGP for model %q: %v", mdl.Name(), err))
		}
	}

	// Identify the compute group from the compute shader's declarations.
	// The animation data binding (simple or skeletal) identifies the correct group.
	computeGroup := 0
	for _, decl := range computeShader.Declarations() {
		if decl.Type != shader.AnnotationTypeBindingGroup || decl.Group == nil {
			continue
		}
		typeArg := string(decl.Args[2])
		if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
			typeArg = strings.TrimSuffix(stripped, ">")
		}
		switch shader.AnnotationArg(typeArg) {
		case shader.AnnotationArgAnimationData, shader.AnnotationArgSkeletalAnimationData:
			computeGroup = *decl.Group
		}
	}

	// Identify the output group in the vertex shader (contains the instance buffer
	// that the compute shader's output feeds into).
	// For static models this is an @oxy:group with AnnotationArgInstanceData.
	// For skinned models this is an @oxy:provider with AnnotationArgAnimator (raw vec4 buffer).
	outputGroup := 0
	outputInstanceBinding := 0
	for _, decl := range vertexShader.Declarations() {
		if decl.Group == nil {
			continue
		}
		switch decl.Type {
		case shader.AnnotationTypeBindingGroup:
			typeArg := string(decl.Args[2])
			if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
				typeArg = strings.TrimSuffix(stripped, ">")
			}
			if shader.AnnotationArg(typeArg) == shader.AnnotationArgInstanceData {
				outputGroup = *decl.Group
				if decl.Binding != nil {
					outputInstanceBinding = *decl.Binding
				}
			}
		case shader.AnnotationTypeProvider:
			if decl.Args[0] == shader.AnnotationArgAnimator {
				outputGroup = *decl.Group
				// Provider identifies the group; the raw instance binding is always 0.
				outputInstanceBinding = 0
			}
		}
	}

	// Derive the per-instance output size from the vertex shader's instance buffer.
	// The compute shader's output buffer and the vertex shader's instance buffer are
	// backed by the same physical GPU buffer, so the per-instance stride must match.
	outputDesc := vertexShader.BindGroupLayoutDescriptor(outputGroup)
	perInstanceOutputSize := uint64(64) // fallback: mat4x4<f32>
	for _, entry := range outputDesc.Entries {
		if int(entry.Binding) == outputInstanceBinding && entry.Buffer.MinBindingSize > 0 {
			perInstanceOutputSize = entry.Buffer.MinBindingSize
			break
		}
	}

	// For skeletal animators the output stride is NOT the array element size (vec4 = 16 bytes)
	// but the full per-instance payload: 1 model matrix + MAX_BONES bone matrices, each mat4x4.
	// The WGSL parser returns the element stride for runtime-sized arrays (array<vec4<f32>> → 16),
	// which must be scaled up to the actual per-instance stride that both the compute and vertex
	// shaders use (FLOATS_PER_INSTANCE × sizeof(vec4) = (1 + MAX_BONES) × 64 bytes).
	if backendType == animator.BackendTypeSkeletal {
		perInstanceOutputSize = (1 + s.maxBonesGPU) * 64
	}

	// Compute skeletal-specific sizing context (bone count, packed buffer size).
	var boneCount uint64
	var packedBufferSize uint64
	if backendType == animator.BackendTypeSkeletal && mdl.Skinned() && mdl.Skeleton() != nil {
		boneCount = uint64(len(mdl.Skeleton().Bones))

		// Compute packed animation buffer size from model data.
		// Packed layout: [clips × 4 u32] [channels × 8 u32] [keyframes × 16 u32]
		totalClips := 0
		totalChannels := 0
		totalKeyframes := 0
		for _, clip := range mdl.Animations() {
			totalClips++
			for _, ch := range clip.Channels {
				totalChannels++
				totalKeyframes += len(ch.PositionKeys) + len(ch.RotationKeys) + len(ch.ScaleKeys)
			}
		}
		totalU32s := totalClips*4 + totalChannels*8 + totalKeyframes*16
		packedBufferSize = uint64(totalU32s) * 4
		if packedBufferSize < 4 {
			packedBufferSize = 4
		}
	}

	// Build buffer size and usage overrides for the compute group.
	// Simple animators: all storage buffers are per-instance (maxInst × element stride).
	// Skeletal animators: bone and packed data are shared (not per-instance), scratch needs
	// extra capacity for blending (2 slots per instance × boneCount matrices).
	maxInst := uint64(anim.MaxInstances())
	computeDesc := computeShader.BindGroupLayoutDescriptor(computeGroup)
	computeSizeOverrides := make(map[int]uint64)
	computeUsageOverrides := make(map[int]wgpu.BufferUsage)

	// Build a binding→type map from the compute shader's declarations for typed bindings.
	computeBindingTypes := make(map[int]shader.AnnotationArg)
	for _, decl := range computeShader.Declarations() {
		if decl.Type != shader.AnnotationTypeBindingGroup || decl.Binding == nil {
			continue
		}
		typeArg := string(decl.Args[2])
		if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
			typeArg = strings.TrimSuffix(stripped, ">")
		}
		computeBindingTypes[*decl.Binding] = shader.AnnotationArg(typeArg)
	}

	// Resolve raw bindings from provider declarations for output, packed, and scratch buffers.
	rawOutputBinding, rawPackedBinding, rawScratchBinding := -1, -1, -1
	for _, decl := range computeShader.Declarations() {
		if decl.Type != shader.AnnotationTypeProvider || decl.Binding == nil {
			continue
		}
		switch decl.Args[0] {
		case shader.AnnotationArgAnimatorOutput:
			rawOutputBinding = *decl.Binding
		case shader.AnnotationArgAnimatorPacked:
			rawPackedBinding = *decl.Binding
		case shader.AnnotationArgAnimatorScratch:
			rawScratchBinding = *decl.Binding
		}
	}

	computeOutputBinding := -1
	for _, entry := range computeDesc.Entries {
		binding := int(entry.Binding)

		// Check annotated bindings first.
		if typeArg, ok := computeBindingTypes[binding]; ok {
			switch typeArg {
			case shader.AnnotationArgIndirectArgs:
				// Indirect args buffer needs the Indirect usage flag for DrawIndexedIndirect.
				computeUsageOverrides[binding] = wgpu.BufferUsageIndirect
			case shader.AnnotationArgBoneInfo:
				// Shared bone info buffer: one entry per bone, not per-instance.
				if entry.Buffer.MinBindingSize > 0 {
					computeSizeOverrides[binding] = boneCount * entry.Buffer.MinBindingSize
				}
			case shader.AnnotationArgModelData:
				// Per-instance model matrices from CPU.
				if entry.Buffer.MinBindingSize > 0 {
					computeSizeOverrides[binding] = maxInst * entry.Buffer.MinBindingSize
				}
			case shader.AnnotationArgAnimationGlobals, shader.AnnotationArgGlobalData:
				// Uniform buffer — fixed size from the parser, no override needed.
			default:
				// Per-instance storage buffers (animation data, skeletal animation data, etc.).
				if (entry.Buffer.Type == wgpu.BufferBindingTypeStorage || entry.Buffer.Type == wgpu.BufferBindingTypeReadOnlyStorage) &&
					entry.Buffer.MinBindingSize > 0 {
					computeSizeOverrides[binding] = maxInst * entry.Buffer.MinBindingSize
				}
			}
			continue
		}

		// Handle raw (un-annotated) bindings by resolved var name.
		switch binding {
		case rawOutputBinding:
			// Output buffer stores per-instance data that the vertex shader reads.
			computeSizeOverrides[binding] = maxInst * perInstanceOutputSize
			computeOutputBinding = binding
		case rawPackedBinding:
			// Packed animation data buffer: clips, channels, keyframes packed as u32 array.
			computeSizeOverrides[binding] = packedBufferSize
		case rawScratchBinding:
			// Scratch bone matrix workspace: 2 slots per instance (for blending) × boneCount × mat4x4.
			computeSizeOverrides[binding] = maxInst * boneCount * 2 * 64
		}
	}

	if err := s.r.InitBindGroup(anim.ComputeBindGroupProvider(), computeDesc, computeUsageOverrides, computeSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init compute BGP for model %q: %v", mdl.Name(), err))
	}

	// Share the compute output buffer with the vertex shader's output BGP.
	// The compute shader writes per-instance data into the output buffer through the compute BGP,
	// and the vertex shader reads it from instance_buffer through the output BGP.
	// These must reference the same physical GPU buffer.
	if computeOutputBinding >= 0 {
		if sharedBuf := anim.ComputeBindGroupProvider().Buffer(computeOutputBinding); sharedBuf != nil {
			anim.OutputBindGroupProvider().SetBuffer(outputInstanceBinding, sharedBuf)
		}
	}

	// Build buffer size overrides for the output group (vertex shader instance buffer).
	// The shared buffer is already set on the output BGP for the instance binding,
	// so InitBindGroup will reuse it rather than creating a new buffer.
	outputSizeOverrides := make(map[int]uint64)
	for _, entry := range outputDesc.Entries {
		if int(entry.Binding) == outputInstanceBinding &&
			(entry.Buffer.Type == wgpu.BufferBindingTypeStorage || entry.Buffer.Type == wgpu.BufferBindingTypeReadOnlyStorage) &&
			entry.Buffer.MinBindingSize > 0 {
			outputSizeOverrides[int(entry.Binding)] = maxInst * perInstanceOutputSize
		}
	}

	if err := s.r.InitBindGroup(anim.OutputBindGroupProvider(), outputDesc, nil, outputSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init output BGP for model %q: %v", mdl.Name(), err))
	}

	// Register compute pipeline
	cp := pipeline.NewPipeline(computeShader.Key(), pipeline.PipelineTypeCompute, pipeline.WithComputeShader(computeShader))
	if err := s.r.RegisterPipelines(cp); err != nil {
		panic(fmt.Sprintf("scene: failed to register compute pipeline for model %q: %v", mdl.Name(), err))
	}
	anim.Model().SetComputePipelineKey(cp.PipelineKey())

	// Register render pipeline with the model name as key, matching Material.PipelineKey()
	renderOpts := append([]pipeline.PipelineBuilderOption{
		pipeline.WithVertexShader(vertexShader),
		pipeline.WithFragmentShader(fragmentShader),
	}, pipelineOpts...)
	rp := pipeline.NewPipeline(mdl.Name(), pipeline.PipelineTypeRender, renderOpts...)
	if err := s.r.RegisterPipelines(rp); err != nil {
		panic(fmt.Sprintf("scene: failed to register render pipeline for model %q: %v", mdl.Name(), err))
	}

	// Init material GPU resources (textures, samplers, bind groups) for each material
	// that doesn't already have a bind group provider set. For materials with a non-empty
	// PipelineKey, a per-material pipeline is registered (or reused if already present).
	for i, mat := range mdl.RenderMaterials() {
		if mat.BindGroupProvider() != nil {
			continue
		}
		// Register or reuse the per-material pipeline.
		matPipelineKey := mat.PipelineKey()
		if matPipelineKey != "" && s.r.Pipeline(matPipelineKey) == nil {
			var matPipeOpts []pipeline.PipelineBuilderOption
			// Check if the material provides explicit pipeline options (shaders, blend, cull, etc.)
			if rawOpts := mat.PipelineOptions(); len(rawOpts) > 0 {
				typedOpts := make([]pipeline.PipelineBuilderOption, 0, len(rawOpts))
				for _, raw := range rawOpts {
					if opt, ok := raw.(pipeline.PipelineBuilderOption); ok {
						typedOpts = append(typedOpts, opt)
					}
				}
				// Prepend the model's standard vertex+fragment shaders so material opts
				// can override them via explicit WithVertexShader/WithFragmentShader entries.
				matPipeOpts = append([]pipeline.PipelineBuilderOption{
					pipeline.WithVertexShader(vertexShader),
					pipeline.WithFragmentShader(fragmentShader),
				}, typedOpts...)
			} else {
				// No material-level options: use the model's standard vertex+fragment shaders.
				matPipeOpts = []pipeline.PipelineBuilderOption{
					pipeline.WithVertexShader(vertexShader),
					pipeline.WithFragmentShader(fragmentShader),
				}
			}
			mp := pipeline.NewPipeline(matPipelineKey, pipeline.PipelineTypeRender, matPipeOpts...)
			if err := s.r.RegisterPipelines(mp); err != nil {
				panic(fmt.Sprintf("scene: failed to register pipeline %q for model %q material %d: %v", matPipelineKey, mdl.Name(), i, err))
			}
		}
		// Determine which fragment shader to use for GPU resource init.
		var fragShaderForMat shader.Shader
		if matPipelineKey != "" {
			if p := s.r.Pipeline(matPipelineKey); p != nil {
				fragShaderForMat = p.Shader(shader.ShaderTypeFragment)
			}
		}
		if fragShaderForMat == nil {
			fragShaderForMat = fragmentShader
		}
		providerName := fmt.Sprintf("%s_material_%d", mdl.Name(), i)
		if err := s.initMaterialGPU(mat, fragShaderForMat, providerName); err != nil {
			panic(fmt.Sprintf("scene: failed to init material GPU for model %q material %d: %v", mdl.Name(), i, err))
		}
	}

	return anim
}

// computeWorkgroupSize2D returns the workgroup size for a 2D compute dispatch.
//
// Falls back to the provided defaults if the pipeline or shader is not available.
func (s *scene) computeWorkgroupSize2D(pipeKey string, defaultX, defaultY uint32) (uint32, uint32) {
	pipe := s.r.Pipeline(pipeKey)
	if pipe == nil {
		return defaultX, defaultY
	}
	shdr := pipe.Shader(shader.ShaderTypeCompute)
	if shdr == nil {
		return defaultX, defaultY
	}
	wgSize := shdr.WorkgroupSize()
	x := wgSize[0]
	y := wgSize[1]
	if x == 0 {
		x = defaultX
	}
	if y == 0 {
		y = defaultY
	}
	return x, y
}

// releaseResolutionDependentResources releases old GPU textures and bind groups
// before resize re-initialization.
func (s *scene) releaseResolutionDependentResources() {
	gbh := s.lightHandler.GBufferHandler()
	if gbh.Enabled() {
		if v := gbh.NormalTextureView(); v != nil {
			v.Release()
		}
		if t := gbh.NormalTexture(); t != nil {
			t.Release()
		}
		gbh.SetNormalTextureView(nil)
		gbh.SetNormalTexture(nil)
		if v := gbh.AlbedoTextureView(); v != nil {
			v.Release()
		}
		if t := gbh.AlbedoTexture(); t != nil {
			t.Release()
		}
		gbh.SetAlbedoTextureView(nil)
		gbh.SetAlbedoTexture(nil)
		if v := gbh.DepthTextureView(); v != nil {
			v.Release()
		}
		if t := gbh.DepthTexture(); t != nil {
			t.Release()
		}
		gbh.SetDepthTextureView(nil)
		gbh.SetDepthTexture(nil)
	}

	ssaoH := s.lightHandler.SSAOHandler()
	if ssaoH.Enabled() {
		if v := ssaoH.RawTextureView(); v != nil {
			v.Release()
		}
		if t := ssaoH.RawTexture(); t != nil {
			t.Release()
		}
		ssaoH.SetRawTextureView(nil)
		ssaoH.SetRawTexture(nil)
		if v := ssaoH.BlurredTextureView(); v != nil {
			v.Release()
		}
		if t := ssaoH.BlurredTexture(); t != nil {
			t.Release()
		}
		ssaoH.SetBlurredTextureView(nil)
		ssaoH.SetBlurredTexture(nil)
		if v := ssaoH.ScratchTextureView(); v != nil {
			v.Release()
		}
		if t := ssaoH.ScratchTexture(); t != nil {
			t.Release()
		}
		ssaoH.SetScratchTextureView(nil)
		ssaoH.SetScratchTexture(nil)
		for _, key := range []string{"ssao_compute", "ssao_blur_h", "ssao_blur_v"} {
			if bgp := ssaoH.Bgp(key); bgp != nil {
				if bg := bgp.BindGroup(); bg != nil {
					bg.Release()
				}
				bgp.SetBindGroup(nil)
			}
		}
	}

	if bgp := s.lightHandler.Bgp("ssao_lit"); bgp != nil {
		if bg := bgp.BindGroup(); bg != nil {
			bg.Release()
		}
		bgp.SetBindGroup(nil)
	}

	csh := s.lightHandler.ContactShadowHandler()
	if csh.Enabled() {
		if v := csh.TextureView(); v != nil {
			v.Release()
		}
		if t := csh.Texture(); t != nil {
			t.Release()
		}
		csh.SetTextureView(nil)
		csh.SetTexture(nil)
		if bgp := csh.Bgp("contact_shadow_compute"); bgp != nil {
			if bg := bgp.BindGroup(); bg != nil {
				bg.Release()
			}
			bgp.SetBindGroup(nil)
		}
	}

	sh := s.lightHandler.ShadowHandler()
	if sh.CSMAtlasTexture() != nil {
		if bgp := sh.Bgp("csm_shadow_lit"); bgp != nil {
			if bg := bgp.BindGroup(); bg != nil {
				bg.Release()
			}
			bgp.SetBindGroup(nil)
		}
	}

	ch := s.lightHandler.CompositionHandler()
	if ch.Enabled() {
		if v := ch.HDRTextureView(); v != nil {
			v.Release()
		}
		if t := ch.HDRTexture(); t != nil {
			t.Release()
		}
		ch.SetHDRTextureView(nil)
		ch.SetHDRTexture(nil)
		if v := ch.MSAATextureView(); v != nil {
			v.Release()
		}
		if t := ch.MSAATexture(); t != nil {
			t.Release()
		}
		ch.SetMSAATextureView(nil)
		ch.SetMSAATexture(nil)
		if v := ch.DepthTextureView(); v != nil {
			v.Release()
		}
		if t := ch.DepthTexture(); t != nil {
			t.Release()
		}
		ch.SetDepthTextureView(nil)
		ch.SetDepthTexture(nil)
		if bgp := ch.Bgp("composition"); bgp != nil {
			if bg := bgp.BindGroup(); bg != nil {
				bg.Release()
			}
			bgp.SetBindGroup(nil)
		}
		if bgp := ch.Bgp("luminance_compute"); bgp != nil {
			if bg := bgp.BindGroup(); bg != nil {
				bg.Release()
			}
			bgp.SetBindGroup(nil)
		}

		mipCount := ch.BloomMipCount()
		if ch.BloomDownTexture() != nil {
			ch.BloomDownTexture().Release()
		}
		for _, v := range ch.BloomDownReadViews() {
			if v != nil {
				v.Release()
			}
		}
		for _, v := range ch.BloomDownStorageViews() {
			if v != nil {
				v.Release()
			}
		}
		ch.SetBloomDownTexture(nil)
		ch.SetBloomDownReadViews(nil)
		ch.SetBloomDownStorageViews(nil)
		if ch.BloomUpTexture() != nil {
			ch.BloomUpTexture().Release()
		}
		for _, v := range ch.BloomUpReadViews() {
			if v != nil {
				v.Release()
			}
		}
		for _, v := range ch.BloomUpStorageViews() {
			if v != nil {
				v.Release()
			}
		}
		if ch.BloomUpMip0View() != nil {
			ch.BloomUpMip0View().Release()
		}
		ch.SetBloomUpTexture(nil)
		ch.SetBloomUpReadViews(nil)
		ch.SetBloomUpStorageViews(nil)
		ch.SetBloomUpMip0View(nil)
		for i := 0; i < mipCount; i++ {
			if bgp := ch.Bgp(fmt.Sprintf("bloom_down_%d", i)); bgp != nil {
				bgp.Release()
			}
			if bgp := ch.Bgp(fmt.Sprintf("bloom_up_%d", i)); bgp != nil {
				bgp.Release()
			}
		}
	}

	ssrH := s.lightHandler.SSRHandler()
	if ssrH.Enabled() {
		if v := ssrH.SSRTextureView(); v != nil {
			v.Release()
		}
		if t := ssrH.SSRTexture(); t != nil {
			t.Release()
		}
		ssrH.SetSSRTextureView(nil)
		ssrH.SetSSRTexture(nil)
		if bgp := ssrH.Bgp("ssr_compute"); bgp != nil {
			if bg := bgp.BindGroup(); bg != nil {
				bg.Release()
			}
			bgp.SetBindGroup(nil)
		}
		hizMipCount := ssrH.HiZMipCount()
		if v := ssrH.HiZTextureView(); v != nil {
			v.Release()
		}
		if t := ssrH.HiZTexture(); t != nil {
			t.Release()
		}
		for _, v := range ssrH.HiZMipReadViews() {
			if v != nil {
				v.Release()
			}
		}
		for _, v := range ssrH.HiZStorageViews() {
			if v != nil {
				v.Release()
			}
		}
		ssrH.SetHiZTextureView(nil)
		ssrH.SetHiZTexture(nil)
		ssrH.SetHiZMipReadViews(nil)
		ssrH.SetHiZStorageViews(nil)
		if bgp := ssrH.Bgp("hiz_init"); bgp != nil {
			bgp.Release()
		}
		for i := 1; i < hizMipCount; i++ {
			if bgp := ssrH.Bgp(fmt.Sprintf("hiz_down_%d", i)); bgp != nil {
				bgp.Release()
			}
		}
	}
}

// resizePostProcessing re-creates resolution-dependent GPU resources (textures
// and bind groups) after a window resize. Pipeline registrations auto-skip
// (RegisterPipelines checks pipelineCache for existing keys). InitBindGroup
// reuses existing layouts and buffers.
//
// Parameters:
//   - w: new width in pixels
//   - h: new height in pixels
func (s *scene) resizePostProcessing(w, h int) {
	if !s.postProcessingInitialized {
		return
	}

	s.r.WaitIdle()
	s.releaseResolutionDependentResources()

	if s.lightHandler.GBufferHandler().Enabled() {
		s.initGBuffer()
	}
	if s.lightHandler.SSAOHandler().Enabled() {
		s.initSSAO()
	}
	if s.lightHandler.ContactShadowHandler().Enabled() {
		s.initContactShadows()
	}

	litFragShader := shader.NewShader("_lit_frag_resize", shader.ShaderTypeFragment,
		"engine/light/assets/lit-frag-csm.wgsl", shader.WithInjections(s.injections))

	s.initSSAOLitBindGroup(litFragShader)

	if s.lightHandler.ShadowHandler().CSMAtlasTexture() != nil {
		s.initCSMShadowLitBindGroup(litFragShader)
	}

	if s.lightHandler.CompositionHandler().Enabled() {
		s.initComposition()
	}
	if s.lightHandler.SSRHandler().Enabled() {
		s.initSSR()
	}

	if s.lightHandler.SSRHandler().Enabled() && s.lightHandler.CompositionHandler().Enabled() {
		compBGP := s.lightHandler.CompositionHandler().Bgp("composition")
		if compBGP != nil && s.lightHandler.SSRHandler().SSRTextureView() != nil {
			compBGP.SetTextureView(2, s.lightHandler.SSRHandler().SSRTextureView())
			compFrag := shader.NewShader("_composition_frag_rebind", shader.ShaderTypeFragment,
				"engine/light/assets/composition-frag.wgsl", shader.WithInjections(s.injections))
			compDesc := compFrag.BindGroupLayoutDescriptor(0)
			sizeOverrides := map[int]uint64{
				4: uint64((&light.GPUCompositionParams{}).Size()),
			}
			if err := s.r.InitBindGroup(compBGP, compDesc, nil, sizeOverrides); err != nil {
				panic(fmt.Sprintf("scene: failed to re-init composition bind group on resize: %v", err))
			}
		}
	}
}
