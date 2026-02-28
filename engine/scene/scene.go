package scene

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

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
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/cogentcore/webgpu/wgpu"
)

// physicsSyncGroup tracks one Animator's physics sync dispatch state. Each unique
// Animator that has rigid body objects gets its own group with a dedicated sync_map
// buffer (initialized to sentinel 0xFFFFFFFF for non-member bodies) and an
// AnimationData reference bound at binding 2.
type physicsSyncGroup struct {
	groupID uint32
	bgp     bind_group_provider.BindGroupProvider
}

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
	common.DelegateImpl[Scene]

	mu *sync.RWMutex

	name   string
	active bool

	animatorPool map[model.Model][]animator.Animator
	registry     map[uint64]game_object.GameObject // non-ephemeral objects by ID
	nextID       uint64

	physicsHandler  physics.Physics
	physicsGPUReady bool // true once initPhysicsGPU has run

	// Per-animator sync dispatch state. Each unique Animator that has physics
	// bodies gets its own physicsSyncGroup with a dedicated sync_map buffer and
	// AnimationData reference. This allows the sync shader to write each body's
	// transform to the correct Animator's AnimationData slot without cross-group
	// interference (bodies not belonging to a group are masked by a sentinel).
	physicsSyncGroups  []*physicsSyncGroup
	physicsSyncAnimMap map[animator.Animator]int         // animator → index in physicsSyncGroups
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

	// Pre-allocated slices reused each frame to avoid per-frame allocations.
	writePool          []bind_group_provider.BufferWrite       // reusable coalesced buffer write slice
	drawBindGroupsPool []bind_group_provider.BindGroupProvider // reusable bind group slice for DrawCalls

	// computePool manages a bounded set of reusable goroutines for the parallel
	// CPU prep phase of PrepareCompute. Workers persist across frames, avoiding
	// per-frame goroutine spawn/teardown overhead.
	computePool    worker.DynamicWorkerPool
	computeWorkers int // stored so we can log/inspect the configured count

	// instanceLookup provides O(1) reverse lookup from (Animator, instanceSlot) → objID.
	// Maintained by Add/Remove so the swap-remove fixup in Remove avoids an O(N) registry scan.
	instanceLookup map[animator.Animator]map[uint32]uint64
}

// Scene manages a collection of Animators (registered implicitly via Add) and an
// optional registry of non-ephemeral GameObjects, with a Camera and Renderer for
// rendering. Rendering is driven entirely by the registered Animator list — each
// Animator owns its instance data and material.
// Scenes can be hot-swapped via the Active flag to switch between different views or levels.
// Thread-safe for concurrent access.
type Scene interface {
	common.Delegate[Scene]

	// Name returns the scene's identifier.
	Name() string

	// SetName sets the scene's identifier.
	SetName(name string)

	// Active returns whether this scene is currently active for rendering.
	Active() bool

	// SetActive sets whether this scene is active for rendering.
	SetActive(active bool)

	// Camera returns the scene's camera.
	Camera() camera.Camera

	// SetCamera replaces the scene's camera.
	//
	// Parameters:
	//   - cam: the new camera
	SetCamera(cam camera.Camera)

	// Renderer returns the scene's renderer.
	Renderer() renderer.Renderer

	// SetRenderer replaces the scene's renderer.
	//
	// Parameters:
	//   - r: the new renderer
	SetRenderer(r renderer.Renderer)

	// SetPhysicsHandler replaces the scene's physics handler. This should be called
	// before adding any rigid body objects. If not set, a default handler is created
	// lazily when the first rigid body object is added.
	//
	// Parameters:
	//   - ph: the pre-configured Physics instance
	SetPhysicsHandler(ph physics.Physics)

	// Count returns the number of persisted GameObjects in the scene's registry. Does not include ephemeral objects.
	//
	// Returns:
	//   - int: count of non-ephemeral GameObjects in the registry
	Count() int

	// CountEphemeral returns the number of ephemeral GameObjects currently being rendered through the scene's animators.
	//
	// Returns:
	//   - int: count of ephemeral GameObjects currently rendered
	CountEphemeral() int

	// Add adds a GameObject to the scene. The scene's Renderer must be attached
	// and the object must carry a Model. The scene automatically creates and manages
	// an Animator for each unique Model, registers its compute and render pipelines,
	// initializes GPU resources, and adds a new instance wired with the object's
	// initial transform data. If the object is not ephemeral it is also persisted
	// in the registry for later lookup or removal by ID.
	//
	// Compute, vertex, and fragment shaders are resolved automatically from the
	// engine's standard shader assets based on whether the model is skinned.
	//
	// Panics if the scene has no Renderer or the object has no Model.
	//
	// Parameters:
	//   - obj: the GameObject to add
	//   - pipelineOpts: optional pipeline builder options for the render pipeline (e.g., blending)
	//
	// Returns:
	//   - uint64: the assigned object ID
	Add(obj game_object.GameObject, pipelineOpts ...pipeline.PipelineBuilderOption) uint64

	// Get retrieves a non-ephemeral GameObject by its ID.
	// Returns nil if not found.
	//
	// Parameters:
	//   - id: the object's unique ID
	//
	// Returns:
	//   - game_object.GameObject: the object or nil
	Get(id uint64) game_object.GameObject

	// Remove removes a non-ephemeral GameObject from the registry by ID
	// and swap-removes the instance data from its animator.
	//
	// Parameters:
	//   - id: the object's unique ID
	Remove(id uint64)

	// PrepareCompute updates camera matrices, advances animation state,
	// uploads staged buffer writes, and dispatches all compute shaders for this scene.
	// Must be called within a BeginComputeFrame/EndComputeFrame block on the renderer.
	//
	// Parameters:
	//   - deltaTime: elapsed time since the last frame in seconds
	PrepareCompute(deltaTime float32)

	// CullingDisabled returns whether GPU frustum culling is explicitly disabled for this scene.
	// When true, the scene will not distribute frustum planes to animators, keeping them in
	// non-culled mode even when a camera is present.
	//
	// Returns:
	//   - bool: true if culling is disabled
	CullingDisabled() bool

	// SetCullingDisabled enables or disables GPU frustum culling for this scene.
	// When set to true, the scene skips frustum plane distribution and animators
	// fall back to non-culled rendering with regular draw calls.
	//
	// Parameters:
	//   - disabled: true to disable culling, false to enable it
	SetCullingDisabled(disabled bool)

	// DrawCalls issues instanced draw calls for each registered animator.
	// Must be called within a BeginFrame/EndFrame block on the renderer.
	//
	// Returns:
	//   - error: error if a draw call fails
	DrawCalls() error

	// AddLight adds a light source to the scene and lazily initializes the full
	// lighting pipeline (light storage buffer, shadow map, Forward+ culling) on
	// the first call. Subsequent calls simply append the light without
	// re-initializing GPU resources. The lighting shaders are loaded internally
	// from the engine's standard light shader assets. Screen dimensions for
	// Forward+ tile culling are taken from the scene's stored screen size
	// (set via Resize or WithScreenSize).
	//
	// Parameters:
	//   - l: the Light to add
	AddLight(l light.Light)

	// Resize updates the scene's stored screen dimensions and propagates the
	// change to the renderer surface, camera aspect ratio, and (when lighting
	// is enabled) the Forward+ tile grid. Call this from the window's resize
	// callback or whenever the surface dimensions change.
	//
	// Parameters:
	//   - width: the new width in pixels
	//   - height: the new height in pixels
	Resize(width, height int)

	// RemoveLight removes a light source from the scene by reference.
	//
	// Parameters:
	//   - l: the Light to remove
	RemoveLight(l light.Light)

	// DetachLight removes a game object's attached light from the scene's tracking
	// and light lists. This is the cleanup counterpart for objects whose lights
	// were auto-registered during Add(). Non-ephemeral objects are cleaned up
	// automatically via Remove(), but ephemeral object owners must call this
	// explicitly when the object's lifetime ends.
	//
	// Parameters:
	//   - obj: the GameObject whose attached light should be detached
	DetachLight(obj game_object.GameObject)

	// Lights returns all lights currently registered in the scene.
	//
	// Returns:
	//   - []light.Light: the scene's light list
	Lights() []light.Light

	// AmbientColor returns the scene's ambient light color.
	//
	// Returns:
	//   - [3]float32: the ambient RGB color
	AmbientColor() [3]float32

	// SetAmbientColor sets the scene's ambient light color.
	//
	// Parameters:
	//   - color: the ambient RGB color
	SetAmbientColor(color [3]float32)

	// PrepareShadows computes the directional light's view-projection, updates the
	// shadow uniform buffer, and renders the depth-only shadow pass for all drawables.
	// Must be called after PrepareCompute and before BeginFrame each frame.
	// No-ops if no shadow map has been initialized or no shadow-casting directional
	// light exists.
	PrepareShadows()

	// PrepareLightCulling updates the light cull uniform buffer and dispatches the
	// light culling compute shader. Must be called after PrepareCompute (so lights
	// are uploaded) and before DrawCalls.
	PrepareLightCulling()
}

// Ensure scene implements Scene interface.
var _ Scene = &scene{}

// NewScene creates a new Scene with the given camera and renderer. Both are
// required and NewScene panics if either is nil. The camera's bind group layout
// is resolved from the pre-processor declarations of the engine's standard
// vertex shader (engine/model/assets/simple-vert.wgsl).
//
// Parameters:
//   - name: the name of the scene
//   - cam: the camera to attach (must not be nil)
//   - r: the renderer to attach (must not be nil)
//   - options: functional options to further configure the scene
//
// Returns:
//   - Scene: the newly created scene
func NewScene(name string, cam camera.Camera, r renderer.Renderer, options ...SceneBuilderOption) Scene {
	if cam == nil {
		panic("scene: NewScene requires a non-nil Camera")
	}
	if r == nil {
		panic("scene: NewScene requires a non-nil Renderer")
	}

	s := &scene{
		mu:                 &sync.RWMutex{},
		name:               name,
		active:             false,
		cam:                cam,
		r:                  r,
		animatorPool:       make(map[model.Model][]animator.Animator),
		registry:           make(map[uint64]game_object.GameObject),
		instanceLookup:     make(map[animator.Animator]map[uint32]uint64),
		nextID:             1,
		computeWorkers:     max(runtime.NumCPU()-1, 1),
		drawBindGroupsPool: make([]bind_group_provider.BindGroupProvider, 0, 3),
		lightHandler:       light.NewLightingHandler(),
		physicsAnimBinding: -1,
	}

	for _, option := range options {
		option(s)
	}

	// Initialize the compute pool after options so WithComputeWorkers can override the default.
	// Queue size of 256 accommodates typical animator group counts with headroom.
	s.computePool = worker.NewDynamicWorkerPool(s.computeWorkers, 256, 1*time.Second)

	// Initialize the camera's bind group on the GPU using the layout from the
	// engine's standard vertex shader. The shader is loaded internally so the
	// caller never needs to supply one. The camera group index is resolved from
	// the shader's pre-processor declarations rather than fuzzy var-name matching.
	cameraVertShader := shader.NewShader("_camera_init_vert", shader.ShaderTypeVertex, "engine/model/assets/simple-vert.wgsl")
	cameraGroup := 0
	for _, decl := range cameraVertShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgCamera {
			cameraGroup = *decl.Group
			break
		}
	}
	if bgp := cam.BindGroupProvider(); bgp != nil {
		if err := r.InitBindGroup(bgp, cameraVertShader.BindGroupLayoutDescriptor(cameraGroup), nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init camera bind group: %v", err))
		}
	}

	s.Delegate = s
	return s
}

func (s *scene) Name() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.name
}

func (s *scene) SetName(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.name = name
}

func (s *scene) Active() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.active
}

func (s *scene) SetActive(active bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.active = active
}

func (s *scene) Camera() camera.Camera {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cam
}

func (s *scene) SetCamera(cam camera.Camera) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cam = cam
}

func (s *scene) Renderer() renderer.Renderer {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.r
}

func (s *scene) SetRenderer(r renderer.Renderer) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.r = r
}

func (s *scene) SetPhysicsHandler(ph physics.Physics) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.physicsHandler = ph
}

func (s *scene) CullingDisabled() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cullingDisabled
}

func (s *scene) SetCullingDisabled(disabled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cullingDisabled = disabled
}

func (s *scene) AddLight(l light.Light) {
	if !s.lightHandler.Enabled() {
		s.initLighting(s.screenWidth, s.screenHeight)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.lightHandler.AddLight(l)
}

func (s *scene) Resize(width, height int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.screenWidth = width
	s.screenHeight = height

	if s.r != nil {
		s.r.Resize(width, height)
	}
	if s.cam != nil && height > 0 {
		s.cam.SetAspect(float32(width) / float32(height))
	}
	if s.lightHandler.Enabled() {
		s.lightHandler.Resize(width, height)
	}
}

func (s *scene) RemoveLight(l light.Light) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lightHandler.RemoveLight(l)
}

func (s *scene) DetachLight(obj game_object.GameObject) {
	s.mu.Lock()
	defer s.mu.Unlock()
	l := obj.Light()
	if l == nil {
		return
	}
	s.lightHandler.RemoveLight(l)
	for i, o := range s.lightObjects {
		if o == obj {
			s.lightObjects = append(s.lightObjects[:i], s.lightObjects[i+1:]...)
			break
		}
	}
}

func (s *scene) Lights() []light.Light {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lightHandler.Lights()
}

func (s *scene) AmbientColor() [3]float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lightHandler.AmbientColor()
}

func (s *scene) SetAmbientColor(color [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lightHandler.SetAmbientColor(color)
}

// initLightBindGroup initializes the GPU resources for the light storage buffer
// using the layout descriptor from the given fragment shader's light group.
func (s *scene) initLightBindGroup(fragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || fragmentShader == nil {
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
			sizeOverrides[binding] = uint64(light.MaxGPULights * (&light.GPULight{}).Size())
		}
	}

	if err := s.r.InitBindGroup(bgp, descriptor, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init light bind group: %v", err))
	}
}

// initShadowMap initializes the shadow mapping resources for the scene.
func (s *scene) initShadowMap(shadowVertexShader, shadowSkinnedVertexShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || shadowVertexShader == nil {
		return
	}

	// Create shadow depth texture.
	res := s.lightHandler.ShadowMapResolution()
	view, tex, err := s.r.CreateShadowDepthTexture(res, res)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create shadow depth texture: %v", err))
	}
	s.lightHandler.SetShadowDepthTexture(tex)
	s.lightHandler.SetShadowDepthTextureView(view)

	// Create comparison sampler for PCF in the lit fragment shader.
	samp, err := s.r.CreateComparisonSampler()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create comparison sampler: %v", err))
	}
	s.lightHandler.SetShadowComparisonSampler(samp)

	// Create shadow data BGP — holds the light VP matrix + texel size + bias.
	// The layout is derived from the shadow vertex shader's group containing the
	// ShadowUniform struct, resolved via pre-processor declarations.
	shadowGroup := 0
	for _, decl := range shadowVertexShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgShadowUniform {
			shadowGroup = *decl.Group
			break
		}
	}
	bgp := s.lightHandler.Bgp("shadow_data")
	desc := shadowVertexShader.BindGroupLayoutDescriptor(shadowGroup)
	sizeOverrides := make(map[int]uint64)
	for _, entry := range desc.Entries {
		if entry.Buffer.Type == wgpu.BufferBindingTypeUniform {
			sizeOverrides[int(entry.Binding)] = uint64((&light.GPUShadowData{}).Size())
		}
	}
	if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init shadow data bind group: %v", err))
	}

	// Register shadow pipelines for each ShadowCullMode variant. Each model
	// can choose its own cull mode via model.WithShadowCullMode, so the scene
	// pre-registers one pipeline per cull mode for both static and skinned shaders.
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
			pipeline.WithVertexShader(shadowVertexShader),
			pipeline.WithDepthBias(2, 1.5),
			pipeline.WithCullMode(cm.wgpu),
		)
		if err := s.r.RegisterShadowPipeline(sp); err != nil {
			panic(fmt.Sprintf("scene: failed to register static shadow pipeline (%s): %v", cm.tag, err))
		}
		s.lightHandler.SetPipelineKey("shadow_static_"+cm.tag, key)
	}

	// Register shadow pipelines for skinned models if a skinned shader is provided.
	if shadowSkinnedVertexShader != nil {
		for _, cm := range cullModes {
			key := "shadow_depth_skinned_" + cm.tag
			ssp := pipeline.NewPipeline(key, pipeline.PipelineTypeRender,
				pipeline.WithVertexShader(shadowSkinnedVertexShader),
				pipeline.WithDepthBias(2, 1.5),
				pipeline.WithCullMode(cm.wgpu),
			)
			if err := s.r.RegisterShadowPipeline(ssp); err != nil {
				panic(fmt.Sprintf("scene: failed to register skinned shadow pipeline (%s): %v", cm.tag, err))
			}
			s.lightHandler.SetPipelineKey("shadow_skinned_"+cm.tag, key)
		}
	}
}

// shadowPipelineKey resolves the shadow depth pipeline key for the given model
// type and cull mode from the lighting handler's pipeline key map.
func (s *scene) shadowPipelineKey(skinned bool, mode model.ShadowCullMode) string {
	prefix := "shadow_static_"
	if skinned && s.lightHandler.PipelineKey("shadow_skinned_back") != "" {
		prefix = "shadow_skinned_"
	}
	tag := "back"
	switch mode {
	case model.ShadowCullModeFront:
		tag = "front"
	case model.ShadowCullModeNone:
		tag = "none"
	}
	return s.lightHandler.PipelineKey(prefix + tag)
}

// initShadowLitBindGroup initializes the bind group provider that lit fragment
// shaders use to sample the shadow map.
func (s *scene) initShadowLitBindGroup(litFragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || litFragmentShader == nil {
		return
	}
	if s.lightHandler.ShadowDepthTextureView() == nil || s.lightHandler.ShadowComparisonSampler() == nil {
		return // InitShadowMap must be called first
	}

	// Resolve the shadow bind group index from the shader's pre-processor
	// declarations by matching the shadow provider identity annotation.
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

	bgp := s.lightHandler.Bgp("shadow_lit")

	// Pre-set the shadow depth texture view and comparison sampler on the BGP
	// so that InitBindGroup can find them when creating the bind group entries.
	desc := litFragmentShader.BindGroupLayoutDescriptor(shadowGroup)
	for _, entry := range desc.Entries {
		binding := int(entry.Binding)
		if entry.Texture.SampleType != wgpu.TextureSampleTypeUndefined {
			bgp.SetTextureView(binding, s.lightHandler.ShadowDepthTextureView())
		}
		if entry.Sampler.Type != wgpu.SamplerBindingTypeUndefined {
			bgp.SetSampler(binding, s.lightHandler.ShadowComparisonSampler())
		}
	}

	// Override the uniform buffer size to 80 bytes (GPUShadowData).
	sizeOverrides := make(map[int]uint64)
	for _, entry := range desc.Entries {
		if entry.Buffer.Type == wgpu.BufferBindingTypeUniform {
			sizeOverrides[int(entry.Binding)] = uint64((&light.GPUShadowData{}).Size())
		}
	}

	if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init shadow lit bind group: %v", err))
	}
}

func (s *scene) PrepareShadows() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.lightHandler.Enabled() || s.r == nil {
		return
	}

	// Find the first enabled, shadow-casting directional light.
	var shadowLight light.Light
	for _, l := range s.lightHandler.Lights() {
		if l.Enabled() && l.CastsShadows() && l.Type() == light.LightTypeDirectional {
			shadowLight = l
			break
		}
	}
	if shadowLight == nil {
		return
	}

	// Compute light VP matrix centered on the camera's look-at target.
	// Using target (not position) keeps the shadow frustum centered on the
	// scene content even when the camera orbits far away.
	centerX, centerY, centerZ := float32(0), float32(0), float32(0)
	if s.cam != nil {
		if ctrl := s.cam.Controller(); ctrl != nil {
			centerX, centerY, centerZ = ctrl.Target()
		}
	}
	// Build and write shadow uniform data.
	texelSize := 1.0 / float32(s.lightHandler.ShadowMapResolution())
	shadowData := light.GPUShadowData{
		TexelSize: [2]float32{texelSize, texelSize},
		Bias:      s.lightHandler.ShadowBias(),
	}
	shadowData.ComputeDirectionalLightVP(
		shadowLight.Direction(),
		centerX, centerY, centerZ,
		s.lightHandler.ShadowHalfExtent(), s.lightHandler.ShadowNear(), s.lightHandler.ShadowFar(),
	)
	shadowData.ComputeNormalBias(s.lightHandler.ShadowHalfExtent(), s.lightHandler.ShadowNormalBiasScale(), s.lightHandler.ShadowMapResolution())
	shadowBytes := shadowData.Marshal()
	shadowDataBGP := s.lightHandler.Bgp("shadow_data")
	writes := []bind_group_provider.BufferWrite{
		{
			Provider: shadowDataBGP,
			Binding:  0,
			Offset:   0,
			Data:     shadowBytes,
		},
	}
	// Also write to the lit-pass shadow BGP if it has a uniform buffer.
	shadowLitBGP := s.lightHandler.Bgp("shadow_lit")
	for binding, buf := range shadowLitBGP.Buffers() {
		if buf != nil {
			writes = append(writes, bind_group_provider.BufferWrite{
				Provider: shadowLitBGP,
				Binding:  binding,
				Offset:   0,
				Data:     shadowBytes,
			})
			break // only one uniform buffer expected
		}
	}
	s.r.WriteBuffers(writes)

	// Execute shadow depth pass.
	if err := s.r.BeginShadowFrame(); err != nil {
		return
	}
	s.r.BeginShadowPass(s.lightHandler.ShadowDepthTextureView())

	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}

			mdl := a.Model()
			if mdl == nil {
				continue
			}
			if !mdl.CastsShadows() {
				continue
			}
			meshProvider := mdl.MeshProvider()
			if meshProvider == nil {
				continue
			}

			// Select the appropriate shadow pipeline based on the model's
			// shadow cull mode (Back/Front/None) and whether it is skinned.
			cullMode := mdl.ShadowCullMode()
			pipeKey := s.shadowPipelineKey(mdl.Skinned(), cullMode)
			if pipeKey == "" {
				continue
			}

			// Build bind groups for the shadow pass:
			//   group(0) = shadow data BGP (light VP uniform)
			//   group(1) = output BGP (instance/bone matrices from compute shader)
			shadowBindGroups := []bind_group_provider.BindGroupProvider{
				shadowDataBGP,
				a.OutputBindGroupProvider(),
			}

			// Use indirect draw when GPU frustum culling is active. The compute
			// shader compacts visible instances into a dense output buffer and
			// writes the visible count to the indirect args buffer. The shadow
			// pass must use the same indirect buffer so instance indices match
			// the compacted output — drawing more instances than the compacted
			// count would read uninitialised / stale slots.
			if a.CullingEnabled() {
				if key := mdl.ComputePipelineKey(); key != "" {
					if rp := s.r.Pipeline(key); rp != nil {
						if cs := rp.Shader(shader.ShaderTypeCompute); cs != nil {
							indirectBinding := 0
							for _, decl := range cs.Declarations() {
								if decl.Type == shader.AnnotationTypeBindingGroup && decl.Binding != nil {
									typeArg := string(decl.Args[2])
									if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
										typeArg = strings.TrimSuffix(stripped, ">")
									}
									if shader.AnnotationArg(typeArg) == shader.AnnotationArgIndirectArgs {
										indirectBinding = *decl.Binding
										break
									}
								}
							}
							if indBuf := a.IndirectBuffer(indirectBinding); indBuf != nil {
								_ = s.r.ShadowDrawCallIndirect(pipeKey, meshProvider, indBuf, shadowBindGroups)
								continue
							}
						}
					}
				}
			}

			_ = s.r.ShadowDrawCall(pipeKey, meshProvider, uint32(a.InstanceCount()), shadowBindGroups)
		}
	}

	s.r.EndShadowPass()
	s.r.EndShadowFrame()
}

// initLightCullResources initializes the Forward+ light culling pipeline and buffer resources.
func (s *scene) initLightCullResources(cullComputeShader, litFragmentShader shader.Shader, screenWidth, screenHeight int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || cullComputeShader == nil || litFragmentShader == nil {
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
		0: uint64((&light.GPULightCullUniforms{}).Size()), // LightCullUniforms
		2: numTiles * 4,                                   // tile_light_counts: one u32 per tile
		3: numTiles * uint64(light.MaxLightsPerTile) * 4,  // tile_light_indices
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
		TileCountX:       tileCountX,
		MaxLightsPerTile: light.MaxLightsPerTile,
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: tileBGP, Binding: 0, Offset: 0, Data: tileUniforms.Marshal()},
	})
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
	if s.cam == nil || litFragShader == nil {
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

func (s *scene) PrepareLightCulling() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.lightHandler.Enabled() || s.r == nil || s.cam == nil {
		return
	}

	// Count enabled lights. Even when zero we must still dispatch the cull
	// shader so that tile counts are zeroed out — otherwise stale tile data
	// from the previous frame causes disabled lights to keep rendering.
	var lightCount uint32
	for _, l := range s.lightHandler.Lights() {
		if l.Enabled() {
			lightCount++
		}
	}

	// Build and write cull uniforms.
	cullBGP := s.lightHandler.Bgp("light_cull")
	uniforms := light.GPULightCullUniforms{
		InvProj:      s.cam.InverseProjectionMatrix(),
		ViewMatrix:   s.cam.ViewMatrix(),
		TileCountX:   s.lightHandler.TileCountX(),
		TileCountY:   s.lightHandler.TileCountY(),
		ScreenWidth:  uint32(s.lightHandler.ScreenWidth()),
		ScreenHeight: uint32(s.lightHandler.ScreenHeight()),
		LightCount:   lightCount,
		Near:         s.cam.Near(),
		Far:          s.cam.Far(),
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: cullBGP, Binding: 0, Offset: 0, Data: uniforms.Marshal()},
	})

	// Dispatch the light culling compute shader.
	if err := s.r.BeginComputeFrame(); err != nil {
		return
	}
	s.r.DispatchCompute(s.lightHandler.PipelineKey("light_cull"), cullBGP, [3]uint32{s.lightHandler.TileCountX(), s.lightHandler.TileCountY(), 1})
	s.r.EndComputeFrame()
}

// initLighting initializes the entire lighting pipeline in the correct order:
// light storage buffer, shadow map resources, shadow lit bind group, and Forward+
// light culling. All lighting shaders are loaded internally from the engine's
// standard light shader assets.
func (s *scene) initLighting(screenWidth, screenHeight int) {
	litFragShader := shader.NewShader("_lit_frag", shader.ShaderTypeFragment, "engine/light/assets/lit-frag.wgsl")
	shadowVertShader := shader.NewShader("_shadow_depth_vert", shader.ShaderTypeVertex, "engine/light/assets/shadow-depth-vert.wgsl")
	shadowSkinnedVertShader := shader.NewShader("_shadow_depth_skinned_vert", shader.ShaderTypeVertex, "engine/light/assets/shadow-depth-skinned-vert.wgsl")
	cullComputeShader := shader.NewShader("_light_cull_compute", shader.ShaderTypeCompute, "engine/light/assets/light-cull-compute.wgsl")

	// 1. Light storage buffer (must be first — other steps share this buffer).
	s.initLightBindGroup(litFragShader)

	// 2. Shadow depth texture, comparison sampler, shadow data BGP, shadow pipelines.
	s.initShadowMap(shadowVertShader, shadowSkinnedVertShader)

	// 3. Shadow lit BGP (fragment-side shadow sampling — references shadow resources from step 2).
	s.initShadowLitBindGroup(litFragShader)

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

	// 6. Mark the lighting subsystem as GPU-initialized.
	s.lightHandler.SetEnabled(true)
}

func (s *scene) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.registry)
}

func (s *scene) CountEphemeral() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	count := 0
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			count += int(a.InstanceCount())
		}
	}
	return count
}

func (s *scene) Add(obj game_object.GameObject, pipelineOpts ...pipeline.PipelineBuilderOption) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil {
		panic("scene: cannot Add without a Renderer attached")
	}

	mdl := obj.Model()
	if mdl == nil {
		panic("scene: cannot Add a GameObject without a Model")
	}

	// Auto-resolve standard shaders based on whether the model uses skeletal animation
	// and whether the scene has lighting enabled.
	var computeShader, vertexShader, fragmentShader shader.Shader
	lit := s.lightHandler.Enabled()
	if mdl.Skinned() {
		computeShader = shader.NewShader(mdl.Name()+"_compute", shader.ShaderTypeCompute, "engine/renderer/animator/assets/skeletal-compute.wgsl")
		if lit {
			vertexShader = shader.NewShader(mdl.Name()+"_vertex", shader.ShaderTypeVertex, "engine/light/assets/lit-skinned-vert.wgsl")
		} else {
			vertexShader = shader.NewShader(mdl.Name()+"_vertex", shader.ShaderTypeVertex, "engine/model/assets/skinned-vert.wgsl")
		}
	} else {
		computeShader = shader.NewShader(mdl.Name()+"_compute", shader.ShaderTypeCompute, "engine/renderer/animator/assets/simple-compute.wgsl")
		if lit {
			vertexShader = shader.NewShader(mdl.Name()+"_vertex", shader.ShaderTypeVertex, "engine/light/assets/lit-vert.wgsl")
		} else {
			vertexShader = shader.NewShader(mdl.Name()+"_vertex", shader.ShaderTypeVertex, "engine/model/assets/simple-vert.wgsl")
		}
	}
	// Resolve fragment shader path: use the first material's custom path if set,
	// otherwise fall back to the lit or standard textured fragment shader.
	fragShaderPath := "engine/model/assets/textured-frag.wgsl"
	if lit {
		fragShaderPath = "engine/light/assets/lit-frag.wgsl"
	}
	for _, mat := range mdl.RenderMaterials() {
		if p := mat.FragmentShaderPath(); p != "" {
			fragShaderPath = p
			break
		}
	}
	fragmentShader = shader.NewShader(mdl.Name()+"_fragment", shader.ShaderTypeFragment, fragShaderPath)

	if obj.ID() == 0 {
		obj.SetID(atomic.AddUint64(&s.nextID, 1) - 1)
	}

	// Lookup or create an Animator for this Model
	animPool, exists := s.animatorPool[mdl]
	var anim animator.Animator
	if !exists {
		anim = s.createAnimator(mdl, computeShader, vertexShader, fragmentShader, pipelineOpts...)
		animPool = []animator.Animator{anim}
		s.animatorPool[mdl] = animPool
	} else {
		for _, a := range animPool {
			if a.InstanceCount() < a.MaxInstances() {
				anim = a
				break
			}
		}
		if anim == nil {
			anim = s.createAnimator(mdl, computeShader, vertexShader, fragmentShader, pipelineOpts...)
			animPool = append(animPool, anim)
			s.animatorPool[mdl] = animPool
		}
	}

	// Capture initial transform from the GameObject BEFORE wiring the animator.
	// TransformData returns the builder-supplied initial values (position, scale,
	// rotation, rotation speed) when the animator is nil. Once SetAnimator is called,
	// it would read from the animator's zero-initialized instance slot instead.
	pos, scale, rot, rotSpeed := obj.TransformData()

	// Wire the object to the animator and add an instance
	obj.SetAnimator(anim)
	idx, err := anim.AddInstance()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to add instance for model %q: %v", mdl.Name(), err))
	}
	obj.SetAnimatorInstanceID(int(idx))

	// Push initial transform data from the GameObject into the animator slot
	anim.SetInstanceData(idx, pos, scale, rotSpeed, rot)

	// Update reverse-index so Remove() can find the swapped object in O(1).
	if s.instanceLookup[anim] == nil {
		s.instanceLookup[anim] = make(map[uint32]uint64)
	}
	s.instanceLookup[anim][idx] = obj.ID()

	// Persist non-ephemeral objects in the registry
	if !obj.Ephemeral() {
		s.registry[obj.ID()] = obj
	}

	// If the object has an attached light, track it for automatic position sync
	// and register the light with the handler's light list.
	if l := obj.Light(); l != nil {
		s.lightObjects = append(s.lightObjects, obj)
		s.lightHandler.AddLight(l)
	}

	if obj.RigidBody() != nil && s.physicsHandler != nil {
		bodyIndex := s.physicsHandler.RegisterBody(obj.ID(), [3]float32{pos[0], pos[1], pos[2]}, [3]float32{rot[0], rot[1], rot[2]}, obj.RigidBody(), uint32(obj.AnimatorInstanceID()))
		if !s.physicsGPUReady {
			s.initPhysicsGPU()
			s.physicsGPUReady = true
		}

		// Ensure this animator has a sync group. Each unique Animator that
		// owns physics bodies gets its own sync dispatch with a per-group
		// sync_map buffer (sentinel-initialized) and the animator's own
		// AnimationData buffer. Bodies not belonging to a group are skipped
		// by the shader via the 0xFFFFFFFF sentinel.
		if s.physicsSyncAnimMap == nil {
			s.physicsSyncAnimMap = make(map[animator.Animator]int)
		}
		sgIdx, exists := s.physicsSyncAnimMap[anim]
		if !exists {
			sgIdx = s.createPhysicsSyncGroup(anim)
		}

		// Stage a write of this body's instance_id into the group's sync_map buffer.
		instanceData := make([]byte, 4)
		binary.LittleEndian.PutUint32(instanceData, uint32(obj.AnimatorInstanceID()))
		s.physicsSyncWrites = append(s.physicsSyncWrites, bind_group_provider.BufferWrite{
			Provider: s.physicsSyncGroups[sgIdx].bgp,
			Binding:  1,
			Offset:   uint64(bodyIndex) * 4,
			Data:     instanceData,
		})

		// For kinematic bodies on skeletal animators, create a bone particle
		// update group so the bone_particle_update shader transforms their
		// particles through the current bone matrices each frame.
		rb := obj.RigidBody()
		mdl := obj.Model()
		if rb.Kinematic() && mdl != nil && mdl.Skinned() && mdl.Skeleton() != nil {
			s.createBoneParticleUpdateGroup(anim, bodyIndex, mdl, uint32(obj.AnimatorInstanceID()))
		}
	}

	return obj.ID()
}

func (s *scene) Get(id uint64) game_object.GameObject {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.registry[id]
}

func (s *scene) Remove(id uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	obj, exists := s.registry[id]
	if !exists {
		return
	}

	delete(s.registry, id)

	// Remove attached light from scene tracking lists
	if l := obj.Light(); l != nil {
		s.lightHandler.RemoveLight(l)
		for i, o := range s.lightObjects {
			if o == obj {
				s.lightObjects = append(s.lightObjects[:i], s.lightObjects[i+1:]...)
				break
			}
		}
	}

	// Swap-remove the instance data from the animator and patch the physics
	// sync_map so the sync shader writes each body's transform to the correct
	// Animator instance slot after the swap.
	anim := obj.Animator()
	if anim != nil {
		removedIdx := obj.AnimatorInstanceID()
		if removedIdx >= 0 {
			swappedFrom, swapped := anim.RemoveInstance(uint32(removedIdx))

			// Remove the deleted object's entry from the reverse-index.
			if lut := s.instanceLookup[anim]; lut != nil {
				delete(lut, uint32(removedIdx))
			}

			if swapped {
				// The instance at swappedFrom was moved into removedIdx.
				// Use the reverse-index for O(1) lookup instead of scanning
				// the entire registry.
				if lut := s.instanceLookup[anim]; lut != nil {
					if swappedObjID, ok := lut[swappedFrom]; ok {
						if o, exists := s.registry[swappedObjID]; exists {
							o.SetAnimatorInstanceID(removedIdx)
							s.patchSyncMapEntry(anim, o.ID(), uint32(removedIdx))
						}
						// Update the reverse-index: swappedFrom is gone, now lives at removedIdx.
						delete(lut, swappedFrom)
						lut[uint32(removedIdx)] = swappedObjID
					}
				}
			}
			obj.SetAnimatorInstanceID(-1)
		}
	}

	// Sentinel the removed body's sync_map entry and deactivate its GPU slot.
	if s.physicsHandler != nil {
		s.patchSyncMapEntry(anim, obj.ID(), 0xFFFFFFFF)
		s.physicsHandler.RemoveBody(obj.ID())
	}
}

// initPhysicsGPU creates GPU buffers, bind groups, and compute pipelines for the
// physics simulation. Called once when the first rigid body is added to the scene.
// Follows the same InitBindGroup + shared buffer + pipeline registration pattern
// used by createAnimator. Caller must hold s.mu write lock.
func (s *scene) initPhysicsGPU() {
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
		shaders[st.name] = shader.NewShader("physics_"+st.name, shader.ShaderTypeCompute, st.path)
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
			if decl.Type != shader.AnnotationTypeBindingGroup || decl.Binding == nil {
				continue
			}
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
			"engine/physics/assets/bone-particle-update.wgsl")
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

// createPhysicsSyncGroup creates a per-animator sync bind group provider for the
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
//   - int: the index of the new sync group in s.physicsSyncGroups
func (s *scene) createPhysicsSyncGroup(anim animator.Animator) int {
	ph := s.physicsHandler

	// Discover the AnimationData binding index from the standard simple compute shader (cached).
	if s.physicsAnimBinding < 0 {
		syncComputeShader := shader.NewShader("_sync_compute_init", shader.ShaderTypeCompute, "engine/renderer/animator/assets/simple-compute.wgsl")
		for _, decl := range syncComputeShader.Declarations() {
			if decl.Type != shader.AnnotationTypeBindingGroup || decl.Binding == nil {
				continue
			}
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
	if s.physicsAnimBinding < 0 {
		panic("scene: AnimationData binding not found in compute shader declarations")
	}

	groupID := uint32(len(s.physicsSyncGroups))
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

	sg := &physicsSyncGroup{
		groupID: groupID,
		bgp:     bgp,
	}
	s.physicsSyncGroups = append(s.physicsSyncGroups, sg)
	s.physicsSyncAnimMap[anim] = int(groupID)

	return int(groupID)
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
		Provider: s.physicsSyncGroups[sgIdx].bgp,
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
		const maxBonesGPU = uint64(64) // must match WGSL MAX_BONES in skeletal-compute.wgsl / skinned-vert.wgsl
		perInstanceOutputSize = (1 + maxBonesGPU) * 64
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
	// that doesn't already have a bind group provider set (loader-produced materials
	// will not have GPU resources yet since the loader is CPU-only).
	for i, mat := range mdl.RenderMaterials() {
		if mat.BindGroupProvider() != nil {
			continue
		}
		providerName := fmt.Sprintf("%s_material_%d", mdl.Name(), i)
		if err := s.r.RegisterMaterial(mat, providerName); err != nil {
			panic(fmt.Sprintf("scene: failed to init material GPU for model %q material %d: %v", mdl.Name(), i, err))
		}
	}

	return anim
}

func (s *scene) PrepareCompute(deltaTime float32) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.r == nil {
		return
	}

	// Update camera matrices and write VP matrix to GPU once per frame
	var gpuPlanes [6]animator.GPUFrustumPlane
	hasFrustum := false
	if s.cam != nil {
		s.cam.Update()
		vpMat := s.cam.ViewProjectionMatrix()
		if camBGP := s.cam.BindGroupProvider(); camBGP != nil {
			camUniform := camera.GPUCameraUniform{ViewProj: vpMat}
			if ctrl := s.cam.Controller(); ctrl != nil {
				camUniform.CameraPosition[0], camUniform.CameraPosition[1], camUniform.CameraPosition[2] = ctrl.Position()
			}
			s.r.WriteBuffers([]bind_group_provider.BufferWrite{
				{
					Provider: camBGP,
					Binding:  0,
					Offset:   0,
					Data:     camUniform.Marshal(),
				},
			})
		}

		// Extract frustum planes from the VP matrix for GPU-side culling
		frustum := common.ExtractFrustumFromMatrix(vpMat[:])
		for i := range 6 {
			gpuPlanes[i] = animator.GPUFrustumPlane{
				Normal:   frustum.Planes[i].Normal,
				Distance: frustum.Planes[i].Distance,
			}
		}
		hasFrustum = !s.cullingDisabled
	}

	// Sync attached lights: copy each game object's world position to its light.
	for _, obj := range s.lightObjects {
		if l := obj.Light(); l != nil && obj.Enabled() {
			x, y, z := obj.Position()
			l.SetPosition(x, y, z)
		}
	}

	// Write light buffer to GPU each frame when lighting is initialized.
	if s.lightHandler.Enabled() {
		lightsBGP := s.lightHandler.Bgp("lights")
		lightData := light.MarshalLightBuffer(s.lightHandler.Lights(), s.lightHandler.AmbientColor())
		writes := []bind_group_provider.BufferWrite{
			{
				Provider: lightsBGP,
				Binding:  0, // light_header uniform
				Offset:   0,
				Data:     lightData[:16], // GPULightHeader is 16 bytes
			},
		}
		if len(lightData) > 16 {
			writes = append(writes, bind_group_provider.BufferWrite{
				Provider: lightsBGP,
				Binding:  1, // lights storage array
				Offset:   0,
				Data:     lightData[16:],
			})
		}
		s.r.WriteBuffers(writes)
	}

	// Process all animator groups in three phases:
	// Pre-pass (serial): rebuild GPU buffers for any groups that grew since last frame.
	// Phase 1 (parallel): fan out CPU-only prep work across goroutines.
	// Phase 2 (serial): coalesce buffer writes and dispatch compute shaders.

	// Pre-pass: serial RebuildGPU for animators that grew — requires GPU access.
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}
			// we don't currently have a RebuildGPU step and we shouldn't have one because we just want to ignore dead animators
			// ideally we should remove the dead animator from the pool, but should that be done here or elsewhere?
			// if a.NeedsRebuild() {
			// 	if err := a.RebuildGPU(s.r.InitBindGroups); err != nil {
			// 		continue
			// 	}
			// }
		}
	}

	// Phase 1: parallel CPU prep — submit each animator's prep work to the
	// compute pool. Workers are reused across frames (no goroutine spawn overhead).
	// A WaitGroup provides per-frame barrier sync since pool.Wait() blocks until
	// workers idle-exit which is unsuitable for frame-rate workloads.
	var wg sync.WaitGroup
	taskID := 0
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}

			mdl := a.Model()
			if mdl == nil {
				continue
			}
			pipeKey := mdl.ComputePipelineKey()
			if pipeKey == "" {
				continue
			}
			pipe := s.r.Pipeline(pipeKey)
			if pipe == nil {
				continue
			}
			shdr := pipe.Shader(shader.ShaderTypeCompute)
			if shdr == nil {
				continue
			}

			wg.Add(1)
			aCap := a // capture for closure
			id := taskID
			taskID++
			s.computePool.SubmitTask(worker.Task{
				ID: id,
				Do: func() (any, error) {
					defer wg.Done()

					uniformBinding, instanceBinding, boneBinding, modelBinding := 0, 0, 0, 0
					for _, decl := range shdr.Declarations() {
						if decl.Type != shader.AnnotationTypeBindingGroup || decl.Binding == nil {
							continue
						}
						typeArg := string(decl.Args[2])
						if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
							typeArg = strings.TrimSuffix(stripped, ">")
						}
						switch shader.AnnotationArg(typeArg) {
						case shader.AnnotationArgGlobalData, shader.AnnotationArgAnimationGlobals:
							uniformBinding = *decl.Binding
						case shader.AnnotationArgAnimationData, shader.AnnotationArgSkeletalAnimationData:
							instanceBinding = *decl.Binding
						case shader.AnnotationArgBoneInfo:
							boneBinding = *decl.Binding
						case shader.AnnotationArgModelData:
							modelBinding = *decl.Binding
						}
					}

					// Feed frustum planes to the animator for GPU-side culling.
					// This must happen before PrepareFrame so the uniform data includes the planes.
					if hasFrustum {
						aCap.SetFrustumPlanes(gpuPlanes)
					}

					aCap.PrepareFrame(deltaTime, uniformBinding)
					aCap.Flush(instanceBinding, boneBinding, modelBinding)
					return nil, nil
				},
			})
		}
	}
	wg.Wait()

	// Phase 2: coalesced GPU submission — collect all buffer writes from all animators into a single
	// slice, then submit once to the renderer. This reduces mutex acquisitions from N to 1 for writes.
	// For each animator with culling enabled, reset the indirect args buffer to zero instance count
	// before collecting its writes, so the compute shader can atomically count visible instances.
	allWrites := s.writePool[:0]
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}
			if a.CullingEnabled() {
				if m := a.Model(); m != nil {
					if mp := m.MeshProvider(); mp != nil {
						pipeKey := m.ComputePipelineKey()
						if pipeKey == "" {
							continue
						}
						pipe := s.r.Pipeline(pipeKey)
						if pipe == nil {
							continue
						}
						shdr := pipe.Shader(shader.ShaderTypeCompute)
						if shdr == nil {
							continue
						}

						indirectBinding := 0
						for _, decl := range shdr.Declarations() {
							if decl.Type != shader.AnnotationTypeBindingGroup || decl.Binding == nil {
								continue
							}
							typeArg := string(decl.Args[2])
							if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
								typeArg = strings.TrimSuffix(stripped, ">")
							}
							if shader.AnnotationArg(typeArg) == shader.AnnotationArgIndirectArgs {
								indirectBinding = *decl.Binding
								break
							}
						}
						a.ResetIndirectArgs(uint32(mp.IndexCount()), indirectBinding)
					}
				}
			}

			allWrites = append(allWrites, a.StagedWriteData()...)
		}
	}
	s.writePool = allWrites

	if len(allWrites) > 0 {
		s.r.WriteBuffers(allWrites)
	}

	// ── Physics compute dispatch ───────────────────────────────────────
	// Runs BEFORE the animator compute dispatches so the sync shader's writes to
	// AnimationData are visible when the animator reads it for model-matrix
	// generation and frustum culling. The 8-stage GPU rigid body pipeline uses a
	// fixed-timestep accumulator. PrepareStep returns the number of substeps for
	// this frame and the marshaled globals uniform. Each substep dispatches the full
	// pipeline: particle values → AABB reduce → grid build params → grid clear →
	// grid insert → collision → compute momenta → integrate. Between substeps,
	// only the AABB atomics buffer is reset; body and particle data persist in the
	// storage buffers across stages.
	if ph := s.physicsHandler; ph != nil && ph.Enabled() {
		// Process any pending GPU→CPU readback from the previous frame's copy command.
		// By this point the compute command buffer containing the CopyBufferToBuffer has
		// been submitted (EndComputeFrame from the prior frame), so the staging buffer is
		// safe to map synchronously. This only runs when game logic called RequestReadback.
		if ph.ReadbackPending() {
			bodySize := uint64((&physics.GPUBody{}).Size())
			readSize := uint64(ph.BodiesCount()) * bodySize
			if readSize > 0 {
				data, err := s.r.ReadMappedBuffer(ph.StagingBuffer(), 0, readSize)
				if err == nil {
					ph.ProcessReadback(data)
				}
			}
			ph.ClearReadbackPending()
		}

		// Collect staged writes (body registrations, removals, force drains).
		// PrepareStep MUST be called first so that force drains from ApplyForce()
		// are staged into the same write batch as body registrations. Without this,
		// newly spawned bodies spend their first physics frame with zero external
		// force (no gravity), causing them to sit at the spawn point while collision
		// forces from overlapping neighbors fling them apart.
		substeps, globalsData := ph.PrepareStep(deltaTime)
		physWrites := ph.StagedWriteData()

		// Append per-group sync_map writes staged during Add() calls.
		if len(s.physicsSyncWrites) > 0 {
			physWrites = append(physWrites, s.physicsSyncWrites...)
			s.physicsSyncWrites = s.physicsSyncWrites[:0]
		}

		if substeps > 0 {
			// Write globals uniform once — it is constant across all substeps
			// since fixedDt does not change within a frame.
			physWrites = append(physWrites, bind_group_provider.BufferWrite{
				Provider: ph.Buffers(),
				Binding:  3,
				Offset:   0,
				Data:     globalsData,
			})

			if len(physWrites) > 0 {
				s.r.WriteBuffers(physWrites)
			}

			// physDispatchGroups computes the number of work groups needed to cover
			// itemCount invocations for the shader behind the given pipeline key.
			// The workgroup size is read from the parsed WGSL source, not hardcoded.
			physDispatchGroups := func(pipeKey string, itemCount uint32) [3]uint32 {
				pipe := s.r.Pipeline(pipeKey)
				if pipe == nil {
					return [3]uint32{1, 1, 1}
				}
				shdr := pipe.Shader(shader.ShaderTypeCompute)
				if shdr == nil {
					return [3]uint32{1, 1, 1}
				}
				wgSize := shdr.WorkgroupSize()
				xSize := wgSize[0]
				if xSize == 0 {
					xSize = 1
				}
				groups := (itemCount + xSize - 1) / xSize
				if groups == 0 {
					groups = 1
				}
				return [3]uint32{groups, 1, 1}
			}

			particleCount := uint32(ph.ParticleCount())
			bodyCount := uint32(ph.BodiesCount())

			// Pre-build AABB atomics reset payload (6 × u32):
			//   indices 0–2 (min): 0xFFFFFFFF (largest sortable uint → will be atomicMin'd down)
			//   indices 3–5 (max): 0x00000000 (smallest sortable uint → will be atomicMax'd up)
			aabbReset := make([]byte, 24)
			binary.LittleEndian.PutUint32(aabbReset[0:4], 0xFFFFFFFF)
			binary.LittleEndian.PutUint32(aabbReset[4:8], 0xFFFFFFFF)
			binary.LittleEndian.PutUint32(aabbReset[8:12], 0xFFFFFFFF)
			// indices 3–5 are already zero from make()

			for sub := 0; sub < substeps; sub++ {
				// Stage 1: Particle value computation (world pos, velocity, rel pos)
				pvKey := ph.PipelineKey("particle_values")
				s.r.DispatchCompute(pvKey, ph.Bgp("particle_values"),
					physDispatchGroups(pvKey, particleCount))

				// Reset AABB atomics before the reduction pass.
				s.r.WriteBuffers([]bind_group_provider.BufferWrite{
					{Provider: ph.Buffers(), Binding: 5, Offset: 0, Data: aabbReset},
				})

				// Stage 1.5a: AABB reduction (parallel atomicMin/Max over particle positions)
				arKey := ph.PipelineKey("aabb_reduce")
				s.r.DispatchCompute(arKey, ph.Bgp("aabb_reduce"),
					physDispatchGroups(arKey, particleCount))

				// Stage 1.5b: Grid build params (single invocation derives grid origin + dims)
				gbKey := ph.PipelineKey("grid_build_params")
				s.r.DispatchCompute(gbKey, ph.Bgp("grid_build_params"),
					physDispatchGroups(gbKey, 1))

				// Stage 2a: Grid clear (fill all cells with sentinel 0xFFFFFFFF)
				gcKey := ph.PipelineKey("grid_clear")
				s.r.DispatchCompute(gcKey, ph.Bgp("grid_clear"),
					physDispatchGroups(gcKey, ph.MaxGridCells()))

				// Stage 2b: Grid insert (hash particles into cells via atomic CAS)
				giKey := ph.PipelineKey("grid_insert")
				s.r.DispatchCompute(giKey, ph.Bgp("grid_insert"),
					physDispatchGroups(giKey, particleCount))

				// Stage 3: Collision detection & DEM force computation
				crKey := ph.PipelineKey("collision")
				s.r.DispatchCompute(crKey, ph.Bgp("collision"),
					physDispatchGroups(crKey, particleCount))

				// Stage 4: Momentum accumulation (sum particle forces → body momenta)
				cmKey := ph.PipelineKey("momenta")
				s.r.DispatchCompute(cmKey, ph.Bgp("momenta"),
					physDispatchGroups(cmKey, bodyCount))

				// Stage 5: Integration (update position & quaternion from momenta)
				iKey := ph.PipelineKey("integrate")
				s.r.DispatchCompute(iKey, ph.Bgp("integrate"),
					physDispatchGroups(iKey, bodyCount))
			}

			// After all substeps, sync physics results back to each Animator's
			// AnimationData buffer. Each sync group dispatches the sync shader with
			// its own BGP that binds the correct AnimationData buffer and per-group
			// sync_map (sentinel-filtered so non-member bodies are skipped).
			if len(s.physicsSyncGroups) > 0 {
				syncKey := ph.PipelineKey("sync")
				wg := physDispatchGroups(syncKey, bodyCount)
				for _, sg := range s.physicsSyncGroups {
					s.r.DispatchCompute(syncKey, sg.bgp, wg)
				}
			}

			// If game logic requested a readback, encode a GPU→GPU copy of the bodies
			// buffer into the staging buffer. The next frame will map and process it.
			if ph.ConsumeReadbackRequest() {
				if staging := ph.StagingBuffer(); staging != nil {
					copySize := uint64(bodyCount) * uint64((&physics.GPUBody{}).Size())
					s.r.CopyBufferToBuffer(ph.Buffers().Buffer(0), staging, 0, 0, copySize)
				}
			}
		} else if len(physWrites) > 0 {
			// No substeps this frame (accumulator hasn't reached fixedDt yet),
			// but we still need to flush registration/removal writes.
			s.r.WriteBuffers(physWrites)
		}
	}

	// Dispatch compute shaders for each registered animator with instances.
	// This runs AFTER the physics block so the sync shader's writes to
	// AnimationData (positions/rotations of physics-controlled bodies) are
	// visible when the animator's compute shader builds model matrices and
	// performs frustum culling.
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}
			mdl := a.Model()
			if mdl == nil {
				continue
			}
			key := mdl.ComputePipelineKey()
			if key == "" {
				continue
			}
			pipe := s.r.Pipeline(key)
			if pipe == nil {
				continue
			}
			shdr := pipe.Shader(shader.ShaderTypeCompute)
			if shdr == nil {
				continue
			}
			// Dispatch the correct number of workgroups to cover all instances.
			// shdr.WorkgroupSize() returns the per-workgroup thread count (e.g. 256),
			// NOT the number of groups. We need ceil(instanceCount / workgroupSize).
			wgSize := shdr.WorkgroupSize()
			xSize := wgSize[0]
			if xSize == 0 {
				xSize = 1
			}
			instCount := a.InstanceCount()
			groups := (instCount + xSize - 1) / xSize
			if groups == 0 {
				groups = 1
			}
			s.r.DispatchCompute(key, a.ComputeBindGroupProvider(), [3]uint32{groups, 1, 1})
		}
	}

	// Dispatch bone particle update for kinematic bodies after animator compute.
	// The animator has populated scratch_matrices with current bone world matrices;
	// this shader transforms each kinematic particle through its bone matrix so the
	// next frame's physics collision pipeline sees the animated pose.
	if ph := s.physicsHandler; ph != nil && len(s.boneParticleUpdateGroups) > 0 {
		boneUpdateKey := ph.PipelineKey("bone_update")
		boneUpdatePipe := s.r.Pipeline(boneUpdateKey)
		if boneUpdatePipe != nil {
			boneUpdateShader := boneUpdatePipe.Shader(shader.ShaderTypeCompute)
			if boneUpdateShader != nil {
				wgSize := boneUpdateShader.WorkgroupSize()
				xSize := wgSize[0]
				if xSize == 0 {
					xSize = 1
				}
				for _, bg := range s.boneParticleUpdateGroups {
					groups := (bg.particleCount + xSize - 1) / xSize
					if groups == 0 {
						groups = 1
					}
					s.r.DispatchCompute(boneUpdateKey, bg.bgp, [3]uint32{groups, 1, 1})
				}
			}
		}
	}
}

func (s *scene) DrawCalls() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.r == nil {
		return fmt.Errorf("scene %q has no renderer attached", s.name)
	}

	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}

			mdl := a.Model()
			if mdl == nil {
				continue
			}
			meshProvider := mdl.MeshProvider()
			if meshProvider == nil {
				continue
			}

			mats := mdl.RenderMaterials()
			if len(mats) == 0 {
				continue
			}

			for _, mat := range mats {
				pipelineKey := mat.PipelineKey()
				if pipelineKey == "" {
					continue
				}

				// Look up the render pipeline to discover bind group layouts from both shaders.
				rp := s.r.Pipeline(pipelineKey)
				if rp == nil {
					continue
				}
				renderShader := rp.Shader(shader.ShaderTypeVertex)
				if renderShader == nil {
					continue
				}

				// Collect declarations from vertex and fragment shaders.
				var allDecls []shader.Annotation
				allDecls = append(allDecls, renderShader.Declarations()...)
				if fragShader := rp.Shader(shader.ShaderTypeFragment); fragShader != nil {
					allDecls = append(allDecls, fragShader.Declarations()...)
				}

				// Build bind groups dynamically by matching each group's var names to a provider.
				// Groups are iterated in index order so bindGroups[i] maps to @group(i).
				maxGroup := -1
				groupProviders := make(map[int]bind_group_provider.BindGroupProvider)
				for _, decl := range allDecls {
					if decl.Group == nil {
						continue
					}
					g := *decl.Group
					if g > maxGroup {
						maxGroup = g
					}
					if _, exists := groupProviders[g]; exists {
						continue
					}

					var provider bind_group_provider.BindGroupProvider
					switch decl.Type {
					case shader.AnnotationTypeProvider:
						switch decl.Args[0] {
						case shader.AnnotationArgCamera:
							if s.cam != nil {
								provider = s.cam.BindGroupProvider()
							}
						case shader.AnnotationArgMaterial:
							if mp := mat.Provider(g); mp != nil {
								provider = mp
							} else {
								provider = mat.BindGroupProvider()
							}
						case shader.AnnotationArgLights:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("lights")
							}
						case shader.AnnotationArgShadow:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("shadow_lit")
							}
						case shader.AnnotationArgTiles:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("tile_lit")
							}
						case shader.AnnotationArgEffect:
							if ep := mdl.EffectProvider(); ep != nil {
								provider = ep
							}
						case shader.AnnotationArgAnimator:
							provider = a.OutputBindGroupProvider()
						}
					case shader.AnnotationTypeBindingGroup:
						typeArg := string(decl.Args[2])
						if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
							typeArg = strings.TrimSuffix(stripped, ">")
						}
						switch shader.AnnotationArg(typeArg) {
						case shader.AnnotationArgCamera:
							if s.cam != nil {
								provider = s.cam.BindGroupProvider()
							}
						case shader.AnnotationArgInstanceData:
							provider = a.OutputBindGroupProvider()
						case shader.AnnotationArgLight, shader.AnnotationArgLightHeader:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("lights")
							}
						case shader.AnnotationArgShadowData, shader.AnnotationArgShadowUniform:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("shadow_lit")
							}
						case shader.AnnotationArgTileUniforms:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("tile_lit")
							}
						case shader.AnnotationArgOverlayParams:
							if bp := mat.BindGroupProvider(); bp != nil {
								provider = bp
							} else if ep := mdl.EffectProvider(); ep != nil {
								provider = ep
							}
						case shader.AnnotationArgEffectParams:
							if ep := mdl.EffectProvider(); ep != nil {
								provider = ep
							} else if mp := mat.Provider(g); mp != nil {
								provider = mp
							}
						}
					}

					if provider != nil {
						groupProviders[g] = provider
					}
				}

				bindGroups := s.drawBindGroupsPool[:0]
				skipMaterial := false
				for g := 0; g <= maxGroup; g++ {
					provider, ok := groupProviders[g]
					if !ok || provider == nil {
						skipMaterial = true
						break
					}
					bindGroups = append(bindGroups, provider)
				}
				if skipMaterial {
					continue
				}

				// Use indirect draw when GPU frustum culling is active — the compute shader writes
				// the visible instance count into the indirect args buffer, avoiding CPU readback.
				if a.CullingEnabled() {
					var indirectBinding int
					if key := mdl.ComputePipelineKey(); key != "" {
						rp := s.r.Pipeline(key)
						if rp == nil {
							continue
						}
						if cs := rp.Shader(shader.ShaderTypeCompute); cs != nil {
							for _, d := range cs.Declarations() {
								if d.Type == shader.AnnotationTypeBindingGroup && d.Binding != nil {
									arg := string(d.Args[2])
									if stripped, ok := strings.CutPrefix(arg, "array<"); ok {
										arg = strings.TrimSuffix(stripped, ">")
									}
									if shader.AnnotationArg(arg) == shader.AnnotationArgIndirectArgs {
										indirectBinding = *d.Binding
										break
									}
								}
							}
						}
					}
					if indBuf := a.IndirectBuffer(indirectBinding); indBuf != nil {
						if err := s.r.DrawCallIndirect(pipelineKey, meshProvider, indBuf, bindGroups); err != nil {
							return fmt.Errorf("indirect draw call failed for animator in scene %q: %w", s.name, err)
						}
						continue
					}
				}

				if err := s.r.DrawCall(pipelineKey, meshProvider, uint32(a.InstanceCount()), bindGroups); err != nil {
					return fmt.Errorf("draw call failed for animator in scene %q: %w", s.name, err)
				}
			}
		}
	}

	return nil
}
