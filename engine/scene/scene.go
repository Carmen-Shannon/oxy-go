package scene

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
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
	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
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

func isTextureBindingEntry(entry wgpu.BindGroupLayoutEntry) bool {
	return entry.Texture != nil && entry.Texture.SampleType != gputypes.TextureSampleTypeUndefined
}

func isSamplerBindingEntry(entry wgpu.BindGroupLayoutEntry) bool {
	return entry.Sampler != nil && entry.Sampler.Type != gputypes.SamplerBindingTypeUndefined
}

func bufferBindingType(entry wgpu.BindGroupLayoutEntry) gputypes.BufferBindingType {
	if entry.Buffer == nil {
		return gputypes.BufferBindingTypeUndefined
	}

	return entry.Buffer.Type
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

	// PrepareShadowBlur runs the VSM post-processing step: either separable Gaussian
	// blur passes (constant-width soft shadows) or SAT generation (PCSS variable-width).
	// Must be called after PrepareShadows and after the shadow render pass has been
	// submitted to the GPU (i.e. after EndGeometryFrame when using merged geometry passes).
	PrepareShadowBlur()

	// PrepareLightCulling updates the light cull uniform buffer and dispatches the
	// light culling compute shader. Must be called after PrepareCompute (so lights
	// are uploaded) and before DrawCalls.
	PrepareLightCulling()

	// PrepareGBuffer renders the G-Buffer MRT pre-pass for all drawables.
	// Writes world position, normals, and albedo to off-screen textures
	// consumed by screen-space effects (SSAO, SSR). Must be called after
	// PrepareCompute and before PrepareSSAO each frame.
	// No-ops if the G-Buffer subsystem has not been initialized.
	PrepareGBuffer()

	// PrepareSSAO dispatches the SSAO hemisphere sampling compute shader
	// and the bilateral blur passes. Must be called after PrepareGBuffer
	// (so G-Buffer textures are populated) and before DrawCalls.
	// No-ops if the SSAO subsystem has not been initialized.
	PrepareSSAO()

	// PrepareProbes performs incremental probe baking for the irradiance probe
	// grid. For each dirty probe, renders 6 cubemap faces into the bake
	// texture, then dispatches the SH projection compute shader to accumulate
	// L2 spherical harmonic coefficients into the probe storage buffer.
	// Must be called after PrepareCompute and before DrawCalls each frame.
	// No-ops if the probe grid subsystem has not been initialized or there
	// are no dirty probes.
	PrepareProbes()

	// PrepareSSR dispatches the SSR compute shader to perform screen-space ray
	// marching against the G-Buffer. Must be called after DrawCalls (so the HDR
	// texture is populated) and before PrepareComposition. No-ops if the SSR
	// subsystem has not been initialized.
	PrepareSSR()

	// PrepareComposition runs the fullscreen composition pass: acquires the
	// swapchain, samples the HDR lit texture and optional SSR texture, applies
	// ACES tone mapping and gamma correction, and writes the final LDR result
	// to the swapchain. Must be called after DrawCalls (and PrepareSSR if active)
	// and before Present. No-ops if the composition subsystem has not been
	// initialized.
	PrepareComposition()

	// BeginHDRFrame starts the HDR render pass using this scene's composition
	// handler textures. Returns an error if the composition handler is
	// not initialized or the render pass cannot be started.
	//
	// Returns:
	//   - error: an error if the HDR frame could not be started
	BeginHDRFrame() error
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
	if s.lightHandler.GBufferHandler().Enabled() {
		s.lightHandler.GBufferHandler().Resize(width, height)
	}
	if s.lightHandler.SSAOHandler().Enabled() {
		s.lightHandler.SSAOHandler().Resize(width, height)
	}
	if s.lightHandler.CompositionHandler().Enabled() {
		s.lightHandler.CompositionHandler().Resize(width, height)
	}
	if s.lightHandler.SSRHandler().Enabled() {
		s.lightHandler.SSRHandler().Resize(width, height)
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

// float32ToFloat16Bits converts a float32 value to an IEEE 754 half-precision
// (float16) bit pattern. Only used for SSAO noise texture generation where the
// values are in [-1, 1] and full precision is not required.
func float32ToFloat16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int((b>>23)&0xFF) - 127 + 15
	frac := b & 0x7FFFFF

	if exp <= 0 {
		return sign
	}
	if exp >= 31 {
		return sign | 0x7C00
	}
	return sign | uint16(exp)<<10 | uint16(frac>>13)
}

// generateSSAONoise generates the 4×4 RGBA16Float noise texture data for SSAO
// kernel rotation. Each texel is a random tangent-space rotation vector (X, Y, 0, 0)
// encoded as four float16 values (8 bytes per texel, 128 bytes total).
func generateSSAONoise() []byte {
	const texels = 4 * 4
	const bytesPerTexel = 8 // 4 × f16
	data := make([]byte, texels*bytesPerTexel)
	off := 0
	for i := 0; i < texels; i++ {
		x := rand.Float32()*2.0 - 1.0
		y := rand.Float32()*2.0 - 1.0
		// Normalize XY to unit length for a pure rotation.
		length := float32(math.Sqrt(float64(x*x + y*y)))
		if length > 0.0001 {
			x /= length
			y /= length
		}
		binary.LittleEndian.PutUint16(data[off:off+2], float32ToFloat16Bits(x))
		binary.LittleEndian.PutUint16(data[off+2:off+4], float32ToFloat16Bits(y))
		binary.LittleEndian.PutUint16(data[off+4:off+6], float32ToFloat16Bits(0))
		binary.LittleEndian.PutUint16(data[off+6:off+8], float32ToFloat16Bits(0))
		off += bytesPerTexel
	}
	return data
}

// generateSSAOKernel generates a hemisphere sample kernel of the given size
// (clamped to [1, 32]) as a flat byte buffer of array<vec4<f32>, 32> (512 bytes).
// Samples are distributed in a unit hemisphere with an accelerating distribution
// that biases samples closer to the origin.
func generateSSAOKernel(sampleCount int) []byte {
	if sampleCount < 1 {
		sampleCount = 1
	}
	if sampleCount > 32 {
		sampleCount = 32
	}

	buf := make([]byte, 32*16) // 32 × vec4<f32>
	off := 0
	for i := 0; i < 32; i++ {
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

// initGBuffer initializes the G-Buffer MRT textures and registers the static
// and skinned G-Buffer render pipelines. The G-Buffer pre-pass writes per-pixel
// position, normal, and albedo data into screen-sized textures consumed by
// downstream screen-space effects (SSAO, SSR).
func (s *scene) initGBuffer() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || s.lightHandler.GBufferHandler() == nil {
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
	gbufferFrag := shader.NewShader("_gbuffer_frag", shader.ShaderTypeFragment, "engine/light/assets/gbuffer-frag.wgsl")
	staticVert := shader.NewShader("_gbuffer_static_vert", shader.ShaderTypeVertex, "engine/light/assets/lit-vert.wgsl")
	skinnedVert := shader.NewShader("_gbuffer_skinned_vert", shader.ShaderTypeVertex, "engine/light/assets/lit-skinned-vert.wgsl")

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

// initSSAO initializes the SSAO subsystem: creates screen-sized occlusion textures,
// the 4×4 noise texture, registers the SSAO compute and bilateral blur pipelines,
// and pre-creates all bind group providers with correctly-sized GPU buffers. The
// G-Buffer must be initialized before calling this method.
func (s *scene) initSSAO() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || s.lightHandler.SSAOHandler() == nil || s.lightHandler.GBufferHandler() == nil {
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

	// 1. Create SSAO textures (raw, blurred, scratch at SSAO res; noise at 4×4).
	rawView, rawTex, blurView, blurTex, scratchView, scratchTex, noiseView, noiseTex, err := s.r.CreateSSAOTextures(ssaoW, ssaoH)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create SSAO textures: %v", err))
	}
	s.lightHandler.SSAOHandler().SetRawTexture(rawTex)
	s.lightHandler.SSAOHandler().SetRawTextureView(rawView)
	s.lightHandler.SSAOHandler().SetBlurredTexture(blurTex)
	s.lightHandler.SSAOHandler().SetBlurredTextureView(blurView)
	s.lightHandler.SSAOHandler().SetScratchTexture(scratchTex)
	s.lightHandler.SSAOHandler().SetScratchTextureView(scratchView)
	s.lightHandler.SSAOHandler().SetNoiseTexture(noiseTex)
	s.lightHandler.SSAOHandler().SetNoiseTextureView(noiseView)

	// 2. Generate and upload 4×4 noise texture data (RGBA16Float).
	noiseData := generateSSAONoise()
	s.r.WriteTexture(noiseTex, noiseData, 4, 4, 4*8) // 4 pixels wide, 8 bytes/pixel (4×f16)

	// 3. Create or reuse linear sampler for the blurred SSAO texture in the lit shader.
	linearSamp, err := s.r.CreateLinearSampler()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create SSAO linear sampler: %v", err))
	}
	s.lightHandler.SSAOHandler().SetLinearSampler(linearSamp)

	// 4. Register SSAO compute pipeline.
	ssaoCompShader := shader.NewShader("_ssao_compute", shader.ShaderTypeCompute, "engine/light/assets/ssao-compute.wgsl")
	ssaoCompKey := "ssao_compute"
	ssaoCompPipe := pipeline.NewPipeline(ssaoCompKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(ssaoCompShader),
	)
	if err := s.r.RegisterPipelines(ssaoCompPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register SSAO compute pipeline: %v", err))
	}
	s.lightHandler.SSAOHandler().SetPipelineKey("ssao_compute", ssaoCompKey)

	// 5. Register bilateral blur compute pipeline.
	blurCompShader := shader.NewShader("_ssao_blur_compute", shader.ShaderTypeCompute, "engine/light/assets/ssao-blur-compute.wgsl")
	blurCompKey := "ssao_blur_compute"
	blurCompPipe := pipeline.NewPipeline(blurCompKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(blurCompShader),
	)
	if err := s.r.RegisterPipelines(blurCompPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register SSAO blur compute pipeline: %v", err))
	}
	s.lightHandler.SSAOHandler().SetPipelineKey("ssao_blur", blurCompKey)

	// 6. Create SSAO compute bind group provider.
	ssaoDesc := ssaoCompShader.BindGroupLayoutDescriptor(0)
	ssaoSizeOverrides := map[int]uint64{
		4: uint64((&light.GPUSSAOParams{}).Size()), // ssao_params uniform
		5: 32 * 16,                                 // ssao_kernel: array<vec4<f32>, 32> = 512 bytes
	}
	ssaoBGP := s.lightHandler.SSAOHandler().Bgp("ssao_compute")
	ssaoBGP.SetTextureView(0, s.lightHandler.GBufferHandler().DepthTextureView())
	ssaoBGP.SetTextureView(1, s.lightHandler.GBufferHandler().NormalTextureView())
	ssaoBGP.SetTextureView(2, noiseView)
	ssaoBGP.SetTextureView(3, rawView)
	if err := s.r.InitBindGroup(ssaoBGP, ssaoDesc, nil, ssaoSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSAO compute bind group: %v", err))
	}

	// 7. Create blur bind group providers (bilateral blur, depth-aware).
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

	// 8. Generate hemisphere sample kernel and write to the SSAO compute BGP buffer.
	kernelData := generateSSAOKernel(s.lightHandler.SSAOHandler().SampleCount())
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

	if s.r == nil || litFragmentShader == nil {
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
			if isTextureBindingEntry(entry) {
				bgp.SetTextureView(binding, s.lightHandler.SSAOHandler().BlurredTextureView())
			}
			if isSamplerBindingEntry(entry) {
				bgp.SetSampler(binding, s.lightHandler.SSAOHandler().LinearSampler())
			}
		}
	} else {
		// Create a 1×1 white fallback texture (ao=1.0, no darkening).
		for _, entry := range desc.Entries {
			binding := int(entry.Binding)
			if isTextureBindingEntry(entry) {
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
			if isSamplerBindingEntry(entry) {
				fallbackSampler := common.SamplerStagingData{
					AddressModeU:  gputypes.AddressModeClampToEdge,
					AddressModeV:  gputypes.AddressModeClampToEdge,
					AddressModeW:  gputypes.AddressModeClampToEdge,
					MagFilter:     gputypes.FilterModeLinear,
					MinFilter:     gputypes.FilterModeLinear,
					MipmapFilter:  wgpu.FilterMode(gputypes.MipmapFilterModeLinear),
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

// initProbeGrid initializes the irradiance probe grid GPU resources. This
// creates bake textures, probe/grid-params buffers, registers the probe bake
// render pipelines (static + skinned) and the SH projection compute pipeline,
// and initializes all associated bind group providers. Must be called after
// G-Buffer and SSAO initialization.
func (s *scene) initProbeGrid() {
	s.mu.Lock()
	defer s.mu.Unlock()

	pg := s.lightHandler.ProbeGrid()
	if s.r == nil || pg == nil {
		return
	}

	resolution := pg.BakeResolution()
	totalProbes := pg.TotalProbes()
	if totalProbes <= 0 || resolution <= 0 {
		return
	}

	// 1. Create bake render-target textures (RGBA8Unorm color + Depth24Plus).
	colorView, colorTex, depthView, depthTex, err := s.r.CreateProbeBakeTextures(resolution)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create probe bake textures: %v", err))
	}
	pg.SetBakeColorTexture(colorTex)
	pg.SetBakeColorTextureView(colorView)
	pg.SetBakeDepthTexture(depthTex)
	pg.SetBakeDepthTextureView(depthView)

	// 2. Create probe storage buffer (storage r/w for SH projection, read for lit shader).
	probeSize := (&light.GPUIrradianceProbe{}).Size()
	probeBufSize := uint64(totalProbes * probeSize)
	probeBuf, err := s.r.CreateBuffer("Probe Storage", probeBufSize, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create probe storage buffer: %v", err))
	}
	pg.SetProbeBuffer(probeBuf)

	// 3. Create grid params uniform buffer.
	gridParamsSize := uint64((&light.GPUProbeGridParams{}).Size())
	gridParamsBuf, err := s.r.CreateBuffer("Probe Grid Params", gridParamsSize, wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create probe grid params buffer: %v", err))
	}
	pg.SetGridParamsBuffer(gridParamsBuf)

	// 4. Load probe bake shaders.
	staticBakeVert := shader.NewShader("_probe_bake_static_vert", shader.ShaderTypeVertex, "engine/light/assets/probe-bake-vert.wgsl")
	skinnedBakeVert := shader.NewShader("_probe_bake_skinned_vert", shader.ShaderTypeVertex, "engine/light/assets/probe-bake-skinned-vert.wgsl")
	bakeFrag := shader.NewShader("_probe_bake_frag", shader.ShaderTypeFragment, "engine/light/assets/probe-bake-frag.wgsl")

	// 5. Register static bake pipeline.
	staticBakeKey := "probe_bake_static"
	staticBakePipe := pipeline.NewPipeline(staticBakeKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(staticBakeVert),
		pipeline.WithFragmentShader(bakeFrag),
	)
	if err := s.r.RegisterProbeBakePipeline(staticBakePipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register static probe bake pipeline: %v", err))
	}
	pg.SetPipelineKey("static", staticBakeKey)

	// 6. Register skinned bake pipeline.
	skinnedBakeKey := "probe_bake_skinned"
	skinnedBakePipe := pipeline.NewPipeline(skinnedBakeKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(skinnedBakeVert),
		pipeline.WithFragmentShader(bakeFrag),
	)
	if err := s.r.RegisterProbeBakePipeline(skinnedBakePipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register skinned probe bake pipeline: %v", err))
	}
	pg.SetPipelineKey("skinned", skinnedBakeKey)

	// 7. Initialize the bake camera BGP (group 0 from the bake vertex shader).
	bakeCameraBGP := pg.Bgp("probe_bake_camera")
	bakeCameraDesc := staticBakeVert.BindGroupLayoutDescriptor(0)
	bakeCameraSizeOverrides := map[int]uint64{
		0: uint64((&light.GPUProbeBakeCamera{}).Size()),
	}
	if err := s.r.InitBindGroup(bakeCameraBGP, bakeCameraDesc, nil, bakeCameraSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init probe bake camera bind group: %v", err))
	}

	// 8. Register SH projection compute pipeline.
	shProjectShader := shader.NewShader("_probe_sh_project", shader.ShaderTypeCompute, "engine/light/assets/probe-sh-project.wgsl")
	shProjectKey := "probe_sh_project"
	shProjectPipe := pipeline.NewPipeline(shProjectKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(shProjectShader),
	)
	if err := s.r.RegisterPipelines(shProjectPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register SH projection compute pipeline: %v", err))
	}
	pg.SetPipelineKey("sh_project", shProjectKey)

	// 9. Initialize SH projection compute BGP. Pre-set the bake texture view
	// (binding 0) and probe storage buffer (binding 1) so InitBindGroup reuses
	// the existing resources instead of creating new ones.
	shProjectBGP := pg.Bgp("probe_sh_project")
	shProjectDesc := shProjectShader.BindGroupLayoutDescriptor(0)
	shProjectBGP.SetTextureView(0, colorView)
	shProjectBGP.SetBuffer(1, probeBuf)
	shProjectSizeOverrides := map[int]uint64{
		2: uint64((&light.GPUSHProjectParams{}).Size()),
	}
	if err := s.r.InitBindGroup(shProjectBGP, shProjectDesc, nil, shProjectSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SH projection bind group: %v", err))
	}

	// 10. Upload initial probe data and grid parameters.
	gridParams := pg.BuildGPUGridParams()
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: shProjectBGP, Binding: 1, Offset: 0, Data: light.MarshalProbeBuffer(pg.Probes())},
	})
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: pg.Bgp("probe_bake_camera"), Binding: 0, Offset: 0, Data: (&light.GPUProbeBakeCamera{}).Marshal()},
	})

	// Write grid params to the standalone uniform buffer (used by the lit shader
	// probe_lit BGP later — the buffer is shared via pre-set on that BGP).
	s.r.WriteRawBuffer(gridParamsBuf, 0, gridParams.Marshal())

	pg.SetEnabled(true)
}

// initProbesLitBindGroup creates the irradiance probe bind group provider used by
// the lit fragment shader at @group(7). When the probe grid subsystem is enabled,
// the real probe storage buffer and grid params uniform are bound. When probes are
// disabled or absent, minimal fallback buffers are created with total_probes = 0
// so the shader reads no indirect illumination, keeping the bind group layout valid.
func (s *scene) initProbesLitBindGroup(litFragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || litFragmentShader == nil {
		return
	}

	// Resolve the probes bind group index from the lit fragment shader's annotations.
	probesGroup := -1
	for _, decl := range litFragmentShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil &&
			(decl.Args[2] == shader.AnnotationArgIrradianceProbe || decl.Args[2] == shader.AnnotationArgProbeGridParams) {
			probesGroup = *decl.Group
			break
		}
	}
	if probesGroup < 0 {
		return
	}

	bgp := s.lightHandler.Bgp("probe_lit")
	desc := litFragmentShader.BindGroupLayoutDescriptor(probesGroup)

	probesReady := s.lightHandler.ProbeGrid() != nil && s.lightHandler.ProbeGrid().Enabled() &&
		s.lightHandler.ProbeGrid().ProbeBuffer() != nil && s.lightHandler.ProbeGrid().GridParamsBuffer() != nil

	if probesReady {
		// Bind the real probe storage buffer and grid params uniform from the probe grid handler.
		for _, entry := range desc.Entries {
			binding := int(entry.Binding)
			if bufferBindingType(entry) == gputypes.BufferBindingTypeReadOnlyStorage || bufferBindingType(entry) == gputypes.BufferBindingTypeStorage {
				bgp.SetBuffer(binding, s.lightHandler.ProbeGrid().ProbeBuffer())
			} else if bufferBindingType(entry) == gputypes.BufferBindingTypeUniform {
				bgp.SetBuffer(binding, s.lightHandler.ProbeGrid().GridParamsBuffer())
			}
		}
	} else {
		// Create fallback buffers. The probe storage buffer gets a single zeroed
		// probe (160 bytes). The grid params uniform gets total_probes = 0 so the
		// shader early-returns vec3(0.0) for indirect illumination.
		for _, entry := range desc.Entries {
			binding := int(entry.Binding)
			if bufferBindingType(entry) == gputypes.BufferBindingTypeReadOnlyStorage || bufferBindingType(entry) == gputypes.BufferBindingTypeStorage {
				fallbackProbe := (&light.GPUIrradianceProbe{}).Marshal()
				buf, bufErr := s.r.CreateBuffer("Probe Lit Fallback Storage", uint64(len(fallbackProbe)), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
				if bufErr != nil {
					panic(fmt.Sprintf("scene: failed to create probe lit fallback storage: %v", bufErr))
				}
				bgp.SetBuffer(binding, buf)
				s.r.WriteRawBuffer(buf, 0, fallbackProbe)
			} else if bufferBindingType(entry) == gputypes.BufferBindingTypeUniform {
				fallbackParams := (&light.GPUProbeGridParams{}).Marshal()
				buf, bufErr := s.r.CreateBuffer("Probe Lit Fallback Uniform", uint64(len(fallbackParams)), wgpu.BufferUsageUniform|wgpu.BufferUsageCopyDst)
				if bufErr != nil {
					panic(fmt.Sprintf("scene: failed to create probe lit fallback uniform: %v", bufErr))
				}
				bgp.SetBuffer(binding, buf)
				s.r.WriteRawBuffer(buf, 0, fallbackParams)
			}
		}
	}

	if err := s.r.InitBindGroup(bgp, desc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init probes lit bind group: %v", err))
	}
}

func (s *scene) PrepareProbes() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	pg := s.lightHandler.ProbeGrid()
	if pg == nil || !pg.Enabled() || s.r == nil {
		return
	}

	dirty := pg.DirtyProbes()
	if len(dirty) == 0 {
		return
	}

	// Rate-limit: bake at most 4 probes per frame to amortize cost.
	const maxProbesPerFrame = 4
	batch := dirty
	if len(batch) > maxProbesPerFrame {
		batch = batch[:maxProbesPerFrame]
	}

	colorView := pg.BakeColorTextureView()
	depthView := pg.BakeDepthTextureView()
	if colorView == nil || depthView == nil {
		return
	}

	bakeCameraBGP := pg.Bgp("probe_bake_camera")
	shProjectBGP := pg.Bgp("probe_sh_project")
	if bakeCameraBGP == nil || shProjectBGP == nil {
		return
	}

	resolution := pg.BakeResolution()
	probeSize := (&light.GPUIrradianceProbe{}).Size()

	// Standard cubemap face directions and up vectors.
	type cubeFace struct {
		dirX, dirY, dirZ float32
		upX, upY, upZ    float32
	}
	faces := [6]cubeFace{
		{1, 0, 0, 0, -1, 0},  // +X
		{-1, 0, 0, 0, -1, 0}, // -X
		{0, 1, 0, 0, 0, 1},   // +Y
		{0, -1, 0, 0, 0, -1}, // -Y
		{0, 0, 1, 0, -1, 0},  // +Z
		{0, 0, -1, 0, -1, 0}, // -Z
	}

	// Perspective projection: 90° FOV, square aspect, shared across all faces.
	proj := make([]float32, 16)
	common.Perspective(proj, float32(math.Pi/2.0), 1.0, 0.1, 100.0)

	view := make([]float32, 16)
	vp := make([]float32, 16)

	staticBakeKey := pg.PipelineKey("static")
	skinnedBakeKey := pg.PipelineKey("skinned")
	shProjectKey := pg.PipelineKey("sh_project")

	for _, probeIdx := range batch {
		probe := pg.Probe(probeIdx)
		px, py, pz := probe.Position[0], probe.Position[1], probe.Position[2]

		// Zero the probe's SH coefficients before accumulating across 6 faces.
		// SH data starts at byte offset 16 within each probe (after position vec4).
		shOffset := uint64(probeIdx*probeSize + 16)
		shSize := probeSize - 16
		s.r.WriteRawBuffer(pg.ProbeBuffer(), shOffset, make([]byte, shSize))

		for face := 0; face < 6; face++ {
			f := faces[face]

			// Build view-projection matrix for this cubemap face.
			common.LookAt(view, px, py, pz, px+f.dirX, py+f.dirY, pz+f.dirZ, f.upX, f.upY, f.upZ)
			common.Mul4(vp, proj, view)

			bakeCam := light.GPUProbeBakeCamera{
				CameraPosition: [3]float32{px, py, pz},
			}
			copy(bakeCam.ViewProj[:], vp)

			// Upload bake camera uniform for this face.
			s.r.WriteBuffers([]bind_group_provider.BufferWrite{
				{Provider: bakeCameraBGP, Binding: 0, Offset: 0, Data: bakeCam.Marshal()},
			})

			// Render the scene from this cubemap face into the bake texture.
			if err := s.r.BeginProbeBakeFrame(); err != nil {
				continue
			}
			s.r.BeginProbeBakePass(colorView, depthView)

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

					pipeKey := staticBakeKey
					if mdl.Skinned() {
						pipeKey = skinnedBakeKey
					}
					if pipeKey == "" {
						continue
					}

					mats := mdl.RenderMaterials()
					if len(mats) == 0 {
						continue
					}
					matBGP := mats[0].BindGroupProvider()
					if matBGP == nil {
						continue
					}

					bakeGroups := []bind_group_provider.BindGroupProvider{
						bakeCameraBGP,
						a.OutputBindGroupProvider(),
						matBGP,
					}

					_ = s.r.ProbeBakeDrawCall(pipeKey, meshProvider, uint32(a.InstanceCount()), bakeGroups)
				}
			}

			s.r.EndProbeBakePass()
			s.r.EndProbeBakeFrame()

			// Dispatch SH projection compute for this face. The compute shader
			// accumulates L2 SH coefficients into the probe storage buffer.
			shParams := light.GPUSHProjectParams{
				ProbeIndex: uint32(probeIdx),
				FaceIndex:  uint32(face),
				Resolution: uint32(resolution),
			}
			s.r.WriteBuffers([]bind_group_provider.BufferWrite{
				{Provider: shProjectBGP, Binding: 2, Offset: 0, Data: shParams.Marshal()},
			})

			if err := s.r.BeginComputeFrame(); err != nil {
				continue
			}
			s.r.DispatchCompute(shProjectKey, shProjectBGP, [3]uint32{1, 1, 1})
			s.r.EndComputeFrame()
		}
	}

	// Remove the processed probes from the dirty list.
	remaining := pg.DirtyProbes()
	if len(batch) >= len(remaining) {
		pg.ClearDirtyProbes()
	} else {
		pg.SetDirtyProbes(remaining[len(batch):])
	}
}

func (s *scene) PrepareGBuffer() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.lightHandler.GBufferHandler() == nil || !s.lightHandler.GBufferHandler().Enabled() || s.r == nil {
		return
	}

	if err := s.r.BeginGBufferFrame(); err != nil {
		return
	}
	s.r.BeginGBufferPass(
		s.lightHandler.GBufferHandler().NormalTextureView(),
		s.lightHandler.GBufferHandler().AlbedoTextureView(),
		s.lightHandler.GBufferHandler().DepthTextureView(),
	)

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

			// Select the appropriate G-Buffer pipeline based on whether the model is skinned.
			pipeKey := s.lightHandler.GBufferHandler().PipelineKey("static")
			if mdl.Skinned() {
				pipeKey = s.lightHandler.GBufferHandler().PipelineKey("skinned")
			}
			if pipeKey == "" {
				continue
			}

			// Build bind groups for the G-Buffer pass:
			//   group(0) = camera BGP
			//   group(1) = output BGP (instance/bone matrices from compute shader)
			//   group(2) = material BGP
			var cameraBGP bind_group_provider.BindGroupProvider
			if s.cam != nil {
				cameraBGP = s.cam.BindGroupProvider()
			}
			if cameraBGP == nil {
				continue
			}

			mats := mdl.RenderMaterials()
			if len(mats) == 0 {
				continue
			}
			matBGP := mats[0].BindGroupProvider()
			if matBGP == nil {
				continue
			}

			gbufferBindGroups := []bind_group_provider.BindGroupProvider{
				cameraBGP,
				a.OutputBindGroupProvider(),
				matBGP,
			}

			// Use indirect draw when GPU frustum culling is active.
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
								_ = s.r.GBufferDrawCallIndirect(pipeKey, meshProvider, indBuf, gbufferBindGroups)
								continue
							}
						}
					}
				}
			}

			_ = s.r.GBufferDrawCall(pipeKey, meshProvider, uint32(a.InstanceCount()), gbufferBindGroups)
		}
	}

	s.r.EndGBufferPass()
	s.r.EndGBufferFrame()
}

func (s *scene) PrepareSSAO() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.lightHandler.SSAOHandler() == nil || !s.lightHandler.SSAOHandler().Enabled() || s.r == nil || s.cam == nil {
		return
	}

	w := s.lightHandler.SSAOHandler().ScreenWidth()
	h := s.lightHandler.SSAOHandler().ScreenHeight()

	// Compute the G-Buffer coordinate scale: 2.0 when running at half-resolution,
	// 1.0 at full resolution. The SSAO compute and blur shaders multiply their
	// texture coordinates by this value when reading from full-res G-Buffer textures.
	var gbufferScale float32 = 1.0
	var gbufferScaleI int32 = 1
	if s.lightHandler.SSAOHandler().HalfResolution() {
		gbufferScale = 2.0
		gbufferScaleI = 2
	}

	// Compute the inverse view-projection matrix for depth-to-world reconstruction.
	vp := s.cam.ViewProjectionMatrix()
	var invVP [16]float32
	common.Invert4(invVP[:], vp[:])

	// Get camera position from the controller.
	var camPos [3]float32
	if ctrl := s.cam.Controller(); ctrl != nil {
		camPos[0], camPos[1], camPos[2] = ctrl.Position()
	}

	// Build and write SSAO uniform parameters.
	ssaoParams := light.GPUSSAOParams{
		Projection:     s.cam.ViewProjectionMatrix(),
		InvViewProj:    invVP,
		Radius:         s.lightHandler.SSAOHandler().Radius(),
		Bias:           s.lightHandler.SSAOHandler().Bias(),
		Power:          s.lightHandler.SSAOHandler().Power(),
		SampleCount:    uint32(s.lightHandler.SSAOHandler().SampleCount()),
		ScreenWidth:    float32(w),
		ScreenHeight:   float32(h),
		GBufferScale:   gbufferScale,
		CameraPosition: camPos,
	}

	ssaoBGP := s.lightHandler.SSAOHandler().Bgp("ssao_compute")
	blurHBGP := s.lightHandler.SSAOHandler().Bgp("ssao_blur_h")
	blurVBGP := s.lightHandler.SSAOHandler().Bgp("ssao_blur_v")

	// Write SSAO params to the compute BGP uniform buffer.
	hParams := light.GPUBlurParams{
		Direction:    [2]int32{1, 0},
		Radius:       int32(s.lightHandler.SSAOHandler().BlurRadius()),
		GBufferScale: gbufferScaleI,
	}
	vParams := light.GPUBlurParams{
		Direction:    [2]int32{0, 1},
		Radius:       int32(s.lightHandler.SSAOHandler().BlurRadius()),
		GBufferScale: gbufferScaleI,
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: ssaoBGP, Binding: 4, Offset: 0, Data: ssaoParams.Marshal()},
		{Provider: blurHBGP, Binding: 2, Offset: 0, Data: hParams.Marshal()},
		{Provider: blurVBGP, Binding: 2, Offset: 0, Data: vParams.Marshal()},
	})

	// Dispatch SSAO compute + bilateral blur.
	workGroupsX := uint32((w + 15) / 16)
	workGroupsY := uint32((h + 15) / 16)
	wg := [3]uint32{workGroupsX, workGroupsY, 1}

	if err := s.r.BeginComputeFrame(); err != nil {
		return
	}
	s.r.DispatchCompute(s.lightHandler.SSAOHandler().PipelineKey("ssao_compute"), ssaoBGP, wg)
	s.r.DispatchCompute(s.lightHandler.SSAOHandler().PipelineKey("ssao_blur"), blurHBGP, wg)
	s.r.DispatchCompute(s.lightHandler.SSAOHandler().PipelineKey("ssao_blur"), blurVBGP, wg)
	s.r.EndComputeFrame()
}

func (s *scene) PrepareSSR() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ssrHandler := s.lightHandler.SSRHandler()
	compHandler := s.lightHandler.CompositionHandler()
	if ssrHandler == nil || !ssrHandler.Enabled() || compHandler == nil || !compHandler.Enabled() || s.r == nil || s.cam == nil {
		return
	}

	w := ssrHandler.ScreenWidth()
	h := ssrHandler.ScreenHeight()

	// Build and write SSR uniform parameters.
	ssrParams := light.GPUSSRParams{
		Projection:      s.cam.ProjectionMatrix(),
		InvProjection:   s.cam.InverseProjectionMatrix(),
		View:            s.cam.ViewMatrix(),
		MaxDistance:     ssrHandler.MaxDistance(),
		Thickness:       ssrHandler.Thickness(),
		Stride:          ssrHandler.Stride(),
		MaxSteps:        uint32(ssrHandler.MaxSteps()),
		ScreenWidth:     float32(w),
		ScreenHeight:    float32(h),
		RoughnessCutoff: ssrHandler.RoughnessCutoff(),
		HiZMipCount:     uint32(ssrHandler.HiZMipCount()),
	}

	ssrBGP := ssrHandler.Bgp("ssr_compute")
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: ssrBGP, Binding: 0, Offset: 0, Data: ssrParams.Marshal()},
	})

	if err := s.r.BeginComputeFrame(); err != nil {
		return
	}

	// --- Hi-Z depth pyramid generation ---
	mipCount := ssrHandler.HiZMipCount()

	// Pass 0: copy GBuffer depth → Hi-Z mip 0 (full resolution).
	hizInitBGP := ssrHandler.Bgp("hiz_init")
	initWGX := uint32((w + 7) / 8)
	initWGY := uint32((h + 7) / 8)
	s.r.DispatchCompute(ssrHandler.PipelineKey("hiz_init"), hizInitBGP, [3]uint32{initWGX, initWGY, 1})

	// Passes 1..N-1: min-downsample each mip level from the previous.
	mipW := w
	mipH := h
	for i := 1; i < mipCount; i++ {
		mipW = max(mipW/2, 1)
		mipH = max(mipH/2, 1)
		bgp := ssrHandler.Bgp(fmt.Sprintf("hiz_down_%d", i))
		wgX := uint32((mipW + 7) / 8)
		wgY := uint32((mipH + 7) / 8)
		s.r.DispatchCompute(ssrHandler.PipelineKey("hiz_downsample"), bgp, [3]uint32{wgX, wgY, 1})
	}

	// --- SSR compute dispatch at half-resolution ---
	halfW := w / 2
	halfH := h / 2
	if halfW <= 0 {
		halfW = 1
	}
	if halfH <= 0 {
		halfH = 1
	}
	workGroupsX := uint32((halfW + 7) / 8)
	workGroupsY := uint32((halfH + 7) / 8)
	s.r.DispatchCompute(ssrHandler.PipelineKey("ssr_compute"), ssrBGP, [3]uint32{workGroupsX, workGroupsY, 1})

	s.r.EndComputeFrame()
}

func (s *scene) PrepareComposition() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	compHandler := s.lightHandler.CompositionHandler()
	if compHandler == nil || !compHandler.Enabled() || s.r == nil {
		return
	}

	// Write composition params uniform.
	compParams := light.GPUCompositionParams{
		Exposure: compHandler.Exposure(),
	}
	if compHandler.ToneMappingEnabled() {
		compParams.ToneMappingEnabled = 1
	}
	compBGP := compHandler.Bgp("composition")
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: compBGP, Binding: 4, Offset: 0, Data: compParams.Marshal()},
	})

	// Run the fullscreen composition pass: acquire swapchain → render → submit.
	if err := s.r.BeginCompositionFrame(); err != nil {
		return
	}
	s.r.BeginCompositionPass()
	_ = s.r.CompositionDrawCall(compHandler.PipelineKey("composition"), []bind_group_provider.BindGroupProvider{compBGP})
	s.r.EndCompositionPass()
	s.r.EndCompositionFrame()
}

func (s *scene) BeginHDRFrame() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.r == nil {
		return fmt.Errorf("scene %q: composition not initialized", s.name)
	}

	ch := s.lightHandler.CompositionHandler()
	if ch == nil || !ch.Enabled() {
		return s.r.BeginFrame()
	}

	sampleCount := s.r.SampleCount()
	if sampleCount > 1 && ch.MSAATextureView() != nil {
		// MSAA active: render to multi-sampled texture, resolve into HDR.
		return s.r.BeginHDRFrame(ch.MSAATextureView(), ch.HDRTextureView(), ch.DepthTextureView(), sampleCount)
	}
	// No MSAA: render directly to HDR texture.
	return s.r.BeginHDRFrame(ch.HDRTextureView(), nil, ch.DepthTextureView(), 1)
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
		if bufferBindingType(entry) == gputypes.BufferBindingTypeReadOnlyStorage || bufferBindingType(entry) == gputypes.BufferBindingTypeStorage {
			// Storage buffer: size it for max lights (header is in a separate uniform binding).
			sizeOverrides[binding] = uint64(light.MaxGPULights * (&light.GPULight{}).Size())
		}
	}

	if err := s.r.InitBindGroup(bgp, descriptor, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init light bind group: %v", err))
	}
}

// initVSMShadowMap initializes the variance shadow mapping resources for the scene.
// This creates the RG32Float moments texture, auxiliary depth texture, scratch blur
// texture, linear sampler, VSM shadow render pipelines (with fragment shader), and
// the separable blur compute pipeline with its horizontal/vertical bind groups.
func (s *scene) initVSMShadowMap(vsmVertShader, vsmSkinnedVertShader, vsmFragShader, blurComputeShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || vsmVertShader == nil || vsmFragShader == nil {
		return
	}

	res := s.lightHandler.ShadowMapResolution()

	// Create VSM textures: RG32Float moments, RG32Float scratch (blur), Depth32Float aux.
	vsmView, vsmTex, scratchView, scratchTex, depthView, depthTex, err := s.r.CreateVSMTextures(res, res)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create VSM textures: %v", err))
	}
	s.lightHandler.SetVSMTexture(vsmTex)
	s.lightHandler.SetVSMTextureView(vsmView)
	s.lightHandler.SetVSMScratchTexture(scratchTex)
	s.lightHandler.SetVSMScratchTextureView(scratchView)
	s.lightHandler.SetVSMAuxDepthTexture(depthTex)
	s.lightHandler.SetVSMAuxDepthTextureView(depthView)

	// Create linear sampler for the lit fragment shader's VSM texture lookup.
	linearSamp, err := s.r.CreateLinearSampler()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create VSM linear sampler: %v", err))
	}
	s.lightHandler.SetVSMLinearSampler(linearSamp)

	// Create shadow data BGP — holds the light VP uniform for the VSM vertex shader.
	shadowGroup := 0
	for _, decl := range vsmVertShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgShadowUniform {
			shadowGroup = *decl.Group
			break
		}
	}
	bgp := s.lightHandler.Bgp("shadow_data")
	desc := vsmVertShader.BindGroupLayoutDescriptor(shadowGroup)
	sizeOverrides := make(map[int]uint64)
	for _, entry := range desc.Entries {
		if bufferBindingType(entry) == gputypes.BufferBindingTypeUniform {
			sizeOverrides[int(entry.Binding)] = uint64((&light.GPUShadowData{}).Size())
		}
	}
	if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init VSM shadow data bind group: %v", err))
	}

	// Register VSM shadow pipelines for each ShadowCullMode variant.
	// Unlike the PCF path, these pipelines include a fragment shader that outputs
	// depth moments to an RG32Float color target and apply no hardware depth bias.
	cullModes := []struct {
		mode model.ShadowCullMode
		wgpu wgpu.CullMode
		tag  string
	}{
		{model.ShadowCullModeBack, gputypes.CullModeBack, "back"},
		{model.ShadowCullModeFront, gputypes.CullModeFront, "front"},
		{model.ShadowCullModeNone, gputypes.CullModeNone, "none"},
	}

	for _, cm := range cullModes {
		key := "vsm_shadow_static_" + cm.tag
		sp := pipeline.NewPipeline(key, pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vsmVertShader),
			pipeline.WithFragmentShader(vsmFragShader),
			pipeline.WithCullMode(cm.wgpu),
		)
		if err := s.r.RegisterVSMShadowPipeline(sp); err != nil {
			panic(fmt.Sprintf("scene: failed to register VSM static shadow pipeline (%s): %v", cm.tag, err))
		}
		s.lightHandler.SetPipelineKey("shadow_static_"+cm.tag, key)
	}

	if vsmSkinnedVertShader != nil {
		for _, cm := range cullModes {
			key := "vsm_shadow_skinned_" + cm.tag
			ssp := pipeline.NewPipeline(key, pipeline.PipelineTypeRender,
				pipeline.WithVertexShader(vsmSkinnedVertShader),
				pipeline.WithFragmentShader(vsmFragShader),
				pipeline.WithCullMode(cm.wgpu),
			)
			if err := s.r.RegisterVSMShadowPipeline(ssp); err != nil {
				panic(fmt.Sprintf("scene: failed to register VSM skinned shadow pipeline (%s): %v", cm.tag, err))
			}
			s.lightHandler.SetPipelineKey("shadow_skinned_"+cm.tag, key)
		}
	}

	// Register the separable blur compute pipeline.
	if blurComputeShader != nil {
		blurPipeKey := "vsm_blur_compute"
		blurPipe := pipeline.NewPipeline(blurPipeKey, pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(blurComputeShader),
		)
		if err := s.r.RegisterPipelines(blurPipe); err != nil {
			panic(fmt.Sprintf("scene: failed to register VSM blur compute pipeline: %v", err))
		}
		s.lightHandler.SetPipelineKey("vsm_blur", blurPipeKey)

		// Initialize blur bind groups. Each pass swaps input/output textures.
		blurDesc := blurComputeShader.BindGroupLayoutDescriptor(0)
		blurSizeOverrides := map[int]uint64{
			2: uint64((&light.GPUBlurParams{}).Size()),
		}

		// Horizontal: read from VSM → write to scratch.
		blurHBGP := s.lightHandler.Bgp("vsm_blur_h")
		blurHBGP.SetTextureView(0, vsmView)
		blurHBGP.SetTextureView(1, scratchView)
		if err := s.r.InitBindGroup(blurHBGP, blurDesc, nil, blurSizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init VSM blur horizontal bind group: %v", err))
		}

		// Vertical: read from scratch → write back to VSM.
		blurVBGP := s.lightHandler.Bgp("vsm_blur_v")
		blurVBGP.SetTextureView(0, scratchView)
		blurVBGP.SetTextureView(1, vsmView)
		if err := s.r.InitBindGroup(blurVBGP, blurDesc, nil, blurSizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init VSM blur vertical bind group: %v", err))
		}
	}
}

// initSATResources initializes the Summed-Area Table resources required for PCSS.
// Creates two RGBA32Float ping-pong textures, registers the SAT compute pipeline,
// and pre-creates one bind group provider per dispatch pass. Each BGP has its own
// uniform buffer so that all params can be written upfront and every dispatch can
// be batched into a single command encoder submission.
func (s *scene) initSATResources(satComputeShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || satComputeShader == nil {
		return
	}

	res := s.lightHandler.ShadowMapResolution()

	// Create two RGBA32Float textures for SAT ping-pong.
	satAView, satATex, satBView, satBTex, err := s.r.CreateSATTextures(res, res)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create SAT textures: %v", err))
	}
	s.lightHandler.SetSATTextureA(satATex)
	s.lightHandler.SetSATTextureAView(satAView)
	s.lightHandler.SetSATTextureB(satBTex)
	s.lightHandler.SetSATTextureBView(satBView)

	// Register the SAT compute pipeline.
	satPipeKey := "vsm_sat_compute"
	satPipe := pipeline.NewPipeline(satPipeKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(satComputeShader),
	)
	if err := s.r.RegisterPipelines(satPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register SAT compute pipeline: %v", err))
	}
	s.lightHandler.SetPipelineKey("vsm_sat", satPipeKey)

	// Shared BGL descriptor and size overrides for all SAT bind groups.
	satDesc := satComputeShader.BindGroupLayoutDescriptor(0)
	satSizeOverrides := map[int]uint64{
		2: uint64((&light.GPUSATParams{}).Size()),
	}

	// Pass 0 — prepare: reads VSM moments (RG32Float), writes SAT A (RGBA32Float).
	prepareBGP := s.lightHandler.Bgp("sat_prepare")
	prepareBGP.SetTextureView(0, s.lightHandler.VSMTextureView())
	prepareBGP.SetTextureView(1, satAView)
	if err := s.r.InitBindGroup(prepareBGP, satDesc, nil, satSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SAT prepare bind group: %v", err))
	}

	// Pre-create one BGP per prefix-sum pass. Each pass reads from one SAT
	// texture and writes to the other (ping-pong). Giving each pass its own
	// BGP (and thus its own uniform buffer) allows all params to be written
	// upfront and all dispatches to be batched into a single command encoder,
	// avoiding the overhead of 23 separate GPU submissions.
	numPasses := 0
	for v := res; v > 1; v >>= 1 {
		numPasses++
	}

	readFromA := true // after prepare, data is in SAT A
	passIdx := 0
	for axis := 0; axis < 2; axis++ {
		for k := 0; k < numPasses; k++ {
			bgpName := fmt.Sprintf("sat_pass_%d", passIdx)
			bgp := s.lightHandler.Bgp(bgpName)
			if readFromA {
				bgp.SetTextureView(0, satAView)
				bgp.SetTextureView(1, satBView)
			} else {
				bgp.SetTextureView(0, satBView)
				bgp.SetTextureView(1, satAView)
			}
			if err := s.r.InitBindGroup(bgp, satDesc, nil, satSizeOverrides); err != nil {
				panic(fmt.Sprintf("scene: failed to init SAT pass %d bind group: %v", passIdx, err))
			}
			readFromA = !readFromA
			passIdx++
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
// shaders use to sample the shadow map. Supports VSM (texture_2d<f32> + linear
// sampler) and PCSS (SAT RGBA32Float texture + linear sampler) modes.
func (s *scene) initShadowLitBindGroup(litFragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || litFragmentShader == nil {
		return
	}

	// Validate that the appropriate shadow resources are available.
	pcssMode := s.lightHandler.PCSSEnabled()
	if pcssMode {
		if s.lightHandler.SATTextureAView() == nil || s.lightHandler.VSMLinearSampler() == nil {
			return // initSATResources must be called first
		}
	} else {
		if s.lightHandler.VSMTextureView() == nil || s.lightHandler.VSMLinearSampler() == nil {
			return // initVSMShadowMap must be called first
		}
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

	// Pre-set the shadow texture view and sampler on the BGP so that
	// InitBindGroup can find them when creating the bind group entries.
	// PCSS uses the SAT texture + linear sampler; VSM uses the moments
	// texture + linear sampler.
	desc := litFragmentShader.BindGroupLayoutDescriptor(shadowGroup)

	for _, entry := range desc.Entries {
		binding := int(entry.Binding)
		if isTextureBindingEntry(entry) {
			if pcssMode {
				bgp.SetTextureView(binding, s.lightHandler.SATTextureAView())
			} else {
				bgp.SetTextureView(binding, s.lightHandler.VSMTextureView())
			}
		}
		if isSamplerBindingEntry(entry) {
			bgp.SetSampler(binding, s.lightHandler.VSMLinearSampler())
		}
	}

	// Override the uniform buffer size to GPUShadowData's size.
	sizeOverrides := make(map[int]uint64)
	for _, entry := range desc.Entries {
		if bufferBindingType(entry) == gputypes.BufferBindingTypeUniform {
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
		TexelSize:           [2]float32{texelSize, texelSize},
		Bias:                s.lightHandler.ShadowBias(),
		MinVariance:         s.lightHandler.VSMMinVariance(),
		LightBleedReduction: s.lightHandler.VSMLightBleedReduction(),
		LightSize:           s.lightHandler.VSMLightSize(),
		ShadowHalfExtent:    s.lightHandler.ShadowHalfExtent(),
	}
	shadowData.ComputeDirectionalLightVP(
		shadowLight.Direction(),
		centerX, centerY, centerZ,
		s.lightHandler.ShadowHalfExtent(), s.lightHandler.ShadowNear(), s.lightHandler.ShadowFar(),
	)
	shadowData.ComputeNormalBias(s.lightHandler.ShadowHalfExtent(), s.lightHandler.ShadowNormalBiasScale(), s.lightHandler.ShadowMapResolution())

	// Build a separate GPUShadowUniform for the depth-pass vertex shader.
	// GPUShadowUniform and GPUShadowData have different struct layouts at offset 128+
	// (ShadowUniform: shadow_near/shadow_far; ShadowData: texel_size), so each BGP
	// must receive the correctly laid-out bytes to avoid misinterpreted fields.
	shadowUniform := light.GPUShadowUniform{
		LightVP:    shadowData.LightVP,
		LightView:  shadowData.LightView,
		ShadowNear: shadowData.ShadowNear,
		ShadowFar:  shadowData.ShadowFar,
	}

	shadowDataBGP := s.lightHandler.Bgp("shadow_data")
	writes := []bind_group_provider.BufferWrite{
		{
			Provider: shadowDataBGP,
			Binding:  0,
			Offset:   0,
			Data:     shadowUniform.Marshal(),
		},
	}
	// Also write the full GPUShadowData to the lit-pass shadow BGP (different layout).
	shadowLitBGP := s.lightHandler.Bgp("shadow_lit")
	shadowBytes := shadowData.Marshal()
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

	// Execute shadow depth pass — VSM targets a color+depth pass.
	if err := s.r.BeginShadowFrame(); err != nil {
		return
	}
	s.r.BeginVSMShadowPass(s.lightHandler.VSMTextureView(), s.lightHandler.VSMAuxDepthTextureView())

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

func (s *scene) PrepareShadowBlur() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.lightHandler.Enabled() || s.r == nil {
		return
	}

	// Post-processing: either blur passes (constant-width VSM) or SAT generation (PCSS variable-width).
	res := s.lightHandler.ShadowMapResolution()
	workGroupsX := uint32((res + 15) / 16)
	workGroupsY := uint32((res + 15) / 16)

	pcssMode := s.lightHandler.PCSSEnabled()
	if pcssMode {
		// PCSS path: generate a Summed-Area Table from the raw moments texture.
		// The SAT replaces the blur pass — variable-width filtering is done in the lit shader.
		s.dispatchSATGeneration(res, workGroupsX, workGroupsY)
	} else {
		// Standard VSM path: separable blur for constant-width soft shadows.
		blurHBGP := s.lightHandler.Bgp("vsm_blur_h")
		blurVBGP := s.lightHandler.Bgp("vsm_blur_v")

		hParams := light.GPUBlurParams{
			Direction:    [2]int32{1, 0},
			Radius:       int32(s.lightHandler.VSMBlurRadius()),
			GBufferScale: 1,
		}
		vParams := light.GPUBlurParams{
			Direction:    [2]int32{0, 1},
			Radius:       int32(s.lightHandler.VSMBlurRadius()),
			GBufferScale: 1,
		}
		s.r.WriteBuffers([]bind_group_provider.BufferWrite{
			{Provider: blurHBGP, Binding: 2, Offset: 0, Data: hParams.Marshal()},
			{Provider: blurVBGP, Binding: 2, Offset: 0, Data: vParams.Marshal()},
		})

		blurPipeKey := s.lightHandler.PipelineKey("vsm_blur")
		if err := s.r.BeginComputeFrame(); err == nil {
			s.r.DispatchCompute(blurPipeKey, blurHBGP, [3]uint32{workGroupsX, workGroupsY, 1})
			s.r.DispatchCompute(blurPipeKey, blurVBGP, [3]uint32{workGroupsX, workGroupsY, 1})
			s.r.EndComputeFrame()
		}
	}
}

// dispatchSATGeneration runs the complete Summed-Area Table generation pipeline:
// one precision-distribution prepare pass followed by 2×log₂(res) recursive-doubling
// prefix-sum passes (horizontal then vertical), ping-ponging between SAT textures A and B.
// After completion, the final SAT resides in SAT texture A.
//
// Because each prefix-sum pass was pre-created with its own BGP and uniform buffer during
// initSATResources, all params are written upfront via a single WriteBuffers call, and
// every dispatch is batched into one command encoder submission. This avoids the overhead
// of 23 separate GPU submissions per frame.
func (s *scene) dispatchSATGeneration(res int, workGroupsX, workGroupsY uint32) {
	satPipeKey := s.lightHandler.PipelineKey("vsm_sat")

	numPasses := 0
	for v := res; v > 1; v >>= 1 {
		numPasses++
	}

	totalPrefixPasses := 2 * numPasses
	wg := [3]uint32{workGroupsX, workGroupsY, 1}

	// Collect all writes for the prepare pass and every prefix-sum pass.
	writes := make([]bind_group_provider.BufferWrite, 0, 1+totalPrefixPasses)

	// Pass 0: prepare — precision distribution.
	prepareBGP := s.lightHandler.Bgp("sat_prepare")
	writes = append(writes, bind_group_provider.BufferWrite{
		Provider: prepareBGP, Binding: 2, Offset: 0,
		Data: (&light.GPUSATParams{Direction: [2]int32{0, 0}, Offset: 0}).Marshal(),
	})

	// Build params for every prefix-sum pass.
	passIdx := 0
	for axis := 0; axis < 2; axis++ {
		dir := [2]int32{0, 0}
		dir[axis] = 1
		for k := 0; k < numPasses; k++ {
			bgp := s.lightHandler.Bgp(fmt.Sprintf("sat_pass_%d", passIdx))
			writes = append(writes, bind_group_provider.BufferWrite{
				Provider: bgp, Binding: 2, Offset: 0,
				Data: (&light.GPUSATParams{Direction: dir, Offset: int32(1) << uint(k)}).Marshal(),
			})
			passIdx++
		}
	}

	// Single WriteBuffers call writes all uniform data to separate GPU buffers.
	s.r.WriteBuffers(writes)

	// Single command encoder batches every dispatch.
	if err := s.r.BeginComputeFrame(); err != nil {
		return
	}

	// Dispatch prepare.
	s.r.DispatchCompute(satPipeKey, prepareBGP, wg)

	// Dispatch all prefix-sum passes.
	for i := 0; i < totalPrefixPasses; i++ {
		bgp := s.lightHandler.Bgp(fmt.Sprintf("sat_pass_%d", i))
		s.r.DispatchCompute(satPipeKey, bgp, wg)
	}

	s.r.EndComputeFrame()
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

// initComposition initializes the composition and tone mapping subsystem: creates
// offscreen HDR render targets (with optional MSAA textures), a linear sampler,
// registers the fullscreen composition pipeline, and creates the composition bind
// group provider with pre-set texture views. The G-Buffer must be initialized
// before calling this method so that SSR textures are available for binding.
func (s *scene) initComposition() {
	s.mu.Lock()
	defer s.mu.Unlock()

	ch := s.lightHandler.CompositionHandler()
	if s.r == nil || ch == nil {
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
	s.r.SetRenderTargetFormat(gputypes.TextureFormatRGBA16Float)

	// 2. Create linear sampler for HDR and SSR texture sampling.
	linearSamp, err := s.r.CreateLinearSampler()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create composition linear sampler: %v", err))
	}
	ch.SetLinearSampler(linearSamp)

	// 3. Load composition shaders and register the fullscreen pipeline.
	compVert := shader.NewShader("_composition_vert", shader.ShaderTypeVertex, "engine/light/assets/composition-vert.wgsl")
	compFrag := shader.NewShader("_composition_frag", shader.ShaderTypeFragment, "engine/light/assets/composition-frag.wgsl")

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

	sizeOverrides := map[int]uint64{
		4: uint64((&light.GPUCompositionParams{}).Size()),
	}
	if err := s.r.InitBindGroup(compBGP, compDesc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init composition bind group: %v", err))
	}

	ch.Resize(w, h)
	ch.SetEnabled(true)
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
	if s.r == nil || ssrHandler == nil || gbHandler == nil || compHandler == nil {
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
	hizInitShader := shader.NewShader("_hiz_init", shader.ShaderTypeCompute, "engine/light/assets/hiz-init-compute.wgsl")
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
	hizDownShader := shader.NewShader("_hiz_downsample", shader.ShaderTypeCompute, "engine/light/assets/hiz-downsample-compute.wgsl")
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
	ssrCompShader := shader.NewShader("_ssr_compute", shader.ShaderTypeCompute, "engine/light/assets/ssr-compute.wgsl")
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
// standard light shader assets. When VSM is enabled on the lighting handler, the
// VSM-specific shaders and resources are used instead of the PCF defaults.
func (s *scene) initLighting(screenWidth, screenHeight int) {
	pcssMode := s.lightHandler.PCSSEnabled()

	// Select the correct lit fragment shader variant based on shadow mode.
	var litFragShader shader.Shader
	if pcssMode {
		litFragShader = shader.NewShader("_lit_frag_pcss", shader.ShaderTypeFragment, "engine/light/assets/lit-frag-pcss.wgsl")
	} else {
		litFragShader = shader.NewShader("_lit_frag_vsm", shader.ShaderTypeFragment, "engine/light/assets/lit-frag-vsm.wgsl")
	}
	cullComputeShader := shader.NewShader("_light_cull_compute", shader.ShaderTypeCompute, "engine/light/assets/light-cull-compute.wgsl")

	// 1. Light storage buffer (must be first — other steps share this buffer).
	s.initLightBindGroup(litFragShader)

	// 2. Shadow resources — VSM creates moments+scratch textures, a linear sampler,
	// VSM render pipelines, and the separable blur compute pipeline.
	// When PCSS is enabled, the SAT compute pipeline and textures are additionally created.
	vsmVertShader := shader.NewShader("_vsm_depth_vert", shader.ShaderTypeVertex, "engine/light/assets/vsm-depth-vert.wgsl")
	vsmSkinnedVertShader := shader.NewShader("_vsm_depth_skinned_vert", shader.ShaderTypeVertex, "engine/light/assets/vsm-depth-skinned-vert.wgsl")
	vsmFragShader := shader.NewShader("_vsm_depth_frag", shader.ShaderTypeFragment, "engine/light/assets/vsm-depth-frag.wgsl")
	blurComputeShader := shader.NewShader("_vsm_blur_compute", shader.ShaderTypeCompute, "engine/light/assets/vsm-blur-compute.wgsl")
	s.initVSMShadowMap(vsmVertShader, vsmSkinnedVertShader, vsmFragShader, blurComputeShader)

	if pcssMode {
		satComputeShader := shader.NewShader("_vsm_sat_compute", shader.ShaderTypeCompute, "engine/light/assets/vsm-sat-compute.wgsl")
		s.initSATResources(satComputeShader)
	}

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

	// 6. G-Buffer MRT pre-pass (required by SSAO and SSR).
	s.initGBuffer()

	// 7. SSAO — hemisphere sampling + bilateral blur (requires G-Buffer).
	s.initSSAO()

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
			compFrag := shader.NewShader("_composition_frag_rebind", shader.ShaderTypeFragment, "engine/light/assets/composition-frag.wgsl")
			compDesc := compFrag.BindGroupLayoutDescriptor(0)
			sizeOverrides := map[int]uint64{
				4: uint64((&light.GPUCompositionParams{}).Size()),
			}
			if err := s.r.InitBindGroup(compBGP, compDesc, nil, sizeOverrides); err != nil {
				panic(fmt.Sprintf("scene: failed to re-init composition bind group with SSR texture: %v", err))
			}
		}
	}

	// 11. Irradiance probe grid (requires lighting GPU resources for bake passes).
	if s.lightHandler.ProbeGrid() != nil {
		s.initProbeGrid()
	}

	// 12. Probes lit bind group — binds probe storage buffer + grid params at
	// @group(7) for the lit fragment shader. When probes are absent, minimal
	// fallback buffers with total_probes = 0 are used (no indirect illumination).
	s.initProbesLitBindGroup(litFragShader)

	// 13. Mark the lighting subsystem as GPU-initialized.
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

	// Auto-resolve standard shaders based on whether the model uses skeletal
	// animation. The full GI lighting pipeline is always active, so lit shaders
	// are always used.
	var computeShader, vertexShader, fragmentShader shader.Shader
	if mdl.Skinned() {
		computeShader = shader.NewShader(mdl.Name()+"_compute", shader.ShaderTypeCompute, "engine/renderer/animator/assets/skeletal-compute.wgsl")
		vertexShader = shader.NewShader(mdl.Name()+"_vertex", shader.ShaderTypeVertex, "engine/light/assets/lit-skinned-vert.wgsl")
	} else {
		computeShader = shader.NewShader(mdl.Name()+"_compute", shader.ShaderTypeCompute, "engine/renderer/animator/assets/simple-compute.wgsl")
		vertexShader = shader.NewShader(mdl.Name()+"_vertex", shader.ShaderTypeVertex, "engine/light/assets/lit-vert.wgsl")
	}
	// Resolve fragment shader path: use the first material's custom path if set,
	// otherwise fall back to the lit VSM or PCSS variant.
	var fragShaderPath string
	if s.lightHandler.PCSSEnabled() {
		fragShaderPath = "engine/light/assets/lit-frag-pcss.wgsl"
	} else {
		fragShaderPath = "engine/light/assets/lit-frag-vsm.wgsl"
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
				if entry.Buffer == nil {
					continue
				}
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
		Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage, MinBindingSize: 4},
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
		if int(entry.Binding) == outputInstanceBinding && entry.Buffer != nil && entry.Buffer.MinBindingSize > 0 {
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
				if entry.Buffer != nil && entry.Buffer.MinBindingSize > 0 {
					computeSizeOverrides[binding] = boneCount * entry.Buffer.MinBindingSize
				}
			case shader.AnnotationArgModelData:
				// Per-instance model matrices from CPU.
				if entry.Buffer != nil && entry.Buffer.MinBindingSize > 0 {
					computeSizeOverrides[binding] = maxInst * entry.Buffer.MinBindingSize
				}
			case shader.AnnotationArgAnimationGlobals, shader.AnnotationArgGlobalData:
				// Uniform buffer — fixed size from the parser, no override needed.
			default:
				// Per-instance storage buffers (animation data, skeletal animation data, etc.).
				if (bufferBindingType(entry) == gputypes.BufferBindingTypeStorage || bufferBindingType(entry) == gputypes.BufferBindingTypeReadOnlyStorage) &&
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
			(bufferBindingType(entry) == gputypes.BufferBindingTypeStorage || bufferBindingType(entry) == gputypes.BufferBindingTypeReadOnlyStorage) &&
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
						case shader.AnnotationArgSSAO:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("ssao_lit")
							}
						case shader.AnnotationArgProbes:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("probe_lit")
							}
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
						case shader.AnnotationArgIrradianceProbe, shader.AnnotationArgProbeGridParams:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("probe_lit")
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
