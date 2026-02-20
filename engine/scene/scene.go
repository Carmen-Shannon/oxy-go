package scene

import (
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
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/cogentcore/webgpu/wgpu"
)

// Scene manages a collection of Animators (registered implicitly via Add) and an
// optional registry of non-ephemeral GameObjects, with a Camera and Renderer for
// rendering. Rendering is driven entirely by the registered Animator list — each
// Animator owns its instance data and material.
// Scenes can be hot-swapped via the Active flag to switch between different views or levels.
// Thread-safe for concurrent access.
type Scene interface {
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
	// Panics if the scene has no Renderer or the object has no Model.
	//
	// Parameters:
	//   - obj: the GameObject to add
	//   - computeShader: the compute shader to use for this object's Animator
	//   - vertexShader: the vertex shader to use for this object's render pipeline
	//   - fragmentShader: the fragment shader to use for this object's render pipeline
	//   - pipelineOpts: optional pipeline builder options for the render pipeline (e.g., blending)
	//
	// Returns:
	//   - uint64: the assigned object ID
	Add(obj game_object.GameObject, computeShader, vertexShader, fragmentShader shader.Shader, pipelineOpts ...pipeline.PipelineBuilderOption) uint64

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

	// Clear removes all objects and animators from the scene.
	// Does not release GPU resources.
	Clear()

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

	// AddLight adds a light source to the scene. Lights are marshaled into a GPU
	// storage buffer each frame and passed to lit fragment shaders.
	//
	// Parameters:
	//   - l: the Light to add
	AddLight(l light.Light)

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

	// LightBindGroupProvider returns the bind group provider holding the GPU light
	// buffer resources, or nil if no light shader has been configured.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the light BGP or nil
	LightBindGroupProvider() bind_group_provider.BindGroupProvider

	// InitLightBindGroup initializes the GPU resources for the light storage buffer
	// using the layout descriptor from the given fragment shader's light group.
	// The fragment shader is scanned for variable names containing "light" to locate
	// the appropriate bind group index.
	//
	// Parameters:
	//   - fragmentShader: the lit fragment shader providing the light bind group layout
	InitLightBindGroup(fragmentShader shader.Shader)

	// InitShadowMap initializes the shadow mapping resources for the scene. Creates
	// the shadow depth texture, comparison sampler, shadow data uniform BGP, and
	// registers shadow pipelines for both static and skinned models. The shadow
	// depth vertex shaders are used to build pipelines that render depth-only passes
	// from the directional light's perspective.
	//
	// Parameters:
	//   - shadowVertexShader: the shadow depth vertex shader for static models
	//   - shadowSkinnedVertexShader: the shadow depth vertex shader for skinned models (may be nil if no skinned models)
	InitShadowMap(shadowVertexShader, shadowSkinnedVertexShader shader.Shader)

	// PrepareShadows computes the directional light's view-projection, updates the
	// shadow uniform buffer, and renders the depth-only shadow pass for all drawables.
	// Must be called after PrepareCompute and before BeginFrame each frame.
	// No-ops if no shadow map has been initialized or no shadow-casting directional
	// light exists.
	PrepareShadows()

	// ShadowDepthTextureView returns the shadow map depth texture view, or nil if
	// shadow mapping has not been initialized.
	//
	// Returns:
	//   - *wgpu.TextureView: the shadow depth texture view or nil
	ShadowDepthTextureView() *wgpu.TextureView

	// ShadowDataBindGroupProvider returns the BGP holding the shadow uniform data
	// (light VP matrix, texel size, bias), or nil if not initialized.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the shadow data BGP or nil
	ShadowDataBindGroupProvider() bind_group_provider.BindGroupProvider

	// ShadowLitBindGroupProvider returns the BGP used by lit fragment shaders
	// to sample the shadow map. It holds the shadow depth texture, comparison
	// sampler, and shadow uniform buffer. Returns nil if not initialized.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the shadow lit BGP or nil
	ShadowLitBindGroupProvider() bind_group_provider.BindGroupProvider

	// InitShadowLitBindGroup initializes the bind group provider that lit fragment
	// shaders use to sample the shadow map. It pre-sets the shadow depth texture view
	// and comparison sampler from InitShadowMap, then creates a uniform buffer for the
	// shadow data. Must be called after InitShadowMap.
	//
	// Parameters:
	//   - litFragmentShader: the lit fragment shader providing the shadow bind group layout
	InitShadowLitBindGroup(litFragmentShader shader.Shader)

	// InitLightCullResources initializes the Forward+ light culling pipeline and
	// buffer resources. Creates the cull compute pipeline, the compute BGP (sharing
	// the lights buffer from InitLightBindGroup), and the fragment-side tile BGP
	// (@group(5)) whose storage buffers are shared with the compute output. Must be
	// called after InitLightBindGroup.
	//
	// Parameters:
	//   - cullComputeShader: the light culling compute shader
	//   - litFragmentShader: the lit fragment shader (for tile bind group layout at @group(5))
	//   - screenWidth: screen width in pixels (determines tile grid sizing)
	//   - screenHeight: screen height in pixels
	InitLightCullResources(cullComputeShader, litFragmentShader shader.Shader, screenWidth, screenHeight int)

	// PrepareLightCulling updates the light cull uniform buffer and dispatches the
	// light culling compute shader. Must be called after PrepareCompute (so lights
	// are uploaded) and before DrawCalls.
	PrepareLightCulling()

	// InitLighting is a convenience method that initializes the entire lighting
	// pipeline in the correct order: light storage buffer, shadow map resources,
	// shadow lit bind group, and Forward+ light culling. Equivalent to calling
	// InitLightBindGroup, InitShadowMap, InitShadowLitBindGroup, and
	// InitLightCullResources individually in that order.
	//
	// Parameters:
	//   - litFragShader: the lit fragment shader (provides light, shadow, and tile bind group layouts)
	//   - shadowVertShader: the shadow depth vertex shader for static models
	//   - shadowSkinnedVertShader: the shadow depth vertex shader for skinned models (may be nil)
	//   - cullComputeShader: the Forward+ light culling compute shader
	//   - screenWidth: screen width in pixels
	//   - screenHeight: screen height in pixels
	InitLighting(litFragShader, shadowVertShader, shadowSkinnedVertShader, cullComputeShader shader.Shader, screenWidth, screenHeight int)
}

type scene struct {
	mu *sync.RWMutex

	name   string
	active bool

	animatorPool map[model.Model][]animator.Animator
	registry     map[uint64]game_object.GameObject // non-ephemeral objects by ID
	nextID       uint64

	cam camera.Camera
	r   renderer.Renderer

	cullingDisabled bool // when true, skips frustum plane distribution to animators

	// Lighting state.
	lights       []light.Light
	lightObjects []game_object.GameObject // objects with attached lights (ephemeral and non-ephemeral)
	ambientColor [3]float32
	lightsBGP    bind_group_provider.BindGroupProvider

	// Shadow mapping state.
	shadowDepthTexture     *wgpu.Texture
	shadowDepthTextureView *wgpu.TextureView
	shadowComparisonSamp   *wgpu.Sampler
	shadowDataBGP          bind_group_provider.BindGroupProvider // used during the shadow depth pass
	shadowLitBGP           bind_group_provider.BindGroupProvider // used during the lit pass (texture + sampler + uniform)
	shadowPipelineKey      string                                // pipeline key for static models
	shadowSkinnedPipeKey   string                                // pipeline key for skinned models
	shadowHalfExtent       float32
	shadowNear             float32
	shadowFar              float32
	shadowBias             float32
	shadowNormalBiasScale  float32
	shadowMapResolution    int

	// Forward+ light culling state.
	lightCullBGP         bind_group_provider.BindGroupProvider // compute shader BGP
	tileLitBGP           bind_group_provider.BindGroupProvider // fragment shader BGP (@group(5))
	lightCullPipelineKey string
	tileCountX           uint32
	tileCountY           uint32
	screenWidth          int
	screenHeight         int

	// Pre-allocated slices reused each frame to avoid per-frame allocations.
	writePool          []bind_group_provider.BufferWrite       // reusable coalesced buffer write slice
	drawBindGroupsPool []bind_group_provider.BindGroupProvider // reusable bind group slice for DrawCalls

	// computePool manages a bounded set of reusable goroutines for the parallel
	// CPU prep phase of PrepareCompute. Workers persist across frames, avoiding
	// per-frame goroutine spawn/teardown overhead.
	computePool    worker.DynamicWorkerPool
	computeWorkers int // stored so we can log/inspect the configured count
}

// Ensure scene implements Scene interface.
var _ Scene = &scene{}

// NewScene creates a new Scene with the given camera, renderer, and a vertex shader
// used to discover the camera's bind group layout. All three are required and NewScene
// panics if any of them is nil. The vertex shader's BindGroupVarNames are scanned for
// a group containing "camera" and its layout descriptor is used to initialize the
// camera's BindGroupProvider on the GPU.
//
// Parameters:
//   - name: the name of the scene
//   - cam: the camera to attach (must not be nil)
//   - r: the renderer to attach (must not be nil)
//   - vertexShader: a vertex shader whose bind groups include the camera uniform layout (must not be nil)
//   - options: functional options to further configure the scene
//
// Returns:
//   - Scene: the newly created scene
func NewScene(name string, cam camera.Camera, r renderer.Renderer, vertexShader shader.Shader, options ...SceneBuilderOption) Scene {
	if cam == nil {
		panic("scene: NewScene requires a non-nil Camera")
	}
	if r == nil {
		panic("scene: NewScene requires a non-nil Renderer")
	}
	if vertexShader == nil {
		panic("scene: NewScene requires a non-nil vertex shader for camera BGP init")
	}

	s := &scene{
		mu:                    &sync.RWMutex{},
		name:                  name,
		active:                false,
		cam:                   cam,
		r:                     r,
		animatorPool:          make(map[model.Model][]animator.Animator),
		registry:              make(map[uint64]game_object.GameObject),
		nextID:                1,
		computeWorkers:        max(runtime.NumCPU()-1, 1),
		drawBindGroupsPool:    make([]bind_group_provider.BindGroupProvider, 0, 3),
		shadowHalfExtent:      light.DefaultShadowHalfExtent,
		shadowNear:            light.DefaultShadowNear,
		shadowFar:             light.DefaultShadowFar,
		shadowBias:            light.DefaultShadowBias,
		shadowNormalBiasScale: light.DefaultShadowNormalBiasScale,
		shadowMapResolution:   light.ShadowMapResolution,
	}

	for _, option := range options {
		option(s)
	}

	// Initialize the compute pool after options so WithComputeWorkers can override the default.
	// Queue size of 256 accommodates typical animator group counts with headroom.
	s.computePool = worker.NewDynamicWorkerPool(s.computeWorkers, 256, 1*time.Second)

	// Initialize the camera's bind group on the GPU using the layout from the vertex shader.
	cameraGroup := 0
	for i, names := range vertexShader.BindGroupVarNames() {
		for _, name := range names {
			if strings.Contains(strings.ToLower(name), "camera") {
				cameraGroup = i
				break
			}
		}
	}
	if bgp := cam.BindGroupProvider(); bgp != nil {
		if err := r.InitBindGroup(bgp, vertexShader.BindGroupLayoutDescriptor(cameraGroup), nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init camera bind group: %v", err))
		}
	}

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
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lights = append(s.lights, l)
}

func (s *scene) RemoveLight(l light.Light) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, existing := range s.lights {
		if existing == l {
			s.lights = append(s.lights[:i], s.lights[i+1:]...)
			return
		}
	}
}

func (s *scene) DetachLight(obj game_object.GameObject) {
	s.mu.Lock()
	defer s.mu.Unlock()
	l := obj.Light()
	if l == nil {
		return
	}
	for i, existing := range s.lights {
		if existing == l {
			s.lights = append(s.lights[:i], s.lights[i+1:]...)
			break
		}
	}
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
	out := make([]light.Light, len(s.lights))
	copy(out, s.lights)
	return out
}

func (s *scene) AmbientColor() [3]float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ambientColor
}

func (s *scene) SetAmbientColor(color [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ambientColor = color
}

func (s *scene) LightBindGroupProvider() bind_group_provider.BindGroupProvider {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lightsBGP
}

func (s *scene) InitLightBindGroup(fragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || fragmentShader == nil {
		return
	}

	// Find the bind group index that contains light-related variables.
	// After renaming tile vars to avoid "light" substring, a simple search
	// for "light" unambiguously matches only the lights group (@group(3)).
	lightGroup := -1
	for groupIdx, bindings := range fragmentShader.BindGroupVarNames() {
		for _, name := range bindings {
			if strings.Contains(strings.ToLower(name), "light") {
				lightGroup = groupIdx
				break
			}
		}
		if lightGroup >= 0 {
			break
		}
	}
	if lightGroup < 0 {
		return
	}

	bgp := bind_group_provider.NewBindGroupProvider(s.name + "_lights")

	// Build buffer size overrides: the light storage buffer (binding 1) must hold
	// MaxGPULights entries so it can accommodate dynamic light counts each frame.
	descriptor := fragmentShader.BindGroupLayoutDescriptor(lightGroup)
	sizeOverrides := make(map[int]uint64)
	for _, entry := range descriptor.Entries {
		binding := int(entry.Binding)
		if entry.Buffer.Type == wgpu.BufferBindingTypeReadOnlyStorage || entry.Buffer.Type == wgpu.BufferBindingTypeStorage {
			// Storage buffer: size it for max lights (header is in a separate uniform binding).
			sizeOverrides[binding] = uint64(light.MaxGPULights) * 64 // 64 bytes per GPULight
		}
	}

	if err := s.r.InitBindGroup(bgp, descriptor, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init light bind group: %v", err))
	}
	s.lightsBGP = bgp
}

func (s *scene) InitShadowMap(shadowVertexShader, shadowSkinnedVertexShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || shadowVertexShader == nil {
		return
	}

	// Create shadow depth texture.
	res := s.shadowMapResolution
	view, tex, err := s.r.CreateShadowDepthTexture(res, res)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create shadow depth texture: %v", err))
	}
	s.shadowDepthTexture = tex
	s.shadowDepthTextureView = view

	// Create comparison sampler for PCF in the lit fragment shader.
	samp, err := s.r.CreateComparisonSampler()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create comparison sampler: %v", err))
	}
	s.shadowComparisonSamp = samp

	// Create shadow data BGP — holds the light VP matrix + texel size + bias.
	// The layout is derived from the shadow vertex shader's group(0) which has the
	// shadow_uniform binding.
	shadowGroup := 0
	for i, names := range shadowVertexShader.BindGroupVarNames() {
		for _, name := range names {
			if strings.Contains(strings.ToLower(name), "shadow") {
				shadowGroup = i
				break
			}
		}
	}
	bgp := bind_group_provider.NewBindGroupProvider(s.name + "_shadow_data")
	desc := shadowVertexShader.BindGroupLayoutDescriptor(shadowGroup)
	// Override buffer size to 80 bytes (GPUShadowData: mat4x4 + vec2 + f32 + f32).
	sizeOverrides := make(map[int]uint64)
	for _, entry := range desc.Entries {
		if entry.Buffer.Type == wgpu.BufferBindingTypeUniform {
			sizeOverrides[int(entry.Binding)] = 80
		}
	}
	if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init shadow data bind group: %v", err))
	}
	s.shadowDataBGP = bgp

	// Register shadow pipeline for static models.
	staticKey := "shadow_depth_static"
	sp := pipeline.NewPipeline(staticKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(shadowVertexShader),
		pipeline.WithDepthBias(2, 1.5),
		pipeline.WithCullMode(wgpu.CullModeFront), // CullModeFront: all static meshes should be closed geometry
	)
	if err := s.r.RegisterShadowPipeline(sp); err != nil {
		panic(fmt.Sprintf("scene: failed to register static shadow pipeline: %v", err))
	}
	s.shadowPipelineKey = staticKey

	// Register shadow pipeline for skinned models if a skinned shader is provided.
	if shadowSkinnedVertexShader != nil {
		skinnedKey := "shadow_depth_skinned"
		ssp := pipeline.NewPipeline(skinnedKey, pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(shadowSkinnedVertexShader),
			pipeline.WithDepthBias(2, 1.5),
			pipeline.WithCullMode(wgpu.CullModeFront), // CullModeFront prevents self-shadowing on closed skinned meshes
		)
		if err := s.r.RegisterShadowPipeline(ssp); err != nil {
			panic(fmt.Sprintf("scene: failed to register skinned shadow pipeline: %v", err))
		}
		s.shadowSkinnedPipeKey = skinnedKey
	}
}

func (s *scene) ShadowDepthTextureView() *wgpu.TextureView {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.shadowDepthTextureView
}

func (s *scene) ShadowDataBindGroupProvider() bind_group_provider.BindGroupProvider {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.shadowDataBGP
}

func (s *scene) ShadowLitBindGroupProvider() bind_group_provider.BindGroupProvider {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.shadowLitBGP
}

func (s *scene) InitShadowLitBindGroup(litFragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || litFragmentShader == nil {
		return
	}
	if s.shadowDepthTextureView == nil || s.shadowComparisonSamp == nil {
		return // InitShadowMap must be called first
	}

	// Find the bind group index that contains shadow-related variables.
	shadowGroup := -1
	for groupIdx, bindings := range litFragmentShader.BindGroupVarNames() {
		for _, name := range bindings {
			if strings.Contains(strings.ToLower(name), "shadow") {
				shadowGroup = groupIdx
				break
			}
		}
		if shadowGroup >= 0 {
			break
		}
	}
	if shadowGroup < 0 {
		return
	}

	bgp := bind_group_provider.NewBindGroupProvider(s.name + "_shadow_lit")

	// Pre-set the shadow depth texture view and comparison sampler on the BGP
	// so that InitBindGroup can find them when creating the bind group entries.
	desc := litFragmentShader.BindGroupLayoutDescriptor(shadowGroup)
	for _, entry := range desc.Entries {
		binding := int(entry.Binding)
		if entry.Texture.SampleType != wgpu.TextureSampleTypeUndefined {
			bgp.SetTextureView(binding, s.shadowDepthTextureView)
		}
		if entry.Sampler.Type != wgpu.SamplerBindingTypeUndefined {
			bgp.SetSampler(binding, s.shadowComparisonSamp)
		}
	}

	// Override the uniform buffer size to 80 bytes (GPUShadowData).
	sizeOverrides := make(map[int]uint64)
	for _, entry := range desc.Entries {
		if entry.Buffer.Type == wgpu.BufferBindingTypeUniform {
			sizeOverrides[int(entry.Binding)] = 80
		}
	}

	if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init shadow lit bind group: %v", err))
	}
	s.shadowLitBGP = bgp
}

func (s *scene) PrepareShadows() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.shadowDepthTextureView == nil || s.shadowDataBGP == nil || s.r == nil {
		return
	}

	// Find the first enabled, shadow-casting directional light.
	var shadowLight light.Light
	for _, l := range s.lights {
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
	texelSize := 1.0 / float32(s.shadowMapResolution)
	shadowData := light.GPUShadowData{
		TexelSize: [2]float32{texelSize, texelSize},
		Bias:      s.shadowBias,
	}
	shadowData.ComputeDirectionalLightVP(
		shadowLight.Direction(),
		centerX, centerY, centerZ,
		s.shadowHalfExtent, s.shadowNear, s.shadowFar,
	)
	shadowData.ComputeNormalBias(s.shadowHalfExtent, s.shadowNormalBiasScale, s.shadowMapResolution)
	shadowBytes := shadowData.Marshal()
	writes := []bind_group_provider.BufferWrite{
		{
			Provider: s.shadowDataBGP,
			Binding:  0,
			Offset:   0,
			Data:     shadowBytes,
		},
	}
	// Also write to the lit-pass shadow BGP if it has a uniform buffer.
	if s.shadowLitBGP != nil {
		for binding, buf := range s.shadowLitBGP.Buffers() {
			if buf != nil {
				writes = append(writes, bind_group_provider.BufferWrite{
					Provider: s.shadowLitBGP,
					Binding:  binding,
					Offset:   0,
					Data:     shadowBytes,
				})
				break // only one uniform buffer expected
			}
		}
	}
	s.r.WriteBuffers(writes)

	// Execute shadow depth pass.
	if err := s.r.BeginShadowFrame(); err != nil {
		return
	}
	s.r.BeginShadowPass(s.shadowDepthTextureView)

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

			// Select the appropriate shadow pipeline.
			pipeKey := s.shadowPipelineKey
			if mdl.Skinned() && s.shadowSkinnedPipeKey != "" {
				pipeKey = s.shadowSkinnedPipeKey
			}
			if pipeKey == "" {
				continue
			}

			// Build bind groups for the shadow pass:
			//   group(0) = shadow data BGP (light VP uniform)
			//   group(1) = output BGP (instance/bone matrices from compute shader)
			shadowBindGroups := []bind_group_provider.BindGroupProvider{
				s.shadowDataBGP,
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
					if cs := s.r.Pipeline(key).Shader(shader.ShaderTypeCompute); cs != nil {
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

			_ = s.r.ShadowDrawCall(pipeKey, meshProvider, uint32(a.InstanceCount()), shadowBindGroups)
		}
	}

	s.r.EndShadowPass()
	s.r.EndShadowFrame()
}

func (s *scene) InitLightCullResources(cullComputeShader, litFragmentShader shader.Shader, screenWidth, screenHeight int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil || cullComputeShader == nil || litFragmentShader == nil {
		return
	}
	if s.lightsBGP == nil {
		return // InitLightBindGroup must be called first
	}

	s.screenWidth = screenWidth
	s.screenHeight = screenHeight
	tileCountX, tileCountY := light.TileCounts(screenWidth, screenHeight)
	s.tileCountX = tileCountX
	s.tileCountY = tileCountY

	numTiles := uint64(tileCountX) * uint64(tileCountY)

	// ── 1. Create compute BGP (cull shader's @group(0)) ────────────────
	// binding 0: cull_uniforms (uniform, 160 bytes)
	// binding 1: cull_lights (storage, read) — shared from lightsBGP binding 1
	// binding 2: tile_light_counts (storage, rw) — new buffer
	// binding 3: tile_light_indices (storage, rw) — new buffer
	cullBGP := bind_group_provider.NewBindGroupProvider(s.name + "_light_cull")

	// Pre-set the lights buffer from lightsBGP so InitBindGroup reuses it.
	if lightsBuffer := s.lightsBGP.Buffer(1); lightsBuffer != nil {
		cullBGP.SetBuffer(1, lightsBuffer)
	}

	cullDesc := cullComputeShader.BindGroupLayoutDescriptor(0)
	sizeOverrides := map[int]uint64{
		0: 160,                                           // LightCullUniforms
		2: numTiles * 4,                                  // tile_light_counts: one u32 per tile
		3: numTiles * uint64(light.MaxLightsPerTile) * 4, // tile_light_indices
	}

	if err := s.r.InitBindGroup(cullBGP, cullDesc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init light cull bind group: %v", err))
	}
	s.lightCullBGP = cullBGP

	// ── 2. Register the cull compute pipeline ──────────────────────────
	pipeKey := "light_cull_compute"
	cp := pipeline.NewPipeline(pipeKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(cullComputeShader),
	)
	if err := s.r.RegisterPipelines(cp); err != nil {
		panic(fmt.Sprintf("scene: failed to register light cull compute pipeline: %v", err))
	}
	s.lightCullPipelineKey = pipeKey

	// ── 3. Create fragment tile BGP (lit frag shader's @group(5)) ──────
	// binding 0: tile_uniforms (uniform, 8 bytes)
	// binding 1: tile_light_counts (storage, read) — shared from cullBGP binding 2
	// binding 2: tile_light_indices (storage, read) — shared from cullBGP binding 3
	tileBGP := bind_group_provider.NewBindGroupProvider(s.name + "_tile_lit")

	if countsBuf := cullBGP.Buffer(2); countsBuf != nil {
		tileBGP.SetBuffer(1, countsBuf)
	}
	if indicesBuf := cullBGP.Buffer(3); indicesBuf != nil {
		tileBGP.SetBuffer(2, indicesBuf)
	}

	// Find the tile bind group index in the lit fragment shader.
	tileGroup := -1
	for groupIdx, bindings := range litFragmentShader.BindGroupVarNames() {
		for _, name := range bindings {
			if strings.Contains(strings.ToLower(name), "tile") {
				tileGroup = groupIdx
				break
			}
		}
		if tileGroup >= 0 {
			break
		}
	}
	if tileGroup < 0 {
		panic("scene: lit fragment shader has no tile bind group")
	}

	tileDesc := litFragmentShader.BindGroupLayoutDescriptor(tileGroup)
	tileSizeOverrides := map[int]uint64{
		0: 8, // TileUniforms (tile_count_x + max_lights_per_tile)
	}
	if err := s.r.InitBindGroup(tileBGP, tileDesc, nil, tileSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init tile lit bind group: %v", err))
	}
	s.tileLitBGP = tileBGP

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

	// Find the camera group in the lit fragment shader.
	cameraGroup := -1
	for groupIdx, bindings := range litFragShader.BindGroupVarNames() {
		for _, name := range bindings {
			if strings.Contains(strings.ToLower(name), "camera") {
				cameraGroup = groupIdx
				break
			}
		}
		if cameraGroup >= 0 {
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

	if s.lightCullBGP == nil || s.r == nil || s.cam == nil {
		return
	}

	// Count enabled lights. Even when zero we must still dispatch the cull
	// shader so that tile counts are zeroed out — otherwise stale tile data
	// from the previous frame causes disabled lights to keep rendering.
	var lightCount uint32
	for _, l := range s.lights {
		if l.Enabled() {
			lightCount++
		}
	}

	// Build and write cull uniforms.
	uniforms := light.GPULightCullUniforms{
		InvProj:      s.cam.InverseProjectionMatrix(),
		ViewMatrix:   s.cam.ViewMatrix(),
		TileCountX:   s.tileCountX,
		TileCountY:   s.tileCountY,
		ScreenWidth:  uint32(s.screenWidth),
		ScreenHeight: uint32(s.screenHeight),
		LightCount:   lightCount,
		Near:         s.cam.Near(),
		Far:          s.cam.Far(),
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: s.lightCullBGP, Binding: 0, Offset: 0, Data: uniforms.Marshal()},
	})

	// Dispatch the light culling compute shader.
	if err := s.r.BeginComputeFrame(); err != nil {
		return
	}
	s.r.DispatchCompute(s.lightCullPipelineKey, s.lightCullBGP, [3]uint32{s.tileCountX, s.tileCountY, 1})
	s.r.EndComputeFrame()
}

func (s *scene) InitLighting(litFragShader, shadowVertShader, shadowSkinnedVertShader, cullComputeShader shader.Shader, screenWidth, screenHeight int) {
	// 1. Light storage buffer (must be first — other steps share this buffer).
	s.InitLightBindGroup(litFragShader)

	// 2. Shadow depth texture, comparison sampler, shadow data BGP, shadow pipelines.
	s.InitShadowMap(shadowVertShader, shadowSkinnedVertShader)

	// 3. Shadow lit BGP (fragment-side shadow sampling — references shadow resources from step 2).
	s.InitShadowLitBindGroup(litFragShader)

	// 4. Forward+ tile culling pipeline and shared tile buffers (references lights buffer from step 1).
	s.InitLightCullResources(cullComputeShader, litFragShader, screenWidth, screenHeight)

	// 5. Re-create the camera bind group with merged VERTEX|FRAGMENT visibility.
	//
	// The camera's bind group was originally created in NewScene from the vertex
	// shader alone (visibility = VERTEX). The lit fragment shader also declares the
	// camera group (visibility = FRAGMENT). The render pipeline merges these into
	// VERTEX|FRAGMENT. WebGPU requires exact bind group layout equivalence, so the
	// camera BGL must be recreated with the combined visibility to pass validation.
	s.reinitCameraBGPForLitPipeline(litFragShader)
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

func (s *scene) Add(obj game_object.GameObject, computeShader, vertexShader, fragmentShader shader.Shader, pipelineOpts ...pipeline.PipelineBuilderOption) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.r == nil {
		panic("scene: cannot Add without a Renderer attached")
	}

	mdl := obj.Model()
	if mdl == nil {
		panic("scene: cannot Add a GameObject without a Model")
	}

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

	// Persist non-ephemeral objects in the registry
	if !obj.Ephemeral() {
		s.registry[obj.ID()] = obj
	}

	// If the object has an attached light, track it for automatic position sync
	// and register the light with the scene's light list.
	if l := obj.Light(); l != nil {
		s.lightObjects = append(s.lightObjects, obj)
		s.lights = append(s.lights, l)
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
		for i, existing := range s.lights {
			if existing == l {
				s.lights = append(s.lights[:i], s.lights[i+1:]...)
				break
			}
		}
		for i, o := range s.lightObjects {
			if o == obj {
				s.lightObjects = append(s.lightObjects[:i], s.lightObjects[i+1:]...)
				break
			}
		}
	}

	// Swap-remove the instance data from the animator
	if anim := obj.Animator(); anim != nil {
		removedIdx := obj.AnimatorInstanceID()
		if removedIdx >= 0 {
			swappedFrom, swapped := anim.RemoveInstance(uint32(removedIdx))
			if swapped {
				// The instance at swappedFrom was moved into removedIdx — find the
				// registry object that owned that slot and update its stored index.
				for _, o := range s.registry {
					if o.Animator() == anim && o.AnimatorInstanceID() == int(swappedFrom) {
						o.SetAnimatorInstanceID(removedIdx)
						break
					}
				}
			}
			obj.SetAnimatorInstanceID(-1)
		}
	}
}

func (s *scene) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.animatorPool = make(map[model.Model][]animator.Animator)
	s.registry = make(map[uint64]game_object.GameObject)
	s.lightObjects = nil
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

	// Write light buffer to GPU each frame when a light BGP is initialized.
	if s.lightsBGP != nil {
		lightData := light.MarshalLightBuffer(s.lights, s.ambientColor)
		writes := []bind_group_provider.BufferWrite{
			{
				Provider: s.lightsBGP,
				Binding:  0, // light_header uniform
				Offset:   0,
				Data:     lightData[:16], // GPULightHeader is 16 bytes
			},
		}
		if len(lightData) > 16 {
			writes = append(writes, bind_group_provider.BufferWrite{
				Provider: s.lightsBGP,
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

			shdr := s.r.Pipeline(a.Model().ComputePipelineKey()).Shader(shader.ShaderTypeCompute)
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
						shdr := s.r.Pipeline(a.Model().ComputePipelineKey()).Shader(shader.ShaderTypeCompute)
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

	// Dispatch compute shaders for each registered animator with instances
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}
			if key := a.Model().ComputePipelineKey(); key != "" {
				s.r.DispatchCompute(key, a.ComputeBindGroupProvider(), s.r.Pipeline(key).Shader(shader.ShaderTypeCompute).WorkgroupSize())
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
							provider = mat.BindGroupProvider()
						case shader.AnnotationArgLights:
							if s.lightsBGP != nil {
								provider = s.lightsBGP
							}
						case shader.AnnotationArgShadow:
							if s.shadowLitBGP != nil {
								provider = s.shadowLitBGP
							}
						case shader.AnnotationArgTiles:
							if s.tileLitBGP != nil {
								provider = s.tileLitBGP
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
							if s.lightsBGP != nil {
								provider = s.lightsBGP
							}
						case shader.AnnotationArgShadowData, shader.AnnotationArgShadowUniform:
							if s.shadowLitBGP != nil {
								provider = s.shadowLitBGP
							}
						case shader.AnnotationArgTileUniforms:
							if s.tileLitBGP != nil {
								provider = s.tileLitBGP
							}
						case shader.AnnotationArgEffectParams, shader.AnnotationArgOverlayParams:
							if ep := mdl.EffectProvider(); ep != nil {
								provider = ep
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
						if cs := s.r.Pipeline(key).Shader(shader.ShaderTypeCompute); cs != nil {
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
