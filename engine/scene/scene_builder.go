package scene

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/Carmen-Shannon/automation/tools/worker"
	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/gbuffer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/composition"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/ssao"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/ssr"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/taa"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/cogentcore/webgpu/wgpu"
)

// SceneBuilderOption is a functional option for configuring a Scene.
// Use the With* functions to create options.
type SceneBuilderOption func(s *scene)

// WithActive sets whether the scene is active for rendering.
//
// Parameters:
//   - active: whether the scene is active
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithActive(active bool) SceneBuilderOption {
	return func(s *scene) {
		s.active = active
	}
}

// WithObjects adds initial objects to the scene.
// Objects without IDs will be assigned new IDs.
// Non-ephemeral objects are persisted in the registry; their animators are auto-registered.
//
// Parameters:
//   - objects: the objects to add
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithObjects(objects ...game_object.GameObject) SceneBuilderOption {
	return func(s *scene) {
		for _, obj := range objects {
			if obj.ID() == 0 {
				obj.SetID(s.nextID)
				s.nextID++
			}
			if !obj.Ephemeral() {
				s.registry[obj.ID()] = obj
			}
		}
	}
}

// WithComputeWorkers sets the number of worker goroutines used during the parallel
// CPU prep phase of PrepareCompute. Defaults to runtime.NumCPU()-1.
// Higher values may improve throughput with many animator groups or skeletal
// animators; lower values reduce scheduling overhead for simple scenes.
//
// Parameters:
//   - n: the number of compute workers (minimum 1)
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithComputeWorkers(n int) SceneBuilderOption {
	return func(s *scene) {
		if n < 1 {
			n = 1
		}
		s.computeWorkers = n
	}
}

// WithCullingDisabled disables GPU frustum culling for the scene. When set to true,
// the scene will not distribute frustum planes to animators, causing them to remain
// in non-culled mode and use regular draw calls instead of indirect draw calls.
// By default culling is enabled (disabled = false).
//
// Parameters:
//   - disabled: true to disable frustum culling, false to enable it (default)
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithCullingDisabled(disabled bool) SceneBuilderOption {
	return func(s *scene) {
		s.cullingDisabled = disabled
	}
}

// WithLighting attaches a pre-configured LightingHandler to the scene, replacing
// the default handler created by NewScene. Use light.NewLightingHandler with
// light.WithShadow* options to configure shadow mapping and ambient color before
// passing the handler here. GPU resources are initialized lazily when the first
// light is added via AddLight.
//
// Parameters:
//   - handler: the pre-configured LightingHandler
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithLighting(handler light.LightingHandler) SceneBuilderOption {
	return func(s *scene) {
		s.lightHandler = handler
	}
}

// WithGBufferHandler attaches a pre-configured GBufferHandler to the scene.
//
// Parameters:
//   - handler: the pre-configured GBufferHandler
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithGBufferHandler(handler gbuffer.Handler) SceneBuilderOption {
	return func(s *scene) {
		s.gBufferHandler = handler
	}
}

// WithSSAOHandler attaches a pre-configured SSAOHandler to the scene,
// replacing the default handler created by NewScene.
//
// Parameters:
//   - handler: the pre-configured SSAOHandler
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithSSAOHandler(handler ssao.Handler) SceneBuilderOption {
	return func(s *scene) {
		s.ssaoHandler = handler
	}
}

// WithCompositionHandler attaches a pre-configured CompositionHandler to the
// scene, replacing the default handler created by NewScene.
//
// Parameters:
//   - handler: the pre-configured CompositionHandler
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithCompositionHandler(handler composition.Handler) SceneBuilderOption {
	return func(s *scene) {
		s.compositionHandler = handler
	}
}

// WithSSRHandler attaches a pre-configured SSRHandler to the scene,
// replacing the default handler created by NewScene.
//
// Parameters:
//   - handler: the pre-configured SSRHandler
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithSSRHandler(handler ssr.Handler) SceneBuilderOption {
	return func(s *scene) {
		s.ssrHandler = handler
	}
}

// WithTAAHandler attaches a pre-configured TAAHandler to the scene,
// replacing the default handler created by NewScene.
//
// Parameters:
//   - handler: the pre-configured TAAHandler
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithTAAHandler(handler taa.Handler) SceneBuilderOption {
	return func(s *scene) {
		s.taaHandler = handler
	}
}

// WithPhysicsHandler creates a Physics instance with the given options and attaches it
// to the scene. GPU resources are initialized lazily when the first rigid body
// object is added via Add. If not called, objects with RigidBodies will be
// rendered without physics simulation — no forces, collisions, or integration
// will be applied.
//
// Parameters:
//   - opts: variadic physics builder options to configure the physics simulation
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithPhysicsHandler(handler physics.Physics) SceneBuilderOption {
	return func(s *scene) {
		s.SetPhysicsHandler(handler)
	}
}

// WithScreenSize sets the initial screen dimensions on the scene. These dimensions
// are used for Forward+ tile culling calculations and are automatically updated
// when Resize is called. If not set, AddLight will use zero dimensions until the
// first Resize call.
//
// Parameters:
//   - width: screen width in pixels
//   - height: screen height in pixels
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithScreenSize(width, height int) SceneBuilderOption {
	return func(s *scene) {
		s.screenWidth = width
		s.screenHeight = height
	}
}

// WithMaxBonesGPU sets the maximum number of bone matrices per skinned mesh
// instance allocated by the GPU compute animator. Defaults to 64.
//
// Parameters:
//   - n: the maximum bone count (minimum 1)
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithMaxBonesGPU(n uint64) SceneBuilderOption {
	return func(s *scene) {
		if n < 1 {
			n = 1
		}
		s.maxBonesGPU = n
	}
}

// WithLODEnabled enables or disables per-frame Level-of-Detail mesh selection.
// When enabled, the scene selects LOD levels based on camera distance each frame.
//
// Parameters:
//   - enabled: true to enable LOD selection
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithLODEnabled(enabled bool) SceneBuilderOption {
	return func(s *scene) {
		s.lodEnabled = enabled
	}
}

// WithLODDistances sets the camera distance thresholds for LOD level transitions.
// Objects farther than lod1 use LOD1; objects farther than lod2 use LOD2.
//
// Parameters:
//   - lod1: distance at which LOD1 activates
//   - lod2: distance at which LOD2 activates (must be > lod1)
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithLODDistances(lod1, lod2 float32) SceneBuilderOption {
	return func(s *scene) {
		s.lod1Distance = lod1
		s.lod2Distance = lod2
	}
}

// WithLODShadowBias sets the number of extra LOD levels applied to shadow
// rendering. A bias of 1 means shadows use one coarser LOD level than the
// visible mesh, reducing shadow pass geometry.
//
// Parameters:
//   - bias: additional LOD levels for shadow passes (default 1)
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithLODShadowBias(bias int) SceneBuilderOption {
	return func(s *scene) {
		s.lodShadowBias = bias
	}
}

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
		mu:                       &sync.RWMutex{},
		name:                     name,
		active:                   false,
		cam:                      cam,
		r:                        r,
		animatorPool:             make(map[model.Model][]animator.Animator),
		registry:                 make(map[uint64]game_object.GameObject),
		instanceLookup:           make(map[animator.Animator]map[uint32]uint64),
		shadowIndirectBuffers:    make(map[animator.Animator]*wgpu.Buffer),
		animIndirectBinding:      make(map[animator.Animator]int),
		shadowAnimationProviders: make(map[animator.Animator]bind_group_provider.BindGroupProvider),
		nextID:                   1,
		computeWorkers:           max(runtime.NumCPU()-1, 1),
		maxBonesGPU:              64,
		drawBindGroupsPool:       make([]bind_group_provider.BindGroupProvider, 0, 3),
		drawDeclsPool:            make([]shader.Annotation, 0, 32),
		drawGroupProvidersPool:   make(map[int]bind_group_provider.BindGroupProvider, 8),
		lightHandler:             light.NewLightingHandler(),
		gBufferHandler:           gbuffer.NewHandler(),
		ssaoHandler:              ssao.NewHandler(),
		compositionHandler:       composition.NewHandler(composition.WithToneMappingEnabled(true), composition.WithExposure(1.0)),
		ssrHandler:               ssr.NewHandler(),
		taaHandler:               taa.NewHandler(),
		physicsSyncGroup:         make(map[int]bind_group_provider.BindGroupProvider),
		physicsAnimBinding:       -1,
		lodLevelCache:            make(map[animator.Animator]int),
		lodShadowBias:            1,
		drawBindGroupCache:       make(map[drawCacheKey][]bind_group_provider.BindGroupProvider),
		drawCacheDirty:           true,
	}

	for _, option := range options {
		option(s)
	}

	s.buildInjectionMap()
	s.r.SetInjections(s.injections)

	// Initialize the compute pool after options so WithComputeWorkers can override the default.
	// Queue size of 256 accommodates typical animator group counts with headroom.
	s.computePool = worker.NewDynamicWorkerPool(s.computeWorkers, 256, 1*time.Second)

	// Initialize the camera's bind group on the GPU using the layout from the
	// engine's standard vertex shader. The shader is loaded internally so the
	// caller never needs to supply one. The camera group index is resolved from
	// the shader's pre-processor declarations rather than fuzzy var-name matching.
	cameraVertShader := shader.NewShader("_camera_init_vert", shader.ShaderTypeVertex, "engine/model/assets/simple-vert.wgsl", shader.WithInjections(s.injections))
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
		bgp.SetSlot(1)
		if err := r.InitBindGroup(bgp, cameraVertShader.BindGroupLayoutDescriptor(cameraGroup), nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init camera bind group slot 1: %v", err))
		}
		bgp.SetSlot(0)
	}

	// Create a 1×1 Hi-Z fallback texture so animators added before lighting
	// is initialized (or in unlit scenes) always have a valid view bound at
	// @group(1) in the occlusion-culling compute shader.
	if hizView, hizTex, _, _, _, err := r.CreateHiZTextures(1, 1); err == nil {
		s.hizFallbackTexture = hizTex
		s.hizFallbackView = hizView
	}

	s.Delegate = s
	return s
}
