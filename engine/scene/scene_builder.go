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
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
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

// WithPhysics creates a Physics instance with the given options and attaches it
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
func WithPhysics(opts ...physics.PhysicsBuilderOption) SceneBuilderOption {
	return func(s *scene) {
		s.physicsHandler = physics.NewPhysics(opts...)
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
		maxBonesGPU:        64,
		drawBindGroupsPool: make([]bind_group_provider.BindGroupProvider, 0, 3),
		lightHandler:       light.NewLightingHandler(),
		physicsSyncGroup:   make(map[int]bind_group_provider.BindGroupProvider),
		physicsAnimBinding: -1,
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
	}
	return s
}
