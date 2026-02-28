package scene

import (
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
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
