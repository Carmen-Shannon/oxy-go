package scene

import (
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
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

// WithShadowHalfExtent sets the orthographic half-extent of the directional shadow
// frustum in world units. Larger values capture more of the scene but reduce shadow
// resolution. Default is light.DefaultShadowHalfExtent (40.0).
//
// Parameters:
//   - halfExtent: half-size of the shadow frustum in world units
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithShadowHalfExtent(halfExtent float32) SceneBuilderOption {
	return func(s *scene) {
		s.shadowHalfExtent = halfExtent
	}
}

// WithShadowNearFar sets the near and far planes for the directional shadow projection.
// Default is light.DefaultShadowNear (0.1) and light.DefaultShadowFar (200.0).
//
// Parameters:
//   - near: near plane distance
//   - far: far plane distance
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithShadowNearFar(near, far float32) SceneBuilderOption {
	return func(s *scene) {
		s.shadowNear = near
		s.shadowFar = far
	}
}

// WithShadowBias sets the depth comparison bias used during shadow sampling to
// reduce shadow acne. Default is light.DefaultShadowBias (0.001).
//
// Parameters:
//   - bias: the depth bias value
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithShadowBias(bias float32) SceneBuilderOption {
	return func(s *scene) {
		s.shadowBias = bias
	}
}

// WithShadowNormalBiasScale sets the multiplier applied to the shadow-map
// texel world-size to derive the normal-offset bias. The normal offset
// shifts the shadow lookup position along the surface normal, preventing
// self-shadowing on concave geometry. Default is light.DefaultShadowNormalBiasScale (3.0).
//
// Parameters:
//   - scale: multiplier on per-texel world size (typically 2.0–4.0)
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithShadowNormalBiasScale(scale float32) SceneBuilderOption {
	return func(s *scene) {
		s.shadowNormalBiasScale = scale
	}
}

// WithShadowMapResolution sets the width and height in texels of the shadow
// depth texture. Higher values produce sharper shadows at the cost of more
// GPU memory and fill-rate. Must be set before InitShadowMap / InitLighting
// is called, as the texture is allocated once. Default is light.ShadowMapResolution (2048).
//
// Parameters:
//   - resolution: shadow map width and height in texels (e.g. 1024, 2048, 4096)
//
// Returns:
//   - SceneBuilderOption: option function to apply
func WithShadowMapResolution(resolution int) SceneBuilderOption {
	return func(s *scene) {
		s.shadowMapResolution = resolution
	}
}
