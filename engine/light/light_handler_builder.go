package light

// LightingHandlerOption is a functional option for configuring a LightingHandler
// during construction via NewLightingHandler.
type LightingHandlerOption func(*lightingHandlerImpl)

// WithShadowHalfExtent sets the orthographic half-extent of the directional shadow
// frustum in world units. Larger values capture more of the scene but reduce shadow
// resolution. Default is DefaultShadowHalfExtent (40.0).
//
// Parameters:
//   - halfExtent: half-size of the shadow frustum in world units
//
// Returns:
//   - LightingHandlerOption: a function that applies the half-extent option to a lightingHandlerImpl
func WithShadowHalfExtent(halfExtent float32) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.shadowHalfExtent = halfExtent
	}
}

// WithShadowNearFar sets the near and far planes for the directional shadow projection.
// Default is DefaultShadowNear (0.1) and DefaultShadowFar (200.0).
//
// Parameters:
//   - near: near plane distance
//   - far: far plane distance
//
// Returns:
//   - LightingHandlerOption: a function that applies the near/far option to a lightingHandlerImpl
func WithShadowNearFar(near, far float32) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.shadowNear = near
		h.shadowFar = far
	}
}

// WithShadowBias sets the depth comparison bias used during shadow sampling to
// reduce shadow acne. Default is DefaultShadowBias (0.001).
//
// Parameters:
//   - bias: the depth bias value
//
// Returns:
//   - LightingHandlerOption: a function that applies the bias option to a lightingHandlerImpl
func WithShadowBias(bias float32) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.shadowBias = bias
	}
}

// WithShadowNormalBiasScale sets the multiplier applied to the shadow-map texel
// world-size to derive the normal-offset bias. The normal offset shifts the
// shadow lookup position along the surface normal, preventing self-shadowing
// on concave geometry. Default is DefaultShadowNormalBiasScale (3.0).
//
// Parameters:
//   - scale: multiplier on per-texel world size (typically 2.0–4.0)
//
// Returns:
//   - LightingHandlerOption: a function that applies the normal bias scale option to a lightingHandlerImpl
func WithShadowNormalBiasScale(scale float32) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.shadowNormalBiasScale = scale
	}
}

// WithShadowMapResolution sets the width and height in texels of the shadow depth
// texture. Higher values produce sharper shadows at the cost of more GPU memory and
// fill-rate. Must be set before GPU initialization. Default is ShadowMapResolution (2048).
//
// Parameters:
//   - resolution: shadow map width and height in texels (e.g. 1024, 2048, 4096)
//
// Returns:
//   - LightingHandlerOption: a function that applies the resolution option to a lightingHandlerImpl
func WithShadowMapResolution(resolution int) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.shadowMapResolution = resolution
	}
}

// WithAmbientColor sets the initial ambient light color for the scene.
// Default is black (no ambient contribution).
//
// Parameters:
//   - color: the ambient RGB color
//
// Returns:
//   - LightingHandlerOption: a function that applies the ambient color option to a lightingHandlerImpl
func WithAmbientColor(color [3]float32) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.ambientColor = color
	}
}
