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

// WithVSMBlurRadius sets the half-width (in texels) of the separable blur applied
// to the variance shadow map. The full kernel width is 2*radius+1. The paper notes a
// minimum filter width of at least 4 is required to eliminate aliasing.
// Default is DefaultVSMBlurRadius (4).
//
// Parameters:
//   - radius: the blur half-width in texels
//
// Returns:
//   - LightingHandlerOption: a function that applies the blur radius option to a lightingHandlerImpl
func WithVSMBlurRadius(radius int) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.vsmBlurRadius = radius
	}
}

// WithVSMMinVariance sets the minimum variance clamped during Chebyshev's inequality
// evaluation. Prevents division by near-zero variance from producing hard shadow edges
// on perfectly planar geometry. Default is DefaultVSMMinVariance (0.00001).
//
// Parameters:
//   - minVariance: the minimum variance clamp value
//
// Returns:
//   - LightingHandlerOption: a function that applies the min variance option to a lightingHandlerImpl
func WithVSMMinVariance(minVariance float32) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.vsmMinVariance = minVariance
	}
}

// WithVSMLightBleedReduction sets the exponent applied to the raw Chebyshev shadow
// probability to reduce light-bleeding artifacts. Higher values reduce light bleeding
// at the cost of darker shadow interiors. Typical range: 0.1–0.6.
// Default is DefaultVSMLightBleedReduction (0.3).
//
// Parameters:
//   - reduction: the light bleed reduction exponent
//
// Returns:
//   - LightingHandlerOption: a function that applies the light bleed reduction option to a lightingHandlerImpl
func WithVSMLightBleedReduction(reduction float32) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.vsmLightBleedReduction = reduction
	}
}

// WithVSMLightSize sets the world-space size of the area light used for PCSS penumbra
// estimation. Larger values produce wider soft-shadow penumbrae. Only relevant when
// PCSS is enabled. Default is DefaultVSMLightSize (1.0).
//
// Parameters:
//   - size: the world-space light size
//
// Returns:
//   - LightingHandlerOption: a function that applies the light size option to a lightingHandlerImpl
func WithVSMLightSize(size float32) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.vsmLightSize = size
	}
}

// WithPCSSEnabled enables or disables Percentage-Closer Soft Shadows. PCSS uses a
// Summed-Area Table built from the VSM moments texture to provide per-pixel variable-width
// shadow filtering, producing contact-hardening soft shadows. Requires VSM to be enabled.
// Default is false.
//
// Parameters:
//   - enabled: true to enable PCSS, false to use constant-width VSM blur
//
// Returns:
//   - LightingHandlerOption: a function that applies the PCSS enabled option to a lightingHandlerImpl
func WithPCSSEnabled(enabled bool) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.pcssEnabled = enabled
	}
}

// WithGBufferHandler attaches a pre-configured GBufferHandler to the lighting
// subsystem, overriding the default that is auto-created by NewLightingHandler.
// GPU resources are initialized lazily during the first lighting initialization.
//
// Parameters:
//   - handler: the pre-configured GBufferHandler
//
// Returns:
//   - LightingHandlerOption: a function that applies the G-Buffer handler option to a lightingHandlerImpl
func WithGBufferHandler(handler GBufferHandler) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.gBufferHandler = handler
	}
}

// WithSSAOHandler attaches a pre-configured SSAOHandler to the lighting
// subsystem, overriding the default that is auto-created by NewLightingHandler.
// GPU resources are initialized lazily during the first lighting initialization.
//
// Parameters:
//   - handler: the pre-configured SSAOHandler
//
// Returns:
//   - LightingHandlerOption: a function that applies the SSAO handler option to a lightingHandlerImpl
func WithSSAOHandler(handler SSAOHandler) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.ssaoHandler = handler
	}
}

// WithProbeGrid attaches a pre-configured IrradianceProbeGrid to the lighting
// subsystem. The probe grid stores L2 spherical harmonic coefficients sampled
// by baking the scene from each probe position. GPU resources are initialized
// lazily during the first lighting initialization. If not set, no probe-based
// diffuse indirect lighting is applied.
//
// Parameters:
//   - handler: the pre-configured IrradianceProbeGrid
//
// Returns:
//   - LightingHandlerOption: a function that applies the probe grid handler option to a lightingHandlerImpl
func WithProbeGrid(handler IrradianceProbeGrid) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.probeGrid = handler
	}
}

// WithCompositionHandler attaches a pre-configured CompositionHandler to the
// lighting subsystem, overriding the default that is auto-created by
// NewLightingHandler. The composition handler manages the offscreen HDR render
// target and the full-screen tone mapping pass. GPU resources are initialized
// lazily during the first lighting initialization.
//
// Parameters:
//   - handler: the pre-configured CompositionHandler
//
// Returns:
//   - LightingHandlerOption: a function that applies the composition handler option to a lightingHandlerImpl
func WithCompositionHandler(handler CompositionHandler) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.compositionHandler = handler
	}
}

// WithSSRHandler attaches a pre-configured SSRHandler to the lighting
// subsystem, overriding the default that is auto-created by NewLightingHandler.
// The SSR handler reads the G-Buffer and writes to a texture sampled by the
// composition pass. GPU resources are initialized lazily during the first
// lighting initialization.
//
// Parameters:
//   - handler: the pre-configured SSRHandler
//
// Returns:
//   - LightingHandlerOption: a function that applies the SSR handler option to a lightingHandlerImpl
func WithSSRHandler(handler SSRHandler) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.ssrHandler = handler
	}
}
