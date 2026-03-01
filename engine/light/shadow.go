package light

// ShadowMapResolution is the default width and height in texels of the shadow
// depth texture. Scenes use this as their initial value but can override it
// via the WithShadowMapResolution builder option.
const ShadowMapResolution = 2048

// DefaultShadowHalfExtent is the default orthographic half-extent (in world units)
// used for the directional light shadow frustum. Controls how much of the scene
// around the camera center is captured in the shadow map.
const DefaultShadowHalfExtent float32 = 40.0

// DefaultShadowNear is the default near plane for the directional light's
// orthographic shadow projection.
const DefaultShadowNear float32 = 0.1

// DefaultShadowFar is the default far plane for the directional light's
// orthographic shadow projection.
const DefaultShadowFar float32 = 200.0

// DefaultShadowBias is the constant depth bias applied to shadow comparisons
// to reduce shadow acne artifacts.
const DefaultShadowBias float32 = 0.001

// DefaultShadowNormalBiasScale is the multiplier applied to the shadow map
// texel world-size to compute the normal-offset bias. Higher values push
// the shadow sample point further along the surface normal, reducing
// self-shadowing on concave geometry at the cost of slight shadow
// detachment from contact points. Typical values are 2.0–4.0.
const DefaultShadowNormalBiasScale float32 = 3.0

// DefaultVSMBlurRadius is the default half-width (in texels) of the separable
// blur applied to the variance shadow map. The paper notes a minimum filter
// width of at least 4 is required to eliminate aliasing. The full kernel width
// is 2*radius+1 texels.
const DefaultVSMBlurRadius int = 4

// DefaultVSMMinVariance is the minimum variance clamped during Chebyshev's
// inequality evaluation in the VSM sampling shader. Prevents division by near-zero
// variance from producing hard shadow edges on perfectly planar geometry.
const DefaultVSMMinVariance float32 = 0.00001

// DefaultVSMLightBleedReduction is the exponent applied to the raw Chebyshev
// shadow probability to reduce light-bleeding artifacts. Higher values reduce
// light bleeding at the cost of darker shadow interiors. Typical range: 0.1–0.6.
const DefaultVSMLightBleedReduction float32 = 0.3

// DefaultVSMLightSize is the default world-space size of the area light used for
// PCSS penumbra estimation. Larger values produce wider soft-shadow penumbrae.
// Only relevant when PCSS (Phase 7) is enabled.
const DefaultVSMLightSize float32 = 1.0
