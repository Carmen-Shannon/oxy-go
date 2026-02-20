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
