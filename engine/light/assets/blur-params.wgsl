// BlurParams holds the configuration for a single separable blur pass.
//
// Fields:
//   direction     — (1,0) for horizontal, (0,1) for vertical
//   radius        — half-width of the box filter kernel in texels
//   gbuffer_scale — coordinate multiplier for depth texture lookups (1 = full-res, 2 = half-res SSAO)
struct BlurParams {
    direction: vec2<i32>,
    radius: i32,
    gbuffer_scale: i32,
};
