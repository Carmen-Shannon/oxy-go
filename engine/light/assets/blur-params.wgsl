// BlurParams holds the configuration for a single separable blur pass.
//
// Fields:
//   direction     — (1,0) for horizontal, (0,1) for vertical
//   radius        — half-width of the box filter kernel in texels
//   gbuffer_scale — coordinate multiplier for depth texture lookups (1 = full-res, 2 = half-res SSAO)
//   cascade_width — per-cascade atlas column width in texels; when > 0, horizontal samples are
//                   clamped to the originating cascade's column to prevent cross-cascade moment bleed
//   _pad          — padding to maintain 8-byte struct alignment (unused)
struct BlurParams {
    direction:     vec2<i32>,
    radius:        i32,
    gbuffer_scale: i32,
    cascade_width: i32,
    _pad:          i32,
};
