// Rainbow fragment shader
//
// Receives per-vertex color interpolated across the triangle by the
// rasterizer and outputs it directly. No textures or lighting — pure
// vertex-color pass-through for rainbow-style rendering.
//
// The FragmentInput layout matches the VertexOutput of simple-vert.wgsl
// so this shader can be used as a drop-in fragment replacement for any
// static (non-skinned) model that uses per-vertex colors.

struct FragmentInput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv:             vec2<f32>,
    @location(1) normal:         vec3<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) world_position: vec3<f32>,
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return in.color;
}
