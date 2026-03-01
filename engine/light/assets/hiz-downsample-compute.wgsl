// Hi-Z depth pyramid downsample pass.
//
// Reads from the previous mip level of the Hi-Z texture and writes the
// minimum of each 2×2 texel block to the current mip level. Using the
// minimum (closest depth) allows the Hi-Z SSR ray march to conservatively
// detect potential intersections: if the ray depth is behind the min-depth
// in a cell, the ray may have hit a surface and must refine to a finer level.

//@oxy:provider 0 0 hiz_down hiz_in
@group(0) @binding(0) var hiz_in: texture_2d<f32>;

//@oxy:provider 0 1 hiz_down hiz_out
@group(0) @binding(1) var hiz_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_size = textureDimensions(hiz_out);
    if (gid.x >= out_size.x || gid.y >= out_size.y) {
        return;
    }

    let src = vec2<i32>(gid.xy) * 2;
    let in_size = vec2<i32>(textureDimensions(hiz_in, 0));

    // Clamp source coordinates to handle odd-sized textures at mip boundaries.
    let s00 = clamp(src, vec2<i32>(0), in_size - 1);
    let s10 = clamp(src + vec2<i32>(1, 0), vec2<i32>(0), in_size - 1);
    let s01 = clamp(src + vec2<i32>(0, 1), vec2<i32>(0), in_size - 1);
    let s11 = clamp(src + vec2<i32>(1, 1), vec2<i32>(0), in_size - 1);

    let d00 = textureLoad(hiz_in, s00, 0).r;
    let d10 = textureLoad(hiz_in, s10, 0).r;
    let d01 = textureLoad(hiz_in, s01, 0).r;
    let d11 = textureLoad(hiz_in, s11, 0).r;

    let min_depth = min(min(d00, d10), min(d01, d11));
    textureStore(hiz_out, vec2<i32>(gid.xy), vec4<f32>(min_depth, 0.0, 0.0, 0.0));
}
