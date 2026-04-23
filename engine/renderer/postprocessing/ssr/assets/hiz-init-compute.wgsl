// Hi-Z depth pyramid initialisation pass.
//
// Copies the hardware Depth24Plus G-Buffer depth values into mip 0 of the
// R32Float Hi-Z texture. This is the first step in building the hierarchical
// depth pyramid used by the Hi-Z SSR ray march.

//@oxy:provider 0 0 hiz_init gbuffer_depth
@group(0) @binding(0) var gbuffer_depth: texture_depth_2d;

//@oxy:provider 0 1 hiz_init hiz_out
@group(0) @binding(1) var hiz_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(hiz_out);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let depth = textureLoad(gbuffer_depth, vec2<i32>(gid.xy), 0);
    textureStore(hiz_out, vec2<i32>(gid.xy), vec4<f32>(depth, 0.0, 0.0, 0.0));
}
