// Simple transform update + frustum culling compute shader
//
// Runs one invocation per instance per frame. Updates rotation animation on
// the GPU and builds a 4x4 model matrix. When frustum culling is active,
// only visible instances are compacted into the output buffer and counted
// via an atomic indirect draw argument, enabling DrawIndexedIndirect without
// CPU readback.

//@oxy:include animation_data
//@oxy:include frustum_plane
//@oxy:include global_data
//@oxy:include indirect_args

//@oxy:group 0 0 storage_uniform globals global_data
//@oxy:group 0 1 storage_read instance_data array<animation_data>
//@oxy:provider 0 2 animator_output
@group(0) @binding(2) var<storage, read_write> output_transforms: array<f32>;
//@oxy:group 0 3 storage_read_write indirect_args indirect_args
//@oxy:provider 1 0 animator_hiz hiz_texture
@group(1) @binding(0) var hiz_texture: texture_2d<f32>;
//@oxy:provider 1 1 animator_max_hiz hiz_max_texture
@group(1) @binding(1) var hiz_max_texture: texture_2d<f32>;

// Returns true if a bounding sphere at `pos` with `radius` is at least
// partially inside all six planes.
fn is_visible(pos: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = globals.planes[i];
        let dist = dot(plane.normal, pos) + plane.distance;
        if (dist < -radius) {
            return false;
        }
    }
    return true;
}

// Builds a column-major 4x4 TRS matrix and writes it plus the instance flag slot
// into the output buffer at the given float offset.
fn build_transform(pos: vec3<f32>, rot: vec3<f32>, scale: vec3<f32>, instance_flags: u32, out_idx: u32) {
    let cx = cos(rot.x); let sx = sin(rot.x);
    let cy = cos(rot.y); let sy = sin(rot.y);
    let cz = cos(rot.z); let sz = sin(rot.z);

    // Combined rotation Z * Y * X (column-major)
    // Column 0
    output_transforms[out_idx +  0u] = scale.x * (cz * cy);
    output_transforms[out_idx +  1u] = scale.x * (sz * cy);
    output_transforms[out_idx +  2u] = scale.x * (-sy);
    output_transforms[out_idx +  3u] = 0.0;
    // Column 1
    output_transforms[out_idx +  4u] = scale.y * (cz * sy * sx - sz * cx);
    output_transforms[out_idx +  5u] = scale.y * (sz * sy * sx + cz * cx);
    output_transforms[out_idx +  6u] = scale.y * (cy * sx);
    output_transforms[out_idx +  7u] = 0.0;
    // Column 2
    output_transforms[out_idx +  8u] = scale.z * (cz * sy * cx + sz * sx);
    output_transforms[out_idx +  9u] = scale.z * (sz * sy * cx - cz * sx);
    output_transforms[out_idx + 10u] = scale.z * (cy * cx);
    output_transforms[out_idx + 11u] = 0.0;
    // Column 3 (translation)
    output_transforms[out_idx + 12u] = pos.x;
    output_transforms[out_idx + 13u] = pos.y;
    output_transforms[out_idx + 14u] = pos.z;
    output_transforms[out_idx + 15u] = 1.0;
    output_transforms[out_idx + 16u] = bitcast<f32>(instance_flags);
    output_transforms[out_idx + 17u] = 0.0;
    output_transforms[out_idx + 18u] = 0.0;
    output_transforms[out_idx + 19u] = 0.0;
}

// Returns true if the model-space AABB (transformed to world space by pos/rot/scale) is
// fully occluded by the previous frame's Hi-Z depth pyramid. Returns false (never occludes)
// when globals.hiz_mip_count == 0 (Hi-Z not yet initialized).
//
// Samples mip 0 (full resolution) at the AABB footprint center: one pixel, no
// contamination from adjacent geometry at coarser mip levels.
fn is_occluded(pos: vec3<f32>, rot: vec3<f32>, scale: vec3<f32>) -> bool {
    if (globals.hiz_mip_count == 0u) {
        return false;
    }

    let cx = cos(rot.x); let sx = sin(rot.x);
    let cy = cos(rot.y); let sy = sin(rot.y);
    let cz = cos(rot.z); let sz = sin(rot.z);
    let col0 = vec4<f32>(scale.x * (cz * cy), scale.x * (sz * cy), scale.x * (-sy), 0.0);
    let col1 = vec4<f32>(scale.y * (cz * sy * sx - sz * cx), scale.y * (sz * sy * sx + cz * cx), scale.y * (cy * sx), 0.0);
    let col2 = vec4<f32>(scale.z * (cz * sy * cx + sz * sx), scale.z * (sz * sy * cx - cz * sx), scale.z * (cy * cx), 0.0);
    let model = mat4x4<f32>(col0, col1, col2, vec4<f32>(pos, 1.0));

    let bmin = globals.bounding_min;
    let bmax = globals.bounding_max;
    let vp = globals.view_proj;

    var min_z = 1.0;
    var min_u = 1.0; var max_u = 0.0;
    var min_v = 1.0; var max_v = 0.0;
    var any_on_screen = false;

    for (var ci = 0u; ci < 8u; ci++) {
        let px = select(bmin.x, bmax.x, (ci & 1u) != 0u);
        let py = select(bmin.y, bmax.y, (ci & 2u) != 0u);
        let pz = select(bmin.z, bmax.z, (ci & 4u) != 0u);
        let world = model * vec4<f32>(px, py, pz, 1.0);
        let clip = vp * world;
        if (clip.w <= 0.0) { return false; }
        let ndc = clip.xyz / clip.w;
        min_z = min(min_z, ndc.z);
        if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) { continue; }
        let u = ndc.x * 0.5 + 0.5;
        let v = 1.0 - (ndc.y * 0.5 + 0.5);
        min_u = min(min_u, u); max_u = max(max_u, u);
        min_v = min(min_v, v); max_v = max(max_v, v);
        any_on_screen = true;
    }

    if (!any_on_screen) { return false; }

    let foot_w = (max_u - min_u) * f32(globals.screen_width);
    let foot_h = (max_v - min_v) * f32(globals.screen_height);
    let foot_max = max(max(foot_w, foot_h), 1.0);
    let mip_level = i32(min(u32(floor(log2(foot_max))), globals.hiz_mip_count - 1u));

    let mip_dims = vec2<i32>(textureDimensions(hiz_max_texture, mip_level));
    let tc_min = clamp(vec2<i32>(vec2<f32>(min_u, min_v) * vec2<f32>(mip_dims)), vec2<i32>(0), mip_dims - vec2<i32>(1));
    let tc_max = clamp(vec2<i32>(vec2<f32>(max_u, max_v) * vec2<f32>(mip_dims)), vec2<i32>(0), mip_dims - vec2<i32>(1));
    let s0 = textureLoad(hiz_max_texture, vec2<i32>(tc_min.x, tc_min.y), mip_level).r;
    let s1 = textureLoad(hiz_max_texture, vec2<i32>(tc_max.x, tc_min.y), mip_level).r;
    let s2 = textureLoad(hiz_max_texture, vec2<i32>(tc_min.x, tc_max.y), mip_level).r;
    let s3 = textureLoad(hiz_max_texture, vec2<i32>(tc_max.x, tc_max.y), mip_level).r;
    let max_hiz = max(max(s0, s1), max(s2, s3));
    return min_z > max_hiz + 0.001;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= globals.instance_count) {
        return;
    }

    // Read instance transform data (rotation is advanced on CPU and uploaded each frame).
    var anim = instance_data[idx];

    if (globals.culling_enabled == 0u) {
        build_transform(anim.pos, anim.rot, anim.scale, anim.instance_flags, idx * 20u);
        return;
    }

    // Frustum cull — scale bounding radius by the instance's largest axis
    // so that non-uniformly scaled instances are not incorrectly culled.
    let max_scale = max(anim.scale.x, max(anim.scale.y, anim.scale.z));
    if (is_visible(anim.pos, globals.bounding_radius * max_scale)) {
        if (!is_occluded(anim.pos, anim.rot, anim.scale)) {
            let out_slot = atomicAdd(&indirect_args.instance_count, 1u);
            build_transform(anim.pos, anim.rot, anim.scale, anim.instance_flags, out_slot * 20u);
        }
    }
}
