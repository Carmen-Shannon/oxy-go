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

// Builds a column-major 4x4 TRS matrix and writes it into the output
// buffer at the given float offset.
fn build_transform(pos: vec3<f32>, rot: vec3<f32>, scale: vec3<f32>, out_idx: u32) {
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
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= globals.instance_count) {
        return;
    }

    // Read instance transform data (rotation is advanced on CPU and uploaded each frame).
    var anim = instance_data[idx];

    // Frustum cull — scale bounding radius by the instance's largest axis
    // so that non-uniformly scaled instances are not incorrectly culled.
    let max_scale = max(anim.scale.x, max(anim.scale.y, anim.scale.z));
    if (is_visible(anim.pos, globals.bounding_radius * max_scale)) {
        let out_slot = atomicAdd(&indirect_args.instance_count, 1u);
        build_transform(anim.pos, anim.rot, anim.scale, out_slot * 16u);
    }
}
