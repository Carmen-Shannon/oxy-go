// Simple transform update + frustum culling compute shader
//
// Runs one invocation per instance per frame. Updates rotation animation on
// the GPU and builds a 4x4 model matrix. When frustum culling is active,
// only visible instances are compacted into the output buffer and counted
// via an atomic indirect draw argument, enabling DrawIndexedIndirect without
// CPU readback.

// ── Per-instance animation data (64 bytes = 4 × vec4) ──────────────
// Must match Go's instanceAnimationData struct exactly.
//@oxy:include animation_data
// struct AnimationData {
//     rot_speed: vec3<f32>,       // offset  0: rotation speed (rad/s)
//     _pad0: f32,                 // offset 12: vec3 pad
//     rot: vec3<f32>,             // offset 16: current rotation angles
//     _pad1: f32,                 // offset 28: vec3 pad
//     pos: vec3<f32>,             // offset 32: world position
//     _pad2: f32,                 // offset 44: vec3 pad
//     scale: vec3<f32>,           // offset 48: non-uniform scale
//     _pad3: f32,                 // offset 60: vec3 pad
// }

// ── Frustum plane ──────────────────────────────────────────────────
//@oxy:include frustum_plane
// struct FrustumPlane {
//     normal: vec3<f32>,
//     distance: f32,
// }

// ── Per-frame global uniform (112 bytes) ───────────────────────────
// Matches Go's simpleCullUniformData struct.
//@oxy:include global_data
// struct GlobalData {
//     instance_count: u32,
//     delta_time: f32,
//     bounding_radius: f32,
//     _padding: f32,
//     planes: array<FrustumPlane, 6>,
// }

// ── Indirect draw arguments ────────────────────────────────────────
// Layout matches WebGPU's DrawIndexedIndirect. instance_count is atomic
// so each visible instance can safely claim an output slot.
//@oxy:include indirect_args
// struct IndirectArgs {
//     index_count: u32,
//     instance_count: atomic<u32>,
//     first_index: u32,
//     base_vertex: u32,
//     first_instance: u32,
// }

// ── Bind group 0 ───────────────────────────────────────────────────
//@oxy:group 0 0 storage_uniform globals global_data
// @group(0) @binding(0) var<uniform> globals: GlobalData;
//@oxy:group 0 1 storage_read_write instance_data array<animation_data>
// @group(0) @binding(1) var<storage, read_write> instance_data: array<AnimationData>;
//@oxy:provider 0 2 animator_output
@group(0) @binding(2) var<storage, read_write> output_transforms: array<f32>;
//@oxy:group 0 3 storage_read_write indirect_args indirect_args
// @group(0) @binding(3) var<storage, read_write> indirect_args: IndirectArgs;

// ── Frustum test ───────────────────────────────────────────────────
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

// ── Matrix builder ─────────────────────────────────────────────────
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

// ── Entry point ────────────────────────────────────────────────────
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= globals.instance_count) {
        return;
    }

    // Read and update rotation (frame-rate independent)
    var anim = instance_data[idx];
    anim.rot = anim.rot + anim.rot_speed * globals.delta_time;

    // Wrap to [0, 2π) to prevent float32 precision loss in sin/cos
    let TWO_PI = vec3<f32>(6.283185307, 6.283185307, 6.283185307);
    anim.rot = fract(anim.rot / TWO_PI) * TWO_PI;

    // Write back updated rotation
    instance_data[idx].rot = anim.rot;

    // Frustum cull — only visible instances are compacted into the output
    if (is_visible(anim.pos, globals.bounding_radius)) {
        let out_slot = atomicAdd(&indirect_args.instance_count, 1u);
        build_transform(anim.pos, anim.rot, anim.scale, out_slot * 16u);
    }
}
