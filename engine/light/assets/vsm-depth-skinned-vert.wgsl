// VSM shadow depth vertex shader (skinned models)
//
// Applies per-instance bone skinning, transforms vertices to light clip space,
// and computes a linear light-space depth for the VSM fragment shader. The
// linear depth is computed from the view-only matrix (no projection) and
// normalized to [0, 1] using near/far.
//
// Bind group layout:
//   @group(0) @binding(0) shadow_uniform — light VP, view matrix, near/far (uniform)
//   @group(1) @binding(0) instance_buffer — per-instance model + bone matrices (storage)

// Maximum number of bones supported per skeleton. Must match the compute
// shader's MAX_BONES constant so the per-instance stride is consistent.
const MAX_BONES: u32 = 64u;

//@oxy:include skinned_vertex
//@oxy:include shadow_uniform

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) light_depth: f32,
};

//@oxy:group 0 0 storage_uniform shadow_uniform shadow_uniform
//@oxy:provider 1 0 animator
@group(1) @binding(0) var<storage, read> instance_buffer: array<vec4<f32>>;

// read_mat4 reconstructs a mat4x4 from 4 consecutive vec4 entries in the flat buffer.
fn read_mat4(base: u32) -> mat4x4<f32> {
    return mat4x4<f32>(
        instance_buffer[base],
        instance_buffer[base + 1u],
        instance_buffer[base + 2u],
        instance_buffer[base + 3u],
    );
}

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let floatsPerInstance = (1u + MAX_BONES) * 4u;
    let base = instance_idx * floatsPerInstance;

    // Model matrix is the first 4 vec4 entries.
    let model_matrix = read_mat4(base);

    // Bone matrices start right after the model matrix.
    let bone_base = base + 4u;

    // Blend skinning: accumulate weighted bone transforms.
    var skinned_pos = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let bone_idx = vertex.bone_indices[i];
        let weight = vertex.bone_weights[i];
        if weight > 0.0 {
            let bone_matrix = read_mat4(bone_base + bone_idx * 4u);
            skinned_pos += weight * (bone_matrix * vec4<f32>(vertex.position, 1.0));
        }
    }

    let world_pos = model_matrix * skinned_pos;

    // Compute linear depth from the view-only matrix, normalized to [0, 1].
    let view_pos = shadow_uniform.light_view * world_pos;
    let linear_depth = (-view_pos.z - shadow_uniform.shadow_near)
                     / (shadow_uniform.shadow_far - shadow_uniform.shadow_near);

    var out: VertexOutput;
    out.clip_position = shadow_uniform.light_vp * world_pos;
    out.light_depth = linear_depth;
    return out;
}
