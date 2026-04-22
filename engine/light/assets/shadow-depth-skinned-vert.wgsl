// Shadow depth vertex shader (skinned models)
//
// Applies per-instance bone skinning and transforms vertices to light clip
// space for depth-only shadow rendering. No fragment shader is needed — the
// hardware depth buffer stores the depth automatically via the rasterizer.
//
// Bind group layout:
//   @group(0) @binding(0) shadow_uniform — light VP matrix (uniform)
//   @group(1) @binding(0) instance_buffer — per-instance model + bone matrices (storage)

// Maximum number of bones supported per skeleton. Must match the compute
// shader's MAX_BONES constant so the per-instance stride is consistent.
//@oxy:inject MAX_BONES u32 max_bones

//@oxy:include skinned_vertex
//@oxy:include shadow_uniform

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// The compute shader writes each instance as a flat sequence of vec4<f32>:
//   [model_matrix: 4 vec4] [instance_flags: 1 vec4] [bone_0: 4 vec4] ... [bone_(MAX_BONES-1): 4 vec4]
// Total per instance: 5 vec4 header entries + 4 vec4 per bone matrix.
const FLOATS_PER_INSTANCE: u32 = 5u + MAX_BONES * 4u;

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
    let base = instance_idx * FLOATS_PER_INSTANCE;

    // Model matrix is the first 4 vec4 entries.
    let model_matrix = read_mat4(base);

    // Bone matrices start right after the model matrix and instance flag slot.
    let bone_base = base + 5u;

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

    var out: VertexOutput;
    out.clip_position = shadow_uniform.light_vp * world_pos;
    return out;
}
