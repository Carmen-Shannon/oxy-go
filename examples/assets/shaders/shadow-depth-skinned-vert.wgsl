// Shadow depth vertex shader (skinned models)
//
// Minimal shader that applies per-instance bone skinning and transforms
// vertices to light clip space for shadow map generation. Outputs only
// @builtin(position) — no color, normal, or UV data is needed since the
// shadow pass only writes depth.
//
// Bind group layout:
//   @group(0) @binding(0) shadow_uniform — light view-projection matrix (uniform)
//   @group(1) @binding(0) instance_buffer — per-instance model + bone matrices (storage)

// Maximum number of bones supported per skeleton. Must match the compute
// shader's MAX_BONES constant so the per-instance stride is consistent.
const MAX_BONES: u32 = 64u;

// ── Vertex attributes ──────────────────────────────────────────────
// Must match Go's model.GPUSkinnedVertex struct layout exactly (96 bytes).
// Only position, bone_indices, and bone_weights are used; other fields
// are declared for stride compatibility.
//@oxy:include skinned_vertex
// struct VertexInput {
//     @location(0) position:     vec3<f32>,
//     @location(1) normal:       vec3<f32>,
//     @location(2) uv:           vec2<f32>,
//     @location(3) color:        vec4<f32>,
//     @location(4) tangent:      vec4<f32>,
//     @location(5) bone_indices: vec4<u32>,
//     @location(6) bone_weights: vec4<f32>,
// };

// ── Output ─────────────────────────────────────────────────────────
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// ── Shadow uniform ─────────────────────────────────────────────────
//@oxy:include shadow_uniform
// struct ShadowUniform {
//     light_vp: mat4x4<f32>,
// };

// ── Per-instance data layout in flat vec4 storage ──────────────────
// The compute shader writes each instance as a flat sequence of vec4<f32>:
//   [model_matrix: 4 vec4] [bone_0: 4 vec4] ... [bone_(MAX_BONES-1): 4 vec4]
// Total per instance: (1 + MAX_BONES) × 4 vec4.
const FLOATS_PER_INSTANCE: u32 = (1u + MAX_BONES) * 4u;

// ── Bind groups ────────────────────────────────────────────────────
//@oxy:group 0 0 storage_uniform shadow_uniform shadow_uniform
// @group(0) @binding(0) var<uniform> shadow_uniform: ShadowUniform;
//@oxy:provider 1 0 animator
@group(1) @binding(0) var<storage, read> instance_buffer: array<vec4<f32>>;

// ── Helpers ────────────────────────────────────────────────────────

// read_mat4 reconstructs a mat4x4 from 4 consecutive vec4 entries in the flat buffer.
fn read_mat4(base: u32) -> mat4x4<f32> {
    return mat4x4<f32>(
        instance_buffer[base],
        instance_buffer[base + 1u],
        instance_buffer[base + 2u],
        instance_buffer[base + 3u],
    );
}

// ── Entry point ────────────────────────────────────────────────────
@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let base = instance_idx * FLOATS_PER_INSTANCE;

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

    var out: VertexOutput;
    out.clip_position = shadow_uniform.light_vp * world_pos;
    return out;
}
