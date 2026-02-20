// Shadow depth vertex shader (static models)
//
// Minimal shader that transforms vertices to light clip space for shadow map
// generation. Outputs only @builtin(position) — no color, normal, or UV data
// is needed since the shadow pass only writes depth.
//
// Bind group layout:
//   @group(0) @binding(0) shadow_uniform — light view-projection matrix (uniform)
//   @group(1) @binding(0) instance_buffer — per-instance model matrices (storage)

// ── Vertex attributes ──────────────────────────────────────────────
// Must match Go's model.GPUVertex struct layout exactly (64 bytes).
// Only position is used; other fields are declared for stride compatibility.
//@oxy:include vertex
// struct VertexInput {
//     @location(0) position: vec3<f32>,
//     @location(1) normal:   vec3<f32>,
//     @location(2) uv:       vec2<f32>,
//     @location(3) color:    vec4<f32>,
//     @location(4) tangent:  vec4<f32>,
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

// ── Per-instance model matrix (produced by compute shader) ─────────
//@oxy:include instance_data
// struct InstanceData {
//     model: mat4x4<f32>,
// };

// ── Bind groups ────────────────────────────────────────────────────
//@oxy:group 0 0 storage_uniform shadow_uniform shadow_uniform
// @group(0) @binding(0) var<uniform> shadow_uniform: ShadowUniform;
//@oxy:group 1 0 storage_read instance_buffer array<instance_data>
// @group(1) @binding(0) var<storage, read> instance_buffer: array<InstanceData>;

// ── Entry point ────────────────────────────────────────────────────
@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let model_matrix = instance_buffer[instance_idx].model;
    let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);

    var out: VertexOutput;
    out.clip_position = shadow_uniform.light_vp * world_pos;
    return out;
}
