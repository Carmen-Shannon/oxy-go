// Shadow depth vertex shader (static models)
//
// Minimal shader that transforms vertices to light clip space for shadow map
// generation. Outputs only @builtin(position) — no color, normal, or UV data
// is needed since the shadow pass only writes depth.
//
// Bind group layout:
//   @group(0) @binding(0) shadow_uniform — light view-projection matrix (uniform)
//   @group(1) @binding(0) instance_buffer — per-instance model matrices (storage)

//@oxy:include vertex
//@oxy:include shadow_uniform
//@oxy:include instance_data

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

//@oxy:group 0 0 storage_uniform shadow_uniform shadow_uniform
//@oxy:group 1 0 storage_read instance_buffer array<instance_data>

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
