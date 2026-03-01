// VSM shadow depth vertex shader (static models)
//
// Transforms vertices to light clip space and computes a linear light-space
// depth for the VSM fragment shader. The linear depth is computed from the
// view-only matrix (no projection) and normalized to [0, 1] using near/far.
//
// Bind group layout:
//   @group(0) @binding(0) shadow_uniform — light VP, view matrix, near/far (uniform)
//   @group(1) @binding(0) instance_buffer — per-instance model matrices (storage)

//@oxy:include vertex
//@oxy:include shadow_uniform
//@oxy:include instance_data

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) light_depth: f32,
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

    // Compute linear depth from the view-only matrix, normalized to [0, 1].
    let view_pos = shadow_uniform.light_view * world_pos;
    let linear_depth = (-view_pos.z - shadow_uniform.shadow_near)
                     / (shadow_uniform.shadow_far - shadow_uniform.shadow_near);

    var out: VertexOutput;
    out.clip_position = shadow_uniform.light_vp * world_pos;
    out.light_depth = linear_depth;
    return out;
}
