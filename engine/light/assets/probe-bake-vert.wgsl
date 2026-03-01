// Probe bake instanced vertex shader (static models)
//
// Transforms each vertex from model space to clip space using a per-instance
// model matrix (from the compute shader's compacted output) and the probe bake
// camera's view-projection matrix. Outputs world-space position, normal, UV,
// and tangent for the probe bake fragment shader.
//
// The bind group layout mirrors the lit vertex shader but substitutes the
// standard CameraUniform with ProbeBakeCamera at @group(0):
//   @group(0) bake_camera — ProbeBakeCamera (view_proj + probe position)
//   @group(1) instances   — per-instance InstanceData (model matrices from compute)
//   @group(2) material    — (consumed by fragment shader only)

//@oxy:include vertex
//@oxy:include probe_bake_camera
//@oxy:include instance_data

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv:             vec2<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) world_tangent:  vec4<f32>,
};

//@oxy:group 0 0 storage_uniform bake_camera probe_bake_camera
//@oxy:group 1 0 storage_read instance_buffer array<instance_data>

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let model_matrix = instance_buffer[instance_idx].model;
    let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);

    let world_normal = (model_matrix * vec4<f32>(vertex.normal, 0.0)).xyz;
    let world_tangent_dir = (model_matrix * vec4<f32>(vertex.tangent.xyz, 0.0)).xyz;

    var out: VertexOutput;
    out.clip_position = bake_camera.view_proj * world_pos;
    out.uv = vertex.uv;
    out.world_normal = normalize(world_normal);
    out.color = vertex.color;
    out.world_position = world_pos.xyz;
    out.world_tangent = vec4<f32>(normalize(world_tangent_dir), vertex.tangent.w);
    return out;
}
