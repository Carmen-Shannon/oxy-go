// Simple instanced vertex shader
//
// Transforms each vertex from model space to clip space using a per-instance
// model matrix (from the compute shader's compacted output) and the camera's
// view-projection matrix. Passes UV, world-space normal, per-vertex color,
// and world-space position through to the fragment stage.

//@oxy:include vertex
//@oxy:include camera
//@oxy:include instance_data

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv:             vec2<f32>,
    @location(1) normal:         vec3<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) world_position: vec3<f32>,
};

//@oxy:group 0 0 storage_uniform camera camera
//@oxy:group 1 0 storage_read instance_buffer array<instance_data>

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let instance = instance_buffer[instance_idx];
    let world_pos = instance.model * vec4<f32>(vertex.position, 1.0);
    let world_normal = (instance.model * vec4<f32>(vertex.normal, 0.0)).xyz;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.uv = vertex.uv;
    out.normal = normalize(world_normal);
    out.color = vertex.color;
    out.world_position = world_pos.xyz;
    return out;
}
