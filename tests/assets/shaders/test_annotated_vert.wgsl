//@oxy:include camera
//@oxy:include instance_data
//@oxy:include model_data
//@oxy:include vertex

//@oxy:group 0 0 storage_uniform camera camera
//@oxy:group 1 0 storage_read instance_buffer array<instance_data>
//@oxy:group 1 1 storage_read model_buffer array<model_data>
//@oxy:provider 2 0 material diffuse_texture
@group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
//@oxy:provider 2 1 material diffuse_sampler
@group(2) @binding(1) var diffuseSampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let instance = instance_buffer[instance_idx];
    let model = model_buffer[instance_idx];
    var out: VertexOutput;
    out.clip_position = camera.view_proj * instance.model * vec4<f32>(vertex.position, 1.0);
    out.uv = vertex.uv;
    return out;
}
