// Minimal VSM shadow depth vertex shader for renderer integration tests.

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) light_depth: f32,
};

struct ShadowUniform {
    light_vp: mat4x4<f32>,
};

struct InstanceData {
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> shadow_uniform: ShadowUniform;
@group(1) @binding(0) var<storage, read> instance_buffer: array<InstanceData>;

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let model_matrix = instance_buffer[instance_idx].model;
    let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);
    let clip = shadow_uniform.light_vp * world_pos;
    var out: VertexOutput;
    out.clip_position = clip;
    out.light_depth = clip.z / clip.w;
    return out;
}
