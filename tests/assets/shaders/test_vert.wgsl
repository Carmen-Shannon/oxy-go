// Minimal vertex shader for renderer integration tests.

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct InstanceData {
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<storage, read> instance_buffer: array<InstanceData>;

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let instance = instance_buffer[instance_idx];
    var out: VertexOutput;
    out.clip_position = camera.view_proj * instance.model * vec4<f32>(vertex.position, 1.0);
    return out;
}
