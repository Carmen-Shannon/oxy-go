// Vertex shader with a struct that has an unrecognized type, causing buildVertexBufferLayout to skip.

struct VertexBad {
    @location(0) position: vec3<f32>,
    @location(1) custom_data: mat4x4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@group(0) @binding(0) var<uniform> dummy: vec4<f32>;

@vertex
fn vs_main(vertex: VertexBad) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(vertex.position, 1.0);
    return out;
}
