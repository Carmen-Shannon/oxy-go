// Simple instanced vertex shader
//
// Transforms each vertex from model space to clip space using a per-instance
// model matrix (from the compute shader's compacted output) and the camera's
// view-projection matrix. Passes per-vertex color through to the fragment stage.

// ── Vertex attributes ──────────────────────────────────────────────
// Must match Go's model.GPUVertex struct layout exactly (64 bytes).
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
    @location(3) color:    vec4<f32>,
    @location(4) tangent:  vec4<f32>,
};

// ── Interpolated output → fragment shader ──────────────────────────
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// ── Camera uniform ─────────────────────────────────────────────────
struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
};

// ── Per-instance model matrix (produced by compute shader) ─────────
struct InstanceData {
    model: mat4x4<f32>,
};
struct InstanceBuffer {
    instances: array<InstanceData>,
};

// ── Bind groups ────────────────────────────────────────────────────
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<storage, read> instance_buffer: InstanceBuffer;

// ── Entry point ────────────────────────────────────────────────────
@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let instance = instance_buffer.instances[instance_idx];

    var out: VertexOutput;
    out.clip_position = camera.view_proj * instance.model * vec4<f32>(vertex.position, 1.0);
    out.color = vertex.color;
    return out;
}
