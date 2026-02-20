// Lit instanced vertex shader (static models)
//
// Transforms each vertex from model space to clip space using a per-instance
// model matrix (from the compute shader's compacted output) and the camera's
// view-projection matrix. Outputs world-space position and normal for
// per-fragment lighting evaluation in the lit fragment shader.

// ── Vertex attributes ──────────────────────────────────────────────
// Must match Go's model.GPUVertex struct layout exactly (64 bytes).
//@oxy:include vertex
// struct VertexInput {
//     @location(0) position: vec3<f32>,
//     @location(1) normal:   vec3<f32>,
//     @location(2) uv:       vec2<f32>,
//     @location(3) color:    vec4<f32>,
//     @location(4) tangent:  vec4<f32>,
// };

// ── Interpolated output → fragment shader ──────────────────────
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv:             vec2<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) world_tangent:  vec4<f32>,
};

// ── Camera uniform ─────────────────────────────────────────────────
//@oxy:include camera
// struct CameraUniform {
//     view_proj: mat4x4<f32>,
//     camera_position: vec3<f32>,
//     _pad: f32,
// };

// ── Per-instance model matrix (produced by compute shader) ─────────
//@oxy:include instance_data
// struct InstanceData {
//     model: mat4x4<f32>,
// };


// ── Bind groups ────────────────────────────────────────────────────
//@oxy:group 0 0 storage_uniform camera camera
// @group(0) @binding(0) var<uniform> camera: CameraUniform;
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

    // Transform normal to world space using the model matrix upper-left 3x3.
    // This is an approximation that works correctly for uniform and non-uniform
    // scale as long as the model matrix is orthogonal (no shear).
    let world_normal = (model_matrix * vec4<f32>(vertex.normal, 0.0)).xyz;

    // Transform tangent to world space, preserve handedness in W.
    let world_tangent_dir = (model_matrix * vec4<f32>(vertex.tangent.xyz, 0.0)).xyz;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.uv = vertex.uv;
    out.world_normal = normalize(world_normal);
    out.color = vertex.color;
    out.world_position = world_pos.xyz;
    out.world_tangent = vec4<f32>(normalize(world_tangent_dir), vertex.tangent.w);
    return out;
}
