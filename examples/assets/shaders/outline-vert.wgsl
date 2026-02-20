// Outline vertex shader (inverted hull technique)
//
// Identical to skinned-vert.wgsl except: after computing the final skinned
// world-space position, each vertex is pushed outward along its skinned
// world-space normal by a configurable thickness. This inflated mesh is
// rendered with front-face culling so only the back faces are visible,
// creating a solid outline / silhouette around the model.

const MAX_BONES: u32 = 64u;
const FLOATS_PER_INSTANCE: u32 = (1u + MAX_BONES) * 4u;

// Outline thickness in world-space units. Adjust to taste.
const OUTLINE_THICKNESS: f32 = 1.5;

// ── Vertex attributes ──────────────────────────────────────────────
// Must match Go's model.GPUSkinnedVertex struct layout exactly (96 bytes).
//@oxy:include skinned_vertex
// struct VertexInput {
//     @location(0) position: vec3<f32>,
//     @location(1) normal:   vec3<f32>,
//     @location(2) uv:       vec2<f32>,
//     @location(3) color:    vec4<f32>,
//     @location(4) tangent:  vec4<f32>,
//     @location(5) bone_indices: vec4<u32>,
//     @location(6) bone_weights: vec4<f32>,
// };

// ── Interpolated output → fragment shader ──────────────────────────
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv:    vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color:  vec4<f32>,
    @location(3) world_position: vec3<f32>,
};

// ── Camera uniform ─────────────────────────────────────────────────
//@oxy:include camera
// struct CameraUniform {
//     view_proj: mat4x4<f32>,
//     camera_position: vec3<f32>,
//     _pad: f32,
// };

// ── Bind groups ────────────────────────────────────────────────────
//@oxy:group 0 0 storage_uniform camera camera
// @group(0) @binding(0) var<uniform> camera: CameraUniform;
//@oxy:provider 1 0 animator
@group(1) @binding(0) var<storage, read> instance_buffer: array<vec4<f32>>;

// ── Helpers ────────────────────────────────────────────────────────
fn read_mat4(base: u32) -> mat4x4<f32> {
    return mat4x4<f32>(
        instance_buffer[base],
        instance_buffer[base + 1u],
        instance_buffer[base + 2u],
        instance_buffer[base + 3u],
    );
}

// ── Entry point ────────────────────────────────────────────────────
@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let base = instance_idx * FLOATS_PER_INSTANCE;
    let model_matrix = read_mat4(base);
    let bone_base = base + 4u;

    let indices = vertex.bone_indices;
    let weights = vertex.bone_weights;

    var skin_matrix = mat4x4<f32>(
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)
    );

    skin_matrix += weights.x * read_mat4(bone_base + indices.x * 4u);
    skin_matrix += weights.y * read_mat4(bone_base + indices.y * 4u);
    skin_matrix += weights.z * read_mat4(bone_base + indices.z * 4u);
    skin_matrix += weights.w * read_mat4(bone_base + indices.w * 4u);

    let skinned_pos = skin_matrix * vec4<f32>(vertex.position, 1.0);
    let world_pos = model_matrix * skinned_pos;

    let skin_normal = (skin_matrix * vec4<f32>(vertex.normal, 0.0)).xyz;
    let raw_normal = (model_matrix * vec4<f32>(skin_normal, 0.0)).xyz;

    // If normals are present use them; otherwise fall back to inflating
    // outward from the model origin using the vertex's world position.
    // Fox.glb (and some other models) have zero-length normals.
    let normal_len = length(raw_normal);
    var inflate_dir: vec3<f32>;
    if normal_len > 0.001 {
        inflate_dir = raw_normal / normal_len;
    } else {
        inflate_dir = normalize(world_pos.xyz);
    }

    // Push vertex outward to inflate the mesh.
    let inflated_pos = world_pos.xyz + inflate_dir * OUTLINE_THICKNESS;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(inflated_pos, 1.0);
    out.uv = vertex.uv;
    out.normal = inflate_dir;
    out.color = vertex.color;
    out.world_position = vertex.position;
    return out;
}
