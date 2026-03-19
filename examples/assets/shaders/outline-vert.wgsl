// Outline vertex shader (inverted hull technique — clip-space extrusion)
//
// Computes the skinned world-space position like the normal vertex shader,
// then projects it to clip space and pushes the clip-space xy outward along
// the screen-space normal direction. This produces a uniform-thickness
// outline that is free of the triangle-seam gaps caused by world-space
// normal inflation on meshes with split/hard-edge normals.
//
// Rendered with front-face culling so only the back faces of the inflated
// mesh are visible, creating a solid outline / silhouette around the model.

//@oxy:inject MAX_BONES u32 max_bones
const FLOATS_PER_INSTANCE: u32 = (1u + MAX_BONES) * 4u;

// Outline thickness in clip-space units (scaled by w for perspective).
// Increase for a thicker outline; decrease for a thinner one.
const OUTLINE_THICKNESS: f32 = 3.0;

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

    // Compute the skinned world-space normal for the extrusion direction.
    let skin_normal = (skin_matrix * vec4<f32>(vertex.normal, 0.0)).xyz;
    let world_normal = (model_matrix * vec4<f32>(skin_normal, 0.0)).xyz;
    let normal_len = length(world_normal);

    // Project the un-inflated vertex to clip space first.
    let clip_pos = camera.view_proj * world_pos;

    // Compute the screen-space extrusion direction by projecting the
    // world-space normal into clip space and normalizing only in xy.
    // When the normal is degenerate, fall back to pushing outward from
    // the model origin in screen space.
    var screen_offset = vec2<f32>(0.0, 0.0);
    if normal_len > 0.001 {
        let n = world_normal / normal_len;
        let clip_n = camera.view_proj * vec4<f32>(n, 0.0);
        let slen = length(clip_n.xy);
        if slen > 0.0001 {
            screen_offset = clip_n.xy / slen;
        }
    }
    // Fallback: extrude away from model center in screen space
    if length(screen_offset) < 0.0001 {
        let model_origin = model_matrix * vec4<f32>(0.0, 0.0, 0.0, 1.0);
        let clip_origin = camera.view_proj * model_origin;
        let dir = clip_pos.xy / clip_pos.w - clip_origin.xy / clip_origin.w;
        let dlen = length(dir);
        if dlen > 0.0001 {
            screen_offset = dir / dlen;
        } else {
            screen_offset = vec2<f32>(0.0, 1.0);
        }
    }

    // Push outward in clip space. Multiplying by clip_pos.w makes the
    // offset perspective-correct so the outline has a consistent
    // screen-pixel width regardless of distance from the camera.
    var out: VertexOutput;
    out.clip_position = clip_pos;
    let extrude = screen_offset * OUTLINE_THICKNESS * 0.002 * clip_pos.w;
    out.clip_position.x = out.clip_position.x + extrude.x;
    out.clip_position.y = out.clip_position.y + extrude.y;
    out.uv = vertex.uv;
    out.normal = select(normalize(world_pos.xyz), world_normal / normal_len, normal_len > 0.001);
    out.color = vertex.color;
    out.world_position = vertex.position;
    return out;
}
