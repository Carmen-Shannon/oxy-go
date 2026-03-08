// Probe bake instanced vertex shader (skinned models)
//
// Transforms each skinned vertex from model space to clip space using
// per-instance bone skinning matrices and model matrix from the compute
// shader's compacted output, combined with the probe bake camera's
// view-projection matrix. Up to 4 bone influences per vertex are blended
// using bone_indices and bone_weights. Outputs world-space position, normal,
// UV, and tangent for the probe bake fragment shader.
//
// The bind group layout mirrors the lit skinned vertex shader but substitutes
// the standard CameraUniform with ProbeBakeCamera at @group(0):
//   @group(0) bake_camera — ProbeBakeCamera (view_proj + probe position)
//   @group(1) animator    — flat vec4 storage (model matrix + bone matrices per instance)
//   @group(2) material    — (consumed by fragment shader only)

const MAX_BONES: u32 = 64u;

//@oxy:include skinned_vertex
//@oxy:include probe_bake_camera

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv:             vec2<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) world_tangent:  vec4<f32>,
};

//@oxy:group 0 0 storage_uniform bake_camera probe_bake_camera
//@oxy:provider 1 0 animator
@group(1) @binding(0) var<storage, read> instance_buffer: array<vec4<f32>>;

fn read_mat4(base: u32) -> mat4x4<f32> {
    return mat4x4<f32>(
        instance_buffer[base],
        instance_buffer[base + 1u],
        instance_buffer[base + 2u],
        instance_buffer[base + 3u],
    );
}

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let floatsPerInstance = (1u + MAX_BONES) * 4u;
    let base = instance_idx * floatsPerInstance;

    let model_matrix = read_mat4(base);

    let bone_base = base + 4u;

    let indices = vertex.bone_indices;
    let weights = vertex.bone_weights;

    // Compute vec4-offset for each bone: bone index * 4 vec4s per matrix.
    let b0 = bone_base + indices.x * 4u;
    let b1 = bone_base + indices.y * 4u;
    let b2 = bone_base + indices.z * 4u;
    let b3 = bone_base + indices.w * 4u;

    // Accumulate skin_matrix column-by-column from raw vec4 buffer reads.
    let c0 = weights.x * instance_buffer[b0]      + weights.y * instance_buffer[b1]      + weights.z * instance_buffer[b2]      + weights.w * instance_buffer[b3];
    let c1 = weights.x * instance_buffer[b0 + 1u] + weights.y * instance_buffer[b1 + 1u] + weights.z * instance_buffer[b2 + 1u] + weights.w * instance_buffer[b3 + 1u];
    let c2 = weights.x * instance_buffer[b0 + 2u] + weights.y * instance_buffer[b1 + 2u] + weights.z * instance_buffer[b2 + 2u] + weights.w * instance_buffer[b3 + 2u];
    let c3 = weights.x * instance_buffer[b0 + 3u] + weights.y * instance_buffer[b1 + 3u] + weights.z * instance_buffer[b2 + 3u] + weights.w * instance_buffer[b3 + 3u];
    var skin_matrix = mat4x4<f32>(c0, c1, c2, c3);

    let skinned_pos = skin_matrix * vec4<f32>(vertex.position, 1.0);
    let world_pos = model_matrix * skinned_pos;

    let skin_normal = (skin_matrix * vec4<f32>(vertex.normal, 0.0)).xyz;
    let world_normal = (model_matrix * vec4<f32>(skin_normal, 0.0)).xyz;

    let skin_tangent = (skin_matrix * vec4<f32>(vertex.tangent.xyz, 0.0)).xyz;
    let world_tangent_dir = (model_matrix * vec4<f32>(skin_tangent, 0.0)).xyz;

    var out: VertexOutput;
    out.clip_position = bake_camera.view_proj * world_pos;
    out.uv = vertex.uv;
    out.world_normal = normalize(world_normal);
    out.color = vertex.color;
    out.world_position = world_pos.xyz;
    out.world_tangent = vec4<f32>(normalize(world_tangent_dir), vertex.tangent.w);
    return out;
}
