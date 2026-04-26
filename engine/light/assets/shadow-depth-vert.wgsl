// Shadow depth vertex shader (static models)
//
// Transforms vertices to light clip space for depth-only shadow rendering.
// No fragment shader is needed — the hardware depth buffer stores the depth
// automatically via the rasterizer.
//
// Bind group layout:
//   @group(0) @binding(0) shadow_uniform — light VP matrix (uniform)
//   @group(1) @binding(0) animation_buffer — per-instance raw animation data (storage)

//@oxy:include vertex
//@oxy:include shadow_uniform
//@oxy:include animation_data

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

//@oxy:group 0 0 storage_uniform shadow_uniform shadow_uniform
//@oxy:group 1 0 storage_read animation_buffer array<animation_data>

fn build_model_matrix(pos: vec3<f32>, rot: vec3<f32>, scale: vec3<f32>) -> mat4x4<f32> {
    let cx = cos(rot.x); let sx = sin(rot.x);
    let cy = cos(rot.y); let sy = sin(rot.y);
    let cz = cos(rot.z); let sz = sin(rot.z);

    let col0 = vec4<f32>(scale.x * (cz * cy), scale.x * (sz * cy), scale.x * (-sy), 0.0);
    let col1 = vec4<f32>(scale.y * (cz * sy * sx - sz * cx), scale.y * (sz * sy * sx + cz * cx), scale.y * (cy * sx), 0.0);
    let col2 = vec4<f32>(scale.z * (cz * sy * cx + sz * sx), scale.z * (sz * sy * cx - cz * sx), scale.z * (cy * cx), 0.0);
    let col3 = vec4<f32>(pos, 1.0);
    return mat4x4<f32>(col0, col1, col2, col3);
}

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let anim = animation_buffer[instance_idx];
    let model_matrix = build_model_matrix(anim.pos, anim.rot, anim.scale);
    let world_pos = model_matrix * vec4<f32>(vertex.position, 1.0);

    var out: VertexOutput;
    out.clip_position = shadow_uniform.light_vp * world_pos;
    return out;
}
