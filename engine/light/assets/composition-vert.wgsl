// composition-vert.wgsl — Full-screen triangle vertex shader for the composition pass.
// Generates a single triangle that covers the entire screen without any vertex buffers.
// Vertex IDs 0, 1, 2 produce clip-space positions and UVs covering the [0,1]² range.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate a full-screen triangle: vertices at (-1,-1), (3,-1), (-1,3).
    // UVs at (0,1), (2,1), (0,-1) — flipped Y for screen-space convention.
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
