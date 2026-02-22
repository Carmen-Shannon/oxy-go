// Compute shader with 3D workgroup size for parser coverage.

struct Uniforms {
    count: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> input: array<vec4<f32>>;

@compute @workgroup_size(8, 4, 2)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + id.y * 8u + id.z * 32u;
    if (idx < uniforms.count) {
        output[idx] = input[idx] * uniforms.scale;
    }
}
