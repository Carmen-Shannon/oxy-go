// Minimal compute shader for renderer integration tests.

struct Params {
    count: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.count) {
        data[id.x] = data[id.x] + 1.0;
    }
}
