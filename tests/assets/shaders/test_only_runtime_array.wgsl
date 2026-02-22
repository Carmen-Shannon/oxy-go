// Shader with only a runtime-sized array (no fixed prefix fields) for parser coverage.

struct Item {
    pos: vec3<f32>,
    weight: f32,
};

struct OnlyArray {
    items: array<Item>,
};

@group(0) @binding(0) var<storage, read> data: OnlyArray;

@compute @workgroup_size(32)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let item = data.items[id.x];
    // use item
}
