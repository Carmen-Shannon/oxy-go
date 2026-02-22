// Shader with a runtime-sized array in a struct for parser coverage.

struct Header {
    count: u32,
    scale: f32,
};

struct Element {
    value: vec4<f32>,
};

struct DataBuffer {
    header: Header,
    elements: array<Element>,
};

@group(0) @binding(0) var<storage, read> data: DataBuffer;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < data.header.count) {
        output[id.x] = data.elements[id.x].value * data.header.scale;
    }
}
