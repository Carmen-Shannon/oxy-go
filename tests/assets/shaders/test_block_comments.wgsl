/* This is a block comment */
/*
    Nested /* block /* comments */ inside */ are handled
*/
// Line comment test

struct Params {
    count: u32,
};

@group(0) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // no-op
}
