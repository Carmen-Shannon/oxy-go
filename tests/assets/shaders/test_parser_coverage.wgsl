// Exercises: fixed-size arrays, reversed struct deps, runtime array fallback, unresolvable structs.

struct Outer {
    inner: Inner,
    extra: f32,
};

struct Inner {
    x: f32,
    y: f32,
};

struct WithFixedArray {
    data: array<f32, 4>,
};

struct WithUnknownRuntime {
    count: u32,
    items: array<NeverDefinedType>,
};

struct Orphan {
    data: CompletelyFake,
};

struct OnlyUnknownArray {
    items: array<AlsoNeverDefined>,
};

@group(0) @binding(0) var<uniform> outer_data: Outer;
@group(0) @binding(1) var<uniform> fixed_data: WithFixedArray;
@group(0) @binding(2) var<storage, read> unknown_data: WithUnknownRuntime;
@group(0) @binding(3) var<storage, read> only_unknown: OnlyUnknownArray;

@compute @workgroup_size(16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // no-op
}
