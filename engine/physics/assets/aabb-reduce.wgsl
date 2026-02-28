// AABB reduction — Stage 1.5a of the GPU rigid body pipeline
//
// Runs one invocation per particle. Each invocation reads the particle's
// world position and atomically updates a global bounding box (6 atomic u32
// representing min.xyz and max.xyz). Uses the float-to-sortable-uint trick
// so that atomicMin/atomicMax on u32 yield correct floating-point ordering.
//
// After this dispatch completes, the 6 atomics hold the tight AABB of all
// active particles (in sortable-uint encoding). The grid_build_params shader
// (Stage 1.5b) converts these back to floats and derives grid parameters.
//
// Reference: GPU Gems 3, Chapter 29 §29.2.4 — Grid Generation
//
// Binding layout (3 bindings):
//   @binding(0) storage, read       — Particle array     (read world_position)
//   @binding(1) storage, read_write — AABB atomics        (6 × atomic<u32>: min.xyz, max.xyz)
//   @binding(2) uniform             — PhysicsGlobals      (particle_count)

//@oxy:include physics_particle
//@oxy:include physics_globals

//@oxy:group 0 0 storage_read particles array<physics_particle>
// AABB atomics buffer declared manually — requires atomic<u32> access
@group(0) @binding(1) var<storage, read_write> aabb: array<atomic<u32>>;
//@oxy:group 0 2 storage_uniform globals physics_globals

// float_to_sortable converts an IEEE-754 f32 to a u32 that preserves ordering
// under unsigned integer comparison. Positive floats are XORed with 0x80000000
// to flip the sign bit; negative floats are XORed with 0xFFFFFFFF to invert all
// bits (making more-negative values smaller in unsigned order).
fn float_to_sortable(f: f32) -> u32 {
    let bits = bitcast<u32>(f);
    return bits ^ select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= globals.particle_count) {
        return;
    }

    // Skip dead particles (sentinel: world_position.w == -1.0). Including
    // them would poison the AABB with extreme positions, making the spatial
    // grid too coarse for any meaningful collision detection.
    if (particles[idx].world_position.w < 0.0) {
        return;
    }

    let pos = particles[idx].world_position.xyz;

    let sx = float_to_sortable(pos.x);
    let sy = float_to_sortable(pos.y);
    let sz = float_to_sortable(pos.z);

    // Indices 0–2: AABB min (atomicMin finds the smallest sortable value)
    atomicMin(&aabb[0], sx);
    atomicMin(&aabb[1], sy);
    atomicMin(&aabb[2], sz);

    // Indices 3–5: AABB max (atomicMax finds the largest sortable value)
    atomicMax(&aabb[3], sx);
    atomicMax(&aabb[4], sy);
    atomicMax(&aabb[5], sz);
}
