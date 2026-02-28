// Grid build params — Stage 1.5b of the GPU rigid body pipeline
//
// Single-invocation dispatch (1,1,1). Reads the AABB atomic accumulators
// written by aabb_reduce (Stage 1.5a), converts sortable u32 values back to
// floats, adds a half-cell-size margin, and computes grid origin and dimensions.
// The result is written to the GPUGridParams storage buffer which is consumed
// by grid_clear (Stage 2a), grid_insert (Stage 2b), and collision_reaction (Stage 3).
//
// Reference: GPU Gems 3, Chapter 29 §29.2.4 — Grid Generation
//
// Binding layout (3 bindings):
//   @binding(0) storage, read       — AABB atomics        (6 × u32: min.xyz, max.xyz in sortable encoding)
//   @binding(1) storage, read_write — GridParams           (output: grid_origin, grid_dims)
//   @binding(2) uniform             — PhysicsGlobals       (particle_diameter as cell size, max_grid_cells)

//@oxy:include physics_grid_params
//@oxy:include physics_globals

// AABB atomics buffer declared manually — plain array<u32> (read-only after reduction)
@group(0) @binding(0) var<storage, read> aabb: array<u32>;
//@oxy:group 0 1 storage_read_write grid_params physics_grid_params
//@oxy:group 0 2 storage_uniform globals physics_globals

// sortable_to_float reverses the float_to_sortable encoding applied in aabb_reduce.
// If the sign bit is set (original was positive), XOR with 0x80000000 to restore.
// Otherwise (original was negative), XOR with 0xFFFFFFFF to invert all bits back.
fn sortable_to_float(s: u32) -> f32 {
    let bits = s ^ select(0xFFFFFFFFu, 0x80000000u, (s & 0x80000000u) != 0u);
    return bitcast<f32>(bits);
}

@compute @workgroup_size(1)
fn main() {
    var cell_size = globals.particle_diameter;
    let margin = cell_size * 0.5;

    // Decode AABB corners from sortable-uint encoding
    let aabb_min = vec3<f32>(
        sortable_to_float(aabb[0]),
        sortable_to_float(aabb[1]),
        sortable_to_float(aabb[2]),
    );
    let aabb_max = vec3<f32>(
        sortable_to_float(aabb[3]),
        sortable_to_float(aabb[4]),
        sortable_to_float(aabb[5]),
    );

    // Grid origin: AABB min with margin
    let origin = aabb_min - vec3<f32>(margin);

    // Padded max corner
    let padded_max = aabb_max + vec3<f32>(margin);

    // Grid dimensions: number of cells along each axis
    let extent = padded_max - origin;
    var dims = vec3<u32>(ceil(extent / vec3<f32>(cell_size)));

    // Enforce minimum of 1 cell per axis
    dims = max(dims, vec3<u32>(1u));

    // Total cell count
    var total = dims.x * dims.y * dims.z;

    // Clamp to configured maximum grid cell count.
    // When dims exceeds the budget, increase cell_size proportionally so that
    // the hash in grid_insert and collision_reaction produces indices that
    // naturally fit within the reduced dims. Without this, those shaders
    // would compute indices using the original (smaller) cell_size, exceeding
    // the reduced dims and clamping all out-of-range particles to edge cells.
    let max_cells = globals.max_grid_cells;
    if (total > max_cells && max_cells > 0u) {
        let scale = pow(f32(max_cells) / f32(total), 1.0 / 3.0);
        cell_size = cell_size / scale;
        dims = max(vec3<u32>(ceil(extent / vec3<f32>(cell_size))), vec3<u32>(1u));
        total = dims.x * dims.y * dims.z;
    }

    // Store effective cell_size in grid_origin.w so grid_insert and
    // collision_reaction use the same cell size that produced these dims.
    grid_params.grid_origin = vec4<f32>(origin, cell_size);
    grid_params.grid_dims = vec4<u32>(dims, total);
}
