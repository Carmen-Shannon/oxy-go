// Grid insert — Stage 2b of the GPU rigid body pipeline
//
// Runs one invocation per particle. For each particle, computes the grid cell
// from its world position and atomically inserts the particle's index into the
// first available slot. Replaces the paper's stencil-routing technique
// (Listing 29-1) with atomicCompareExchangeWeak on 4 cell slots.
//
// Reference: GPU Gems 3, Chapter 29 §29.2.4 — Grid Generation
//   Eq. 9: g = floor((x_i - s) / d)  — particle-to-cell mapping
//
// Binding layout (4 bindings):
//   @binding(0) storage, read_write — Grid as flat array<atomic<u32>> (4 slots per cell)
//   @binding(1) storage, read       — Particle array  (read world_position)
//   @binding(2) storage, read       — GridParams      (grid origin, dims)
//   @binding(3) uniform             — PhysicsGlobals  (particle_diameter, particle_count)

//@oxy:include physics_particle
//@oxy:include physics_grid_params
//@oxy:include physics_globals

// Grid binding is declared manually because cell slots require atomic<u32>
// for concurrent insertion. The underlying buffer is the same as grid_clear's
// GridCell array — vec4<u32> maps to 4 consecutive u32 per cell.
@group(0) @binding(0) var<storage, read_write> grid: array<atomic<u32>>;
//@oxy:group 0 1 storage_read particles array<physics_particle>
//@oxy:group 0 2 storage_read grid_params physics_grid_params
//@oxy:group 0 3 storage_uniform globals physics_globals

const EMPTY: u32 = 0xFFFFFFFFu;
const SLOTS_PER_CELL: u32 = 16u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= globals.particle_count) {
        return;
    }

    // Skip dead particles — they must not occupy grid slots or they'd
    // block live particles from being inserted (only 4 slots per cell).
    if (particles[idx].world_position.w < 0.0) {
        return;
    }

    // Eq. 9: compute grid coordinates from world position.
    // Uses the effective cell_size stored by grid_build_params (grid_origin.w)
    // rather than particle_diameter. When the grid is proportionally scaled to
    // fit max_grid_cells, cell_size increases so indices stay within dims.
    let pos = particles[idx].world_position.xyz;
    let cell_size = grid_params.grid_origin.w;
    let g = vec3<i32>(floor((pos - grid_params.grid_origin.xyz) / cell_size));

    // Clamp to valid grid range
    let dims = vec3<i32>(grid_params.grid_dims.xyz);
    let gc = clamp(g, vec3<i32>(0), dims - vec3<i32>(1));

    // 1D cell index: x + y * dimX + z * dimX * dimY
    let cell = u32(gc.x) + u32(gc.y) * grid_params.grid_dims.x + u32(gc.z) * grid_params.grid_dims.x * grid_params.grid_dims.y;

    // Atomically insert into the first available slot (§29.2.4 — 4 slots per cell)
    let base = cell * SLOTS_PER_CELL;
    for (var slot = 0u; slot < SLOTS_PER_CELL; slot++) {
        let result = atomicCompareExchangeWeak(&grid[base + slot], EMPTY, idx);
        if (result.exchanged) {
            return;
        }
    }
    // All 16 slots occupied — particle dropped (rare when cell_size ≈ particle_diameter)
}
