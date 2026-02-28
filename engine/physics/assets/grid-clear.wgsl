// Grid clear — Stage 2a of the GPU rigid body pipeline
//
// Runs one invocation per grid cell. Fills every cell's 16 particle-index
// slots with the empty sentinel 0xFFFFFFFF. This must complete before
// the grid insert dispatch.
//
// Reference: GPU Gems 3, Chapter 29 §29.2.4 — Grid Generation
//
// Binding layout (3 bindings):
//   @binding(0) storage, read_write — GridCell array   (cleared to sentinel)
//   @binding(1) storage, read       — GridParams       (grid_dims.w = total cell count)
//   @binding(2) uniform             — PhysicsGlobals   (reserved for future use)

//@oxy:include physics_grid
//@oxy:include physics_grid_params
//@oxy:include physics_globals

//@oxy:group 0 0 storage_read_write grid array<physics_grid>
//@oxy:group 0 1 storage_read grid_params physics_grid_params
//@oxy:group 0 2 storage_uniform globals physics_globals

const EMPTY: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= grid_params.grid_dims.w) {
        return;
    }

    grid[idx].indices_0 = vec4<u32>(EMPTY, EMPTY, EMPTY, EMPTY);
    grid[idx].indices_1 = vec4<u32>(EMPTY, EMPTY, EMPTY, EMPTY);
    grid[idx].indices_2 = vec4<u32>(EMPTY, EMPTY, EMPTY, EMPTY);
    grid[idx].indices_3 = vec4<u32>(EMPTY, EMPTY, EMPTY, EMPTY);
}
