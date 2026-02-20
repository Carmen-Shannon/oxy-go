// Forward+ light culling compute shader
//
// Divides the screen into a grid of 16×16 pixel tiles and assigns lights to
// each tile using workgroup-cooperative culling. Each workgroup handles one tile
// with 256 threads (16×16). Threads cooperatively test lights against the tile's
// view-space frustum planes, accumulating visible light indices into shared
// memory. The final per-tile light list and count are written to global storage
// buffers that the lit fragment shader reads during the render pass.
//
// Bind group layout (@group(0)):
//   @binding(0) uniform  — LightCullUniforms (inv_proj, view_matrix, tile/screen dims, etc.)
//   @binding(1) storage  — Light array (same buffer as the fragment shader's lights)
//   @binding(2) storage  — tile_light_counts: per-tile visible light count
//   @binding(3) storage  — tile_light_indices: per-tile light index list

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 256u;
const NUM_THREADS: u32 = 256u; // TILE_SIZE * TILE_SIZE

// ── Light struct ───────────────────────────────────────────────────
// Must match Go's light.GPULight and the fragment shader's Light struct.
//@oxy:include light
// struct Light {
//     position:       vec3<f32>,
//     light_type:     u32,
//     color:          vec3<f32>,
//     intensity:      f32,
//     direction:      vec3<f32>,
//     light_range:    f32,
//     inner_cone:     f32,
//     outer_cone:     f32,
//     casts_shadows:  u32,
//     _pad:           u32,
// };

// ── Per-frame uniforms ─────────────────────────────────────────────
// Must match Go's light.GPULightCullUniforms (160 bytes).
//@oxy:include light_cull_uniforms
// struct LightCullUniforms {
//     inv_proj:       mat4x4<f32>,
//     view_matrix:    mat4x4<f32>,
//     tile_count_x:   u32,
//     tile_count_y:   u32,
//     screen_width:   u32,
//     screen_height:  u32,
//     light_count:    u32,
//     near:           f32,
//     far:            f32,
//     _pad:           u32,
// };

// Light type constants matching the fragment shader.
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;

// ── Bind group 0 ───────────────────────────────────────────────────
//@oxy:group 0 0 storage_uniform cull_uniforms light_cull_uniforms
// @group(0) @binding(0) var<uniform> cull_uniforms: LightCullUniforms;
//@oxy:group 0 1 storage_read cull_lights array<light>
// @group(0) @binding(1) var<storage, read> cull_lights: array<Light>;
@group(0) @binding(2) var<storage, read_write> tile_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_indices: array<u32>;

// ── Workgroup shared memory ────────────────────────────────────────
var<workgroup> shared_count: atomic<u32>;
var<workgroup> shared_list: array<u32, 256>; // MAX_LIGHTS_PER_TILE

// ── Tile frustum (4 planes + near/far) ─────────────────────────────
struct TileFrustum {
    planes: array<vec4<f32>, 4>, // normal.xyz, d=0 (planes through origin)
};

// unproject_ndc converts an NDC xy coordinate at the near plane (z=0)
// to view space using the inverse projection matrix.
fn unproject_ndc(ndc_xy: vec2<f32>) -> vec3<f32> {
    let clip = vec4<f32>(ndc_xy, 0.0, 1.0);
    let view = cull_uniforms.inv_proj * clip;
    return view.xyz / view.w;
}

// build_tile_frustum constructs 4 side frustum planes for a tile in view space.
// Planes pass through the origin (camera position in view space) with
// inward-pointing normals.
fn build_tile_frustum(tile_x: u32, tile_y: u32) -> TileFrustum {
    let sw = f32(cull_uniforms.screen_width);
    let sh = f32(cull_uniforms.screen_height);

    // Tile corners in screen pixels.
    let min_px = f32(tile_x * TILE_SIZE);
    let max_px = f32(min(tile_x * TILE_SIZE + TILE_SIZE, cull_uniforms.screen_width));
    let min_py = f32(tile_y * TILE_SIZE);
    let max_py = f32(min(tile_y * TILE_SIZE + TILE_SIZE, cull_uniforms.screen_height));

    // Convert to NDC [-1, 1]. Screen Y is top-down, NDC Y is bottom-up.
    let min_x_ndc = min_px / sw * 2.0 - 1.0;
    let max_x_ndc = max_px / sw * 2.0 - 1.0;
    let min_y_ndc = 1.0 - max_py / sh * 2.0; // bottom of tile in NDC
    let max_y_ndc = 1.0 - min_py / sh * 2.0; // top of tile in NDC

    // Unproject tile corners to view space at the near plane.
    let tl = unproject_ndc(vec2<f32>(min_x_ndc, max_y_ndc));
    let tr = unproject_ndc(vec2<f32>(max_x_ndc, max_y_ndc));
    let bl = unproject_ndc(vec2<f32>(min_x_ndc, min_y_ndc));
    let br = unproject_ndc(vec2<f32>(max_x_ndc, min_y_ndc));

    // Build inward-pointing plane normals from pairs of adjacent corner rays.
    // All planes pass through the origin so d=0.
    var frustum: TileFrustum;
    frustum.planes[0] = vec4<f32>(normalize(cross(bl, tl)), 0.0); // left
    frustum.planes[1] = vec4<f32>(normalize(cross(tr, br)), 0.0); // right
    frustum.planes[2] = vec4<f32>(normalize(cross(tl, tr)), 0.0); // top
    frustum.planes[3] = vec4<f32>(normalize(cross(br, bl)), 0.0); // bottom
    return frustum;
}

// sphere_in_frustum tests whether a bounding sphere overlaps the tile frustum.
// The test checks 4 side planes (through origin) and the camera's near/far
// range along the view-space Z axis.
fn sphere_in_frustum(frustum: TileFrustum, center: vec3<f32>, radius: f32) -> bool {
    // Side plane tests — planes pass through origin so d=0.
    // Unrolled because naga forbids dynamic indexing into fixed-size arrays in structs.
    if dot(frustum.planes[0].xyz, center) < -radius { return false; }
    if dot(frustum.planes[1].xyz, center) < -radius { return false; }
    if dot(frustum.planes[2].xyz, center) < -radius { return false; }
    if dot(frustum.planes[3].xyz, center) < -radius { return false; }

    // Near/far test in view space (right-handed: -Z is forward).
    // Sphere fully behind camera (positive Z side).
    if center.z - radius > 0.0 {
        return false;
    }
    // Sphere fully beyond far plane.
    if center.z + radius < -cull_uniforms.far {
        return false;
    }
    return true;
}

// ── Entry point ────────────────────────────────────────────────────
// Dispatch: (tile_count_x, tile_count_y, 1)
// Each workgroup = one tile, 256 threads cooperatively cull all lights.
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let tile_x = workgroup_id.x;
    let tile_y = workgroup_id.y;
    let tile_idx = tile_y * cull_uniforms.tile_count_x + tile_x;

    // Reset shared counter.
    if local_idx == 0u {
        atomicStore(&shared_count, 0u);
    }
    workgroupBarrier();

    // Build the tile's 4-plane frustum in view space.
    let frustum = build_tile_frustum(tile_x, tile_y);

    // Cooperative culling: each thread tests a strided subset of lights.
    let total_lights = cull_uniforms.light_count;
    for (var i = local_idx; i < total_lights; i += NUM_THREADS) {
        let light = cull_lights[i];

        var visible = false;
        if light.light_type == LIGHT_TYPE_DIRECTIONAL {
            // Directional lights have no position — they affect every tile.
            visible = true;
        } else {
            // Transform light world position to view space for frustum testing.
            let world_pos = vec4<f32>(light.position, 1.0);
            let view_pos = (cull_uniforms.view_matrix * world_pos).xyz;
            visible = sphere_in_frustum(frustum, view_pos, light.light_range);
        }

        if visible {
            let slot = atomicAdd(&shared_count, 1u);
            if slot < MAX_LIGHTS_PER_TILE {
                shared_list[slot] = i;
            }
        }
    }
    workgroupBarrier();

    // Write shared results to global memory.
    let count = min(atomicLoad(&shared_count), MAX_LIGHTS_PER_TILE);
    let base = tile_idx * MAX_LIGHTS_PER_TILE;
    for (var i = local_idx; i < count; i += NUM_THREADS) {
        tile_indices[base + i] = shared_list[i];
    }

    // Thread 0 writes the per-tile light count.
    if local_idx == 0u {
        tile_counts[tile_idx] = count;
    }
}
