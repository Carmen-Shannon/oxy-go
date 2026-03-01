// SH projection compute shader — cubemap-face-to-L2-spherical-harmonics.
//
// After the scene is rendered into the probe bake color texture for a single
// cubemap face, this compute shader is dispatched to project the captured
// radiance into L2 spherical harmonics and accumulate the coefficients into
// the probe storage buffer. One dispatch is performed per face (6 total per
// probe). Before the first face dispatch the caller must zero the target
// probe's SH fields in the buffer.
//
// Each dispatch uses a single workgroup of 256 threads. Every thread
// processes ceil(resolution² / 256) texels, accumulates partial SH sums
// locally, then participates in a tree reduction via workgroup shared memory.
// Thread 0 writes the reduced sums back to the probe buffer with +=.
//
// Dispatch: 1 × 1 × 1

//@oxy:include irradiance_probe
//@oxy:include sh_project_params

@group(0) @binding(0) var bake_texture: texture_2d<f32>;
//@oxy:group 0 1 storage_read_write probes array<irradiance_probe>
//@oxy:group 0 2 storage_uniform sh_params sh_project_params

const WG_SIZE: u32 = 256u;
const SH_COUNT: u32 = 9u;

var<workgroup> reduce_buf: array<f32, 256>;

// cube_dir converts a cubemap face index and texel UV (in [-1, 1]) to a
// world-space direction vector (unnormalized).
fn cube_dir(face: u32, u: f32, v: f32) -> vec3<f32> {
    switch face {
        case 0u { return vec3<f32>( 1.0, -v, -u); }  // +X
        case 1u { return vec3<f32>(-1.0, -v,  u); }  // -X
        case 2u { return vec3<f32>( u,  1.0,  v); }   // +Y
        case 3u { return vec3<f32>( u, -1.0, -v); }   // -Y
        case 4u { return vec3<f32>( u, -v,  1.0); }   // +Z
        default { return vec3<f32>(-u, -v, -1.0); }   // -Z
    }
}

// eval_sh evaluates the 9 real L2 spherical harmonic basis functions for a
// normalized direction vector.
fn eval_sh(d: vec3<f32>) -> array<f32, 9> {
    var sh: array<f32, 9>;
    sh[0] = 0.282095;                                  // Y_0^0
    sh[1] = 0.488603 * d.y;                            // Y_1^-1
    sh[2] = 0.488603 * d.z;                            // Y_1^0
    sh[3] = 0.488603 * d.x;                            // Y_1^1
    sh[4] = 1.092548 * d.x * d.y;                      // Y_2^-2
    sh[5] = 1.092548 * d.y * d.z;                      // Y_2^-1
    sh[6] = 0.315392 * (3.0 * d.z * d.z - 1.0);       // Y_2^0
    sh[7] = 1.092548 * d.x * d.z;                      // Y_2^1
    sh[8] = 0.546274 * (d.x * d.x - d.y * d.y);       // Y_2^2
    return sh;
}

// workgroup_reduce performs an in-place parallel tree reduction on reduce_buf.
// After the call, reduce_buf[0] holds the sum of all 256 entries. The caller
// must have written its value into reduce_buf[lid] and called
// workgroupBarrier() before invoking this function.
fn workgroup_reduce(lid: u32) {
    for (var stride = WG_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            reduce_buf[lid] += reduce_buf[lid + stride];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn cs_main(@builtin(local_invocation_index) lid: u32) {
    let res   = sh_params.resolution;
    let total = res * res;
    let face  = sh_params.face_index;
    let pidx  = sh_params.probe_index;

    // Thread-local accumulators for 9 SH coefficients × 3 RGB channels.
    var acc_r: array<f32, 9>;
    var acc_g: array<f32, 9>;
    var acc_b: array<f32, 9>;
    for (var k = 0u; k < SH_COUNT; k++) {
        acc_r[k] = 0.0;
        acc_g[k] = 0.0;
        acc_b[k] = 0.0;
    }

    // Each thread processes ceil(total / WG_SIZE) texels.
    for (var t = lid; t < total; t += WG_SIZE) {
        let px = t % res;
        let py = t / res;

        // UV in [-1, 1] for the texel center.
        let u = 2.0 * (f32(px) + 0.5) / f32(res) - 1.0;
        let v = 2.0 * (f32(py) + 0.5) / f32(res) - 1.0;

        // Cubemap direction (normalized) and solid-angle weight.
        let dir = normalize(cube_dir(face, u, v));
        let weight = 4.0 / (f32(total) * pow(1.0 + u * u + v * v, 1.5));

        // Sample radiance from the bake texture.
        let color = textureLoad(bake_texture, vec2<u32>(px, py), 0);

        // Evaluate SH basis and accumulate weighted radiance.
        let sh = eval_sh(dir);
        for (var c = 0u; c < SH_COUNT; c++) {
            let w = sh[c] * weight;
            acc_r[c] += color.r * w;
            acc_g[c] += color.g * w;
            acc_b[c] += color.b * w;
        }
    }

    // Parallel reduction: one coefficient at a time (27 reductions total).
    // Shared memory usage: 256 × 4 = 1024 bytes.
    for (var c = 0u; c < SH_COUNT; c++) {
        // Red channel
        reduce_buf[lid] = acc_r[c];
        workgroupBarrier();
        workgroup_reduce(lid);
        if lid == 0u {
            probes[pidx].sh_r[c] += reduce_buf[0];
        }
        workgroupBarrier();

        // Green channel
        reduce_buf[lid] = acc_g[c];
        workgroupBarrier();
        workgroup_reduce(lid);
        if lid == 0u {
            probes[pidx].sh_g[c] += reduce_buf[0];
        }
        workgroupBarrier();

        // Blue channel
        reduce_buf[lid] = acc_b[c];
        workgroupBarrier();
        workgroup_reduce(lid);
        if lid == 0u {
            probes[pidx].sh_b[c] += reduce_buf[0];
        }
        workgroupBarrier();
    }
}
