// taa-sharpen.wgsl — Contrast Adaptive Sharpening (CAS) post-TAA compute shader.
//
// Reads the TAA resolved texture and writes a sharpened RGBA16Float result to a
// separate output texture. Applied OUTSIDE the temporal feedback loop so that
// no negative-lobe sharpening is accumulated across frames.
//
// Algorithm: AMD FidelityFX CAS (Contrast Adaptive Sharpening).
// Reference: Kramer [Kra19] — https://gpuopen.com/fidelityfx-cas/  (survey §6.1.2)
//
// Per pixel:
//   1. Load center + 4 cardinal neighbors from the TAA resolved texture.
//   2. Compute the local luminance min/max across the 5 samples.
//   3. Derive a sharpening weight from the contrast ratio.
//   4. Apply: sharpened = center + weight * (4*center - sum(neighbors))
//
// The sharpness constant (0.0–1.0) controls intensity. In flat regions the weight
// converges to 0 (no sharpening); in high-contrast regions it reaches maximum.
// Default: 0.5.

//@oxy:provider 0 0 taa_sharpen taa_sharpen_input
@group(0) @binding(0) var taa_resolved_in: texture_2d<f32>;

//@oxy:provider 0 1 taa_sharpen taa_sharpen_output
@group(0) @binding(1) var taa_sharpened_out: texture_storage_2d<rgba16float, write>;

@group(0) @binding(2) var linear_sampler: sampler;

// luminance returns the BT.709 perceptual luma for a linear RGB color.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(taa_sharpened_out);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let maxc  = vec2<i32>(size) - vec2<i32>(1);

    // Load center and 4 cardinal neighbors. Clamp to texture bounds.
    let c = textureLoad(taa_resolved_in, coord,                              0).rgb;
    let n = textureLoad(taa_resolved_in, clamp(coord + vec2<i32>( 0, -1), vec2<i32>(0), maxc), 0).rgb;
    let s = textureLoad(taa_resolved_in, clamp(coord + vec2<i32>( 0,  1), vec2<i32>(0), maxc), 0).rgb;
    let e = textureLoad(taa_resolved_in, clamp(coord + vec2<i32>( 1,  0), vec2<i32>(0), maxc), 0).rgb;
    let w = textureLoad(taa_resolved_in, clamp(coord + vec2<i32>(-1,  0), vec2<i32>(0), maxc), 0).rgb;

    // Compute luminance min/max over the 5-tap neighborhood.
    let lc   = luminance(c);
    let ln   = luminance(n);
    let ls   = luminance(s);
    let le   = luminance(e);
    let lw   = luminance(w);
    let lmin = min(lc, min(min(ln, ls), min(le, lw)));
    let lmax = max(lc, max(max(ln, ls), max(le, lw)));

    // Derive sharpening weight from contrast ratio.
    // In flat regions (lmax ≈ lmin): ratio → 0 → weight → 0 (no sharpening).
    // In high-contrast regions: ratio → 1 → weight approaches maximum.
    var weight = 0.0;
    let lrange = lmax - lmin;
    if (lrange > 0.0) {
        let sharpness = 0.5;
        let ratio = min(lmin, 1.0 - lmax) / lrange;
        weight = saturate(ratio * (-sharpness * 2.0 + 8.0 * sharpness));
        weight = weight * (-2.0 * weight + 3.0); // smooth cubic ramp
    }

    // Apply: sharpened = center + weight * (4*center - sum(neighbors))
    let sharpened = c + weight * (4.0 * c - (n + s + e + w));

    textureStore(taa_sharpened_out, coord, vec4<f32>(sharpened, 1.0));
}
