// Summed-Area Table (SAT) generator for Variance Shadow Maps.
//
// This shader implements the recursive-doubling prefix-sum algorithm from
// Hensley et al. 2005, used to build a SAT over the VSM moments texture.
// For an N×N texture, log₂(N) passes are dispatched horizontally and then
// log₂(N) passes vertically, for a total of 2·log₂(N) dispatches.
//
// The shader operates in two modes controlled by the `params.offset` uniform:
//
//   offset == 0 — Precision distribution (prepare) pass:
//     Reads the RG32Float moments texture (M₁, M₂) and distributes each moment
//     across two channels (integer-aligned part + fractional remainder) to double
//     the effective mantissa precision for SAT accumulation.
//     Reference: GPU Gems 3, §8.5.2, Listing 8-5.
//
//   offset > 0 — Recursive-doubling prefix-sum pass:
//     Reads the RGBA32Float SAT working texture and adds the value at
//     (x − offset·dir.x, y − offset·dir.y) to produce the next partial sum.
//
// Bind group layout (@group(0)):
//   @binding(0) texture_2d<f32>                      — input texture (read)
//   @binding(1) texture_storage_2d<rgba32float,write> — output texture (write)
//   @binding(2) uniform                              — SATParams (direction, offset)

//@oxy:include sat_params

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
//@oxy:group 0 2 storage_uniform params sat_params

// distribute_precision splits two moment values into four channels for increased
// SAT precision. Each moment is decomposed into floor(M × 256) / 256 (coarse)
// and fract(M × 256) (fine). The coarse parts occupy R and G; the fine parts
// occupy B and A.
//
// Reference: GPU Gems 3, Listing 8-5.
fn distribute_precision(moments: vec2<f32>) -> vec4<f32> {
    let scaled = moments * 256.0;
    let int_part = floor(scaled);
    let frac_part = scaled - int_part;
    return vec4<f32>(int_part / 256.0, frac_part);
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(gid.xy);

    if params.offset == 0 {
        // Prepare pass: read RG32Float moments, distribute precision, write RGBA32Float.
        let moments = textureLoad(input_tex, coord, 0).rg;
        let distributed = distribute_precision(moments);
        textureStore(output_tex, coord, distributed);
    } else {
        // Prefix-sum pass: accumulate from the texel at (coord − offset × direction).
        let current = textureLoad(input_tex, coord, 0);
        let prev_coord = coord - params.direction * params.offset;

        var prev = vec4<f32>(0.0);
        if prev_coord.x >= 0 && prev_coord.y >= 0 {
            prev = textureLoad(input_tex, prev_coord, 0);
        }

        textureStore(output_tex, coord, current + prev);
    }
}
