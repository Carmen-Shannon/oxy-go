// Separable box-filter blur for Variance Shadow Maps.
//
// This shader implements a configurable-radius box filter over a two-channel
// (RG32Float) moments texture. It is dispatched twice per frame: once
// horizontal (direction = (1,0)) and once vertical (direction = (0,1)).
//
// The uniform `params` controls the blur direction and kernel half-width.
// A radius of 4 produces a 9-texel kernel (2*4+1). The shader uses
// textureLoad for precise texel access and clamps coordinates at borders.
//
// Bind group layout (@group(0)):
//   @binding(0) texture_2d<f32>                    — input moments texture (read)
//   @binding(1) texture_storage_2d<rg32float,write> — output moments texture (write)
//   @binding(2) uniform                            — BlurParams (direction, radius)

//@oxy:include blur_params

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rg32float, write>;
//@oxy:group 0 2 storage_uniform params blur_params

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let dir = params.direction;
    let r = params.radius;
    let count = 2 * r + 1;

    var sum = vec2<f32>(0.0);
    for (var i = -r; i <= r; i++) {
        let sample_coord = coord + dir * i;
        let clamped = clamp(sample_coord, vec2<i32>(0), vec2<i32>(dims) - 1);
        sum += textureLoad(input_tex, clamped, 0).rg;
    }

    let avg = sum / f32(count);
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(avg, 0.0, 0.0));
}
