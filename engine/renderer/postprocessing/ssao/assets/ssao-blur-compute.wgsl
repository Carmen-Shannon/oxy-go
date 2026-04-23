// SSAO bilateral blur compute shader — edge-preserving smoothing.
//
// Performs a separable bilateral blur on the raw SSAO occlusion texture.
// The blur kernel is depth-aware: samples whose hardware depth differs from
// the center pixel by more than a threshold contribute less, preventing
// occlusion from bleeding across geometric boundaries.
//
// Dispatch twice per frame: once horizontal (direction = (1,0)), once
// vertical (direction = (0,1)), matching the separable blur architecture.
//
// Dispatch: ceil(width/16) × ceil(height/16) × 1

//@oxy:include blur_params

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<r32float, write>;
//@oxy:group 0 2 storage_uniform params blur_params
@group(0) @binding(3) var depth_tex: texture_depth_2d;

const DEPTH_THRESHOLD: f32 = 0.005;

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let dir = params.direction;
    let r = params.radius;
    let scale = max(params.gbuffer_scale, 1);

    let center_ao = textureLoad(input_tex, coord, 0).r;
    let center_depth = textureLoad(depth_tex, coord * scale, 0);

    // Early out for sky pixels (no geometry).
    if center_depth >= 1.0 {
        textureStore(output_tex, coord, vec4<f32>(1.0));
        return;
    }

    var weighted_sum = center_ao;
    var total_weight = 1.0;

    for (var i = -r; i <= r; i++) {
        if i == 0 {
            continue;
        }
        let sample_coord = coord + dir * i;
        let clamped = clamp(sample_coord, vec2<i32>(0), vec2<i32>(dims) - 1);

        let sample_ao = textureLoad(input_tex, clamped, 0).r;
        let sample_depth = textureLoad(depth_tex, clamped * scale, 0);

        // Bilateral weight: reduce contribution when depth differs significantly
        // from the center pixel to preserve geometric edges.
        let depth_diff = abs(center_depth - sample_depth);
        let bilateral_weight = 1.0 - smoothstep(0.0, DEPTH_THRESHOLD + DEPTH_THRESHOLD * center_depth, depth_diff);

        weighted_sum += sample_ao * bilateral_weight;
        total_weight += bilateral_weight;
    }

    let result = weighted_sum / max(total_weight, 0.001);
    textureStore(output_tex, coord, vec4<f32>(result, 0.0, 0.0, 0.0));
}
