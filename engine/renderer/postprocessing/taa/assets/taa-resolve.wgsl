// taa-resolve.wgsl — Temporal Anti-Aliasing resolve compute shader.
//
// Blends the current HDR frame with the previously accumulated history texture
// using depth-based reprojection (Nehab et al. 2007; Yang et al. 2009) and
// neighborhood AABB clamping in YCoCg space (Karis [Kar14]) to reject stale history.
// Luminance-weighted blending (Karis [Kar13]) prevents high-energy samples (fireflies,
// bright speculars) from polluting the history buffer for many subsequent frames.
//
// History is sampled with bilinear filtering (textureSampleLevel at mip 0). Any
// reconstruction filter with negative lobes (Catmull-Rom, Mitchell-Netravali, Lanczos)
// applied inside the temporal feedback loop causes resonant oscillation on alternating
// Halton jitter phases. Post-TAA sharpening via a separate CAS pass (see Phase 10)
// is applied outside the feedback loop to recover fine detail.
//
// Outputs the blended result to the TAA resolved texture. The composition pass reads
// from the CAS-sharpened texture, NOT directly from this resolved texture.

//@oxy:include taa_params

@group(0) @binding(0) var<uniform> taa_params: TAAParams;

//@oxy:provider 0 1 taa taa_hdr_texture
@group(0) @binding(1) var hdr_texture: texture_2d<f32>;

//@oxy:provider 0 2 taa taa_history_texture
@group(0) @binding(2) var history_texture: texture_2d<f32>;

//@oxy:provider 0 3 taa taa_depth
@group(0) @binding(3) var gbuffer_depth: texture_depth_2d;

//@oxy:provider 0 4 taa taa_resolved
@group(0) @binding(4) var taa_resolved: texture_storage_2d<rgba16float, write>;

@group(0) @binding(5) var linear_sampler: sampler;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn rgbToYCoCg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
         0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b,
         0.5  * rgb.r               - 0.5  * rgb.b,
        -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b,
    );
}

fn yCoCgToRgb(ycocg: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        ycocg.x + ycocg.y - ycocg.z,
        ycocg.x            + ycocg.z,
        ycocg.x - ycocg.y - ycocg.z,
    );
}

fn reconstructWorldPos(uv: vec2<f32>, depth: f32) -> vec4<f32> {
    let ndc_x =  uv.x * 2.0 - 1.0;
    let ndc_y =  1.0 - uv.y * 2.0; // WebGPU UV row 0 = top; NDC Y=-1 = bottom
    let clip = vec4<f32>(ndc_x, ndc_y, depth, 1.0);
    let world_h = taa_params.inv_curr_view_proj * clip;
    return world_h / world_h.w;
}

fn worldToPrevUV(world_pos: vec4<f32>) -> vec2<f32> {
    let prev_clip = taa_params.prev_view_proj * world_pos;
    let prev_ndc  = prev_clip.xy / prev_clip.w;
    return vec2<f32>(
        prev_ndc.x *  0.5 + 0.5,
        prev_ndc.y * -0.5 + 0.5,
    );
}

fn currentJitterUV(uv: vec2<f32>) -> vec2<f32> {
    let inv_screen = vec2<f32>(
        1.0 / f32(taa_params.screen_width),
        1.0 / f32(taa_params.screen_height),
    );
    let half_texel = 0.5 * inv_screen;
    // Current upload order advances jitter before params upload, so the current
    // HDR/depth frame corresponds to jitter_prev rather than jitter_curr.
    let unjittered_uv = uv + vec2<f32>(
        -taa_params.jitter_prev.x * inv_screen.x,
         taa_params.jitter_prev.y * inv_screen.y,
    );
    return clamp(unjittered_uv, half_texel, vec2<f32>(1.0) - half_texel);
}

fn sampleCurrentReconstructed(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(hdr_texture, linear_sampler, currentJitterUV(uv), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(taa_resolved);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(taa_params.screen_width, taa_params.screen_height);
    let current_source_uv = currentJitterUV(uv);
    let current_depth_texel = vec2<i32>(
        current_source_uv * vec2<f32>(taa_params.screen_width, taa_params.screen_height),
    );

    let curr_color = textureSampleLevel(hdr_texture, linear_sampler, current_source_uv, 0.0).rgb;

    let depth = textureLoad(gbuffer_depth, current_depth_texel, 0);
    if (depth >= 1.0) {
        textureStore(taa_resolved, vec2<i32>(gid.xy), vec4<f32>(curr_color, 1.0));
        return;
    }

    let world_pos = reconstructWorldPos(current_source_uv, depth);
    let prev_uv   = worldToPrevUV(world_pos);

    let in_frame = prev_uv.x >= 0.0 && prev_uv.x <= 1.0 &&
                   prev_uv.y >= 0.0 && prev_uv.y <= 1.0;
    if (!in_frame) {
        textureStore(taa_resolved, vec2<i32>(gid.xy), vec4<f32>(curr_color, 1.0));
        return;
    }

    let history_color = textureSampleLevel(history_texture, linear_sampler, prev_uv, 0.0).rgb;
    if (taa_params.raw_history_only != 0.0) {
        textureStore(taa_resolved, vec2<i32>(gid.xy), vec4<f32>(history_color, 1.0));
        return;
    }

    var nmin = rgbToYCoCg(curr_color);
    var nmax = rgbToYCoCg(curr_color);
    var nsum = vec3<f32>(0.0);
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nc = clamp(
                vec2<i32>(gid.xy) + vec2<i32>(dx, dy),
                vec2<i32>(0),
                vec2<i32>(size) - vec2<i32>(1),
            );
            let nc_uv = (vec2<f32>(nc) + 0.5) / vec2<f32>(taa_params.screen_width, taa_params.screen_height);
            let s = rgbToYCoCg(sampleCurrentReconstructed(nc_uv));
            nmin = min(nmin, s);
            nmax = max(nmax, s);
            nsum = nsum + s;
        }
    }

    let history_ycocg = rgbToYCoCg(history_color);
    let rectification_scale = taa_params.history_rectification_scale;
    var clamp_min = nmin;
    var clamp_max = nmax;
    if (rectification_scale != 1.0) {
        let nmean = nsum / 9.0;
        clamp_min = nmean + (nmin - nmean) * rectification_scale;
        clamp_max = nmean + (nmax - nmean) * rectification_scale;
    }
    let clamped_history = yCoCgToRgb(clamp(history_ycocg, clamp_min, clamp_max));

    let w_curr = 1.0 / (1.0 + luminance(curr_color));
    let w_hist = 1.0 / (1.0 + luminance(clamped_history));
    let alpha = taa_params.blend_factor;
    let blended = (clamped_history * w_hist * (1.0 - alpha) + curr_color * w_curr * alpha)
                / (w_hist * (1.0 - alpha) + w_curr * alpha);

    textureStore(taa_resolved, vec2<i32>(gid.xy), vec4<f32>(blended, 1.0));
}
