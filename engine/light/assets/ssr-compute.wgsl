// SSR (Screen-Space Reflections) compute shader — Hi-Z ray march variant.
//
// Reads the G-Buffer normal/roughness and depth textures, the hierarchical
// depth pyramid (Hi-Z), and the HDR lit result, then performs a hierarchical
// screen-space ray march to find reflected pixels. Outputs RGBA16Float where
// RGB = reflected color and A = hit confidence (0 = miss, 1 = perfect hit).
//
// The Hi-Z march starts at a coarse mip level and advances by cell boundaries,
// dropping to finer levels on potential intersections and rising to coarser
// levels in empty space. This dramatically reduces the number of texture
// samples compared to uniform-stride linear marching.

//@oxy:include ssr_params

@group(0) @binding(0) var<uniform> ssr_params: SSRParams;

// G-Buffer textures (from the G-Buffer pre-pass)
//@oxy:provider 0 1 ssr gbuffer_normal
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
//@oxy:provider 0 2 ssr gbuffer_depth
@group(0) @binding(2) var gbuffer_depth: texture_depth_2d;

// HDR lit result texture (output of the forward pass)
//@oxy:provider 0 3 ssr hdr_texture
@group(0) @binding(3) var hdr_texture: texture_2d<f32>;

// SSR output texture (RGBA16Float, half-resolution)
//@oxy:provider 0 4 ssr ssr_output
@group(0) @binding(4) var ssr_output: texture_storage_2d<rgba16float, write>;

// Hierarchical depth pyramid (R32Float, full mip chain, min-depth per cell)
//@oxy:provider 0 5 ssr hiz_texture
@group(0) @binding(5) var hiz_texture: texture_2d<f32>;

// reconstructViewPos reconstructs a view-space position from screen UV and depth.
fn reconstructViewPos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let clip = ssr_params.inv_projection * vec4<f32>(ndc.x, -ndc.y, ndc.z, 1.0);
    return clip.xyz / clip.w;
}

// projectToScreen projects a view-space position to screen UV and NDC depth.
fn projectToScreen(pos_vs: vec3<f32>) -> vec3<f32> {
    let clip = ssr_params.projection * vec4<f32>(pos_vs, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec3<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5, ndc.z);
}

// crossCell computes the parametric distance along a screen-space ray to the
// next Hi-Z cell boundary at the given cell size.
fn crossCell(pos: vec2<f32>, dir: vec2<f32>, cell_size: vec2<f32>) -> f32 {
    let cell = floor(pos / cell_size);
    let next = select(cell, cell + 1.0, dir > vec2<f32>(0.0)) * cell_size;
    let delta = (next - pos) / dir;

    var t = 1e10;
    if (abs(dir.x) > 1e-8 && delta.x > 0.0) {
        t = min(t, delta.x);
    }
    if (abs(dir.y) > 1e-8 && delta.y > 0.0) {
        t = min(t, delta.y);
    }
    return t;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_size = textureDimensions(ssr_output);
    if (gid.x >= output_size.x || gid.y >= output_size.y) {
        return;
    }

    // SSR runs at half-resolution; map to full-resolution coords for G-Buffer reads.
    let full_size = vec2<f32>(ssr_params.screen_width, ssr_params.screen_height);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(output_size);

    // Sample G-Buffer normal + roughness at corresponding full-res texel.
    let full_coord = vec2<i32>(vec2<f32>(gid.xy) * 2.0);
    let normal_rough = textureLoad(gbuffer_normal, full_coord, 0);
    let world_normal = normal_rough.xyz * 2.0 - 1.0;
    let roughness = normal_rough.w;

    // Early out for rough surfaces.
    if (roughness > ssr_params.roughness_cutoff) {
        textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Read depth and reconstruct view-space position.
    let depth = textureLoad(gbuffer_depth, full_coord, 0);
    if (depth >= 1.0) {
        textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    let view_pos = reconstructViewPos(uv, depth);

    // Transform normal to view space and compute reflection direction.
    let view_normal = normalize((ssr_params.view * vec4<f32>(world_normal, 0.0)).xyz);
    let view_dir = normalize(view_pos);
    let reflect_dir = reflect(view_dir, view_normal);

    // Project ray start and end to screen space (UV + NDC depth).
    let start_screen = projectToScreen(view_pos);
    let end_vs = view_pos + reflect_dir * ssr_params.max_distance;
    let end_screen = projectToScreen(end_vs);

    let ray_uv = end_screen.xy - start_screen.xy;
    let ray_uv_len = length(ray_uv);

    // Degenerate ray guard.
    if (ray_uv_len < 0.0001) {
        textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Normalised ray direction in UV space and corresponding depth rate.
    let dir_uv = ray_uv / ray_uv_len;
    let depth_start = start_screen.z;
    let depth_end = end_screen.z;

    // Hi-Z march parameters.
    let max_level = max(i32(ssr_params.hiz_mip_count) - 1, 0);
    let start_level = clamp(3, 0, max_level);

    // Small initial offset to avoid self-intersection (advance ~2 pixels).
    let pixel_stride = 1.0 / max(full_size.x, full_size.y);
    var t = pixel_stride * 2.0 / ray_uv_len;
    var level = start_level;

    var hit_color = vec3<f32>(0.0);
    var confidence = 0.0;

    for (var i = 0u; i < ssr_params.max_steps; i = i + 1u) {
        let pos_uv = start_screen.xy + dir_uv * (ray_uv_len * t);
        let pos_depth = mix(depth_start, depth_end, t);

        // Bounds check.
        if (pos_uv.x < 0.0 || pos_uv.x >= 1.0 || pos_uv.y < 0.0 || pos_uv.y >= 1.0 || t >= 1.0) {
            break;
        }

        // Sample Hi-Z at current mip level.
        let mip_size = vec2<f32>(textureDimensions(hiz_texture, level));
        let cell_coord = clamp(
            vec2<i32>(floor(pos_uv * mip_size)),
            vec2<i32>(0),
            vec2<i32>(mip_size) - 1,
        );
        let hiz_depth = textureLoad(hiz_texture, cell_coord, level).r;

        if (pos_depth > hiz_depth) {
            // Ray is behind the min-depth at this level — potential intersection.
            if (level == 0) {
                // Full-resolution hit test using view-space thickness comparison.
                let surface_pos = reconstructViewPos(pos_uv, hiz_depth);
                let ray_vs = reconstructViewPos(pos_uv, pos_depth);
                let depth_diff = ray_vs.z - surface_pos.z;

                if (depth_diff > 0.0 && depth_diff < ssr_params.thickness) {
                    // Hit — sample the HDR texture.
                    let sample_coord = vec2<i32>(pos_uv * full_size);
                    hit_color = textureLoad(hdr_texture, sample_coord, 0).rgb;

                    // Confidence: edge fade × roughness fade × distance fade.
                    let edge_fade = smoothstep(0.0, 0.05, pos_uv.x)
                                  * smoothstep(0.0, 0.05, pos_uv.y)
                                  * (1.0 - smoothstep(0.95, 1.0, pos_uv.x))
                                  * (1.0 - smoothstep(0.95, 1.0, pos_uv.y));
                    let roughness_fade = 1.0 - smoothstep(0.0, ssr_params.roughness_cutoff, roughness);
                    let distance_fade = 1.0 - t;
                    confidence = edge_fade * roughness_fade * distance_fade;
                    break;
                }

                // Not a real hit — advance past this cell.
                let cell_size = 1.0 / mip_size;
                let dt = crossCell(pos_uv, dir_uv, cell_size);
                t += max(dt, pixel_stride) / ray_uv_len;
            } else {
                // Refine: drop to a finer mip level.
                level -= 1;
            }
        } else {
            // Ray is in front of all surfaces at this level — advance to next cell.
            let cell_size = 1.0 / mip_size;
            let dt = crossCell(pos_uv, dir_uv, cell_size);
            t += max(dt, pixel_stride) / ray_uv_len;
            level = min(level + 1, max_level);
        }
    }

    textureStore(ssr_output, vec2<i32>(gid.xy), vec4<f32>(hit_color, confidence));
}
