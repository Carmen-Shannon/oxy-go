// SSAO compute shader — hemisphere sampling with depth-aware occlusion.
//
// Reads the G-Buffer hardware depth texture and normal texture (world normal
// packed [0,1] + roughness). For each pixel, reconstructs the world-space
// position from depth using the inverse view-projection matrix, generates
// sample points in a hemisphere around the surface normal, projects them to
// screen space, and tests against the depth buffer. Occluded samples
// contribute to the occlusion factor.
//
// A 4×4 noise texture provides per-pixel random rotation of the sample
// kernel to reduce banding artifacts, which the subsequent bilateral blur
// pass smooths out.
//
// Dispatch: ceil(width/16) × ceil(height/16) × 1

//@oxy:include ssao_params

@group(0) @binding(0) var gbuffer_depth: texture_depth_2d;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var noise_texture: texture_2d<f32>;
@group(0) @binding(3) var output_tex: texture_storage_2d<r32float, write>;
//@oxy:group 0 4 storage_uniform ssao_params ssao_params
@group(0) @binding(5) var<uniform> ssao_kernel: array<vec4<f32>, 32>;

// reconstructWorldPos reconstructs the world-space position and linear depth
// from a screen UV and hardware depth value using the inverse view-projection
// matrix. Returns (world_x, world_y, world_z, linear_depth).
fn reconstructWorldPos(uv: vec2<f32>, depth: f32) -> vec4<f32> {
    let ndc = vec4<f32>(uv * 2.0 - vec2<f32>(1.0), depth, 1.0);
    let world = ssao_params.inv_view_proj * ndc;
    let world_pos = world.xyz / world.w;
    let linear_depth = length(world_pos - ssao_params.camera_position);
    return vec4<f32>(world_pos, linear_depth);
}

// world_to_screen projects a world-space position to screen-space pixel
// coordinates and NDC depth using the view-projection matrix.
fn world_to_screen(world_pos: vec3<f32>) -> vec3<f32> {
    let clip = ssao_params.projection * vec4<f32>(world_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let screen_x = (ndc.x * 0.5 + 0.5) * ssao_params.screen_width;
    let screen_y = (1.0 - (ndc.y * 0.5 + 0.5)) * ssao_params.screen_height;
    return vec3<f32>(screen_x, screen_y, ndc.z);
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if gid.x >= u32(dims.x) || gid.y >= u32(dims.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let scale = i32(ssao_params.gbuffer_scale);
    let gb_coord = coord * scale;
    let gb_dims = vec2<i32>(textureDimensions(gbuffer_depth));

    // Read hardware depth and reconstruct world position.
    let hw_depth = textureLoad(gbuffer_depth, gb_coord, 0);
    let center_uv = (vec2<f32>(gb_coord) + 0.5) / vec2<f32>(gb_dims);
    let pos_sample = reconstructWorldPos(center_uv, hw_depth);
    let frag_pos = pos_sample.xyz;
    let frag_depth = pos_sample.w;

    // Early out for sky/background pixels (no geometry written).
    if hw_depth >= 1.0 {
        textureStore(output_tex, coord, vec4<f32>(1.0));
        return;
    }

    // Unpack normal from [0,1] to [-1,1].
    let norm_sample = textureLoad(gbuffer_normal, gb_coord, 0);
    let normal = normalize(norm_sample.xyz * 2.0 - 1.0);

    // Read noise for random kernel rotation (tiled 4×4).
    let noise_coord = vec2<i32>(vec2<u32>(gid.xy) % 4u);
    let noise_val = textureLoad(noise_texture, noise_coord, 0).xyz;

    // Build TBN from the surface normal and the random noise vector.
    let tangent = normalize(noise_val - normal * dot(noise_val, normal));
    let bitangent = cross(normal, tangent);
    let TBN = mat3x3<f32>(tangent, bitangent, normal);

    let sample_count = min(ssao_params.sample_count, 32u);
    var occlusion = 0.0;

    for (var i = 0u; i < sample_count; i++) {
        // Transform hemisphere sample to world space around the fragment.
        let sample_offset = TBN * ssao_kernel[i].xyz;
        let sample_world = frag_pos + sample_offset * ssao_params.radius;

        // Project sample to screen space to look up the depth buffer.
        let sample_screen = world_to_screen(sample_world);
        let sample_pixel = vec2<i32>(vec2<f32>(sample_screen.xy));

        // Bounds check — samples outside the SSAO output are not occluded.
        if sample_pixel.x < 0 || sample_pixel.x >= i32(dims.x) ||
           sample_pixel.y < 0 || sample_pixel.y >= i32(dims.y) {
            continue;
        }

        // Read depth at the projected sample position, scaled to G-Buffer resolution.
        let gb_sample = clamp(sample_pixel * scale, vec2<i32>(0), gb_dims - 1);
        let sample_hw_depth = textureLoad(gbuffer_depth, gb_sample, 0);

        // Skip sky pixels at the sample location.
        if sample_hw_depth >= 1.0 {
            continue;
        }

        // Reconstruct the linear depth at the sample screen position.
        let sample_uv = (vec2<f32>(gb_sample) + 0.5) / vec2<f32>(gb_dims);
        let scene_pos = reconstructWorldPos(sample_uv, sample_hw_depth);
        let scene_depth = scene_pos.w;

        // Range check: only count occlusion within the configured radius.
        let range_check = smoothstep(0.0, 1.0, ssao_params.radius / max(abs(frag_depth - scene_depth), 0.0001));

        // Compare linear depths: if the scene surface at the sample pixel is
        // closer than the hemisphere sample point, the sample is occluded.
        let sample_depth_from_cam = length(sample_world - ssao_params.camera_position);
        if scene_depth < sample_depth_from_cam - ssao_params.bias {
            occlusion += range_check;
        }
    }

    occlusion = 1.0 - (occlusion / f32(sample_count));
    let ao = pow(occlusion, ssao_params.power);

    textureStore(output_tex, coord, vec4<f32>(ao, 0.0, 0.0, 0.0));
}
