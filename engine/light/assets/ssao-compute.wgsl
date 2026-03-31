// SSAO compute shader — hemisphere sampling with depth-aware occlusion
// and normal-based rejection.
//
// Reads the G-Buffer hardware depth texture and normal texture (world normal
// packed [0,1] + roughness). For each pixel, reconstructs the world-space
// position from depth using the inverse view-projection matrix, generates
// sample points in a hemisphere around the surface normal, projects them to
// screen space, and tests against the depth buffer. Occluded samples
// contribute to the occlusion factor.
//
// Normal-based rejection: after projecting a sample to screen space, the
// shader reads the G-Buffer normal at that pixel. If the sample-pixel normal
// is similar to the fragment normal (high dot product), the sample is on the
// same continuous surface and is rejected as self-occlusion. Only samples
// whose normals diverge (corners, creases, different surfaces) contribute
// full occlusion.
//
// TBN construction uses a stable cross-product tangent frame from the
// surface normal (non-parallel reference vector), then rotates the tangent
// and bitangent in the tangent plane by a per-pixel PCG hash angle. This
// gives proper per-pixel hemisphere rotation on all surface orientations,
// avoiding the axis-aligned degeneracy of Gram-Schmidt with a Z=0 noise
// vector. The subsequent bilateral blur smooths the resulting per-pixel noise.
//
// Dispatch: ceil(width/16) × ceil(height/16) × 1

//@oxy:include ssao_params

@group(0) @binding(0) var gbuffer_depth: texture_depth_2d;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(3) var output_tex: texture_storage_2d<r32float, write>;
//@oxy:group 0 4 storage_uniform ssao_params ssao_params
//@oxy:inject MAX_SSAO_SAMPLES u32 max_ssao_samples
@group(0) @binding(5) var<uniform> ssao_kernel: array<vec4<f32>, MAX_SSAO_SAMPLES>;

// reconstructWorldPos reconstructs the world-space position and linear depth
// from a screen UV and hardware depth value using the inverse view-projection
// matrix. Returns (world_x, world_y, world_z, linear_depth).
fn reconstructWorldPos(uv: vec2<f32>, depth: f32) -> vec4<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = ssao_params.inv_view_proj * ndc;
    let world_pos = world.xyz / world.w;
    let linear_depth = length(world_pos - ssao_params.camera_position);
    return vec4<f32>(world_pos, linear_depth);
}

// Per-pixel hash — produces a pseudo-random float in [0,1) from 2D
// pixel coordinates. Uses a PCG-style integer hash to eliminate
// spatial correlation, ensuring every pixel gets an independent
// hemisphere rotation for the SSAO sample kernel.
fn hash_pixel(px: u32, py: u32) -> f32 {
    var state = px * 1597334677u + py * 3812015801u + 2891336453u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    return f32(state) / 4294967295.0;
}

// build_tbn constructs a stable tangent frame from a surface normal. Uses a
// reference vector guaranteed non-parallel to the normal to produce orthogonal
// tangent and bitangent vectors via cross products.
fn build_tbn(n: vec3<f32>) -> mat3x3<f32> {
    let ref_vec = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) > 0.999);
    let t = normalize(cross(ref_vec, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
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

    // Per-pixel hash rotation — each pixel gets a unique hemisphere rotation
    // angle. build_tbn produces a stable tangent frame from the normal, then
    // the tangent and bitangent are rotated in the tangent plane by the hash
    // angle so every pixel gets a unique hemisphere orientation.
    let angle = hash_pixel(gid.x, gid.y) * 6.283185307;
    let base_tbn = build_tbn(normal);
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let t_rot = base_tbn[0] * cos_a + base_tbn[1] * sin_a;
    let b_rot = base_tbn[1] * cos_a - base_tbn[0] * sin_a;
    let TBN = mat3x3<f32>(t_rot, b_rot, normal);

    let sample_count = min(ssao_params.sample_count, MAX_SSAO_SAMPLES);
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

        // Read normal at the sample's projected screen position.
        let scene_normal_raw = textureLoad(gbuffer_normal, gb_sample, 0);
        let scene_normal = normalize(scene_normal_raw.xyz * 2.0 - 1.0);

        // Normal-based rejection: surfaces facing the same direction as
        // the fragment are on the same continuous surface and should not
        // count as occluders. The weight is zero when normals match and
        // increases as normals diverge.
        let normal_weight = 1.0 - max(dot(normal, scene_normal), 0.0);

        // Reconstruct the linear depth at the sample screen position.
        let sample_uv = (vec2<f32>(gb_sample) + 0.5) / vec2<f32>(gb_dims);
        let scene_pos = reconstructWorldPos(sample_uv, sample_hw_depth);
        let scene_depth = scene_pos.w;

        // Range check: only count occlusion from surfaces within radius in 3D world space.
        let scene_world_dist = length(scene_pos.xyz - frag_pos);
        let range_check = 1.0 - smoothstep(0.0, 1.0, scene_world_dist / ssao_params.radius);

        // Compare linear depths: if the scene surface at the sample pixel is
        // closer than the hemisphere sample point, the sample is occluded.
        let sample_depth_from_cam = length(sample_world - ssao_params.camera_position);
        if scene_depth < sample_depth_from_cam - ssao_params.bias * ssao_params.radius {
            occlusion += range_check * normal_weight;
        }
    }

    occlusion = 1.0 - (occlusion / f32(sample_count));
    let ao = pow(occlusion, ssao_params.power);
    textureStore(output_tex, coord, vec4<f32>(ao, 0.0, 0.0, 0.0));
}
