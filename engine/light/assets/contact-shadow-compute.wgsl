// Contact shadow compute shader — screen-space ray march along the directional
// light direction to detect fine-detail occlusion at surface contacts (feet on
// ground, object creases, etc.).
//
// Reads the G-Buffer hardware depth texture. For each pixel, reconstructs the
// world-space position from depth, then marches along the light direction in
// world space, projecting each step back to screen space and comparing against
// the depth buffer. If the scene depth is closer than the marched point within
// a thickness tolerance, the pixel is marked as in contact shadow.
//
// Output: R32Float texture where 1.0 = fully lit, 0.0 = in contact shadow.
//
// Dispatch: ceil(width/16) × ceil(height/16) × 1

//@oxy:include contact_shadow_params

@group(0) @binding(0) var gbuffer_depth: texture_depth_2d;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(3) var output_tex: texture_storage_2d<r32float, write>;
//@oxy:group 0 4 storage_uniform contact_shadow_params contact_shadow_params

fn is_physics_scene_droplet(albedo_sample: vec4<f32>) -> bool {
    let albedo = albedo_sample.xyz;
    return albedo.z > 0.75 && albedo.y > 0.35 && albedo.x < 0.35;
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if gid.x >= u32(dims.x) || gid.y >= u32(dims.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(contact_shadow_params.screen_width, contact_shadow_params.screen_height);

    let depth = textureLoad(gbuffer_depth, vec2<i32>(gid.xy), 0);

    // Skip sky pixels (no geometry).
    if depth >= 1.0 || depth == 0.0 {
        textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    // Reconstruct world position from depth (same Y-flip as SSAO).
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = contact_shadow_params.inv_view_proj * ndc;
    let world_pos = world.xyz / world.w;
    let receiver_norm_sample = textureLoad(gbuffer_normal, vec2<i32>(gid.xy), 0);
    let receiver_normal = normalize(receiver_norm_sample.xyz * 2.0 - 1.0);
    let receiver_albedo = textureLoad(gbuffer_albedo, vec2<i32>(gid.xy), 0);

    if is_physics_scene_droplet(receiver_albedo) {
        textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    // Ray march from the surface toward the light source in world space.
    let step_size = contact_shadow_params.max_distance / f32(contact_shadow_params.step_count);
    var occlusion = 0.0;

    for (var i = 1u; i <= contact_shadow_params.step_count; i++) {
        let march_pos = world_pos - contact_shadow_params.light_direction * step_size * f32(i);

        // Project marched position to screen space.
        let clip = contact_shadow_params.view_proj * vec4<f32>(march_pos, 1.0);

        // Guard against points behind the camera.
        if clip.w <= 0.0 {
            continue;
        }

        let march_ndc = clip.xyz / clip.w;

        // Convert NDC to UV (Y-flip).
        let sample_uv = vec2<f32>((march_ndc.x + 1.0) * 0.5, (1.0 - march_ndc.y) * 0.5);

        // Bounds check.
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }

        // Sample depth buffer at the projected UV.
        let sample_coord = vec2<i32>(sample_uv * vec2<f32>(contact_shadow_params.screen_width, contact_shadow_params.screen_height));
        let scene_depth = textureLoad(gbuffer_depth, sample_coord, 0);
        let sample_albedo = textureLoad(gbuffer_albedo, sample_coord, 0);
        if is_physics_scene_droplet(sample_albedo) {
            continue;
        }
        let sample_norm = textureLoad(gbuffer_normal, sample_coord, 0);
        let sample_normal = normalize(sample_norm.xyz * 2.0 - 1.0);
        let depth_delta = march_ndc.z - scene_depth;

        if scene_depth < march_ndc.z && depth_delta > 0.0 && depth_delta < contact_shadow_params.thickness {
            let depth_weight = 1.0 - clamp(depth_delta / contact_shadow_params.thickness, 0.0, 1.0);
            let normal_dot = dot(receiver_normal, sample_normal);
            let normal_weight = clamp((0.9 - normal_dot) / 0.9, 0.0, 1.0);
            let hit_weight = clamp(depth_weight * normal_weight, 0.0, 1.0);
            let occlusion_gain = hit_weight * 0.4;
            occlusion = occlusion + (1.0 - occlusion) * occlusion_gain;
        }
    }

    let shadow = 1.0 - clamp(occlusion, 0.0, 1.0);
    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(shadow, 0.0, 0.0, 0.0));
}
