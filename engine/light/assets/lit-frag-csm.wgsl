// Lit fragment shader — CSM variant (Forward+ Blinn-Phong with cascaded shadow maps)
//
// Uses a dual-cascade shadow map system:
//   - Cascade 0 (inner): fixed-radius sphere centered on the camera providing
//     high-fidelity shadows regardless of zoom level.
//   - Cascade 1 (outer): frustum-fit projection covering the remaining depth
//     range at lower resolution.
//   - Cascade selection uses 3D world-space distance from the camera position
//     to the fragment, compared against csm_data.inner_radius.
//   - Atlas layout: cascade i occupies column i of 2 columns.
//   - Shadow sampling uses hardware depth comparison with PCF (Percentage
//     Closer Filtering) via a 16-tap Poisson disk kernel for soft edges.
//
// Bind group layout:
//   @group(0) camera     — CameraUniform (view_proj + camera_position)
//   @group(2) material   — diffuse texture + sampler, normal map, metallic-roughness map
//   @group(3) lights     — LightHeader + Light array (storage buffer)
//   @group(4) shadow     — CSM depth atlas, spot/point atlas, comparison sampler, CSMData uniform, light shadow entries
//   @group(5) tiles      — TileUniforms + per-tile light counts + per-tile light indices
//   @group(6) ssao       — blurred SSAO occlusion texture + sampler (fallback: 1×1 white)
//   @group(4) shadow     — (cont.) contact shadow texture + sampler at bindings 5–6

struct FragmentInput {
    @builtin(position) position: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) uv:             vec2<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) color:          vec4<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) world_tangent:  vec4<f32>,
};

//@oxy:include camera
//@oxy:include light
//@oxy:include light_header
//@oxy:include csm_data
//@oxy:include tile_uniforms
//@oxy:include light_shadow_entry

struct MaterialParams {
    alpha_cutoff: f32,
}

// ── Bind groups ────────────────────────────────────────────────────
//@oxy:group 0 0 storage_uniform camera camera
//@oxy:provider 2 0 material diffuse_texture
@group(2) @binding(0) var diffuse_texture: texture_2d<f32>;
//@oxy:provider 2 1 material diffuse_sampler
@group(2) @binding(1) var diffuse_sampler: sampler;
//@oxy:provider 2 2 material normal_texture
@group(2) @binding(2) var normal_texture: texture_2d<f32>;
//@oxy:provider 2 3 material normal_sampler
@group(2) @binding(3) var normal_sampler: sampler;
//@oxy:provider 2 4 material metallic_roughness_texture
@group(2) @binding(4) var metallic_roughness_texture: texture_2d<f32>;
//@oxy:provider 2 5 material metallic_roughness_sampler
@group(2) @binding(5) var metallic_roughness_sampler: sampler;
//@oxy:provider 2 6 material material_params
@group(2) @binding(6) var<uniform> material_params: MaterialParams;

//@oxy:group 3 0 storage_uniform light_header light_header
//@oxy:group 3 1 storage_read lights array<light>

//@oxy:provider 4 0 shadow
@group(4) @binding(0) var shadow_texture: texture_depth_2d;
//@oxy:provider 4 1 shadow
@group(4) @binding(1) var shadow_sampler: sampler_comparison;
//@oxy:group 4 2 storage_uniform csm_data csm_data
//@oxy:provider 4 3 shadow spot_shadow_texture
@group(4) @binding(3) var spot_shadow_texture: texture_depth_2d;
//@oxy:group 4 4 storage_read light_shadow_entries array<light_shadow_entry>

//@oxy:group 5 0 storage_uniform tile_uniforms tile_uniforms
//@oxy:provider 5 1 tiles
@group(5) @binding(1) var<storage, read> tile_counts: array<u32>;
@group(5) @binding(2) var<storage, read> tile_indices: array<u32>;

//@oxy:provider 6 0 ssao ssao_texture
@group(6) @binding(0) var ssao_texture: texture_2d<f32>;
//@oxy:provider 6 1 ssao ssao_sampler
@group(6) @binding(1) var ssao_sampler: sampler;

//@oxy:provider 4 5 shadow contact_shadow_texture
@group(4) @binding(5) var contact_shadow_texture: texture_2d<f32>;
//@oxy:provider 4 6 shadow contact_shadow_sampler
@group(4) @binding(6) var contact_shadow_sampler: sampler;

// ── Constants ──────────────────────────────────────────────────────
//@oxy:inject LIGHT_TYPE_DIRECTIONAL u32 light_type_directional
//@oxy:inject LIGHT_TYPE_POINT u32 light_type_point
//@oxy:inject LIGHT_TYPE_SPOT u32 light_type_spot
//@oxy:inject TILE_SIZE u32 tile_size

// ── Attenuation ────────────────────────────────────────────────────
fn attenuation(distance: f32, light_range: f32) -> f32 {
    if light_range <= 0.0 {
        return 0.0;
    }
    let ratio = saturate(distance / light_range);
    let window = pow(saturate(1.0 - pow(ratio, 4.0)), 2.0);
    let norm_d2 = ratio * ratio;
    return window / (norm_d2 + 1.0);
}

// ── Spot cone falloff ──────────────────────────────────────────────
fn spot_falloff(cos_angle: f32, inner_cone: f32, outer_cone: f32) -> f32 {
    return saturate((cos_angle - outer_cone) / max(inner_cone - outer_cone, 0.0001));
}

// PCF Poisson disk offsets — 16-tap with good distribution for shadow PCF.
//@oxy:inject PCF_SAMPLES u32 pcf_samples
//@oxy:inject PCF_SAMPLES_SPOT u32 pcf_samples_spot
var<private> POISSON_DISK: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845),
    vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554),
    vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507),
    vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367),
    vec2<f32>( 0.14383161, -0.14100790),
);

// sample_single_cascade projects offset_pos into cascade idx and returns the
// PCF shadow factor. Returns 1.0 (fully lit) if the point falls outside the
// cascade's ortho frustum or depth range.
fn sample_single_cascade(idx: u32, offset_pos: vec3<f32>) -> f32 {
    let c = csm_data.cascades[idx];

    let clip     = c.light_vp * vec4<f32>(offset_pos, 1.0);
    let ndc      = clip.xyz / clip.w;
    let local_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if local_uv.x < 0.0 || local_uv.x > 1.0 ||
       local_uv.y < 0.0 || local_uv.y > 1.0 {
        return 1.0;
    }

    // Projective depth for comparison against the hardware depth buffer.
    // The depth buffer stores clip.z/clip.w values written by the shadow pass.
    let ref_depth = saturate(ndc.z);

    // PCF: Poisson disk sampling with configurable radius.
    // Offsets are applied in cascade-local UV space (where texel_size is correct),
    // then each sample point is converted to atlas UV individually to avoid 2:1
    // horizontal asymmetry and cross-cascade bleeding.
    let depth_range = c.shadow_far - c.shadow_near;
    let ndc_bias = csm_data.bias / depth_range;
    let texel_scale = csm_data.texel_size * csm_data.pcf_radius;
    var shadow = 0.0;
    for (var i = 0u; i < PCF_SAMPLES; i = i + 1u) {
        let offset = POISSON_DISK[i] * texel_scale;
        let sample_uv = local_uv + offset;
        let sample_atlas_u = (sample_uv.x + f32(idx)) / 2.0;
        shadow += textureSampleCompareLevel(
            shadow_texture, shadow_sampler,
            vec2<f32>(sample_atlas_u, sample_uv.y),
            ref_depth - ndc_bias,
        );
    }
    shadow /= f32(PCF_SAMPLES);

    return shadow;
}

fn sample_shadow_csm(world_pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>,
                     camera_depth: f32) -> f32 {
    // Select cascade based on 3D world-space distance from camera.
    let dist = length(world_pos - camera.camera_position);

    var cascade_idx: u32 = 1u;
    if dist < csm_data.inner_radius {
        cascade_idx = 0u;
    }

    let c          = csm_data.cascades[cascade_idx];
    let n_dot_l    = dot(normal, -light_dir);
    let offset_pos = world_pos + normal * (c.normal_bias * (1.0 - n_dot_l));

    var shadow = sample_single_cascade(cascade_idx, offset_pos);

    // Smooth blend from cascade 0 to cascade 1 over the final 15% of the
    // inner radius, eliminating hard transition lines.
    if cascade_idx == 0u {
        let blend_start = csm_data.inner_radius * 0.85;
        if dist > blend_start {
            let blend_factor = saturate((dist - blend_start) / (csm_data.inner_radius - blend_start));
            let next_c      = csm_data.cascades[1u];
            let next_offset = world_pos + normal * (next_c.normal_bias * (1.0 - n_dot_l));
            let shadow_next = sample_single_cascade(1u, next_offset);
            shadow = mix(shadow, shadow_next, blend_factor);
        }
    }

    // Distance fade: blend shadow to 1.0 (fully lit) over the last 10% of max distance.
    if csm_data.shadow_max_distance > 0.0 {
        let fade_start = csm_data.shadow_max_distance * 0.9;
        let fade = saturate((camera_depth - fade_start) / (csm_data.shadow_max_distance - fade_start));
        shadow = mix(shadow, 1.0, fade);
    }

    return shadow;
}

// sample_shadow_spot projects the fragment into the spot light's shadow map and
// returns the PCF shadow factor (0.0 = fully shadowed, 1.0 = fully lit).
fn sample_shadow_spot(frag_pos: vec3<f32>, entry: LightShadowEntry, normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let light_clip = entry.light_vp * vec4<f32>(frag_pos, 1.0);
    let ndc = light_clip.xyz / light_clip.w;

    // NDC → tile-local UV. WebGPU: y=-1 at bottom, UV y=0 at top.
    let local_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));

    // Outside light frustum → fully lit.
    if (local_uv.x < 0.0 || local_uv.x > 1.0 ||
        local_uv.y < 0.0 || local_uv.y > 1.0 ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    let NdotL = max(dot(normal, light_dir), 0.0);
    let ref_depth = ndc.z - entry.bias * (1.0 + 1.0 * (1.0 - NdotL));

    // Tile pixel width from atlas rect: atlas_width_px * atlas_rect.z.
    let atlas_w = f32(textureDimensions(spot_shadow_texture).x);
    let tile_pixels = atlas_w * entry.atlas_rect.z;
    let texel_size = 1.0 / tile_pixels;

    // PCF in tile-local UV space, then convert each sample to atlas UV.
    let pcf_scale = texel_size * 1.5;
    var shadow = 0.0;
    for (var i = 0u; i < PCF_SAMPLES_SPOT; i = i + 1u) {
        let offset = POISSON_DISK[i] * pcf_scale;
        let sample_local = local_uv + offset;
        let atlas_uv = vec2<f32>(
            entry.atlas_rect.x + sample_local.x * entry.atlas_rect.z,
            entry.atlas_rect.y + sample_local.y * entry.atlas_rect.w,
        );
        shadow += textureSampleCompareLevel(
            spot_shadow_texture, shadow_sampler,
            atlas_uv,
            ref_depth,
        );
    }
    shadow /= f32(PCF_SAMPLES_SPOT);

    return shadow;
}

fn sample_shadow_point(frag_pos: vec3<f32>, light_pos: vec3<f32>, shadow_index: u32, normal: vec3<f32>) -> f32 {
    let dir = frag_pos - light_pos;
    let abs_dir = abs(dir);

    // Select cube face by dominant axis.
    var face_index: u32;
    if abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z {
        if dir.x >= 0.0 { face_index = 0u; } else { face_index = 1u; } // +X / -X
    } else if abs_dir.y >= abs_dir.x && abs_dir.y >= abs_dir.z {
        if dir.y >= 0.0 { face_index = 2u; } else { face_index = 3u; } // +Y / -Y
    } else {
        if dir.z >= 0.0 { face_index = 4u; } else { face_index = 5u; } // +Z / -Z
    }

    let entry = light_shadow_entries[shadow_index + face_index];
    let point_light_dir = normalize(light_pos - frag_pos);
    return sample_shadow_spot(frag_pos, entry, normal, point_light_dir);
}

// ── PBR BRDF ───────────────────────────────────────────────────────
const PI: f32 = 3.14159265359;

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

// ── Per-light contribution ─────────────────────────────────────────
fn evaluate_light(
    light: Light,
    surface_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    roughness: f32,
    metallic: f32,
    albedo: vec3<f32>,
) -> vec3<f32> {
    var light_dir: vec3<f32>;
    var atten: f32 = 1.0;

    switch light.light_type {
        case LIGHT_TYPE_DIRECTIONAL: {
            light_dir = normalize(-light.direction);
        }
        case LIGHT_TYPE_POINT: {
            let to_light = light.position - surface_pos;
            let dist = length(to_light);
            light_dir = to_light / max(dist, 0.0001);
            atten = attenuation(dist, light.light_range);
        }
        case LIGHT_TYPE_SPOT: {
            let to_light = light.position - surface_pos;
            let dist = length(to_light);
            light_dir = to_light / max(dist, 0.0001);
            atten = attenuation(dist, light.light_range);

            let cos_angle = dot(-light_dir, normalize(light.direction));
            atten *= spot_falloff(cos_angle, light.inner_cone, light.outer_cone);
        }
        default: {
            return vec3<f32>(0.0);
        }
    }

    let n_dot_l = max(dot(normal, light_dir), 0.0);
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }

    let half_dir = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 0.001);
    let h_dot_v = max(dot(half_dir, view_dir), 0.0);

    // F0: 0.04 for dielectrics, albedo for metals.
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let D = distribution_ggx(n_dot_h, roughness);
    let F = fresnel_schlick(h_dot_v, f0);
    let G = geometry_smith(n_dot_v, n_dot_l, roughness);

    // Specular: Cook-Torrance
    let specular = (D * G) * F / (4.0 * n_dot_v * n_dot_l);

    // Diffuse: energy-conserving Lambertian
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let diffuse = kD * albedo / PI;

    let radiance = light.color * light.intensity * atten;
    return (diffuse + specular) * radiance * n_dot_l;
}

// ── Entry point ────────────────────────────────────────────────────
@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(diffuse_texture, diffuse_sampler, in.uv);

    if (tex_color.a < material_params.alpha_cutoff) {
        discard;
    }

    let albedo = tex_color.rgb * in.color.rgb;

    let normal_sample = textureSample(normal_texture, normal_sampler, in.uv).rgb;
    let mapped_normal = normal_sample * 2.0 - 1.0;

    var N = normalize(in.world_normal);
    if !in.front_facing {
        N = -N;
    }
    let T = normalize(in.world_tangent.xyz);
    let B = cross(N, T) * in.world_tangent.w;
    let TBN = mat3x3<f32>(T, B, N);
    var normal = normalize(TBN * mapped_normal);

    let mr_sample = textureSample(metallic_roughness_texture, metallic_roughness_sampler, in.uv);
    let roughness = mr_sample.g;
    let metallic = mr_sample.b;

    let view_dir = normalize(camera.camera_position - in.world_position);

    // Camera-space depth for CSM cascade selection.
    let cam_depth = -( camera.view * vec4<f32>(in.world_position, 1.0) ).z;

    // ── Forward+ tiled light loop ──────────────────────────────────
    let frag_coord = vec2<u32>(in.position.xy);
    let tile_x = frag_coord.x / TILE_SIZE;
    let tile_y = frag_coord.y / TILE_SIZE;
    let tile_index = tile_y * tile_uniforms.tile_count_x + tile_x;

    let num_tile_lights = tile_counts[tile_index];
    let tile_base = tile_index * tile_uniforms.max_lights_per_tile;

    // Sample screen-space ambient occlusion from the blurred SSAO texture.
    let screen_uv = in.position.xy / vec2<f32>(f32(tile_uniforms.screen_width), f32(tile_uniforms.screen_height));
    let ao = textureSample(ssao_texture, ssao_sampler, screen_uv).r;

    var total_light = albedo * (light_header.ambient_color * ao);
    for (var i = 0u; i < num_tile_lights; i++) {
        let light_idx = tile_indices[tile_base + i];
        let light = lights[light_idx];

        var contribution = evaluate_light(light, in.world_position, normal, view_dir, roughness, metallic, albedo);
        if (all(contribution < vec3<f32>(0.001))) { continue; }

        if light.light_type == LIGHT_TYPE_DIRECTIONAL && light.casts_shadows == 1u {
            contribution *= sample_shadow_csm(in.world_position, normal, light.direction, cam_depth);
            let contact = textureSample(contact_shadow_texture, contact_shadow_sampler, screen_uv).r;
            contribution *= contact;
        }

        if light.light_type == LIGHT_TYPE_SPOT && light.shadow_index != 0xFFFFFFFFu {
            let entry = light_shadow_entries[light.shadow_index];
            let spot_light_dir = normalize(light.position - in.world_position);
            contribution *= sample_shadow_spot(in.world_position, entry, normal, spot_light_dir);
        }

        if light.light_type == LIGHT_TYPE_POINT && light.shadow_index != 0xFFFFFFFFu {
            contribution *= sample_shadow_point(in.world_position, light.position, light.shadow_index, normal);
        }

        total_light += contribution;
    }

    let final_color = total_light;
    return vec4<f32>(final_color, tex_color.a * in.color.a);
}
