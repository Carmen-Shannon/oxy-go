// Lit fragment shader — VSM variant (Forward+ Blinn-Phong with variance shadow mapping)
//
// Identical to lit-frag.wgsl except for the shadow sampling path:
//   - Uses a texture_2d<f32> moments texture (RG32Float) + linear sampler
//     instead of a texture_depth_2d + comparison sampler.
//   - Replaces the 3×3 PCF kernel with Chebyshev's inequality on the first
//     two depth moments, yielding smooth, filter-width-independent soft shadows.
//   - Applies a light-bleeding reduction step (linstep) to suppress artifacts
//     caused by overlapping occluders at different depths.
//
// Bind group layout:
//   @group(0) camera     — CameraUniform (view_proj + camera_position)
//   @group(2) material   — diffuse texture + sampler, normal map, metallic-roughness map
//   @group(3) lights     — LightHeader + Light array (storage buffer)
//   @group(4) shadow     — VSM moments texture, linear sampler, ShadowData uniform
//   @group(5) tiles      — TileUniforms + per-tile light counts + per-tile light indices
//   @group(6) ssao       — blurred SSAO occlusion texture + sampler (fallback: 1×1 white)
//   @group(7) probes     — irradiance probe SH storage + grid params uniform

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
//@oxy:include shadow_data
//@oxy:include tile_uniforms
//@oxy:include irradiance_probe
//@oxy:include probe_grid_params

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

//@oxy:group 3 0 storage_uniform light_header light_header
//@oxy:group 3 1 storage_read lights array<light>

//@oxy:provider 4 0 shadow
@group(4) @binding(0) var shadow_texture: texture_2d<f32>;
//@oxy:provider 4 1 shadow
@group(4) @binding(1) var shadow_sampler: sampler;
//@oxy:group 4 2 storage_uniform shadow_data shadow_data

//@oxy:group 5 0 storage_uniform tile_uniforms tile_uniforms
//@oxy:provider 5 1 tiles
@group(5) @binding(1) var<storage, read> tile_counts: array<u32>;
@group(5) @binding(2) var<storage, read> tile_indices: array<u32>;

//@oxy:provider 6 0 ssao ssao_texture
@group(6) @binding(0) var ssao_texture: texture_2d<f32>;
//@oxy:provider 6 1 ssao ssao_sampler
@group(6) @binding(1) var ssao_sampler: sampler;

//@oxy:group 7 0 storage_read probes array<irradiance_probe>
//@oxy:group 7 1 storage_uniform probe_grid_params probe_grid_params

// ── Constants ──────────────────────────────────────────────────────
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT:       u32 = 1u;
const LIGHT_TYPE_SPOT:        u32 = 2u;

const SPECULAR_STRENGTH: f32 = 0.5;

// ── Attenuation ────────────────────────────────────────────────────
fn attenuation(distance: f32, light_range: f32) -> f32 {
    if light_range <= 0.0 {
        return 0.0;
    }
    let ratio = clamp(distance / light_range, 0.0, 1.0);
    let window = 1.0 - ratio * ratio;
    return window * window;
}

// ── Spot cone falloff ──────────────────────────────────────────────
fn spot_falloff(cos_angle: f32, inner_cone: f32, outer_cone: f32) -> f32 {
    return clamp((cos_angle - outer_cone) / max(inner_cone - outer_cone, 0.0001), 0.0, 1.0);
}

// ── VSM helpers ────────────────────────────────────────────────────

// linstep performs a linear interpolation that returns 0 when v <= lo,
// 1 when v >= hi, and a linearly interpolated value in between.
fn linstep(lo: f32, hi: f32, v: f32) -> f32 {
    return clamp((v - lo) / (hi - lo), 0.0, 1.0);
}

// reduce_light_bleeding suppresses light-bleeding artefacts that occur
// when multiple occluders at different depths share a texel. The amount
// parameter controls how aggressively small p_max values are clamped
// toward zero — higher values produce stronger reduction at the cost of
// slightly darker umbrae.
fn reduce_light_bleeding(p_max: f32, amount: f32) -> f32 {
    return linstep(amount, 1.0, p_max);
}

// chebyshev_upper_bound applies Chebyshev's inequality to the first two
// depth moments to obtain an upper bound on the probability that the
// current fragment is in shadow. Returns 1.0 (fully lit) when the
// fragment depth is in front of or equal to the mean blocker depth.
//
// Reference: Donnelly & Lauritzen, "Variance Shadow Maps", I3D 2006.
fn chebyshev_upper_bound(moments: vec2<f32>, depth: f32, min_variance: f32) -> f32 {
    // If the fragment is in front of the mean blocker depth, it is lit.
    let d = depth - moments.x;
    if d <= 0.0 {
        return 1.0;
    }

    // Variance = E(x²) - E(x)². Clamp to min_variance to prevent
    // division by zero and numerical noise on flat surfaces.
    let variance = max(moments.y - moments.x * moments.x, min_variance);

    // One-tailed Chebyshev's inequality: P(x >= depth) <= variance / (variance + d²)
    let p_max = variance / (variance + d * d);

    return p_max;
}

// ── VSM shadow sampling ────────────────────────────────────────────
// Projects the world position into light space, computes the same linear
// depth metric used by the VSM depth pass, samples the blurred moments
// texture, and evaluates Chebyshev's inequality with light-bleeding
// reduction.
fn sample_shadow(world_pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    // Normal-offset bias (same as PCF variant) to reduce self-shadowing
    // on surfaces nearly parallel to the light direction.
    let n_dot_l = dot(normal, -light_dir);
    let offset_scale = shadow_data.normal_bias * (1.0 - n_dot_l);
    let offset_pos = world_pos + normal * offset_scale;

    // Project into light clip space.
    let clip = shadow_data.light_vp * vec4<f32>(offset_pos, 1.0);
    let ndc = clip.xyz / clip.w;

    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    // Fragments outside the shadow map receive no shadow (fully lit).
    if shadow_uv.x < 0.0 || shadow_uv.x > 1.0 ||
       shadow_uv.y < 0.0 || shadow_uv.y > 1.0 {
        return 1.0;
    }

    // Compute linear depth matching the VSM depth pass:
    //   linear_depth = (-view_z - near) / (far - near)
    let view_pos = shadow_data.light_view * vec4<f32>(offset_pos, 1.0);
    let linear_depth = (-view_pos.z - shadow_data.shadow_near)
                     / (shadow_data.shadow_far - shadow_data.shadow_near);

    // Clamp to [0,1] — anything outside the frustum is fully lit.
    if linear_depth < 0.0 || linear_depth > 1.0 {
        return 1.0;
    }

    // Sample the blurred moments texture (R = E(depth), G = E(depth²)).
    let moments = textureSample(shadow_texture, shadow_sampler, shadow_uv).rg;

    // Chebyshev upper-bound test.
    var p_max = chebyshev_upper_bound(moments, linear_depth, shadow_data.min_variance);

    // Light-bleeding reduction.
    p_max = reduce_light_bleeding(p_max, shadow_data.light_bleed_reduction);

    return p_max;
}

// ── Per-light contribution ─────────────────────────────────────────
fn evaluate_light(
    light: Light,
    surface_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    roughness: f32,
    metallic: f32,
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
    let diffuse = n_dot_l * light.color * light.intensity;

    let shininess = mix(4.0, 128.0, pow(1.0 - roughness, 2.0));
    let spec_strength = mix(SPECULAR_STRENGTH, 1.0, metallic);
    let half_dir = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let specular = select(vec3<f32>(0.0), spec_strength * pow(n_dot_h, shininess) * light.color * light.intensity, n_dot_l > 0.0);

    return (diffuse + specular) * atten;
}

// ── Irradiance probe helpers ───────────────────────────────────────

// eval_sh_irradiance evaluates L2 spherical harmonics for a normalized direction
// and reconstructs per-channel irradiance from a probe's SH coefficients.
fn eval_sh_irradiance(n: vec3<f32>, probe: IrradianceProbe) -> vec3<f32> {
    var sh: array<f32, 9>;
    sh[0] = 0.282095;
    sh[1] = 0.488603 * n.y;
    sh[2] = 0.488603 * n.z;
    sh[3] = 0.488603 * n.x;
    sh[4] = 1.092548 * n.x * n.y;
    sh[5] = 1.092548 * n.y * n.z;
    sh[6] = 0.315392 * (3.0 * n.z * n.z - 1.0);
    sh[7] = 1.092548 * n.x * n.z;
    sh[8] = 0.546274 * (n.x * n.x - n.y * n.y);

    // Copy SH arrays to locals so Naga allows runtime indexing.
    var r_coeff = probe.sh_r;
    var g_coeff = probe.sh_g;
    var b_coeff = probe.sh_b;

    var r = 0.0; var g = 0.0; var b = 0.0;
    for (var i = 0u; i < 9u; i++) {
        r += r_coeff[i] * sh[i];
        g += g_coeff[i] * sh[i];
        b += b_coeff[i] * sh[i];
    }
    return max(vec3<f32>(r, g, b), vec3<f32>(0.0));
}

// sample_probe_grid performs trilinear interpolation of the 8 surrounding
// probes for the given world-space position and surface normal to compute
// smooth indirect diffuse irradiance.
fn sample_probe_grid(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    if probe_grid_params.total_probes == 0u {
        return vec3<f32>(0.0);
    }

    let grid_min = probe_grid_params.grid_min;
    let spacing  = probe_grid_params.spacing;
    let cx = probe_grid_params.probe_count_x;
    let cy = probe_grid_params.probe_count_y;
    let cz = probe_grid_params.probe_count_z;

    // Continuous grid coordinates.
    let rel = (world_pos - grid_min) / spacing;
    let fx = clamp(rel.x, 0.0, f32(cx - 1u));
    let fy = clamp(rel.y, 0.0, f32(cy - 1u));
    let fz = clamp(rel.z, 0.0, f32(cz - 1u));

    let ix = u32(floor(fx));
    let iy = u32(floor(fy));
    let iz = u32(floor(fz));
    let ix1 = min(ix + 1u, cx - 1u);
    let iy1 = min(iy + 1u, cy - 1u);
    let iz1 = min(iz + 1u, cz - 1u);

    let tx = fx - f32(ix);
    let ty = fy - f32(iy);
    let tz = fz - f32(iz);

    // Trilinear interpolation over 8 corner probes.
    let i000 = ix  + iy  * cx + iz  * cx * cy;
    let i100 = ix1 + iy  * cx + iz  * cx * cy;
    let i010 = ix  + iy1 * cx + iz  * cx * cy;
    let i110 = ix1 + iy1 * cx + iz  * cx * cy;
    let i001 = ix  + iy  * cx + iz1 * cx * cy;
    let i101 = ix1 + iy  * cx + iz1 * cx * cy;
    let i011 = ix  + iy1 * cx + iz1 * cx * cy;
    let i111 = ix1 + iy1 * cx + iz1 * cx * cy;

    let c000 = eval_sh_irradiance(normal, probes[i000]);
    let c100 = eval_sh_irradiance(normal, probes[i100]);
    let c010 = eval_sh_irradiance(normal, probes[i010]);
    let c110 = eval_sh_irradiance(normal, probes[i110]);
    let c001 = eval_sh_irradiance(normal, probes[i001]);
    let c101 = eval_sh_irradiance(normal, probes[i101]);
    let c011 = eval_sh_irradiance(normal, probes[i011]);
    let c111 = eval_sh_irradiance(normal, probes[i111]);

    let c00 = mix(c000, c100, tx);
    let c10 = mix(c010, c110, tx);
    let c01 = mix(c001, c101, tx);
    let c11 = mix(c011, c111, tx);

    let c0 = mix(c00, c10, ty);
    let c1 = mix(c01, c11, ty);

    return mix(c0, c1, tz);
}

// ── Entry point ────────────────────────────────────────────────────
@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(diffuse_texture, diffuse_sampler, in.uv);

    if tex_color.a < 0.01 {
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

    // ── Forward+ tiled light loop ──────────────────────────────────
    let frag_coord = vec2<u32>(in.position.xy);
    let tile_x = frag_coord.x / 16u;
    let tile_y = frag_coord.y / 16u;
    let tile_index = tile_y * tile_uniforms.tile_count_x + tile_x;

    let num_tile_lights = tile_counts[tile_index];
    let tile_base = tile_index * tile_uniforms.max_lights_per_tile;

    // Sample screen-space ambient occlusion from the blurred SSAO texture.
    let ssao_dims = textureDimensions(ssao_texture);
    let screen_uv = in.position.xy / vec2<f32>(f32(ssao_dims.x), f32(ssao_dims.y));
    let ao = textureSample(ssao_texture, ssao_sampler, screen_uv).r;

    // Indirect diffuse illumination from the irradiance probe grid.
    let indirect = sample_probe_grid(in.world_position, normal);

    var total_light = light_header.ambient_color * ao + indirect * ao;
    for (var i = 0u; i < num_tile_lights; i++) {
        let light_idx = tile_indices[tile_base + i];
        let light = lights[light_idx];

        var contribution = evaluate_light(light, in.world_position, normal, view_dir, roughness, metallic);

        if light.light_type == LIGHT_TYPE_DIRECTIONAL && light.casts_shadows == 1u {
            let face_dot = dot(normal, normalize(-light.direction));
            if face_dot > 0.1 {
                contribution *= sample_shadow(in.world_position, normal, light.direction);
            }
        }

        total_light += contribution;
    }

    let final_color = albedo * total_light;
    return vec4<f32>(final_color, tex_color.a * in.color.a);
}
