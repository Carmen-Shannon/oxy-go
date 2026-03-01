// Lit fragment shader — PCSS variant (Forward+ Blinn-Phong with SAT-backed contact-hardening shadows)
//
// Extends the VSM shadow path with Percentage-Closer Soft Shadows (PCSS) using
// a Summed-Area Table (SAT) for constant-time variable-width shadow filtering.
// Shadows are sharp near contact points and grow progressively softer with
// increasing caster-receiver distance, mimicking real-world area-light penumbrae.
//
// The SAT texture stores precision-distributed moments (see GPU Gems 3, §8.5.2):
//   R = floor(M₁ × 256) / 256   (coarse depth)
//   G = floor(M₂ × 256) / 256   (coarse depth²)
//   B = fract(M₁ × 256)         (fine depth)
//   A = fract(M₂ × 256)         (fine depth²)
//
// Bind group layout:
//   @group(0) camera     — CameraUniform (view_proj + camera_position)
//   @group(2) material   — diffuse texture + sampler, normal map, metallic-roughness map
//   @group(3) lights     — LightHeader + Light array (storage buffer)
//   @group(4) shadow     — SAT texture (RGBA32Float), linear sampler, ShadowData uniform
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

// Precision distribution scale factor (must match vsm-sat-compute.wgsl).
const SAT_PRECISION_SCALE: f32 = 256.0;

// ── Attenuation ────────────────────────────────────────────────────
fn attenuation(distance: f32, light_range: f32) -> f32 {
    if light_range <= 0.0 {
        return 0.0;
    }
    let ratio = saturate(distance / light_range);
    let window = 1.0 - ratio * ratio;
    return window * window;
}

// ── Spot cone falloff ──────────────────────────────────────────────
fn spot_falloff(cos_angle: f32, inner_cone: f32, outer_cone: f32) -> f32 {
    return saturate((cos_angle - outer_cone) / max(inner_cone - outer_cone, 0.0001));
}

// ── VSM helpers ────────────────────────────────────────────────────

fn linstep(lo: f32, hi: f32, v: f32) -> f32 {
    return saturate((v - lo) / (hi - lo));
}

fn reduce_light_bleeding(p_max: f32, amount: f32) -> f32 {
    return linstep(amount, 1.0, p_max);
}

fn chebyshev_upper_bound(moments: vec2<f32>, depth: f32, min_variance: f32) -> f32 {
    let d = depth - moments.x;
    if d <= 0.0 {
        return 1.0;
    }
    let variance = max(moments.y - moments.x * moments.x, min_variance);
    let p_max = variance / (variance + d * d);
    return p_max;
}

// ── SAT lookup helpers ─────────────────────────────────────────────

// recombine_precision recombines precision-distributed RGBA into two moments.
// R + B/Scale = M₁, G + A/Scale = M₂.
fn recombine_precision(v: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(
        v.r + v.b / SAT_PRECISION_SCALE,
        v.g + v.a / SAT_PRECISION_SCALE,
    );
}

// sat_load performs a safe textureLoad on the SAT texture. Returns vec4(0) for
// coordinates outside the texture bounds (SAT convention: prefix sum of empty
// region is zero).
fn sat_load(coord: vec2<i32>) -> vec4<f32> {
    if coord.x < 0 || coord.y < 0 {
        return vec4<f32>(0.0);
    }
    let dims = vec2<i32>(textureDimensions(shadow_texture));
    let c = min(coord, dims - 1);
    return textureLoad(shadow_texture, c, 0);
}

// sat_area_query computes the average moments over an axis-aligned rectangle
// [min_coord, max_coord] (inclusive) using four SAT lookups and the standard
// inclusion-exclusion formula.
//
// Returns vec2(avg_M1, avg_M2) for the queried rectangle.
fn sat_area_query(min_coord: vec2<i32>, max_coord: vec2<i32>) -> vec2<f32> {
    let dims = vec2<i32>(textureDimensions(shadow_texture));
    let lo = clamp(min_coord, vec2<i32>(0), dims - 1);
    let hi = clamp(max_coord, vec2<i32>(0), dims - 1);

    let area = f32((hi.x - lo.x + 1) * (hi.y - lo.y + 1));
    if area <= 0.0 {
        return vec2<f32>(0.0);
    }

    // Inclusion-exclusion: sum = SAT[D] − SAT[B] − SAT[C] + SAT[A]
    // where A = (lo.x-1, lo.y-1), B = (hi.x, lo.y-1), C = (lo.x-1, hi.y), D = (hi.x, hi.y)
    let sum = sat_load(hi)
            - sat_load(vec2<i32>(lo.x - 1, hi.y))
            - sat_load(vec2<i32>(hi.x, lo.y - 1))
            + sat_load(vec2<i32>(lo.x - 1, lo.y - 1));

    return recombine_precision(sum / area);
}

// ── PCSS shadow sampling ───────────────────────────────────────────

// pcss_shadow projects the world position into light space, performs a SAT-backed
// blocker search, estimates the penumbra width using similar triangles, and
// evaluates Chebyshev's inequality over the variable-width filter region.
fn pcss_shadow(world_pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    // Normal-offset bias to reduce self-shadowing on surfaces nearly parallel to the light.
    let n_dot_l = dot(normal, -light_dir);
    let offset_scale = shadow_data.normal_bias * (1.0 - n_dot_l);
    let offset_pos = world_pos + normal * offset_scale;

    // Project into light clip space.
    let clip = shadow_data.light_vp * vec4<f32>(offset_pos, 1.0);
    let ndc = clip.xyz / clip.w;

    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    // Fragments outside the shadow map are fully lit.
    if shadow_uv.x < 0.0 || shadow_uv.x > 1.0 ||
       shadow_uv.y < 0.0 || shadow_uv.y > 1.0 {
        return 1.0;
    }

    // Compute linear depth matching the VSM depth pass.
    let view_pos = shadow_data.light_view * vec4<f32>(offset_pos, 1.0);
    let linear_depth = (-view_pos.z - shadow_data.shadow_near)
                     / (shadow_data.shadow_far - shadow_data.shadow_near);

    if linear_depth < 0.0 || linear_depth > 1.0 {
        return 1.0;
    }

    let dims = vec2<f32>(textureDimensions(shadow_texture));
    let texel_coord = vec2<i32>(shadow_uv * dims);

    // ── Step 1: Blocker search via SAT ─────────────────────────────
    // Convert light_size from world-space to shadow-map texels using the
    // orthographic half-extent and UV texel size:
    //   world_per_texel = 2 * half_extent * texel_size_uv
    let world_per_texel = 2.0 * shadow_data.shadow_half_extent * shadow_data.texel_size.x;
    let search_half = max(i32(shadow_data.light_size / world_per_texel * 0.5), 3);
    let search_min = texel_coord - vec2<i32>(search_half);
    let search_max = texel_coord + vec2<i32>(search_half);

    let search_moments = sat_area_query(search_min, search_max);
    let avg_blocker_depth = search_moments.x;

    // Smooth transition when the blocker depth approaches the receiver depth.
    // Instead of a hard early-out that causes granular flicker at shadow edges,
    // use a small epsilon and smoothstep to blend towards fully lit.
    let blocker_margin = 0.002;
    if avg_blocker_depth >= linear_depth - blocker_margin {
        return 1.0;
    }

    // Guard against near-zero blocker depth to prevent extreme penumbra widths.
    if avg_blocker_depth < 0.002 {
        return 1.0;
    }

    // ── Step 2: Penumbra estimation ────────────────────────────────
    // Similar-triangles formula from PCSS:
    //   w_penumbra = light_size × (d_receiver − d_blocker) / d_blocker
    let penumbra_world = shadow_data.light_size
                       * (linear_depth - avg_blocker_depth)
                       / avg_blocker_depth;

    // Convert world-space penumbra width to texel-space filter half-size.
    // Minimum 3 texels (7×7 region) to smooth out SAT precision noise.
    let filter_half = max(i32(penumbra_world / world_per_texel * 0.5), 3);
    // Clamp to a reasonable maximum to avoid querying the entire shadow map.
    let clamped_half = min(filter_half, i32(dims.x) / 4);

    // ── Step 3: Variable-width VSM lookup via SAT ──────────────────
    let filter_min = texel_coord - vec2<i32>(clamped_half);
    let filter_max = texel_coord + vec2<i32>(clamped_half);

    let filtered_moments = sat_area_query(filter_min, filter_max);

    // Chebyshev upper-bound test with light-bleeding reduction.
    var p_max = chebyshev_upper_bound(filtered_moments, linear_depth, shadow_data.min_variance);
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
            // Smooth fade at grazing angles to avoid hard stipple boundaries.
            let face_dot = dot(normal, normalize(-light.direction));
            let shadow_fade = smoothstep(0.0, 0.25, face_dot);
            if shadow_fade > 0.001 {
                let shadow = pcss_shadow(in.world_position, normal, light.direction);
                contribution *= mix(1.0, shadow, shadow_fade);
            }
        }

        total_light += contribution;
    }

    let final_color = albedo * total_light;
    return vec4<f32>(final_color, tex_color.a * in.color.a);
}
