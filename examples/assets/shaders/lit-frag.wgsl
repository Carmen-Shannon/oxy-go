// Lit fragment shader (Forward+ Blinn-Phong with material maps, normal mapping, and shadow mapping)
//
// Uses the Forward+ (tiled forward) rendering technique: a light culling compute
// shader assigns lights to 16×16 pixel screen tiles, and this fragment shader only
// evaluates the lights relevant to each fragment's tile. Samples diffuse, normal,
// and metallic-roughness textures from the material bind group. Uses the camera
// position from the camera uniform for specular highlights. Constructs a TBN
// matrix from interpolated world-space tangent and normal vectors to transform
// normal map samples from tangent space to world space. Directional lights that
// cast shadows are attenuated by a 3×3 PCF shadow map lookup.
//
// Bind group layout:
//   @group(0) camera     — CameraUniform (view_proj + camera_position)
//   @group(2) material   — diffuse texture + sampler, normal map, metallic-roughness map
//   @group(3) lights     — LightHeader + Light array (storage buffer)
//   @group(4) shadow     — shadow depth texture, comparison sampler, ShadowData uniform
//   @group(5) tiles      — TileUniforms + per-tile light counts + per-tile light indices

// ── Fragment input (from vertex shader) ────────────────────────────
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
@group(4) @binding(0) var shadow_texture: texture_depth_2d;
@group(4) @binding(1) var shadow_sampler: sampler_comparison;
//@oxy:group 4 2 storage_uniform shadow_data shadow_data

//@oxy:group 5 0 storage_uniform tile_uniforms tile_uniforms
//@oxy:provider 5 1 tiles
@group(5) @binding(1) var<storage, read> tile_counts: array<u32>;
@group(5) @binding(2) var<storage, read> tile_indices: array<u32>;

// ── Constants ──────────────────────────────────────────────────────
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT:       u32 = 1u;
const LIGHT_TYPE_SPOT:        u32 = 2u;

const SPECULAR_STRENGTH: f32 = 0.5;  // base specular contribution scale (dielectric)

// ── Attenuation ────────────────────────────────────────────────────
// Smooth range-normalized attenuation. Returns 1.0 at distance 0 and
// falls smoothly to 0.0 at light_range using a squared windowing
// function. Avoids the raw 1/d² approach which produces vanishingly
// small values at typical scene distances.
fn attenuation(distance: f32, light_range: f32) -> f32 {
    if light_range <= 0.0 {
        return 0.0;
    }
    let ratio = saturate(distance / light_range);
    let window = 1.0 - ratio * ratio;
    return window * window;
}

// ── Spot cone falloff ──────────────────────────────────────────────
// Smooth falloff between inner and outer cone angles.
fn spot_falloff(cos_angle: f32, inner_cone: f32, outer_cone: f32) -> f32 {
    return saturate((cos_angle - outer_cone) / max(inner_cone - outer_cone, 0.0001));
}

// ── Shadow sampling ────────────────────────────────────────────────
// 3×3 PCF (Percentage-Closer Filtering) shadow map lookup with normal-
// offset bias. The world position is shifted along the surface normal
// before projecting into light clip space. The offset is largest when
// the surface is nearly parallel to the light direction (grazing angles),
// which is exactly where concave-geometry self-shadowing artifacts are
// worst. A small constant depth bias is applied on top for residual acne.
fn sample_shadow(world_pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    // Offset the world position along the surface normal to reduce shadow acne
    // on surfaces nearly parallel to the light direction.
    let n_dot_l = dot(normal, -light_dir);
    let offset_scale = shadow_data.normal_bias * (1.0 - n_dot_l);
    let offset_pos = world_pos + normal * offset_scale;

    let clip = shadow_data.light_vp * vec4<f32>(offset_pos, 1.0);
    let ndc = clip.xyz / clip.w;

    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    let depth = ndc.z;

    // Fragments outside the shadow map receive no shadow (fully lit).
    if shadow_uv.x < 0.0 || shadow_uv.x > 1.0 ||
       shadow_uv.y < 0.0 || shadow_uv.y > 1.0 ||
       depth < 0.0 || depth > 1.0 {
        return 1.0;
    }

    // 3×3 PCF (percentage-closer filtering) for soft shadow edges.
    let bias = shadow_data.bias;
    var total = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * shadow_data.texel_size;
            total += textureSampleCompare(
                shadow_texture,
                shadow_sampler,
                shadow_uv + offset,
                depth - bias,
            );
        }
    }
    return total / 9.0;
}

// ── Per-light contribution ─────────────────────────────────────────
// Computes diffuse + specular for a single light using Blinn-Phong.
// Roughness modulates the specular exponent: shininess = mix(4, 128, (1-roughness)^2).
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
            // Directional: light direction points FROM the light toward the scene,
            // so we negate it to get the direction toward the light.
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

            // Spot cone attenuation
            let cos_angle = dot(-light_dir, normalize(light.direction));
            atten *= spot_falloff(cos_angle, light.inner_cone, light.outer_cone);
        }
        default: {
            return vec3<f32>(0.0);
        }
    }

    // Diffuse (Lambertian)
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse = n_dot_l * light.color * light.intensity;

    // Specular (Blinn-Phong) — roughness modulates the exponent.
    // Smooth surfaces (roughness≈0) get a tight highlight, rough surfaces a broad one.
    // Gated on n_dot_l > 0: when the light is behind the surface there should be
    // no specular highlight at all, preventing shadow bleed-through artifacts.
    let shininess = mix(4.0, 128.0, pow(1.0 - roughness, 2.0));
    let spec_strength = mix(SPECULAR_STRENGTH, 1.0, metallic);
    let half_dir = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let specular = select(vec3<f32>(0.0), spec_strength * pow(n_dot_h, shininess) * light.color * light.intensity, n_dot_l > 0.0);

    return (diffuse + specular) * atten;
}

// ── Entry point ────────────────────────────────────────────────────
@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    // Sample diffuse texture
    let tex_color = textureSample(diffuse_texture, diffuse_sampler, in.uv);

    // Discard fully transparent fragments
    if tex_color.a < 0.01 {
        discard;
    }

    // Surface albedo: texture × vertex color
    let albedo = tex_color.rgb * in.color.rgb;

    // Sample normal map and transform from tangent space to world space via TBN matrix.
    // The tangent and bitangent are derived from the vertex shader's world_tangent output,
    // where W stores the handedness sign (±1) for correct bitangent orientation.
    let normal_sample = textureSample(normal_texture, normal_sampler, in.uv).rgb;
    let mapped_normal = normal_sample * 2.0 - 1.0; // [0,1] → [-1,1]

    // Flip the geometric normal for back-facing fragments so that surfaces
    // viewed from behind are correctly shaded (e.g. underside of a canopy).
    var N = normalize(in.world_normal);
    if !in.front_facing {
        N = -N;
    }
    let T = normalize(in.world_tangent.xyz);
    let B = cross(N, T) * in.world_tangent.w; // handedness from glTF/MikkTSpace
    let TBN = mat3x3<f32>(T, B, N);
    var normal = normalize(TBN * mapped_normal);

    // Sample metallic-roughness (glTF packing: R=unused, G=roughness, B=metallic)
    let mr_sample = textureSample(metallic_roughness_texture, metallic_roughness_sampler, in.uv);
    let roughness = mr_sample.g;
    let metallic = mr_sample.b;

    // View direction (fragment → camera)
    let view_dir = normalize(camera.camera_position - in.world_position);

    // ── Forward+ tiled light loop ──────────────────────────────────
    // Determine which screen tile this fragment belongs to.
    let frag_coord = vec2<u32>(in.position.xy);
    let tile_x = frag_coord.x / 16u;
    let tile_y = frag_coord.y / 16u;
    let tile_index = tile_y * tile_uniforms.tile_count_x + tile_x;

    // Number of lights affecting this tile (written by the cull compute shader).
    let num_tile_lights = tile_counts[tile_index];

    // Base offset into the flat light-index array for this tile.
    let tile_base = tile_index * tile_uniforms.max_lights_per_tile;

    // Accumulate lighting from all lights in this tile.
    var total_light = light_header.ambient_color;
    for (var i = 0u; i < num_tile_lights; i++) {
        let light_idx = tile_indices[tile_base + i];
        let light = lights[light_idx];

        var contribution = evaluate_light(light, in.world_position, normal, view_dir, roughness, metallic);

        // Apply shadow map attenuation for shadow-casting directional lights.
        // Skip shadow sampling when the surface barely faces the light (N·L < threshold).
        // At grazing angles the diffuse contribution is negligible and the shadow map
        // projection can produce false silhouettes from geometry on the other side.
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
