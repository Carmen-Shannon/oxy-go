// Skeletal animation compute shader with GPU frustum culling
//
// Runs one invocation per instance per frame. For each instance: samples keyframes
// for the active animation clip, builds the bone hierarchy (parent-to-child), optionally
// blends between two clips, tests the instance against the frustum, and compacts visible
// instances into dense output arrays for DrawIndexedIndirect.
//
// Binding layout (8 bindings: 1 uniform + 7 storage):
//   @binding(0) uniform: globals (AnimationGlobals with frustum planes + packed-buffer offsets)
//   @binding(1) rw:      instance_data (per-instance animation state)
//   @binding(2) read:    bone_data (shared skeleton)
//   @binding(3) read:    anim_packed (flat u32 array: clips | channels | keyframes)
//   @binding(4) rw:      output_transforms (compacted per-instance output, shared with vertex shader)
//   @binding(5) rw:      scratch_matrices (full-sized bone matrix workspace for hierarchy)
//   @binding(6) read:    model_data (per-instance model matrices from CPU)
//   @binding(7) rw:      indirect_args (DrawIndexedIndirect arguments)

// Maximum number of bones supported per skeleton. Must match the vertex shader's
// InstanceData.bone_matrices array size so the output stride is consistent.
//@oxy:inject MAX_BONES u32 max_bones

//@oxy:include skeletal_animation_data
//@oxy:include frustum_plane
//@oxy:include animation_globals
//@oxy:include bone_info
//@oxy:include indirect_args
//@oxy:include model_data

//@oxy:group 0 0 storage_uniform globals animation_globals
//@oxy:group 0 1 storage_read_write instance_data array<skeletal_animation_data>
//@oxy:group 0 2 storage_read bone_data array<bone_info>
//@oxy:provider 0 3 animator_packed
@group(0) @binding(3) var<storage, read> anim_packed: array<u32>;
//@oxy:provider 0 4 animator_output
@group(0) @binding(4) var<storage, read_write> output_transforms: array<f32>;
//@oxy:provider 0 5 animator_scratch
@group(0) @binding(5) var<storage, read_write> scratch_matrices: array<mat4x4<f32>>;
//@oxy:group 0 6 storage_read model_data array<model_data>
//@oxy:group 0 7 storage_read_write indirect_args indirect_args
//@oxy:provider 1 0 animator_hiz hiz_texture
@group(1) @binding(0) var hiz_texture: texture_2d<f32>;
//@oxy:provider 1 1 animator_max_hiz hiz_max_texture
@group(1) @binding(1) var hiz_max_texture: texture_2d<f32>;

fn get_clip_duration(clip_idx: u32) -> f32 {
    return bitcast<f32>(anim_packed[clip_idx * 4u + 0u]);
}

fn get_clip_channel_offset(clip_idx: u32) -> u32 {
    return anim_packed[clip_idx * 4u + 2u];
}

fn get_clip_channel_count(clip_idx: u32) -> u32 {
    return anim_packed[clip_idx * 4u + 3u];
}

fn get_channel_bone_index(ch_idx: u32) -> u32 {
    let base = globals.channel_data_offset + ch_idx * 8u;
    return anim_packed[base + 0u];
}

fn get_channel_pos_key_offset(ch_idx: u32) -> u32 {
    let base = globals.channel_data_offset + ch_idx * 8u;
    return anim_packed[base + 1u];
}

fn get_channel_pos_key_count(ch_idx: u32) -> u32 {
    let base = globals.channel_data_offset + ch_idx * 8u;
    return anim_packed[base + 2u];
}

fn get_channel_rot_key_offset(ch_idx: u32) -> u32 {
    let base = globals.channel_data_offset + ch_idx * 8u;
    return anim_packed[base + 3u];
}

fn get_channel_rot_key_count(ch_idx: u32) -> u32 {
    let base = globals.channel_data_offset + ch_idx * 8u;
    return anim_packed[base + 4u];
}

fn get_channel_scale_key_offset(ch_idx: u32) -> u32 {
    let base = globals.channel_data_offset + ch_idx * 8u;
    return anim_packed[base + 5u];
}

fn get_channel_scale_key_count(ch_idx: u32) -> u32 {
    let base = globals.channel_data_offset + ch_idx * 8u;
    return anim_packed[base + 6u];
}

// Keyframe layout (16 u32 per keyframe):
// [time, pad, pad, pad, tx, ty, tz, pad, rx, ry, rz, rw, sx, sy, sz, pad]
fn get_keyframe_time(kf_idx: u32) -> f32 {
    let base = globals.keyframe_data_offset + kf_idx * 16u;
    return bitcast<f32>(anim_packed[base + 0u]);
}

fn get_keyframe_translation(kf_idx: u32) -> vec3<f32> {
    let base = globals.keyframe_data_offset + kf_idx * 16u;
    return vec3<f32>(
        bitcast<f32>(anim_packed[base + 4u]),
        bitcast<f32>(anim_packed[base + 5u]),
        bitcast<f32>(anim_packed[base + 6u])
    );
}

fn get_keyframe_rotation(kf_idx: u32) -> vec4<f32> {
    let base = globals.keyframe_data_offset + kf_idx * 16u;
    return vec4<f32>(
        bitcast<f32>(anim_packed[base + 8u]),
        bitcast<f32>(anim_packed[base + 9u]),
        bitcast<f32>(anim_packed[base + 10u]),
        bitcast<f32>(anim_packed[base + 11u])
    );
}

fn get_keyframe_scale(kf_idx: u32) -> vec3<f32> {
    let base = globals.keyframe_data_offset + kf_idx * 16u;
    return vec3<f32>(
        bitcast<f32>(anim_packed[base + 12u]),
        bitcast<f32>(anim_packed[base + 13u]),
        bitcast<f32>(anim_packed[base + 14u])
    );
}

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

fn quat_to_mat(q: vec4<f32>) -> mat4x4<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;
    return mat4x4<f32>(
        vec4<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 0.0),
        vec4<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 0.0),
        vec4<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );
}

fn slerp(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    var b_adj = b;
    var dot_val = dot(a, b);
    if dot_val < 0.0 {
        b_adj = -b;
        dot_val = -dot_val;
    }
    if dot_val > 0.9995 {
        return normalize(mix(a, b_adj, t));
    }
    let theta_0 = acos(dot_val);
    let theta = theta_0 * t;
    let sin_theta = sin(theta);
    let sin_theta_0 = sin(theta_0);
    let s0 = cos(theta) - dot_val * sin_theta / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;
    return normalize(s0 * a + s1 * b_adj);
}

fn build_trs(translation: vec3<f32>, rotation: vec4<f32>, scale: vec3<f32>) -> mat4x4<f32> {
    let rot_mat = quat_to_mat(rotation);
    var result = rot_mat;
    result[0] = result[0] * scale.x;
    result[1] = result[1] * scale.y;
    result[2] = result[2] * scale.z;
    result[3] = vec4<f32>(translation, 1.0);
    return result;
}

fn sample_vec3_keyframes(time: f32, offset: u32, count: u32, is_scale: bool) -> vec3<f32> {
    if count == 0u {
        if is_scale { return vec3<f32>(1.0, 1.0, 1.0); }
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    if count == 1u {
        if is_scale { return get_keyframe_scale(offset); }
        return get_keyframe_translation(offset);
    }

    var key0_idx = offset;
    var key1_idx = offset;
    for (var i = 0u; i < count - 1u; i = i + 1u) {
        if get_keyframe_time(offset + i + 1u) > time {
            key0_idx = offset + i;
            key1_idx = offset + i + 1u;
            break;
        }
        key0_idx = offset + count - 1u;
        key1_idx = key0_idx;
    }

    if key0_idx == key1_idx {
        if is_scale { return get_keyframe_scale(key0_idx); }
        return get_keyframe_translation(key0_idx);
    }

    let t0 = get_keyframe_time(key0_idx);
    let t1 = get_keyframe_time(key1_idx);
    let t = (time - t0) / (t1 - t0);

    if is_scale {
        return mix(get_keyframe_scale(key0_idx), get_keyframe_scale(key1_idx), t);
    }
    return mix(get_keyframe_translation(key0_idx), get_keyframe_translation(key1_idx), t);
}

fn sample_quat_keyframes(time: f32, offset: u32, count: u32) -> vec4<f32> {
    if count == 0u { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    if count == 1u { return get_keyframe_rotation(offset); }

    var key0_idx = offset;
    var key1_idx = offset;
    for (var i = 0u; i < count - 1u; i = i + 1u) {
        if get_keyframe_time(offset + i + 1u) > time {
            key0_idx = offset + i;
            key1_idx = offset + i + 1u;
            break;
        }
        key0_idx = offset + count - 1u;
        key1_idx = key0_idx;
    }

    if key0_idx == key1_idx { return get_keyframe_rotation(key0_idx); }

    let t0 = get_keyframe_time(key0_idx);
    let t1 = get_keyframe_time(key1_idx);
    let t = (time - t0) / (t1 - t0);
    return slerp(get_keyframe_rotation(key0_idx), get_keyframe_rotation(key1_idx), t);
}

fn sample_bone_transform(clip_idx: u32, bone_idx: u32, time: f32) -> mat4x4<f32> {
    let clip_duration = get_clip_duration(clip_idx);
    let clip_channel_offset = get_clip_channel_offset(clip_idx);
    let clip_channel_count = get_clip_channel_count(clip_idx);
    let bone = bone_data[bone_idx];

    var local_translation = bone.local_translation;
    var local_rotation = bone.local_rotation;
    var local_scale = bone.local_scale;

    var current_time = time;
    if current_time > clip_duration && clip_duration > 0.0 {
        current_time = current_time % clip_duration;
    }

    for (var ch_idx = 0u; ch_idx < clip_channel_count; ch_idx = ch_idx + 1u) {
        let abs_ch = clip_channel_offset + ch_idx;
        if get_channel_bone_index(abs_ch) == bone_idx {
            let pos_count = get_channel_pos_key_count(abs_ch);
            if pos_count > 0u {
                local_translation = sample_vec3_keyframes(
                    current_time, get_channel_pos_key_offset(abs_ch), pos_count, false
                );
            }
            let rot_count = get_channel_rot_key_count(abs_ch);
            if rot_count > 0u {
                local_rotation = sample_quat_keyframes(
                    current_time, get_channel_rot_key_offset(abs_ch), rot_count
                );
            }
            let scale_count = get_channel_scale_key_count(abs_ch);
            if scale_count > 0u {
                local_scale = sample_vec3_keyframes(
                    current_time, get_channel_scale_key_offset(abs_ch), scale_count, true
                );
            }
            break;
        }
    }

    return build_trs(local_translation, local_rotation, local_scale);
}

fn scratch_index(instance_idx: u32, slot: u32, bone_idx: u32) -> u32 {
    return instance_idx * (globals.bone_count * 2u) + slot * globals.bone_count + bone_idx;
}

fn compute_bone_world_matrices(clip_idx: u32, time: f32, instance_idx: u32, slot: u32) {
    for (var bone_idx = 0u; bone_idx < globals.bone_count; bone_idx = bone_idx + 1u) {
        let local_matrix = sample_bone_transform(clip_idx, bone_idx, time);
        let out_idx = scratch_index(instance_idx, slot, bone_idx);
        let parent_idx = bone_data[bone_idx].parent_index;
        if parent_idx < 0 {
            scratch_matrices[out_idx] = local_matrix;
        } else {
            let parent_out_idx = scratch_index(instance_idx, slot, u32(parent_idx));
            scratch_matrices[out_idx] = scratch_matrices[parent_out_idx] * local_matrix;
        }
    }
}

fn blend_matrices(mat_a: mat4x4<f32>, mat_b: mat4x4<f32>, weight: f32) -> mat4x4<f32> {
    let trans_a = mat_a[3].xyz;
    let trans_b = mat_b[3].xyz;
    let blended_trans = mix(trans_a, trans_b, weight);
    let rot_a = mat3x3<f32>(normalize(mat_a[0].xyz), normalize(mat_a[1].xyz), normalize(mat_a[2].xyz));
    let rot_b = mat3x3<f32>(normalize(mat_b[0].xyz), normalize(mat_b[1].xyz), normalize(mat_b[2].xyz));
    let blended_rot = mat3x3<f32>(
        normalize(mix(rot_a[0], rot_b[0], weight)),
        normalize(mix(rot_a[1], rot_b[1], weight)),
        normalize(mix(rot_a[2], rot_b[2], weight))
    );
    let scale_a = vec3<f32>(length(mat_a[0].xyz), length(mat_a[1].xyz), length(mat_a[2].xyz));
    let scale_b = vec3<f32>(length(mat_b[0].xyz), length(mat_b[1].xyz), length(mat_b[2].xyz));
    let blended_scale = mix(scale_a, scale_b, weight);
    return mat4x4<f32>(
        vec4<f32>(blended_rot[0] * blended_scale.x, 0.0),
        vec4<f32>(blended_rot[1] * blended_scale.y, 0.0),
        vec4<f32>(blended_rot[2] * blended_scale.z, 0.0),
        vec4<f32>(blended_trans, 1.0)
    );
}

fn is_visible(pos: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = globals.planes[i];
        let dist = dot(plane.normal, pos) + plane.distance;
        if (dist < -radius) {
            return false;
        }
    }
    return true;
}

// Writes a mat4x4 column-major into the flat output buffer at the given float offset.
fn write_mat4(base: u32, m: mat4x4<f32>) {
    output_transforms[base +  0u] = m[0].x; output_transforms[base +  1u] = m[0].y;
    output_transforms[base +  2u] = m[0].z; output_transforms[base +  3u] = m[0].w;
    output_transforms[base +  4u] = m[1].x; output_transforms[base +  5u] = m[1].y;
    output_transforms[base +  6u] = m[1].z; output_transforms[base +  7u] = m[1].w;
    output_transforms[base +  8u] = m[2].x; output_transforms[base +  9u] = m[2].y;
    output_transforms[base + 10u] = m[2].z; output_transforms[base + 11u] = m[2].w;
    output_transforms[base + 12u] = m[3].x; output_transforms[base + 13u] = m[3].y;
    output_transforms[base + 14u] = m[3].z; output_transforms[base + 15u] = m[3].w;
}

fn write_instance_flags(base: u32, flags: u32) {
    output_transforms[base + 0u] = bitcast<f32>(flags);
    output_transforms[base + 1u] = 0.0;
    output_transforms[base + 2u] = 0.0;
    output_transforms[base + 3u] = 0.0;
}

// Returns true if the model-space AABB (transformed to world space by model_mat) is
// fully occluded by the previous frame's Hi-Z depth pyramid. Returns false (never occludes)
// when globals.hiz_mip_count == 0 (Hi-Z not yet initialized).
//
// Samples mip 0 (full resolution) at the AABB footprint center: one pixel, no
// contamination from adjacent geometry at coarser mip levels.
fn is_occluded(model_mat: mat4x4<f32>) -> bool {
    if (globals.hiz_mip_count == 0u) {
        return false;
    }

    let bmin = globals.bounding_min;
    let bmax = globals.bounding_max;
    let vp = globals.view_proj;

    var min_z = 1.0;
    var min_u = 1.0; var max_u = 0.0;
    var min_v = 1.0; var max_v = 0.0;
    var any_on_screen = false;

    for (var ci = 0u; ci < 8u; ci++) {
        let px = select(bmin.x, bmax.x, (ci & 1u) != 0u);
        let py = select(bmin.y, bmax.y, (ci & 2u) != 0u);
        let pz = select(bmin.z, bmax.z, (ci & 4u) != 0u);
        let world = model_mat * vec4<f32>(px, py, pz, 1.0);
        let clip = vp * world;
        if (clip.w  <= 0.0) { return false; }
        let ndc = clip.xyz / clip.w;
        min_z = min(min_z, ndc.z);
        if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) { continue; }
        let u = ndc.x * 0.5 + 0.5;
        let v = 1.0 - (ndc.y * 0.5 + 0.5);
        min_u = min(min_u, u); max_u = max(max_u, u);
        min_v = min(min_v, v); max_v = max(max_v, v);
        any_on_screen = true;
    }

    if (!any_on_screen) { return false; }

    let foot_w = (max_u - min_u) * f32(globals.screen_width);
    let foot_h = (max_v - min_v) * f32(globals.screen_height);
    let foot_max = max(max(foot_w, foot_h), 1.0);
    let mip_level = i32(min(u32(floor(log2(foot_max))), globals.hiz_mip_count - 1u));

    let mip_dims = vec2<i32>(textureDimensions(hiz_max_texture, mip_level));
    let tc_min = clamp(vec2<i32>(vec2<f32>(min_u, min_v) * vec2<f32>(mip_dims)), vec2<i32>(0), mip_dims - vec2<i32>(1));
    let tc_max = clamp(vec2<i32>(vec2<f32>(max_u, max_v) * vec2<f32>(mip_dims)), vec2<i32>(0), mip_dims - vec2<i32>(1));
    let s0 = textureLoad(hiz_max_texture, vec2<i32>(tc_min.x, tc_min.y), mip_level).r;
    let s1 = textureLoad(hiz_max_texture, vec2<i32>(tc_max.x, tc_min.y), mip_level).r;
    let s2 = textureLoad(hiz_max_texture, vec2<i32>(tc_min.x, tc_max.y), mip_level).r;
    let s3 = textureLoad(hiz_max_texture, vec2<i32>(tc_max.x, tc_max.y), mip_level).r;
    let max_hiz = max(max(s0, s1), max(s2, s3));
    return min_z > max_hiz + 0.001;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let instance_idx = global_id.x;
    if instance_idx >= globals.instance_count {
        return;
    }

    let anim = instance_data[instance_idx];

    // Compute primary animation bone matrices into scratch slot 0
    compute_bone_world_matrices(anim.animation_index, anim.animation_time, instance_idx, 0u);

    // Check if we need to blend with secondary animation
    let is_blending = anim.blend_weight > 0.0 && anim.blend_weight < 1.0;
    if is_blending {
        compute_bone_world_matrices(anim.secondary_anim_index, anim.secondary_anim_time, instance_idx, 1u);
    }

    // Read world position from the model matrix (column 3 = translation)
    let model_matrix = model_data[instance_idx].model;
    let world_pos = model_matrix[3].xyz;

    // Frustum cull — derive per-instance scale from model matrix column
    // lengths so that scaled instances are not incorrectly culled.
    let sx = length(model_matrix[0].xyz);
    let sy = length(model_matrix[1].xyz);
    let sz = length(model_matrix[2].xyz);
    let max_scale = max(sx, max(sy, sz));
    if (!is_visible(world_pos, globals.bounding_radius * max_scale)) {
        return;
    }
    if (!is_occluded(model_matrix)) {
        // Visible — atomically claim an output slot
        let out_slot = atomicAdd(&indirect_args.instance_count, 1u);

        // Per-instance output stride in floats: 1 model matrix (16) + 1 flag vec4 (4) + MAX_BONES bone matrices (16 each).
        let stride = 20u + MAX_BONES * 16u;
        let out_base = out_slot * stride;

        // Write compacted model matrix first
        write_mat4(out_base, model_matrix);
        write_instance_flags(out_base + 16u, anim.instance_flags);

        // Write compacted bone skinning matrices (world × inverse_bind)
        for (var bone_idx = 0u; bone_idx < globals.bone_count; bone_idx = bone_idx + 1u) {
            var world_matrix: mat4x4<f32>;
            if is_blending {
                world_matrix = blend_matrices(
                    scratch_matrices[scratch_index(instance_idx, 0u, bone_idx)],
                    scratch_matrices[scratch_index(instance_idx, 1u, bone_idx)],
                    anim.blend_weight
                );
            } else {
                world_matrix = scratch_matrices[scratch_index(instance_idx, 0u, bone_idx)];
            }
            let final_matrix = world_matrix * bone_data[bone_idx].inverse_bind_matrix;
            write_mat4(out_base + 20u + bone_idx * 16u, final_matrix);
        }

        // Pad remaining bone slots with identity so the vertex shader stride is consistent
        for (var b = globals.bone_count; b < MAX_BONES; b = b + 1u) {
            let off = out_base + 20u + b * 16u;
            output_transforms[off +  0u] = 1.0; output_transforms[off +  1u] = 0.0;
            output_transforms[off +  2u] = 0.0; output_transforms[off +  3u] = 0.0;
            output_transforms[off +  4u] = 0.0; output_transforms[off +  5u] = 1.0;
            output_transforms[off +  6u] = 0.0; output_transforms[off +  7u] = 0.0;
            output_transforms[off +  8u] = 0.0; output_transforms[off +  9u] = 0.0;
            output_transforms[off + 10u] = 1.0; output_transforms[off + 11u] = 0.0;
            output_transforms[off + 12u] = 0.0; output_transforms[off + 13u] = 0.0;
            output_transforms[off + 14u] = 0.0; output_transforms[off + 15u] = 1.0;
        }
    }
}
