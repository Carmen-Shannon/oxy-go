// Bone-driven particle position update — dispatched after the skeletal animator
// compute shader so that kinematic body particles track their parent bones.
//
// Each invocation handles one particle belonging to a single kinematic body.
// The particle's bone-local position (stored in local_position.xyz by
// AssignBoneIndices on the CPU) is transformed through the bone's current
// world matrix from the animator's scratch_matrices buffer. The resulting
// world-space position is written to world_position so the next frame's
// collision pipeline sees the animated pose.
//
// scratch_matrices indexing (from skeletal-compute.wgsl):
//   scratch_index = instance * (bone_count * 2) + slot * bone_count + bone
// We always use slot 0 (primary animation).
//
// Binding layout (5 bindings):
//   @binding(0) storage, read_write — Particle array     (write world_position, velocity, rel_position)
//   @binding(1) storage, read       — Body array          (read position for rel_position delta)
//   @binding(2) storage, read       — scratch_matrices    (bone hierarchy matrices from animator)
//   @binding(3) uniform             — BoneUpdateParams    (particle range, bone/instance counts)
//   @binding(4) storage, read       — model_data          (per-instance model matrices from animator)

//@oxy:include physics_particle
//@oxy:include physics_body
//@oxy:include model_data

struct BoneUpdateParams {
    particle_start:  u32,
    particle_count:  u32,
    bone_count:      u32,
    instance_index:  u32,
};

//@oxy:group 0 0 storage_read_write particles array<physics_particle>
//@oxy:group 0 1 storage_read bodies array<physics_body>
@group(0) @binding(2) var<storage, read> scratch_matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<uniform> params: BoneUpdateParams;
//@oxy:group 0 4 storage_read model_instances array<model_data>

//@oxy:inject BODY_IDX_MASK u32 body_idx_mask

// scratch_index computes the flat index into the scratch_matrices buffer.
// Uses slot 0 (primary animation) exclusively.
fn scratch_index(instance: u32, bone: u32) -> u32 {
    return instance * (params.bone_count * 2u) + bone;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let local_idx = global_id.x;
    if (local_idx >= params.particle_count) {
        return;
    }

    let global_idx = params.particle_start + local_idx;

    // Unpack bone index from upper 8 bits of local_position.w
    let packed = bitcast<u32>(particles[global_idx].local_position.w);
    let body_idx = packed & BODY_IDX_MASK;
    let bone_idx = (packed >> 24u) & 0xFFu;

    // Read the bone's current hierarchy matrix from the animator's scratch buffer.
    // scratch_matrices stores bone-to-model-space transforms (the bone hierarchy),
    // NOT world-space transforms. We must also multiply by the model matrix.
    let bone_matrix = scratch_matrices[scratch_index(params.instance_index, bone_idx)];

    // Read the instance's model matrix (object-to-world transform, includes scale).
    let model_matrix = model_instances[params.instance_index].model;

    // Transform bone-local position to world space:
    //   bone_local → model space (via bone_matrix) → world space (via model_matrix)
    let bone_local_pos = particles[global_idx].local_position.xyz;
    let world_pos = (model_matrix * bone_matrix * vec4<f32>(bone_local_pos, 1.0)).xyz;

    // Compute relative position from body center (used by compute_momenta for torque)
    let body_pos = bodies[body_idx].position.xyz;
    let rel_pos = world_pos - body_pos;

    // Write outputs — world_position.w = 1.0 marks this as a wall/static particle
    // for collision_reaction's force averaging logic.
    particles[global_idx].world_position = vec4<f32>(world_pos, 1.0);
    particles[global_idx].velocity       = vec4<f32>(0.0);
    particles[global_idx].rel_position   = vec4<f32>(rel_pos, 0.0);
}
