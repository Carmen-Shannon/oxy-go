// Particle value computation — Stage 1 of the GPU rigid body pipeline
//
// Runs one invocation per particle per frame. For each particle, reads the
// owning body's current state (position, quaternion, momentum) and derives
// the particle's world-space position, velocity, and relative offset from
// the center of mass.
//
// Reference: GPU Gems 3, Chapter 29 §29.2.3 — Computation of Particle Values
//   Eq. 18: r_i  = rotate(Q_j, r_i^0)            — current relative position
//   Eq. 19: x_i  = X_j + r_i                      — world-space position
//   Eq. 20: v_i  = V_j + W_j × r_i                — particle velocity
//
// Binding layout (3 bindings):
//   @binding(0) storage, read       — Body array   (read position, quaternion, momentum, inertia)
//   @binding(1) storage, read_write — Particle array (read local_position; write world_position, velocity, rel_position)
//   @binding(2) uniform             — PhysicsGlobals (particle_count for bounds check)

//@oxy:include physics_body
//@oxy:include physics_particle
//@oxy:include physics_globals

//@oxy:group 0 0 storage_read bodies array<physics_body>
//@oxy:group 0 1 storage_read_write particles array<physics_particle>
//@oxy:group 0 2 storage_uniform globals physics_globals

//@oxy:inject FLAG_ACTIVE u32 flag_active
//@oxy:inject FLAG_STATIC u32 flag_static
//@oxy:inject FLAG_KINEMATIC u32 flag_kinematic

// quat_rotate_vec3 rotates a vector by a unit quaternion.
// Quaternion layout: (x, y, z, w) where w is the scalar part.
// Uses the Rodrigues-style formula: v' = v + 2w(u × v) + 2(u × (u × v))
// where u = q.xyz and w = q.w.
fn quat_rotate_vec3(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let w = q.w;
    let uv = cross(u, v);
    return v + 2.0 * (w * uv + cross(u, uv));
}

// quat_to_mat3 converts a unit quaternion to a 3×3 rotation matrix.
// Returns column-major mat3x3 matching WGSL convention.
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;

    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;

    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;

    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz,         xz - wy),         // column 0
        vec3<f32>(xy - wz,          1.0 - (xx + zz), yz + wx),         // column 1
        vec3<f32>(xz + wy,          yz - wx,          1.0 - (xx + yy)) // column 2
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= globals.particle_count) {
        return;
    }

    // Read the particle's body-local rest position and owning body index.
    // The body index is packed into the low 24 bits of local_position.w via
    // bitcast<u32>; the upper 8 bits store the bone index for kinematic bodies.
    let local_pos = particles[idx].local_position.xyz;
    let packed_idx = bitcast<u32>(particles[idx].local_position.w);
    let body_idx = packed_idx & 0xFFFFFFu;

    // Read body state
    let body = bodies[body_idx];

    // Skip particles belonging to deactivated bodies (flags == 0). Mark them
    // with world_position.w = -1.0 as a sentinel so aabb_reduce, grid_insert,
    // and collision_reaction skip them. Without this, dead particles would
    // poison the AABB and break the spatial grid for all live particles.
    if (body.flags == 0u) {
        particles[idx].world_position = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        particles[idx].velocity       = vec4<f32>(0.0);
        particles[idx].rel_position   = vec4<f32>(0.0);
        particles[idx].force          = vec4<f32>(0.0);
        return;
    }

    // Skip kinematic particles — their world positions are set by the
    // bone_particle_update shader after the animator compute dispatch.
    if ((body.flags & FLAG_KINEMATIC) != 0u) {
        return;
    }

    let body_pos  = body.position.xyz;
    let body_quat = body.quaternion;

    // Eq. 18: rotate the rest-pose local offset by the body's current quaternion
    // r_i = rotate(Q_j, r_i^0)
    let rel_pos = quat_rotate_vec3(body_quat, local_pos);

    // Eq. 19: world position = body center of mass + rotated offset
    // x_i = X_j + r_i
    let world_pos = body_pos + rel_pos;

    // Derive linear and angular velocity from momentum (§29.1.1 Eq. 2, §29.1.2 Eq. 5-6)
    var lin_vel = vec3<f32>(0.0);
    var ang_vel = vec3<f32>(0.0);

    let is_static = (body.flags & FLAG_STATIC) != 0u;
    if (!is_static) {
        // v = P * inverseMass
        lin_vel = body.linear_momentum.xyz * body.inverse_mass;

        // ω = I(t)^-1 * L
        // I(t)^-1 = R * I_body^-1 * R^T   (Eq. 6)
        let r_mat = quat_to_mat3(body_quat);
        let inv_i_world = r_mat * body.inv_inertia_tensor_body * transpose(r_mat);
        ang_vel = inv_i_world * body.angular_momentum.xyz;
    }

    // Eq. 20: particle velocity = linear velocity + angular velocity × relative position
    // v_i = V_j + W_j × r_i
    let velocity = lin_vel + cross(ang_vel, rel_pos);

    // Write outputs
    // world_position.w encodes static flag: 1.0 for static body particles,
    // 0.0 for dynamic. Used by collision_reaction to separate wall contacts
    // (full force) from fluid-fluid contacts (averaged).
    var wp_w = 0.0;
    if ((body.flags & FLAG_STATIC) != 0u) {
        wp_w = 1.0;
    }
    particles[idx].world_position = vec4<f32>(world_pos, wp_w);
    particles[idx].velocity       = vec4<f32>(velocity, 0.0);
    particles[idx].rel_position   = vec4<f32>(rel_pos, 0.0);
}
