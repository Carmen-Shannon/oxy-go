// Integration — Stage 5 of the GPU rigid body pipeline
//
// Runs one invocation per rigid body. Derives velocity from updated momenta,
// then integrates position and quaternion forward by dt using semi-implicit
// Euler. Normalizes the quaternion to prevent drift.
//
// Reference: GPU Gems 3, Chapter 29 §29.2.7 — Computation of Position and Quaternion
//   Eq. 2: v = P / m                              — linear velocity from momentum
//   Eq. 3: x += v * dt                            — position integration
//   Eq. 5: ω = I(t)^-1 * L                        — angular velocity from angular momentum
//   Eq. 6: I(t)^-1 = R * I_body^-1 * R^T          — world-space inverse inertia tensor
//   Eq. 7: dq = (dt/2) * [ω, 0] * q               — quaternion derivative
//   Eq. 8: q' = normalize(q + dq)                  — quaternion update
//
// Binding layout (2 bindings):
//   @binding(0) storage, read_write — Body array     (read momenta, inertia; write position, quaternion)
//   @binding(1) uniform             — PhysicsGlobals (delta_time, body_count)

//@oxy:include physics_body
//@oxy:include physics_globals

//@oxy:group 0 0 storage_read_write bodies array<physics_body>
//@oxy:group 0 1 storage_uniform globals physics_globals

const FLAG_STATIC: u32    = 2u;
const FLAG_KINEMATIC: u32 = 4u;

// Per-second velocity retention targets. These define how much momentum is
// preserved after one full second of simulation, independent of timestep.
// The per-substep damping factor is derived via pow(target, dt) so that
// changing fixedDt does not alter the effective damping rate.
//   LINEAR:  0.403 → ~60% energy loss per second (viscous drag)
//   ANGULAR: 0.296 → ~70% spin loss per second (rotational drag)
// Original per-step values at dt=1/240: linear=0.99622, angular=0.99494.
const LINEAR_DAMP_PER_SEC: f32  = 0.403;
const ANGULAR_DAMP_PER_SEC: f32 = 0.296;

// CFL velocity limiter — maximum displacement per substep is half a
// particle radius, computed dynamically from globals so any change to
// particle_diameter or delta_time is automatically safe.
// V_MAX = particle_diameter * 0.5 / dt  (set in main after dt is known)

// quat_to_mat3 converts a unit quaternion to a 3×3 rotation matrix.
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
        vec3<f32>(1.0 - (yy + zz), xy + wz,         xz - wy),
        vec3<f32>(xy - wz,          1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy,          yz - wx,          1.0 - (xx + yy))
    );
}

// quat_mul multiplies two quaternions (Hamilton product).
// Layout: (x, y, z, w) where w is the scalar part.
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= globals.body_count) {
        return;
    }

    let body = bodies[idx];
    let flags = body.flags;

    // Static and kinematic bodies do not integrate
    if ((flags & (FLAG_STATIC | FLAG_KINEMATIC)) != 0u) {
        return;
    }

    let dt = globals.delta_time;

    // Compute per-substep damping from per-second retention targets.
    // pow(target, dt) yields the correct multiplicative factor for any dt.
    let linear_damping  = pow(LINEAR_DAMP_PER_SEC, dt);
    let angular_damping = pow(ANGULAR_DAMP_PER_SEC, dt);

    // Apply velocity damping to momenta (models air resistance / energy loss).
    // This must happen before deriving velocity so the damped values feed into
    // the position/quaternion integration.
    let damped_linear  = body.linear_momentum.xyz  * linear_damping;
    let damped_angular = body.angular_momentum.xyz * angular_damping;
    bodies[idx].linear_momentum  = vec4<f32>(damped_linear, 0.0);
    bodies[idx].angular_momentum = vec4<f32>(damped_angular, 0.0);

    // Eq. 2: linear velocity from (damped) momentum
    var v = damped_linear * body.inverse_mass;

    // CFL velocity limiter — clamp max displacement per substep to half
    // a particle radius. Derived from globals so it adapts automatically
    // to any particle_diameter or delta_time combination.
    let v_max = globals.particle_diameter * 0.5 / dt;
    let speed = length(v);
    if (speed > v_max) {
        v = v * (v_max / speed);
        // Write back clamped momentum for consistency
        bodies[idx].linear_momentum = vec4<f32>(v / body.inverse_mass, 0.0);
    }

    // Eq. 6: world-space inverse inertia tensor
    let q = body.quaternion;
    let r_mat = quat_to_mat3(q);
    let inv_i_world = r_mat * body.inv_inertia_tensor_body * transpose(r_mat);

    // Eq. 5: angular velocity from (damped) angular momentum
    let omega = inv_i_world * damped_angular;

    // Eq. 3: semi-implicit Euler position update
    var new_pos = body.position.xyz + v * dt;

    // Boundary position correction — prevents tunneling through analytical
    // boundary planes. If the body center has penetrated a plane, clamp it
    // to the surface (offset by particle radius) and zero the normal momentum
    // component. Without this, tunneled particles accumulate huge spring forces
    // pointing toward the box center, causing the "magnetization" effect.
    let radius = globals.particle_diameter * 0.5;
    var corrected_momentum = bodies[idx].linear_momentum.xyz;

    for (var bp = 0u; bp < globals.boundary_count; bp++) {
        let plane = globals.boundary_planes[bp];
        let n_bp = plane.xyz;
        let d_bp = plane.w;

        let yr = globals.boundary_y_ranges[bp];
        if (new_pos.y < yr.x || new_pos.y > yr.y) {
            continue;
        }

        // Check whether the particle center was inside this plane BEFORE
        // integration. If it was already outside (old_sd <= 0), the particle
        // has escaped the container and must not be teleported back in.
        let old_sd = dot(n_bp, body.position.xyz) + d_bp;
        if (old_sd <= 0.0) {
            continue;
        }

        let sd = dot(n_bp, new_pos) + d_bp;
        if (sd < radius) {
            // Push position back to the plane surface + radius
            new_pos += (radius - sd) * n_bp;

            // Kill normal momentum component to prevent re-penetration
            let v_n = dot(corrected_momentum, n_bp);
            if (v_n < 0.0) {
                corrected_momentum -= v_n * n_bp;
            }
        }
    }

    bodies[idx].linear_momentum = vec4<f32>(corrected_momentum, 0.0);
    bodies[idx].position = vec4<f32>(new_pos, 0.0);

    // Eq. 7–8: quaternion integration
    // dq = (dt/2) * [ω, 0] * q
    let omega_quat = vec4<f32>(omega, 0.0);
    let dq = quat_mul(omega_quat, q) * (dt * 0.5);
    let q_new = normalize(q + dq);
    bodies[idx].quaternion = q_new;
}
