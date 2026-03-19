// Momentum accumulation — Stage 4 of the GPU rigid body pipeline
//
// Runs one invocation per rigid body. For each body, loops over its owned
// particles, sums collision forces into linear momentum and torques into
// angular momentum, then adds CPU-uploaded external forces/torques.
//
// Reference: GPU Gems 3, Chapter 29 §29.2.6 — Computation of Momenta
//   Eq. 1:  P += F_total * dt           — linear momentum update
//   Eq. 4:  L += τ_total * dt           — angular momentum update
//   Eq. 14: F_total = Σ f_i             — sum of particle forces
//   Eq. 15: τ_total = Σ (r_i × f_i)    — sum of particle torques
//
// Binding layout (3 bindings):
//   @binding(0) storage, read_write — Body array     (read particle range, flags, external forces; write momenta)
//   @binding(1) storage, read       — Particle array (read force, rel_position)
//   @binding(2) uniform             — PhysicsGlobals (delta_time, body_count)

//@oxy:include physics_body
//@oxy:include physics_particle
//@oxy:include physics_globals

//@oxy:group 0 0 storage_read_write bodies array<physics_body>
//@oxy:group 0 1 storage_read particles array<physics_particle>
//@oxy:group 0 2 storage_uniform globals physics_globals

//@oxy:inject FLAG_STATIC u32 flag_static
//@oxy:inject FLAG_KINEMATIC u32 flag_kinematic

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= globals.body_count) {
        return;
    }

    let body = bodies[idx];

    if ((body.flags & (FLAG_STATIC | FLAG_KINEMATIC)) != 0u) {
        return;
    }

    let start = body.particle_start;
    let count = body.particle_count;
    let dt = globals.delta_time;

    // Eq. 14 & 15: accumulate particle forces and torques
    var f_total = vec3<f32>(0.0);
    var tau_total = vec3<f32>(0.0);

    for (var i = 0u; i < count; i++) {
        let p = particles[start + i];
        let f_i = p.force.xyz;
        let r_i = p.rel_position.xyz;

        f_total += f_i;

        // Standard torque: arm from body center to particle center
        tau_total += cross(r_i, f_i);

        // Surface contact torque correction: the collision shader computes the
        // torque contribution from the contact-surface offset (-radius * n)
        // and packs it into the .w components of force, velocity, and
        // rel_position. For multi-particle bodies this is a small correction;
        // for single-particle bodies (where r_i = 0) this is the sole source
        // of angular momentum from contact forces.
        let surface_torque = vec3<f32>(p.force.w, p.velocity.w, p.rel_position.w);
        tau_total += surface_torque;
    }

    // Apply gravity as a persistent field force (F = m * g = g / inverse_mass)
    // This is applied every substep, unlike external_force which is one-shot.
    if (body.inverse_mass > 0.0) {
        let gravity = vec3<f32>(globals.gravity_x, globals.gravity_y, globals.gravity_z);
        f_total += gravity / body.inverse_mass;
    }

    // Add CPU-uploaded external forces/torques (user impulses, etc.)
    f_total += body.external_force.xyz;
    tau_total += body.external_torque.xyz;

    // Eq. 1: P += F_total * dt
    bodies[idx].linear_momentum = vec4<f32>(body.linear_momentum.xyz + f_total * dt, 0.0);

    // Eq. 4: L += τ_total * dt
    bodies[idx].angular_momentum = vec4<f32>(body.angular_momentum.xyz + tau_total * dt, 0.0);

    // Zero external forces so they don't persist to the next frame
    bodies[idx].external_force = vec4<f32>(0.0);
    bodies[idx].external_torque = vec4<f32>(0.0);
}
