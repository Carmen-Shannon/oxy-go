// Collision detection & DEM force computation — Stage 3 of the GPU rigid body pipeline
//
// Runs one invocation per particle. For each particle, searches the 27
// neighboring grid cells for nearby particles, performs sphere-sphere
// distance checks, and accumulates contact forces using the Discrete
// Element Method (DEM) spring-dashpot-shear model.
//
// Reference: GPU Gems 3, Chapter 29 §29.2.5 — Collision Detection and Reaction
//   Eq. 10: f_spring = -k (d - |r_ij|) n_ij        — repulsive spring
//   Eq. 11: f_damp   = η v_ij                       — velocity damping
//   Eq. 12: f_shear  = μ_t v_ij^t                   — tangential friction
//
// Binding layout (4 bindings):
//   @binding(0) storage, read       — GridCell array  (neighbor lookup)
//   @binding(1) storage, read_write — Particle array  (read positions/velocities; write force)
//   @binding(2) storage, read       — GridParams      (grid origin, dims)
//   @binding(3) uniform             — PhysicsGlobals  (DEM coefficients, particle_diameter)

//@oxy:include physics_grid
//@oxy:include physics_particle
//@oxy:include physics_grid_params
//@oxy:include physics_globals

//@oxy:group 0 0 storage_read grid array<physics_grid>
//@oxy:group 0 1 storage_read_write particles array<physics_particle>
//@oxy:group 0 2 storage_read grid_params physics_grid_params
//@oxy:group 0 3 storage_uniform globals physics_globals

const EMPTY: u32 = 0xFFFFFFFFu;
const SLOTS_PER_CELL: u32 = 16u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= globals.particle_count) {
        return;
    }

    // Skip dead particles — no forces to compute.
    if (particles[idx].world_position.w < 0.0) {
        particles[idx].force = vec4<f32>(0.0);
        return;
    }

    let pos_i = particles[idx].world_position.xyz;
    let vel_i = particles[idx].velocity.xyz;
    let body_i = bitcast<u32>(particles[idx].local_position.w) & 0xFFFFFFu;

    let diameter = globals.particle_diameter;
    let radius = diameter * 0.5;
    let cell_size = grid_params.grid_origin.w;
    let dims = grid_params.grid_dims.xyz;

    // Eq. 9: particle's grid cell using the effective cell_size from
    // grid_build_params. When the grid is proportionally scaled to fit
    // max_grid_cells, cell_size > diameter so indices naturally fit within
    // dims. Clamp retained as a float-precision safety net.
    let g_raw = vec3<i32>(floor((pos_i - grid_params.grid_origin.xyz) / cell_size));
    let g = clamp(g_raw, vec3<i32>(0), vec3<i32>(dims) - vec3<i32>(1));

    var force = vec3<f32>(0.0);

    // Per-axis wall force accumulators (Fix B). Contacts from perpendicular
    // wall faces are bucketed by the dominant axis of the wall's surface
    // normal and averaged independently. This prevents floor contacts from
    // diluting side-wall containment and vice versa at corners.
    var wall_force_x = vec3<f32>(0.0);
    var wall_count_x = 0u;
    var wall_force_y = vec3<f32>(0.0);
    var wall_count_y = 0u;
    var wall_force_z = vec3<f32>(0.0);
    var wall_count_z = 0u;

    var pp_force = vec3<f32>(0.0);    // force from dynamic body (fluid) particles
    var pp_contacts = 0u;             // dynamic particle-particle contact count

    // Surface contact torque accumulators — the contact arm from body center
    // to the contact surface is -radius * n. The torque from each contact is
    // cross(-radius * n, f_contact). This produces rolling torque from shear
    // (friction) forces and is essential for single-particle bodies where the
    // standard torque formula cross(rel_pos, f) yields zero.
    var wall_torque_x = vec3<f32>(0.0);
    var wall_torque_y = vec3<f32>(0.0);
    var wall_torque_z = vec3<f32>(0.0);
    var pp_torque = vec3<f32>(0.0);
    var bp_torque = vec3<f32>(0.0);

    // Search 27 neighbor cells (3×3×3 block centered on particle's cell)
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nc = g + vec3<i32>(dx, dy, dz);

                // Skip out-of-bounds cells
                if (any(nc < vec3<i32>(0)) || any(nc >= vec3<i32>(dims))) {
                    continue;
                }

                let cell = u32(nc.x) + u32(nc.y) * dims.x + u32(nc.z) * dims.x * dims.y;
                let s0 = grid[cell].indices_0;
                let s1 = grid[cell].indices_1;
                let s2 = grid[cell].indices_2;
                let s3 = grid[cell].indices_3;

                // Check all 16 slots in this cell (4 × vec4)
                for (var slot = 0u; slot < SLOTS_PER_CELL; slot++) {
                    var j: u32;
                    let group = slot / 4u;
                    let lane  = slot % 4u;
                    if (group == 0u) {
                        j = s0[lane];
                    } else if (group == 1u) {
                        j = s1[lane];
                    } else if (group == 2u) {
                        j = s2[lane];
                    } else {
                        j = s3[lane];
                    }
                    if (j == EMPTY || j == idx) {
                        continue;
                    }

                    // Skip same-body collisions
                    let body_j = bitcast<u32>(particles[j].local_position.w) & 0xFFFFFFu;
                    if (body_j == body_i) {
                        continue;
                    }

                    let pos_j = particles[j].world_position.xyz;
                    let r_ij = pos_i - pos_j;
                    let dist = length(r_ij);

                    if (dist >= diameter || dist < 1e-8) {
                        continue;
                    }

                    // Contact normal (points from j toward i)
                    let n = r_ij / dist;
                    let penetration = diameter - dist;

                    // Relative velocity
                    let vel_j = particles[j].velocity.xyz;
                    let v_rel = vel_i - vel_j;

                    // Eq. 10: repulsive spring force
                    // n points from j toward i (outward), so positive coefficient
                    // pushes i away from j (repulsive).
                    let f_spring = globals.spring_coeff * penetration * n;

                    // Penetration-weighted damping — scales with contact depth so that
                    // shallow (first-touch) contacts produce negligible damping while
                    // deep overlaps get full dissipation. This prevents the classic
                    // DEM artifact where high velocity + tiny overlap → huge damping
                    // force that reverses velocity (artificial bounce).
                    let damp_weight = penetration / diameter;

                    // Eq. 11: damping force (penetration-weighted, opposes normal velocity)
                    let v_n = dot(v_rel, n) * n;
                    let f_damp = -globals.damping_coeff * damp_weight * v_n;

                    // Eq. 12: tangential shear/friction (penetration-weighted)
                    let v_t = v_rel - v_n;
                    let f_shear = -globals.shear_coeff * damp_weight * v_t;

                    let f_contact = f_spring + f_damp + f_shear;

                    let is_wall = particles[j].world_position.w > 0.5;
                    if (is_wall) {
                        let sn = particles[j].surface_normal;

                        // Determine wall contact force and bucketing axis.
                        var wall_f: vec3<f32>;
                        var wall_t: vec3<f32>;
                        var bucket_axis: vec3<f32>;

                        if (sn.w > 0.5) {
                            // Fix A: Surface-normal-directed DEM for static walls.
                            // Instead of using the center-to-center contact normal n
                            // (which reverses direction once the fluid particle passes
                            // the wall particle, actively ejecting it through the wall),
                            // redirect all three DEM components along the wall's stored
                            // surface normal sn. The surface normal always points toward
                            // the box interior, so the spring ALWAYS pushes the fluid
                            // particle back inside — whether it's approaching the wall,
                            // resting on it, or has penetrated past the surface.
                            let sn_dir = sn.xyz;

                            // Spring along surface normal — always restoring
                            let f_spring_sn = globals.spring_coeff * penetration * sn_dir;

                            // Decompose velocity relative to wall surface plane
                            let v_n_sn = dot(v_rel, sn_dir) * sn_dir;
                            let v_t_sn = v_rel - v_n_sn;

                            let f_damp_sn = -globals.damping_coeff * damp_weight * v_n_sn;
                            let f_shear_sn = -globals.shear_coeff * damp_weight * v_t_sn;

                            wall_f = f_spring_sn + f_damp_sn + f_shear_sn;
                            wall_t = cross(-radius * sn_dir, wall_f);
                            bucket_axis = abs(sn_dir);
                        } else {
                            // Kinematic/dynamic wall without surface normal: use
                            // original DEM model with contact normal n.
                            wall_f = f_contact;
                            wall_t = cross(-radius * n, f_contact);
                            bucket_axis = abs(n);
                        }

                        // Fix B: per-axis bucketing by dominant axis
                        if (bucket_axis.x >= bucket_axis.y && bucket_axis.x >= bucket_axis.z) {
                            wall_force_x += wall_f;
                            wall_count_x += 1u;
                            wall_torque_x += wall_t;
                        } else if (bucket_axis.y >= bucket_axis.z) {
                            wall_force_y += wall_f;
                            wall_count_y += 1u;
                            wall_torque_y += wall_t;
                        } else {
                            wall_force_z += wall_f;
                            wall_count_z += 1u;
                            wall_torque_z += wall_t;
                        }
                    } else {
                        let contact_arm = -radius * n;
                        let torque_contrib = cross(contact_arm, f_contact);
                        pp_force += f_contact;
                        pp_contacts += 1u;
                        pp_torque += torque_contrib;
                    }
                }
            }
        }
    }

    // Per-axis wall force averaging (Fix B). Each axis group is averaged
    // independently so perpendicular wall faces maintain full containment
    // force at corners. With Fix A redirecting forces along the surface
    // normal, all contacts in a group push in the same direction — averaging
    // yields effective k = k_spring (500), well within the stability limit
    // of 4m/dt² = 2880.
    //
    // Multi-axis attenuation: when multiple axis groups are active (corners),
    // the combined wall force magnitude would be √2× or √3× a single wall.
    // Dividing by √(active_count) normalizes the total magnitude so that
    // corners don't eject particles upward — matching observed fluid behavior
    // where corners are calm zones, not pressure amplifiers.
    var wall_axes_active = 0u;
    if (wall_count_x > 0u) { wall_axes_active += 1u; }
    if (wall_count_y > 0u) { wall_axes_active += 1u; }
    if (wall_count_z > 0u) { wall_axes_active += 1u; }
    let corner_atten = select(1.0, inverseSqrt(f32(wall_axes_active)), wall_axes_active > 1u);

    if (wall_count_x > 0u) {
        force += wall_force_x / f32(wall_count_x) * corner_atten;
    }
    if (wall_count_y > 0u) {
        force += wall_force_y / f32(wall_count_y) * corner_atten;
    }
    if (wall_count_z > 0u) {
        force += wall_force_z / f32(wall_count_z) * corner_atten;
    }

    // Average fluid-fluid forces across contacts to prevent multi-contact
    // instability. With N contacts, the raw accumulated DEM force has
    // effective coefficients N×k, N×c, N×μ — easily blowing past the
    // explicit Euler stability limit c < 2m/dt. Dividing by N keeps
    // per-particle effective coefficients bounded.
    if (pp_contacts > 0u) {
        force += pp_force / f32(pp_contacts);
    }

    // Analytical boundary plane collision — each plane yields exactly one contact,
    // eliminating the multi-contact force amplification caused by particle-based walls.
    // Plane convention: (normal.xyz, d) where normal points INTO the containment volume
    // and dot(n, p) + d >= 0 means the point p is inside.
    // Each plane has a Y activation range (y_min, y_max) so walls stop at a finite height.
    for (var bp = 0u; bp < globals.boundary_count; bp++) {
        let plane = globals.boundary_planes[bp];
        let n_bp = plane.xyz;
        let d_bp = plane.w;

        // Y activation range — skip this plane when the particle is outside the range
        let yr = globals.boundary_y_ranges[bp];
        if (pos_i.y < yr.x || pos_i.y > yr.y) {
            continue;
        }

        // Signed distance from particle center to plane surface
        let sd = dot(n_bp, pos_i) + d_bp;

        // Only interact when particle center is inside the container (sd > 0).
        // If sd <= 0 the particle has already escaped the boundary — let it
        // free-fall rather than pulling it back in like a vacuum.
        if (sd <= 0.0 || sd >= radius) {
            continue;
        }

        // Clamp penetration to one full diameter to prevent runaway forces
        let pen_bp = min(radius - sd, diameter);

        // Spring: push particle along inward normal (away from wall)
        let f_bp_spring = globals.spring_coeff * pen_bp * n_bp;

        // Penetration-weighted boundary damping — same depth-scaling as
        // particle-particle contacts to prevent velocity reversal on impact.
        let bp_damp_weight = pen_bp / diameter;

        // Damping: oppose normal velocity component (wall is stationary)
        let v_n_bp = dot(vel_i, n_bp) * n_bp;
        let f_bp_damp = -globals.damping_coeff * bp_damp_weight * v_n_bp;

        // Tangential shear on boundary planes — provides friction so that
        // sliding particles experience a rolling torque. Uses the same
        // penetration-weighted scaling as particle-particle shear.
        let v_t_bp = vel_i - dot(vel_i, n_bp) * n_bp;
        let f_bp_shear = -globals.shear_coeff * bp_damp_weight * v_t_bp;

        let f_bp = f_bp_spring + f_bp_damp + f_bp_shear;
        force += f_bp;

        // Surface contact torque from boundary plane
        let bp_arm = -radius * n_bp;
        bp_torque += cross(bp_arm, f_bp);
    }

    // Compute total surface contact torque with per-axis wall averaging
    // matching the force averaging above, including corner attenuation.
    var contact_torque = vec3<f32>(0.0);
    if (wall_count_x > 0u) {
        contact_torque += wall_torque_x / f32(wall_count_x) * corner_atten;
    }
    if (wall_count_y > 0u) {
        contact_torque += wall_torque_y / f32(wall_count_y) * corner_atten;
    }
    if (wall_count_z > 0u) {
        contact_torque += wall_torque_z / f32(wall_count_z) * corner_atten;
    }
    if (pp_contacts > 0u) {
        contact_torque += pp_torque / f32(pp_contacts);
    }
    contact_torque += bp_torque;

    // Pack contact torque into unused .w components for compute_momenta:
    //   force.w        = contact_torque.x
    //   velocity.w     = contact_torque.y
    //   rel_position.w = contact_torque.z
    // These .w slots are set to 0.0 by particle_values each substep and are
    // not read by any shader between particle_values and compute_momenta.
    particles[idx].force = vec4<f32>(force, contact_torque.x);
    particles[idx].velocity = vec4<f32>(vel_i, contact_torque.y);
    particles[idx].rel_position = vec4<f32>(particles[idx].rel_position.xyz, contact_torque.z);
}
