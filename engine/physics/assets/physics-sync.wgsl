// Physics sync — bridges GPU physics output to the Animator's AnimationData buffer
//
// Runs one invocation per rigid body after all substeps complete. For each
// body, reads the updated position and quaternion, converts the quaternion
// to Euler angles (XYZ convention, matching the Animator's simple-compute
// shader), and writes position, rotation, and zero rotation speed to the
// Animator's AnimationData buffer at the instance slot determined by the
// sync mapping.
//
// This runs ONCE per frame after all physics substeps, before the Animator's
// own compute shader processes the data.
//
// Reference: GPU Gems 3, Chapter 29 §29.2.8 — Rendering
//
// Binding layout (4 bindings):
//   @binding(0) storage, read       — Body array          (read position, quaternion, flags)
//   @binding(1) storage, read       — Sync mapping        (body index → Animator instance ID, manual binding)
//   @binding(2) storage, read_write — AnimationData array  (write position, rotation, rot_speed=0)
//   @binding(3) uniform             — PhysicsGlobals       (body_count)

//@oxy:include physics_body
//@oxy:include physics_globals
//@oxy:include animation_data

//@oxy:group 0 0 storage_read bodies array<physics_body>
@group(0) @binding(1) var<storage, read> sync_map: array<u32>;
//@oxy:group 0 2 storage_read_write anim_data array<animation_data>
//@oxy:group 0 3 storage_uniform globals physics_globals

//@oxy:inject FLAG_ACTIVE u32 flag_active
//@oxy:inject FLAG_KINEMATIC u32 flag_kinematic

// quat_to_euler extracts XYZ Euler angles (in radians) from a unit quaternion.
// Returns (roll, pitch, yaw) matching the Animator's rotation convention.
fn quat_to_euler(q: vec4<f32>) -> vec3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;

    // Roll (X-axis rotation)
    let sinr_cosp = 2.0 * (w * x + y * z);
    let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    let roll = atan2(sinr_cosp, cosr_cosp);

    // Pitch (Y-axis rotation)
    let sinp = 2.0 * (w * y - z * x);
    var pitch: f32;
    if (abs(sinp) >= 1.0) {
        pitch = sign(sinp) * 1.5707963; // clamp at ±π/2 (gimbal lock)
    } else {
        pitch = asin(sinp);
    }

    // Yaw (Z-axis rotation)
    let siny_cosp = 2.0 * (w * z + x * y);
    let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    let yaw = atan2(siny_cosp, cosy_cosp);

    return vec3<f32>(roll, pitch, yaw);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= globals.body_count) {
        return;
    }

    let body = bodies[idx];

    // Only sync active bodies
    if ((body.flags & FLAG_ACTIVE) == 0u) {
        return;
    }

    // Skip kinematic bodies — their transforms come from skeletal animation,
    // not from physics.  Writing pos/rot here would overwrite the animator's
    // AnimationData and cause ghosting (two conflicting transform sources).
    if ((body.flags & FLAG_KINEMATIC) != 0u) {
        return;
    }

    // Each sync group owns a separate sync_map buffer where entries for bodies
    // NOT belonging to this group are set to 0xFFFFFFFF (sentinel). Skip them
    // so we only write to AnimationData slots that belong to our Animator.
    let instance_id = sync_map[idx];
    if (instance_id == 0xFFFFFFFFu) {
        return;
    }

    let euler = quat_to_euler(body.quaternion);

    // Write updated position from physics simulation
    anim_data[instance_id].pos = body.position.xyz;

    // Write rotation as Euler angles converted from the body quaternion
    anim_data[instance_id].rot = euler;

    // Zero rotation speed — physics controls rotation directly,
    // preventing the Animator's rot += rot_speed * dt from interfering
    anim_data[instance_id].rot_speed = vec3<f32>(0.0);
}
