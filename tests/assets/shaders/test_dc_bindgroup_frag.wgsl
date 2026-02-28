// Test fragment shader for binding-group type annotation branches in DrawCalls.
// Groups 0-1 are occupied by the vertex shader (camera, instance_data).
// Groups 2-5 exercise binding-group type resolution.

//@oxy:include light
//@oxy:include shadow_data
//@oxy:include overlay_params
//@oxy:include effect_params

//@oxy:group 2 0 storage_read light_buffer array<light>
//@oxy:group 3 0 storage_uniform shadow_data_buffer shadow_data
//@oxy:group 4 0 storage_uniform overlay_uniform overlay_params
//@oxy:group 5 0 storage_uniform effect_uniform effect_params

@fragment
fn main() -> @location(0) vec4f {
    return vec4f(1.0);
}
