// Test fragment shader for effect provider and shadow_uniform bindgroup branches.
// Groups 0-1 are occupied by the vertex shader (camera, instance_data).

//@oxy:include shadow_uniform

//@oxy:provider 2 0 effect
//@oxy:group 3 0 storage_uniform shadow_uniform_buffer shadow_uniform

@fragment
fn main() -> @location(0) vec4f {
    return vec4f(1.0);
}
