// Test fragment shader for provider-type annotation branches in DrawCalls.
// Groups 0-1 are occupied by the vertex shader (camera, instance_data).
// Groups 2-5 exercise provider resolution: camera, lights, tiles, animator.

//@oxy:provider 2 0 camera
//@oxy:provider 3 0 lights
//@oxy:provider 4 0 tiles
//@oxy:provider 5 0 animator

@fragment
fn main() -> @location(0) vec4f {
    return vec4f(1.0);
}
