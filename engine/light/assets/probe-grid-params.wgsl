struct ProbeGridParams {
    grid_min:      vec3<f32>,
    probe_count_x: u32,
    grid_max:      vec3<f32>,
    probe_count_y: u32,
    spacing:       vec3<f32>,
    probe_count_z: u32,
    total_probes:  u32,
    _pad:          vec3<u32>,
};
