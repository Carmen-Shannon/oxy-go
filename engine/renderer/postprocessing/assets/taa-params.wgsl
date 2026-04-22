struct TAAParams {
    inv_curr_view_proj: mat4x4<f32>,  // offset   0: inverse of jittered current VP (64 bytes)
    prev_view_proj:     mat4x4<f32>,  // offset  64: previous frame's jittered VP   (64 bytes)
    jitter_curr:        vec2<f32>,    // offset 128: current NDC jitter (x, y)       ( 8 bytes)
    jitter_prev:        vec2<f32>,    // offset 136: previous NDC jitter (x, y)      ( 8 bytes)
    screen_width:       f32,          // offset 144:                                  ( 4 bytes)
    screen_height:      f32,          // offset 148:                                  ( 4 bytes)
    blend_factor:       f32,          // offset 152: new-frame weight (0.1 typical)   ( 4 bytes)
    history_rectification_scale: f32, // offset 156: YCoCg clamp expansion scale      ( 4 bytes)
    raw_history_only:   f32,          // offset 160: 1.0 = output raw history         ( 4 bytes)
    _pad0:              f32,          // offset 164: uniform padding                  ( 4 bytes)
    _pad1:              f32,          // offset 168: uniform padding                  ( 4 bytes)
    _pad2:              f32,          // offset 172: uniform padding                  ( 4 bytes)
};
// Total: 176 bytes
