struct LightShadowEntry {
    light_vp: mat4x4<f32>,
    atlas_rect: vec4<f32>,
    bias: f32,
    near_plane: f32,
    far_plane: f32,
    shadow_type: u32,
};
