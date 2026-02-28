struct SkeletalAnimationData {
    animation_index:      u32,
    animation_time:       f32,
    blend_weight:         f32,
    secondary_anim_index: u32,
    secondary_anim_time:  f32,
    _pad:                 vec3<f32>,
}
