struct VertexInput {
    @location(0) position:     vec3<f32>,
    @location(1) normal:       vec3<f32>,
    @location(2) uv:           vec2<f32>,
    @location(3) color:        vec4<f32>,
    @location(4) tangent:      vec4<f32>,
    @location(5) bone_indices: vec4<u32>,
    @location(6) bone_weights: vec4<f32>,
};
