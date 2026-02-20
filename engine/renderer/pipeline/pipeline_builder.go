package pipeline

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/cogentcore/webgpu/wgpu"
)

// PipelineBuilderOption is a functional option used to configure a Pipeline during construction.
type PipelineBuilderOption func(*pipeline)

// WithVertexShader sets the vertex shader for this pipeline.
//
// Parameters:
//   - s: the vertex shader to use for this pipeline
//
// Returns:
//   - PipelineBuilderOption: a function that sets the vertex shader for this pipeline
func WithVertexShader(s shader.Shader) PipelineBuilderOption {
	return func(p *pipeline) {
		p.vertexShader = s
	}
}

// WithFragmentShader sets the fragment shader for this pipeline.
//
// Parameters:
//   - s: the fragment shader to use for this pipeline
//
// Returns:
//   - PipelineBuilderOption: a function that sets the fragment shader for this pipeline
func WithFragmentShader(s shader.Shader) PipelineBuilderOption {
	return func(p *pipeline) {
		p.fragmentShader = s
	}
}

// WithComputeShader sets the compute shader for this pipeline.
//
// Parameters:
//   - s: the compute shader to use for this pipeline
//
// Returns:
//   - PipelineBuilderOption: a function that sets the compute shader for this pipeline
func WithComputeShader(s shader.Shader) PipelineBuilderOption {
	return func(p *pipeline) {
		p.computeShader = s
	}
}

// WithDepthTestEnabled sets whether depth testing is enabled for this pipeline.
//
// Parameters:
//   - enabled: a boolean indicating whether depth testing should be enabled
//
// Returns:
//   - PipelineBuilderOption: a function that sets the depth test enabled state for this pipeline
func WithDepthTestEnabled(enabled bool) PipelineBuilderOption {
	return func(p *pipeline) {
		p.depthTestEnabled = enabled
	}
}

// WithDepthWriteEnabled sets whether depth writing is enabled for this pipeline.
//
// Parameters:
//   - enabled: a boolean indicating whether depth writing should be enabled
//
// Returns:
//   - PipelineBuilderOption: a function that sets the depth write enabled state for this pipeline
func WithDepthWriteEnabled(enabled bool) PipelineBuilderOption {
	return func(p *pipeline) {
		p.depthWriteEnabled = enabled
	}
}

// WithDepthBias sets the depth bias parameters for this pipeline.
//
// Parameters:
//   - bias: the constant depth bias to apply
//   - slopeScale: the slope scale depth bias to apply
//
// Returns:
//   - PipelineBuilderOption: a function that sets the depth bias parameters for this pipeline
func WithDepthBias(bias int32, slopeScale float32) PipelineBuilderOption {
	return func(p *pipeline) {
		p.depthBias = bias
		p.depthBiasSlopeScale = slopeScale
	}
}

// WithBlendEnabled sets whether blending is enabled for this pipeline.
//
// Parameters:
//   - enabled: a boolean indicating whether blending should be enabled
//
// Returns:
//   - PipelineBuilderOption: a function that sets the blend enabled state for this pipeline
func WithBlendEnabled(enabled bool) PipelineBuilderOption {
	return func(p *pipeline) {
		p.blendEnabled = enabled
	}
}

// WithCullMode sets the cull mode for this pipeline.
//
// Parameters:
//   - mode: the cull mode to use for this pipeline (e.g., wgpu.CullModeNone, wgpu.CullModeFront, wgpu.CullModeBack)
//
// Returns:
//   - PipelineBuilderOption: a function that sets the cull mode for this pipeline
func WithCullMode(mode wgpu.CullMode) PipelineBuilderOption {
	return func(p *pipeline) {
		p.cullMode = mode
	}
}

// WithTopology sets the primitive topology for this pipeline.
//
// Parameters:
//   - topology: the primitive topology to use for this pipeline (e.g., wgpu.PrimitiveTopologyPointList, wgpu.PrimitiveTopologyLineList, wgpu.PrimitiveTopologyTriangleList)
//
// Returns:
//   - PipelineBuilderOption: a function that sets the primitive topology for this pipeline
func WithTopology(topology wgpu.PrimitiveTopology) PipelineBuilderOption {
	return func(p *pipeline) {
		p.topology = topology
	}
}

// WithFrontFace sets the front face winding order for this pipeline.
//
// Parameters:
//   - frontFace: the front face to use for this pipeline (e.g., wgpu.FrontFaceCCW, wgpu.FrontFaceCW)
//
// Returns:
//   - PipelineBuilderOption: a function that sets the front face for this pipeline
func WithFrontFace(frontFace wgpu.FrontFace) PipelineBuilderOption {
	return func(p *pipeline) {
		p.frontFace = frontFace
	}
}

// WithWriteMask sets the color write mask for this pipeline.
//
// Parameters:
//   - writeMask: the color write mask to use for this pipeline (e.g., wgpu.ColorWriteMaskAll, wgpu.ColorWriteMaskRed, wgpu.ColorWriteMaskGreen, wgpu.ColorWriteMaskBlue, wgpu.ColorWriteMaskAlpha)
//
// Returns:
//   - PipelineBuilderOption: a function that sets the color write mask for this pipeline
func WithWriteMask(writeMask wgpu.ColorWriteMask) PipelineBuilderOption {
	return func(p *pipeline) {
		p.writeMask = writeMask
	}
}

// WithBlendState sets the blend state for this pipeline.
//
// Parameters:
//   - blendState: the blend state to use for this pipeline (e.g., &wgpu.BlendState{Color: wgpu.BlendComponent{Operation: wgpu.BlendOperationAdd, SrcFactor: wgpu.BlendFactorSrcAlpha, DstFactor: wgpu.BlendFactorOneMinusSrcAlpha}, Alpha: wgpu.BlendComponent{Operation: wgpu.BlendOperationAdd, SrcFactor: wgpu.BlendFactorOne, DstFactor: wgpu.BlendFactorZero}})
//
// Returns:
//   - PipelineBuilderOption: a function that sets the blend state for this pipeline
func WithBlendState(blendState *wgpu.BlendState) PipelineBuilderOption {
	return func(p *pipeline) {
		p.blendState = blendState
	}
}
