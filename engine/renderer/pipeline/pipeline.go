// Package pipeline defines renderer pipeline abstractions centered on the [Pipeline] interface.
//
// It models render and compute pipeline configuration and pipeline references used by renderer
// pipeline registration and lookup.
package pipeline

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/oliverbestmann/webgpu/wgpu"
)

// PipelineType identifies whether a pipeline is a compute pipeline or a render pipeline.
type PipelineType int

const (
	// PipelineTypeCompute indicates a compute pipeline with a single compute shader entry point.
	PipelineTypeCompute PipelineType = iota

	// PipelineTypeRender indicates a render pipeline with vertex and fragment shader entry points.
	PipelineTypeRender
)

// Pipeline defines the interface for a GPU pipeline, encapsulating either a render pipeline
// (vertex + fragment shaders) or a compute pipeline (compute shader). It holds all configuration
// state required for pipeline creation including depth, blend, cull, and topology settings.
type Pipeline interface {
	common.Delegate[Pipeline]

	// Type returns the type of the pipeline
	//
	// Returns:
	//   - PipelineType: the type of the pipeline (render or compute)
	Type() PipelineType

	// PipelineKey returns the unique key associated with this pipeline, used for caching and lookups.
	//
	// Returns:
	//   - string: the unique key for this pipeline
	PipelineKey() string

	// Shader retrieves the shader associated with the specified type if it exists, nil otherwise.
	//
	// Parameters:
	//   - shaderType: the type of shader to retrieve (vertex, fragment, or compute)
	//
	// Returns:
	//   - shader.Shader: the shader associated with the specified type, or nil if not set
	Shader(shaderType shader.ShaderType) shader.Shader

	// Pipeline returns the underlying pipeline object, either *wgpu.RenderPipeline or *wgpu.ComputePipeline
	// Note: The caller is responsible for type asserting the returned value as either pipeline type.
	//
	// Returns:
	//   - any: the underlying pipeline object.
	Pipeline() any

	// DepthTestEnabled returns whether depth testing is enabled for this pipeline.
	//
	// Returns:
	//   - bool: true if depth testing is enabled, false otherwise
	DepthTestEnabled() bool

	// DepthWriteEnabled returns whether depth writing is enabled for this pipeline.
	//
	// Returns:
	//   - bool: true if depth writing is enabled, false otherwise
	DepthWriteEnabled() bool

	// DepthCompare returns the explicit depth comparison function for this pipeline.
	// When set to wgpu.CompareFunctionUndefined the backend derives the function
	// from DepthTestEnabled (Less when enabled, Always when disabled).
	//
	// Returns:
	//   - wgpu.CompareFunction: the depth comparison function for this pipeline
	DepthCompare() wgpu.CompareFunction

	// DepthBias returns the depth bias value configured for this pipeline.
	//
	// Returns:
	//   - int32: the depth bias value for this pipeline
	DepthBias() int32

	// DepthBiasSlopeScale returns the depth bias slope scale configured for this pipeline.
	//
	// Returns:
	//   - float32: the depth bias slope scale for this pipeline
	DepthBiasSlopeScale() float32

	// BlendEnabled returns whether blending is enabled for this pipeline.
	//
	// Returns:
	//   - bool: true if blending is enabled, false otherwise
	BlendEnabled() bool

	// CullMode returns the cull mode configured for this pipeline.
	//
	// Returns:
	//   - wgpu.CullMode: the cull mode for this pipeline (e.g., wgpu.CullModeNone, wgpu.CullModeFront, wgpu.CullModeBack)
	CullMode() wgpu.CullMode

	// Topology returns the primitive topology configured for this pipeline.
	//
	// Returns:
	//   - wgpu.PrimitiveTopology: the primitive topology for this pipeline (e.g., wgpu.PrimitiveTopologyTriangleList)
	Topology() wgpu.PrimitiveTopology

	// FrontFace returns the front face winding order configured for this pipeline.
	//
	// Returns:
	//   - wgpu.FrontFace: the front face winding order for this pipeline (e.g., wgpu.FrontFaceCCW, wgpu.FrontFaceCW)
	FrontFace() wgpu.FrontFace

	// WriteMask returns the color write mask configured for this pipeline.
	//
	// Returns:
	//   - wgpu.ColorWriteMask: the color write mask for this pipeline (e.g., wgpu.ColorWriteMaskAll)
	WriteMask() wgpu.ColorWriteMask

	// BlendState returns the blend state configured for this pipeline.
	//
	// Returns:
	//   - *wgpu.BlendState: the blend state for this pipeline, or nil if blending is not enabled
	BlendState() *wgpu.BlendState

	// SetRenderPipeline sets the render pipeline
	//
	// Parameters:
	//   - p: the WebGPU render pipeline to set
	SetRenderPipeline(p *wgpu.RenderPipeline)

	// SetComputePipeline sets the compute pipeline
	//
	// Parameters:
	//   - p: the WebGPU compute pipeline to set
	SetComputePipeline(p *wgpu.ComputePipeline)
}

var _ Pipeline = &pipeline{}

func (p *pipeline) Type() PipelineType { return p.pipelineType }

func (p *pipeline) PipelineKey() string { return p.pipelineKey }

func (p *pipeline) Pipeline() any {
	switch p.pipelineType {
	case PipelineTypeRender:
		return p.renderPipeline
	case PipelineTypeCompute:
		return p.computePipeline
	default:
		return nil
	}
}

func (p *pipeline) DepthTestEnabled() bool { return p.depthTestEnabled }

func (p *pipeline) DepthWriteEnabled() bool { return p.depthWriteEnabled }

func (p *pipeline) DepthCompare() wgpu.CompareFunction { return p.depthCompare }

func (p *pipeline) DepthBias() int32 { return p.depthBias }

func (p *pipeline) DepthBiasSlopeScale() float32 { return p.depthBiasSlopeScale }

func (p *pipeline) BlendEnabled() bool { return p.blendEnabled }

func (p *pipeline) CullMode() wgpu.CullMode { return p.cullMode }

func (p *pipeline) Topology() wgpu.PrimitiveTopology { return p.topology }

func (p *pipeline) FrontFace() wgpu.FrontFace { return p.frontFace }

func (p *pipeline) WriteMask() wgpu.ColorWriteMask { return p.writeMask }

func (p *pipeline) BlendState() *wgpu.BlendState { return p.blendState }

func (p *pipeline) Shader(shaderType shader.ShaderType) shader.Shader {
	switch shaderType {
	case shader.ShaderTypeVertex:
		return p.vertexShader
	case shader.ShaderTypeFragment:
		return p.fragmentShader
	case shader.ShaderTypeCompute:
		return p.computeShader
	default:
		return nil
	}
}

func (p *pipeline) SetRenderPipeline(rp *wgpu.RenderPipeline) { p.renderPipeline = rp }

func (p *pipeline) SetComputePipeline(cp *wgpu.ComputePipeline) { p.computePipeline = cp }
