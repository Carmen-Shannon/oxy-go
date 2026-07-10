package pipeline

import (
	"github.com/oliverbestmann/webgpu/wgpu"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
)

// pipeline is the implementation of the Pipeline interface.
// It holds the underlying WebGPU pipeline objects and related data for both render and compute pipelines.
type pipeline struct {
	common.DelegateImpl[Pipeline]

	// pipelineType indicates the type of pipeline this is; compute or render
	pipelineType PipelineType
	// pipelineKey is the unique identifier for this pipeline, used for caching and lookups
	pipelineKey string

	// the following shader references are used for pipeline creation and material binding, they are required to be set before initializing a pipeline.

	vertexShader, fragmentShader, computeShader shader.Shader

	// renderPipeline is the render pipeline if this is a render pipeline, nil otherwise
	renderPipeline *wgpu.RenderPipeline
	// computePipeline is the compute pipeline if this is a compute pipeline, nil otherwise
	computePipeline *wgpu.ComputePipeline

	// The following properties are used to configure the pipeline during creation and can be toggled/set with the builder options.
	// These are only used for renderer pipelines, compute pipelines still set defaults but do not utilize them.

	depthTestEnabled    bool
	depthWriteEnabled   bool
	depthCompare        wgpu.CompareFunction
	depthBias           int32
	depthBiasSlopeScale float32
	blendEnabled        bool
	cullMode            wgpu.CullMode
	topology            wgpu.PrimitiveTopology
	frontFace           wgpu.FrontFace
	writeMask           wgpu.ColorWriteMask
	blendState          *wgpu.BlendState
}
