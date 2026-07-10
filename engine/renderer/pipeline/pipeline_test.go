package pipeline_test

import (
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
)

type pipelineTest struct {
	suite.Suite
	p pipeline.Pipeline
	c pipeline.Pipeline
}

func TestPipeline(t *testing.T) {
	suite.Run(t, new(pipelineTest))
}

func (suite *pipelineTest) SetupSubTest() {
	suite.p = pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
	suite.c = pipeline.NewPipeline("ckey", pipeline.PipelineTypeCompute)
}

func (suite *pipelineTest) TestPipelineTypeConstants() {
	suite.Run("PipelineTypeCompute is 0", func() {
		suite.Equal(pipeline.PipelineType(0), pipeline.PipelineTypeCompute)
	})
	suite.Run("PipelineTypeRender is 1", func() {
		suite.Equal(pipeline.PipelineType(1), pipeline.PipelineTypeRender)
	})
}

func (suite *pipelineTest) TestNewPipelineDefaults() {
	suite.Run("key is set", func() {
		suite.Equal("key", suite.p.PipelineKey())
	})
	suite.Run("type is render", func() {
		suite.Equal(pipeline.PipelineTypeRender, suite.p.Type())
	})
	suite.Run("depth test enabled by default", func() {
		suite.True(suite.p.DepthTestEnabled())
	})
	suite.Run("depth write enabled by default", func() {
		suite.True(suite.p.DepthWriteEnabled())
	})
	suite.Run("blend disabled by default", func() {
		suite.False(suite.p.BlendEnabled())
	})
	suite.Run("cull mode none by default", func() {
		suite.Equal(wgpu.CullModeNone, suite.p.CullMode())
	})
	suite.Run("topology triangle list by default", func() {
		suite.Equal(wgpu.PrimitiveTopologyTriangleList, suite.p.Topology())
	})
	suite.Run("front face CCW by default", func() {
		suite.Equal(wgpu.FrontFaceCCW, suite.p.FrontFace())
	})
	suite.Run("write mask all by default", func() {
		suite.Equal(wgpu.ColorWriteMaskAll, suite.p.WriteMask())
	})
	suite.Run("blend state non-nil by default", func() {
		suite.NotNil(suite.p.BlendState())
	})
	suite.Run("depth bias zero by default", func() {
		suite.Equal(int32(0), suite.p.DepthBias())
	})
	suite.Run("depth bias slope scale zero by default", func() {
		suite.InDelta(float32(0), suite.p.DepthBiasSlopeScale(), 1e-6)
	})
	suite.Run("depth compare zero by default", func() {
		suite.Equal(wgpu.CompareFunction(0), suite.p.DepthCompare())
	})
}

func (suite *pipelineTest) TestPipelineMethod() {
	suite.Run("render pipeline returns nil when not set", func() {
		suite.Nil(suite.p.Pipeline())
	})
	suite.Run("compute pipeline returns nil when not set", func() {
		suite.Nil(suite.c.Pipeline())
	})
	suite.Run("unknown pipeline type returns nil", func() {
		p := pipeline.NewPipeline("x", pipeline.PipelineType(99))
		suite.Nil(p.Pipeline())
	})
}

func (suite *pipelineTest) TestSetRenderPipeline() {
	suite.Run("set nil render pipeline returns nil from Pipeline", func() {
		suite.p.SetRenderPipeline(nil)
		suite.Nil(suite.p.Pipeline())
	})
}

func (suite *pipelineTest) TestSetComputePipeline() {
	suite.Run("set nil compute pipeline returns nil from Pipeline", func() {
		suite.c.SetComputePipeline(nil)
		suite.Nil(suite.c.Pipeline())
	})
}

func (suite *pipelineTest) TestShader() {
	suite.Run("vertex shader returns nil when not set", func() {
		suite.Nil(suite.p.Shader(shader.ShaderTypeVertex))
	})
	suite.Run("fragment shader returns nil when not set", func() {
		suite.Nil(suite.p.Shader(shader.ShaderTypeFragment))
	})
	suite.Run("compute shader returns nil when not set", func() {
		suite.Nil(suite.p.Shader(shader.ShaderTypeCompute))
	})
	suite.Run("unknown shader type returns nil", func() {
		suite.Nil(suite.p.Shader(shader.ShaderType(99)))
	})
}

func (suite *pipelineTest) TestWithVertexShader() {
	suite.Run("vertex shader set to nil", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithVertexShader(nil))
		suite.Nil(p.Shader(shader.ShaderTypeVertex))
	})
}

func (suite *pipelineTest) TestWithFragmentShader() {
	suite.Run("fragment shader set to nil", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithFragmentShader(nil))
		suite.Nil(p.Shader(shader.ShaderTypeFragment))
	})
}

func (suite *pipelineTest) TestWithComputeShader() {
	suite.Run("compute shader set to nil", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeCompute, pipeline.WithComputeShader(nil))
		suite.Nil(p.Shader(shader.ShaderTypeCompute))
	})
}

func (suite *pipelineTest) TestWithDepthTestEnabled() {
	suite.Run("depth test disabled", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithDepthTestEnabled(false))
		suite.False(p.DepthTestEnabled())
	})
}

func (suite *pipelineTest) TestWithDepthWriteEnabled() {
	suite.Run("depth write disabled", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithDepthWriteEnabled(false))
		suite.False(p.DepthWriteEnabled())
	})
}

func (suite *pipelineTest) TestWithDepthCompare() {
	suite.Run("depth compare set to LessEqual", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithDepthCompare(wgpu.CompareFunctionLessEqual))
		suite.Equal(wgpu.CompareFunctionLessEqual, p.DepthCompare())
	})
}

func (suite *pipelineTest) TestWithDepthBias() {
	suite.Run("depth bias set", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithDepthBias(2, 2.0))
		suite.Equal(int32(2), p.DepthBias())
	})
	suite.Run("depth bias slope scale set", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithDepthBias(2, 2.0))
		suite.InDelta(float32(2.0), p.DepthBiasSlopeScale(), 1e-6)
	})
}

func (suite *pipelineTest) TestWithBlendEnabled() {
	suite.Run("blend enabled", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithBlendEnabled(true))
		suite.True(p.BlendEnabled())
	})
}

func (suite *pipelineTest) TestWithCullMode() {
	suite.Run("cull mode back", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithCullMode(wgpu.CullModeBack))
		suite.Equal(wgpu.CullModeBack, p.CullMode())
	})
}

func (suite *pipelineTest) TestWithTopology() {
	suite.Run("topology line list", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithTopology(wgpu.PrimitiveTopologyLineList))
		suite.Equal(wgpu.PrimitiveTopologyLineList, p.Topology())
	})
}

func (suite *pipelineTest) TestWithFrontFace() {
	suite.Run("front face CW", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithFrontFace(wgpu.FrontFaceCW))
		suite.Equal(wgpu.FrontFaceCW, p.FrontFace())
	})
}

func (suite *pipelineTest) TestWithWriteMask() {
	suite.Run("write mask none", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithWriteMask(wgpu.ColorWriteMaskNone))
		suite.Equal(wgpu.ColorWriteMaskNone, p.WriteMask())
	})
}

func (suite *pipelineTest) TestWithBlendState() {
	suite.Run("blend state nil", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithBlendState(nil))
		suite.Nil(p.BlendState())
	})
	suite.Run("blend state non-nil", func() {
		p := pipeline.NewPipeline("k", pipeline.PipelineTypeRender, pipeline.WithBlendState(&wgpu.BlendState{}))
		suite.NotNil(p.BlendState())
	})
}

func (suite *pipelineTest) TestType() {
	suite.Run("render pipeline type", func() {
		suite.Equal(pipeline.PipelineTypeRender, suite.p.Type())
	})
	suite.Run("compute pipeline type", func() {
		suite.Equal(pipeline.PipelineTypeCompute, suite.c.Type())
	})
}

func (suite *pipelineTest) TestPipelineKey() {
	suite.Run("render pipeline key", func() {
		suite.Equal("key", suite.p.PipelineKey())
	})
}
