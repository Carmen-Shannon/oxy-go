package pipeline_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	shadermocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/shader"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

type pipelineTest struct {
	suite.Suite
}

func TestPipeline(t *testing.T) {
	suite.Run(t, new(pipelineTest))
}

func (suite *pipelineTest) TestPipelineTypeConstants() {
	suite.Run("PipelineTypeCompute is zero value", func() {
		suite.Equal(pipeline.PipelineType(0), pipeline.PipelineTypeCompute)
	})

	suite.Run("PipelineTypeRender is one", func() {
		suite.Equal(pipeline.PipelineType(1), pipeline.PipelineTypeRender)
	})

	suite.Run("compute and render are distinct", func() {
		suite.NotEqual(pipeline.PipelineTypeCompute, pipeline.PipelineTypeRender)
	})
}

func (suite *pipelineTest) TestNewPipelineDefaults() {
	suite.Run("pipeline key is preserved", func() {
		p := pipeline.NewPipeline("test-key", pipeline.PipelineTypeRender)
		suite.Equal("test-key", p.PipelineKey())
	})

	suite.Run("pipeline type is preserved", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Equal(pipeline.PipelineTypeRender, p.Type())
	})

	suite.Run("compute type is preserved", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeCompute)
		suite.Equal(pipeline.PipelineTypeCompute, p.Type())
	})

	suite.Run("depth test is enabled by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.True(p.DepthTestEnabled())
	})

	suite.Run("depth write is enabled by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.True(p.DepthWriteEnabled())
	})

	suite.Run("depth bias is zero by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Equal(int32(0), p.DepthBias())
	})

	suite.Run("depth bias slope scale is zero by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Equal(float32(0), p.DepthBiasSlopeScale())
	})

	suite.Run("blend is disabled by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.False(p.BlendEnabled())
	})

	suite.Run("cull mode is none by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Equal(wgpu.CullModeNone, p.CullMode())
	})

	suite.Run("topology is triangle list by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Equal(wgpu.PrimitiveTopologyTriangleList, p.Topology())
	})

	suite.Run("front face is CCW by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Equal(wgpu.FrontFaceCCW, p.FrontFace())
	})

	suite.Run("write mask is all by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Equal(wgpu.ColorWriteMaskAll, p.WriteMask())
	})

	suite.Run("blend state has default alpha-blend config", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		bs := p.BlendState()
		suite.NotNil(bs)
		suite.Equal(wgpu.BlendFactorSrcAlpha, bs.Color.SrcFactor)
		suite.Equal(wgpu.BlendFactorOneMinusSrcAlpha, bs.Color.DstFactor)
		suite.Equal(wgpu.BlendOperationAdd, bs.Color.Operation)
		suite.Equal(wgpu.BlendFactorOne, bs.Alpha.SrcFactor)
		suite.Equal(wgpu.BlendFactorOneMinusSrcAlpha, bs.Alpha.DstFactor)
		suite.Equal(wgpu.BlendOperationAdd, bs.Alpha.Operation)
	})

	suite.Run("vertex shader is nil by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Nil(p.Shader(shader.ShaderTypeVertex))
	})

	suite.Run("fragment shader is nil by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Nil(p.Shader(shader.ShaderTypeFragment))
	})

	suite.Run("compute shader is nil by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeCompute)
		suite.Nil(p.Shader(shader.ShaderTypeCompute))
	})

	suite.Run("render pipeline returns nil by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Nil(p.Pipeline())
	})

	suite.Run("compute pipeline returns nil by default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeCompute)
		suite.Nil(p.Pipeline())
	})
}

func (suite *pipelineTest) TestPipelineReturnsCorrectType() {
	suite.Run("render pipeline returns render pipeline value", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		var rp *wgpu.RenderPipeline
		p.SetRenderPipeline(rp)
		suite.Equal(rp, p.Pipeline())
	})

	suite.Run("compute pipeline returns compute pipeline value", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeCompute)
		var cp *wgpu.ComputePipeline
		p.SetComputePipeline(cp)
		suite.Equal(cp, p.Pipeline())
	})

	suite.Run("unknown pipeline type returns nil", func() {
		// Create a render pipeline but force an invalid type scenario
		// by checking boundary: PipelineType(99) is neither render nor compute
		p := pipeline.NewPipeline("key", pipeline.PipelineType(99))
		suite.Nil(p.Pipeline())
	})
}

func (suite *pipelineTest) TestShaderReturnsCorrectType() {
	suite.Run("returns vertex shader when requested", func() {
		vs := newMockShader(shader.ShaderTypeVertex)
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithVertexShader(vs))
		suite.Equal(vs, p.Shader(shader.ShaderTypeVertex))
	})

	suite.Run("returns fragment shader when requested", func() {
		fs := newMockShader(shader.ShaderTypeFragment)
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithFragmentShader(fs))
		suite.Equal(fs, p.Shader(shader.ShaderTypeFragment))
	})

	suite.Run("returns compute shader when requested", func() {
		cs := newMockShader(shader.ShaderTypeCompute)
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeCompute, pipeline.WithComputeShader(cs))
		suite.Equal(cs, p.Shader(shader.ShaderTypeCompute))
	})

	suite.Run("returns nil for unset shader type", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Nil(p.Shader(shader.ShaderTypeVertex))
		suite.Nil(p.Shader(shader.ShaderTypeFragment))
		suite.Nil(p.Shader(shader.ShaderTypeCompute))
	})

	suite.Run("returns nil for unknown shader type", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.Nil(p.Shader(shader.ShaderType(99)))
	})
}

func (suite *pipelineTest) TestWithVertexShaderOption() {
	suite.Run("sets vertex shader", func() {
		vs := newMockShader(shader.ShaderTypeVertex)
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithVertexShader(vs))
		suite.Equal(vs, p.Shader(shader.ShaderTypeVertex))
	})

	suite.Run("nil vertex shader is accepted", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithVertexShader(nil))
		suite.Nil(p.Shader(shader.ShaderTypeVertex))
	})
}

func (suite *pipelineTest) TestWithFragmentShaderOption() {
	suite.Run("sets fragment shader", func() {
		fs := newMockShader(shader.ShaderTypeFragment)
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithFragmentShader(fs))
		suite.Equal(fs, p.Shader(shader.ShaderTypeFragment))
	})

	suite.Run("nil fragment shader is accepted", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithFragmentShader(nil))
		suite.Nil(p.Shader(shader.ShaderTypeFragment))
	})
}

func (suite *pipelineTest) TestWithComputeShaderOption() {
	suite.Run("sets compute shader", func() {
		cs := newMockShader(shader.ShaderTypeCompute)
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeCompute, pipeline.WithComputeShader(cs))
		suite.Equal(cs, p.Shader(shader.ShaderTypeCompute))
	})

	suite.Run("nil compute shader is accepted", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeCompute, pipeline.WithComputeShader(nil))
		suite.Nil(p.Shader(shader.ShaderTypeCompute))
	})
}

func (suite *pipelineTest) TestWithDepthTestEnabledOption() {
	suite.Run("disables depth test", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithDepthTestEnabled(false))
		suite.False(p.DepthTestEnabled())
	})

	suite.Run("explicitly enables depth test", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithDepthTestEnabled(true))
		suite.True(p.DepthTestEnabled())
	})
}

func (suite *pipelineTest) TestWithDepthWriteEnabledOption() {
	suite.Run("disables depth write", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithDepthWriteEnabled(false))
		suite.False(p.DepthWriteEnabled())
	})

	suite.Run("explicitly enables depth write", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithDepthWriteEnabled(true))
		suite.True(p.DepthWriteEnabled())
	})
}

func (suite *pipelineTest) TestWithDepthBiasOption() {
	suite.Run("sets depth bias and slope scale", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithDepthBias(4, 1.5))
		suite.Equal(int32(4), p.DepthBias())
		suite.Equal(float32(1.5), p.DepthBiasSlopeScale())
	})

	suite.Run("zero bias and slope scale are accepted", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithDepthBias(0, 0))
		suite.Equal(int32(0), p.DepthBias())
		suite.Equal(float32(0), p.DepthBiasSlopeScale())
	})

	suite.Run("negative bias is accepted", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithDepthBias(-2, -0.5))
		suite.Equal(int32(-2), p.DepthBias())
		suite.Equal(float32(-0.5), p.DepthBiasSlopeScale())
	})
}

func (suite *pipelineTest) TestWithBlendEnabledOption() {
	suite.Run("enables blending", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithBlendEnabled(true))
		suite.True(p.BlendEnabled())
	})

	suite.Run("explicitly disables blending", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithBlendEnabled(false))
		suite.False(p.BlendEnabled())
	})
}

func (suite *pipelineTest) TestWithCullModeOption() {
	suite.Run("sets cull mode to back", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithCullMode(wgpu.CullModeBack))
		suite.Equal(wgpu.CullModeBack, p.CullMode())
	})

	suite.Run("sets cull mode to front", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithCullMode(wgpu.CullModeFront))
		suite.Equal(wgpu.CullModeFront, p.CullMode())
	})

	suite.Run("sets cull mode to none", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithCullMode(wgpu.CullModeNone))
		suite.Equal(wgpu.CullModeNone, p.CullMode())
	})
}

func (suite *pipelineTest) TestWithTopologyOption() {
	suite.Run("sets topology to point list", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithTopology(wgpu.PrimitiveTopologyPointList))
		suite.Equal(wgpu.PrimitiveTopologyPointList, p.Topology())
	})

	suite.Run("sets topology to line list", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithTopology(wgpu.PrimitiveTopologyLineList))
		suite.Equal(wgpu.PrimitiveTopologyLineList, p.Topology())
	})

	suite.Run("sets topology to triangle strip", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithTopology(wgpu.PrimitiveTopologyTriangleStrip))
		suite.Equal(wgpu.PrimitiveTopologyTriangleStrip, p.Topology())
	})
}

func (suite *pipelineTest) TestWithFrontFaceOption() {
	suite.Run("sets front face to CW", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithFrontFace(wgpu.FrontFaceCW))
		suite.Equal(wgpu.FrontFaceCW, p.FrontFace())
	})

	suite.Run("sets front face to CCW", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithFrontFace(wgpu.FrontFaceCCW))
		suite.Equal(wgpu.FrontFaceCCW, p.FrontFace())
	})
}

func (suite *pipelineTest) TestWithWriteMaskOption() {
	suite.Run("sets write mask to red only", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithWriteMask(wgpu.ColorWriteMaskRed))
		suite.Equal(wgpu.ColorWriteMaskRed, p.WriteMask())
	})

	suite.Run("sets write mask to all", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithWriteMask(wgpu.ColorWriteMaskAll))
		suite.Equal(wgpu.ColorWriteMaskAll, p.WriteMask())
	})

	suite.Run("sets write mask to none", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithWriteMask(wgpu.ColorWriteMaskNone))
		suite.Equal(wgpu.ColorWriteMaskNone, p.WriteMask())
	})
}

func (suite *pipelineTest) TestWithBlendStateOption() {
	suite.Run("sets custom blend state", func() {
		bs := &wgpu.BlendState{
			Color: wgpu.BlendComponent{
				SrcFactor: wgpu.BlendFactorOne,
				DstFactor: wgpu.BlendFactorZero,
				Operation: wgpu.BlendOperationAdd,
			},
			Alpha: wgpu.BlendComponent{
				SrcFactor: wgpu.BlendFactorOne,
				DstFactor: wgpu.BlendFactorZero,
				Operation: wgpu.BlendOperationAdd,
			},
		}
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithBlendState(bs))
		suite.Equal(bs, p.BlendState())
	})

	suite.Run("nil blend state replaces default", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithBlendState(nil))
		suite.Nil(p.BlendState())
	})

	suite.Run("overwrites default blend state", func() {
		bs := &wgpu.BlendState{
			Color: wgpu.BlendComponent{
				SrcFactor: wgpu.BlendFactorSrcAlpha,
				DstFactor: wgpu.BlendFactorOne,
				Operation: wgpu.BlendOperationSubtract,
			},
		}
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender, pipeline.WithBlendState(bs))
		suite.Equal(wgpu.BlendFactorSrcAlpha, p.BlendState().Color.SrcFactor)
		suite.Equal(wgpu.BlendOperationSubtract, p.BlendState().Color.Operation)
	})
}

func (suite *pipelineTest) TestSetRenderPipeline() {
	suite.Run("set and get render pipeline round-trips", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		var rp *wgpu.RenderPipeline
		p.SetRenderPipeline(rp)
		suite.Equal(rp, p.Pipeline())
	})

	suite.Run("setting nil clears render pipeline", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		p.SetRenderPipeline(nil)
		suite.Nil(p.Pipeline())
	})
}

func (suite *pipelineTest) TestSetComputePipeline() {
	suite.Run("set and get compute pipeline round-trips", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeCompute)
		var cp *wgpu.ComputePipeline
		p.SetComputePipeline(cp)
		suite.Equal(cp, p.Pipeline())
	})

	suite.Run("setting nil clears compute pipeline", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeCompute)
		p.SetComputePipeline(nil)
		suite.Nil(p.Pipeline())
	})
}

func (suite *pipelineTest) TestSetDelegate() {
	suite.Run("set delegate does not panic", func() {
		p := pipeline.NewPipeline("key", pipeline.PipelineTypeRender)
		suite.NotPanics(func() {
			p.SetDelegate(p)
		})
	})
}

func (suite *pipelineTest) TestAllOptionsComposed() {
	suite.Run("all options can be composed in single call", func() {
		vs := newMockShader(shader.ShaderTypeVertex)
		fs := newMockShader(shader.ShaderTypeFragment)
		bs := &wgpu.BlendState{
			Color: wgpu.BlendComponent{
				SrcFactor: wgpu.BlendFactorOne,
				DstFactor: wgpu.BlendFactorZero,
				Operation: wgpu.BlendOperationAdd,
			},
		}

		p := pipeline.NewPipeline("composed", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithDepthTestEnabled(false),
			pipeline.WithDepthWriteEnabled(false),
			pipeline.WithDepthBias(5, 2.0),
			pipeline.WithBlendEnabled(true),
			pipeline.WithCullMode(wgpu.CullModeBack),
			pipeline.WithTopology(wgpu.PrimitiveTopologyLineList),
			pipeline.WithFrontFace(wgpu.FrontFaceCW),
			pipeline.WithWriteMask(wgpu.ColorWriteMaskRed),
			pipeline.WithBlendState(bs),
		)

		suite.Equal("composed", p.PipelineKey())
		suite.Equal(pipeline.PipelineTypeRender, p.Type())
		suite.Equal(vs, p.Shader(shader.ShaderTypeVertex))
		suite.Equal(fs, p.Shader(shader.ShaderTypeFragment))
		suite.False(p.DepthTestEnabled())
		suite.False(p.DepthWriteEnabled())
		suite.Equal(int32(5), p.DepthBias())
		suite.Equal(float32(2.0), p.DepthBiasSlopeScale())
		suite.True(p.BlendEnabled())
		suite.Equal(wgpu.CullModeBack, p.CullMode())
		suite.Equal(wgpu.PrimitiveTopologyLineList, p.Topology())
		suite.Equal(wgpu.FrontFaceCW, p.FrontFace())
		suite.Equal(wgpu.ColorWriteMaskRed, p.WriteMask())
		suite.Equal(bs, p.BlendState())
	})
}

func (suite *pipelineTest) TestEmptyPipelineKey() {
	suite.Run("empty pipeline key is accepted", func() {
		p := pipeline.NewPipeline("", pipeline.PipelineTypeRender)
		suite.Equal("", p.PipelineKey())
	})
}

func (suite *pipelineTest) TestComputePipelineWithRenderOptions() {
	suite.Run("compute pipeline still stores render options", func() {
		p := pipeline.NewPipeline("compute-key", pipeline.PipelineTypeCompute,
			pipeline.WithDepthTestEnabled(false),
			pipeline.WithCullMode(wgpu.CullModeBack),
			pipeline.WithBlendEnabled(true),
		)
		suite.Equal(pipeline.PipelineTypeCompute, p.Type())
		suite.False(p.DepthTestEnabled())
		suite.Equal(wgpu.CullModeBack, p.CullMode())
		suite.True(p.BlendEnabled())
	})
}

// newMockShader creates a mock shader with the given type and sets up the SetDelegate expectation.
//
// Parameters:
//   - shaderType: the type of shader to mock
//
// Returns:
//   - *shadermocks.MockShader: the configured mock shader
func newMockShader(shaderType shader.ShaderType) *shadermocks.MockShader {
	s := &shadermocks.MockShader{}
	s.EXPECT().ShaderType().Return(shaderType).Maybe()
	s.EXPECT().SetDelegate(mock.Anything).Maybe()
	return s
}
