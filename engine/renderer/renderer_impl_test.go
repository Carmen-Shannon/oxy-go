package renderer_test

import (
	"fmt"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	pipeline_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline/mocks"
)

type rendererImplTest struct {
	suite.Suite
	backendMock *mocks.MockRendererBackend
	r           renderer.Renderer
}

func TestRunRendererImplTests(t *testing.T) {
	suite.Run(t, new(rendererImplTest))
}

func (suite *rendererImplTest) SetupSubTest() {
	suite.backendMock = mocks.NewMockRendererBackend(suite.T())
	suite.r = renderer.NewRendererWithBackend(suite.backendMock)
}

// --- Pipeline cache methods ---

func (suite *rendererImplTest) TestPipeline() {
	suite.Run("should return nil for a missing key", func() {
		result := suite.r.Pipeline("missing")
		suite.Nil(result)
	})
	suite.Run("should return the cached pipeline for an existing key", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		result := suite.r.Pipeline("key")
		suite.Equal(mockPipeline, result)
	})
}

func (suite *rendererImplTest) TestPipelines() {
	suite.Run("should return the full pipeline cache map", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		result := suite.r.Pipelines()
		suite.Equal(mockPipeline, result["key"])
	})
}

func (suite *rendererImplTest) TestSetPipeline() {
	suite.Run("should insert a pipeline under the given key", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		suite.Equal(mockPipeline, suite.r.Pipeline("key"))
	})
	suite.Run("should overwrite an existing pipeline under the same key", func() {
		first := pipeline_mocks.NewMockPipeline(suite.T())
		second := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", first)
		suite.r.SetPipeline("key", second)
		suite.Equal(second, suite.r.Pipeline("key"))
	})
}

func (suite *rendererImplTest) TestSetPipelines() {
	suite.Run("should replace the entire pipeline cache", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		newCache := map[string]pipeline.Pipeline{"newkey": mockPipeline}
		suite.r.SetPipelines(newCache)
		suite.Equal(newCache, suite.r.Pipelines())
	})
}

func (suite *rendererImplTest) TestSetInjections() {
	suite.Run("should store the injection map on the renderer", func() {
		suite.r.SetInjections(map[string]string{"key": "value"})
	})
}

// --- RegisterPipelines ---

func (suite *rendererImplTest) TestRegisterPipelines() {
	suite.Run("should skip registration for a pipeline whose key already exists in the cache", func() {
		existingPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", existingPipeline)
		incoming := pipeline_mocks.NewMockPipeline(suite.T())
		incoming.EXPECT().PipelineKey().Return("key").Once()
		err := suite.r.RegisterPipelines(incoming)
		suite.NoError(err)
		suite.Equal(existingPipeline, suite.r.Pipeline("key"))
	})
	suite.Run("should register a compute pipeline and cache it", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("compute-key").Once()
		mockPipeline.EXPECT().Type().Return(pipeline.PipelineTypeCompute).Once()
		suite.backendMock.EXPECT().RegisterComputePipeline(mockPipeline).Return(nil).Once()
		err := suite.r.RegisterPipelines(mockPipeline)
		suite.NoError(err)
		suite.Equal(mockPipeline, suite.r.Pipeline("compute-key"))
	})
	suite.Run("should register a render pipeline and cache it", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("render-key").Once()
		mockPipeline.EXPECT().Type().Return(pipeline.PipelineTypeRender).Once()
		suite.backendMock.EXPECT().RegisterRenderPipeline(mockPipeline).Return(nil).Once()
		err := suite.r.RegisterPipelines(mockPipeline)
		suite.NoError(err)
		suite.Equal(mockPipeline, suite.r.Pipeline("render-key"))
	})
	suite.Run("should return an error when compute pipeline registration fails", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("compute-key").Once()
		mockPipeline.EXPECT().Type().Return(pipeline.PipelineTypeCompute).Once()
		suite.backendMock.EXPECT().RegisterComputePipeline(mockPipeline).Return(fmt.Errorf("test error")).Once()
		err := suite.r.RegisterPipelines(mockPipeline)
		suite.Error(err)
		suite.Nil(suite.r.Pipeline("compute-key"))
	})
	suite.Run("should return an error when render pipeline registration fails", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("render-key").Once()
		mockPipeline.EXPECT().Type().Return(pipeline.PipelineTypeRender).Once()
		suite.backendMock.EXPECT().RegisterRenderPipeline(mockPipeline).Return(fmt.Errorf("test error")).Once()
		err := suite.r.RegisterPipelines(mockPipeline)
		suite.Error(err)
		suite.Nil(suite.r.Pipeline("render-key"))
	})
	suite.Run("should register multiple pipelines in one call", func() {
		computePipe := pipeline_mocks.NewMockPipeline(suite.T())
		computePipe.EXPECT().PipelineKey().Return("compute-key").Once()
		computePipe.EXPECT().Type().Return(pipeline.PipelineTypeCompute).Once()
		suite.backendMock.EXPECT().RegisterComputePipeline(computePipe).Return(nil).Once()

		renderPipe := pipeline_mocks.NewMockPipeline(suite.T())
		renderPipe.EXPECT().PipelineKey().Return("render-key").Once()
		renderPipe.EXPECT().Type().Return(pipeline.PipelineTypeRender).Once()
		suite.backendMock.EXPECT().RegisterRenderPipeline(renderPipe).Return(nil).Once()

		err := suite.r.RegisterPipelines(computePipe, renderPipe)
		suite.NoError(err)
		suite.Equal(computePipe, suite.r.Pipeline("compute-key"))
		suite.Equal(renderPipe, suite.r.Pipeline("render-key"))
	})
	suite.Run("should cache pipeline with unknown type without backend registration", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("unknown-key").Once()
		mockPipeline.EXPECT().Type().Return(pipeline.PipelineType(99)).Once()
		err := suite.r.RegisterPipelines(mockPipeline)
		suite.NoError(err)
		suite.Equal(mockPipeline, suite.r.Pipeline("unknown-key"))
	})
}

// --- RegisterShadowDepthPipeline ---

func (suite *rendererImplTest) TestRegisterShadowDepthPipeline() {
	suite.Run("should skip when the key already exists in the cache", func() {
		existingPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("shadow-key", existingPipeline)
		incoming := pipeline_mocks.NewMockPipeline(suite.T())
		incoming.EXPECT().PipelineKey().Return("shadow-key").Once()
		err := suite.r.RegisterShadowDepthPipeline(incoming)
		suite.NoError(err)
		suite.Equal(existingPipeline, suite.r.Pipeline("shadow-key"))
	})
	suite.Run("should register the shadow depth pipeline and cache it", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("shadow-key").Once()
		suite.backendMock.EXPECT().RegisterShadowDepthPipeline(mockPipeline).Return(nil).Once()
		err := suite.r.RegisterShadowDepthPipeline(mockPipeline)
		suite.NoError(err)
		suite.Equal(mockPipeline, suite.r.Pipeline("shadow-key"))
	})
	suite.Run("should return an error when backend registration fails", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("shadow-key").Once()
		suite.backendMock.EXPECT().RegisterShadowDepthPipeline(mockPipeline).Return(fmt.Errorf("test error")).Once()
		err := suite.r.RegisterShadowDepthPipeline(mockPipeline)
		suite.Error(err)
		suite.Nil(suite.r.Pipeline("shadow-key"))
	})
}

// --- RegisterGBufferPipeline ---

func (suite *rendererImplTest) TestRegisterGBufferPipeline() {
	suite.Run("should skip when the key already exists in the cache", func() {
		existingPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("gbuffer-key", existingPipeline)
		incoming := pipeline_mocks.NewMockPipeline(suite.T())
		incoming.EXPECT().PipelineKey().Return("gbuffer-key").Once()
		err := suite.r.RegisterGBufferPipeline(incoming)
		suite.NoError(err)
		suite.Equal(existingPipeline, suite.r.Pipeline("gbuffer-key"))
	})
	suite.Run("should register the gbuffer pipeline and cache it", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("gbuffer-key").Once()
		suite.backendMock.EXPECT().RegisterGBufferPipeline(mockPipeline).Return(nil).Once()
		err := suite.r.RegisterGBufferPipeline(mockPipeline)
		suite.NoError(err)
		suite.Equal(mockPipeline, suite.r.Pipeline("gbuffer-key"))
	})
	suite.Run("should return an error when backend registration fails", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("gbuffer-key").Once()
		suite.backendMock.EXPECT().RegisterGBufferPipeline(mockPipeline).Return(fmt.Errorf("test error")).Once()
		err := suite.r.RegisterGBufferPipeline(mockPipeline)
		suite.Error(err)
		suite.Nil(suite.r.Pipeline("gbuffer-key"))
	})
}

// --- RegisterCompositionPipeline ---

func (suite *rendererImplTest) TestRegisterCompositionPipeline() {
	suite.Run("should skip when the key already exists in the cache", func() {
		existingPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("comp-key", existingPipeline)
		incoming := pipeline_mocks.NewMockPipeline(suite.T())
		incoming.EXPECT().PipelineKey().Return("comp-key").Once()
		err := suite.r.RegisterCompositionPipeline(incoming)
		suite.NoError(err)
		suite.Equal(existingPipeline, suite.r.Pipeline("comp-key"))
	})
	suite.Run("should register the composition pipeline and cache it", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("comp-key").Once()
		suite.backendMock.EXPECT().RegisterCompositionPipeline(mockPipeline).Return(nil).Once()
		err := suite.r.RegisterCompositionPipeline(mockPipeline)
		suite.NoError(err)
		suite.Equal(mockPipeline, suite.r.Pipeline("comp-key"))
	})
	suite.Run("should return an error when backend registration fails", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().PipelineKey().Return("comp-key").Once()
		suite.backendMock.EXPECT().RegisterCompositionPipeline(mockPipeline).Return(fmt.Errorf("test error")).Once()
		err := suite.r.RegisterCompositionPipeline(mockPipeline)
		suite.Error(err)
		suite.Nil(suite.r.Pipeline("comp-key"))
	})
}

func (suite *rendererImplTest) TestDispatchComputeBatch() {
	suite.Run("should call backend with empty entries when the pipeline key is not in the cache", func() {
		suite.backendMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(e []renderer.ComputeDispatchEntry) bool {
			return len(e) == 0
		})).Return().Once()
		suite.r.DispatchComputeBatch([]renderer.ComputeDispatch{
			{PipelineKey: "missing", Providers: nil, WorkGroupCount: [3]uint32{1, 1, 1}},
		})
	})
	suite.Run("should dispatch compute when the pipeline key exists", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		suite.backendMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(e []renderer.ComputeDispatchEntry) bool {
			return len(e) == 1 && e[0].Pipeline == mockPipeline
		})).Return().Once()
		suite.r.DispatchComputeBatch([]renderer.ComputeDispatch{
			{PipelineKey: "key", Providers: nil, WorkGroupCount: [3]uint32{1, 1, 1}},
		})
	})
	suite.Run("should dispatch only found pipelines in a mixed batch", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("found", mockPipeline)
		suite.backendMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(e []renderer.ComputeDispatchEntry) bool {
			return len(e) == 1 && e[0].Pipeline == mockPipeline
		})).Return().Once()
		suite.r.DispatchComputeBatch([]renderer.ComputeDispatch{
			{PipelineKey: "found", Providers: nil, WorkGroupCount: [3]uint32{1, 1, 1}},
			{PipelineKey: "missing", Providers: nil, WorkGroupCount: [3]uint32{2, 2, 2}},
		})
	})
}

// --- DrawCall ---

func (suite *rendererImplTest) TestDrawCall() {
	suite.Run("should return an error when the pipeline key is not found", func() {
		err := suite.r.DrawCall("missing", nil, 1, nil)
		suite.Error(err)
	})
	suite.Run("should call the backend DrawCall and return nil when the pipeline exists", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		suite.backendMock.EXPECT().DrawCall(mockPipeline, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		err := suite.r.DrawCall("key", nil, 1, nil)
		suite.NoError(err)
	})
}

// --- DrawCallIndirect ---

func (suite *rendererImplTest) TestDrawCallIndirect() {
	suite.Run("should return an error when the pipeline key is not found", func() {
		err := suite.r.DrawCallIndirect("missing", nil, nil, nil)
		suite.Error(err)
	})
	suite.Run("should call the backend DrawCallIndirect and return nil when the pipeline exists", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		suite.backendMock.EXPECT().DrawCallIndirect(mockPipeline, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		err := suite.r.DrawCallIndirect("key", nil, nil, nil)
		suite.NoError(err)
	})
}

// --- ShadowDrawCall ---

func (suite *rendererImplTest) TestShadowDrawCall() {
	suite.Run("should return an error when the pipeline key is not found", func() {
		err := suite.r.ShadowDrawCall("missing", nil, 1, nil)
		suite.Error(err)
	})
	suite.Run("should call the backend ShadowDrawCall and return nil when the pipeline exists", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		suite.backendMock.EXPECT().ShadowDrawCall(mockPipeline, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		err := suite.r.ShadowDrawCall("key", nil, 1, nil)
		suite.NoError(err)
	})
}

// --- ShadowDrawCallIndirect ---

func (suite *rendererImplTest) TestShadowDrawCallIndirect() {
	suite.Run("should return an error when the pipeline key is not found", func() {
		err := suite.r.ShadowDrawCallIndirect("missing", nil, nil, nil)
		suite.Error(err)
	})
	suite.Run("should call the backend ShadowDrawCallIndirect and return nil when the pipeline exists", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		suite.backendMock.EXPECT().ShadowDrawCallIndirect(mockPipeline, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		err := suite.r.ShadowDrawCallIndirect("key", nil, nil, nil)
		suite.NoError(err)
	})
}

// --- GBufferDrawCall ---

func (suite *rendererImplTest) TestGBufferDrawCall() {
	suite.Run("should return an error when the pipeline key is not found", func() {
		err := suite.r.GBufferDrawCall("missing", nil, 1, nil)
		suite.Error(err)
	})
	suite.Run("should call the backend GBufferDrawCall and return nil when the pipeline exists", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		suite.backendMock.EXPECT().GBufferDrawCall(mockPipeline, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		err := suite.r.GBufferDrawCall("key", nil, 1, nil)
		suite.NoError(err)
	})
}

// --- GBufferDrawCallIndirect ---

func (suite *rendererImplTest) TestGBufferDrawCallIndirect() {
	suite.Run("should return an error when the pipeline key is not found", func() {
		err := suite.r.GBufferDrawCallIndirect("missing", nil, nil, nil)
		suite.Error(err)
	})
	suite.Run("should call the backend GBufferDrawCallIndirect and return nil when the pipeline exists", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		suite.backendMock.EXPECT().GBufferDrawCallIndirect(mockPipeline, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		err := suite.r.GBufferDrawCallIndirect("key", nil, nil, nil)
		suite.NoError(err)
	})
}

// --- CompositionDrawCall ---

func (suite *rendererImplTest) TestCompositionDrawCall() {
	suite.Run("should return an error when the pipeline key is not found", func() {
		err := suite.r.CompositionDrawCall("missing", nil)
		suite.Error(err)
	})
	suite.Run("should call the backend CompositionDrawCall and return nil when the pipeline exists", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		suite.r.SetPipeline("key", mockPipeline)
		suite.backendMock.EXPECT().CompositionDrawCall(mockPipeline, mock.Anything).Return().Once()
		err := suite.r.CompositionDrawCall("key", nil)
		suite.NoError(err)
	})
}

// --- WriteBuffers / WriteRawBuffer ---

func (suite *rendererImplTest) TestWriteBuffers() {
	suite.Run("should delegate to the backend with the given writes", func() {
		suite.backendMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.r.WriteBuffers(nil)
	})
}

func (suite *rendererImplTest) TestWriteRawBuffer() {
	suite.Run("should delegate to the backend with the given buffer, offset, and data", func() {
		suite.backendMock.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.r.WriteRawBuffer(nil, 0, nil)
	})
}

// --- Pure delegation methods ---

func (suite *rendererImplTest) TestResize() {
	suite.Run("should call ConfigureSurface on the backend", func() {
		suite.backendMock.EXPECT().ConfigureSurface(800, 600).Return().Once()
		suite.r.Resize(800, 600)
	})
}

func (suite *rendererImplTest) TestSetPresentMode() {
	suite.Run("should call SetPresentMode on the backend", func() {
		suite.backendMock.EXPECT().SetPresentMode(renderer.PresentModeVSync).Return().Once()
		suite.r.SetPresentMode(renderer.PresentModeVSync)
	})
}

func (suite *rendererImplTest) TestBeginComputeFrame() {
	suite.Run("should call the backend", func() {
		suite.backendMock.EXPECT().BeginComputeFrame().Return().Once()
		suite.r.BeginComputeFrame()
	})
}

func (suite *rendererImplTest) TestEndComputeFrame() {
	suite.Run("should call EndComputeFrame on the backend", func() {
		suite.backendMock.EXPECT().EndComputeFrame().Return().Once()
		suite.r.EndComputeFrame()
	})
}

func (suite *rendererImplTest) TestBeginFrame() {
	suite.Run("should return the backend result", func() {
		suite.backendMock.EXPECT().BeginFrame().Return(nil).Once()
		err := suite.r.BeginFrame()
		suite.NoError(err)
	})
}

func (suite *rendererImplTest) TestEndFrame() {
	suite.Run("should call EndFrame on the backend", func() {
		suite.backendMock.EXPECT().EndFrame().Return().Once()
		suite.r.EndFrame()
	})
}

func (suite *rendererImplTest) TestPresent() {
	suite.Run("should call Present on the backend", func() {
		suite.backendMock.EXPECT().Present().Return().Once()
		suite.r.Present()
	})
}

func (suite *rendererImplTest) TestBeginGeometryFrame() {
	suite.Run("should call the backend", func() {
		suite.backendMock.EXPECT().BeginGeometryFrame().Return().Once()
		suite.r.BeginGeometryFrame()
	})
}

func (suite *rendererImplTest) TestEndGeometryFrame() {
	suite.Run("should call EndGeometryFrame on the backend", func() {
		suite.backendMock.EXPECT().EndGeometryFrame().Return().Once()
		suite.r.EndGeometryFrame()
	})
}

func (suite *rendererImplTest) TestBeginShadowFrame() {
	suite.Run("should call the backend", func() {
		suite.backendMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.r.BeginShadowFrame()
	})
}

func (suite *rendererImplTest) TestEndShadowFrame() {
	suite.Run("should call EndShadowFrame on the backend", func() {
		suite.backendMock.EXPECT().EndShadowFrame().Return().Once()
		suite.r.EndShadowFrame()
	})
}

func (suite *rendererImplTest) TestBeginShadowAtlasPass() {
	suite.Run("should delegate to the backend", func() {
		suite.backendMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.r.BeginShadowAtlasPass(nil)
	})
}

func (suite *rendererImplTest) TestSetShadowViewport() {
	suite.Run("should delegate to the backend", func() {
		suite.backendMock.EXPECT().SetShadowViewport(uint32(0), uint32(0), uint32(512), uint32(512)).Return().Once()
		suite.r.SetShadowViewport(0, 0, 512, 512)
	})
}

func (suite *rendererImplTest) TestEndShadowAtlasPass() {
	suite.Run("should delegate to the backend", func() {
		suite.backendMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.r.EndShadowAtlasPass()
	})
}

func (suite *rendererImplTest) TestEndGBufferPass() {
	suite.Run("should call EndGBufferPass on the backend", func() {
		suite.backendMock.EXPECT().EndGBufferPass().Return().Once()
		suite.r.EndGBufferPass()
	})
}

func (suite *rendererImplTest) TestEndGBufferFrame() {
	suite.Run("should call EndGBufferFrame on the backend", func() {
		suite.backendMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.r.EndGBufferFrame()
	})
}

func (suite *rendererImplTest) TestBeginGBufferFrame() {
	suite.Run("should call the backend", func() {
		suite.backendMock.EXPECT().BeginGBufferFrame().Return().Once()
		suite.r.BeginGBufferFrame()
	})
}

func (suite *rendererImplTest) TestBeginGBufferPass() {
	suite.Run("should call BeginGBufferPass on the backend", func() {
		suite.backendMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.r.BeginGBufferPass(nil, nil, nil)
	})
}

func (suite *rendererImplTest) TestBeginCompositionFrame() {
	suite.Run("should return the backend result", func() {
		suite.backendMock.EXPECT().BeginCompositionFrame().Return(nil).Once()
		err := suite.r.BeginCompositionFrame()
		suite.NoError(err)
	})
}

func (suite *rendererImplTest) TestBeginCompositionPass() {
	suite.Run("should call BeginCompositionPass on the backend", func() {
		suite.backendMock.EXPECT().BeginCompositionPass().Return().Once()
		suite.r.BeginCompositionPass()
	})
}

func (suite *rendererImplTest) TestEndCompositionPass() {
	suite.Run("should call EndCompositionPass on the backend", func() {
		suite.backendMock.EXPECT().EndCompositionPass().Return().Once()
		suite.r.EndCompositionPass()
	})
}

func (suite *rendererImplTest) TestEndCompositionFrame() {
	suite.Run("should call EndCompositionFrame on the backend", func() {
		suite.backendMock.EXPECT().EndCompositionFrame().Return().Once()
		suite.r.EndCompositionFrame()
	})
}

func (suite *rendererImplTest) TestFlushFrame() {
	suite.Run("should call FlushFrame on the backend", func() {
		suite.backendMock.EXPECT().FlushFrame().Return(wgpu.SubmissionIndex(0)).Once()
		suite.r.FlushFrame()
	})
}

func (suite *rendererImplTest) TestCurrentFrameSlot() {
	suite.Run("should call CurrentFrameSlot on the backend and return the result", func() {
		suite.backendMock.EXPECT().CurrentFrameSlot().Return(1).Once()
		result := suite.r.CurrentFrameSlot()
		suite.Equal(1, result)
	})
}

func (suite *rendererImplTest) TestWaitIdle() {
	suite.Run("should call WaitIdle on the backend", func() {
		suite.backendMock.EXPECT().WaitIdle().Return().Once()
		suite.r.WaitIdle()
	})
}

func (suite *rendererImplTest) TestSyncGPUTimestamps() {
	suite.Run("should call SyncGPUTimestamps on the backend", func() {
		suite.backendMock.EXPECT().SyncGPUTimestamps().Return().Once()
		suite.r.SyncGPUTimestamps()
	})
}

func (suite *rendererImplTest) TestGPUTimings() {
	suite.Run("should return the timings map when backend provides non nil data", func() {
		expected := map[string]float64{"compute": 1.25, "geometry": 2.5}
		suite.backendMock.EXPECT().GPUTimings().Return(expected).Once()

		timings := suite.r.GPUTimings()

		suite.Equal(expected, timings)
	})

	suite.Run("should return nil when backend timings are unavailable", func() {
		suite.backendMock.EXPECT().GPUTimings().Return(nil).Once()

		timings := suite.r.GPUTimings()

		suite.Nil(timings)
	})
}

func (suite *rendererImplTest) TestSampleCount() {
	suite.Run("should return the backend sample count", func() {
		suite.backendMock.EXPECT().SampleCount().Return(uint32(4)).Once()
		count := suite.r.SampleCount()
		suite.Equal(uint32(4), count)
	})
}

func (suite *rendererImplTest) TestMaxTextureDimension2D() {
	suite.Run("should return the backend max texture dimension", func() {
		suite.backendMock.EXPECT().MaxTextureDimension2D().Return(uint32(8192)).Once()
		dim := suite.r.MaxTextureDimension2D()
		suite.Equal(uint32(8192), dim)
	})
}

func (suite *rendererImplTest) TestSetRenderTargetFormat() {
	suite.Run("should call SetRenderTargetFormat on the backend", func() {
		suite.backendMock.EXPECT().SetRenderTargetFormat(mock.Anything).Return().Once()
		suite.r.SetRenderTargetFormat(wgpu.TextureFormat(1))
	})
}

func (suite *rendererImplTest) TestInitMeshBuffers() {
	suite.Run("should delegate to the backend and return its result", func() {
		suite.backendMock.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		err := suite.r.InitMeshBuffers(nil, nil, nil, 0)
		suite.NoError(err)
	})
}

func (suite *rendererImplTest) TestInitBindGroup() {
	suite.Run("should delegate to the backend and return its result", func() {
		suite.backendMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		err := suite.r.InitBindGroup(nil, wgpu.BindGroupLayoutDescriptor{}, nil, nil)
		suite.NoError(err)
	})
}

func (suite *rendererImplTest) TestInitTextureView() {
	suite.Run("should delegate to the backend and return its result", func() {
		suite.backendMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		err := suite.r.InitTextureView(nil, 0, common.TextureStagingData{})
		suite.NoError(err)
	})
}

func (suite *rendererImplTest) TestInitSampler() {
	suite.Run("should delegate to the backend and return its result", func() {
		suite.backendMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		err := suite.r.InitSampler(nil, 0, common.SamplerStagingData{})
		suite.NoError(err)
	})
}

func (suite *rendererImplTest) TestCreateBuffer() {
	suite.Run("should delegate to the backend and return the buffer", func() {
		suite.backendMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		buf := suite.r.CreateBuffer("label", 64, 0)
		suite.Nil(buf)
	})
}

func (suite *rendererImplTest) TestCopyBufferToBuffer() {
	suite.Run("should delegate to the backend", func() {
		suite.backendMock.EXPECT().CopyBufferToBuffer(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.r.CopyBufferToBuffer(nil, nil, 0, 0, 0)
	})
}

func (suite *rendererImplTest) TestReadMappedBuffer() {
	suite.Run("should delegate to the backend and return the byte slice and error", func() {
		expected := []byte{1, 2, 3}
		suite.backendMock.EXPECT().ReadMappedBuffer(mock.Anything, mock.Anything, mock.Anything).Return(expected, nil).Once()
		result, err := suite.r.ReadMappedBuffer(nil, 0, 3)
		suite.NoError(err)
		suite.Equal(expected, result)
	})
}

func (suite *rendererImplTest) TestWriteTexture() {
	suite.Run("should delegate to the backend", func() {
		suite.backendMock.EXPECT().WriteTexture(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.r.WriteTexture(nil, nil, 0, 0, 0)
	})
}

func (suite *rendererImplTest) TestCreateShadowDepthTexture() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(nil, nil).Once()
		view, tex := suite.r.CreateShadowDepthTexture(512, 512)
		suite.Nil(view)
		suite.Nil(tex)
	})
}

func (suite *rendererImplTest) TestCreateComparisonSampler() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateComparisonSampler().Return(nil).Once()
		sampler := suite.r.CreateComparisonSampler()
		suite.Nil(sampler)
	})
}

func (suite *rendererImplTest) TestCreateLinearSampler() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateLinearSampler().Return(nil).Once()
		sampler := suite.r.CreateLinearSampler()
		suite.Nil(sampler)
	})
}

func (suite *rendererImplTest) TestCreateGBufferTextures() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateGBufferTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil).Once()
		normView, normTex, albedoView, albedoTex, depthView, depthTex := suite.r.CreateGBufferTextures(1920, 1080)
		suite.Nil(normView)
		suite.Nil(normTex)
		suite.Nil(albedoView)
		suite.Nil(albedoTex)
		suite.Nil(depthView)
		suite.Nil(depthTex)
	})
}

func (suite *rendererImplTest) TestCreateSSAOTextures() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateSSAOTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil).Once()
		rawView, rawTex, blurredView, blurredTex, scratchView, scratchTex := suite.r.CreateSSAOTextures(1920, 1080)
		suite.Nil(rawView)
		suite.Nil(rawTex)
		suite.Nil(blurredView)
		suite.Nil(blurredTex)
		suite.Nil(scratchView)
		suite.Nil(scratchTex)
	})
}

func (suite *rendererImplTest) TestBeginHDRFrame() {
	suite.Run("should delegate to the backend", func() {
		suite.backendMock.EXPECT().BeginHDRFrame(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.r.BeginHDRFrame(nil, nil, nil, 1)
	})
}

func (suite *rendererImplTest) TestCreateCompositionTextures() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateCompositionTextures(mock.Anything, mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil).Once()
		hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex := suite.r.CreateCompositionTextures(1920, 1080, 1)
		suite.Nil(hdrView)
		suite.Nil(hdrTex)
		suite.Nil(msaaView)
		suite.Nil(msaaTex)
		suite.Nil(depthView)
		suite.Nil(depthTex)
	})
}

func (suite *rendererImplTest) TestCreateSSRTextures() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateSSRTextures(mock.Anything, mock.Anything).Return(nil, nil).Once()
		ssrView, ssrTex := suite.r.CreateSSRTextures(1920, 1080)
		suite.Nil(ssrView)
		suite.Nil(ssrTex)
	})
}

func (suite *rendererImplTest) TestCreateContactShadowTextures() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateContactShadowTextures(mock.Anything, mock.Anything).Return(nil, nil).Once()
		csView, csTex := suite.r.CreateContactShadowTextures(1920, 1080)
		suite.Nil(csView)
		suite.Nil(csTex)
	})
}

func (suite *rendererImplTest) TestCreateBloomTextures() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateBloomTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil, nil, 0, nil).Once()
		downTex, downReadViews, downStorageViews, upTex, upReadViews, upStorageViews, upMip0View, mipCount, err := suite.r.CreateBloomTextures(1920, 1080)
		suite.NoError(err)
		suite.Nil(downTex)
		suite.Nil(downReadViews)
		suite.Nil(downStorageViews)
		suite.Nil(upTex)
		suite.Nil(upReadViews)
		suite.Nil(upStorageViews)
		suite.Nil(upMip0View)
		suite.Equal(0, mipCount)
	})
}

func (suite *rendererImplTest) TestCreateHiZTextures() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateHiZTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, 0).Once()
		hizView, hizTex, mipReadViews, mipStorageViews, mipCount := suite.r.CreateHiZTextures(1920, 1080)
		suite.Nil(hizView)
		suite.Nil(hizTex)
		suite.Nil(mipReadViews)
		suite.Nil(mipStorageViews)
		suite.Equal(0, mipCount)
	})
}

func (suite *rendererImplTest) TestCreateTAATextures() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateTAATextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil).Once()
		view0, tex0, view1, tex1 := suite.r.CreateTAATextures(1920, 1080)
		suite.Nil(view0)
		suite.Nil(tex0)
		suite.Nil(view1)
		suite.Nil(tex1)
	})
}

func (suite *rendererImplTest) TestCreateSharpenTexture() {
	suite.Run("should delegate to the backend and return results", func() {
		suite.backendMock.EXPECT().CreateSharpenTexture(mock.Anything, mock.Anything).Return(nil, nil).Once()
		view, tex := suite.r.CreateSharpenTexture(1920, 1080)
		suite.Nil(view)
		suite.Nil(tex)
	})
}

func (suite *rendererImplTest) TestNewRendererNilWindowPanics() {
	suite.Run("should panic when constructing a renderer with a nil window", func() {
		suite.Panics(func() {
			renderer.NewRenderer(renderer.BackendTypeWGPU, nil)
		})
	})

	suite.Run("should still panic with nil window when builder options are provided", func() {
		suite.Panics(func() {
			renderer.NewRenderer(
				renderer.BackendTypeWGPU,
				nil,
				renderer.WithGPUSerializedProfiling(true),
				renderer.WithForceSoftwareRenderer(true),
			)
		})
	})

	suite.Run("should still panic with nil window when msaa option is provided", func() {
		suite.Panics(func() {
			renderer.NewRenderer(renderer.BackendTypeWGPU, nil, renderer.WithMSAA(renderer.MSAA8x))
		})
	})
}

func (suite *rendererImplTest) TestWithPipeline() {
	suite.Run("should add the pipeline to the cache under the given key", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		renderer.ApplyOption(suite.r, renderer.WithPipeline("key", mockPipeline))
		suite.Equal(mockPipeline, suite.r.Pipeline("key"))
	})
}

func (suite *rendererImplTest) TestWithPipelines() {
	suite.Run("should replace the pipeline cache with the provided map", func() {
		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		newCache := map[string]pipeline.Pipeline{"k": mockPipeline}
		renderer.ApplyOption(suite.r, renderer.WithPipelines(newCache))
		suite.Equal(newCache, suite.r.Pipelines())
	})
}

func (suite *rendererImplTest) TestWithPresentMode() {
	suite.Run("should set the pending present mode", func() {
		renderer.ApplyOption(suite.r, renderer.WithPresentMode(renderer.PresentModeVSync))
		mode := renderer.RendererPendingPresentMode(suite.r)
		suite.NotNil(mode)
		suite.Equal(renderer.PresentModeVSync, *mode)
	})
}

func (suite *rendererImplTest) TestWithMSAA() {
	suite.Run("should set the pending MSAA sample count", func() {
		renderer.ApplyOption(suite.r, renderer.WithMSAA(renderer.MSAAOff))
		msaa := renderer.RendererPendingMSAA(suite.r)
		suite.NotNil(msaa)
		suite.Equal(renderer.MSAAOff, *msaa)
	})
}

func (suite *rendererImplTest) TestWithForceSoftwareRenderer() {
	suite.Run("should set the force fallback adapter flag to true", func() {
		renderer.ApplyOption(suite.r, renderer.WithForceSoftwareRenderer(true))
		suite.True(renderer.RendererForceFallbackAdapter(suite.r))
	})
	suite.Run("should set the force fallback adapter flag to false", func() {
		renderer.ApplyOption(suite.r, renderer.WithForceSoftwareRenderer(false))
		suite.False(renderer.RendererForceFallbackAdapter(suite.r))
	})
}

func (suite *rendererImplTest) TestWithGPUSerializedProfiling() {
	suite.Run("should set gpu serialized profiling to true", func() {
		renderer.ApplyOption(suite.r, renderer.WithGPUSerializedProfiling(true))
		suite.True(renderer.RendererGPUSerializedProfiling(suite.r))
	})

	suite.Run("should set gpu serialized profiling to false", func() {
		renderer.ApplyOption(suite.r, renderer.WithGPUSerializedProfiling(false))
		suite.False(renderer.RendererGPUSerializedProfiling(suite.r))
	})

	suite.Run("last applied serialized profiling option wins", func() {
		renderer.ApplyOption(suite.r, renderer.WithGPUSerializedProfiling(true))
		renderer.ApplyOption(suite.r, renderer.WithGPUSerializedProfiling(false))

		suite.False(renderer.RendererGPUSerializedProfiling(suite.r))
	})
}
