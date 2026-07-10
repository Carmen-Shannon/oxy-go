package renderer

import (
	"sync"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	shadermocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/shader/mocks"
)

type wgpuRendererBackendTest struct {
	suite.Suite
}

func TestRunWGPURendererBackendTests(t *testing.T) {
	suite.Run(t, new(wgpuRendererBackendTest))
}

func (suite *wgpuRendererBackendTest) TestSampleAndRenderTargetState() {
	suite.Run("sample count getter returns stored msaa value", func() {
		backend := newTestWGPURendererBackend()
		backend.sampleCount = MSAA8x

		suite.Equal(uint32(MSAA8x), backend.SampleCount())
	})

	suite.Run("max texture dimension getter returns stored value", func() {
		backend := newTestWGPURendererBackend()
		backend.maxTextureDimension2D = 16384

		suite.Equal(uint32(16384), backend.MaxTextureDimension2D())
	})

	suite.Run("set render target format stores non nil pointer with exact format", func() {
		backend := newTestWGPURendererBackend()

		backend.SetRenderTargetFormat(wgpu.TextureFormatRGBA16Float)

		suite.NotNil(backend.renderTargetFormat)
		suite.Equal(wgpu.TextureFormatRGBA16Float, *backend.renderTargetFormat)
	})
}

func (suite *wgpuRendererBackendTest) TestSetPresentMode() {
	suite.Run("vsync maps to fifo", func() {
		backend := newTestWGPURendererBackend()

		backend.SetPresentMode(PresentModeVSync)

		suite.Equal(wgpu.PresentModeFifo, backend.presentMode)
	})

	suite.Run("mailbox maps to mailbox", func() {
		backend := newTestWGPURendererBackend()

		backend.SetPresentMode(PresentModeMailbox)

		suite.Equal(wgpu.PresentModeMailbox, backend.presentMode)
	})

	suite.Run("uncapped maps to immediate", func() {
		backend := newTestWGPURendererBackend()

		backend.SetPresentMode(PresentModeUncapped)

		suite.Equal(wgpu.PresentModeImmediate, backend.presentMode)
	})

	suite.Run("unknown present mode maps to immediate", func() {
		backend := newTestWGPURendererBackend()

		backend.SetPresentMode(PresentMode(999))

		suite.Equal(wgpu.PresentModeImmediate, backend.presentMode)
	})
}

func (suite *wgpuRendererBackendTest) TestComputeGuards() {
	suite.Run("begin compute frame nested call increments depth and returns nil", func() {
		backend := newTestWGPURendererBackend()
		backend.computeFrameDepth = 1

		backend.BeginComputeFrame()

		suite.Equal(2, backend.computeFrameDepth)
	})

	suite.Run("end compute frame with nil encoder resets depth to zero", func() {
		backend := newTestWGPURendererBackend()
		backend.computeFrameDepth = 4

		backend.EndComputeFrame()

		suite.Equal(0, backend.computeFrameDepth)
	})

	suite.Run("end compute frame nested depth decrements and retains encoder", func() {
		backend := newTestWGPURendererBackend()
		encoder := new(wgpu.CommandEncoder)
		backend.computeFrameEncoder = encoder
		backend.computeFrameDepth = 2

		backend.EndComputeFrame()

		suite.Equal(1, backend.computeFrameDepth)
		suite.Equal(encoder, backend.computeFrameEncoder)
	})

	suite.Run("copy buffer to buffer returns safely when compute encoder is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.CopyBufferToBuffer(nil, nil, 0, 0, 0)
		})
	})

	suite.Run("dispatch compute batch returns safely when compute encoder is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.DispatchComputeBatch([]ComputeDispatchEntry{{}})
		})
	})

	suite.Run("dispatch compute batch returns safely when dispatch list is empty", func() {
		backend := newTestWGPURendererBackend()
		backend.computeFrameEncoder = new(wgpu.CommandEncoder)

		suite.NotPanics(func() {
			backend.DispatchComputeBatch([]ComputeDispatchEntry{})
		})
	})

	suite.Run("end compute frame with gpu serialized profiling and nil encoder resets depth to zero", func() {
		backend := newTestWGPURendererBackend()
		backend.gpuSerializedProfiling = true
		backend.computeFrameDepth = 4

		backend.EndComputeFrame()

		suite.Equal(0, backend.computeFrameDepth)
	})

	suite.Run("end compute frame with gpu serialized profiling and nested depth decrements and retains encoder", func() {
		backend := newTestWGPURendererBackend()
		backend.gpuSerializedProfiling = true
		encoder := new(wgpu.CommandEncoder)
		backend.computeFrameEncoder = encoder
		backend.computeFrameDepth = 2

		backend.EndComputeFrame()

		suite.Equal(1, backend.computeFrameDepth)
		suite.Equal(encoder, backend.computeFrameEncoder)
	})

	suite.Run("begin compute frame triple nested call increments depth to three", func() {
		backend := newTestWGPURendererBackend()
		backend.computeFrameDepth = 1

		backend.BeginComputeFrame()
		backend.BeginComputeFrame()

		suite.Equal(3, backend.computeFrameDepth)
	})
}

func (suite *wgpuRendererBackendTest) TestFrameAndPresentGuards() {
	suite.Run("begin frame returns error when previous frame surface exists", func() {
		backend := newTestWGPURendererBackend()
		backend.frameSurface = new(wgpu.Texture)

		err := backend.BeginFrame()

		suite.EqualError(err, "previous frame surface not yet presented")
	})

	suite.Run("present returns safely when both frame and composition surfaces are nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.Present()
		})
	})

	suite.Run("begin composition frame returns error when previous composition surface exists", func() {
		backend := newTestWGPURendererBackend()
		backend.compositionSurface = new(wgpu.Texture)

		err := backend.BeginCompositionFrame()

		suite.EqualError(err, "previous composition frame surface not yet presented")
	})
}

func (suite *wgpuRendererBackendTest) TestGeometryFrameGuards() {
	suite.Run("begin geometry frame nested call increments depth and returns nil", func() {
		backend := newTestWGPURendererBackend()
		backend.geometryFrameDepth = 1

		backend.BeginGeometryFrame()

		suite.Equal(2, backend.geometryFrameDepth)
	})

	suite.Run("end geometry frame returns immediately when depth is non positive", func() {
		backend := newTestWGPURendererBackend()
		backend.geometryFrameDepth = 0

		backend.EndGeometryFrame()

		suite.Equal(0, backend.geometryFrameDepth)
	})

	suite.Run("end geometry frame decrements and returns when depth remains positive", func() {
		backend := newTestWGPURendererBackend()
		backend.geometryFrameDepth = 2

		backend.EndGeometryFrame()

		suite.Equal(1, backend.geometryFrameDepth)
	})

	suite.Run("end geometry frame returns when depth reaches zero and encoder is nil", func() {
		backend := newTestWGPURendererBackend()
		backend.geometryFrameDepth = 1

		backend.EndGeometryFrame()

		suite.Equal(0, backend.geometryFrameDepth)
		suite.Nil(backend.geometryFrameEncoder)
	})

	suite.Run("end geometry frame with gpu serialized profiling returns immediately when depth is non positive", func() {
		backend := newTestWGPURendererBackend()
		backend.gpuSerializedProfiling = true
		backend.geometryFrameDepth = 0

		backend.EndGeometryFrame()

		suite.Equal(0, backend.geometryFrameDepth)
	})

	suite.Run("end geometry frame with gpu serialized profiling decrements and returns when depth remains positive", func() {
		backend := newTestWGPURendererBackend()
		backend.gpuSerializedProfiling = true
		backend.geometryFrameDepth = 2

		backend.EndGeometryFrame()

		suite.Equal(1, backend.geometryFrameDepth)
	})
}

func (suite *wgpuRendererBackendTest) TestShadowFrameGuards() {
	suite.Run("begin shadow frame aliases geometry encoder when geometry frame is active", func() {
		backend := newTestWGPURendererBackend()
		backend.geometryFrameEncoder = new(wgpu.CommandEncoder)

		backend.BeginShadowFrame()

		suite.Equal(backend.geometryFrameEncoder, backend.shadowFrameEncoder)
	})

	suite.Run("shadow draw call returns safely when shadow pass is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.ShadowDrawCall(nil, nil, 1, nil)
		})
	})

	suite.Run("shadow draw call indirect returns safely when shadow pass is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.ShadowDrawCallIndirect(nil, nil, nil, nil)
		})
	})

	suite.Run("end shadow frame returns safely when shadow encoder is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.EndShadowFrame()
		})
	})

	suite.Run("end shadow frame clears alias when geometry frame encoder is active", func() {
		backend := newTestWGPURendererBackend()
		shared := new(wgpu.CommandEncoder)
		backend.geometryFrameEncoder = shared
		backend.shadowFrameEncoder = shared

		backend.EndShadowFrame()

		suite.Nil(backend.shadowFrameEncoder)
		suite.Equal(shared, backend.geometryFrameEncoder)
	})

	suite.Run("begin shadow atlas pass returns safely when shadow frame encoder is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.BeginShadowAtlasPass(nil)
		})
	})

	suite.Run("set shadow viewport returns safely when shadow pass is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.SetShadowViewport(0, 0, 128, 128)
		})
	})

	suite.Run("end shadow atlas pass returns safely when shadow pass is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.EndShadowAtlasPass()
		})
	})
}

func (suite *wgpuRendererBackendTest) TestGBufferGuards() {
	suite.Run("begin gbuffer frame aliases geometry encoder when geometry frame is active", func() {
		backend := newTestWGPURendererBackend()
		backend.geometryFrameEncoder = new(wgpu.CommandEncoder)

		backend.BeginGBufferFrame()

		suite.Equal(backend.geometryFrameEncoder, backend.gbufferFrameEncoder)
	})

	suite.Run("begin gbuffer pass returns safely when gbuffer frame encoder is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.BeginGBufferPass(nil, nil, nil)
		})
	})

	suite.Run("gbuffer draw call returns safely when gbuffer pass is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.GBufferDrawCall(nil, nil, 1, nil)
		})
	})

	suite.Run("gbuffer draw call indirect returns safely when gbuffer pass is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.GBufferDrawCallIndirect(nil, nil, nil, nil)
		})
	})

	suite.Run("end gbuffer pass returns safely when gbuffer pass is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.EndGBufferPass()
		})
	})

	suite.Run("end gbuffer frame returns safely when frame encoder is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.EndGBufferFrame()
		})
	})

	suite.Run("end gbuffer frame clears alias when geometry frame encoder is active", func() {
		backend := newTestWGPURendererBackend()
		shared := new(wgpu.CommandEncoder)
		backend.geometryFrameEncoder = shared
		backend.gbufferFrameEncoder = shared

		backend.EndGBufferFrame()

		suite.Nil(backend.gbufferFrameEncoder)
		suite.Equal(shared, backend.geometryFrameEncoder)
	})
}

func (suite *wgpuRendererBackendTest) TestCompositionGuards() {
	suite.Run("begin composition pass returns safely when composition frame encoder is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.BeginCompositionPass()
		})
	})

	suite.Run("composition draw call returns safely when composition pass is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.CompositionDrawCall(nil, nil)
		})
	})

	suite.Run("end composition pass returns safely when composition pass is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.EndCompositionPass()
		})
	})

	suite.Run("end composition frame returns safely when composition frame encoder is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.EndCompositionFrame()
		})
	})
}

func (suite *wgpuRendererBackendTest) TestPipelineRegistrationValidationErrors() {
	suite.Run("register render pipeline errors when vertex and fragment shaders are missing", func() {
		backend := newTestWGPURendererBackend()
		renderPipeline := pipeline.NewPipeline("render-missing-both", pipeline.PipelineTypeRender)

		err := backend.RegisterRenderPipeline(renderPipeline)

		suite.EqualError(err, "both vertex and fragment shaders must be set to create a render pipeline")
	})

	suite.Run("register render pipeline errors when fragment shader is missing", func() {
		backend := newTestWGPURendererBackend()
		vertexShader := shadermocks.NewMockShader(suite.T())
		renderPipeline := pipeline.NewPipeline(
			"render-missing-fragment",
			pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vertexShader),
		)

		err := backend.RegisterRenderPipeline(renderPipeline)

		suite.EqualError(err, "both vertex and fragment shaders must be set to create a render pipeline")
	})

	suite.Run("register compute pipeline errors when compute shader is missing", func() {
		backend := newTestWGPURendererBackend()
		computePipeline := pipeline.NewPipeline("compute-missing", pipeline.PipelineTypeCompute)

		err := backend.RegisterComputePipeline(computePipeline)

		suite.EqualError(err, "compute shader must be set to create a compute pipeline")
	})

	suite.Run("register shadow depth pipeline errors when vertex shader is missing", func() {
		backend := newTestWGPURendererBackend()
		shadowPipeline := pipeline.NewPipeline("shadow-missing-vertex", pipeline.PipelineTypeRender)

		err := backend.RegisterShadowDepthPipeline(shadowPipeline)

		suite.EqualError(err, "vertex shader must be set to create a shadow depth pipeline")
	})

	suite.Run("register gbuffer pipeline errors when vertex shader is missing", func() {
		backend := newTestWGPURendererBackend()
		gbufferPipeline := pipeline.NewPipeline("gbuffer-missing-vertex", pipeline.PipelineTypeRender)

		err := backend.RegisterGBufferPipeline(gbufferPipeline)

		suite.EqualError(err, "vertex shader must be set to create a G-Buffer pipeline")
	})

	suite.Run("register gbuffer pipeline errors when fragment shader is missing", func() {
		backend := newTestWGPURendererBackend()
		vertexShader := shadermocks.NewMockShader(suite.T())
		gbufferPipeline := pipeline.NewPipeline(
			"gbuffer-missing-fragment",
			pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vertexShader),
		)

		err := backend.RegisterGBufferPipeline(gbufferPipeline)

		suite.EqualError(err, "fragment shader must be set to create a G-Buffer pipeline")
	})

	suite.Run("register composition pipeline errors when vertex shader is missing", func() {
		backend := newTestWGPURendererBackend()
		compositionPipeline := pipeline.NewPipeline("composition-missing-vertex", pipeline.PipelineTypeRender)

		err := backend.RegisterCompositionPipeline(compositionPipeline)

		suite.EqualError(err, "vertex shader must be set to create a composition pipeline")
	})

	suite.Run("register composition pipeline errors when fragment shader is missing", func() {
		backend := newTestWGPURendererBackend()
		vertexShader := shadermocks.NewMockShader(suite.T())
		compositionPipeline := pipeline.NewPipeline(
			"composition-missing-fragment",
			pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vertexShader),
		)

		err := backend.RegisterCompositionPipeline(compositionPipeline)

		suite.EqualError(err, "fragment shader must be set to create a composition pipeline")
	})
}

func (suite *wgpuRendererBackendTest) TestCreateBloomTexturesValidation() {
	suite.Run("width zero returns validation error and nil outputs", func() {
		backend := newTestWGPURendererBackend()

		downTex, downReadViews, downStorageViews, upTex, upReadViews, upStorageViews, upMip0View, mipCount, err :=
			backend.CreateBloomTextures(0, 128)

		suite.EqualError(err, "bloom texture dimensions must be non-zero")
		suite.Nil(downTex)
		suite.Nil(downReadViews)
		suite.Nil(downStorageViews)
		suite.Nil(upTex)
		suite.Nil(upReadViews)
		suite.Nil(upStorageViews)
		suite.Nil(upMip0View)
		suite.Equal(0, mipCount)
	})

	suite.Run("height zero returns validation error and nil outputs", func() {
		backend := newTestWGPURendererBackend()

		downTex, downReadViews, downStorageViews, upTex, upReadViews, upStorageViews, upMip0View, mipCount, err :=
			backend.CreateBloomTextures(128, 0)

		suite.EqualError(err, "bloom texture dimensions must be non-zero")
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

func (suite *wgpuRendererBackendTest) TestBindGroupAndBufferNoGPUBranches() {
	suite.Run("init bind group returns nil when descriptor has no entries", func() {
		backend := newTestWGPURendererBackend()
		provider := bind_group_provider.NewBindGroupProvider("empty-descriptor")

		err := backend.InitBindGroup(provider, wgpu.BindGroupLayoutDescriptor{}, nil, nil)

		suite.NoError(err)
		suite.Nil(provider.BindGroup())
	})

	suite.Run("init bind group returns validation error when sampled texture view is missing", func() {
		backend := newTestWGPURendererBackend()
		provider := bind_group_provider.NewBindGroupProvider("missing-sampled-view")
		provider.SetBindGroupLayout(new(wgpu.BindGroupLayout))

		err := backend.InitBindGroup(provider, wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding: 3,
					Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat},
				},
			},
		}, nil, nil)

		suite.ErrorContains(err, "texture binding 3 has no texture view")
		suite.ErrorContains(err, "call InitTextureView first")
	})

	suite.Run("init bind group returns validation error when storage texture view is missing", func() {
		backend := newTestWGPURendererBackend()
		provider := bind_group_provider.NewBindGroupProvider("missing-storage-view")
		provider.SetBindGroupLayout(new(wgpu.BindGroupLayout))

		err := backend.InitBindGroup(provider, wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:        4,
					StorageTexture: wgpu.StorageTextureBindingLayout{Access: wgpu.StorageTextureAccess(1)},
				},
			},
		}, nil, nil)

		suite.ErrorContains(err, "texture binding 4 has no texture view")
		suite.ErrorContains(err, "call InitTextureView first")
	})

	suite.Run("init bind group returns validation error when sampler is missing", func() {
		backend := newTestWGPURendererBackend()
		provider := bind_group_provider.NewBindGroupProvider("missing-sampler")
		provider.SetBindGroupLayout(new(wgpu.BindGroupLayout))

		err := backend.InitBindGroup(provider, wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding: 5,
					Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering},
				},
			},
		}, nil, nil)

		suite.ErrorContains(err, "sampler binding 5 has no sampler")
		suite.ErrorContains(err, "call InitSampler first")
	})

	suite.Run("init mesh buffers with empty data still sets index count", func() {
		backend := newTestWGPURendererBackend()
		provider := bind_group_provider.NewBindGroupProvider("empty-mesh")

		err := backend.InitMeshBuffers(provider, nil, nil, 7)

		suite.NoError(err)
		suite.Nil(provider.VertexBuffer())
		suite.Nil(provider.IndexBuffer())
		suite.Equal(7, provider.IndexCount())
	})

	suite.Run("write buffers skips nil provider buffer without panic", func() {
		backend := newTestWGPURendererBackend()
		provider := bind_group_provider.NewBindGroupProvider("nil-write-buffer")

		suite.NotPanics(func() {
			backend.WriteBuffers([]bind_group_provider.BufferWrite{
				{Provider: provider, Binding: 9, Offset: 0, Data: []byte{1, 2, 3}},
			})
		})
	})

	suite.Run("write raw buffer returns when buffer is nil", func() {
		backend := newTestWGPURendererBackend()

		suite.NotPanics(func() {
			backend.WriteRawBuffer(nil, 0, []byte{1})
		})
	})
}

func (suite *wgpuRendererBackendTest) TestFlushAndTimingHelpers() {
	suite.Run("flush frame with no pending command buffers returns zero and advances slot", func() {
		backend := newTestWGPURendererBackend()
		backend.frameInFlightCount = 2
		backend.currentFrameSlot = 1

		index := backend.FlushFrame()

		suite.Equal(wgpu.SubmissionIndex(0), index)
		suite.Equal(0, backend.currentFrameSlot)
	})

	suite.Run("sync gpu timestamps returns safely when current slot has no submission", func() {
		backend := newTestWGPURendererBackend()
		backend.currentFrameSlot = 1
		backend.slotSubmitValid = [2]bool{true, false}

		suite.NotPanics(func() {
			backend.SyncGPUTimestamps()
		})
	})

	suite.Run("gpu timings returns nil", func() {
		backend := newTestWGPURendererBackend()

		suite.Nil(backend.GPUTimings())
	})

	suite.Run("current frame slot returns stored slot", func() {
		backend := newTestWGPURendererBackend()
		backend.currentFrameSlot = 1

		suite.Equal(1, backend.CurrentFrameSlot())
	})

	suite.Run("flush frame with no pending wraps slot from last index back to zero", func() {
		backend := newTestWGPURendererBackend()
		backend.frameInFlightCount = 2
		backend.currentFrameSlot = 1

		index := backend.FlushFrame()

		suite.Equal(wgpu.SubmissionIndex(0), index)
		suite.Equal(0, backend.currentFrameSlot)
	})

	suite.Run("flush frame with no pending advances slot from zero to one", func() {
		backend := newTestWGPURendererBackend()
		backend.frameInFlightCount = 2
		backend.currentFrameSlot = 0

		index := backend.FlushFrame()

		suite.Equal(wgpu.SubmissionIndex(0), index)
		suite.Equal(1, backend.currentFrameSlot)
	})

	suite.Run("flush frame with no pending leaves slot submit state untouched", func() {
		backend := newTestWGPURendererBackend()
		backend.frameInFlightCount = 2
		backend.currentFrameSlot = 0
		backend.slotSubmitValid = [2]bool{true, false}
		backend.slotSubmitIndex = [2]wgpu.SubmissionIndex{42, 0}

		backend.FlushFrame()

		suite.True(backend.slotSubmitValid[0])
		suite.Equal(wgpu.SubmissionIndex(42), backend.slotSubmitIndex[0])
		suite.False(backend.slotSubmitValid[1])
		suite.Equal(1, backend.currentFrameSlot)
	})
}

func (suite *wgpuRendererBackendTest) TestMergeBindGroupLayouts() {
	suite.Run("nil vertex and fragment maps return an empty merged map", func() {
		merged := mergeBindGroupLayouts(nil, nil)

		suite.NotNil(merged)
		suite.Len(merged, 0)
	})

	suite.Run("vertex only group is preserved", func() {
		vertexLayouts := map[int]wgpu.BindGroupLayoutDescriptor{
			0: {
				Label: "vertex-only",
				Entries: []wgpu.BindGroupLayoutEntry{
					{Binding: 2, Visibility: wgpu.ShaderStageVertex},
				},
			},
		}

		merged := mergeBindGroupLayouts(vertexLayouts, nil)
		desc, ok := merged[0]

		suite.True(ok)
		suite.Equal("vertex-only", desc.Label)
		suite.Len(desc.Entries, 1)
		suite.Equal(uint32(2), desc.Entries[0].Binding)
		suite.Equal(wgpu.ShaderStageVertex, desc.Entries[0].Visibility)
	})

	suite.Run("fragment only group is preserved", func() {
		fragmentLayouts := map[int]wgpu.BindGroupLayoutDescriptor{
			1: {
				Label: "fragment-only",
				Entries: []wgpu.BindGroupLayoutEntry{
					{Binding: 7, Visibility: wgpu.ShaderStageFragment},
				},
			},
		}

		merged := mergeBindGroupLayouts(nil, fragmentLayouts)
		desc, ok := merged[1]

		suite.True(ok)
		suite.Equal("fragment-only", desc.Label)
		suite.Len(desc.Entries, 1)
		suite.Equal(uint32(7), desc.Entries[0].Binding)
		suite.Equal(wgpu.ShaderStageFragment, desc.Entries[0].Visibility)
	})

	suite.Run("overlapping bindings have OR visibility and merged entries are sorted", func() {
		vertexLayouts := map[int]wgpu.BindGroupLayoutDescriptor{
			2: {
				Label: "vertex-overlap",
				Entries: []wgpu.BindGroupLayoutEntry{
					{Binding: 5, Visibility: wgpu.ShaderStageVertex},
					{Binding: 1, Visibility: wgpu.ShaderStageVertex},
				},
			},
		}
		fragmentLayouts := map[int]wgpu.BindGroupLayoutDescriptor{
			2: {
				Label: "fragment-overlap",
				Entries: []wgpu.BindGroupLayoutEntry{
					{Binding: 1, Visibility: wgpu.ShaderStageFragment},
					{Binding: 3, Visibility: wgpu.ShaderStageFragment},
				},
			},
		}

		merged := mergeBindGroupLayouts(vertexLayouts, fragmentLayouts)
		desc, ok := merged[2]

		suite.True(ok)
		suite.Len(desc.Entries, 3)
		suite.Equal(uint32(1), desc.Entries[0].Binding)
		suite.Equal(uint32(3), desc.Entries[1].Binding)
		suite.Equal(uint32(5), desc.Entries[2].Binding)
		suite.Equal(wgpu.ShaderStageVertex|wgpu.ShaderStageFragment, desc.Entries[0].Visibility)
		suite.Equal(wgpu.ShaderStageFragment, desc.Entries[1].Visibility)
		suite.Equal(wgpu.ShaderStageVertex, desc.Entries[2].Visibility)
	})
}

func newTestWGPURendererBackend() *wgpuRendererBackendImpl {
	return &wgpuRendererBackendImpl{
		mu:                 &sync.Mutex{},
		frameInFlightCount: 2,
	}
}
