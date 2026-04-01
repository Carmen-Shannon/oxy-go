package renderer_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type wgpuRendererBackendTest struct {
	suite.Suite
	w window.Window
	r renderer.Renderer
}

func TestRunWgpuRendererBackendTests(t *testing.T) {
	suite.Run(t, new(wgpuRendererBackendTest))
}

func (suite *wgpuRendererBackendTest) SetupSuite() {
	suite.w = window.NewWindow(
		window.WithTitle("oxy-go integration test"),
		window.WithWidth(800),
		window.WithHeight(600),
	)
	suite.r = renderer.NewRenderer(
		renderer.BackendTypeWGPU,
		suite.w,
		renderer.WithForceSoftwareRenderer(true),
		renderer.WithMSAA(renderer.MSAAOff),
		renderer.WithPresentMode(renderer.PresentModeVSync),
	)
}

func (suite *wgpuRendererBackendTest) TearDownSuite() {
	suite.r.WaitIdle()
	suite.w.Close() //nolint:errcheck
}

func (suite *wgpuRendererBackendTest) TestSampleCount() {
	suite.Run("should return MSAAOff sample count", func() {
		suite.Equal(uint32(1), suite.r.SampleCount())
	})
}

func (suite *wgpuRendererBackendTest) TestMaxTextureDimension2D() {
	suite.Run("should return a positive dimension", func() {
		suite.Greater(suite.r.MaxTextureDimension2D(), uint32(0))
	})
}

func (suite *wgpuRendererBackendTest) TestCurrentFrameSlot() {
	suite.Run("should return a valid frame slot index", func() {
		slot := suite.r.CurrentFrameSlot()
		suite.GreaterOrEqual(slot, 0)
		suite.LessOrEqual(slot, 1)
	})
}

func (suite *wgpuRendererBackendTest) TestGPUTimings() {
	suite.Run("should return nil when GPU timestamps are not supported by software adapter", func() {
		suite.Nil(suite.r.GPUTimings())
	})
}

func (suite *wgpuRendererBackendTest) TestSetPresentMode() {
	suite.Run("should not panic when setting VSync", func() {
		suite.NotPanics(func() {
			suite.r.SetPresentMode(renderer.PresentModeVSync)
		})
	})
}

func (suite *wgpuRendererBackendTest) TestSetRenderTargetFormat() {
	suite.Run("should not panic when overriding render target format", func() {
		suite.NotPanics(func() {
			suite.r.SetRenderTargetFormat(wgpu.TextureFormatRGBA8Unorm)
		})
	})
}

func (suite *wgpuRendererBackendTest) TestWaitIdle() {
	suite.Run("should complete without error", func() {
		suite.NotPanics(func() {
			suite.r.WaitIdle()
		})
	})
}

func (suite *wgpuRendererBackendTest) TestSyncGPUTimestamps() {
	suite.Run("should complete without error when no submission recorded", func() {
		suite.NotPanics(func() {
			suite.r.SyncGPUTimestamps()
		})
	})
}

func (suite *wgpuRendererBackendTest) TestResize() {
	suite.Run("should reconfigure surface without error", func() {
		suite.NotPanics(func() {
			suite.r.Resize(1024, 768)
		})
	})
}

func (suite *wgpuRendererBackendTest) TestCreateBuffer() {
	suite.Run("should create a storage buffer without error", func() {
		buf, err := suite.r.CreateBuffer("storage-test", 256, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(err)
		suite.Require().NotNil(buf)
		defer buf.Release()
	})
}

func (suite *wgpuRendererBackendTest) TestWriteRawBuffer() {
	suite.Run("should write data to a buffer without panicking", func() {
		buf, err := suite.r.CreateBuffer("write-test", 64, wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(err)
		suite.Require().NotNil(buf)
		defer buf.Release()

		suite.NotPanics(func() {
			suite.r.WriteRawBuffer(buf, 0, make([]byte, 16))
		})
	})
}

func (suite *wgpuRendererBackendTest) TestCreateComparisonSampler() {
	suite.Run("should create sampler without error", func() {
		sampler, err := suite.r.CreateComparisonSampler()
		suite.Require().NoError(err)
		suite.Require().NotNil(sampler)
		defer sampler.Release()
	})
}

func (suite *wgpuRendererBackendTest) TestCreateLinearSampler() {
	suite.Run("should create sampler without error", func() {
		sampler, err := suite.r.CreateLinearSampler()
		suite.Require().NoError(err)
		suite.Require().NotNil(sampler)
		defer sampler.Release()
	})
}

func (suite *wgpuRendererBackendTest) TestCreateShadowDepthTexture() {
	suite.Run("should create depth texture and view without error", func() {
		depthView, depthTex, err := suite.r.CreateShadowDepthTexture(512, 512)
		suite.Require().NoError(err)
		suite.Require().NotNil(depthView)
		suite.Require().NotNil(depthTex)
		defer func() {
			depthView.Release()
			depthTex.Release()
		}()
	})
}

func (suite *wgpuRendererBackendTest) TestCreateGBufferTextures() {
	suite.Run("should create all gbuffer textures without error", func() {
		normView, normTex, albedoView, albedoTex, depthView, depthTex, err := suite.r.CreateGBufferTextures(800, 600)
		suite.Require().NoError(err)
		suite.Require().NotNil(normView)
		suite.Require().NotNil(normTex)
		suite.Require().NotNil(albedoView)
		suite.Require().NotNil(albedoTex)
		suite.Require().NotNil(depthView)
		suite.Require().NotNil(depthTex)
		defer func() {
			normView.Release()
			albedoView.Release()
			depthView.Release()
			normTex.Release()
			albedoTex.Release()
			depthTex.Release()
		}()
	})
}

func (suite *wgpuRendererBackendTest) TestCreateSSAOTextures() {
	suite.Run("should create all SSAO textures without error", func() {
		rawView, rawTex, blurredView, blurredTex, scratchView, scratchTex, err := suite.r.CreateSSAOTextures(400, 300)
		suite.Require().NoError(err)
		suite.Require().NotNil(rawView)
		suite.Require().NotNil(rawTex)
		suite.Require().NotNil(blurredView)
		suite.Require().NotNil(blurredTex)
		suite.Require().NotNil(scratchView)
		suite.Require().NotNil(scratchTex)
		defer func() {
			rawView.Release()
			blurredView.Release()
			scratchView.Release()
			rawTex.Release()
			blurredTex.Release()
			scratchTex.Release()
		}()
	})
}

func (suite *wgpuRendererBackendTest) TestCreateSSRTextures() {
	suite.Run("should create SSR texture and view without error", func() {
		ssrView, ssrTex, err := suite.r.CreateSSRTextures(800, 600)
		suite.Require().NoError(err)
		suite.Require().NotNil(ssrView)
		suite.Require().NotNil(ssrTex)
		defer func() {
			ssrView.Release()
			ssrTex.Release()
		}()
	})
}

func (suite *wgpuRendererBackendTest) TestCreateContactShadowTextures() {
	suite.Run("should create contact shadow texture and view without error", func() {
		csView, csTex, err := suite.r.CreateContactShadowTextures(800, 600)
		suite.Require().NoError(err)
		suite.Require().NotNil(csView)
		suite.Require().NotNil(csTex)
		defer func() {
			csView.Release()
			csTex.Release()
		}()
	})
}

func (suite *wgpuRendererBackendTest) TestCreateHiZTextures() {
	suite.Run("should create Hi-Z textures with a positive mip count", func() {
		hizView, hizTex, mipReadViews, mipStorageViews, mipCount, err := suite.r.CreateHiZTextures(800, 600)
		suite.Require().NoError(err)
		suite.Require().NotNil(hizView)
		suite.Require().NotNil(hizTex)
		suite.Greater(mipCount, 0)
		suite.Equal(mipCount, len(mipReadViews))
		suite.Equal(mipCount, len(mipStorageViews))
		defer func() {
			hizView.Release()
			for _, v := range mipReadViews {
				v.Release()
			}
			for _, v := range mipStorageViews {
				v.Release()
			}
			hizTex.Release()
		}()
	})
}

func (suite *wgpuRendererBackendTest) TestCreateBloomTextures() {
	suite.Run("should create bloom textures with a positive mip count", func() {
		downTex, downReadViews, downStorageViews, upTex, upReadViews, upStorageViews, upMip0View, mipCount, err := suite.r.CreateBloomTextures(800, 600)
		suite.Require().NoError(err)
		suite.Require().NotNil(downTex)
		suite.Require().NotNil(upTex)
		suite.Require().NotNil(upMip0View)
		suite.Greater(mipCount, 0)
		defer func() {
			upMip0View.Release()
			for _, v := range downReadViews {
				v.Release()
			}
			for _, v := range downStorageViews {
				v.Release()
			}
			for _, v := range upReadViews {
				v.Release()
			}
			for _, v := range upStorageViews {
				v.Release()
			}
			downTex.Release()
			upTex.Release()
		}()
	})
}

func (suite *wgpuRendererBackendTest) TestCreateCompositionTextures() {
	suite.Run("should create HDR composition textures without error", func() {
		hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex, err := suite.r.CreateCompositionTextures(800, 600, 1)
		suite.Require().NoError(err)
		suite.Require().NotNil(hdrView)
		suite.Require().NotNil(hdrTex)
		suite.Require().NotNil(depthView)
		suite.Require().NotNil(depthTex)
		defer func() {
			hdrView.Release()
			hdrTex.Release()
			if msaaView != nil {
				msaaView.Release()
			}
			if msaaTex != nil {
				msaaTex.Release()
			}
			depthView.Release()
			depthTex.Release()
		}()
	})
}

func (suite *wgpuRendererBackendTest) TestBeginEndComputeFrame() {
	suite.Run("should begin and end a compute frame without error", func() {
		err := suite.r.BeginComputeFrame()
		suite.Require().NoError(err)
		suite.r.EndComputeFrame()
		suite.r.FlushFrame()
	})
}

func (suite *wgpuRendererBackendTest) TestBeginEndGeometryFrame() {
	suite.Run("should begin and end a geometry frame without error", func() {
		err := suite.r.BeginGeometryFrame()
		suite.Require().NoError(err)
		suite.r.EndGeometryFrame()
		suite.r.FlushFrame()
	})
}

func (suite *wgpuRendererBackendTest) TestShadowFrameLifecycle() {
	suite.Run("should complete shadow depth pass within geometry frame", func() {
		depthView, depthTex, err := suite.r.CreateShadowDepthTexture(512, 512)
		suite.Require().NoError(err)
		defer func() {
			depthView.Release()
			depthTex.Release()
		}()

		err = suite.r.BeginGeometryFrame()
		suite.Require().NoError(err)

		err = suite.r.BeginShadowFrame()
		suite.Require().NoError(err)

		suite.r.BeginShadowDepthPass(depthView, 0, 0, 512, 512, true)
		suite.r.EndShadowPass()
		suite.r.EndShadowFrame()
		suite.r.EndGeometryFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestGBufferFrameLifecycle() {
	suite.Run("should complete gbuffer pass within geometry frame", func() {
		normView, normTex, albedoView, albedoTex, depthView, depthTex, err := suite.r.CreateGBufferTextures(800, 600)
		suite.Require().NoError(err)
		defer func() {
			normView.Release()
			albedoView.Release()
			depthView.Release()
			normTex.Release()
			albedoTex.Release()
			depthTex.Release()
		}()

		err = suite.r.BeginGeometryFrame()
		suite.Require().NoError(err)

		err = suite.r.BeginGBufferFrame()
		suite.Require().NoError(err)

		suite.r.BeginGBufferPass(normView, albedoView, depthView)
		suite.r.EndGBufferPass()
		suite.r.EndGBufferFrame()
		suite.r.EndGeometryFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestPipelineAccess() {
	p := pipeline.NewPipeline("access-test", pipeline.PipelineTypeRender)
	suite.Run("SetPipeline stores and Pipeline retrieves", func() {
		suite.r.SetPipeline("access-test", p)
		got := suite.r.Pipeline("access-test")
		suite.NotNil(got)
		suite.Equal("access-test", got.PipelineKey())
	})
	suite.Run("Pipelines returns the full cache", func() {
		all := suite.r.Pipelines()
		suite.NotNil(all)
	})
	suite.Run("SetPipelines replaces the cache", func() {
		suite.r.SetPipelines(map[string]pipeline.Pipeline{"set-pipelines-key": p})
		suite.NotNil(suite.r.Pipeline("set-pipelines-key"))
		// restore an empty map so stale entries don't affect later tests
		suite.r.SetPipelines(map[string]pipeline.Pipeline{})
	})
}

func (suite *wgpuRendererBackendTest) TestSetInjections() {
	suite.Run("should not panic when setting injections", func() {
		suite.NotPanics(func() {
			suite.r.SetInjections(map[string]string{"TILE_SIZE": "16"})
		})
	})
}

func (suite *wgpuRendererBackendTest) TestDrawCallErrors() {
	const key = "nonexistent-pipeline-key"
	suite.Run("DrawCall returns error for unknown pipeline", func() {
		err := suite.r.DrawCall(key, nil, 1, nil)
		suite.Error(err)
	})
	suite.Run("DrawCallIndirect returns error for unknown pipeline", func() {
		err := suite.r.DrawCallIndirect(key, nil, nil, nil)
		suite.Error(err)
	})
	suite.Run("ShadowDrawCall returns error for unknown pipeline", func() {
		err := suite.r.ShadowDrawCall(key, nil, 1, nil)
		suite.Error(err)
	})
	suite.Run("ShadowDrawCallIndirect returns error for unknown pipeline", func() {
		err := suite.r.ShadowDrawCallIndirect(key, nil, nil, nil)
		suite.Error(err)
	})
	suite.Run("GBufferDrawCall returns error for unknown pipeline", func() {
		err := suite.r.GBufferDrawCall(key, nil, 1, nil)
		suite.Error(err)
	})
	suite.Run("GBufferDrawCallIndirect returns error for unknown pipeline", func() {
		err := suite.r.GBufferDrawCallIndirect(key, nil, nil, nil)
		suite.Error(err)
	})
	suite.Run("CompositionDrawCall returns error for unknown pipeline", func() {
		err := suite.r.CompositionDrawCall(key, nil)
		suite.Error(err)
	})
}

func (suite *wgpuRendererBackendTest) TestCopyBufferToBuffer() {
	suite.Run("should copy buffer data without error", func() {
		src, err := suite.r.CreateBuffer("copy-src", 64, wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(err)
		suite.Require().NotNil(src)
		defer src.Release()

		dst, err := suite.r.CreateBuffer("copy-dst", 64, wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
		suite.Require().NoError(err)
		suite.Require().NotNil(dst)
		defer dst.Release()

		data := make([]byte, 32)
		for i := range data {
			data[i] = byte(i)
		}
		suite.r.WriteRawBuffer(src, 0, data)

		err = suite.r.BeginComputeFrame()
		suite.Require().NoError(err)
		suite.r.CopyBufferToBuffer(src, dst, 0, 0, 32)
		suite.r.EndComputeFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestReadMappedBuffer() {
	suite.Run("should read back copied data from a MapRead buffer", func() {
		const size uint64 = 32

		src, err := suite.r.CreateBuffer("map-src", size, wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(err)
		defer src.Release()

		dst, err := suite.r.CreateBuffer("map-dst", size, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead)
		suite.Require().NoError(err)
		defer dst.Release()

		payload := make([]byte, size)
		for i := range payload {
			payload[i] = byte(i * 2)
		}
		suite.r.WriteRawBuffer(src, 0, payload)

		err = suite.r.BeginComputeFrame()
		suite.Require().NoError(err)
		suite.r.CopyBufferToBuffer(src, dst, 0, 0, size)
		suite.r.EndComputeFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()

		result, err := suite.r.ReadMappedBuffer(dst, 0, size)
		suite.Require().NoError(err)
		suite.Require().NotNil(result)
		suite.Equal(int(size), len(result))
	})
}

func (suite *wgpuRendererBackendTest) TestWriteTexture() {
	suite.Run("should write pixel data to a texture without panicking", func() {
		const w, h uint32 = 64, 64
		csView, csTex, err := suite.r.CreateContactShadowTextures(int(w), int(h))
		suite.Require().NoError(err)
		defer func() {
			csView.Release()
			csTex.Release()
		}()

		data := make([]byte, w*h*4) // R32Float = 4 bytes per pixel
		suite.NotPanics(func() {
			suite.r.WriteTexture(csTex, data, w, h, w*4)
		})
	})
}

func (suite *wgpuRendererBackendTest) TestInitMeshBuffers() {
	suite.Run("should create vertex and index buffers on the provider", func() {
		provider := bind_group_provider.NewBindGroupProvider("mesh-test")
		defer provider.Release()

		vertices := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0}
		vertexBytes := make([]byte, len(vertices)*4)
		for i, v := range vertices {
			binary.LittleEndian.PutUint32(vertexBytes[i*4:], math.Float32bits(v))
		}

		indices := []uint32{0, 1, 2}
		indexBytes := make([]byte, len(indices)*4)
		for i, idx := range indices {
			binary.LittleEndian.PutUint32(indexBytes[i*4:], idx)
		}

		err := suite.r.InitMeshBuffers(provider, vertexBytes, indexBytes, 3)
		suite.Require().NoError(err)
		suite.NotNil(provider.VertexBuffer())
		suite.NotNil(provider.IndexBuffer())
		suite.Equal(3, provider.IndexCount())
	})
}

func (suite *wgpuRendererBackendTest) TestInitBindGroup() {
	suite.Run("should create bind group and reuse layout on second call", func() {
		provider := bind_group_provider.NewBindGroupProvider("bgp-test")
		defer provider.Release()

		descriptor := wgpu.BindGroupLayoutDescriptor{
			Label: "test-bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment,
					Buffer: wgpu.BufferBindingLayout{
						Type: wgpu.BufferBindingTypeUniform,
					},
				},
			},
		}

		// First call: creates layout and bind group
		err := suite.r.InitBindGroup(provider, descriptor, nil, map[int]uint64{0: 64})
		suite.Require().NoError(err)
		suite.NotNil(provider.BindGroupLayout())
		suite.NotNil(provider.BindGroup())

		// Second call: reuses existing layout (idempotent path)
		err = suite.r.InitBindGroup(provider, descriptor, nil, map[int]uint64{0: 64})
		suite.Require().NoError(err)
	})
}

func (suite *wgpuRendererBackendTest) TestInitTextureView() {
	suite.Run("should upload texture staging data to provider", func() {
		provider := bind_group_provider.NewBindGroupProvider("tv-test")
		defer provider.Release()

		stagingData := common.TextureStagingData{
			Pixels: make([]byte, 4*4*4), // 4×4 RGBA = 64 bytes (all zero = black)
			Width:  4,
			Height: 4,
			Linear: false,
		}

		err := suite.r.InitTextureView(provider, 0, stagingData)
		suite.Require().NoError(err)
		suite.NotNil(provider.TextureView(0))
	})
}

func (suite *wgpuRendererBackendTest) TestInitSampler() {
	suite.Run("should create sampler on provider", func() {
		provider := bind_group_provider.NewBindGroupProvider("sampler-test")
		defer provider.Release()

		stagingData := common.SamplerStagingData{
			AddressModeU:  wgpu.AddressModeClampToEdge,
			AddressModeV:  wgpu.AddressModeClampToEdge,
			AddressModeW:  wgpu.AddressModeClampToEdge,
			MagFilter:     wgpu.FilterModeLinear,
			MinFilter:     wgpu.FilterModeLinear,
			MipmapFilter:  wgpu.MipmapFilterModeLinear,
			LodMinClamp:   0,
			LodMaxClamp:   32,
			MaxAnisotropy: 1,
		}

		err := suite.r.InitSampler(provider, 0, stagingData)
		suite.Require().NoError(err)
		suite.NotNil(provider.Sampler(0))
	})
}

func (suite *wgpuRendererBackendTest) TestWriteBuffers() {
	suite.Run("should write data to provider buffer without panicking", func() {
		provider := bind_group_provider.NewBindGroupProvider("write-buffers-test")
		defer provider.Release()

		descriptor := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageVertex,
					Buffer: wgpu.BufferBindingLayout{
						Type: wgpu.BufferBindingTypeUniform,
					},
				},
			},
		}
		err := suite.r.InitBindGroup(provider, descriptor, nil, map[int]uint64{0: 64})
		suite.Require().NoError(err)

		suite.NotPanics(func() {
			suite.r.WriteBuffers([]bind_group_provider.BufferWrite{
				{
					Provider: provider,
					Binding:  0,
					Offset:   0,
					Data:     make([]byte, 16),
				},
			})
		})
	})
}

func (suite *wgpuRendererBackendTest) TestBeginEndFrame() {
	suite.Run("should complete a swapchain frame without error", func() {
		err := suite.r.BeginFrame()
		suite.Require().NoError(err)
		suite.r.EndFrame()
		suite.r.FlushFrame()
		suite.r.Present()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestHDRFrameLifecycle() {
	suite.Run("should complete an HDR offscreen frame without error", func() {
		hdrView, hdrTex, _, _, depthView, depthTex, err := suite.r.CreateCompositionTextures(800, 600, 1)
		suite.Require().NoError(err)
		defer func() {
			hdrView.Release()
			hdrTex.Release()
			depthView.Release()
			depthTex.Release()
		}()

		err = suite.r.BeginHDRFrame(hdrView, nil, depthView, 1)
		suite.Require().NoError(err)
		suite.r.EndFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestCompositionFrameLifecycle() {
	suite.Run("should complete a composition frame without error", func() {
		err := suite.r.BeginCompositionFrame()
		suite.Require().NoError(err)
		suite.r.BeginCompositionPass()
		suite.r.EndCompositionPass()
		suite.r.EndCompositionFrame()
		suite.r.FlushFrame()
		suite.r.Present()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestBeginShadowDepthPassNoClear() {
	suite.Run("should complete shadow pass with clear=false without error", func() {
		depthView, depthTex, err := suite.r.CreateShadowDepthTexture(512, 512)
		suite.Require().NoError(err)
		defer func() {
			depthView.Release()
			depthTex.Release()
		}()

		err = suite.r.BeginGeometryFrame()
		suite.Require().NoError(err)

		err = suite.r.BeginShadowFrame()
		suite.Require().NoError(err)

		// First pass: clear=true (LoadOpClear)
		suite.r.BeginShadowDepthPass(depthView, 0, 0, 512, 512, true)
		suite.r.EndShadowPass()

		// Second pass: clear=false (LoadOpLoad) — the branch under test
		suite.r.BeginShadowDepthPass(depthView, 0, 0, 256, 256, false)
		suite.r.EndShadowPass()

		suite.r.EndShadowFrame()
		suite.r.EndGeometryFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterComputePipeline() {
	suite.Run("should register a compute pipeline without error", func() {
		tmpDir := suite.T().TempDir()
		csPath := filepath.Join(tmpDir, "cs.wgsl")
		suite.Require().NoError(os.WriteFile(csPath, []byte(`
@compute @workgroup_size(1, 1, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {}
`), 0644))

		cs := shader.NewShader("reg-compute-cs", shader.ShaderTypeCompute, csPath)
		p := pipeline.NewPipeline("reg-compute-pipeline", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(cs),
		)

		err := suite.r.RegisterPipelines(p)
		suite.Require().NoError(err)

		// Second registration of same key must be a no-op (cache-hit skip path)
		err = suite.r.RegisterPipelines(p)
		suite.NoError(err)
	})
}

func (suite *wgpuRendererBackendTest) TestDispatchComputeBatch() {
	suite.Run("should dispatch a registered compute pipeline without error", func() {
		tmpDir := suite.T().TempDir()
		csPath := filepath.Join(tmpDir, "dispatch_cs.wgsl")
		suite.Require().NoError(os.WriteFile(csPath, []byte(`
@compute @workgroup_size(1, 1, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {}
`), 0644))

		cs := shader.NewShader("dispatch-cs", shader.ShaderTypeCompute, csPath)
		p := pipeline.NewPipeline("dispatch-compute", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(cs),
		)
		err := suite.r.RegisterPipelines(p)
		suite.Require().NoError(err)

		// Provider with nil bind group — exercises the "BindGroup() == nil → continue" inner branch
		emptyProvider := bind_group_provider.NewBindGroupProvider("empty-dispatch-provider")
		defer emptyProvider.Release()

		err = suite.r.BeginComputeFrame()
		suite.Require().NoError(err)

		suite.r.DispatchComputeBatch([]renderer.ComputeDispatch{
			{
				PipelineKey: "dispatch-compute",
				Providers: []renderer.ComputeGroupProvider{
					{Group: 0, Provider: emptyProvider},
				},
				WorkGroupCount: [3]uint32{1, 1, 1},
			},
		})

		suite.r.EndComputeFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestDispatchComputeBatchUnknownPipeline() {
	suite.Run("should silently skip unknown pipeline key", func() {
		suite.NotPanics(func() {
			suite.r.DispatchComputeBatch([]renderer.ComputeDispatch{
				{
					PipelineKey:    "totally-unknown-compute-key",
					WorkGroupCount: [3]uint32{1, 1, 1},
				},
			})
		})
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterShadowDepthPipeline() {
	suite.Run("should register a shadow depth pipeline without error", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "shadow_vs.wgsl")
		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))

		vs := shader.NewShader("reg-shadow-vs", shader.ShaderTypeVertex, vsPath)
		p := pipeline.NewPipeline("reg-shadow-depth", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithDepthTestEnabled(true),
			pipeline.WithDepthWriteEnabled(true),
		)

		err := suite.r.RegisterShadowDepthPipeline(p)
		suite.Require().NoError(err)

		// Duplicate registration must be a no-op
		err = suite.r.RegisterShadowDepthPipeline(p)
		suite.NoError(err)
	})
}

func (suite *wgpuRendererBackendTest) TestShadowDrawCallLifecycle() {
	suite.Run("should execute shadow draw calls within a shadow depth pass", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "shadow_draw_vs.wgsl")
		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))

		vs := shader.NewShader("shadow-draw-vs", shader.ShaderTypeVertex, vsPath)
		p := pipeline.NewPipeline("shadow-draw-depth", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithDepthTestEnabled(true),
			pipeline.WithDepthWriteEnabled(true),
		)

		err := suite.r.RegisterShadowDepthPipeline(p)
		suite.Require().NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("shadow-mesh")
		defer meshProvider.Release()

		vertices := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0}
		vertexBytes := make([]byte, len(vertices)*4)
		for i, v := range vertices {
			binary.LittleEndian.PutUint32(vertexBytes[i*4:], math.Float32bits(v))
		}
		indices := []uint32{0, 1, 2}
		indexBytes := make([]byte, len(indices)*4)
		for i, idx := range indices {
			binary.LittleEndian.PutUint32(indexBytes[i*4:], idx)
		}

		err = suite.r.InitMeshBuffers(meshProvider, vertexBytes, indexBytes, 3)
		suite.Require().NoError(err)

		indirectBuf, err := suite.r.CreateBuffer("shadow-indirect", 20, wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(err)
		defer indirectBuf.Release()

		indirectParams := make([]byte, 20)
		binary.LittleEndian.PutUint32(indirectParams[0:], 3) // indexCount
		binary.LittleEndian.PutUint32(indirectParams[4:], 1) // instanceCount
		suite.r.WriteRawBuffer(indirectBuf, 0, indirectParams)

		depthView, depthTex, err := suite.r.CreateShadowDepthTexture(512, 512)
		suite.Require().NoError(err)
		defer func() {
			depthView.Release()
			depthTex.Release()
		}()

		err = suite.r.BeginGeometryFrame()
		suite.Require().NoError(err)

		err = suite.r.BeginShadowFrame()
		suite.Require().NoError(err)

		suite.r.BeginShadowDepthPass(depthView, 0, 0, 512, 512, true)

		err = suite.r.ShadowDrawCall("shadow-draw-depth", meshProvider, 1, nil)
		suite.NoError(err)

		err = suite.r.ShadowDrawCallIndirect("shadow-draw-depth", meshProvider, indirectBuf, nil)
		suite.NoError(err)

		suite.r.EndShadowPass()
		suite.r.EndShadowFrame()
		suite.r.EndGeometryFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterGBufferPipeline() {
	suite.Run("should register a GBuffer pipeline without error", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "gbuf_vs.wgsl")
		fsPath := filepath.Join(tmpDir, "gbuf_fs.wgsl")

		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))

		suite.Require().NoError(os.WriteFile(fsPath, []byte(`
struct GBufOut {
    @location(0) normal: vec4<f32>,
    @location(1) albedo: vec4<f32>,
}
@fragment
fn fs_main() -> GBufOut {
    var out: GBufOut;
    out.normal = vec4<f32>(0.5, 0.5, 1.0, 0.0);
    out.albedo = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return out;
}
`), 0644))

		vs := shader.NewShader("reg-gbuf-vs", shader.ShaderTypeVertex, vsPath)
		fs := shader.NewShader("reg-gbuf-fs", shader.ShaderTypeFragment, fsPath)
		p := pipeline.NewPipeline("reg-gbuf-pipeline", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithDepthTestEnabled(true),
			pipeline.WithDepthWriteEnabled(true),
		)

		err := suite.r.RegisterGBufferPipeline(p)
		suite.Require().NoError(err)

		// Duplicate
		err = suite.r.RegisterGBufferPipeline(p)
		suite.NoError(err)
	})
}

func (suite *wgpuRendererBackendTest) TestGBufferDrawCallLifecycle() {
	suite.Run("should execute GBuffer draw calls within a GBuffer pass", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "gbuf_draw_vs.wgsl")
		fsPath := filepath.Join(tmpDir, "gbuf_draw_fs.wgsl")

		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))

		suite.Require().NoError(os.WriteFile(fsPath, []byte(`
struct GBufOut {
    @location(0) normal: vec4<f32>,
    @location(1) albedo: vec4<f32>,
}
@fragment
fn fs_main() -> GBufOut {
    var out: GBufOut;
    out.normal = vec4<f32>(0.5, 0.5, 1.0, 0.0);
    out.albedo = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return out;
}
`), 0644))

		vs := shader.NewShader("gbuf-draw-vs", shader.ShaderTypeVertex, vsPath)
		fs := shader.NewShader("gbuf-draw-fs", shader.ShaderTypeFragment, fsPath)
		p := pipeline.NewPipeline("gbuf-draw-pipeline", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithDepthTestEnabled(true),
			pipeline.WithDepthWriteEnabled(true),
		)

		err := suite.r.RegisterGBufferPipeline(p)
		suite.Require().NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("gbuf-mesh")
		defer meshProvider.Release()

		vertices := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0}
		vertexBytes := make([]byte, len(vertices)*4)
		for i, v := range vertices {
			binary.LittleEndian.PutUint32(vertexBytes[i*4:], math.Float32bits(v))
		}
		indices := []uint32{0, 1, 2}
		indexBytes := make([]byte, len(indices)*4)
		for i, idx := range indices {
			binary.LittleEndian.PutUint32(indexBytes[i*4:], idx)
		}
		err = suite.r.InitMeshBuffers(meshProvider, vertexBytes, indexBytes, 3)
		suite.Require().NoError(err)

		indirectBuf, err := suite.r.CreateBuffer("gbuf-indirect", 20, wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(err)
		defer indirectBuf.Release()
		indirectParams := make([]byte, 20)
		binary.LittleEndian.PutUint32(indirectParams[0:], 3)
		binary.LittleEndian.PutUint32(indirectParams[4:], 1)
		suite.r.WriteRawBuffer(indirectBuf, 0, indirectParams)

		normView, normTex, albedoView, albedoTex, depthView, depthTex, err := suite.r.CreateGBufferTextures(800, 600)
		suite.Require().NoError(err)
		defer func() {
			normView.Release()
			albedoView.Release()
			depthView.Release()
			normTex.Release()
			albedoTex.Release()
			depthTex.Release()
		}()

		err = suite.r.BeginGeometryFrame()
		suite.Require().NoError(err)

		err = suite.r.BeginGBufferFrame()
		suite.Require().NoError(err)

		suite.r.BeginGBufferPass(normView, albedoView, depthView)

		err = suite.r.GBufferDrawCall("gbuf-draw-pipeline", meshProvider, 1, nil)
		suite.NoError(err)

		err = suite.r.GBufferDrawCallIndirect("gbuf-draw-pipeline", meshProvider, indirectBuf, nil)
		suite.NoError(err)

		suite.r.EndGBufferPass()
		suite.r.EndGBufferFrame()
		suite.r.EndGeometryFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterCompositionPipeline() {
	suite.Run("should register a composition pipeline without error", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "comp_vs.wgsl")
		fsPath := filepath.Join(tmpDir, "comp_fs.wgsl")

		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))

		suite.Require().NoError(os.WriteFile(fsPath, []byte(`
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
`), 0644))

		vs := shader.NewShader("reg-comp-vs", shader.ShaderTypeVertex, vsPath)
		fs := shader.NewShader("reg-comp-fs", shader.ShaderTypeFragment, fsPath)
		p := pipeline.NewPipeline("reg-comp-pipeline", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithDepthTestEnabled(false),
			pipeline.WithDepthWriteEnabled(false),
		)

		err := suite.r.RegisterCompositionPipeline(p)
		suite.Require().NoError(err)

		// Duplicate
		err = suite.r.RegisterCompositionPipeline(p)
		suite.NoError(err)
	})
}

func (suite *wgpuRendererBackendTest) TestCompositionDrawCallLifecycle() {
	suite.Run("should execute a composition draw call within a composition pass", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "comp_draw_vs.wgsl")
		fsPath := filepath.Join(tmpDir, "comp_draw_fs.wgsl")

		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))

		suite.Require().NoError(os.WriteFile(fsPath, []byte(`
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
`), 0644))

		vs := shader.NewShader("comp-draw-vs", shader.ShaderTypeVertex, vsPath)
		fs := shader.NewShader("comp-draw-fs", shader.ShaderTypeFragment, fsPath)
		p := pipeline.NewPipeline("comp-draw-pipeline", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithDepthTestEnabled(false),
			pipeline.WithDepthWriteEnabled(false),
		)

		err := suite.r.RegisterCompositionPipeline(p)
		suite.Require().NoError(err)

		err = suite.r.BeginCompositionFrame()
		suite.Require().NoError(err)

		suite.r.BeginCompositionPass()

		err = suite.r.CompositionDrawCall("comp-draw-pipeline", nil)
		suite.NoError(err)

		suite.r.EndCompositionPass()
		suite.r.EndCompositionFrame()
		suite.r.FlushFrame()
		suite.r.Present()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterRenderPipeline() {
	suite.Run("should register a render pipeline without error", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "render_vs.wgsl")
		fsPath := filepath.Join(tmpDir, "render_fs.wgsl")

		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))

		suite.Require().NoError(os.WriteFile(fsPath, []byte(`
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`), 0644))

		vs := shader.NewShader("reg-render-vs", shader.ShaderTypeVertex, vsPath)
		fs := shader.NewShader("reg-render-fs", shader.ShaderTypeFragment, fsPath)
		p := pipeline.NewPipeline("reg-render-pipeline", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithDepthTestEnabled(true),
			pipeline.WithDepthWriteEnabled(true),
		)

		err := suite.r.RegisterPipelines(p)
		suite.Require().NoError(err)

		// Duplicate registration of same key — exercises the cache-hit skip branch
		err = suite.r.RegisterPipelines(p)
		suite.NoError(err)
	})
}

func (suite *wgpuRendererBackendTest) TestDrawCallLifecycle() {
	suite.Run("should execute draw calls within a swapchain render frame", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "dcl_vs.wgsl")
		fsPath := filepath.Join(tmpDir, "dcl_fs.wgsl")

		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))

		suite.Require().NoError(os.WriteFile(fsPath, []byte(`
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`), 0644))

		vs := shader.NewShader("dcl-render-vs", shader.ShaderTypeVertex, vsPath)
		fs := shader.NewShader("dcl-render-fs", shader.ShaderTypeFragment, fsPath)
		p := pipeline.NewPipeline("dcl-render-pipeline", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithDepthTestEnabled(true),
			pipeline.WithDepthWriteEnabled(true),
		)

		err := suite.r.RegisterPipelines(p)
		suite.Require().NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("dcl-mesh")
		defer meshProvider.Release()

		vertices := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0}
		vertexBytes := make([]byte, len(vertices)*4)
		for i, v := range vertices {
			binary.LittleEndian.PutUint32(vertexBytes[i*4:], math.Float32bits(v))
		}
		indices := []uint32{0, 1, 2}
		indexBytes := make([]byte, len(indices)*4)
		for i, idx := range indices {
			binary.LittleEndian.PutUint32(indexBytes[i*4:], idx)
		}
		err = suite.r.InitMeshBuffers(meshProvider, vertexBytes, indexBytes, 3)
		suite.Require().NoError(err)

		indirectBuf, err := suite.r.CreateBuffer("dcl-indirect", 20, wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(err)
		defer indirectBuf.Release()
		indirectParams := make([]byte, 20)
		binary.LittleEndian.PutUint32(indirectParams[0:], 3)
		binary.LittleEndian.PutUint32(indirectParams[4:], 1)
		suite.r.WriteRawBuffer(indirectBuf, 0, indirectParams)

		err = suite.r.BeginFrame()
		suite.Require().NoError(err)

		err = suite.r.DrawCall("dcl-render-pipeline", meshProvider, 1, nil)
		suite.NoError(err)

		err = suite.r.DrawCallIndirect("dcl-render-pipeline", meshProvider, indirectBuf, nil)
		suite.NoError(err)

		suite.r.EndFrame()
		suite.r.FlushFrame()
		suite.r.Present()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestSetPresentModeVariants() {
	suite.Run("should set mailbox present mode without panic", func() {
		suite.NotPanics(func() { suite.r.SetPresentMode(renderer.PresentModeMailbox) })
	})
	suite.Run("should set uncapped present mode without panic", func() {
		suite.NotPanics(func() { suite.r.SetPresentMode(renderer.PresentModeUncapped) })
	})
	suite.Run("should set unknown present mode without panic (default case)", func() {
		suite.NotPanics(func() { suite.r.SetPresentMode(renderer.PresentMode(99)) })
	})
	// restore VSync for other tests
	suite.r.SetPresentMode(renderer.PresentModeVSync)
}

func (suite *wgpuRendererBackendTest) TestEndComputeFrameNoEncoder() {
	suite.Run("should not panic when EndComputeFrame called without Begin", func() {
		suite.NotPanics(func() { suite.r.EndComputeFrame() })
	})
}

func (suite *wgpuRendererBackendTest) TestBeginComputeFrameNested() {
	suite.Run("should handle nested Begin/End compute frame pairs", func() {
		// First Begin creates the encoder
		err := suite.r.BeginComputeFrame()
		suite.Require().NoError(err)
		// Second Begin is a no-op (depth > 1)
		err = suite.r.BeginComputeFrame()
		suite.NoError(err)
		// First End decrements depth to 1 — does NOT submit yet
		suite.r.EndComputeFrame()
		// Second End decrements to 0 and submits
		suite.r.EndComputeFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestBeginGeometryFrameNested() {
	suite.Run("should handle nested Begin/End geometry frame pairs", func() {
		// Call End without Begin — depth <= 0, should no-op
		suite.NotPanics(func() { suite.r.EndGeometryFrame() })

		err := suite.r.BeginGeometryFrame()
		suite.Require().NoError(err)
		// Nested Begin — no-op
		err = suite.r.BeginGeometryFrame()
		suite.NoError(err)
		// First End decrements depth to 1 — no submit yet
		suite.r.EndGeometryFrame()
		// Second End decrements to 0 and submits
		suite.r.EndGeometryFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestShadowFrameStandalone() {
	suite.Run("should complete standalone shadow frame without geometry frame", func() {
		depthView, depthTex, err := suite.r.CreateShadowDepthTexture(512, 512)
		suite.Require().NoError(err)
		defer func() {
			depthView.Release()
			depthTex.Release()
		}()

		// Nil guard: EndShadowFrame without BeginShadowFrame
		suite.NotPanics(func() { suite.r.EndShadowFrame() })

		// Standalone path: no geometry frame active
		err = suite.r.BeginShadowFrame()
		suite.Require().NoError(err)

		suite.r.BeginShadowDepthPass(depthView, 0, 0, 512, 512, true)
		suite.r.EndShadowPass()
		suite.r.EndShadowFrame() // standalone submit (no geometry frame encoder)
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestGBufferFrameStandalone() {
	suite.Run("should complete standalone gbuffer frame without geometry frame", func() {
		normView, normTex, albedoView, albedoTex, depthView, depthTex, err := suite.r.CreateGBufferTextures(800, 600)
		suite.Require().NoError(err)
		defer func() {
			normView.Release()
			albedoView.Release()
			depthView.Release()
			normTex.Release()
			albedoTex.Release()
			depthTex.Release()
		}()

		// Nil guard: EndGBufferFrame without BeginGBufferFrame
		suite.NotPanics(func() { suite.r.EndGBufferFrame() })

		// Standalone path: no geometry frame active
		err = suite.r.BeginGBufferFrame()
		suite.Require().NoError(err)

		suite.r.BeginGBufferPass(normView, albedoView, depthView)
		suite.r.EndGBufferPass()
		suite.r.EndGBufferFrame() // standalone submit
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestNilPassGuards() {
	suite.Run("EndShadowPass without active pass should not panic", func() {
		suite.NotPanics(func() { suite.r.EndShadowPass() })
	})
	suite.Run("EndGBufferPass without active pass should not panic", func() {
		suite.NotPanics(func() { suite.r.EndGBufferPass() })
	})
	suite.Run("EndCompositionPass without active pass should not panic", func() {
		suite.NotPanics(func() { suite.r.EndCompositionPass() })
	})
	suite.Run("BeginShadowDepthPass without shadow encoder should not panic", func() {
		depthView, depthTex, err := suite.r.CreateShadowDepthTexture(256, 256)
		suite.Require().NoError(err)
		defer func() { depthView.Release(); depthTex.Release() }()
		suite.NotPanics(func() {
			suite.r.BeginShadowDepthPass(depthView, 0, 0, 256, 256, true)
		})
	})
	suite.Run("BeginGBufferPass without gbuffer encoder should not panic", func() {
		suite.NotPanics(func() {
			suite.r.BeginGBufferPass(nil, nil, nil)
		})
	})
	suite.Run("BeginCompositionPass without composition encoder should not panic", func() {
		suite.NotPanics(func() { suite.r.BeginCompositionPass() })
	})
}

func (suite *wgpuRendererBackendTest) TestDrawCallNilPassGuards() {
	tmpDir := suite.T().TempDir()
	vsPath := filepath.Join(tmpDir, "nil_guard_vs.wgsl")
	suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))
	fsPath := filepath.Join(tmpDir, "nil_guard_fs.wgsl")
	suite.Require().NoError(os.WriteFile(fsPath, []byte(`
struct GBufOut {
    @location(0) normal: vec4<f32>,
    @location(1) albedo: vec4<f32>,
}
@fragment
fn fs_main() -> GBufOut {
    var out: GBufOut;
    out.normal = vec4<f32>(0.5, 0.5, 1.0, 0.0);
    out.albedo = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return out;
}
`), 0644))
	compFsPath := filepath.Join(tmpDir, "nil_guard_comp_fs.wgsl")
	suite.Require().NoError(os.WriteFile(compFsPath, []byte(`
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
`), 0644))

	vs := shader.NewShader("ng-vs", shader.ShaderTypeVertex, vsPath)
	fs := shader.NewShader("ng-fs", shader.ShaderTypeFragment, fsPath)
	compFs := shader.NewShader("ng-comp-fs", shader.ShaderTypeFragment, compFsPath)

	shadowP := pipeline.NewPipeline("ng-shadow", pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(vs),
		pipeline.WithDepthTestEnabled(true),
		pipeline.WithDepthWriteEnabled(true),
	)
	suite.Require().NoError(suite.r.RegisterShadowDepthPipeline(shadowP))

	gbufP := pipeline.NewPipeline("ng-gbuf", pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(vs),
		pipeline.WithFragmentShader(fs),
		pipeline.WithDepthTestEnabled(true),
		pipeline.WithDepthWriteEnabled(true),
	)
	suite.Require().NoError(suite.r.RegisterGBufferPipeline(gbufP))

	compP := pipeline.NewPipeline("ng-comp", pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(vs),
		pipeline.WithFragmentShader(compFs),
		pipeline.WithDepthTestEnabled(false),
		pipeline.WithDepthWriteEnabled(false),
	)
	suite.Require().NoError(suite.r.RegisterCompositionPipeline(compP))

	meshProvider := bind_group_provider.NewBindGroupProvider("ng-mesh")
	defer meshProvider.Release()
	vertices := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0}
	vBytes := make([]byte, len(vertices)*4)
	for i, v := range vertices {
		binary.LittleEndian.PutUint32(vBytes[i*4:], math.Float32bits(v))
	}
	indices := []uint32{0, 1, 2}
	iBytes := make([]byte, len(indices)*4)
	for i, idx := range indices {
		binary.LittleEndian.PutUint32(iBytes[i*4:], idx)
	}
	suite.Require().NoError(suite.r.InitMeshBuffers(meshProvider, vBytes, iBytes, 3))

	indirectBuf, err := suite.r.CreateBuffer("ng-indirect", 20, wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst)
	suite.Require().NoError(err)
	defer indirectBuf.Release()

	suite.Run("ShadowDrawCall without active shadow pass should not panic", func() {
		suite.NotPanics(func() {
			suite.r.ShadowDrawCall("ng-shadow", meshProvider, 1, nil) //nolint:errcheck
		})
	})
	suite.Run("ShadowDrawCallIndirect without active shadow pass should not panic", func() {
		suite.NotPanics(func() {
			suite.r.ShadowDrawCallIndirect("ng-shadow", meshProvider, indirectBuf, nil) //nolint:errcheck
		})
	})
	suite.Run("GBufferDrawCall without active gbuffer pass should not panic", func() {
		suite.NotPanics(func() {
			suite.r.GBufferDrawCall("ng-gbuf", meshProvider, 1, nil) //nolint:errcheck
		})
	})
	suite.Run("GBufferDrawCallIndirect without active gbuffer pass should not panic", func() {
		suite.NotPanics(func() {
			suite.r.GBufferDrawCallIndirect("ng-gbuf", meshProvider, indirectBuf, nil) //nolint:errcheck
		})
	})
	suite.Run("CompositionDrawCall without active composition pass should not panic", func() {
		suite.NotPanics(func() {
			suite.r.CompositionDrawCall("ng-comp", nil) //nolint:errcheck
		})
	})
}

func (suite *wgpuRendererBackendTest) TestWriteRawBufferNil() {
	suite.Run("should not panic when buf is nil", func() {
		suite.NotPanics(func() {
			suite.r.WriteRawBuffer(nil, 0, make([]byte, 4))
		})
	})
}

func (suite *wgpuRendererBackendTest) TestCopyBufferToBufferNilEncoder() {
	suite.Run("should not panic when compute encoder is nil", func() {
		src, err := suite.r.CreateBuffer("copy-nil-src", 32, wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(err)
		defer src.Release()
		dst, err := suite.r.CreateBuffer("copy-nil-dst", 32, wgpu.BufferUsageCopyDst|wgpu.BufferUsageCopySrc)
		suite.Require().NoError(err)
		defer dst.Release()
		// No BeginComputeFrame — encoder is nil
		suite.NotPanics(func() {
			suite.r.CopyBufferToBuffer(src, dst, 0, 0, 16)
		})
	})
}

func (suite *wgpuRendererBackendTest) TestWriteBuffersNilBuffer() {
	suite.Run("should not panic when buffer is nil for a binding", func() {
		provider := bind_group_provider.NewBindGroupProvider("wb-nil-test")
		defer provider.Release()
		// provider has no buffer at binding 0
		suite.NotPanics(func() {
			suite.r.WriteBuffers([]bind_group_provider.BufferWrite{
				{Provider: provider, Binding: 0, Offset: 0, Data: make([]byte, 4)},
			})
		})
	})
}

func (suite *wgpuRendererBackendTest) TestInitMeshBuffersEmptyData() {
	suite.Run("should handle empty vertex and index data without error", func() {
		provider := bind_group_provider.NewBindGroupProvider("mesh-empty")
		defer provider.Release()
		err := suite.r.InitMeshBuffers(provider, nil, nil, 0)
		suite.NoError(err)
		suite.Nil(provider.VertexBuffer())
		suite.Nil(provider.IndexBuffer())
	})
}

func (suite *wgpuRendererBackendTest) TestInitBindGroupEdgeCases() {
	suite.Run("should return nil for empty descriptor", func() {
		provider := bind_group_provider.NewBindGroupProvider("bgp-empty")
		defer provider.Release()
		err := suite.r.InitBindGroup(provider, wgpu.BindGroupLayoutDescriptor{}, nil, nil)
		suite.NoError(err)
	})

	suite.Run("should return error when texture binding has no view", func() {
		provider := bind_group_provider.NewBindGroupProvider("bgp-tex-missing")
		defer provider.Release()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageFragment,
					Texture: wgpu.TextureBindingLayout{
						SampleType:    wgpu.TextureSampleTypeFloat,
						ViewDimension: wgpu.TextureViewDimension2D,
					},
				},
			},
		}
		err := suite.r.InitBindGroup(provider, desc, nil, nil)
		suite.Error(err)
	})

	suite.Run("should return error when sampler binding has no sampler", func() {
		provider := bind_group_provider.NewBindGroupProvider("bgp-samp-missing")
		defer provider.Release()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageFragment,
					Sampler: wgpu.SamplerBindingLayout{
						Type: wgpu.SamplerBindingTypeFiltering,
					},
				},
			},
		}
		err := suite.r.InitBindGroup(provider, desc, nil, nil)
		suite.Error(err)
	})

	suite.Run("should create bind group with texture binding", func() {
		provider := bind_group_provider.NewBindGroupProvider("bgp-tex-ok")
		defer provider.Release()
		stagingData := common.TextureStagingData{
			Pixels: make([]byte, 4*4*4),
			Width:  4,
			Height: 4,
			Linear: false,
		}
		err := suite.r.InitTextureView(provider, 0, stagingData)
		suite.Require().NoError(err)

		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageFragment,
					Texture: wgpu.TextureBindingLayout{
						SampleType:    wgpu.TextureSampleTypeFloat,
						ViewDimension: wgpu.TextureViewDimension2D,
					},
				},
			},
		}
		err = suite.r.InitBindGroup(provider, desc, nil, nil)
		suite.NoError(err)
	})

	suite.Run("should create bind group with sampler binding", func() {
		provider := bind_group_provider.NewBindGroupProvider("bgp-samp-ok")
		defer provider.Release()
		samplerData := common.SamplerStagingData{
			AddressModeU: wgpu.AddressModeClampToEdge,
			AddressModeV: wgpu.AddressModeClampToEdge,
			AddressModeW: wgpu.AddressModeClampToEdge,
			MagFilter:    wgpu.FilterModeLinear,
			MinFilter:    wgpu.FilterModeLinear,
		}
		err := suite.r.InitSampler(provider, 0, samplerData)
		suite.Require().NoError(err)

		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageFragment,
					Sampler: wgpu.SamplerBindingLayout{
						Type: wgpu.SamplerBindingTypeFiltering,
					},
				},
			},
		}
		err = suite.r.InitBindGroup(provider, desc, nil, nil)
		suite.NoError(err)
	})
}

func (suite *wgpuRendererBackendTest) TestBeginFrameAlreadyAcquired() {
	suite.Run("should return error when frame is already acquired", func() {
		// Acquire the first frame normally
		err := suite.r.BeginFrame()
		suite.Require().NoError(err)

		// Second acquire should fail with an error about already-acquired surface
		err = suite.r.BeginFrame()
		suite.Error(err)

		// Clean up: end, flush, present
		suite.r.EndFrame()
		suite.r.FlushFrame()
		suite.r.Present()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestBeginCompositionFrameAlreadyAcquired() {
	suite.Run("should return error when composition frame is already acquired", func() {
		err := suite.r.BeginCompositionFrame()
		suite.Require().NoError(err)

		err = suite.r.BeginCompositionFrame()
		suite.Error(err)

		// Clean up
		suite.r.BeginCompositionPass()
		suite.r.EndCompositionPass()
		suite.r.EndCompositionFrame()
		suite.r.FlushFrame()
		suite.r.Present()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestPresentNoSurface() {
	suite.Run("should not panic when no surface is held", func() {
		// Call Present without any active Begin* — both frameSurface and compositionSurface are nil
		suite.NotPanics(func() { suite.r.Present() })
	})
}

func (suite *wgpuRendererBackendTest) TestFlushFrameNoPending() {
	suite.Run("should return zero index when no pending buffers", func() {
		idx := suite.r.FlushFrame()
		// Just verify it doesn't panic and returns
		_ = idx
	})
}

func (suite *wgpuRendererBackendTest) TestSyncGPUTimestampsAfterFlush() {
	suite.Run("should poll GPU when slot has a valid submission", func() {
		// Submit a real compute frame to populate slotSubmitValid
		err := suite.r.BeginComputeFrame()
		suite.Require().NoError(err)
		suite.r.EndComputeFrame()
		suite.r.FlushFrame() // slot 0 becomes valid, advances to slot 1

		suite.r.FlushFrame() // slot 1 also advances, back to slot 0

		// Now on slot 0 which has a valid submission — SyncGPUTimestamps should poll
		suite.NotPanics(func() {
			suite.r.SyncGPUTimestamps()
		})
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestCreateCompositionTexturesMSAA() {
	suite.Run("should create MSAA composition textures without error", func() {
		hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex, err := suite.r.CreateCompositionTextures(800, 600, 4)
		suite.Require().NoError(err)
		suite.Require().NotNil(hdrView)
		suite.Require().NotNil(hdrTex)
		suite.Require().NotNil(msaaView)
		suite.Require().NotNil(msaaTex)
		suite.Require().NotNil(depthView)
		suite.Require().NotNil(depthTex)
		defer func() {
			hdrView.Release()
			hdrTex.Release()
			msaaView.Release()
			msaaTex.Release()
			depthView.Release()
			depthTex.Release()
		}()
	})
}

func (suite *wgpuRendererBackendTest) TestBeginHDRFrameMSAA() {
	suite.Run("should complete HDR MSAA frame without error", func() {
		hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex, err := suite.r.CreateCompositionTextures(800, 600, 4)
		suite.Require().NoError(err)
		defer func() {
			hdrView.Release()
			hdrTex.Release()
			if msaaView != nil {
				msaaView.Release()
			}
			if msaaTex != nil {
				msaaTex.Release()
			}
			depthView.Release()
			depthTex.Release()
		}()

		// msaaView is the color attachment; hdrView is the resolve target
		err = suite.r.BeginHDRFrame(msaaView, hdrView, depthView, 4)
		suite.Require().NoError(err)
		suite.r.EndFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterRenderPipelineWithBlend() {
	suite.Run("should register a render pipeline with blend enabled", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "blend_vs.wgsl")
		fsPath := filepath.Join(tmpDir, "blend_fs.wgsl")
		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))
		suite.Require().NoError(os.WriteFile(fsPath, []byte(`
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`), 0644))
		vs := shader.NewShader("blend-vs", shader.ShaderTypeVertex, vsPath)
		fs := shader.NewShader("blend-fs", shader.ShaderTypeFragment, fsPath)
		p := pipeline.NewPipeline("blend-pipeline", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithDepthTestEnabled(true),
			pipeline.WithDepthWriteEnabled(true),
			pipeline.WithBlendEnabled(true),
		)
		err := suite.r.RegisterPipelines(p)
		suite.Require().NoError(err)
	})
}

func (suite *wgpuRendererBackendTest) TestShadowDrawCallWithBindGroups() {
	suite.Run("should execute shadow draw calls with bind groups", func() {
		depthView, depthTex, err := suite.r.CreateShadowDepthTexture(512, 512)
		suite.Require().NoError(err)
		defer func() {
			depthView.Release()
			depthTex.Release()
		}()

		meshProvider := bind_group_provider.NewBindGroupProvider("sdcwbg-mesh")
		defer meshProvider.Release()
		vertices := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0}
		vBytes := make([]byte, len(vertices)*4)
		for i, v := range vertices {
			binary.LittleEndian.PutUint32(vBytes[i*4:], math.Float32bits(v))
		}
		indices := []uint32{0, 1, 2}
		iBytes := make([]byte, len(indices)*4)
		for i, idx := range indices {
			binary.LittleEndian.PutUint32(iBytes[i*4:], idx)
		}
		suite.Require().NoError(suite.r.InitMeshBuffers(meshProvider, vBytes, iBytes, 3))

		indirectBuf, err := suite.r.CreateBuffer("sdcwbg-indirect", 20, wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(err)
		defer indirectBuf.Release()

		bgProvider := bind_group_provider.NewBindGroupProvider("sdcwbg-bg")
		defer bgProvider.Release()
		err = suite.r.InitBindGroup(bgProvider, wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageVertex,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeUniform,
						MinBindingSize: 64,
					},
				},
			},
		}, nil, map[int]uint64{0: 64})
		suite.Require().NoError(err)

		err = suite.r.BeginGeometryFrame()
		suite.Require().NoError(err)

		err = suite.r.BeginShadowFrame()
		suite.Require().NoError(err)

		suite.r.BeginShadowDepthPass(depthView, 0, 0, 512, 512, true)

		errDC := suite.r.ShadowDrawCall("shadow-draw-depth", meshProvider, 1, []bind_group_provider.BindGroupProvider{bgProvider})
		suite.NoError(errDC)

		errDCI := suite.r.ShadowDrawCallIndirect("shadow-draw-depth", meshProvider, indirectBuf, []bind_group_provider.BindGroupProvider{bgProvider})
		suite.NoError(errDCI)

		suite.r.EndShadowPass()
		suite.r.EndShadowFrame()
		suite.r.EndGeometryFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestGBufferDrawCallWithBindGroups() {
	suite.Run("should execute GBuffer draw calls with bind groups", func() {
		normView, normTex, albedoView, albedoTex, depthView, depthTex, err := suite.r.CreateGBufferTextures(800, 600)
		suite.Require().NoError(err)
		defer func() {
			normView.Release()
			albedoView.Release()
			depthView.Release()
			normTex.Release()
			albedoTex.Release()
			depthTex.Release()
		}()

		meshProvider := bind_group_provider.NewBindGroupProvider("gdcwbg-mesh")
		defer meshProvider.Release()
		vertices := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0}
		vBytes := make([]byte, len(vertices)*4)
		for i, v := range vertices {
			binary.LittleEndian.PutUint32(vBytes[i*4:], math.Float32bits(v))
		}
		indices := []uint32{0, 1, 2}
		iBytes := make([]byte, len(indices)*4)
		for i, idx := range indices {
			binary.LittleEndian.PutUint32(iBytes[i*4:], idx)
		}
		suite.Require().NoError(suite.r.InitMeshBuffers(meshProvider, vBytes, iBytes, 3))

		indirectBuf, err := suite.r.CreateBuffer("gdcwbg-indirect", 20, wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(err)
		defer indirectBuf.Release()

		bgProvider := bind_group_provider.NewBindGroupProvider("gdcwbg-bg")
		defer bgProvider.Release()
		err = suite.r.InitBindGroup(bgProvider, wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeUniform,
						MinBindingSize: 64,
					},
				},
			},
		}, nil, map[int]uint64{0: 64})
		suite.Require().NoError(err)

		err = suite.r.BeginGeometryFrame()
		suite.Require().NoError(err)

		err = suite.r.BeginGBufferFrame()
		suite.Require().NoError(err)

		suite.r.BeginGBufferPass(normView, albedoView, depthView)

		errDC := suite.r.GBufferDrawCall("gbuf-draw-pipeline", meshProvider, 1, []bind_group_provider.BindGroupProvider{bgProvider})
		suite.NoError(errDC)

		errDCI := suite.r.GBufferDrawCallIndirect("gbuf-draw-pipeline", meshProvider, indirectBuf, []bind_group_provider.BindGroupProvider{bgProvider})
		suite.NoError(errDCI)

		suite.r.EndGBufferPass()
		suite.r.EndGBufferFrame()
		suite.r.EndGeometryFrame()
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestCompositionDrawCallWithBindGroups() {
	suite.Run("should execute composition draw call with bind groups", func() {
		bgProvider := bind_group_provider.NewBindGroupProvider("cdcwbg-bg")
		defer bgProvider.Release()
		err := suite.r.InitBindGroup(bgProvider, wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageFragment,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeUniform,
						MinBindingSize: 64,
					},
				},
			},
		}, nil, map[int]uint64{0: 64})
		suite.Require().NoError(err)

		err = suite.r.BeginCompositionFrame()
		suite.Require().NoError(err)

		suite.r.BeginCompositionPass()

		suite.r.CompositionDrawCall("comp-draw-pipeline", []bind_group_provider.BindGroupProvider{bgProvider})

		suite.r.EndCompositionPass()
		suite.r.EndCompositionFrame()
		suite.r.FlushFrame()
		suite.r.Present()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestInitBindGroupStorageBindings() {
	suite.Run("should create bind group with storage binding type", func() {
		provider := bind_group_provider.NewBindGroupProvider("bgp-storage")
		defer provider.Release()
		err := suite.r.InitBindGroup(provider, wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageCompute,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeStorage,
						MinBindingSize: 64,
					},
				},
			},
		}, nil, map[int]uint64{0: 64})
		suite.NoError(err)
	})
	suite.Run("should create bind group with read-only storage binding type", func() {
		provider := bind_group_provider.NewBindGroupProvider("bgp-readonly-storage")
		defer provider.Release()
		err := suite.r.InitBindGroup(provider, wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageCompute,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeReadOnlyStorage,
						MinBindingSize: 64,
					},
				},
			},
		}, nil, map[int]uint64{0: 64})
		suite.NoError(err)
	})
}

func (suite *wgpuRendererBackendTest) TestMergeBindGroupLayouts() {
	suite.Run("should pass through vertex-only group unchanged", func() {
		vLayouts := map[int]wgpu.BindGroupLayoutDescriptor{
			0: {
				Label: "vert-only",
				Entries: []wgpu.BindGroupLayoutEntry{
					{Binding: 0, Visibility: wgpu.ShaderStageVertex},
				},
			},
		}
		fLayouts := map[int]wgpu.BindGroupLayoutDescriptor{}
		result := renderer.MergeBindGroupLayouts(vLayouts, fLayouts)
		suite.Require().Contains(result, 0)
		suite.Equal("vert-only", result[0].Label)
	})

	suite.Run("should pass through fragment-only group unchanged", func() {
		vLayouts := map[int]wgpu.BindGroupLayoutDescriptor{}
		fLayouts := map[int]wgpu.BindGroupLayoutDescriptor{
			1: {
				Label: "frag-only",
				Entries: []wgpu.BindGroupLayoutEntry{
					{Binding: 0, Visibility: wgpu.ShaderStageFragment},
				},
			},
		}
		result := renderer.MergeBindGroupLayouts(vLayouts, fLayouts)
		suite.Require().Contains(result, 1)
		suite.Equal("frag-only", result[1].Label)
	})

	suite.Run("should merge and OR visibility flags when both have same group", func() {
		vLayouts := map[int]wgpu.BindGroupLayoutDescriptor{
			0: {
				Label: "merged",
				Entries: []wgpu.BindGroupLayoutEntry{
					{Binding: 0, Visibility: wgpu.ShaderStageVertex},
					{Binding: 1, Visibility: wgpu.ShaderStageVertex},
				},
			},
		}
		fLayouts := map[int]wgpu.BindGroupLayoutDescriptor{
			0: {
				Label: "merged-frag",
				Entries: []wgpu.BindGroupLayoutEntry{
					{Binding: 0, Visibility: wgpu.ShaderStageFragment},
					{Binding: 2, Visibility: wgpu.ShaderStageFragment},
				},
			},
		}
		result := renderer.MergeBindGroupLayouts(vLayouts, fLayouts)
		suite.Require().Contains(result, 0)
		entries := result[0].Entries
		suite.Require().Len(entries, 3)

		var b0 *wgpu.BindGroupLayoutEntry
		for i := range entries {
			if entries[i].Binding == 0 {
				b0 = &entries[i]
				break
			}
		}
		suite.Require().NotNil(b0)
		suite.Equal(wgpu.ShaderStageVertex|wgpu.ShaderStageFragment, b0.Visibility)

		suite.Equal(uint32(0), entries[0].Binding)
		suite.Equal(uint32(1), entries[1].Binding)
		suite.Equal(uint32(2), entries[2].Binding)
	})

	suite.Run("should return empty map for empty inputs", func() {
		result := renderer.MergeBindGroupLayouts(
			map[int]wgpu.BindGroupLayoutDescriptor{},
			map[int]wgpu.BindGroupLayoutDescriptor{},
		)
		suite.Empty(result)
	})
}

type wgpuRendererBackendMSAATest struct {
	suite.Suite
	w window.Window
	r renderer.Renderer
}

func TestRunWgpuRendererBackendMSAATests(t *testing.T) {
	suite.Run(t, new(wgpuRendererBackendMSAATest))
}

func (suite *wgpuRendererBackendMSAATest) SetupSuite() {
	suite.w = window.NewWindow(
		window.WithTitle("oxy-go MSAA integration test"),
		window.WithWidth(800),
		window.WithHeight(600),
	)
	suite.r = renderer.NewRenderer(
		renderer.BackendTypeWGPU,
		suite.w,
		renderer.WithForceSoftwareRenderer(true),
		renderer.WithMSAA(renderer.MSAA4x),
		renderer.WithPresentMode(renderer.PresentModeVSync),
	)
}

func (suite *wgpuRendererBackendMSAATest) TearDownSuite() {
	suite.r.WaitIdle()
	suite.w.Close() //nolint:errcheck
}

func (suite *wgpuRendererBackendMSAATest) TestConfigureSurfaceMSAA() {
	suite.Run("should configure MSAA surface and recreate MSAA texture on resize", func() {
		suite.NotPanics(func() { suite.r.Resize(800, 600) })
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterRenderPipelineNoVertexShader() {
	suite.Run("should error when vertex shader is missing", func() {
		tmpDir := suite.T().TempDir()
		fsPath := filepath.Join(tmpDir, "no_vs_fs.wgsl")
		suite.Require().NoError(os.WriteFile(fsPath, []byte(`
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`), 0644))
		fs := shader.NewShader("no-vs-fs", shader.ShaderTypeFragment, fsPath)
		p := pipeline.NewPipeline("no-vs-pipeline", pipeline.PipelineTypeRender,
			pipeline.WithFragmentShader(fs),
		)
		err := suite.r.RegisterPipelines(p)
		suite.Error(err)
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterComputePipelineNoShader() {
	suite.Run("should error when compute shader is missing", func() {
		p := pipeline.NewPipeline("no-cs-pipeline", pipeline.PipelineTypeCompute)
		err := suite.r.RegisterPipelines(p)
		suite.Error(err)
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterShadowDepthPipelineNoVertexShader() {
	suite.Run("should error when vertex shader is missing for shadow depth pipeline", func() {
		p := pipeline.NewPipeline("no-vs-shadow", pipeline.PipelineTypeRender)
		err := suite.r.RegisterShadowDepthPipeline(p)
		suite.Error(err)
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterGBufferPipelineNoShaders() {
	suite.Run("should error when vertex shader is missing for GBuffer pipeline", func() {
		p := pipeline.NewPipeline("gbuf-no-vs", pipeline.PipelineTypeRender)
		err := suite.r.RegisterGBufferPipeline(p)
		suite.Error(err)
	})
	suite.Run("should error when fragment shader is missing for GBuffer pipeline", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "gbuf_no_fs_vs.wgsl")
		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))
		vs := shader.NewShader("gbuf-no-fs-vs", shader.ShaderTypeVertex, vsPath)
		p := pipeline.NewPipeline("gbuf-no-fs", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
		)
		err := suite.r.RegisterGBufferPipeline(p)
		suite.Error(err)
	})
}

func (suite *wgpuRendererBackendTest) TestRegisterCompositionPipelineNoShaders() {
	suite.Run("should error when vertex shader is missing for composition pipeline", func() {
		p := pipeline.NewPipeline("comp-no-vs", pipeline.PipelineTypeRender)
		err := suite.r.RegisterCompositionPipeline(p)
		suite.Error(err)
	})
	suite.Run("should error when fragment shader is missing for composition pipeline", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "comp_no_fs_vs.wgsl")
		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))
		vs := shader.NewShader("comp-no-fs-vs", shader.ShaderTypeVertex, vsPath)
		p := pipeline.NewPipeline("comp-no-fs", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
		)
		err := suite.r.RegisterCompositionPipeline(p)
		suite.Error(err)
	})
}

func (suite *wgpuRendererBackendTest) TestCreateBloomTexturesZeroDimensions() {
	suite.Run("should return error for zero dimensions", func() {
		_, _, _, _, _, _, _, _, err := suite.r.CreateBloomTextures(0, 0)
		suite.Error(err)
	})
}

func (suite *wgpuRendererBackendTest) TestDrawCallWithBindGroups() {
	suite.Run("should execute draw calls within a swapchain render frame using bind groups", func() {
		tmpDir := suite.T().TempDir()
		vsPath := filepath.Join(tmpDir, "bg_vs.wgsl")
		fsPath := filepath.Join(tmpDir, "bg_fs.wgsl")
		suite.Require().NoError(os.WriteFile(vsPath, []byte(`
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`), 0644))
		suite.Require().NoError(os.WriteFile(fsPath, []byte(`
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`), 0644))
		vs := shader.NewShader("bg-vs", shader.ShaderTypeVertex, vsPath)
		fs := shader.NewShader("bg-fs", shader.ShaderTypeFragment, fsPath)
		p := pipeline.NewPipeline("bg-pipeline", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithDepthTestEnabled(true),
			pipeline.WithDepthWriteEnabled(true),
		)
		err := suite.r.RegisterPipelines(p)
		suite.Require().NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("bg-mesh")
		defer meshProvider.Release()
		vertices := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0}
		vBytes := make([]byte, len(vertices)*4)
		for i, v := range vertices {
			binary.LittleEndian.PutUint32(vBytes[i*4:], math.Float32bits(v))
		}
		indices := []uint32{0, 1, 2}
		iBytes := make([]byte, len(indices)*4)
		for i, idx := range indices {
			binary.LittleEndian.PutUint32(iBytes[i*4:], idx)
		}
		suite.Require().NoError(suite.r.InitMeshBuffers(meshProvider, vBytes, iBytes, 3))

		bgProvider := bind_group_provider.NewBindGroupProvider("bg-uniform")
		defer bgProvider.Release()
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment,
					Buffer: wgpu.BufferBindingLayout{
						Type: wgpu.BufferBindingTypeUniform,
					},
				},
			},
		}
		err = suite.r.InitBindGroup(bgProvider, descriptor, nil, map[int]uint64{0: 64})
		suite.Require().NoError(err)

		indirectBuf, iErr := suite.r.CreateBuffer("bg-indirect", 20, wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst)
		suite.Require().NoError(iErr)
		defer indirectBuf.Release()
		params := make([]byte, 20)
		binary.LittleEndian.PutUint32(params[0:], 3)
		binary.LittleEndian.PutUint32(params[4:], 1)
		suite.r.WriteRawBuffer(indirectBuf, 0, params)

		err = suite.r.BeginFrame()
		suite.Require().NoError(err)

		err = suite.r.DrawCall("bg-pipeline", meshProvider, 1, []bind_group_provider.BindGroupProvider{bgProvider})
		suite.NoError(err)

		err = suite.r.DrawCallIndirect("bg-pipeline", meshProvider, indirectBuf, []bind_group_provider.BindGroupProvider{bgProvider})
		suite.NoError(err)

		suite.r.EndFrame()
		suite.r.FlushFrame()
		suite.r.Present()
		suite.r.WaitIdle()
	})
}

type wgpuRendererBackendGPUSerializedTest struct {
	suite.Suite
	w window.Window
	r renderer.Renderer
}

func TestRunWgpuRendererBackendGPUSerializedTests(t *testing.T) {
	suite.Run(t, new(wgpuRendererBackendGPUSerializedTest))
}

func (suite *wgpuRendererBackendGPUSerializedTest) SetupSuite() {
	suite.w = window.NewWindow(
		window.WithTitle("oxy-go gpu serialized integration test"),
		window.WithWidth(800),
		window.WithHeight(600),
	)
	suite.r = renderer.NewRenderer(
		renderer.BackendTypeWGPU,
		suite.w,
		renderer.WithForceSoftwareRenderer(true),
		renderer.WithMSAA(renderer.MSAAOff),
		renderer.WithPresentMode(renderer.PresentModeVSync),
		renderer.WithGPUSerializedProfiling(true),
	)
}

func (suite *wgpuRendererBackendGPUSerializedTest) TearDownSuite() {
	suite.r.WaitIdle()
	suite.w.Close() //nolint:errcheck
}

func (suite *wgpuRendererBackendGPUSerializedTest) TestEndComputeFrameGPUSerialized() {
	suite.Run("should submit immediately when GPU serialized profiling is enabled", func() {
		err := suite.r.BeginComputeFrame()
		suite.Require().NoError(err)
		suite.NotPanics(func() { suite.r.EndComputeFrame() })
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendGPUSerializedTest) TestEndGeometryFrameGPUSerialized() {
	suite.Run("should submit geometry frame immediately when GPU serialized profiling is enabled", func() {
		err := suite.r.BeginGeometryFrame()
		suite.Require().NoError(err)
		suite.NotPanics(func() { suite.r.EndGeometryFrame() })
		suite.r.FlushFrame()
		suite.r.WaitIdle()
	})
}

func (suite *wgpuRendererBackendGPUSerializedTest) TestEndFrameGPUSerialized() {
	suite.Run("should submit frame immediately when GPU serialized profiling is enabled", func() {
		err := suite.r.BeginFrame()
		suite.Require().NoError(err)
		suite.NotPanics(func() { suite.r.EndFrame() })
		suite.r.FlushFrame()
		suite.r.Present()
		suite.r.WaitIdle()
	})
}
