package renderer_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
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
