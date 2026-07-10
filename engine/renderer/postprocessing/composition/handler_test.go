package composition_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/composition"
	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

func TestRunCompositionHandlerTests(t *testing.T) {
	suite.Run(t, new(compositionHandlerTest))
}

type compositionHandlerTest struct {
	suite.Suite
	handler composition.Handler
}

func (suite *compositionHandlerTest) SetupSubTest() {
	suite.handler = composition.NewHandler()
}

func (suite *compositionHandlerTest) TestNewCompositionHandler() {
	suite.Run("should create a new composition handler with provided options", func() {
		h := composition.NewHandler(
			composition.WithCompositionScreenSize(1920, 1080),
			composition.WithToneMappingEnabled(false),
			composition.WithExposure(2.0),
		)
		suite.NotNil(h)
	})
}

func (suite *compositionHandlerTest) TestEnabled() {
	suite.Run("should return false by default", func() {
		suite.Equal(false, suite.handler.Enabled())
	})
}

func (suite *compositionHandlerTest) TestSetEnabled() {
	suite.Run("should update the enabled state", func() {
		suite.handler.SetEnabled(true)
		suite.Equal(true, suite.handler.Enabled())
	})
}

func (suite *compositionHandlerTest) TestScreenWidth() {
	suite.Run("should return the screen width", func() {
		suite.Equal(0, suite.handler.ScreenWidth())
	})
}

func (suite *compositionHandlerTest) TestScreenHeight() {
	suite.Run("should return the screen height", func() {
		suite.Equal(0, suite.handler.ScreenHeight())
	})
}

func (suite *compositionHandlerTest) TestToneMappingEnabled() {
	suite.Run("should return true by default", func() {
		suite.Equal(true, suite.handler.ToneMappingEnabled())
	})
}

func (suite *compositionHandlerTest) TestExposure() {
	suite.Run("should return the exposure value", func() {
		suite.Equal(float32(1.0), suite.handler.Exposure())
	})
}

func (suite *compositionHandlerTest) TestSetExposure() {
	suite.Run("should update the exposure value", func() {
		suite.handler.SetExposure(2.5)
		suite.Equal(float32(2.5), suite.handler.Exposure())
	})
}

func (suite *compositionHandlerTest) TestPipelineKey() {
	suite.Run("should return an empty string for an unknown key", func() {
		suite.Equal("", suite.handler.PipelineKey("unknown"))
	})

	suite.Run("should return the pipeline key after it is set", func() {
		suite.handler.SetPipelineKey("test", "test-key")
		suite.Equal("test-key", suite.handler.PipelineKey("test"))
	})
}

func (suite *compositionHandlerTest) TestPipelineKeys() {
	suite.Run("should return the full pipeline keys map", func() {
		suite.handler.SetPipelineKey("a", "akey")
		m := suite.handler.PipelineKeys()
		suite.Equal("akey", m["a"])
	})
}

func (suite *compositionHandlerTest) TestSetPipelineKey() {
	suite.Run("should store a pipeline key under the given name", func() {
		suite.handler.SetPipelineKey("pipe", "pipekey")
		suite.Equal("pipekey", suite.handler.PipelineKey("pipe"))
	})
}

func (suite *compositionHandlerTest) TestBgp() {
	suite.Run("should return nil for an unknown key", func() {
		suite.Nil(suite.handler.Bgp("unknown"))
	})

	suite.Run("should return the provider after it is set", func() {
		bgp := bind_group_provider.NewBindGroupProvider("test")
		suite.handler.SetBgp("test", bgp)
		suite.Equal(bgp, suite.handler.Bgp("test"))
	})
}

func (suite *compositionHandlerTest) TestBgps() {
	suite.Run("should return the full bgps map", func() {
		m := suite.handler.Bgps()
		suite.NotNil(m)
		suite.NotNil(m["composition"])
	})
}

func (suite *compositionHandlerTest) TestSetBgp() {
	suite.Run("should store a bind group provider under the given key", func() {
		bgp := bind_group_provider.NewBindGroupProvider("key")
		suite.handler.SetBgp("key", bgp)
		suite.Equal(bgp, suite.handler.Bgp("key"))
	})
}

func (suite *compositionHandlerTest) TestHDRTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.HDRTexture())
	})
}

func (suite *compositionHandlerTest) TestSetHDRTexture() {
	suite.Run("should update the HDR texture", func() {
		suite.handler.SetHDRTexture(nil)
		suite.Nil(suite.handler.HDRTexture())
	})
}

func (suite *compositionHandlerTest) TestHDRTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.HDRTextureView())
	})
}

func (suite *compositionHandlerTest) TestSetHDRTextureView() {
	suite.Run("should update the HDR texture view", func() {
		suite.handler.SetHDRTextureView(nil)
		suite.Nil(suite.handler.HDRTextureView())
	})
}

func (suite *compositionHandlerTest) TestMSAATexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.MSAATexture())
	})
}

func (suite *compositionHandlerTest) TestSetMSAATexture() {
	suite.Run("should update the MSAA texture", func() {
		suite.handler.SetMSAATexture(nil)
		suite.Nil(suite.handler.MSAATexture())
	})
}

func (suite *compositionHandlerTest) TestMSAATextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.MSAATextureView())
	})
}

func (suite *compositionHandlerTest) TestSetMSAATextureView() {
	suite.Run("should update the MSAA texture view", func() {
		suite.handler.SetMSAATextureView(nil)
		suite.Nil(suite.handler.MSAATextureView())
	})
}

func (suite *compositionHandlerTest) TestDepthTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.DepthTexture())
	})
}

func (suite *compositionHandlerTest) TestSetDepthTexture() {
	suite.Run("should update the depth texture", func() {
		suite.handler.SetDepthTexture(nil)
		suite.Nil(suite.handler.DepthTexture())
	})
}

func (suite *compositionHandlerTest) TestDepthTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.DepthTextureView())
	})
}

func (suite *compositionHandlerTest) TestSetDepthTextureView() {
	suite.Run("should update the depth texture view", func() {
		suite.handler.SetDepthTextureView(nil)
		suite.Nil(suite.handler.DepthTextureView())
	})
}

func (suite *compositionHandlerTest) TestLinearSampler() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.LinearSampler())
	})
}

func (suite *compositionHandlerTest) TestSetLinearSampler() {
	suite.Run("should update the linear sampler", func() {
		suite.handler.SetLinearSampler(nil)
		suite.Nil(suite.handler.LinearSampler())
	})
}

func (suite *compositionHandlerTest) TestResize() {
	suite.Run("should update the screen dimensions", func() {
		suite.handler.Resize(1920, 1080)
		suite.Equal(1920, suite.handler.ScreenWidth())
		suite.Equal(1080, suite.handler.ScreenHeight())
	})
}

func (suite *compositionHandlerTest) TestAutoExposureEnabled() {
	suite.Run("should return false by default", func() {
		suite.Equal(false, suite.handler.AutoExposureEnabled())
	})
}

func (suite *compositionHandlerTest) TestSetAutoExposureEnabled() {
	suite.Run("should update the auto exposure enabled state", func() {
		suite.handler.SetAutoExposureEnabled(true)
		suite.Equal(true, suite.handler.AutoExposureEnabled())
	})
}

func (suite *compositionHandlerTest) TestAdaptSpeed() {
	suite.Run("should return the default adapt speed", func() {
		suite.Equal(float32(1.0), suite.handler.AdaptSpeed())
	})
}

func (suite *compositionHandlerTest) TestSetAdaptSpeed() {
	suite.Run("should update the adapt speed", func() {
		suite.handler.SetAdaptSpeed(2.0)
		suite.Equal(float32(2.0), suite.handler.AdaptSpeed())
	})
}

func (suite *compositionHandlerTest) TestMinExposure() {
	suite.Run("should return the default min exposure", func() {
		suite.Equal(float32(0.1), suite.handler.MinExposure())
	})
}

func (suite *compositionHandlerTest) TestSetMinExposure() {
	suite.Run("should update the min exposure", func() {
		suite.handler.SetMinExposure(0.5)
		suite.Equal(float32(0.5), suite.handler.MinExposure())
	})
}

func (suite *compositionHandlerTest) TestMaxExposure() {
	suite.Run("should return the default max exposure", func() {
		suite.Equal(float32(10.0), suite.handler.MaxExposure())
	})
}

func (suite *compositionHandlerTest) TestSetMaxExposure() {
	suite.Run("should update the max exposure", func() {
		suite.handler.SetMaxExposure(20.0)
		suite.Equal(float32(20.0), suite.handler.MaxExposure())
	})
}

func (suite *compositionHandlerTest) TestLuminanceWorkgroupSize() {
	suite.Run("should return the default luminance workgroup size", func() {
		suite.Equal(16, suite.handler.LuminanceWorkgroupSize())
	})
}

func (suite *compositionHandlerTest) TestExposureBuffer() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.ExposureBuffer())
	})
}

func (suite *compositionHandlerTest) TestSetExposureBuffer() {
	suite.Run("should update the exposure buffer", func() {
		suite.handler.SetExposureBuffer(nil)
		suite.Nil(suite.handler.ExposureBuffer())
	})
}

func (suite *compositionHandlerTest) TestNewCompositionHandlerWithAutoExposureOptions() {
	suite.Run("should create handler with auto exposure enabled", func() {
		h := composition.NewHandler(
			composition.WithAutoExposure(true),
		)
		suite.Equal(true, h.AutoExposureEnabled())
	})

	suite.Run("should create handler with custom adapt speed", func() {
		h := composition.NewHandler(
			composition.WithAdaptSpeed(3.0),
		)
		suite.Equal(float32(3.0), h.AdaptSpeed())
	})

	suite.Run("should create handler with custom min exposure", func() {
		h := composition.NewHandler(
			composition.WithMinExposure(0.2),
		)
		suite.Equal(float32(0.2), h.MinExposure())
	})

	suite.Run("should create handler with custom max exposure", func() {
		h := composition.NewHandler(
			composition.WithMaxExposure(15.0),
		)
		suite.Equal(float32(15.0), h.MaxExposure())
	})

	suite.Run("should create handler with custom luminance workgroup size", func() {
		h := composition.NewHandler(
			composition.WithLuminanceWorkgroupSize(8),
		)
		suite.Equal(8, h.LuminanceWorkgroupSize())
	})
}

func (suite *compositionHandlerTest) TestBloomEnabled() {
	suite.Run("should return false by default", func() {
		suite.Equal(false, suite.handler.BloomEnabled())
	})
}

func (suite *compositionHandlerTest) TestSetBloomEnabled() {
	suite.Run("should update the bloom enabled state", func() {
		suite.handler.SetBloomEnabled(true)
		suite.Equal(true, suite.handler.BloomEnabled())
	})
}

func (suite *compositionHandlerTest) TestBloomThreshold() {
	suite.Run("should return the default bloom threshold", func() {
		suite.Equal(float32(1.0), suite.handler.BloomThreshold())
	})
}

func (suite *compositionHandlerTest) TestSetBloomThreshold() {
	suite.Run("should update the bloom threshold", func() {
		suite.handler.SetBloomThreshold(0.5)
		suite.Equal(float32(0.5), suite.handler.BloomThreshold())
	})
}

func (suite *compositionHandlerTest) TestBloomIntensity() {
	suite.Run("should return the default bloom intensity", func() {
		suite.Equal(float32(0.5), suite.handler.BloomIntensity())
	})
}

func (suite *compositionHandlerTest) TestSetBloomIntensity() {
	suite.Run("should update the bloom intensity", func() {
		suite.handler.SetBloomIntensity(0.8)
		suite.Equal(float32(0.8), suite.handler.BloomIntensity())
	})
}

func (suite *compositionHandlerTest) TestBloomMipCount() {
	suite.Run("should return zero by default", func() {
		suite.Equal(0, suite.handler.BloomMipCount())
	})
}

func (suite *compositionHandlerTest) TestSetBloomMipCount() {
	suite.Run("should update the bloom mip count", func() {
		suite.handler.SetBloomMipCount(5)
		suite.Equal(5, suite.handler.BloomMipCount())
	})
}

func (suite *compositionHandlerTest) TestBloomDownTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.BloomDownTexture())
	})
}

func (suite *compositionHandlerTest) TestSetBloomDownTexture() {
	suite.Run("should update the bloom down texture", func() {
		suite.handler.SetBloomDownTexture(nil)
		suite.Nil(suite.handler.BloomDownTexture())
	})
}

func (suite *compositionHandlerTest) TestBloomDownReadViews() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.BloomDownReadViews())
	})
}

func (suite *compositionHandlerTest) TestSetBloomDownReadViews() {
	suite.Run("should update the bloom down read views", func() {
		suite.handler.SetBloomDownReadViews(nil)
		suite.Nil(suite.handler.BloomDownReadViews())
	})
}

func (suite *compositionHandlerTest) TestBloomDownStorageViews() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.BloomDownStorageViews())
	})
}

func (suite *compositionHandlerTest) TestSetBloomDownStorageViews() {
	suite.Run("should update the bloom down storage views", func() {
		suite.handler.SetBloomDownStorageViews(nil)
		suite.Nil(suite.handler.BloomDownStorageViews())
	})
}

func (suite *compositionHandlerTest) TestBloomUpTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.BloomUpTexture())
	})
}

func (suite *compositionHandlerTest) TestSetBloomUpTexture() {
	suite.Run("should update the bloom up texture", func() {
		suite.handler.SetBloomUpTexture(nil)
		suite.Nil(suite.handler.BloomUpTexture())
	})
}

func (suite *compositionHandlerTest) TestBloomUpReadViews() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.BloomUpReadViews())
	})
}

func (suite *compositionHandlerTest) TestSetBloomUpReadViews() {
	suite.Run("should update the bloom up read views", func() {
		suite.handler.SetBloomUpReadViews(nil)
		suite.Nil(suite.handler.BloomUpReadViews())
	})
}

func (suite *compositionHandlerTest) TestBloomUpStorageViews() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.BloomUpStorageViews())
	})
}

func (suite *compositionHandlerTest) TestSetBloomUpStorageViews() {
	suite.Run("should update the bloom up storage views", func() {
		suite.handler.SetBloomUpStorageViews(nil)
		suite.Nil(suite.handler.BloomUpStorageViews())
	})
}

func (suite *compositionHandlerTest) TestBloomUpMip0View() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.BloomUpMip0View())
	})
}

func (suite *compositionHandlerTest) TestSetBloomUpMip0View() {
	suite.Run("should update the bloom up mip0 view", func() {
		suite.handler.SetBloomUpMip0View(nil)
		suite.Nil(suite.handler.BloomUpMip0View())
	})
}

func (suite *compositionHandlerTest) TestNewCompositionHandlerWithBloomOptions() {
	suite.Run("should create handler with bloom options applied", func() {
		h := composition.NewHandler(
			composition.WithBloomEnabled(true),
			composition.WithBloomThreshold(0.8),
			composition.WithBloomIntensity(0.6),
		)
		suite.Equal(true, h.BloomEnabled())
		suite.Equal(float32(0.8), h.BloomThreshold())
		suite.Equal(float32(0.6), h.BloomIntensity())
	})
}

func (suite *compositionHandlerTest) TestNewCompositionHandlerBloomDefaults() {
	suite.Run("should have correct bloom defaults with no bloom options", func() {
		h := composition.NewHandler()
		suite.Equal(false, h.BloomEnabled())
		suite.Equal(float32(1.0), h.BloomThreshold())
		suite.Equal(float32(0.5), h.BloomIntensity())
		suite.Equal(0, h.BloomMipCount())
	})
}

func (suite *compositionHandlerTest) TestSetSlot() {
	suite.Run("should set the active slot", func() {
		suite.handler.SetSlot(0)
		suite.handler.SetBloomDownReadViews([]*wgpu.TextureView{nil, nil})
		suite.handler.SetSlot(1)
		suite.Nil(suite.handler.BloomDownReadViews())
	})

	suite.Run("should isolate data between slots", func() {
		suite.handler.SetSlot(0)
		suite.handler.SetBloomDownReadViews([]*wgpu.TextureView{nil, nil})
		suite.handler.SetSlot(1)
		suite.handler.SetBloomDownReadViews([]*wgpu.TextureView{nil, nil, nil})
		suite.handler.SetSlot(0)
		suite.Equal(2, len(suite.handler.BloomDownReadViews()))
		suite.handler.SetSlot(1)
		suite.Equal(3, len(suite.handler.BloomDownReadViews()))
	})
}
