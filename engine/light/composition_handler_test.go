package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/stretchr/testify/suite"
)

func TestRunCompositionHandlerTests(t *testing.T) {
	suite.Run(t, new(compositionHandlerTest))
}

type compositionHandlerTest struct {
	suite.Suite
	handler light.CompositionHandler
}

func (suite *compositionHandlerTest) SetupSubTest() {
	suite.handler = light.NewCompositionHandler()
}

func (suite *compositionHandlerTest) TestNewCompositionHandler() {
	suite.Run("should create a new composition handler with provided options", func() {
		h := light.NewCompositionHandler(
			light.WithCompositionScreenSize(1920, 1080),
			light.WithToneMappingEnabled(false),
			light.WithExposure(2.0),
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
