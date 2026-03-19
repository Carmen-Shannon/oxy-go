package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/stretchr/testify/suite"
)

func TestRunSSAOHandlerTests(t *testing.T) {
	suite.Run(t, new(ssaoHandlerTest))
}

type ssaoHandlerTest struct {
	suite.Suite
	handler light.SSAOHandler
}

func (suite *ssaoHandlerTest) SetupSubTest() {
	suite.handler = light.NewSSAOHandler()
}

func (suite *ssaoHandlerTest) TestNewSSAOHandler() {
	suite.Run("should create a new SSAO handler with provided options", func() {
		h := light.NewSSAOHandler(
			light.WithSSAOScreenSize(1920, 1080),
			light.WithSSAOSampleCount(32),
			light.WithSSAOMaxSamples(64),
			light.WithSSAORadius(1.0),
			light.WithSSAOBias(0.05),
			light.WithSSAOPower(3.0),
			light.WithSSAOBlurRadius(8),
			light.WithSSAOHalfResolution(true),
		)
		suite.NotNil(h)
	})
}

func (suite *ssaoHandlerTest) TestEnabled() {
	suite.Run("should return false by default", func() {
		suite.False(suite.handler.Enabled())
	})
}

func (suite *ssaoHandlerTest) TestSetEnabled() {
	suite.Run("should update the enabled state", func() {
		suite.handler.SetEnabled(true)
		suite.True(suite.handler.Enabled())
	})
}

func (suite *ssaoHandlerTest) TestScreenWidth() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.ScreenWidth())
	})
}

func (suite *ssaoHandlerTest) TestScreenHeight() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.ScreenHeight())
	})
}

func (suite *ssaoHandlerTest) TestSampleCount() {
	suite.Run("should return 16 by default", func() {
		suite.Equal(16, suite.handler.SampleCount())
	})
}

func (suite *ssaoHandlerTest) TestMaxSamples() {
	suite.Run("should return 32 by default", func() {
		suite.Equal(32, suite.handler.MaxSamples())
	})
}

func (suite *ssaoHandlerTest) TestRadius() {
	suite.Run("should return 0.5 by default", func() {
		suite.InDelta(0.5, suite.handler.Radius(), 1e-6)
	})
}

func (suite *ssaoHandlerTest) TestBias() {
	suite.Run("should return 0.025 by default", func() {
		suite.InDelta(0.025, suite.handler.Bias(), 1e-6)
	})
}

func (suite *ssaoHandlerTest) TestPower() {
	suite.Run("should return 2.0 by default", func() {
		suite.InDelta(2.0, suite.handler.Power(), 1e-6)
	})
}

func (suite *ssaoHandlerTest) TestBlurRadius() {
	suite.Run("should return 4 by default", func() {
		suite.Equal(4, suite.handler.BlurRadius())
	})
}

func (suite *ssaoHandlerTest) TestHalfResolution() {
	suite.Run("should return false by default", func() {
		suite.False(suite.handler.HalfResolution())
	})
}

func (suite *ssaoHandlerTest) TestSetHalfResolution() {
	suite.Run("should update the half resolution flag", func() {
		suite.handler.SetHalfResolution(true)
		suite.True(suite.handler.HalfResolution())
	})
}

func (suite *ssaoHandlerTest) TestPipelineKey() {
	suite.Run("should return empty string for unknown key", func() {
		suite.Equal("", suite.handler.PipelineKey("nonexistent"))
	})

	suite.Run("should return value after SetPipelineKey", func() {
		suite.handler.SetPipelineKey("test_pipeline", "test_key")
		suite.Equal("test_key", suite.handler.PipelineKey("test_pipeline"))
	})
}

func (suite *ssaoHandlerTest) TestPipelineKeys() {
	suite.Run("should return the full pipeline keys map", func() {
		suite.handler.SetPipelineKey("test_pipeline", "test_key")
		suite.Len(suite.handler.PipelineKeys(), 1)
	})
}

func (suite *ssaoHandlerTest) TestSetPipelineKey() {
	suite.Run("should store a pipeline key", func() {
		suite.handler.SetPipelineKey("test_pipeline", "test_key")
		suite.Equal("test_key", suite.handler.PipelineKey("test_pipeline"))
	})
}

func (suite *ssaoHandlerTest) TestBgp() {
	suite.Run("should return non-nil for a known default key", func() {
		suite.NotNil(suite.handler.Bgp("ssao_compute"))
	})

	suite.Run("should return nil for an unknown key", func() {
		suite.Nil(suite.handler.Bgp("unknown"))
	})
}

func (suite *ssaoHandlerTest) TestBgps() {
	suite.Run("should return the full bgp map", func() {
		suite.Len(suite.handler.Bgps(), 3)
	})
}

func (suite *ssaoHandlerTest) TestSetBgp() {
	suite.Run("should store a bgp under the given key", func() {
		suite.handler.SetBgp("k", bind_group_provider.NewBindGroupProvider("k"))
		suite.NotNil(suite.handler.Bgp("k"))
	})
}

func (suite *ssaoHandlerTest) TestResize() {
	suite.Run("should update the screen dimensions", func() {
		suite.handler.Resize(1920, 1080)
		suite.Equal(1920, suite.handler.ScreenWidth())
		suite.Equal(1080, suite.handler.ScreenHeight())
	})
}

func (suite *ssaoHandlerTest) TestRawTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.RawTexture())
	})

	suite.Run("should update after SetRawTexture", func() {
		suite.handler.SetRawTexture(nil)
		suite.Nil(suite.handler.RawTexture())
	})
}

func (suite *ssaoHandlerTest) TestRawTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.RawTextureView())
	})

	suite.Run("should update after SetRawTextureView", func() {
		suite.handler.SetRawTextureView(nil)
		suite.Nil(suite.handler.RawTextureView())
	})
}

func (suite *ssaoHandlerTest) TestBlurredTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.BlurredTexture())
	})

	suite.Run("should update after SetBlurredTexture", func() {
		suite.handler.SetBlurredTexture(nil)
		suite.Nil(suite.handler.BlurredTexture())
	})
}

func (suite *ssaoHandlerTest) TestBlurredTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.BlurredTextureView())
	})

	suite.Run("should update after SetBlurredTextureView", func() {
		suite.handler.SetBlurredTextureView(nil)
		suite.Nil(suite.handler.BlurredTextureView())
	})
}

func (suite *ssaoHandlerTest) TestScratchTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.ScratchTexture())
	})

	suite.Run("should update after SetScratchTexture", func() {
		suite.handler.SetScratchTexture(nil)
		suite.Nil(suite.handler.ScratchTexture())
	})
}

func (suite *ssaoHandlerTest) TestScratchTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.ScratchTextureView())
	})

	suite.Run("should update after SetScratchTextureView", func() {
		suite.handler.SetScratchTextureView(nil)
		suite.Nil(suite.handler.ScratchTextureView())
	})
}

func (suite *ssaoHandlerTest) TestNoiseTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.NoiseTexture())
	})

	suite.Run("should update after SetNoiseTexture", func() {
		suite.handler.SetNoiseTexture(nil)
		suite.Nil(suite.handler.NoiseTexture())
	})
}

func (suite *ssaoHandlerTest) TestNoiseTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.NoiseTextureView())
	})

	suite.Run("should update after SetNoiseTextureView", func() {
		suite.handler.SetNoiseTextureView(nil)
		suite.Nil(suite.handler.NoiseTextureView())
	})
}

func (suite *ssaoHandlerTest) TestLinearSampler() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.LinearSampler())
	})

	suite.Run("should update after SetLinearSampler", func() {
		suite.handler.SetLinearSampler(nil)
		suite.Nil(suite.handler.LinearSampler())
	})
}
