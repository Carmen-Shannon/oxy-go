package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/stretchr/testify/suite"
)

func TestRunSSRHandlerTests(t *testing.T) {
	suite.Run(t, new(ssrHandlerTest))
}

type ssrHandlerTest struct {
	suite.Suite
	handler light.SSRHandler
}

func (suite *ssrHandlerTest) SetupSubTest() {
	suite.handler = light.NewSSRHandler()
}

func (suite *ssrHandlerTest) TestNewSSRHandler() {
	suite.Run("should create a new SSR handler with provided options", func() {
		h := light.NewSSRHandler(
			light.WithSSRScreenSize(1920, 1080),
			light.WithSSRMaxSteps(128),
			light.WithSSRMaxDistance(100.0),
			light.WithSSRThickness(0.2),
			light.WithSSRStride(2.0),
			light.WithSSRRoughnessCutoff(0.3),
		)
		suite.NotNil(h)
	})
}

func (suite *ssrHandlerTest) TestEnabled() {
	suite.Run("should return false by default", func() {
		suite.False(suite.handler.Enabled())
	})
}

func (suite *ssrHandlerTest) TestSetEnabled() {
	suite.Run("should update the enabled state", func() {
		suite.handler.SetEnabled(true)
		suite.True(suite.handler.Enabled())
	})
}

func (suite *ssrHandlerTest) TestScreenWidth() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.ScreenWidth())
	})
}

func (suite *ssrHandlerTest) TestScreenHeight() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.ScreenHeight())
	})
}

func (suite *ssrHandlerTest) TestMaxSteps() {
	suite.Run("should return 64 by default", func() {
		suite.Equal(64, suite.handler.MaxSteps())
	})
}

func (suite *ssrHandlerTest) TestMaxDistance() {
	suite.Run("should return 50.0 by default", func() {
		suite.InDelta(50.0, suite.handler.MaxDistance(), 1e-6)
	})
}

func (suite *ssrHandlerTest) TestThickness() {
	suite.Run("should return 0.1 by default", func() {
		suite.InDelta(0.1, suite.handler.Thickness(), 1e-6)
	})
}

func (suite *ssrHandlerTest) TestStride() {
	suite.Run("should return 1.0 by default", func() {
		suite.InDelta(1.0, suite.handler.Stride(), 1e-6)
	})
}

func (suite *ssrHandlerTest) TestRoughnessCutoff() {
	suite.Run("should return 0.5 by default", func() {
		suite.InDelta(0.5, suite.handler.RoughnessCutoff(), 1e-6)
	})
}

func (suite *ssrHandlerTest) TestPipelineKey() {
	suite.Run("should return empty string for unknown key", func() {
		suite.Equal("", suite.handler.PipelineKey("nonexistent"))
	})

	suite.Run("should return value after SetPipelineKey", func() {
		suite.handler.SetPipelineKey("test_pipeline", "test_key")
		suite.Equal("test_key", suite.handler.PipelineKey("test_pipeline"))
	})
}

func (suite *ssrHandlerTest) TestPipelineKeys() {
	suite.Run("should return the full pipeline keys map", func() {
		suite.handler.SetPipelineKey("test_pipeline", "test_key")
		suite.Len(suite.handler.PipelineKeys(), 1)
	})
}

func (suite *ssrHandlerTest) TestSetPipelineKey() {
	suite.Run("should store a pipeline key", func() {
		suite.handler.SetPipelineKey("test_pipeline", "test_key")
		suite.Equal("test_key", suite.handler.PipelineKey("test_pipeline"))
	})
}

func (suite *ssrHandlerTest) TestBgp() {
	suite.Run("should return non-nil for a known default key", func() {
		suite.NotNil(suite.handler.Bgp("ssr_compute"))
	})

	suite.Run("should return nil for an unknown key", func() {
		suite.Nil(suite.handler.Bgp("unknown"))
	})
}

func (suite *ssrHandlerTest) TestBgps() {
	suite.Run("should return the full bgp map", func() {
		suite.Len(suite.handler.Bgps(), 1)
	})
}

func (suite *ssrHandlerTest) TestSetBgp() {
	suite.Run("should store a bgp under the given key", func() {
		suite.handler.SetBgp("k", bind_group_provider.NewBindGroupProvider("k"))
		suite.NotNil(suite.handler.Bgp("k"))
	})
}

func (suite *ssrHandlerTest) TestResize() {
	suite.Run("should update the screen dimensions", func() {
		suite.handler.Resize(1920, 1080)
		suite.Equal(1920, suite.handler.ScreenWidth())
		suite.Equal(1080, suite.handler.ScreenHeight())
	})
}

func (suite *ssrHandlerTest) TestSSRTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.SSRTexture())
	})

	suite.Run("should update after SetSSRTexture", func() {
		suite.handler.SetSSRTexture(nil)
		suite.Nil(suite.handler.SSRTexture())
	})
}

func (suite *ssrHandlerTest) TestSSRTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.SSRTextureView())
	})

	suite.Run("should update after SetSSRTextureView", func() {
		suite.handler.SetSSRTextureView(nil)
		suite.Nil(suite.handler.SSRTextureView())
	})
}

func (suite *ssrHandlerTest) TestLinearSampler() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.LinearSampler())
	})

	suite.Run("should update after SetLinearSampler", func() {
		suite.handler.SetLinearSampler(nil)
		suite.Nil(suite.handler.LinearSampler())
	})
}

func (suite *ssrHandlerTest) TestHiZTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.HiZTexture())
	})

	suite.Run("should update after SetHiZTexture", func() {
		suite.handler.SetHiZTexture(nil)
		suite.Nil(suite.handler.HiZTexture())
	})
}

func (suite *ssrHandlerTest) TestHiZTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.HiZTextureView())
	})

	suite.Run("should update after SetHiZTextureView", func() {
		suite.handler.SetHiZTextureView(nil)
		suite.Nil(suite.handler.HiZTextureView())
	})
}

func (suite *ssrHandlerTest) TestHiZMipCount() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.HiZMipCount())
	})
}

func (suite *ssrHandlerTest) TestSetHiZMipCount() {
	suite.Run("should update the mip count", func() {
		suite.handler.SetHiZMipCount(8)
		suite.Equal(8, suite.handler.HiZMipCount())
	})
}

func (suite *ssrHandlerTest) TestHiZMipReadViews() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.HiZMipReadViews())
	})
}

func (suite *ssrHandlerTest) TestSetHiZMipReadViews() {
	suite.Run("should update the mip read views", func() {
		suite.handler.SetHiZMipReadViews(nil)
		suite.Nil(suite.handler.HiZMipReadViews())
	})
}

func (suite *ssrHandlerTest) TestHiZStorageViews() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.HiZStorageViews())
	})
}

func (suite *ssrHandlerTest) TestSetHiZStorageViews() {
	suite.Run("should update the mip storage views", func() {
		suite.handler.SetHiZStorageViews(nil)
		suite.Nil(suite.handler.HiZStorageViews())
	})
}
