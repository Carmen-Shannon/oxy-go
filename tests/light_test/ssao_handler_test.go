package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type ssaoHandlerTest struct {
	suite.Suite
}

func TestSSAOHandler(t *testing.T) {
	suite.Run(t, new(ssaoHandlerTest))
}

func (suite *ssaoHandlerTest) TestNewSSAOHandler() {
	suite.Run("default enabled is false", func() {
		h := light.NewSSAOHandler()
		suite.False(h.Enabled())
	})

	suite.Run("default sample count is 16", func() {
		h := light.NewSSAOHandler()
		suite.Equal(16, h.SampleCount())
	})

	suite.Run("default radius is 0.5", func() {
		h := light.NewSSAOHandler()
		suite.InDelta(0.5, h.Radius(), 1e-6)
	})

	suite.Run("default bias is 0.025", func() {
		h := light.NewSSAOHandler()
		suite.InDelta(0.025, h.Bias(), 1e-6)
	})

	suite.Run("default power is 2.0", func() {
		h := light.NewSSAOHandler()
		suite.InDelta(2.0, h.Power(), 1e-6)
	})

	suite.Run("default blur radius is 4", func() {
		h := light.NewSSAOHandler()
		suite.Equal(4, h.BlurRadius())
	})

	suite.Run("default half resolution is false", func() {
		h := light.NewSSAOHandler()
		suite.False(h.HalfResolution())
	})

	suite.Run("default screen dimensions are zero", func() {
		h := light.NewSSAOHandler()
		suite.Equal(0, h.ScreenWidth())
		suite.Equal(0, h.ScreenHeight())
	})

	suite.Run("default pipeline keys are empty", func() {
		h := light.NewSSAOHandler()
		suite.Empty(h.PipelineKeys())
	})

	suite.Run("default bgps contain required keys", func() {
		h := light.NewSSAOHandler()
		suite.NotNil(h.Bgp("ssao_compute"))
		suite.NotNil(h.Bgp("ssao_blur_h"))
		suite.NotNil(h.Bgp("ssao_blur_v"))
	})

	suite.Run("bgp labels match keys", func() {
		h := light.NewSSAOHandler()
		suite.Equal("ssao_compute", h.Bgp("ssao_compute").Label())
		suite.Equal("ssao_blur_h", h.Bgp("ssao_blur_h").Label())
		suite.Equal("ssao_blur_v", h.Bgp("ssao_blur_v").Label())
	})

	suite.Run("default textures are nil", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.RawTexture())
		suite.Nil(h.RawTextureView())
		suite.Nil(h.BlurredTexture())
		suite.Nil(h.BlurredTextureView())
		suite.Nil(h.ScratchTexture())
		suite.Nil(h.ScratchTextureView())
		suite.Nil(h.NoiseTexture())
		suite.Nil(h.NoiseTextureView())
	})

	suite.Run("default linear sampler is nil", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.LinearSampler())
	})
}

func (suite *ssaoHandlerTest) TestWithSSAOScreenSize() {
	suite.Run("sets screen dimensions", func() {
		h := light.NewSSAOHandler(light.WithSSAOScreenSize(1920, 1080))
		suite.Equal(1920, h.ScreenWidth())
		suite.Equal(1080, h.ScreenHeight())
	})
}

func (suite *ssaoHandlerTest) TestWithSSAOSampleCount() {
	suite.Run("overrides sample count", func() {
		h := light.NewSSAOHandler(light.WithSSAOSampleCount(32))
		suite.Equal(32, h.SampleCount())
	})
}

func (suite *ssaoHandlerTest) TestWithSSAORadius() {
	suite.Run("overrides radius", func() {
		h := light.NewSSAOHandler(light.WithSSAORadius(1.5))
		suite.InDelta(1.5, h.Radius(), 1e-6)
	})
}

func (suite *ssaoHandlerTest) TestWithSSAOBias() {
	suite.Run("overrides bias", func() {
		h := light.NewSSAOHandler(light.WithSSAOBias(0.05))
		suite.InDelta(0.05, h.Bias(), 1e-6)
	})
}

func (suite *ssaoHandlerTest) TestWithSSAOPower() {
	suite.Run("overrides power", func() {
		h := light.NewSSAOHandler(light.WithSSAOPower(3.0))
		suite.InDelta(3.0, h.Power(), 1e-6)
	})
}

func (suite *ssaoHandlerTest) TestWithSSAOBlurRadius() {
	suite.Run("overrides blur radius", func() {
		h := light.NewSSAOHandler(light.WithSSAOBlurRadius(8))
		suite.Equal(8, h.BlurRadius())
	})
}

func (suite *ssaoHandlerTest) TestWithSSAOHalfResolution() {
	suite.Run("enables half resolution", func() {
		h := light.NewSSAOHandler(light.WithSSAOHalfResolution(true))
		suite.True(h.HalfResolution())
	})
}

func (suite *ssaoHandlerTest) TestSetEnabled() {
	suite.Run("toggles enabled state", func() {
		h := light.NewSSAOHandler()
		suite.False(h.Enabled())
		h.SetEnabled(true)
		suite.True(h.Enabled())
		h.SetEnabled(false)
		suite.False(h.Enabled())
	})
}

func (suite *ssaoHandlerTest) TestSetHalfResolution() {
	suite.Run("toggles half resolution", func() {
		h := light.NewSSAOHandler()
		suite.False(h.HalfResolution())
		h.SetHalfResolution(true)
		suite.True(h.HalfResolution())
		h.SetHalfResolution(false)
		suite.False(h.HalfResolution())
	})
}

func (suite *ssaoHandlerTest) TestPipelineKeys() {
	suite.Run("set and retrieve pipeline key", func() {
		h := light.NewSSAOHandler()
		h.SetPipelineKey("ssao", "pipeline-ssao")
		suite.Equal("pipeline-ssao", h.PipelineKey("ssao"))
	})

	suite.Run("missing key returns empty string", func() {
		h := light.NewSSAOHandler()
		suite.Equal("", h.PipelineKey("nonexistent"))
	})

	suite.Run("pipeline keys map returns all entries", func() {
		h := light.NewSSAOHandler()
		h.SetPipelineKey("a", "key-a")
		h.SetPipelineKey("b", "key-b")
		keys := h.PipelineKeys()
		suite.Len(keys, 2)
	})
}

func (suite *ssaoHandlerTest) TestBgps() {
	suite.Run("returns full bgp map", func() {
		h := light.NewSSAOHandler()
		bgps := h.Bgps()
		suite.Len(bgps, 3)
	})
}

func (suite *ssaoHandlerTest) TestResize() {
	suite.Run("updates screen dimensions", func() {
		h := light.NewSSAOHandler()
		h.Resize(2560, 1440)
		suite.Equal(2560, h.ScreenWidth())
		suite.Equal(1440, h.ScreenHeight())
	})
}

func (suite *ssaoHandlerTest) TestSetBgp() {
	suite.Run("adds new bgp entry", func() {
		h := light.NewSSAOHandler()
		bgp := bind_group_provider.NewBindGroupProvider("custom_ssao_bgp")
		h.SetBgp("custom", bgp)
		suite.Equal(bgp, h.Bgp("custom"))
	})

	suite.Run("overwrites existing bgp entry", func() {
		h := light.NewSSAOHandler()
		replacement := bind_group_provider.NewBindGroupProvider("replaced_ssao_compute")
		h.SetBgp("ssao_compute", replacement)
		suite.Equal(replacement, h.Bgp("ssao_compute"))
		suite.Equal("replaced_ssao_compute", h.Bgp("ssao_compute").Label())
	})
}

func (suite *ssaoHandlerTest) TestSetRawTexture() {
	suite.Run("sets and retrieves raw texture", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.RawTexture())
		tex := &wgpu.Texture{}
		h.SetRawTexture(tex)
		suite.Equal(tex, h.RawTexture())
	})
}

func (suite *ssaoHandlerTest) TestSetRawTextureView() {
	suite.Run("sets and retrieves raw texture view", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.RawTextureView())
		tv := &wgpu.TextureView{}
		h.SetRawTextureView(tv)
		suite.Equal(tv, h.RawTextureView())
	})
}

func (suite *ssaoHandlerTest) TestSetBlurredTexture() {
	suite.Run("sets and retrieves blurred texture", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.BlurredTexture())
		tex := &wgpu.Texture{}
		h.SetBlurredTexture(tex)
		suite.Equal(tex, h.BlurredTexture())
	})
}

func (suite *ssaoHandlerTest) TestSetBlurredTextureView() {
	suite.Run("sets and retrieves blurred texture view", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.BlurredTextureView())
		tv := &wgpu.TextureView{}
		h.SetBlurredTextureView(tv)
		suite.Equal(tv, h.BlurredTextureView())
	})
}

func (suite *ssaoHandlerTest) TestSetScratchTexture() {
	suite.Run("sets and retrieves scratch texture", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.ScratchTexture())
		tex := &wgpu.Texture{}
		h.SetScratchTexture(tex)
		suite.Equal(tex, h.ScratchTexture())
	})
}

func (suite *ssaoHandlerTest) TestSetScratchTextureView() {
	suite.Run("sets and retrieves scratch texture view", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.ScratchTextureView())
		tv := &wgpu.TextureView{}
		h.SetScratchTextureView(tv)
		suite.Equal(tv, h.ScratchTextureView())
	})
}

func (suite *ssaoHandlerTest) TestSetNoiseTexture() {
	suite.Run("sets and retrieves noise texture", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.NoiseTexture())
		tex := &wgpu.Texture{}
		h.SetNoiseTexture(tex)
		suite.Equal(tex, h.NoiseTexture())
	})
}

func (suite *ssaoHandlerTest) TestSetNoiseTextureView() {
	suite.Run("sets and retrieves noise texture view", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.NoiseTextureView())
		tv := &wgpu.TextureView{}
		h.SetNoiseTextureView(tv)
		suite.Equal(tv, h.NoiseTextureView())
	})
}

func (suite *ssaoHandlerTest) TestSetLinearSampler() {
	suite.Run("sets and retrieves linear sampler", func() {
		h := light.NewSSAOHandler()
		suite.Nil(h.LinearSampler())
		s := &wgpu.Sampler{}
		h.SetLinearSampler(s)
		suite.Equal(s, h.LinearSampler())
	})
}
