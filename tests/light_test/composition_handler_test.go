package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type compositionHandlerTest struct {
	suite.Suite
}

func TestCompositionHandler(t *testing.T) {
	suite.Run(t, new(compositionHandlerTest))
}

func (suite *compositionHandlerTest) TestNewCompositionHandler() {
	suite.Run("default tone mapping is enabled", func() {
		h := light.NewCompositionHandler()
		suite.True(h.ToneMappingEnabled())
	})

	suite.Run("default exposure is 1.0", func() {
		h := light.NewCompositionHandler()
		suite.InDelta(1.0, h.Exposure(), 1e-6)
	})

	suite.Run("default enabled is false", func() {
		h := light.NewCompositionHandler()
		suite.False(h.Enabled())
	})

	suite.Run("default screen dimensions are zero", func() {
		h := light.NewCompositionHandler()
		suite.Equal(0, h.ScreenWidth())
		suite.Equal(0, h.ScreenHeight())
	})

	suite.Run("default pipeline keys are empty", func() {
		h := light.NewCompositionHandler()
		suite.Empty(h.PipelineKeys())
	})

	suite.Run("default bgps contain composition key", func() {
		h := light.NewCompositionHandler()
		suite.NotNil(h.Bgp("composition"))
		suite.Equal("composition", h.Bgp("composition").Label())
	})

	suite.Run("default HDR texture is nil", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.HDRTexture())
		suite.Nil(h.HDRTextureView())
	})

	suite.Run("default MSAA texture is nil", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.MSAATexture())
		suite.Nil(h.MSAATextureView())
	})

	suite.Run("default depth texture is nil", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.DepthTexture())
		suite.Nil(h.DepthTextureView())
	})

	suite.Run("default linear sampler is nil", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.LinearSampler())
	})
}

func (suite *compositionHandlerTest) TestWithCompositionScreenSize() {
	suite.Run("sets screen dimensions", func() {
		h := light.NewCompositionHandler(light.WithCompositionScreenSize(1920, 1080))
		suite.Equal(1920, h.ScreenWidth())
		suite.Equal(1080, h.ScreenHeight())
	})
}

func (suite *compositionHandlerTest) TestWithToneMappingEnabled() {
	suite.Run("disables tone mapping", func() {
		h := light.NewCompositionHandler(light.WithToneMappingEnabled(false))
		suite.False(h.ToneMappingEnabled())
	})
}

func (suite *compositionHandlerTest) TestWithExposure() {
	suite.Run("sets custom exposure", func() {
		h := light.NewCompositionHandler(light.WithExposure(2.5))
		suite.InDelta(2.5, h.Exposure(), 1e-6)
	})
}

func (suite *compositionHandlerTest) TestSetEnabled() {
	suite.Run("toggles enabled state", func() {
		h := light.NewCompositionHandler()
		suite.False(h.Enabled())
		h.SetEnabled(true)
		suite.True(h.Enabled())
		h.SetEnabled(false)
		suite.False(h.Enabled())
	})
}

func (suite *compositionHandlerTest) TestSetExposure() {
	suite.Run("updates exposure value", func() {
		h := light.NewCompositionHandler()
		h.SetExposure(3.0)
		suite.InDelta(3.0, h.Exposure(), 1e-6)
	})
}

func (suite *compositionHandlerTest) TestPipelineKeys() {
	suite.Run("set and retrieve pipeline key", func() {
		h := light.NewCompositionHandler()
		h.SetPipelineKey("comp", "pipeline-123")
		suite.Equal("pipeline-123", h.PipelineKey("comp"))
	})

	suite.Run("missing key returns empty string", func() {
		h := light.NewCompositionHandler()
		suite.Equal("", h.PipelineKey("nonexistent"))
	})

	suite.Run("pipeline keys map returns all entries", func() {
		h := light.NewCompositionHandler()
		h.SetPipelineKey("a", "key-a")
		h.SetPipelineKey("b", "key-b")
		keys := h.PipelineKeys()
		suite.Len(keys, 2)
		suite.Equal("key-a", keys["a"])
		suite.Equal("key-b", keys["b"])
	})
}

func (suite *compositionHandlerTest) TestBgps() {
	suite.Run("returns full bgp map", func() {
		h := light.NewCompositionHandler()
		bgps := h.Bgps()
		suite.Len(bgps, 1)
		suite.NotNil(bgps["composition"])
	})
}

func (suite *compositionHandlerTest) TestResize() {
	suite.Run("updates screen dimensions", func() {
		h := light.NewCompositionHandler()
		h.Resize(2560, 1440)
		suite.Equal(2560, h.ScreenWidth())
		suite.Equal(1440, h.ScreenHeight())
	})
}

func (suite *compositionHandlerTest) TestSetBgp() {
	suite.Run("adds new bgp entry", func() {
		h := light.NewCompositionHandler()
		bgp := bind_group_provider.NewBindGroupProvider("custom_bgp")
		h.SetBgp("custom", bgp)
		suite.Equal(bgp, h.Bgp("custom"))
	})

	suite.Run("overwrites existing bgp entry", func() {
		h := light.NewCompositionHandler()
		replacement := bind_group_provider.NewBindGroupProvider("replacement")
		h.SetBgp("composition", replacement)
		suite.Equal(replacement, h.Bgp("composition"))
		suite.Equal("replacement", h.Bgp("composition").Label())
	})
}

func (suite *compositionHandlerTest) TestSetHDRTexture() {
	suite.Run("sets and retrieves HDR texture", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.HDRTexture())
		tex := &wgpu.Texture{}
		h.SetHDRTexture(tex)
		suite.Equal(tex, h.HDRTexture())
	})
}

func (suite *compositionHandlerTest) TestSetHDRTextureView() {
	suite.Run("sets and retrieves HDR texture view", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.HDRTextureView())
		tv := &wgpu.TextureView{}
		h.SetHDRTextureView(tv)
		suite.Equal(tv, h.HDRTextureView())
	})
}

func (suite *compositionHandlerTest) TestSetMSAATexture() {
	suite.Run("sets and retrieves MSAA texture", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.MSAATexture())
		tex := &wgpu.Texture{}
		h.SetMSAATexture(tex)
		suite.Equal(tex, h.MSAATexture())
	})
}

func (suite *compositionHandlerTest) TestSetMSAATextureView() {
	suite.Run("sets and retrieves MSAA texture view", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.MSAATextureView())
		tv := &wgpu.TextureView{}
		h.SetMSAATextureView(tv)
		suite.Equal(tv, h.MSAATextureView())
	})
}

func (suite *compositionHandlerTest) TestSetDepthTexture() {
	suite.Run("sets and retrieves depth texture", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.DepthTexture())
		tex := &wgpu.Texture{}
		h.SetDepthTexture(tex)
		suite.Equal(tex, h.DepthTexture())
	})
}

func (suite *compositionHandlerTest) TestSetDepthTextureView() {
	suite.Run("sets and retrieves depth texture view", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.DepthTextureView())
		tv := &wgpu.TextureView{}
		h.SetDepthTextureView(tv)
		suite.Equal(tv, h.DepthTextureView())
	})
}

func (suite *compositionHandlerTest) TestSetLinearSampler() {
	suite.Run("sets and retrieves linear sampler", func() {
		h := light.NewCompositionHandler()
		suite.Nil(h.LinearSampler())
		s := &wgpu.Sampler{}
		h.SetLinearSampler(s)
		suite.Equal(s, h.LinearSampler())
	})
}
