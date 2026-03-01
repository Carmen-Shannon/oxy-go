package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type ssrHandlerTest struct {
	suite.Suite
}

func TestSSRHandler(t *testing.T) {
	suite.Run(t, new(ssrHandlerTest))
}

func (suite *ssrHandlerTest) TestNewSSRHandler() {
	suite.Run("default enabled is false", func() {
		h := light.NewSSRHandler()
		suite.False(h.Enabled())
	})

	suite.Run("default max steps is 64", func() {
		h := light.NewSSRHandler()
		suite.Equal(64, h.MaxSteps())
	})

	suite.Run("default max distance is 50.0", func() {
		h := light.NewSSRHandler()
		suite.InDelta(50.0, h.MaxDistance(), 1e-6)
	})

	suite.Run("default thickness is 0.1", func() {
		h := light.NewSSRHandler()
		suite.InDelta(0.1, h.Thickness(), 1e-6)
	})

	suite.Run("default stride is 1.0", func() {
		h := light.NewSSRHandler()
		suite.InDelta(1.0, h.Stride(), 1e-6)
	})

	suite.Run("default roughness cutoff is 0.5", func() {
		h := light.NewSSRHandler()
		suite.InDelta(0.5, h.RoughnessCutoff(), 1e-6)
	})

	suite.Run("default screen dimensions are zero", func() {
		h := light.NewSSRHandler()
		suite.Equal(0, h.ScreenWidth())
		suite.Equal(0, h.ScreenHeight())
	})

	suite.Run("default pipeline keys are empty", func() {
		h := light.NewSSRHandler()
		suite.Empty(h.PipelineKeys())
	})

	suite.Run("default bgps contain ssr_compute key", func() {
		h := light.NewSSRHandler()
		suite.NotNil(h.Bgp("ssr_compute"))
		suite.Equal("ssr_compute", h.Bgp("ssr_compute").Label())
	})

	suite.Run("default SSR texture is nil", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.SSRTexture())
		suite.Nil(h.SSRTextureView())
	})

	suite.Run("default linear sampler is nil", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.LinearSampler())
	})

	suite.Run("default HiZ texture is nil", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.HiZTexture())
		suite.Nil(h.HiZTextureView())
	})

	suite.Run("default HiZ mip count is zero", func() {
		h := light.NewSSRHandler()
		suite.Equal(0, h.HiZMipCount())
	})

	suite.Run("default HiZ mip views are nil", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.HiZMipReadViews())
		suite.Nil(h.HiZStorageViews())
	})
}

func (suite *ssrHandlerTest) TestWithSSRScreenSize() {
	suite.Run("sets screen dimensions", func() {
		h := light.NewSSRHandler(light.WithSSRScreenSize(1920, 1080))
		suite.Equal(1920, h.ScreenWidth())
		suite.Equal(1080, h.ScreenHeight())
	})
}

func (suite *ssrHandlerTest) TestWithSSRMaxSteps() {
	suite.Run("overrides max steps", func() {
		h := light.NewSSRHandler(light.WithSSRMaxSteps(128))
		suite.Equal(128, h.MaxSteps())
	})
}

func (suite *ssrHandlerTest) TestWithSSRMaxDistance() {
	suite.Run("overrides max distance", func() {
		h := light.NewSSRHandler(light.WithSSRMaxDistance(100.0))
		suite.InDelta(100.0, h.MaxDistance(), 1e-6)
	})
}

func (suite *ssrHandlerTest) TestWithSSRThickness() {
	suite.Run("overrides thickness", func() {
		h := light.NewSSRHandler(light.WithSSRThickness(0.5))
		suite.InDelta(0.5, h.Thickness(), 1e-6)
	})
}

func (suite *ssrHandlerTest) TestWithSSRStride() {
	suite.Run("overrides stride", func() {
		h := light.NewSSRHandler(light.WithSSRStride(2.0))
		suite.InDelta(2.0, h.Stride(), 1e-6)
	})
}

func (suite *ssrHandlerTest) TestWithSSRRoughnessCutoff() {
	suite.Run("overrides roughness cutoff", func() {
		h := light.NewSSRHandler(light.WithSSRRoughnessCutoff(0.8))
		suite.InDelta(0.8, h.RoughnessCutoff(), 1e-6)
	})
}

func (suite *ssrHandlerTest) TestSetEnabled() {
	suite.Run("toggles enabled state", func() {
		h := light.NewSSRHandler()
		suite.False(h.Enabled())
		h.SetEnabled(true)
		suite.True(h.Enabled())
		h.SetEnabled(false)
		suite.False(h.Enabled())
	})
}

func (suite *ssrHandlerTest) TestPipelineKeys() {
	suite.Run("set and retrieve pipeline key", func() {
		h := light.NewSSRHandler()
		h.SetPipelineKey("ssr", "pipeline-ssr")
		suite.Equal("pipeline-ssr", h.PipelineKey("ssr"))
	})

	suite.Run("missing key returns empty string", func() {
		h := light.NewSSRHandler()
		suite.Equal("", h.PipelineKey("nonexistent"))
	})

	suite.Run("pipeline keys map returns all entries", func() {
		h := light.NewSSRHandler()
		h.SetPipelineKey("a", "key-a")
		h.SetPipelineKey("b", "key-b")
		keys := h.PipelineKeys()
		suite.Len(keys, 2)
	})
}

func (suite *ssrHandlerTest) TestBgps() {
	suite.Run("returns full bgp map", func() {
		h := light.NewSSRHandler()
		bgps := h.Bgps()
		suite.Len(bgps, 1)
	})
}

func (suite *ssrHandlerTest) TestSetHiZMipCount() {
	suite.Run("sets and retrieves mip count", func() {
		h := light.NewSSRHandler()
		h.SetHiZMipCount(10)
		suite.Equal(10, h.HiZMipCount())
	})
}

func (suite *ssrHandlerTest) TestResize() {
	suite.Run("updates screen dimensions", func() {
		h := light.NewSSRHandler()
		h.Resize(2560, 1440)
		suite.Equal(2560, h.ScreenWidth())
		suite.Equal(1440, h.ScreenHeight())
	})
}

func (suite *ssrHandlerTest) TestSetBgp() {
	suite.Run("adds new bgp entry", func() {
		h := light.NewSSRHandler()
		bgp := bind_group_provider.NewBindGroupProvider("custom_ssr_bgp")
		h.SetBgp("custom", bgp)
		suite.Equal(bgp, h.Bgp("custom"))
	})

	suite.Run("overwrites existing bgp entry", func() {
		h := light.NewSSRHandler()
		replacement := bind_group_provider.NewBindGroupProvider("replaced_ssr")
		h.SetBgp("ssr_compute", replacement)
		suite.Equal(replacement, h.Bgp("ssr_compute"))
		suite.Equal("replaced_ssr", h.Bgp("ssr_compute").Label())
	})
}

func (suite *ssrHandlerTest) TestSetSSRTexture() {
	suite.Run("sets and retrieves SSR texture", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.SSRTexture())
		tex := &wgpu.Texture{}
		h.SetSSRTexture(tex)
		suite.Equal(tex, h.SSRTexture())
	})
}

func (suite *ssrHandlerTest) TestSetSSRTextureView() {
	suite.Run("sets and retrieves SSR texture view", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.SSRTextureView())
		tv := &wgpu.TextureView{}
		h.SetSSRTextureView(tv)
		suite.Equal(tv, h.SSRTextureView())
	})
}

func (suite *ssrHandlerTest) TestSetLinearSampler() {
	suite.Run("sets and retrieves linear sampler", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.LinearSampler())
		s := &wgpu.Sampler{}
		h.SetLinearSampler(s)
		suite.Equal(s, h.LinearSampler())
	})
}

func (suite *ssrHandlerTest) TestSetHiZTexture() {
	suite.Run("sets and retrieves HiZ texture", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.HiZTexture())
		tex := &wgpu.Texture{}
		h.SetHiZTexture(tex)
		suite.Equal(tex, h.HiZTexture())
	})
}

func (suite *ssrHandlerTest) TestSetHiZTextureView() {
	suite.Run("sets and retrieves HiZ texture view", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.HiZTextureView())
		tv := &wgpu.TextureView{}
		h.SetHiZTextureView(tv)
		suite.Equal(tv, h.HiZTextureView())
	})
}

func (suite *ssrHandlerTest) TestSetHiZMipReadViews() {
	suite.Run("sets and retrieves HiZ mip read views", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.HiZMipReadViews())
		views := []*wgpu.TextureView{{}, {}, {}}
		h.SetHiZMipReadViews(views)
		suite.Equal(views, h.HiZMipReadViews())
		suite.Len(h.HiZMipReadViews(), 3)
	})
}

func (suite *ssrHandlerTest) TestSetHiZStorageViews() {
	suite.Run("sets and retrieves HiZ storage views", func() {
		h := light.NewSSRHandler()
		suite.Nil(h.HiZStorageViews())
		views := []*wgpu.TextureView{{}, {}}
		h.SetHiZStorageViews(views)
		suite.Equal(views, h.HiZStorageViews())
		suite.Len(h.HiZStorageViews(), 2)
	})
}
