package postprocessing_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

func TestRunTAAHandlerTests(t *testing.T) {
	suite.Run(t, new(taaHandlerTest))
}

type taaHandlerTest struct {
	suite.Suite
	handler postprocessing.TAAHandler
}

func (suite *taaHandlerTest) SetupSubTest() {
	suite.handler = postprocessing.NewTAAHandler()
}

func (suite *taaHandlerTest) TestNewTAAHandler() {
	suite.Run("should create a handler with sensible defaults", func() {
		suite.False(suite.handler.Enabled())
		suite.Equal(float32(0.1), suite.handler.BlendFactor())
		suite.Equal(0, suite.handler.ScreenWidth())
		suite.Equal(0, suite.handler.ScreenHeight())
		suite.Equal(uint64(0), suite.handler.FrameIndex())
		suite.Equal(float32(1.0), suite.handler.JitterScale())
	})

	suite.Run("should apply screen size and blend factor options", func() {
		h := postprocessing.NewTAAHandler(
			postprocessing.WithTAAScreenSize(1280, 720),
			postprocessing.WithTAABlendFactor(0.2),
		)
		suite.Equal(1280, h.ScreenWidth())
		suite.Equal(720, h.ScreenHeight())
		suite.Equal(float32(0.2), h.BlendFactor())
	})

	suite.Run("should apply jitter scale option", func() {
		h := postprocessing.NewTAAHandler(
			postprocessing.WithTAAJitterScale(0.5),
		)
		suite.Equal(float32(0.5), h.JitterScale())
	})

	suite.Run("should apply history rectification scale option", func() {
		h := postprocessing.NewTAAHandler(
			postprocessing.WithTAAHistoryRectificationScale(2.0),
		)
		suite.Equal(float32(2.0), h.HistoryRectificationScale())
	})
}

func (suite *taaHandlerTest) TestAdvanceFrame() {
	suite.Run("should move current jitter into previous jitter and increment the frame index", func() {
		suite.handler.AdvanceFrame(0.25, -0.5)
		suite.Equal(uint64(1), suite.handler.FrameIndex())
		suite.Equal(float32(0.25), suite.handler.JitterX())
		suite.Equal(float32(-0.5), suite.handler.JitterY())
		suite.Equal(float32(0), suite.handler.PrevJitterX())
		suite.Equal(float32(0), suite.handler.PrevJitterY())

		suite.handler.AdvanceFrame(-0.125, 0.75)
		suite.Equal(uint64(2), suite.handler.FrameIndex())
		suite.Equal(float32(-0.125), suite.handler.JitterX())
		suite.Equal(float32(0.75), suite.handler.JitterY())
		suite.Equal(float32(0.25), suite.handler.PrevJitterX())
		suite.Equal(float32(-0.5), suite.handler.PrevJitterY())
	})
}

func (suite *taaHandlerTest) TestBindGroupProviders() {
	suite.Run("should expose the default taa bind group providers", func() {
		bgps := suite.handler.Bgps()
		suite.Equal(4, len(bgps))
		suite.Contains(bgps, "taa_resolve_0")
		suite.Contains(bgps, "taa_resolve_1")
		suite.Contains(bgps, "taa_sharpen_0")
		suite.Contains(bgps, "taa_sharpen_1")
	})

	suite.Run("should allow replacing a bind group provider", func() {
		bgp := bind_group_provider.NewBindGroupProvider("custom-taa")
		suite.handler.SetBgp("taa_resolve_0", bgp)
		suite.Equal(bgp, suite.handler.Bgp("taa_resolve_0"))
	})
}

func (suite *taaHandlerTest) TestPipelineKeys() {
	suite.Run("should store pipeline keys by name", func() {
		suite.handler.SetPipelineKey("taa_resolve", "resolve-key")
		suite.handler.SetPipelineKey("taa_sharpen", "sharpen-key")

		suite.Equal("resolve-key", suite.handler.PipelineKey("taa_resolve"))
		suite.Equal("sharpen-key", suite.handler.PipelineKey("taa_sharpen"))
	})

	suite.Run("should return all pipeline keys as a map", func() {
		suite.handler.SetPipelineKey("taa_resolve", "resolve-key")
		suite.handler.SetPipelineKey("taa_sharpen", "sharpen-key")

		keys := suite.handler.PipelineKeys()
		suite.Equal(2, len(keys))
		suite.Equal("resolve-key", keys["taa_resolve"])
		suite.Equal("sharpen-key", keys["taa_sharpen"])
	})
}

func (suite *taaHandlerTest) TestSlotResources() {
	suite.Run("should store taa textures per active slot", func() {
		tex0 := new(wgpu.Texture)
		view0 := new(wgpu.TextureView)
		tex1 := new(wgpu.Texture)
		view1 := new(wgpu.TextureView)

		suite.handler.SetSlot(0)
		suite.handler.SetTAATexture(tex0)
		suite.handler.SetTAATextureView(view0)

		suite.handler.SetSlot(1)
		suite.handler.SetTAATexture(tex1)
		suite.handler.SetTAATextureView(view1)

		suite.handler.SetSlot(0)
		suite.True(suite.handler.TAATexture() == tex0)
		suite.True(suite.handler.TAATextureView() == view0)

		suite.handler.SetSlot(1)
		suite.True(suite.handler.TAATexture() == tex1)
		suite.True(suite.handler.TAATextureView() == view1)
	})

	suite.Run("should store the shared sampler and sharpen outputs", func() {
		sampler := new(wgpu.Sampler)
		sharpenTexture := new(wgpu.Texture)
		sharpenView := new(wgpu.TextureView)

		suite.handler.SetLinearSampler(sampler)
		suite.handler.SetSharpenTexture(sharpenTexture)
		suite.handler.SetSharpenTextureView(sharpenView)

		suite.True(suite.handler.LinearSampler() == sampler)
		suite.True(suite.handler.SharpenTexture() == sharpenTexture)
		suite.True(suite.handler.SharpenTextureView() == sharpenView)
	})
}

func (suite *taaHandlerTest) TestEnabledAndResize() {
	suite.Run("should toggle enabled state and update stored screen size", func() {
		suite.handler.SetEnabled(true)
		suite.handler.Resize(800, 600)

		suite.True(suite.handler.Enabled())
		suite.Equal(800, suite.handler.ScreenWidth())
		suite.Equal(600, suite.handler.ScreenHeight())
	})
}

func (suite *taaHandlerTest) TestJitterScale() {
	suite.Run("should return the default jitter scale", func() {
		suite.Equal(float32(1.0), suite.handler.JitterScale())
	})

	suite.Run("should update jitter scale via SetJitterScale", func() {
		suite.handler.SetJitterScale(0.75)
		suite.Equal(float32(0.75), suite.handler.JitterScale())
	})

	suite.Run("should allow setting jitter scale to zero", func() {
		suite.handler.SetJitterScale(0)
		suite.Equal(float32(0), suite.handler.JitterScale())
	})
}

func (suite *taaHandlerTest) TestRawHistoryOnly() {
	suite.Run("should default to false", func() {
		suite.False(suite.handler.RawHistoryOnly())
	})

	suite.Run("should return true after setting to true", func() {
		suite.handler.SetRawHistoryOnly(true)
		suite.True(suite.handler.RawHistoryOnly())
	})

	suite.Run("should return false after setting back to false", func() {
		suite.handler.SetRawHistoryOnly(true)
		suite.handler.SetRawHistoryOnly(false)
		suite.False(suite.handler.RawHistoryOnly())
	})
}

func (suite *taaHandlerTest) TestHistoryRectificationScale() {
	suite.Run("should default to 1.0", func() {
		suite.Equal(float32(1.0), suite.handler.HistoryRectificationScale())
	})

	suite.Run("should update via SetHistoryRectificationScale", func() {
		suite.handler.SetHistoryRectificationScale(2.5)
		suite.Equal(float32(2.5), suite.handler.HistoryRectificationScale())
	})

	suite.Run("should allow setting to zero", func() {
		suite.handler.SetHistoryRectificationScale(0)
		suite.Equal(float32(0), suite.handler.HistoryRectificationScale())
	})
}
