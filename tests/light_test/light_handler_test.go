package light_test

import (
	"fmt"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/stretchr/testify/suite"
)

type lightHandlerTest struct {
	suite.Suite
}

func TestLightHandler(t *testing.T) {
	suite.Run(t, new(lightHandlerTest))
}

func (suite *lightHandlerTest) TestNewLightingHandler() {
	suite.Run("default enabled is false", func() {
		h := light.NewLightingHandler()
		suite.False(h.Enabled())
	})

	suite.Run("default lights list is empty", func() {
		h := light.NewLightingHandler()
		suite.Empty(h.Lights())
	})

	suite.Run("default ambient color is black", func() {
		h := light.NewLightingHandler()
		suite.Equal([3]float32{0, 0, 0}, h.AmbientColor())
	})

	suite.Run("default bgps contain all required keys", func() {
		h := light.NewLightingHandler()
		bgps := h.Bgps()
		suite.NotNil(bgps["lights"])
		suite.NotNil(bgps["shadow_data"])
		suite.NotNil(bgps["shadow_lit"])
		suite.NotNil(bgps["light_cull"])
		suite.NotNil(bgps["tile_lit"])
		suite.NotNil(bgps["vsm_blur_h"])
		suite.NotNil(bgps["vsm_blur_v"])
		suite.NotNil(bgps["sat_prepare"])
	})

	suite.Run("bgp labels match their keys", func() {
		h := light.NewLightingHandler()
		suite.Equal("lights", h.Bgp("lights").Label())
		suite.Equal("shadow_data", h.Bgp("shadow_data").Label())
		suite.Equal("shadow_lit", h.Bgp("shadow_lit").Label())
		suite.Equal("light_cull", h.Bgp("light_cull").Label())
		suite.Equal("tile_lit", h.Bgp("tile_lit").Label())
		suite.Equal("vsm_blur_h", h.Bgp("vsm_blur_h").Label())
		suite.Equal("vsm_blur_v", h.Bgp("vsm_blur_v").Label())
		suite.Equal("sat_prepare", h.Bgp("sat_prepare").Label())
	})

	suite.Run("default pipeline keys map is empty", func() {
		h := light.NewLightingHandler()
		suite.Empty(h.PipelineKeys())
	})

	suite.Run("default shadow half extent matches constant", func() {
		h := light.NewLightingHandler()
		suite.Equal(light.DefaultShadowHalfExtent, h.ShadowHalfExtent())
	})

	suite.Run("default shadow near matches constant", func() {
		h := light.NewLightingHandler()
		suite.Equal(light.DefaultShadowNear, h.ShadowNear())
	})

	suite.Run("default shadow far matches constant", func() {
		h := light.NewLightingHandler()
		suite.Equal(light.DefaultShadowFar, h.ShadowFar())
	})

	suite.Run("default shadow bias matches constant", func() {
		h := light.NewLightingHandler()
		suite.Equal(light.DefaultShadowBias, h.ShadowBias())
	})

	suite.Run("default shadow normal bias scale matches constant", func() {
		h := light.NewLightingHandler()
		suite.Equal(light.DefaultShadowNormalBiasScale, h.ShadowNormalBiasScale())
	})

	suite.Run("default shadow map resolution matches constant", func() {
		h := light.NewLightingHandler()
		suite.Equal(light.ShadowMapResolution, h.ShadowMapResolution())
	})

	suite.Run("default VSM texture is nil", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMTexture())
	})

	suite.Run("default VSM texture view is nil", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMTextureView())
	})

	suite.Run("default VSM scratch texture is nil", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMScratchTexture())
	})

	suite.Run("default VSM scratch texture view is nil", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMScratchTextureView())
	})

	suite.Run("default VSM aux depth texture is nil", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMAuxDepthTexture())
	})

	suite.Run("default VSM aux depth texture view is nil", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMAuxDepthTextureView())
	})

	suite.Run("default VSM linear sampler is nil", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMLinearSampler())
	})

	suite.Run("default VSM blur radius matches constant", func() {
		h := light.NewLightingHandler()
		suite.Equal(light.DefaultVSMBlurRadius, h.VSMBlurRadius())
	})

	suite.Run("default VSM min variance matches constant", func() {
		h := light.NewLightingHandler()
		suite.Equal(light.DefaultVSMMinVariance, h.VSMMinVariance())
	})

	suite.Run("default VSM light bleed reduction matches constant", func() {
		h := light.NewLightingHandler()
		suite.Equal(light.DefaultVSMLightBleedReduction, h.VSMLightBleedReduction())
	})

	suite.Run("default VSM light size matches constant", func() {
		h := light.NewLightingHandler()
		suite.Equal(light.DefaultVSMLightSize, h.VSMLightSize())
	})

	suite.Run("default PCSS is disabled", func() {
		h := light.NewLightingHandler()
		suite.False(h.PCSSEnabled())
	})

	suite.Run("default SAT textures are nil", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.SATTextureA())
		suite.Nil(h.SATTextureAView())
		suite.Nil(h.SATTextureB())
		suite.Nil(h.SATTextureBView())
	})

	suite.Run("default screen dimensions are zero", func() {
		h := light.NewLightingHandler()
		suite.Equal(0, h.ScreenWidth())
		suite.Equal(0, h.ScreenHeight())
	})

	suite.Run("default tile counts are zero", func() {
		h := light.NewLightingHandler()
		suite.Equal(uint32(0), h.TileCountX())
		suite.Equal(uint32(0), h.TileCountY())
	})
}

func (suite *lightHandlerTest) TestNewLightingHandlerWithOptions() {
	suite.Run("WithShadowHalfExtent overrides default", func() {
		h := light.NewLightingHandler(light.WithShadowHalfExtent(80.0))
		suite.Equal(float32(80.0), h.ShadowHalfExtent())
	})

	suite.Run("WithShadowNearFar overrides defaults", func() {
		h := light.NewLightingHandler(light.WithShadowNearFar(0.5, 500.0))
		suite.Equal(float32(0.5), h.ShadowNear())
		suite.Equal(float32(500.0), h.ShadowFar())
	})

	suite.Run("WithShadowBias overrides default", func() {
		h := light.NewLightingHandler(light.WithShadowBias(0.005))
		suite.Equal(float32(0.005), h.ShadowBias())
	})

	suite.Run("WithShadowNormalBiasScale overrides default", func() {
		h := light.NewLightingHandler(light.WithShadowNormalBiasScale(4.0))
		suite.Equal(float32(4.0), h.ShadowNormalBiasScale())
	})

	suite.Run("WithShadowMapResolution overrides default", func() {
		h := light.NewLightingHandler(light.WithShadowMapResolution(4096))
		suite.Equal(4096, h.ShadowMapResolution())
	})

	suite.Run("WithAmbientColor overrides default", func() {
		h := light.NewLightingHandler(light.WithAmbientColor([3]float32{0.1, 0.2, 0.3}))
		suite.Equal([3]float32{0.1, 0.2, 0.3}, h.AmbientColor())
	})

	suite.Run("WithVSMBlurRadius overrides default", func() {
		h := light.NewLightingHandler(light.WithVSMBlurRadius(8))
		suite.Equal(8, h.VSMBlurRadius())
	})

	suite.Run("WithVSMMinVariance overrides default", func() {
		h := light.NewLightingHandler(light.WithVSMMinVariance(0.001))
		suite.Equal(float32(0.001), h.VSMMinVariance())
	})

	suite.Run("WithVSMLightBleedReduction overrides default", func() {
		h := light.NewLightingHandler(light.WithVSMLightBleedReduction(0.5))
		suite.Equal(float32(0.5), h.VSMLightBleedReduction())
	})

	suite.Run("WithVSMLightSize overrides default", func() {
		h := light.NewLightingHandler(light.WithVSMLightSize(2.5))
		suite.Equal(float32(2.5), h.VSMLightSize())
	})

	suite.Run("WithPCSSEnabled enables PCSS", func() {
		h := light.NewLightingHandler(light.WithPCSSEnabled(true))
		suite.True(h.PCSSEnabled())
	})

	suite.Run("WithPCSSEnabled pre-creates SAT pass BGPs", func() {
		h := light.NewLightingHandler(light.WithPCSSEnabled(true))
		numPasses := 0
		for v := light.ShadowMapResolution; v > 1; v >>= 1 {
			numPasses++
		}
		for i := 0; i < 2*numPasses; i++ {
			suite.NotNil(h.Bgp(fmt.Sprintf("sat_pass_%d", i)))
		}
	})

	suite.Run("combined options apply correctly", func() {
		h := light.NewLightingHandler(
			light.WithShadowHalfExtent(60.0),
			light.WithShadowNearFar(1.0, 300.0),
			light.WithShadowBias(0.002),
			light.WithShadowNormalBiasScale(2.5),
			light.WithShadowMapResolution(1024),
			light.WithAmbientColor([3]float32{0.5, 0.5, 0.5}),
			light.WithVSMBlurRadius(6),
			light.WithVSMMinVariance(0.0001),
			light.WithVSMLightBleedReduction(0.4),
			light.WithVSMLightSize(2.0),
			light.WithPCSSEnabled(true),
		)
		suite.Equal(float32(60.0), h.ShadowHalfExtent())
		suite.Equal(float32(1.0), h.ShadowNear())
		suite.Equal(float32(300.0), h.ShadowFar())
		suite.Equal(float32(0.002), h.ShadowBias())
		suite.Equal(float32(2.5), h.ShadowNormalBiasScale())
		suite.Equal(1024, h.ShadowMapResolution())
		suite.Equal([3]float32{0.5, 0.5, 0.5}, h.AmbientColor())
		suite.Equal(6, h.VSMBlurRadius())
		suite.Equal(float32(0.0001), h.VSMMinVariance())
		suite.Equal(float32(0.4), h.VSMLightBleedReduction())
		suite.Equal(float32(2.0), h.VSMLightSize())
		suite.True(h.PCSSEnabled())
	})
}

func (suite *lightHandlerTest) TestSetEnabled() {
	suite.Run("enables the handler", func() {
		h := light.NewLightingHandler()
		h.SetEnabled(true)
		suite.True(h.Enabled())
	})

	suite.Run("disables the handler", func() {
		h := light.NewLightingHandler()
		h.SetEnabled(true)
		h.SetEnabled(false)
		suite.False(h.Enabled())
	})
}

func (suite *lightHandlerTest) TestAddLight() {
	suite.Run("adds a single light", func() {
		h := light.NewLightingHandler()
		l := light.NewLight(light.LightTypePoint)
		h.AddLight(l)
		suite.Len(h.Lights(), 1)
	})

	suite.Run("adds multiple lights in order", func() {
		h := light.NewLightingHandler()
		l1 := light.NewLight(light.LightTypePoint, light.WithPosition(1, 0, 0))
		l2 := light.NewLight(light.LightTypeDirectional)
		l3 := light.NewLight(light.LightTypeSpot, light.WithPosition(3, 0, 0))
		h.AddLight(l1)
		h.AddLight(l2)
		h.AddLight(l3)
		lights := h.Lights()
		suite.Len(lights, 3)
		suite.Equal([3]float32{1, 0, 0}, lights[0].Position())
		suite.Equal(light.LightTypeDirectional, lights[1].Type())
		suite.Equal([3]float32{3, 0, 0}, lights[2].Position())
	})
}

func (suite *lightHandlerTest) TestRemoveLight() {
	suite.Run("removes an existing light", func() {
		h := light.NewLightingHandler()
		l1 := light.NewLight(light.LightTypePoint, light.WithPosition(1, 0, 0))
		l2 := light.NewLight(light.LightTypePoint, light.WithPosition(2, 0, 0))
		h.AddLight(l1)
		h.AddLight(l2)
		h.RemoveLight(l1)
		lights := h.Lights()
		suite.Len(lights, 1)
		suite.Equal([3]float32{2, 0, 0}, lights[0].Position())
	})

	suite.Run("removing non-existent light does nothing", func() {
		h := light.NewLightingHandler()
		l1 := light.NewLight(light.LightTypePoint)
		l2 := light.NewLight(light.LightTypePoint)
		h.AddLight(l1)
		h.RemoveLight(l2)
		suite.Len(h.Lights(), 1)
	})

	suite.Run("removing from empty list does nothing", func() {
		h := light.NewLightingHandler()
		l := light.NewLight(light.LightTypePoint)
		h.RemoveLight(l)
		suite.Empty(h.Lights())
	})
}

func (suite *lightHandlerTest) TestLights() {
	suite.Run("returns a copy not a reference", func() {
		h := light.NewLightingHandler()
		l := light.NewLight(light.LightTypePoint)
		h.AddLight(l)
		cp := h.Lights()
		cp[0] = nil
		suite.NotNil(h.Lights()[0])
	})
}

func (suite *lightHandlerTest) TestSetAmbientColor() {
	suite.Run("updates ambient color", func() {
		h := light.NewLightingHandler()
		h.SetAmbientColor([3]float32{0.4, 0.5, 0.6})
		suite.Equal([3]float32{0.4, 0.5, 0.6}, h.AmbientColor())
	})

	suite.Run("overwrites previous ambient color", func() {
		h := light.NewLightingHandler(light.WithAmbientColor([3]float32{1, 1, 1}))
		h.SetAmbientColor([3]float32{0, 0, 0})
		suite.Equal([3]float32{0, 0, 0}, h.AmbientColor())
	})
}

func (suite *lightHandlerTest) TestBgp() {
	suite.Run("returns nil for unknown key", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.Bgp("nonexistent"))
	})

	suite.Run("returns correct provider for known key", func() {
		h := light.NewLightingHandler()
		suite.NotNil(h.Bgp("lights"))
		suite.Equal("lights", h.Bgp("lights").Label())
	})
}

func (suite *lightHandlerTest) TestPipelineKey() {
	suite.Run("returns empty string for unknown name", func() {
		h := light.NewLightingHandler()
		suite.Equal("", h.PipelineKey("unknown"))
	})

	suite.Run("returns value after being set", func() {
		h := light.NewLightingHandler()
		h.SetPipelineKey("shadow", "shadow-depth-pipeline")
		suite.Equal("shadow-depth-pipeline", h.PipelineKey("shadow"))
	})
}

func (suite *lightHandlerTest) TestSetPipelineKey() {
	suite.Run("sets a new key", func() {
		h := light.NewLightingHandler()
		h.SetPipelineKey("lit", "lit-forward-pipeline")
		suite.Equal("lit-forward-pipeline", h.PipelineKey("lit"))
	})

	suite.Run("overwrites existing key", func() {
		h := light.NewLightingHandler()
		h.SetPipelineKey("lit", "first")
		h.SetPipelineKey("lit", "second")
		suite.Equal("second", h.PipelineKey("lit"))
	})

	suite.Run("multiple keys are independent", func() {
		h := light.NewLightingHandler()
		h.SetPipelineKey("shadow", "shadow-key")
		h.SetPipelineKey("lit", "lit-key")
		suite.Equal("shadow-key", h.PipelineKey("shadow"))
		suite.Equal("lit-key", h.PipelineKey("lit"))
	})
}

func (suite *lightHandlerTest) TestPipelineKeys() {
	suite.Run("returns all set keys", func() {
		h := light.NewLightingHandler()
		h.SetPipelineKey("a", "key-a")
		h.SetPipelineKey("b", "key-b")
		keys := h.PipelineKeys()
		suite.Len(keys, 2)
		suite.Equal("key-a", keys["a"])
		suite.Equal("key-b", keys["b"])
	})
}

func (suite *lightHandlerTest) TestVSMTextureSetters() {
	suite.Run("VSMTexture set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMTexture())
		h.SetVSMTexture(nil)
		suite.Nil(h.VSMTexture())
	})

	suite.Run("VSMTextureView set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMTextureView())
		h.SetVSMTextureView(nil)
		suite.Nil(h.VSMTextureView())
	})

	suite.Run("VSMScratchTexture set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMScratchTexture())
		h.SetVSMScratchTexture(nil)
		suite.Nil(h.VSMScratchTexture())
	})

	suite.Run("VSMScratchTextureView set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMScratchTextureView())
		h.SetVSMScratchTextureView(nil)
		suite.Nil(h.VSMScratchTextureView())
	})

	suite.Run("VSMAuxDepthTexture set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMAuxDepthTexture())
		h.SetVSMAuxDepthTexture(nil)
		suite.Nil(h.VSMAuxDepthTexture())
	})

	suite.Run("VSMAuxDepthTextureView set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMAuxDepthTextureView())
		h.SetVSMAuxDepthTextureView(nil)
		suite.Nil(h.VSMAuxDepthTextureView())
	})

	suite.Run("VSMLinearSampler set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.VSMLinearSampler())
		h.SetVSMLinearSampler(nil)
		suite.Nil(h.VSMLinearSampler())
	})
}

func (suite *lightHandlerTest) TestSATTextureSetters() {
	suite.Run("SATTextureA set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.SATTextureA())
		h.SetSATTextureA(nil)
		suite.Nil(h.SATTextureA())
	})

	suite.Run("SATTextureAView set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.SATTextureAView())
		h.SetSATTextureAView(nil)
		suite.Nil(h.SATTextureAView())
	})

	suite.Run("SATTextureB set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.SATTextureB())
		h.SetSATTextureB(nil)
		suite.Nil(h.SATTextureB())
	})

	suite.Run("SATTextureBView set and get round-trips", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.SATTextureBView())
		h.SetSATTextureBView(nil)
		suite.Nil(h.SATTextureBView())
	})
}

func (suite *lightHandlerTest) TestPCSSEnabled() {
	suite.Run("enables PCSS", func() {
		h := light.NewLightingHandler()
		h.SetPCSSEnabled(true)
		suite.True(h.PCSSEnabled())
	})

	suite.Run("disables PCSS", func() {
		h := light.NewLightingHandler()
		h.SetPCSSEnabled(true)
		h.SetPCSSEnabled(false)
		suite.False(h.PCSSEnabled())
	})
}

func (suite *lightHandlerTest) TestResize() {
	suite.Run("updates screen dimensions", func() {
		h := light.NewLightingHandler()
		h.Resize(1920, 1080)
		suite.Equal(1920, h.ScreenWidth())
		suite.Equal(1080, h.ScreenHeight())
	})

	suite.Run("recalculates tile counts", func() {
		h := light.NewLightingHandler()
		h.Resize(1920, 1080)
		expectedTX, expectedTY := light.TileCounts(1920, 1080)
		suite.Equal(expectedTX, h.TileCountX())
		suite.Equal(expectedTY, h.TileCountY())
	})

	suite.Run("resize to different dimensions updates all fields", func() {
		h := light.NewLightingHandler()
		h.Resize(1920, 1080)
		h.Resize(2560, 1440)
		suite.Equal(2560, h.ScreenWidth())
		suite.Equal(1440, h.ScreenHeight())
		expectedTX, expectedTY := light.TileCounts(2560, 1440)
		suite.Equal(expectedTX, h.TileCountX())
		suite.Equal(expectedTY, h.TileCountY())
	})

	suite.Run("single pixel screen produces one tile each", func() {
		h := light.NewLightingHandler()
		h.Resize(1, 1)
		suite.Equal(uint32(1), h.TileCountX())
		suite.Equal(uint32(1), h.TileCountY())
	})
}

func (suite *lightHandlerTest) TestSubHandlerGetters() {
	suite.Run("default GBufferHandler is not nil", func() {
		h := light.NewLightingHandler()
		suite.NotNil(h.GBufferHandler())
	})

	suite.Run("default SSAOHandler is not nil", func() {
		h := light.NewLightingHandler()
		suite.NotNil(h.SSAOHandler())
	})

	suite.Run("default CompositionHandler is not nil", func() {
		h := light.NewLightingHandler()
		suite.NotNil(h.CompositionHandler())
	})

	suite.Run("default SSRHandler is not nil", func() {
		h := light.NewLightingHandler()
		suite.NotNil(h.SSRHandler())
	})

	suite.Run("default ProbeGrid is nil", func() {
		h := light.NewLightingHandler()
		suite.Nil(h.ProbeGrid())
	})
}

func (suite *lightHandlerTest) TestWithGBufferHandler() {
	suite.Run("overrides default GBufferHandler", func() {
		customGB := light.NewGBufferHandler(light.WithGBufferScreenSize(1920, 1080))
		h := light.NewLightingHandler(light.WithGBufferHandler(customGB))
		suite.Equal(1920, h.GBufferHandler().ScreenWidth())
		suite.Equal(1080, h.GBufferHandler().ScreenHeight())
	})
}

func (suite *lightHandlerTest) TestWithSSAOHandler() {
	suite.Run("overrides default SSAOHandler", func() {
		customSSAO := light.NewSSAOHandler(light.WithSSAOSampleCount(32))
		h := light.NewLightingHandler(light.WithSSAOHandler(customSSAO))
		suite.Equal(32, h.SSAOHandler().SampleCount())
	})
}

func (suite *lightHandlerTest) TestWithProbeGrid() {
	suite.Run("attaches probe grid", func() {
		grid := light.NewIrradianceProbeGrid()
		h := light.NewLightingHandler(light.WithProbeGrid(grid))
		suite.NotNil(h.ProbeGrid())
		suite.Equal(256, h.ProbeGrid().TotalProbes())
	})
}

func (suite *lightHandlerTest) TestWithCompositionHandler() {
	suite.Run("overrides default CompositionHandler", func() {
		customComp := light.NewCompositionHandler(light.WithExposure(3.0))
		h := light.NewLightingHandler(light.WithCompositionHandler(customComp))
		suite.InDelta(3.0, h.CompositionHandler().Exposure(), 1e-6)
	})
}

func (suite *lightHandlerTest) TestWithSSRHandler() {
	suite.Run("overrides default SSRHandler", func() {
		customSSR := light.NewSSRHandler(light.WithSSRMaxSteps(128))
		h := light.NewLightingHandler(light.WithSSRHandler(customSSR))
		suite.Equal(128, h.SSRHandler().MaxSteps())
	})
}
