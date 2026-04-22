package light_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/stretchr/testify/suite"
)

func TestRunLightingHandlerTests(t *testing.T) {
	suite.Run(t, new(lightingHandlerTest))
}

type lightingHandlerTest struct {
	suite.Suite
	handler light.LightingHandler
}

func (suite *lightingHandlerTest) SetupSubTest() {
	suite.handler = light.NewLightingHandler()
}

func (suite *lightingHandlerTest) TestNewLightingHandler() {
	suite.Run("should create a new lighting handler with provided options", func() {
		h := light.NewLightingHandler(
			light.WithAmbientColor([3]float32{0.1, 0.2, 0.3}),
			light.WithTileSize(32),
			light.WithMaxLightsPerTile(128),
			light.WithMaxGPULights(512),
			light.WithShadowHandler(light.NewShadowHandler()),
			light.WithContactShadowHandler(light.NewContactShadowHandler()),
		)
		suite.NotNil(h)
	})
}

func (suite *lightingHandlerTest) TestWithAmbientColor() {
	suite.Run("should apply ambient color option", func() {
		h := light.NewLightingHandler(light.WithAmbientColor([3]float32{0.1, 0.2, 0.3}))
		suite.Equal([3]float32{0.1, 0.2, 0.3}, h.AmbientColor())
	})
}

func (suite *lightingHandlerTest) TestWithTileSize() {
	suite.Run("should apply tile size option", func() {
		h := light.NewLightingHandler(light.WithTileSize(32))
		suite.Equal(32, h.TileSize())
	})
}

func (suite *lightingHandlerTest) TestWithMaxLightsPerTile() {
	suite.Run("should apply max lights per tile option", func() {
		h := light.NewLightingHandler(light.WithMaxLightsPerTile(128))
		suite.Equal(128, h.MaxLightsPerTile())
	})
}

func (suite *lightingHandlerTest) TestWithMaxGPULights() {
	suite.Run("should apply max GPU lights option", func() {
		h := light.NewLightingHandler(light.WithMaxGPULights(512))
		suite.Equal(512, h.MaxGPULights())
	})
}

func (suite *lightingHandlerTest) TestWithShadowHandler() {
	suite.Run("should apply custom shadow handler", func() {
		h := light.NewLightingHandler(light.WithShadowHandler(light.NewShadowHandler()))
		suite.NotNil(h.ShadowHandler())
	})
}

func (suite *lightingHandlerTest) TestWithContactShadowHandler() {
	suite.Run("should apply custom contact shadow handler", func() {
		h := light.NewLightingHandler(light.WithContactShadowHandler(light.NewContactShadowHandler()))
		suite.NotNil(h.ContactShadowHandler())
	})
}

func (suite *lightingHandlerTest) TestEnabled() {
	suite.Run("should return false by default", func() {
		suite.Equal(false, suite.handler.Enabled())
	})
}

func (suite *lightingHandlerTest) TestSetEnabled() {
	suite.Run("should update the enabled state", func() {
		suite.handler.SetEnabled(true)
		suite.Equal(true, suite.handler.Enabled())
	})
}

func (suite *lightingHandlerTest) TestAmbientColor() {
	suite.Run("should return zero color by default", func() {
		suite.Equal([3]float32{}, suite.handler.AmbientColor())
	})
}

func (suite *lightingHandlerTest) TestSetAmbientColor() {
	suite.Run("should update the ambient color", func() {
		suite.handler.SetAmbientColor([3]float32{1, 0, 0})
		suite.Equal([3]float32{1, 0, 0}, suite.handler.AmbientColor())
	})
}

func (suite *lightingHandlerTest) TestLights() {
	suite.Run("should return an empty slice by default", func() {
		suite.Equal(0, len(suite.handler.Lights()))
	})
}

func (suite *lightingHandlerTest) TestAddLight() {
	suite.Run("should append a light to the light list", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.AddLight(l)
		suite.Equal(1, len(suite.handler.Lights()))
	})
}

func (suite *lightingHandlerTest) TestRemoveLight() {
	suite.Run("should remove a light from the list by reference", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.AddLight(l)
		suite.handler.RemoveLight(l)
		suite.Equal(0, len(suite.handler.Lights()))
	})

	suite.Run("should do nothing when the light is not in the list", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.RemoveLight(l)
		suite.Equal(0, len(suite.handler.Lights()))
	})
}

func (suite *lightingHandlerTest) TestBgp() {
	suite.Run("should return a non-nil provider for a known key", func() {
		suite.NotNil(suite.handler.Bgp("lights"))
	})

	suite.Run("should return nil for an unknown key", func() {
		suite.Nil(suite.handler.Bgp("unknown"))
	})
}

func (suite *lightingHandlerTest) TestBgps() {
	suite.Run("should return the full bgp map", func() {
		bgps := suite.handler.Bgps()
		suite.Equal(8, len(bgps))
		suite.Contains(bgps, "lights")
		suite.Contains(bgps, "light_cull")
		suite.Contains(bgps, "tile_lit")
		suite.Contains(bgps, "ssao_lit")
		suite.Contains(bgps, "probe_lit")
		suite.Contains(bgps, "composition_lit")
		suite.Contains(bgps, "ssr_lit")
		suite.Contains(bgps, "taa_lit")
	})
}

func (suite *lightingHandlerTest) TestPipelineKey() {
	suite.Run("should return empty string for unknown key", func() {
		suite.Equal("", suite.handler.PipelineKey("x"))
	})

	suite.Run("should return the value after SetPipelineKey", func() {
		suite.handler.SetPipelineKey("k", "v")
		suite.Equal("v", suite.handler.PipelineKey("k"))
	})
}

func (suite *lightingHandlerTest) TestPipelineKeys() {
	suite.Run("should return the full pipeline keys map", func() {
		suite.handler.SetPipelineKey("mykey", "myval")
		keys := suite.handler.PipelineKeys()
		suite.Equal("myval", keys["mykey"])
	})
}

func (suite *lightingHandlerTest) TestSetPipelineKey() {
	suite.Run("should store a pipeline key", func() {
		suite.handler.SetPipelineKey("p", "pkey")
		suite.Equal("pkey", suite.handler.PipelineKey("p"))
	})
}

func (suite *lightingHandlerTest) TestScreenWidth() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.ScreenWidth())
	})
}

func (suite *lightingHandlerTest) TestScreenHeight() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.ScreenHeight())
	})
}

func (suite *lightingHandlerTest) TestTileCountX() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.TileCountX())
	})
}

func (suite *lightingHandlerTest) TestTileCountY() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.TileCountY())
	})
}

func (suite *lightingHandlerTest) TestTileSize() {
	suite.Run("should return 16 by default", func() {
		suite.Equal(16, suite.handler.TileSize())
	})
}

func (suite *lightingHandlerTest) TestMaxLightsPerTile() {
	suite.Run("should return 256 by default", func() {
		suite.Equal(256, suite.handler.MaxLightsPerTile())
	})
}

func (suite *lightingHandlerTest) TestResize() {
	suite.Run("should update screen dimensions and recompute tile counts", func() {
		suite.handler.Resize(1920, 1080)
		suite.Equal(1920, suite.handler.ScreenWidth())
		suite.Equal(1080, suite.handler.ScreenHeight())
		suite.Equal((1920+15)/16, suite.handler.TileCountX())
		suite.Equal((1080+15)/16, suite.handler.TileCountY())
	})
}

func (suite *lightingHandlerTest) TestShadowHandler() {
	suite.Run("should return a non-nil handler by default", func() {
		suite.NotNil(suite.handler.ShadowHandler())
	})
}

func (suite *lightingHandlerTest) TestContactShadowHandler() {
	suite.Run("should return a non-nil handler by default", func() {
		suite.NotNil(suite.handler.ContactShadowHandler())
	})
}

func (suite *lightingHandlerTest) TestMaxGPULights() {
	suite.Run("should return 1024 by default", func() {
		suite.Equal(1024, suite.handler.MaxGPULights())
	})
}

func (suite *lightingHandlerTest) TestSetMaxGPULights() {
	suite.Run("should update the max GPU lights cap", func() {
		suite.handler.SetMaxGPULights(512)
		suite.Equal(512, suite.handler.MaxGPULights())
	})
}

func (suite *lightingHandlerTest) TestMarshalLightBuffer() {
	const headerSize = 16
	const lightSize = 64

	suite.Run("should produce a header-only buffer when no lights are provided", func() {
		buf := suite.handler.MarshalLightBuffer([]light.Light{}, nil)
		suite.Equal(headerSize, len(buf))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[12:16]))
	})

	suite.Run("should include only enabled lights in the buffer", func() {
		l1 := light.NewLight(light.LightTypePoint)
		l2 := light.NewLight(light.LightTypePoint, light.WithEnabled(false))
		buf := suite.handler.MarshalLightBuffer([]light.Light{l1, l2}, nil)
		suite.Equal(headerSize+lightSize, len(buf))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[12:16]))
	})

	suite.Run("should respect the maxGPULights cap", func() {
		suite.handler.SetMaxGPULights(1)
		l1 := light.NewLight(light.LightTypePoint)
		l2 := light.NewLight(light.LightTypePoint)
		l3 := light.NewLight(light.LightTypePoint)
		buf := suite.handler.MarshalLightBuffer([]light.Light{l1, l2, l3}, nil)
		suite.Equal(headerSize+lightSize, len(buf))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[12:16]))
	})

	suite.Run("should write the ambient color into the header", func() {
		suite.handler.SetAmbientColor([3]float32{1.0, 0.5, 0.0})
		buf := suite.handler.MarshalLightBuffer([]light.Light{}, nil)
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(0.5), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(math.Float32bits(0.0), binary.LittleEndian.Uint32(buf[8:12]))
	})

	suite.Run("should override ShadowIndex from the shadowIndices map when provided", func() {
		l := light.NewLight(light.LightTypePoint)
		buf := suite.handler.MarshalLightBuffer([]light.Light{l}, map[light.Light]uint32{l: 3})
		suite.Equal(uint32(3), binary.LittleEndian.Uint32(buf[headerSize+60:headerSize+64]))
	})

	suite.Run("should leave ShadowIndex as 0xFFFFFFFF when shadowIndices is nil", func() {
		l := light.NewLight(light.LightTypePoint)
		buf := suite.handler.MarshalLightBuffer([]light.Light{l}, nil)
		suite.Equal(uint32(0xFFFFFFFF), binary.LittleEndian.Uint32(buf[headerSize+60:headerSize+64]))
	})
}
