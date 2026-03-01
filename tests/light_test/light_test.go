package light_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/stretchr/testify/suite"
)

// ─────────────────────────────────────────────────────────────────────────────
// Light interface suite
// ─────────────────────────────────────────────────────────────────────────────

type lightTest struct {
	suite.Suite
}

func TestLight(t *testing.T) {
	suite.Run(t, new(lightTest))
}

func (suite *lightTest) TestNewLight() {
	suite.Run("type is preserved", func() {
		suite.Equal(light.LightTypeDirectional, light.NewLight(light.LightTypeDirectional).Type())
		suite.Equal(light.LightTypePoint, light.NewLight(light.LightTypePoint).Type())
		suite.Equal(light.LightTypeSpot, light.NewLight(light.LightTypeSpot).Type())
	})

	suite.Run("default position is zero", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.Equal([3]float32{0, 0, 0}, l.Position())
	})

	suite.Run("default direction is negative y", func() {
		l := light.NewLight(light.LightTypeDirectional)
		suite.Equal([3]float32{0, -1, 0}, l.Direction())
	})

	suite.Run("default color is white", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.Equal([3]float32{1, 1, 1}, l.Color())
	})

	suite.Run("default intensity is one", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.Equal(float32(1.0), l.Intensity())
	})

	suite.Run("default range is ten", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.Equal(float32(10.0), l.Range())
	})

	suite.Run("default inner cone is cos 25 degrees", func() {
		l := light.NewLight(light.LightTypeSpot)
		expected := float32(math.Cos(25.0 * math.Pi / 180.0))
		suite.InDelta(float64(expected), float64(l.InnerCone()), 1e-3)
	})

	suite.Run("default outer cone is cos 35 degrees", func() {
		l := light.NewLight(light.LightTypeSpot)
		expected := float32(math.Cos(35.0 * math.Pi / 180.0))
		suite.InDelta(float64(expected), float64(l.OuterCone()), 1e-3)
	})

	suite.Run("default enabled is true", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.True(l.Enabled())
	})

	suite.Run("default ephemeral is false", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.False(l.Ephemeral())
	})

	suite.Run("default casts shadows is false", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.False(l.CastsShadows())
	})
}

func (suite *lightTest) TestNewLightWithOptions() {
	suite.Run("WithPosition sets position", func() {
		l := light.NewLight(light.LightTypePoint, light.WithPosition(1, 2, 3))
		suite.Equal([3]float32{1, 2, 3}, l.Position())
	})

	suite.Run("WithDirection normalizes and sets direction", func() {
		l := light.NewLight(light.LightTypeDirectional, light.WithDirection(3, 0, 0))
		d := l.Direction()
		suite.InDelta(1.0, float64(d[0]), 1e-6)
		suite.InDelta(0.0, float64(d[1]), 1e-6)
		suite.InDelta(0.0, float64(d[2]), 1e-6)
	})

	suite.Run("WithDirection zero vector yields zero", func() {
		l := light.NewLight(light.LightTypeDirectional, light.WithDirection(0, 0, 0))
		suite.Equal([3]float32{0, 0, 0}, l.Direction())
	})

	suite.Run("WithColor sets color", func() {
		l := light.NewLight(light.LightTypePoint, light.WithColor(0.5, 0.6, 0.7))
		suite.Equal([3]float32{0.5, 0.6, 0.7}, l.Color())
	})

	suite.Run("WithIntensity sets intensity", func() {
		l := light.NewLight(light.LightTypePoint, light.WithIntensity(5.0))
		suite.Equal(float32(5.0), l.Intensity())
	})

	suite.Run("WithRange sets range", func() {
		l := light.NewLight(light.LightTypePoint, light.WithRange(25.0))
		suite.Equal(float32(25.0), l.Range())
	})

	suite.Run("WithSpotCone sets cone cosines", func() {
		l := light.NewLight(light.LightTypeSpot, light.WithSpotCone(20, 40))
		expectedInner := float32(math.Cos(20.0 * math.Pi / 180.0))
		expectedOuter := float32(math.Cos(40.0 * math.Pi / 180.0))
		suite.InDelta(float64(expectedInner), float64(l.InnerCone()), 1e-6)
		suite.InDelta(float64(expectedOuter), float64(l.OuterCone()), 1e-6)
	})

	suite.Run("WithEnabled false disables light", func() {
		l := light.NewLight(light.LightTypePoint, light.WithEnabled(false))
		suite.False(l.Enabled())
	})

	suite.Run("WithEphemeral true marks ephemeral", func() {
		l := light.NewLight(light.LightTypePoint, light.WithEphemeral(true))
		suite.True(l.Ephemeral())
	})

	suite.Run("WithCastsShadows true enables shadow casting", func() {
		l := light.NewLight(light.LightTypePoint, light.WithCastsShadows(true))
		suite.True(l.CastsShadows())
	})

	suite.Run("combined options apply correctly", func() {
		l := light.NewLight(light.LightTypeSpot,
			light.WithPosition(10, 20, 30),
			light.WithDirection(0, 0, -1),
			light.WithColor(1, 0, 0),
			light.WithIntensity(3.0),
			light.WithRange(50.0),
			light.WithSpotCone(15, 30),
			light.WithEnabled(true),
			light.WithEphemeral(true),
			light.WithCastsShadows(true),
		)
		suite.Equal(light.LightTypeSpot, l.Type())
		suite.Equal([3]float32{10, 20, 30}, l.Position())
		suite.Equal([3]float32{1, 0, 0}, l.Color())
		suite.Equal(float32(3.0), l.Intensity())
		suite.Equal(float32(50.0), l.Range())
		suite.True(l.Enabled())
		suite.True(l.Ephemeral())
		suite.True(l.CastsShadows())

		d := l.Direction()
		suite.InDelta(0.0, float64(d[0]), 1e-6)
		suite.InDelta(0.0, float64(d[1]), 1e-6)
		suite.InDelta(-1.0, float64(d[2]), 1e-6)
	})
}

func (suite *lightTest) TestSetters() {
	suite.Run("SetPosition updates position", func() {
		l := light.NewLight(light.LightTypePoint)
		l.SetPosition(4, 5, 6)
		suite.Equal([3]float32{4, 5, 6}, l.Position())
	})

	suite.Run("SetDirection normalizes and updates direction", func() {
		l := light.NewLight(light.LightTypeDirectional)
		l.SetDirection(0, 0, -5)
		d := l.Direction()
		suite.InDelta(0.0, float64(d[0]), 1e-6)
		suite.InDelta(0.0, float64(d[1]), 1e-6)
		suite.InDelta(-1.0, float64(d[2]), 1e-6)
	})

	suite.Run("SetDirection with zero vector yields zero", func() {
		l := light.NewLight(light.LightTypeDirectional)
		l.SetDirection(0, 0, 0)
		suite.Equal([3]float32{0, 0, 0}, l.Direction())
	})

	suite.Run("SetColor updates color", func() {
		l := light.NewLight(light.LightTypePoint)
		l.SetColor(0.2, 0.3, 0.4)
		suite.Equal([3]float32{0.2, 0.3, 0.4}, l.Color())
	})

	suite.Run("SetIntensity updates intensity", func() {
		l := light.NewLight(light.LightTypePoint)
		l.SetIntensity(7.5)
		suite.Equal(float32(7.5), l.Intensity())
	})

	suite.Run("SetRange updates range", func() {
		l := light.NewLight(light.LightTypePoint)
		l.SetRange(100)
		suite.Equal(float32(100), l.Range())
	})

	suite.Run("SetSpotCone updates cone cosines", func() {
		l := light.NewLight(light.LightTypeSpot)
		l.SetSpotCone(10, 20)
		expectedInner := float32(math.Cos(10.0 * math.Pi / 180.0))
		expectedOuter := float32(math.Cos(20.0 * math.Pi / 180.0))
		suite.InDelta(float64(expectedInner), float64(l.InnerCone()), 1e-6)
		suite.InDelta(float64(expectedOuter), float64(l.OuterCone()), 1e-6)
	})

	suite.Run("SetEnabled toggles enabled", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.True(l.Enabled())
		l.SetEnabled(false)
		suite.False(l.Enabled())
		l.SetEnabled(true)
		suite.True(l.Enabled())
	})

	suite.Run("SetEphemeral toggles ephemeral", func() {
		l := light.NewLight(light.LightTypePoint)
		l.SetEphemeral(true)
		suite.True(l.Ephemeral())
		l.SetEphemeral(false)
		suite.False(l.Ephemeral())
	})

	suite.Run("SetCastsShadows toggles shadow casting", func() {
		l := light.NewLight(light.LightTypePoint)
		l.SetCastsShadows(true)
		suite.True(l.CastsShadows())
		l.SetCastsShadows(false)
		suite.False(l.CastsShadows())
	})
}

func (suite *lightTest) TestLightTypeConstants() {
	suite.Run("directional is 0", func() {
		suite.Equal(light.LightType(0), light.LightTypeDirectional)
	})

	suite.Run("point is 1", func() {
		suite.Equal(light.LightType(1), light.LightTypePoint)
	})

	suite.Run("spot is 2", func() {
		suite.Equal(light.LightType(2), light.LightTypeSpot)
	})
}

func (suite *lightTest) TestTileCounts() {
	suite.Run("exact multiple of tile size", func() {
		tx, ty := light.TileCounts(1920, 1080)
		// 1920/16 = 120, 1080/16 = 67.5 → ceil = 68
		suite.Equal(uint32(120), tx)
		suite.Equal(uint32(68), ty)
	})

	suite.Run("non-multiple rounds up", func() {
		tx, ty := light.TileCounts(1921, 1081)
		// (1921+15)/16 = 121, (1081+15)/16 = 68 (ceil division)
		suite.Equal(uint32(121), tx)
		suite.Equal(uint32(68), ty)
	})

	suite.Run("single pixel screen", func() {
		tx, ty := light.TileCounts(1, 1)
		suite.Equal(uint32(1), tx)
		suite.Equal(uint32(1), ty)
	})

	suite.Run("exactly one tile", func() {
		tx, ty := light.TileCounts(16, 16)
		suite.Equal(uint32(1), tx)
		suite.Equal(uint32(1), ty)
	})
}

func (suite *lightTest) TestShadowConstants() {
	suite.Run("shadow map resolution is 2048", func() {
		suite.Equal(2048, light.ShadowMapResolution)
	})

	suite.Run("default shadow half extent is 40", func() {
		suite.Equal(float32(40.0), light.DefaultShadowHalfExtent)
	})

	suite.Run("default shadow near is 0.1", func() {
		suite.Equal(float32(0.1), light.DefaultShadowNear)
	})

	suite.Run("default shadow far is 200", func() {
		suite.Equal(float32(200.0), light.DefaultShadowFar)
	})

	suite.Run("default shadow bias is 0.001", func() {
		suite.Equal(float32(0.001), light.DefaultShadowBias)
	})

	suite.Run("default shadow normal bias scale is 3.0", func() {
		suite.Equal(float32(3.0), light.DefaultShadowNormalBiasScale)
	})
}

func (suite *lightTest) TestLightCullConstants() {
	suite.Run("tile size is 16", func() {
		suite.Equal(16, light.TileSize)
	})

	suite.Run("max lights per tile is 256", func() {
		suite.Equal(256, light.MaxLightsPerTile)
	})

	suite.Run("max gpu lights is 1024", func() {
		suite.Equal(1024, light.MaxGPULights)
	})
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU types suite
// ─────────────────────────────────────────────────────────────────────────────

type gpuTypesTest struct {
	suite.Suite
}

func TestGPUTypes(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

func (suite *gpuTypesTest) TestGPULightSize() {
	suite.Run("size is 64 bytes", func() {
		g := &light.GPULight{}
		suite.Equal(64, g.Size())
	})
}

func (suite *gpuTypesTest) TestGPULightMarshal() {
	suite.Run("byte length is 64", func() {
		g := &light.GPULight{}
		suite.Len(g.Marshal(), 64)
	})

	suite.Run("position is encoded at offset 0", func() {
		g := &light.GPULight{Position: [3]float32{1.0, 2.0, 3.0}}
		buf := g.Marshal()
		suite.Equal(float32(1.0), readF32(buf, 0))
		suite.Equal(float32(2.0), readF32(buf, 4))
		suite.Equal(float32(3.0), readF32(buf, 8))
	})

	suite.Run("light type is encoded at offset 12", func() {
		g := &light.GPULight{LightType: 2}
		buf := g.Marshal()
		suite.Equal(uint32(2), binary.LittleEndian.Uint32(buf[12:16]))
	})

	suite.Run("color and intensity are encoded at offset 16", func() {
		g := &light.GPULight{
			Color:     [3]float32{0.5, 0.6, 0.7},
			Intensity: 3.0,
		}
		buf := g.Marshal()
		suite.Equal(float32(0.5), readF32(buf, 16))
		suite.Equal(float32(0.6), readF32(buf, 20))
		suite.Equal(float32(0.7), readF32(buf, 24))
		suite.Equal(float32(3.0), readF32(buf, 28))
	})

	suite.Run("direction and range are encoded at offset 32", func() {
		g := &light.GPULight{
			Direction:  [3]float32{0, -1, 0},
			LightRange: 25.0,
		}
		buf := g.Marshal()
		suite.Equal(float32(0), readF32(buf, 32))
		suite.Equal(float32(-1), readF32(buf, 36))
		suite.Equal(float32(0), readF32(buf, 40))
		suite.Equal(float32(25.0), readF32(buf, 44))
	})

	suite.Run("cone angles and shadow flag at offset 48", func() {
		g := &light.GPULight{
			InnerCone:    0.9,
			OuterCone:    0.8,
			CastsShadows: 1,
		}
		buf := g.Marshal()
		suite.InDelta(0.9, float64(readF32(buf, 48)), 1e-6)
		suite.InDelta(0.8, float64(readF32(buf, 52)), 1e-6)
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[56:60]))
	})

	suite.Run("padding at offset 60 is zero", func() {
		g := &light.GPULight{
			Position:     [3]float32{1, 2, 3},
			CastsShadows: 1,
		}
		buf := g.Marshal()
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[60:64]))
	})
}

func (suite *gpuTypesTest) TestGPULightHeaderSize() {
	suite.Run("size is 16 bytes", func() {
		h := &light.GPULightHeader{}
		suite.Equal(16, h.Size())
	})
}

func (suite *gpuTypesTest) TestGPULightHeaderMarshal() {
	suite.Run("byte length is 16", func() {
		h := &light.GPULightHeader{}
		suite.Len(h.Marshal(), 16)
	})

	suite.Run("ambient color is encoded at offset 0", func() {
		h := &light.GPULightHeader{AmbientColor: [3]float32{0.1, 0.2, 0.3}}
		buf := h.Marshal()
		suite.InDelta(0.1, float64(readF32(buf, 0)), 1e-6)
		suite.InDelta(0.2, float64(readF32(buf, 4)), 1e-6)
		suite.InDelta(0.3, float64(readF32(buf, 8)), 1e-6)
	})

	suite.Run("light count is encoded at offset 12", func() {
		h := &light.GPULightHeader{LightCount: 42}
		buf := h.Marshal()
		suite.Equal(uint32(42), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

func (suite *gpuTypesTest) TestGPUShadowDataSize() {
	suite.Run("size is 176 bytes", func() {
		s := &light.GPUShadowData{}
		suite.Equal(176, s.Size())
	})
}

func (suite *gpuTypesTest) TestGPUShadowDataMarshal() {
	suite.Run("byte length is 176", func() {
		s := &light.GPUShadowData{}
		suite.Len(s.Marshal(), 176)
	})

	suite.Run("light VP matrix is encoded in first 64 bytes", func() {
		s := &light.GPUShadowData{}
		// set an identity-like pattern
		s.LightVP[0] = 1.0
		s.LightVP[5] = 1.0
		s.LightVP[10] = 1.0
		s.LightVP[15] = 1.0
		buf := s.Marshal()
		suite.Equal(float32(1.0), readF32(buf, 0))
		suite.Equal(float32(1.0), readF32(buf, 20))
		suite.Equal(float32(1.0), readF32(buf, 40))
		suite.Equal(float32(1.0), readF32(buf, 60))
	})

	suite.Run("texel size is encoded at offset 128", func() {
		s := &light.GPUShadowData{TexelSize: [2]float32{0.001, 0.002}}
		buf := s.Marshal()
		suite.InDelta(0.001, float64(readF32(buf, 128)), 1e-6)
		suite.InDelta(0.002, float64(readF32(buf, 132)), 1e-6)
	})

	suite.Run("bias and normal bias at offset 136", func() {
		s := &light.GPUShadowData{Bias: 0.005, NormalBias: 0.01}
		buf := s.Marshal()
		suite.InDelta(0.005, float64(readF32(buf, 136)), 1e-6)
		suite.InDelta(0.01, float64(readF32(buf, 140)), 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUShadowDataComputeNormalBias() {
	suite.Run("computes correct normal bias", func() {
		s := &light.GPUShadowData{}
		// texelWorldSize = 2.0 * 40 / 2048 = 0.0390625
		// normalBias = 0.0390625 * 3.0 = 0.1171875
		s.ComputeNormalBias(40.0, 3.0, 2048)
		suite.InDelta(0.1171875, float64(s.NormalBias), 1e-6)
	})

	suite.Run("scales linearly with half extent", func() {
		s1 := &light.GPUShadowData{}
		s2 := &light.GPUShadowData{}
		s1.ComputeNormalBias(20.0, 3.0, 2048)
		s2.ComputeNormalBias(40.0, 3.0, 2048)
		suite.InDelta(float64(s1.NormalBias)*2, float64(s2.NormalBias), 1e-6)
	})

	suite.Run("scales linearly with scale factor", func() {
		s1 := &light.GPUShadowData{}
		s2 := &light.GPUShadowData{}
		s1.ComputeNormalBias(40.0, 2.0, 2048)
		s2.ComputeNormalBias(40.0, 4.0, 2048)
		suite.InDelta(float64(s1.NormalBias)*2, float64(s2.NormalBias), 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUShadowDataComputeDirectionalLightVP() {
	suite.Run("produces a non-zero VP matrix for downward light", func() {
		s := &light.GPUShadowData{}
		s.ComputeDirectionalLightVP(
			[3]float32{0, -1, 0}, // pointing down
			0, 0, 0,              // center at origin
			40, 0.1, 200,
		)
		nonZero := false
		for _, v := range s.LightVP {
			if v != 0 {
				nonZero = true
				break
			}
		}
		suite.True(nonZero)
	})

	suite.Run("uses x-axis up when light points nearly vertical", func() {
		s := &light.GPUShadowData{}
		// Light direction nearly straight down — triggers the up-vector override
		s.ComputeDirectionalLightVP(
			[3]float32{0, -0.999, 0.01},
			0, 10, 0,
			40, 0.1, 200,
		)
		nonZero := false
		for _, v := range s.LightVP {
			if v != 0 {
				nonZero = true
				break
			}
		}
		suite.True(nonZero)
	})

	suite.Run("centered on provided position", func() {
		s1 := &light.GPUShadowData{}
		s2 := &light.GPUShadowData{}
		dir := [3]float32{0, -1, 0}
		s1.ComputeDirectionalLightVP(dir, 0, 0, 0, 40, 0.1, 200)
		s2.ComputeDirectionalLightVP(dir, 100, 0, 0, 40, 0.1, 200)
		// Different centers must produce different VP matrices
		differ := false
		for i := range s1.LightVP {
			if s1.LightVP[i] != s2.LightVP[i] {
				differ = true
				break
			}
		}
		suite.True(differ)
	})
}

func (suite *gpuTypesTest) TestGPUShadowUniformSize() {
	suite.Run("size is 136 bytes", func() {
		u := &light.GPUShadowUniform{}
		suite.Equal(136, u.Size())
	})
}

func (suite *gpuTypesTest) TestGPUShadowUniformMarshal() {
	suite.Run("byte length is 136", func() {
		u := &light.GPUShadowUniform{}
		suite.Len(u.Marshal(), 136)
	})

	suite.Run("matrix values are encoded correctly", func() {
		u := &light.GPUShadowUniform{}
		u.LightVP[0] = 1.0
		u.LightVP[5] = 2.0
		u.LightVP[10] = 3.0
		u.LightVP[15] = 4.0
		buf := u.Marshal()
		suite.Equal(float32(1.0), readF32(buf, 0))
		suite.Equal(float32(2.0), readF32(buf, 20))
		suite.Equal(float32(3.0), readF32(buf, 40))
		suite.Equal(float32(4.0), readF32(buf, 60))
	})
}

func (suite *gpuTypesTest) TestGPULightCullUniformsSize() {
	suite.Run("size is 160 bytes", func() {
		u := &light.GPULightCullUniforms{}
		suite.Equal(160, u.Size())
	})
}

func (suite *gpuTypesTest) TestGPULightCullUniformsMarshal() {
	suite.Run("byte length is 160", func() {
		u := &light.GPULightCullUniforms{}
		suite.Len(u.Marshal(), 160)
	})

	suite.Run("inv proj matrix starts at offset 0", func() {
		u := &light.GPULightCullUniforms{}
		u.InvProj[0] = 5.0
		buf := u.Marshal()
		suite.Equal(float32(5.0), readF32(buf, 0))
	})

	suite.Run("view matrix starts at offset 64", func() {
		u := &light.GPULightCullUniforms{}
		u.ViewMatrix[0] = 7.0
		buf := u.Marshal()
		suite.Equal(float32(7.0), readF32(buf, 64))
	})

	suite.Run("tile counts at offset 128", func() {
		u := &light.GPULightCullUniforms{TileCountX: 120, TileCountY: 68}
		buf := u.Marshal()
		suite.Equal(uint32(120), binary.LittleEndian.Uint32(buf[128:132]))
		suite.Equal(uint32(68), binary.LittleEndian.Uint32(buf[132:136]))
	})

	suite.Run("screen dimensions at offset 136", func() {
		u := &light.GPULightCullUniforms{ScreenWidth: 1920, ScreenHeight: 1080}
		buf := u.Marshal()
		suite.Equal(uint32(1920), binary.LittleEndian.Uint32(buf[136:140]))
		suite.Equal(uint32(1080), binary.LittleEndian.Uint32(buf[140:144]))
	})

	suite.Run("light count at offset 144", func() {
		u := &light.GPULightCullUniforms{LightCount: 42}
		buf := u.Marshal()
		suite.Equal(uint32(42), binary.LittleEndian.Uint32(buf[144:148]))
	})

	suite.Run("near and far at offset 148", func() {
		u := &light.GPULightCullUniforms{Near: 0.1, Far: 200.0}
		buf := u.Marshal()
		suite.InDelta(0.1, float64(readF32(buf, 148)), 1e-6)
		suite.Equal(float32(200.0), readF32(buf, 152))
	})

	suite.Run("padding at offset 156 is zero", func() {
		u := &light.GPULightCullUniforms{LightCount: 10, Near: 1, Far: 100}
		buf := u.Marshal()
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[156:160]))
	})
}

func (suite *gpuTypesTest) TestGPUSATParamsSize() {
	suite.Run("size is 16 bytes", func() {
		p := &light.GPUSATParams{}
		suite.Equal(16, p.Size())
	})
}

func (suite *gpuTypesTest) TestGPUSATParamsMarshal() {
	suite.Run("byte length is 16", func() {
		p := &light.GPUSATParams{}
		suite.Len(p.Marshal(), 16)
	})

	suite.Run("zero value produces all-zero bytes", func() {
		p := &light.GPUSATParams{}
		buf := p.Marshal()
		for i, b := range buf {
			suite.Equal(byte(0), b, "byte %d should be zero", i)
		}
	})

	suite.Run("horizontal direction encoded at offset 0", func() {
		p := &light.GPUSATParams{Direction: [2]int32{1, 0}, Offset: 0}
		buf := p.Marshal()
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[4:8]))
	})

	suite.Run("vertical direction encoded at offset 0", func() {
		p := &light.GPUSATParams{Direction: [2]int32{0, 1}, Offset: 0}
		buf := p.Marshal()
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[4:8]))
	})

	suite.Run("offset encoded at offset 8", func() {
		p := &light.GPUSATParams{Offset: 16}
		buf := p.Marshal()
		suite.Equal(uint32(16), binary.LittleEndian.Uint32(buf[8:12]))
	})

	suite.Run("padding at offset 12 is zero", func() {
		p := &light.GPUSATParams{Direction: [2]int32{1, 1}, Offset: 128}
		buf := p.Marshal()
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[12:16]))
	})

	suite.Run("negative direction values encode correctly", func() {
		p := &light.GPUSATParams{Direction: [2]int32{-1, 0}, Offset: 0}
		buf := p.Marshal()
		// -1 as uint32 in two's complement is 0xFFFFFFFF
		suite.Equal(uint32(0xFFFFFFFF), binary.LittleEndian.Uint32(buf[0:4]))
	})

	suite.Run("power-of-two offset for recursive doubling", func() {
		p := &light.GPUSATParams{Direction: [2]int32{1, 0}, Offset: 1024}
		buf := p.Marshal()
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(uint32(1024), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

func (suite *gpuTypesTest) TestGPUTileUniformsSize() {
	suite.Run("size is 8 bytes", func() {
		u := &light.GPUTileUniforms{}
		suite.Equal(8, u.Size())
	})
}

func (suite *gpuTypesTest) TestGPUTileUniformsMarshal() {
	suite.Run("byte length is 8", func() {
		u := &light.GPUTileUniforms{}
		suite.Len(u.Marshal(), 8)
	})

	suite.Run("tile count x at offset 0", func() {
		u := &light.GPUTileUniforms{TileCountX: 120}
		buf := u.Marshal()
		suite.Equal(uint32(120), binary.LittleEndian.Uint32(buf[0:4]))
	})

	suite.Run("max lights per tile at offset 4", func() {
		u := &light.GPUTileUniforms{MaxLightsPerTile: 256}
		buf := u.Marshal()
		suite.Equal(uint32(256), binary.LittleEndian.Uint32(buf[4:8]))
	})
}

func (suite *gpuTypesTest) TestToGPULight() {
	suite.Run("converts directional light correctly", func() {
		l := light.NewLight(light.LightTypeDirectional,
			light.WithDirection(0, -1, 0),
			light.WithColor(1, 1, 1),
			light.WithIntensity(2.0),
		)
		g := light.ToGPULight(l)
		suite.Equal(uint32(0), g.LightType)
		suite.Equal([3]float32{1, 1, 1}, g.Color)
		suite.Equal(float32(2.0), g.Intensity)
		suite.Equal(uint32(0), g.CastsShadows)
	})

	suite.Run("converts point light correctly", func() {
		l := light.NewLight(light.LightTypePoint,
			light.WithPosition(1, 2, 3),
			light.WithColor(1, 0, 0),
			light.WithIntensity(5.0),
			light.WithRange(20.0),
		)
		g := light.ToGPULight(l)
		suite.Equal(uint32(1), g.LightType)
		suite.Equal([3]float32{1, 2, 3}, g.Position)
		suite.Equal([3]float32{1, 0, 0}, g.Color)
		suite.Equal(float32(5.0), g.Intensity)
		suite.Equal(float32(20.0), g.LightRange)
	})

	suite.Run("converts spot light with shadows", func() {
		l := light.NewLight(light.LightTypeSpot,
			light.WithPosition(0, 5, 0),
			light.WithDirection(0, -1, 0),
			light.WithCastsShadows(true),
			light.WithSpotCone(20, 40),
		)
		g := light.ToGPULight(l)
		suite.Equal(uint32(2), g.LightType)
		suite.Equal(uint32(1), g.CastsShadows)
		suite.InDelta(float64(l.InnerCone()), float64(g.InnerCone), 1e-6)
		suite.InDelta(float64(l.OuterCone()), float64(g.OuterCone), 1e-6)
	})

	suite.Run("casts shadows false yields zero", func() {
		l := light.NewLight(light.LightTypePoint, light.WithCastsShadows(false))
		g := light.ToGPULight(l)
		suite.Equal(uint32(0), g.CastsShadows)
	})
}

func (suite *gpuTypesTest) TestMarshalLightBuffer() {
	suite.Run("empty light list produces header only", func() {
		buf := light.MarshalLightBuffer(nil, [3]float32{0.1, 0.1, 0.1})
		suite.Len(buf, 16)
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[12:16]))
	})

	suite.Run("disabled lights are excluded", func() {
		lights := []light.Light{
			light.NewLight(light.LightTypePoint, light.WithEnabled(false)),
			light.NewLight(light.LightTypePoint, light.WithEnabled(true)),
			light.NewLight(light.LightTypePoint, light.WithEnabled(false)),
		}
		buf := light.MarshalLightBuffer(lights, [3]float32{0, 0, 0})
		// header (16) + 1 enabled light (64) = 80
		suite.Len(buf, 80)
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[12:16]))
	})

	suite.Run("all enabled lights are included", func() {
		lights := []light.Light{
			light.NewLight(light.LightTypePoint, light.WithPosition(1, 0, 0)),
			light.NewLight(light.LightTypePoint, light.WithPosition(2, 0, 0)),
			light.NewLight(light.LightTypeDirectional),
		}
		buf := light.MarshalLightBuffer(lights, [3]float32{0.5, 0.5, 0.5})
		// header (16) + 3 lights (64 each) = 208
		suite.Len(buf, 208)
		suite.Equal(uint32(3), binary.LittleEndian.Uint32(buf[12:16]))
	})

	suite.Run("ambient color is written in header", func() {
		buf := light.MarshalLightBuffer(nil, [3]float32{0.2, 0.3, 0.4})
		suite.InDelta(0.2, float64(readF32(buf, 0)), 1e-6)
		suite.InDelta(0.3, float64(readF32(buf, 4)), 1e-6)
		suite.InDelta(0.4, float64(readF32(buf, 8)), 1e-6)
	})

	suite.Run("light data follows header at correct offsets", func() {
		l := light.NewLight(light.LightTypePoint,
			light.WithPosition(9, 8, 7),
			light.WithColor(1, 0, 0),
			light.WithIntensity(3.0),
		)
		buf := light.MarshalLightBuffer([]light.Light{l}, [3]float32{0, 0, 0})
		// First light starts at offset 16
		suite.Equal(float32(9), readF32(buf, 16))
		suite.Equal(float32(8), readF32(buf, 20))
		suite.Equal(float32(7), readF32(buf, 24))
		suite.Equal(float32(3.0), readF32(buf, 44)) // intensity at light offset 28 -> buf offset 16+28=44
	})
}

func (suite *gpuTypesTest) TestWGSLSources() {
	suite.Run("GPULightSource is non-empty", func() {
		suite.NotEmpty(light.GPULightSource)
	})

	suite.Run("GPULightSource contains Light struct", func() {
		suite.Contains(light.GPULightSource, "Light")
	})

	suite.Run("GPULightHeaderSource is non-empty", func() {
		suite.NotEmpty(light.GPULightHeaderSource)
	})

	suite.Run("GPUShadowDataSource is non-empty", func() {
		suite.NotEmpty(light.GPUShadowDataSource)
	})

	suite.Run("GPUShadowUniformSource is non-empty", func() {
		suite.NotEmpty(light.GPUShadowUniformSource)
	})

	suite.Run("GPULightCullUniformsSource is non-empty", func() {
		suite.NotEmpty(light.GPULightCullUniformsSource)
	})

	suite.Run("GPUTileUniformsSource is non-empty", func() {
		suite.NotEmpty(light.GPUTileUniformsSource)
	})

	suite.Run("GPUSATParamsSource is non-empty", func() {
		suite.NotEmpty(light.GPUSATParamsSource)
	})

	suite.Run("GPUSATParamsSource contains SATParams struct", func() {
		suite.Contains(light.GPUSATParamsSource, "SATParams")
	})

	suite.Run("GPUGBufferOutputSource is non-empty", func() {
		suite.NotEmpty(light.GPUGBufferOutputSource)
	})

	suite.Run("GPUSSAOParamsSource is non-empty", func() {
		suite.NotEmpty(light.GPUSSAOParamsSource)
	})

	suite.Run("GPUIrradianceProbeSource is non-empty", func() {
		suite.NotEmpty(light.GPUIrradianceProbeSource)
	})

	suite.Run("GPUProbeGridParamsSource is non-empty", func() {
		suite.NotEmpty(light.GPUProbeGridParamsSource)
	})

	suite.Run("GPUProbeBakeCameraSource is non-empty", func() {
		suite.NotEmpty(light.GPUProbeBakeCameraSource)
	})

	suite.Run("GPUSHProjectParamsSource is non-empty", func() {
		suite.NotEmpty(light.GPUSHProjectParamsSource)
	})

	suite.Run("GPUCompositionParamsSource is non-empty", func() {
		suite.NotEmpty(light.GPUCompositionParamsSource)
	})

	suite.Run("GPUSSRParamsSource is non-empty", func() {
		suite.NotEmpty(light.GPUSSRParamsSource)
	})

	suite.Run("GPUBlurParamsSource is non-empty", func() {
		suite.NotEmpty(light.GPUBlurParamsSource)
	})
}

func (suite *gpuTypesTest) TestGPUBlurParamsSize() {
	suite.Run("size is 16 bytes", func() {
		p := &light.GPUBlurParams{}
		suite.Equal(16, p.Size())
	})
}

func (suite *gpuTypesTest) TestGPUBlurParamsMarshal() {
	suite.Run("round-trips direction and parameters", func() {
		p := &light.GPUBlurParams{
			Direction:    [2]int32{1, 0},
			Radius:       4,
			GBufferScale: 2,
		}
		buf := p.Marshal()
		suite.Len(buf, 16)
		suite.Equal(int32(1), int32(binary.LittleEndian.Uint32(buf[0:4])))
		suite.Equal(int32(0), int32(binary.LittleEndian.Uint32(buf[4:8])))
		suite.Equal(int32(4), int32(binary.LittleEndian.Uint32(buf[8:12])))
		suite.Equal(int32(2), int32(binary.LittleEndian.Uint32(buf[12:16])))
	})

	suite.Run("vertical direction is serialized correctly", func() {
		p := &light.GPUBlurParams{
			Direction:    [2]int32{0, 1},
			Radius:       8,
			GBufferScale: 1,
		}
		buf := p.Marshal()
		suite.Equal(int32(0), int32(binary.LittleEndian.Uint32(buf[0:4])))
		suite.Equal(int32(1), int32(binary.LittleEndian.Uint32(buf[4:8])))
	})
}

func (suite *gpuTypesTest) TestGPUGBufferOutputSize() {
	suite.Run("size is 48 bytes", func() {
		g := &light.GPUGBufferOutput{}
		suite.Equal(48, g.Size())
	})
}

func (suite *gpuTypesTest) TestGPUGBufferOutputMarshal() {
	suite.Run("round-trips position normal and albedo", func() {
		g := &light.GPUGBufferOutput{
			Position: [4]float32{1.0, 2.0, 3.0, 0.5},
			Normal:   [4]float32{0.0, 1.0, 0.0, 0.25},
			Albedo:   [4]float32{0.8, 0.2, 0.1, 0.9},
		}
		buf := g.Marshal()
		suite.Len(buf, 48)
		suite.InDelta(1.0, float64(readF32(buf, 0)), 1e-6)
		suite.InDelta(2.0, float64(readF32(buf, 4)), 1e-6)
		suite.InDelta(3.0, float64(readF32(buf, 8)), 1e-6)
		suite.InDelta(0.5, float64(readF32(buf, 12)), 1e-6)
		suite.InDelta(0.0, float64(readF32(buf, 16)), 1e-6)
		suite.InDelta(1.0, float64(readF32(buf, 20)), 1e-6)
		suite.InDelta(0.0, float64(readF32(buf, 24)), 1e-6)
		suite.InDelta(0.25, float64(readF32(buf, 28)), 1e-6)
		suite.InDelta(0.8, float64(readF32(buf, 32)), 1e-6)
		suite.InDelta(0.2, float64(readF32(buf, 36)), 1e-6)
		suite.InDelta(0.1, float64(readF32(buf, 40)), 1e-6)
		suite.InDelta(0.9, float64(readF32(buf, 44)), 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUSSAOParamsSize() {
	suite.Run("size is 176 bytes", func() {
		p := &light.GPUSSAOParams{}
		suite.Equal(176, p.Size())
	})
}

func (suite *gpuTypesTest) TestGPUSSAOParamsMarshal() {
	suite.Run("round-trips scalar fields", func() {
		p := &light.GPUSSAOParams{
			Radius:         0.5,
			Bias:           0.025,
			Power:          2.0,
			SampleCount:    16,
			ScreenWidth:    1920.0,
			ScreenHeight:   1080.0,
			GBufferScale:   1.0,
			CameraPosition: [3]float32{5, 10, 15},
		}
		buf := p.Marshal()
		suite.Len(buf, 176)
		// Scalar fields start at offset 128
		suite.InDelta(0.5, float64(readF32(buf, 128)), 1e-6)
		suite.InDelta(0.025, float64(readF32(buf, 132)), 1e-6)
		suite.InDelta(2.0, float64(readF32(buf, 136)), 1e-6)
		suite.Equal(uint32(16), binary.LittleEndian.Uint32(buf[140:144]))
		suite.InDelta(1920.0, float64(readF32(buf, 144)), 1e-6)
		suite.InDelta(1080.0, float64(readF32(buf, 148)), 1e-6)
		suite.InDelta(1.0, float64(readF32(buf, 152)), 1e-6)
		// Camera position at offset 160
		suite.InDelta(5.0, float64(readF32(buf, 160)), 1e-6)
		suite.InDelta(10.0, float64(readF32(buf, 164)), 1e-6)
		suite.InDelta(15.0, float64(readF32(buf, 168)), 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUIrradianceProbeSize() {
	suite.Run("size is 160 bytes", func() {
		p := &light.GPUIrradianceProbe{}
		suite.Equal(160, p.Size())
	})
}

func (suite *gpuTypesTest) TestGPUIrradianceProbeMarshal() {
	suite.Run("round-trips position and SH coefficients", func() {
		p := &light.GPUIrradianceProbe{
			Position: [4]float32{1, 2, 3, 1},
		}
		// Set first SH coefficient for each channel
		p.SH_R[0] = 0.5
		p.SH_G[0] = 0.6
		p.SH_B[0] = 0.7
		buf := p.Marshal()
		suite.Len(buf, 160)
		suite.InDelta(1.0, float64(readF32(buf, 0)), 1e-6)
		suite.InDelta(2.0, float64(readF32(buf, 4)), 1e-6)
		suite.InDelta(3.0, float64(readF32(buf, 8)), 1e-6)
		suite.InDelta(1.0, float64(readF32(buf, 12)), 1e-6)
		// SH_R[0] at offset 16
		suite.InDelta(0.5, float64(readF32(buf, 16)), 1e-6)
		// SH_G[0] at offset 64
		suite.InDelta(0.6, float64(readF32(buf, 64)), 1e-6)
		// SH_B[0] at offset 112
		suite.InDelta(0.7, float64(readF32(buf, 112)), 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUProbeGridParamsSize() {
	suite.Run("size is 80 bytes", func() {
		p := &light.GPUProbeGridParams{}
		suite.Equal(80, p.Size())
	})
}

func (suite *gpuTypesTest) TestGPUProbeGridParamsMarshal() {
	suite.Run("round-trips grid parameters", func() {
		p := &light.GPUProbeGridParams{
			GridMin:     [3]float32{-10, -2, -10},
			ProbeCountX: 8,
			GridMax:     [3]float32{10, 6, 10},
			ProbeCountY: 4,
			Spacing:     [3]float32{2.857, 2.667, 2.857},
			ProbeCountZ: 8,
			TotalProbes: 256,
		}
		buf := p.Marshal()
		suite.Len(buf, 80)
		suite.InDelta(-10.0, float64(readF32(buf, 0)), 1e-3)
		suite.InDelta(-2.0, float64(readF32(buf, 4)), 1e-3)
		suite.InDelta(-10.0, float64(readF32(buf, 8)), 1e-3)
		suite.Equal(uint32(8), binary.LittleEndian.Uint32(buf[12:16]))
		suite.InDelta(10.0, float64(readF32(buf, 16)), 1e-3)
		suite.InDelta(6.0, float64(readF32(buf, 20)), 1e-3)
		suite.InDelta(10.0, float64(readF32(buf, 24)), 1e-3)
		suite.Equal(uint32(4), binary.LittleEndian.Uint32(buf[28:32]))
		suite.Equal(uint32(8), binary.LittleEndian.Uint32(buf[44:48]))
		suite.Equal(uint32(256), binary.LittleEndian.Uint32(buf[48:52]))
	})
}

func (suite *gpuTypesTest) TestGPUProbeBakeCameraSize() {
	suite.Run("size is 80 bytes", func() {
		c := &light.GPUProbeBakeCamera{}
		suite.Equal(80, c.Size())
	})
}

func (suite *gpuTypesTest) TestGPUProbeBakeCameraMarshal() {
	suite.Run("round-trips camera data", func() {
		c := &light.GPUProbeBakeCamera{
			CameraPosition: [3]float32{5, 10, 15},
		}
		// Set identity-like view*proj matrix
		c.ViewProj[0] = 1.0
		c.ViewProj[5] = 1.0
		c.ViewProj[10] = 1.0
		c.ViewProj[15] = 1.0
		buf := c.Marshal()
		suite.Len(buf, 80)
		suite.InDelta(1.0, float64(readF32(buf, 0)), 1e-6)
		suite.InDelta(1.0, float64(readF32(buf, 20)), 1e-6)
		suite.InDelta(1.0, float64(readF32(buf, 40)), 1e-6)
		suite.InDelta(1.0, float64(readF32(buf, 60)), 1e-6)
		// Camera position at offset 64
		suite.InDelta(5.0, float64(readF32(buf, 64)), 1e-6)
		suite.InDelta(10.0, float64(readF32(buf, 68)), 1e-6)
		suite.InDelta(15.0, float64(readF32(buf, 72)), 1e-6)
		// Padding at offset 76 should be zero
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[76:80]))
	})
}

func (suite *gpuTypesTest) TestMarshalProbeBuffer() {
	suite.Run("empty slice returns empty buffer", func() {
		buf := light.MarshalProbeBuffer(nil)
		suite.Len(buf, 0)
	})

	suite.Run("single probe marshals correctly", func() {
		p := light.GPUIrradianceProbe{
			Position: [4]float32{1, 2, 3, 1},
		}
		buf := light.MarshalProbeBuffer([]light.GPUIrradianceProbe{p})
		suite.Len(buf, 160)
		suite.InDelta(1.0, float64(readF32(buf, 0)), 1e-6)
		suite.InDelta(2.0, float64(readF32(buf, 4)), 1e-6)
	})

	suite.Run("multiple probes are tightly packed", func() {
		probes := []light.GPUIrradianceProbe{
			{Position: [4]float32{1, 0, 0, 1}},
			{Position: [4]float32{2, 0, 0, 1}},
		}
		buf := light.MarshalProbeBuffer(probes)
		suite.Len(buf, 320)
		suite.InDelta(1.0, float64(readF32(buf, 0)), 1e-6)
		suite.InDelta(2.0, float64(readF32(buf, 160)), 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUSHProjectParamsSize() {
	suite.Run("size is 16 bytes", func() {
		p := &light.GPUSHProjectParams{}
		suite.Equal(16, p.Size())
	})
}

func (suite *gpuTypesTest) TestGPUSHProjectParamsMarshal() {
	suite.Run("round-trips projection params", func() {
		p := &light.GPUSHProjectParams{
			ProbeIndex: 5,
			FaceIndex:  3,
			Resolution: 32,
		}
		buf := p.Marshal()
		suite.Len(buf, 16)
		suite.Equal(uint32(5), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(3), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(uint32(32), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

func (suite *gpuTypesTest) TestGPUCompositionParamsSize() {
	suite.Run("size is 16 bytes", func() {
		p := &light.GPUCompositionParams{}
		suite.Equal(16, p.Size())
	})
}

func (suite *gpuTypesTest) TestGPUCompositionParamsMarshal() {
	suite.Run("round-trips composition params", func() {
		p := &light.GPUCompositionParams{
			ToneMappingEnabled: 1,
			Exposure:           2.0,
		}
		buf := p.Marshal()
		suite.Len(buf, 16)
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[0:4]))
		suite.InDelta(2.0, float64(readF32(buf, 4)), 1e-6)
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[12:16]))
	})

	suite.Run("disabled tone mapping is serialized", func() {
		p := &light.GPUCompositionParams{
			ToneMappingEnabled: 0,
			Exposure:           1.0,
		}
		buf := p.Marshal()
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[0:4]))
	})
}

func (suite *gpuTypesTest) TestGPUSSRParamsSize() {
	suite.Run("size is 224 bytes", func() {
		p := &light.GPUSSRParams{}
		suite.Equal(224, p.Size())
	})
}

func (suite *gpuTypesTest) TestGPUSSRParamsMarshal() {
	suite.Run("round-trips scalar fields", func() {
		p := &light.GPUSSRParams{
			MaxDistance:     50.0,
			Thickness:       0.1,
			Stride:          1.0,
			MaxSteps:        64,
			ScreenWidth:     1920.0,
			ScreenHeight:    1080.0,
			RoughnessCutoff: 0.5,
			HiZMipCount:     10,
		}
		buf := p.Marshal()
		suite.Len(buf, 224)
		// Scalars start at offset 192 (after 3 × mat4x4 = 192 bytes)
		suite.InDelta(50.0, float64(readF32(buf, 192)), 1e-6)
		suite.InDelta(0.1, float64(readF32(buf, 196)), 1e-6)
		suite.InDelta(1.0, float64(readF32(buf, 200)), 1e-6)
		suite.Equal(uint32(64), binary.LittleEndian.Uint32(buf[204:208]))
		suite.InDelta(1920.0, float64(readF32(buf, 208)), 1e-6)
		suite.InDelta(1080.0, float64(readF32(buf, 212)), 1e-6)
		suite.InDelta(0.5, float64(readF32(buf, 216)), 1e-6)
		suite.Equal(uint32(10), binary.LittleEndian.Uint32(buf[220:224]))
	})

	suite.Run("identity projection matrix round-trips", func() {
		p := &light.GPUSSRParams{}
		p.Projection[0] = 1
		p.Projection[5] = 1
		p.Projection[10] = 1
		p.Projection[15] = 1
		buf := p.Marshal()
		suite.InDelta(1.0, float64(readF32(buf, 0)), 1e-6)
		suite.InDelta(0.0, float64(readF32(buf, 4)), 1e-6)
		suite.InDelta(1.0, float64(readF32(buf, 20)), 1e-6)
	})
}

// readF32 reads a little-endian float32 from buf at the given byte offset.
func readF32(buf []byte, offset int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(buf[offset : offset+4]))
}
