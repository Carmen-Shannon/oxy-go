package light_test

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/stretchr/testify/suite"
)

func TestRunGPUTypesTests(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

type gpuTypesTest struct {
	suite.Suite
}

func (suite *gpuTypesTest) TestGPULight() {
	suite.Run("Size should return 64 bytes", func() {
		g := &light.GPULight{}
		suite.Equal(64, g.Size())
		suite.Equal(64, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 64-byte buffer with correct encoding", func() {
		g := &light.GPULight{
			Position:     [3]float32{1, 2, 3},
			LightType:    1,
			Color:        [3]float32{0.5, 0.6, 0.7},
			Intensity:    2.5,
			CastsShadows: 1,
			ShadowIndex:  0xFFFFFFFF,
		}
		buf := g.Marshal()
		suite.Equal(64, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[12:16]))
		suite.Equal(math.Float32bits(2.5), binary.LittleEndian.Uint32(buf[28:32]))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[56:60]))
		suite.Equal(uint32(0xFFFFFFFF), binary.LittleEndian.Uint32(buf[60:64]))
	})
}

func (suite *gpuTypesTest) TestGPULightHeader() {
	suite.Run("Size should return 16 bytes", func() {
		h := &light.GPULightHeader{}
		suite.Equal(16, h.Size())
		suite.Equal(16, int(unsafe.Sizeof(*h)))
	})

	suite.Run("Marshal should return a 16-byte buffer with correct encoding", func() {
		h := &light.GPULightHeader{
			AmbientColor: [3]float32{0.1, 0.2, 0.3},
			LightCount:   5,
		}
		buf := h.Marshal()
		suite.Equal(16, len(buf))
		suite.Equal(math.Float32bits(0.1), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(0.3), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(uint32(5), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

func (suite *gpuTypesTest) TestGPUShadowUniform() {
	suite.Run("Size should return 64 bytes", func() {
		u := &light.GPUShadowUniform{}
		suite.Equal(64, u.Size())
		suite.Equal(64, int(unsafe.Sizeof(*u)))
	})

	suite.Run("Marshal should return a 64-byte buffer with correct encoding", func() {
		var vp [16]float32
		for i := range vp {
			vp[i] = float32(i + 1)
		}
		u := &light.GPUShadowUniform{LightVP: vp}
		buf := u.Marshal()
		suite.Equal(64, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(9.0), binary.LittleEndian.Uint32(buf[32:36]))
		suite.Equal(math.Float32bits(16.0), binary.LittleEndian.Uint32(buf[60:64]))
	})
}

func (suite *gpuTypesTest) TestGPUCSMCascade() {
	suite.Run("Size should return 80 bytes", func() {
		c := &light.GPUCSMCascade{}
		suite.Equal(80, c.Size())
		suite.Equal(80, int(unsafe.Sizeof(*c)))
	})

	suite.Run("Marshal should return an 80-byte buffer with correct encoding", func() {
		var vp [16]float32
		for i := range vp {
			vp[i] = float32(i + 1)
		}
		c := &light.GPUCSMCascade{
			LightVP:    vp,
			ShadowNear: 0.1,
			ShadowFar:  100.0,
			CamFar:     200.0,
			NormalBias: 0.005,
		}
		buf := c.Marshal()
		suite.Equal(80, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(0.1), binary.LittleEndian.Uint32(buf[64:68]))
		suite.Equal(math.Float32bits(100.0), binary.LittleEndian.Uint32(buf[68:72]))
		suite.Equal(math.Float32bits(0.005), binary.LittleEndian.Uint32(buf[76:80]))
	})
}

func (suite *gpuTypesTest) TestGPUCSMData() {
	suite.Run("Size should return 192 bytes", func() {
		g := &light.GPUCSMData{}
		suite.Equal(192, g.Size())
	})

	suite.Run("Marshal should return a 192-byte buffer with correct encoding", func() {
		g := &light.GPUCSMData{
			TexelSize:   [2]float32{0.001, 0.002},
			Bias:        0.005,
			InnerRadius: 10.0,
		}
		buf := g.Marshal()
		suite.Equal(192, len(buf))
		suite.Equal(math.Float32bits(0.001), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(0.002), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(math.Float32bits(0.005), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(math.Float32bits(10.0), binary.LittleEndian.Uint32(buf[12:16]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[24:28]))
	})
}

func (suite *gpuTypesTest) TestGPULightShadowEntry() {
	suite.Run("Size should return 96 bytes", func() {
		g := &light.GPULightShadowEntry{}
		suite.Equal(96, g.Size())
		suite.Equal(96, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 96-byte buffer with correct encoding", func() {
		var vp [16]float32
		for i := range vp {
			vp[i] = float32(i + 1)
		}
		g := &light.GPULightShadowEntry{
			LightVP:    vp,
			AtlasRect:  [4]float32{0.1, 0.2, 0.3, 0.4},
			Bias:       0.001,
			Near:       0.1,
			Far:        50.0,
			ShadowType: light.ShadowTypeSpot,
		}
		buf := g.Marshal()
		suite.Equal(96, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(0.1), binary.LittleEndian.Uint32(buf[64:68]))
		suite.Equal(math.Float32bits(0.001), binary.LittleEndian.Uint32(buf[80:84]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[92:96]))
	})
}

func (suite *gpuTypesTest) TestGPULightCullUniforms() {
	suite.Run("Size should return 160 bytes", func() {
		u := &light.GPULightCullUniforms{}
		suite.Equal(160, u.Size())
		suite.Equal(160, int(unsafe.Sizeof(*u)))
	})

	suite.Run("Marshal should return a 160-byte buffer with correct encoding", func() {
		var inv [16]float32
		for i := range inv {
			inv[i] = float32(i + 1)
		}
		var view [16]float32
		for i := range view {
			view[i] = float32(16 - i)
		}
		u := &light.GPULightCullUniforms{
			InvProj:    inv,
			ViewMatrix: view,
			TileCountX: 20,
			TileCountY: 10,
			LightCount: 5,
			Near:       0.1,
			Far:        100.0,
		}
		buf := u.Marshal()
		suite.Equal(160, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(20), binary.LittleEndian.Uint32(buf[128:132]))
		suite.Equal(uint32(5), binary.LittleEndian.Uint32(buf[144:148]))
		suite.Equal(math.Float32bits(0.1), binary.LittleEndian.Uint32(buf[148:152]))
		suite.Equal(math.Float32bits(100.0), binary.LittleEndian.Uint32(buf[152:156]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[156:160]))
	})
}

func (suite *gpuTypesTest) TestGPUTileUniforms() {
	suite.Run("Size should return 16 bytes", func() {
		u := &light.GPUTileUniforms{}
		suite.Equal(16, u.Size())
		suite.Equal(16, int(unsafe.Sizeof(*u)))
	})

	suite.Run("Marshal should return a 16-byte buffer with correct encoding", func() {
		u := &light.GPUTileUniforms{
			TileCountX:       20,
			MaxLightsPerTile: 64,
			ScreenWidth:      1920,
			ScreenHeight:     1080,
		}
		buf := u.Marshal()
		suite.Equal(16, len(buf))
		suite.Equal(uint32(20), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(64), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(uint32(1920), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(uint32(1080), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

func (suite *gpuTypesTest) TestGPUContactShadowParams() {
	suite.Run("Size should return 176 bytes", func() {
		p := &light.GPUContactShadowParams{}
		suite.Equal(176, p.Size())
		suite.Equal(176, int(unsafe.Sizeof(*p)))
	})

	suite.Run("Marshal should return a 176-byte buffer with correct encoding", func() {
		var vp [16]float32
		for i := range vp {
			vp[i] = float32(i + 1)
		}
		var ones [16]float32
		for i := range ones {
			ones[i] = 1.0
		}
		p := &light.GPUContactShadowParams{
			ViewProj:       vp,
			InvViewProj:    ones,
			LightDirection: [3]float32{0, -1, 0},
			StepCount:      16,
			MaxDistance:    1.0,
			Thickness:      0.05,
			ScreenWidth:    1920,
			ScreenHeight:   1080,
			CameraPosition: [3]float32{0, 5, 10},
		}
		buf := p.Marshal()
		suite.Equal(176, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(0.0), binary.LittleEndian.Uint32(buf[128:132]))
		suite.Equal(math.Float32bits(-1.0), binary.LittleEndian.Uint32(buf[132:136]))
		suite.Equal(uint32(16), binary.LittleEndian.Uint32(buf[140:144]))
		suite.Equal(math.Float32bits(1920.0), binary.LittleEndian.Uint32(buf[152:156]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[172:176]))
	})
}

func (suite *gpuTypesTest) TestToGPULight() {
	suite.Run("should convert a Light to a GPULight with correct field values", func() {
		l := light.NewLight(light.LightTypeSpot,
			light.WithPosition(1, 2, 3),
			light.WithColor(1, 0, 0),
			light.WithIntensity(2.0),
			light.WithRange(50.0),
			light.WithCastsShadows(true),
		)
		gpu := light.ToGPULight(l)
		suite.Equal([3]float32{1, 2, 3}, gpu.Position)
		suite.Equal(uint32(light.LightTypeSpot), gpu.LightType)
		suite.Equal([3]float32{1, 0, 0}, gpu.Color)
		suite.Equal(float32(2.0), gpu.Intensity)
		suite.Equal(float32(50.0), gpu.LightRange)
		suite.Equal(uint32(1), gpu.CastsShadows)
		suite.Equal(uint32(0xFFFFFFFF), gpu.ShadowIndex)
	})

	suite.Run("should set CastsShadows to 0 when shadows are not enabled", func() {
		l := light.NewLight(light.LightTypePoint,
			light.WithIntensity(1.0),
		)
		gpu := light.ToGPULight(l)
		suite.Equal(uint32(0), gpu.CastsShadows)
		suite.Equal(uint32(0xFFFFFFFF), gpu.ShadowIndex)
	})
}

func (suite *gpuTypesTest) TestComputeCascades() {
	suite.Run("should compute non-zero cascade matrices for a standard scene", func() {
		g := &light.GPUCSMData{}
		lightDir := [3]float32{0, -1, 0}
		camView := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		g.ComputeCascades(lightDir, 0.1, 100.0, float32(math.Pi/4), 16.0/9.0, camView, [3]float32{0, 10, 0}, 20.0, 1.0, 2048)
		suite.NotEqual([16]float32{}, g.Cascades[0].LightVP)
		suite.NotEqual([16]float32{}, g.Cascades[1].LightVP)
		suite.Greater(g.Cascades[0].ShadowNear, float32(0))
		suite.Greater(g.Cascades[1].ShadowNear, float32(0))
	})

	suite.Run("should handle a light direction near the up axis by using a fallback up vector", func() {
		g := &light.GPUCSMData{}
		lightDir := [3]float32{0, 1, 0}
		camView := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		g.ComputeCascades(lightDir, 0.1, 100.0, float32(math.Pi/4), 16.0/9.0, camView, [3]float32{0, 10, 0}, 20.0, 1.0, 2048)
		suite.NotEqual([16]float32{}, g.Cascades[0].LightVP)
	})

	suite.Run("should clamp cascade 0 sceneDepthPadding to minimum when innerRadius is small", func() {
		g := &light.GPUCSMData{}
		lightDir := [3]float32{0, 0, -1}
		camView := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		g.ComputeCascades(lightDir, 0.1, 100.0, float32(math.Pi/4), 1.0, camView, [3]float32{0, 0, 0}, 1.0, 1.0, 2048)
		// innerRadius=1.0 → cascade 0: sceneDepthPadding = 0.5 < 5.0 → clamped to 5.0
		suite.NotEqual([16]float32{}, g.Cascades[0].LightVP)
		suite.Greater(g.Cascades[0].ShadowNear, float32(0))
	})

	suite.Run("should use identity fallback and clamp cascade 1 values when matrix is singular and frustum is tiny", func() {
		g := &light.GPUCSMData{}
		lightDir := [3]float32{0, 0, -1}
		camView := [16]float32{} // all zeros → Invert4 returns false → identity fallback
		g.ComputeCascades(lightDir, 0.1, 0.3, 0.01, 1.0, camView, [3]float32{0, 0, 0}, 1.0, 1.0, 2048)
		// Invert4 fails → common.Identity(invView[:]) is called
		// tiny frustum (camFar=0.3) → sphereRadius ≈ 0.1 < 0.5 → clamped to 0.5
		// tiny frustum depth → sceneDepthPadding ≈ 0.1 < 5.0 → clamped to 5.0
		suite.NotEqual([16]float32{}, g.Cascades[1].LightVP)
		suite.Greater(g.Cascades[1].ShadowNear, float32(0))
	})
}
