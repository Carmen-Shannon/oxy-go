package postprocessing_test

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing"
	"github.com/stretchr/testify/suite"
)

func TestRunGPUTypesTests(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

type gpuTypesTest struct {
	suite.Suite
}

func (suite *gpuTypesTest) TestGPUBlurParams() {
	suite.Run("Size should return 24 bytes", func() {
		p := &postprocessing.GPUBlurParams{}
		suite.Equal(24, p.Size())
		suite.Equal(24, int(unsafe.Sizeof(*p)))
	})

	suite.Run("Marshal should return a 24-byte buffer with correct encoding", func() {
		p := &postprocessing.GPUBlurParams{
			Direction:    [2]int32{1, 0},
			Radius:       4,
			GBufferScale: 2,
			CascadeWidth: 512,
		}
		buf := p.Marshal()
		suite.Equal(24, len(buf))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(uint32(4), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(uint32(2), binary.LittleEndian.Uint32(buf[12:16]))
		suite.Equal(uint32(512), binary.LittleEndian.Uint32(buf[16:20]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[20:24]))
	})
}

func (suite *gpuTypesTest) TestGPUTAAParams() {
	suite.Run("Size should return 176 bytes", func() {
		p := &postprocessing.GPUTAAParams{}
		suite.Equal(176, p.Size())
		suite.Equal(176, int(unsafe.Sizeof(*p)))
	})

	suite.Run("Marshal should return a 176-byte buffer with correct encoding", func() {
		var inv [16]float32
		var prev [16]float32
		for i := range inv {
			inv[i] = float32(i + 1)
			prev[i] = float32(i + 17)
		}

		p := &postprocessing.GPUTAAParams{
			InvCurrViewProj: inv,
			PrevViewProj:    prev,
			JitterCurr:      [2]float32{0.25, -0.5},
			JitterPrev:      [2]float32{-0.125, 0.75},
			ScreenWidth:     1920,
			ScreenHeight:    1080,
			BlendFactor:     0.1,
		}
		buf := p.Marshal()
		suite.Equal(176, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(16.0), binary.LittleEndian.Uint32(buf[60:64]))
		suite.Equal(math.Float32bits(17.0), binary.LittleEndian.Uint32(buf[64:68]))
		suite.Equal(math.Float32bits(32.0), binary.LittleEndian.Uint32(buf[124:128]))
		suite.Equal(math.Float32bits(0.25), binary.LittleEndian.Uint32(buf[128:132]))
		suite.Equal(math.Float32bits(-0.5), binary.LittleEndian.Uint32(buf[132:136]))
		suite.Equal(math.Float32bits(-0.125), binary.LittleEndian.Uint32(buf[136:140]))
		suite.Equal(math.Float32bits(0.75), binary.LittleEndian.Uint32(buf[140:144]))
		suite.Equal(math.Float32bits(1920.0), binary.LittleEndian.Uint32(buf[144:148]))
		suite.Equal(math.Float32bits(1080.0), binary.LittleEndian.Uint32(buf[148:152]))
		suite.Equal(math.Float32bits(0.1), binary.LittleEndian.Uint32(buf[152:156]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[156:160])) // HistoryRectificationScale (0)
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[160:164])) // RawHistoryOnly (0)
	})

	suite.Run("Marshal should encode non-zero HistoryRectificationScale and RawHistoryOnly", func() {
		var inv [16]float32
		var prev [16]float32
		for i := range inv {
			inv[i] = float32(i + 1)
			prev[i] = float32(i + 17)
		}

		p := &postprocessing.GPUTAAParams{
			InvCurrViewProj:           inv,
			PrevViewProj:              prev,
			JitterCurr:                [2]float32{0.25, -0.5},
			JitterPrev:                [2]float32{-0.125, 0.75},
			ScreenWidth:               1920,
			ScreenHeight:              1080,
			BlendFactor:               0.1,
			HistoryRectificationScale: 2.5,
			RawHistoryOnly:            1.0,
		}
		buf := p.Marshal()
		suite.Equal(176, len(buf))
		suite.Equal(math.Float32bits(2.5), binary.LittleEndian.Uint32(buf[156:160]))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[160:164]))
	})
}

func (suite *gpuTypesTest) TestGPUSSAOParams() {
	suite.Run("Size should return 176 bytes", func() {
		p := &postprocessing.GPUSSAOParams{}
		suite.Equal(176, p.Size())
		suite.Equal(176, int(unsafe.Sizeof(*p)))
	})

	suite.Run("Marshal should return a 176-byte buffer with correct encoding", func() {
		var proj [16]float32
		for i := range proj {
			proj[i] = float32(i + 1)
		}
		var invVP [16]float32
		for i := range invVP {
			invVP[i] = 2.0
		}
		p := &postprocessing.GPUSSAOParams{
			Projection:     proj,
			InvViewProj:    invVP,
			Radius:         0.5,
			Bias:           0.025,
			SampleCount:    16,
			ScreenWidth:    1920,
			ScreenHeight:   1080,
			GBufferScale:   1.0,
			CameraPosition: [3]float32{10, 20, 30},
		}
		buf := p.Marshal()
		suite.Equal(176, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(0.5), binary.LittleEndian.Uint32(buf[128:132]))
		suite.Equal(uint32(16), binary.LittleEndian.Uint32(buf[140:144]))
		suite.Equal(math.Float32bits(1920.0), binary.LittleEndian.Uint32(buf[144:148]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[156:160]))
		suite.Equal(math.Float32bits(10.0), binary.LittleEndian.Uint32(buf[160:164]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[172:176]))
	})
}

func (suite *gpuTypesTest) TestGPUCompositionParams() {
	suite.Run("Size should return 32 bytes", func() {
		p := &postprocessing.GPUCompositionParams{}
		suite.Equal(32, p.Size())
		suite.Equal(32, int(unsafe.Sizeof(*p)))
	})

	suite.Run("Marshal should return a 32-byte buffer with correct encoding", func() {
		p := &postprocessing.GPUCompositionParams{
			ToneMappingEnabled:  1,
			Exposure:            2.0,
			AutoExposureEnabled: 1,
			BloomEnabled:        1,
			BloomIntensity:      0.75,
		}
		buf := p.Marshal()
		suite.Equal(32, len(buf))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(2.0), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[12:16]))
		suite.Equal(math.Float32bits(0.75), binary.LittleEndian.Uint32(buf[16:20]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[20:24]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[24:28]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[28:32]))
	})
}

func (suite *gpuTypesTest) TestGPUBloomParams() {
	suite.Run("Size should return 16 bytes", func() {
		p := &postprocessing.GPUBloomParams{}
		suite.Equal(16, p.Size())
		suite.Equal(16, int(unsafe.Sizeof(*p)))
	})

	suite.Run("Marshal should return a 16-byte buffer with correct encoding", func() {
		p := &postprocessing.GPUBloomParams{
			Threshold: 1.5,
		}
		buf := p.Marshal()
		suite.Equal(16, len(buf))
		suite.Equal(math.Float32bits(1.5), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

func (suite *gpuTypesTest) TestGPUSSRParams() {
	suite.Run("Size should return 224 bytes", func() {
		p := &postprocessing.GPUSSRParams{}
		suite.Equal(224, p.Size())
		suite.Equal(224, int(unsafe.Sizeof(*p)))
	})

	suite.Run("Marshal should return a 224-byte buffer with correct encoding", func() {
		var proj [16]float32
		for i := range proj {
			proj[i] = float32(i + 1)
		}
		var ones [16]float32
		for i := range ones {
			ones[i] = 1.0
		}
		p := &postprocessing.GPUSSRParams{
			Projection:    proj,
			InvProjection: ones,
			View:          ones,
			MaxDistance:   5.0,
			Thickness:     0.1,
			MaxSteps:      64,
			ScreenWidth:   1920,
			ScreenHeight:  1080,
			HiZMipCount:   8,
		}
		buf := p.Marshal()
		suite.Equal(224, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(5.0), binary.LittleEndian.Uint32(buf[192:196]))
		suite.Equal(uint32(64), binary.LittleEndian.Uint32(buf[204:208]))
		suite.Equal(math.Float32bits(1920.0), binary.LittleEndian.Uint32(buf[208:212]))
		suite.Equal(uint32(8), binary.LittleEndian.Uint32(buf[220:224]))
	})
}

func (suite *gpuTypesTest) TestGPULuminanceParamsSize() {
	suite.Run("Size should return 32 bytes", func() {
		p := &postprocessing.GPULuminanceParams{}
		suite.Equal(uint64(32), p.Size())
		suite.Equal(32, int(unsafe.Sizeof(*p)))
	})
}

func (suite *gpuTypesTest) TestGPULuminanceParamsMarshal() {
	suite.Run("Marshal should return a 32-byte buffer with correct encoding", func() {
		p := &postprocessing.GPULuminanceParams{
			ScreenWidth:         1920,
			ScreenHeight:        1080,
			AdaptSpeed:          1.5,
			DeltaTime:           0.016,
			MinExposure:         0.1,
			MaxExposure:         10.0,
			KeyValue:            0.18,
			AutoExposureEnabled: 1,
		}
		buf := p.Marshal()
		suite.Equal(32, len(buf))
		suite.Equal(uint32(1920), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(1080), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(math.Float32bits(1.5), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(math.Float32bits(0.016), binary.LittleEndian.Uint32(buf[12:16]))
		suite.Equal(math.Float32bits(0.1), binary.LittleEndian.Uint32(buf[16:20]))
		suite.Equal(math.Float32bits(10.0), binary.LittleEndian.Uint32(buf[20:24]))
		suite.Equal(math.Float32bits(0.18), binary.LittleEndian.Uint32(buf[24:28]))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[28:32]))
	})
}
