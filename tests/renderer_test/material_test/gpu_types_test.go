package material_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/stretchr/testify/suite"
)

type gpuTypesTest struct {
	suite.Suite
}

func TestGPUTypes(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

func (suite *gpuTypesTest) TestGPUOverlayParamsSize() {
	suite.Run("size is 16 bytes", func() {
		p := &material.GPUOverlayParams{}
		suite.Equal(16, p.Size())
	})
}

func (suite *gpuTypesTest) TestGPUOverlayParamsMarshal() {
	suite.Run("marshal produces 16-byte buffer", func() {
		p := &material.GPUOverlayParams{}
		buf := p.Marshal()
		suite.Len(buf, 16)
	})

	suite.Run("zero values marshal to all zeros", func() {
		p := &material.GPUOverlayParams{
			OverlayColor: [4]float32{0, 0, 0, 0},
		}
		buf := p.Marshal()
		for i, b := range buf {
			suite.Equal(byte(0), b, "byte %d should be zero", i)
		}
	})

	suite.Run("overlay color values are correctly serialized", func() {
		p := &material.GPUOverlayParams{
			OverlayColor: [4]float32{1.0, 0.5, 0.25, 0.75},
		}
		buf := p.Marshal()

		suite.Equal(float32(1.0), readFloat32LE(buf, 0))
		suite.Equal(float32(0.5), readFloat32LE(buf, 4))
		suite.Equal(float32(0.25), readFloat32LE(buf, 8))
		suite.Equal(float32(0.75), readFloat32LE(buf, 12))
	})

	suite.Run("negative values are correctly serialized", func() {
		p := &material.GPUOverlayParams{
			OverlayColor: [4]float32{-1.0, -0.5, -0.25, -0.75},
		}
		buf := p.Marshal()

		suite.Equal(float32(-1.0), readFloat32LE(buf, 0))
		suite.Equal(float32(-0.5), readFloat32LE(buf, 4))
		suite.Equal(float32(-0.25), readFloat32LE(buf, 8))
		suite.Equal(float32(-0.75), readFloat32LE(buf, 12))
	})

	suite.Run("opaque red overlay round-trips through marshal", func() {
		p := &material.GPUOverlayParams{
			OverlayColor: [4]float32{1, 0, 0, 1},
		}
		buf := p.Marshal()

		suite.Equal(float32(1), readFloat32LE(buf, 0))
		suite.Equal(float32(0), readFloat32LE(buf, 4))
		suite.Equal(float32(0), readFloat32LE(buf, 8))
		suite.Equal(float32(1), readFloat32LE(buf, 12))
	})
}

func (suite *gpuTypesTest) TestGPUOverlayParamsSource() {
	suite.Run("embedded WGSL source is not empty", func() {
		suite.NotEmpty(material.GPUOverlayParamsSource)
	})

	suite.Run("embedded WGSL contains OverlayParams struct", func() {
		suite.Contains(material.GPUOverlayParamsSource, "OverlayParams")
	})

	suite.Run("embedded WGSL contains overlay_color field", func() {
		suite.Contains(material.GPUOverlayParamsSource, "overlay_color")
	})
}

func (suite *gpuTypesTest) TestGPUEffectParamsSize() {
	suite.Run("size is 16 bytes", func() {
		p := &material.GPUEffectParams{}
		suite.Equal(16, p.Size())
	})
}

func (suite *gpuTypesTest) TestGPUEffectParamsMarshal() {
	suite.Run("marshal produces 16-byte buffer", func() {
		p := &material.GPUEffectParams{}
		buf := p.Marshal()
		suite.Len(buf, 16)
	})

	suite.Run("zero values marshal to all zeros", func() {
		p := &material.GPUEffectParams{
			TintColor: [4]float32{0, 0, 0, 0},
		}
		buf := p.Marshal()
		for i, b := range buf {
			suite.Equal(byte(0), b, "byte %d should be zero", i)
		}
	})

	suite.Run("tint color values are correctly serialized", func() {
		p := &material.GPUEffectParams{
			TintColor: [4]float32{0.8, 0.2, 0.6, 1.0},
		}
		buf := p.Marshal()

		suite.InDelta(float64(0.8), float64(readFloat32LE(buf, 0)), 1e-6)
		suite.InDelta(float64(0.2), float64(readFloat32LE(buf, 4)), 1e-6)
		suite.InDelta(float64(0.6), float64(readFloat32LE(buf, 8)), 1e-6)
		suite.Equal(float32(1.0), readFloat32LE(buf, 12))
	})

	suite.Run("negative values are correctly serialized", func() {
		p := &material.GPUEffectParams{
			TintColor: [4]float32{-1.0, -0.5, -0.25, -0.75},
		}
		buf := p.Marshal()

		suite.Equal(float32(-1.0), readFloat32LE(buf, 0))
		suite.Equal(float32(-0.5), readFloat32LE(buf, 4))
		suite.Equal(float32(-0.25), readFloat32LE(buf, 8))
		suite.Equal(float32(-0.75), readFloat32LE(buf, 12))
	})

	suite.Run("full tint with zero alpha round-trips through marshal", func() {
		p := &material.GPUEffectParams{
			TintColor: [4]float32{1, 1, 1, 0},
		}
		buf := p.Marshal()

		suite.Equal(float32(1), readFloat32LE(buf, 0))
		suite.Equal(float32(1), readFloat32LE(buf, 4))
		suite.Equal(float32(1), readFloat32LE(buf, 8))
		suite.Equal(float32(0), readFloat32LE(buf, 12))
	})

	suite.Run("alpha controls tint blend intensity", func() {
		// alpha=0 means no tint, alpha=1 means fully tinted
		noTint := &material.GPUEffectParams{
			TintColor: [4]float32{1, 0, 0, 0},
		}
		fullTint := &material.GPUEffectParams{
			TintColor: [4]float32{1, 0, 0, 1},
		}

		noBuf := noTint.Marshal()
		fullBuf := fullTint.Marshal()

		// RGB channels are the same
		suite.Equal(readFloat32LE(noBuf, 0), readFloat32LE(fullBuf, 0))
		suite.Equal(readFloat32LE(noBuf, 4), readFloat32LE(fullBuf, 4))
		suite.Equal(readFloat32LE(noBuf, 8), readFloat32LE(fullBuf, 8))

		// Alpha differs
		suite.Equal(float32(0), readFloat32LE(noBuf, 12))
		suite.Equal(float32(1), readFloat32LE(fullBuf, 12))
	})
}

func (suite *gpuTypesTest) TestGPUEffectParamsSource() {
	suite.Run("embedded WGSL source is not empty", func() {
		suite.NotEmpty(material.GPUEffectParamsSource)
	})

	suite.Run("embedded WGSL contains EffectParams struct", func() {
		suite.Contains(material.GPUEffectParamsSource, "EffectParams")
	})

	suite.Run("embedded WGSL contains tint_color field", func() {
		suite.Contains(material.GPUEffectParamsSource, "tint_color")
	})
}

func (suite *gpuTypesTest) TestOverlayAndEffectHaveSameSize() {
	suite.Run("both GPU structs are 16 bytes", func() {
		overlay := &material.GPUOverlayParams{}
		effect := &material.GPUEffectParams{}
		suite.Equal(overlay.Size(), effect.Size())
		suite.Equal(16, overlay.Size())
	})
}

// readFloat32LE reads a little-endian float32 from a byte slice at the given offset.
//
// Parameters:
//   - buf: the byte slice to read from
//   - offset: the byte offset to start reading
//
// Returns:
//   - float32: the decoded float32 value
func readFloat32LE(buf []byte, offset int) float32 {
	bits := binary.LittleEndian.Uint32(buf[offset : offset+4])
	return math.Float32frombits(bits)
}
