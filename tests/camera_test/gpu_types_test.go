package camera_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/stretchr/testify/suite"
)

type gpuTypesTest struct {
	suite.Suite
}

func TestGPUTypes(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

func (suite *gpuTypesTest) TestGPUCameraUniformSize() {
	suite.Run("size is 80 bytes", func() {
		u := &camera.GPUCameraUniform{}
		suite.Equal(80, u.Size())
	})
}

func (suite *gpuTypesTest) TestGPUCameraUniformMarshal() {
	suite.Run("marshal returns correct byte length", func() {
		u := &camera.GPUCameraUniform{}
		buf := u.Marshal()
		suite.Len(buf, 80)
	})

	suite.Run("marshal encodes view-proj matrix correctly", func() {
		u := &camera.GPUCameraUniform{}
		// Set identity-like values in ViewProj
		u.ViewProj = [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		buf := u.Marshal()

		// Check first element (1.0)
		bits := binary.LittleEndian.Uint32(buf[0:4])
		suite.InDelta(1.0, float64(math.Float32frombits(bits)), 1e-6)

		// Check second element (0.0)
		bits = binary.LittleEndian.Uint32(buf[4:8])
		suite.InDelta(0.0, float64(math.Float32frombits(bits)), 1e-6)

		// Check diagonal element at index 5 (offset 20 bytes) = 1.0
		bits = binary.LittleEndian.Uint32(buf[20:24])
		suite.InDelta(1.0, float64(math.Float32frombits(bits)), 1e-6)

		// Check last ViewProj element at index 15 (offset 60) = 1.0
		bits = binary.LittleEndian.Uint32(buf[60:64])
		suite.InDelta(1.0, float64(math.Float32frombits(bits)), 1e-6)
	})

	suite.Run("marshal encodes camera position correctly", func() {
		u := &camera.GPUCameraUniform{}
		u.CameraPosition = [3]float32{1.5, 2.5, 3.5}
		buf := u.Marshal()

		// Camera position starts at offset 64
		xBits := binary.LittleEndian.Uint32(buf[64:68])
		yBits := binary.LittleEndian.Uint32(buf[68:72])
		zBits := binary.LittleEndian.Uint32(buf[72:76])

		suite.InDelta(1.5, float64(math.Float32frombits(xBits)), 1e-6)
		suite.InDelta(2.5, float64(math.Float32frombits(yBits)), 1e-6)
		suite.InDelta(3.5, float64(math.Float32frombits(zBits)), 1e-6)
	})

	suite.Run("marshal sets padding byte to zero", func() {
		u := &camera.GPUCameraUniform{}
		u.CameraPosition = [3]float32{10, 20, 30}
		buf := u.Marshal()

		// Padding at offset 76 should be zero
		padBits := binary.LittleEndian.Uint32(buf[76:80])
		suite.Equal(uint32(0), padBits)
	})

	suite.Run("marshal round-trips specific values", func() {
		u := &camera.GPUCameraUniform{}
		u.ViewProj = [16]float32{
			0.5, 0.1, 0.2, 0.3,
			0.4, 0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2,
			1.3, 1.4, 1.5, 1.6,
		}
		u.CameraPosition = [3]float32{100.0, -50.0, 0.001}

		buf := u.Marshal()

		// Verify all 16 ViewProj elements
		for i := 0; i < 16; i++ {
			bits := binary.LittleEndian.Uint32(buf[i*4 : i*4+4])
			suite.InDelta(float64(u.ViewProj[i]), float64(math.Float32frombits(bits)), 1e-6)
		}

		// Verify all 3 CameraPosition elements
		for i := 0; i < 3; i++ {
			bits := binary.LittleEndian.Uint32(buf[64+i*4 : 64+i*4+4])
			suite.InDelta(float64(u.CameraPosition[i]), float64(math.Float32frombits(bits)), 1e-6)
		}
	})
}

func (suite *gpuTypesTest) TestGPUCameraUniformSource() {
	suite.Run("wgsl source is non-empty", func() {
		suite.NotEmpty(camera.GPUCameraUniformSource)
	})

	suite.Run("wgsl source contains CameraUniform struct", func() {
		suite.Contains(camera.GPUCameraUniformSource, "CameraUniform")
	})
}
