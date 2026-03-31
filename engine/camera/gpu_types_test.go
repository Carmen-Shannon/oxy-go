package camera

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/suite"
)

func TestRunGPUTypesTests(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

type gpuTypesTest struct {
	suite.Suite
}

func (suite *gpuTypesTest) TestGPUCameraUniform() {
	suite.Run("Size should return 144 bytes", func() {
		g := &GPUCameraUniform{}
		suite.Equal(144, g.Size())
		suite.Equal(144, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 144-byte buffer with correct float encoding", func() {
		g := &GPUCameraUniform{
			ViewProj:       [16]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			View:           [16]float32{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
			CameraPosition: [3]float32{1.0, 2.0, 3.0},
		}

		buf := g.Marshal()
		suite.Equal(144, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(16.0), binary.LittleEndian.Uint32(buf[60:64]))
		suite.Equal(math.Float32bits(16.0), binary.LittleEndian.Uint32(buf[64:68]))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[124:128]))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[128:132]))
		suite.Equal(math.Float32bits(3.0), binary.LittleEndian.Uint32(buf[136:140]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[140:144]))
	})
}
