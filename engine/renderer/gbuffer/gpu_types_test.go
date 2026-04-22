package gbuffer_test

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/gbuffer"
	"github.com/stretchr/testify/suite"
)

func TestRunGPUTypesTests(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

type gpuTypesTest struct {
	suite.Suite
}

func (suite *gpuTypesTest) TestGPUGBufferOutput() {
	suite.Run("Size should return 48 bytes", func() {
		g := &gbuffer.GPUGBufferOutput{}
		suite.Equal(48, g.Size())
		suite.Equal(48, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 48-byte buffer with correct encoding", func() {
		g := &gbuffer.GPUGBufferOutput{
			Position: [4]float32{1, 2, 3, 4},
			Normal:   [4]float32{0.5, 0.5, 0.5, 0.8},
			Albedo:   [4]float32{1, 0, 0, 0.2},
		}
		buf := g.Marshal()
		suite.Equal(48, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(0.5), binary.LittleEndian.Uint32(buf[16:20]))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[32:36]))
		suite.Equal(math.Float32bits(0.2), binary.LittleEndian.Uint32(buf[44:48]))
	})
}
