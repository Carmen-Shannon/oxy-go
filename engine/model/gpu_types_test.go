package model_test

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/stretchr/testify/suite"
)

func TestRunGPUTypesTests(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

type gpuTypesTest struct {
	suite.Suite
}

func (suite *gpuTypesTest) TestGPUVertex() {
	suite.Run("Size should return 64 bytes", func() {
		g := &model.GPUVertex{}
		suite.Equal(64, g.Size())
		suite.Equal(64, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 64-byte buffer with correct encoding", func() {
		g := &model.GPUVertex{
			Position: [3]float32{1.0, 2.0, 3.0},
			Normal:   [3]float32{0.1, 0.2, 0.3},
			TexCoord: [2]float32{0.5, 0.6},
			Color:    [4]float32{1.0, 0.0, 0.0, 1.0},
			Tangent:  [4]float32{0.7, 0.8, 0.9, 1.0},
		}
		buf := g.Marshal()
		suite.Equal(64, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(3.0), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(math.Float32bits(0.5), binary.LittleEndian.Uint32(buf[24:28]))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[32:36]))
		suite.Equal(math.Float32bits(0.7), binary.LittleEndian.Uint32(buf[48:52]))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[60:64]))
	})
}

func (suite *gpuTypesTest) TestGPUSkinnedVertex() {
	suite.Run("Size should return 96 bytes", func() {
		g := &model.GPUSkinnedVertex{}
		suite.Equal(96, g.Size())
		suite.Equal(96, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 96-byte buffer with correct encoding", func() {
		g := &model.GPUSkinnedVertex{
			GPUVertex: model.GPUVertex{
				Position: [3]float32{1.0, 2.0, 3.0},
				Normal:   [3]float32{0.1, 0.2, 0.3},
				TexCoord: [2]float32{0.5, 0.6},
				Color:    [4]float32{1.0, 0.0, 0.0, 1.0},
				Tangent:  [4]float32{0.7, 0.8, 0.9, 1.0},
			},
			BoneIndices: [4]uint32{0, 1, 2, 3},
			BoneWeights: [4]float32{0.4, 0.3, 0.2, 0.1},
		}
		buf := g.Marshal()
		suite.Equal(96, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[64:68]))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[68:72]))
		suite.Equal(uint32(3), binary.LittleEndian.Uint32(buf[76:80]))
		suite.Equal(math.Float32bits(0.4), binary.LittleEndian.Uint32(buf[80:84]))
		suite.Equal(math.Float32bits(0.1), binary.LittleEndian.Uint32(buf[92:96]))
	})
}

func (suite *gpuTypesTest) TestComputeBoundingRadius() {
	suite.Run("returns zero for empty slice", func() {
		result := model.ComputeBoundingRadius(nil)
		suite.InDelta(0.0, float64(result), 1e-6)
	})

	suite.Run("returns max distance from origin across vertices", func() {
		vertices := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{3, 4, 0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 5}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{1, 1, 1}}},
		}
		result := model.ComputeBoundingRadius(vertices)
		suite.InDelta(5.0, float64(result), 1e-5)
	})

	suite.Run("returns distance of single vertex", func() {
		vertices := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 3, 4}}},
		}
		result := model.ComputeBoundingRadius(vertices)
		suite.InDelta(5.0, float64(result), 1e-5)
	})
}

func (suite *gpuTypesTest) TestGPUModelData() {
	suite.Run("Size should return 64 bytes", func() {
		g := &model.GPUModelData{}
		suite.Equal(64, g.Size())
		suite.Equal(64, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 64-byte buffer with correct encoding", func() {
		g := &model.GPUModelData{}
		for i := range 16 {
			g.Model[i] = float32(i + 1)
		}
		buf := g.Marshal()
		suite.Equal(64, len(buf))
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(9.0), binary.LittleEndian.Uint32(buf[32:36]))
		suite.Equal(math.Float32bits(16.0), binary.LittleEndian.Uint32(buf[60:64]))
	})
}
