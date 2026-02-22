package model_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/stretchr/testify/suite"
)

type gpuTypesTest struct {
	suite.Suite
}

func TestGPUTypes(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

func (suite *gpuTypesTest) TestGPUVertexSize() {
	suite.Run("returns 64 bytes", func() {
		v := &model.GPUVertex{}
		suite.Equal(64, v.Size())
	})
}

func (suite *gpuTypesTest) TestGPUVertexMarshal() {
	suite.Run("zero value produces 64-byte buffer", func() {
		v := &model.GPUVertex{}
		buf := v.Marshal()
		suite.Len(buf, 64)
	})

	suite.Run("all zero fields produce all-zero bytes", func() {
		v := &model.GPUVertex{}
		buf := v.Marshal()
		for i, b := range buf {
			suite.Equal(byte(0), b, "byte %d should be zero", i)
		}
	})

	suite.Run("position encoded at offset 0", func() {
		v := &model.GPUVertex{Position: [3]float32{1.0, 2.0, 3.0}}
		buf := v.Marshal()
		suite.InDelta(1.0, readFloat32(buf, 0), 1e-6)
		suite.InDelta(2.0, readFloat32(buf, 4), 1e-6)
		suite.InDelta(3.0, readFloat32(buf, 8), 1e-6)
	})

	suite.Run("normal encoded at offset 12", func() {
		v := &model.GPUVertex{Normal: [3]float32{0.0, 1.0, 0.0}}
		buf := v.Marshal()
		suite.InDelta(0.0, readFloat32(buf, 12), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 16), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 20), 1e-6)
	})

	suite.Run("texcoord encoded at offset 24", func() {
		v := &model.GPUVertex{TexCoord: [2]float32{0.5, 0.75}}
		buf := v.Marshal()
		suite.InDelta(0.5, readFloat32(buf, 24), 1e-6)
		suite.InDelta(0.75, readFloat32(buf, 28), 1e-6)
	})

	suite.Run("color encoded at offset 32", func() {
		v := &model.GPUVertex{Color: [4]float32{1.0, 0.5, 0.25, 1.0}}
		buf := v.Marshal()
		suite.InDelta(1.0, readFloat32(buf, 32), 1e-6)
		suite.InDelta(0.5, readFloat32(buf, 36), 1e-6)
		suite.InDelta(0.25, readFloat32(buf, 40), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 44), 1e-6)
	})

	suite.Run("tangent encoded at offset 48", func() {
		v := &model.GPUVertex{Tangent: [4]float32{1.0, 0.0, 0.0, 1.0}}
		buf := v.Marshal()
		suite.InDelta(1.0, readFloat32(buf, 48), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 52), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 56), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 60), 1e-6)
	})

	suite.Run("all fields round-trip correctly", func() {
		v := &model.GPUVertex{
			Position: [3]float32{-1.5, 2.5, 0.0},
			Normal:   [3]float32{0.0, 0.0, 1.0},
			TexCoord: [2]float32{0.25, 0.75},
			Color:    [4]float32{1.0, 0.0, 0.0, 1.0},
			Tangent:  [4]float32{0.0, 1.0, 0.0, -1.0},
		}
		buf := v.Marshal()

		suite.InDelta(-1.5, readFloat32(buf, 0), 1e-6)
		suite.InDelta(2.5, readFloat32(buf, 4), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 8), 1e-6)

		suite.InDelta(0.0, readFloat32(buf, 12), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 16), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 20), 1e-6)

		suite.InDelta(0.25, readFloat32(buf, 24), 1e-6)
		suite.InDelta(0.75, readFloat32(buf, 28), 1e-6)

		suite.InDelta(1.0, readFloat32(buf, 32), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 36), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 40), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 44), 1e-6)

		suite.InDelta(0.0, readFloat32(buf, 48), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 52), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 56), 1e-6)
		suite.InDelta(-1.0, readFloat32(buf, 60), 1e-6)
	})

	suite.Run("negative values serialize correctly", func() {
		v := &model.GPUVertex{Position: [3]float32{-100.0, -200.0, -300.0}}
		buf := v.Marshal()
		suite.InDelta(-100.0, readFloat32(buf, 0), 1e-6)
		suite.InDelta(-200.0, readFloat32(buf, 4), 1e-6)
		suite.InDelta(-300.0, readFloat32(buf, 8), 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUSkinnedVertexSize() {
	suite.Run("returns 96 bytes", func() {
		v := &model.GPUSkinnedVertex{}
		suite.Equal(96, v.Size())
	})
}

func (suite *gpuTypesTest) TestGPUSkinnedVertexMarshal() {
	suite.Run("zero value produces 96-byte buffer", func() {
		v := &model.GPUSkinnedVertex{}
		buf := v.Marshal()
		suite.Len(buf, 96)
	})

	suite.Run("base vertex fields encoded in first 64 bytes", func() {
		v := &model.GPUSkinnedVertex{}
		v.Position = [3]float32{1.0, 2.0, 3.0}
		v.Normal = [3]float32{0.0, 1.0, 0.0}
		buf := v.Marshal()
		suite.InDelta(1.0, readFloat32(buf, 0), 1e-6)
		suite.InDelta(2.0, readFloat32(buf, 4), 1e-6)
		suite.InDelta(3.0, readFloat32(buf, 8), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 12), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 16), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 20), 1e-6)
	})

	suite.Run("bone indices encoded at offset 64", func() {
		v := &model.GPUSkinnedVertex{BoneIndices: [4]uint32{0, 1, 2, 3}}
		buf := v.Marshal()
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[64:68]))
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[68:72]))
		suite.Equal(uint32(2), binary.LittleEndian.Uint32(buf[72:76]))
		suite.Equal(uint32(3), binary.LittleEndian.Uint32(buf[76:80]))
	})

	suite.Run("bone weights encoded at offset 80", func() {
		v := &model.GPUSkinnedVertex{BoneWeights: [4]float32{0.5, 0.3, 0.15, 0.05}}
		buf := v.Marshal()
		suite.InDelta(0.5, readFloat32(buf, 80), 1e-6)
		suite.InDelta(0.3, readFloat32(buf, 84), 1e-6)
		suite.InDelta(0.15, readFloat32(buf, 88), 1e-6)
		suite.InDelta(0.05, readFloat32(buf, 92), 1e-6)
	})

	suite.Run("all fields round-trip correctly", func() {
		v := &model.GPUSkinnedVertex{
			GPUVertex: model.GPUVertex{
				Position: [3]float32{5.0, 6.0, 7.0},
				Normal:   [3]float32{0.0, 0.0, 1.0},
				TexCoord: [2]float32{0.1, 0.9},
				Color:    [4]float32{1.0, 1.0, 1.0, 1.0},
				Tangent:  [4]float32{1.0, 0.0, 0.0, 1.0},
			},
			BoneIndices: [4]uint32{10, 20, 30, 40},
			BoneWeights: [4]float32{0.4, 0.3, 0.2, 0.1},
		}
		buf := v.Marshal()

		suite.InDelta(5.0, readFloat32(buf, 0), 1e-6)
		suite.InDelta(6.0, readFloat32(buf, 4), 1e-6)
		suite.InDelta(7.0, readFloat32(buf, 8), 1e-6)
		suite.InDelta(0.1, readFloat32(buf, 24), 1e-6)
		suite.InDelta(0.9, readFloat32(buf, 28), 1e-6)

		suite.Equal(uint32(10), binary.LittleEndian.Uint32(buf[64:68]))
		suite.Equal(uint32(20), binary.LittleEndian.Uint32(buf[68:72]))
		suite.Equal(uint32(30), binary.LittleEndian.Uint32(buf[72:76]))
		suite.Equal(uint32(40), binary.LittleEndian.Uint32(buf[76:80]))

		suite.InDelta(0.4, readFloat32(buf, 80), 1e-6)
		suite.InDelta(0.3, readFloat32(buf, 84), 1e-6)
		suite.InDelta(0.2, readFloat32(buf, 88), 1e-6)
		suite.InDelta(0.1, readFloat32(buf, 92), 1e-6)
	})

	suite.Run("large bone indices serialize correctly", func() {
		v := &model.GPUSkinnedVertex{BoneIndices: [4]uint32{255, 1024, 65535, 0}}
		buf := v.Marshal()
		suite.Equal(uint32(255), binary.LittleEndian.Uint32(buf[64:68]))
		suite.Equal(uint32(1024), binary.LittleEndian.Uint32(buf[68:72]))
		suite.Equal(uint32(65535), binary.LittleEndian.Uint32(buf[72:76]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[76:80]))
	})
}

func (suite *gpuTypesTest) TestComputeBoundingRadius() {
	suite.Run("empty slice returns zero", func() {
		r := model.ComputeBoundingRadius(nil)
		suite.InDelta(0.0, r, 1e-6)
	})

	suite.Run("single vertex at origin returns zero", func() {
		verts := []model.GPUSkinnedVertex{{}}
		r := model.ComputeBoundingRadius(verts)
		suite.InDelta(0.0, r, 1e-6)
	})

	suite.Run("single vertex on x axis returns distance", func() {
		verts := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{3.0, 0.0, 0.0}}},
		}
		r := model.ComputeBoundingRadius(verts)
		suite.InDelta(3.0, r, 1e-6)
	})

	suite.Run("returns max distance from origin across all vertices", func() {
		verts := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{1.0, 0.0, 0.0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{0.0, 5.0, 0.0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{0.0, 0.0, 2.0}}},
		}
		r := model.ComputeBoundingRadius(verts)
		suite.InDelta(5.0, r, 1e-6)
	})

	suite.Run("negative coordinates produce correct distance", func() {
		verts := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{-4.0, 0.0, 0.0}}},
		}
		r := model.ComputeBoundingRadius(verts)
		suite.InDelta(4.0, r, 1e-6)
	})

	suite.Run("diagonal vertex produces correct euclidean distance", func() {
		// sqrt(1^2 + 2^2 + 2^2) = sqrt(9) = 3
		verts := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{1.0, 2.0, 2.0}}},
		}
		r := model.ComputeBoundingRadius(verts)
		suite.InDelta(3.0, r, 1e-6)
	})

	suite.Run("empty position slice returns zero", func() {
		verts := []model.GPUSkinnedVertex{}
		r := model.ComputeBoundingRadius(verts)
		suite.InDelta(0.0, r, 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUModelDataSize() {
	suite.Run("returns 64 bytes", func() {
		d := &model.GPUModelData{}
		suite.Equal(64, d.Size())
	})
}

func (suite *gpuTypesTest) TestGPUModelDataMarshal() {
	suite.Run("zero value produces 64-byte buffer", func() {
		d := &model.GPUModelData{}
		buf := d.Marshal()
		suite.Len(buf, 64)
	})

	suite.Run("identity matrix encodes correctly", func() {
		d := &model.GPUModelData{
			Model: [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
		}
		buf := d.Marshal()
		// diagonal elements = 1.0
		suite.InDelta(1.0, readFloat32(buf, 0), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 20), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 40), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 60), 1e-6)
		// off-diagonal elements = 0.0
		suite.InDelta(0.0, readFloat32(buf, 4), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 8), 1e-6)
		suite.InDelta(0.0, readFloat32(buf, 12), 1e-6)
	})

	suite.Run("translation matrix round-trips correctly", func() {
		// column-major translation matrix: translate (10, 20, 30)
		d := &model.GPUModelData{
			Model: [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1},
		}
		buf := d.Marshal()
		suite.InDelta(10.0, readFloat32(buf, 48), 1e-6)
		suite.InDelta(20.0, readFloat32(buf, 52), 1e-6)
		suite.InDelta(30.0, readFloat32(buf, 56), 1e-6)
		suite.InDelta(1.0, readFloat32(buf, 60), 1e-6)
	})

	suite.Run("all elements serialize in order", func() {
		var m [16]float32
		for i := range m {
			m[i] = float32(i + 1)
		}
		d := &model.GPUModelData{Model: m}
		buf := d.Marshal()

		for i := 0; i < 16; i++ {
			suite.InDelta(float64(i+1), readFloat32(buf, i*4), 1e-6)
		}
	})
}

func (suite *gpuTypesTest) TestGPUVertexSourceEmbed() {
	suite.Run("vertex source is non-empty", func() {
		suite.NotEmpty(model.GPUVertexSource)
	})
}

func (suite *gpuTypesTest) TestGPUSkinnedVertexSourceEmbed() {
	suite.Run("skinned vertex source is non-empty", func() {
		suite.NotEmpty(model.GPUSkinnedVertexSource)
	})
}

func (suite *gpuTypesTest) TestGPUModelDataSourceEmbed() {
	suite.Run("model data source is non-empty", func() {
		suite.NotEmpty(model.GPUModelDataSource)
	})
}

// readFloat32 reads a little-endian float32 from buf at the given byte offset.
func readFloat32(buf []byte, offset int) float64 {
	bits := binary.LittleEndian.Uint32(buf[offset : offset+4])
	return float64(math.Float32frombits(bits))
}
