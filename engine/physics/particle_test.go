package physics_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/stretchr/testify/suite"
)

func TestRunParticleTests(t *testing.T) {
	suite.Run(t, new(particleTest))
}

type particleTest struct {
	suite.Suite
}

func (suite *particleTest) TestVoxelizeMesh() {
	suite.Run("returns nil when vertex data is too short", func() {
		m := model.NewModel(
			model.WithVertexData([]byte{0, 0, 0, 0}),
			model.WithIndexData(makeIdxBuf(0, 1, 2)),
		)
		suite.Nil(physics.VoxelizeMesh(m, 0.2, false))
	})

	suite.Run("returns nil when index data has fewer than 3 indices", func() {
		m := model.NewModel(
			model.WithVertexData(make([]byte, 64)),
			model.WithIndexData(makeIdxBuf(0, 1)),
		)
		suite.Nil(physics.VoxelizeMesh(m, 0.2, false))
	})

	suite.Run("returns nil for degenerate geometry", func() {
		m := model.NewModel(
			model.WithVertexData(make([]byte, 3*64)),
			model.WithIndexData(makeIdxBuf(0, 1, 2)),
		)
		suite.Nil(physics.VoxelizeMesh(m, 0.2, false))
	})

	suite.Run("returns particles for a closed mesh", func() {
		m := buildCubeModel(false, nil)
		result := physics.VoxelizeMesh(m, 0.2, false)
		suite.NotNil(result)
		suite.NotEmpty(result)
	})

	suite.Run("surfaceOnly returns no more particles than full fill", func() {
		m := buildCubeModel(false, nil)
		full := physics.VoxelizeMesh(m, 0.2, false)
		surface := physics.VoxelizeMesh(m, 0.2, true)
		suite.LessOrEqual(len(surface), len(full))
	})

	suite.Run("skips triangles with out-of-bounds vertex indices", func() {
		cube := buildCubeModel(false, nil)
		// Append one triangle whose indices exceed the vertex count (8 vertices).
		idxBuf := append(cube.IndexData(), makeIdxBuf(0, 8, 9)...)
		m := model.NewModel(
			model.WithVertexData(cube.VertexData()),
			model.WithIndexData(idxBuf),
		)
		result := physics.VoxelizeMesh(m, 0.2, false)
		suite.NotEmpty(result)
	})

	suite.Run("surfaceOnly prunes interior particles with smaller radius", func() {
		m := buildCubeModel(false, nil)
		full := physics.VoxelizeMesh(m, 0.1, false)
		surface := physics.VoxelizeMesh(m, 0.1, true)
		suite.NotEmpty(full)
		suite.Less(len(surface), len(full))
	})
}

func (suite *particleTest) TestAssignBoneIndices() {
	suite.Run("returns early when model is not skinned", func() {
		m := buildCubeModel(false, nil)
		p := physics.Particle{LocalPosition: [3]float32{0.1, 0.2, 0.3}}
		particles := []physics.Particle{p}
		physics.AssignBoneIndices(particles, m)
		suite.Equal(uint32(0), particles[0].BoneIndex)
		suite.InDelta(float64(p.LocalPosition[0]), float64(particles[0].LocalPosition[0]), 1e-6)
		suite.InDelta(float64(p.LocalPosition[1]), float64(particles[0].LocalPosition[1]), 1e-6)
		suite.InDelta(float64(p.LocalPosition[2]), float64(particles[0].LocalPosition[2]), 1e-6)
	})

	suite.Run("returns early when skeleton is nil", func() {
		m := model.NewModel(
			model.WithVertexData(make([]byte, 96)),
			model.WithSkinned(true),
		)
		p := physics.Particle{LocalPosition: [3]float32{0.1, 0.2, 0.3}}
		particles := []physics.Particle{p}
		physics.AssignBoneIndices(particles, m)
		suite.Equal(uint32(0), particles[0].BoneIndex)
		suite.InDelta(float64(p.LocalPosition[0]), float64(particles[0].LocalPosition[0]), 1e-6)
		suite.InDelta(float64(p.LocalPosition[1]), float64(particles[0].LocalPosition[1]), 1e-6)
		suite.InDelta(float64(p.LocalPosition[2]), float64(particles[0].LocalPosition[2]), 1e-6)
	})

	suite.Run("returns early when vertex data is empty", func() {
		skel := &model.Skeleton{
			Bones: []model.Bone{{
				Name:              "root",
				InverseBindMatrix: [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
			}},
		}
		m := model.NewModel(
			model.WithVertexData([]byte{}),
			model.WithSkinned(true),
			model.WithSkeleton(skel),
		)
		p := physics.Particle{LocalPosition: [3]float32{0.1, 0.2, 0.3}}
		particles := []physics.Particle{p}
		physics.AssignBoneIndices(particles, m)
		suite.Equal(uint32(0), particles[0].BoneIndex)
		suite.InDelta(float64(p.LocalPosition[0]), float64(particles[0].LocalPosition[0]), 1e-6)
		suite.InDelta(float64(p.LocalPosition[1]), float64(particles[0].LocalPosition[1]), 1e-6)
		suite.InDelta(float64(p.LocalPosition[2]), float64(particles[0].LocalPosition[2]), 1e-6)
	})

	suite.Run("assigns bone index 0 and preserves position under identity inverse bind matrix", func() {
		skel := &model.Skeleton{
			Bones: []model.Bone{{
				Name:              "root",
				InverseBindMatrix: [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
			}},
		}
		m := buildCubeModel(true, skel)
		particles := physics.VoxelizeMesh(m, 0.2, false)
		suite.NotEmpty(particles)
		if len(particles) == 0 {
			return
		}
		before := particles[0].LocalPosition
		physics.AssignBoneIndices(particles, m)
		suite.Equal(uint32(0), particles[0].BoneIndex)
		suite.InDelta(float64(before[0]), float64(particles[0].LocalPosition[0]), 1e-6)
		suite.InDelta(float64(before[1]), float64(particles[0].LocalPosition[1]), 1e-6)
		suite.InDelta(float64(before[2]), float64(particles[0].LocalPosition[2]), 1e-6)
	})
}

// buildCubeModel constructs a unit cube model with 8 vertices and 12 triangles.
// If skinned is true, vertices include bone index and weight data (all assigned to bone 0).
// If skel is non-nil, it is attached as the skeleton.
func buildCubeModel(skinned bool, skel *model.Skeleton) model.Model {
	positions := [8][3]float32{
		{-0.5, -0.5, -0.5},
		{0.5, -0.5, -0.5},
		{0.5, 0.5, -0.5},
		{-0.5, 0.5, -0.5},
		{-0.5, -0.5, 0.5},
		{0.5, -0.5, 0.5},
		{0.5, 0.5, 0.5},
		{-0.5, 0.5, 0.5},
	}

	stride := 64
	if skinned {
		stride = 96
	}
	vertBuf := make([]byte, 8*stride)
	for i, pos := range positions {
		off := i * stride
		binary.LittleEndian.PutUint32(vertBuf[off:], math.Float32bits(pos[0]))
		binary.LittleEndian.PutUint32(vertBuf[off+4:], math.Float32bits(pos[1]))
		binary.LittleEndian.PutUint32(vertBuf[off+8:], math.Float32bits(pos[2]))
		if skinned {
			// BoneIndices at offset 64 (4 × uint32), all referencing bone 0.
			binary.LittleEndian.PutUint32(vertBuf[off+64:], 0)
			binary.LittleEndian.PutUint32(vertBuf[off+68:], 0)
			binary.LittleEndian.PutUint32(vertBuf[off+72:], 0)
			binary.LittleEndian.PutUint32(vertBuf[off+76:], 0)
			// BoneWeights at offset 80 (4 × float32), full weight on bone 0.
			binary.LittleEndian.PutUint32(vertBuf[off+80:], math.Float32bits(1.0))
			binary.LittleEndian.PutUint32(vertBuf[off+84:], math.Float32bits(0.0))
			binary.LittleEndian.PutUint32(vertBuf[off+88:], math.Float32bits(0.0))
			binary.LittleEndian.PutUint32(vertBuf[off+92:], math.Float32bits(0.0))
		}
	}

	// 12 triangles (CCW winding, outward-facing normals).
	triangles := [12][3]uint32{
		{0, 2, 1}, {0, 3, 2},
		{4, 5, 6}, {4, 6, 7},
		{0, 1, 5}, {0, 5, 4},
		{3, 6, 2}, {3, 7, 6},
		{0, 4, 7}, {0, 7, 3},
		{1, 2, 6}, {1, 6, 5},
	}
	idxBuf := make([]byte, 36*4)
	for i, tri := range triangles {
		binary.LittleEndian.PutUint32(idxBuf[(i*3)*4:], tri[0])
		binary.LittleEndian.PutUint32(idxBuf[(i*3+1)*4:], tri[1])
		binary.LittleEndian.PutUint32(idxBuf[(i*3+2)*4:], tri[2])
	}

	opts := []model.ModelBuilderOption{
		model.WithVertexData(vertBuf),
		model.WithIndexData(idxBuf),
		model.WithSkinned(skinned),
	}
	if skel != nil {
		opts = append(opts, model.WithSkeleton(skel))
	}
	return model.NewModel(opts...)
}

func makeIdxBuf(indices ...uint32) []byte {
	buf := make([]byte, len(indices)*4)
	for i, idx := range indices {
		binary.LittleEndian.PutUint32(buf[i*4:], idx)
	}
	return buf
}
