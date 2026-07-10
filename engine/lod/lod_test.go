package lod_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/lod"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

func TestRunLODTests(t *testing.T) {
	suite.Run(t, new(lodTest))
}

type lodTest struct {
	suite.Suite
}

// makeGridMesh returns a rows×cols vertex grid (spacing between adjacent vertices)
// and (rows-1)*(cols-1)*2 triangle indices. All Z values are 0.
func makeGridMesh(rows, cols int, spacing float32) ([]model.GPUSkinnedVertex, []uint32) {
	verts := make([]model.GPUSkinnedVertex, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			verts[r*cols+c].Position = [3]float32{float32(c) * spacing, float32(r) * spacing, 0}
		}
	}
	var indices []uint32
	for r := 0; r < rows-1; r++ {
		for c := 0; c < cols-1; c++ {
			tl := uint32(r*cols + c)
			tr := uint32(r*cols + c + 1)
			bl := uint32((r+1)*cols + c)
			br := uint32((r+1)*cols + c + 1)
			indices = append(indices, tl, tr, bl)
			indices = append(indices, tr, br, bl)
		}
	}
	return verts, indices
}

func (suite *lodTest) TestDecimate() {
	suite.Run("returns original vertices and indices when triangle count is less than 16", func() {
		// 3×3 grid → (3-1)*(3-1)*2 = 8 triangles < 16
		verts, indices := makeGridMesh(3, 3, 0.1)
		outV, outI := lod.Decimate(verts, indices, 0.5)
		suite.Equal(len(verts), len(outV))
		suite.Equal(len(indices), len(outI))
	})

	suite.Run("returns original when exactly 15 triangles", func() {
		// 4×4 grid → 18 triangles; truncate to 45 indices = 15 triangles
		verts, indices := makeGridMesh(4, 4, 0.1)
		indices = indices[:45]
		outV, outI := lod.Decimate(verts, indices, 0.5)
		suite.Equal(len(verts), len(outV))
		suite.Equal(45, len(outI))
	})

	suite.Run("reduces vertex count with aggressive clustering", func() {
		// 5×5 grid → 32 triangles; targetRatio=0.02 → gridRes=2 → 4 cells
		verts, indices := makeGridMesh(5, 5, 0.2)
		outV, _ := lod.Decimate(verts, indices, 0.02)
		suite.Less(len(outV), len(verts))
	})

	suite.Run("output index count is a multiple of 3", func() {
		verts, indices := makeGridMesh(5, 5, 0.2)
		_, outI := lod.Decimate(verts, indices, 0.5)
		suite.Equal(0, len(outI)%3)
	})

	suite.Run("output triangles contain no degenerate indices", func() {
		// With heavy clustering, cells merge vertices; degenerate triangles must be filtered
		verts, indices := makeGridMesh(5, 5, 0.2)
		_, outI := lod.Decimate(verts, indices, 0.02)
		for i := 0; i+2 < len(outI); i += 3 {
			a, b, c := outI[i], outI[i+1], outI[i+2]
			suite.NotEqual(a, b)
			suite.NotEqual(a, c)
			suite.NotEqual(b, c)
		}
	})

	suite.Run("gridRes minimum of 2 is enforced for very low target ratio", func() {
		// targetRatio=0.001 → int(0.001*100)=0 → clamped to 2; must not panic
		verts, indices := makeGridMesh(5, 5, 0.2)
		suite.NotPanics(func() {
			outV, outI := lod.Decimate(verts, indices, 0.001)
			suite.NotNil(outV)
			suite.Equal(0, len(outI)%3)
		})
	})

	suite.Run("handles flat mesh where z delta is zero without panic", func() {
		// All vertices have z=0 → delta[2]=0 → algorithm replaces with 1.0
		verts, indices := makeGridMesh(5, 5, 0.2)
		suite.NotPanics(func() {
			lod.Decimate(verts, indices, 0.5)
		})
	})

	suite.Run("representative vertex position is within source bounding box", func() {
		verts, indices := makeGridMesh(5, 5, 0.2)
		outV, _ := lod.Decimate(verts, indices, 0.02)
		const maxCoord = float32(0.8)
		for _, v := range outV {
			suite.GreaterOrEqual(v.Position[0], float32(0))
			suite.LessOrEqual(v.Position[0], maxCoord+1e-5)
			suite.GreaterOrEqual(v.Position[1], float32(0))
			suite.LessOrEqual(v.Position[1], maxCoord+1e-5)
		}
	})

	suite.Run("nearest-to-centroid bone data is preserved in clustered cell", func() {
		// 5×5 grid with gridRes=2 merges vertices into 4 cells.
		// Vertex 0 is the first processed in cell (cx=0,cy=0) and has distSq=0
		// from the running centroid at that moment → its bone data is retained.
		verts, indices := makeGridMesh(5, 5, 0.2)
		verts[0].BoneIndices = [4]uint32{77, 0, 0, 0}
		verts[0].BoneWeights = [4]float32{1, 0, 0, 0}

		outV, _ := lod.Decimate(verts, indices, 0.02)

		var found bool
		for _, v := range outV {
			if v.BoneIndices[0] == 77 {
				found = true
				break
			}
		}
		suite.True(found)
	})

	suite.Run("normals are averaged when vertex normals are non-zero", func() {
		// Build a 5×5 grid mesh (32 triangles). Set all vertex normals to (0,1,0) so
		// normalizeF64 receives a non-zero accumulated sum and takes the normal return path.
		verts, indices := makeGridMesh(5, 5, 0.2)
		for i := range verts {
			verts[i].Normal = [3]float32{0, 1, 0}
		}
		outV, outI := lod.Decimate(verts, indices, 0.02)
		suite.NotEmpty(outV)
		suite.Equal(0, len(outI)%3)
		// All output normals should be unit-length (roughly Y-up).
		for _, v := range outV {
			suite.InDelta(float32(1.0), v.Normal[1], 0.01)
		}
	})
}
