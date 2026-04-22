package lod

import (
	"math"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

type cellAccumulator struct {
	posSum         [3]float64
	normalSum      [3]float64
	texSum         [2]float64
	colorSum       [4]float64
	tangentSum     [4]float64
	tangentW       float32
	count          int
	nearestDistSq  float32
	nearestBones   [4]uint32
	nearestWeights [4]float32
	hasBoneData    bool
}

// decimate performs vertex-clustering LOD reduction on the given mesh.
//
// Parameters:
//   - vertices: source skinned vertex array
//   - indices: source triangle index array
//   - targetRatio: detail retention fraction (0.0–1.0)
//
// Returns:
//   - []model.GPUSkinnedVertex: reduced vertex array
//   - []uint32: reduced triangle index array
func decimate(vertices []model.GPUSkinnedVertex, indices []uint32, targetRatio float32) ([]model.GPUSkinnedVertex, []uint32) {
	triCount := len(indices) / 3
	if triCount < 16 {
		return vertices, indices
	}

	// Compute bounding box.
	vMin := [3]float32{float32(math.MaxFloat32), float32(math.MaxFloat32), float32(math.MaxFloat32)}
	vMax := [3]float32{-float32(math.MaxFloat32), -float32(math.MaxFloat32), -float32(math.MaxFloat32)}
	for i := range vertices {
		p := vertices[i].Position
		for a := 0; a < 3; a++ {
			if p[a] < vMin[a] {
				vMin[a] = p[a]
			}
			if p[a] > vMax[a] {
				vMax[a] = p[a]
			}
		}
	}

	// Grid resolution from target ratio.
	gridRes := int(targetRatio * 100)
	if gridRes < 2 {
		gridRes = 2
	}

	// Cell size per axis.
	var delta [3]float32
	for a := 0; a < 3; a++ {
		delta[a] = (vMax[a] - vMin[a]) / float32(gridRes)
		if delta[a] == 0 {
			delta[a] = 1.0
		}
	}

	// Map each vertex to a cell key and accumulate.
	vertexKeys := make([]int, len(vertices))
	cells := make(map[int]*cellAccumulator)

	for i := range vertices {
		v := &vertices[i]
		key := common.CellKey(v.Position, vMin, delta, gridRes)
		vertexKeys[i] = key

		acc, ok := cells[key]
		if !ok {
			acc = &cellAccumulator{
				nearestDistSq: math.MaxFloat32,
			}
			cells[key] = acc
		}

		acc.posSum[0] += float64(v.Position[0])
		acc.posSum[1] += float64(v.Position[1])
		acc.posSum[2] += float64(v.Position[2])
		acc.normalSum[0] += float64(v.Normal[0])
		acc.normalSum[1] += float64(v.Normal[1])
		acc.normalSum[2] += float64(v.Normal[2])
		acc.texSum[0] += float64(v.TexCoord[0])
		acc.texSum[1] += float64(v.TexCoord[1])
		acc.colorSum[0] += float64(v.Color[0])
		acc.colorSum[1] += float64(v.Color[1])
		acc.colorSum[2] += float64(v.Color[2])
		acc.colorSum[3] += float64(v.Color[3])
		acc.tangentSum[0] += float64(v.Tangent[0])
		acc.tangentSum[1] += float64(v.Tangent[1])
		acc.tangentSum[2] += float64(v.Tangent[2])
		acc.tangentSum[3] += float64(v.Tangent[3])

		if acc.count == 0 {
			acc.tangentW = v.Tangent[3]
		}

		acc.count++

		// Track nearest vertex to running centroid for bone data.
		cx := float32(acc.posSum[0] / float64(acc.count))
		cy := float32(acc.posSum[1] / float64(acc.count))
		cz := float32(acc.posSum[2] / float64(acc.count))
		dx := v.Position[0] - cx
		dy := v.Position[1] - cy
		dz := v.Position[2] - cz
		distSq := dx*dx + dy*dy + dz*dz
		if distSq < acc.nearestDistSq || !acc.hasBoneData {
			acc.nearestDistSq = distSq
			acc.nearestBones = v.BoneIndices
			acc.nearestWeights = v.BoneWeights
			acc.hasBoneData = true
		}
	}

	// Build representative vertices.
	cellToRep := make(map[int]uint32, len(cells))
	repVertices := make([]model.GPUSkinnedVertex, 0, len(cells))

	for key, acc := range cells {
		n := float64(acc.count)
		norm := common.NormalizeF64(acc.normalSum)

		var sv model.GPUSkinnedVertex
		sv.Position[0] = float32(acc.posSum[0] / n)
		sv.Position[1] = float32(acc.posSum[1] / n)
		sv.Position[2] = float32(acc.posSum[2] / n)
		sv.Normal = norm
		sv.TexCoord[0] = float32(acc.texSum[0] / n)
		sv.TexCoord[1] = float32(acc.texSum[1] / n)
		sv.Color[0] = float32(acc.colorSum[0] / n)
		sv.Color[1] = float32(acc.colorSum[1] / n)
		sv.Color[2] = float32(acc.colorSum[2] / n)
		sv.Color[3] = float32(acc.colorSum[3] / n)
		sv.Tangent[0] = float32(acc.tangentSum[0] / n)
		sv.Tangent[1] = float32(acc.tangentSum[1] / n)
		sv.Tangent[2] = float32(acc.tangentSum[2] / n)
		sv.Tangent[3] = acc.tangentW
		sv.BoneIndices = acc.nearestBones
		sv.BoneWeights = acc.nearestWeights

		cellToRep[key] = uint32(len(repVertices))
		repVertices = append(repVertices, sv)
	}

	// Remap indices and discard degenerate triangles.
	outIndices := make([]uint32, 0, len(indices))
	for i := 0; i+2 < len(indices); i += 3 {
		a := cellToRep[vertexKeys[indices[i]]]
		b := cellToRep[vertexKeys[indices[i+1]]]
		c := cellToRep[vertexKeys[indices[i+2]]]
		if a == b || a == c || b == c {
			continue
		}
		outIndices = append(outIndices, a, b, c)
	}

	return repVertices, outIndices
}
