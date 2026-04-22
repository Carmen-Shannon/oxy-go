// Package lod provides mesh Level-of-Detail generation via vertex clustering.
//
// [Decimate] reduces mesh triangle count by spatially clustering vertices into
// a uniform 3D grid and collapsing each cell into a single representative vertex.
// The algorithm runs at load time and produces independent vertex/index arrays
// suitable for GPU upload.
package lod

import "github.com/Carmen-Shannon/oxy-go/engine/model"

// Decimate reduces mesh complexity by clustering vertices into a uniform 3D grid
// and collapsing each cell into a single representative vertex.
//
// Parameters:
//   - vertices: source skinned vertex array
//   - indices: source triangle index array (must be a multiple of 3)
//   - targetRatio: fraction of detail to retain (0.0–1.0); lower values produce coarser meshes
//
// Returns:
//   - []model.GPUSkinnedVertex: reduced vertex array
//   - []uint32: reduced triangle index array
func Decimate(vertices []model.GPUSkinnedVertex, indices []uint32, targetRatio float32) ([]model.GPUSkinnedVertex, []uint32) {
	return decimate(vertices, indices, targetRatio)
}
