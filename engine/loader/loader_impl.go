package loader

import (
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
)

// loader is the implementation of the Loader interface.
type loader struct {
	common.DelegateImpl[Loader]

	mu sync.RWMutex

	modelCache map[string]model.Model

	backend loaderBackend
}

// resolveBackend selects an appropriate loader backend based on the file extension.
// Currently only glTF/GLB is supported.
func (l *loader) resolveBackend(path string) (loaderBackend, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".gltf", ".glb":
		return l.backend, nil
	default:
		return nil, fmt.Errorf("unsupported model format: %s", ext)
	}
}

// importedToModel converts an ImportedModel (CPU data) into a Model with CPU-side
// vertex/index buffers and materials prepared for later GPU initialization.
// No GPU resources are created here; the Scene handles all GPU init when
// the model is added via Add().
//
// Parameters:
//   - imported: the CPU-side ImportedModel containing mesh, skeleton, animation, and material data
//
// Returns:
//   - model.Model: the CPU-ready Model
//   - error: error if conversion fails
func (l *loader) importedToModel(imported *model.ImportedModel) (model.Model, error) {
	skinned := imported.Skeleton != nil && len(imported.Skeleton.Bones) > 0

	// Combine all meshes into one vertex + index buffer
	var allVertexBytes []byte
	var allIndexBytes []byte
	totalIndices := 0
	indexOffset := uint32(0)

	var allVertices []model.GPUSkinnedVertex
	for _, mesh := range imported.Meshes {
		allVertices = append(allVertices, mesh.Vertices...)
		allVertexBytes = append(allVertexBytes, common.SliceToBytes(mesh.Vertices)...)

		// Reindex: offset each index by the running vertex count across meshes
		adjusted := make([]uint32, len(mesh.Indices))
		for i, idx := range mesh.Indices {
			adjusted[i] = idx + indexOffset
		}
		allIndexBytes = append(allIndexBytes, common.SliceToBytes(adjusted)...)

		totalIndices += len(mesh.Indices)
		indexOffset += uint32(len(mesh.Vertices))
	}

	boundingRadius := model.ComputeBoundingRadius(allVertices)

	// Create BindGroupProvider for mesh data (GPU buffers created later by Scene)
	provider := bind_group_provider.NewBindGroupProvider(
		imported.Name + "_mesh",
	)

	mdl := model.NewModel(
		model.WithName(imported.Name),
		model.WithSkinned(skinned),
		model.WithVertexData(allVertexBytes),
		model.WithIndexData(allIndexBytes),
		model.WithIndexCount(totalIndices),
		model.WithSkeleton(imported.Skeleton),
		model.WithAnimations(imported.Animations),
		model.WithImportedMaterials(imported.Materials),
		model.WithMeshProvider(provider),
		model.WithBoundingRadius(boundingRadius),
	)

	// Convert imported materials into Material objects (CPU-only; GPU init deferred to Scene).
	renderMats := make([]material.Material, len(imported.Materials))
	for i, imp := range imported.Materials {
		mat := material.NewMaterial(
			material.WithName(imp.Name),
			material.WithBaseColor(imp.BaseColor),
			material.WithMetallic(imp.Metallic),
			material.WithRoughness(imp.Roughness),
			material.WithAlphaCutoff(imp.AlphaCutoff),
			material.WithDiffuseTexture(imp.DiffuseTexture),
			material.WithNormalTexture(imp.NormalTexture),
			material.WithMetallicRoughnessTexture(imp.MetallicRoughnessTexture),
			material.WithPipelineKey(imported.Name),
		)
		renderMats[i] = mat
	}
	mdl.SetRenderMaterials(renderMats)

	return mdl, nil
}
