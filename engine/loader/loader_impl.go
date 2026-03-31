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

		if skinned {
			allVertexBytes = append(allVertexBytes, common.SliceToBytes(mesh.Vertices)...)
		} else {
			gpuVerts := make([]model.GPUVertex, len(mesh.Vertices))
			for i, v := range mesh.Vertices {
				gpuVerts[i] = v.GPUVertex
			}
			allVertexBytes = append(allVertexBytes, common.SliceToBytes(gpuVerts)...)
		}

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
	boundingMin, boundingMax := model.ComputeBoundingAABB(allVertices)

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
		model.WithBoundingMin(boundingMin),
		model.WithBoundingMax(boundingMax),
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

func (l *loader) LoadAll(path string) ([]model.Model, error) {
	backend, err := l.resolveBackend(path)
	if err != nil {
		return nil, err
	}

	imported, err := backend.Load(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load %s: %w", path, err)
	}

	return l.importedToModels(imported)
}

// importedToModels splits an ImportedModel into one model.Model per unique material
// index. Each returned model contains only the primitives for that material, merged
// into a single vertex+index buffer, with exactly one render material at index 0.
//
// Parameters:
//   - imported: the CPU-side ImportedModel containing mesh, skeleton, animation, and material data
//
// Returns:
//   - []model.Model: one model per unique material index, in order of first appearance
//   - error: error if conversion fails
func (l *loader) importedToModels(imported *model.ImportedModel) ([]model.Model, error) {
	skinned := imported.Skeleton != nil && len(imported.Skeleton.Bones) > 0

	// Group meshes by their MaterialIndex, preserving first-seen order.
	type group struct {
		materialIndex int
		meshes        []model.ImportedMesh
	}
	groupMap := map[int]*group{}
	var groupOrder []int
	for _, mesh := range imported.Meshes {
		mi := mesh.MaterialIndex
		if _, ok := groupMap[mi]; !ok {
			groupMap[mi] = &group{materialIndex: mi}
			groupOrder = append(groupOrder, mi)
		}
		groupMap[mi].meshes = append(groupMap[mi].meshes, mesh)
	}

	result := make([]model.Model, 0, len(groupOrder))
	for _, mi := range groupOrder {
		grp := groupMap[mi]

		var allVertexBytes []byte
		var allIndexBytes []byte
		var allVertices []model.GPUSkinnedVertex
		totalIndices := 0
		indexOffset := uint32(0)

		for _, mesh := range grp.meshes {
			allVertices = append(allVertices, mesh.Vertices...)

			if skinned {
				allVertexBytes = append(allVertexBytes, common.SliceToBytes(mesh.Vertices)...)
			} else {
				gpuVerts := make([]model.GPUVertex, len(mesh.Vertices))
				for i, v := range mesh.Vertices {
					gpuVerts[i] = v.GPUVertex
				}
				allVertexBytes = append(allVertexBytes, common.SliceToBytes(gpuVerts)...)
			}

			adjusted := make([]uint32, len(mesh.Indices))
			for i, idx := range mesh.Indices {
				adjusted[i] = idx + indexOffset
			}
			allIndexBytes = append(allIndexBytes, common.SliceToBytes(adjusted)...)
			totalIndices += len(mesh.Indices)
			indexOffset += uint32(len(mesh.Vertices))
		}

		boundingRadius := model.ComputeBoundingRadius(allVertices)
		var boundingMin, boundingMax [3]float32
		if len(grp.meshes) > 0 {
			boundingMin = grp.meshes[0].BoundingMin
			boundingMax = grp.meshes[0].BoundingMax
			for _, mesh := range grp.meshes[1:] {
				for i := range 3 {
					if mesh.BoundingMin[i] < boundingMin[i] {
						boundingMin[i] = mesh.BoundingMin[i]
					}
					if mesh.BoundingMax[i] > boundingMax[i] {
						boundingMax[i] = mesh.BoundingMax[i]
					}
				}
			}
		}
		provider := bind_group_provider.NewBindGroupProvider(
			fmt.Sprintf("%s_mat_%d_mesh", imported.Name, mi),
		)

		var imp common.ImportedMaterial
		if mi < len(imported.Materials) {
			imp = imported.Materials[mi]
		}
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

		mdl := model.NewModel(
			model.WithName(imported.Name),
			model.WithSkinned(skinned),
			model.WithVertexData(allVertexBytes),
			model.WithIndexData(allIndexBytes),
			model.WithIndexCount(totalIndices),
			model.WithSkeleton(imported.Skeleton),
			model.WithAnimations(imported.Animations),
			model.WithImportedMaterials([]common.ImportedMaterial{imp}),
			model.WithMeshProvider(provider),
			model.WithBoundingRadius(boundingRadius),
			model.WithBoundingMin(boundingMin),
			model.WithBoundingMax(boundingMax),
		)
		mdl.SetRenderMaterials([]material.Material{mat})

		result = append(result, mdl)
	}

	return result, nil
}
