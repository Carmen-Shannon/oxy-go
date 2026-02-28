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

// LoaderBackendType identifies the model file format backend to use.
type LoaderBackendType int

const (
	// BackendTypeGLTF selects the glTF/GLB loader backend.
	BackendTypeGLTF LoaderBackendType = iota
)

// loader is the implementation of the Loader interface.
type loader struct {
	common.DelegateImpl[Loader]

	mu sync.RWMutex

	modelCache map[string]model.Model

	backend loaderBackend
}

// Loader defines the public-facing interface for loading and caching 3D models.
// It abstracts the file format (glTF, GLB, etc.) behind a generic backend and
// manages a cache of previously loaded models.
type Loader interface {
	common.Delegate[Loader]

	// Load imports a model file and caches the result.
	// If the model is already cached (by file path), the cached version is returned.
	// The backend is selected based on the file extension (.gltf/.glb → glTF backend).
	// The returned model contains CPU-side data only; GPU resources (mesh buffers,
	// material textures/samplers/bind groups) are initialized by the Scene when
	// the model is added via Add().
	//
	// Parameters:
	//   - path: the file path to the model file
	//
	// Returns:
	//   - model.Model: the loaded and cached model
	//   - error: error if loading fails
	Load(path string) (model.Model, error)

	// Get retrieves a cached model by name. Returns nil if not found.
	//
	// Parameters:
	//   - name: the cache key to look up
	//
	// Returns:
	//   - model.Model: the cached model or nil
	Get(name string) model.Model

	// Models returns the full model cache.
	//
	// Returns:
	//   - map[string]model.Model: all cached models keyed by name
	Models() map[string]model.Model
}

var _ Loader = &loader{}

// NewLoader creates a new Loader instance with the specified backend type and options applied.
//
// Parameters:
//   - backendType: the type of loader backend to use (e.g., BackendTypeGLTF)
//   - options: a variadic list of LoaderBuilderOption functions to configure the Loader
//
// Returns:
//   - Loader: a new instance of Loader configured with the provided backend and options
func NewLoader(backendType LoaderBackendType, options ...LoaderBuilderOption) Loader {
	l := &loader{
		mu:         sync.RWMutex{},
		modelCache: make(map[string]model.Model),
	}

	switch backendType {
	case BackendTypeGLTF:
		l.backend = newGLTFLoaderBackend()
	}

	for _, option := range options {
		option(l)
	}
	l.Delegate = l
	return l
}

func (l *loader) Load(path string) (model.Model, error) {
	l.mu.RLock()
	if cached, ok := l.modelCache[path]; ok {
		l.mu.RUnlock()
		return cached, nil
	}
	l.mu.RUnlock()

	backend, err := l.resolveBackend(path)
	if err != nil {
		return nil, err
	}

	imported, err := backend.Load(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load %s: %w", path, err)
	}

	m, err := l.importedToModel(imported)
	if err != nil {
		return nil, err
	}

	l.mu.Lock()
	l.modelCache[path] = m
	l.mu.Unlock()

	return m, nil
}

func (l *loader) Get(name string) model.Model {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.modelCache[name]
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

func (l *loader) Models() map[string]model.Model {
	l.mu.RLock()
	defer l.mu.RUnlock()

	result := make(map[string]model.Model, len(l.modelCache))
	for k, v := range l.modelCache {
		result[k] = v
	}
	return result
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
