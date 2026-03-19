// Package loader provides the [Loader] interface for importing, parsing, and
// caching 3D model files from disk.
//
// File-format details are abstracted behind an internal backend system selected
// by [LoaderBackendType]; the only supported backend is [BackendTypeGLTF]
// (glTF/GLB). Loaded models contain CPU-side data only — GPU resource
// initialization happens when a model is added to a scene. Instances are
// created with [NewLoader] using the option-builder pattern.
package loader

import (
	"fmt"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// LoaderBackendType identifies the model file format backend to use.
type LoaderBackendType int

const (
	// BackendTypeGLTF selects the glTF/GLB loader backend.
	BackendTypeGLTF LoaderBackendType = iota
)

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

func (l *loader) Models() map[string]model.Model { return l.modelCache }

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

	m, _ := l.importedToModel(imported)

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
