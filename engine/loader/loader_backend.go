package loader

import (
	"io"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// loaderBackend defines the generic interface for loading models from files or streams.
// Concrete implementations (e.g., gltfLoaderBackend) handle format-specific details.
type loaderBackend interface {
	// Load performs a full model import from the given file path.
	// This extracts meshes, skeleton, animations, and materials.
	//
	// Parameters:
	//   - path: the file path to load
	//
	// Returns:
	//   - *model.ImportedModel: the imported model data
	//   - error: error if loading fails
	Load(path string) (*model.ImportedModel, error)

	// LoadMeshOnly imports only mesh and material data from the given file path.
	// Skeleton and animation extraction is skipped for faster loading of static models.
	//
	// Parameters:
	//   - path: the file path to load
	//
	// Returns:
	//   - *model.ImportedModel: the imported model with meshes and materials only
	//   - error: error if loading fails
	LoadMeshOnly(path string) (*model.ImportedModel, error)

	// LoadReader imports a model from a reader stream.
	//
	// Parameters:
	//   - r: the reader providing model data
	//   - isGLB: true if the reader provides GLB binary data, false for text-based formats
	//
	// Returns:
	//   - *model.ImportedModel: the imported model data
	//   - error: error if loading fails
	LoadReader(r io.Reader, isGLB bool) (*model.ImportedModel, error)
}
