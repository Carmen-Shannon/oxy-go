package loader

import (
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// loaderBackend defines the generic interface for loading models from files.
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
}
