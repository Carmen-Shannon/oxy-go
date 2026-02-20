package loader

import (
	"io"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// gltfLoaderBackendImpl is the implementation of gltfLoaderBackend.
type gltfLoaderBackendImpl struct {
	importer gltfImporter
}

// gltfLoaderBackend is a loaderBackend implementation for glTF/GLB files.
// It delegates to the gltfImporter for parsing and extraction.
type gltfLoaderBackend interface {
	loaderBackend
}

var _ gltfLoaderBackend = &gltfLoaderBackendImpl{}

// newGLTFLoaderBackend creates a new glTF loader backend.
//
// Returns:
//   - gltfLoaderBackend: the loader backend for glTF/GLB files
func newGLTFLoaderBackend() gltfLoaderBackend {
	return &gltfLoaderBackendImpl{
		importer: newGLTFImporter(),
	}
}

func (b *gltfLoaderBackendImpl) Load(path string) (*model.ImportedModel, error) {
	return b.importer.Import(path)
}

func (b *gltfLoaderBackendImpl) LoadMeshOnly(path string) (*model.ImportedModel, error) {
	return b.importer.ImportMeshOnly(path)
}

func (b *gltfLoaderBackendImpl) LoadReader(r io.Reader, isGLB bool) (*model.ImportedModel, error) {
	return b.importer.ImportReader(r, isGLB)
}
