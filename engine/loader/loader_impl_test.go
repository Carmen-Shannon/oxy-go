package loader

import (
	"sync"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/stretchr/testify/suite"
)

func TestRunLoaderImplTests(t *testing.T) {
	suite.Run(t, new(loaderImplTest))
}

type loaderImplTest struct {
	suite.Suite
	l  *loader
	mb *mockLoaderBackend
}

func (suite *loaderImplTest) SetupSubTest() {
	suite.mb = &mockLoaderBackend{}
	suite.l = &loader{
		mu:         sync.RWMutex{},
		modelCache: make(map[string]model.Model),
		backend:    suite.mb,
	}
}

func (suite *loaderImplTest) TestResolveBackend() {
	suite.Run(".gltf extension returns backend", func() {
		backend, err := suite.l.resolveBackend("model.gltf")
		suite.NoError(err)
		suite.NotNil(backend)
	})

	suite.Run(".glb extension returns backend", func() {
		backend, err := suite.l.resolveBackend("model.glb")
		suite.NoError(err)
		suite.NotNil(backend)
	})

	suite.Run("unsupported extension returns error", func() {
		backend, err := suite.l.resolveBackend("model.xyz")
		suite.Error(err)
		suite.Nil(backend)
	})
}

func (suite *loaderImplTest) TestImportedToModel() {
	suite.Run("returns non-nil model for empty imported model", func() {
		result, err := suite.l.importedToModel(&model.ImportedModel{Name: "empty"})
		suite.NoError(err)
		suite.NotNil(result)
	})

	suite.Run("static model when no skeleton", func() {
		result, err := suite.l.importedToModel(&model.ImportedModel{Name: "static"})
		suite.NoError(err)
		suite.False(result.Skinned())
	})

	suite.Run("skinned model when skeleton has bones", func() {
		result, err := suite.l.importedToModel(&model.ImportedModel{
			Name: "skinned",
			Skeleton: &model.Skeleton{
				Bones: []model.Bone{{Name: "root"}},
			},
		})
		suite.NoError(err)
		suite.True(result.Skinned())
	})

	suite.Run("combines multiple mesh vertices", func() {
		imported := &model.ImportedModel{
			Name: "multi",
			Meshes: []model.ImportedMesh{
				{
					Vertices: []model.GPUSkinnedVertex{{}, {}},
					Indices:  []uint32{0, 1},
				},
				{
					Vertices: []model.GPUSkinnedVertex{{}, {}},
					Indices:  []uint32{0, 1},
				},
			},
		}
		result, err := suite.l.importedToModel(imported)
		suite.NoError(err)
		suite.Equal(4, result.IndexCount())
	})

	suite.Run("adjusts index offset across meshes", func() {
		imported := &model.ImportedModel{
			Name: "offset",
			Meshes: []model.ImportedMesh{
				{
					Vertices: []model.GPUSkinnedVertex{{}, {}},
					Indices:  []uint32{0, 1},
				},
				{
					Vertices: []model.GPUSkinnedVertex{{}, {}},
					Indices:  []uint32{0, 1},
				},
			},
		}
		result, err := suite.l.importedToModel(imported)
		suite.NoError(err)
		suite.Equal(4, result.IndexCount())
	})

	suite.Run("creates render materials for each imported material", func() {
		imported := &model.ImportedModel{
			Name:      "with-mat",
			Materials: []common.ImportedMaterial{{Name: "mat0"}},
		}
		result, err := suite.l.importedToModel(imported)
		suite.NoError(err)
		suite.Len(result.RenderMaterials(), 1)
	})
}

func (suite *loaderImplTest) TestLoad() {
	suite.Run("caches and returns model when backend succeeds", func() {
		suite.mb.result = &model.ImportedModel{Name: "fox"}
		result, err := suite.l.Load("fox.gltf")
		suite.NoError(err)
		suite.NotNil(result)
		suite.NotNil(suite.l.modelCache["fox.gltf"])
	})
}

type mockLoaderBackend struct {
	result *model.ImportedModel
	err    error
}

func (m *mockLoaderBackend) Load(_ string) (*model.ImportedModel, error) {
	return m.result, m.err
}
