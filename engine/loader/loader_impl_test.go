package loader

import (
	"errors"
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

	suite.Run("skinned model with mesh vertices appends skinned byte buffer", func() {
		imported := &model.ImportedModel{
			Name: "skinned-mesh",
			Skeleton: &model.Skeleton{
				Bones: []model.Bone{{Name: "root"}},
			},
			Meshes: []model.ImportedMesh{
				{
					Vertices: []model.GPUSkinnedVertex{
						{GPUVertex: model.GPUVertex{Position: [3]float32{1, 2, 3}}},
					},
				},
			},
			Materials: nil,
		}
		result, err := suite.l.importedToModel(imported)
		suite.NoError(err)
		suite.True(result.Skinned())
		suite.Greater(len(result.VertexData()), 0)
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

	suite.Run("computes AABB from vertex positions", func() {
		imported := &model.ImportedModel{
			Name: "aabb",
			Meshes: []model.ImportedMesh{
				{
					Vertices: []model.GPUSkinnedVertex{
						{GPUVertex: model.GPUVertex{Position: [3]float32{-1, -2, -3}}},
						{GPUVertex: model.GPUVertex{Position: [3]float32{4, 5, 6}}},
					},
					Indices: []uint32{0, 1},
				},
			},
		}
		result, err := suite.l.importedToModel(imported)
		suite.NoError(err)
		suite.Equal([3]float32{-1, -2, -3}, result.BoundingMin())
		suite.Equal([3]float32{4, 5, 6}, result.BoundingMax())
	})

	suite.Run("generates LOD levels when total triangle count is at least 16", func() {
		verts, indices := makeLODGridVerts(5, 5, 0.2)
		imported := &model.ImportedModel{
			Name: "lod-mesh",
			Meshes: []model.ImportedMesh{
				{Vertices: verts, Indices: indices},
			},
		}
		result, err := suite.l.importedToModel(imported)
		suite.NoError(err)
		suite.Greater(result.LODCount(), 1)
	})

	suite.Run("skips LOD generation when total triangle count is less than 16", func() {
		// 3×3 grid → 8 triangles < 16; no LOD levels should be added
		verts, indices := makeLODGridVerts(3, 3, 0.2)
		imported := &model.ImportedModel{
			Name: "small-mesh",
			Meshes: []model.ImportedMesh{
				{Vertices: verts, Indices: indices},
			},
		}
		result, err := suite.l.importedToModel(imported)
		suite.NoError(err)
		suite.Equal(1, result.LODCount())
	})

	suite.Run("generates LOD1 for skinned mesh with sufficient triangles", func() {
		verts, indices := makeLODGridVerts(5, 5, 0.2)
		imported := &model.ImportedModel{
			Name: "skinned-lod1",
			Skeleton: &model.Skeleton{
				Bones: []model.Bone{{Name: "root"}},
			},
			Meshes: []model.ImportedMesh{
				{Vertices: verts, Indices: indices},
			},
		}
		result, err := suite.l.importedToModel(imported)
		suite.NoError(err)
		suite.Greater(result.LODCount(), 1)
	})

	suite.Run("generates LOD2 for skinned mesh with sufficient triangles", func() {
		verts, indices := makeLODGridVerts(5, 5, 0.2)
		imported := &model.ImportedModel{
			Name: "skinned-lod2",
			Skeleton: &model.Skeleton{
				Bones: []model.Bone{{Name: "root"}},
			},
			Meshes: []model.ImportedMesh{
				{Vertices: verts, Indices: indices},
			},
		}
		result, err := suite.l.importedToModel(imported)
		suite.NoError(err)
		suite.Equal(3, result.LODCount())
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

func (suite *loaderImplTest) TestLoadAll() {
	suite.Run("unsupported extension returns error", func() {
		_, err := suite.l.LoadAll("model.xyz")
		suite.Error(err)
	})

	suite.Run("backend load error returns error", func() {
		suite.mb.err = errors.New("backend error")
		_, err := suite.l.LoadAll("model.gltf")
		suite.Error(err)
	})

	suite.Run("single-material model returns one entry", func() {
		suite.mb.result = &model.ImportedModel{
			Name: "single",
			Meshes: []model.ImportedMesh{
				{Vertices: []model.GPUSkinnedVertex{{}}, Indices: []uint32{0}, MaterialIndex: 0},
			},
			Materials: []common.ImportedMaterial{{Name: "mat0"}},
		}
		result, err := suite.l.LoadAll("model.gltf")
		suite.NoError(err)
		suite.Len(result, 1)
	})

	suite.Run("multi-material model returns one entry per unique material", func() {
		suite.mb.result = &model.ImportedModel{
			Name: "multi",
			Meshes: []model.ImportedMesh{
				{Vertices: []model.GPUSkinnedVertex{{}}, Indices: []uint32{0}, MaterialIndex: 0},
				{Vertices: []model.GPUSkinnedVertex{{}}, Indices: []uint32{0}, MaterialIndex: 1},
			},
			Materials: []common.ImportedMaterial{{Name: "mat0"}, {Name: "mat1"}},
		}
		result, err := suite.l.LoadAll("model.gltf")
		suite.NoError(err)
		suite.Len(result, 2)
	})
}

func (suite *loaderImplTest) TestImportedToModels() {
	suite.Run("empty meshes returns empty slice", func() {
		result, err := suite.l.importedToModels(&model.ImportedModel{Name: "empty"})
		suite.NoError(err)
		suite.Len(result, 0)
	})

	suite.Run("single material group produces one model", func() {
		imported := &model.ImportedModel{
			Name: "one",
			Meshes: []model.ImportedMesh{
				{Vertices: []model.GPUSkinnedVertex{{}}, Indices: []uint32{0}, MaterialIndex: 0},
				{Vertices: []model.GPUSkinnedVertex{{}}, Indices: []uint32{0}, MaterialIndex: 0},
			},
			Materials: []common.ImportedMaterial{{Name: "mat0"}},
		}
		result, err := suite.l.importedToModels(imported)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.Equal(2, result[0].IndexCount())
	})

	suite.Run("two material groups produce two models", func() {
		imported := &model.ImportedModel{
			Name: "two",
			Meshes: []model.ImportedMesh{
				{Vertices: []model.GPUSkinnedVertex{{}}, Indices: []uint32{0}, MaterialIndex: 0},
				{Vertices: []model.GPUSkinnedVertex{{}}, Indices: []uint32{0}, MaterialIndex: 1},
			},
			Materials: []common.ImportedMaterial{{Name: "mat0"}, {Name: "mat1"}},
		}
		result, err := suite.l.importedToModels(imported)
		suite.NoError(err)
		suite.Len(result, 2)
	})

	suite.Run("skinned model when skeleton has bones", func() {
		imported := &model.ImportedModel{
			Name: "skinned",
			Meshes: []model.ImportedMesh{
				{Vertices: []model.GPUSkinnedVertex{{}}, Indices: []uint32{0}, MaterialIndex: 0},
			},
			Skeleton: &model.Skeleton{Bones: []model.Bone{{Name: "root"}}},
		}
		result, err := suite.l.importedToModels(imported)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.True(result[0].Skinned())
	})

	suite.Run("non-skinned model when no skeleton", func() {
		imported := &model.ImportedModel{
			Name: "static",
			Meshes: []model.ImportedMesh{
				{Vertices: []model.GPUSkinnedVertex{{}}, Indices: []uint32{0}, MaterialIndex: 0},
			},
		}
		result, err := suite.l.importedToModels(imported)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.False(result[0].Skinned())
	})

	suite.Run("material index out of bounds uses empty material without panic", func() {
		imported := &model.ImportedModel{
			Name: "oob",
			Meshes: []model.ImportedMesh{
				{Vertices: []model.GPUSkinnedVertex{{}}, Indices: []uint32{0}, MaterialIndex: 5},
			},
			Materials: []common.ImportedMaterial{},
		}
		suite.NotPanics(func() {
			result, err := suite.l.importedToModels(imported)
			suite.NoError(err)
			suite.Len(result, 1)
		})
	})

	suite.Run("unions BoundingMin and BoundingMax from per-mesh fields", func() {
		imported := &model.ImportedModel{
			Name: "union",
			Meshes: []model.ImportedMesh{
				{
					Vertices:      []model.GPUSkinnedVertex{{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 0}}}},
					Indices:       []uint32{0},
					MaterialIndex: 0,
					BoundingMin:   [3]float32{-1, -2, -3},
					BoundingMax:   [3]float32{2, 3, 4},
				},
				{
					Vertices:      []model.GPUSkinnedVertex{{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 0}}}},
					Indices:       []uint32{0},
					MaterialIndex: 0,
					BoundingMin:   [3]float32{-5, 0, -1},
					BoundingMax:   [3]float32{1, 6, 2},
				},
			},
			Materials: []common.ImportedMaterial{{Name: "mat0"}},
		}
		result, err := suite.l.importedToModels(imported)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.Equal([3]float32{-5, -2, -3}, result[0].BoundingMin())
		suite.Equal([3]float32{2, 6, 4}, result[0].BoundingMax())
	})

	suite.Run("BoundingMin and BoundingMax are zero when mesh AABB fields are zero", func() {
		imported := &model.ImportedModel{
			Name: "zero-aabb",
			Meshes: []model.ImportedMesh{
				{
					Vertices:      []model.GPUSkinnedVertex{{GPUVertex: model.GPUVertex{Position: [3]float32{1, 2, 3}}}},
					Indices:       []uint32{0},
					MaterialIndex: 0,
					BoundingMin:   [3]float32{},
					BoundingMax:   [3]float32{},
				},
			},
			Materials: []common.ImportedMaterial{{Name: "mat0"}},
		}
		result, err := suite.l.importedToModels(imported)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.Equal([3]float32{}, result[0].BoundingMin())
		suite.Equal([3]float32{}, result[0].BoundingMax())
	})

	suite.Run("generates LOD levels for non-skinned multi-triangle group", func() {
		verts, indices := makeLODGridVerts(5, 5, 0.2)
		imported := &model.ImportedModel{
			Name: "lod-group",
			Meshes: []model.ImportedMesh{
				{Vertices: verts, Indices: indices, MaterialIndex: 0},
			},
			Materials: []common.ImportedMaterial{{Name: "mat0"}},
		}
		result, err := suite.l.importedToModels(imported)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.Greater(result[0].LODCount(), 1)
	})

	suite.Run("generates LOD levels for skinned multi-triangle group", func() {
		verts, indices := makeLODGridVerts(5, 5, 0.2)
		imported := &model.ImportedModel{
			Name: "skinned-lod-group",
			Skeleton: &model.Skeleton{
				Bones: []model.Bone{{Name: "root"}},
			},
			Meshes: []model.ImportedMesh{
				{Vertices: verts, Indices: indices, MaterialIndex: 0},
			},
			Materials: []common.ImportedMaterial{{Name: "mat0"}},
		}
		result, err := suite.l.importedToModels(imported)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.Greater(result[0].LODCount(), 1)
	})

	suite.Run("skips LOD generation in importedToModels when triangle count is less than 16", func() {
		verts, indices := makeLODGridVerts(3, 3, 0.2)
		imported := &model.ImportedModel{
			Name: "small-lod-group",
			Meshes: []model.ImportedMesh{
				{Vertices: verts, Indices: indices, MaterialIndex: 0},
			},
			Materials: []common.ImportedMaterial{{Name: "mat0"}},
		}
		result, err := suite.l.importedToModels(imported)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.Equal(1, result[0].LODCount())
	})
}

type mockLoaderBackend struct {
	result *model.ImportedModel
	err    error
}

func (m *mockLoaderBackend) Load(_ string) (*model.ImportedModel, error) {
	return m.result, m.err
}

// makeLODGridVerts returns a rows×cols grid of GPUSkinnedVertex and triangle indices
// for use in LOD generation tests. Produces (rows-1)*(cols-1)*2 triangles.
func makeLODGridVerts(rows, cols int, spacing float32) ([]model.GPUSkinnedVertex, []uint32) {
	verts := make([]model.GPUSkinnedVertex, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			verts[r*cols+c].Position = [3]float32{float32(c) * spacing, float32(r) * spacing, 0}
		}
	}
	var indices []uint32
	for r := 0; r < rows-1; r++ {
		for c := 0; c < cols-1; c++ {
			tl := uint32(r*cols + c)
			tr := uint32(r*cols + c + 1)
			bl := uint32((r+1)*cols + c)
			br := uint32((r+1)*cols + c + 1)
			indices = append(indices, tl, tr, bl)
			indices = append(indices, tr, br, bl)
		}
	}
	return verts, indices
}
