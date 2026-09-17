package loader

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

type mockGltfParserFull struct {
	mockGltfParser
	mat4Result  [][16]float32
	readMat4Err error
}

func (m *mockGltfParserFull) ReadMat4Accessor(_ int) ([][16]float32, error) {
	return m.mat4Result, m.readMat4Err
}

type mockGltfParserNilDocAfterTwo struct {
	mockGltfParser
	calls int
}

func (m *mockGltfParserNilDocAfterTwo) Document() *gltfDocument {
	m.calls++
	if m.calls > 2 {
		return nil
	}
	return m.doc
}

type gltfImporterImplTest struct {
	suite.Suite
	mockParser *mockGltfParser
	importer   *gltfImporterImpl
}

func TestGltfImporterImpl(t *testing.T) {
	suite.Run(t, new(gltfImporterImplTest))
}

func (suite *gltfImporterImplTest) SetupSubTest() {
	suite.mockParser = &mockGltfParser{}
	suite.importer = &gltfImporterImpl{}
}

func (suite *gltfImporterImplTest) TestImportFromParser() {
	suite.Run("nil document returns error", func() {
		suite.mockParser.doc = nil
		_, err := suite.importer.importFromParser(suite.mockParser, "test.gltf")
		suite.Error(err)
	})

	suite.Run("empty document returns model with fallback name", func() {
		suite.mockParser.doc = &gltfDocument{}
		result, err := suite.importer.importFromParser(suite.mockParser, "model.gltf")
		suite.NoError(err)
		suite.Equal("model.gltf", result.Name)
		suite.Empty(result.Meshes)
		suite.Nil(result.Skeleton)
		suite.Nil(result.Animations)
		suite.Nil(result.Materials)
	})

	suite.Run("empty fallback path returns unnamed_model", func() {
		suite.mockParser.doc = &gltfDocument{}
		result, err := suite.importer.importFromParser(suite.mockParser, "")
		suite.NoError(err)
		suite.Equal("unnamed_model", result.Name)
	})

	suite.Run("scene name is used as model name", func() {
		suite.mockParser.doc = &gltfDocument{
			Scene:  common.ToPtr(0),
			Scenes: []gltfScene{{Name: "FoxScene"}},
		}
		result, err := suite.importer.importFromParser(suite.mockParser, "fallback.gltf")
		suite.NoError(err)
		suite.Equal("FoxScene", result.Name)
	})

	suite.Run("mesh extraction error propagates", func() {
		suite.mockParser.readVec3Err = errors.New("fail")
		suite.mockParser.doc = &gltfDocument{
			Meshes: []gltfMesh{
				{Primitives: []gltfPrimitive{{Attributes: map[string]int{"POSITION": 0}}}},
			},
		}
		_, err := suite.importer.importFromParser(suite.mockParser, "test.gltf")
		suite.Error(err)
	})

	suite.Run("skeleton extraction error propagates", func() {
		fullParser := &mockGltfParserFull{
			mockGltfParser: mockGltfParser{
				doc: &gltfDocument{
					Skins: []gltfSkin{{InverseBindMatrices: common.ToPtr(0), Joints: []int{0}}},
					Nodes: []gltfNode{{Name: "bone0"}},
				},
			},
			readMat4Err: errors.New("fail"),
		}
		_, err := suite.importer.importFromParser(fullParser, "test.gltf")
		suite.Error(err)
	})

	suite.Run("animation extraction error propagates", func() {
		suite.mockParser.readScalarErr = errors.New("fail")
		suite.mockParser.doc = &gltfDocument{
			Skins: []gltfSkin{{Joints: []int{0}}},
			Nodes: []gltfNode{{Name: "bone0"}},
			Animations: []gltfAnimation{
				{
					Name: "Walk",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		_, err := suite.importer.importFromParser(suite.mockParser, "test.gltf")
		suite.Error(err)
	})

	suite.Run("animations without skeleton are extracted", func() {
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{Name: "Idle", Channels: []gltfAnimChannel{}, Samplers: []gltfAnimSampler{}},
			},
		}
		result, err := suite.importer.importFromParser(suite.mockParser, "test.gltf")
		suite.NoError(err)
		suite.Len(result.Animations, 1)
		suite.Equal("Idle", result.Animations[0].Name)
	})

	suite.Run("skin with mesh covers FindSkeletonForMesh branch", func() {
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		suite.mockParser.doc = &gltfDocument{
			Meshes: []gltfMesh{{Primitives: []gltfPrimitive{{Attributes: map[string]int{"POSITION": 0}}}}},
			Skins:  []gltfSkin{{Joints: []int{1}}},
			Nodes: []gltfNode{
				{Name: "meshNode", Mesh: common.ToPtr(0), Skin: common.ToPtr(0)},
				{Name: "bone0"},
			},
		}
		result, err := suite.importer.importFromParser(suite.mockParser, "test.gltf")
		suite.NoError(err)
		suite.NotNil(result.Skeleton)
		suite.Len(result.Meshes, 1)
	})

	suite.Run("animation with skeleton and mesh covers FindSkeletonForMesh branch", func() {
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		suite.mockParser.doc = &gltfDocument{
			Meshes: []gltfMesh{{Primitives: []gltfPrimitive{{Attributes: map[string]int{"POSITION": 0}}}}},
			Skins:  []gltfSkin{{Joints: []int{1}}},
			Nodes: []gltfNode{
				{Name: "meshNode", Mesh: common.ToPtr(0), Skin: common.ToPtr(0)},
				{Name: "bone0"},
			},
			Animations: []gltfAnimation{
				{Name: "Walk", Channels: []gltfAnimChannel{}, Samplers: []gltfAnimSampler{}},
			},
		}
		result, err := suite.importer.importFromParser(suite.mockParser, "test.gltf")
		suite.NoError(err)
		suite.NotNil(result.Skeleton)
	})

	suite.Run("materials are extracted successfully", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{Name: "mat1"}},
		}
		result, err := suite.importer.importFromParser(suite.mockParser, "test.gltf")
		suite.NoError(err)
		suite.Len(result.Materials, 1)
		suite.Equal("mat1", result.Materials[0].Name)
	})

	suite.Run("animation extraction no-skeleton error propagates", func() {
		p := &mockGltfParserNilDocAfterTwo{
			mockGltfParser: mockGltfParser{
				doc: &gltfDocument{
					Animations: []gltfAnimation{{Name: "Idle"}},
				},
			},
		}
		_, err := suite.importer.importFromParser(p, "test.gltf")
		suite.Error(err)
	})

	suite.Run("material extraction error propagates", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				Name: "mat1",
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					BaseColorTexture: &gltfTextureInfo{Index: 0},
				},
			}},
		}
		_, err := suite.importer.importFromParser(suite.mockParser, "test.gltf")
		suite.Error(err)
	})
}

func (suite *gltfImporterImplTest) TestGltfRemapMeshBoneIndices() {
	suite.Run("empty oldToNew is a no-op", func() {
		meshes := []model.ImportedMesh{
			{Vertices: []model.GPUSkinnedVertex{{BoneIndices: [4]uint32{0, 1, 2, 3}}}},
		}
		gltfRemapMeshBoneIndices(meshes, map[int32]int32{})
		suite.Equal([4]uint32{0, 1, 2, 3}, meshes[0].Vertices[0].BoneIndices)
	})

	suite.Run("single bone index is remapped", func() {
		meshes := []model.ImportedMesh{
			{Vertices: []model.GPUSkinnedVertex{{BoneIndices: [4]uint32{0, 0, 0, 0}}}},
		}
		gltfRemapMeshBoneIndices(meshes, map[int32]int32{0: 5})
		suite.Equal([4]uint32{5, 5, 5, 5}, meshes[0].Vertices[0].BoneIndices)
	})

	suite.Run("bone index not in map is unchanged", func() {
		meshes := []model.ImportedMesh{
			{Vertices: []model.GPUSkinnedVertex{{BoneIndices: [4]uint32{7, 7, 7, 7}}}},
		}
		gltfRemapMeshBoneIndices(meshes, map[int32]int32{0: 5})
		suite.Equal([4]uint32{7, 7, 7, 7}, meshes[0].Vertices[0].BoneIndices)
	})

	suite.Run("multiple meshes and vertices all remapped", func() {
		meshes := []model.ImportedMesh{
			{Vertices: []model.GPUSkinnedVertex{
				{BoneIndices: [4]uint32{0, 1, 0, 1}},
				{BoneIndices: [4]uint32{1, 0, 1, 0}},
			}},
			{Vertices: []model.GPUSkinnedVertex{
				{BoneIndices: [4]uint32{0, 0, 1, 1}},
			}},
		}
		gltfRemapMeshBoneIndices(meshes, map[int32]int32{0: 2, 1: 3})
		suite.Equal([4]uint32{2, 3, 2, 3}, meshes[0].Vertices[0].BoneIndices)
		suite.Equal([4]uint32{3, 2, 3, 2}, meshes[0].Vertices[1].BoneIndices)
		suite.Equal([4]uint32{2, 2, 3, 3}, meshes[1].Vertices[0].BoneIndices)
	})
}

func (suite *gltfImporterImplTest) TestGltfExtractModelName() {
	suite.Run("scene name is used when available", func() {
		doc := &gltfDocument{Scene: common.ToPtr(0), Scenes: []gltfScene{{Name: "MyScene"}}}
		suite.Equal("MyScene", gltfExtractModelName(doc, "fallback.gltf"))
	})

	suite.Run("nil scene pointer falls through to fallback", func() {
		doc := &gltfDocument{Scenes: []gltfScene{{Name: "MyScene"}}}
		suite.Equal("fallback.gltf", gltfExtractModelName(doc, "fallback.gltf"))
	})

	suite.Run("empty scene name falls through to fallback", func() {
		doc := &gltfDocument{Scene: common.ToPtr(0), Scenes: []gltfScene{{Name: ""}}}
		suite.Equal("fallback.gltf", gltfExtractModelName(doc, "fallback.gltf"))
	})

	suite.Run("scene index out of range falls through to fallback", func() {
		doc := &gltfDocument{Scene: common.ToPtr(5), Scenes: []gltfScene{{Name: "Only"}}}
		suite.Equal("fallback.gltf", gltfExtractModelName(doc, "fallback.gltf"))
	})

	suite.Run("non-empty fallback path is returned", func() {
		doc := &gltfDocument{}
		suite.Equal("model.gltf", gltfExtractModelName(doc, "model.gltf"))
	})

	suite.Run("empty fallback and no scene returns unnamed_model", func() {
		doc := &gltfDocument{}
		suite.Equal("unnamed_model", gltfExtractModelName(doc, ""))
	})
}

func (suite *gltfImporterImplTest) TestGltfFlattenMaterials() {
	suite.Run("nil input returns nil", func() {
		result := gltfFlattenMaterials(nil)
		suite.Nil(result)
	})

	suite.Run("non-nil pointers are correctly flattened", func() {
		m1 := &common.ImportedMaterial{Name: "mat1"}
		m2 := &common.ImportedMaterial{Name: "mat2"}
		result := gltfFlattenMaterials([]*common.ImportedMaterial{m1, m2})
		suite.Equal("mat1", result[0].Name)
		suite.Equal("mat2", result[1].Name)
	})

	suite.Run("nil pointer produces zero-value slot", func() {
		result := gltfFlattenMaterials([]*common.ImportedMaterial{nil})
		suite.Equal(common.ImportedMaterial{}, result[0])
	})

	suite.Run("empty non-nil slice returns empty non-nil slice", func() {
		result := gltfFlattenMaterials([]*common.ImportedMaterial{})
		suite.NotNil(result)
		suite.Len(result, 0)
	})
}

func (suite *gltfImporterImplTest) TestNewGLTFImporter() {
	suite.Run("returns non-nil importer", func() {
		suite.NotNil(newGLTFImporter())
	})
}

func (suite *gltfImporterImplTest) TestImport() {
	suite.Run("invalid path returns parse error", func() {
		_, err := suite.importer.Import("nonexistent_path_that_does_not_exist.gltf")
		suite.Error(err)
	})

	suite.Run("valid minimal gltf returns model without error", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "minimal.gltf")
		err := os.WriteFile(path, []byte(`{"asset":{"version":"2.0"}}`), 0644)
		suite.Require().NoError(err)
		result, err := suite.importer.Import(path)
		suite.NoError(err)
		suite.NotNil(result)
	})
}
