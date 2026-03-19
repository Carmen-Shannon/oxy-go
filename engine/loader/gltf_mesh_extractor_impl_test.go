package loader

import (
	"errors"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/stretchr/testify/suite"
)

// mockGltfParserMesh embeds mockGltfParser and adds error/result control for the
// methods that mockGltfParser returns nil/nil by default (Vec2, Joints, Indices, AccessorData).
type mockGltfParserMesh struct {
	mockGltfParser
	vec2Err            error
	jointsErr          error
	indicesErr         error
	accessorDataErr    error
	vec2Result         [][2]float32
	jointsResult       [][4]uint32
	indicesResult      []uint32
	accessorDataResult []byte
}

func (m *mockGltfParserMesh) ReadVec2Accessor(_ int) ([][2]float32, error) {
	return m.vec2Result, m.vec2Err
}

func (m *mockGltfParserMesh) ReadJointsAccessor(_ int) ([][4]uint32, error) {
	return m.jointsResult, m.jointsErr
}

func (m *mockGltfParserMesh) ReadIndicesAccessor(_ int) ([]uint32, error) {
	return m.indicesResult, m.indicesErr
}

func (m *mockGltfParserMesh) ReadAccessorData(_ int) ([]byte, error) {
	return m.accessorDataResult, m.accessorDataErr
}

// mockGltfParserVec3FailOnSecond returns vec3Result on the first call and an error on all
// subsequent calls. This is the only viable approach for testing NORMAL read failures
// because ReadVec3Accessor is called for both POSITION (must succeed) and NORMAL (must fail).
type mockGltfParserVec3FailOnSecond struct {
	mockGltfParser
	callCount int
}

func (m *mockGltfParserVec3FailOnSecond) ReadVec3Accessor(_ int) ([][3]float32, error) {
	m.callCount++
	if m.callCount > 1 {
		return nil, errors.New("vec3 read error")
	}
	return m.vec3Result, nil
}

type gltfMeshExtractorImplTest struct {
	suite.Suite
	mockParser *mockGltfParser
	extractor  gltfMeshExtractor
}

func (suite *gltfMeshExtractorImplTest) SetupSubTest() {
	suite.mockParser = &mockGltfParser{}
	suite.extractor = newGLTFMeshExtractor(suite.mockParser)
}

func TestGltfMeshExtractorImpl(t *testing.T) {
	suite.Run(t, new(gltfMeshExtractorImplTest))
}

func (suite *gltfMeshExtractorImplTest) TestNewGLTFMeshExtractor() {
	suite.Run("returns non-nil", func() {
		suite.NotNil(suite.extractor)
	})
}

func (suite *gltfMeshExtractorImplTest) TestExtractMesh() {
	suite.Run("nil document returns error", func() {
		suite.mockParser.doc = nil
		_, err := suite.extractor.ExtractMesh(0)
		suite.Error(err)
	})

	suite.Run("negative meshIndex returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Meshes: []gltfMesh{{Name: "fox", Primitives: []gltfPrimitive{{Attributes: map[string]int{"POSITION": 0}}}}},
		}
		_, err := suite.extractor.ExtractMesh(-1)
		suite.Error(err)
	})

	suite.Run("meshIndex out of range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Meshes: []gltfMesh{{Name: "fox"}},
		}
		_, err := suite.extractor.ExtractMesh(1)
		suite.Error(err)
	})

	suite.Run("extractPrimitive error propagates", func() {
		suite.mockParser.doc = &gltfDocument{
			Meshes: []gltfMesh{{
				Name:       "fox",
				Primitives: []gltfPrimitive{{Attributes: map[string]int{}}},
			}},
		}
		_, err := suite.extractor.ExtractMesh(0)
		suite.Error(err)
	})

	suite.Run("single primitive mesh returns one result", func() {
		suite.mockParser.doc = &gltfDocument{
			Meshes: []gltfMesh{{
				Name:       "fox",
				Primitives: []gltfPrimitive{{Attributes: map[string]int{"POSITION": 0}}},
			}},
			Accessors: []gltfAccessor{},
		}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		result, err := suite.extractor.ExtractMesh(0)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.Equal("fox", result[0].Name)
	})

	suite.Run("multi-primitive mesh returns multiple results", func() {
		suite.mockParser.doc = &gltfDocument{
			Meshes: []gltfMesh{{
				Name: "fox",
				Primitives: []gltfPrimitive{
					{Attributes: map[string]int{"POSITION": 0}},
					{Attributes: map[string]int{"POSITION": 0}},
				},
			}},
		}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		result, err := suite.extractor.ExtractMesh(0)
		suite.NoError(err)
		suite.Len(result, 2)
	})
}

func (suite *gltfMeshExtractorImplTest) TestExtractAllMeshes() {
	suite.Run("nil document returns error", func() {
		suite.mockParser.doc = nil
		_, err := suite.extractor.ExtractAllMeshes()
		suite.Error(err)
	})

	suite.Run("empty meshes returns empty slice", func() {
		suite.mockParser.doc = &gltfDocument{Meshes: nil}
		result, err := suite.extractor.ExtractAllMeshes()
		suite.NoError(err)
		suite.Len(result, 0)
	})

	suite.Run("error from ExtractMesh propagates", func() {
		suite.mockParser.doc = &gltfDocument{
			Meshes: []gltfMesh{{
				Primitives: []gltfPrimitive{{Attributes: map[string]int{}}},
			}},
		}
		_, err := suite.extractor.ExtractAllMeshes()
		suite.Error(err)
	})

	suite.Run("all meshes extracted and flattened", func() {
		suite.mockParser.doc = &gltfDocument{
			Meshes: []gltfMesh{
				{Name: "mesh0", Primitives: []gltfPrimitive{{Attributes: map[string]int{"POSITION": 0}}}},
				{Name: "mesh1", Primitives: []gltfPrimitive{{Attributes: map[string]int{"POSITION": 0}}}},
			},
		}
		suite.mockParser.vec3Result = [][3]float32{{1, 2, 3}}
		result, err := suite.extractor.ExtractAllMeshes()
		suite.NoError(err)
		suite.Len(result, 2)
		suite.Equal("mesh0", result[0].Name)
		suite.Equal("mesh1", result[1].Name)
	})
}

func (suite *gltfMeshExtractorImplTest) TestExtractPrimitive() {
	impl := func() *gltfMeshExtractorImpl {
		return suite.extractor.(*gltfMeshExtractorImpl)
	}

	suite.Run("unsupported mode returns error", func() {
		prim := &gltfPrimitive{
			Attributes: map[string]int{"POSITION": 0},
			Mode:       common.ToPtr(5),
		}
		_, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.Error(err)
	})

	suite.Run("missing POSITION attribute returns error", func() {
		suite.mockParser.doc = &gltfDocument{}
		prim := &gltfPrimitive{Attributes: map[string]int{}}
		_, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.Error(err)
	})

	suite.Run("ReadVec3Accessor error for positions returns error", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.readVec3Err = errors.New("vec3 err")
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0}}
		_, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.Error(err)
	})

	suite.Run("ReadVec3Accessor error for normals returns error", func() {
		mp := &mockGltfParserVec3FailOnSecond{
			mockGltfParser: mockGltfParser{
				doc:        &gltfDocument{},
				vec3Result: [][3]float32{{0, 0, 0}},
			},
		}
		extractor := newGLTFMeshExtractor(mp)
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "NORMAL": 1}}
		_, err := extractor.(*gltfMeshExtractorImpl).extractPrimitive(prim, "mesh", 0)
		suite.Error(err)
	})

	suite.Run("ReadVec2Accessor error for texcoords propagates", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc:        &gltfDocument{},
				vec3Result: [][3]float32{{0, 0, 0}},
			},
			vec2Err: errors.New("vec2 err"),
		}
		extractor := newGLTFMeshExtractor(mp)
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "TEXCOORD_0": 1}}
		_, err := extractor.(*gltfMeshExtractorImpl).extractPrimitive(prim, "mesh", 0)
		suite.Error(err)
	})

	suite.Run("readColorAccessor error for COLOR_0 propagates", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc: &gltfDocument{
					Accessors: []gltfAccessor{
						{},
						{Type: gltfAccessorTypeVec4, ComponentType: gltfComponentTypeFloat},
					},
				},
				vec3Result:  [][3]float32{{0, 0, 0}},
				readVec4Err: errors.New("vec4 err"),
			},
		}
		extractor := newGLTFMeshExtractor(mp)
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "COLOR_0": 1}}
		_, err := extractor.(*gltfMeshExtractorImpl).extractPrimitive(prim, "mesh", 0)
		suite.Error(err)
	})

	suite.Run("ReadVec4Accessor error for tangents propagates", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		suite.mockParser.readVec4Err = errors.New("vec4 err")
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "TANGENT": 1}}
		_, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.Error(err)
	})

	suite.Run("ReadJointsAccessor error propagates", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc:        &gltfDocument{},
				vec3Result: [][3]float32{{0, 0, 0}},
			},
			jointsErr: errors.New("joints err"),
		}
		extractor := newGLTFMeshExtractor(mp)
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "JOINTS_0": 1}}
		_, err := extractor.(*gltfMeshExtractorImpl).extractPrimitive(prim, "mesh", 0)
		suite.Error(err)
	})

	suite.Run("ReadVec4Accessor error for weights propagates", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		suite.mockParser.readVec4Err = errors.New("vec4 err")
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "WEIGHTS_0": 1}}
		_, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.Error(err)
	})

	suite.Run("ReadIndicesAccessor error propagates", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc:        &gltfDocument{},
				vec3Result: [][3]float32{{0, 0, 0}},
			},
			indicesErr: errors.New("indices err"),
		}
		extractor := newGLTFMeshExtractor(mp)
		prim := &gltfPrimitive{
			Attributes: map[string]int{"POSITION": 0},
			Indices:    common.ToPtr(0),
		}
		_, err := extractor.(*gltfMeshExtractorImpl).extractPrimitive(prim, "mesh", 0)
		suite.Error(err)
	})

	suite.Run("nil indices generates sequential indices", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0}}
		result, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		suite.Equal([]uint32{0, 1, 2}, result.Indices)
	})

	suite.Run("explicit material index is applied", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		prim := &gltfPrimitive{
			Attributes: map[string]int{"POSITION": 0},
			Material:   common.ToPtr(2),
		}
		result, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		suite.Equal(2, result.MaterialIndex)
	})

	suite.Run("empty mesh name uses mesh_N naming", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0}}
		result, err := impl().extractPrimitive(prim, "", 0)
		suite.NoError(err)
		suite.Equal("mesh_0", result.Name)
	})

	suite.Run("primIndex greater than zero appends prim suffix", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0}}
		result, err := impl().extractPrimitive(prim, "fox", 1)
		suite.NoError(err)
		suite.Equal("fox_prim1", result.Name)
	})

	suite.Run("NORMAL attribute writes normals to vertices", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 1, 0}}
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "NORMAL": 0}}
		result, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		suite.Equal([3]float32{0, 1, 0}, result.Vertices[0].Normal)
	})

	suite.Run("TEXCOORD_0 attribute writes texcoords to vertices", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc:        &gltfDocument{},
				vec3Result: [][3]float32{{0, 0, 0}},
			},
			vec2Result: [][2]float32{{0.5, 0.5}},
		}
		extractor := newGLTFMeshExtractor(mp)
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "TEXCOORD_0": 0}}
		result, err := extractor.(*gltfMeshExtractorImpl).extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		suite.Equal([2]float32{0.5, 0.5}, result.Vertices[0].TexCoord)
	})

	suite.Run("COLOR_0 attribute writes colors to vertices", func() {
		suite.mockParser.doc = &gltfDocument{
			Accessors: []gltfAccessor{
				{},
				{Type: gltfAccessorTypeVec4, ComponentType: gltfComponentTypeFloat},
			},
		}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		suite.mockParser.vec4Result = [][4]float32{{0.2, 0.4, 0.6, 1.0}}
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "COLOR_0": 1}}
		result, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		suite.InDelta(float32(0.2), result.Vertices[0].Color[0], 1e-5)
	})

	suite.Run("TANGENT attribute writes tangents to vertices", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		suite.mockParser.vec4Result = [][4]float32{{1, 0, 0, 1}}
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "TANGENT": 1}}
		result, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		suite.Equal([4]float32{1, 0, 0, 1}, result.Vertices[0].Tangent)
	})

	suite.Run("JOINTS_0 attribute writes joint indices to vertices", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc:        &gltfDocument{},
				vec3Result: [][3]float32{{0, 0, 0}},
			},
			jointsResult: [][4]uint32{{1, 2, 3, 4}},
		}
		extractor := newGLTFMeshExtractor(mp)
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "JOINTS_0": 0}}
		result, err := extractor.(*gltfMeshExtractorImpl).extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		suite.Equal([4]uint32{1, 2, 3, 4}, result.Vertices[0].BoneIndices)
	})

	suite.Run("WEIGHTS_0 attribute writes bone weights to vertices", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}}
		suite.mockParser.vec4Result = [][4]float32{{0.25, 0.25, 0.25, 0.25}}
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0, "WEIGHTS_0": 1}}
		result, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		suite.Equal([4]float32{0.25, 0.25, 0.25, 0.25}, result.Vertices[0].BoneWeights)
	})

	suite.Run("explicit Indices are read from parser", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc:        &gltfDocument{},
				vec3Result: [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
			},
			indicesResult: []uint32{2, 1, 0},
		}
		extractor := newGLTFMeshExtractor(mp)
		prim := &gltfPrimitive{
			Attributes: map[string]int{"POSITION": 0},
			Indices:    common.ToPtr(0),
		}
		result, err := extractor.(*gltfMeshExtractorImpl).extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		suite.Equal([]uint32{2, 1, 0}, result.Indices)
	})

	suite.Run("normals generated when not in attributes", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0}}
		result, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		for _, v := range result.Vertices {
			zero := v.Normal[0] == 0 && v.Normal[1] == 0 && v.Normal[2] == 0
			suite.False(zero, "expected non-zero generated normal")
		}
	})

	suite.Run("tangents generated when not in attributes", func() {
		suite.mockParser.doc = &gltfDocument{}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
		prim := &gltfPrimitive{Attributes: map[string]int{"POSITION": 0}}
		result, err := impl().extractPrimitive(prim, "mesh", 0)
		suite.NoError(err)
		for _, v := range result.Vertices {
			zero := v.Tangent[0] == 0 && v.Tangent[1] == 0 && v.Tangent[2] == 0 && v.Tangent[3] == 0
			suite.False(zero, "expected non-zero generated tangent")
		}
	})
}

func (suite *gltfMeshExtractorImplTest) TestReadColorAccessor() {
	suite.Run("VEC4 FLOAT delegates to ReadVec4Accessor success", func() {
		suite.mockParser.doc = &gltfDocument{
			Accessors: []gltfAccessor{
				{Type: gltfAccessorTypeVec4, ComponentType: gltfComponentTypeFloat},
			},
		}
		suite.mockParser.vec4Result = [][4]float32{{0.5, 0.5, 0.5, 1.0}}
		result, err := suite.extractor.(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.NoError(err)
		suite.Equal([][4]float32{{0.5, 0.5, 0.5, 1.0}}, result)
	})

	suite.Run("VEC4 FLOAT ReadVec4Accessor error propagates", func() {
		suite.mockParser.doc = &gltfDocument{
			Accessors: []gltfAccessor{
				{Type: gltfAccessorTypeVec4, ComponentType: gltfComponentTypeFloat},
			},
		}
		suite.mockParser.readVec4Err = errors.New("vec4 err")
		_, err := suite.extractor.(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.Error(err)
	})

	suite.Run("VEC3 FLOAT delegates to ReadVec3Accessor success", func() {
		suite.mockParser.doc = &gltfDocument{
			Accessors: []gltfAccessor{
				{Type: gltfAccessorTypeVec3, ComponentType: gltfComponentTypeFloat},
			},
		}
		suite.mockParser.vec3Result = [][3]float32{{0.5, 0.5, 0.5}}
		result, err := suite.extractor.(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.InDelta(float32(0.5), result[0][0], 1e-6)
		suite.InDelta(float32(1.0), result[0][3], 1e-6)
	})

	suite.Run("VEC3 FLOAT ReadVec3Accessor error propagates", func() {
		suite.mockParser.doc = &gltfDocument{
			Accessors: []gltfAccessor{
				{Type: gltfAccessorTypeVec3, ComponentType: gltfComponentTypeFloat},
			},
		}
		suite.mockParser.readVec3Err = errors.New("vec3 err")
		_, err := suite.extractor.(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.Error(err)
	})

	suite.Run("UNSIGNED_BYTE VEC4 normalizes correctly", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc: &gltfDocument{
					Accessors: []gltfAccessor{
						{Type: gltfAccessorTypeVec4, ComponentType: gltfComponentTypeUnsignedByte, Count: 1},
					},
				},
			},
			accessorDataResult: []byte{255, 128, 0, 255},
		}
		result, err := newGLTFMeshExtractor(mp).(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.InDelta(float32(1.0), result[0][0], 1e-4)
		suite.InDelta(float32(128.0/255.0), result[0][1], 1e-4)
		suite.InDelta(float32(0.0), result[0][2], 1e-4)
		suite.InDelta(float32(1.0), result[0][3], 1e-4)
	})

	suite.Run("UNSIGNED_BYTE VEC3 normalizes correctly", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc: &gltfDocument{
					Accessors: []gltfAccessor{
						{Type: gltfAccessorTypeVec3, ComponentType: gltfComponentTypeUnsignedByte, Count: 1},
					},
				},
			},
			accessorDataResult: []byte{255, 0, 128},
		}
		result, err := newGLTFMeshExtractor(mp).(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.InDelta(float32(1.0), result[0][0], 1e-4)
		suite.InDelta(float32(0.0), result[0][1], 1e-4)
		suite.InDelta(float32(128.0/255.0), result[0][2], 1e-4)
		suite.InDelta(float32(1.0), result[0][3], 1e-4)
	})

	suite.Run("UNSIGNED_BYTE ReadAccessorData error propagates", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc: &gltfDocument{
					Accessors: []gltfAccessor{
						{Type: gltfAccessorTypeVec4, ComponentType: gltfComponentTypeUnsignedByte, Count: 1},
					},
				},
			},
			accessorDataErr: errors.New("read error"),
		}
		_, err := newGLTFMeshExtractor(mp).(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.Error(err)
	})

	suite.Run("UNSIGNED_SHORT VEC4 normalizes correctly", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc: &gltfDocument{
					Accessors: []gltfAccessor{
						{Type: gltfAccessorTypeVec4, ComponentType: gltfComponentTypeUnsignedShort, Count: 1},
					},
				},
			},
			// 0xFFFF LE = 65535 → 1.0, 0x0000 LE = 0 → 0.0
			accessorDataResult: []byte{0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF},
		}
		result, err := newGLTFMeshExtractor(mp).(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.InDelta(float32(1.0), result[0][0], 1e-4)
		suite.InDelta(float32(0.0), result[0][1], 1e-4)
		suite.InDelta(float32(0.0), result[0][2], 1e-4)
		suite.InDelta(float32(1.0), result[0][3], 1e-4)
	})

	suite.Run("UNSIGNED_SHORT VEC3 normalizes correctly", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc: &gltfDocument{
					Accessors: []gltfAccessor{
						{Type: gltfAccessorTypeVec3, ComponentType: gltfComponentTypeUnsignedShort, Count: 1},
					},
				},
			},
			// 0xFFFF LE, 0x0000 LE, 0x0000 LE → {1.0, 0.0, 0.0, 1.0}
			accessorDataResult: []byte{0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00},
		}
		result, err := newGLTFMeshExtractor(mp).(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.NoError(err)
		suite.Len(result, 1)
		suite.InDelta(float32(1.0), result[0][0], 1e-4)
		suite.InDelta(float32(0.0), result[0][1], 1e-4)
		suite.InDelta(float32(0.0), result[0][2], 1e-4)
		suite.InDelta(float32(1.0), result[0][3], 1e-4)
	})

	suite.Run("UNSIGNED_SHORT ReadAccessorData error propagates", func() {
		mp := &mockGltfParserMesh{
			mockGltfParser: mockGltfParser{
				doc: &gltfDocument{
					Accessors: []gltfAccessor{
						{Type: gltfAccessorTypeVec4, ComponentType: gltfComponentTypeUnsignedShort, Count: 1},
					},
				},
			},
			accessorDataErr: errors.New("read error"),
		}
		_, err := newGLTFMeshExtractor(mp).(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.Error(err)
	})

	suite.Run("unsupported format returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Accessors: []gltfAccessor{
				{Type: "SCALAR", ComponentType: 5120},
			},
		}
		_, err := suite.extractor.(*gltfMeshExtractorImpl).readColorAccessor(0)
		suite.Error(err)
	})
}

func (suite *gltfMeshExtractorImplTest) TestGltfCalculateBoundingBox() {
	suite.Run("empty positions returns zero bounds", func() {
		bmin, bmax := gltfCalculateBoundingBox(nil)
		suite.Equal([3]float32{}, bmin)
		suite.Equal([3]float32{}, bmax)
	})

	suite.Run("single position returns same min and max", func() {
		bmin, bmax := gltfCalculateBoundingBox([][3]float32{{1, 2, 3}})
		suite.Equal([3]float32{1, 2, 3}, bmin)
		suite.Equal([3]float32{1, 2, 3}, bmax)
	})

	suite.Run("multiple positions returns correct bounds", func() {
		bmin, bmax := gltfCalculateBoundingBox([][3]float32{
			{-1, 0, 2},
			{3, -2, 0},
			{0, 1, -3},
		})
		suite.Equal([3]float32{-1, -2, -3}, bmin)
		suite.Equal([3]float32{3, 1, 2}, bmax)
	})
}

func (suite *gltfMeshExtractorImplTest) TestGenerateNormals() {
	suite.Run("triangle in XY plane generates Z normals", func() {
		v := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{1, 0, 0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 1, 0}}},
		}
		idx := []uint32{0, 1, 2}
		generateNormals(v, idx)
		for i := range v {
			suite.InDelta(float32(0), v[i].Normal[0], 1e-5)
			suite.InDelta(float32(0), v[i].Normal[1], 1e-5)
			suite.InDelta(float32(1), v[i].Normal[2], 1e-5)
		}
	})

	suite.Run("degenerate triangle defaults to up vector", func() {
		v := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 0}}},
		}
		idx := []uint32{0, 1, 2}
		generateNormals(v, idx)
		for i := range v {
			suite.Equal([3]float32{0, 1, 0}, v[i].Normal)
		}
	})

	suite.Run("out of bounds index is skipped without panic", func() {
		v := []model.GPUSkinnedVertex{{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 0}}}}
		idx := []uint32{0, 1, 2}
		suite.NotPanics(func() { generateNormals(v, idx) })
		suite.Equal([3]float32{0, 1, 0}, v[0].Normal)
	})
}

func (suite *gltfMeshExtractorImplTest) TestGenerateTangents() {
	suite.Run("positive handedness tangent", func() {
		// Triangle in XY plane with UV (0,0),(1,0),(0,1) → T={1,0,0}, w=+1
		v := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 0}, Normal: [3]float32{0, 0, 1}, TexCoord: [2]float32{0, 0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{1, 0, 0}, Normal: [3]float32{0, 0, 1}, TexCoord: [2]float32{1, 0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 1, 0}, Normal: [3]float32{0, 0, 1}, TexCoord: [2]float32{0, 1}}},
		}
		idx := []uint32{0, 1, 2}
		generateTangents(v, idx)
		for i := range v {
			suite.InDelta(float32(1), v[i].Tangent[0], 1e-5)
			suite.InDelta(float32(0), v[i].Tangent[1], 1e-5)
			suite.InDelta(float32(0), v[i].Tangent[2], 1e-5)
			suite.InDelta(float32(1), v[i].Tangent[3], 1e-5)
		}
	})

	suite.Run("negative handedness tangent", func() {
		// Transposed UV: (0,0),(0,1),(1,0) → det=-1 → w=-1
		v := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 0}, Normal: [3]float32{0, 0, 1}, TexCoord: [2]float32{0, 0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{1, 0, 0}, Normal: [3]float32{0, 0, 1}, TexCoord: [2]float32{0, 1}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 1, 0}, Normal: [3]float32{0, 0, 1}, TexCoord: [2]float32{1, 0}}},
		}
		idx := []uint32{0, 1, 2}
		generateTangents(v, idx)
		for i := range v {
			suite.InDelta(float32(-1), v[i].Tangent[3], 1e-5)
		}
	})

	suite.Run("degenerate UV produces default tangent", func() {
		// All UVs identical → duv1=duv2={0,0} → det=0 → triangle skipped → default {1,0,0,1}
		v := []model.GPUSkinnedVertex{
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 0, 0}, Normal: [3]float32{0, 0, 1}, TexCoord: [2]float32{0, 0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{1, 0, 0}, Normal: [3]float32{0, 0, 1}, TexCoord: [2]float32{0, 0}}},
			{GPUVertex: model.GPUVertex{Position: [3]float32{0, 1, 0}, Normal: [3]float32{0, 0, 1}, TexCoord: [2]float32{0, 0}}},
		}
		idx := []uint32{0, 1, 2}
		generateTangents(v, idx)
		for i := range v {
			suite.Equal([4]float32{1, 0, 0, 1}, v[i].Tangent)
		}
	})

	suite.Run("out of bounds index is skipped without panic", func() {
		v := []model.GPUSkinnedVertex{{GPUVertex: model.GPUVertex{Normal: [3]float32{0, 0, 1}}}}
		idx := []uint32{0, 1, 2}
		suite.NotPanics(func() { generateTangents(v, idx) })
		suite.Equal([4]float32{1, 0, 0, 1}, v[0].Tangent)
	})
}
