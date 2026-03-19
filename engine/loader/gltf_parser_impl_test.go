package loader

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/suite"
)

type gltfParserImplTest struct {
	suite.Suite
	parser gltfParser
}

func (suite *gltfParserImplTest) SetupSubTest() {
	suite.parser = newGLTFParser()
}

func TestGltfParserImpl(t *testing.T) {
	suite.Run(t, new(gltfParserImplTest))
}

func (suite *gltfParserImplTest) impl() *gltfParserImpl {
	return suite.parser.(*gltfParserImpl)
}

func (suite *gltfParserImplTest) TestNewGLTFParser() {
	suite.Run("returns non-nil parser", func() {
		suite.NotNil(suite.parser)
	})
}

func (suite *gltfParserImplTest) TestDocument() {
	suite.Run("nil before Parse", func() {
		suite.Nil(suite.parser.Document())
	})

	suite.Run("non-nil after successful Parse", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.gltf")
		suite.NoError(os.WriteFile(path, []byte(`{"asset":{"version":"2.0"}}`), 0644))
		suite.NoError(suite.parser.Parse(path))
		suite.NotNil(suite.parser.Document())
	})
}

func (suite *gltfParserImplTest) TestBaseDir() {
	suite.Run("empty before Parse", func() {
		suite.Equal("", suite.parser.BaseDir())
	})

	suite.Run("set to file dir after Parse", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.gltf")
		suite.NoError(os.WriteFile(path, []byte(`{"asset":{"version":"2.0"}}`), 0644))
		suite.NoError(suite.parser.Parse(path))
		suite.Equal(filepath.Dir(path), suite.parser.BaseDir())
	})
}

func (suite *gltfParserImplTest) TestParse() {
	suite.Run("non-existent file returns error", func() {
		err := suite.parser.Parse("/nonexistent/path/file.gltf")
		suite.Error(err)
	})

	suite.Run("parses valid .gltf file successfully", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.gltf")
		suite.NoError(os.WriteFile(path, []byte(`{"asset":{"version":"2.0"}}`), 0644))
		suite.NoError(suite.parser.Parse(path))
	})

	suite.Run("parses valid .glb file successfully", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.glb")
		glbData := makeGLBData(`{"asset":{"version":"2.0"}}`, nil)
		suite.NoError(os.WriteFile(path, glbData, 0644))
		suite.NoError(suite.parser.Parse(path))
	})

	suite.Run("detects GLB by magic even with .gltf extension", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.gltf")
		glbData := makeGLBData(`{"asset":{"version":"2.0"}}`, nil)
		suite.NoError(os.WriteFile(path, glbData, 0644))
		suite.NoError(suite.parser.Parse(path))
	})
}

func (suite *gltfParserImplTest) TestParseGLTF() {
	suite.Run("invalid JSON returns error", func() {
		err := suite.impl().parseGLTF([]byte("not-valid-json"))
		suite.Error(err)
	})

	suite.Run("wrong glTF version returns error", func() {
		err := suite.impl().parseGLTF([]byte(`{"asset":{"version":"1.0"}}`))
		suite.ErrorIs(err, errInvalidGLTFVersion)
	})

	suite.Run("loadBuffers failure returns error", func() {
		json := []byte(`{"asset":{"version":"2.0"},"buffers":[{"uri":"nonexistent.bin","byteLength":4}]}`)
		err := suite.impl().parseGLTF(json)
		suite.Error(err)
	})

	suite.Run("valid glTF JSON succeeds", func() {
		err := suite.impl().parseGLTF([]byte(`{"asset":{"version":"2.0"}}`))
		suite.NoError(err)
	})
}

func (suite *gltfParserImplTest) TestParseGLB() {
	suite.Run("data too small returns error", func() {
		err := suite.impl().parseGLB([]byte{1, 2, 3})
		suite.Error(err)
		suite.Contains(err.Error(), "too small")
	})

	suite.Run("invalid magic returns error", func() {
		data := make([]byte, 12)
		binary.LittleEndian.PutUint32(data[0:], 0xDEADBEEF)
		binary.LittleEndian.PutUint32(data[4:], gltfGLBVersion)
		binary.LittleEndian.PutUint32(data[8:], 12)
		err := suite.impl().parseGLB(data)
		suite.ErrorIs(err, errInvalidGLBMagic)
	})

	suite.Run("invalid version returns error", func() {
		data := make([]byte, 12)
		binary.LittleEndian.PutUint32(data[0:], gltfGLBMagic)
		binary.LittleEndian.PutUint32(data[4:], 1)
		binary.LittleEndian.PutUint32(data[8:], 12)
		err := suite.impl().parseGLB(data)
		suite.ErrorIs(err, errInvalidGLBVersion)
	})

	suite.Run("partial chunk header returns error", func() {
		// 12 header + 4 bytes = partial chunk header (needs 8 bytes) → io.ErrUnexpectedEOF
		data := make([]byte, 16)
		binary.LittleEndian.PutUint32(data[0:], gltfGLBMagic)
		binary.LittleEndian.PutUint32(data[4:], gltfGLBVersion)
		binary.LittleEndian.PutUint32(data[8:], 16)
		// bytes 12-15 = only 4 bytes of 8-byte chunk header
		err := suite.impl().parseGLB(data)
		suite.Error(err)
		suite.Contains(err.Error(), "failed to read chunk header")
	})

	suite.Run("chunk data read failure returns error", func() {
		var buf bytes.Buffer
		_ = binary.Write(&buf, binary.LittleEndian, gltfGLBHeader{Magic: gltfGLBMagic, Version: gltfGLBVersion, Length: 30})
		_ = binary.Write(&buf, binary.LittleEndian, gltfGLBChunkHeader{ChunkLength: 100000, ChunkType: gltfGLBChunkJSON})
		buf.Write(make([]byte, 10))
		err := suite.impl().parseGLB(buf.Bytes())
		suite.Error(err)
		suite.Contains(err.Error(), "failed to read chunk data")
	})

	suite.Run("missing JSON chunk returns error", func() {
		// exact 12-byte header → no chunks → io.EOF on first chunk read → loop exits → errMissingJSONChunk
		data := make([]byte, 12)
		binary.LittleEndian.PutUint32(data[0:], gltfGLBMagic)
		binary.LittleEndian.PutUint32(data[4:], gltfGLBVersion)
		binary.LittleEndian.PutUint32(data[8:], 12)
		err := suite.impl().parseGLB(data)
		suite.ErrorIs(err, errMissingJSONChunk)
	})

	suite.Run("invalid JSON in chunk returns error", func() {
		glbData := makeGLBData("not-valid-json", nil)
		err := suite.impl().parseGLB(glbData)
		suite.Error(err)
		suite.Contains(err.Error(), "failed to parse")
	})

	suite.Run("wrong glTF version in GLB returns error", func() {
		glbData := makeGLBData(`{"asset":{"version":"1.0"}}`, nil)
		err := suite.impl().parseGLB(glbData)
		suite.ErrorIs(err, errInvalidGLTFVersion)
	})

	suite.Run("loadBuffers failure in GLB returns error", func() {
		glbData := makeGLBData(`{"asset":{"version":"2.0"},"buffers":[{"uri":"nonexistent.bin","byteLength":4}]}`, nil)
		err := suite.impl().parseGLB(glbData)
		suite.Error(err)
	})

	suite.Run("valid GLB without BIN chunk succeeds", func() {
		glbData := makeGLBData(`{"asset":{"version":"2.0"}}`, nil)
		err := suite.impl().parseGLB(glbData)
		suite.NoError(err)
	})

	suite.Run("valid GLB with BIN chunk succeeds", func() {
		binData := []byte{1, 2, 3, 4}
		glbData := makeGLBData(`{"asset":{"version":"2.0"},"buffers":[{"byteLength":4}]}`, binData)
		err := suite.impl().parseGLB(glbData)
		suite.NoError(err)
	})
}

func (suite *gltfParserImplTest) TestLoadBuffers() {
	suite.Run("buffer with no URI at index 0 with GLB chunk size ok", func() {
		impl := suite.impl()
		impl.glbBinaryChunk = []byte{1, 2, 3, 4}
		doc := &gltfDocument{Buffers: []gltfBuffer{{ByteLength: 4}}}
		suite.NoError(impl.loadBuffers(doc))
		suite.Equal([]byte{1, 2, 3, 4}, doc.Buffers[0].Data)
	})

	suite.Run("buffer with no URI at index 0 with GLB chunk but size mismatch", func() {
		impl := suite.impl()
		impl.glbBinaryChunk = []byte{1, 2, 3}
		doc := &gltfDocument{Buffers: []gltfBuffer{{ByteLength: 100}}}
		suite.ErrorIs(impl.loadBuffers(doc), errBufferSizeMismatch)
	})

	suite.Run("buffer with no URI at index 0 and no GLB chunk returns error", func() {
		doc := &gltfDocument{Buffers: []gltfBuffer{{ByteLength: 4}}}
		err := suite.impl().loadBuffers(doc)
		suite.Error(err)
		suite.Contains(err.Error(), "no URI")
	})

	suite.Run("buffer with no URI at non-zero index returns error", func() {
		encoded := base64.StdEncoding.EncodeToString([]byte{1, 2, 3})
		doc := &gltfDocument{Buffers: []gltfBuffer{
			{URI: "data:application/octet-stream;base64," + encoded, ByteLength: 3},
			{ByteLength: 4},
		}}
		err := suite.impl().loadBuffers(doc)
		suite.Error(err)
	})

	suite.Run("buffer with data URI is loaded", func() {
		encoded := base64.StdEncoding.EncodeToString([]byte{1, 2, 3})
		doc := &gltfDocument{Buffers: []gltfBuffer{
			{URI: "data:application/octet-stream;base64," + encoded, ByteLength: 3},
		}}
		suite.NoError(suite.impl().loadBuffers(doc))
		suite.Equal([]byte{1, 2, 3}, doc.Buffers[0].Data)
	})

	suite.Run("buffer with file URI is loaded", func() {
		dir := suite.T().TempDir()
		content := []byte{10, 20, 30}
		suite.NoError(os.WriteFile(filepath.Join(dir, "buf.bin"), content, 0644))
		impl := suite.impl()
		impl.baseDir = dir
		doc := &gltfDocument{Buffers: []gltfBuffer{{URI: "buf.bin", ByteLength: 3}}}
		suite.NoError(impl.loadBuffers(doc))
		suite.Equal(content, doc.Buffers[0].Data)
	})

	suite.Run("buffer file URI not found returns error", func() {
		impl := suite.impl()
		impl.baseDir = suite.T().TempDir()
		doc := &gltfDocument{Buffers: []gltfBuffer{{URI: "missing.bin", ByteLength: 4}}}
		suite.Error(impl.loadBuffers(doc))
	})

	suite.Run("loaded buffer but ByteLength mismatch returns error", func() {
		encoded := base64.StdEncoding.EncodeToString([]byte{1, 2, 3})
		doc := &gltfDocument{Buffers: []gltfBuffer{
			{URI: "data:application/octet-stream;base64," + encoded, ByteLength: 100},
		}}
		suite.ErrorIs(suite.impl().loadBuffers(doc), errBufferSizeMismatch)
	})
}

func (suite *gltfParserImplTest) TestLoadBufferURI() {
	suite.Run("data URI is decoded", func() {
		encoded := base64.StdEncoding.EncodeToString([]byte{5, 6, 7})
		data, err := suite.impl().loadBufferURI("data:application/octet-stream;base64," + encoded)
		suite.NoError(err)
		suite.Equal([]byte{5, 6, 7}, data)
	})

	suite.Run("relative file path is resolved with baseDir", func() {
		dir := suite.T().TempDir()
		content := []byte{1, 2}
		suite.NoError(os.WriteFile(filepath.Join(dir, "buf.bin"), content, 0644))
		impl := suite.impl()
		impl.baseDir = dir
		data, err := impl.loadBufferURI("buf.bin")
		suite.NoError(err)
		suite.Equal(content, data)
	})

	suite.Run("non-existent file URI returns error", func() {
		impl := suite.impl()
		impl.baseDir = suite.T().TempDir()
		_, err := impl.loadBufferURI("nonexistent.bin")
		suite.Error(err)
	})
}

func (suite *gltfParserImplTest) TestLoadDataURI() {
	suite.Run("valid base64 data URI is decoded", func() {
		encoded := base64.StdEncoding.EncodeToString([]byte{10, 20, 30})
		data, err := suite.impl().loadDataURI("data:application/octet-stream;base64," + encoded)
		suite.NoError(err)
		suite.Equal([]byte{10, 20, 30}, data)
	})

	suite.Run("invalid URI returns error", func() {
		_, err := suite.impl().loadDataURI("not-a-data-uri")
		suite.Error(err)
	})
}

func (suite *gltfParserImplTest) TestReadAccessorData() {
	suite.Run("nil document returns error", func() {
		_, err := suite.impl().ReadAccessorData(0)
		suite.Error(err)
		suite.Contains(err.Error(), "no document loaded")
	})

	suite.Run("negative index returns error", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{Accessors: []gltfAccessor{{Type: gltfAccessorTypeScalar, ComponentType: gltfComponentTypeFloat, Count: 1}}}
		_, err := impl.ReadAccessorData(-1)
		suite.Error(err)
		suite.Contains(err.Error(), "out of range")
	})

	suite.Run("index out of range returns error", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{Accessors: []gltfAccessor{{Type: gltfAccessorTypeScalar, ComponentType: gltfComponentTypeFloat, Count: 1}}}
		_, err := impl.ReadAccessorData(1)
		suite.Error(err)
		suite.Contains(err.Error(), "out of range")
	})

	suite.Run("sparse accessor returns error", func() {
		impl := suite.impl()
		sparse := &gltfAccessorSparse{Count: 1}
		impl.document = &gltfDocument{
			Accessors: []gltfAccessor{{Type: gltfAccessorTypeScalar, ComponentType: gltfComponentTypeFloat, Count: 1, Sparse: sparse}},
		}
		_, err := impl.ReadAccessorData(0)
		suite.Error(err)
		suite.Contains(err.Error(), "sparse")
	})

	suite.Run("nil BufferView returns error", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{
			Accessors: []gltfAccessor{{Type: gltfAccessorTypeScalar, ComponentType: gltfComponentTypeFloat, Count: 1}},
		}
		_, err := impl.ReadAccessorData(0)
		suite.Error(err)
		suite.Contains(err.Error(), "bufferView")
	})

	suite.Run("default stride reads correctly", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{
			Accessors:   []gltfAccessor{{Type: gltfAccessorTypeVec3, ComponentType: gltfComponentTypeFloat, Count: 2, BufferView: ptrInt(0)}},
			BufferViews: []gltfBufferView{{Buffer: 0, ByteOffset: 0, ByteLength: 24}},
			Buffers:     []gltfBuffer{{Data: f32ToBytes(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)}},
		}
		data, err := impl.ReadAccessorData(0)
		suite.NoError(err)
		suite.Equal(f32ToBytes(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), data)
	})

	suite.Run("custom stride reads correctly", func() {
		impl := suite.impl()
		stride := 24 // element is 12 bytes, 12 bytes padding
		bufData := make([]byte, 36)
		copy(bufData[0:], f32ToBytes(1.0, 2.0, 3.0))
		copy(bufData[24:], f32ToBytes(4.0, 5.0, 6.0))
		impl.document = &gltfDocument{
			Accessors:   []gltfAccessor{{Type: gltfAccessorTypeVec3, ComponentType: gltfComponentTypeFloat, Count: 2, BufferView: ptrInt(0)}},
			BufferViews: []gltfBufferView{{Buffer: 0, ByteOffset: 0, ByteLength: 36, ByteStride: &stride}},
			Buffers:     []gltfBuffer{{Data: bufData}},
		}
		data, err := impl.ReadAccessorData(0)
		suite.NoError(err)
		suite.Equal(f32ToBytes(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), data)
	})
}

func (suite *gltfParserImplTest) TestReadVec2Accessor() {
	suite.Run("wrong type returns error", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec3, gltfComponentTypeFloat, 1, f32ToBytes(1.0, 2.0, 3.0))
		_, err := impl.ReadVec2Accessor(0)
		suite.Error(err)
	})

	suite.Run("ReadAccessorData error returns error", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{
			Accessors: []gltfAccessor{{Type: gltfAccessorTypeVec2, ComponentType: gltfComponentTypeFloat, Count: 1}},
		}
		_, err := impl.ReadVec2Accessor(0)
		suite.Error(err)
	})

	suite.Run("success reads VEC2 data", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec2, gltfComponentTypeFloat, 1, f32ToBytes(1.0, 2.0))
		result, err := impl.ReadVec2Accessor(0)
		suite.NoError(err)
		suite.Equal([][2]float32{{1.0, 2.0}}, result)
	})
}

func (suite *gltfParserImplTest) TestReadVec3Accessor() {
	suite.Run("wrong type returns error", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec2, gltfComponentTypeFloat, 1, f32ToBytes(1.0, 2.0))
		_, err := impl.ReadVec3Accessor(0)
		suite.Error(err)
	})

	suite.Run("ReadAccessorData error returns error", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{
			Accessors: []gltfAccessor{{Type: gltfAccessorTypeVec3, ComponentType: gltfComponentTypeFloat, Count: 1}},
		}
		_, err := impl.ReadVec3Accessor(0)
		suite.Error(err)
	})

	suite.Run("success reads VEC3 data", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec3, gltfComponentTypeFloat, 1, f32ToBytes(1.0, 2.0, 3.0))
		result, err := impl.ReadVec3Accessor(0)
		suite.NoError(err)
		suite.Equal([][3]float32{{1.0, 2.0, 3.0}}, result)
	})
}

func (suite *gltfParserImplTest) TestReadVec4Accessor() {
	suite.Run("wrong type returns error", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec3, gltfComponentTypeFloat, 1, f32ToBytes(1.0, 2.0, 3.0))
		_, err := impl.ReadVec4Accessor(0)
		suite.Error(err)
	})

	suite.Run("ReadAccessorData error returns error", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{
			Accessors: []gltfAccessor{{Type: gltfAccessorTypeVec4, ComponentType: gltfComponentTypeFloat, Count: 1}},
		}
		_, err := impl.ReadVec4Accessor(0)
		suite.Error(err)
	})

	suite.Run("success reads VEC4 data", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec4, gltfComponentTypeFloat, 1, f32ToBytes(1.0, 2.0, 3.0, 4.0))
		result, err := impl.ReadVec4Accessor(0)
		suite.NoError(err)
		suite.Equal([][4]float32{{1.0, 2.0, 3.0, 4.0}}, result)
	})
}

func (suite *gltfParserImplTest) TestReadScalarAccessor() {
	suite.Run("wrong type returns error", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec2, gltfComponentTypeFloat, 1, f32ToBytes(1.0, 2.0))
		_, err := impl.ReadScalarAccessor(0)
		suite.Error(err)
	})

	suite.Run("ReadAccessorData error returns error", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{
			Accessors: []gltfAccessor{{Type: gltfAccessorTypeScalar, ComponentType: gltfComponentTypeFloat, Count: 1}},
		}
		_, err := impl.ReadScalarAccessor(0)
		suite.Error(err)
	})

	suite.Run("success reads SCALAR data", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeScalar, gltfComponentTypeFloat, 2, f32ToBytes(1.5, 2.5))
		result, err := impl.ReadScalarAccessor(0)
		suite.NoError(err)
		suite.Equal([]float32{1.5, 2.5}, result)
	})
}

func (suite *gltfParserImplTest) TestReadMat4Accessor() {
	suite.Run("wrong type returns error", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec3, gltfComponentTypeFloat, 1, f32ToBytes(1.0, 2.0, 3.0))
		_, err := impl.ReadMat4Accessor(0)
		suite.Error(err)
	})

	suite.Run("ReadAccessorData error returns error", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{
			Accessors: []gltfAccessor{{Type: gltfAccessorTypeMat4, ComponentType: gltfComponentTypeFloat, Count: 1}},
		}
		_, err := impl.ReadMat4Accessor(0)
		suite.Error(err)
	})

	suite.Run("success reads MAT4 data", func() {
		vals := [16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		var allVals []float32
		for _, v := range vals {
			allVals = append(allVals, v)
		}
		impl := setupAccessorDoc(suite, gltfAccessorTypeMat4, gltfComponentTypeFloat, 1, f32ToBytes(allVals...))
		result, err := impl.ReadMat4Accessor(0)
		suite.NoError(err)
		suite.Equal([][16]float32{vals}, result)
	})
}

func (suite *gltfParserImplTest) TestReadIndicesAccessor() {
	suite.Run("not SCALAR returns error", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec3, gltfComponentTypeUnsignedInt, 1, u32ToBytes(1))
		_, err := impl.ReadIndicesAccessor(0)
		suite.Error(err)
	})

	suite.Run("ReadAccessorData error returns error", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{
			Accessors: []gltfAccessor{{Type: gltfAccessorTypeScalar, ComponentType: gltfComponentTypeUnsignedByte, Count: 1}},
		}
		_, err := impl.ReadIndicesAccessor(0)
		suite.Error(err)
	})

	suite.Run("UNSIGNED_BYTE reads correctly", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeScalar, gltfComponentTypeUnsignedByte, 3, []byte{1, 2, 3})
		result, err := impl.ReadIndicesAccessor(0)
		suite.NoError(err)
		suite.Equal([]uint32{1, 2, 3}, result)
	})

	suite.Run("UNSIGNED_SHORT reads correctly", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeScalar, gltfComponentTypeUnsignedShort, 2, u16ToBytes(100, 200))
		result, err := impl.ReadIndicesAccessor(0)
		suite.NoError(err)
		suite.Equal([]uint32{100, 200}, result)
	})

	suite.Run("UNSIGNED_INT reads correctly", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeScalar, gltfComponentTypeUnsignedInt, 2, u32ToBytes(1000, 2000))
		result, err := impl.ReadIndicesAccessor(0)
		suite.NoError(err)
		suite.Equal([]uint32{1000, 2000}, result)
	})

	suite.Run("unsupported component type returns error", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeScalar, 9999, 1, make([]byte, 4))
		_, err := impl.ReadIndicesAccessor(0)
		suite.Error(err)
	})
}

func (suite *gltfParserImplTest) TestReadJointsAccessor() {
	suite.Run("not VEC4 returns error", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec3, gltfComponentTypeUnsignedByte, 1, []byte{1, 2, 3})
		_, err := impl.ReadJointsAccessor(0)
		suite.Error(err)
	})

	suite.Run("ReadAccessorData error returns error", func() {
		impl := suite.impl()
		impl.document = &gltfDocument{
			Accessors: []gltfAccessor{{Type: gltfAccessorTypeVec4, ComponentType: gltfComponentTypeUnsignedByte, Count: 1}},
		}
		_, err := impl.ReadJointsAccessor(0)
		suite.Error(err)
	})

	suite.Run("UNSIGNED_BYTE reads correctly", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec4, gltfComponentTypeUnsignedByte, 1, []byte{1, 2, 3, 4})
		result, err := impl.ReadJointsAccessor(0)
		suite.NoError(err)
		suite.Equal([][4]uint32{{1, 2, 3, 4}}, result)
	})

	suite.Run("UNSIGNED_SHORT reads correctly", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec4, gltfComponentTypeUnsignedShort, 1, u16ToBytes(10, 20, 30, 40))
		result, err := impl.ReadJointsAccessor(0)
		suite.NoError(err)
		suite.Equal([][4]uint32{{10, 20, 30, 40}}, result)
	})

	suite.Run("unsupported component type returns error", func() {
		impl := setupAccessorDoc(suite, gltfAccessorTypeVec4, gltfComponentTypeFloat, 1, f32ToBytes(1, 2, 3, 4))
		_, err := impl.ReadJointsAccessor(0)
		suite.Error(err)
	})
}

func (suite *gltfParserImplTest) TestGltfComponentTypeSize() {
	suite.Run("UNSIGNED_BYTE returns 1", func() {
		suite.Equal(1, gltfComponentTypeSize(gltfComponentTypeUnsignedByte))
	})
	suite.Run("UNSIGNED_SHORT returns 2", func() {
		suite.Equal(2, gltfComponentTypeSize(gltfComponentTypeUnsignedShort))
	})
	suite.Run("UNSIGNED_INT returns 4", func() {
		suite.Equal(4, gltfComponentTypeSize(gltfComponentTypeUnsignedInt))
	})
	suite.Run("FLOAT returns 4", func() {
		suite.Equal(4, gltfComponentTypeSize(gltfComponentTypeFloat))
	})
	suite.Run("unknown returns 0", func() {
		suite.Equal(0, gltfComponentTypeSize(9999))
	})
}

func (suite *gltfParserImplTest) TestGltfAccessorTypeComponentCount() {
	suite.Run("SCALAR returns 1", func() {
		suite.Equal(1, gltfAccessorTypeComponentCount(gltfAccessorTypeScalar))
	})
	suite.Run("VEC2 returns 2", func() {
		suite.Equal(2, gltfAccessorTypeComponentCount(gltfAccessorTypeVec2))
	})
	suite.Run("VEC3 returns 3", func() {
		suite.Equal(3, gltfAccessorTypeComponentCount(gltfAccessorTypeVec3))
	})
	suite.Run("VEC4 returns 4", func() {
		suite.Equal(4, gltfAccessorTypeComponentCount(gltfAccessorTypeVec4))
	})
	suite.Run("MAT4 returns 16", func() {
		suite.Equal(16, gltfAccessorTypeComponentCount(gltfAccessorTypeMat4))
	})
	suite.Run("unknown returns 0", func() {
		suite.Equal(0, gltfAccessorTypeComponentCount("UNKNOWN"))
	})
}

// makeGLBData builds a valid GLB binary. binData may be nil for JSON-only GLB.
func makeGLBData(jsonStr string, binData []byte) []byte {
	var buf bytes.Buffer
	jsonBytes := []byte(jsonStr)
	for len(jsonBytes)%4 != 0 {
		jsonBytes = append(jsonBytes, 0x20)
	}
	totalLen := uint32(12 + 8 + len(jsonBytes))
	var paddedBin []byte
	if binData != nil {
		paddedBin = make([]byte, (len(binData)+3)&^3)
		copy(paddedBin, binData)
		totalLen += uint32(8 + len(paddedBin))
	}
	_ = binary.Write(&buf, binary.LittleEndian, gltfGLBHeader{Magic: gltfGLBMagic, Version: gltfGLBVersion, Length: totalLen})
	_ = binary.Write(&buf, binary.LittleEndian, gltfGLBChunkHeader{ChunkLength: uint32(len(jsonBytes)), ChunkType: gltfGLBChunkJSON})
	buf.Write(jsonBytes)
	if paddedBin != nil {
		_ = binary.Write(&buf, binary.LittleEndian, gltfGLBChunkHeader{ChunkLength: uint32(len(paddedBin)), ChunkType: gltfGLBChunkBIN})
		buf.Write(paddedBin)
	}
	return buf.Bytes()
}

// f32ToBytes encodes float32 values to little-endian bytes.
func f32ToBytes(vals ...float32) []byte {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// u16ToBytes encodes uint16 values to little-endian bytes.
func u16ToBytes(vals ...uint16) []byte {
	b := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(b[i*2:], v)
	}
	return b
}

// u32ToBytes encodes uint32 values to little-endian bytes.
func u32ToBytes(vals ...uint32) []byte {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], v)
	}
	return b
}

// ptrInt returns a *int pointing to v.
func ptrInt(v int) *int { return &v }

// setupAccessorDoc builds a minimal gltfParserImpl with a document containing one accessor.
func setupAccessorDoc(suite *gltfParserImplTest, accType string, compType int, count int, data []byte) *gltfParserImpl {
	impl := suite.impl()
	impl.document = &gltfDocument{
		Accessors:   []gltfAccessor{{Type: accType, ComponentType: compType, Count: count, BufferView: ptrInt(0)}},
		BufferViews: []gltfBufferView{{Buffer: 0, ByteLength: len(data)}},
		Buffers:     []gltfBuffer{{Data: data}},
	}
	return impl
}
