package loader

import (
	"encoding/base64"
	"os"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type gltfMaterialExtractorImplTest struct {
	suite.Suite
	mockParser *mockGltfParser
	extractor  gltfMaterialExtractor
}

func (suite *gltfMaterialExtractorImplTest) SetupSubTest() {
	suite.mockParser = &mockGltfParser{}
	suite.extractor = newGLTFMaterialExtractor(suite.mockParser)
}

func TestGltfMaterialExtractorImpl(t *testing.T) {
	suite.Run(t, new(gltfMaterialExtractorImplTest))
}

func (suite *gltfMaterialExtractorImplTest) TestExtractMaterial() {
	suite.Run("nil document returns error", func() {
		suite.mockParser.doc = nil
		_, err := suite.extractor.ExtractMaterial(0)
		suite.Error(err)
	})

	suite.Run("negative index returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{Name: "mat"}},
		}
		_, err := suite.extractor.ExtractMaterial(-1)
		suite.Error(err)
	})

	suite.Run("index out of range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{Name: "mat"}},
		}
		_, err := suite.extractor.ExtractMaterial(1)
		suite.Error(err)
	})

	suite.Run("material with no PBR uses defaults", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{Name: "mat"}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.Equal("mat", result.Name)
		suite.Equal([4]float32{1, 1, 1, 1}, result.BaseColor)
		suite.InDelta(float32(1.0), result.Metallic, 1e-6)
		suite.InDelta(float32(1.0), result.Roughness, 1e-6)
	})

	suite.Run("PBR BaseColorFactor is applied", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					BaseColorFactor: &[4]float32{0.5, 0.5, 0.5, 1},
				},
			}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.Equal([4]float32{0.5, 0.5, 0.5, 1}, result.BaseColor)
	})

	suite.Run("PBR MetallicFactor is applied", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					MetallicFactor: common.ToPtr(float32(0.3)),
				},
			}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.InDelta(float32(0.3), result.Metallic, 1e-6)
	})

	suite.Run("PBR RoughnessFactor is applied", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					RoughnessFactor: common.ToPtr(float32(0.7)),
				},
			}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.InDelta(float32(0.7), result.Roughness, 1e-6)
	})

	suite.Run("PBR BaseColorTexture out-of-range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					BaseColorTexture: &gltfTextureInfo{Index: 0},
				},
			}},
			Textures: []gltfTexture{},
		}
		_, err := suite.extractor.ExtractMaterial(0)
		suite.Error(err)
	})

	suite.Run("PBR MetallicRoughnessTexture out-of-range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					MetallicRoughnessTexture: &gltfTextureInfo{Index: 0},
				},
			}},
			Textures: []gltfTexture{},
		}
		_, err := suite.extractor.ExtractMaterial(0)
		suite.Error(err)
	})

	suite.Run("NormalTexture out-of-range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				NormalTexture: &gltfNormalTextureInfo{gltfTextureInfo: gltfTextureInfo{Index: 0}},
			}},
			Textures: []gltfTexture{},
		}
		_, err := suite.extractor.ExtractMaterial(0)
		suite.Error(err)
	})

	suite.Run("AlphaCutoff is applied", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				AlphaCutoff: common.ToPtr(float32(0.5)),
			}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.InDelta(float32(0.5), result.AlphaCutoff, 1e-6)
	})

	suite.Run("AlphaMode is applied", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				AlphaMode: "MASK",
			}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.Equal("MASK", result.AlphaMode)
	})

	suite.Run("PBR BaseColorTexture with nil source returns nil texture", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					BaseColorTexture: &gltfTextureInfo{Index: 0},
				},
			}},
			Textures: []gltfTexture{{Source: nil}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.Nil(result.DiffuseTexture)
		suite.Equal("", result.DiffuseTexturePath)
	})

	suite.Run("PBR MetallicRoughnessTexture with nil source returns nil texture", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					MetallicRoughnessTexture: &gltfTextureInfo{Index: 0},
				},
			}},
			Textures: []gltfTexture{{Source: nil}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.Nil(result.MetallicRoughnessTexture)
		suite.Equal("", result.MetallicTexturePath)
	})

	suite.Run("NormalTexture with nil source returns nil texture", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				NormalTexture: &gltfNormalTextureInfo{gltfTextureInfo: gltfTextureInfo{Index: 0}},
			}},
			Textures: []gltfTexture{{Source: nil}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.Nil(result.NormalTexture)
		suite.Equal("", result.NormalTexturePath)
	})

	suite.Run("PBR BaseColorTexture embedded sets DiffuseTexture", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					BaseColorTexture: &gltfTextureInfo{Index: 0},
				},
			}},
			Textures:    []gltfTexture{{Source: common.ToPtr(0)}},
			Images:      []gltfImage{{BufferView: common.ToPtr(0)}},
			BufferViews: []gltfBufferView{{Buffer: 0, ByteOffset: 0, ByteLength: 2}},
			Buffers:     []gltfBuffer{{Data: []byte{0xFF, 0xD8}}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.NotNil(result.DiffuseTexture)
	})

	suite.Run("PBR BaseColorTexture external sets DiffuseTexturePath", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					BaseColorTexture: &gltfTextureInfo{Index: 0},
				},
			}},
			Textures: []gltfTexture{{Source: common.ToPtr(0)}},
			Images:   []gltfImage{{URI: "diffuse.png"}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.Equal("diffuse.png", result.DiffuseTexturePath)
	})

	suite.Run("PBR MetallicRoughnessTexture embedded sets MetallicRoughnessTexture", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					MetallicRoughnessTexture: &gltfTextureInfo{Index: 0},
				},
			}},
			Textures:    []gltfTexture{{Source: common.ToPtr(0)}},
			Images:      []gltfImage{{BufferView: common.ToPtr(0)}},
			BufferViews: []gltfBufferView{{Buffer: 0, ByteOffset: 0, ByteLength: 2}},
			Buffers:     []gltfBuffer{{Data: []byte{0xFF, 0xD8}}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.NotNil(result.MetallicRoughnessTexture)
	})

	suite.Run("PBR MetallicRoughnessTexture external sets MetallicTexturePath", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					MetallicRoughnessTexture: &gltfTextureInfo{Index: 0},
				},
			}},
			Textures: []gltfTexture{{Source: common.ToPtr(0)}},
			Images:   []gltfImage{{URI: "metallic.png"}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.Equal("metallic.png", result.MetallicTexturePath)
	})

	suite.Run("NormalTexture embedded sets NormalTexture", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				NormalTexture: &gltfNormalTextureInfo{gltfTextureInfo: gltfTextureInfo{Index: 0}},
			}},
			Textures:    []gltfTexture{{Source: common.ToPtr(0)}},
			Images:      []gltfImage{{BufferView: common.ToPtr(0)}},
			BufferViews: []gltfBufferView{{Buffer: 0, ByteOffset: 0, ByteLength: 2}},
			Buffers:     []gltfBuffer{{Data: []byte{0xFF, 0xD8}}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.NotNil(result.NormalTexture)
	})

	suite.Run("NormalTexture external sets NormalTexturePath", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				NormalTexture: &gltfNormalTextureInfo{gltfTextureInfo: gltfTextureInfo{Index: 0}},
			}},
			Textures: []gltfTexture{{Source: common.ToPtr(0)}},
			Images:   []gltfImage{{URI: "normal.png"}},
		}
		result, err := suite.extractor.ExtractMaterial(0)
		suite.NoError(err)
		suite.Equal("normal.png", result.NormalTexturePath)
	})
}

func (suite *gltfMaterialExtractorImplTest) TestExtractAllMaterials() {
	suite.Run("nil document returns error", func() {
		suite.mockParser.doc = nil
		_, err := suite.extractor.ExtractAllMaterials()
		suite.Error(err)
	})

	suite.Run("empty materials returns empty slice", func() {
		suite.mockParser.doc = &gltfDocument{Materials: nil}
		result, err := suite.extractor.ExtractAllMaterials()
		suite.NoError(err)
		suite.Len(result, 0)
	})

	suite.Run("all materials extracted", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{
				{Name: "mat0"},
				{Name: "mat1"},
			},
		}
		result, err := suite.extractor.ExtractAllMaterials()
		suite.NoError(err)
		suite.Len(result, 2)
		suite.Equal("mat0", result[0].Name)
		suite.Equal("mat1", result[1].Name)
	})

	suite.Run("error from ExtractMaterial propagates", func() {
		suite.mockParser.doc = &gltfDocument{
			Materials: []gltfMaterial{{
				PbrMetallicRoughness: &gltfPbrMetallicRoughness{
					BaseColorTexture: &gltfTextureInfo{Index: 0},
				},
			}},
			Textures: []gltfTexture{},
		}
		_, err := suite.extractor.ExtractAllMaterials()
		suite.Error(err)
	})
}

func (suite *gltfMaterialExtractorImplTest) TestLoadTexture() {
	impl := func() *gltfMaterialExtractorImpl {
		return suite.extractor.(*gltfMaterialExtractorImpl)
	}

	suite.Run("negative index returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Textures: []gltfTexture{{Source: common.ToPtr(0)}},
		}
		_, _, err := impl().loadTexture(-1)
		suite.Error(err)
	})

	suite.Run("index out of range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Textures: []gltfTexture{{Source: common.ToPtr(0)}},
		}
		_, _, err := impl().loadTexture(1)
		suite.Error(err)
	})

	suite.Run("texture with nil source returns nil", func() {
		suite.mockParser.doc = &gltfDocument{
			Textures: []gltfTexture{{Source: nil}},
		}
		tex, path, err := impl().loadTexture(0)
		suite.NoError(err)
		suite.Nil(tex)
		suite.Equal("", path)
	})

	suite.Run("texture with sampler out of range is tolerated", func() {
		suite.mockParser.doc = &gltfDocument{
			Textures: []gltfTexture{{Sampler: common.ToPtr(5), Source: common.ToPtr(0)}},
			Images:   []gltfImage{{Name: "img"}},
			Samplers: []gltfSampler{},
		}
		// No URI, no BufferView — falls through to nil return; no error
		tex, path, err := impl().loadTexture(0)
		suite.NoError(err)
		suite.Nil(tex)
		suite.Equal("", path)
	})

	suite.Run("image index out of range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Textures: []gltfTexture{{Source: common.ToPtr(99)}},
			Images:   []gltfImage{{Name: "img"}},
		}
		_, _, err := impl().loadTexture(0)
		suite.Error(err)
	})

	suite.Run("valid sampler sets samplerData on returned texture", func() {
		suite.mockParser.doc = &gltfDocument{
			Textures: []gltfTexture{{Sampler: common.ToPtr(0), Source: common.ToPtr(0)}},
			Images:   []gltfImage{{URI: "nonexistent.png"}},
			Samplers: []gltfSampler{{}},
		}
		tex, _, err := impl().loadTexture(0)
		suite.NoError(err)
		suite.NotNil(tex)
		suite.NotNil(tex.SamplerData)
	})

	suite.Run("image with buffer view returns data", func() {
		suite.mockParser.doc = &gltfDocument{
			Textures:    []gltfTexture{{Source: common.ToPtr(0)}},
			Images:      []gltfImage{{BufferView: common.ToPtr(0)}},
			BufferViews: []gltfBufferView{{Buffer: 0, ByteOffset: 0, ByteLength: 3}},
			Buffers:     []gltfBuffer{{Data: []byte{1, 2, 3}}},
		}
		tex, path, err := impl().loadTexture(0)
		suite.NoError(err)
		suite.NotNil(tex)
		suite.Equal([]byte{1, 2, 3}, tex.Data)
		suite.Equal("", path)
	})

	suite.Run("buffer view read error propagates", func() {
		suite.mockParser.doc = &gltfDocument{
			Textures:    []gltfTexture{{Source: common.ToPtr(0)}},
			Images:      []gltfImage{{BufferView: common.ToPtr(0)}},
			BufferViews: []gltfBufferView{{Buffer: 99, ByteOffset: 0, ByteLength: 3}},
			Buffers:     []gltfBuffer{},
		}
		_, _, err := impl().loadTexture(0)
		suite.Error(err)
	})

	suite.Run("data URI is decoded", func() {
		encoded := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte{10, 20, 30})
		suite.mockParser.doc = &gltfDocument{
			Textures: []gltfTexture{{Source: common.ToPtr(0)}},
			Images:   []gltfImage{{URI: encoded}},
		}
		tex, _, err := impl().loadTexture(0)
		suite.NoError(err)
		suite.NotNil(tex)
		suite.Equal([]byte{10, 20, 30}, tex.Data)
	})

	suite.Run("malformed data URI returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Textures: []gltfTexture{{Source: common.ToPtr(0)}},
			Images:   []gltfImage{{URI: "data:no-comma"}},
		}
		_, _, err := impl().loadTexture(0)
		suite.Error(err)
	})

	suite.Run("external file path is returned when file missing", func() {
		suite.mockParser.doc = &gltfDocument{
			Textures: []gltfTexture{{Source: common.ToPtr(0)}},
			Images:   []gltfImage{{URI: "missing.png"}},
		}
		tex, path, err := impl().loadTexture(0)
		suite.NoError(err)
		suite.NotNil(tex)
		suite.Equal("missing.png", path)
	})

	suite.Run("external file data is loaded when file exists", func() {
		expected := []byte{7, 8, 9}
		f, ferr := os.CreateTemp("", "oxy_loader_test_*.bin")
		suite.NoError(ferr)
		defer os.Remove(f.Name())
		_, werr := f.Write(expected)
		suite.NoError(werr)
		f.Close()

		suite.mockParser.doc = &gltfDocument{
			Textures: []gltfTexture{{Source: common.ToPtr(0)}},
			Images:   []gltfImage{{URI: f.Name()}},
		}
		tex, path, err := impl().loadTexture(0)
		suite.NoError(err)
		suite.NotNil(tex)
		suite.Equal(expected, tex.Data)
		suite.Equal(f.Name(), path)
	})
}

func (suite *gltfMaterialExtractorImplTest) TestReadBufferViewRaw() {
	impl := func() *gltfMaterialExtractorImpl {
		return suite.extractor.(*gltfMaterialExtractorImpl)
	}

	suite.Run("negative index returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			BufferViews: []gltfBufferView{{Buffer: 0, ByteOffset: 0, ByteLength: 3}},
			Buffers:     []gltfBuffer{{Data: []byte{1, 2, 3}}},
		}
		_, err := impl().readBufferViewRaw(-1)
		suite.Error(err)
	})

	suite.Run("index out of range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			BufferViews: []gltfBufferView{{Buffer: 0, ByteOffset: 0, ByteLength: 3}},
			Buffers:     []gltfBuffer{{Data: []byte{1, 2, 3}}},
		}
		_, err := impl().readBufferViewRaw(1)
		suite.Error(err)
	})

	suite.Run("buffer index out of range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			BufferViews: []gltfBufferView{{Buffer: 5, ByteOffset: 0, ByteLength: 3}},
			Buffers:     []gltfBuffer{},
		}
		_, err := impl().readBufferViewRaw(0)
		suite.Error(err)
	})

	suite.Run("bufferView exceeds buffer bounds returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			BufferViews: []gltfBufferView{{Buffer: 0, ByteOffset: 0, ByteLength: 10}},
			Buffers:     []gltfBuffer{{Data: []byte{1, 2, 3}}},
		}
		_, err := impl().readBufferViewRaw(0)
		suite.Error(err)
	})

	suite.Run("successful read returns correct bytes", func() {
		suite.mockParser.doc = &gltfDocument{
			BufferViews: []gltfBufferView{{Buffer: 0, ByteOffset: 1, ByteLength: 2}},
			Buffers:     []gltfBuffer{{Data: []byte{10, 20, 30}}},
		}
		data, err := impl().readBufferViewRaw(0)
		suite.NoError(err)
		suite.Equal([]byte{20, 30}, data)
	})
}

func (suite *gltfMaterialExtractorImplTest) TestGltfDecodeDataURI() {
	suite.Run("not a data URI returns error", func() {
		_, _, err := gltfDecodeDataURI("http://example.com")
		suite.Error(err)
	})

	suite.Run("no comma returns error", func() {
		_, _, err := gltfDecodeDataURI("data:image/png;base64")
		suite.Error(err)
	})

	suite.Run("valid base64 with mime type is decoded", func() {
		encoded := base64.StdEncoding.EncodeToString([]byte{1, 2, 3})
		data, mimeType, err := gltfDecodeDataURI("data:image/png;base64," + encoded)
		suite.NoError(err)
		suite.Equal([]byte{1, 2, 3}, data)
		suite.Equal("image/png", mimeType)
	})

	suite.Run("header without base64 prefix returns plain mime type", func() {
		encoded := base64.StdEncoding.EncodeToString([]byte{65})
		data, mimeType, err := gltfDecodeDataURI("data:text/plain," + encoded)
		suite.NoError(err)
		suite.Equal([]byte{65}, data)
		suite.Equal("text/plain", mimeType)
	})

	suite.Run("invalid base64 returns error", func() {
		_, _, err := gltfDecodeDataURI("data:image/png;base64,!!!invalid!!!")
		suite.Error(err)
	})
}

func (suite *gltfMaterialExtractorImplTest) TestGltfSamplerToStagingData() {
	suite.Run("nil fields produce linear repeat defaults", func() {
		result := gltfSamplerToStagingData(&gltfSampler{})
		suite.Equal(wgpu.FilterModeLinear, result.MagFilter)
		suite.Equal(wgpu.FilterModeLinear, result.MinFilter)
		suite.Equal(wgpu.MipmapFilterModeLinear, result.MipmapFilter)
		suite.Equal(wgpu.AddressModeRepeat, result.AddressModeU)
		suite.Equal(wgpu.AddressModeRepeat, result.AddressModeV)
		suite.Equal(wgpu.AddressModeRepeat, result.AddressModeW)
	})

	suite.Run("MagFilter nearest is applied", func() {
		result := gltfSamplerToStagingData(&gltfSampler{MagFilter: common.ToPtr(gltfFilterNearest)})
		suite.Equal(wgpu.FilterModeNearest, result.MagFilter)
	})

	suite.Run("MagFilter linear is applied", func() {
		result := gltfSamplerToStagingData(&gltfSampler{MagFilter: common.ToPtr(gltfFilterLinear)})
		suite.Equal(wgpu.FilterModeLinear, result.MagFilter)
	})

	suite.Run("MinFilter nearest sets MinFilter nearest and mipmap nearest", func() {
		result := gltfSamplerToStagingData(&gltfSampler{MinFilter: common.ToPtr(gltfFilterNearest)})
		suite.Equal(wgpu.FilterModeNearest, result.MinFilter)
		suite.Equal(wgpu.MipmapFilterModeNearest, result.MipmapFilter)
	})

	suite.Run("MinFilter NearestMipmapNearest sets mipmap nearest", func() {
		result := gltfSamplerToStagingData(&gltfSampler{MinFilter: common.ToPtr(gltfFilterNearestMipmapNearest)})
		suite.Equal(wgpu.FilterModeNearest, result.MinFilter)
		suite.Equal(wgpu.MipmapFilterModeNearest, result.MipmapFilter)
	})

	suite.Run("MinFilter NearestMipmapLinear sets mipmap linear", func() {
		result := gltfSamplerToStagingData(&gltfSampler{MinFilter: common.ToPtr(gltfFilterNearestMipmapLinear)})
		suite.Equal(wgpu.FilterModeNearest, result.MinFilter)
		suite.Equal(wgpu.MipmapFilterModeLinear, result.MipmapFilter)
	})

	suite.Run("MinFilter LinearMipmapNearest sets mipmap nearest", func() {
		result := gltfSamplerToStagingData(&gltfSampler{MinFilter: common.ToPtr(gltfFilterLinearMipmapNearest)})
		suite.Equal(wgpu.FilterModeLinear, result.MinFilter)
		suite.Equal(wgpu.MipmapFilterModeNearest, result.MipmapFilter)
	})

	suite.Run("MinFilter LinearMipmapLinear sets mipmap linear", func() {
		result := gltfSamplerToStagingData(&gltfSampler{MinFilter: common.ToPtr(gltfFilterLinearMipmapLinear)})
		suite.Equal(wgpu.FilterModeLinear, result.MinFilter)
		suite.Equal(wgpu.MipmapFilterModeLinear, result.MipmapFilter)
	})

	suite.Run("MinFilter linear sets mipmap nearest", func() {
		result := gltfSamplerToStagingData(&gltfSampler{MinFilter: common.ToPtr(gltfFilterLinear)})
		suite.Equal(wgpu.FilterModeLinear, result.MinFilter)
		suite.Equal(wgpu.MipmapFilterModeNearest, result.MipmapFilter)
	})

	suite.Run("WrapS ClampToEdge sets AddressModeU ClampToEdge", func() {
		result := gltfSamplerToStagingData(&gltfSampler{WrapS: common.ToPtr(gltfWrapClampToEdge)})
		suite.Equal(wgpu.AddressModeClampToEdge, result.AddressModeU)
	})

	suite.Run("WrapS MirroredRepeat sets AddressModeU MirrorRepeat", func() {
		result := gltfSamplerToStagingData(&gltfSampler{WrapS: common.ToPtr(gltfWrapMirroredRepeat)})
		suite.Equal(wgpu.AddressModeMirrorRepeat, result.AddressModeU)
	})

	suite.Run("WrapT ClampToEdge sets AddressModeV ClampToEdge", func() {
		result := gltfSamplerToStagingData(&gltfSampler{WrapT: common.ToPtr(gltfWrapClampToEdge)})
		suite.Equal(wgpu.AddressModeClampToEdge, result.AddressModeV)
	})
}

func (suite *gltfMaterialExtractorImplTest) TestGltfWrapToAddressMode() {
	suite.Run("ClampToEdge", func() {
		suite.Equal(wgpu.AddressModeClampToEdge, gltfWrapToAddressMode(gltfWrapClampToEdge))
	})

	suite.Run("MirroredRepeat", func() {
		suite.Equal(wgpu.AddressModeMirrorRepeat, gltfWrapToAddressMode(gltfWrapMirroredRepeat))
	})

	suite.Run("Repeat", func() {
		suite.Equal(wgpu.AddressModeRepeat, gltfWrapToAddressMode(gltfWrapRepeat))
	})

	suite.Run("unknown defaults to Repeat", func() {
		suite.Equal(wgpu.AddressModeRepeat, gltfWrapToAddressMode(0))
	})
}
