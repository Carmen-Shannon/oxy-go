package loader

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Carmen-Shannon/oxy-go/common"

	"github.com/cogentcore/webgpu/wgpu"
)

// gltfMaterialExtractorImpl is the implementation of the gltfMaterialExtractor interface.
type gltfMaterialExtractorImpl struct {
	parser gltfParser
}

// gltfMaterialExtractor defines the interface for extracting material and texture data
// from a parsed glTF document into engine-ready ImportedMaterial structs.
type gltfMaterialExtractor interface {
	// ExtractMaterial extracts a single material by index, including loading any referenced texture data.
	//
	// Parameters:
	//   - materialIndex: the index of the material in the document
	//
	// Returns:
	//   - *common.ImportedMaterial: the extracted material with any embedded texture data loaded
	//   - error: error if extraction fails
	ExtractMaterial(materialIndex int) (*common.ImportedMaterial, error)

	// ExtractAllMaterials extracts all materials from the document.
	//
	// Returns:
	//   - []*common.ImportedMaterial: all extracted materials
	//   - error: error if extraction fails
	ExtractAllMaterials() ([]*common.ImportedMaterial, error)
}

var _ gltfMaterialExtractor = &gltfMaterialExtractorImpl{}

// newGLTFMaterialExtractor creates a new material extractor for a parsed document.
//
// Parameters:
//   - parser: the parser containing a loaded document
//
// Returns:
//   - gltfMaterialExtractor: the material extractor
func newGLTFMaterialExtractor(parser gltfParser) gltfMaterialExtractor {
	return &gltfMaterialExtractorImpl{parser: parser}
}

func (e *gltfMaterialExtractorImpl) ExtractMaterial(materialIndex int) (*common.ImportedMaterial, error) {
	doc := e.parser.Document()
	if doc == nil {
		return nil, fmt.Errorf("no document loaded")
	}
	if materialIndex < 0 || materialIndex >= len(doc.Materials) {
		return nil, fmt.Errorf("material index %d out of range", materialIndex)
	}

	mat := &doc.Materials[materialIndex]

	result := &common.ImportedMaterial{
		Name:      mat.Name,
		BaseColor: [4]float32{1, 1, 1, 1},
		Metallic:  1.0,
		Roughness: 1.0,
	}

	if mat.PbrMetallicRoughness != nil {
		pbr := mat.PbrMetallicRoughness

		if pbr.BaseColorFactor != nil {
			result.BaseColor = *pbr.BaseColorFactor
		}
		if pbr.MetallicFactor != nil {
			result.Metallic = *pbr.MetallicFactor
		}
		if pbr.RoughnessFactor != nil {
			result.Roughness = *pbr.RoughnessFactor
		}

		// Base color / diffuse texture
		if pbr.BaseColorTexture != nil {
			tex, path, err := e.loadTexture(pbr.BaseColorTexture.Index)
			if err != nil {
				return nil, fmt.Errorf("material %q: base color texture: %w", mat.Name, err)
			}
			if tex != nil {
				result.DiffuseTexture = tex
			}
			if path != "" {
				result.DiffuseTexturePath = path
			}
		}

		// Metallic-roughness texture
		if pbr.MetallicRoughnessTexture != nil {
			tex, path, err := e.loadTexture(pbr.MetallicRoughnessTexture.Index)
			if err != nil {
				return nil, fmt.Errorf("material %q: metallic-roughness texture: %w", mat.Name, err)
			}
			if tex != nil {
				result.MetallicRoughnessTexture = tex
			}
			if path != "" {
				result.MetallicTexturePath = path
			}
		}
	}

	// Normal map
	if mat.NormalTexture != nil {
		tex, path, err := e.loadTexture(mat.NormalTexture.Index)
		if err != nil {
			return nil, fmt.Errorf("material %q: normal texture: %w", mat.Name, err)
		}
		if tex != nil {
			result.NormalTexture = tex
		}
		if path != "" {
			result.NormalTexturePath = path
		}
	}

	return result, nil
}

func (e *gltfMaterialExtractorImpl) ExtractAllMaterials() ([]*common.ImportedMaterial, error) {
	doc := e.parser.Document()
	if doc == nil {
		return nil, fmt.Errorf("no document loaded")
	}

	materials := make([]*common.ImportedMaterial, len(doc.Materials))
	for i := range doc.Materials {
		mat, err := e.ExtractMaterial(i)
		if err != nil {
			return nil, fmt.Errorf("material %d: %w", i, err)
		}
		materials[i] = mat
	}

	return materials, nil
}

// loadTexture resolves a glTF texture index into an ImportedTexture with loaded image data.
// Returns the texture (for embedded data) and/or a file path (for external references).
// For embedded images (buffer view or data URI), the raw bytes are loaded into the texture.
// For external file references, the path is resolved relative to the glTF base directory.
func (e *gltfMaterialExtractorImpl) loadTexture(textureIndex int) (*common.ImportedTexture, string, error) {
	doc := e.parser.Document()
	if textureIndex < 0 || textureIndex >= len(doc.Textures) {
		return nil, "", fmt.Errorf("texture index %d out of range", textureIndex)
	}

	tex := &doc.Textures[textureIndex]
	if tex.Source == nil {
		return nil, "", nil
	}

	// Resolve glTF sampler parameters if this texture references one.
	var samplerData *common.SamplerStagingData
	if tex.Sampler != nil {
		samplerIdx := *tex.Sampler
		if samplerIdx >= 0 && samplerIdx < len(doc.Samplers) {
			samplerData = gltfSamplerToStagingData(&doc.Samplers[samplerIdx])
		}
	}

	imageIndex := *tex.Source
	if imageIndex < 0 || imageIndex >= len(doc.Images) {
		return nil, "", fmt.Errorf("image index %d out of range", imageIndex)
	}

	img := &doc.Images[imageIndex]

	result := &common.ImportedTexture{
		Name:        img.Name,
		MimeType:    img.MimeType,
		SamplerData: samplerData,
	}

	// Case 1: Image embedded in a buffer view (common in GLB)
	if img.BufferView != nil {
		data, err := e.readBufferViewRaw(*img.BufferView)
		if err != nil {
			return nil, "", fmt.Errorf("failed to read image buffer view: %w", err)
		}
		result.Data = data
		return result, "", nil
	}

	// Case 2: Data URI (base64 encoded inline)
	if strings.HasPrefix(img.URI, "data:") {
		data, mimeType, err := gltfDecodeDataURI(img.URI)
		if err != nil {
			return nil, "", fmt.Errorf("failed to decode image data URI: %w", err)
		}
		result.Data = data
		if result.MimeType == "" {
			result.MimeType = mimeType
		}
		return result, "", nil
	}

	// Case 3: External file reference
	if img.URI != "" {
		absPath := filepath.Join(e.parser.BaseDir(), img.URI)
		result.Path = absPath

		// Attempt to load file contents
		data, err := os.ReadFile(absPath)
		if err != nil {
			// File may not be available yet; return path only
			return result, absPath, nil
		}
		result.Data = data
		return result, absPath, nil
	}

	return nil, "", nil
}

// readBufferViewRaw reads raw bytes from a buffer view by index (not through an accessor).
// This is used for image data which is stored directly in buffer views without accessor interpretation.
func (e *gltfMaterialExtractorImpl) readBufferViewRaw(bufferViewIndex int) ([]byte, error) {
	doc := e.parser.Document()
	if bufferViewIndex < 0 || bufferViewIndex >= len(doc.BufferViews) {
		return nil, fmt.Errorf("bufferView index %d out of range", bufferViewIndex)
	}

	bv := &doc.BufferViews[bufferViewIndex]
	if bv.Buffer < 0 || bv.Buffer >= len(doc.Buffers) {
		return nil, fmt.Errorf("buffer index %d out of range", bv.Buffer)
	}

	buf := &doc.Buffers[bv.Buffer]
	start := bv.ByteOffset
	length := bv.ByteLength
	end := start + length

	if end > len(buf.Data) {
		return nil, fmt.Errorf("bufferView exceeds buffer bounds: offset=%d length=%d bufSize=%d", start, length, len(buf.Data))
	}

	data := make([]byte, length)
	copy(data, buf.Data[start:end])
	return data, nil
}

// gltfDecodeDataURI decodes a base64 data URI into raw bytes and extracts the MIME type.
func gltfDecodeDataURI(uri string) ([]byte, string, error) {
	// Format: data:[<mediatype>][;base64],<data>
	if !strings.HasPrefix(uri, "data:") {
		return nil, "", fmt.Errorf("not a data URI")
	}

	commaIdx := strings.Index(uri, ",")
	if commaIdx < 0 {
		return nil, "", fmt.Errorf("malformed data URI: no comma found")
	}

	header := uri[5:commaIdx] // after "data:", before ","
	encoded := uri[commaIdx+1:]

	var mimeType string
	if strings.Contains(header, ";base64") {
		mimeType = strings.TrimSuffix(header, ";base64")
	} else {
		mimeType = header
	}

	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, "", fmt.Errorf("failed to decode base64: %w", err)
	}

	return data, mimeType, nil
}

// gltfSamplerToStagingData converts a glTF sampler definition into engine-ready SamplerStagingData.
// Any unset fields in the glTF sampler fall back to the glTF spec defaults (linear filtering, repeat wrapping).
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-sampler
//
// Parameters:
//   - s: the glTF sampler to convert
//
// Returns:
//   - *common.SamplerStagingData: the converted sampler staging data
func gltfSamplerToStagingData(s *gltfSampler) *common.SamplerStagingData {
	result := &common.SamplerStagingData{
		AddressModeU:  wgpu.AddressModeRepeat,
		AddressModeV:  wgpu.AddressModeRepeat,
		AddressModeW:  wgpu.AddressModeRepeat,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeLinear,
		LodMinClamp:   0,
		LodMaxClamp:   32,
		MaxAnisotropy: 1,
	}

	if s.MagFilter != nil {
		switch *s.MagFilter {
		case gltfFilterNearest:
			result.MagFilter = wgpu.FilterModeNearest
		case gltfFilterLinear:
			result.MagFilter = wgpu.FilterModeLinear
		}
	}

	if s.MinFilter != nil {
		switch *s.MinFilter {
		case gltfFilterNearest, gltfFilterNearestMipmapNearest, gltfFilterNearestMipmapLinear:
			result.MinFilter = wgpu.FilterModeNearest
		case gltfFilterLinear, gltfFilterLinearMipmapNearest, gltfFilterLinearMipmapLinear:
			result.MinFilter = wgpu.FilterModeLinear
		}
		// Also set the mipmap filter based on the minification filter variant
		switch *s.MinFilter {
		case gltfFilterNearestMipmapNearest, gltfFilterLinearMipmapNearest:
			result.MipmapFilter = wgpu.MipmapFilterModeNearest
		case gltfFilterNearestMipmapLinear, gltfFilterLinearMipmapLinear:
			result.MipmapFilter = wgpu.MipmapFilterModeLinear
		case gltfFilterNearest, gltfFilterLinear:
			// Non-mipmapped filters: set mipmap to nearest as a conservative default
			result.MipmapFilter = wgpu.MipmapFilterModeNearest
		}
	}

	if s.WrapS != nil {
		result.AddressModeU = gltfWrapToAddressMode(*s.WrapS)
	}
	if s.WrapT != nil {
		result.AddressModeV = gltfWrapToAddressMode(*s.WrapT)
	}

	return result
}

// gltfWrapToAddressMode converts a glTF wrap mode constant to a wgpu AddressMode.
//
// Parameters:
//   - wrap: the glTF wrap mode constant
//
// Returns:
//   - wgpu.AddressMode: the corresponding wgpu address mode
func gltfWrapToAddressMode(wrap int) wgpu.AddressMode {
	switch wrap {
	case gltfWrapClampToEdge:
		return wgpu.AddressModeClampToEdge
	case gltfWrapMirroredRepeat:
		return wgpu.AddressModeMirrorRepeat
	case gltfWrapRepeat:
		return wgpu.AddressModeRepeat
	default:
		return wgpu.AddressModeRepeat
	}
}
