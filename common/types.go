// package common contains common types that are used throughout this engine. They are not interface-wrapped structs, just plain structs that express
// commonly used data-types.
package common

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/cogentcore/webgpu/wgpu"
)

// TextureStagingData holds RGBA pixel data for a texture binding pending GPU upload.
// This is primarily used in the BindGroupProvider to stage texture data before creating the GPU texture and bind group.
type TextureStagingData struct {
	// Pixels is the byte slice representing the actual pixel data for the texture. It should be in RGBA format, with 4 bytes per pixel.
	Pixels []byte
	// Width is the width of the texture in pixels. This is required to correctly create the GPU texture and interpret the pixel data.
	Width uint32
	// Height is the height of the texture in pixels. This is required to correctly create the GPU texture and interpret the pixel data.
	Height uint32
}

// SamplerStagingData holds the configuration for a sampler binding pending GPU creation.
// This is primarily used in the BindGroupProvider to stage sampler data before creating the GPU sampler and bind group.
type SamplerStagingData struct {
	// AddressModeU, AddressModeV, AddressModeW specify the addressing mode for texture coordinates outside the [0, 1] range in each dimension (U, V, W).
	AddressModeU, AddressModeV, AddressModeW wgpu.AddressMode
	// MagFilter and MinFilter specify the filtering mode for magnification and minification.
	MagFilter, MinFilter wgpu.FilterMode
	// MipmapFilter specifies the filtering mode for mipmap level selection.
	MipmapFilter wgpu.MipmapFilterMode
	// LodMinClamp and LodMaxClamp specify the minimum and maximum level of detail (LOD) for mipmapping.
	LodMinClamp, LodMaxClamp float32
	// Compare specifies the comparison function for comparison samplers, used in shadow mapping and similar techniques.
	Compare wgpu.CompareFunction
	// MaxAnisotropy specifies the maximum anisotropy level for anisotropic filtering, which can improve texture quality at oblique viewing angles.
	MaxAnisotropy uint16
}

// ImportedMaterial represents material properties from an imported model file.
type ImportedMaterial struct {
	// Name is the material identifier.
	Name string

	// BaseColor is the albedo/diffuse color (RGBA).
	BaseColor [4]float32

	// Metallic factor (0.0 = dielectric, 1.0 = metal).
	Metallic float32

	// Roughness factor (0.0 = smooth, 1.0 = rough).
	Roughness float32

	// DiffuseTexturePath is the file path for the diffuse/albedo texture.
	DiffuseTexturePath string

	// NormalTexturePath is the file path for the normal map texture.
	NormalTexturePath string

	// MetallicTexturePath is the file path for the metallic-roughness texture.
	MetallicTexturePath string

	// DiffuseTexture holds embedded texture data (if present).
	DiffuseTexture *ImportedTexture

	// NormalTexture holds embedded normal map data (if present).
	NormalTexture *ImportedTexture

	// MetallicRoughnessTexture holds embedded metallic/roughness data (if present).
	MetallicRoughnessTexture *ImportedTexture
}

// ImportedTexture represents texture data extracted from a model file.
// For embedded textures (GLB), the Data field contains raw image bytes.
// For external textures, the Path field contains the file path.
type ImportedTexture struct {
	// Name is an identifier for this texture (e.g., "diffuse", "normal").
	Name string

	// Path is the file path for external textures (empty for embedded).
	Path string

	// Data contains raw image bytes for embedded textures (PNG/JPEG).
	Data []byte

	// MimeType indicates the image format (e.g., "image/png", "image/jpeg").
	MimeType string

	// Width is the texture width in pixels (populated after Decode).
	Width int

	// Height is the texture height in pixels (populated after Decode).
	Height int

	// SamplerData holds GPU sampler parameters extracted from the model file.
	// When non-nil, these values override the default linear/repeat settings
	// used during material GPU initialization.
	SamplerData *SamplerStagingData
}

// Decode decodes the texture to raw RGBA pixel data.
// Uses either embedded Data bytes or loads from Path on disk.
// Supports PNG and JPEG formats.
// Reference: https://pkg.go.dev/image
//
// Returns:
//   - []byte: raw RGBA pixel data (4 bytes per pixel, row-major order)
//   - uint32: texture width in pixels
//   - uint32: texture height in pixels
//   - error: error if decoding fails
func (t *ImportedTexture) Decode() ([]byte, uint32, uint32, error) {
	if t == nil {
		return nil, 0, 0, fmt.Errorf("texture is nil")
	}

	var img image.Image
	var err error

	if len(t.Data) > 0 {
		img, _, err = image.Decode(bytes.NewReader(t.Data))
		if err != nil {
			return nil, 0, 0, fmt.Errorf("failed to decode embedded image: %w", err)
		}
	} else if t.Path != "" {
		file, fileErr := os.Open(t.Path)
		if fileErr != nil {
			return nil, 0, 0, fmt.Errorf("failed to open texture file %s: %w", t.Path, fileErr)
		}
		defer file.Close()

		img, _, err = image.Decode(file)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("failed to decode texture file %s: %w", t.Path, err)
		}
	} else {
		return nil, 0, 0, fmt.Errorf("texture has neither data nor path")
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)

	t.Width = width
	t.Height = height

	return rgba.Pix, uint32(width), uint32(height), nil
}
