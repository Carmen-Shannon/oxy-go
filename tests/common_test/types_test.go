package common_test

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

func TestTypes(t *testing.T) {
	suite.Run(t, new(typesTest))
}

type typesTest struct {
	suite.Suite
	tmpDir string
}

func (suite *typesTest) SetupSuite() {
	dir, err := os.MkdirTemp("", "types_test")
	suite.Require().NoError(err)
	suite.tmpDir = dir
}

func (suite *typesTest) TearDownSuite() {
	os.RemoveAll(suite.tmpDir)
}

func (suite *typesTest) TestTextureStagingData() {
	suite.Run("fields are set correctly", func() {
		pixels := []byte{255, 0, 0, 255, 0, 255, 0, 255}
		data := common.TextureStagingData{
			Pixels: pixels,
			Width:  2,
			Height: 1,
		}
		suite.Equal(uint32(2), data.Width)
		suite.Equal(uint32(1), data.Height)
		suite.Len(data.Pixels, 8)
	})

	suite.Run("zero value has nil pixels and zero dimensions", func() {
		var data common.TextureStagingData
		suite.Nil(data.Pixels)
		suite.Equal(uint32(0), data.Width)
		suite.Equal(uint32(0), data.Height)
	})
}

func (suite *typesTest) TestSamplerStagingData() {
	suite.Run("fields are set correctly", func() {
		data := common.SamplerStagingData{
			AddressModeU:  wgpu.AddressModeRepeat,
			AddressModeV:  wgpu.AddressModeMirrorRepeat,
			AddressModeW:  wgpu.AddressModeClampToEdge,
			MagFilter:     wgpu.FilterModeLinear,
			MinFilter:     wgpu.FilterModeNearest,
			MipmapFilter:  wgpu.MipmapFilterModeLinear,
			LodMinClamp:   0.0,
			LodMaxClamp:   32.0,
			Compare:       wgpu.CompareFunctionLess,
			MaxAnisotropy: 16,
		}
		suite.Equal(wgpu.AddressModeRepeat, data.AddressModeU)
		suite.Equal(wgpu.AddressModeMirrorRepeat, data.AddressModeV)
		suite.Equal(wgpu.AddressModeClampToEdge, data.AddressModeW)
		suite.Equal(wgpu.FilterModeLinear, data.MagFilter)
		suite.Equal(wgpu.FilterModeNearest, data.MinFilter)
		suite.Equal(wgpu.MipmapFilterModeLinear, data.MipmapFilter)
		suite.Equal(float32(0.0), data.LodMinClamp)
		suite.Equal(float32(32.0), data.LodMaxClamp)
		suite.Equal(wgpu.CompareFunctionLess, data.Compare)
		suite.Equal(uint16(16), data.MaxAnisotropy)
	})

	suite.Run("zero value uses default enum values", func() {
		var data common.SamplerStagingData
		suite.Equal(float32(0), data.LodMinClamp)
		suite.Equal(float32(0), data.LodMaxClamp)
		suite.Equal(uint16(0), data.MaxAnisotropy)
	})
}

func (suite *typesTest) TestImportedMaterial() {
	suite.Run("fields are set correctly", func() {
		mat := common.ImportedMaterial{
			Name:      "test_material",
			BaseColor: [4]float32{1.0, 0.5, 0.25, 1.0},
			Metallic:  0.8,
			Roughness: 0.2,
		}
		suite.Equal("test_material", mat.Name)
		suite.InDelta(float32(1.0), mat.BaseColor[0], 1e-6)
		suite.InDelta(float32(0.5), mat.BaseColor[1], 1e-6)
		suite.InDelta(float32(0.25), mat.BaseColor[2], 1e-6)
		suite.InDelta(float32(1.0), mat.BaseColor[3], 1e-6)
		suite.InDelta(float32(0.8), mat.Metallic, 1e-6)
		suite.InDelta(float32(0.2), mat.Roughness, 1e-6)
	})

	suite.Run("texture pointers are nil by default", func() {
		var mat common.ImportedMaterial
		suite.Nil(mat.DiffuseTexture)
		suite.Nil(mat.NormalTexture)
		suite.Nil(mat.MetallicRoughnessTexture)
	})

	suite.Run("texture paths are empty by default", func() {
		var mat common.ImportedMaterial
		suite.Empty(mat.DiffuseTexturePath)
		suite.Empty(mat.NormalTexturePath)
		suite.Empty(mat.MetallicTexturePath)
	})
}

func (suite *typesTest) TestDecode() {
	suite.Run("nil texture returns error", func() {
		var t *common.ImportedTexture
		pixels, w, h, err := t.Decode()
		suite.Error(err)
		suite.Nil(pixels)
		suite.Equal(uint32(0), w)
		suite.Equal(uint32(0), h)
		suite.Contains(err.Error(), "nil")
	})

	suite.Run("texture with neither data nor path returns error", func() {
		t := &common.ImportedTexture{Name: "empty"}
		pixels, w, h, err := t.Decode()
		suite.Error(err)
		suite.Nil(pixels)
		suite.Equal(uint32(0), w)
		suite.Equal(uint32(0), h)
		suite.Contains(err.Error(), "neither data nor path")
	})

	suite.Run("embedded PNG data decodes correctly", func() {
		// Create a 4x2 red PNG in memory
		pngData := encodePNG(4, 2, color.NRGBA{R: 255, G: 0, B: 0, A: 255})
		t := &common.ImportedTexture{
			Name:     "red_texture",
			Data:     pngData,
			MimeType: "image/png",
		}
		pixels, w, h, err := t.Decode()
		suite.NoError(err)
		suite.Equal(uint32(4), w)
		suite.Equal(uint32(2), h)
		// 4 * 2 * 4 bytes (RGBA) = 32 bytes total
		suite.Len(pixels, 32)
	})

	suite.Run("decode sets Width and Height on the texture struct", func() {
		pngData := encodePNG(8, 6, color.NRGBA{R: 0, G: 255, B: 0, A: 255})
		t := &common.ImportedTexture{
			Name: "green_texture",
			Data: pngData,
		}
		_, _, _, err := t.Decode()
		suite.NoError(err)
		suite.Equal(8, t.Width)
		suite.Equal(6, t.Height)
	})

	suite.Run("embedded RGBA pixels match expected color", func() {
		// Single white pixel
		pngData := encodePNG(1, 1, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
		t := &common.ImportedTexture{Data: pngData}
		pixels, _, _, err := t.Decode()
		suite.NoError(err)
		suite.Len(pixels, 4)
		suite.Equal(byte(255), pixels[0]) // R
		suite.Equal(byte(255), pixels[1]) // G
		suite.Equal(byte(255), pixels[2]) // B
		suite.Equal(byte(255), pixels[3]) // A
	})

	suite.Run("invalid embedded data returns error", func() {
		t := &common.ImportedTexture{
			Data: []byte{0xFF, 0xFE, 0xFD, 0xFC},
		}
		pixels, w, h, err := t.Decode()
		suite.Error(err)
		suite.Nil(pixels)
		suite.Equal(uint32(0), w)
		suite.Equal(uint32(0), h)
		suite.Contains(err.Error(), "decode embedded image")
	})

	suite.Run("file path PNG decodes correctly", func() {
		pngData := encodePNG(3, 3, color.NRGBA{R: 0, G: 0, B: 255, A: 255})
		path := filepath.Join(suite.tmpDir, "blue.png")
		err := os.WriteFile(path, pngData, 0644)
		suite.Require().NoError(err)

		t := &common.ImportedTexture{Path: path}
		pixels, w, h, err := t.Decode()
		suite.NoError(err)
		suite.Equal(uint32(3), w)
		suite.Equal(uint32(3), h)
		// 3 * 3 * 4 = 36 bytes
		suite.Len(pixels, 36)
	})

	suite.Run("file path sets Width and Height on the texture struct", func() {
		pngData := encodePNG(5, 7, color.NRGBA{R: 128, G: 128, B: 128, A: 255})
		path := filepath.Join(suite.tmpDir, "gray.png")
		err := os.WriteFile(path, pngData, 0644)
		suite.Require().NoError(err)

		t := &common.ImportedTexture{Path: path}
		_, _, _, decErr := t.Decode()
		suite.NoError(decErr)
		suite.Equal(5, t.Width)
		suite.Equal(7, t.Height)
	})

	suite.Run("nonexistent file path returns error", func() {
		t := &common.ImportedTexture{Path: filepath.Join(suite.tmpDir, "does_not_exist.png")}
		pixels, w, h, err := t.Decode()
		suite.Error(err)
		suite.Nil(pixels)
		suite.Equal(uint32(0), w)
		suite.Equal(uint32(0), h)
		suite.Contains(err.Error(), "failed to open texture file")
	})

	suite.Run("file with invalid image data returns error", func() {
		path := filepath.Join(suite.tmpDir, "garbage.png")
		err := os.WriteFile(path, []byte{0x00, 0x01, 0x02}, 0644)
		suite.Require().NoError(err)

		t := &common.ImportedTexture{Path: path}
		pixels, w, h, decErr := t.Decode()
		suite.Error(decErr)
		suite.Nil(pixels)
		suite.Equal(uint32(0), w)
		suite.Equal(uint32(0), h)
		suite.Contains(decErr.Error(), "failed to decode texture file")
	})

	suite.Run("embedded data takes priority over path", func() {
		// Data is valid, Path points to nonexistent file — should succeed via Data
		pngData := encodePNG(2, 2, color.NRGBA{R: 255, G: 0, B: 255, A: 255})
		t := &common.ImportedTexture{
			Data: pngData,
			Path: filepath.Join(suite.tmpDir, "nonexistent.png"),
		}
		pixels, w, h, err := t.Decode()
		suite.NoError(err)
		suite.Equal(uint32(2), w)
		suite.Equal(uint32(2), h)
		suite.Len(pixels, 16)
	})
}

// encodePNG creates a PNG-encoded byte slice for a solid-color image with the given dimensions.
//
// Parameters:
//   - width: image width in pixels
//   - height: image height in pixels
//   - c: the solid color to fill every pixel with
//
// Returns:
//   - []byte: PNG-encoded image data
func encodePNG(width, height int, c color.NRGBA) []byte {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}
