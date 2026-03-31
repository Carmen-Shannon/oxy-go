package common

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/suite"
)

func TestRunCommonTests(t *testing.T) {
	suite.Run(t, new(commonTest))
}

type commonTest struct {
	suite.Suite
}

func (suite *commonTest) TestDelegate() {
	suite.Run("should set delegate", func() {
		type dummyInterface interface {
			Delegate[dummyInterface]
		}
		type dummyImpl struct {
			DelegateImpl[dummyInterface]
		}
		d := &dummyImpl{}
		d.Delegate = d
		d.SetDelegate(d)
		suite.NotNil(d.Delegate)
	})
}

func (suite *commonTest) TestFrustum() {
	suite.Run("should return a Frustum from a view projection matrix", func() {
		viewProj := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		frustum := ExtractFrustumFromMatrix(viewProj)
		suite.NotEmpty(frustum)
	})

	suite.Run("IntersectSphere returns true for a sphere inside the frustum", func() {
		view := make([]float32, 16)
		proj := make([]float32, 16)
		vp := make([]float32, 16)
		LookAt(view, 0, 0, 5, 0, 0, 0, 0, 1, 0)
		Perspective(proj, float32(math.Pi/2), 1.0, 0.1, 100.0)
		Mul4(vp, proj, view)
		f := ExtractFrustumFromMatrix(vp)
		suite.True(f.IntersectSphere([3]float32{0, 0, 0}, 1.0))
	})

	suite.Run("IntersectSphere returns false for a sphere behind the camera", func() {
		view := make([]float32, 16)
		proj := make([]float32, 16)
		vp := make([]float32, 16)
		LookAt(view, 0, 0, 5, 0, 0, 0, 0, 1, 0)
		Perspective(proj, float32(math.Pi/2), 1.0, 0.1, 100.0)
		Mul4(vp, proj, view)
		f := ExtractFrustumFromMatrix(vp)
		suite.False(f.IntersectSphere([3]float32{0, 0, 1000}, 0.1))
	})

	suite.Run("IntersectAABB returns true for an AABB inside the frustum", func() {
		view := make([]float32, 16)
		proj := make([]float32, 16)
		vp := make([]float32, 16)
		LookAt(view, 0, 0, 5, 0, 0, 0, 0, 1, 0)
		Perspective(proj, float32(math.Pi/2), 1.0, 0.1, 100.0)
		Mul4(vp, proj, view)
		f := ExtractFrustumFromMatrix(vp)
		suite.True(f.IntersectAABB([3]float32{-1, -1, -1}, [3]float32{1, 1, 1}))
	})

	suite.Run("IntersectAABB returns false for an AABB entirely behind the camera", func() {
		view := make([]float32, 16)
		proj := make([]float32, 16)
		vp := make([]float32, 16)
		LookAt(view, 0, 0, 5, 0, 0, 0, 0, 1, 0)
		Perspective(proj, float32(math.Pi/2), 1.0, 0.1, 100.0)
		Mul4(vp, proj, view)
		f := ExtractFrustumFromMatrix(vp)
		suite.False(f.IntersectAABB([3]float32{0, 0, 1000}, [3]float32{1, 1, 1001}))
	})
}

func (suite *commonTest) TestMath() {
	type testStruct struct {
		A int
		B float32
	}

	suite.Run("should reset a 4x4 matrix to the identity matrix with column-major order", func() {
		m := make([]float32, 16)
		Identity(m)
		expected := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		suite.Equal(expected, m)
	})

	suite.Run("should convert a slice of any type to a byte slice", func() {
		data := []testStruct{
			{A: 1, B: 2.5},
			{A: 3, B: 4.5},
		}
		bytes := SliceToBytes(data)
		suite.NotNil(bytes)
		suite.Equal(len(data)*int(unsafe.Sizeof(testStruct{})), len(bytes))
	})

	suite.Run("should return nil when converting an empty slice to bytes", func() {
		var data []testStruct
		bytes := SliceToBytes(data)
		suite.Nil(bytes)
	})

	suite.Run("should convert a struct of any type to a byte slice", func() {
		v := testStruct{A: 1, B: 2.5}
		bytes := StructToBytes(&v)
		suite.NotNil(bytes)
		suite.Equal(int(unsafe.Sizeof(testStruct{})), len(bytes))
	})

	suite.Run("should multiply two 4x4 matrices in column-major order", func() {
		a := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		b := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			1, 2, 3, 1,
		}
		out := make([]float32, 16)
		Mul4(out, a, b)
		expected := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			1, 2, 3, 1,
		}
		suite.Equal(expected, out)
	})

	suite.Run("should produce a perspective projection matrix", func() {
		fovY := float32(math.Pi / 2)
		aspect := float32(1.0)
		near := float32(1.0)
		far := float32(10.0)
		out := make([]float32, 16)
		Perspective(out, fovY, aspect, near, far)
		expected := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, float32(-10.0 / 9.0), -1,
			0, 0, float32(-10.0 / 9.0), 0,
		}
		for i := range expected {
			suite.InDelta(expected[i], out[i], 1e-5)
		}
	})

	suite.Run("should produce a model matrix with translation and no rotation", func() {
		out := make([]float32, 16)
		BuildModelMatrix(out, 2, 3, 4, 0, 0, 0, 1, 1, 1)
		expected := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			2, 3, 4, 1,
		}
		for i := range expected {
			suite.InDelta(expected[i], out[i], 1e-5)
		}
	})

	suite.Run("should invert the identity matrix to itself", func() {
		m := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		out := make([]float32, 16)
		ok := Invert4(out, m)
		suite.Equal(true, ok)
		expected := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		for i := range expected {
			suite.InDelta(expected[i], out[i], 1e-5)
		}
	})

	suite.Run("should return false for a singular matrix", func() {
		m := make([]float32, 16)
		out := make([]float32, 16)
		ok := Invert4(out, m)
		suite.Equal(false, ok)
		expected := make([]float32, 16)
		for i := range expected {
			suite.InDelta(expected[i], out[i], 1e-5)
		}
	})

	suite.Run("should produce a view matrix looking along the negative Z axis", func() {
		out := make([]float32, 16)
		LookAt(out, 0, 0, 5, 0, 0, 0, 0, 1, 0)
		expected := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, -5, 1,
		}
		for i := range expected {
			suite.InDelta(expected[i], out[i], 1e-5)
		}
	})

	suite.Run("should not panic when eye equals center (zero-length z vector)", func() {
		out := make([]float32, 16)
		suite.NotPanics(func() {
			LookAt(out, 0, 0, 0, 0, 0, 0, 0, 1, 0)
		})
		suite.Equal(float32(1), out[15])
	})

	suite.Run("should invert the 3x3 identity matrix to itself", func() {
		m := [9]float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
		result, ok := Invert3x3(m)
		suite.True(ok)
		expected := [9]float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
		for i := range expected {
			suite.InDelta(expected[i], result[i], 1e-5)
		}
	})

	suite.Run("should return false for a singular 3x3 matrix", func() {
		m := [9]float32{}
		_, ok := Invert3x3(m)
		suite.False(ok)
	})
}

func (suite *commonTest) TestTypes() {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 0, B: 128, A: 255})
	var buf bytes.Buffer
	err := png.Encode(&buf, img)
	suite.NoError(err)

	suite.Run("should decode an ImportedTexture into it's raw rgba pixel data with width and height", func() {
		tex := &ImportedTexture{
			Name: "test-texture",
			Data: buf.Bytes(),
		}

		pixels, width, height, err := tex.Decode()
		suite.NoError(err)
		suite.Equal(uint32(1), width)
		suite.Equal(uint32(1), height)
		suite.Equal(4, len(pixels))
		suite.Equal([]byte{255, 0, 128, 255}, pixels)
	})

	suite.Run("should decode an ImportedTexture from a file path into raw rgba pixel data with width and height", func() {
		tmpDir := suite.T().TempDir()
		path := filepath.Join(tmpDir, "test.png")
		err := os.WriteFile(path, buf.Bytes(), 0644)
		suite.NoError(err)

		tex := &ImportedTexture{
			Name: "test-texture",
			Path: path,
		}

		pixels, width, height, err := tex.Decode()
		suite.NoError(err)
		suite.Equal(uint32(1), width)
		suite.Equal(uint32(1), height)
		suite.Equal(4, len(pixels))
		suite.Equal([]byte{255, 0, 128, 255}, pixels)
	})

	suite.Run("should return an error when called with a nil ImportedTexture", func() {
		var tex *ImportedTexture
		_, _, _, err := tex.Decode()
		suite.ErrorContains(err, "texture is nil")
	})

	suite.Run("should return an error when the ImportedTexture data is invalid", func() {
		tex := &ImportedTexture{
			Name: "test-texture",
			Data: []byte{1, 2, 3, 4},
		}
		_, _, _, err := tex.Decode()
		suite.ErrorContains(err, "failed to decode embedded image")
	})

	suite.Run("should return an error when the ImportedTexture file path is invalid", func() {
		tex := &ImportedTexture{
			Name: "test-texture",
			Path: "nonexistent.png",
		}
		_, _, _, err := tex.Decode()
		suite.ErrorContains(err, "failed to open texture file")
	})

	suite.Run("should return an error when the ImportedTexture path is an invalid image", func() {
		tmpDir := suite.T().TempDir()
		path := filepath.Join(tmpDir, "invalid.png")
		err := os.WriteFile(path, []byte{1, 2, 3, 4}, 0644)
		suite.NoError(err)

		tex := &ImportedTexture{
			Name: "test-texture",
			Path: path,
		}
		_, _, _, err = tex.Decode()
		suite.ErrorContains(err, "failed to decode texture file")
	})

	suite.Run("should return an error when the ImportedTexture has neither data nor path", func() {
		tex := &ImportedTexture{
			Name: "test-texture",
		}
		_, _, _, err := tex.Decode()
		suite.ErrorContains(err, "texture has neither data nor path")
	})
}

func (suite *commonTest) TestUtils() {
	suite.Run("should return the first non-zero value from a list of values", func() {
		result := Coalesce(0, 0, 5, 0, 10)
		suite.Equal(5, result)
	})

	suite.Run("should return the zero value if all values are zero", func() {
		result := Coalesce(0, 0, 0)
		suite.Equal(0, result)
	})
}
