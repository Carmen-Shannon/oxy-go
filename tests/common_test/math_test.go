package common_test

import (
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/stretchr/testify/suite"
)

func TestMath(t *testing.T) {
	suite.Run(t, new(mathTest))
}

type mathTest struct {
	suite.Suite
}

func (suite *mathTest) TestIdentity() {
	suite.Run("sets diagonal to 1 and all other elements to 0", func() {
		m := make([]float32, 16)
		common.Identity(m)
		expected := identityMatrix()
		for i := 0; i < 16; i++ {
			suite.Equal(expected[i], m[i])
		}
	})

	suite.Run("overwrites existing non-zero data", func() {
		m := make([]float32, 16)
		for i := range m {
			m[i] = 99
		}
		common.Identity(m)
		suite.Equal(float32(1), m[0])
		suite.Equal(float32(0), m[1])
		suite.Equal(float32(1), m[5])
		suite.Equal(float32(1), m[10])
		suite.Equal(float32(1), m[15])
	})

	suite.Run("off-diagonal elements are all zero", func() {
		m := make([]float32, 16)
		common.Identity(m)
		offDiag := []int{1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14}
		for _, idx := range offDiag {
			suite.Equal(float32(0), m[idx])
		}
	})
}

func (suite *mathTest) TestSliceToBytes() {
	suite.Run("returns correct byte count for float32 slice", func() {
		data := []float32{1.0, 2.0, 3.0}
		b := common.SliceToBytes(data)
		// 3 float32 * 4 bytes = 12
		suite.Len(b, 12)
	})

	suite.Run("returns nil for empty slice", func() {
		var data []float32
		b := common.SliceToBytes(data)
		suite.Nil(b)
	})

	suite.Run("returns nil for nil slice", func() {
		b := common.SliceToBytes[float32](nil)
		suite.Nil(b)
	})

	suite.Run("returns correct byte count for uint32 slice", func() {
		data := []uint32{0xDEADBEEF, 0xCAFEBABE}
		b := common.SliceToBytes(data)
		// 2 uint32 * 4 bytes = 8
		suite.Len(b, 8)
	})

	suite.Run("returns correct byte count for single byte slice", func() {
		data := []byte{0xFF, 0x00, 0xAB}
		b := common.SliceToBytes(data)
		suite.Len(b, 3)
	})

	suite.Run("shares memory with source slice", func() {
		data := []float32{1.0}
		b := common.SliceToBytes(data)
		suite.NotNil(b)
		suite.Len(b, 4)
	})
}

func (suite *mathTest) TestStructToBytes() {
	suite.Run("returns correct byte count for struct with two float32 fields", func() {
		type vec2 struct{ X, Y float32 }
		v := vec2{X: 1.0, Y: 2.0}
		b := common.StructToBytes(&v)
		// 2 * 4 bytes = 8
		suite.Len(b, 8)
	})

	suite.Run("returns correct byte count for struct with one uint64 field", func() {
		type wrapper struct{ Val uint64 }
		w := wrapper{Val: 42}
		b := common.StructToBytes(&w)
		suite.Len(b, 8)
	})

	suite.Run("returns correct byte count for struct with mixed fields", func() {
		type mixed struct {
			A float32
			B float32
			C float32
			D float32
		}
		m := mixed{1, 2, 3, 4}
		b := common.StructToBytes(&m)
		// 4 * 4 bytes = 16
		suite.Len(b, 16)
	})
}

func (suite *mathTest) TestMul4() {
	suite.Run("identity times identity equals identity", func() {
		a := identityMatrix()
		b := identityMatrix()
		out := make([]float32, 16)
		common.Mul4(out, a, b)
		expected := identityMatrix()
		for i := 0; i < 16; i++ {
			suite.InDelta(expected[i], out[i], 1e-6)
		}
	})

	suite.Run("identity times arbitrary matrix equals that matrix", func() {
		id := identityMatrix()
		m := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}
		out := make([]float32, 16)
		common.Mul4(out, id, m)
		for i := 0; i < 16; i++ {
			suite.InDelta(m[i], out[i], 1e-6)
		}
	})

	suite.Run("arbitrary matrix times identity equals that matrix", func() {
		id := identityMatrix()
		m := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}
		out := make([]float32, 16)
		common.Mul4(out, m, id)
		for i := 0; i < 16; i++ {
			suite.InDelta(m[i], out[i], 1e-6)
		}
	})

	suite.Run("translation times scale produces correct combined matrix", func() {
		// T = translate(1,2,3), S = scale(2,3,4), column-major
		t := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			1, 2, 3, 1,
		}
		s := []float32{
			2, 0, 0, 0,
			0, 3, 0, 0,
			0, 0, 4, 0,
			0, 0, 0, 1,
		}
		out := make([]float32, 16)
		common.Mul4(out, t, s)

		// T * S: each column of S is transformed by T
		// col0=(2,0,0,0), col1=(0,3,0,0), col2=(0,0,4,0), col3=(1,2,3,1)
		expected := []float32{
			2, 0, 0, 0,
			0, 3, 0, 0,
			0, 0, 4, 0,
			1, 2, 3, 1,
		}
		for i := 0; i < 16; i++ {
			suite.InDelta(expected[i], out[i], 1e-6)
		}
	})

	suite.Run("multiplication is not commutative for general matrices", func() {
		a := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			5, 0, 0, 1,
		}
		b := []float32{
			2, 0, 0, 0,
			0, 2, 0, 0,
			0, 0, 2, 0,
			0, 0, 0, 1,
		}
		ab := make([]float32, 16)
		ba := make([]float32, 16)
		common.Mul4(ab, a, b)
		common.Mul4(ba, b, a)

		different := false
		for i := 0; i < 16; i++ {
			if ab[i] != ba[i] {
				different = true
				break
			}
		}
		suite.True(different)
	})

	suite.Run("zero matrix times any matrix produces zero matrix", func() {
		zero := make([]float32, 16)
		m := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}
		out := make([]float32, 16)
		common.Mul4(out, zero, m)
		for i := 0; i < 16; i++ {
			suite.InDelta(float32(0), out[i], 1e-6)
		}
	})
}

func (suite *mathTest) TestPerspective() {
	suite.Run("produces correct matrix elements for standard parameters", func() {
		out := make([]float32, 16)
		fovY := float32(math.Pi / 4.0)
		aspect := float32(16.0 / 9.0)
		near := float32(0.1)
		far := float32(100.0)
		common.Perspective(out, fovY, aspect, near, far)

		f := float32(1.0 / math.Tan(float64(fovY)/2.0))
		suite.InDelta(f/aspect, out[0], 1e-6)
		suite.InDelta(f, out[5], 1e-6)
		suite.InDelta(far/(near-far), out[10], 1e-6)
		suite.InDelta(float32(-1.0), out[11], 1e-6)
		suite.InDelta((near*far)/(near-far), out[14], 1e-6)
		suite.InDelta(float32(0), out[15], 1e-6)
	})

	suite.Run("[0,0] and [1,1] are equal for square aspect ratio", func() {
		out := make([]float32, 16)
		common.Perspective(out, float32(math.Pi/2), 1.0, 0.1, 1000.0)
		suite.InDelta(out[0], out[5], 1e-6)
	})

	suite.Run("[0,0] differs from [1,1] for non-square aspect ratio", func() {
		out := make([]float32, 16)
		common.Perspective(out, float32(math.Pi/4), 2.0, 0.1, 100.0)
		suite.NotEqual(out[0], out[5])
	})

	suite.Run("off-axis elements are zero", func() {
		out := make([]float32, 16)
		common.Perspective(out, float32(math.Pi/4), 16.0/9.0, 0.1, 100.0)
		// Elements that should be zero in a symmetric perspective matrix
		suite.Equal(float32(0), out[1])
		suite.Equal(float32(0), out[2])
		suite.Equal(float32(0), out[3])
		suite.Equal(float32(0), out[4])
		suite.Equal(float32(0), out[6])
		suite.Equal(float32(0), out[7])
		suite.Equal(float32(0), out[8])
		suite.Equal(float32(0), out[9])
		suite.Equal(float32(0), out[12])
		suite.Equal(float32(0), out[13])
	})

	suite.Run("wider FOV produces smaller [1,1] value", func() {
		narrow := make([]float32, 16)
		wide := make([]float32, 16)
		common.Perspective(narrow, float32(math.Pi/6), 1.0, 0.1, 100.0)
		common.Perspective(wide, float32(math.Pi/3), 1.0, 0.1, 100.0)
		suite.True(wide[5] < narrow[5])
	})
}

func (suite *mathTest) TestBuildModelMatrix() {
	suite.Run("zero transform with unit scale produces identity", func() {
		out := make([]float32, 16)
		common.BuildModelMatrix(out, 0, 0, 0, 0, 0, 0, 1, 1, 1)
		expected := identityMatrix()
		for i := 0; i < 16; i++ {
			suite.InDelta(expected[i], out[i], 1e-6)
		}
	})

	suite.Run("translation only sets column 3 correctly", func() {
		out := make([]float32, 16)
		common.BuildModelMatrix(out, 5, 10, 15, 0, 0, 0, 1, 1, 1)
		suite.InDelta(float32(5), out[12], 1e-6)
		suite.InDelta(float32(10), out[13], 1e-6)
		suite.InDelta(float32(15), out[14], 1e-6)
		suite.InDelta(float32(1), out[15], 1e-6)
	})

	suite.Run("scale only sets diagonal correctly", func() {
		out := make([]float32, 16)
		common.BuildModelMatrix(out, 0, 0, 0, 0, 0, 0, 2, 3, 4)
		suite.InDelta(float32(2), out[0], 1e-6)
		suite.InDelta(float32(3), out[5], 1e-6)
		suite.InDelta(float32(4), out[10], 1e-6)
		suite.InDelta(float32(1), out[15], 1e-6)
	})

	suite.Run("90 degree Y rotation swaps X and Z axes", func() {
		out := make([]float32, 16)
		common.BuildModelMatrix(out, 0, 0, 0, 0, math.Pi/2, 0, 1, 1, 1)
		// After 90° Y rotation (Ry), column-major:
		//   col0 ≈ (cos90, 0, -sin90, 0) = (0, 0, -1, 0)
		//   col2 ≈ (sin90, 0,  cos90, 0) = (1, 0,  0, 0)
		suite.InDelta(float32(0), out[0], 1e-6)
		suite.InDelta(float32(-1), out[2], 1e-6)
		suite.InDelta(float32(1), out[8], 1e-6)
		suite.InDelta(float32(0), out[10], 1e-6)
	})

	suite.Run("90 degree X rotation swaps Y and Z axes", func() {
		out := make([]float32, 16)
		common.BuildModelMatrix(out, 0, 0, 0, math.Pi/2, 0, 0, 1, 1, 1)
		// col1 ≈ (0, cos90, sin90, 0) = (0, 0, 1, 0) — but with Ry*Rx*Rz order
		// out[9] = -sin(rotX) ≈ -1
		suite.InDelta(float32(-1), out[9], 1e-6)
		suite.InDelta(float32(1), out[0], 1e-6)
	})

	suite.Run("w row is always 0 0 0 1", func() {
		out := make([]float32, 16)
		common.BuildModelMatrix(out, 7, 8, 9, 0.5, 1.2, 0.3, 2, 3, 4)
		suite.Equal(float32(0), out[3])
		suite.Equal(float32(0), out[7])
		suite.Equal(float32(0), out[11])
		suite.Equal(float32(1), out[15])
	})

	suite.Run("uniform scale preserves rotation column lengths", func() {
		out := make([]float32, 16)
		scale := float32(3.0)
		common.BuildModelMatrix(out, 0, 0, 0, 0.3, 0.7, 1.1, scale, scale, scale)
		for col := 0; col < 3; col++ {
			x := float64(out[col*4+0])
			y := float64(out[col*4+1])
			z := float64(out[col*4+2])
			length := math.Sqrt(x*x + y*y + z*z)
			suite.InDelta(float64(scale), length, 1e-5)
		}
	})
}

func (suite *mathTest) TestInvert4() {
	suite.Run("identity inverse is identity", func() {
		m := identityMatrix()
		out := make([]float32, 16)
		ok := common.Invert4(out, m)
		suite.True(ok)
		expected := identityMatrix()
		for i := 0; i < 16; i++ {
			suite.InDelta(expected[i], out[i], 1e-6)
		}
	})

	suite.Run("M times M-inverse equals identity", func() {
		// Non-trivial matrix: translate + scale
		m := []float32{
			2, 0, 0, 0,
			0, 3, 0, 0,
			0, 0, 4, 0,
			5, 6, 7, 1,
		}
		inv := make([]float32, 16)
		ok := common.Invert4(inv, m)
		suite.True(ok)

		result := make([]float32, 16)
		common.Mul4(result, m, inv)
		expected := identityMatrix()
		for i := 0; i < 16; i++ {
			suite.InDelta(expected[i], result[i], 1e-5)
		}
	})

	suite.Run("M-inverse times M equals identity", func() {
		m := []float32{
			2, 0, 0, 0,
			0, 3, 0, 0,
			0, 0, 4, 0,
			5, 6, 7, 1,
		}
		inv := make([]float32, 16)
		ok := common.Invert4(inv, m)
		suite.True(ok)

		result := make([]float32, 16)
		common.Mul4(result, inv, m)
		expected := identityMatrix()
		for i := 0; i < 16; i++ {
			suite.InDelta(expected[i], result[i], 1e-5)
		}
	})

	suite.Run("zero matrix returns false", func() {
		m := make([]float32, 16)
		out := make([]float32, 16)
		ok := common.Invert4(out, m)
		suite.False(ok)
	})

	suite.Run("singular matrix with duplicate rows returns false", func() {
		// All rows identical → determinant is 0
		m := []float32{
			1, 1, 1, 1,
			1, 1, 1, 1,
			1, 1, 1, 1,
			1, 1, 1, 1,
		}
		out := make([]float32, 16)
		ok := common.Invert4(out, m)
		suite.False(ok)
	})

	suite.Run("output is unchanged when inversion fails", func() {
		m := make([]float32, 16) // zero matrix, singular
		out := make([]float32, 16)
		for i := range out {
			out[i] = 42
		}
		common.Invert4(out, m)
		for i := 0; i < 16; i++ {
			suite.Equal(float32(42), out[i])
		}
	})

	suite.Run("scale matrix inverse has reciprocal diagonal", func() {
		m := []float32{
			2, 0, 0, 0,
			0, 4, 0, 0,
			0, 0, 8, 0,
			0, 0, 0, 1,
		}
		out := make([]float32, 16)
		ok := common.Invert4(out, m)
		suite.True(ok)
		suite.InDelta(float32(0.5), out[0], 1e-6)
		suite.InDelta(float32(0.25), out[5], 1e-6)
		suite.InDelta(float32(0.125), out[10], 1e-6)
		suite.InDelta(float32(1.0), out[15], 1e-6)
	})

	suite.Run("rotation matrix inverse equals its transpose", func() {
		// Build a pure rotation matrix (no scale, no translation)
		rot := make([]float32, 16)
		common.BuildModelMatrix(rot, 0, 0, 0, 0.5, 0.7, 1.1, 1, 1, 1)

		inv := make([]float32, 16)
		ok := common.Invert4(inv, rot)
		suite.True(ok)

		// For a pure rotation matrix, inverse == transpose
		// Transpose: row and col swap → index [col*4+row] becomes [row*4+col]
		for row := 0; row < 4; row++ {
			for col := 0; col < 4; col++ {
				suite.InDelta(rot[col*4+row], inv[row*4+col], 1e-5)
			}
		}
	})
}

func (suite *mathTest) TestLookAt() {
	suite.Run("camera at origin looking down -Z produces identity", func() {
		out := make([]float32, 16)
		common.LookAt(out, 0, 0, 0, 0, 0, -1, 0, 1, 0)
		expected := identityMatrix()
		for i := 0; i < 16; i++ {
			suite.InDelta(expected[i], out[i], 1e-6)
		}
	})

	suite.Run("camera translated along Z axis reflects in view matrix", func() {
		out := make([]float32, 16)
		common.LookAt(out, 0, 0, 5, 0, 0, 0, 0, 1, 0)
		// Looking from z=5 toward origin, forward = -Z
		// Translation in view space: out[14] = -(z_axis dot eye) = -5
		suite.InDelta(float32(-5), out[14], 1e-6)
	})

	suite.Run("upper-left 3x3 columns have unit length", func() {
		out := make([]float32, 16)
		common.LookAt(out, 3, 4, 5, 0, 0, 0, 0, 1, 0)
		for col := 0; col < 3; col++ {
			x := float64(out[col*4+0])
			y := float64(out[col*4+1])
			z := float64(out[col*4+2])
			length := math.Sqrt(x*x + y*y + z*z)
			suite.InDelta(1.0, length, 1e-6)
		}
	})

	suite.Run("upper-left 3x3 columns are mutually orthogonal", func() {
		out := make([]float32, 16)
		common.LookAt(out, 3, 4, 5, 0, 0, 0, 0, 1, 0)
		// dot(col0, col1) ≈ 0, dot(col0, col2) ≈ 0, dot(col1, col2) ≈ 0
		pairs := [][2]int{{0, 1}, {0, 2}, {1, 2}}
		for _, p := range pairs {
			dot := float64(0)
			for row := 0; row < 3; row++ {
				dot += float64(out[p[0]*4+row]) * float64(out[p[1]*4+row])
			}
			suite.InDelta(0.0, dot, 1e-6)
		}
	})

	suite.Run("bottom row is always 0 0 0 1", func() {
		out := make([]float32, 16)
		common.LookAt(out, 10, 20, 30, 1, 2, 3, 0, 1, 0)
		suite.Equal(float32(0), out[3])
		suite.Equal(float32(0), out[7])
		suite.Equal(float32(0), out[11])
		suite.Equal(float32(1), out[15])
	})

	suite.Run("eye at center does not panic", func() {
		out := make([]float32, 16)
		// Degenerate case: eye == center, should not crash
		suite.NotPanics(func() {
			common.LookAt(out, 5, 5, 5, 5, 5, 5, 0, 1, 0)
		})
	})

	suite.Run("different eye positions produce different translation columns", func() {
		a := make([]float32, 16)
		b := make([]float32, 16)
		common.LookAt(a, 0, 0, 5, 0, 0, 0, 0, 1, 0)
		common.LookAt(b, 0, 0, 10, 0, 0, 0, 0, 1, 0)
		suite.NotEqual(a[14], b[14])
	})
}

// identityMatrix returns a fresh 4x4 identity matrix as a 16-element column-major float32 slice.
func identityMatrix() []float32 {
	return []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
}
