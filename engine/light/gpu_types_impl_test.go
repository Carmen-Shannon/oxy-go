package light

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

func TestRunGPUTypesImplTests(t *testing.T) {
	suite.Run(t, new(gpuTypesImplTest))
}

type gpuTypesImplTest struct {
	suite.Suite
}

func (suite *gpuTypesImplTest) TestAbsF32() {
	suite.Run("should return the value unchanged when positive", func() {
		suite.Equal(float32(5.0), absF32(5.0))
	})

	suite.Run("should return the negated value when negative", func() {
		suite.Equal(float32(3.14), absF32(-3.14))
	})
}

func (suite *gpuTypesImplTest) TestTransformPoint() {
	suite.Run("should return the point unchanged when multiplied by an identity matrix", func() {
		identity := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		result := transformPoint(identity, [3]float32{1, 2, 3})
		suite.Equal([3]float32{1, 2, 3}, result)
	})

	suite.Run("should translate a point by the matrix's translation column", func() {
		m := []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			10, 20, 30, 1,
		}
		result := transformPoint(m, [3]float32{1, 2, 3})
		suite.Equal([3]float32{11, 22, 33}, result)
	})
}

func (suite *gpuTypesImplTest) TestOrtho() {
	suite.Run("should produce a diagonal matrix for a symmetric frustum", func() {
		out := make([]float32, 16)
		ortho(out, -1, 1, -1, 1, 0.1, 100.0)
		suite.InDelta(1.0, float64(out[0]), 1e-5)
		suite.InDelta(1.0, float64(out[5]), 1e-5)
		suite.InDelta(float64(float32(-1.0/99.9)), float64(out[10]), 1e-5)
		suite.InDelta(0.0, float64(out[12]), 1e-5)
		suite.InDelta(0.0, float64(out[13]), 1e-5)
		suite.InDelta(float64(float32(-0.1/99.9)), float64(out[14]), 1e-5)
		suite.InDelta(1.0, float64(out[15]), 1e-5)
	})

	suite.Run("should set the identity diagonal before applying ortho values", func() {
		out := make([]float32, 16)
		ortho(out, -1, 1, -1, 1, 0.1, 100.0)
		suite.Equal(float32(1.0), out[15])
	})

	suite.Run("should produce non-zero translation values for an asymmetric frustum", func() {
		out := make([]float32, 16)
		ortho(out, 0, 10, 0, 5, 1, 50)
		suite.InDelta(-1.0, float64(out[12]), 1e-5)
		suite.InDelta(-1.0, float64(out[13]), 1e-5)
	})
}
