package light

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

func TestRunLightImplTests(t *testing.T) {
	suite.Run(t, new(lightImplTest))
}

type lightImplTest struct {
	suite.Suite
}

func (suite *lightImplTest) TestNormalize3() {
	suite.Run("should return a zero vector when the input length is zero", func() {
		result := normalize3(0, 0, 0)
		suite.Equal([3]float32{0, 0, 0}, result)
	})

	suite.Run("should return a unit vector for a non-zero input", func() {
		result := normalize3(3, 4, 0)
		suite.InDelta(float64(float32(3.0/5.0)), float64(result[0]), 1e-6)
		suite.InDelta(float64(float32(4.0/5.0)), float64(result[1]), 1e-6)
		suite.InDelta(0.0, float64(result[2]), 1e-6)
	})

	suite.Run("should return a unit vector for a negative input", func() {
		result := normalize3(-1, 0, 0)
		suite.Equal([3]float32{-1, 0, 0}, result)
	})
}
