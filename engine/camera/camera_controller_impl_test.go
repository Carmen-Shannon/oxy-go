package camera

import (
	"math"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"
)

func TestRunCameraImplTests(t *testing.T) {
	suite.Run(t, new(cameraImplTest))
}

type cameraImplTest struct {
	suite.Suite
}

func (suite *cameraImplTest) TestLocalAxes() {
	suite.Run("should return all zeros when position equals target", func() {
		cc := &cameraControllerImpl{
			mu:       &sync.Mutex{},
			position: [3]float32{0, 0, 0},
			target:   [3]float32{0, 0, 0},
		}
		rx, ry, rz, ux, uy, uz, fx, fy, fz := cc.localAxes()
		suite.Equal(float32(0), rx)
		suite.Equal(float32(0), ry)
		suite.Equal(float32(0), rz)
		suite.Equal(float32(0), ux)
		suite.Equal(float32(0), uy)
		suite.Equal(float32(0), uz)
		suite.Equal(float32(0), fx)
		suite.Equal(float32(0), fy)
		suite.Equal(float32(0), fz)
	})

	suite.Run("should return all zeros when camera is directly above target", func() {
		// Set position directly to ensure bx=0, bz=0 exactly.
		// updatePosition() cannot be used here because float32(math.Pi/2) != π/2
		// exactly — cos returns ~-4.37e-8, making rLen > 1e-8 and bypassing Branch 2.
		cc := &cameraControllerImpl{
			mu:       &sync.Mutex{},
			target:   [3]float32{0, 0, 0},
			position: [3]float32{0, 250, 0},
		}
		rx, ry, rz, ux, uy, uz, fx, fy, fz := cc.localAxes()
		suite.Equal(float32(0), rx)
		suite.Equal(float32(0), ry)
		suite.Equal(float32(0), rz)
		suite.Equal(float32(0), ux)
		suite.Equal(float32(0), uy)
		suite.Equal(float32(0), uz)
		suite.Equal(float32(0), fx)
		suite.Equal(float32(0), fy)
		suite.Equal(float32(0), fz)
	})

	suite.Run("should return valid orthogonal axes for a standard camera position", func() {
		cc := &cameraControllerImpl{
			mu:        &sync.Mutex{},
			target:    [3]float32{0, 0, 0},
			radius:    250.0,
			azimuth:   0.0,
			elevation: float32(math.Pi / 6),
		}
		cc.updatePosition()
		rx, _, rz, _, uy, _, fx, fy, fz := cc.localAxes()
		suite.Greater(rx*rx+rz*rz, float32(0))
		suite.Greater(uy, float32(0))
		suite.Greater(fx*fx+fy*fy+fz*fz, float32(0))
	})
}
