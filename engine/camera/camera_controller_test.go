package camera_test

import (
	"math"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/camera"
)

func TestRunCameraControllerTests(t *testing.T) {
	suite.Run(t, new(cameraControllerTest))
}

type cameraControllerTest struct {
	suite.Suite
	controller camera.CameraController
}

func (suite *cameraControllerTest) SetupSubTest() {
	suite.controller = camera.NewCameraController()
}

func (suite *cameraControllerTest) TestNewCameraController() {
	suite.Run("should create a controller with default values", func() {
		suite.NotNil(suite.controller)
		suite.Equal(float32(250.0), suite.controller.Radius())
		suite.Equal(float32(0.0), suite.controller.Azimuth())
		suite.Equal(float32(math.Pi/6), suite.controller.Elevation())
		suite.Equal(float32(20.0), suite.controller.MinRadius())
		suite.Equal(float32(2000.0), suite.controller.MaxRadius())
		suite.Equal(float32(0.05), suite.controller.MinElevation())
		suite.Equal(float32(math.Pi/2-0.1), suite.controller.MaxElevation())
		suite.Equal(float32(0.03), suite.controller.OrbitSpeed())
		suite.Equal(float32(0.005), suite.controller.MouseSensitivity())
		suite.Equal(float32(15.0), suite.controller.ZoomSpeed())
		suite.Equal(float32(1.0), suite.controller.PanSpeed())
	})

	suite.Run("should apply all provided builder options", func() {
		ctrl := camera.NewCameraController(
			camera.WithRadius(500.0),
			camera.WithAzimuth(float32(math.Pi/4)),
			camera.WithElevation(float32(math.Pi/4)),
			camera.WithTarget(1, 2, 3),
			camera.WithRadiusBounds(10.0, 1000.0),
			camera.WithElevationBounds(0.1, float32(math.Pi/2-0.2)),
			camera.WithOrbitSpeed(0.05),
			camera.WithMouseSensitivity(0.01),
			camera.WithZoomSpeed(20.0),
			camera.WithPanSpeed(2.0),
		)
		suite.Equal(float32(500.0), ctrl.Radius())
		suite.Equal(float32(math.Pi/4), ctrl.Azimuth())
		suite.Equal(float32(math.Pi/4), ctrl.Elevation())
		tx, ty, tz := ctrl.Target()
		suite.Equal(float32(1), tx)
		suite.Equal(float32(2), ty)
		suite.Equal(float32(3), tz)
		suite.Equal(float32(10.0), ctrl.MinRadius())
		suite.Equal(float32(1000.0), ctrl.MaxRadius())
		suite.Equal(float32(0.1), ctrl.MinElevation())
		suite.Equal(float32(math.Pi/2-0.2), ctrl.MaxElevation())
		suite.Equal(float32(0.05), ctrl.OrbitSpeed())
		suite.Equal(float32(0.01), ctrl.MouseSensitivity())
		suite.Equal(float32(20.0), ctrl.ZoomSpeed())
		suite.Equal(float32(2.0), ctrl.PanSpeed())
	})
}

func (suite *cameraControllerTest) TestNewOrbitController() {
	suite.Run("should create a controller identical to NewCameraController", func() {
		ctrl := camera.NewOrbitController()
		suite.NotNil(ctrl)
		suite.Equal(float32(250.0), ctrl.Radius())
	})
}

func (suite *cameraControllerTest) TestPosition() {
	suite.Run("should return the computed camera position", func() {
		x, y, z := suite.controller.Position()
		suite.NotEqual(float32(0), x*x+y*y+z*z)
	})
}

func (suite *cameraControllerTest) TestSetPosition() {
	suite.Run("should set the camera position directly", func() {
		suite.controller.SetPosition(10, 20, 30)
		x, y, z := suite.controller.Position()
		suite.Equal(float32(10), x)
		suite.Equal(float32(20), y)
		suite.Equal(float32(30), z)
	})
}

func (suite *cameraControllerTest) TestTarget() {
	suite.Run("should return the default target at origin", func() {
		x, y, z := suite.controller.Target()
		suite.Equal(float32(0), x)
		suite.Equal(float32(0), y)
		suite.Equal(float32(0), z)
	})
}

func (suite *cameraControllerTest) TestSetTarget() {
	suite.Run("should update the target and recompute position", func() {
		suite.controller.SetTarget(5, 10, 15)
		x, y, z := suite.controller.Target()
		suite.Equal(float32(5), x)
		suite.Equal(float32(10), y)
		suite.Equal(float32(15), z)
	})
}

func (suite *cameraControllerTest) TestZoom() {
	suite.Run("should decrease radius on positive delta", func() {
		before := suite.controller.Radius()
		suite.controller.Zoom(1.0)
		suite.Less(suite.controller.Radius(), before)
	})

	suite.Run("should clamp radius to minRadius", func() {
		suite.controller.Zoom(10000.0)
		suite.Equal(suite.controller.MinRadius(), suite.controller.Radius())
	})

	suite.Run("should clamp radius to maxRadius", func() {
		suite.controller.Zoom(-10000.0)
		suite.Equal(suite.controller.MaxRadius(), suite.controller.Radius())
	})
}

func (suite *cameraControllerTest) TestOrbitLeft() {
	suite.Run("should decrease azimuth", func() {
		before := suite.controller.Azimuth()
		suite.controller.OrbitLeft()
		suite.Less(suite.controller.Azimuth(), before)
	})
}

func (suite *cameraControllerTest) TestOrbitRight() {
	suite.Run("should increase azimuth", func() {
		before := suite.controller.Azimuth()
		suite.controller.OrbitRight()
		suite.Greater(suite.controller.Azimuth(), before)
	})
}

func (suite *cameraControllerTest) TestOrbitUp() {
	suite.Run("should increase elevation", func() {
		before := suite.controller.Elevation()
		suite.controller.OrbitUp()
		suite.Greater(suite.controller.Elevation(), before)
	})

	suite.Run("should clamp elevation to maxElevation", func() {
		suite.controller.SetElevation(suite.controller.MaxElevation() - 0.001)
		suite.controller.OrbitUp()
		suite.Equal(suite.controller.MaxElevation(), suite.controller.Elevation())
	})
}

func (suite *cameraControllerTest) TestOrbitDown() {
	suite.Run("should decrease elevation", func() {
		before := suite.controller.Elevation()
		suite.controller.OrbitDown()
		suite.Less(suite.controller.Elevation(), before)
	})

	suite.Run("should clamp elevation to minElevation", func() {
		suite.controller.SetElevation(suite.controller.MinElevation() + 0.001)
		suite.controller.OrbitDown()
		suite.Equal(suite.controller.MinElevation(), suite.controller.Elevation())
	})
}

func (suite *cameraControllerTest) TestRadius() {
	suite.Run("should return the current radius", func() {
		suite.Equal(float32(250.0), suite.controller.Radius())
	})
}

func (suite *cameraControllerTest) TestSetRadius() {
	suite.Run("should update the radius", func() {
		suite.controller.SetRadius(300.0)
		suite.Equal(float32(300.0), suite.controller.Radius())
	})

	suite.Run("should clamp to minRadius", func() {
		suite.controller.SetRadius(1.0)
		suite.Equal(suite.controller.MinRadius(), suite.controller.Radius())
	})

	suite.Run("should clamp to maxRadius", func() {
		suite.controller.SetRadius(99999.0)
		suite.Equal(suite.controller.MaxRadius(), suite.controller.Radius())
	})
}

func (suite *cameraControllerTest) TestAzimuth() {
	suite.Run("should return the current azimuth", func() {
		suite.Equal(float32(0.0), suite.controller.Azimuth())
	})
}

func (suite *cameraControllerTest) TestSetAzimuth() {
	suite.Run("should update the azimuth", func() {
		suite.controller.SetAzimuth(float32(math.Pi / 2))
		suite.Equal(float32(math.Pi/2), suite.controller.Azimuth())
	})
}

func (suite *cameraControllerTest) TestElevation() {
	suite.Run("should return the current elevation", func() {
		suite.Equal(float32(math.Pi/6), suite.controller.Elevation())
	})
}

func (suite *cameraControllerTest) TestSetElevation() {
	suite.Run("should update the elevation", func() {
		suite.controller.SetElevation(float32(math.Pi / 4))
		suite.Equal(float32(math.Pi/4), suite.controller.Elevation())
	})

	suite.Run("should clamp to minElevation", func() {
		suite.controller.SetElevation(-1.0)
		suite.Equal(suite.controller.MinElevation(), suite.controller.Elevation())
	})

	suite.Run("should clamp to maxElevation", func() {
		suite.controller.SetElevation(float32(math.Pi))
		suite.Equal(suite.controller.MaxElevation(), suite.controller.Elevation())
	})
}

func (suite *cameraControllerTest) TestMinRadius() {
	suite.Run("should return the minimum radius", func() {
		suite.Equal(float32(20.0), suite.controller.MinRadius())
	})
}

func (suite *cameraControllerTest) TestMaxRadius() {
	suite.Run("should return the maximum radius", func() {
		suite.Equal(float32(2000.0), suite.controller.MaxRadius())
	})
}

func (suite *cameraControllerTest) TestMinElevation() {
	suite.Run("should return the minimum elevation", func() {
		suite.Equal(float32(0.05), suite.controller.MinElevation())
	})
}

func (suite *cameraControllerTest) TestMaxElevation() {
	suite.Run("should return the maximum elevation", func() {
		suite.Equal(float32(math.Pi/2-0.1), suite.controller.MaxElevation())
	})
}

func (suite *cameraControllerTest) TestOrbitSpeed() {
	suite.Run("should return the orbit speed", func() {
		suite.Equal(float32(0.03), suite.controller.OrbitSpeed())
	})
}

func (suite *cameraControllerTest) TestMouseSensitivity() {
	suite.Run("should return the mouse sensitivity", func() {
		suite.Equal(float32(0.005), suite.controller.MouseSensitivity())
	})
}

func (suite *cameraControllerTest) TestZoomSpeed() {
	suite.Run("should return the zoom speed", func() {
		suite.Equal(float32(15.0), suite.controller.ZoomSpeed())
	})
}

func (suite *cameraControllerTest) TestPanRight() {
	suite.Run("should translate target and position along the right axis", func() {
		tx0, _, tz0 := suite.controller.Target()
		suite.controller.PanRight(10.0)
		tx1, _, tz1 := suite.controller.Target()
		suite.NotEqual(tx0, tx1+tz1, (tx0 + tz0))
	})
}

func (suite *cameraControllerTest) TestPanUp() {
	suite.Run("should translate target and position along the up axis", func() {
		tx0, ty0, tz0 := suite.controller.Target()
		suite.controller.PanUp(10.0)
		tx1, ty1, tz1 := suite.controller.Target()
		suite.NotEqual(tx0+ty0+tz0, tx1+ty1+tz1)
	})
}

func (suite *cameraControllerTest) TestPanForward() {
	suite.Run("should translate target and position along the forward axis", func() {
		tx0, ty0, tz0 := suite.controller.Target()
		suite.controller.PanForward(10.0)
		tx1, ty1, tz1 := suite.controller.Target()
		suite.NotEqual(tx0+ty0+tz0, tx1+ty1+tz1)
	})
}

func (suite *cameraControllerTest) TestPanSpeed() {
	suite.Run("should return the pan speed", func() {
		suite.Equal(float32(1.0), suite.controller.PanSpeed())
	})
}
