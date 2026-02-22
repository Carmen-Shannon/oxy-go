package camera_test

import (
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/stretchr/testify/suite"
)

type cameraControllerTest struct {
	suite.Suite
}

func TestCameraController(t *testing.T) {
	suite.Run(t, new(cameraControllerTest))
}

func (suite *cameraControllerTest) TestNewCameraController() {
	suite.Run("default construction returns non-nil controller", func() {
		cc := camera.NewCameraController()
		suite.NotNil(cc)
	})

	suite.Run("default radius is 250", func() {
		cc := camera.NewCameraController()
		suite.InDelta(250.0, cc.Radius(), 1e-6)
	})

	suite.Run("default azimuth is 0", func() {
		cc := camera.NewCameraController()
		suite.InDelta(0.0, cc.Azimuth(), 1e-6)
	})

	suite.Run("default elevation is pi/6", func() {
		cc := camera.NewCameraController()
		suite.InDelta(math.Pi/6, float64(cc.Elevation()), 1e-6)
	})

	suite.Run("default target is origin", func() {
		cc := camera.NewCameraController()
		x, y, z := cc.Target()
		suite.InDelta(0.0, x, 1e-6)
		suite.InDelta(0.0, y, 1e-6)
		suite.InDelta(0.0, z, 1e-6)
	})

	suite.Run("default radius bounds are 20 to 2000", func() {
		cc := camera.NewCameraController()
		suite.InDelta(20.0, cc.MinRadius(), 1e-6)
		suite.InDelta(2000.0, cc.MaxRadius(), 1e-6)
	})

	suite.Run("default elevation bounds", func() {
		cc := camera.NewCameraController()
		suite.InDelta(0.05, cc.MinElevation(), 1e-6)
		suite.InDelta(math.Pi/2-0.1, float64(cc.MaxElevation()), 1e-6)
	})

	suite.Run("default orbit speed is 0.03", func() {
		cc := camera.NewCameraController()
		suite.InDelta(0.03, cc.OrbitSpeed(), 1e-6)
	})

	suite.Run("default mouse sensitivity is 0.005", func() {
		cc := camera.NewCameraController()
		suite.InDelta(0.005, cc.MouseSensitivity(), 1e-6)
	})

	suite.Run("default zoom speed is 15", func() {
		cc := camera.NewCameraController()
		suite.InDelta(15.0, cc.ZoomSpeed(), 1e-6)
	})

	suite.Run("default pan speed is 1.0", func() {
		cc := camera.NewCameraController()
		suite.InDelta(1.0, cc.PanSpeed(), 1e-6)
	})

	suite.Run("initial position is computed from spherical coordinates", func() {
		// With azimuth=0, elevation=pi/6, radius=250, target=origin:
		// x = 250 * cos(pi/6) * sin(0) = 0
		// y = 250 * sin(pi/6) = 125
		// z = 250 * cos(pi/6) * cos(0) = 250 * cos(pi/6)
		cc := camera.NewCameraController()
		x, y, z := cc.Position()
		suite.InDelta(0.0, x, 1e-3)
		suite.InDelta(125.0, y, 1e-3)
		expectedZ := float32(250.0 * math.Cos(math.Pi/6))
		suite.InDelta(expectedZ, z, 1e-3)
	})
}

func (suite *cameraControllerTest) TestNewOrbitController() {
	suite.Run("new orbit controller is equivalent to new camera controller", func() {
		cc := camera.NewOrbitController()
		suite.NotNil(cc)
		suite.InDelta(250.0, cc.Radius(), 1e-6)
	})

	suite.Run("new orbit controller with options applies them", func() {
		cc := camera.NewOrbitController(camera.WithRadius(100))
		suite.InDelta(100.0, cc.Radius(), 1e-6)
	})
}

func (suite *cameraControllerTest) TestBuilderOptions() {
	suite.Run("with radius sets custom radius", func() {
		cc := camera.NewCameraController(camera.WithRadius(100))
		suite.InDelta(100.0, cc.Radius(), 1e-6)
	})

	suite.Run("with azimuth sets custom azimuth", func() {
		cc := camera.NewCameraController(camera.WithAzimuth(float32(math.Pi / 4)))
		suite.InDelta(math.Pi/4, float64(cc.Azimuth()), 1e-6)
	})

	suite.Run("with elevation sets custom elevation", func() {
		cc := camera.NewCameraController(camera.WithElevation(float32(math.Pi / 3)))
		suite.InDelta(math.Pi/3, float64(cc.Elevation()), 1e-6)
	})

	suite.Run("with target sets custom target", func() {
		cc := camera.NewCameraController(camera.WithTarget(5, 10, 15))
		x, y, z := cc.Target()
		suite.InDelta(5.0, x, 1e-6)
		suite.InDelta(10.0, y, 1e-6)
		suite.InDelta(15.0, z, 1e-6)
	})

	suite.Run("with radius bounds sets min and max", func() {
		cc := camera.NewCameraController(camera.WithRadiusBounds(5, 500))
		suite.InDelta(5.0, cc.MinRadius(), 1e-6)
		suite.InDelta(500.0, cc.MaxRadius(), 1e-6)
	})

	suite.Run("with elevation bounds sets min and max", func() {
		cc := camera.NewCameraController(camera.WithElevationBounds(0.1, 1.2))
		suite.InDelta(0.1, cc.MinElevation(), 1e-6)
		suite.InDelta(1.2, cc.MaxElevation(), 1e-6)
	})

	suite.Run("with orbit speed sets custom orbit speed", func() {
		cc := camera.NewCameraController(camera.WithOrbitSpeed(0.1))
		suite.InDelta(0.1, cc.OrbitSpeed(), 1e-6)
	})

	suite.Run("with mouse sensitivity sets custom sensitivity", func() {
		cc := camera.NewCameraController(camera.WithMouseSensitivity(0.01))
		suite.InDelta(0.01, cc.MouseSensitivity(), 1e-6)
	})

	suite.Run("with zoom speed sets custom zoom speed", func() {
		cc := camera.NewCameraController(camera.WithZoomSpeed(25.0))
		suite.InDelta(25.0, cc.ZoomSpeed(), 1e-6)
	})

	suite.Run("with pan speed sets custom pan speed", func() {
		cc := camera.NewCameraController(camera.WithPanSpeed(2.5))
		suite.InDelta(2.5, cc.PanSpeed(), 1e-6)
	})

	suite.Run("multiple options combined apply correctly", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(50),
			camera.WithAzimuth(1.0),
			camera.WithElevation(0.5),
			camera.WithTarget(1, 2, 3),
			camera.WithOrbitSpeed(0.05),
			camera.WithZoomSpeed(10),
			camera.WithPanSpeed(0.5),
		)
		suite.InDelta(50.0, cc.Radius(), 1e-6)
		suite.InDelta(1.0, cc.Azimuth(), 1e-6)
		suite.InDelta(0.5, cc.Elevation(), 1e-6)
		tx, ty, tz := cc.Target()
		suite.InDelta(1.0, tx, 1e-6)
		suite.InDelta(2.0, ty, 1e-6)
		suite.InDelta(3.0, tz, 1e-6)
	})
}

func (suite *cameraControllerTest) TestPositionAndTarget() {
	suite.Run("set position directly updates position", func() {
		cc := camera.NewCameraController()
		cc.SetPosition(1, 2, 3)
		x, y, z := cc.Position()
		suite.InDelta(1.0, x, 1e-6)
		suite.InDelta(2.0, y, 1e-6)
		suite.InDelta(3.0, z, 1e-6)
	})

	suite.Run("set target recomputes position from spherical coordinates", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(10),
			camera.WithAzimuth(0),
			camera.WithElevation(0),
		)
		// With azimuth=0, elevation=0, radius=10: position = target + (0, 0, 10)
		cc.SetTarget(5, 5, 5)
		x, y, z := cc.Position()
		suite.InDelta(5.0, x, 1e-3)
		suite.InDelta(5.0, y, 1e-3)
		suite.InDelta(15.0, z, 1e-3)
	})

	suite.Run("position is consistent with spherical coordinates", func() {
		azimuth := float32(math.Pi / 4)
		elevation := float32(math.Pi / 6)
		radius := float32(100)
		cc := camera.NewCameraController(
			camera.WithRadius(radius),
			camera.WithAzimuth(azimuth),
			camera.WithElevation(elevation),
			camera.WithTarget(0, 0, 0),
		)
		x, y, z := cc.Position()

		// Expected: x = r*cos(elev)*sin(azim), y = r*sin(elev), z = r*cos(elev)*cos(azim)
		cosElev := float32(math.Cos(float64(elevation)))
		sinElev := float32(math.Sin(float64(elevation)))
		cosAzim := float32(math.Cos(float64(azimuth)))
		sinAzim := float32(math.Sin(float64(azimuth)))

		suite.InDelta(radius*cosElev*sinAzim, x, 1e-3)
		suite.InDelta(radius*sinElev, y, 1e-3)
		suite.InDelta(radius*cosElev*cosAzim, z, 1e-3)
	})
}

func (suite *cameraControllerTest) TestOrbit() {
	suite.Run("orbit left decreases azimuth", func() {
		cc := camera.NewCameraController(camera.WithAzimuth(1.0))
		before := cc.Azimuth()
		cc.OrbitLeft()
		after := cc.Azimuth()
		suite.Less(float64(after), float64(before))
	})

	suite.Run("orbit right increases azimuth", func() {
		cc := camera.NewCameraController(camera.WithAzimuth(0))
		before := cc.Azimuth()
		cc.OrbitRight()
		after := cc.Azimuth()
		suite.Greater(float64(after), float64(before))
	})

	suite.Run("orbit up increases elevation", func() {
		cc := camera.NewCameraController(camera.WithElevation(0.2))
		before := cc.Elevation()
		cc.OrbitUp()
		after := cc.Elevation()
		suite.Greater(float64(after), float64(before))
	})

	suite.Run("orbit down decreases elevation", func() {
		cc := camera.NewCameraController(camera.WithElevation(0.5))
		before := cc.Elevation()
		cc.OrbitDown()
		after := cc.Elevation()
		suite.Less(float64(after), float64(before))
	})

	suite.Run("orbit up clamps to max elevation", func() {
		maxElev := float32(math.Pi/2 - 0.1)
		cc := camera.NewCameraController(
			camera.WithElevation(maxElev),
			camera.WithOrbitSpeed(0.5),
		)
		cc.OrbitUp()
		suite.InDelta(maxElev, cc.Elevation(), 1e-6)
	})

	suite.Run("orbit down clamps to min elevation", func() {
		minElev := float32(0.05)
		cc := camera.NewCameraController(
			camera.WithElevation(minElev),
			camera.WithOrbitSpeed(0.5),
		)
		cc.OrbitDown()
		suite.InDelta(minElev, cc.Elevation(), 1e-6)
	})

	suite.Run("orbit left and right are inverse operations", func() {
		cc := camera.NewCameraController(camera.WithAzimuth(1.0))
		before := cc.Azimuth()
		cc.OrbitLeft()
		cc.OrbitRight()
		suite.InDelta(before, cc.Azimuth(), 1e-6)
	})

	suite.Run("orbit changes position", func() {
		cc := camera.NewCameraController(camera.WithRadius(100))
		x0, y0, z0 := cc.Position()
		cc.OrbitRight()
		x1, y1, z1 := cc.Position()
		// At least one coordinate should have changed
		posChanged := math.Abs(float64(x1-x0)) > 1e-6 ||
			math.Abs(float64(y1-y0)) > 1e-6 ||
			math.Abs(float64(z1-z0)) > 1e-6
		suite.True(posChanged)
	})

	suite.Run("orbit preserves radius", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithTarget(0, 0, 0),
		)
		cc.OrbitRight()
		cc.OrbitUp()
		// Verify distance from target == radius
		x, y, z := cc.Position()
		dist := float32(math.Sqrt(float64(x*x + y*y + z*z)))
		suite.InDelta(100.0, dist, 1e-3)
	})
}

func (suite *cameraControllerTest) TestSetRadius() {
	suite.Run("set radius updates radius", func() {
		cc := camera.NewCameraController()
		cc.SetRadius(50)
		suite.InDelta(50.0, cc.Radius(), 1e-6)
	})

	suite.Run("set radius below min clamps to min", func() {
		cc := camera.NewCameraController(camera.WithRadiusBounds(10, 500))
		cc.SetRadius(5)
		suite.InDelta(10.0, cc.Radius(), 1e-6)
	})

	suite.Run("set radius above max clamps to max", func() {
		cc := camera.NewCameraController(camera.WithRadiusBounds(10, 500))
		cc.SetRadius(1000)
		suite.InDelta(500.0, cc.Radius(), 1e-6)
	})

	suite.Run("set radius updates position", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithAzimuth(0),
			camera.WithElevation(0),
			camera.WithTarget(0, 0, 0),
		)
		_, _, z0 := cc.Position()
		cc.SetRadius(200)
		_, _, z1 := cc.Position()
		suite.Greater(float64(z1), float64(z0))
	})
}

func (suite *cameraControllerTest) TestSetAzimuth() {
	suite.Run("set azimuth updates azimuth", func() {
		cc := camera.NewCameraController()
		cc.SetAzimuth(float32(math.Pi))
		suite.InDelta(math.Pi, float64(cc.Azimuth()), 1e-6)
	})

	suite.Run("set azimuth recomputes position", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithTarget(0, 0, 0),
			camera.WithElevation(0),
		)
		cc.SetAzimuth(float32(math.Pi / 2))
		x, _, z := cc.Position()
		// azimuth=pi/2, elev=0: x=100*sin(pi/2)=100, z=100*cos(pi/2)=~0
		suite.InDelta(100.0, x, 1e-3)
		suite.InDelta(0.0, z, 1e-3)
	})
}

func (suite *cameraControllerTest) TestSetElevation() {
	suite.Run("set elevation updates elevation", func() {
		cc := camera.NewCameraController()
		cc.SetElevation(0.3)
		suite.InDelta(0.3, cc.Elevation(), 1e-6)
	})

	suite.Run("set elevation below min clamps to min", func() {
		cc := camera.NewCameraController(camera.WithElevationBounds(0.1, 1.4))
		cc.SetElevation(0.01)
		suite.InDelta(0.1, cc.Elevation(), 1e-6)
	})

	suite.Run("set elevation above max clamps to max", func() {
		cc := camera.NewCameraController(camera.WithElevationBounds(0.1, 1.4))
		cc.SetElevation(2.0)
		suite.InDelta(1.4, cc.Elevation(), 1e-6)
	})

	suite.Run("set elevation recomputes position", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithTarget(0, 0, 0),
			camera.WithAzimuth(0),
		)
		cc.SetElevation(float32(math.Pi / 4))
		_, y, _ := cc.Position()
		// elevation=pi/4: y = 100*sin(pi/4) ≈ 70.71
		suite.InDelta(100.0*math.Sin(math.Pi/4), float64(y), 1e-2)
	})
}

func (suite *cameraControllerTest) TestZoom() {
	suite.Run("positive delta zooms in reducing radius", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithZoomSpeed(10),
		)
		cc.Zoom(1)
		// radius = 100 - 1*10 = 90
		suite.InDelta(90.0, cc.Radius(), 1e-6)
	})

	suite.Run("negative delta zooms out increasing radius", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithZoomSpeed(10),
		)
		cc.Zoom(-1)
		// radius = 100 - (-1)*10 = 110
		suite.InDelta(110.0, cc.Radius(), 1e-6)
	})

	suite.Run("zoom clamps to min radius", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(25),
			camera.WithRadiusBounds(20, 2000),
			camera.WithZoomSpeed(10),
		)
		cc.Zoom(1) // 25 - 10 = 15 → clamped to 20
		suite.InDelta(20.0, cc.Radius(), 1e-6)
	})

	suite.Run("zoom clamps to max radius", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(1995),
			camera.WithRadiusBounds(20, 2000),
			camera.WithZoomSpeed(10),
		)
		cc.Zoom(-1) // 1995 + 10 = 2005 → clamped to 2000
		suite.InDelta(2000.0, cc.Radius(), 1e-6)
	})

	suite.Run("zoom updates position", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithTarget(0, 0, 0),
			camera.WithAzimuth(0),
			camera.WithElevation(0),
		)
		_, _, z0 := cc.Position()
		cc.Zoom(1) // zoom in → smaller radius → closer z
		_, _, z1 := cc.Position()
		suite.Less(float64(z1), float64(z0))
	})
}

func (suite *cameraControllerTest) TestPanRight() {
	suite.Run("pan right shifts position and target along local right axis", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(10),
			camera.WithAzimuth(0),
			camera.WithElevation(0),
			camera.WithTarget(0, 0, 0),
		)
		x0, _, _ := cc.Position()
		tx0, _, _ := cc.Target()
		cc.PanRight(1)
		x1, _, _ := cc.Position()
		tx1, _, _ := cc.Target()

		// Position and target should have shifted equally
		dx := x1 - x0
		dtx := tx1 - tx0
		suite.InDelta(dx, dtx, 1e-6)
		suite.True(math.Abs(float64(dx)) > 1e-6)
	})

	suite.Run("pan right preserves orbit distance", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithTarget(0, 0, 0),
		)
		cc.PanRight(5)
		px, py, pz := cc.Position()
		tx, ty, tz := cc.Target()
		dx := px - tx
		dy := py - ty
		dz := pz - tz
		dist := float32(math.Sqrt(float64(dx*dx + dy*dy + dz*dz)))
		suite.InDelta(100.0, dist, 1e-2)
	})
}

func (suite *cameraControllerTest) TestPanUp() {
	suite.Run("pan up shifts position and target along local up axis", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(10),
			camera.WithAzimuth(0),
			camera.WithElevation(float32(math.Pi/6)),
			camera.WithTarget(0, 0, 0),
		)
		_, y0, _ := cc.Position()
		_, ty0, _ := cc.Target()
		cc.PanUp(1)
		_, y1, _ := cc.Position()
		_, ty1, _ := cc.Target()

		// Both should have shifted upward
		dy := y1 - y0
		dty := ty1 - ty0
		suite.InDelta(dy, dty, 1e-6)
	})

	suite.Run("pan up preserves orbit distance", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithTarget(0, 0, 0),
		)
		cc.PanUp(5)
		px, py, pz := cc.Position()
		tx, ty, tz := cc.Target()
		dx := px - tx
		dy := py - ty
		dz := pz - tz
		dist := float32(math.Sqrt(float64(dx*dx + dy*dy + dz*dz)))
		suite.InDelta(100.0, dist, 1e-2)
	})
}

func (suite *cameraControllerTest) TestPanForward() {
	suite.Run("pan forward shifts position and target along local forward axis", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(10),
			camera.WithAzimuth(0),
			camera.WithElevation(0),
			camera.WithTarget(0, 0, 0),
		)
		_, _, z0 := cc.Position()
		_, _, tz0 := cc.Target()
		cc.PanForward(1)
		_, _, z1 := cc.Position()
		_, _, tz1 := cc.Target()

		dz := z1 - z0
		dtz := tz1 - tz0
		suite.InDelta(dz, dtz, 1e-6)
	})

	suite.Run("pan forward preserves orbit distance", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithTarget(0, 0, 0),
		)
		cc.PanForward(5)
		px, py, pz := cc.Position()
		tx, ty, tz := cc.Target()
		dx := px - tx
		dy := py - ty
		dz := pz - tz
		dist := float32(math.Sqrt(float64(dx*dx + dy*dy + dz*dz)))
		suite.InDelta(100.0, dist, 1e-2)
	})

	suite.Run("pan forward with zero delta is a no-op", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(10),
			camera.WithTarget(0, 0, 0),
		)
		x0, y0, z0 := cc.Position()
		cc.PanForward(0)
		x1, y1, z1 := cc.Position()
		suite.InDelta(x0, x1, 1e-6)
		suite.InDelta(y0, y1, 1e-6)
		suite.InDelta(z0, z1, 1e-6)
	})
}

func (suite *cameraControllerTest) TestCombinedOperations() {
	suite.Run("orbit then zoom maintains coherent state", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithTarget(0, 0, 0),
		)
		cc.OrbitRight()
		cc.OrbitRight()
		cc.OrbitUp()
		cc.Zoom(2)

		// Verify radius decreased by 2*zoomSpeed
		expectedRadius := 100.0 - 2.0*15.0 // 70
		suite.InDelta(expectedRadius, cc.Radius(), 1e-3)

		// Verify position is at expected distance from target
		x, y, z := cc.Position()
		dist := float32(math.Sqrt(float64(x*x + y*y + z*z)))
		suite.InDelta(expectedRadius, float64(dist), 1e-2)
	})

	suite.Run("pan then orbit maintains correct orbit geometry", func() {
		cc := camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithTarget(0, 0, 0),
		)
		cc.PanRight(10)
		cc.OrbitRight()

		// Orbit should still be around the updated target at expected radius
		px, py, pz := cc.Position()
		tx, ty, tz := cc.Target()
		dx := px - tx
		dy := py - ty
		dz := pz - tz
		dist := float32(math.Sqrt(float64(dx*dx + dy*dy + dz*dz)))
		suite.InDelta(100.0, dist, 1e-1)
	})
}
