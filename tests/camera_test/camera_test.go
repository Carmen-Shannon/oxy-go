package camera_test

import (
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/stretchr/testify/suite"
)

type cameraTest struct {
	suite.Suite
}

func TestCamera(t *testing.T) {
	suite.Run(t, new(cameraTest))
}

func (suite *cameraTest) TestNewCamera() {
	suite.Run("default construction returns non-nil camera", func() {
		cam := camera.NewCamera()
		suite.NotNil(cam)
	})

	suite.Run("default up vector is Y-up", func() {
		cam := camera.NewCamera()
		x, y, z := cam.Up()
		suite.InDelta(0.0, x, 1e-6)
		suite.InDelta(1.0, y, 1e-6)
		suite.InDelta(0.0, z, 1e-6)
	})

	suite.Run("default fov is 45 degrees in radians", func() {
		cam := camera.NewCamera()
		expected := float32(45.0 * (math.Pi / 180.0))
		suite.InDelta(expected, cam.Fov(), 1e-6)
	})

	suite.Run("default aspect ratio is 1.0", func() {
		cam := camera.NewCamera()
		suite.InDelta(1.0, cam.Aspect(), 1e-6)
	})

	suite.Run("default near plane is 0.1", func() {
		cam := camera.NewCamera()
		suite.InDelta(0.1, cam.Near(), 1e-6)
	})

	suite.Run("default far plane is 100.0", func() {
		cam := camera.NewCamera()
		suite.InDelta(100.0, cam.Far(), 1e-6)
	})

	suite.Run("default controller is nil", func() {
		cam := camera.NewCamera()
		suite.Nil(cam.Controller())
	})

	suite.Run("default bind group provider is non-nil", func() {
		cam := camera.NewCamera()
		suite.NotNil(cam.BindGroupProvider())
	})

	suite.Run("default matrices are identity", func() {
		cam := camera.NewCamera()
		identity := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		suite.Equal(identity, cam.ViewMatrix())
		suite.Equal(identity, cam.ProjectionMatrix())
		suite.Equal(identity, cam.ViewProjectionMatrix())
	})
}

func (suite *cameraTest) TestNewCameraWithOptions() {
	suite.Run("with up option sets custom up vector", func() {
		cam := camera.NewCamera(camera.WithUp(1, 0, 0))
		x, y, z := cam.Up()
		suite.InDelta(1.0, x, 1e-6)
		suite.InDelta(0.0, y, 1e-6)
		suite.InDelta(0.0, z, 1e-6)
	})

	suite.Run("with fov option sets custom fov", func() {
		fov := float32(math.Pi / 3.0)
		cam := camera.NewCamera(camera.WithFov(fov))
		suite.InDelta(fov, cam.Fov(), 1e-6)
	})

	suite.Run("with aspect option sets custom aspect ratio", func() {
		cam := camera.NewCamera(camera.WithAspect(16.0 / 9.0))
		suite.InDelta(16.0/9.0, cam.Aspect(), 1e-6)
	})

	suite.Run("with near option sets custom near plane", func() {
		cam := camera.NewCamera(camera.WithNear(0.01))
		suite.InDelta(0.01, cam.Near(), 1e-6)
	})

	suite.Run("with far option sets custom far plane", func() {
		cam := camera.NewCamera(camera.WithFar(1000.0))
		suite.InDelta(1000.0, cam.Far(), 1e-6)
	})

	suite.Run("with controller option attaches controller", func() {
		ctrl := camera.NewCameraController()
		cam := camera.NewCamera(camera.WithController(ctrl))
		suite.NotNil(cam.Controller())
	})

	suite.Run("with controller option triggers matrix computation", func() {
		ctrl := camera.NewCameraController(
			camera.WithRadius(10),
			camera.WithTarget(0, 0, 0),
			camera.WithElevation(0),
			camera.WithAzimuth(0),
		)
		cam := camera.NewCamera(camera.WithController(ctrl))
		// With a controller, matrices should no longer be identity
		identity := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		suite.NotEqual(identity, cam.ViewMatrix())
		suite.NotEqual(identity, cam.ProjectionMatrix())
	})

	suite.Run("with bind group provider option overrides default", func() {
		bgp := bind_group_provider.NewBindGroupProvider("custom_bgp")
		cam := camera.NewCamera(camera.WithBindGroupProvider(bgp))
		suite.Equal(bgp, cam.BindGroupProvider())
	})

	suite.Run("multiple options combined apply without panic", func() {
		ctrl := camera.NewCameraController()
		suite.NotPanics(func() {
			_ = camera.NewCamera(
				camera.WithUp(0, 1, 0),
				camera.WithFov(float32(math.Pi/4)),
				camera.WithAspect(1.5),
				camera.WithNear(0.5),
				camera.WithFar(500.0),
				camera.WithController(ctrl),
			)
		})
	})
}

func (suite *cameraTest) TestSettersAndGetters() {
	suite.Run("set up updates the up vector", func() {
		cam := camera.NewCamera()
		cam.SetUp(0, 0, 1)
		x, y, z := cam.Up()
		suite.InDelta(0.0, x, 1e-6)
		suite.InDelta(0.0, y, 1e-6)
		suite.InDelta(1.0, z, 1e-6)
	})

	suite.Run("set fov updates fov", func() {
		cam := camera.NewCamera()
		newFov := float32(math.Pi / 3)
		cam.SetFov(newFov)
		suite.InDelta(newFov, cam.Fov(), 1e-6)
	})

	suite.Run("set aspect updates aspect ratio", func() {
		cam := camera.NewCamera()
		cam.SetAspect(2.0)
		suite.InDelta(2.0, cam.Aspect(), 1e-6)
	})

	suite.Run("set near updates near plane", func() {
		cam := camera.NewCamera()
		cam.SetNear(0.5)
		suite.InDelta(0.5, cam.Near(), 1e-6)
	})

	suite.Run("set far updates far plane", func() {
		cam := camera.NewCamera()
		cam.SetFar(500.0)
		suite.InDelta(500.0, cam.Far(), 1e-6)
	})

	suite.Run("set controller attaches controller", func() {
		cam := camera.NewCamera()
		ctrl := camera.NewCameraController()
		cam.SetController(ctrl)
		suite.NotNil(cam.Controller())
	})

	suite.Run("set controller to nil detaches controller", func() {
		ctrl := camera.NewCameraController()
		cam := camera.NewCamera(camera.WithController(ctrl))
		cam.SetController(nil)
		suite.Nil(cam.Controller())
	})

	suite.Run("set bind group provider updates provider", func() {
		cam := camera.NewCamera()
		bgp := bind_group_provider.NewBindGroupProvider("test_bgp")
		cam.SetBindGroupProvider(bgp)
		suite.Equal(bgp, cam.BindGroupProvider())
	})

	suite.Run("set bind group provider to nil clears provider", func() {
		cam := camera.NewCamera()
		cam.SetBindGroupProvider(nil)
		suite.Nil(cam.BindGroupProvider())
	})
}

func (suite *cameraTest) TestUpdate() {
	suite.Run("update with no controller is a no-op", func() {
		cam := camera.NewCamera()
		identity := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		cam.Update()
		suite.Equal(identity, cam.ViewMatrix())
	})

	suite.Run("update with controller recomputes matrices", func() {
		ctrl := camera.NewCameraController(
			camera.WithRadius(10),
			camera.WithTarget(0, 0, 0),
			camera.WithElevation(0),
			camera.WithAzimuth(0),
		)
		cam := camera.NewCamera(camera.WithController(ctrl))

		// Capture initial matrices
		initialView := cam.ViewMatrix()

		// Move the controller
		ctrl.OrbitRight()
		cam.Update()

		// Matrices should have changed
		suite.NotEqual(initialView, cam.ViewMatrix())
	})

	suite.Run("update recomputes projection from current fov and aspect", func() {
		ctrl := camera.NewCameraController(
			camera.WithRadius(10),
			camera.WithTarget(0, 0, 0),
		)
		cam := camera.NewCamera(camera.WithController(ctrl))

		initialProj := cam.ProjectionMatrix()
		cam.SetFov(float32(math.Pi / 3))
		cam.Update()

		// Projection should reflect the new fov
		suite.NotEqual(initialProj, cam.ProjectionMatrix())
	})

	suite.Run("update produces valid view-projection as product of view and projection", func() {
		ctrl := camera.NewCameraController(
			camera.WithRadius(10),
			camera.WithTarget(0, 0, 0),
			camera.WithElevation(float32(math.Pi/6)),
			camera.WithAzimuth(float32(math.Pi/4)),
		)
		cam := camera.NewCamera(
			camera.WithController(ctrl),
			camera.WithAspect(1.5),
			camera.WithFov(float32(math.Pi/4)),
			camera.WithNear(0.1),
			camera.WithFar(100),
		)
		cam.Update()

		// The view-projection matrix should be the product of projection * view
		view := cam.ViewMatrix()
		proj := cam.ProjectionMatrix()
		var expected [16]float32
		mul4(&expected, &proj, &view)

		vp := cam.ViewProjectionMatrix()
		for i := 0; i < 16; i++ {
			suite.InDelta(expected[i], vp[i], 1e-4)
		}
	})

	suite.Run("inverse projection times projection yields identity", func() {
		ctrl := camera.NewCameraController(
			camera.WithRadius(10),
			camera.WithTarget(0, 0, 0),
		)
		cam := camera.NewCamera(
			camera.WithController(ctrl),
			camera.WithAspect(16.0/9.0),
			camera.WithFov(float32(math.Pi/4)),
			camera.WithNear(0.1),
			camera.WithFar(100),
		)
		cam.Update()

		proj := cam.ProjectionMatrix()
		invProj := cam.InverseProjectionMatrix()
		var result [16]float32
		mul4(&result, &proj, &invProj)

		identity := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		for i := 0; i < 16; i++ {
			suite.InDelta(identity[i], result[i], 1e-4)
		}
	})
}

func (suite *cameraTest) TestMatrixUpdatesOnSetters() {
	suite.Run("set fov triggers matrix recomputation with controller", func() {
		ctrl := camera.NewCameraController(camera.WithRadius(10))
		cam := camera.NewCamera(camera.WithController(ctrl))

		before := cam.ProjectionMatrix()
		cam.SetFov(float32(math.Pi / 2))
		after := cam.ProjectionMatrix()
		suite.NotEqual(before, after)
	})

	suite.Run("set aspect triggers matrix recomputation with controller", func() {
		ctrl := camera.NewCameraController(camera.WithRadius(10))
		cam := camera.NewCamera(camera.WithController(ctrl))

		before := cam.ProjectionMatrix()
		cam.SetAspect(2.5)
		after := cam.ProjectionMatrix()
		suite.NotEqual(before, after)
	})

	suite.Run("set near triggers matrix recomputation with controller", func() {
		ctrl := camera.NewCameraController(camera.WithRadius(10))
		cam := camera.NewCamera(camera.WithController(ctrl))

		before := cam.ProjectionMatrix()
		cam.SetNear(1.0)
		after := cam.ProjectionMatrix()
		suite.NotEqual(before, after)
	})

	suite.Run("set far triggers matrix recomputation with controller", func() {
		ctrl := camera.NewCameraController(camera.WithRadius(10))
		cam := camera.NewCamera(camera.WithController(ctrl))

		before := cam.ProjectionMatrix()
		cam.SetFar(500.0)
		after := cam.ProjectionMatrix()
		suite.NotEqual(before, after)
	})

	suite.Run("set up triggers matrix recomputation with controller", func() {
		ctrl := camera.NewCameraController(camera.WithRadius(10))
		cam := camera.NewCamera(camera.WithController(ctrl))

		before := cam.ViewMatrix()
		cam.SetUp(1, 0, 0)
		after := cam.ViewMatrix()
		suite.NotEqual(before, after)
	})

	suite.Run("set fov without controller does not panic", func() {
		cam := camera.NewCamera()
		suite.NotPanics(func() {
			cam.SetFov(float32(math.Pi / 3))
		})
	})

	suite.Run("set aspect without controller does not panic", func() {
		cam := camera.NewCamera()
		suite.NotPanics(func() {
			cam.SetAspect(2.0)
		})
	})
}

func (suite *cameraTest) TestMultipleCamerasGetUniqueBGP() {
	suite.Run("two cameras have different bind group provider instances", func() {
		cam1 := camera.NewCamera()
		cam2 := camera.NewCamera()
		// Each camera creates its own bind group provider; verify they are distinct instances.
		suite.False(cam1.BindGroupProvider() == cam2.BindGroupProvider())
	})
}

// mul4 computes out = a * b for 4x4 column-major matrices.
func mul4(out, a, b *[16]float32) {
	for col := range 4 {
		for row := range 4 {
			var sum float32
			for k := range 4 {
				sum += a[k*4+row] * b[col*4+k]
			}
			out[col*4+row] = sum
		}
	}
}
