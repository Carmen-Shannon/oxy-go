package camera_test

import (
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	camera_mocks "github.com/Carmen-Shannon/oxy-go/engine/camera/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/stretchr/testify/suite"
)

func TestRunCameraTests(t *testing.T) {
	suite.Run(t, new(cameraTest))
}

type cameraTest struct {
	suite.Suite
	controllerMock *camera_mocks.MockCameraController
	camera         camera.Camera
}

func (suite *cameraTest) SetupSubTest() {
	suite.controllerMock = camera_mocks.NewMockCameraController(suite.T())
	suite.controllerMock.EXPECT().Position().Return(float32(0), float32(0), float32(0)).Maybe()
	suite.controllerMock.EXPECT().Target().Return(float32(0), float32(0), float32(-1)).Maybe()
	suite.camera = camera.NewCamera(
		camera.WithController(suite.controllerMock),
	)
}

func (suite *cameraTest) TestNewCamera() {
	suite.Run("should create a new camera with provided options", func() {
		dummyCamera := camera.NewCamera(
			camera.WithUp(0, 1, 0),
			camera.WithFov(float32(45.0*(math.Pi/180.0))),
			camera.WithAspect(16.0/9.0),
			camera.WithNear(0.1),
			camera.WithFar(100.0),
			camera.WithController(suite.controllerMock),
			camera.WithBindGroupProvider(bind_group_provider.NewBindGroupProvider("test")),
		)
		suite.NotNil(dummyCamera)
	})
}

func (suite *cameraTest) TestUp() {
	suite.Run("should return the camera up vector", func() {
		x, y, z := suite.camera.Up()
		suite.Equal(float32(0), x)
		suite.Equal(float32(1), y)
		suite.Equal(float32(0), z)
	})
}

func (suite *cameraTest) TestFov() {
	suite.Run("should return the camera field of view", func() {
		fov := suite.camera.Fov()
		suite.Equal(float32(45.0*(math.Pi/180.0)), fov)
	})
}

func (suite *cameraTest) TestAspect() {
	suite.Run("should return the camera aspect ratio", func() {
		aspect := suite.camera.Aspect()
		suite.Equal(float32(1.0), aspect)
	})
}

func (suite *cameraTest) TestNear() {
	suite.Run("should return the camera near plane distance", func() {
		near := suite.camera.Near()
		suite.Equal(float32(0.1), near)
	})
}

func (suite *cameraTest) TestFar() {
	suite.Run("should return the camera far plane distance", func() {
		far := suite.camera.Far()
		suite.Equal(float32(100.0), far)
	})
}

func (suite *cameraTest) TestViewMatrix() {
	suite.Run("should return the camera view matrix", func() {
		m := suite.camera.ViewMatrix()
		suite.Equal(16, len(m))
	})
}

func (suite *cameraTest) TestProjectionMatrix() {
	suite.Run("should return the camera projection matrix", func() {
		m := suite.camera.ProjectionMatrix()
		suite.Equal(16, len(m))
	})
}

func (suite *cameraTest) TestViewProjectionMatrix() {
	suite.Run("should return the combined view-projection matrix", func() {
		m := suite.camera.ViewProjectionMatrix()
		suite.Equal(16, len(m))
	})
}

func (suite *cameraTest) TestInverseProjectionMatrix() {
	suite.Run("should return the inverse projection matrix", func() {
		m := suite.camera.InverseProjectionMatrix()
		suite.Equal(16, len(m))
	})
}

func (suite *cameraTest) TestController() {
	suite.Run("should return the attached controller", func() {
		ctrl := suite.camera.Controller()
		suite.Equal(suite.controllerMock, ctrl)
	})
}

func (suite *cameraTest) TestBindGroupProvider() {
	suite.Run("should return a non-nil bind group provider", func() {
		bgp := suite.camera.BindGroupProvider()
		suite.NotNil(bgp)
	})
}

func (suite *cameraTest) TestUpdate() {
	suite.Run("should recompute matrices when a controller is attached", func() {
		suite.camera.Update()
	})

	suite.Run("should be a no-op when no controller is attached", func() {
		c := camera.NewCamera()
		c.Update()
	})
}

func (suite *cameraTest) TestSetUp() {
	suite.Run("should update the camera up vector", func() {
		suite.camera.SetUp(0, 0, 1)
		x, y, z := suite.camera.Up()
		suite.Equal(float32(0), x)
		suite.Equal(float32(0), y)
		suite.Equal(float32(1), z)
	})
}

func (suite *cameraTest) TestSetFov() {
	suite.Run("should update the camera field of view", func() {
		suite.camera.SetFov(float32(90.0 * (math.Pi / 180.0)))
		suite.Equal(float32(90.0*(math.Pi/180.0)), suite.camera.Fov())
	})
}

func (suite *cameraTest) TestSetAspect() {
	suite.Run("should update the camera aspect ratio", func() {
		suite.camera.SetAspect(4.0 / 3.0)
		suite.Equal(float32(4.0/3.0), suite.camera.Aspect())
	})
}

func (suite *cameraTest) TestSetNear() {
	suite.Run("should update the camera near plane distance", func() {
		suite.camera.SetNear(0.5)
		suite.Equal(float32(0.5), suite.camera.Near())
	})
}

func (suite *cameraTest) TestSetFar() {
	suite.Run("should update the camera far plane distance", func() {
		suite.camera.SetFar(500.0)
		suite.Equal(float32(500.0), suite.camera.Far())
	})
}

func (suite *cameraTest) TestSetController() {
	suite.Run("should update the attached controller", func() {
		newMock := camera_mocks.NewMockCameraController(suite.T())
		suite.camera.SetController(newMock)
		suite.Equal(newMock, suite.camera.Controller())
	})
}

func (suite *cameraTest) TestSetBindGroupProvider() {
	suite.Run("should update the bind group provider", func() {
		newBGP := bind_group_provider.NewBindGroupProvider("new-test")
		suite.camera.SetBindGroupProvider(newBGP)
		suite.Equal(newBGP, suite.camera.BindGroupProvider())
	})
}
