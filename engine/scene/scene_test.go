package scene_test

import (
	"os"
	"strings"
	"testing"

	camera_mocks "github.com/Carmen-Shannon/oxy-go/engine/camera/mocks"
	game_object_mocks "github.com/Carmen-Shannon/oxy-go/engine/game_object/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	renderer_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

func TestRunSceneTests(t *testing.T) {
	suite.Run(t, new(sceneTest))
}

type sceneTest struct {
	suite.Suite
	cameraMock   *camera_mocks.MockCamera
	rendererMock *renderer_mocks.MockRenderer
	scene        scene.Scene
}

func (suite *sceneTest) SetupSuite() {
	for {
		if _, err := os.Stat("go.mod"); err == nil {
			break
		}
		if err := os.Chdir(".."); err != nil {
			suite.T().Fatalf("failed to locate module root: %v", err)
		}
	}
}

func (suite *sceneTest) SetupSubTest() {
	suite.cameraMock = camera_mocks.NewMockCamera(suite.T())
	suite.cameraMock.EXPECT().BindGroupProvider().Return(nil).Maybe()
	suite.rendererMock = renderer_mocks.NewMockRenderer(suite.T())
	suite.rendererMock.EXPECT().SetInjections(mock.Anything).Return().Maybe()
	suite.scene = scene.NewScene("test", suite.cameraMock, suite.rendererMock)
}

func (suite *sceneTest) TestNewScene() {
	suite.Run("panics when camera is nil", func() {
		suite.Panics(func() {
			scene.NewScene("test", nil, suite.rendererMock)
		})
	})

	suite.Run("panics when renderer is nil", func() {
		suite.Panics(func() {
			scene.NewScene("test", suite.cameraMock, nil)
		})
	})

	suite.Run("creates scene with valid args", func() {
		suite.NotNil(suite.scene)
	})

	suite.Run("scene name is set from constructor arg", func() {
		suite.Equal("test", suite.scene.Name())
	})
}

func (suite *sceneTest) TestName() {
	suite.Run("returns name set at construction", func() {
		suite.Equal("test", suite.scene.Name())
	})
}

func (suite *sceneTest) TestSetName() {
	suite.Run("updates the scene name", func() {
		suite.scene.SetName("renamed")
		suite.Equal("renamed", suite.scene.Name())
	})
}

func (suite *sceneTest) TestActive() {
	suite.Run("default is false", func() {
		suite.False(suite.scene.Active())
	})

	suite.Run("WithActive true sets initial state to true", func() {
		cam := camera_mocks.NewMockCamera(suite.T())
		cam.EXPECT().BindGroupProvider().Return(nil).Maybe()
		r := renderer_mocks.NewMockRenderer(suite.T())
		r.EXPECT().SetInjections(mock.Anything).Return().Maybe()
		s := scene.NewScene("active-test", cam, r, scene.WithActive(true))
		suite.True(s.Active())
	})
}

func (suite *sceneTest) TestSetActive() {
	suite.Run("can toggle active to true", func() {
		suite.scene.SetActive(true)
		suite.True(suite.scene.Active())
	})

	suite.Run("can toggle active back to false", func() {
		suite.scene.SetActive(true)
		suite.scene.SetActive(false)
		suite.False(suite.scene.Active())
	})
}

func (suite *sceneTest) TestCamera() {
	suite.Run("returns the camera provided at construction", func() {
		suite.Equal(suite.cameraMock, suite.scene.Camera())
	})
}

func (suite *sceneTest) TestSetCamera() {
	suite.Run("replaces the camera", func() {
		newCam := camera_mocks.NewMockCamera(suite.T())
		suite.scene.SetCamera(newCam)
		suite.Equal(newCam, suite.scene.Camera())
	})
}

func (suite *sceneTest) TestRenderer() {
	suite.Run("returns the renderer provided at construction", func() {
		suite.Equal(suite.rendererMock, suite.scene.Renderer())
	})
}

func (suite *sceneTest) TestSetRenderer() {
	suite.Run("replaces the renderer", func() {
		newR := renderer_mocks.NewMockRenderer(suite.T())
		suite.scene.SetRenderer(newR)
		suite.Equal(newR, suite.scene.Renderer())
	})
}

func (suite *sceneTest) TestSetPhysicsHandler() {
	suite.Run("does not panic when setting a physics handler", func() {
		suite.NotPanics(func() {
			ph := physics.NewPhysics()
			suite.scene.SetPhysicsHandler(ph)
		})
	})
}

func (suite *sceneTest) TestCullingDisabled() {
	suite.Run("default is false", func() {
		suite.False(suite.scene.CullingDisabled())
	})

	suite.Run("WithCullingDisabled true sets initial state to true", func() {
		cam := camera_mocks.NewMockCamera(suite.T())
		cam.EXPECT().BindGroupProvider().Return(nil).Maybe()
		r := renderer_mocks.NewMockRenderer(suite.T())
		r.EXPECT().SetInjections(mock.Anything).Return().Maybe()
		s := scene.NewScene("cull-test", cam, r, scene.WithCullingDisabled(true))
		suite.True(s.CullingDisabled())
	})
}

func (suite *sceneTest) TestSetCullingDisabled() {
	suite.Run("can enable culling disabled", func() {
		suite.scene.SetCullingDisabled(true)
		suite.True(suite.scene.CullingDisabled())
	})

	suite.Run("can re-enable culling", func() {
		suite.scene.SetCullingDisabled(true)
		suite.scene.SetCullingDisabled(false)
		suite.False(suite.scene.CullingDisabled())
	})
}

func (suite *sceneTest) TestCount() {
	suite.Run("returns zero for empty scene", func() {
		suite.Equal(0, suite.scene.Count())
	})
}

func (suite *sceneTest) TestCountEphemeral() {
	suite.Run("returns zero for empty scene", func() {
		suite.Equal(0, suite.scene.CountEphemeral())
	})
}

func (suite *sceneTest) TestGet() {
	suite.Run("returns nil for non-existent id", func() {
		suite.Nil(suite.scene.Get(9999))
	})
}

func (suite *sceneTest) TestRemove() {
	suite.Run("no-ops without panicking for non-existent id", func() {
		suite.NotPanics(func() {
			suite.scene.RemoveGameObject(9999)
		})
	})
}

func (suite *sceneTest) TestAmbientColor() {
	suite.Run("default ambient color is zero", func() {
		c := suite.scene.AmbientColor()
		suite.Equal([3]float32{0, 0, 0}, c)
	})
}

func (suite *sceneTest) TestSetAmbientColor() {
	suite.Run("round-trips the ambient color through the light handler", func() {
		color := [3]float32{0.1, 0.2, 0.3}
		suite.scene.SetAmbientColor(color)
		suite.Equal(color, suite.scene.AmbientColor())
	})
}

func (suite *sceneTest) TestLights() {
	suite.Run("returns empty slice for uninitialized scene", func() {
		lights := suite.scene.Lights()
		suite.Empty(lights)
	})
}

func (suite *sceneTest) TestRemoveLight() {
	suite.Run("does not panic when no lights are present", func() {
		suite.NotPanics(func() {
			suite.scene.RemoveLight(nil)
		})
	})
}

func (suite *sceneTest) TestDetachLight() {
	suite.Run("no-op when game object light is nil", func() {
		obj := game_object_mocks.NewMockGameObject(suite.T())
		obj.EXPECT().Light().Return(nil)
		suite.NotPanics(func() {
			suite.scene.DetachLight(obj)
		})
	})
}

func (suite *sceneTest) TestResize() {
	suite.Run("calls renderer Resize with provided dimensions", func() {
		suite.rendererMock.EXPECT().Resize(800, 600).Return().Once()
		suite.cameraMock.EXPECT().SetAspect(float32(800) / float32(600)).Return().Once()
		suite.scene.Resize(800, 600)
	})

	suite.Run("does not call SetAspect when height is zero", func() {
		suite.rendererMock.EXPECT().Resize(800, 0).Return().Once()
		suite.scene.Resize(800, 0)
	})
}

func (suite *sceneTest) TestDrawCalls() {
	suite.Run("returns nil for empty animator pool", func() {
		err := suite.scene.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestPrepareCompute() {
	suite.Run("completes without panic for empty scene with nil bgp", func() {
		suite.cameraMock.EXPECT().Update().Return().Once()
		suite.cameraMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.cameraMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		suite.NotPanics(func() {
			suite.scene.PrepareCompute(0.016)
		})
	})
}

func (suite *sceneTest) TestPrepareShadows() {
	suite.Run("no-ops without panic when lighting not enabled", func() {
		suite.NotPanics(func() {
			suite.scene.PrepareShadows()
		})
	})
}

func (suite *sceneTest) TestPrepareLightCulling() {
	suite.Run("no-ops without panic when lighting not enabled", func() {
		suite.NotPanics(func() {
			suite.scene.PrepareLightCulling()
		})
	})
}

func (suite *sceneTest) TestPrepareGBuffer() {
	suite.Run("no-ops without panic when gbuffer not enabled", func() {
		suite.NotPanics(func() {
			suite.scene.PrepareGBuffer()
		})
	})
}

func (suite *sceneTest) TestPrepareSSAO() {
	suite.Run("no-ops without panic when ssao not enabled", func() {
		suite.NotPanics(func() {
			suite.scene.PrepareSSAO()
		})
	})
}

func (suite *sceneTest) TestPrepareContactShadows() {
	suite.Run("no-ops without panic when contact shadows not enabled", func() {
		suite.NotPanics(func() {
			suite.scene.PrepareContactShadows()
		})
	})
}

func (suite *sceneTest) TestPrepareSSR() {
	suite.Run("no-ops without panic when ssr not enabled", func() {
		suite.NotPanics(func() {
			suite.scene.PrepareSSR()
		})
	})
}

func (suite *sceneTest) TestPrepareComposition() {
	suite.Run("no-ops without panic when composition not enabled", func() {
		suite.NotPanics(func() {
			suite.scene.PrepareComposition()
		})
	})
}

func (suite *sceneTest) TestBeginHDRFrame() {
	suite.Run("returns non-nil error containing composition not initialized", func() {
		err := suite.scene.BeginHDRFrame()
		suite.Error(err)
		suite.True(strings.Contains(err.Error(), "composition not initialized"))
	})
}
