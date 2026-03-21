package renderer_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/stretchr/testify/suite"
)

type rendererTest struct {
	suite.Suite
}

func TestRunRendererTests(t *testing.T) {
	suite.Run(t, new(rendererTest))
}

func (suite *rendererTest) TestRendererBackendTypeConstants() {
	suite.Run("BackendTypeWGPU should equal zero", func() {
		suite.Equal(renderer.RendererBackendType(0), renderer.BackendTypeWGPU)
	})
}

func (suite *rendererTest) TestPresentModeConstants() {
	suite.Run("PresentModeVSync should equal zero", func() {
		suite.Equal(renderer.PresentMode(0), renderer.PresentModeVSync)
	})
	suite.Run("PresentModeUncapped should equal one", func() {
		suite.Equal(renderer.PresentMode(1), renderer.PresentModeUncapped)
	})
}

func (suite *rendererTest) TestMSAASampleCountConstants() {
	suite.Run("MSAAOff should equal one", func() {
		suite.Equal(renderer.MSAASampleCount(1), renderer.MSAAOff)
	})
	suite.Run("MSAA4x should equal four", func() {
		suite.Equal(renderer.MSAASampleCount(4), renderer.MSAA4x)
	})
	suite.Run("MSAA8x should equal eight", func() {
		suite.Equal(renderer.MSAASampleCount(8), renderer.MSAA8x)
	})
	suite.Run("MSAA16x should equal sixteen", func() {
		suite.Equal(renderer.MSAASampleCount(16), renderer.MSAA16x)
	})
}
