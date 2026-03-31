package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/stretchr/testify/suite"
)

func TestRunGBufferHandlerTests(t *testing.T) {
	suite.Run(t, new(gBufferHandlerTest))
}

type gBufferHandlerTest struct {
	suite.Suite
	handler light.GBufferHandler
}

func (suite *gBufferHandlerTest) SetupSubTest() {
	suite.handler = light.NewGBufferHandler()
}

func (suite *gBufferHandlerTest) TestNewGBufferHandler() {
	suite.Run("should create a new gbuffer handler with provided options", func() {
		h := light.NewGBufferHandler(
			light.WithGBufferScreenSize(1920, 1080),
		)
		suite.NotNil(h)
	})
}

func (suite *gBufferHandlerTest) TestEnabled() {
	suite.Run("should return false by default", func() {
		suite.Equal(false, suite.handler.Enabled())
	})
}

func (suite *gBufferHandlerTest) TestSetEnabled() {
	suite.Run("should update the enabled state", func() {
		suite.handler.SetEnabled(true)
		suite.Equal(true, suite.handler.Enabled())
	})
}

func (suite *gBufferHandlerTest) TestScreenWidth() {
	suite.Run("should return the screen width", func() {
		suite.Equal(0, suite.handler.ScreenWidth())
	})
}

func (suite *gBufferHandlerTest) TestScreenHeight() {
	suite.Run("should return the screen height", func() {
		suite.Equal(0, suite.handler.ScreenHeight())
	})
}

func (suite *gBufferHandlerTest) TestPipelineKey() {
	suite.Run("should return an empty string for an unknown key", func() {
		suite.Equal("", suite.handler.PipelineKey("unknown"))
	})

	suite.Run("should return the pipeline key after it is set", func() {
		suite.handler.SetPipelineKey("test", "test-key")
		suite.Equal("test-key", suite.handler.PipelineKey("test"))
	})
}

func (suite *gBufferHandlerTest) TestPipelineKeys() {
	suite.Run("should return the full pipeline keys map", func() {
		suite.handler.SetPipelineKey("a", "akey")
		m := suite.handler.PipelineKeys()
		suite.Equal("akey", m["a"])
	})
}

func (suite *gBufferHandlerTest) TestSetPipelineKey() {
	suite.Run("should store a pipeline key under the given name", func() {
		suite.handler.SetPipelineKey("pipe", "pipekey")
		suite.Equal("pipekey", suite.handler.PipelineKey("pipe"))
	})
}

func (suite *gBufferHandlerTest) TestNormalTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.NormalTexture())
	})
}

func (suite *gBufferHandlerTest) TestSetNormalTexture() {
	suite.Run("should update the normal texture", func() {
		suite.handler.SetNormalTexture(nil)
		suite.Nil(suite.handler.NormalTexture())
	})
}

func (suite *gBufferHandlerTest) TestNormalTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.NormalTextureView())
	})
}

func (suite *gBufferHandlerTest) TestSetNormalTextureView() {
	suite.Run("should update the normal texture view", func() {
		suite.handler.SetNormalTextureView(nil)
		suite.Nil(suite.handler.NormalTextureView())
	})
}

func (suite *gBufferHandlerTest) TestAlbedoTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.AlbedoTexture())
	})
}

func (suite *gBufferHandlerTest) TestSetAlbedoTexture() {
	suite.Run("should update the albedo texture", func() {
		suite.handler.SetAlbedoTexture(nil)
		suite.Nil(suite.handler.AlbedoTexture())
	})
}

func (suite *gBufferHandlerTest) TestAlbedoTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.AlbedoTextureView())
	})
}

func (suite *gBufferHandlerTest) TestSetAlbedoTextureView() {
	suite.Run("should update the albedo texture view", func() {
		suite.handler.SetAlbedoTextureView(nil)
		suite.Nil(suite.handler.AlbedoTextureView())
	})
}

func (suite *gBufferHandlerTest) TestDepthTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.DepthTexture())
	})
}

func (suite *gBufferHandlerTest) TestSetDepthTexture() {
	suite.Run("should update the depth texture", func() {
		suite.handler.SetDepthTexture(nil)
		suite.Nil(suite.handler.DepthTexture())
	})
}

func (suite *gBufferHandlerTest) TestDepthTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.DepthTextureView())
	})
}

func (suite *gBufferHandlerTest) TestSetDepthTextureView() {
	suite.Run("should update the depth texture view", func() {
		suite.handler.SetDepthTextureView(nil)
		suite.Nil(suite.handler.DepthTextureView())
	})
}

func (suite *gBufferHandlerTest) TestResize() {
	suite.Run("should update the screen dimensions", func() {
		suite.handler.Resize(1920, 1080)
		suite.Equal(1920, suite.handler.ScreenWidth())
		suite.Equal(1080, suite.handler.ScreenHeight())
	})
}

func (suite *gBufferHandlerTest) TestSetSlot() {
	suite.Run("should set the active slot", func() {
		suite.handler.SetSlot(1)
		suite.Nil(suite.handler.NormalTexture())
	})

	suite.Run("should not affect other slot data when switching slots", func() {
		suite.handler.SetSlot(0)
		suite.Nil(suite.handler.NormalTexture())
		suite.handler.SetSlot(1)
		suite.Nil(suite.handler.NormalTexture())
	})
}
