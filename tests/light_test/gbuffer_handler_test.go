package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type gBufferHandlerTest struct {
	suite.Suite
}

func TestGBufferHandler(t *testing.T) {
	suite.Run(t, new(gBufferHandlerTest))
}

func (suite *gBufferHandlerTest) TestNewGBufferHandler() {
	suite.Run("default enabled is false", func() {
		h := light.NewGBufferHandler()
		suite.False(h.Enabled())
	})

	suite.Run("default screen dimensions are zero", func() {
		h := light.NewGBufferHandler()
		suite.Equal(0, h.ScreenWidth())
		suite.Equal(0, h.ScreenHeight())
	})

	suite.Run("default pipeline keys are empty", func() {
		h := light.NewGBufferHandler()
		suite.Empty(h.PipelineKeys())
	})

	suite.Run("default position texture is nil", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.PositionTexture())
		suite.Nil(h.PositionTextureView())
	})

	suite.Run("default normal texture is nil", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.NormalTexture())
		suite.Nil(h.NormalTextureView())
	})

	suite.Run("default albedo texture is nil", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.AlbedoTexture())
		suite.Nil(h.AlbedoTextureView())
	})

	suite.Run("default depth texture is nil", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.DepthTexture())
		suite.Nil(h.DepthTextureView())
	})
}

func (suite *gBufferHandlerTest) TestWithGBufferScreenSize() {
	suite.Run("sets screen dimensions", func() {
		h := light.NewGBufferHandler(light.WithGBufferScreenSize(1920, 1080))
		suite.Equal(1920, h.ScreenWidth())
		suite.Equal(1080, h.ScreenHeight())
	})
}

func (suite *gBufferHandlerTest) TestSetEnabled() {
	suite.Run("toggles enabled state", func() {
		h := light.NewGBufferHandler()
		suite.False(h.Enabled())
		h.SetEnabled(true)
		suite.True(h.Enabled())
		h.SetEnabled(false)
		suite.False(h.Enabled())
	})
}

func (suite *gBufferHandlerTest) TestPipelineKeys() {
	suite.Run("set and retrieve pipeline key", func() {
		h := light.NewGBufferHandler()
		h.SetPipelineKey("gbuffer", "pipeline-456")
		suite.Equal("pipeline-456", h.PipelineKey("gbuffer"))
	})

	suite.Run("missing key returns empty string", func() {
		h := light.NewGBufferHandler()
		suite.Equal("", h.PipelineKey("nonexistent"))
	})

	suite.Run("pipeline keys map returns all entries", func() {
		h := light.NewGBufferHandler()
		h.SetPipelineKey("a", "key-a")
		h.SetPipelineKey("b", "key-b")
		keys := h.PipelineKeys()
		suite.Len(keys, 2)
		suite.Equal("key-a", keys["a"])
		suite.Equal("key-b", keys["b"])
	})
}

func (suite *gBufferHandlerTest) TestResize() {
	suite.Run("updates screen dimensions", func() {
		h := light.NewGBufferHandler()
		h.Resize(3840, 2160)
		suite.Equal(3840, h.ScreenWidth())
		suite.Equal(2160, h.ScreenHeight())
	})
}

func (suite *gBufferHandlerTest) TestSetPositionTexture() {
	suite.Run("sets and retrieves position texture", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.PositionTexture())
		tex := &wgpu.Texture{}
		h.SetPositionTexture(tex)
		suite.Equal(tex, h.PositionTexture())
	})
}

func (suite *gBufferHandlerTest) TestSetPositionTextureView() {
	suite.Run("sets and retrieves position texture view", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.PositionTextureView())
		tv := &wgpu.TextureView{}
		h.SetPositionTextureView(tv)
		suite.Equal(tv, h.PositionTextureView())
	})
}

func (suite *gBufferHandlerTest) TestSetNormalTexture() {
	suite.Run("sets and retrieves normal texture", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.NormalTexture())
		tex := &wgpu.Texture{}
		h.SetNormalTexture(tex)
		suite.Equal(tex, h.NormalTexture())
	})
}

func (suite *gBufferHandlerTest) TestSetNormalTextureView() {
	suite.Run("sets and retrieves normal texture view", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.NormalTextureView())
		tv := &wgpu.TextureView{}
		h.SetNormalTextureView(tv)
		suite.Equal(tv, h.NormalTextureView())
	})
}

func (suite *gBufferHandlerTest) TestSetAlbedoTexture() {
	suite.Run("sets and retrieves albedo texture", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.AlbedoTexture())
		tex := &wgpu.Texture{}
		h.SetAlbedoTexture(tex)
		suite.Equal(tex, h.AlbedoTexture())
	})
}

func (suite *gBufferHandlerTest) TestSetAlbedoTextureView() {
	suite.Run("sets and retrieves albedo texture view", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.AlbedoTextureView())
		tv := &wgpu.TextureView{}
		h.SetAlbedoTextureView(tv)
		suite.Equal(tv, h.AlbedoTextureView())
	})
}

func (suite *gBufferHandlerTest) TestSetDepthTexture() {
	suite.Run("sets and retrieves depth texture", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.DepthTexture())
		tex := &wgpu.Texture{}
		h.SetDepthTexture(tex)
		suite.Equal(tex, h.DepthTexture())
	})
}

func (suite *gBufferHandlerTest) TestSetDepthTextureView() {
	suite.Run("sets and retrieves depth texture view", func() {
		h := light.NewGBufferHandler()
		suite.Nil(h.DepthTextureView())
		tv := &wgpu.TextureView{}
		h.SetDepthTextureView(tv)
		suite.Equal(tv, h.DepthTextureView())
	})
}
