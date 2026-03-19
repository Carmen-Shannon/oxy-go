package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/stretchr/testify/suite"
)

func TestRunContactShadowHandlerTests(t *testing.T) {
	suite.Run(t, new(contactShadowHandlerTest))
}

type contactShadowHandlerTest struct {
	suite.Suite
	handler light.ContactShadowHandler
}

func (suite *contactShadowHandlerTest) SetupSubTest() {
	suite.handler = light.NewContactShadowHandler()
}

func (suite *contactShadowHandlerTest) TestNewContactShadowHandler() {
	suite.Run("should create a new contact shadow handler with provided options", func() {
		h := light.NewContactShadowHandler(
			light.WithContactShadowsEnabled(false),
			light.WithContactShadowStepCount(32),
			light.WithContactShadowMaxDistance(2.0),
			light.WithContactShadowThickness(0.1),
		)
		suite.NotNil(h)
	})
}

func (suite *contactShadowHandlerTest) TestEnabled() {
	suite.Run("should return true by default", func() {
		suite.Equal(true, suite.handler.Enabled())
	})
}

func (suite *contactShadowHandlerTest) TestSetEnabled() {
	suite.Run("should update the enabled state", func() {
		suite.handler.SetEnabled(false)
		suite.Equal(false, suite.handler.Enabled())
	})
}

func (suite *contactShadowHandlerTest) TestStepCount() {
	suite.Run("should return the step count", func() {
		suite.Equal(16, suite.handler.StepCount())
	})
}

func (suite *contactShadowHandlerTest) TestMaxDistance() {
	suite.Run("should return the max distance", func() {
		suite.Equal(float32(1.0), suite.handler.MaxDistance())
	})
}

func (suite *contactShadowHandlerTest) TestThickness() {
	suite.Run("should return the thickness", func() {
		suite.Equal(float32(0.05), suite.handler.Thickness())
	})
}

func (suite *contactShadowHandlerTest) TestPipelineKey() {
	suite.Run("should return an empty string for an unknown key", func() {
		suite.Equal("", suite.handler.PipelineKey("unknown"))
	})

	suite.Run("should return the pipeline key after it is set", func() {
		suite.handler.SetPipelineKey("test", "test-key")
		suite.Equal("test-key", suite.handler.PipelineKey("test"))
	})
}

func (suite *contactShadowHandlerTest) TestPipelineKeys() {
	suite.Run("should return the full pipeline keys map", func() {
		suite.handler.SetPipelineKey("a", "akey")
		m := suite.handler.PipelineKeys()
		suite.Equal("akey", m["a"])
	})
}

func (suite *contactShadowHandlerTest) TestSetPipelineKey() {
	suite.Run("should store a pipeline key under the given name", func() {
		suite.handler.SetPipelineKey("pipe", "pipekey")
		suite.Equal("pipekey", suite.handler.PipelineKey("pipe"))
	})
}

func (suite *contactShadowHandlerTest) TestBgp() {
	suite.Run("should return nil for an unknown key", func() {
		suite.Nil(suite.handler.Bgp("unknown"))
	})

	suite.Run("should return the provider after it is set", func() {
		bgp := bind_group_provider.NewBindGroupProvider("test")
		suite.handler.SetBgp("test", bgp)
		suite.Equal(bgp, suite.handler.Bgp("test"))
	})
}

func (suite *contactShadowHandlerTest) TestBgps() {
	suite.Run("should return the full bgps map", func() {
		m := suite.handler.Bgps()
		suite.NotNil(m)
		suite.NotNil(m["contact_shadow_compute"])
	})
}

func (suite *contactShadowHandlerTest) TestSetBgp() {
	suite.Run("should store a bind group provider under the given key", func() {
		bgp := bind_group_provider.NewBindGroupProvider("key")
		suite.handler.SetBgp("key", bgp)
		suite.Equal(bgp, suite.handler.Bgp("key"))
	})
}

func (suite *contactShadowHandlerTest) TestTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.Texture())
	})
}

func (suite *contactShadowHandlerTest) TestSetTexture() {
	suite.Run("should update the texture", func() {
		suite.handler.SetTexture(nil)
		suite.Nil(suite.handler.Texture())
	})
}

func (suite *contactShadowHandlerTest) TestTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.TextureView())
	})
}

func (suite *contactShadowHandlerTest) TestSetTextureView() {
	suite.Run("should update the texture view", func() {
		suite.handler.SetTextureView(nil)
		suite.Nil(suite.handler.TextureView())
	})
}

func (suite *contactShadowHandlerTest) TestLinearSampler() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.LinearSampler())
	})
}

func (suite *contactShadowHandlerTest) TestSetLinearSampler() {
	suite.Run("should update the linear sampler", func() {
		suite.handler.SetLinearSampler(nil)
		suite.Nil(suite.handler.LinearSampler())
	})
}
