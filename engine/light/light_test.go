package light_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
)

func TestRunLightTests(t *testing.T) {
	suite.Run(t, new(lightTest))
}

type lightTest struct {
	suite.Suite
	light light.Light
}

func (suite *lightTest) SetupSubTest() {
	suite.light = light.NewLight(light.LightTypeDirectional)
}

func (suite *lightTest) TestNewLight() {
	suite.Run("should create a new light with provided options", func() {
		l := light.NewLight(
			light.LightTypeSpot,
			light.WithPosition(1, 2, 3),
			light.WithDirection(0, -1, 0),
			light.WithColor(1, 0, 0),
			light.WithIntensity(2.0),
			light.WithRange(50.0),
			light.WithSpotCone(25.0, 35.0),
			light.WithEnabled(false),
			light.WithEphemeral(true),
			light.WithCastsShadows(true),
			light.WithShadowBias(0.001),
		)
		suite.NotNil(l)
	})
}

func (suite *lightTest) TestType() {
	suite.Run("should return LightTypeDirectional", func() {
		l := light.NewLight(light.LightTypeDirectional)
		suite.Equal(light.LightTypeDirectional, l.Type())
	})

	suite.Run("should return LightTypePoint", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.Equal(light.LightTypePoint, l.Type())
	})

	suite.Run("should return LightTypeSpot", func() {
		l := light.NewLight(light.LightTypeSpot)
		suite.Equal(light.LightTypeSpot, l.Type())
	})
}

func (suite *lightTest) TestPosition() {
	suite.Run("should return the light position", func() {
		pos := suite.light.Position()
		suite.Equal(float32(0), pos[0])
		suite.Equal(float32(0), pos[1])
		suite.Equal(float32(0), pos[2])
	})
}

func (suite *lightTest) TestSetPosition() {
	suite.Run("should update the light position", func() {
		suite.light.SetPosition(1, 2, 3)
		suite.Equal([3]float32{1, 2, 3}, suite.light.Position())
	})
}

func (suite *lightTest) TestDirection() {
	suite.Run("should return the light direction", func() {
		suite.Equal([3]float32{0, -1, 0}, suite.light.Direction())
	})
}

func (suite *lightTest) TestSetDirection() {
	suite.Run("should normalize a non-zero direction vector", func() {
		suite.light.SetDirection(1, 0, 0)
		suite.Equal([3]float32{1, 0, 0}, suite.light.Direction())
	})

	suite.Run("should return a zero vector for a zero-length direction", func() {
		suite.light.SetDirection(0, 0, 0)
		suite.Equal([3]float32{0, 0, 0}, suite.light.Direction())
	})
}

func (suite *lightTest) TestColor() {
	suite.Run("should return the light color", func() {
		suite.Equal([3]float32{1, 1, 1}, suite.light.Color())
	})
}

func (suite *lightTest) TestSetColor() {
	suite.Run("should update the light color", func() {
		suite.light.SetColor(1, 0, 0)
		suite.Equal([3]float32{1, 0, 0}, suite.light.Color())
	})
}

func (suite *lightTest) TestIntensity() {
	suite.Run("should return the light intensity", func() {
		suite.Equal(float32(1.0), suite.light.Intensity())
	})
}

func (suite *lightTest) TestSetIntensity() {
	suite.Run("should update the light intensity", func() {
		suite.light.SetIntensity(5.0)
		suite.Equal(float32(5.0), suite.light.Intensity())
	})
}

func (suite *lightTest) TestRange() {
	suite.Run("should return the light range", func() {
		suite.Equal(float32(10.0), suite.light.Range())
	})
}

func (suite *lightTest) TestSetRange() {
	suite.Run("should update the light range", func() {
		suite.light.SetRange(50.0)
		suite.Equal(float32(50.0), suite.light.Range())
	})
}

func (suite *lightTest) TestInnerCone() {
	suite.Run("should return the inner cone value", func() {
		suite.Equal(float32(0.9063), suite.light.InnerCone())
	})
}

func (suite *lightTest) TestOuterCone() {
	suite.Run("should return the outer cone value", func() {
		suite.Equal(float32(0.8192), suite.light.OuterCone())
	})
}

func (suite *lightTest) TestSetSpotCone() {
	suite.Run("should update the inner and outer cone values", func() {
		suite.light.SetSpotCone(0.0, 90.0)
		suite.Equal(float32(1.0), suite.light.InnerCone())
		suite.InDelta(float64(suite.light.OuterCone()), 0.0, 0.0001)
	})
}

func (suite *lightTest) TestEnabled() {
	suite.Run("should return true by default", func() {
		suite.Equal(true, suite.light.Enabled())
	})
}

func (suite *lightTest) TestSetEnabled() {
	suite.Run("should update the enabled state", func() {
		suite.light.SetEnabled(false)
		suite.Equal(false, suite.light.Enabled())
	})
}

func (suite *lightTest) TestEphemeral() {
	suite.Run("should return false by default", func() {
		suite.Equal(false, suite.light.Ephemeral())
	})
}

func (suite *lightTest) TestSetEphemeral() {
	suite.Run("should update the ephemeral state", func() {
		suite.light.SetEphemeral(true)
		suite.Equal(true, suite.light.Ephemeral())
	})
}

func (suite *lightTest) TestCastsShadows() {
	suite.Run("should return false by default", func() {
		suite.Equal(false, suite.light.CastsShadows())
	})
}

func (suite *lightTest) TestSetCastsShadows() {
	suite.Run("should update the casts shadows state", func() {
		suite.light.SetCastsShadows(true)
		suite.Equal(true, suite.light.CastsShadows())
	})
}

func (suite *lightTest) TestShadowBias() {
	suite.Run("should return the shadow bias", func() {
		suite.Equal(float32(0.0002), suite.light.ShadowBias())
	})
}

func (suite *lightTest) TestSetShadowBias() {
	suite.Run("should update the shadow bias", func() {
		suite.light.SetShadowBias(0.001)
		suite.Equal(float32(0.001), suite.light.ShadowBias())
	})
}
