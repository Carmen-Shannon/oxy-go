package material_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/stretchr/testify/suite"
)

type materialTest struct {
	suite.Suite
}

func TestMaterial(t *testing.T) {
	suite.Run(t, new(materialTest))
}

func (suite *materialTest) TestNewMaterialDefaults() {
	suite.Run("name is empty by default", func() {
		m := material.NewMaterial()
		suite.Equal("", m.Name())
	})

	suite.Run("base color defaults to opaque white", func() {
		m := material.NewMaterial()
		suite.Equal([4]float32{1, 1, 1, 1}, m.BaseColor())
	})

	suite.Run("metallic defaults to zero", func() {
		m := material.NewMaterial()
		suite.Equal(float32(0.0), m.Metallic())
	})

	suite.Run("roughness defaults to one", func() {
		m := material.NewMaterial()
		suite.Equal(float32(1.0), m.Roughness())
	})

	suite.Run("diffuse texture is nil by default", func() {
		m := material.NewMaterial()
		suite.Nil(m.DiffuseTexture())
	})

	suite.Run("normal texture is nil by default", func() {
		m := material.NewMaterial()
		suite.Nil(m.NormalTexture())
	})

	suite.Run("metallic roughness texture is nil by default", func() {
		m := material.NewMaterial()
		suite.Nil(m.MetallicRoughnessTexture())
	})

	suite.Run("pipeline key is empty by default", func() {
		m := material.NewMaterial()
		suite.Equal("", m.PipelineKey())
	})

	suite.Run("bind group provider is nil by default", func() {
		m := material.NewMaterial()
		suite.Nil(m.BindGroupProvider())
	})

	suite.Run("fragment shader path is empty by default", func() {
		m := material.NewMaterial()
		suite.Equal("", m.FragmentShaderPath())
	})

	suite.Run("provider returns nil for any group by default", func() {
		m := material.NewMaterial()
		suite.Nil(m.Provider(0))
		suite.Nil(m.Provider(1))
	})
}

func (suite *materialTest) TestWithNameOption() {
	suite.Run("sets material name", func() {
		m := material.NewMaterial(material.WithName("gold"))
		suite.Equal("gold", m.Name())
	})

	suite.Run("empty name is accepted", func() {
		m := material.NewMaterial(material.WithName(""))
		suite.Equal("", m.Name())
	})

	suite.Run("later name option overwrites earlier", func() {
		m := material.NewMaterial(
			material.WithName("first"),
			material.WithName("second"),
		)
		suite.Equal("second", m.Name())
	})
}

func (suite *materialTest) TestWithBaseColorOption() {
	suite.Run("sets base color", func() {
		color := [4]float32{0.5, 0.3, 0.1, 0.9}
		m := material.NewMaterial(material.WithBaseColor(color))
		suite.Equal(color, m.BaseColor())
	})

	suite.Run("zero color is accepted", func() {
		color := [4]float32{0, 0, 0, 0}
		m := material.NewMaterial(material.WithBaseColor(color))
		suite.Equal(color, m.BaseColor())
	})

	suite.Run("overwrites default white", func() {
		color := [4]float32{1, 0, 0, 1}
		m := material.NewMaterial(material.WithBaseColor(color))
		suite.NotEqual([4]float32{1, 1, 1, 1}, m.BaseColor())
		suite.Equal(color, m.BaseColor())
	})
}

func (suite *materialTest) TestWithMetallicOption() {
	suite.Run("sets metallic factor", func() {
		m := material.NewMaterial(material.WithMetallic(0.75))
		suite.Equal(float32(0.75), m.Metallic())
	})

	suite.Run("zero metallic is dielectric", func() {
		m := material.NewMaterial(material.WithMetallic(0.0))
		suite.Equal(float32(0.0), m.Metallic())
	})

	suite.Run("one metallic is fully metallic", func() {
		m := material.NewMaterial(material.WithMetallic(1.0))
		suite.Equal(float32(1.0), m.Metallic())
	})
}

func (suite *materialTest) TestWithRoughnessOption() {
	suite.Run("sets roughness factor", func() {
		m := material.NewMaterial(material.WithRoughness(0.5))
		suite.Equal(float32(0.5), m.Roughness())
	})

	suite.Run("zero roughness is smooth", func() {
		m := material.NewMaterial(material.WithRoughness(0.0))
		suite.Equal(float32(0.0), m.Roughness())
	})

	suite.Run("one roughness is fully rough", func() {
		m := material.NewMaterial(material.WithRoughness(1.0))
		suite.Equal(float32(1.0), m.Roughness())
	})
}

func (suite *materialTest) TestWithDiffuseTextureOption() {
	suite.Run("sets diffuse texture", func() {
		tex := &common.ImportedTexture{Name: "diffuse", MimeType: "image/png"}
		m := material.NewMaterial(material.WithDiffuseTexture(tex))
		suite.NotNil(m.DiffuseTexture())
		suite.Equal("diffuse", m.DiffuseTexture().Name)
	})

	suite.Run("nil diffuse texture is accepted", func() {
		m := material.NewMaterial(material.WithDiffuseTexture(nil))
		suite.Nil(m.DiffuseTexture())
	})
}

func (suite *materialTest) TestWithNormalTextureOption() {
	suite.Run("sets normal texture", func() {
		tex := &common.ImportedTexture{Name: "normal", MimeType: "image/png"}
		m := material.NewMaterial(material.WithNormalTexture(tex))
		suite.NotNil(m.NormalTexture())
		suite.Equal("normal", m.NormalTexture().Name)
	})

	suite.Run("nil normal texture is accepted", func() {
		m := material.NewMaterial(material.WithNormalTexture(nil))
		suite.Nil(m.NormalTexture())
	})
}

func (suite *materialTest) TestWithMetallicRoughnessTextureOption() {
	suite.Run("sets metallic roughness texture", func() {
		tex := &common.ImportedTexture{Name: "metallic-roughness", MimeType: "image/png"}
		m := material.NewMaterial(material.WithMetallicRoughnessTexture(tex))
		suite.NotNil(m.MetallicRoughnessTexture())
		suite.Equal("metallic-roughness", m.MetallicRoughnessTexture().Name)
	})

	suite.Run("nil metallic roughness texture is accepted", func() {
		m := material.NewMaterial(material.WithMetallicRoughnessTexture(nil))
		suite.Nil(m.MetallicRoughnessTexture())
	})
}

func (suite *materialTest) TestWithPipelineKeyOption() {
	suite.Run("sets pipeline key", func() {
		m := material.NewMaterial(material.WithPipelineKey("lit-pbr"))
		suite.Equal("lit-pbr", m.PipelineKey())
	})

	suite.Run("empty pipeline key is accepted", func() {
		m := material.NewMaterial(material.WithPipelineKey(""))
		suite.Equal("", m.PipelineKey())
	})
}

func (suite *materialTest) TestWithBindGroupProviderOption() {
	suite.Run("sets bind group provider", func() {
		bgp := bind_group_provider.NewBindGroupProvider("mat-bgp")
		m := material.NewMaterial(material.WithBindGroupProvider(bgp))
		suite.NotNil(m.BindGroupProvider())
		suite.Equal("mat-bgp", m.BindGroupProvider().Label())
	})

	suite.Run("nil bind group provider is accepted", func() {
		m := material.NewMaterial(material.WithBindGroupProvider(nil))
		suite.Nil(m.BindGroupProvider())
	})
}

func (suite *materialTest) TestSetPipelineKey() {
	suite.Run("set and get pipeline key round-trips", func() {
		m := material.NewMaterial()
		m.SetPipelineKey("textured")
		suite.Equal("textured", m.PipelineKey())
	})

	suite.Run("overwriting pipeline key replaces previous", func() {
		m := material.NewMaterial(material.WithPipelineKey("first"))
		suite.Equal("first", m.PipelineKey())
		m.SetPipelineKey("second")
		suite.Equal("second", m.PipelineKey())
	})

	suite.Run("setting to empty clears pipeline key", func() {
		m := material.NewMaterial(material.WithPipelineKey("something"))
		m.SetPipelineKey("")
		suite.Equal("", m.PipelineKey())
	})
}

func (suite *materialTest) TestSetBindGroupProvider() {
	suite.Run("set and get bind group provider round-trips", func() {
		m := material.NewMaterial()
		bgp := bind_group_provider.NewBindGroupProvider("provider")
		m.SetBindGroupProvider(bgp)
		suite.NotNil(m.BindGroupProvider())
		suite.Equal("provider", m.BindGroupProvider().Label())
	})

	suite.Run("overwriting bind group provider replaces previous", func() {
		bgp1 := bind_group_provider.NewBindGroupProvider("first")
		bgp2 := bind_group_provider.NewBindGroupProvider("second")
		m := material.NewMaterial(material.WithBindGroupProvider(bgp1))
		suite.Equal("first", m.BindGroupProvider().Label())
		m.SetBindGroupProvider(bgp2)
		suite.Equal("second", m.BindGroupProvider().Label())
	})

	suite.Run("setting to nil clears bind group provider", func() {
		bgp := bind_group_provider.NewBindGroupProvider("temp")
		m := material.NewMaterial(material.WithBindGroupProvider(bgp))
		m.SetBindGroupProvider(nil)
		suite.Nil(m.BindGroupProvider())
	})
}

func (suite *materialTest) TestWithFragmentShaderPathOption() {
	suite.Run("sets fragment shader path", func() {
		m := material.NewMaterial(material.WithFragmentShaderPath("shaders/custom.wgsl"))
		suite.Equal("shaders/custom.wgsl", m.FragmentShaderPath())
	})

	suite.Run("empty path uses engine default", func() {
		m := material.NewMaterial(material.WithFragmentShaderPath(""))
		suite.Equal("", m.FragmentShaderPath())
	})

	suite.Run("later option overwrites earlier", func() {
		m := material.NewMaterial(
			material.WithFragmentShaderPath("first.wgsl"),
			material.WithFragmentShaderPath("second.wgsl"),
		)
		suite.Equal("second.wgsl", m.FragmentShaderPath())
	})
}

func (suite *materialTest) TestFragmentShaderPath() {
	suite.Run("default is empty string", func() {
		m := material.NewMaterial()
		suite.Equal("", m.FragmentShaderPath())
	})

	suite.Run("returns value set by builder option", func() {
		m := material.NewMaterial(material.WithFragmentShaderPath("shaders/overlay.wgsl"))
		suite.Equal("shaders/overlay.wgsl", m.FragmentShaderPath())
	})
}

func (suite *materialTest) TestSetFragmentShaderPath() {
	suite.Run("set and get round-trips", func() {
		m := material.NewMaterial()
		m.SetFragmentShaderPath("shaders/toon.wgsl")
		suite.Equal("shaders/toon.wgsl", m.FragmentShaderPath())
	})

	suite.Run("overwrites previous value", func() {
		m := material.NewMaterial(material.WithFragmentShaderPath("first.wgsl"))
		m.SetFragmentShaderPath("second.wgsl")
		suite.Equal("second.wgsl", m.FragmentShaderPath())
	})

	suite.Run("setting to empty clears the path", func() {
		m := material.NewMaterial(material.WithFragmentShaderPath("some.wgsl"))
		m.SetFragmentShaderPath("")
		suite.Equal("", m.FragmentShaderPath())
	})
}

func (suite *materialTest) TestProvider() {
	suite.Run("returns nil for unset group", func() {
		m := material.NewMaterial()
		suite.Nil(m.Provider(0))
	})

	suite.Run("returns nil for non-existent group index", func() {
		m := material.NewMaterial()
		bgp := bind_group_provider.NewBindGroupProvider("group-1")
		m.SetProvider(1, bgp)
		suite.Nil(m.Provider(99))
	})

	suite.Run("returns provider set at group index", func() {
		m := material.NewMaterial()
		bgp := bind_group_provider.NewBindGroupProvider("tex-group")
		m.SetProvider(2, bgp)
		suite.NotNil(m.Provider(2))
		suite.Equal("tex-group", m.Provider(2).Label())
	})
}

func (suite *materialTest) TestSetProvider() {
	suite.Run("set and get round-trips", func() {
		m := material.NewMaterial()
		bgp := bind_group_provider.NewBindGroupProvider("group-0")
		m.SetProvider(0, bgp)
		suite.NotNil(m.Provider(0))
		suite.Equal("group-0", m.Provider(0).Label())
	})

	suite.Run("overwriting group replaces previous provider", func() {
		m := material.NewMaterial()
		bgp1 := bind_group_provider.NewBindGroupProvider("first")
		bgp2 := bind_group_provider.NewBindGroupProvider("second")
		m.SetProvider(0, bgp1)
		suite.Equal("first", m.Provider(0).Label())
		m.SetProvider(0, bgp2)
		suite.Equal("second", m.Provider(0).Label())
	})

	suite.Run("multiple groups are independent", func() {
		m := material.NewMaterial()
		bgp0 := bind_group_provider.NewBindGroupProvider("group-0")
		bgp1 := bind_group_provider.NewBindGroupProvider("group-1")
		m.SetProvider(0, bgp0)
		m.SetProvider(1, bgp1)
		suite.Equal("group-0", m.Provider(0).Label())
		suite.Equal("group-1", m.Provider(1).Label())
	})

	suite.Run("setting nil clears the provider for that group", func() {
		m := material.NewMaterial()
		bgp := bind_group_provider.NewBindGroupProvider("temp")
		m.SetProvider(0, bgp)
		m.SetProvider(0, nil)
		suite.Nil(m.Provider(0))
	})
}

func (suite *materialTest) TestSetDelegate() {
	suite.Run("set delegate does not panic", func() {
		m := material.NewMaterial()
		suite.NotPanics(func() {
			m.SetDelegate(m)
		})
	})
}

func (suite *materialTest) TestAllOptionsComposed() {
	suite.Run("all options can be composed in single call", func() {
		diffTex := &common.ImportedTexture{Name: "diffuse"}
		normTex := &common.ImportedTexture{Name: "normal"}
		mrTex := &common.ImportedTexture{Name: "mr"}
		bgp := bind_group_provider.NewBindGroupProvider("composed-bgp")

		m := material.NewMaterial(
			material.WithName("pbr-gold"),
			material.WithBaseColor([4]float32{1, 0.84, 0, 1}),
			material.WithMetallic(1.0),
			material.WithRoughness(0.3),
			material.WithDiffuseTexture(diffTex),
			material.WithNormalTexture(normTex),
			material.WithMetallicRoughnessTexture(mrTex),
			material.WithPipelineKey("lit-pbr"),
			material.WithBindGroupProvider(bgp),
			material.WithFragmentShaderPath("shaders/custom.wgsl"),
		)

		suite.Equal("pbr-gold", m.Name())
		suite.Equal([4]float32{1, 0.84, 0, 1}, m.BaseColor())
		suite.Equal(float32(1.0), m.Metallic())
		suite.Equal(float32(0.3), m.Roughness())
		suite.NotNil(m.DiffuseTexture())
		suite.Equal("diffuse", m.DiffuseTexture().Name)
		suite.NotNil(m.NormalTexture())
		suite.Equal("normal", m.NormalTexture().Name)
		suite.NotNil(m.MetallicRoughnessTexture())
		suite.Equal("mr", m.MetallicRoughnessTexture().Name)
		suite.Equal("lit-pbr", m.PipelineKey())
		suite.NotNil(m.BindGroupProvider())
		suite.Equal("composed-bgp", m.BindGroupProvider().Label())
		suite.Equal("shaders/custom.wgsl", m.FragmentShaderPath())
	})
}
