package material_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/stretchr/testify/suite"
)

func TestRunMaterialTests(t *testing.T) {
	suite.Run(t, new(materialTest))
}

type materialTest struct {
	suite.Suite
	m material.Material
}

func (suite *materialTest) SetupSubTest() {
	suite.m = material.NewMaterial()
}

// --- NewMaterial defaults ---

func (suite *materialTest) TestNewMaterialDefaults() {
	suite.Run("base color should default to opaque white", func() {
		suite.Equal([4]float32{1, 1, 1, 1}, suite.m.BaseColor())
	})
	suite.Run("metallic should default to zero", func() {
		suite.InDelta(0.0, float64(suite.m.Metallic()), 1e-6)
	})
	suite.Run("roughness should default to one", func() {
		suite.InDelta(1.0, float64(suite.m.Roughness()), 1e-6)
	})
	suite.Run("alpha cutoff should default to 0.01", func() {
		suite.InDelta(0.01, float64(suite.m.AlphaCutoff()), 1e-6)
	})
	suite.Run("name should default to empty string", func() {
		suite.Equal("", suite.m.Name())
	})
	suite.Run("pipeline key should default to empty string", func() {
		suite.Equal("", suite.m.PipelineKey())
	})
	suite.Run("bind group provider should default to nil", func() {
		suite.Nil(suite.m.BindGroupProvider())
	})
	suite.Run("diffuse texture should default to nil", func() {
		suite.Nil(suite.m.DiffuseTexture())
	})
	suite.Run("normal texture should default to nil", func() {
		suite.Nil(suite.m.NormalTexture())
	})
	suite.Run("metallic roughness texture should default to nil", func() {
		suite.Nil(suite.m.MetallicRoughnessTexture())
	})
	suite.Run("pipeline options should default to nil", func() {
		suite.Nil(suite.m.PipelineOptions())
	})
}

// --- WithName ---

func (suite *materialTest) TestWithName() {
	suite.Run("should set the material name", func() {
		m := material.NewMaterial(material.WithName("mat1"))
		suite.Equal("mat1", m.Name())
	})
}

// --- WithBaseColor ---

func (suite *materialTest) TestWithBaseColor() {
	suite.Run("should set the base color", func() {
		m := material.NewMaterial(material.WithBaseColor([4]float32{0.5, 0.6, 0.7, 0.8}))
		suite.Equal([4]float32{0.5, 0.6, 0.7, 0.8}, m.BaseColor())
	})
}

// --- WithMetallic ---

func (suite *materialTest) TestWithMetallic() {
	suite.Run("should set the metallic factor", func() {
		m := material.NewMaterial(material.WithMetallic(0.9))
		suite.InDelta(0.9, float64(m.Metallic()), 1e-6)
	})
}

// --- WithRoughness ---

func (suite *materialTest) TestWithRoughness() {
	suite.Run("should set the roughness factor", func() {
		m := material.NewMaterial(material.WithRoughness(0.3))
		suite.InDelta(0.3, float64(m.Roughness()), 1e-6)
	})
}

// --- WithAlphaCutoff ---

func (suite *materialTest) TestWithAlphaCutoff() {
	suite.Run("should set the alpha cutoff threshold", func() {
		m := material.NewMaterial(material.WithAlphaCutoff(0.5))
		suite.InDelta(0.5, float64(m.AlphaCutoff()), 1e-6)
	})
}

// --- WithDiffuseTexture ---

func (suite *materialTest) TestWithDiffuseTexture() {
	suite.Run("should set a non-nil diffuse texture", func() {
		m := material.NewMaterial(material.WithDiffuseTexture(&common.ImportedTexture{}))
		suite.NotNil(m.DiffuseTexture())
	})
	suite.Run("should allow setting diffuse texture to nil explicitly", func() {
		m := material.NewMaterial(material.WithDiffuseTexture(nil))
		suite.Nil(m.DiffuseTexture())
	})
}

// --- WithNormalTexture ---

func (suite *materialTest) TestWithNormalTexture() {
	suite.Run("should set a non-nil normal texture", func() {
		m := material.NewMaterial(material.WithNormalTexture(&common.ImportedTexture{}))
		suite.NotNil(m.NormalTexture())
	})
}

// --- WithMetallicRoughnessTexture ---

func (suite *materialTest) TestWithMetallicRoughnessTexture() {
	suite.Run("should set a non-nil metallic roughness texture", func() {
		m := material.NewMaterial(material.WithMetallicRoughnessTexture(&common.ImportedTexture{}))
		suite.NotNil(m.MetallicRoughnessTexture())
	})
}

// --- WithPipelineKey ---

func (suite *materialTest) TestWithPipelineKey() {
	suite.Run("should set the pipeline key", func() {
		m := material.NewMaterial(material.WithPipelineKey("lit"))
		suite.Equal("lit", m.PipelineKey())
	})
}

// --- WithBindGroupProvider ---

func (suite *materialTest) TestWithBindGroupProvider() {
	suite.Run("should set a non-nil bind group provider", func() {
		m := material.NewMaterial(material.WithBindGroupProvider(bind_group_provider.NewBindGroupProvider("test")))
		suite.NotNil(m.BindGroupProvider())
	})
}

// --- WithPipelineOptions ---

func (suite *materialTest) TestWithPipelineOptions() {
	suite.Run("should set pipeline options when args are provided", func() {
		m := material.NewMaterial(material.WithPipelineOptions("opt1", 42))
		suite.Equal([]any{"opt1", 42}, m.PipelineOptions())
	})
	suite.Run("should set nil when no args are provided", func() {
		m := material.NewMaterial(material.WithPipelineOptions())
		suite.Nil(m.PipelineOptions())
	})
}

// --- SetPipelineKey ---

func (suite *materialTest) TestSetPipelineKey() {
	suite.Run("should update the pipeline key on the material", func() {
		suite.m.SetPipelineKey("gbuffer")
		suite.Equal("gbuffer", suite.m.PipelineKey())
	})
}

// --- SetBindGroupProvider ---

func (suite *materialTest) TestSetBindGroupProvider() {
	suite.Run("should set a non-nil bind group provider", func() {
		suite.m.SetBindGroupProvider(bind_group_provider.NewBindGroupProvider("x"))
		suite.NotNil(suite.m.BindGroupProvider())
	})
	suite.Run("should allow setting bind group provider to nil", func() {
		suite.m.SetBindGroupProvider(nil)
		suite.Nil(suite.m.BindGroupProvider())
	})
}

// --- Provider / SetProvider ---

func (suite *materialTest) TestProviderGetSet() {
	suite.Run("should return provider stored at group 0", func() {
		suite.m.SetProvider(0, bind_group_provider.NewBindGroupProvider("g0"))
		suite.NotNil(suite.m.Provider(0))
	})
	suite.Run("should return nil for an unset group", func() {
		suite.Nil(suite.m.Provider(1))
	})
	suite.Run("should allow storing nil for a group", func() {
		suite.m.SetProvider(1, nil)
		suite.Nil(suite.m.Provider(1))
	})
}

// --- GPUMaterialParams ---

func (suite *materialTest) TestGPUMaterialParamsSize() {
	suite.Run("size should be greater than zero", func() {
		p := &material.GPUMaterialParams{}
		suite.Greater(p.Size(), 0)
	})
}

func (suite *materialTest) TestGPUMaterialParamsMarshal() {
	suite.Run("marshal length should equal Size()", func() {
		p := &material.GPUMaterialParams{AlphaCutoff: 0.5}
		buf := p.Marshal()
		suite.Len(buf, p.Size())
	})
	suite.Run("offset 0 should encode AlphaCutoff as float32 LE", func() {
		p := &material.GPUMaterialParams{AlphaCutoff: 0.5}
		buf := p.Marshal()
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[0:4]))
		suite.InDelta(0.5, float64(got), 1e-6)
	})
}

// --- GPUOverlayParams ---

func (suite *materialTest) TestGPUOverlayParamsSize() {
	suite.Run("size should equal 16 bytes", func() {
		p := &material.GPUOverlayParams{}
		suite.Equal(16, p.Size())
	})
}

func (suite *materialTest) TestGPUOverlayParamsMarshal() {
	suite.Run("marshal length should equal 16", func() {
		p := &material.GPUOverlayParams{OverlayColor: [4]float32{1, 2, 3, 4}}
		suite.Len(p.Marshal(), 16)
	})
	suite.Run("offset 0 should encode OverlayColor[0] as float32 LE", func() {
		p := &material.GPUOverlayParams{OverlayColor: [4]float32{1, 2, 3, 4}}
		buf := p.Marshal()
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[0:4]))
		suite.InDelta(1.0, float64(got), 1e-6)
	})
	suite.Run("offset 4 should encode OverlayColor[1] as float32 LE", func() {
		p := &material.GPUOverlayParams{OverlayColor: [4]float32{1, 2, 3, 4}}
		buf := p.Marshal()
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[4:8]))
		suite.InDelta(2.0, float64(got), 1e-6)
	})
	suite.Run("offset 8 should encode OverlayColor[2] as float32 LE", func() {
		p := &material.GPUOverlayParams{OverlayColor: [4]float32{1, 2, 3, 4}}
		buf := p.Marshal()
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[8:12]))
		suite.InDelta(3.0, float64(got), 1e-6)
	})
	suite.Run("offset 12 should encode OverlayColor[3] as float32 LE", func() {
		p := &material.GPUOverlayParams{OverlayColor: [4]float32{1, 2, 3, 4}}
		buf := p.Marshal()
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[12:16]))
		suite.InDelta(4.0, float64(got), 1e-6)
	})
}

// --- GPUEffectParams ---

func (suite *materialTest) TestGPUEffectParamsSize() {
	suite.Run("size should equal 16 bytes", func() {
		p := &material.GPUEffectParams{}
		suite.Equal(16, p.Size())
	})
}

func (suite *materialTest) TestGPUEffectParamsMarshal() {
	suite.Run("marshal length should equal 16", func() {
		p := &material.GPUEffectParams{TintColor: [4]float32{0.1, 0.2, 0.3, 0.4}}
		suite.Len(p.Marshal(), 16)
	})
	suite.Run("offset 0 should encode TintColor[0] as float32 LE", func() {
		p := &material.GPUEffectParams{TintColor: [4]float32{0.1, 0.2, 0.3, 0.4}}
		buf := p.Marshal()
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[0:4]))
		suite.InDelta(0.1, float64(got), 1e-6)
	})
	suite.Run("offset 4 should encode TintColor[1] as float32 LE", func() {
		p := &material.GPUEffectParams{TintColor: [4]float32{0.1, 0.2, 0.3, 0.4}}
		buf := p.Marshal()
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[4:8]))
		suite.InDelta(0.2, float64(got), 1e-6)
	})
	suite.Run("offset 8 should encode TintColor[2] as float32 LE", func() {
		p := &material.GPUEffectParams{TintColor: [4]float32{0.1, 0.2, 0.3, 0.4}}
		buf := p.Marshal()
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[8:12]))
		suite.InDelta(0.3, float64(got), 1e-6)
	})
	suite.Run("offset 12 should encode TintColor[3] as float32 LE", func() {
		p := &material.GPUEffectParams{TintColor: [4]float32{0.1, 0.2, 0.3, 0.4}}
		buf := p.Marshal()
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[12:16]))
		suite.InDelta(0.4, float64(got), 1e-6)
	})
}

// --- Embedded WGSL sources ---

func (suite *materialTest) TestEmbeddedWGSLSourcesNonEmpty() {
	suite.Run("GPUOverlayParamsSource should be non-empty", func() {
		suite.NotEmpty(material.GPUOverlayParamsSource)
	})
	suite.Run("GPUEffectParamsSource should be non-empty", func() {
		suite.NotEmpty(material.GPUEffectParamsSource)
	})
}
