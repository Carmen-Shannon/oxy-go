package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/stretchr/testify/suite"
)

func TestRunShadowHandlerTests(t *testing.T) {
	suite.Run(t, new(shadowHandlerTest))
}

type shadowHandlerTest struct {
	suite.Suite
	handler light.ShadowHandler
}

func (suite *shadowHandlerTest) SetupSubTest() {
	suite.handler = light.NewShadowHandler()
}

func (suite *shadowHandlerTest) TestNewShadowHandler() {
	suite.Run("should create a new shadow handler with provided options", func() {
		h := light.NewShadowHandler(
			light.WithShadowNearFar(0.5, 300.0),
			light.WithShadowNormalBiasScale(2.0),
			light.WithShadowMapResolution(4096),
			light.WithPCFRadius(2.0),
			light.WithPCFSamples(32), light.WithPCFSamplesSpot(4), light.WithShadowInnerRadius(50.0),
			light.WithLightShadowTileSize(512),
		)
		suite.NotNil(h)
	})
}

func (suite *shadowHandlerTest) TestShadowNear() {
	suite.Run("should return 0.1 by default", func() {
		suite.InDelta(0.1, suite.handler.ShadowNear(), 1e-6)
	})
}

func (suite *shadowHandlerTest) TestShadowFar() {
	suite.Run("should return 200.0 by default", func() {
		suite.InDelta(200.0, suite.handler.ShadowFar(), 1e-6)
	})
}

func (suite *shadowHandlerTest) TestShadowNormalBiasScale() {
	suite.Run("should return 3.0 by default", func() {
		suite.InDelta(3.0, suite.handler.ShadowNormalBiasScale(), 1e-6)
	})
}

func (suite *shadowHandlerTest) TestShadowMapResolution() {
	suite.Run("should return 2048 by default", func() {
		suite.Equal(2048, suite.handler.ShadowMapResolution())
	})
}

func (suite *shadowHandlerTest) TestPCFRadius() {
	suite.Run("should return 1.0 by default", func() {
		suite.InDelta(1.0, suite.handler.PCFRadius(), 1e-6)
	})
}

func (suite *shadowHandlerTest) TestPCFSamples() {
	suite.Run("should return 16 by default", func() {
		suite.Equal(uint32(16), suite.handler.PCFSamples())
	})
}

func (suite *shadowHandlerTest) TestPCFSamplesSpot() {
	suite.Run("should return 8 by default", func() {
		suite.Equal(uint32(8), suite.handler.PCFSamplesSpot())
	})
}

func (suite *shadowHandlerTest) TestCascadeCount() {
	suite.Run("should always return 2", func() {
		suite.Equal(2, suite.handler.CascadeCount())
	})
}

func (suite *shadowHandlerTest) TestShadowInnerRadius() {
	suite.Run("should return 100.0 by default", func() {
		suite.InDelta(100.0, suite.handler.ShadowInnerRadius(), 1e-6)
	})
}

func (suite *shadowHandlerTest) TestLightShadowTileSize() {
	suite.Run("should return 512 by default", func() {
		suite.Equal(512, suite.handler.LightShadowTileSize())
	})
}

func (suite *shadowHandlerTest) TestComparisonSampler() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.ComparisonSampler())
	})
}

func (suite *shadowHandlerTest) TestSetComparisonSampler() {
	suite.Run("should update the comparison sampler", func() {
		suite.handler.SetComparisonSampler(nil)
		suite.Nil(suite.handler.ComparisonSampler())
	})
}

func (suite *shadowHandlerTest) TestCSMAtlasTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.CSMAtlasTexture())
	})
}

func (suite *shadowHandlerTest) TestSetCSMAtlasTexture() {
	suite.Run("should update the CSM atlas texture", func() {
		suite.handler.SetCSMAtlasTexture(nil)
		suite.Nil(suite.handler.CSMAtlasTexture())
	})
}

func (suite *shadowHandlerTest) TestCSMAtlasTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.CSMAtlasTextureView())
	})
}

func (suite *shadowHandlerTest) TestSetCSMAtlasTextureView() {
	suite.Run("should update the CSM atlas texture view", func() {
		suite.handler.SetCSMAtlasTextureView(nil)
		suite.Nil(suite.handler.CSMAtlasTextureView())
	})
}

func (suite *shadowHandlerTest) TestBgp() {
	suite.Run("should return nil for an unknown key", func() {
		suite.Nil(suite.handler.Bgp("nonexistent"))
	})

	suite.Run("should return the provider after SetBgp", func() {
		suite.handler.SetBgp("test_key", bind_group_provider.NewBindGroupProvider("test_key"))
		suite.NotNil(suite.handler.Bgp("test_key"))
	})
}

func (suite *shadowHandlerTest) TestBgps() {
	suite.Run("should return the full bgp map", func() {
		suite.handler.SetBgp("test_key", bind_group_provider.NewBindGroupProvider("test_key"))
		suite.Equal(1, len(suite.handler.Bgps()))
	})
}

func (suite *shadowHandlerTest) TestSetBgp() {
	suite.Run("should store a bgp under the given key", func() {
		suite.handler.SetBgp("test_key", bind_group_provider.NewBindGroupProvider("test_key"))
		suite.NotNil(suite.handler.Bgp("test_key"))
	})
}

func (suite *shadowHandlerTest) TestPipelineKey() {
	suite.Run("should return empty string for unknown key", func() {
		suite.Equal("", suite.handler.PipelineKey("nonexistent"))
	})

	suite.Run("should return value after SetPipelineKey", func() {
		suite.handler.SetPipelineKey("shadow_pass", "depth_only_pipeline")
		suite.Equal("depth_only_pipeline", suite.handler.PipelineKey("shadow_pass"))
	})
}

func (suite *shadowHandlerTest) TestPipelineKeys() {
	suite.Run("should return the full pipeline keys map", func() {
		suite.handler.SetPipelineKey("shadow_pass", "depth_only_pipeline")
		suite.Equal(1, len(suite.handler.PipelineKeys()))
	})
}

func (suite *shadowHandlerTest) TestSetPipelineKey() {
	suite.Run("should store a pipeline key", func() {
		suite.handler.SetPipelineKey("shadow_pass", "depth_only_pipeline")
		suite.Equal("depth_only_pipeline", suite.handler.PipelineKey("shadow_pass"))
	})
}

func (suite *shadowHandlerTest) TestLightShadowAtlasSlots() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.LightShadowAtlasSlots())
	})
}

func (suite *shadowHandlerTest) TestSetLightShadowAtlasSlots() {
	suite.Run("should update the atlas slot count", func() {
		suite.handler.SetLightShadowAtlasSlots(64)
		suite.Equal(64, suite.handler.LightShadowAtlasSlots())
	})
}

func (suite *shadowHandlerTest) TestLightShadowAtlasCols() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.LightShadowAtlasCols())
	})
}

func (suite *shadowHandlerTest) TestSetLightShadowAtlasCols() {
	suite.Run("should update the atlas column count", func() {
		suite.handler.SetLightShadowAtlasCols(8)
		suite.Equal(8, suite.handler.LightShadowAtlasCols())
	})
}

func (suite *shadowHandlerTest) TestLightShadowAtlas() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.LightShadowAtlas())
	})
}

func (suite *shadowHandlerTest) TestSetLightShadowAtlas() {
	suite.Run("should update the light shadow atlas", func() {
		suite.handler.SetLightShadowAtlas(nil)
		suite.Nil(suite.handler.LightShadowAtlas())
	})
}

func (suite *shadowHandlerTest) TestLightShadowAtlasView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.LightShadowAtlasView())
	})
}

func (suite *shadowHandlerTest) TestSetLightShadowAtlasView() {
	suite.Run("should update the light shadow atlas view", func() {
		suite.handler.SetLightShadowAtlasView(nil)
		suite.Nil(suite.handler.LightShadowAtlasView())
	})
}

func (suite *shadowHandlerTest) TestCheckAndMarkDirty() {
	suite.Run("no prior snapshot marks dirty and returns true", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.True(suite.handler.CheckAndMarkDirty(l))
	})

	suite.Run("unchanged light with committed snapshot returns false", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.CommitSnapshot(l)
		suite.False(suite.handler.CheckAndMarkDirty(l))
	})

	suite.Run("changed position marks dirty and returns true", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.CommitSnapshot(l)
		l.SetPosition(10, 20, 30)
		suite.True(suite.handler.CheckAndMarkDirty(l))
	})

	suite.Run("previously externally marked dirty returns true even if snapshot matches", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.CommitSnapshot(l)
		suite.handler.MarkAllDirty()
		suite.True(suite.handler.CheckAndMarkDirty(l))
	})
}

func (suite *shadowHandlerTest) TestMarkAllDirty() {
	suite.Run("marks all committed lights dirty", func() {
		l1 := light.NewLight(light.LightTypePoint)
		l2 := light.NewLight(light.LightTypeSpot)
		suite.handler.CommitSnapshot(l1)
		suite.handler.CommitSnapshot(l2)
		suite.handler.MarkAllDirty()
		suite.True(suite.handler.CheckAndMarkDirty(l1))
		suite.True(suite.handler.CheckAndMarkDirty(l2))
	})

	suite.Run("lights without snapshots are unaffected", func() {
		suite.NotPanics(func() { suite.handler.MarkAllDirty() })
	})
}

func (suite *shadowHandlerTest) TestCommitSnapshot() {
	suite.Run("commit stores snapshot and clears dirty flag", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.CheckAndMarkDirty(l)
		suite.handler.CommitSnapshot(l)
		suite.False(suite.handler.CheckAndMarkDirty(l))
	})

	suite.Run("re-commit after change clears dirty", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.CommitSnapshot(l)
		l.SetPosition(5, 5, 5)
		suite.handler.CheckAndMarkDirty(l)
		suite.handler.CommitSnapshot(l)
		suite.False(suite.handler.CheckAndMarkDirty(l))
	})
}

func (suite *shadowHandlerTest) TestOnLightRemoved() {
	suite.Run("removes snapshot and dirty flag", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.CommitSnapshot(l)
		suite.handler.OnLightRemoved(l)
		suite.True(suite.handler.CheckAndMarkDirty(l))
		suite.handler.CommitSnapshot(l)
		suite.False(suite.handler.CheckAndMarkDirty(l))
	})

	suite.Run("removing unknown light does not panic", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.NotPanics(func() { suite.handler.OnLightRemoved(l) })
	})
}

func (suite *shadowHandlerTest) TestForceMarkDirty() {
	suite.Run("should mark a clean light dirty", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.CommitSnapshot(l)
		suite.handler.CommitSnapshot(l)
		suite.handler.ForceMarkDirty(l)
		suite.True(suite.handler.CheckAndMarkDirty(l))
	})

	suite.Run("should not panic when called on a light with no prior snapshot", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.NotPanics(func() { suite.handler.ForceMarkDirty(l) })
		suite.True(suite.handler.CheckAndMarkDirty(l))
	})
}
