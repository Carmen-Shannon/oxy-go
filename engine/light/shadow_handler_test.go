package light_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
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

	suite.Run("should set near and far from WithShadowNearFar", func() {
		h := light.NewShadowHandler(light.WithShadowNearFar(5.0, 500.0))
		suite.InDelta(5.0, h.ShadowNear(), 1e-6)
		suite.InDelta(500.0, h.ShadowFar(), 1e-6)
	})

	suite.Run("should set normal bias scale from WithShadowNormalBiasScale", func() {
		h := light.NewShadowHandler(light.WithShadowNormalBiasScale(7.0))
		suite.InDelta(7.0, h.ShadowNormalBiasScale(), 1e-6)
	})

	suite.Run("should set map resolution from WithShadowMapResolution", func() {
		h := light.NewShadowHandler(light.WithShadowMapResolution(1024))
		suite.Equal(1024, h.ShadowMapResolution())
	})

	suite.Run("should set PCF radius from WithPCFRadius", func() {
		h := light.NewShadowHandler(light.WithPCFRadius(3.0))
		suite.InDelta(3.0, h.PCFRadius(), 1e-6)
	})

	suite.Run("should set PCF samples from WithPCFSamples", func() {
		h := light.NewShadowHandler(light.WithPCFSamples(32))
		suite.Equal(uint32(32), h.PCFSamples())
	})

	suite.Run("should set PCF samples spot from WithPCFSamplesSpot", func() {
		h := light.NewShadowHandler(light.WithPCFSamplesSpot(4))
		suite.Equal(uint32(4), h.PCFSamplesSpot())
	})

	suite.Run("should set inner radius from WithShadowInnerRadius", func() {
		h := light.NewShadowHandler(light.WithShadowInnerRadius(50.0))
		suite.InDelta(50.0, h.ShadowInnerRadius(), 1e-6)
	})

	suite.Run("should set tile size from WithLightShadowTileSize", func() {
		h := light.NewShadowHandler(light.WithLightShadowTileSize(256))
		suite.Equal(256, h.LightShadowTileSize())
	})

	suite.Run("should use defaults when no options are provided", func() {
		h := light.NewShadowHandler()
		suite.InDelta(0.1, h.ShadowNear(), 1e-6)
		suite.InDelta(200.0, h.ShadowFar(), 1e-6)
		suite.InDelta(3.0, h.ShadowNormalBiasScale(), 1e-6)
		suite.Equal(2048, h.ShadowMapResolution())
		suite.InDelta(1.0, h.PCFRadius(), 1e-6)
		suite.Equal(uint32(16), h.PCFSamples())
		suite.Equal(uint32(8), h.PCFSamplesSpot())
		suite.InDelta(100.0, h.ShadowInnerRadius(), 1e-6)
		suite.Equal(512, h.LightShadowTileSize())
	})
}

func (suite *shadowHandlerTest) TestShadowNear() {
	suite.Run("should return 0.1 by default", func() {
		suite.InDelta(0.1, suite.handler.ShadowNear(), 1e-6)
	})

	suite.Run("should return custom near from WithShadowNearFar", func() {
		h := light.NewShadowHandler(light.WithShadowNearFar(5.0, 500.0))
		suite.InDelta(5.0, h.ShadowNear(), 1e-6)
	})
}

func (suite *shadowHandlerTest) TestShadowFar() {
	suite.Run("should return 200.0 by default", func() {
		suite.InDelta(200.0, suite.handler.ShadowFar(), 1e-6)
	})

	suite.Run("should return custom far from WithShadowNearFar", func() {
		h := light.NewShadowHandler(light.WithShadowNearFar(5.0, 500.0))
		suite.InDelta(500.0, h.ShadowFar(), 1e-6)
	})
}

func (suite *shadowHandlerTest) TestShadowNormalBiasScale() {
	suite.Run("should return 3.0 by default", func() {
		suite.InDelta(3.0, suite.handler.ShadowNormalBiasScale(), 1e-6)
	})

	suite.Run("should return custom scale from WithShadowNormalBiasScale", func() {
		h := light.NewShadowHandler(light.WithShadowNormalBiasScale(7.0))
		suite.InDelta(7.0, h.ShadowNormalBiasScale(), 1e-6)
	})
}

func (suite *shadowHandlerTest) TestShadowMapResolution() {
	suite.Run("should return 2048 by default", func() {
		suite.Equal(2048, suite.handler.ShadowMapResolution())
	})

	suite.Run("should return custom resolution from WithShadowMapResolution", func() {
		h := light.NewShadowHandler(light.WithShadowMapResolution(1024))
		suite.Equal(1024, h.ShadowMapResolution())
	})
}

func (suite *shadowHandlerTest) TestPCFRadius() {
	suite.Run("should return 1.0 by default", func() {
		suite.InDelta(1.0, suite.handler.PCFRadius(), 1e-6)
	})

	suite.Run("should return custom radius from WithPCFRadius", func() {
		h := light.NewShadowHandler(light.WithPCFRadius(3.0))
		suite.InDelta(3.0, h.PCFRadius(), 1e-6)
	})
}

func (suite *shadowHandlerTest) TestPCFSamples() {
	suite.Run("should return 16 by default", func() {
		suite.Equal(uint32(16), suite.handler.PCFSamples())
	})

	suite.Run("should return custom samples from WithPCFSamples", func() {
		h := light.NewShadowHandler(light.WithPCFSamples(32))
		suite.Equal(uint32(32), h.PCFSamples())
	})
}

func (suite *shadowHandlerTest) TestPCFSamplesSpot() {
	suite.Run("should return 8 by default", func() {
		suite.Equal(uint32(8), suite.handler.PCFSamplesSpot())
	})

	suite.Run("should return custom spot samples from WithPCFSamplesSpot", func() {
		h := light.NewShadowHandler(light.WithPCFSamplesSpot(4))
		suite.Equal(uint32(4), h.PCFSamplesSpot())
	})
}

func (suite *shadowHandlerTest) TestCascadeCount() {
	suite.Run("should always return 2", func() {
		suite.Equal(2, suite.handler.CascadeCount())
	})

	suite.Run("should still return 2 with options provided", func() {
		h := light.NewShadowHandler(light.WithShadowNearFar(1.0, 100.0))
		suite.Equal(2, h.CascadeCount())
	})
}

func (suite *shadowHandlerTest) TestShadowInnerRadius() {
	suite.Run("should return 100.0 by default", func() {
		suite.InDelta(100.0, suite.handler.ShadowInnerRadius(), 1e-6)
	})

	suite.Run("should return custom inner radius from WithShadowInnerRadius", func() {
		h := light.NewShadowHandler(light.WithShadowInnerRadius(50.0))
		suite.InDelta(50.0, h.ShadowInnerRadius(), 1e-6)
	})
}

func (suite *shadowHandlerTest) TestLightShadowTileSize() {
	suite.Run("should return 512 by default", func() {
		suite.Equal(512, suite.handler.LightShadowTileSize())
	})

	suite.Run("should return custom tile size from WithLightShadowTileSize", func() {
		h := light.NewShadowHandler(light.WithLightShadowTileSize(256))
		suite.Equal(256, h.LightShadowTileSize())
	})
}

func (suite *shadowHandlerTest) TestComparisonSampler() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.ComparisonSampler())
	})

	suite.Run("should return nil after SetComparisonSampler nil", func() {
		suite.handler.SetComparisonSampler(nil)
		suite.Nil(suite.handler.ComparisonSampler())
	})
}

func (suite *shadowHandlerTest) TestSetComparisonSampler() {
	suite.Run("should update the comparison sampler", func() {
		suite.handler.SetComparisonSampler(nil)
		suite.Nil(suite.handler.ComparisonSampler())
	})

	suite.Run("should round-trip nil through setter and getter", func() {
		suite.handler.SetComparisonSampler(nil)
		retrieved := suite.handler.ComparisonSampler()
		suite.Nil(retrieved)
	})
}

func (suite *shadowHandlerTest) TestCSMAtlasTexture() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.CSMAtlasTexture())
	})

	suite.Run("should return nil after SetCSMAtlasTexture nil", func() {
		suite.handler.SetCSMAtlasTexture(nil)
		suite.Nil(suite.handler.CSMAtlasTexture())
	})
}

func (suite *shadowHandlerTest) TestSetCSMAtlasTexture() {
	suite.Run("should update the CSM atlas texture", func() {
		suite.handler.SetCSMAtlasTexture(nil)
		suite.Nil(suite.handler.CSMAtlasTexture())
	})

	suite.Run("should round-trip nil through setter and getter", func() {
		suite.handler.SetCSMAtlasTexture(nil)
		retrieved := suite.handler.CSMAtlasTexture()
		suite.Nil(retrieved)
	})
}

func (suite *shadowHandlerTest) TestCSMAtlasTextureView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.CSMAtlasTextureView())
	})

	suite.Run("should return nil after SetCSMAtlasTextureView nil", func() {
		suite.handler.SetCSMAtlasTextureView(nil)
		suite.Nil(suite.handler.CSMAtlasTextureView())
	})
}

func (suite *shadowHandlerTest) TestSetCSMAtlasTextureView() {
	suite.Run("should update the CSM atlas texture view", func() {
		suite.handler.SetCSMAtlasTextureView(nil)
		suite.Nil(suite.handler.CSMAtlasTextureView())
	})

	suite.Run("should round-trip nil through setter and getter", func() {
		suite.handler.SetCSMAtlasTextureView(nil)
		retrieved := suite.handler.CSMAtlasTextureView()
		suite.Nil(retrieved)
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

	suite.Run("should return updated provider after overwrite", func() {
		p1 := bind_group_provider.NewBindGroupProvider("test_key")
		suite.handler.SetBgp("test_key", p1)
		p2 := bind_group_provider.NewBindGroupProvider("test_key")
		suite.handler.SetBgp("test_key", p2)
		suite.Equal(p2, suite.handler.Bgp("test_key"))
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

	suite.Run("should return updated value after overwrite", func() {
		suite.handler.SetPipelineKey("shadow_pass", "depth_only_pipeline")
		suite.handler.SetPipelineKey("shadow_pass", "updated_pipeline")
		suite.Equal("updated_pipeline", suite.handler.PipelineKey("shadow_pass"))
	})
}

func (suite *shadowHandlerTest) TestPipelineKeys() {
	suite.Run("should return the full pipeline keys map", func() {
		suite.handler.SetPipelineKey("shadow_pass", "depth_only_pipeline")
		suite.Equal(1, len(suite.handler.PipelineKeys()))
	})

	suite.Run("should return empty map by default", func() {
		suite.Equal(0, len(suite.handler.PipelineKeys()))
	})
}

func (suite *shadowHandlerTest) TestSetPipelineKey() {
	suite.Run("should store a pipeline key", func() {
		suite.handler.SetPipelineKey("shadow_pass", "depth_only_pipeline")
		suite.Equal("depth_only_pipeline", suite.handler.PipelineKey("shadow_pass"))
	})

	suite.Run("should overwrite existing key with new value", func() {
		suite.handler.SetPipelineKey("shadow_pass", "depth_only_pipeline")
		suite.handler.SetPipelineKey("shadow_pass", "updated_pipeline")
		suite.Equal("updated_pipeline", suite.handler.PipelineKey("shadow_pass"))
	})
}

func (suite *shadowHandlerTest) TestLightShadowAtlasSlots() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.handler.LightShadowAtlasSlots())
	})

	suite.Run("should return 96 after SetLightShadowAtlasSlots", func() {
		suite.handler.SetLightShadowAtlasSlots(96)
		suite.Equal(96, suite.handler.LightShadowAtlasSlots())
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

	suite.Run("should return 12 after SetLightShadowAtlasCols", func() {
		suite.handler.SetLightShadowAtlasCols(12)
		suite.Equal(12, suite.handler.LightShadowAtlasCols())
	})
}

func (suite *shadowHandlerTest) TestSetLightShadowAtlasCols() {
	suite.Run("should update the atlas column count", func() {
		suite.handler.SetLightShadowAtlasCols(8)
		suite.Equal(8, suite.handler.LightShadowAtlasCols())
	})

	suite.Run("should set 12 and read back 12", func() {
		suite.handler.SetLightShadowAtlasCols(12)
		suite.Equal(12, suite.handler.LightShadowAtlasCols())
	})
}

func (suite *shadowHandlerTest) TestLightShadowAtlas() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.LightShadowAtlas())
	})

	suite.Run("should return nil after SetLightShadowAtlas nil", func() {
		suite.handler.SetLightShadowAtlas(nil)
		suite.Nil(suite.handler.LightShadowAtlas())
	})
}

func (suite *shadowHandlerTest) TestSetLightShadowAtlas() {
	suite.Run("should update the light shadow atlas", func() {
		suite.handler.SetLightShadowAtlas(nil)
		suite.Nil(suite.handler.LightShadowAtlas())
	})

	suite.Run("should round-trip nil through setter and getter", func() {
		suite.handler.SetLightShadowAtlas(nil)
		retrieved := suite.handler.LightShadowAtlas()
		suite.Nil(retrieved)
	})
}

func (suite *shadowHandlerTest) TestLightShadowAtlasView() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.handler.LightShadowAtlasView())
	})

	suite.Run("should return nil after SetLightShadowAtlasView nil", func() {
		suite.handler.SetLightShadowAtlasView(nil)
		suite.Nil(suite.handler.LightShadowAtlasView())
	})
}

func (suite *shadowHandlerTest) TestSetLightShadowAtlasView() {
	suite.Run("should update the light shadow atlas view", func() {
		suite.handler.SetLightShadowAtlasView(nil)
		suite.Nil(suite.handler.LightShadowAtlasView())
	})

	suite.Run("should round-trip nil through setter and getter", func() {
		suite.handler.SetLightShadowAtlasView(nil)
		retrieved := suite.handler.LightShadowAtlasView()
		suite.Nil(retrieved)
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

	suite.Run("force-dirty then commit clears dirty flag and returns false", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.ForceMarkDirty(l)
		suite.handler.CommitSnapshot(l)
		suite.False(suite.handler.CheckAndMarkDirty(l))
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

	suite.Run("committed light after MarkAllDirty returns false after CommitSnapshot", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.CommitSnapshot(l)
		suite.handler.MarkAllDirty()
		suite.handler.CommitSnapshot(l)
		suite.False(suite.handler.CheckAndMarkDirty(l))
	})
}

func (suite *shadowHandlerTest) TestWithPCFSamplesSpotSetsValue() {
	suite.Run("should set PCFSamplesSpot to the given value", func() {
		h := light.NewShadowHandler(light.WithPCFSamplesSpot(4))
		suite.Equal(uint32(4), h.PCFSamplesSpot())
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

	suite.Run("commit after ForceMarkDirty clears dirty flag", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.CommitSnapshot(l)
		suite.handler.ForceMarkDirty(l)
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

	suite.Run("after OnLightRemoved force-dirtied light has no snapshot so CheckAndMarkDirty returns true", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.CommitSnapshot(l)
		suite.handler.ForceMarkDirty(l)
		suite.handler.OnLightRemoved(l)
		suite.True(suite.handler.CheckAndMarkDirty(l))
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

	suite.Run("force dirty commit force dirty again returns true", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.handler.ForceMarkDirty(l)
		suite.handler.CommitSnapshot(l)
		suite.handler.ForceMarkDirty(l)
		suite.True(suite.handler.CheckAndMarkDirty(l))
	})
}
