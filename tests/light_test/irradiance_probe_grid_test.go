package light_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type irradianceProbeGridTest struct {
	suite.Suite
}

func TestIrradianceProbeGrid(t *testing.T) {
	suite.Run(t, new(irradianceProbeGridTest))
}

func (suite *irradianceProbeGridTest) TestNewIrradianceProbeGrid() {
	suite.Run("default enabled is false", func() {
		g := light.NewIrradianceProbeGrid()
		suite.False(g.Enabled())
	})

	suite.Run("default grid counts are 8x4x8", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal(8, g.CountX())
		suite.Equal(4, g.CountY())
		suite.Equal(8, g.CountZ())
	})

	suite.Run("default total probes is 256", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal(256, g.TotalProbes())
	})

	suite.Run("default grid min is correct", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal([3]float32{-10, -2, -10}, g.GridMin())
	})

	suite.Run("default grid max is correct", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal([3]float32{10, 6, 10}, g.GridMax())
	})

	suite.Run("default bake resolution is 32", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal(32, g.BakeResolution())
	})

	suite.Run("spacing is derived from bounds and counts", func() {
		g := light.NewIrradianceProbeGrid()
		spacing := g.Spacing()
		// X: (10 - (-10)) / (8-1) = 20/7 ≈ 2.857
		suite.InDelta(20.0/7.0, spacing[0], 1e-6)
		// Y: (6 - (-2)) / (4-1) = 8/3 ≈ 2.667
		suite.InDelta(8.0/3.0, spacing[1], 1e-6)
		// Z: (10 - (-10)) / (8-1) = 20/7 ≈ 2.857
		suite.InDelta(20.0/7.0, spacing[2], 1e-6)
	})

	suite.Run("probes are pre-allocated with correct count", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Len(g.Probes(), 256)
	})

	suite.Run("first probe position matches grid min", func() {
		g := light.NewIrradianceProbeGrid()
		p := g.Probe(0)
		suite.InDelta(-10.0, p.Position[0], 1e-6)
		suite.InDelta(-2.0, p.Position[1], 1e-6)
		suite.InDelta(-10.0, p.Position[2], 1e-6)
		suite.InDelta(1.0, p.Position[3], 1e-6) // active status
	})

	suite.Run("all probes start dirty", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Len(g.DirtyProbes(), 256)
	})

	suite.Run("default pipeline keys are empty", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Empty(g.PipelineKeys())
	})

	suite.Run("default bgps contain required keys", func() {
		g := light.NewIrradianceProbeGrid()
		suite.NotNil(g.Bgp("probe_grid"))
		suite.NotNil(g.Bgp("probe_sh_project"))
		suite.NotNil(g.Bgp("probe_bake_camera"))
	})

	suite.Run("bgp labels match keys", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal("probe_grid", g.Bgp("probe_grid").Label())
		suite.Equal("probe_sh_project", g.Bgp("probe_sh_project").Label())
		suite.Equal("probe_bake_camera", g.Bgp("probe_bake_camera").Label())
	})

	suite.Run("default bake textures are nil", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Nil(g.BakeColorTexture())
		suite.Nil(g.BakeColorTextureView())
		suite.Nil(g.BakeDepthTexture())
		suite.Nil(g.BakeDepthTextureView())
	})

	suite.Run("default GPU buffers are nil", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Nil(g.ProbeBuffer())
		suite.Nil(g.GridParamsBuffer())
	})
}

func (suite *irradianceProbeGridTest) TestWithProbeGridCounts() {
	suite.Run("overrides grid counts", func() {
		g := light.NewIrradianceProbeGrid(light.WithProbeGridCounts(4, 2, 4))
		suite.Equal(4, g.CountX())
		suite.Equal(2, g.CountY())
		suite.Equal(4, g.CountZ())
		suite.Equal(32, g.TotalProbes())
		suite.Len(g.Probes(), 32)
	})
}

func (suite *irradianceProbeGridTest) TestWithProbeGridBounds() {
	suite.Run("overrides grid bounds", func() {
		min := [3]float32{-5, -1, -5}
		max := [3]float32{5, 3, 5}
		g := light.NewIrradianceProbeGrid(light.WithProbeGridBounds(min, max))
		suite.Equal(min, g.GridMin())
		suite.Equal(max, g.GridMax())
	})
}

func (suite *irradianceProbeGridTest) TestWithProbeBakeResolution() {
	suite.Run("overrides bake resolution", func() {
		g := light.NewIrradianceProbeGrid(light.WithProbeBakeResolution(64))
		suite.Equal(64, g.BakeResolution())
	})
}

func (suite *irradianceProbeGridTest) TestProbeIndex() {
	suite.Run("origin index is zero", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal(0, g.ProbeIndex(0, 0, 0))
	})

	suite.Run("x increments index by 1", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal(1, g.ProbeIndex(1, 0, 0))
	})

	suite.Run("y increments index by countX", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal(8, g.ProbeIndex(0, 1, 0))
	})

	suite.Run("z increments index by countX times countY", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal(32, g.ProbeIndex(0, 0, 1))
	})

	suite.Run("arbitrary position computes correctly", func() {
		g := light.NewIrradianceProbeGrid()
		// index = x + y*countX + z*countX*countY = 3 + 2*8 + 1*8*4 = 3+16+32 = 51
		suite.Equal(51, g.ProbeIndex(3, 2, 1))
	})
}

func (suite *irradianceProbeGridTest) TestSetProbe() {
	suite.Run("updates probe and marks dirty", func() {
		g := light.NewIrradianceProbeGrid()
		g.ClearDirtyProbes()
		suite.Empty(g.DirtyProbes())

		probe := light.GPUIrradianceProbe{
			Position: [4]float32{1, 2, 3, 1},
		}
		g.SetProbe(5, probe)
		suite.Equal(probe, g.Probe(5))
		suite.Contains(g.DirtyProbes(), 5)
	})
}

func (suite *irradianceProbeGridTest) TestSetProbes() {
	suite.Run("replaces entire probe slice", func() {
		g := light.NewIrradianceProbeGrid(light.WithProbeGridCounts(2, 1, 1))
		newProbes := []light.GPUIrradianceProbe{
			{Position: [4]float32{0, 0, 0, 1}},
			{Position: [4]float32{1, 0, 0, 1}},
		}
		g.SetProbes(newProbes)
		suite.Len(g.Probes(), 2)
		suite.Equal(newProbes[0], g.Probe(0))
		suite.Equal(newProbes[1], g.Probe(1))
	})
}

func (suite *irradianceProbeGridTest) TestDirtyProbes() {
	suite.Run("set dirty probes replaces list", func() {
		g := light.NewIrradianceProbeGrid()
		g.SetDirtyProbes([]int{1, 5, 10})
		suite.Equal([]int{1, 5, 10}, g.DirtyProbes())
	})
}

func (suite *irradianceProbeGridTest) TestMarkAllDirty() {
	suite.Run("marks every probe as dirty", func() {
		g := light.NewIrradianceProbeGrid(light.WithProbeGridCounts(2, 2, 2))
		g.ClearDirtyProbes()
		suite.Empty(g.DirtyProbes())
		g.MarkAllDirty()
		suite.Len(g.DirtyProbes(), 8)
	})
}

func (suite *irradianceProbeGridTest) TestClearDirtyProbes() {
	suite.Run("empties the dirty list", func() {
		g := light.NewIrradianceProbeGrid()
		suite.NotEmpty(g.DirtyProbes())
		g.ClearDirtyProbes()
		suite.Empty(g.DirtyProbes())
	})
}

func (suite *irradianceProbeGridTest) TestPipelineKeys() {
	suite.Run("set and retrieve pipeline key", func() {
		g := light.NewIrradianceProbeGrid()
		g.SetPipelineKey("probe_bake", "pipeline-probe")
		suite.Equal("pipeline-probe", g.PipelineKey("probe_bake"))
	})

	suite.Run("missing key returns empty string", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Equal("", g.PipelineKey("nonexistent"))
	})

	suite.Run("pipeline keys map returns all entries", func() {
		g := light.NewIrradianceProbeGrid()
		g.SetPipelineKey("a", "key-a")
		g.SetPipelineKey("b", "key-b")
		keys := g.PipelineKeys()
		suite.Len(keys, 2)
	})
}

func (suite *irradianceProbeGridTest) TestBgps() {
	suite.Run("returns full bgp map", func() {
		g := light.NewIrradianceProbeGrid()
		bgps := g.Bgps()
		suite.Len(bgps, 3)
	})
}

func (suite *irradianceProbeGridTest) TestSetEnabled() {
	suite.Run("toggles enabled state", func() {
		g := light.NewIrradianceProbeGrid()
		suite.False(g.Enabled())
		g.SetEnabled(true)
		suite.True(g.Enabled())
	})
}

func (suite *irradianceProbeGridTest) TestBuildGPUGridParams() {
	suite.Run("returns correct default params", func() {
		g := light.NewIrradianceProbeGrid()
		params := g.BuildGPUGridParams()
		suite.Equal([3]float32{-10, -2, -10}, params.GridMin)
		suite.Equal([3]float32{10, 6, 10}, params.GridMax)
		suite.Equal(uint32(8), params.ProbeCountX)
		suite.Equal(uint32(4), params.ProbeCountY)
		suite.Equal(uint32(8), params.ProbeCountZ)
		suite.Equal(uint32(256), params.TotalProbes)
		spacing := g.Spacing()
		suite.Equal(spacing, params.Spacing)
	})

	suite.Run("returns correct custom params", func() {
		g := light.NewIrradianceProbeGrid(
			light.WithProbeGridCounts(4, 2, 4),
			light.WithProbeGridBounds([3]float32{0, 0, 0}, [3]float32{9, 3, 9}),
		)
		params := g.BuildGPUGridParams()
		suite.Equal([3]float32{0, 0, 0}, params.GridMin)
		suite.Equal([3]float32{9, 3, 9}, params.GridMax)
		suite.Equal(uint32(4), params.ProbeCountX)
		suite.Equal(uint32(2), params.ProbeCountY)
		suite.Equal(uint32(4), params.ProbeCountZ)
		suite.Equal(uint32(32), params.TotalProbes)
	})
}

func (suite *irradianceProbeGridTest) TestSpacingWithSingleCount() {
	suite.Run("single count axis yields zero spacing", func() {
		g := light.NewIrradianceProbeGrid(light.WithProbeGridCounts(1, 1, 1))
		spacing := g.Spacing()
		suite.InDelta(0.0, spacing[0], 1e-6)
		suite.InDelta(0.0, spacing[1], 1e-6)
		suite.InDelta(0.0, spacing[2], 1e-6)
	})
}

func (suite *irradianceProbeGridTest) TestSetProbeBuffer() {
	suite.Run("sets and retrieves probe buffer", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Nil(g.ProbeBuffer())
		buf := &wgpu.Buffer{}
		g.SetProbeBuffer(buf)
		suite.Equal(buf, g.ProbeBuffer())
	})
}

func (suite *irradianceProbeGridTest) TestSetGridParamsBuffer() {
	suite.Run("sets and retrieves grid params buffer", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Nil(g.GridParamsBuffer())
		buf := &wgpu.Buffer{}
		g.SetGridParamsBuffer(buf)
		suite.Equal(buf, g.GridParamsBuffer())
	})
}

func (suite *irradianceProbeGridTest) TestSetBgp() {
	suite.Run("adds new bgp entry", func() {
		g := light.NewIrradianceProbeGrid()
		bgp := bind_group_provider.NewBindGroupProvider("custom_probe_bgp")
		g.SetBgp("custom", bgp)
		suite.Equal(bgp, g.Bgp("custom"))
	})

	suite.Run("overwrites existing bgp entry", func() {
		g := light.NewIrradianceProbeGrid()
		replacement := bind_group_provider.NewBindGroupProvider("replaced_probe_grid")
		g.SetBgp("probe_grid", replacement)
		suite.Equal(replacement, g.Bgp("probe_grid"))
		suite.Equal("replaced_probe_grid", g.Bgp("probe_grid").Label())
	})
}

func (suite *irradianceProbeGridTest) TestSetBakeColorTexture() {
	suite.Run("sets and retrieves bake color texture", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Nil(g.BakeColorTexture())
		tex := &wgpu.Texture{}
		g.SetBakeColorTexture(tex)
		suite.Equal(tex, g.BakeColorTexture())
	})
}

func (suite *irradianceProbeGridTest) TestSetBakeColorTextureView() {
	suite.Run("sets and retrieves bake color texture view", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Nil(g.BakeColorTextureView())
		tv := &wgpu.TextureView{}
		g.SetBakeColorTextureView(tv)
		suite.Equal(tv, g.BakeColorTextureView())
	})
}

func (suite *irradianceProbeGridTest) TestSetBakeDepthTexture() {
	suite.Run("sets and retrieves bake depth texture", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Nil(g.BakeDepthTexture())
		tex := &wgpu.Texture{}
		g.SetBakeDepthTexture(tex)
		suite.Equal(tex, g.BakeDepthTexture())
	})
}

func (suite *irradianceProbeGridTest) TestSetBakeDepthTextureView() {
	suite.Run("sets and retrieves bake depth texture view", func() {
		g := light.NewIrradianceProbeGrid()
		suite.Nil(g.BakeDepthTextureView())
		tv := &wgpu.TextureView{}
		g.SetBakeDepthTextureView(tv)
		suite.Equal(tv, g.BakeDepthTextureView())
	})
}
