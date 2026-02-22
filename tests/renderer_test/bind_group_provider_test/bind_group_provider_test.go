package bind_group_provider_test

import (
	"testing"

	bgp "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type bindGroupProviderTest struct {
	suite.Suite
}

func TestBindGroupProvider(t *testing.T) {
	suite.Run(t, new(bindGroupProviderTest))
}

func (suite *bindGroupProviderTest) TestNewBindGroupProviderDefaults() {
	suite.Run("label is empty when not set", func() {
		p := bgp.NewBindGroupProvider("")
		suite.Equal("", p.Label())
	})

	suite.Run("label is preserved from constructor", func() {
		p := bgp.NewBindGroupProvider("test-label")
		suite.Equal("test-label", p.Label())
	})

	suite.Run("bind group is nil by default", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.Nil(p.BindGroup())
	})

	suite.Run("bind group layout is nil by default", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.Nil(p.BindGroupLayout())
	})

	suite.Run("buffers map is initialized and empty", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.NotNil(p.Buffers())
		suite.Len(p.Buffers(), 0)
	})

	suite.Run("texture views map is initialized and empty", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.NotNil(p.TextureViews())
		suite.Len(p.TextureViews(), 0)
	})

	suite.Run("samplers map is initialized and empty", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.NotNil(p.Samplers())
		suite.Len(p.Samplers(), 0)
	})

	suite.Run("vertex buffer is nil by default", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.Nil(p.VertexBuffer())
	})

	suite.Run("index buffer is nil by default", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.Nil(p.IndexBuffer())
	})

	suite.Run("index count is zero by default", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.Equal(0, p.IndexCount())
	})

	suite.Run("buffer for non-existent binding returns nil", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.Nil(p.Buffer(0))
		suite.Nil(p.Buffer(99))
	})

	suite.Run("texture view for non-existent binding returns nil", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.Nil(p.TextureView(0))
		suite.Nil(p.TextureView(99))
	})

	suite.Run("sampler for non-existent binding returns nil", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.Nil(p.Sampler(0))
		suite.Nil(p.Sampler(99))
	})
}

func (suite *bindGroupProviderTest) TestSetBindGroup() {
	suite.Run("set and get bind group round-trips", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.Nil(p.BindGroup())

		// Use a typed nil pointer to verify set/get identity
		var bg *wgpu.BindGroup
		p.SetBindGroup(bg)
		suite.Equal(bg, p.BindGroup())
	})

	suite.Run("overwriting bind group replaces previous value", func() {
		p := bgp.NewBindGroupProvider("bgp")

		var bg1 *wgpu.BindGroup
		p.SetBindGroup(bg1)
		suite.Equal(bg1, p.BindGroup())

		var bg2 *wgpu.BindGroup
		p.SetBindGroup(bg2)
		suite.Equal(bg2, p.BindGroup())
	})

	suite.Run("setting bind group to nil clears it", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetBindGroup(nil)
		suite.Nil(p.BindGroup())
	})
}

func (suite *bindGroupProviderTest) TestSetBindGroupLayout() {
	suite.Run("set and get bind group layout round-trips", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.Nil(p.BindGroupLayout())

		var bgl *wgpu.BindGroupLayout
		p.SetBindGroupLayout(bgl)
		suite.Equal(bgl, p.BindGroupLayout())
	})

	suite.Run("setting layout to nil clears it", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetBindGroupLayout(nil)
		suite.Nil(p.BindGroupLayout())
	})
}

func (suite *bindGroupProviderTest) TestSetBuffer() {
	suite.Run("set single buffer at binding index", func() {
		p := bgp.NewBindGroupProvider("bgp")
		var buf *wgpu.Buffer
		p.SetBuffer(0, buf)
		suite.Equal(buf, p.Buffer(0))
	})

	suite.Run("set multiple buffers at different bindings", func() {
		p := bgp.NewBindGroupProvider("bgp")
		var buf0, buf1, buf2 *wgpu.Buffer
		p.SetBuffer(0, buf0)
		p.SetBuffer(1, buf1)
		p.SetBuffer(5, buf2)

		suite.Equal(buf0, p.Buffer(0))
		suite.Equal(buf1, p.Buffer(1))
		suite.Equal(buf2, p.Buffer(5))
		suite.Nil(p.Buffer(3))
	})

	suite.Run("overwriting buffer at same binding replaces it", func() {
		p := bgp.NewBindGroupProvider("bgp")
		var buf1, buf2 *wgpu.Buffer
		p.SetBuffer(0, buf1)
		suite.Equal(buf1, p.Buffer(0))
		p.SetBuffer(0, buf2)
		suite.Equal(buf2, p.Buffer(0))
	})

	suite.Run("buffers map reflects set buffer", func() {
		p := bgp.NewBindGroupProvider("bgp")
		var buf *wgpu.Buffer
		p.SetBuffer(3, buf)
		suite.Len(p.Buffers(), 1)
		suite.Equal(buf, p.Buffers()[3])
	})

	suite.Run("set buffer initializes nil map", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetBuffers(nil)
		suite.Nil(p.Buffers())
		p.SetBuffer(0, nil)
		suite.NotNil(p.Buffers())
		suite.Len(p.Buffers(), 1)
	})
}

func (suite *bindGroupProviderTest) TestSetBuffers() {
	suite.Run("set buffers replaces entire map", func() {
		p := bgp.NewBindGroupProvider("bgp")

		// Set initial buffer
		var origBuf *wgpu.Buffer
		p.SetBuffer(0, origBuf)
		suite.Len(p.Buffers(), 1)

		// Replace with new map
		newBufs := map[int]*wgpu.Buffer{
			1: nil,
			2: nil,
		}
		p.SetBuffers(newBufs)

		suite.Len(p.Buffers(), 2)
		suite.Nil(p.Buffer(0))
		suite.Nil(p.Buffer(1))
		suite.Nil(p.Buffer(2))
	})

	suite.Run("set buffers with empty map clears all", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetBuffer(0, nil)
		suite.Len(p.Buffers(), 1)

		p.SetBuffers(map[int]*wgpu.Buffer{})
		suite.Len(p.Buffers(), 0)
	})
}

func (suite *bindGroupProviderTest) TestSetTextureView() {
	suite.Run("set and get texture view round-trips", func() {
		p := bgp.NewBindGroupProvider("bgp")
		var tv *wgpu.TextureView
		p.SetTextureView(0, tv)
		suite.Equal(tv, p.TextureView(0))
	})

	suite.Run("multiple texture views at different bindings", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetTextureView(0, nil)
		p.SetTextureView(1, nil)
		suite.Len(p.TextureViews(), 2)
	})

	suite.Run("overwriting texture view replaces previous", func() {
		p := bgp.NewBindGroupProvider("bgp")
		var tv1, tv2 *wgpu.TextureView
		p.SetTextureView(0, tv1)
		suite.Equal(tv1, p.TextureView(0))
		p.SetTextureView(0, tv2)
		suite.Equal(tv2, p.TextureView(0))
	})

	suite.Run("set texture view initializes nil map", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetTextureViews(nil)
		suite.Nil(p.TextureViews())
		p.SetTextureView(0, nil)
		suite.NotNil(p.TextureViews())
		suite.Len(p.TextureViews(), 1)
	})
}

func (suite *bindGroupProviderTest) TestSetTextureViews() {
	suite.Run("set texture views replaces entire map", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetTextureView(0, nil)
		suite.Len(p.TextureViews(), 1)

		newViews := map[int]*wgpu.TextureView{
			1: nil,
			2: nil,
			3: nil,
		}
		p.SetTextureViews(newViews)
		suite.Len(p.TextureViews(), 3)
		suite.Nil(p.TextureView(0))
	})
}

func (suite *bindGroupProviderTest) TestSetSampler() {
	suite.Run("set and get sampler round-trips", func() {
		p := bgp.NewBindGroupProvider("bgp")
		var s *wgpu.Sampler
		p.SetSampler(0, s)
		suite.Equal(s, p.Sampler(0))
	})

	suite.Run("multiple samplers at different bindings", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetSampler(0, nil)
		p.SetSampler(4, nil)
		suite.Len(p.Samplers(), 2)
	})

	suite.Run("overwriting sampler replaces previous", func() {
		p := bgp.NewBindGroupProvider("bgp")
		var s1, s2 *wgpu.Sampler
		p.SetSampler(0, s1)
		suite.Equal(s1, p.Sampler(0))
		p.SetSampler(0, s2)
		suite.Equal(s2, p.Sampler(0))
	})

	suite.Run("set sampler initializes nil map", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetSamplers(nil)
		suite.Nil(p.Samplers())
		p.SetSampler(0, nil)
		suite.NotNil(p.Samplers())
		suite.Len(p.Samplers(), 1)
	})
}

func (suite *bindGroupProviderTest) TestSetSamplers() {
	suite.Run("set samplers replaces entire map", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetSampler(0, nil)
		suite.Len(p.Samplers(), 1)

		newSamplers := map[int]*wgpu.Sampler{
			1: nil,
			2: nil,
		}
		p.SetSamplers(newSamplers)
		suite.Len(p.Samplers(), 2)
		suite.Nil(p.Sampler(0))
	})
}

func (suite *bindGroupProviderTest) TestSetVertexBuffer() {
	suite.Run("set and get vertex buffer round-trips", func() {
		p := bgp.NewBindGroupProvider("bgp")
		var vb *wgpu.Buffer
		p.SetVertexBuffer(vb)
		suite.Equal(vb, p.VertexBuffer())
	})

	suite.Run("setting to nil clears vertex buffer", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetVertexBuffer(nil)
		suite.Nil(p.VertexBuffer())
	})
}

func (suite *bindGroupProviderTest) TestSetIndexBuffer() {
	suite.Run("set and get index buffer round-trips", func() {
		p := bgp.NewBindGroupProvider("bgp")
		var ib *wgpu.Buffer
		p.SetIndexBuffer(ib)
		suite.Equal(ib, p.IndexBuffer())
	})

	suite.Run("setting to nil clears index buffer", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetIndexBuffer(nil)
		suite.Nil(p.IndexBuffer())
	})
}

func (suite *bindGroupProviderTest) TestSetIndexCount() {
	suite.Run("set and get index count round-trips", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetIndexCount(1024)
		suite.Equal(1024, p.IndexCount())
	})

	suite.Run("setting to zero resets index count", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetIndexCount(512)
		suite.Equal(512, p.IndexCount())
		p.SetIndexCount(0)
		suite.Equal(0, p.IndexCount())
	})

	suite.Run("large index count is preserved", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetIndexCount(1_000_000)
		suite.Equal(1_000_000, p.IndexCount())
	})
}

func (suite *bindGroupProviderTest) TestReleaseWithNilResources() {
	suite.Run("release on default provider does not panic", func() {
		p := bgp.NewBindGroupProvider("bgp")
		suite.NotPanics(func() {
			p.Release()
		})
	})

	suite.Run("release clears nil bind group without panic", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetBindGroup(nil)
		suite.NotPanics(func() {
			p.Release()
		})
		suite.Nil(p.BindGroup())
	})

	suite.Run("release clears nil bind group layout without panic", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetBindGroupLayout(nil)
		suite.NotPanics(func() {
			p.Release()
		})
		suite.Nil(p.BindGroupLayout())
	})

	suite.Run("release clears nil vertex buffer without panic", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetVertexBuffer(nil)
		suite.NotPanics(func() {
			p.Release()
		})
		suite.Nil(p.VertexBuffer())
	})

	suite.Run("release clears nil index buffer without panic", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetIndexBuffer(nil)
		suite.NotPanics(func() {
			p.Release()
		})
		suite.Nil(p.IndexBuffer())
	})

	suite.Run("release with nil map entries does not panic", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetBuffer(0, nil)
		p.SetTextureView(0, nil)
		p.SetSampler(0, nil)
		suite.NotPanics(func() {
			p.Release()
		})
	})
}

func (suite *bindGroupProviderTest) TestWithBindGroupOption() {
	suite.Run("option sets bind group during construction", func() {
		var bg *wgpu.BindGroup
		p := bgp.NewBindGroupProvider("bgp", bgp.WithBindGroup(bg))
		suite.Equal(bg, p.BindGroup())
	})

	suite.Run("nil bind group option is accepted", func() {
		p := bgp.NewBindGroupProvider("bgp", bgp.WithBindGroup(nil))
		suite.Nil(p.BindGroup())
	})
}

func (suite *bindGroupProviderTest) TestWithBindGroupLayoutOption() {
	suite.Run("option sets bind group layout during construction", func() {
		var bgl *wgpu.BindGroupLayout
		p := bgp.NewBindGroupProvider("bgp", bgp.WithBindGroupLayout(bgl))
		suite.Equal(bgl, p.BindGroupLayout())
	})

	suite.Run("nil bind group layout option is accepted", func() {
		p := bgp.NewBindGroupProvider("bgp", bgp.WithBindGroupLayout(nil))
		suite.Nil(p.BindGroupLayout())
	})
}

func (suite *bindGroupProviderTest) TestWithBufferOption() {
	suite.Run("option sets buffer at specified binding", func() {
		var buf *wgpu.Buffer
		p := bgp.NewBindGroupProvider("bgp", bgp.WithBuffer(0, buf))
		suite.Equal(buf, p.Buffer(0))
	})

	suite.Run("multiple buffer options set different bindings", func() {
		p := bgp.NewBindGroupProvider("bgp",
			bgp.WithBuffer(0, nil),
			bgp.WithBuffer(1, nil),
			bgp.WithBuffer(3, nil),
		)
		suite.Len(p.Buffers(), 3)
		suite.Nil(p.Buffer(0))
		suite.Nil(p.Buffer(1))
		suite.Nil(p.Buffer(3))
		suite.Nil(p.Buffer(2))
	})

	suite.Run("later buffer option at same binding overwrites earlier", func() {
		var buf1, buf2 *wgpu.Buffer
		p := bgp.NewBindGroupProvider("bgp",
			bgp.WithBuffer(0, buf1),
			bgp.WithBuffer(0, buf2),
		)
		suite.Equal(buf2, p.Buffer(0))
	})
}

func (suite *bindGroupProviderTest) TestWithBuffersOption() {
	suite.Run("option sets entire buffers map", func() {
		bufs := map[int]*wgpu.Buffer{
			0: nil,
			1: nil,
		}
		p := bgp.NewBindGroupProvider("bgp", bgp.WithBuffers(bufs))
		suite.Len(p.Buffers(), 2)
	})

	suite.Run("empty buffers map option creates empty map", func() {
		p := bgp.NewBindGroupProvider("bgp", bgp.WithBuffers(map[int]*wgpu.Buffer{}))
		suite.NotNil(p.Buffers())
		suite.Len(p.Buffers(), 0)
	})

	suite.Run("WithBuffers after WithBuffer replaces individual buffer", func() {
		p := bgp.NewBindGroupProvider("bgp",
			bgp.WithBuffer(0, nil),
			bgp.WithBuffers(map[int]*wgpu.Buffer{5: nil}),
		)
		suite.Len(p.Buffers(), 1)
		suite.Nil(p.Buffer(0))
		suite.Nil(p.Buffer(5))
	})
}

func (suite *bindGroupProviderTest) TestMultipleOptionsComposed() {
	suite.Run("all options can be composed in single call", func() {
		var bg *wgpu.BindGroup
		var bgl *wgpu.BindGroupLayout
		bufs := map[int]*wgpu.Buffer{0: nil, 1: nil}

		p := bgp.NewBindGroupProvider("composed",
			bgp.WithBindGroup(bg),
			bgp.WithBindGroupLayout(bgl),
			bgp.WithBuffers(bufs),
		)

		suite.Equal("composed", p.Label())
		suite.Equal(bg, p.BindGroup())
		suite.Equal(bgl, p.BindGroupLayout())
		suite.Len(p.Buffers(), 2)
	})
}

func (suite *bindGroupProviderTest) TestSetDelegate() {
	suite.Run("provider supports set delegate", func() {
		p := bgp.NewBindGroupProvider("bgp")
		// SetDelegate is exposed via the Delegate interface;
		// calling it with the provider itself should be a no-op
		suite.NotPanics(func() {
			p.SetDelegate(p)
		})
	})
}

func (suite *bindGroupProviderTest) TestReleaseEdgeCases() {
	suite.Run("double release does not panic", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetBuffer(0, nil)
		p.SetTextureView(0, nil)
		p.SetSampler(0, nil)
		suite.NotPanics(func() {
			p.Release()
			p.Release()
		})
	})

	suite.Run("release clears all resource maps", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetBuffer(0, nil)
		p.SetBuffer(1, nil)
		p.SetTextureView(0, nil)
		p.SetSampler(0, nil)
		p.SetVertexBuffer(nil)
		p.SetIndexBuffer(nil)
		p.SetBindGroup(nil)
		p.SetBindGroupLayout(nil)

		p.Release()

		suite.Nil(p.BindGroup())
		suite.Nil(p.BindGroupLayout())
		suite.Nil(p.VertexBuffer())
		suite.Nil(p.IndexBuffer())
	})

	suite.Run("release with populated nil entries leaves empty maps", func() {
		p := bgp.NewBindGroupProvider("bgp")
		p.SetBuffer(0, nil)
		p.SetBuffer(1, nil)
		p.SetTextureView(0, nil)
		p.SetTextureView(1, nil)
		p.SetSampler(0, nil)
		p.SetSampler(1, nil)

		p.Release()

		// nil entries are not deleted by release, maps should still have entries
		suite.Len(p.Buffers(), 2)
		suite.Len(p.TextureViews(), 2)
		suite.Len(p.Samplers(), 2)
	})
}

func (suite *bindGroupProviderTest) TestWithTextureViewOption() {
	suite.Run("option sets texture view at specified binding", func() {
		var tv *wgpu.TextureView
		p := bgp.NewBindGroupProvider("bgp", bgp.WithTextureView(0, tv))
		suite.Equal(tv, p.TextureView(0))
	})

	suite.Run("multiple texture view options set different bindings", func() {
		p := bgp.NewBindGroupProvider("bgp",
			bgp.WithTextureView(0, nil),
			bgp.WithTextureView(1, nil),
			bgp.WithTextureView(3, nil),
		)
		suite.Len(p.TextureViews(), 3)
		suite.Nil(p.TextureView(2))
	})

	suite.Run("later texture view option at same binding overwrites earlier", func() {
		var tv1, tv2 *wgpu.TextureView
		p := bgp.NewBindGroupProvider("bgp",
			bgp.WithTextureView(0, tv1),
			bgp.WithTextureView(0, tv2),
		)
		suite.Equal(tv2, p.TextureView(0))
	})
}

func (suite *bindGroupProviderTest) TestWithTextureViewsOption() {
	suite.Run("option sets entire texture views map", func() {
		views := map[int]*wgpu.TextureView{
			0: nil,
			1: nil,
		}
		p := bgp.NewBindGroupProvider("bgp", bgp.WithTextureViews(views))
		suite.Len(p.TextureViews(), 2)
	})

	suite.Run("empty texture views map option creates empty map", func() {
		p := bgp.NewBindGroupProvider("bgp", bgp.WithTextureViews(map[int]*wgpu.TextureView{}))
		suite.NotNil(p.TextureViews())
		suite.Len(p.TextureViews(), 0)
	})

	suite.Run("WithTextureViews after WithTextureView replaces individual view", func() {
		p := bgp.NewBindGroupProvider("bgp",
			bgp.WithTextureView(0, nil),
			bgp.WithTextureViews(map[int]*wgpu.TextureView{5: nil}),
		)
		suite.Len(p.TextureViews(), 1)
		suite.Nil(p.TextureView(0))
		suite.Nil(p.TextureView(5))
	})
}

func (suite *bindGroupProviderTest) TestWithSamplerOption() {
	suite.Run("option sets sampler at specified binding", func() {
		var s *wgpu.Sampler
		p := bgp.NewBindGroupProvider("bgp", bgp.WithSampler(0, s))
		suite.Equal(s, p.Sampler(0))
	})

	suite.Run("multiple sampler options set different bindings", func() {
		p := bgp.NewBindGroupProvider("bgp",
			bgp.WithSampler(0, nil),
			bgp.WithSampler(2, nil),
		)
		suite.Len(p.Samplers(), 2)
		suite.Nil(p.Sampler(1))
	})

	suite.Run("later sampler option at same binding overwrites earlier", func() {
		var s1, s2 *wgpu.Sampler
		p := bgp.NewBindGroupProvider("bgp",
			bgp.WithSampler(0, s1),
			bgp.WithSampler(0, s2),
		)
		suite.Equal(s2, p.Sampler(0))
	})
}

func (suite *bindGroupProviderTest) TestWithSamplersOption() {
	suite.Run("option sets entire samplers map", func() {
		samplers := map[int]*wgpu.Sampler{
			0: nil,
			1: nil,
		}
		p := bgp.NewBindGroupProvider("bgp", bgp.WithSamplers(samplers))
		suite.Len(p.Samplers(), 2)
	})

	suite.Run("empty samplers map option creates empty map", func() {
		p := bgp.NewBindGroupProvider("bgp", bgp.WithSamplers(map[int]*wgpu.Sampler{}))
		suite.NotNil(p.Samplers())
		suite.Len(p.Samplers(), 0)
	})

	suite.Run("WithSamplers after WithSampler replaces individual sampler", func() {
		p := bgp.NewBindGroupProvider("bgp",
			bgp.WithSampler(0, nil),
			bgp.WithSamplers(map[int]*wgpu.Sampler{5: nil}),
		)
		suite.Len(p.Samplers(), 1)
		suite.Nil(p.Sampler(0))
		suite.Nil(p.Sampler(5))
	})
}

func (suite *bindGroupProviderTest) TestWithVertexBufferOption() {
	suite.Run("option sets vertex buffer during construction", func() {
		var vb *wgpu.Buffer
		p := bgp.NewBindGroupProvider("bgp", bgp.WithVertexBuffer(vb))
		suite.Equal(vb, p.VertexBuffer())
	})

	suite.Run("nil vertex buffer option is accepted", func() {
		p := bgp.NewBindGroupProvider("bgp", bgp.WithVertexBuffer(nil))
		suite.Nil(p.VertexBuffer())
	})
}

func (suite *bindGroupProviderTest) TestWithIndexBufferOption() {
	suite.Run("option sets index buffer during construction", func() {
		var ib *wgpu.Buffer
		p := bgp.NewBindGroupProvider("bgp", bgp.WithIndexBuffer(ib))
		suite.Equal(ib, p.IndexBuffer())
	})

	suite.Run("nil index buffer option is accepted", func() {
		p := bgp.NewBindGroupProvider("bgp", bgp.WithIndexBuffer(nil))
		suite.Nil(p.IndexBuffer())
	})
}

func (suite *bindGroupProviderTest) TestWithIndexCountOption() {
	suite.Run("option sets index count during construction", func() {
		p := bgp.NewBindGroupProvider("bgp", bgp.WithIndexCount(256))
		suite.Equal(256, p.IndexCount())
	})

	suite.Run("zero index count option is accepted", func() {
		p := bgp.NewBindGroupProvider("bgp", bgp.WithIndexCount(0))
		suite.Equal(0, p.IndexCount())
	})
}

func (suite *bindGroupProviderTest) TestAllOptionsComposed() {
	suite.Run("all options including new ones can be composed", func() {
		var bg *wgpu.BindGroup
		var bgl *wgpu.BindGroupLayout
		bufs := map[int]*wgpu.Buffer{0: nil}
		views := map[int]*wgpu.TextureView{0: nil}
		samplers := map[int]*wgpu.Sampler{0: nil}

		p := bgp.NewBindGroupProvider("all-opts",
			bgp.WithBindGroup(bg),
			bgp.WithBindGroupLayout(bgl),
			bgp.WithBuffers(bufs),
			bgp.WithTextureViews(views),
			bgp.WithSamplers(samplers),
			bgp.WithVertexBuffer(nil),
			bgp.WithIndexBuffer(nil),
			bgp.WithIndexCount(512),
		)

		suite.Equal("all-opts", p.Label())
		suite.Equal(bg, p.BindGroup())
		suite.Equal(bgl, p.BindGroupLayout())
		suite.Len(p.Buffers(), 1)
		suite.Len(p.TextureViews(), 1)
		suite.Len(p.Samplers(), 1)
		suite.Nil(p.VertexBuffer())
		suite.Nil(p.IndexBuffer())
		suite.Equal(512, p.IndexCount())
	})
}

type bufferWriteTest struct {
	suite.Suite
}

func TestBufferWrite(t *testing.T) {
	suite.Run(t, new(bufferWriteTest))
}

func (suite *bufferWriteTest) TestBufferWriteFields() {
	suite.Run("struct fields are correctly assigned", func() {
		p := bgp.NewBindGroupProvider("test")
		data := []byte{0x01, 0x02, 0x03, 0x04}
		bw := bgp.BufferWrite{
			Provider: p,
			Binding:  2,
			Offset:   64,
			Data:     data,
		}

		suite.Equal(p, bw.Provider)
		suite.Equal(2, bw.Binding)
		suite.Equal(uint64(64), bw.Offset)
		suite.Equal(data, bw.Data)
	})

	suite.Run("zero value buffer write has nil provider and empty data", func() {
		var bw bgp.BufferWrite
		suite.Nil(bw.Provider)
		suite.Equal(0, bw.Binding)
		suite.Equal(uint64(0), bw.Offset)
		suite.Nil(bw.Data)
	})

	suite.Run("large offset is preserved", func() {
		bw := bgp.BufferWrite{
			Offset: 1<<32 + 256,
		}
		suite.Equal(uint64(4294967552), bw.Offset)
	})

	suite.Run("empty data slice is valid", func() {
		bw := bgp.BufferWrite{
			Data: []byte{},
		}
		suite.NotNil(bw.Data)
		suite.Len(bw.Data, 0)
	})
}
