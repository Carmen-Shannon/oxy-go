package bind_group_provider_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

func TestRunBindGroupProviderTests(t *testing.T) {
	suite.Run(t, new(bindGroupProviderTest))
}

type bindGroupProviderTest struct {
	suite.Suite
	provider bind_group_provider.BindGroupProvider
}

func (suite *bindGroupProviderTest) SetupSubTest() {
	suite.provider = bind_group_provider.NewBindGroupProvider("test_label")
}

// --- Constructor ---

func (suite *bindGroupProviderTest) TestNewBindGroupProvider() {
	suite.Run("should set label correctly", func() {
		p := bind_group_provider.NewBindGroupProvider("my_label")
		suite.Equal("my_label", p.Label())
	})
	suite.Run("should initialize buffers map as non-nil", func() {
		suite.NotNil(suite.provider.Buffers())
	})
	suite.Run("should initialize textureViews map as non-nil", func() {
		suite.NotNil(suite.provider.TextureViews())
	})
	suite.Run("should initialize samplers map as non-nil", func() {
		suite.NotNil(suite.provider.Samplers())
	})
	suite.Run("should have nil BindGroup by default", func() {
		suite.Nil(suite.provider.BindGroup())
	})
	suite.Run("should have nil BindGroupLayout by default", func() {
		suite.Nil(suite.provider.BindGroupLayout())
	})
	suite.Run("should have nil VertexBuffer by default", func() {
		suite.Nil(suite.provider.VertexBuffer())
	})
	suite.Run("should have nil IndexBuffer by default", func() {
		suite.Nil(suite.provider.IndexBuffer())
	})
	suite.Run("should have zero IndexCount by default", func() {
		suite.Equal(0, suite.provider.IndexCount())
	})
}

// --- Label ---

func (suite *bindGroupProviderTest) TestLabel() {
	suite.Run("should return the label passed to NewBindGroupProvider", func() {
		suite.Equal("test_label", suite.provider.Label())
	})
}

// --- BindGroup get / set ---

func (suite *bindGroupProviderTest) TestBindGroupGetSet() {
	suite.Run("should return nil after SetBindGroup(nil)", func() {
		suite.provider.SetBindGroup(nil)
		suite.Nil(suite.provider.BindGroup())
	})
}

// --- BindGroupLayout get / set ---

func (suite *bindGroupProviderTest) TestBindGroupLayoutGetSet() {
	suite.Run("should return nil after SetBindGroupLayout(nil)", func() {
		suite.provider.SetBindGroupLayout(nil)
		suite.Nil(suite.provider.BindGroupLayout())
	})
}

// --- Buffer get / set ---

func (suite *bindGroupProviderTest) TestBufferGetSet() {
	suite.Run("should return nil for a nil buffer set at binding 0", func() {
		suite.provider.SetBuffer(0, nil)
		suite.Nil(suite.provider.Buffer(0))
	})
	suite.Run("should return nil for a nil buffer set at binding 1", func() {
		suite.provider.SetBuffer(1, nil)
		suite.Nil(suite.provider.Buffer(1))
	})
	suite.Run("should return nil for a missing binding key", func() {
		suite.Nil(suite.provider.Buffer(99))
	})
}

// --- Buffers get / set ---

func (suite *bindGroupProviderTest) TestBuffersGetSet() {
	suite.Run("should return a non-nil map after construction", func() {
		suite.NotNil(suite.provider.Buffers())
	})
	suite.Run("should return nil after SetBuffers(nil)", func() {
		suite.provider.SetBuffers(nil)
		suite.Nil(suite.provider.Buffers())
	})
}

// --- SetBuffer nil-map guard ---

func (suite *bindGroupProviderTest) TestSetBufferNilMapGuard() {
	suite.Run("should re-initialize nil buffers map on SetBuffer call", func() {
		suite.provider.SetBuffers(nil)
		suite.provider.SetBuffer(0, nil)
		suite.NotNil(suite.provider.Buffers())
		suite.Nil(suite.provider.Buffer(0))
	})
}

// --- TextureView get / set ---

func (suite *bindGroupProviderTest) TestTextureViewGetSet() {
	suite.Run("should return nil for a nil texture view set at binding 0", func() {
		suite.provider.SetTextureView(0, nil)
		suite.Nil(suite.provider.TextureView(0))
	})
	suite.Run("should return nil for a missing binding key", func() {
		suite.Nil(suite.provider.TextureView(99))
	})
}

// --- TextureViews get / set ---

func (suite *bindGroupProviderTest) TestTextureViewsGetSet() {
	suite.Run("should return a non-nil map after construction", func() {
		suite.NotNil(suite.provider.TextureViews())
	})
	suite.Run("should return nil after SetTextureViews(nil)", func() {
		suite.provider.SetTextureViews(nil)
		suite.Nil(suite.provider.TextureViews())
	})
}

// --- SetTextureView nil-map guard ---

func (suite *bindGroupProviderTest) TestSetTextureViewNilMapGuard() {
	suite.Run("should re-initialize nil textureViews map on SetTextureView call", func() {
		suite.provider.SetTextureViews(nil)
		suite.provider.SetTextureView(0, nil)
		suite.NotNil(suite.provider.TextureViews())
		suite.Nil(suite.provider.TextureView(0))
	})
}

// --- Sampler get / set ---

func (suite *bindGroupProviderTest) TestSamplerGetSet() {
	suite.Run("should return nil for a nil sampler set at binding 0", func() {
		suite.provider.SetSampler(0, nil)
		suite.Nil(suite.provider.Sampler(0))
	})
	suite.Run("should return nil for a missing binding key", func() {
		suite.Nil(suite.provider.Sampler(99))
	})
}

// --- Samplers get / set ---

func (suite *bindGroupProviderTest) TestSamplersGetSet() {
	suite.Run("should return a non-nil map after construction", func() {
		suite.NotNil(suite.provider.Samplers())
	})
	suite.Run("should return nil after SetSamplers(nil)", func() {
		suite.provider.SetSamplers(nil)
		suite.Nil(suite.provider.Samplers())
	})
}

// --- SetSampler nil-map guard ---

func (suite *bindGroupProviderTest) TestSetSamplerNilMapGuard() {
	suite.Run("should re-initialize nil samplers map on SetSampler call", func() {
		suite.provider.SetSamplers(nil)
		suite.provider.SetSampler(0, nil)
		suite.NotNil(suite.provider.Samplers())
		suite.Nil(suite.provider.Sampler(0))
	})
}

// --- VertexBuffer get / set ---

func (suite *bindGroupProviderTest) TestVertexBufferGetSet() {
	suite.Run("should return nil after SetVertexBuffer(nil)", func() {
		suite.provider.SetVertexBuffer(nil)
		suite.Nil(suite.provider.VertexBuffer())
	})
}

// --- IndexBuffer get / set ---

func (suite *bindGroupProviderTest) TestIndexBufferGetSet() {
	suite.Run("should return nil after SetIndexBuffer(nil)", func() {
		suite.provider.SetIndexBuffer(nil)
		suite.Nil(suite.provider.IndexBuffer())
	})
}

// --- IndexCount get / set ---

func (suite *bindGroupProviderTest) TestIndexCountGetSet() {
	suite.Run("should return 42 after SetIndexCount(42)", func() {
		suite.provider.SetIndexCount(42)
		suite.Equal(42, suite.provider.IndexCount())
	})
	suite.Run("should return 0 after SetIndexCount(0)", func() {
		suite.provider.SetIndexCount(0)
		suite.Equal(0, suite.provider.IndexCount())
	})
}

// --- Release ---

func (suite *bindGroupProviderTest) TestRelease() {
	suite.Run("should not panic when all GPU fields are nil", func() {
		suite.NotPanics(func() {
			suite.provider.Release()
		})
	})
	suite.Run("nil texture view key remains stable and release does not panic", func() {
		suite.provider.SetTextureView(7, nil)

		suite.Contains(suite.provider.TextureViews(), 7)
		suite.Nil(suite.provider.TextureView(7))

		suite.NotPanics(func() {
			suite.provider.Release()
		})

		suite.Contains(suite.provider.TextureViews(), 7)
		suite.Nil(suite.provider.TextureView(7))
	})
	suite.Run("nil sampler key remains stable and release does not panic", func() {
		suite.provider.SetSampler(11, nil)

		suite.Contains(suite.provider.Samplers(), 11)
		suite.Nil(suite.provider.Sampler(11))

		suite.NotPanics(func() {
			suite.provider.Release()
		})

		suite.Contains(suite.provider.Samplers(), 11)
		suite.Nil(suite.provider.Sampler(11))
	})
	suite.Run("nil buffer entries remain stable across both frame slots and release does not panic", func() {
		suite.provider.SetSlot(0)
		suite.provider.SetBuffer(3, nil)
		suite.provider.SetSlot(1)
		suite.provider.SetBuffer(5, nil)

		suite.provider.SetSlot(0)
		suite.Contains(suite.provider.Buffers(), 3)
		suite.Nil(suite.provider.Buffer(3))
		suite.provider.SetSlot(1)
		suite.Contains(suite.provider.Buffers(), 5)
		suite.Nil(suite.provider.Buffer(5))

		suite.NotPanics(func() {
			suite.provider.Release()
		})

		suite.provider.SetSlot(0)
		suite.Contains(suite.provider.Buffers(), 3)
		suite.Nil(suite.provider.Buffer(3))
		suite.provider.SetSlot(1)
		suite.Contains(suite.provider.Buffers(), 5)
		suite.Nil(suite.provider.Buffer(5))
	})
}

// --- WithBindGroup ---

func (suite *bindGroupProviderTest) TestWithBindGroup() {
	suite.Run("should set bind group to nil when WithBindGroup(nil) is used", func() {
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithBindGroup(nil))
		suite.Nil(p.BindGroup())
	})
}

// --- WithBindGroupLayout ---

func (suite *bindGroupProviderTest) TestWithBindGroupLayout() {
	suite.Run("should set bind group layout to nil when WithBindGroupLayout(nil) is used", func() {
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithBindGroupLayout(nil))
		suite.Nil(p.BindGroupLayout())
	})
}

// --- WithBuffer ---

func (suite *bindGroupProviderTest) TestWithBuffer() {
	suite.Run("should set buffer to nil for binding 0 when WithBuffer(0, nil) is used", func() {
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithBuffer(0, nil))
		suite.Nil(p.Buffer(0))
	})
}

// --- WithBuffers ---

func (suite *bindGroupProviderTest) TestWithBuffers() {
	suite.Run("should set buffers map when WithBuffers is used", func() {
		m := map[int]*wgpu.Buffer{}
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithBuffers(m))
		suite.NotNil(p.Buffers())
	})
}

// --- WithTextureView ---

func (suite *bindGroupProviderTest) TestWithTextureView() {
	suite.Run("should set texture view to nil for binding 0 when WithTextureView(0, nil) is used", func() {
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithTextureView(0, nil))
		suite.Nil(p.TextureView(0))
	})
}

// --- WithTextureViews ---

func (suite *bindGroupProviderTest) TestWithTextureViews() {
	suite.Run("should set texture views map when WithTextureViews is used", func() {
		m := map[int]*wgpu.TextureView{}
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithTextureViews(m))
		suite.NotNil(p.TextureViews())
	})
}

// --- WithSampler ---

func (suite *bindGroupProviderTest) TestWithSampler() {
	suite.Run("should set sampler to nil for binding 0 when WithSampler(0, nil) is used", func() {
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithSampler(0, nil))
		suite.Nil(p.Sampler(0))
	})
}

// --- WithSamplers ---

func (suite *bindGroupProviderTest) TestWithSamplers() {
	suite.Run("should set samplers map when WithSamplers is used", func() {
		m := map[int]*wgpu.Sampler{}
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithSamplers(m))
		suite.NotNil(p.Samplers())
	})
}

// --- WithVertexBuffer ---

func (suite *bindGroupProviderTest) TestWithVertexBuffer() {
	suite.Run("should set vertex buffer to nil when WithVertexBuffer(nil) is used", func() {
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithVertexBuffer(nil))
		suite.Nil(p.VertexBuffer())
	})
}

// --- WithIndexBuffer ---

func (suite *bindGroupProviderTest) TestWithIndexBuffer() {
	suite.Run("should set index buffer to nil when WithIndexBuffer(nil) is used", func() {
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithIndexBuffer(nil))
		suite.Nil(p.IndexBuffer())
	})
}

// --- WithIndexCount ---

func (suite *bindGroupProviderTest) TestWithIndexCount() {
	suite.Run("should set index count to 10 when WithIndexCount(10) is used", func() {
		p := bind_group_provider.NewBindGroupProvider("x", bind_group_provider.WithIndexCount(10))
		suite.Equal(10, p.IndexCount())
	})
}

// --- BufferWrite ---

func (suite *bindGroupProviderTest) TestBufferWrite() {
	suite.Run("should store all fields correctly", func() {
		p := bind_group_provider.NewBindGroupProvider("bw_label")
		data := []byte{1, 2, 3, 4}
		bw := bind_group_provider.BufferWrite{
			Provider: p,
			Binding:  3,
			Offset:   16,
			Data:     data,
		}
		suite.Equal(p, bw.Provider)
		suite.Equal(3, bw.Binding)
		suite.Equal(uint64(16), bw.Offset)
		suite.Equal(data, bw.Data)
	})
}
