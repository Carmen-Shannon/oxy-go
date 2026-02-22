package renderer_test

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type rendererTest struct {
	suite.Suite
}

func TestRenderer(t *testing.T) {
	suite.Run(t, new(rendererTest))
}

func (suite *rendererTest) TestBackendTypeConstants() {
	suite.Run("BackendTypeWGPU is zero value", func() {
		suite.Equal(renderer.RendererBackendType(0), renderer.BackendTypeWGPU)
	})
}

func (suite *rendererTest) TestPresentModeConstants() {
	suite.Run("VSync is 0 and Uncapped is 1", func() {
		suite.Equal(renderer.PresentMode(0), renderer.PresentModeVSync)
		suite.Equal(renderer.PresentMode(1), renderer.PresentModeUncapped)
	})
}

func (suite *rendererTest) TestMSAASampleCountConstants() {
	suite.Run("MSAAOff is 1", func() {
		suite.Equal(renderer.MSAASampleCount(1), renderer.MSAAOff)
	})

	suite.Run("MSAA4x is 4", func() {
		suite.Equal(renderer.MSAASampleCount(4), renderer.MSAA4x)
	})

	suite.Run("MSAA8x is 8", func() {
		suite.Equal(renderer.MSAASampleCount(8), renderer.MSAA8x)
	})

	suite.Run("MSAA16x is 16", func() {
		suite.Equal(renderer.MSAASampleCount(16), renderer.MSAA16x)
	})
}

func (suite *rendererTest) TestNewRenderer() {
	suite.Run("creates renderer with default options", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotNil(r)
	})

	suite.Run("applies WithPipeline builder option", func() {
		p := pipeline.NewPipeline("test-pipe", pipeline.PipelineTypeRender)
		w, r := newTestRenderer(renderer.WithPipeline("test-pipe", p))
		defer w.Close()
		suite.NotNil(r.Pipeline("test-pipe"))
	})

	suite.Run("applies WithPipelines builder option", func() {
		p1 := pipeline.NewPipeline("pipe-a", pipeline.PipelineTypeRender)
		p2 := pipeline.NewPipeline("pipe-b", pipeline.PipelineTypeCompute)
		cache := map[string]pipeline.Pipeline{
			"pipe-a": p1,
			"pipe-b": p2,
		}
		w, r := newTestRenderer(renderer.WithPipelines(cache))
		defer w.Close()
		suite.Len(r.Pipelines(), 2)
		suite.NotNil(r.Pipeline("pipe-a"))
		suite.NotNil(r.Pipeline("pipe-b"))
	})

	suite.Run("applies WithPresentMode VSync option", func() {
		w, r := newTestRenderer(renderer.WithPresentMode(renderer.PresentModeVSync))
		defer w.Close()
		suite.NotNil(r)
	})

	suite.Run("applies WithPresentMode Uncapped option", func() {
		w, r := newTestRenderer(renderer.WithPresentMode(renderer.PresentModeUncapped))
		defer w.Close()
		suite.NotNil(r)
	})

	suite.Run("applies WithMSAA off option", func() {
		w, r := newTestRenderer(renderer.WithMSAA(renderer.MSAAOff))
		defer w.Close()
		suite.NotNil(r)
	})

	suite.Run("applies WithMSAA 4x option", func() {
		w, r := newTestRenderer(renderer.WithMSAA(renderer.MSAA4x))
		defer w.Close()
		suite.NotNil(r)
	})

	suite.Run("applies WithForceSoftwareRenderer false option", func() {
		w, r := newTestRenderer(renderer.WithForceSoftwareRenderer(false))
		defer w.Close()
		suite.NotNil(r)
	})

	suite.Run("applies multiple builder options simultaneously", func() {
		p := pipeline.NewPipeline("combo-pipe", pipeline.PipelineTypeRender)
		w, r := newTestRenderer(
			renderer.WithPipeline("combo-pipe", p),
			renderer.WithPresentMode(renderer.PresentModeVSync),
			renderer.WithMSAA(renderer.MSAA4x),
			renderer.WithForceSoftwareRenderer(false),
		)
		defer w.Close()
		suite.NotNil(r)
		suite.NotNil(r.Pipeline("combo-pipe"))
	})
}

func (suite *rendererTest) TestPipeline() {
	suite.Run("returns nil when key does not exist", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.Nil(r.Pipeline("nonexistent"))
	})

	suite.Run("returns pipeline when key exists", func() {
		p := pipeline.NewPipeline("existing", pipeline.PipelineTypeRender)
		w, r := newTestRenderer(renderer.WithPipeline("existing", p))
		defer w.Close()
		suite.NotNil(r.Pipeline("existing"))
		suite.Equal("existing", r.Pipeline("existing").PipelineKey())
	})
}

func (suite *rendererTest) TestPipelines() {
	suite.Run("returns empty map for new renderer", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.Empty(r.Pipelines())
	})

	suite.Run("returns populated map after SetPipeline", func() {
		w, r := newTestRenderer()
		defer w.Close()
		p := pipeline.NewPipeline("added", pipeline.PipelineTypeRender)
		r.SetPipeline("added", p)
		suite.Len(r.Pipelines(), 1)
	})
}

func (suite *rendererTest) TestSetPipeline() {
	suite.Run("stores pipeline retrievable by key", func() {
		w, r := newTestRenderer()
		defer w.Close()
		p := pipeline.NewPipeline("set-key", pipeline.PipelineTypeRender)
		r.SetPipeline("set-key", p)
		suite.NotNil(r.Pipeline("set-key"))
		suite.Equal("set-key", r.Pipeline("set-key").PipelineKey())
	})

	suite.Run("overwrites existing pipeline with same key", func() {
		w, r := newTestRenderer()
		defer w.Close()
		p1 := pipeline.NewPipeline("dup-key", pipeline.PipelineTypeRender)
		p2 := pipeline.NewPipeline("dup-key", pipeline.PipelineTypeCompute)
		r.SetPipeline("dup-key", p1)
		suite.Equal(pipeline.PipelineTypeRender, r.Pipeline("dup-key").Type())
		r.SetPipeline("dup-key", p2)
		suite.Equal(pipeline.PipelineTypeCompute, r.Pipeline("dup-key").Type())
	})
}

func (suite *rendererTest) TestSetPipelines() {
	suite.Run("replaces entire pipeline cache", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p1 := pipeline.NewPipeline("first", pipeline.PipelineTypeRender)
		r.SetPipeline("first", p1)
		suite.Len(r.Pipelines(), 1)

		p2 := pipeline.NewPipeline("second", pipeline.PipelineTypeCompute)
		p3 := pipeline.NewPipeline("third", pipeline.PipelineTypeRender)
		r.SetPipelines(map[string]pipeline.Pipeline{
			"second": p2,
			"third":  p3,
		})
		suite.Nil(r.Pipeline("first"))
		suite.NotNil(r.Pipeline("second"))
		suite.NotNil(r.Pipeline("third"))
		suite.Len(r.Pipelines(), 2)
	})
}

func (suite *rendererTest) TestRegisterPipelines() {
	suite.Run("skips pipeline with key already in cache", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("preloaded", pipeline.PipelineTypeRender)
		r.SetPipeline("preloaded", p)

		err := r.RegisterPipelines(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("preloaded"))
	})

	suite.Run("registers a real render pipeline with vertex and fragment shaders", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("test-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("test-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))

		p := pipeline.NewPipeline("render-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("render-pipe"))
		suite.NotNil(r.Pipeline("render-pipe").Pipeline())
	})

	suite.Run("registers a real compute pipeline", func() {
		w, r := newTestRenderer()
		defer w.Close()

		cs := shader.NewShader("test-compute", shader.ShaderTypeCompute, shaderPath("test_compute.wgsl"))
		p := pipeline.NewPipeline("compute-pipe", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(cs),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("compute-pipe"))
		suite.NotNil(r.Pipeline("compute-pipe").Pipeline())
	})

	suite.Run("registers both render and compute pipelines in one call", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("test-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("test-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		cs := shader.NewShader("test-compute", shader.ShaderTypeCompute, shaderPath("test_compute.wgsl"))

		rp := pipeline.NewPipeline("render-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)
		cp := pipeline.NewPipeline("compute-pipe", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(cs),
		)

		err := r.RegisterPipelines(rp, cp)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("render-pipe"))
		suite.NotNil(r.Pipeline("compute-pipe"))
	})

	suite.Run("returns error when vertex shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		fs := shader.NewShader("test-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		p := pipeline.NewPipeline("bad-pipe", pipeline.PipelineTypeRender,
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(p)
		suite.Error(err)
	})

	suite.Run("returns error when fragment shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("test-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		p := pipeline.NewPipeline("bad-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
		)

		err := r.RegisterPipelines(p)
		suite.Error(err)
	})

	suite.Run("returns error when compute shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("bad-compute", pipeline.PipelineTypeCompute)
		err := r.RegisterPipelines(p)
		suite.Error(err)
	})

	suite.Run("registers render pipeline with blend enabled", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("test-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("test-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))

		p := pipeline.NewPipeline("blend-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithBlendEnabled(true),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)
		suite.True(r.Pipeline("blend-pipe").BlendEnabled())
	})

	suite.Run("registers render pipeline with depth test disabled", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("test-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("test-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))

		p := pipeline.NewPipeline("nodepth-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
			pipeline.WithDepthTestEnabled(false),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)
		suite.False(r.Pipeline("nodepth-pipe").DepthTestEnabled())
	})
}

func (suite *rendererTest) TestRegisterShadowPipeline() {
	suite.Run("skips pipeline with key already in cache", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("shadow-preloaded", pipeline.PipelineTypeRender)
		r.SetPipeline("shadow-preloaded", p)

		err := r.RegisterShadowPipeline(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("shadow-preloaded"))
	})

	suite.Run("registers a real shadow pipeline with vertex shader", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("shadow-vert", shader.ShaderTypeVertex, shaderPath("test_shadow_vert.wgsl"))
		p := pipeline.NewPipeline("shadow-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithDepthBias(2, 1.5),
		)

		err := r.RegisterShadowPipeline(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("shadow-pipe"))
		suite.NotNil(r.Pipeline("shadow-pipe").Pipeline())
	})

	suite.Run("returns error when vertex shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("bad-shadow", pipeline.PipelineTypeRender)
		err := r.RegisterShadowPipeline(p)
		suite.Error(err)
	})
}

func (suite *rendererTest) TestResize() {
	suite.Run("does not panic on positive dimensions", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.Resize(128, 128)
		})
	})
}

func (suite *rendererTest) TestSetPresentMode() {
	suite.Run("does not panic when setting VSync", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.SetPresentMode(renderer.PresentModeVSync)
		})
	})

	suite.Run("does not panic when setting Uncapped", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.SetPresentMode(renderer.PresentModeUncapped)
		})
	})
}

func (suite *rendererTest) TestInitMeshBuffers() {
	suite.Run("creates vertex and index buffers from raw bytes", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("mesh-test")
		vertexData, indexData := buildTriangleGeometry()

		err := r.InitMeshBuffers(provider, vertexData, indexData, 3)
		suite.NoError(err)
		suite.NotNil(provider.VertexBuffer())
		suite.NotNil(provider.IndexBuffer())
		suite.Equal(3, provider.IndexCount())
	})

	suite.Run("handles empty vertex data without error", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("empty-vert")
		_, indexData := buildTriangleGeometry()

		err := r.InitMeshBuffers(provider, nil, indexData, 3)
		suite.NoError(err)
		suite.Nil(provider.VertexBuffer())
		suite.NotNil(provider.IndexBuffer())
	})

	suite.Run("handles empty index data without error", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("empty-idx")
		vertexData, _ := buildTriangleGeometry()

		err := r.InitMeshBuffers(provider, vertexData, nil, 0)
		suite.NoError(err)
		suite.NotNil(provider.VertexBuffer())
		suite.Nil(provider.IndexBuffer())
	})
}

func (suite *rendererTest) TestInitBindGroup() {
	suite.Run("creates bind group with uniform buffer", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("uniform-test")
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Label: "test-layout",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageVertex,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeUniform,
						MinBindingSize: 80,
					},
				},
			},
		}

		err := r.InitBindGroup(provider, descriptor, nil, nil)
		suite.NoError(err)
		suite.NotNil(provider.BindGroup())
		suite.NotNil(provider.BindGroupLayout())
		suite.NotNil(provider.Buffer(0))
	})

	suite.Run("creates bind group with storage buffer", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("storage-test")
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Label: "storage-layout",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageVertex,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeReadOnlyStorage,
						MinBindingSize: 64,
					},
				},
			},
		}

		err := r.InitBindGroup(provider, descriptor, nil, nil)
		suite.NoError(err)
		suite.NotNil(provider.BindGroup())
		suite.NotNil(provider.Buffer(0))
	})

	suite.Run("creates bind group with buffer usage override", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("override-test")
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Label: "override-layout",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageCompute,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeStorage,
						MinBindingSize: 64,
					},
				},
			},
		}

		overrides := map[int]wgpu.BufferUsage{
			0: wgpu.BufferUsageIndirect,
		}

		err := r.InitBindGroup(provider, descriptor, overrides, nil)
		suite.NoError(err)
		suite.NotNil(provider.Buffer(0))
	})

	suite.Run("creates bind group with buffer size override", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("size-override-test")
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Label: "size-override-layout",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageVertex,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeUniform,
						MinBindingSize: 16,
					},
				},
			},
		}

		sizeOverrides := map[int]uint64{
			0: 256,
		}

		err := r.InitBindGroup(provider, descriptor, nil, sizeOverrides)
		suite.NoError(err)
		suite.NotNil(provider.Buffer(0))
	})

	suite.Run("returns nil error for empty descriptor entries", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("empty-desc")
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Label:   "empty-layout",
			Entries: []wgpu.BindGroupLayoutEntry{},
		}

		err := r.InitBindGroup(provider, descriptor, nil, nil)
		suite.NoError(err)
	})

	suite.Run("creates bind group with texture and sampler entries", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("tex-sampler-test")

		pixels := make([]byte, 4*4*4) // 4x4 RGBA
		for i := range pixels {
			pixels[i] = 0xFF
		}
		err := r.InitTextureView(provider, 0, common.TextureStagingData{
			Pixels: pixels,
			Width:  4,
			Height: 4,
		})
		suite.NoError(err)

		err = r.InitSampler(provider, 1, common.SamplerStagingData{})
		suite.NoError(err)

		descriptor := wgpu.BindGroupLayoutDescriptor{
			Label: "tex-sampler-layout",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageFragment,
					Texture: wgpu.TextureBindingLayout{
						SampleType:    wgpu.TextureSampleTypeFloat,
						ViewDimension: wgpu.TextureViewDimension2D,
					},
				},
				{
					Binding:    1,
					Visibility: wgpu.ShaderStageFragment,
					Sampler: wgpu.SamplerBindingLayout{
						Type: wgpu.SamplerBindingTypeFiltering,
					},
				},
			},
		}

		err = r.InitBindGroup(provider, descriptor, nil, nil)
		suite.NoError(err)
		suite.NotNil(provider.BindGroup())
	})
}

func (suite *rendererTest) TestInitTextureView() {
	suite.Run("creates texture view from RGBA staging data", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("tex-test")
		pixels := make([]byte, 8*8*4) // 8x8 RGBA
		for i := 0; i < len(pixels); i += 4 {
			pixels[i] = 0xFF   // R
			pixels[i+1] = 0x00 // G
			pixels[i+2] = 0x00 // B
			pixels[i+3] = 0xFF // A
		}

		err := r.InitTextureView(provider, 0, common.TextureStagingData{
			Pixels: pixels,
			Width:  8,
			Height: 8,
		})
		suite.NoError(err)
		suite.NotNil(provider.TextureView(0))
	})

	suite.Run("creates multiple texture views on different bindings", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("multi-tex")
		pixels := make([]byte, 2*2*4) // 2x2 RGBA

		err := r.InitTextureView(provider, 0, common.TextureStagingData{
			Pixels: pixels, Width: 2, Height: 2,
		})
		suite.NoError(err)

		err = r.InitTextureView(provider, 1, common.TextureStagingData{
			Pixels: pixels, Width: 2, Height: 2,
		})
		suite.NoError(err)

		suite.NotNil(provider.TextureView(0))
		suite.NotNil(provider.TextureView(1))
	})
}

func (suite *rendererTest) TestInitSampler() {
	suite.Run("creates sampler with default staging data", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("sampler-test")

		err := r.InitSampler(provider, 0, common.SamplerStagingData{})
		suite.NoError(err)
		suite.NotNil(provider.Sampler(0))
	})

	suite.Run("creates sampler with explicit configuration", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("sampler-config-test")

		err := r.InitSampler(provider, 0, common.SamplerStagingData{
			AddressModeU:  wgpu.AddressModeClampToEdge,
			AddressModeV:  wgpu.AddressModeClampToEdge,
			AddressModeW:  wgpu.AddressModeClampToEdge,
			MagFilter:     wgpu.FilterModeNearest,
			MinFilter:     wgpu.FilterModeNearest,
			MipmapFilter:  wgpu.MipmapFilterModeNearest,
			LodMinClamp:   0.0,
			LodMaxClamp:   16.0,
			MaxAnisotropy: 1,
		})
		suite.NoError(err)
		suite.NotNil(provider.Sampler(0))
	})
}

func (suite *rendererTest) TestWriteBuffers() {
	suite.Run("does not panic with empty writes", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.WriteBuffers([]bind_group_provider.BufferWrite{})
		})
	})

	suite.Run("does not panic with nil writes", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.WriteBuffers(nil)
		})
	})

	suite.Run("writes data to an initialized buffer", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("write-test")
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Label: "write-layout",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageVertex,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeUniform,
						MinBindingSize: 64,
					},
				},
			},
		}

		err := r.InitBindGroup(provider, descriptor, nil, nil)
		suite.NoError(err)

		data := make([]byte, 64)
		for i := 0; i < 4; i++ {
			binary.LittleEndian.PutUint32(data[i*20:], math.Float32bits(1.0))
		}

		suite.NotPanics(func() {
			r.WriteBuffers([]bind_group_provider.BufferWrite{
				{
					Provider: provider,
					Binding:  0,
					Offset:   0,
					Data:     data,
				},
			})
		})
	})

	suite.Run("skips write when buffer does not exist for binding", func() {
		w, r := newTestRenderer()
		defer w.Close()

		provider := bind_group_provider.NewBindGroupProvider("no-buf")
		suite.NotPanics(func() {
			r.WriteBuffers([]bind_group_provider.BufferWrite{
				{
					Provider: provider,
					Binding:  99,
					Offset:   0,
					Data:     []byte{1, 2, 3, 4},
				},
			})
		})
	})
}

func (suite *rendererTest) TestDrawCall() {
	suite.Run("returns error when pipeline key is not found", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.DrawCall("nonexistent", nil, 0, nil)
		suite.Error(err)
		suite.Contains(err.Error(), "nonexistent")
	})

	suite.Run("succeeds with registered pipeline and initialized resources", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("test-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("test-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		p := pipeline.NewPipeline("draw-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("draw-mesh")
		vertexData, indexData := buildTriangleGeometry()
		err = r.InitMeshBuffers(meshProvider, vertexData, indexData, 3)
		suite.NoError(err)

		cameraProvider := bind_group_provider.NewBindGroupProvider("draw-camera")
		err = r.InitBindGroup(cameraProvider, vs.BindGroupLayoutDescriptor(0), nil, nil)
		suite.NoError(err)

		instanceProvider := bind_group_provider.NewBindGroupProvider("draw-instance")
		err = r.InitBindGroup(instanceProvider, vs.BindGroupLayoutDescriptor(1), nil, map[int]uint64{0: 64})
		suite.NoError(err)

		err = r.BeginFrame()
		suite.NoError(err)

		err = r.DrawCall("draw-pipe", meshProvider, 1, []bind_group_provider.BindGroupProvider{
			cameraProvider,
			instanceProvider,
		})
		suite.NoError(err)

		r.EndFrame()
		r.Present()
	})
}

func (suite *rendererTest) TestDrawCallIndirect() {
	suite.Run("returns error when pipeline key is not found", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.DrawCallIndirect("nonexistent", nil, nil, nil)
		suite.Error(err)
		suite.Contains(err.Error(), "nonexistent")
	})

	suite.Run("succeeds with registered pipeline and indirect buffer", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("test-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("test-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		p := pipeline.NewPipeline("indirect-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("indirect-mesh")
		vertexData, indexData := buildTriangleGeometry()
		err = r.InitMeshBuffers(meshProvider, vertexData, indexData, 3)
		suite.NoError(err)

		cameraProvider := bind_group_provider.NewBindGroupProvider("indirect-camera")
		err = r.InitBindGroup(cameraProvider, vs.BindGroupLayoutDescriptor(0), nil, nil)
		suite.NoError(err)

		instanceProvider := bind_group_provider.NewBindGroupProvider("indirect-instance")
		err = r.InitBindGroup(instanceProvider, vs.BindGroupLayoutDescriptor(1), nil, map[int]uint64{0: 64})
		suite.NoError(err)

		// Create an indirect buffer with BufferUsageIndirect via a storage bind group
		indirectProvider := bind_group_provider.NewBindGroupProvider("indirect-buf")
		indirectDesc := wgpu.BindGroupLayoutDescriptor{
			Label: "indirect-layout",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageCompute,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeStorage,
						MinBindingSize: 20,
					},
				},
			},
		}
		err = r.InitBindGroup(indirectProvider, indirectDesc,
			map[int]wgpu.BufferUsage{0: wgpu.BufferUsageIndirect},
			nil,
		)
		suite.NoError(err)

		// Write DrawIndexedIndirect args: indexCount=3, instanceCount=1, firstIndex=0, baseVertex=0, firstInstance=0
		indirectArgs := buildIndirectArgs(3, 1, 0, 0, 0)
		r.WriteBuffers([]bind_group_provider.BufferWrite{
			{Provider: indirectProvider, Binding: 0, Offset: 0, Data: indirectArgs},
		})

		indirectBuffer := indirectProvider.Buffer(0)
		suite.NotNil(indirectBuffer)

		err = r.BeginFrame()
		suite.NoError(err)

		err = r.DrawCallIndirect("indirect-pipe", meshProvider, indirectBuffer, []bind_group_provider.BindGroupProvider{
			cameraProvider,
			instanceProvider,
		})
		suite.NoError(err)

		r.EndFrame()
		r.Present()
	})
}

func (suite *rendererTest) TestShadowDrawCall() {
	suite.Run("returns error when pipeline key is not found", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.ShadowDrawCall("nonexistent", nil, 0, nil)
		suite.Error(err)
		suite.Contains(err.Error(), "nonexistent")
	})
}

func (suite *rendererTest) TestShadowDrawCallIndirect() {
	suite.Run("returns error when pipeline key is not found", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.ShadowDrawCallIndirect("nonexistent", nil, nil, nil)
		suite.Error(err)
		suite.Contains(err.Error(), "nonexistent")
	})

	suite.Run("succeeds with registered shadow pipeline and indirect buffer", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("shadow-vert", shader.ShaderTypeVertex, shaderPath("test_shadow_vert.wgsl"))
		p := pipeline.NewPipeline("shadow-indirect-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithDepthBias(2, 1.5),
		)

		err := r.RegisterShadowPipeline(p)
		suite.NoError(err)

		view, _, err := r.CreateShadowDepthTexture(256, 256)
		suite.NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("shadow-indirect-mesh")
		vertexData, indexData := buildTriangleGeometry()
		err = r.InitMeshBuffers(meshProvider, vertexData, indexData, 3)
		suite.NoError(err)

		shadowUniformProvider := bind_group_provider.NewBindGroupProvider("shadow-indirect-uniform")
		err = r.InitBindGroup(shadowUniformProvider, vs.BindGroupLayoutDescriptor(0), nil, nil)
		suite.NoError(err)

		instanceProvider := bind_group_provider.NewBindGroupProvider("shadow-indirect-instance")
		err = r.InitBindGroup(instanceProvider, vs.BindGroupLayoutDescriptor(1), nil, map[int]uint64{0: 64})
		suite.NoError(err)

		// Create an indirect buffer with BufferUsageIndirect via a storage bind group
		indirectProvider := bind_group_provider.NewBindGroupProvider("shadow-indirect-buf")
		indirectDesc := wgpu.BindGroupLayoutDescriptor{
			Label: "shadow-indirect-layout",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageCompute,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeStorage,
						MinBindingSize: 20,
					},
				},
			},
		}
		err = r.InitBindGroup(indirectProvider, indirectDesc,
			map[int]wgpu.BufferUsage{0: wgpu.BufferUsageIndirect},
			nil,
		)
		suite.NoError(err)

		indirectArgs := buildIndirectArgs(3, 1, 0, 0, 0)
		r.WriteBuffers([]bind_group_provider.BufferWrite{
			{Provider: indirectProvider, Binding: 0, Offset: 0, Data: indirectArgs},
		})

		indirectBuffer := indirectProvider.Buffer(0)
		suite.NotNil(indirectBuffer)

		err = r.BeginShadowFrame()
		suite.NoError(err)

		r.BeginShadowPass(view)

		err = r.ShadowDrawCallIndirect("shadow-indirect-pipe", meshProvider, indirectBuffer, []bind_group_provider.BindGroupProvider{
			shadowUniformProvider,
			instanceProvider,
		})
		suite.NoError(err)

		r.EndShadowPass()
		r.EndShadowFrame()
	})
}

func (suite *rendererTest) TestDispatchCompute() {
	suite.Run("does not panic when pipeline key is not found", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.DispatchCompute("nonexistent", nil, [3]uint32{1, 1, 1})
		})
	})

	suite.Run("dispatches compute with registered pipeline and bind group", func() {
		w, r := newTestRenderer()
		defer w.Close()

		cs := shader.NewShader("test-compute", shader.ShaderTypeCompute, shaderPath("test_compute.wgsl"))
		p := pipeline.NewPipeline("dispatch-pipe", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(cs),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)

		computeProvider := bind_group_provider.NewBindGroupProvider("compute-bg")
		err = r.InitBindGroup(computeProvider, cs.BindGroupLayoutDescriptor(0), nil, map[int]uint64{
			0: 16,
			1: 256,
		})
		suite.NoError(err)

		err = r.BeginComputeFrame()
		suite.NoError(err)

		suite.NotPanics(func() {
			r.DispatchCompute("dispatch-pipe", computeProvider, [3]uint32{1, 1, 1})
		})

		suite.NotPanics(func() {
			r.EndComputeFrame()
		})
	})
}

func (suite *rendererTest) TestBeginFrameEndFramePresent() {
	suite.Run("full frame lifecycle completes without panic", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.BeginFrame()
		if err == nil {
			suite.NotPanics(func() {
				r.EndFrame()
				r.Present()
			})
		}
	})
}

func (suite *rendererTest) TestBeginComputeFrameEndComputeFrame() {
	suite.Run("compute lifecycle completes without error", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.BeginComputeFrame()
		suite.NoError(err)
		suite.NotPanics(func() {
			r.EndComputeFrame()
		})
	})
}

func (suite *rendererTest) TestBeginShadowFrameEndShadowFrame() {
	suite.Run("shadow lifecycle completes without error", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.BeginShadowFrame()
		suite.NoError(err)
		suite.NotPanics(func() {
			r.EndShadowFrame()
		})
	})
}

func (suite *rendererTest) TestCreateShadowDepthTexture() {
	suite.Run("returns non-nil texture view and texture", func() {
		w, r := newTestRenderer()
		defer w.Close()
		view, tex, err := r.CreateShadowDepthTexture(256, 256)
		suite.NoError(err)
		suite.NotNil(view)
		suite.NotNil(tex)
	})
}

func (suite *rendererTest) TestCreateComparisonSampler() {
	suite.Run("returns non-nil sampler", func() {
		w, r := newTestRenderer()
		defer w.Close()
		samp, err := r.CreateComparisonSampler()
		suite.NoError(err)
		suite.NotNil(samp)
	})
}

func (suite *rendererTest) TestEndComputeFrameWithoutBegin() {
	suite.Run("does not panic when called without BeginComputeFrame", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.EndComputeFrame()
		})
	})
}

func (suite *rendererTest) TestEndShadowFrameWithoutBegin() {
	suite.Run("does not panic when called without BeginShadowFrame", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.EndShadowFrame()
		})
	})
}

func (suite *rendererTest) TestEndShadowPassWithoutBegin() {
	suite.Run("does not panic when called without BeginShadowPass", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.EndShadowPass()
		})
	})
}

func (suite *rendererTest) TestPresentWithoutBeginFrame() {
	suite.Run("does not panic when no frame is active", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.Present()
		})
	})
}

func (suite *rendererTest) TestShadowPassLifecycle() {
	suite.Run("full shadow lifecycle with registered pipeline and draw call", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("shadow-vert", shader.ShaderTypeVertex, shaderPath("test_shadow_vert.wgsl"))
		p := pipeline.NewPipeline("shadow-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
		)

		err := r.RegisterShadowPipeline(p)
		suite.NoError(err)

		view, _, err := r.CreateShadowDepthTexture(256, 256)
		suite.NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("shadow-mesh")
		vertexData, indexData := buildTriangleGeometry()
		err = r.InitMeshBuffers(meshProvider, vertexData, indexData, 3)
		suite.NoError(err)

		shadowUniformProvider := bind_group_provider.NewBindGroupProvider("shadow-uniform")
		err = r.InitBindGroup(shadowUniformProvider, vs.BindGroupLayoutDescriptor(0), nil, nil)
		suite.NoError(err)

		instanceProvider := bind_group_provider.NewBindGroupProvider("shadow-instance")
		err = r.InitBindGroup(instanceProvider, vs.BindGroupLayoutDescriptor(1), nil, map[int]uint64{0: 64})
		suite.NoError(err)

		err = r.BeginShadowFrame()
		suite.NoError(err)

		r.BeginShadowPass(view)

		err = r.ShadowDrawCall("shadow-pipe", meshProvider, 1, []bind_group_provider.BindGroupProvider{
			shadowUniformProvider,
			instanceProvider,
		})
		suite.NoError(err)

		r.EndShadowPass()
		r.EndShadowFrame()
	})
}

func (suite *rendererTest) TestFullRenderFrameLifecycle() {
	suite.Run("register pipelines, init resources, render frame, and present", func() {
		w, r := newTestRenderer(renderer.WithMSAA(renderer.MSAAOff))
		defer w.Close()

		vs := shader.NewShader("lc-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("lc-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		rp := pipeline.NewPipeline("lifecycle-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(rp)
		suite.NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("lc-mesh")
		vertexData, indexData := buildTriangleGeometry()
		err = r.InitMeshBuffers(meshProvider, vertexData, indexData, 3)
		suite.NoError(err)

		cameraProvider := bind_group_provider.NewBindGroupProvider("lc-camera")
		err = r.InitBindGroup(cameraProvider, vs.BindGroupLayoutDescriptor(0), nil, nil)
		suite.NoError(err)

		instanceProvider := bind_group_provider.NewBindGroupProvider("lc-instance")
		err = r.InitBindGroup(instanceProvider, vs.BindGroupLayoutDescriptor(1), nil, map[int]uint64{0: 64})
		suite.NoError(err)

		cameraData := make([]byte, 80)
		identity := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		for i, v := range identity {
			binary.LittleEndian.PutUint32(cameraData[i*4:], math.Float32bits(v))
		}
		r.WriteBuffers([]bind_group_provider.BufferWrite{
			{Provider: cameraProvider, Binding: 0, Offset: 0, Data: cameraData},
		})

		instanceData := make([]byte, 64)
		for i, v := range identity {
			binary.LittleEndian.PutUint32(instanceData[i*4:], math.Float32bits(v))
		}
		r.WriteBuffers([]bind_group_provider.BufferWrite{
			{Provider: instanceProvider, Binding: 0, Offset: 0, Data: instanceData},
		})

		err = r.BeginFrame()
		suite.NoError(err)

		err = r.DrawCall("lifecycle-pipe", meshProvider, 1, []bind_group_provider.BindGroupProvider{
			cameraProvider,
			instanceProvider,
		})
		suite.NoError(err)

		r.EndFrame()
		r.Present()
	})
}

func (suite *rendererTest) TestFullComputeLifecycle() {
	suite.Run("register compute pipeline, init bind group, dispatch, and submit", func() {
		w, r := newTestRenderer()
		defer w.Close()

		cs := shader.NewShader("lc-compute", shader.ShaderTypeCompute, shaderPath("test_compute.wgsl"))
		cp := pipeline.NewPipeline("lc-compute-pipe", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(cs),
		)

		err := r.RegisterPipelines(cp)
		suite.NoError(err)

		computeProvider := bind_group_provider.NewBindGroupProvider("lc-compute-bg")
		err = r.InitBindGroup(computeProvider, cs.BindGroupLayoutDescriptor(0), nil, map[int]uint64{
			0: 16,
			1: 256,
		})
		suite.NoError(err)

		paramsData := make([]byte, 4)
		binary.LittleEndian.PutUint32(paramsData, 4)
		r.WriteBuffers([]bind_group_provider.BufferWrite{
			{Provider: computeProvider, Binding: 0, Offset: 0, Data: paramsData},
		})

		err = r.BeginComputeFrame()
		suite.NoError(err)

		r.DispatchCompute("lc-compute-pipe", computeProvider, [3]uint32{1, 1, 1})

		r.EndComputeFrame()
	})
}

func (suite *rendererTest) TestSetDelegate() {
	suite.Run("accepts a delegate without panic", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.SetDelegate(r)
		})
	})
}

// newTestWindow creates a small GLFW window for renderer tests.
func newTestWindow() window.Window {
	return window.NewWindow(
		window.WithWidth(64),
		window.WithHeight(64),
		window.WithTitle("renderer-test"),
	)
}

// newTestRenderer creates a small window and renderer pair for testing.
func newTestRenderer(opts ...renderer.RendererBuilderOption) (window.Window, renderer.Renderer) {
	w := newTestWindow()
	r := renderer.NewRenderer(renderer.BackendTypeWGPU, w, opts...)
	return w, r
}

// shaderPath returns the absolute path to a test shader file in the test assets directory.
func shaderPath(name string) string {
	_, file, _, _ := runtime.Caller(0)
	dir := filepath.Dir(file)
	return filepath.Join(dir, "..", "assets", "shaders", name)
}

// buildIndirectArgs produces a 20-byte DrawIndexedIndirect argument buffer.
//
// Parameters:
//   - indexCount: number of indices to draw
//   - instanceCount: number of instances to draw
//   - firstIndex: offset into the index buffer
//   - baseVertex: value added to each index before lookup
//   - firstInstance: first instance ID
func buildIndirectArgs(indexCount, instanceCount, firstIndex uint32, baseVertex int32, firstInstance uint32) []byte {
	buf := make([]byte, 20)
	binary.LittleEndian.PutUint32(buf[0:], indexCount)
	binary.LittleEndian.PutUint32(buf[4:], instanceCount)
	binary.LittleEndian.PutUint32(buf[8:], firstIndex)
	binary.LittleEndian.PutUint32(buf[12:], uint32(baseVertex))
	binary.LittleEndian.PutUint32(buf[16:], firstInstance)
	return buf
}

// buildTriangleGeometry produces raw vertex and index byte slices for a minimal triangle.
// Each vertex has only a vec3<f32> position (12 bytes).
func buildTriangleGeometry() (vertexData []byte, indexData []byte) {
	positions := [9]float32{
		0.0, 0.5, 0.0,
		-0.5, -0.5, 0.0,
		0.5, -0.5, 0.0,
	}
	vertexData = make([]byte, len(positions)*4)
	for i, v := range positions {
		binary.LittleEndian.PutUint32(vertexData[i*4:], math.Float32bits(v))
	}

	indices := [3]uint32{0, 1, 2}
	indexData = make([]byte, len(indices)*4)
	for i, idx := range indices {
		binary.LittleEndian.PutUint32(indexData[i*4:], idx)
	}

	return vertexData, indexData
}
