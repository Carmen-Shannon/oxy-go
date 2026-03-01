package renderer_test

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"math"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
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

	suite.Run("succeeds with registered VSM shadow pipeline and indirect buffer", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("vsm-shadow-vert", shader.ShaderTypeVertex, shaderPath("test_vsm_shadow_vert.wgsl"))
		fs := shader.NewShader("vsm-shadow-frag", shader.ShaderTypeFragment, shaderPath("test_vsm_shadow_frag.wgsl"))
		p := pipeline.NewPipeline("shadow-indirect-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterVSMShadowPipeline(p)
		suite.NoError(err)

		vsmView, _, _, _, depthView, _, err := r.CreateVSMTextures(256, 256)
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

		r.BeginVSMShadowPass(vsmView, depthView)

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

func (suite *rendererTest) TestCreateVSMTextures() {
	suite.Run("returns non-nil textures and views", func() {
		w, r := newTestRenderer()
		defer w.Close()
		vsmView, vsmTex, scratchView, scratchTex, depthView, depthTex, err := r.CreateVSMTextures(256, 256)
		suite.NoError(err)
		suite.NotNil(vsmView)
		suite.NotNil(vsmTex)
		suite.NotNil(scratchView)
		suite.NotNil(scratchTex)
		suite.NotNil(depthView)
		suite.NotNil(depthTex)
	})
}

func (suite *rendererTest) TestCreateSATTextures() {
	suite.Run("returns non-nil textures and views", func() {
		w, r := newTestRenderer()
		defer w.Close()
		satAView, satATex, satBView, satBTex, err := r.CreateSATTextures(256, 256)
		suite.NoError(err)
		suite.NotNil(satAView)
		suite.NotNil(satATex)
		suite.NotNil(satBView)
		suite.NotNil(satBTex)
	})
}

func (suite *rendererTest) TestCreateLinearSampler() {
	suite.Run("returns non-nil sampler", func() {
		w, r := newTestRenderer()
		defer w.Close()
		samp, err := r.CreateLinearSampler()
		suite.NoError(err)
		suite.NotNil(samp)
	})
}

func (suite *rendererTest) TestRegisterVSMShadowPipeline() {
	suite.Run("skips pipeline with key already in cache", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("vsm-preloaded", pipeline.PipelineTypeRender)
		r.SetPipeline("vsm-preloaded", p)

		err := r.RegisterVSMShadowPipeline(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("vsm-preloaded"))
	})

	suite.Run("registers a VSM shadow pipeline with vertex and fragment shaders", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("vsm-vert", shader.ShaderTypeVertex, shaderPath("test_vsm_shadow_vert.wgsl"))
		fs := shader.NewShader("vsm-frag", shader.ShaderTypeFragment, shaderPath("test_vsm_shadow_frag.wgsl"))
		p := pipeline.NewPipeline("vsm-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterVSMShadowPipeline(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("vsm-pipe"))
		suite.NotNil(r.Pipeline("vsm-pipe").Pipeline())
	})

	suite.Run("returns error when vertex shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("bad-vsm", pipeline.PipelineTypeRender)
		err := r.RegisterVSMShadowPipeline(p)
		suite.Error(err)
	})
}

func (suite *rendererTest) TestVSMShadowPassLifecycle() {
	suite.Run("full VSM shadow lifecycle with registered pipeline and draw call", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("vsm-vert", shader.ShaderTypeVertex, shaderPath("test_vsm_shadow_vert.wgsl"))
		fs := shader.NewShader("vsm-frag", shader.ShaderTypeFragment, shaderPath("test_vsm_shadow_frag.wgsl"))
		p := pipeline.NewPipeline("vsm-shadow-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterVSMShadowPipeline(p)
		suite.NoError(err)

		vsmView, _, _, _, depthView, _, err := r.CreateVSMTextures(256, 256)
		suite.NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("vsm-shadow-mesh")
		vertexData, indexData := buildTriangleGeometry()
		err = r.InitMeshBuffers(meshProvider, vertexData, indexData, 3)
		suite.NoError(err)

		shadowUniformProvider := bind_group_provider.NewBindGroupProvider("vsm-shadow-uniform")
		err = r.InitBindGroup(shadowUniformProvider, vs.BindGroupLayoutDescriptor(0), nil, nil)
		suite.NoError(err)

		instanceProvider := bind_group_provider.NewBindGroupProvider("vsm-shadow-instance")
		err = r.InitBindGroup(instanceProvider, vs.BindGroupLayoutDescriptor(1), nil, map[int]uint64{0: 64})
		suite.NoError(err)

		err = r.BeginShadowFrame()
		suite.NoError(err)

		r.BeginVSMShadowPass(vsmView, depthView)

		err = r.ShadowDrawCall("vsm-shadow-pipe", meshProvider, 1, []bind_group_provider.BindGroupProvider{
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

func (suite *rendererTest) TestMaterial() {
	suite.Run("returns nil for unregistered material", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.Nil(r.Material("nonexistent"))
	})

	suite.Run("returns cached material after RegisterMaterial", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("mat-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("mat-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		p := pipeline.NewPipeline("mat-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)

		mat := material.NewMaterial(
			material.WithName("test-material"),
			material.WithPipelineKey("mat-pipe"),
		)

		err = r.RegisterMaterial(mat, "test-mat")
		suite.NoError(err)
		suite.NotNil(r.Material("test-material"))
		suite.Equal("test-material", r.Material("test-material").Name())
	})
}

func (suite *rendererTest) TestRegisterMaterial() {
	suite.Run("returns error when pipeline key is empty", func() {
		w, r := newTestRenderer()
		defer w.Close()

		mat := material.NewMaterial(material.WithName("no-key-mat"))
		err := r.RegisterMaterial(mat, "prefix")
		suite.Error(err)
		suite.Contains(err.Error(), "no pipeline key set")
	})

	suite.Run("returns error when pipeline not found and no opts provided", func() {
		w, r := newTestRenderer()
		defer w.Close()

		mat := material.NewMaterial(
			material.WithName("orphan-mat"),
			material.WithPipelineKey("nonexistent-pipe"),
		)

		err := r.RegisterMaterial(mat, "prefix")
		suite.Error(err)
		suite.Contains(err.Error(), "not found")
	})

	suite.Run("succeeds when pipeline exists and fragment has no material annotations", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("rm-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("rm-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		p := pipeline.NewPipeline("rm-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)

		mat := material.NewMaterial(
			material.WithName("simple-mat"),
			material.WithPipelineKey("rm-pipe"),
		)

		err = r.RegisterMaterial(mat, "simple")
		suite.NoError(err)
		suite.NotNil(r.Material("simple-mat"))
	})

	suite.Run("returns error when pipeline has no fragment shader", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("no-frag-pipe", pipeline.PipelineTypeRender)
		r.SetPipeline("no-frag-pipe", p)

		mat := material.NewMaterial(
			material.WithName("no-frag-mat"),
			material.WithPipelineKey("no-frag-pipe"),
		)

		err := r.RegisterMaterial(mat, "prefix")
		suite.Error(err)
		suite.Contains(err.Error(), "no fragment shader")
	})

	suite.Run("succeeds with material provider annotations creating fallback textures", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("matprov-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("matprov-frag", shader.ShaderTypeFragment, shaderPath("test_material_frag.wgsl"))
		p := pipeline.NewPipeline("matprov-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)

		mat := material.NewMaterial(
			material.WithName("textured-mat"),
			material.WithPipelineKey("matprov-pipe"),
		)

		err = r.RegisterMaterial(mat, "tex-mat")
		suite.NoError(err)
		suite.NotNil(r.Material("textured-mat"))
		suite.NotNil(mat.Provider(2))
	})

	suite.Run("auto-derives pipeline from base when FragmentShaderPath is set", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("base-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("base-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		basePipe := pipeline.NewPipeline("base", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(basePipe)
		suite.NoError(err)

		mat := material.NewMaterial(
			material.WithName("derived-mat"),
			material.WithPipelineKey("base_variant"),
			material.WithFragmentShaderPath(shaderPath("test_material_frag.wgsl")),
		)

		err = r.RegisterMaterial(mat, "derived")
		suite.NoError(err)
		suite.NotNil(r.Pipeline("base_variant"))
		suite.NotNil(r.Material("derived-mat"))
	})

	suite.Run("returns error when auto-deriving with no base pipeline", func() {
		w, r := newTestRenderer()
		defer w.Close()

		mat := material.NewMaterial(
			material.WithName("no-base-mat"),
			material.WithPipelineKey("orphan_variant"),
			material.WithFragmentShaderPath(shaderPath("test_material_frag.wgsl")),
		)

		err := r.RegisterMaterial(mat, "prefix")
		suite.Error(err)
		suite.Contains(err.Error(), "no base pipeline exists")
	})

	suite.Run("succeeds with real diffuse texture on material", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("tex-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("tex-frag", shader.ShaderTypeFragment, shaderPath("test_material_frag.wgsl"))
		p := pipeline.NewPipeline("tex-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)

		diffuseTex := &common.ImportedTexture{
			Name:     "diffuse",
			Data:     buildTestPNG(4, 4, color.NRGBA{R: 255, G: 0, B: 0, A: 255}),
			MimeType: "image/png",
		}

		mat := material.NewMaterial(
			material.WithName("real-tex-mat"),
			material.WithPipelineKey("tex-pipe"),
			material.WithDiffuseTexture(diffuseTex),
		)

		err = r.RegisterMaterial(mat, "real-tex")
		suite.NoError(err)
		suite.NotNil(r.Material("real-tex-mat"))
		suite.NotNil(mat.Provider(2))
	})

	suite.Run("succeeds with diffuse texture and custom sampler data", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("samp-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("samp-frag", shader.ShaderTypeFragment, shaderPath("test_material_frag.wgsl"))
		p := pipeline.NewPipeline("samp-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterPipelines(p)
		suite.NoError(err)

		diffuseTex := &common.ImportedTexture{
			Name:     "diffuse",
			Data:     buildTestPNG(2, 2, color.NRGBA{R: 128, G: 128, B: 128, A: 255}),
			MimeType: "image/png",
			SamplerData: &common.SamplerStagingData{
				AddressModeU:  wgpu.AddressModeClampToEdge,
				AddressModeV:  wgpu.AddressModeClampToEdge,
				AddressModeW:  wgpu.AddressModeClampToEdge,
				MagFilter:     wgpu.FilterModeNearest,
				MinFilter:     wgpu.FilterModeNearest,
				MipmapFilter:  wgpu.MipmapFilterModeNearest,
				LodMinClamp:   0,
				LodMaxClamp:   1,
				MaxAnisotropy: 1,
			},
		}

		mat := material.NewMaterial(
			material.WithName("sampler-mat"),
			material.WithPipelineKey("samp-pipe"),
			material.WithDiffuseTexture(diffuseTex),
		)

		err = r.RegisterMaterial(mat, "samp")
		suite.NoError(err)
		suite.NotNil(r.Material("sampler-mat"))
	})
}

func (suite *rendererTest) TestCreateBuffer() {
	suite.Run("creates a buffer with the specified usage", func() {
		w, r := newTestRenderer()
		defer w.Close()

		buf, err := r.CreateBuffer("test-buf", 256, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead)
		suite.NoError(err)
		suite.NotNil(buf)
	})

	suite.Run("creates a storage buffer", func() {
		w, r := newTestRenderer()
		defer w.Close()

		buf, err := r.CreateBuffer("storage-buf", 64, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc)
		suite.NoError(err)
		suite.NotNil(buf)
	})
}

func (suite *rendererTest) TestCopyBufferToBuffer() {
	suite.Run("copies data between buffers within compute frame", func() {
		w, r := newTestRenderer()
		defer w.Close()

		srcProvider := bind_group_provider.NewBindGroupProvider("copy-src")
		srcDesc := wgpu.BindGroupLayoutDescriptor{
			Label: "copy-src-layout",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageCompute,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeStorage,
						MinBindingSize: 16,
					},
				},
			},
		}
		err := r.InitBindGroup(srcProvider, srcDesc,
			map[int]wgpu.BufferUsage{0: wgpu.BufferUsageCopySrc},
			nil,
		)
		suite.NoError(err)

		srcData := make([]byte, 16)
		for i := range srcData {
			srcData[i] = byte(i + 1)
		}
		r.WriteBuffers([]bind_group_provider.BufferWrite{
			{Provider: srcProvider, Binding: 0, Offset: 0, Data: srcData},
		})

		dstBuf, err := r.CreateBuffer("copy-dst", 16, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead)
		suite.NoError(err)

		err = r.BeginComputeFrame()
		suite.NoError(err)

		suite.NotPanics(func() {
			r.CopyBufferToBuffer(srcProvider.Buffer(0), dstBuf, 0, 0, 16)
		})

		r.EndComputeFrame()
	})
}

func (suite *rendererTest) TestReadMappedBuffer() {
	suite.Run("reads data from a mapped buffer", func() {
		w, r := newTestRenderer()
		defer w.Close()

		srcProvider := bind_group_provider.NewBindGroupProvider("read-src")
		srcDesc := wgpu.BindGroupLayoutDescriptor{
			Label: "read-src-layout",
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    0,
					Visibility: wgpu.ShaderStageCompute,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeStorage,
						MinBindingSize: 16,
					},
				},
			},
		}
		err := r.InitBindGroup(srcProvider, srcDesc,
			map[int]wgpu.BufferUsage{0: wgpu.BufferUsageCopySrc},
			nil,
		)
		suite.NoError(err)

		srcData := make([]byte, 16)
		binary.LittleEndian.PutUint32(srcData[0:], math.Float32bits(1.0))
		binary.LittleEndian.PutUint32(srcData[4:], math.Float32bits(2.0))
		binary.LittleEndian.PutUint32(srcData[8:], math.Float32bits(3.0))
		binary.LittleEndian.PutUint32(srcData[12:], math.Float32bits(4.0))
		r.WriteBuffers([]bind_group_provider.BufferWrite{
			{Provider: srcProvider, Binding: 0, Offset: 0, Data: srcData},
		})

		dstBuf, err := r.CreateBuffer("read-dst", 16, wgpu.BufferUsageCopyDst|wgpu.BufferUsageMapRead)
		suite.NoError(err)

		err = r.BeginComputeFrame()
		suite.NoError(err)
		r.CopyBufferToBuffer(srcProvider.Buffer(0), dstBuf, 0, 0, 16)
		r.EndComputeFrame()

		data, err := r.ReadMappedBuffer(dstBuf, 0, 16)
		suite.NoError(err)
		suite.Len(data, 16)

		v0 := math.Float32frombits(binary.LittleEndian.Uint32(data[0:4]))
		v1 := math.Float32frombits(binary.LittleEndian.Uint32(data[4:8]))
		v2 := math.Float32frombits(binary.LittleEndian.Uint32(data[8:12]))
		v3 := math.Float32frombits(binary.LittleEndian.Uint32(data[12:16]))
		suite.InDelta(1.0, v0, 0.001)
		suite.InDelta(2.0, v1, 0.001)
		suite.InDelta(3.0, v2, 0.001)
		suite.InDelta(4.0, v3, 0.001)
	})
}

func (suite *rendererTest) TestSampleCount() {
	suite.Run("returns default sample count", func() {
		w, r := newTestRenderer()
		defer w.Close()
		count := r.SampleCount()
		suite.Greater(count, uint32(0))
	})

	suite.Run("returns 1 when MSAA is off", func() {
		w, r := newTestRenderer(renderer.WithMSAA(renderer.MSAAOff))
		defer w.Close()
		suite.Equal(uint32(1), r.SampleCount())
	})
}

func (suite *rendererTest) TestSetRenderTargetFormat() {
	suite.Run("does not panic with a valid format", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.SetRenderTargetFormat(wgpu.TextureFormatRGBA8Unorm)
		})
	})
}

func (suite *rendererTest) TestWriteRawBuffer() {
	suite.Run("writes raw data to a GPU buffer", func() {
		w, r := newTestRenderer()
		defer w.Close()

		buf, err := r.CreateBuffer("raw-write-test", 64, wgpu.BufferUsageCopyDst|wgpu.BufferUsageStorage)
		suite.NoError(err)
		suite.NotNil(buf)

		data := make([]byte, 16)
		binary.LittleEndian.PutUint32(data[0:], math.Float32bits(1.0))
		binary.LittleEndian.PutUint32(data[4:], math.Float32bits(2.0))
		binary.LittleEndian.PutUint32(data[8:], math.Float32bits(3.0))
		binary.LittleEndian.PutUint32(data[12:], math.Float32bits(4.0))

		suite.NotPanics(func() {
			r.WriteRawBuffer(buf, 0, data)
		})
	})
}

func (suite *rendererTest) TestWriteTexture() {
	suite.Run("writes pixel data to a GPU texture", func() {
		w, r := newTestRenderer()
		defer w.Close()

		ssrView, ssrTex, err := r.CreateSSRTextures(4, 4)
		suite.NoError(err)
		suite.NotNil(ssrView)
		suite.NotNil(ssrTex)

		pixelData := make([]byte, 4*4*8)
		suite.NotPanics(func() {
			r.WriteTexture(ssrTex, pixelData, 4, 4, 4*8)
		})
	})
}

func (suite *rendererTest) TestBeginGeometryFrameEndGeometryFrame() {
	suite.Run("geometry lifecycle completes without error", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.BeginGeometryFrame()
		suite.NoError(err)
		suite.NotPanics(func() {
			r.EndGeometryFrame()
		})
	})
}

func (suite *rendererTest) TestEndGeometryFrameWithoutBegin() {
	suite.Run("does not panic when called without BeginGeometryFrame", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.EndGeometryFrame()
		})
	})
}

func (suite *rendererTest) TestCreateGBufferTextures() {
	suite.Run("returns non-nil textures and views", func() {
		w, r := newTestRenderer()
		defer w.Close()
		normView, normTex, albedoView, albedoTex, depthView, depthTex, err := r.CreateGBufferTextures(64, 64)
		suite.NoError(err)
		suite.NotNil(normView)
		suite.NotNil(normTex)
		suite.NotNil(albedoView)
		suite.NotNil(albedoTex)
		suite.NotNil(depthView)
		suite.NotNil(depthTex)
	})
}

func (suite *rendererTest) TestRegisterGBufferPipeline() {
	suite.Run("skips pipeline with key already in cache", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("gbuf-preloaded", pipeline.PipelineTypeRender)
		r.SetPipeline("gbuf-preloaded", p)

		err := r.RegisterGBufferPipeline(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("gbuf-preloaded"))
	})

	suite.Run("registers a GBuffer pipeline with vertex and fragment shaders", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("gbuf-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("gbuf-frag", shader.ShaderTypeFragment, shaderPath("test_gbuffer_frag.wgsl"))
		p := pipeline.NewPipeline("gbuf-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterGBufferPipeline(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("gbuf-pipe"))
		suite.NotNil(r.Pipeline("gbuf-pipe").Pipeline())
	})

	suite.Run("returns error when vertex shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("bad-gbuf", pipeline.PipelineTypeRender)
		err := r.RegisterGBufferPipeline(p)
		suite.Error(err)
	})

	suite.Run("returns error when fragment shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("gbuf-vert-only", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		p := pipeline.NewPipeline("no-frag-gbuf", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
		)
		err := r.RegisterGBufferPipeline(p)
		suite.Error(err)
	})
}

func (suite *rendererTest) TestBeginGBufferFrameEndGBufferFrame() {
	suite.Run("GBuffer lifecycle completes without error", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.BeginGBufferFrame()
		suite.NoError(err)
		suite.NotPanics(func() {
			r.EndGBufferFrame()
		})
	})
}

func (suite *rendererTest) TestEndGBufferFrameWithoutBegin() {
	suite.Run("does not panic when called without BeginGBufferFrame", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.EndGBufferFrame()
		})
	})
}

func (suite *rendererTest) TestEndGBufferPassWithoutBegin() {
	suite.Run("does not panic when called without BeginGBufferPass", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.EndGBufferPass()
		})
	})
}

func (suite *rendererTest) TestGBufferDrawCall() {
	suite.Run("returns error when pipeline key is not found", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.GBufferDrawCall("nonexistent", nil, 0, nil)
		suite.Error(err)
		suite.Contains(err.Error(), "nonexistent")
	})
}

func (suite *rendererTest) TestGBufferDrawCallIndirect() {
	suite.Run("returns error when pipeline key is not found", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.GBufferDrawCallIndirect("nonexistent", nil, nil, nil)
		suite.Error(err)
		suite.Contains(err.Error(), "nonexistent")
	})

	suite.Run("full GBuffer indirect draw lifecycle", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("gbuf-ind-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("gbuf-ind-frag", shader.ShaderTypeFragment, shaderPath("test_gbuffer_frag.wgsl"))
		p := pipeline.NewPipeline("gbuf-ind-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterGBufferPipeline(p)
		suite.NoError(err)

		normView, _, albedoView, _, depthView, _, err := r.CreateGBufferTextures(64, 64)
		suite.NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("gbuf-ind-mesh")
		vertexData, indexData := buildTriangleGeometry()
		err = r.InitMeshBuffers(meshProvider, vertexData, indexData, 3)
		suite.NoError(err)

		cameraProvider := bind_group_provider.NewBindGroupProvider("gbuf-ind-camera")
		err = r.InitBindGroup(cameraProvider, vs.BindGroupLayoutDescriptor(0), nil, nil)
		suite.NoError(err)

		instanceProvider := bind_group_provider.NewBindGroupProvider("gbuf-ind-instance")
		err = r.InitBindGroup(instanceProvider, vs.BindGroupLayoutDescriptor(1), nil, map[int]uint64{0: 64})
		suite.NoError(err)

		indirectArgs := buildIndirectArgs(3, 1, 0, 0, 0)
		indirectBuf, err := r.CreateBuffer("gbuf-indirect", uint64(len(indirectArgs)), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst)
		suite.NoError(err)
		r.WriteRawBuffer(indirectBuf, 0, indirectArgs)

		err = r.BeginGBufferFrame()
		suite.NoError(err)

		r.BeginGBufferPass(normView, albedoView, depthView)

		err = r.GBufferDrawCallIndirect("gbuf-ind-pipe", meshProvider, indirectBuf, []bind_group_provider.BindGroupProvider{
			cameraProvider,
			instanceProvider,
		})
		suite.NoError(err)

		r.EndGBufferPass()
		r.EndGBufferFrame()
	})
}

func (suite *rendererTest) TestGBufferFullLifecycle() {
	suite.Run("full GBuffer lifecycle with registered pipeline and draw call", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("gbuf-lc-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("gbuf-lc-frag", shader.ShaderTypeFragment, shaderPath("test_gbuffer_frag.wgsl"))
		p := pipeline.NewPipeline("gbuf-lc-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterGBufferPipeline(p)
		suite.NoError(err)

		normView, _, albedoView, _, depthView, _, err := r.CreateGBufferTextures(64, 64)
		suite.NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("gbuf-mesh")
		vertexData, indexData := buildTriangleGeometry()
		err = r.InitMeshBuffers(meshProvider, vertexData, indexData, 3)
		suite.NoError(err)

		cameraProvider := bind_group_provider.NewBindGroupProvider("gbuf-camera")
		err = r.InitBindGroup(cameraProvider, vs.BindGroupLayoutDescriptor(0), nil, nil)
		suite.NoError(err)

		instanceProvider := bind_group_provider.NewBindGroupProvider("gbuf-instance")
		err = r.InitBindGroup(instanceProvider, vs.BindGroupLayoutDescriptor(1), nil, map[int]uint64{0: 64})
		suite.NoError(err)

		err = r.BeginGBufferFrame()
		suite.NoError(err)

		r.BeginGBufferPass(normView, albedoView, depthView)

		err = r.GBufferDrawCall("gbuf-lc-pipe", meshProvider, 1, []bind_group_provider.BindGroupProvider{
			cameraProvider,
			instanceProvider,
		})
		suite.NoError(err)

		r.EndGBufferPass()
		r.EndGBufferFrame()
	})
}

func (suite *rendererTest) TestCreateSSAOTextures() {
	suite.Run("returns non-nil textures and views", func() {
		w, r := newTestRenderer()
		defer w.Close()
		rawView, rawTex, blurredView, blurredTex, scratchView, scratchTex, noiseView, noiseTex, err := r.CreateSSAOTextures(64, 64)
		suite.NoError(err)
		suite.NotNil(rawView)
		suite.NotNil(rawTex)
		suite.NotNil(blurredView)
		suite.NotNil(blurredTex)
		suite.NotNil(scratchView)
		suite.NotNil(scratchTex)
		suite.NotNil(noiseView)
		suite.NotNil(noiseTex)
	})
}

func (suite *rendererTest) TestCreateProbeBakeTextures() {
	suite.Run("returns non-nil textures and views", func() {
		w, r := newTestRenderer()
		defer w.Close()
		colorView, colorTex, depthView, depthTex, err := r.CreateProbeBakeTextures(32)
		suite.NoError(err)
		suite.NotNil(colorView)
		suite.NotNil(colorTex)
		suite.NotNil(depthView)
		suite.NotNil(depthTex)
	})
}

func (suite *rendererTest) TestRegisterProbeBakePipeline() {
	suite.Run("skips pipeline with key already in cache", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("probe-preloaded", pipeline.PipelineTypeRender)
		r.SetPipeline("probe-preloaded", p)

		err := r.RegisterProbeBakePipeline(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("probe-preloaded"))
	})

	suite.Run("registers a probe bake pipeline with vertex and fragment shaders", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("probe-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("probe-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		p := pipeline.NewPipeline("probe-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterProbeBakePipeline(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("probe-pipe"))
		suite.NotNil(r.Pipeline("probe-pipe").Pipeline())
	})

	suite.Run("returns error when vertex shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("bad-probe", pipeline.PipelineTypeRender)
		err := r.RegisterProbeBakePipeline(p)
		suite.Error(err)
	})

	suite.Run("returns error when fragment shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("probe-vert-only", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		p := pipeline.NewPipeline("no-frag-probe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
		)
		err := r.RegisterProbeBakePipeline(p)
		suite.Error(err)
	})
}

func (suite *rendererTest) TestBeginProbeBakeFrameEndProbeBakeFrame() {
	suite.Run("probe bake lifecycle completes without error", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.BeginProbeBakeFrame()
		suite.NoError(err)
		suite.NotPanics(func() {
			r.EndProbeBakeFrame()
		})
	})
}

func (suite *rendererTest) TestEndProbeBakeFrameWithoutBegin() {
	suite.Run("does not panic when called without BeginProbeBakeFrame", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.EndProbeBakeFrame()
		})
	})
}

func (suite *rendererTest) TestEndProbeBakePassWithoutBegin() {
	suite.Run("does not panic when called without BeginProbeBakePass", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.EndProbeBakePass()
		})
	})
}

func (suite *rendererTest) TestProbeBakeDrawCall() {
	suite.Run("returns error when pipeline key is not found", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.ProbeBakeDrawCall("nonexistent", nil, 0, nil)
		suite.Error(err)
		suite.Contains(err.Error(), "nonexistent")
	})
}

func (suite *rendererTest) TestProbeBakeDrawCallIndirect() {
	suite.Run("returns error when pipeline key is not found", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.ProbeBakeDrawCallIndirect("nonexistent", nil, nil, nil)
		suite.Error(err)
		suite.Contains(err.Error(), "nonexistent")
	})

	suite.Run("full probe bake indirect draw lifecycle", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("probe-ind-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("probe-ind-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		p := pipeline.NewPipeline("probe-ind-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterProbeBakePipeline(p)
		suite.NoError(err)

		colorView, _, depthView, _, err := r.CreateProbeBakeTextures(32)
		suite.NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("probe-ind-mesh")
		vertexData, indexData := buildTriangleGeometry()
		err = r.InitMeshBuffers(meshProvider, vertexData, indexData, 3)
		suite.NoError(err)

		cameraProvider := bind_group_provider.NewBindGroupProvider("probe-ind-camera")
		err = r.InitBindGroup(cameraProvider, vs.BindGroupLayoutDescriptor(0), nil, nil)
		suite.NoError(err)

		instanceProvider := bind_group_provider.NewBindGroupProvider("probe-ind-instance")
		err = r.InitBindGroup(instanceProvider, vs.BindGroupLayoutDescriptor(1), nil, map[int]uint64{0: 64})
		suite.NoError(err)

		indirectArgs := buildIndirectArgs(3, 1, 0, 0, 0)
		indirectBuf, err := r.CreateBuffer("probe-indirect", uint64(len(indirectArgs)), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst)
		suite.NoError(err)
		r.WriteRawBuffer(indirectBuf, 0, indirectArgs)

		err = r.BeginProbeBakeFrame()
		suite.NoError(err)

		r.BeginProbeBakePass(colorView, depthView)

		err = r.ProbeBakeDrawCallIndirect("probe-ind-pipe", meshProvider, indirectBuf, []bind_group_provider.BindGroupProvider{
			cameraProvider,
			instanceProvider,
		})
		suite.NoError(err)

		r.EndProbeBakePass()
		r.EndProbeBakeFrame()
	})
}

func (suite *rendererTest) TestProbeBakeFullLifecycle() {
	suite.Run("full probe bake lifecycle with registered pipeline and draw call", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("probe-lc-vert", shader.ShaderTypeVertex, shaderPath("test_vert.wgsl"))
		fs := shader.NewShader("probe-lc-frag", shader.ShaderTypeFragment, shaderPath("test_frag.wgsl"))
		p := pipeline.NewPipeline("probe-lc-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterProbeBakePipeline(p)
		suite.NoError(err)

		colorView, _, depthView, _, err := r.CreateProbeBakeTextures(32)
		suite.NoError(err)

		meshProvider := bind_group_provider.NewBindGroupProvider("probe-mesh")
		vertexData, indexData := buildTriangleGeometry()
		err = r.InitMeshBuffers(meshProvider, vertexData, indexData, 3)
		suite.NoError(err)

		cameraProvider := bind_group_provider.NewBindGroupProvider("probe-camera")
		err = r.InitBindGroup(cameraProvider, vs.BindGroupLayoutDescriptor(0), nil, nil)
		suite.NoError(err)

		instanceProvider := bind_group_provider.NewBindGroupProvider("probe-instance")
		err = r.InitBindGroup(instanceProvider, vs.BindGroupLayoutDescriptor(1), nil, map[int]uint64{0: 64})
		suite.NoError(err)

		err = r.BeginProbeBakeFrame()
		suite.NoError(err)

		r.BeginProbeBakePass(colorView, depthView)

		err = r.ProbeBakeDrawCall("probe-lc-pipe", meshProvider, 1, []bind_group_provider.BindGroupProvider{
			cameraProvider,
			instanceProvider,
		})
		suite.NoError(err)

		r.EndProbeBakePass()
		r.EndProbeBakeFrame()
	})
}

func (suite *rendererTest) TestCreateCompositionTextures() {
	suite.Run("returns non-nil textures and views with MSAAOff", func() {
		w, r := newTestRenderer(renderer.WithMSAA(renderer.MSAAOff))
		defer w.Close()
		hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex, err := r.CreateCompositionTextures(64, 64, 1)
		suite.NoError(err)
		suite.NotNil(hdrView)
		suite.NotNil(hdrTex)
		suite.NotNil(depthView)
		suite.NotNil(depthTex)
		// msaa textures are nil when sample count is 1
		_ = msaaView
		_ = msaaTex
	})

	suite.Run("returns non-nil MSAA textures and views with sampleCount 4", func() {
		w, r := newTestRenderer(renderer.WithMSAA(renderer.MSAA4x))
		defer w.Close()
		hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex, err := r.CreateCompositionTextures(64, 64, 4)
		suite.NoError(err)
		suite.NotNil(hdrView)
		suite.NotNil(hdrTex)
		suite.NotNil(msaaView)
		suite.NotNil(msaaTex)
		suite.NotNil(depthView)
		suite.NotNil(depthTex)
	})
}

func (suite *rendererTest) TestCreateSSRTextures() {
	suite.Run("returns non-nil textures and views", func() {
		w, r := newTestRenderer()
		defer w.Close()
		ssrView, ssrTex, err := r.CreateSSRTextures(64, 64)
		suite.NoError(err)
		suite.NotNil(ssrView)
		suite.NotNil(ssrTex)
	})
}

func (suite *rendererTest) TestCreateHiZTextures() {
	suite.Run("returns non-nil textures, views and positive mip count", func() {
		w, r := newTestRenderer()
		defer w.Close()
		hizView, hizTex, mipReadViews, mipStorageViews, mipCount, err := r.CreateHiZTextures(64, 64)
		suite.NoError(err)
		suite.NotNil(hizView)
		suite.NotNil(hizTex)
		suite.NotEmpty(mipReadViews)
		suite.NotEmpty(mipStorageViews)
		suite.Greater(mipCount, 0)
	})

	suite.Run("handles non-square dimensions where height exceeds width", func() {
		w, r := newTestRenderer()
		defer w.Close()
		hizView, hizTex, mipReadViews, mipStorageViews, mipCount, err := r.CreateHiZTextures(32, 128)
		suite.NoError(err)
		suite.NotNil(hizView)
		suite.NotNil(hizTex)
		suite.NotEmpty(mipReadViews)
		suite.NotEmpty(mipStorageViews)
		suite.Greater(mipCount, 0)
	})
}

func (suite *rendererTest) TestRegisterCompositionPipeline() {
	suite.Run("skips pipeline with key already in cache", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("comp-preloaded", pipeline.PipelineTypeRender)
		r.SetPipeline("comp-preloaded", p)

		err := r.RegisterCompositionPipeline(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("comp-preloaded"))
	})

	suite.Run("registers a composition pipeline with fullscreen shaders", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("comp-vert", shader.ShaderTypeVertex, shaderPath("test_composition_vert.wgsl"))
		fs := shader.NewShader("comp-frag", shader.ShaderTypeFragment, shaderPath("test_composition_frag.wgsl"))
		p := pipeline.NewPipeline("comp-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterCompositionPipeline(p)
		suite.NoError(err)
		suite.NotNil(r.Pipeline("comp-pipe"))
		suite.NotNil(r.Pipeline("comp-pipe").Pipeline())
	})

	suite.Run("returns error when vertex shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		p := pipeline.NewPipeline("bad-comp", pipeline.PipelineTypeRender)
		err := r.RegisterCompositionPipeline(p)
		suite.Error(err)
	})

	suite.Run("returns error when fragment shader is missing", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("comp-vert-only", shader.ShaderTypeVertex, shaderPath("test_composition_vert.wgsl"))
		p := pipeline.NewPipeline("no-frag-comp", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
		)
		err := r.RegisterCompositionPipeline(p)
		suite.Error(err)
	})
}

func (suite *rendererTest) TestBeginCompositionFrameEndCompositionFrame() {
	suite.Run("composition lifecycle completes without error", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.BeginCompositionFrame()
		suite.NoError(err)
		suite.NotPanics(func() {
			r.EndCompositionFrame()
		})
	})

	suite.Run("returns error when called twice without end", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.BeginCompositionFrame()
		suite.NoError(err)
		err = r.BeginCompositionFrame()
		suite.Error(err)
		suite.Contains(err.Error(), "previous composition frame")
		r.EndCompositionFrame()
	})
}

func (suite *rendererTest) TestEndCompositionFrameWithoutBegin() {
	suite.Run("does not panic when called without BeginCompositionFrame", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.EndCompositionFrame()
		})
	})
}

func (suite *rendererTest) TestEndCompositionPassWithoutBegin() {
	suite.Run("does not panic when called without BeginCompositionPass", func() {
		w, r := newTestRenderer()
		defer w.Close()
		suite.NotPanics(func() {
			r.EndCompositionPass()
		})
	})
}

func (suite *rendererTest) TestCompositionDrawCall() {
	suite.Run("returns error when pipeline key is not found", func() {
		w, r := newTestRenderer()
		defer w.Close()
		err := r.CompositionDrawCall("nonexistent", nil)
		suite.Error(err)
		suite.Contains(err.Error(), "nonexistent")
	})
}

func (suite *rendererTest) TestBeginHDRFrame() {
	suite.Run("HDR frame with created composition textures", func() {
		w, r := newTestRenderer(renderer.WithMSAA(renderer.MSAAOff))
		defer w.Close()

		hdrView, _, _, _, depthView, _, err := r.CreateCompositionTextures(64, 64, 1)
		suite.NoError(err)

		err = r.BeginHDRFrame(hdrView, nil, depthView, 1)
		suite.NoError(err)

		r.EndFrame()
		r.Present()
	})

	suite.Run("MSAA HDR frame uses resolve view", func() {
		w, r := newTestRenderer(renderer.WithMSAA(renderer.MSAA4x))
		defer w.Close()

		hdrView, _, msaaView, _, depthView, _, err := r.CreateCompositionTextures(64, 64, 4)
		suite.NoError(err)

		err = r.BeginHDRFrame(msaaView, hdrView, depthView, 4)
		suite.NoError(err)

		r.EndFrame()
		r.Present()
	})
}

func (suite *rendererTest) TestCompositionFullLifecycle() {
	suite.Run("full composition lifecycle with registered pipeline and draw call", func() {
		w, r := newTestRenderer()
		defer w.Close()

		vs := shader.NewShader("comp-lc-vert", shader.ShaderTypeVertex, shaderPath("test_composition_vert.wgsl"))
		fs := shader.NewShader("comp-lc-frag", shader.ShaderTypeFragment, shaderPath("test_composition_frag.wgsl"))
		p := pipeline.NewPipeline("comp-lc-pipe", pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(vs),
			pipeline.WithFragmentShader(fs),
		)

		err := r.RegisterCompositionPipeline(p)
		suite.NoError(err)

		err = r.BeginCompositionFrame()
		suite.NoError(err)

		r.BeginCompositionPass()

		err = r.CompositionDrawCall("comp-lc-pipe", nil)
		suite.NoError(err)

		r.EndCompositionPass()
		r.EndCompositionFrame()
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

// buildTestPNG generates an in-memory PNG image of the given dimensions filled with a solid color.
func buildTestPNG(width, height int, c color.NRGBA) []byte {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	return buf.Bytes()
}
