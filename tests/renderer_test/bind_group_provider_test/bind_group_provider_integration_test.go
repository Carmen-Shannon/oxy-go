package bind_group_provider_test

import (
	"runtime"
	"testing"

	bgp "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type bindGroupProviderIntegrationTest struct {
	suite.Suite
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue
}

func TestBindGroupProviderIntegration(t *testing.T) {
	suite.Run(t, new(bindGroupProviderIntegrationTest))
}

func (suite *bindGroupProviderIntegrationTest) SetupSuite() {
	runtime.LockOSThread()

	suite.instance = wgpu.CreateInstance(nil)
	suite.Require().NotNil(suite.instance)

	adapter, err := suite.instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		ForceFallbackAdapter: true,
	})
	suite.Require().NoError(err)
	suite.Require().NotNil(adapter)
	suite.adapter = adapter

	device, err := adapter.RequestDevice(nil)
	suite.Require().NoError(err)
	suite.Require().NotNil(device)
	suite.device = device
	suite.queue = device.GetQueue()
}

func (suite *bindGroupProviderIntegrationTest) TearDownSuite() {
	if suite.device != nil {
		suite.device.Release()
	}
	if suite.adapter != nil {
		suite.adapter.Release()
	}
	if suite.instance != nil {
		suite.instance.Release()
	}
}

// createBuffer is a test helper that creates a real GPU buffer with the given size.
//
// Parameters:
//   - label: debug label for the buffer
//   - size: buffer size in bytes
//
// Returns:
//   - *wgpu.Buffer: the created GPU buffer
func (suite *bindGroupProviderIntegrationTest) createBuffer(label string, size uint64) *wgpu.Buffer {
	buf, err := suite.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label:            label,
		Size:             size,
		Usage:            wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		MappedAtCreation: false,
	})
	suite.Require().NoError(err)
	suite.Require().NotNil(buf)
	return buf
}

// createTextureView is a test helper that creates a real GPU texture and returns its default view.
//
// Parameters:
//   - label: debug label for the texture
//
// Returns:
//   - *wgpu.TextureView: the texture view created from the texture
func (suite *bindGroupProviderIntegrationTest) createTextureView(label string) *wgpu.TextureView {
	tex, err := suite.device.CreateTexture(&wgpu.TextureDescriptor{
		Label:         label,
		Usage:         wgpu.TextureUsageTextureBinding,
		Dimension:     wgpu.TextureDimension2D,
		Size:          wgpu.Extent3D{Width: 1, Height: 1, DepthOrArrayLayers: 1},
		Format:        wgpu.TextureFormatRGBA8UnormSrgb,
		MipLevelCount: 1,
		SampleCount:   1,
	})
	suite.Require().NoError(err)

	view, err := tex.CreateView(nil)
	suite.Require().NoError(err)
	suite.Require().NotNil(view)
	return view
}

// createSampler is a test helper that creates a real GPU sampler with default settings.
//
// Parameters:
//   - label: debug label for the sampler
//
// Returns:
//   - *wgpu.Sampler: the created GPU sampler
func (suite *bindGroupProviderIntegrationTest) createSampler(label string) *wgpu.Sampler {
	samp, err := suite.device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         label,
		AddressModeU:  wgpu.AddressModeRepeat,
		AddressModeV:  wgpu.AddressModeRepeat,
		AddressModeW:  wgpu.AddressModeRepeat,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MaxAnisotropy: 1,
	})
	suite.Require().NoError(err)
	suite.Require().NotNil(samp)
	return samp
}

// createBindGroupLayout is a test helper that creates a real GPU bind group layout with a single uniform buffer binding.
//
// Parameters:
//   - label: debug label for the layout
//
// Returns:
//   - *wgpu.BindGroupLayout: the created bind group layout
func (suite *bindGroupProviderIntegrationTest) createBindGroupLayout(label string) *wgpu.BindGroupLayout {
	bgl, err := suite.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: label,
		Entries: []wgpu.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: wgpu.ShaderStageVertex | wgpu.ShaderStageFragment,
				Buffer: wgpu.BufferBindingLayout{
					Type:           wgpu.BufferBindingTypeUniform,
					MinBindingSize: 64,
				},
			},
		},
	})
	suite.Require().NoError(err)
	suite.Require().NotNil(bgl)
	return bgl
}

// createBindGroup is a test helper that creates a real GPU bind group from a layout and buffer.
//
// Parameters:
//   - label: debug label for the bind group
//   - layout: the bind group layout to use
//   - buf: the buffer to bind at index 0
//
// Returns:
//   - *wgpu.BindGroup: the created bind group
func (suite *bindGroupProviderIntegrationTest) createBindGroup(label string, layout *wgpu.BindGroupLayout, buf *wgpu.Buffer) *wgpu.BindGroup {
	bg, err := suite.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  label,
		Layout: layout,
		Entries: []wgpu.BindGroupEntry{
			{
				Binding: 0,
				Buffer:  buf,
				Size:    wgpu.WholeSize,
			},
		},
	})
	suite.Require().NoError(err)
	suite.Require().NotNil(bg)
	return bg
}

func (suite *bindGroupProviderIntegrationTest) TestReleaseWithRealGPUResources() {
	suite.Run("release frees all GPU resources", func() {
		buf0 := suite.createBuffer("buf-0", 64)
		buf1 := suite.createBuffer("buf-1", 128)
		tv := suite.createTextureView("tex-0")
		samp := suite.createSampler("samp-0")
		bgl := suite.createBindGroupLayout("bgl")
		bg := suite.createBindGroup("bg", bgl, buf0)
		vb := suite.createBuffer("vertex-buf", 256)
		ib := suite.createBuffer("index-buf", 128)

		p := bgp.NewBindGroupProvider("release-test")
		p.SetBuffer(0, buf0)
		p.SetBuffer(1, buf1)
		p.SetTextureView(0, tv)
		p.SetSampler(0, samp)
		p.SetBindGroupLayout(bgl)
		p.SetBindGroup(bg)
		p.SetVertexBuffer(vb)
		p.SetIndexBuffer(ib)

		suite.NotPanics(func() {
			p.Release()
		})

		suite.Nil(p.BindGroup())
		suite.Nil(p.BindGroupLayout())
		suite.Nil(p.VertexBuffer())
		suite.Nil(p.IndexBuffer())
	})

	suite.Run("release with only buffers frees them", func() {
		buf := suite.createBuffer("solo-buf", 64)

		p := bgp.NewBindGroupProvider("buf-only")
		p.SetBuffer(0, buf)

		suite.NotPanics(func() {
			p.Release()
		})

		suite.Len(p.Buffers(), 0)
	})

	suite.Run("release with only texture views frees them", func() {
		tv := suite.createTextureView("solo-tex")

		p := bgp.NewBindGroupProvider("tv-only")
		p.SetTextureView(0, tv)

		suite.NotPanics(func() {
			p.Release()
		})

		suite.Len(p.TextureViews(), 0)
	})

	suite.Run("release with only samplers frees them", func() {
		samp := suite.createSampler("solo-samp")

		p := bgp.NewBindGroupProvider("samp-only")
		p.SetSampler(0, samp)

		suite.NotPanics(func() {
			p.Release()
		})

		suite.Len(p.Samplers(), 0)
	})

	suite.Run("release with only vertex and index buffers frees them", func() {
		vb := suite.createBuffer("vb", 64)
		ib := suite.createBuffer("ib", 64)

		p := bgp.NewBindGroupProvider("vb-ib-only")
		p.SetVertexBuffer(vb)
		p.SetIndexBuffer(ib)

		suite.NotPanics(func() {
			p.Release()
		})

		suite.Nil(p.VertexBuffer())
		suite.Nil(p.IndexBuffer())
	})

	suite.Run("release with only bind group and layout frees them", func() {
		buf := suite.createBuffer("bg-buf", 64)
		bgl := suite.createBindGroupLayout("bg-layout")
		bg := suite.createBindGroup("bg", bgl, buf)

		p := bgp.NewBindGroupProvider("bg-only")
		p.SetBindGroupLayout(bgl)
		p.SetBindGroup(bg)
		// buf is not managed by this provider, release it manually
		defer buf.Release()

		suite.NotPanics(func() {
			p.Release()
		})

		suite.Nil(p.BindGroup())
		suite.Nil(p.BindGroupLayout())
	})

	suite.Run("double release with real resources does not panic", func() {
		buf := suite.createBuffer("double-buf", 64)
		tv := suite.createTextureView("double-tex")
		samp := suite.createSampler("double-samp")

		p := bgp.NewBindGroupProvider("double-release")
		p.SetBuffer(0, buf)
		p.SetTextureView(0, tv)
		p.SetSampler(0, samp)

		suite.NotPanics(func() {
			p.Release()
			p.Release()
		})
	})

	suite.Run("release with multiple buffers at different bindings", func() {
		buf0 := suite.createBuffer("multi-buf-0", 64)
		buf1 := suite.createBuffer("multi-buf-1", 128)
		buf2 := suite.createBuffer("multi-buf-2", 256)

		p := bgp.NewBindGroupProvider("multi-buf")
		p.SetBuffer(0, buf0)
		p.SetBuffer(1, buf1)
		p.SetBuffer(5, buf2)

		suite.NotPanics(func() {
			p.Release()
		})

		suite.Len(p.Buffers(), 0)
	})

	suite.Run("release with multiple texture views and samplers at different bindings", func() {
		tv0 := suite.createTextureView("multi-tv-0")
		tv1 := suite.createTextureView("multi-tv-1")
		samp0 := suite.createSampler("multi-samp-0")
		samp1 := suite.createSampler("multi-samp-1")

		p := bgp.NewBindGroupProvider("multi-resources")
		p.SetTextureView(0, tv0)
		p.SetTextureView(1, tv1)
		p.SetSampler(0, samp0)
		p.SetSampler(1, samp1)

		suite.NotPanics(func() {
			p.Release()
		})

		suite.Len(p.TextureViews(), 0)
		suite.Len(p.Samplers(), 0)
	})
}
