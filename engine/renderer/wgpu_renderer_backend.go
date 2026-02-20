package renderer

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/cogentcore/webgpu/wgpu"
)

type wgpuRendererBackendImpl struct {
	mu     *sync.Mutex
	device *wgpu.Device
	queue  *wgpu.Queue

	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	surface  *wgpu.Surface

	surfaceFormat        *wgpu.TextureFormat
	msaaTextureView      *wgpu.TextureView
	depthTextureView     *wgpu.TextureView
	renderPassDescriptor *wgpu.RenderPassDescriptor

	presentMode wgpu.PresentMode // defaults to PresentModeImmediate (Uncapped)
	sampleCount MSAASampleCount  // MSAA sample count for the main render pass

	// Frame state for batched rendering across multiple draw calls
	frameEncoder *wgpu.CommandEncoder
	framePass    *wgpu.RenderPassEncoder
	frameSurface *wgpu.Texture
	frameView    *wgpu.TextureView

	// Compute frame state for batching all compute dispatches into a single GPU submission
	computeFrameEncoder *wgpu.CommandEncoder

	// Shadow pass state for rendering depth-only passes from a light's perspective.
	// Shadow passes use their own command encoder, a Depth32Float texture (no color),
	// sample count 1 (no MSAA), and front-face culling to reduce self-shadowing.
	shadowFrameEncoder *wgpu.CommandEncoder
	shadowPass         *wgpu.RenderPassEncoder
}

type wgpuRendererBackend interface {
	Device() *wgpu.Device
	Queue() *wgpu.Queue
	Instance() *wgpu.Instance
	Adapter() *wgpu.Adapter
	Surface() *wgpu.Surface
	SetDevice(device *wgpu.Device)
	SetQueue(queue *wgpu.Queue)
	SetInstance(instance *wgpu.Instance)
	SetAdapter(adapter *wgpu.Adapter)
	SetSurface(surface *wgpu.Surface)

	// ConfigureSurface is a wrapper for boilerplate logic required when calling ConfigureSurface on a surface.
	// This is required when the surface size changes, such as when the window is resized.
	//
	// Parameters:
	//   - width: the new width of the surface in pixels
	//   - height: the new height of the surface in pixels
	ConfigureSurface(width, height int)

	// SetPresentMode sets the surface present mode which controls how frames are delivered to the display.
	//
	// Parameters:
	//   - mode: the PresentMode to use (VSync, Uncapped, or TripleBuffered)
	SetPresentMode(mode PresentMode)

	// BeginComputeFrame creates a single command encoder for batching all compute dispatches
	// within a frame into one GPU submission. Must be paired with EndComputeFrame after all
	// DispatchCompute calls for the frame.
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginComputeFrame() error

	// EndComputeFrame finishes the batched compute command encoder and submits the resulting
	// command buffer to the GPU queue. Must be called after BeginComputeFrame and all
	// DispatchCompute calls for the frame.
	EndComputeFrame()

	// DispatchCompute encodes a compute pass within the current batched compute frame.
	// BeginComputeFrame must be called before any DispatchCompute calls.
	//
	// Parameters:
	//   - p: the cached Pipeline containing the compute pipeline to use for dispatching
	//   - computeProvider: the BindGroupProvider whose BindGroup will be set on the compute pass
	//   - workGroupCount: the number of workgroups to dispatch in the x, y, and z dimensions
	DispatchCompute(p pipeline.Pipeline, computeProvider bind_group_provider.BindGroupProvider, workGroupCount [3]uint32)

	// RegisterRenderPipeline is a high-level function that creates a render pipeline based on the provided pipeline.
	// It handles creating the shader module, pipeline layout, and render pipeline based on the pipeline's configuration.
	//
	// Parameters:
	//   - p: the pipeline object containing the source code and configuration for the pipeline
	//
	// Returns:
	//   - error: an error if the pipeline could not be created, otherwise nil
	RegisterRenderPipeline(p pipeline.Pipeline) error

	// RegisterComputePipeline is a high-level function that creates a compute pipeline based on the provided pipeline.
	// It handles creating the shader module and compute pipeline based on the pipeline's configuration.
	//
	// Parameters:
	//   - p: the pipeline object containing the source code and configuration for the pipeline
	//
	// Returns:
	//   - error: an error if the pipeline could not be created, otherwise nil
	RegisterComputePipeline(p pipeline.Pipeline) error

	// InitMeshBuffers inits the vertex and index buffers for a mesh based on the provided vertex and index data, and stores them on the given BindGroupProvider.
	//
	// Parameters:
	//   - provider: the BindGroupProvider to store the created vertex and index buffers on
	//   - vertexData: the raw vertex data bytes to upload to the GPU
	//   - indexData: the raw index data bytes to upload to the GPU
	//   - indexCount: the number of indices represented in the indexData, used for draw calls
	//
	// Returns:
	//   - error: an error if the buffers could not be created or initialized, otherwise nil
	InitMeshBuffers(provider bind_group_provider.BindGroupProvider, vertexData, indexData []byte, indexCount int) error

	// InitBindGroup is a high-level function that creates GPU buffers and a bind group based on a BindGroupProvider's layout entries.
	// It handles creating the necessary GPU resources and storing them back on the provider for later use.
	//
	// Parameters:
	//   - provider: the BindGroupProvider describing the layout entries and storage for the bind group
	//   - descriptor: the BindGroupLayoutDescriptor describing the layout of the bind group
	//   - bufferUsageOverrides: a map of binding indices to buffer usage flags, allowing customization of buffer usage
	//   - bufferSizeOverrides: a map of binding indices to buffer sizes, allowing customization of buffer sizes
	//
	// Returns:
	//   - error: an error if the bind group could not be initialized, otherwise nil
	InitBindGroup(provider bind_group_provider.BindGroupProvider, descriptor wgpu.BindGroupLayoutDescriptor, bufferUsageOverrides map[int]wgpu.BufferUsage, bufferSizeOverrides map[int]uint64) error

	// InitTextureView creates a GPU texture and texture view based on the provided staging data, and stores the view on the given BindGroupProvider.
	//
	// Parameters:
	//   - provider: the BindGroupProvider to store the created texture view on
	//   - bindingKey: the integer key identifying the bind group layout entry for this texture
	//   - stagingData: the TextureStagingData containing the raw texture data and metadata for creating the texture
	//
	// Returns:
	//   - error: an error if the texture view could not be created or initialized, otherwise nil
	InitTextureView(provider bind_group_provider.BindGroupProvider, bindingKey int, stagingData common.TextureStagingData) error

	// InitSampler creates a GPU sampler based on the provided staging data, and stores it on the given BindGroupProvider.
	//
	// Parameters:
	//   - provider: the BindGroupProvider to store the created sampler on
	//   - bindingKey: the integer key identifying the bind group layout entry for this sampler
	//   - stagingData: the SamplerStagingData containing the configuration for creating the sampler
	//
	// Returns:
	//   - error: an error if the sampler could not be created or initialized, otherwise nil
	InitSampler(provider bind_group_provider.BindGroupProvider, bindingKey int, samplerStagingData common.SamplerStagingData) error

	// WriteBuffers writes all staged buffer writes to the GPU queue.
	// Each BufferWrite targets a specific buffer on a BindGroupProvider at a given binding and offset.
	//
	// Parameters:
	//   - writes: a slice of BufferWrite structs describing the data to write
	WriteBuffers(writes []bind_group_provider.BufferWrite)

	// BeginFrame acquires the next swapchain texture, creates a command encoder, and begins
	// the main render pass. Must be paired with EndFrame after all DrawCall invocations.
	//
	// Returns:
	//   - error: an error if the swapchain texture could not be acquired
	BeginFrame() error

	// DrawCall encodes a single instanced draw command within the current render pass started by BeginFrame.
	// Multiple DrawCall invocations can be made between BeginFrame and EndFrame.
	//
	// Parameters:
	//   - p: the cached Pipeline containing the render pipeline to use
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - instanceCount: the number of instances to draw
	//   - bindGroups: a slice of BindGroupProviders whose BindGroups will be set on the render pass
	DrawCall(p pipeline.Pipeline, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider)

	// DrawCallIndirect encodes a single indirect instanced draw command within the current render pass.
	// The instance count is read from the indirectBuffer on the GPU, allowing the compute shader to
	// control how many instances are drawn without CPU readback.
	//
	// Parameters:
	//   - p: the cached Pipeline containing the render pipeline to use
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - indirectBuffer: the GPU buffer containing DrawIndexedIndirect arguments (20 bytes)
	//   - bindGroups: a slice of BindGroupProviders whose BindGroups will be set on the render pass
	DrawCallIndirect(p pipeline.Pipeline, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider)

	// EndFrame ends the current render pass and submits the command buffer to the GPU.
	// Does not present the surface — call Present() after EndFrame to display the frame.
	// Must be called after BeginFrame and all DrawCall invocations.
	EndFrame()

	// Present presents the surface to the display and releases the swapchain texture.
	// Must be called once per frame after EndFrame.
	Present()

	// CreateShadowDepthTexture creates a Depth32Float texture and view for shadow mapping.
	// The texture has sample count 1 (no MSAA) and can be sampled as a depth texture
	// in the lit fragment shader.
	//
	// Parameters:
	//   - width: shadow map width in texels
	//   - height: shadow map height in texels
	//
	// Returns:
	//   - *wgpu.TextureView: the depth texture view for the shadow render pass
	//   - *wgpu.Texture: the underlying texture (caller must release when done)
	//   - error: an error if texture creation fails
	CreateShadowDepthTexture(width, height int) (*wgpu.TextureView, *wgpu.Texture, error)

	// CreateComparisonSampler creates a comparison sampler suitable for PCF shadow mapping.
	// Uses CompareFunction Less for standard shadow depth comparison.
	//
	// Returns:
	//   - *wgpu.Sampler: the comparison sampler
	//   - error: an error if sampler creation fails
	CreateComparisonSampler() (*wgpu.Sampler, error)

	// RegisterShadowPipeline creates a depth-only render pipeline for shadow map generation.
	// Unlike RegisterRenderPipeline, shadow pipelines have no fragment shader, no color
	// target, sample count 1, Depth32Float format, front-face culling, and a depth bias.
	// The pipeline is stored on the Pipeline via SetRenderPipeline.
	//
	// Parameters:
	//   - p: the pipeline object containing the vertex shader and depth bias configuration
	//
	// Returns:
	//   - error: an error if pipeline creation fails
	RegisterShadowPipeline(p pipeline.Pipeline) error

	// BeginShadowFrame creates a command encoder for batching all shadow depth passes
	// within a frame. Must be paired with EndShadowFrame after all shadow passes.
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginShadowFrame() error

	// BeginShadowPass starts a depth-only render pass targeting the given shadow depth
	// texture view. Must be called between BeginShadowFrame and EndShadowFrame.
	//
	// Parameters:
	//   - depthView: the shadow map depth texture view to render into
	BeginShadowPass(depthView *wgpu.TextureView)

	// ShadowDrawCall encodes a single instanced draw command within the current shadow pass.
	//
	// Parameters:
	//   - p: the cached shadow Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - instanceCount: the number of instances to draw
	//   - bindGroups: bind group providers for the shadow pass (shadow uniform + instance buffer)
	ShadowDrawCall(p pipeline.Pipeline, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider)

	// ShadowDrawCallIndirect encodes a single indirect instanced draw command within the
	// current shadow pass. The instance count is read from the indirectBuffer on the GPU.
	//
	// Parameters:
	//   - p: the cached shadow Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - indirectBuffer: the GPU buffer containing DrawIndexedIndirect arguments
	//   - bindGroups: bind group providers for the shadow pass
	ShadowDrawCallIndirect(p pipeline.Pipeline, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider)

	// EndShadowPass ends the current shadow depth render pass.
	EndShadowPass()

	// EndShadowFrame finishes the shadow command encoder and submits to the GPU queue.
	EndShadowFrame()
}

var _ RendererBackend = &wgpuRendererBackendImpl{}

func newWGPURendererBackend(surfaceDescriptor *wgpu.SurfaceDescriptor, forceFallbackAdapter bool, sampleCount MSAASampleCount) wgpuRendererBackend {
	runtime.LockOSThread()
	w := &wgpuRendererBackendImpl{
		mu:          &sync.Mutex{},
		instance:    wgpu.CreateInstance(nil),
		presentMode: wgpu.PresentModeImmediate,
		sampleCount: sampleCount,
	}
	w.SetSurface(w.instance.CreateSurface(surfaceDescriptor))

	a, err := w.instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		ForceFallbackAdapter: forceFallbackAdapter,
		CompatibleSurface:    w.surface,
	})
	if err != nil {
		panic(err)
	}
	w.SetAdapter(a)

	// Start from the WebGPU spec default limits and raise MaxBindGroups to 8
	// so the lit fragment shader's 6 bind groups (0–5) are allowed.
	limits := wgpu.DefaultLimits()
	limits.MaxBindGroups = 8

	d, err := a.RequestDevice(&wgpu.DeviceDescriptor{
		Label: "Main Device",
		RequiredLimits: &wgpu.RequiredLimits{
			Limits: limits,
		},
	})
	if err != nil {
		panic(err)
	}
	w.SetDevice(d)
	w.SetQueue(d.GetQueue())

	return w
}

func (b *wgpuRendererBackendImpl) ConfigureSurface(width, height int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	capabilities := b.surface.GetCapabilities(b.adapter)
	b.surfaceFormat = &capabilities.Formats[0]

	b.surface.Configure(b.adapter, b.device, &wgpu.SurfaceConfiguration{
		Usage:       wgpu.TextureUsageRenderAttachment,
		Format:      *b.surfaceFormat,
		Width:       uint32(width),
		Height:      uint32(height),
		PresentMode: b.presentMode,
		AlphaMode:   capabilities.AlphaModes[0],
	})

	count := uint32(b.sampleCount)
	msaaEnabled := count > 1

	if msaaEnabled {
		// Create the MSAA texture that the render pass draws into; the resolved
		// result is written to the swapchain view as the ResolveTarget.
		msaaTexture, err := b.device.CreateTexture(&wgpu.TextureDescriptor{
			Label: "MSAA Texture",
			Size: wgpu.Extent3D{
				Width:              uint32(width),
				Height:             uint32(height),
				DepthOrArrayLayers: 1,
			},
			MipLevelCount: 1,
			SampleCount:   count,
			Dimension:     wgpu.TextureDimension2D,
			Format:        *b.surfaceFormat,
			Usage:         wgpu.TextureUsageRenderAttachment,
		})
		if err != nil {
			panic(err)
		}
		b.msaaTextureView, err = msaaTexture.CreateView(nil)
		if err != nil {
			panic(err)
		}
	} else {
		// No MSAA — the render pass draws directly to the swapchain view.
		b.msaaTextureView = nil
	}

	// Depth texture sample count must match the color attachment.
	depthTexture, err := b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "Depth Texture",
		Size: wgpu.Extent3D{
			Width:              uint32(width),
			Height:             uint32(height),
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   count,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatDepth24Plus,
		Usage:         wgpu.TextureUsageRenderAttachment,
	})
	if err != nil {
		panic(err)
	}
	b.depthTextureView, err = depthTexture.CreateView(nil)
	if err != nil {
		panic(err)
	}

	// Build the cached render pass descriptor for the main render target.
	// When MSAA is enabled, View is the MSAA texture and ResolveTarget is
	// set per-frame to the swapchain view. When disabled, View is set
	// per-frame to the swapchain view and ResolveTarget remains nil.
	storeOp := wgpu.StoreOpStore
	if msaaEnabled {
		storeOp = wgpu.StoreOpDiscard // Don't store MSAA data, just resolve
	}
	b.renderPassDescriptor = &wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{
			{
				View:          b.msaaTextureView, // nil when MSAA is off; set in BeginFrame
				ResolveTarget: nil,               // set per-frame when MSAA is on
				LoadOp:        wgpu.LoadOpClear,
				StoreOp:       storeOp,
				ClearValue: wgpu.Color{
					R: 0.1, G: 0.1, B: 0.1, A: 1.0,
				},
			},
		},
		DepthStencilAttachment: &wgpu.RenderPassDepthStencilAttachment{
			View:            b.depthTextureView, // Persistent until resize
			DepthLoadOp:     wgpu.LoadOpClear,
			DepthStoreOp:    wgpu.StoreOpDiscard, // Depth not needed after resolving
			DepthClearValue: 1.0,
		},
	}
}

func (b *wgpuRendererBackendImpl) SetPresentMode(mode PresentMode) {
	b.mu.Lock()
	defer b.mu.Unlock()

	switch mode {
	case PresentModeVSync:
		b.presentMode = wgpu.PresentModeFifo
	case PresentModeUncapped:
		fallthrough
	default:
		b.presentMode = wgpu.PresentModeImmediate
	}
}

func (b *wgpuRendererBackendImpl) BeginComputeFrame() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	b.computeFrameEncoder = encoder
	return nil
}

func (b *wgpuRendererBackendImpl) EndComputeFrame() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.computeFrameEncoder == nil {
		return
	}

	commandBuffer, err := b.computeFrameEncoder.Finish(nil)
	if err != nil {
		b.computeFrameEncoder.Release()
		b.computeFrameEncoder = nil
		return
	}

	b.queue.Submit(commandBuffer)
	commandBuffer.Release()
	b.computeFrameEncoder.Release()
	b.computeFrameEncoder = nil
}

func (b *wgpuRendererBackendImpl) DispatchCompute(
	p pipeline.Pipeline,
	computeProvider bind_group_provider.BindGroupProvider,
	workGroupCount [3]uint32,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.computeFrameEncoder == nil {
		return
	}

	computePipeline := p.Pipeline().(*wgpu.ComputePipeline)
	bindGroup := computeProvider.BindGroup()

	pass := b.computeFrameEncoder.BeginComputePass(nil)
	pass.SetPipeline(computePipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.DispatchWorkgroups(workGroupCount[0], workGroupCount[1], workGroupCount[2])
	pass.End()
}

func (b *wgpuRendererBackendImpl) RegisterRenderPipeline(p pipeline.Pipeline) error {
	if p.Shader(shader.ShaderTypeVertex) == nil || p.Shader(shader.ShaderTypeFragment) == nil {
		return errors.New("both vertex and fragment shaders must be set to create a render pipeline")
	}

	vertexShader := p.Shader(shader.ShaderTypeVertex)
	fragmentShader := p.Shader(shader.ShaderTypeFragment)

	vs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: vertexShader.Key(),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: vertexShader.Source(),
		},
	})
	if err != nil {
		return err
	}
	fs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: fragmentShader.Key(),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: fragmentShader.Source(),
		},
	})
	if err != nil {
		return err
	}

	merged := mergeBindGroupLayouts(vertexShader.BindGroupLayoutDescriptors(), fragmentShader.BindGroupLayoutDescriptors())
	maxGroup := -1
	for g := range merged {
		if g > maxGroup {
			maxGroup = g
		}
	}
	bindGroupLayouts := make([]*wgpu.BindGroupLayout, maxGroup+1)
	for g, desc := range merged {
		layout, layoutErr := b.device.CreateBindGroupLayout(&desc)
		if layoutErr != nil {
			return fmt.Errorf("failed to create bind group layout for group %d: %w", g, layoutErr)
		}
		bindGroupLayouts[g] = layout
	}

	pipelineLayout, err := b.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            p.PipelineKey(),
		BindGroupLayouts: bindGroupLayouts,
	})
	if err != nil {
		return err
	}

	vertexLayouts := make([]wgpu.VertexBufferLayout, 0, len(vertexShader.VertexLayouts()))
	for i := range vertexShader.VertexLayouts() {
		vertexLayouts = append(vertexLayouts, vertexShader.VertexLayout(i)...)
	}

	created, err := b.device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  p.PipelineKey() + " Render Pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     vs,
			EntryPoint: vertexShader.EntryPoint(),
			Buffers:    vertexLayouts,
		},
		Fragment: &wgpu.FragmentState{
			Module:     fs,
			EntryPoint: fragmentShader.EntryPoint(),
			Targets: []wgpu.ColorTargetState{
				func() wgpu.ColorTargetState {
					state := wgpu.ColorTargetState{
						Format:    *b.surfaceFormat,
						WriteMask: p.WriteMask(),
					}
					if p.BlendEnabled() {
						state.Blend = p.BlendState()
					}
					return state
				}(),
			},
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  p.Topology(),
			FrontFace: p.FrontFace(),
			CullMode:  p.CullMode(),
		},
		Multisample: wgpu.MultisampleState{
			Count: uint32(b.sampleCount),
			Mask:  0xFFFFFFFF,
		},
		DepthStencil: func() *wgpu.DepthStencilState {
			depthCompare := wgpu.CompareFunctionLess
			if !p.DepthTestEnabled() {
				depthCompare = wgpu.CompareFunctionAlways
			}
			return &wgpu.DepthStencilState{
				Format:              wgpu.TextureFormatDepth24Plus,
				DepthWriteEnabled:   p.DepthWriteEnabled(),
				DepthCompare:        depthCompare,
				DepthBias:           p.DepthBias(),
				DepthBiasSlopeScale: p.DepthBiasSlopeScale(),
				StencilFront: wgpu.StencilFaceState{
					Compare: wgpu.CompareFunctionAlways,
				},
				StencilBack: wgpu.StencilFaceState{
					Compare: wgpu.CompareFunctionAlways,
				},
			}
		}(),
	})
	if err != nil {
		return err
	}

	p.SetRenderPipeline(created)

	return nil
}

func (b *wgpuRendererBackendImpl) RegisterComputePipeline(p pipeline.Pipeline) error {
	if p.Shader(shader.ShaderTypeCompute) == nil {
		return errors.New("compute shader must be set to create a compute pipeline")
	}

	computeShader := p.Shader(shader.ShaderTypeCompute)
	s, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: computeShader.Key(),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: computeShader.Source(),
		},
	})
	if err != nil {
		return err
	}

	descriptors := computeShader.BindGroupLayoutDescriptors()
	maxGroup := -1
	for g := range descriptors {
		if g > maxGroup {
			maxGroup = g
		}
	}
	bindGroupLayouts := make([]*wgpu.BindGroupLayout, maxGroup+1)
	for g, desc := range descriptors {
		bgl, bglErr := b.device.CreateBindGroupLayout(&desc)
		if bglErr != nil {
			return fmt.Errorf("failed to create bind group layout for group %d: %w", g, bglErr)
		}
		bindGroupLayouts[g] = bgl
	}

	layout, err := b.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            p.PipelineKey(),
		BindGroupLayouts: bindGroupLayouts,
	})
	if err != nil {
		return err
	}

	created, err := b.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:  p.PipelineKey() + " Compute Pipeline",
		Layout: layout,
		Compute: wgpu.ProgrammableStageDescriptor{
			Module:     s,
			EntryPoint: computeShader.EntryPoint(),
		},
	})
	if err != nil {
		return err
	}

	p.SetComputePipeline(created)

	return nil
}

func (b *wgpuRendererBackendImpl) InitMeshBuffers(provider bind_group_provider.BindGroupProvider, vertexData, indexData []byte, indexCount int) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if len(vertexData) > 0 {
		buf, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label:            provider.Label() + " Vertex Buffer",
			Size:             uint64(len(vertexData)),
			Usage:            wgpu.BufferUsageVertex | wgpu.BufferUsageCopyDst,
			MappedAtCreation: false,
		})
		if err != nil {
			return err
		}
		b.queue.WriteBuffer(buf, 0, vertexData)
		provider.SetVertexBuffer(buf)
	}

	if len(indexData) > 0 {
		buf, err := b.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label:            provider.Label() + " Index Buffer",
			Size:             uint64(len(indexData)),
			Usage:            wgpu.BufferUsageIndex | wgpu.BufferUsageCopyDst,
			MappedAtCreation: false,
		})
		if err != nil {
			return err
		}
		b.queue.WriteBuffer(buf, 0, indexData)
		provider.SetIndexBuffer(buf)
	}

	provider.SetIndexCount(indexCount)

	return nil
}

func (b *wgpuRendererBackendImpl) InitBindGroup(provider bind_group_provider.BindGroupProvider, descriptor wgpu.BindGroupLayoutDescriptor, bufferUsageOverrides map[int]wgpu.BufferUsage, bufferSizeOverrides map[int]uint64) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if len(descriptor.Entries) == 0 {
		return nil
	}

	layout := provider.BindGroupLayout()
	if layout == nil {
		var err error
		layout, err = b.device.CreateBindGroupLayout(&descriptor)
		if err != nil {
			return err
		}
		provider.SetBindGroupLayout(layout)
	}

	bindGroupEntries := make([]wgpu.BindGroupEntry, len(descriptor.Entries))
	for i, entry := range descriptor.Entries {
		binding := int(entry.Binding)

		isTexture := entry.Texture.SampleType != wgpu.TextureSampleTypeUndefined
		isSampler := entry.Sampler.Type != wgpu.SamplerBindingTypeUndefined

		if isTexture {
			tv := provider.TextureView(binding)
			if tv == nil {
				return fmt.Errorf("texture binding %d has no texture view — call InitTextureView first", binding)
			}
			bindGroupEntries[i] = wgpu.BindGroupEntry{
				Binding:     entry.Binding,
				TextureView: tv,
			}
		} else if isSampler {
			samp := provider.Sampler(binding)
			if samp == nil {
				return fmt.Errorf("sampler binding %d has no sampler — call InitSampler first", binding)
			}
			bindGroupEntries[i] = wgpu.BindGroupEntry{
				Binding: entry.Binding,
				Sampler: samp,
			}
		} else {
			// Buffer binding — create if not already present
			var usage wgpu.BufferUsage
			switch entry.Buffer.Type {
			case wgpu.BufferBindingTypeUniform:
				usage = wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst
			case wgpu.BufferBindingTypeStorage:
				usage = wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst
			case wgpu.BufferBindingTypeReadOnlyStorage:
				usage = wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst
			}
			if overrideUsage, ok := bufferUsageOverrides[binding]; ok {
				usage |= overrideUsage
			}

			buf := provider.Buffer(binding)
			if buf == nil {
				var bufErr error
				bufSize := entry.Buffer.MinBindingSize
				if overrideSize, ok := bufferSizeOverrides[binding]; ok {
					bufSize = overrideSize
				}
				buf, bufErr = b.device.CreateBuffer(&wgpu.BufferDescriptor{
					Label: provider.Label() + " Buffer",
					Size:  bufSize,
					Usage: usage,
				})
				if bufErr != nil {
					return bufErr
				}
				provider.SetBuffer(binding, buf)
			}
			bindGroupEntries[i] = wgpu.BindGroupEntry{
				Binding: entry.Binding,
				Buffer:  buf,
				Offset:  0,
				Size:    wgpu.WholeSize,
			}
		}
	}

	bindGroup, err := b.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:   provider.Label() + " Bind Group",
		Layout:  layout,
		Entries: bindGroupEntries,
	})
	if err != nil {
		return err
	}
	provider.SetBindGroup(bindGroup)

	return nil
}

func (b *wgpuRendererBackendImpl) InitTextureView(provider bind_group_provider.BindGroupProvider, bindingKey int, stagingData common.TextureStagingData) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	tex, err := b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label:     provider.Label() + " Texture",
		Usage:     wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
		Dimension: wgpu.TextureDimension2D,
		Size: wgpu.Extent3D{
			Width:              stagingData.Width,
			Height:             stagingData.Height,
			DepthOrArrayLayers: 1,
		},
		Format:        wgpu.TextureFormatRGBA8UnormSrgb,
		MipLevelCount: 1,
		SampleCount:   1,
	})
	if err != nil {
		return err
	}

	b.queue.WriteTexture(
		&wgpu.ImageCopyTexture{
			Texture:  tex,
			MipLevel: 0,
			Origin:   wgpu.Origin3D{},
			Aspect:   wgpu.TextureAspectAll,
		},
		stagingData.Pixels,
		&wgpu.TextureDataLayout{
			Offset:       0,
			BytesPerRow:  stagingData.Width * 4,
			RowsPerImage: stagingData.Height,
		},
		&wgpu.Extent3D{
			Width:              stagingData.Width,
			Height:             stagingData.Height,
			DepthOrArrayLayers: 1,
		},
	)

	view, err := tex.CreateView(nil)
	if err != nil {
		return err
	}
	provider.SetTextureView(bindingKey, view)

	return nil
}

func (b *wgpuRendererBackendImpl) InitSampler(provider bind_group_provider.BindGroupProvider, bindingKey int, samplerStagingData common.SamplerStagingData) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	samp, err := b.device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         provider.Label() + " Sampler",
		AddressModeU:  common.Coalesce(samplerStagingData.AddressModeU, wgpu.AddressModeRepeat),
		AddressModeV:  common.Coalesce(samplerStagingData.AddressModeV, wgpu.AddressModeRepeat),
		AddressModeW:  common.Coalesce(samplerStagingData.AddressModeW, wgpu.AddressModeRepeat),
		MagFilter:     common.Coalesce(samplerStagingData.MagFilter, wgpu.FilterModeLinear),
		MinFilter:     common.Coalesce(samplerStagingData.MinFilter, wgpu.FilterModeLinear),
		MipmapFilter:  common.Coalesce(samplerStagingData.MipmapFilter, wgpu.MipmapFilterModeLinear),
		LodMinClamp:   common.Coalesce(samplerStagingData.LodMinClamp, 0.0),
		LodMaxClamp:   common.Coalesce(samplerStagingData.LodMaxClamp, 32.0),
		MaxAnisotropy: common.Coalesce(samplerStagingData.MaxAnisotropy, 1),
		Compare:       samplerStagingData.Compare,
	})
	if err != nil {
		return err
	}
	provider.SetSampler(bindingKey, samp)

	return nil
}

func (b *wgpuRendererBackendImpl) WriteBuffers(writes []bind_group_provider.BufferWrite) {
	b.mu.Lock()
	defer b.mu.Unlock()

	for _, w := range writes {
		buf := w.Provider.Buffer(w.Binding)
		if buf == nil {
			continue
		}
		b.queue.WriteBuffer(buf, w.Offset, w.Data)
	}
}

func (b *wgpuRendererBackendImpl) BeginFrame() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Defensive: if a previous frame's surface texture is still held, avoid
	// attempting to acquire another one. This prevents wgpu-native validation
	// errors like "Surface image is already acquired" when frames overlap.
	if b.frameSurface != nil {
		return fmt.Errorf("previous frame surface not yet presented")
	}

	surfaceTexture, err := b.surface.GetCurrentTexture()
	if err != nil {
		return err
	}

	view, err := surfaceTexture.CreateView(nil)
	if err != nil {
		surfaceTexture.Release()
		return err
	}

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		view.Release()
		surfaceTexture.Release()
		return err
	}

	// When MSAA is enabled, the MSAA texture is the color attachment View and
	// the swapchain view is the ResolveTarget. When MSAA is off, the swapchain
	// view is the color attachment View directly and ResolveTarget is nil.
	if b.sampleCount > 1 {
		b.renderPassDescriptor.ColorAttachments[0].ResolveTarget = view
	} else {
		b.renderPassDescriptor.ColorAttachments[0].View = view
	}
	pass := encoder.BeginRenderPass(b.renderPassDescriptor)

	b.frameEncoder = encoder
	b.framePass = pass
	b.frameSurface = surfaceTexture
	b.frameView = view

	return nil
}

func (b *wgpuRendererBackendImpl) DrawCall(
	p pipeline.Pipeline,
	meshProvider bind_group_provider.BindGroupProvider,
	instanceCount uint32,
	bindGroups []bind_group_provider.BindGroupProvider,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	renderPipeline := p.Pipeline().(*wgpu.RenderPipeline)
	b.framePass.SetPipeline(renderPipeline)

	for i, bg := range bindGroups {
		b.framePass.SetBindGroup(uint32(i), bg.BindGroup(), nil)
	}

	b.framePass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0, wgpu.WholeSize)
	b.framePass.SetIndexBuffer(meshProvider.IndexBuffer(), wgpu.IndexFormatUint32, 0, wgpu.WholeSize)
	b.framePass.DrawIndexed(uint32(meshProvider.IndexCount()), instanceCount, 0, 0, 0)
}

func (b *wgpuRendererBackendImpl) DrawCallIndirect(
	p pipeline.Pipeline,
	meshProvider bind_group_provider.BindGroupProvider,
	indirectBuffer *wgpu.Buffer,
	bindGroups []bind_group_provider.BindGroupProvider,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	renderPipeline := p.Pipeline().(*wgpu.RenderPipeline)
	b.framePass.SetPipeline(renderPipeline)

	for i, bg := range bindGroups {
		b.framePass.SetBindGroup(uint32(i), bg.BindGroup(), nil)
	}

	b.framePass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0, wgpu.WholeSize)
	b.framePass.SetIndexBuffer(meshProvider.IndexBuffer(), wgpu.IndexFormatUint32, 0, wgpu.WholeSize)
	b.framePass.DrawIndexedIndirect(indirectBuffer, 0)
}

func (b *wgpuRendererBackendImpl) EndFrame() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.framePass.End()

	commandBuffer, err := b.frameEncoder.Finish(nil)
	if err != nil {
		b.frameEncoder.Release()
		b.frameView.Release()
		b.frameSurface.Release()
		b.frameEncoder = nil
		b.framePass = nil
		b.frameSurface = nil
		b.frameView = nil
		return
	}

	b.queue.Submit(commandBuffer)

	commandBuffer.Release()
	b.frameEncoder.Release()
	b.frameEncoder = nil
	b.framePass = nil
}

func (b *wgpuRendererBackendImpl) Present() {
	b.mu.Lock()
	defer b.mu.Unlock()

	// If no frame surface is held, nothing to present.
	if b.frameSurface == nil {
		return
	}

	// Present the acquired surface image and release local references.
	b.surface.Present()

	if b.frameView != nil {
		b.frameView.Release()
		b.frameView = nil
	}
	if b.frameSurface != nil {
		b.frameSurface.Release()
		b.frameSurface = nil
	}
}

func (b *wgpuRendererBackendImpl) Device() *wgpu.Device {
	return b.device
}

func (b *wgpuRendererBackendImpl) Queue() *wgpu.Queue {
	return b.queue
}

func (b *wgpuRendererBackendImpl) Instance() *wgpu.Instance {
	return b.instance
}

func (b *wgpuRendererBackendImpl) Adapter() *wgpu.Adapter {
	return b.adapter
}

func (b *wgpuRendererBackendImpl) Surface() *wgpu.Surface {
	return b.surface
}

func (b *wgpuRendererBackendImpl) SetDevice(device *wgpu.Device) {
	b.device = device
}

func (b *wgpuRendererBackendImpl) SetQueue(queue *wgpu.Queue) {
	b.queue = queue
}

func (b *wgpuRendererBackendImpl) SetInstance(instance *wgpu.Instance) {
	b.instance = instance
}

func (b *wgpuRendererBackendImpl) SetAdapter(adapter *wgpu.Adapter) {
	b.adapter = adapter
}

func (b *wgpuRendererBackendImpl) SetSurface(surface *wgpu.Surface) {
	b.surface = surface
}

func (b *wgpuRendererBackendImpl) CreateShadowDepthTexture(width, height int) (*wgpu.TextureView, *wgpu.Texture, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	tex, err := b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "Shadow Depth Texture",
		Size: wgpu.Extent3D{
			Width:              uint32(width),
			Height:             uint32(height),
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     wgpu.TextureDimension2D,
		Format:        wgpu.TextureFormatDepth32Float,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create shadow depth texture: %w", err)
	}

	view, err := tex.CreateView(nil)
	if err != nil {
		tex.Release()
		return nil, nil, fmt.Errorf("failed to create shadow depth texture view: %w", err)
	}

	return view, tex, nil
}

func (b *wgpuRendererBackendImpl) CreateComparisonSampler() (*wgpu.Sampler, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	samp, err := b.device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:         "Shadow Comparison Sampler",
		AddressModeU:  wgpu.AddressModeClampToEdge,
		AddressModeV:  wgpu.AddressModeClampToEdge,
		AddressModeW:  wgpu.AddressModeClampToEdge,
		MagFilter:     wgpu.FilterModeLinear,
		MinFilter:     wgpu.FilterModeLinear,
		MipmapFilter:  wgpu.MipmapFilterModeNearest,
		Compare:       wgpu.CompareFunctionLess,
		MaxAnisotropy: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create comparison sampler: %w", err)
	}

	return samp, nil
}

func (b *wgpuRendererBackendImpl) RegisterShadowPipeline(p pipeline.Pipeline) error {
	vertexShader := p.Shader(shader.ShaderTypeVertex)
	if vertexShader == nil {
		return errors.New("vertex shader must be set to create a shadow pipeline")
	}

	vs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: vertexShader.Key(),
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: vertexShader.Source(),
		},
	})
	if err != nil {
		return err
	}

	descriptors := vertexShader.BindGroupLayoutDescriptors()
	maxGroup := -1
	for g := range descriptors {
		if g > maxGroup {
			maxGroup = g
		}
	}
	bindGroupLayouts := make([]*wgpu.BindGroupLayout, maxGroup+1)
	for g, desc := range descriptors {
		layout, layoutErr := b.device.CreateBindGroupLayout(&desc)
		if layoutErr != nil {
			return fmt.Errorf("shadow: failed to create bind group layout for group %d: %w", g, layoutErr)
		}
		bindGroupLayouts[g] = layout
	}

	pipelineLayout, err := b.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            p.PipelineKey(),
		BindGroupLayouts: bindGroupLayouts,
	})
	if err != nil {
		return err
	}

	vertexLayouts := make([]wgpu.VertexBufferLayout, 0, len(vertexShader.VertexLayouts()))
	for i := range vertexShader.VertexLayouts() {
		vertexLayouts = append(vertexLayouts, vertexShader.VertexLayout(i)...)
	}

	created, err := b.device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  p.PipelineKey() + " Shadow Pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     vs,
			EntryPoint: vertexShader.EntryPoint(),
			Buffers:    vertexLayouts,
		},
		// No fragment shader — depth-only pass
		Fragment: nil,
		Primitive: wgpu.PrimitiveState{
			Topology:  wgpu.PrimitiveTopologyTriangleList,
			FrontFace: wgpu.FrontFaceCCW,
			CullMode:  p.CullMode(), // Configurable: CullModeFront for closed meshes, CullModeNone for flat geometry
		},
		Multisample: wgpu.MultisampleState{
			Count: 1, // No MSAA for shadow maps
			Mask:  0xFFFFFFFF,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:              wgpu.TextureFormatDepth32Float,
			DepthWriteEnabled:   true,
			DepthCompare:        wgpu.CompareFunctionLess,
			DepthBias:           p.DepthBias(),
			DepthBiasSlopeScale: p.DepthBiasSlopeScale(),
			StencilFront: wgpu.StencilFaceState{
				Compare: wgpu.CompareFunctionAlways,
			},
			StencilBack: wgpu.StencilFaceState{
				Compare: wgpu.CompareFunctionAlways,
			},
		},
	})
	if err != nil {
		return err
	}

	p.SetRenderPipeline(created)
	return nil
}

func (b *wgpuRendererBackendImpl) BeginShadowFrame() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	b.shadowFrameEncoder = encoder
	return nil
}

func (b *wgpuRendererBackendImpl) BeginShadowPass(depthView *wgpu.TextureView) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.shadowFrameEncoder == nil {
		return
	}

	pass := b.shadowFrameEncoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		// No color attachments — depth-only pass
		ColorAttachments: nil,
		DepthStencilAttachment: &wgpu.RenderPassDepthStencilAttachment{
			View:            depthView,
			DepthLoadOp:     wgpu.LoadOpClear,
			DepthStoreOp:    wgpu.StoreOpStore, // Must store — this is the shadow map
			DepthClearValue: 1.0,
		},
	})
	b.shadowPass = pass
}

func (b *wgpuRendererBackendImpl) ShadowDrawCall(
	p pipeline.Pipeline,
	meshProvider bind_group_provider.BindGroupProvider,
	instanceCount uint32,
	bindGroups []bind_group_provider.BindGroupProvider,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.shadowPass == nil {
		return
	}

	renderPipeline := p.Pipeline().(*wgpu.RenderPipeline)
	b.shadowPass.SetPipeline(renderPipeline)

	for i, bg := range bindGroups {
		b.shadowPass.SetBindGroup(uint32(i), bg.BindGroup(), nil)
	}

	b.shadowPass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0, wgpu.WholeSize)
	b.shadowPass.SetIndexBuffer(meshProvider.IndexBuffer(), wgpu.IndexFormatUint32, 0, wgpu.WholeSize)
	b.shadowPass.DrawIndexed(uint32(meshProvider.IndexCount()), instanceCount, 0, 0, 0)
}

func (b *wgpuRendererBackendImpl) ShadowDrawCallIndirect(
	p pipeline.Pipeline,
	meshProvider bind_group_provider.BindGroupProvider,
	indirectBuffer *wgpu.Buffer,
	bindGroups []bind_group_provider.BindGroupProvider,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.shadowPass == nil {
		return
	}

	renderPipeline := p.Pipeline().(*wgpu.RenderPipeline)
	b.shadowPass.SetPipeline(renderPipeline)

	for i, bg := range bindGroups {
		b.shadowPass.SetBindGroup(uint32(i), bg.BindGroup(), nil)
	}

	b.shadowPass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0, wgpu.WholeSize)
	b.shadowPass.SetIndexBuffer(meshProvider.IndexBuffer(), wgpu.IndexFormatUint32, 0, wgpu.WholeSize)
	b.shadowPass.DrawIndexedIndirect(indirectBuffer, 0)
}

func (b *wgpuRendererBackendImpl) EndShadowPass() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.shadowPass == nil {
		return
	}

	b.shadowPass.End()
	b.shadowPass = nil
}

func (b *wgpuRendererBackendImpl) EndShadowFrame() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.shadowFrameEncoder == nil {
		return
	}

	commandBuffer, err := b.shadowFrameEncoder.Finish(nil)
	if err != nil {
		b.shadowFrameEncoder.Release()
		b.shadowFrameEncoder = nil
		return
	}

	b.queue.Submit(commandBuffer)
	commandBuffer.Release()
	b.shadowFrameEncoder.Release()
	b.shadowFrameEncoder = nil
}

// mergeBindGroupLayouts merges the bind group layout descriptors from a vertex and fragment shader
// into a unified set of descriptors suitable for a render pipeline layout.
//
// For each group index present in either shader:
//   - Entries with the same binding number have their Visibility flags ORed together
//   - Entries unique to one shader are included with their original visibility
//
// Parameters:
//   - vertexLayouts: bind group layout descriptors from the vertex shader
//   - fragmentLayouts: bind group layout descriptors from the fragment shader
//
// Returns:
//   - map[int]wgpu.BindGroupLayoutDescriptor: the merged descriptors keyed by group index
func mergeBindGroupLayouts(
	vertexLayouts, fragmentLayouts map[int]wgpu.BindGroupLayoutDescriptor,
) map[int]wgpu.BindGroupLayoutDescriptor {
	merged := make(map[int]wgpu.BindGroupLayoutDescriptor)

	// collect all group indices from both maps
	groupIndices := make(map[int]bool)
	for g := range vertexLayouts {
		groupIndices[g] = true
	}
	for g := range fragmentLayouts {
		groupIndices[g] = true
	}

	for g := range groupIndices {
		vDesc, hasV := vertexLayouts[g]
		fDesc, hasF := fragmentLayouts[g]

		switch {
		case hasV && !hasF:
			// group only in vertex shader — use as-is
			merged[g] = vDesc
		case hasF && !hasV:
			// group only in fragment shader — use as-is
			merged[g] = fDesc
		default:
			// group in both — merge entries by binding number
			entryMap := make(map[uint32]wgpu.BindGroupLayoutEntry)
			for _, e := range vDesc.Entries {
				entryMap[e.Binding] = e
			}
			for _, e := range fDesc.Entries {
				if existing, ok := entryMap[e.Binding]; ok {
					// same binding in both stages — OR the visibility
					existing.Visibility |= e.Visibility
					entryMap[e.Binding] = existing
				} else {
					entryMap[e.Binding] = e
				}
			}

			// flatten back to a sorted slice
			entries := make([]wgpu.BindGroupLayoutEntry, 0, len(entryMap))
			for _, e := range entryMap {
				entries = append(entries, e)
			}
			// sort by binding for deterministic layout
			sort.Slice(entries, func(i, j int) bool {
				return entries[i].Binding < entries[j].Binding
			})

			merged[g] = wgpu.BindGroupLayoutDescriptor{
				Label:   vDesc.Label, // or generate a composite label
				Entries: entries,
			}
		}
	}

	return merged
}
