package renderer

import (
	"fmt"
	"sync"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
	"github.com/cogentcore/webgpu/wgpu"
)

// renderer is the implementation of the Renderer interface.
type renderer struct {
	mu *sync.Mutex

	pipelineCache map[string]pipeline.Pipeline

	backendType RendererBackendType
	backend     RendererBackend

	// Pre-creation config collected from builder options
	forceFallbackAdapter bool
	pendingPresentMode   *PresentMode
	pendingMSAA          *MSAASampleCount
}

// Renderer defines the interface for the rendering system.
//
// This is a high-level API designed to simplify rendering tasks into a streamlined and idiomatic flow.
// The Renderer manages a cache of shaders and pipelines, allowing for easy retrieval and management of these resources.
// The Renderer also implements a backend which allows for multiple backend API implementations to exist.
type Renderer interface {
	// Pipeline retrieves the cached Pipeline associated with the given key.
	// If the Pipeline does not exist, this will return nil.
	//
	// Parameters:
	//   - key: the unique identifier for the Pipeline to retrieve
	//
	// Returns:
	//   - pipeline.Pipeline: the Pipeline associated with the key, or nil if not found
	Pipeline(key string) pipeline.Pipeline

	// Pipelines retrieves the entire cache of Pipelines.
	//
	// Returns:
	//   - map[string]pipeline.Pipeline: a map of pipeline keys to their corresponding Pipeline objects
	Pipelines() map[string]pipeline.Pipeline

	// RegisterPipelines registers one or more pipelines by creating the corresponding GPU
	// pipeline objects (render or compute) via the backend, then caching them by PipelineKey.
	// Pipelines whose keys are already registered are skipped to avoid duplicate GPU resource creation.
	//
	// Parameters:
	//   - pipelines: the Pipelines to register
	//
	// Returns:
	//   - error: an error if pipeline creation fails
	RegisterPipelines(pipelines ...pipeline.Pipeline) error

	// SetPipeline adds or updates a Pipeline in the cache with the given key.
	//
	// Parameters:
	//   - key: the unique identifier for the Pipeline to add or update in the cache
	//   - p: the Pipeline to add or update in the cache
	SetPipeline(key string, p pipeline.Pipeline)

	// SetPipelines replaces the entire pipeline cache with the provided map of Pipelines.
	//
	// Parameters:
	//   - pipelines: a map of pipeline keys to their corresponding Pipeline objects to set as the new cache
	SetPipelines(pipelines map[string]pipeline.Pipeline)

	// Resize configures the underlying backend to handle a new surface size.
	// This should be called when re-sizing the window or when the surface size should change.
	//
	// Parameters:
	//   - width: the new width of the surface in pixels
	//   - height: the new height of the surface in pixels
	Resize(width, height int)

	// InitMeshBuffers creates GPU vertex and index buffers from raw byte data and stores them
	// on the given BindGroupProvider for later use in draw calls.
	//
	// Parameters:
	//   - provider: the BindGroupProvider to store the created buffers on
	//   - vertexData: the raw vertex data bytes to upload to the GPU
	//   - indexData: the raw index data bytes to upload to the GPU
	//   - indexCount: the number of indices, used for draw calls
	//
	// Returns:
	//   - error: an error if buffer creation fails
	InitMeshBuffers(provider bind_group_provider.BindGroupProvider, vertexData, indexData []byte, indexCount int) error

	// InitBindGroup creates GPU buffers and a bind group from a layout descriptor and stores them
	// on the given BindGroupProvider. Textures and samplers must be initialized via InitTextureView
	// and InitSampler before calling this method. Buffer usage and size can be overridden per binding.
	//
	// Parameters:
	//   - provider: the BindGroupProvider to store the created bind group on
	//   - descriptor: the layout descriptor defining the bind group entries
	//   - bufferUsageOverrides: additional buffer usage flags to OR into the derived usage, keyed by binding index (nil safe)
	//   - bufferSizeOverrides: custom buffer sizes to use instead of MinBindingSize, keyed by binding index (nil safe)
	//
	// Returns:
	//   - error: an error if bind group creation fails
	InitBindGroup(provider bind_group_provider.BindGroupProvider, descriptor wgpu.BindGroupLayoutDescriptor, bufferUsageOverrides map[int]wgpu.BufferUsage, bufferSizeOverrides map[int]uint64) error

	// InitTextureView creates a GPU texture from staging data and stores the resulting texture view
	// on the given BindGroupProvider at the specified binding index. Must be called before InitBindGroup
	// for any texture bindings.
	//
	// Parameters:
	//   - provider: the BindGroupProvider to store the created texture view on
	//   - bindingKey: the binding index for this texture
	//   - stagingData: the pixel data and dimensions for the texture
	//
	// Returns:
	//   - error: an error if texture creation fails
	InitTextureView(provider bind_group_provider.BindGroupProvider, bindingKey int, stagingData common.TextureStagingData) error

	// InitSampler creates a GPU sampler from staging data and stores it on the given BindGroupProvider
	// at the specified binding index. Must be called before InitBindGroup for any sampler bindings.
	//
	// Parameters:
	//   - provider: the BindGroupProvider to store the created sampler on
	//   - bindingKey: the binding index for this sampler
	//   - samplerStagingData: the sampler configuration
	//
	// Returns:
	//   - error: an error if sampler creation fails
	InitSampler(provider bind_group_provider.BindGroupProvider, bindingKey int, samplerStagingData common.SamplerStagingData) error

	// WriteBuffers writes all staged buffer writes to the GPU queue.
	// Each BufferWrite targets a specific buffer on a BindGroupProvider at a given binding and offset.
	//
	// Parameters:
	//   - writes: a slice of BufferWrite structs describing the data to write
	WriteBuffers(writes []bind_group_provider.BufferWrite)

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

	// DispatchCompute looks up the cached compute Pipeline by key, then encodes a compute pass
	// within the current batched compute frame started by BeginComputeFrame.
	//
	// Parameters:
	//   - pipelineKey: the unique identifier for the cached compute Pipeline to use
	//   - computeProvider: the BindGroupProvider whose BindGroup will be set on the compute pass
	//   - workGroupCount: the number of workgroups to dispatch in the x, y, and z dimensions
	DispatchCompute(pipelineKey string, computeProvider bind_group_provider.BindGroupProvider, workGroupCount [3]uint32)

	// BeginFrame acquires the swapchain texture and begins the main render pass.
	// Must be paired with EndFrame after all DrawCall invocations within a single frame.
	//
	// Returns:
	//   - error: an error if the swapchain texture could not be acquired
	BeginFrame() error

	// DrawCall encodes a single instanced draw command within the current render pass.
	// Multiple DrawCall invocations can be made between BeginFrame and EndFrame.
	//
	// Parameters:
	//   - pipelineKey: the unique identifier for the cached render Pipeline to use
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - instanceCount: the number of instances to draw
	//   - bindGroups: a slice of BindGroupProviders whose BindGroups will be set on the render pass
	//
	// Returns:
	//   - error: an error if the pipeline is not found
	DrawCall(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider) error

	// DrawCallIndirect encodes a single indirect instanced draw command within the current render pass.
	// The instance count is read from the indirectBuffer on the GPU, allowing the compute shader to
	// control how many instances are drawn without CPU readback.
	//
	// Parameters:
	//   - pipelineKey: the unique identifier for the cached render Pipeline to use
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - indirectBuffer: the GPU buffer containing DrawIndexedIndirect arguments (20 bytes)
	//   - bindGroups: a slice of BindGroupProviders whose BindGroups will be set on the render pass
	//
	// Returns:
	//   - error: an error if the pipeline is not found
	DrawCallIndirect(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider) error

	// EndFrame ends the current render pass and submits the command buffer to the GPU.
	// Does not present the surface — call Present() after EndFrame to display the frame.
	// Must be called after BeginFrame and all DrawCall invocations within a single frame.
	EndFrame()

	// Present presents the surface to the display and releases the swapchain texture.
	// Must be called once per frame after EndFrame.
	Present()

	// SetPresentMode sets the surface present mode which controls how frames are delivered to the display.
	// A call to Resize is required after changing this for the new mode to take effect.
	//
	// Parameters:
	//   - mode: the PresentMode to use (VSync, Uncapped, or TripleBuffered)
	SetPresentMode(mode PresentMode)

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
	//
	// Returns:
	//   - *wgpu.Sampler: the comparison sampler
	//   - error: an error if sampler creation fails
	CreateComparisonSampler() (*wgpu.Sampler, error)

	// RegisterShadowPipeline registers a depth-only render pipeline for shadow map generation.
	// Uses no fragment shader, sample count 1, Depth32Float format, and front-face culling.
	//
	// Parameters:
	//   - p: the pipeline object containing the shadow vertex shader
	//
	// Returns:
	//   - error: an error if pipeline creation fails
	RegisterShadowPipeline(p pipeline.Pipeline) error

	// BeginShadowFrame creates a command encoder for batching shadow depth passes.
	// Must be paired with EndShadowFrame.
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginShadowFrame() error

	// BeginShadowPass starts a depth-only render pass targeting the given shadow depth view.
	//
	// Parameters:
	//   - depthView: the shadow map depth texture view to render into
	BeginShadowPass(depthView *wgpu.TextureView)

	// ShadowDrawCall encodes a single instanced draw command within the current shadow pass.
	//
	// Parameters:
	//   - pipelineKey: the unique identifier for the cached shadow Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - instanceCount: the number of instances to draw
	//   - bindGroups: bind group providers for the shadow pass
	//
	// Returns:
	//   - error: an error if the pipeline is not found
	ShadowDrawCall(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider) error

	// ShadowDrawCallIndirect encodes a single indirect instanced draw command within the
	// current shadow pass. The instance count is read from the indirectBuffer on the GPU.
	//
	// Parameters:
	//   - pipelineKey: the unique identifier for the cached shadow Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - indirectBuffer: the GPU buffer containing DrawIndexedIndirect arguments
	//   - bindGroups: bind group providers for the shadow pass
	//
	// Returns:
	//   - error: an error if the pipeline is not found
	ShadowDrawCallIndirect(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider) error

	// EndShadowPass ends the current shadow depth render pass.
	EndShadowPass()

	// EndShadowFrame finishes the shadow command encoder and submits to the GPU queue.
	EndShadowFrame()
}

var _ Renderer = &renderer{}

// NewRenderer creates a new Renderer instance with the specified backend type and surface descriptor.
// The surface descriptor is platform-specific and is typically obtained from Window.GetSurfaceDescriptor().
//
// Parameters:
//   - backendType: the type of rendering backend to use (e.g., WGPU)
//   - surfaceDescriptor: the platform-specific surface descriptor for WebGPU surface creation
//   - options: variadic list of RendererBuilderOption functions to configure the Renderer
//
// Returns:
//   - Renderer: a new instance of Renderer configured with the specified backend and options
func NewRenderer(backendType RendererBackendType, window window.Window, options ...RendererBuilderOption) Renderer {
	r := &renderer{
		mu:            &sync.Mutex{},
		pipelineCache: make(map[string]pipeline.Pipeline),
		backendType:   backendType,
	}

	// Apply options first so config flags (e.g. forceFallbackAdapter) are
	// available before the backend requests a GPU adapter.
	for _, opt := range options {
		opt(r)
	}

	msaa := MSAA4x // default
	if r.pendingMSAA != nil {
		msaa = *r.pendingMSAA
	}

	switch backendType {
	case BackendTypeWGPU:
		fallthrough
	default:
		r.backend = newWGPURendererBackend(window.SurfaceDescriptor(), r.forceFallbackAdapter, msaa)
	}

	if r.pendingPresentMode != nil {
		r.backend.SetPresentMode(*r.pendingPresentMode)
	}

	r.backend.ConfigureSurface(window.Width(), window.Height())
	return r
}

func (r *renderer) Resize(width, height int) {
	r.backend.ConfigureSurface(width, height)
}

func (r *renderer) SetPresentMode(mode PresentMode) {
	r.backend.SetPresentMode(mode)
}

func (r *renderer) Pipeline(key string) pipeline.Pipeline {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pipelineCache[key]
}

func (r *renderer) Pipelines() map[string]pipeline.Pipeline {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pipelineCache
}

func (r *renderer) RegisterPipelines(pipelines ...pipeline.Pipeline) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, p := range pipelines {
		key := p.PipelineKey()
		if _, exists := r.pipelineCache[key]; exists {
			continue
		}
		switch p.Type() {
		case pipeline.PipelineTypeCompute:
			if err := r.backend.RegisterComputePipeline(p); err != nil {
				return err
			}
		case pipeline.PipelineTypeRender:
			if err := r.backend.RegisterRenderPipeline(p); err != nil {
				return err
			}
		}
		r.pipelineCache[key] = p
	}
	return nil
}

func (r *renderer) SetPipeline(key string, p pipeline.Pipeline) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pipelineCache[key] = p
}

func (r *renderer) SetPipelines(pipelines map[string]pipeline.Pipeline) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pipelineCache = pipelines
}

func (r *renderer) InitMeshBuffers(provider bind_group_provider.BindGroupProvider, vertexData, indexData []byte, indexCount int) error {
	return r.backend.InitMeshBuffers(provider, vertexData, indexData, indexCount)
}

func (r *renderer) InitBindGroup(provider bind_group_provider.BindGroupProvider, descriptor wgpu.BindGroupLayoutDescriptor, bufferUsageOverrides map[int]wgpu.BufferUsage, bufferSizeOverrides map[int]uint64) error {
	return r.backend.InitBindGroup(provider, descriptor, bufferUsageOverrides, bufferSizeOverrides)
}

func (r *renderer) InitTextureView(provider bind_group_provider.BindGroupProvider, bindingKey int, stagingData common.TextureStagingData) error {
	return r.backend.InitTextureView(provider, bindingKey, stagingData)
}

func (r *renderer) InitSampler(provider bind_group_provider.BindGroupProvider, bindingKey int, samplerStagingData common.SamplerStagingData) error {
	return r.backend.InitSampler(provider, bindingKey, samplerStagingData)
}

func (r *renderer) WriteBuffers(writes []bind_group_provider.BufferWrite) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.backend.WriteBuffers(writes)
}

func (r *renderer) BeginComputeFrame() error {
	return r.backend.BeginComputeFrame()
}

func (r *renderer) EndComputeFrame() {
	r.backend.EndComputeFrame()
}

func (r *renderer) DispatchCompute(pipelineKey string, computeProvider bind_group_provider.BindGroupProvider, workGroupCount [3]uint32) {
	r.mu.Lock()
	defer r.mu.Unlock()

	p, exists := r.pipelineCache[pipelineKey]
	if !exists {
		return
	}

	r.backend.DispatchCompute(p, computeProvider, workGroupCount)
}

func (r *renderer) BeginFrame() error {
	return r.backend.BeginFrame()
}

func (r *renderer) DrawCall(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider) error {
	r.mu.Lock()
	p, exists := r.pipelineCache[pipelineKey]
	r.mu.Unlock()

	if !exists {
		return fmt.Errorf("render pipeline %q not found in cache", pipelineKey)
	}

	r.backend.DrawCall(p, meshProvider, instanceCount, bindGroups)
	return nil
}

func (r *renderer) DrawCallIndirect(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider) error {
	r.mu.Lock()
	p, exists := r.pipelineCache[pipelineKey]
	r.mu.Unlock()

	if !exists {
		return fmt.Errorf("render pipeline %q not found in cache", pipelineKey)
	}

	r.backend.DrawCallIndirect(p, meshProvider, indirectBuffer, bindGroups)
	return nil
}

func (r *renderer) EndFrame() {
	r.backend.EndFrame()
}

func (r *renderer) Present() {
	r.backend.Present()
}

func (r *renderer) CreateShadowDepthTexture(width, height int) (*wgpu.TextureView, *wgpu.Texture, error) {
	return r.backend.CreateShadowDepthTexture(width, height)
}

func (r *renderer) CreateComparisonSampler() (*wgpu.Sampler, error) {
	return r.backend.CreateComparisonSampler()
}

func (r *renderer) RegisterShadowPipeline(p pipeline.Pipeline) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	key := p.PipelineKey()
	if _, exists := r.pipelineCache[key]; exists {
		return nil
	}

	if err := r.backend.RegisterShadowPipeline(p); err != nil {
		return err
	}
	r.pipelineCache[key] = p
	return nil
}

func (r *renderer) BeginShadowFrame() error {
	return r.backend.BeginShadowFrame()
}

func (r *renderer) BeginShadowPass(depthView *wgpu.TextureView) {
	r.backend.BeginShadowPass(depthView)
}

func (r *renderer) ShadowDrawCall(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider) error {
	r.mu.Lock()
	p, exists := r.pipelineCache[pipelineKey]
	r.mu.Unlock()

	if !exists {
		return fmt.Errorf("shadow pipeline %q not found in cache", pipelineKey)
	}

	r.backend.ShadowDrawCall(p, meshProvider, instanceCount, bindGroups)
	return nil
}

func (r *renderer) ShadowDrawCallIndirect(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider) error {
	r.mu.Lock()
	p, exists := r.pipelineCache[pipelineKey]
	r.mu.Unlock()

	if !exists {
		return fmt.Errorf("shadow pipeline %q not found in cache", pipelineKey)
	}

	r.backend.ShadowDrawCallIndirect(p, meshProvider, indirectBuffer, bindGroups)
	return nil
}

func (r *renderer) EndShadowPass() {
	r.backend.EndShadowPass()
}

func (r *renderer) EndShadowFrame() {
	r.backend.EndShadowFrame()
}
