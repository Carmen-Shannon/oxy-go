package renderer

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
)

// renderer is the implementation of the Renderer interface.
type renderer struct {
	common.DelegateImpl[Renderer]

	mu *sync.Mutex

	pipelineCache map[string]pipeline.Pipeline
	materialCache map[string]material.Material

	backendType RendererBackendType
	backend     RendererBackend

	// Pre-creation config collected from builder options
	forceFallbackAdapter bool
	pendingPresentMode   *PresentMode
	pendingMSAA          *MSAASampleCount
}

func isTextureBindingEntry(entry wgpu.BindGroupLayoutEntry) bool {
	return entry.Texture != nil && entry.Texture.SampleType != gputypes.TextureSampleTypeUndefined
}

func isStorageTextureBindingEntry(entry wgpu.BindGroupLayoutEntry) bool {
	return entry.StorageTexture != nil && entry.StorageTexture.Access != 0
}

func isSamplerBindingEntry(entry wgpu.BindGroupLayoutEntry) bool {
	return entry.Sampler != nil && entry.Sampler.Type != gputypes.SamplerBindingTypeUndefined
}

func bufferBindingType(entry wgpu.BindGroupLayoutEntry) gputypes.BufferBindingType {
	if entry.Buffer == nil {
		return gputypes.BufferBindingTypeUndefined
	}

	return entry.Buffer.Type
}

// Renderer defines the interface for the rendering system.
//
// This is a high-level API designed to simplify rendering tasks into a streamlined and idiomatic flow.
// The Renderer manages a cache of shaders and pipelines, allowing for easy retrieval and management of these resources.
// The Renderer also implements a backend which allows for multiple backend API implementations to exist.
type Renderer interface {
	common.Delegate[Renderer]

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

	// CreateBuffer creates a GPU buffer with the specified label, size, and usage flags.
	// This is a low-level operation for creating buffers outside of BindGroupProviders,
	// such as staging buffers for GPU→CPU readback.
	//
	// Parameters:
	//   - label: a debug label for the buffer
	//   - size: the buffer size in bytes
	//   - usage: the buffer usage flags (e.g. MapRead | CopyDst for staging)
	//
	// Returns:
	//   - *wgpu.Buffer: the created GPU buffer
	//   - error: an error if buffer creation fails
	CreateBuffer(label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error)

	// CopyBufferToBuffer encodes a buffer-to-buffer copy on the current compute frame encoder.
	// Must be called between BeginComputeFrame and EndComputeFrame, outside of any compute pass.
	//
	// Parameters:
	//   - src: the source buffer to copy from
	//   - dst: the destination buffer to copy to
	//   - srcOffset: byte offset in the source buffer
	//   - dstOffset: byte offset in the destination buffer
	//   - size: the number of bytes to copy
	CopyBufferToBuffer(src, dst *wgpu.Buffer, srcOffset, dstOffset, size uint64)

	// ReadMappedBuffer synchronously maps a buffer for reading, copies the data into a new
	// byte slice, and unmaps the buffer. Blocks via Device.Poll until the mapping completes.
	// The buffer must have been created with BufferUsageMapRead and the GPU work that wrote
	// to it must have been submitted before this call.
	//
	// Parameters:
	//   - buf: the buffer to map and read (must have MapRead usage)
	//   - offset: byte offset to start reading from
	//   - size: number of bytes to read
	//
	// Returns:
	//   - []byte: a copy of the mapped buffer data
	//   - error: an error if mapping fails
	ReadMappedBuffer(buf *wgpu.Buffer, offset, size uint64) ([]byte, error)

	// WriteBuffers writes all staged buffer writes to the GPU queue.
	// Each BufferWrite targets a specific buffer on a BindGroupProvider at a given binding and offset.
	//
	// Parameters:
	//   - writes: a slice of BufferWrite structs describing the data to write
	WriteBuffers(writes []bind_group_provider.BufferWrite)

	// WriteRawBuffer writes data directly to a GPU buffer at the given byte offset
	// using the device queue. This bypasses the BindGroupProvider lookup and is
	// useful for updating standalone buffers not yet associated with any provider.
	//
	// Parameters:
	//   - buf: the GPU buffer to write to
	//   - offset: byte offset within the buffer
	//   - data: the raw bytes to upload
	WriteRawBuffer(buf *wgpu.Buffer, offset uint64, data []byte)

	// WriteTexture queues a data upload to a GPU texture region. Wraps
	// wgpu.Queue.WriteTexture for raw texture writes (e.g. noise data).
	//
	// Parameters:
	//   - tex: the destination texture
	//   - data: the raw byte data to write
	//   - width: the width of the region in texels
	//   - height: the height of the region in texels
	//   - bytesPerRow: the stride in bytes between consecutive rows
	WriteTexture(tex *wgpu.Texture, data []byte, width, height, bytesPerRow uint32)

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

	// RegisterMaterial creates GPU resources (textures, samplers, bind group) for a Material
	// and optionally registers a new render pipeline from the supplied pipeline builder options.
	// When pipelineOpts are provided and no pipeline exists for the material's PipelineKey,
	// a new render pipeline is created, registered, and the material's PipelineKey is updated.
	//
	// When no pipelineOpts are provided but the pipeline does not yet exist and the material
	// has a FragmentShaderPath set, RegisterMaterial automatically derives a new pipeline by
	// cloning the vertex shader from the best-matching existing pipeline (longest pipeline key
	// that is a prefix of the material's PipelineKey) and pairing it with a new fragment shader
	// built from the material's FragmentShaderPath. This allows callers to create variant
	// pipelines without referencing internal engine shader paths.
	//
	// The fragment shader from the resolved pipeline is inspected for @oxy:provider annotations
	// with the "material" identity to determine per-binding texture and sampler assignments.
	//
	// Parameters:
	//   - mat: the Material to initialize GPU resources for
	//   - key: a unique identifier prefix for the GPU bind group provider
	//   - pipelineOpts: optional pipeline builder options for creating and registering a new render pipeline
	//
	// Returns:
	//   - error: an error if GPU resource creation or pipeline registration fails
	RegisterMaterial(mat material.Material, key string, pipelineOpts ...pipeline.PipelineBuilderOption) error

	// Material returns the associated material from the material cache by name, provided it has already been registered.
	//
	// Parameters:
	//   - name: the unique identifier for the Material to retrieve
	//
	// Returns:
	//   - material.Material: the Material associated with the name, or nil if not found
	Material(name string) material.Material

	// BeginGeometryFrame opens a shared command encoder that merges shadow and G-Buffer
	// render passes into a single GPU submission, reducing per-frame driver overhead.
	// Uses reference counting so nested Begin/End pairs are safe. Must be paired with
	// EndGeometryFrame after all shadow and G-Buffer passes.
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginGeometryFrame() error

	// EndGeometryFrame decrements the geometry frame reference count and, when it
	// reaches zero, finishes the shared command encoder and submits the resulting
	// command buffer to the GPU queue.
	EndGeometryFrame()

	// BeginShadowFrame creates a command encoder for batching shadow depth passes.
	// Must be paired with EndShadowFrame. When a geometry frame is active, the
	// shadow encoder aliases the shared geometry encoder.
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginShadowFrame() error

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

	// CreateVSMTextures creates the GPU textures required for variance shadow mapping:
	// an RG32Float color texture for depth moments, a Depth32Float auxiliary depth texture
	// for hardware z-testing, and an RG32Float scratch texture for the intermediate blur result.
	//
	// Parameters:
	//   - width: shadow map width in texels
	//   - height: shadow map height in texels
	//
	// Returns:
	//   - vsmView: texture view for the VSM moments texture
	//   - vsmTex: the underlying VSM moments texture
	//   - scratchView: texture view for the scratch blur texture
	//   - scratchTex: the underlying scratch blur texture
	//   - depthView: texture view for the auxiliary depth texture
	//   - depthTex: the underlying auxiliary depth texture
	//   - err: an error if texture creation fails
	CreateVSMTextures(width, height int) (vsmView *wgpu.TextureView, vsmTex *wgpu.Texture, scratchView *wgpu.TextureView, scratchTex *wgpu.Texture, depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error)

	// CreateLinearSampler creates a linear filtering sampler suitable for VSM texture lookups.
	// Unlike the comparison sampler used for PCF, this sampler performs standard bilinear
	// filtering on the RG32Float moments texture.
	//
	// Returns:
	//   - *wgpu.Sampler: the linear sampler
	//   - error: an error if sampler creation fails
	CreateLinearSampler() (*wgpu.Sampler, error)

	// RegisterVSMShadowPipeline registers a render pipeline for VSM shadow map generation.
	// Unlike the depth-only PCF pipeline, this pipeline includes a fragment shader that
	// outputs depth moments to an RG32Float color target, uses a Depth32Float depth-stencil
	// for hardware z-testing, and applies no hardware depth bias.
	//
	// Parameters:
	//   - p: the pipeline object containing the VSM vertex and fragment shaders
	//
	// Returns:
	//   - error: an error if pipeline creation fails
	RegisterVSMShadowPipeline(p pipeline.Pipeline) error

	// BeginVSMShadowPass starts a render pass targeting both the VSM color texture (RG32Float)
	// and the auxiliary depth texture (Depth32Float). Must be called between BeginShadowFrame
	// and EndShadowFrame.
	//
	// Parameters:
	//   - vsmView: the VSM moments texture view (color attachment)
	//   - depthView: the auxiliary depth texture view (depth-stencil attachment)
	BeginVSMShadowPass(vsmView *wgpu.TextureView, depthView *wgpu.TextureView)

	// CreateSATTextures creates two RGBA32Float textures for the Summed-Area Table
	// ping-pong passes used by PCSS. Each texture has TextureBinding + StorageBinding
	// usage. Returns the texture views and textures for both ping-pong targets.
	//
	// Parameters:
	//   - width: the SAT texture width in texels
	//   - height: the SAT texture height in texels
	//
	// Returns:
	//   - satAView: texture view for SAT texture A
	//   - satATex: SAT texture A
	//   - satBView: texture view for SAT texture B
	//   - satBTex: SAT texture B
	//   - err: error if texture creation fails
	CreateSATTextures(width, height int) (satAView *wgpu.TextureView, satATex *wgpu.Texture, satBView *wgpu.TextureView, satBTex *wgpu.Texture, err error)

	// BeginGBufferFrame creates a command encoder for batching G-Buffer geometry
	// pre-pass draw calls. Must be paired with EndGBufferFrame.
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginGBufferFrame() error

	// GBufferDrawCall encodes a single instanced draw command within the current
	// G-Buffer MRT render pass.
	//
	// Parameters:
	//   - pipelineKey: the unique identifier for the cached G-Buffer Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - instanceCount: the number of instances to draw
	//   - bindGroups: bind group providers for the G-Buffer pass
	//
	// Returns:
	//   - error: an error if the pipeline is not found
	GBufferDrawCall(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider) error

	// GBufferDrawCallIndirect encodes a single indirect instanced draw command within
	// the current G-Buffer MRT render pass. The instance count is read from the
	// indirectBuffer on the GPU.
	//
	// Parameters:
	//   - pipelineKey: the unique identifier for the cached G-Buffer Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - indirectBuffer: the GPU buffer containing DrawIndexedIndirect arguments
	//   - bindGroups: bind group providers for the G-Buffer pass
	//
	// Returns:
	//   - error: an error if the pipeline is not found
	GBufferDrawCallIndirect(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider) error

	// EndGBufferPass ends the current G-Buffer MRT render pass.
	EndGBufferPass()

	// EndGBufferFrame finishes the G-Buffer command encoder and submits to the GPU queue.
	EndGBufferFrame()

	// BeginGBufferPass starts an MRT render pass targeting the G-Buffer textures.
	// Must be called between BeginGBufferFrame and EndGBufferFrame.
	//
	// Parameters:
	//   - normView: texture view for the normal MRT attachment
	//   - albedoView: texture view for the albedo MRT attachment
	//   - depthView: texture view for the depth-stencil attachment
	BeginGBufferPass(normView, albedoView, depthView *wgpu.TextureView)

	// CreateGBufferTextures creates the GPU textures required for the G-Buffer
	// geometry pre-pass.
	//
	// Parameters:
	//   - width: texture width in pixels
	//   - height: texture height in pixels
	//
	// Returns:
	//   - normView: texture view for the normal texture
	//   - normTex: the underlying normal texture
	//   - albedoView: texture view for the albedo texture
	//   - albedoTex: the underlying albedo texture
	//   - depthView: texture view for the depth texture
	//   - depthTex: the underlying depth texture
	//   - err: an error if texture creation fails
	CreateGBufferTextures(width, height int) (normView *wgpu.TextureView, normTex *wgpu.Texture, albedoView *wgpu.TextureView, albedoTex *wgpu.Texture, depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error)

	// RegisterGBufferPipeline registers a render pipeline for the G-Buffer
	// geometry pre-pass with MRT color targets and caches it.
	//
	// Parameters:
	//   - p: the pipeline object containing the G-Buffer vertex and fragment shaders
	//
	// Returns:
	//   - error: an error if pipeline creation fails
	RegisterGBufferPipeline(p pipeline.Pipeline) error

	// CreateSSAOTextures creates the GPU textures required for screen-space
	// ambient occlusion.
	//
	// Parameters:
	//   - width: texture width in pixels (screen resolution)
	//   - height: texture height in pixels (screen resolution)
	//
	// Returns:
	//   - rawView: texture view for the raw SSAO texture
	//   - rawTex: the underlying raw SSAO texture
	//   - blurredView: texture view for the blurred SSAO texture
	//   - blurredTex: the underlying blurred SSAO texture
	//   - scratchView: texture view for the scratch blur texture
	//   - scratchTex: the underlying scratch blur texture
	//   - noiseView: texture view for the 4×4 noise texture
	//   - noiseTex: the underlying noise texture
	//   - err: an error if texture creation fails
	CreateSSAOTextures(width, height int) (rawView *wgpu.TextureView, rawTex *wgpu.Texture, blurredView *wgpu.TextureView, blurredTex *wgpu.Texture, scratchView *wgpu.TextureView, scratchTex *wgpu.Texture, noiseView *wgpu.TextureView, noiseTex *wgpu.Texture, err error)

	// CreateProbeBakeTextures creates the GPU textures required for rendering
	// cubemap faces during irradiance probe baking.
	//
	// Parameters:
	//   - resolution: the cubemap face edge size in pixels
	//
	// Returns:
	//   - colorView: texture view for the bake color texture
	//   - colorTex: the underlying bake color texture
	//   - depthView: texture view for the bake depth texture
	//   - depthTex: the underlying bake depth texture
	//   - err: an error if texture creation fails
	CreateProbeBakeTextures(resolution int) (colorView *wgpu.TextureView, colorTex *wgpu.Texture, depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error)

	// RegisterProbeBakePipeline registers a render pipeline for irradiance
	// probe cubemap baking and caches it by PipelineKey. The pipeline writes
	// to a single RGBA8Unorm color target and Depth24Plus depth-stencil.
	//
	// Parameters:
	//   - p: the pipeline containing vertex and fragment shaders for probe baking
	//
	// Returns:
	//   - error: an error if pipeline creation fails
	RegisterProbeBakePipeline(p pipeline.Pipeline) error

	// BeginProbeBakeFrame creates a command encoder for batching probe bake
	// draw calls. Must be paired with EndProbeBakeFrame.
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginProbeBakeFrame() error

	// BeginProbeBakePass starts a render pass targeting the probe bake textures.
	// Must be called between BeginProbeBakeFrame and EndProbeBakeFrame.
	//
	// Parameters:
	//   - colorView: texture view for the RGBA8Unorm bake color attachment
	//   - depthView: texture view for the Depth24Plus bake depth attachment
	BeginProbeBakePass(colorView, depthView *wgpu.TextureView)

	// ProbeBakeDrawCall encodes a single instanced draw command within the
	// current probe bake render pass.
	//
	// Parameters:
	//   - pipelineKey: the unique identifier for the cached probe bake Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - instanceCount: the number of instances to draw
	//   - bindGroups: bind group providers for the probe bake pass
	//
	// Returns:
	//   - error: an error if the pipeline is not found
	ProbeBakeDrawCall(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider) error

	// ProbeBakeDrawCallIndirect encodes a single indirect instanced draw
	// command within the current probe bake render pass.
	//
	// Parameters:
	//   - pipelineKey: the unique identifier for the cached probe bake Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - indirectBuffer: the GPU buffer containing DrawIndexedIndirect arguments
	//   - bindGroups: bind group providers for the probe bake pass
	//
	// Returns:
	//   - error: an error if the pipeline is not found
	ProbeBakeDrawCallIndirect(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider) error

	// EndProbeBakePass ends the current probe bake render pass.
	EndProbeBakePass()

	// EndProbeBakeFrame finishes the probe bake command encoder and submits
	// to the GPU queue.
	EndProbeBakeFrame()

	// BeginHDRFrame creates a command encoder and begins a render pass targeting an
	// offscreen RGBA16Float HDR texture instead of the swapchain. When MSAA is active,
	// colorView is the multi-sampled texture and resolveView is the single-sample HDR
	// resolve target. Uses the main frame state so that DrawCall and EndFrame work
	// without modification.
	//
	// Parameters:
	//   - colorView: the render target texture view (MSAA or HDR)
	//   - resolveView: the HDR resolve target (nil when MSAA is off)
	//   - depthView: the depth-stencil attachment texture view
	//   - sampleCount: the MSAA sample count (1 when disabled)
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginHDRFrame(colorView, resolveView, depthView *wgpu.TextureView, sampleCount uint32) error

	// CreateCompositionTextures creates the GPU textures required for the HDR
	// composition pipeline: an RGBA16Float HDR texture, an optional MSAA texture,
	// and a Depth24Plus depth texture.
	//
	// Parameters:
	//   - width: texture width in pixels
	//   - height: texture height in pixels
	//   - sampleCount: the MSAA sample count (1 when disabled)
	//
	// Returns:
	//   - hdrView: texture view for the HDR texture
	//   - hdrTex: the underlying HDR texture
	//   - msaaView: texture view for the MSAA texture (nil when sampleCount <= 1)
	//   - msaaTex: the underlying MSAA texture (nil when sampleCount <= 1)
	//   - depthView: texture view for the depth texture
	//   - depthTex: the underlying depth texture
	//   - err: an error if texture creation fails
	CreateCompositionTextures(width, height int, sampleCount uint32) (hdrView *wgpu.TextureView, hdrTex *wgpu.Texture, msaaView *wgpu.TextureView, msaaTex *wgpu.Texture, depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error)

	// CreateSSRTextures creates the GPU texture required for screen-space
	// reflection output.
	//
	// Parameters:
	//   - width: texture width in pixels
	//   - height: texture height in pixels
	//
	// Returns:
	//   - ssrView: texture view for the SSR output texture
	//   - ssrTex: the underlying SSR output texture
	//   - err: an error if texture creation fails
	CreateSSRTextures(width, height int) (ssrView *wgpu.TextureView, ssrTex *wgpu.Texture, err error)

	// CreateHiZTextures creates the R32Float Hi-Z depth pyramid texture with a
	// full mip chain, plus per-mip read and storage texture views.
	//
	// Parameters:
	//   - width: texture width in pixels (should match G-Buffer depth width)
	//   - height: texture height in pixels (should match G-Buffer depth height)
	//
	// Returns:
	//   - hizView: full mip chain texture view for SSR reads
	//   - hizTex: the underlying Hi-Z texture
	//   - mipReadViews: per-mip texture views for downsample input
	//   - mipStorageViews: per-mip storage texture views for downsample output
	//   - mipCount: the number of mip levels generated
	//   - err: an error if texture creation fails
	CreateHiZTextures(width, height int) (hizView *wgpu.TextureView, hizTex *wgpu.Texture, mipReadViews []*wgpu.TextureView, mipStorageViews []*wgpu.TextureView, mipCount int, err error)

	// RegisterCompositionPipeline registers a render pipeline for the fullscreen
	// composition / tone mapping pass and caches it by PipelineKey.
	//
	// Parameters:
	//   - p: the pipeline containing vertex and fragment shaders for composition
	//
	// Returns:
	//   - error: an error if pipeline creation fails
	RegisterCompositionPipeline(p pipeline.Pipeline) error

	// BeginCompositionFrame acquires the swapchain texture and creates a command
	// encoder for the composition pass. Must be paired with EndCompositionFrame.
	//
	// Returns:
	//   - error: an error if the swapchain texture could not be acquired
	BeginCompositionFrame() error

	// BeginCompositionPass starts a render pass targeting the swapchain for
	// the fullscreen composition draw.
	BeginCompositionPass()

	// CompositionDrawCall encodes a fullscreen triangle draw command within the
	// current composition render pass.
	//
	// Parameters:
	//   - pipelineKey: the unique identifier for the cached composition Pipeline
	//   - bindGroups: bind group providers for the composition pass
	//
	// Returns:
	//   - error: an error if the pipeline is not found
	CompositionDrawCall(pipelineKey string, bindGroups []bind_group_provider.BindGroupProvider) error

	// EndCompositionPass ends the current composition render pass.
	EndCompositionPass()

	// EndCompositionFrame finishes the composition command encoder and submits
	// to the GPU queue.
	EndCompositionFrame()

	// SampleCount returns the MSAA sample count for the main render pass.
	//
	// Returns:
	//   - uint32: the MSAA sample count (1 when disabled)
	SampleCount() uint32

	// SetRenderTargetFormat overrides the color target format used when creating
	// render pipelines. Defaults to the swapchain surface format. Set to
	// RGBA16Float when composition (HDR tone mapping) is active so that the
	// lit render pass targets the offscreen HDR texture instead of the swapchain.
	//
	// Parameters:
	//   - format: the wgpu.TextureFormat to use for render pipeline color targets
	SetRenderTargetFormat(format wgpu.TextureFormat)
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
		materialCache: make(map[string]material.Material),
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
		displayHandle, windowHandle, err := window.SurfaceHandles()
		if err != nil {
			panic(fmt.Sprintf("failed to get surface descriptor from window: %v", err))
		}
		r.backend = newWGPURendererBackend(displayHandle, windowHandle, r.forceFallbackAdapter, msaa)
	}

	if r.pendingPresentMode != nil {
		r.backend.SetPresentMode(*r.pendingPresentMode)
	}

	r.backend.ConfigureSurface(window.Width(), window.Height())
	r.Delegate = r
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

func (r *renderer) RegisterMaterial(mat material.Material, key string, pipelineOpts ...pipeline.PipelineBuilderOption) error {
	pipelineKey := mat.PipelineKey()
	if pipelineKey == "" {
		return fmt.Errorf("material %q has no pipeline key set", mat.Name())
	}

	// If the pipeline doesn't exist yet, create one. When the material specifies a
	// fragment shader path, vertex and fragment shaders are auto-derived from the
	// best-matching base pipeline so callers never need to reference internal engine
	// shader paths. Any additional pipelineOpts (blend state, cull mode, etc.) are
	// merged on top of the auto-derived shaders.
	if r.Pipeline(pipelineKey) == nil {
		var opts []pipeline.PipelineBuilderOption

		// Auto-derive shaders when the material carries a fragment shader path.
		// The derived vertex+fragment shader options are prepended so that any
		// explicit shader options supplied by the caller take precedence.
		if mat.FragmentShaderPath() != "" {
			basePipeline := r.findBasePipeline(pipelineKey)
			if basePipeline == nil {
				return fmt.Errorf("pipeline %q not found and no base pipeline exists to derive from", pipelineKey)
			}
			vertShader := basePipeline.Shader(shader.ShaderTypeVertex)
			if vertShader == nil {
				return fmt.Errorf("base pipeline %q has no vertex shader to derive from", basePipeline.PipelineKey())
			}
			fragShader := shader.NewShader(pipelineKey+"_fragment", shader.ShaderTypeFragment, mat.FragmentShaderPath())
			opts = append(opts, pipeline.WithVertexShader(vertShader), pipeline.WithFragmentShader(fragShader))
		}

		// Append caller-supplied options so they can override defaults (e.g. blend, cull mode).
		opts = append(opts, pipelineOpts...)

		if len(opts) > 0 {
			p := pipeline.NewPipeline(pipelineKey, pipeline.PipelineTypeRender, opts...)
			if err := r.RegisterPipelines(p); err != nil {
				return fmt.Errorf("failed to register pipeline %q: %w", pipelineKey, err)
			}
		}
	}

	p := r.Pipeline(pipelineKey)
	if p == nil {
		return fmt.Errorf("pipeline %q not found", pipelineKey)
	}
	fragShader := p.Shader(shader.ShaderTypeFragment)
	if fragShader == nil {
		return fmt.Errorf("pipeline %q has no fragment shader", pipelineKey)
	}

	if err := r.initMaterialGPU(mat, fragShader, key); err != nil {
		return err
	}

	// Always cache the material by name so it can be retrieved via Material(),
	// even when initMaterialGPU is a no-op (e.g. shaders without @oxy:provider material).
	r.mu.Lock()
	r.materialCache[mat.Name()] = mat
	r.mu.Unlock()

	return nil
}

// findBasePipeline searches the pipeline cache for the longest-matching key that is a prefix
// of the given pipelineKey and belongs to a render pipeline with a vertex shader. This is used
// to derive new pipelines for materials that only differ in their fragment shader.
func (r *renderer) findBasePipeline(pipelineKey string) pipeline.Pipeline {
	r.mu.Lock()
	defer r.mu.Unlock()

	var best pipeline.Pipeline
	bestLen := 0
	for k, p := range r.pipelineCache {
		if p.Type() != pipeline.PipelineTypeRender {
			continue
		}
		if p.Shader(shader.ShaderTypeVertex) == nil {
			continue
		}
		if strings.HasPrefix(pipelineKey, k) && len(k) > bestLen {
			best = p
			bestLen = len(k)
		}
	}
	return best
}

func (r *renderer) Material(name string) material.Material {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.materialCache[name]
}

// initMaterialGPU creates GPU resources (textures, samplers, bind group) for a single Material
// by inspecting the fragment shader's pre-processed Declarations for @oxy:provider annotations
// with the "material" identity. Multiple material groups are supported: each group with an
// @oxy:provider material annotation gets its own BindGroupProvider, enabling a single material
// to own resources across several bind groups (e.g. textures at group 2, effect uniforms at group 3).
// Per-binding roles (diffuse_texture, normal_texture, etc.) are resolved from the declaration Args,
// eliminating the need for variable-name string matching.
func (r *renderer) initMaterialGPU(mat material.Material, fragmentShader shader.Shader, providerName string) error {
	// Phase 1: Collect all groups annotated with @oxy:provider material and their binding roles.
	type groupInfo struct {
		bindingRoles map[int]shader.AnnotationArg
	}
	materialGroups := make(map[int]*groupInfo)

	for _, decl := range fragmentShader.Declarations() {
		if decl.Type != shader.AnnotationTypeProvider || decl.Group == nil {
			continue
		}
		if decl.Args[0] != shader.AnnotationArgMaterial {
			continue
		}
		g := *decl.Group
		if _, exists := materialGroups[g]; !exists {
			materialGroups[g] = &groupInfo{bindingRoles: make(map[int]shader.AnnotationArg)}
		}
		if len(decl.Args) > 1 && decl.Binding != nil {
			materialGroups[g].bindingRoles[*decl.Binding] = decl.Args[1]
		}
	}

	if len(materialGroups) == 0 {
		return nil
	}

	// Sort group indices so the lowest group (typically the texture group) becomes the
	// primary BindGroupProvider for backward-compatible access via mat.BindGroupProvider().
	groupIndices := make([]int, 0, len(materialGroups))
	for g := range materialGroups {
		groupIndices = append(groupIndices, g)
	}
	sort.Ints(groupIndices)

	// Texture role → material texture lookup
	type textureBinding struct {
		tex  *common.ImportedTexture
		role shader.AnnotationArg
	}
	roleToTexture := map[shader.AnnotationArg]textureBinding{
		shader.AnnotationArgDiffuseTexture:           {tex: mat.DiffuseTexture(), role: shader.AnnotationArgDiffuseTexture},
		shader.AnnotationArgNormalTexture:            {tex: mat.NormalTexture(), role: shader.AnnotationArgNormalTexture},
		shader.AnnotationArgMetallicRoughnessTexture: {tex: mat.MetallicRoughnessTexture(), role: shader.AnnotationArgMetallicRoughnessTexture},
	}
	textureSamplerPairs := map[shader.AnnotationArg]shader.AnnotationArg{
		shader.AnnotationArgDiffuseTexture:           shader.AnnotationArgDiffuseSampler,
		shader.AnnotationArgNormalTexture:            shader.AnnotationArgNormalSampler,
		shader.AnnotationArgMetallicRoughnessTexture: shader.AnnotationArgMetallicRoughnessSampler,
	}

	// Phase 2: For each material group, create a BGP, process textures/samplers, init bind group.
	firstGroup := true
	for _, groupIdx := range groupIndices {
		gi := materialGroups[groupIdx]
		provName := fmt.Sprintf("%s_g%d", providerName, groupIdx)
		provider := bind_group_provider.NewBindGroupProvider(provName)

		// Build binding→role reverse map for this group
		roleToBinding := make(map[shader.AnnotationArg]int)
		for binding, role := range gi.bindingRoles {
			roleToBinding[role] = binding
		}

		// Process user-supplied textures + their paired samplers
		for texRole, tb := range roleToTexture {
			if tb.tex == nil {
				continue
			}
			texBindingIdx, hasTexBinding := roleToBinding[texRole]
			if !hasTexBinding {
				continue
			}

			samplerRole := textureSamplerPairs[texRole]
			samplerBindingIdx, hasSamplerBinding := roleToBinding[samplerRole]

			pixels, width, height, err := tb.tex.Decode()
			if err != nil {
				return fmt.Errorf("failed to decode %s texture: %w", texRole, err)
			}
			isLinear := texRole == shader.AnnotationArgNormalTexture || texRole == shader.AnnotationArgMetallicRoughnessTexture
			stagingData := common.TextureStagingData{
				Pixels: pixels,
				Width:  width,
				Height: height,
				Linear: isLinear,
			}
			if err := r.InitTextureView(provider, texBindingIdx, stagingData); err != nil {
				return fmt.Errorf("failed to init %s texture view: %w", texRole, err)
			}
			if hasSamplerBinding {
				samplerData := common.SamplerStagingData{
					AddressModeU:  gputypes.AddressModeRepeat,
					AddressModeV:  gputypes.AddressModeRepeat,
					AddressModeW:  gputypes.AddressModeRepeat,
					MagFilter:     gputypes.FilterModeLinear,
					MinFilter:     gputypes.FilterModeLinear,
					MipmapFilter:  wgpu.FilterMode(gputypes.MipmapFilterModeLinear),
					LodMinClamp:   0,
					LodMaxClamp:   32,
					MaxAnisotropy: 1,
				}
				if tb.tex.SamplerData != nil {
					samplerData = *tb.tex.SamplerData
				}
				if err := r.InitSampler(provider, samplerBindingIdx, samplerData); err != nil {
					return fmt.Errorf("failed to init %s sampler: %w", samplerRole, err)
				}
			}
		}

		// Fill in fallback textures/samplers for any texture or sampler bindings without data.
		descriptor := fragmentShader.BindGroupLayoutDescriptor(groupIdx)
		for _, entry := range descriptor.Entries {
			binding := int(entry.Binding)
			isTexture := isTextureBindingEntry(entry)
			isSampler := isSamplerBindingEntry(entry)

			if isTexture && provider.TextureView(binding) == nil {
				role := gi.bindingRoles[binding]
				var pixel [4]byte
				switch role {
				case shader.AnnotationArgNormalTexture:
					pixel = [4]byte{128, 128, 255, 255}
				case shader.AnnotationArgMetallicRoughnessTexture:
					// Encode the material's scalar roughness (G) and metallic (B) into the
					// fallback 1×1 texture so shaders that read these values from the texture
					// (e.g. the G-Buffer pass for SSR) get the correct material properties.
					roughByte := byte(mat.Roughness() * 255)
					metalByte := byte(mat.Metallic() * 255)
					pixel = [4]byte{0, roughByte, metalByte, 255}
				default:
					pixel = [4]byte{255, 255, 255, 255}
				}
				isLinear := role == shader.AnnotationArgNormalTexture || role == shader.AnnotationArgMetallicRoughnessTexture
				fallback := common.TextureStagingData{
					Pixels: pixel[:],
					Width:  1,
					Height: 1,
					Linear: isLinear,
				}
				if err := r.InitTextureView(provider, binding, fallback); err != nil {
					return fmt.Errorf("failed to init fallback texture at binding %d: %w", binding, err)
				}
			}

			if isSampler && provider.Sampler(binding) == nil {
				fallbackSampler := common.SamplerStagingData{
					AddressModeU:  gputypes.AddressModeRepeat,
					AddressModeV:  gputypes.AddressModeRepeat,
					AddressModeW:  gputypes.AddressModeRepeat,
					MagFilter:     gputypes.FilterModeLinear,
					MinFilter:     gputypes.FilterModeLinear,
					MipmapFilter:  wgpu.FilterMode(gputypes.MipmapFilterModeLinear),
					LodMinClamp:   0,
					LodMaxClamp:   32,
					MaxAnisotropy: 1,
				}
				if err := r.InitSampler(provider, binding, fallbackSampler); err != nil {
					return fmt.Errorf("failed to init fallback sampler at binding %d: %w", binding, err)
				}
			}
		}

		if err := r.InitBindGroup(provider, descriptor, nil, nil); err != nil {
			return fmt.Errorf("failed to init material bind group for group %d: %w", groupIdx, err)
		}

		// First material group becomes the primary BindGroupProvider for backward compat.
		if firstGroup {
			mat.SetBindGroupProvider(provider)
			firstGroup = false
		}
		mat.SetProvider(groupIdx, provider)
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

func (r *renderer) CreateBuffer(label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
	return r.backend.CreateBuffer(label, size, usage)
}

func (r *renderer) CopyBufferToBuffer(src, dst *wgpu.Buffer, srcOffset, dstOffset, size uint64) {
	r.backend.CopyBufferToBuffer(src, dst, srcOffset, dstOffset, size)
}

func (r *renderer) ReadMappedBuffer(buf *wgpu.Buffer, offset, size uint64) ([]byte, error) {
	return r.backend.ReadMappedBuffer(buf, offset, size)
}

func (r *renderer) WriteBuffers(writes []bind_group_provider.BufferWrite) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.backend.WriteBuffers(writes)
}

func (r *renderer) WriteRawBuffer(buf *wgpu.Buffer, offset uint64, data []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.backend.WriteRawBuffer(buf, offset, data)
}

func (r *renderer) WriteTexture(tex *wgpu.Texture, data []byte, width, height, bytesPerRow uint32) {
	r.backend.WriteTexture(tex, data, width, height, bytesPerRow)
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

func (r *renderer) BeginGeometryFrame() error {
	return r.backend.BeginGeometryFrame()
}

func (r *renderer) EndGeometryFrame() {
	r.backend.EndGeometryFrame()
}

func (r *renderer) BeginShadowFrame() error {
	return r.backend.BeginShadowFrame()
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

func (r *renderer) CreateVSMTextures(width, height int) (vsmView *wgpu.TextureView, vsmTex *wgpu.Texture, scratchView *wgpu.TextureView, scratchTex *wgpu.Texture, depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error) {
	return r.backend.CreateVSMTextures(width, height)
}

func (r *renderer) CreateLinearSampler() (*wgpu.Sampler, error) {
	return r.backend.CreateLinearSampler()
}

func (r *renderer) RegisterVSMShadowPipeline(p pipeline.Pipeline) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	key := p.PipelineKey()
	if _, exists := r.pipelineCache[key]; exists {
		return nil
	}

	if err := r.backend.RegisterVSMShadowPipeline(p); err != nil {
		return err
	}
	r.pipelineCache[key] = p
	return nil
}

func (r *renderer) BeginVSMShadowPass(vsmView *wgpu.TextureView, depthView *wgpu.TextureView) {
	r.backend.BeginVSMShadowPass(vsmView, depthView)
}

func (r *renderer) CreateSATTextures(width, height int) (satAView *wgpu.TextureView, satATex *wgpu.Texture, satBView *wgpu.TextureView, satBTex *wgpu.Texture, err error) {
	return r.backend.CreateSATTextures(width, height)
}

func (r *renderer) BeginGBufferFrame() error {
	return r.backend.BeginGBufferFrame()
}

func (r *renderer) GBufferDrawCall(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider) error {
	r.mu.Lock()
	p, exists := r.pipelineCache[pipelineKey]
	r.mu.Unlock()

	if !exists {
		return fmt.Errorf("gbuffer pipeline %q not found in cache", pipelineKey)
	}

	r.backend.GBufferDrawCall(p, meshProvider, instanceCount, bindGroups)
	return nil
}

func (r *renderer) GBufferDrawCallIndirect(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider) error {
	r.mu.Lock()
	p, exists := r.pipelineCache[pipelineKey]
	r.mu.Unlock()

	if !exists {
		return fmt.Errorf("gbuffer pipeline %q not found in cache", pipelineKey)
	}

	r.backend.GBufferDrawCallIndirect(p, meshProvider, indirectBuffer, bindGroups)
	return nil
}

func (r *renderer) EndGBufferPass() {
	r.backend.EndGBufferPass()
}

func (r *renderer) EndGBufferFrame() {
	r.backend.EndGBufferFrame()
}

func (r *renderer) BeginGBufferPass(normView, albedoView, depthView *wgpu.TextureView) {
	r.backend.BeginGBufferPass(normView, albedoView, depthView)
}

func (r *renderer) CreateGBufferTextures(width, height int) (normView *wgpu.TextureView, normTex *wgpu.Texture, albedoView *wgpu.TextureView, albedoTex *wgpu.Texture, depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error) {
	return r.backend.CreateGBufferTextures(width, height)
}

func (r *renderer) RegisterGBufferPipeline(p pipeline.Pipeline) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	key := p.PipelineKey()
	if _, exists := r.pipelineCache[key]; exists {
		return nil
	}

	if err := r.backend.RegisterGBufferPipeline(p); err != nil {
		return err
	}
	r.pipelineCache[key] = p
	return nil
}

func (r *renderer) CreateSSAOTextures(width, height int) (rawView *wgpu.TextureView, rawTex *wgpu.Texture, blurredView *wgpu.TextureView, blurredTex *wgpu.Texture, scratchView *wgpu.TextureView, scratchTex *wgpu.Texture, noiseView *wgpu.TextureView, noiseTex *wgpu.Texture, err error) {
	return r.backend.CreateSSAOTextures(width, height)
}

func (r *renderer) CreateProbeBakeTextures(resolution int) (colorView *wgpu.TextureView, colorTex *wgpu.Texture, depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error) {
	return r.backend.CreateProbeBakeTextures(resolution)
}

func (r *renderer) RegisterProbeBakePipeline(p pipeline.Pipeline) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	key := p.PipelineKey()
	if _, exists := r.pipelineCache[key]; exists {
		return nil
	}

	if err := r.backend.RegisterProbeBakePipeline(p); err != nil {
		return err
	}
	r.pipelineCache[key] = p
	return nil
}

func (r *renderer) BeginProbeBakeFrame() error {
	return r.backend.BeginProbeBakeFrame()
}

func (r *renderer) BeginProbeBakePass(colorView, depthView *wgpu.TextureView) {
	r.backend.BeginProbeBakePass(colorView, depthView)
}

func (r *renderer) ProbeBakeDrawCall(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider) error {
	r.mu.Lock()
	p, exists := r.pipelineCache[pipelineKey]
	r.mu.Unlock()

	if !exists {
		return fmt.Errorf("probe bake pipeline %q not found in cache", pipelineKey)
	}

	r.backend.ProbeBakeDrawCall(p, meshProvider, instanceCount, bindGroups)
	return nil
}

func (r *renderer) ProbeBakeDrawCallIndirect(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider) error {
	r.mu.Lock()
	p, exists := r.pipelineCache[pipelineKey]
	r.mu.Unlock()

	if !exists {
		return fmt.Errorf("probe bake pipeline %q not found in cache", pipelineKey)
	}

	r.backend.ProbeBakeDrawCallIndirect(p, meshProvider, indirectBuffer, bindGroups)
	return nil
}

func (r *renderer) EndProbeBakePass() {
	r.backend.EndProbeBakePass()
}

func (r *renderer) EndProbeBakeFrame() {
	r.backend.EndProbeBakeFrame()
}

func (r *renderer) BeginHDRFrame(colorView, resolveView, depthView *wgpu.TextureView, sampleCount uint32) error {
	return r.backend.BeginHDRFrame(colorView, resolveView, depthView, sampleCount)
}

func (r *renderer) CreateCompositionTextures(width, height int, sampleCount uint32) (hdrView *wgpu.TextureView, hdrTex *wgpu.Texture, msaaView *wgpu.TextureView, msaaTex *wgpu.Texture, depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error) {
	return r.backend.CreateCompositionTextures(width, height, sampleCount)
}

func (r *renderer) CreateSSRTextures(width, height int) (ssrView *wgpu.TextureView, ssrTex *wgpu.Texture, err error) {
	return r.backend.CreateSSRTextures(width, height)
}

func (r *renderer) CreateHiZTextures(width, height int) (hizView *wgpu.TextureView, hizTex *wgpu.Texture, mipReadViews []*wgpu.TextureView, mipStorageViews []*wgpu.TextureView, mipCount int, err error) {
	return r.backend.CreateHiZTextures(width, height)
}

func (r *renderer) RegisterCompositionPipeline(p pipeline.Pipeline) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	key := p.PipelineKey()
	if _, exists := r.pipelineCache[key]; exists {
		return nil
	}

	if err := r.backend.RegisterCompositionPipeline(p); err != nil {
		return err
	}
	r.pipelineCache[key] = p
	return nil
}

func (r *renderer) BeginCompositionFrame() error {
	return r.backend.BeginCompositionFrame()
}

func (r *renderer) BeginCompositionPass() {
	r.backend.BeginCompositionPass()
}

func (r *renderer) CompositionDrawCall(pipelineKey string, bindGroups []bind_group_provider.BindGroupProvider) error {
	r.mu.Lock()
	p, exists := r.pipelineCache[pipelineKey]
	r.mu.Unlock()

	if !exists {
		return fmt.Errorf("composition pipeline %q not found in cache", pipelineKey)
	}

	r.backend.CompositionDrawCall(p, bindGroups)
	return nil
}

func (r *renderer) EndCompositionPass() {
	r.backend.EndCompositionPass()
}

func (r *renderer) EndCompositionFrame() {
	r.backend.EndCompositionFrame()
}

func (r *renderer) SampleCount() uint32 {
	return r.backend.SampleCount()
}

func (r *renderer) SetRenderTargetFormat(format wgpu.TextureFormat) {
	r.backend.SetRenderTargetFormat(format)
}
