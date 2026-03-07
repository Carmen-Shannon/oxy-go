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
	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	"github.com/gogpu/wgpu/hal"
	_ "github.com/gogpu/wgpu/hal/allbackends"
)

type wgpuRendererBackendImpl struct {
	mu     *sync.Mutex
	device *wgpu.Device
	queue  *wgpu.Queue

	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	surface  *wgpu.Surface

	surfaceFormat        *wgpu.TextureFormat
	renderTargetFormat   *wgpu.TextureFormat // defaults to surfaceFormat; overridden to RGBA16Float when composition is active
	msaaTextureView      *wgpu.TextureView
	depthTextureView     *wgpu.TextureView
	renderPassDescriptor *wgpu.RenderPassDescriptor

	presentMode wgpu.PresentMode // defaults to PresentModeImmediate (Uncapped)
	sampleCount MSAASampleCount  // MSAA sample count for the main render pass

	// Frame state for batched rendering across multiple draw calls
	frameEncoder *wgpu.CommandEncoder
	framePass    *wgpu.RenderPassEncoder
	frameSurface *wgpu.SurfaceTexture
	frameView    *wgpu.TextureView

	// Compute frame state for batching all compute dispatches into a single GPU submission.
	// computeFrameDepth tracks nested Begin/End pairs so that multiple Prepare* methods
	// can share a single encoder when the engine wraps them in an outer Begin/End.
	computeFrameEncoder *wgpu.CommandEncoder
	computeFrameDepth   int

	// Geometry pre-pass state for merging shadow and G-Buffer render passes
	// into a single command encoder and GPU submission. geometryFrameDepth
	// tracks nested Begin/End pairs so BeginShadowFrame and BeginGBufferFrame
	// can share the encoder when wrapped in an outer BeginGeometryFrame.
	geometryFrameEncoder *wgpu.CommandEncoder
	geometryFrameDepth   int

	// Shadow pass state for rendering depth-only passes from a light's perspective.
	// Shadow passes use their own command encoder, a Depth32Float texture (no color),
	// sample count 1 (no MSAA), and front-face culling to reduce self-shadowing.
	// When a geometry frame is active, shadowFrameEncoder aliases geometryFrameEncoder.
	shadowFrameEncoder *wgpu.CommandEncoder
	shadowPass         *wgpu.RenderPassEncoder

	// G-Buffer pass state for rendering the geometry pre-pass that outputs
	// normals and albedo to multiple render targets (MRT).
	// The G-Buffer pass uses its own command encoder, sample count 1, and
	// produces textures consumed by screen-space effects (SSAO, SSR).
	// When a geometry frame is active, gbufferFrameEncoder aliases geometryFrameEncoder.
	gbufferFrameEncoder *wgpu.CommandEncoder
	gbufferPass         *wgpu.RenderPassEncoder

	// Probe bake pass state for rendering cubemap faces from each irradiance
	// probe position. Each face produces an RGBA8Unorm color attachment at the
	// bake resolution, which is later projected into SH coefficients by a
	// compute shader.
	probeBakeFrameEncoder *wgpu.CommandEncoder
	probeBakePass         *wgpu.RenderPassEncoder

	// Composition pass state for the fullscreen tone mapping / compositing pass
	// that reads the HDR lit texture and optional SSR texture, then renders the
	// final LDR result to the swapchain.
	compositionFrameEncoder *wgpu.CommandEncoder
	compositionPass         *wgpu.RenderPassEncoder
	compositionSurface      *wgpu.SurfaceTexture
	compositionView         *wgpu.TextureView
}

type wgpuRendererBackend interface {
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

	// WriteRawBuffer writes data directly to a GPU buffer at the given byte offset
	// using the device queue. This bypasses the BindGroupProvider lookup and is
	// useful for updating standalone buffers not yet associated with any provider.
	//
	// Parameters:
	//   - buf: the GPU buffer to write to
	//   - offset: byte offset within the buffer
	//   - data: the raw bytes to upload
	WriteRawBuffer(buf *wgpu.Buffer, offset uint64, data []byte)

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

	// BeginShadowFrame creates a command encoder for batching all shadow depth passes
	// within a frame. Must be paired with EndShadowFrame after all shadow passes.
	// When a geometry frame is active, the shadow encoder aliases the shared geometry
	// encoder and EndShadowFrame becomes a no-op.
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginShadowFrame() error

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
	// When a geometry frame is active, this is a no-op (the geometry frame handles submission).
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

	// RegisterVSMShadowPipeline creates a render pipeline for VSM shadow map generation.
	// Unlike the depth-only PCF pipeline, this pipeline includes a fragment shader that
	// outputs depth moments to an RG32Float color target, uses a Depth32Float depth-stencil
	// for hardware z-testing, and applies no hardware depth bias (the partial-derivative
	// moment computation replaces it).
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
	// pre-pass draw calls into a single GPU submission. Must be paired with
	// EndGBufferFrame after all G-Buffer passes. When a geometry frame is active,
	// the G-Buffer encoder aliases the shared geometry encoder and EndGBufferFrame
	// becomes a no-op.
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginGBufferFrame() error

	// BeginGBufferPass starts an MRT render pass targeting the G-Buffer textures:
	// normal (RGBA16Float), albedo (RGBA8Unorm), and depth (Depth24Plus).
	// World-space position is reconstructed from depth at read time by compute
	// shaders, so no position MRT is needed. Must be called between
	// BeginGBufferFrame and EndGBufferFrame.
	//
	// Parameters:
	//   - normView: texture view for the normal MRT attachment
	//   - albedoView: texture view for the albedo MRT attachment
	//   - depthView: texture view for the depth-stencil attachment
	BeginGBufferPass(normView, albedoView, depthView *wgpu.TextureView)

	// GBufferDrawCall encodes a single instanced draw command within the current
	// G-Buffer MRT render pass.
	//
	// Parameters:
	//   - p: the cached G-Buffer Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - instanceCount: the number of instances to draw
	//   - bindGroups: bind group providers for the G-Buffer pass
	GBufferDrawCall(p pipeline.Pipeline, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider)

	// GBufferDrawCallIndirect encodes a single indirect instanced draw command within
	// the current G-Buffer MRT render pass. The instance count is read from the
	// indirectBuffer on the GPU.
	//
	// Parameters:
	//   - p: the cached G-Buffer Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - indirectBuffer: the GPU buffer containing DrawIndexedIndirect arguments
	//   - bindGroups: bind group providers for the G-Buffer pass
	GBufferDrawCallIndirect(p pipeline.Pipeline, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider)

	// EndGBufferPass ends the current G-Buffer MRT render pass.
	EndGBufferPass()

	// EndGBufferFrame finishes the G-Buffer command encoder and submits to the GPU queue.
	// When a geometry frame is active, this is a no-op (the geometry frame handles submission).
	EndGBufferFrame()

	// CreateGBufferTextures creates the GPU textures required for the G-Buffer
	// geometry pre-pass: an RGBA16Float normal texture, an RGBA8Unorm albedo
	// texture, and a Depth24Plus depth texture. World-space position is
	// reconstructed from depth at read time. Color textures also have
	// TextureBinding for downstream read access.
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

	// RegisterGBufferPipeline creates a render pipeline for the G-Buffer geometry
	// pre-pass. The pipeline writes to three MRT color targets (RGBA16Float,
	// RGBA16Float, RGBA8Unorm) and a Depth24Plus depth-stencil, with sample count 1.
	//
	// Parameters:
	//   - p: the pipeline object containing the G-Buffer vertex and fragment shaders
	//
	// Returns:
	//   - error: an error if pipeline creation fails
	RegisterGBufferPipeline(p pipeline.Pipeline) error

	// CreateSSAOTextures creates the GPU textures required for screen-space
	// ambient occlusion: a raw R32Float occlusion texture, a blurred R32Float
	// output texture, a scratch R32Float intermediate texture for the separable
	// bilateral blur, and a 4×4 RGBA16Float noise texture for kernel rotation.
	// R32Float is used because R8Unorm does not support StorageBinding in WebGPU.
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
	// cubemap faces during irradiance probe baking: an RGBA8Unorm color
	// texture for capturing radiance, and a Depth24Plus depth texture for
	// hardware z-testing. Both are square with the specified resolution.
	// The color texture also has TextureBinding and CopySrc for downstream
	// SH projection compute reads.
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

	// RegisterProbeBakePipeline creates a render pipeline for irradiance probe
	// cubemap baking. The pipeline writes to a single RGBA8Unorm color target
	// and a Depth24Plus depth-stencil with sample count 1.
	//
	// Parameters:
	//   - p: the pipeline containing vertex and fragment shaders for probe baking
	//
	// Returns:
	//   - error: an error if pipeline creation fails
	RegisterProbeBakePipeline(p pipeline.Pipeline) error

	// BeginProbeBakeFrame creates a command encoder for batching probe bake
	// draw calls into a single GPU submission. Must be paired with
	// EndProbeBakeFrame.
	//
	// Returns:
	//   - error: an error if the command encoder could not be created
	BeginProbeBakeFrame() error

	// BeginProbeBakePass starts a render pass targeting the probe bake
	// textures. Must be called between BeginProbeBakeFrame and
	// EndProbeBakeFrame.
	//
	// Parameters:
	//   - colorView: texture view for the RGBA8Unorm bake color attachment
	//   - depthView: texture view for the Depth24Plus bake depth attachment
	BeginProbeBakePass(colorView, depthView *wgpu.TextureView)

	// ProbeBakeDrawCall encodes a single instanced draw command within the
	// current probe bake render pass.
	//
	// Parameters:
	//   - p: the cached probe bake Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - instanceCount: the number of instances to draw
	//   - bindGroups: bind group providers for the probe bake pass
	ProbeBakeDrawCall(p pipeline.Pipeline, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider)

	// ProbeBakeDrawCallIndirect encodes a single indirect instanced draw
	// command within the current probe bake render pass. The instance count
	// is read from the indirectBuffer on the GPU.
	//
	// Parameters:
	//   - p: the cached probe bake Pipeline
	//   - meshProvider: the BindGroupProvider holding vertex and index buffers
	//   - indirectBuffer: the GPU buffer containing DrawIndexedIndirect arguments
	//   - bindGroups: bind group providers for the probe bake pass
	ProbeBakeDrawCallIndirect(p pipeline.Pipeline, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider)

	// EndProbeBakePass ends the current probe bake render pass.
	EndProbeBakePass()

	// EndProbeBakeFrame finishes the probe bake command encoder and submits
	// to the GPU queue.
	EndProbeBakeFrame()

	// BeginHDRFrame creates a command encoder and begins a render pass targeting an
	// offscreen RGBA16Float HDR texture instead of the swapchain. When MSAA is active,
	// colorView is the multi-sampled texture and resolveView is the single-sample HDR
	// resolve target. When MSAA is off, colorView is the HDR texture directly and
	// resolveView should be nil. Uses the main frame state (frameEncoder/framePass)
	// so that DrawCall and DrawCallIndirect work without modification. EndFrame submits
	// the resulting command buffer.
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
	// composition pipeline: an RGBA16Float HDR texture (resolve target / direct
	// render target), an optional MSAA texture when sampleCount > 1, and a
	// Depth24Plus depth texture matching the sample count.
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
	// reflection output: an RGBA16Float texture with StorageBinding for
	// compute writes and TextureBinding for composition fragment reads.
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
	// full mip chain, plus per-mip read and storage texture views. The full-chain
	// view is used by the SSR compute shader, per-mip read views are downsample
	// inputs, and per-mip storage views are downsample outputs.
	//
	// Parameters:
	//   - width: texture width in pixels (should match G-Buffer depth width)
	//   - height: texture height in pixels (should match G-Buffer depth height)
	//
	// Returns:
	//   - hizView: full mip chain texture view for SSR reads
	//   - hizTex: the underlying Hi-Z texture
	//   - mipReadViews: per-mip texture views for downsample input (texture_2d)
	//   - mipStorageViews: per-mip storage texture views for downsample output (texture_storage_2d)
	//   - mipCount: the number of mip levels generated
	//   - err: an error if texture creation fails
	CreateHiZTextures(width, height int) (hizView *wgpu.TextureView, hizTex *wgpu.Texture, mipReadViews []*wgpu.TextureView, mipStorageViews []*wgpu.TextureView, mipCount int, err error)

	// RegisterCompositionPipeline creates a render pipeline for the fullscreen
	// composition / tone mapping pass. The pipeline uses a full-screen triangle
	// (no vertex buffers), targets the swapchain format, and has no depth-stencil.
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
	// the fullscreen composition draw. Must be called between
	// BeginCompositionFrame and EndCompositionFrame.
	BeginCompositionPass()

	// CompositionDrawCall encodes a fullscreen triangle draw command within the
	// current composition render pass. No vertex or index buffers are used; the
	// vertex shader generates positions from vertex_index.
	//
	// Parameters:
	//   - p: the cached composition Pipeline
	//   - bindGroups: bind group providers for the composition pass
	CompositionDrawCall(p pipeline.Pipeline, bindGroups []bind_group_provider.BindGroupProvider)

	// EndCompositionPass ends the current composition render pass.
	EndCompositionPass()

	// EndCompositionFrame finishes the composition command encoder and submits
	// to the GPU queue. The swapchain texture remains held until Present is called.
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

var _ RendererBackend = &wgpuRendererBackendImpl{}

func (b *wgpuRendererBackendImpl) SampleCount() uint32 {
	return uint32(b.sampleCount)
}

func (b *wgpuRendererBackendImpl) SetRenderTargetFormat(format wgpu.TextureFormat) {
	b.mu.Lock()
	defer b.mu.Unlock()
	f := format
	b.renderTargetFormat = &f
}

func newWGPURendererBackend(displayHandle, windowHandle uintptr, forceFallbackAdapter bool, sampleCount MSAASampleCount) wgpuRendererBackend {
	runtime.LockOSThread()
	w := &wgpuRendererBackendImpl{
		mu:          &sync.Mutex{},
		presentMode: wgpu.PresentModeImmediate,
		sampleCount: sampleCount,
	}
	var err error
	w.instance, err = wgpu.CreateInstance(&wgpu.InstanceDescriptor{
		Backends: gputypes.BackendsAll,
	})
	if err != nil {
		panic(err)
	}
	w.surface, err = w.instance.CreateSurface(displayHandle, windowHandle)
	if err != nil {
		panic(err)
	}

	a, err := w.instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		ForceFallbackAdapter: forceFallbackAdapter,
	})
	if err != nil {
		panic(err)
	}
	w.adapter = a

	// Start from the WebGPU spec default limits and raise MaxBindGroups to 8
	// so the lit fragment shader's 6 bind groups (0–5) are allowed.
	limits := wgpu.DefaultLimits()
	limits.MaxBindGroups = 8

	d, err := a.RequestDevice(&wgpu.DeviceDescriptor{
		Label:            "Main Device",
		RequiredFeatures: wgpu.Features(gputypes.FeatureFloat32Filterable),
		RequiredLimits:   wgpu.Limits(limits),
	})
	if err != nil {
		panic(err)
	}
	w.device = d
	w.queue = d.Queue()

	return w
}

func (b *wgpuRendererBackendImpl) ConfigureSurface(width, height int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.surfaceFormat == nil {
		format := wgpu.TextureFormatBGRA8Unorm
		b.surfaceFormat = &format
	}
	if b.renderTargetFormat == nil {
		b.renderTargetFormat = b.surfaceFormat
	}

	err := b.surface.Configure(b.device, &wgpu.SurfaceConfiguration{
		Usage:       wgpu.TextureUsageRenderAttachment,
		Format:      *b.surfaceFormat,
		Width:       uint32(width),
		Height:      uint32(height),
		PresentMode: b.presentMode,
		AlphaMode:   gputypes.CompositeAlphaModeAuto,
	})
	if err != nil {
		panic(err)
	}

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
			Dimension:     gputypes.TextureDimension2D,
			Format:        *b.surfaceFormat,
			Usage:         wgpu.TextureUsageRenderAttachment,
		})
		if err != nil {
			panic(err)
		}
		b.msaaTextureView, err = b.device.CreateTextureView(msaaTexture, nil)
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
		Dimension:     gputypes.TextureDimension2D,
		Format:        wgpu.TextureFormatDepth24Plus,
		Usage:         wgpu.TextureUsageRenderAttachment,
	})
	if err != nil {
		panic(err)
	}
	b.depthTextureView, err = b.device.CreateTextureView(depthTexture, nil)
	if err != nil {
		panic(err)
	}

	// Build the cached render pass descriptor for the main render target.
	// When MSAA is enabled, View is the MSAA texture and ResolveTarget is
	// set per-frame to the swapchain view. When disabled, View is set
	// per-frame to the swapchain view and ResolveTarget remains nil.
	storeOp := gputypes.StoreOpStore
	if msaaEnabled {
		storeOp = gputypes.StoreOpDiscard // Don't store MSAA data, just resolve
	}
	b.renderPassDescriptor = &wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{
			{
				View:          b.msaaTextureView, // nil when MSAA is off; set in BeginFrame
				ResolveTarget: nil,               // set per-frame when MSAA is on
				LoadOp:        gputypes.LoadOpClear,
				StoreOp:       storeOp,
				ClearValue: wgpu.Color{
					R: 0.1, G: 0.1, B: 0.1, A: 1.0,
				},
			},
		},
		DepthStencilAttachment: &wgpu.RenderPassDepthStencilAttachment{
			View:            b.depthTextureView, // Persistent until resize
			DepthLoadOp:     gputypes.LoadOpClear,
			DepthStoreOp:    gputypes.StoreOpDiscard, // Depth not needed after resolving
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

func (b *wgpuRendererBackendImpl) CreateBuffer(label string, size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	return b.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: label,
		Size:  size,
		Usage: usage,
	})
}

func (b *wgpuRendererBackendImpl) CopyBufferToBuffer(src, dst *wgpu.Buffer, srcOffset, dstOffset, size uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.computeFrameEncoder == nil {
		return
	}
	b.computeFrameEncoder.CopyBufferToBuffer(src, srcOffset, dst, dstOffset, size)
}

func (b *wgpuRendererBackendImpl) ReadMappedBuffer(buf *wgpu.Buffer, offset, size uint64) ([]byte, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if buf == nil {
		return nil, fmt.Errorf("buffer is nil")
	}
	if b.queue == nil {
		return nil, fmt.Errorf("queue is nil")
	}
	if size > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("read size too large: %d", size)
	}

	out := make([]byte, int(size))
	if err := b.queue.ReadBuffer(buf, offset, out); err != nil {
		return nil, err
	}

	return out, nil
}

func (b *wgpuRendererBackendImpl) BeginComputeFrame() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Support nested Begin/End pairs: only create a new encoder on the
	// outermost call so that multiple Prepare* methods can be batched
	// into a single GPU submission when wrapped by an outer Begin/End.
	b.computeFrameDepth++
	if b.computeFrameDepth > 1 {
		return nil
	}

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		b.computeFrameDepth--
		return err
	}
	b.computeFrameEncoder = encoder
	return nil
}

func (b *wgpuRendererBackendImpl) EndComputeFrame() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.computeFrameEncoder == nil {
		b.computeFrameDepth = 0
		return
	}

	// Only submit on the outermost End call that balances the outermost Begin.
	b.computeFrameDepth--
	if b.computeFrameDepth > 0 {
		return
	}
	b.computeFrameDepth = 0

	commandBuffer, err := b.computeFrameEncoder.Finish()
	if err != nil {
		b.computeFrameEncoder = nil
		return
	}

	b.queue.Submit(commandBuffer)
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

	pass, err := b.computeFrameEncoder.BeginComputePass(nil)
	if err != nil {
		return
	}
	pass.SetPipeline(computePipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.Dispatch(workGroupCount[0], workGroupCount[1], workGroupCount[2])
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
		WGSL:  vertexShader.Source(),
	})
	if err != nil {
		return err
	}
	fs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: fragmentShader.Key(),
		WGSL:  fragmentShader.Source(),
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
						Format:    *b.renderTargetFormat,
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
			depthCompare := p.DepthCompare()
			if depthCompare == gputypes.CompareFunctionUndefined {
				depthCompare = gputypes.CompareFunctionLess
				if !p.DepthTestEnabled() {
					depthCompare = gputypes.CompareFunctionAlways
				}
			}
			return &wgpu.DepthStencilState{
				Format:              wgpu.TextureFormatDepth24Plus,
				DepthWriteEnabled:   p.DepthWriteEnabled(),
				DepthCompare:        depthCompare,
				DepthBias:           p.DepthBias(),
				DepthBiasSlopeScale: p.DepthBiasSlopeScale(),
				StencilFront: hal.StencilFaceState{
					Compare: gputypes.CompareFunctionAlways,
				},
				StencilBack: hal.StencilFaceState{
					Compare: gputypes.CompareFunctionAlways,
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
		WGSL:  computeShader.Source(),
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
		Label:      p.PipelineKey() + " Compute Pipeline",
		Layout:     layout,
		Module:     s,
		EntryPoint: computeShader.EntryPoint(),
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

		isTexture := isTextureBindingEntry(entry)
		isStorageTexture := isStorageTextureBindingEntry(entry)
		isSampler := isSamplerBindingEntry(entry)

		if isTexture || isStorageTexture {
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
			switch bufferBindingType(entry) {
			case gputypes.BufferBindingTypeUniform:
				usage = wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst
			case gputypes.BufferBindingTypeStorage:
				usage = wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst
			case gputypes.BufferBindingTypeReadOnlyStorage:
				usage = wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst
			}
			if overrideUsage, ok := bufferUsageOverrides[binding]; ok {
				usage |= overrideUsage
			}

			buf := provider.Buffer(binding)
			var bufSize uint64
			if buf == nil {
				var bufErr error
				if entry.Buffer != nil {
					bufSize = entry.Buffer.MinBindingSize
				}
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
				Size:    bufSize,
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

	format := wgpu.TextureFormatRGBA8UnormSrgb
	if stagingData.Linear {
		format = wgpu.TextureFormatRGBA8Unorm
	}

	tex, err := b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label:     provider.Label() + " Texture",
		Usage:     wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
		Dimension: gputypes.TextureDimension2D,
		Size: wgpu.Extent3D{
			Width:              stagingData.Width,
			Height:             stagingData.Height,
			DepthOrArrayLayers: 1,
		},
		Format:        format,
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
			Aspect:   gputypes.TextureAspectAll,
		},
		stagingData.Pixels,
		&wgpu.ImageDataLayout{
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

	view, err := b.device.CreateTextureView(tex, nil)
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
		Label:        provider.Label() + " Sampler",
		AddressModeU: common.Coalesce(samplerStagingData.AddressModeU, gputypes.AddressModeRepeat),
		AddressModeV: common.Coalesce(samplerStagingData.AddressModeV, gputypes.AddressModeRepeat),
		AddressModeW: common.Coalesce(samplerStagingData.AddressModeW, gputypes.AddressModeRepeat),
		MagFilter:    common.Coalesce(samplerStagingData.MagFilter, gputypes.FilterModeLinear),
		MinFilter:    common.Coalesce(samplerStagingData.MinFilter, gputypes.FilterModeLinear),
		MipmapFilter: common.Coalesce(samplerStagingData.MipmapFilter, wgpu.FilterMode(gputypes.MipmapFilterModeLinear)),
		LodMinClamp:  common.Coalesce(samplerStagingData.LodMinClamp, 0.0),
		LodMaxClamp:  common.Coalesce(samplerStagingData.LodMaxClamp, 32.0),
		Anisotropy:   common.Coalesce(samplerStagingData.MaxAnisotropy, 1),
		Compare:      samplerStagingData.Compare,
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

func (b *wgpuRendererBackendImpl) WriteRawBuffer(buf *wgpu.Buffer, offset uint64, data []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if buf == nil {
		return
	}
	b.queue.WriteBuffer(buf, offset, data)
}

func (b *wgpuRendererBackendImpl) WriteTexture(tex *wgpu.Texture, data []byte, width, height, bytesPerRow uint32) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.queue.WriteTexture(
		&wgpu.ImageCopyTexture{
			Texture:  tex,
			MipLevel: 0,
			Origin:   wgpu.Origin3D{},
			Aspect:   gputypes.TextureAspectAll,
		},
		data,
		&wgpu.ImageDataLayout{
			Offset:       0,
			BytesPerRow:  bytesPerRow,
			RowsPerImage: height,
		},
		&wgpu.Extent3D{
			Width:              width,
			Height:             height,
			DepthOrArrayLayers: 1,
		},
	)
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

	surfaceTexture, sub, err := b.surface.GetCurrentTexture()
	if err != nil {
		return err
	} else if sub {
		return fmt.Errorf("suboptimal surface texture acquired")
	}

	view, err := surfaceTexture.CreateView(nil)
	if err != nil {
		return err
	}

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		view.Release()
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
	pass, err := encoder.BeginRenderPass(b.renderPassDescriptor)
	if err != nil {
		view.Release()
		return err
	}

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

	b.framePass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0)
	b.framePass.SetIndexBuffer(meshProvider.IndexBuffer(), gputypes.IndexFormatUint32, 0)
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

	b.framePass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0)
	b.framePass.SetIndexBuffer(meshProvider.IndexBuffer(), gputypes.IndexFormatUint32, 0)
	b.framePass.DrawIndexedIndirect(indirectBuffer, 0)
}

func (b *wgpuRendererBackendImpl) EndFrame() {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.framePass.End()

	commandBuffer, err := b.frameEncoder.Finish()
	if err != nil {
		if b.frameView != nil {
			b.frameView.Release()
		}
		b.frameEncoder = nil
		b.framePass = nil
		b.frameSurface = nil
		b.frameView = nil
		return
	}

	b.queue.Submit(commandBuffer)

	b.frameEncoder = nil
	b.framePass = nil
}

func (b *wgpuRendererBackendImpl) Present() {
	b.mu.Lock()
	defer b.mu.Unlock()

	var surfaceTexture *wgpu.SurfaceTexture
	if b.frameSurface != nil {
		surfaceTexture = b.frameSurface
	} else if b.compositionSurface != nil {
		surfaceTexture = b.compositionSurface
	} else {
		return
	}

	b.surface.Present(surfaceTexture)

	// Release normal frame state.
	if b.frameView != nil {
		b.frameView.Release()
		b.frameView = nil
	}
	if b.frameSurface != nil {
		b.frameSurface = nil
	}

	// Release composition frame state.
	if b.compositionView != nil {
		b.compositionView.Release()
		b.compositionView = nil
	}
	if b.compositionSurface != nil {
		b.compositionSurface = nil
	}
}

func (b *wgpuRendererBackendImpl) BeginGeometryFrame() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.geometryFrameDepth++
	if b.geometryFrameDepth > 1 {
		return nil
	}

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		b.geometryFrameDepth--
		return err
	}
	b.geometryFrameEncoder = encoder
	return nil
}

func (b *wgpuRendererBackendImpl) EndGeometryFrame() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.geometryFrameDepth <= 0 {
		return
	}

	b.geometryFrameDepth--
	if b.geometryFrameDepth > 0 {
		return
	}

	if b.geometryFrameEncoder == nil {
		return
	}

	commandBuffer, err := b.geometryFrameEncoder.Finish()
	if err != nil {
		b.geometryFrameEncoder = nil
		return
	}

	b.queue.Submit(commandBuffer)

	b.geometryFrameEncoder = nil
}

func (b *wgpuRendererBackendImpl) BeginShadowFrame() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// When a geometry frame is active, alias the shared encoder instead of creating a new one.
	if b.geometryFrameEncoder != nil {
		b.shadowFrameEncoder = b.geometryFrameEncoder
		return nil
	}

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	b.shadowFrameEncoder = encoder
	return nil
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

	b.shadowPass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0)
	b.shadowPass.SetIndexBuffer(meshProvider.IndexBuffer(), gputypes.IndexFormatUint32, 0)
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

	b.shadowPass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0)
	b.shadowPass.SetIndexBuffer(meshProvider.IndexBuffer(), gputypes.IndexFormatUint32, 0)
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

	// When a geometry frame is active, just clear the alias; the geometry frame handles submission.
	if b.geometryFrameEncoder != nil {
		b.shadowFrameEncoder = nil
		return
	}

	commandBuffer, err := b.shadowFrameEncoder.Finish()
	if err != nil {
		b.shadowFrameEncoder = nil
		return
	}

	b.queue.Submit(commandBuffer)

	b.shadowFrameEncoder = nil
}

func (b *wgpuRendererBackendImpl) CreateVSMTextures(width, height int) (
	vsmView *wgpu.TextureView, vsmTex *wgpu.Texture,
	scratchView *wgpu.TextureView, scratchTex *wgpu.Texture,
	depthView *wgpu.TextureView, depthTex *wgpu.Texture,
	err error,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	w := uint32(width)
	h := uint32(height)

	// VSM moments texture: RG32Float for (depth, depth²).
	// StorageBinding is required for the blur compute pass.
	vsmTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "VSM Moments Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRG32Float,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding | wgpu.TextureUsageStorageBinding,
	})
	if err != nil {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create VSM moments texture: %w", err)
	}

	vsmView, err = b.device.CreateTextureView(vsmTex, nil)
	if err != nil {
		vsmTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create VSM moments texture view: %w", err)
	}

	// Scratch texture for the intermediate separable blur result.
	scratchTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "VSM Scratch Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRG32Float,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageStorageBinding,
	})
	if err != nil {
		vsmView.Release()
		vsmTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create VSM scratch texture: %w", err)
	}

	scratchView, err = b.device.CreateTextureView(scratchTex, nil)
	if err != nil {
		scratchTex.Release()
		vsmView.Release()
		vsmTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create VSM scratch texture view: %w", err)
	}

	// Auxiliary depth texture for hardware z-testing during the VSM render pass.
	depthTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "VSM Auxiliary Depth Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        wgpu.TextureFormatDepth32Float,
		Usage:         wgpu.TextureUsageRenderAttachment,
	})
	if err != nil {
		scratchView.Release()
		scratchTex.Release()
		vsmView.Release()
		vsmTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create VSM auxiliary depth texture: %w", err)
	}

	depthView, err = b.device.CreateTextureView(depthTex, nil)
	if err != nil {
		depthTex.Release()
		scratchView.Release()
		scratchTex.Release()
		vsmView.Release()
		vsmTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create VSM auxiliary depth texture view: %w", err)
	}

	return vsmView, vsmTex, scratchView, scratchTex, depthView, depthTex, nil
}

func (b *wgpuRendererBackendImpl) CreateLinearSampler() (*wgpu.Sampler, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	samp, err := b.device.CreateSampler(&wgpu.SamplerDescriptor{
		Label:        "VSM Linear Sampler",
		AddressModeU: gputypes.AddressModeClampToEdge,
		AddressModeV: gputypes.AddressModeClampToEdge,
		AddressModeW: gputypes.AddressModeClampToEdge,
		MagFilter:    gputypes.FilterModeLinear,
		MinFilter:    gputypes.FilterModeLinear,
		MipmapFilter: wgpu.FilterMode(gputypes.MipmapFilterModeNearest),
		Anisotropy:   1,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create linear sampler: %w", err)
	}

	return samp, nil
}

func (b *wgpuRendererBackendImpl) RegisterVSMShadowPipeline(p pipeline.Pipeline) error {
	vertexShader := p.Shader(shader.ShaderTypeVertex)
	if vertexShader == nil {
		return errors.New("vertex shader must be set to create a VSM shadow pipeline")
	}

	fragmentShader := p.Shader(shader.ShaderTypeFragment)
	if fragmentShader == nil {
		return errors.New("fragment shader must be set to create a VSM shadow pipeline")
	}

	vs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: vertexShader.Key(),
		WGSL:  vertexShader.Source(),
	})
	if err != nil {
		return fmt.Errorf("vsm shadow: failed to create vertex shader module: %w", err)
	}

	fs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: fragmentShader.Key(),
		WGSL:  fragmentShader.Source(),
	})
	if err != nil {
		return fmt.Errorf("vsm shadow: failed to create fragment shader module: %w", err)
	}

	// Merge bind group layouts from both vertex and fragment shaders.
	mergedDescriptors := mergeBindGroupLayouts(
		vertexShader.BindGroupLayoutDescriptors(),
		fragmentShader.BindGroupLayoutDescriptors(),
	)

	maxGroup := -1
	for g := range mergedDescriptors {
		if g > maxGroup {
			maxGroup = g
		}
	}
	bindGroupLayouts := make([]*wgpu.BindGroupLayout, maxGroup+1)
	for g, desc := range mergedDescriptors {
		layout, layoutErr := b.device.CreateBindGroupLayout(&desc)
		if layoutErr != nil {
			return fmt.Errorf("vsm shadow: failed to create bind group layout for group %d: %w", g, layoutErr)
		}
		bindGroupLayouts[g] = layout
	}

	pipelineLayout, err := b.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            p.PipelineKey(),
		BindGroupLayouts: bindGroupLayouts,
	})
	if err != nil {
		return fmt.Errorf("vsm shadow: failed to create pipeline layout: %w", err)
	}

	vertexLayouts := make([]wgpu.VertexBufferLayout, 0, len(vertexShader.VertexLayouts()))
	for i := range vertexShader.VertexLayouts() {
		vertexLayouts = append(vertexLayouts, vertexShader.VertexLayout(i)...)
	}

	created, err := b.device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  p.PipelineKey() + " VSM Shadow Pipeline",
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
				{
					Format:    gputypes.TextureFormatRG32Float,
					WriteMask: gputypes.ColorWriteMaskAll,
				},
			},
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  gputypes.PrimitiveTopologyTriangleList,
			FrontFace: gputypes.FrontFaceCCW,
			CullMode:  p.CullMode(),
		},
		Multisample: wgpu.MultisampleState{
			Count: 1,
			Mask:  0xFFFFFFFF,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            wgpu.TextureFormatDepth32Float,
			DepthWriteEnabled: true,
			DepthCompare:      gputypes.CompareFunctionLess,
			// No depth bias — the partial-derivative moment computation replaces it.
			DepthBias:           0,
			DepthBiasSlopeScale: 0,
			StencilFront: hal.StencilFaceState{
				Compare: gputypes.CompareFunctionAlways,
			},
			StencilBack: hal.StencilFaceState{
				Compare: gputypes.CompareFunctionAlways,
			},
		},
	})
	if err != nil {
		return fmt.Errorf("vsm shadow: failed to create render pipeline: %w", err)
	}

	p.SetRenderPipeline(created)
	return nil
}

func (b *wgpuRendererBackendImpl) CreateSATTextures(width, height int) (
	satAView *wgpu.TextureView, satATex *wgpu.Texture,
	satBView *wgpu.TextureView, satBTex *wgpu.Texture,
	err error,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	w := uint32(width)
	h := uint32(height)

	// SAT texture A: RGBA32Float for precision-distributed moments.
	satATex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "SAT Texture A",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRGBA32Float,
		Usage:         gputypes.TextureUsageTextureBinding | gputypes.TextureUsageStorageBinding,
	})
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to create SAT texture A: %w", err)
	}

	satAView, err = b.device.CreateTextureView(satATex, nil)
	if err != nil {
		satATex.Release()
		return nil, nil, nil, nil, fmt.Errorf("failed to create SAT texture A view: %w", err)
	}

	// SAT texture B: RGBA32Float ping-pong target.
	satBTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "SAT Texture B",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRGBA32Float,
		Usage:         gputypes.TextureUsageTextureBinding | gputypes.TextureUsageStorageBinding,
	})
	if err != nil {
		satAView.Release()
		satATex.Release()
		return nil, nil, nil, nil, fmt.Errorf("failed to create SAT texture B: %w", err)
	}

	satBView, err = b.device.CreateTextureView(satBTex, nil)
	if err != nil {
		satBTex.Release()
		satAView.Release()
		satATex.Release()
		return nil, nil, nil, nil, fmt.Errorf("failed to create SAT texture B view: %w", err)
	}

	return satAView, satATex, satBView, satBTex, nil
}

func (b *wgpuRendererBackendImpl) BeginVSMShadowPass(vsmView *wgpu.TextureView, depthView *wgpu.TextureView) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.shadowFrameEncoder == nil {
		return
	}

	pass, err := b.shadowFrameEncoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{
			{
				View:       vsmView,
				LoadOp:     gputypes.LoadOpClear,
				StoreOp:    gputypes.StoreOpStore,
				ClearValue: wgpu.Color{R: 1.0, G: 1.0, B: 0.0, A: 0.0},
			},
		},
		DepthStencilAttachment: &wgpu.RenderPassDepthStencilAttachment{
			View:            depthView,
			DepthLoadOp:     gputypes.LoadOpClear,
			DepthStoreOp:    gputypes.StoreOpStore,
			DepthClearValue: 1.0,
		},
	})
	if err != nil {
		return
	}
	b.shadowPass = pass
}

func (b *wgpuRendererBackendImpl) BeginGBufferFrame() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	// When a geometry frame is active, alias the shared encoder instead of creating a new one.
	if b.geometryFrameEncoder != nil {
		b.gbufferFrameEncoder = b.geometryFrameEncoder
		return nil
	}

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	b.gbufferFrameEncoder = encoder
	return nil
}

func (b *wgpuRendererBackendImpl) BeginGBufferPass(normView, albedoView, depthView *wgpu.TextureView) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.gbufferFrameEncoder == nil {
		return
	}

	pass, err := b.gbufferFrameEncoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{
			{
				View:       normView,
				LoadOp:     gputypes.LoadOpClear,
				StoreOp:    gputypes.StoreOpStore,
				ClearValue: wgpu.Color{R: 0.5, G: 0.5, B: 1.0, A: 0.0},
			},
			{
				View:       albedoView,
				LoadOp:     gputypes.LoadOpClear,
				StoreOp:    gputypes.StoreOpStore,
				ClearValue: wgpu.Color{R: 0.0, G: 0.0, B: 0.0, A: 0.0},
			},
		},
		DepthStencilAttachment: &wgpu.RenderPassDepthStencilAttachment{
			View:            depthView,
			DepthLoadOp:     gputypes.LoadOpClear,
			DepthStoreOp:    gputypes.StoreOpStore,
			DepthClearValue: 1.0,
		},
	})
	if err != nil {
		return
	}
	b.gbufferPass = pass
}

func (b *wgpuRendererBackendImpl) GBufferDrawCall(
	p pipeline.Pipeline,
	meshProvider bind_group_provider.BindGroupProvider,
	instanceCount uint32,
	bindGroups []bind_group_provider.BindGroupProvider,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.gbufferPass == nil {
		return
	}

	renderPipeline := p.Pipeline().(*wgpu.RenderPipeline)
	b.gbufferPass.SetPipeline(renderPipeline)

	for i, bg := range bindGroups {
		b.gbufferPass.SetBindGroup(uint32(i), bg.BindGroup(), nil)
	}

	b.gbufferPass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0)
	b.gbufferPass.SetIndexBuffer(meshProvider.IndexBuffer(), gputypes.IndexFormatUint32, 0)
	b.gbufferPass.DrawIndexed(uint32(meshProvider.IndexCount()), instanceCount, 0, 0, 0)
}

func (b *wgpuRendererBackendImpl) GBufferDrawCallIndirect(
	p pipeline.Pipeline,
	meshProvider bind_group_provider.BindGroupProvider,
	indirectBuffer *wgpu.Buffer,
	bindGroups []bind_group_provider.BindGroupProvider,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.gbufferPass == nil {
		return
	}

	renderPipeline := p.Pipeline().(*wgpu.RenderPipeline)
	b.gbufferPass.SetPipeline(renderPipeline)

	for i, bg := range bindGroups {
		b.gbufferPass.SetBindGroup(uint32(i), bg.BindGroup(), nil)
	}

	b.gbufferPass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0)
	b.gbufferPass.SetIndexBuffer(meshProvider.IndexBuffer(), gputypes.IndexFormatUint32, 0)
	b.gbufferPass.DrawIndexedIndirect(indirectBuffer, 0)
}

func (b *wgpuRendererBackendImpl) EndGBufferPass() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.gbufferPass == nil {
		return
	}

	b.gbufferPass.End()
	b.gbufferPass = nil
}

func (b *wgpuRendererBackendImpl) EndGBufferFrame() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.gbufferFrameEncoder == nil {
		return
	}

	// When a geometry frame is active, just clear the alias; the geometry frame handles submission.
	if b.geometryFrameEncoder != nil {
		b.gbufferFrameEncoder = nil
		return
	}

	commandBuffer, err := b.gbufferFrameEncoder.Finish()
	if err != nil {
		b.gbufferFrameEncoder = nil
		return
	}

	b.queue.Submit(commandBuffer)

	b.gbufferFrameEncoder = nil
}

func (b *wgpuRendererBackendImpl) CreateGBufferTextures(width, height int) (
	normView *wgpu.TextureView, normTex *wgpu.Texture,
	albedoView *wgpu.TextureView, albedoTex *wgpu.Texture,
	depthView *wgpu.TextureView, depthTex *wgpu.Texture,
	err error,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	w := uint32(width)
	h := uint32(height)

	// Normal MRT: RGBA16Float for packed normals [0,1] XYZ + roughness W.
	normTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "GBuffer Normal Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRGBA16Float,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create G-Buffer normal texture: %w", err)
	}

	normView, err = b.device.CreateTextureView(normTex, nil)
	if err != nil {
		normTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create G-Buffer normal texture view: %w", err)
	}

	// Albedo MRT: RGBA8Unorm for albedo RGB + metallic A.
	albedoTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "GBuffer Albedo Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        wgpu.TextureFormatRGBA8Unorm,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		normView.Release()
		normTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create G-Buffer albedo texture: %w", err)
	}

	albedoView, err = b.device.CreateTextureView(albedoTex, nil)
	if err != nil {
		albedoTex.Release()
		normView.Release()
		normTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create G-Buffer albedo texture view: %w", err)
	}

	// Depth for hardware z-testing during the G-Buffer pass.
	// TextureBinding is required so SSR and SSAO compute shaders can sample it.
	depthTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "GBuffer Depth Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        wgpu.TextureFormatDepth24Plus,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		albedoView.Release()
		albedoTex.Release()
		normView.Release()
		normTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create G-Buffer depth texture: %w", err)
	}

	depthView, err = b.device.CreateTextureView(depthTex, nil)
	if err != nil {
		depthTex.Release()
		albedoView.Release()
		albedoTex.Release()
		normView.Release()
		normTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create G-Buffer depth texture view: %w", err)
	}

	return normView, normTex, albedoView, albedoTex, depthView, depthTex, nil
}

func (b *wgpuRendererBackendImpl) RegisterGBufferPipeline(p pipeline.Pipeline) error {
	vertexShader := p.Shader(shader.ShaderTypeVertex)
	if vertexShader == nil {
		return errors.New("vertex shader must be set to create a G-Buffer pipeline")
	}

	fragmentShader := p.Shader(shader.ShaderTypeFragment)
	if fragmentShader == nil {
		return errors.New("fragment shader must be set to create a G-Buffer pipeline")
	}

	vs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: vertexShader.Key(),
		WGSL:  vertexShader.Source(),
	})
	if err != nil {
		return fmt.Errorf("gbuffer: failed to create vertex shader module: %w", err)
	}

	fs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: fragmentShader.Key(),
		WGSL:  fragmentShader.Source(),
	})
	if err != nil {
		return fmt.Errorf("gbuffer: failed to create fragment shader module: %w", err)
	}

	mergedDescriptors := mergeBindGroupLayouts(
		vertexShader.BindGroupLayoutDescriptors(),
		fragmentShader.BindGroupLayoutDescriptors(),
	)

	maxGroup := -1
	for g := range mergedDescriptors {
		if g > maxGroup {
			maxGroup = g
		}
	}
	bindGroupLayouts := make([]*wgpu.BindGroupLayout, maxGroup+1)
	for g, desc := range mergedDescriptors {
		layout, layoutErr := b.device.CreateBindGroupLayout(&desc)
		if layoutErr != nil {
			return fmt.Errorf("gbuffer: failed to create bind group layout for group %d: %w", g, layoutErr)
		}
		bindGroupLayouts[g] = layout
	}

	pipelineLayout, err := b.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            p.PipelineKey(),
		BindGroupLayouts: bindGroupLayouts,
	})
	if err != nil {
		return fmt.Errorf("gbuffer: failed to create pipeline layout: %w", err)
	}

	vertexLayouts := make([]wgpu.VertexBufferLayout, 0, len(vertexShader.VertexLayouts()))
	for i := range vertexShader.VertexLayouts() {
		vertexLayouts = append(vertexLayouts, vertexShader.VertexLayout(i)...)
	}

	created, err := b.device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  p.PipelineKey() + " GBuffer Pipeline",
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
				{
					Format:    gputypes.TextureFormatRGBA16Float,
					WriteMask: gputypes.ColorWriteMaskAll,
				},
				{
					Format:    wgpu.TextureFormatRGBA8Unorm,
					WriteMask: gputypes.ColorWriteMaskAll,
				},
			},
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  gputypes.PrimitiveTopologyTriangleList,
			FrontFace: gputypes.FrontFaceCCW,
			CullMode:  p.CullMode(),
		},
		Multisample: wgpu.MultisampleState{
			Count: 1,
			Mask:  0xFFFFFFFF,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            wgpu.TextureFormatDepth24Plus,
			DepthWriteEnabled: true,
			DepthCompare:      gputypes.CompareFunctionLess,
			StencilFront: hal.StencilFaceState{
				Compare: gputypes.CompareFunctionAlways,
			},
			StencilBack: hal.StencilFaceState{
				Compare: gputypes.CompareFunctionAlways,
			},
		},
	})
	if err != nil {
		return fmt.Errorf("gbuffer: failed to create render pipeline: %w", err)
	}

	p.SetRenderPipeline(created)
	return nil
}

func (b *wgpuRendererBackendImpl) CreateSSAOTextures(width, height int) (
	rawView *wgpu.TextureView, rawTex *wgpu.Texture,
	blurredView *wgpu.TextureView, blurredTex *wgpu.Texture,
	scratchView *wgpu.TextureView, scratchTex *wgpu.Texture,
	noiseView *wgpu.TextureView, noiseTex *wgpu.Texture,
	err error,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	w := uint32(width)
	h := uint32(height)

	// Raw SSAO output: R32Float, written by the SSAO compute shader.
	// StorageBinding for compute writes, TextureBinding for blur reads.
	// R32Float is used instead of R8Unorm because R8Unorm does not support StorageBinding.
	rawTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "SSAO Raw Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatR32Float,
		Usage:         wgpu.TextureUsageStorageBinding | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create SSAO raw texture: %w", err)
	}

	rawView, err = b.device.CreateTextureView(rawTex, nil)
	if err != nil {
		rawTex.Release()
		return nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create SSAO raw texture view: %w", err)
	}

	// Blurred SSAO output: R32Float, bound to the lit fragment shader.
	// R32Float is used instead of R8Unorm because R8Unorm does not support StorageBinding.
	blurredTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "SSAO Blurred Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatR32Float,
		Usage:         wgpu.TextureUsageStorageBinding | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		rawView.Release()
		rawTex.Release()
		return nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create SSAO blurred texture: %w", err)
	}

	blurredView, err = b.device.CreateTextureView(blurredTex, nil)
	if err != nil {
		blurredTex.Release()
		rawView.Release()
		rawTex.Release()
		return nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create SSAO blurred texture view: %w", err)
	}

	// Scratch texture for the separable bilateral blur intermediate step.
	// R32Float is used instead of R8Unorm because R8Unorm does not support StorageBinding.
	scratchTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "SSAO Scratch Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatR32Float,
		Usage:         wgpu.TextureUsageStorageBinding | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		blurredView.Release()
		blurredTex.Release()
		rawView.Release()
		rawTex.Release()
		return nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create SSAO scratch texture: %w", err)
	}

	scratchView, err = b.device.CreateTextureView(scratchTex, nil)
	if err != nil {
		scratchTex.Release()
		blurredView.Release()
		blurredTex.Release()
		rawView.Release()
		rawTex.Release()
		return nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create SSAO scratch texture view: %w", err)
	}

	// 4×4 noise texture: RGBA16Float for random rotation vectors.
	// Only needs TextureBinding — written once via queue.WriteTexture.
	noiseTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "SSAO Noise Texture",
		Size: wgpu.Extent3D{
			Width:              4,
			Height:             4,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRGBA16Float,
		Usage:         wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopyDst,
	})
	if err != nil {
		scratchView.Release()
		scratchTex.Release()
		blurredView.Release()
		blurredTex.Release()
		rawView.Release()
		rawTex.Release()
		return nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create SSAO noise texture: %w", err)
	}

	noiseView, err = b.device.CreateTextureView(noiseTex, nil)
	if err != nil {
		noiseTex.Release()
		scratchView.Release()
		scratchTex.Release()
		blurredView.Release()
		blurredTex.Release()
		rawView.Release()
		rawTex.Release()
		return nil, nil, nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create SSAO noise texture view: %w", err)
	}

	return rawView, rawTex, blurredView, blurredTex, scratchView, scratchTex, noiseView, noiseTex, nil
}

func (b *wgpuRendererBackendImpl) CreateProbeBakeTextures(resolution int) (
	colorView *wgpu.TextureView, colorTex *wgpu.Texture,
	depthView *wgpu.TextureView, depthTex *wgpu.Texture,
	err error,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	size := uint32(resolution)

	// RGBA8Unorm color texture for capturing lit radiance from each cubemap face.
	// TextureBinding for SH projection compute reads, CopySrc for potential readback.
	colorTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "Probe Bake Color Texture",
		Size: wgpu.Extent3D{
			Width:              size,
			Height:             size,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        wgpu.TextureFormatRGBA8Unorm,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding | wgpu.TextureUsageCopySrc,
	})
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to create probe bake color texture: %w", err)
	}

	colorView, err = b.device.CreateTextureView(colorTex, nil)
	if err != nil {
		colorTex.Release()
		return nil, nil, nil, nil, fmt.Errorf("failed to create probe bake color texture view: %w", err)
	}

	// Depth24Plus for hardware z-testing during the bake pass.
	depthTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "Probe Bake Depth Texture",
		Size: wgpu.Extent3D{
			Width:              size,
			Height:             size,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        wgpu.TextureFormatDepth24Plus,
		Usage:         wgpu.TextureUsageRenderAttachment,
	})
	if err != nil {
		colorView.Release()
		colorTex.Release()
		return nil, nil, nil, nil, fmt.Errorf("failed to create probe bake depth texture: %w", err)
	}

	depthView, err = b.device.CreateTextureView(depthTex, nil)
	if err != nil {
		depthTex.Release()
		colorView.Release()
		colorTex.Release()
		return nil, nil, nil, nil, fmt.Errorf("failed to create probe bake depth texture view: %w", err)
	}

	return colorView, colorTex, depthView, depthTex, nil
}

func (b *wgpuRendererBackendImpl) RegisterProbeBakePipeline(p pipeline.Pipeline) error {
	vertexShader := p.Shader(shader.ShaderTypeVertex)
	if vertexShader == nil {
		return errors.New("vertex shader must be set to create a probe bake pipeline")
	}

	fragmentShader := p.Shader(shader.ShaderTypeFragment)
	if fragmentShader == nil {
		return errors.New("fragment shader must be set to create a probe bake pipeline")
	}

	vs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: vertexShader.Key(),
		WGSL:  vertexShader.Source(),
	})
	if err != nil {
		return fmt.Errorf("probe bake: failed to create vertex shader module: %w", err)
	}

	fs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: fragmentShader.Key(),
		WGSL:  fragmentShader.Source(),
	})
	if err != nil {
		return fmt.Errorf("probe bake: failed to create fragment shader module: %w", err)
	}

	mergedDescriptors := mergeBindGroupLayouts(
		vertexShader.BindGroupLayoutDescriptors(),
		fragmentShader.BindGroupLayoutDescriptors(),
	)

	maxGroup := -1
	for g := range mergedDescriptors {
		if g > maxGroup {
			maxGroup = g
		}
	}
	bindGroupLayouts := make([]*wgpu.BindGroupLayout, maxGroup+1)
	for g, desc := range mergedDescriptors {
		layout, layoutErr := b.device.CreateBindGroupLayout(&desc)
		if layoutErr != nil {
			return fmt.Errorf("probe bake: failed to create bind group layout for group %d: %w", g, layoutErr)
		}
		bindGroupLayouts[g] = layout
	}

	pipelineLayout, err := b.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            p.PipelineKey(),
		BindGroupLayouts: bindGroupLayouts,
	})
	if err != nil {
		return fmt.Errorf("probe bake: failed to create pipeline layout: %w", err)
	}

	vertexLayouts := make([]wgpu.VertexBufferLayout, 0, len(vertexShader.VertexLayouts()))
	for i := range vertexShader.VertexLayouts() {
		vertexLayouts = append(vertexLayouts, vertexShader.VertexLayout(i)...)
	}

	created, err := b.device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  p.PipelineKey() + " Probe Bake Pipeline",
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
				{
					Format:    wgpu.TextureFormatRGBA8Unorm,
					WriteMask: gputypes.ColorWriteMaskAll,
				},
			},
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  gputypes.PrimitiveTopologyTriangleList,
			FrontFace: gputypes.FrontFaceCCW,
			CullMode:  p.CullMode(),
		},
		Multisample: wgpu.MultisampleState{
			Count: 1,
			Mask:  0xFFFFFFFF,
		},
		DepthStencil: &wgpu.DepthStencilState{
			Format:            wgpu.TextureFormatDepth24Plus,
			DepthWriteEnabled: true,
			DepthCompare:      gputypes.CompareFunctionLess,
			StencilFront: hal.StencilFaceState{
				Compare: gputypes.CompareFunctionAlways,
			},
			StencilBack: hal.StencilFaceState{
				Compare: gputypes.CompareFunctionAlways,
			},
		},
	})
	if err != nil {
		return fmt.Errorf("probe bake: failed to create render pipeline: %w", err)
	}

	p.SetRenderPipeline(created)
	return nil
}

func (b *wgpuRendererBackendImpl) BeginProbeBakeFrame() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	b.probeBakeFrameEncoder = encoder
	return nil
}

func (b *wgpuRendererBackendImpl) BeginProbeBakePass(colorView, depthView *wgpu.TextureView) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.probeBakeFrameEncoder == nil {
		return
	}

	pass, err := b.probeBakeFrameEncoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{
			{
				View:       colorView,
				LoadOp:     gputypes.LoadOpClear,
				StoreOp:    gputypes.StoreOpStore,
				ClearValue: wgpu.Color{R: 0.0, G: 0.0, B: 0.0, A: 1.0},
			},
		},
		DepthStencilAttachment: &wgpu.RenderPassDepthStencilAttachment{
			View:            depthView,
			DepthLoadOp:     gputypes.LoadOpClear,
			DepthStoreOp:    gputypes.StoreOpStore,
			DepthClearValue: 1.0,
		},
	})
	if err != nil {
		return
	}
	b.probeBakePass = pass
}

func (b *wgpuRendererBackendImpl) ProbeBakeDrawCall(
	p pipeline.Pipeline,
	meshProvider bind_group_provider.BindGroupProvider,
	instanceCount uint32,
	bindGroups []bind_group_provider.BindGroupProvider,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.probeBakePass == nil {
		return
	}

	renderPipeline := p.Pipeline().(*wgpu.RenderPipeline)
	b.probeBakePass.SetPipeline(renderPipeline)

	for i, bg := range bindGroups {
		b.probeBakePass.SetBindGroup(uint32(i), bg.BindGroup(), nil)
	}

	b.probeBakePass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0)
	b.probeBakePass.SetIndexBuffer(meshProvider.IndexBuffer(), gputypes.IndexFormatUint32, 0)
	b.probeBakePass.DrawIndexed(uint32(meshProvider.IndexCount()), instanceCount, 0, 0, 0)
}

func (b *wgpuRendererBackendImpl) ProbeBakeDrawCallIndirect(
	p pipeline.Pipeline,
	meshProvider bind_group_provider.BindGroupProvider,
	indirectBuffer *wgpu.Buffer,
	bindGroups []bind_group_provider.BindGroupProvider,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.probeBakePass == nil {
		return
	}

	renderPipeline := p.Pipeline().(*wgpu.RenderPipeline)
	b.probeBakePass.SetPipeline(renderPipeline)

	for i, bg := range bindGroups {
		b.probeBakePass.SetBindGroup(uint32(i), bg.BindGroup(), nil)
	}

	b.probeBakePass.SetVertexBuffer(0, meshProvider.VertexBuffer(), 0)
	b.probeBakePass.SetIndexBuffer(meshProvider.IndexBuffer(), gputypes.IndexFormatUint32, 0)
	b.probeBakePass.DrawIndexedIndirect(indirectBuffer, 0)
}

func (b *wgpuRendererBackendImpl) EndProbeBakePass() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.probeBakePass == nil {
		return
	}

	b.probeBakePass.End()
	b.probeBakePass = nil
}

func (b *wgpuRendererBackendImpl) EndProbeBakeFrame() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.probeBakeFrameEncoder == nil {
		return
	}

	commandBuffer, err := b.probeBakeFrameEncoder.Finish()
	if err != nil {
		b.probeBakeFrameEncoder = nil
		return
	}

	b.queue.Submit(commandBuffer)

	b.probeBakeFrameEncoder = nil
}

func (b *wgpuRendererBackendImpl) BeginHDRFrame(colorView, resolveView, depthView *wgpu.TextureView, sampleCount uint32) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}

	colorAttachment := wgpu.RenderPassColorAttachment{
		LoadOp:     gputypes.LoadOpClear,
		ClearValue: b.renderPassDescriptor.ColorAttachments[0].ClearValue,
	}
	if sampleCount > 1 && resolveView != nil {
		colorAttachment.View = colorView
		colorAttachment.ResolveTarget = resolveView
		colorAttachment.StoreOp = gputypes.StoreOpDiscard
	} else {
		colorAttachment.View = colorView
		colorAttachment.StoreOp = gputypes.StoreOpStore
	}

	pass, err := encoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{colorAttachment},
		DepthStencilAttachment: &wgpu.RenderPassDepthStencilAttachment{
			View:            depthView,
			DepthLoadOp:     gputypes.LoadOpClear,
			DepthStoreOp:    gputypes.StoreOpDiscard,
			DepthClearValue: 1.0,
		},
	})
	if err != nil {
		return err
	}

	b.frameEncoder = encoder
	b.framePass = pass
	return nil
}

func (b *wgpuRendererBackendImpl) CreateCompositionTextures(width, height int, sampleCount uint32) (
	hdrView *wgpu.TextureView, hdrTex *wgpu.Texture,
	msaaView *wgpu.TextureView, msaaTex *wgpu.Texture,
	depthView *wgpu.TextureView, depthTex *wgpu.Texture,
	err error,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	w := uint32(width)
	h := uint32(height)

	// RGBA16Float HDR texture: resolve target when MSAA is active, direct
	// render target otherwise. TextureBinding for downstream SSR compute
	// and the composition fragment reads.
	hdrTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "Composition HDR Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRGBA16Float,
		Usage:         wgpu.TextureUsageRenderAttachment | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create composition HDR texture: %w", err)
	}

	hdrView, err = b.device.CreateTextureView(hdrTex, nil)
	if err != nil {
		hdrTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create composition HDR texture view: %w", err)
	}

	// MSAA texture: only created when sampleCount > 1, used as the View in
	// the HDR render pass with the HDR texture as ResolveTarget.
	if sampleCount > 1 {
		msaaTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
			Label: "Composition MSAA Texture",
			Size: wgpu.Extent3D{
				Width:              w,
				Height:             h,
				DepthOrArrayLayers: 1,
			},
			MipLevelCount: 1,
			SampleCount:   sampleCount,
			Dimension:     gputypes.TextureDimension2D,
			Format:        gputypes.TextureFormatRGBA16Float,
			Usage:         wgpu.TextureUsageRenderAttachment,
		})
		if err != nil {
			hdrView.Release()
			hdrTex.Release()
			return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create composition MSAA texture: %w", err)
		}

		msaaView, err = b.device.CreateTextureView(msaaTex, nil)
		if err != nil {
			msaaTex.Release()
			hdrView.Release()
			hdrTex.Release()
			return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create composition MSAA texture view: %w", err)
		}
	}

	// Depth texture matching the MSAA sample count of the lit pass.
	depthTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "Composition Depth Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   sampleCount,
		Dimension:     gputypes.TextureDimension2D,
		Format:        wgpu.TextureFormatDepth24Plus,
		Usage:         wgpu.TextureUsageRenderAttachment,
	})
	if err != nil {
		if msaaView != nil {
			msaaView.Release()
		}
		if msaaTex != nil {
			msaaTex.Release()
		}
		hdrView.Release()
		hdrTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create composition depth texture: %w", err)
	}

	depthView, err = b.device.CreateTextureView(depthTex, nil)
	if err != nil {
		depthTex.Release()
		if msaaView != nil {
			msaaView.Release()
		}
		if msaaTex != nil {
			msaaTex.Release()
		}
		hdrView.Release()
		hdrTex.Release()
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("failed to create composition depth texture view: %w", err)
	}

	return hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex, nil
}

func (b *wgpuRendererBackendImpl) CreateSSRTextures(width, height int) (
	ssrView *wgpu.TextureView, ssrTex *wgpu.Texture, err error,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	w := uint32(width)
	h := uint32(height)

	// RGBA16Float SSR output: written by the SSR compute shader (StorageBinding),
	// read by the composition fragment shader (TextureBinding).
	ssrTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "SSR Output Texture",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRGBA16Float,
		Usage:         wgpu.TextureUsageStorageBinding | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create SSR output texture: %w", err)
	}

	ssrView, err = b.device.CreateTextureView(ssrTex, nil)
	if err != nil {
		ssrTex.Release()
		return nil, nil, fmt.Errorf("failed to create SSR output texture view: %w", err)
	}

	return ssrView, ssrTex, nil
}

func (b *wgpuRendererBackendImpl) CreateHiZTextures(width, height int) (
	hizView *wgpu.TextureView, hizTex *wgpu.Texture,
	mipReadViews []*wgpu.TextureView, mipStorageViews []*wgpu.TextureView,
	mipCount int, err error,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	w := uint32(width)
	h := uint32(height)

	// Compute full mip chain count: floor(log2(max(w,h))) + 1.
	maxDim := w
	if h > maxDim {
		maxDim = h
	}
	mipCount = 1
	for d := maxDim; d > 1; d >>= 1 {
		mipCount++
	}

	// R32Float Hi-Z pyramid: written by the Hi-Z init+downsample compute shaders
	// (StorageBinding per mip), read by the SSR compute shader (TextureBinding).
	hizTex, err = b.device.CreateTexture(&wgpu.TextureDescriptor{
		Label: "Hi-Z Depth Pyramid",
		Size: wgpu.Extent3D{
			Width:              w,
			Height:             h,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: uint32(mipCount),
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatR32Float,
		Usage:         wgpu.TextureUsageStorageBinding | wgpu.TextureUsageTextureBinding,
	})
	if err != nil {
		return nil, nil, nil, nil, 0, fmt.Errorf("failed to create Hi-Z texture: %w", err)
	}

	// Full mip chain view for SSR shader reads.
	hizView, err = b.device.CreateTextureView(hizTex, nil)
	if err != nil {
		hizTex.Release()
		return nil, nil, nil, nil, 0, fmt.Errorf("failed to create Hi-Z full view: %w", err)
	}

	// Per-mip read views (texture_2d<f32>) for downsample input.
	mipReadViews = make([]*wgpu.TextureView, mipCount)
	for i := 0; i < mipCount; i++ {
		view, viewErr := b.device.CreateTextureView(hizTex, &wgpu.TextureViewDescriptor{
			Label:           fmt.Sprintf("Hi-Z Mip %d Read", i),
			Format:          gputypes.TextureFormatR32Float,
			Dimension:       gputypes.TextureViewDimension2D,
			BaseMipLevel:    uint32(i),
			MipLevelCount:   1,
			BaseArrayLayer:  0,
			ArrayLayerCount: 1,
		})
		if viewErr != nil {
			// Release already-created views on error.
			for j := 0; j < i; j++ {
				mipReadViews[j].Release()
			}
			hizView.Release()
			hizTex.Release()
			return nil, nil, nil, nil, 0, fmt.Errorf("failed to create Hi-Z mip %d read view: %w", i, viewErr)
		}
		mipReadViews[i] = view
	}

	// Per-mip storage views (texture_storage_2d<r32float, write>) for downsample output.
	mipStorageViews = make([]*wgpu.TextureView, mipCount)
	for i := 0; i < mipCount; i++ {
		view, viewErr := b.device.CreateTextureView(hizTex, &wgpu.TextureViewDescriptor{
			Label:           fmt.Sprintf("Hi-Z Mip %d Storage", i),
			Format:          gputypes.TextureFormatR32Float,
			Dimension:       gputypes.TextureViewDimension2D,
			BaseMipLevel:    uint32(i),
			MipLevelCount:   1,
			BaseArrayLayer:  0,
			ArrayLayerCount: 1,
		})
		if viewErr != nil {
			for j := 0; j < i; j++ {
				mipStorageViews[j].Release()
			}
			for j := 0; j < mipCount; j++ {
				mipReadViews[j].Release()
			}
			hizView.Release()
			hizTex.Release()
			return nil, nil, nil, nil, 0, fmt.Errorf("failed to create Hi-Z mip %d storage view: %w", i, viewErr)
		}
		mipStorageViews[i] = view
	}

	return hizView, hizTex, mipReadViews, mipStorageViews, mipCount, nil
}

func (b *wgpuRendererBackendImpl) RegisterCompositionPipeline(p pipeline.Pipeline) error {
	vertexShader := p.Shader(shader.ShaderTypeVertex)
	if vertexShader == nil {
		return errors.New("vertex shader must be set to create a composition pipeline")
	}

	fragmentShader := p.Shader(shader.ShaderTypeFragment)
	if fragmentShader == nil {
		return errors.New("fragment shader must be set to create a composition pipeline")
	}

	vs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: vertexShader.Key(),
		WGSL:  vertexShader.Source(),
	})
	if err != nil {
		return fmt.Errorf("composition: failed to create vertex shader module: %w", err)
	}

	fs, err := b.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: fragmentShader.Key(),
		WGSL:  fragmentShader.Source(),
	})
	if err != nil {
		return fmt.Errorf("composition: failed to create fragment shader module: %w", err)
	}

	mergedDescriptors := mergeBindGroupLayouts(
		vertexShader.BindGroupLayoutDescriptors(),
		fragmentShader.BindGroupLayoutDescriptors(),
	)

	maxGroup := -1
	for g := range mergedDescriptors {
		if g > maxGroup {
			maxGroup = g
		}
	}
	bindGroupLayouts := make([]*wgpu.BindGroupLayout, maxGroup+1)
	for g, desc := range mergedDescriptors {
		layout, layoutErr := b.device.CreateBindGroupLayout(&desc)
		if layoutErr != nil {
			return fmt.Errorf("composition: failed to create bind group layout for group %d: %w", g, layoutErr)
		}
		bindGroupLayouts[g] = layout
	}

	pipelineLayout, err := b.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            p.PipelineKey(),
		BindGroupLayouts: bindGroupLayouts,
	})
	if err != nil {
		return fmt.Errorf("composition: failed to create pipeline layout: %w", err)
	}

	// Full-screen triangle: no vertex buffers needed. The vertex shader
	// generates positions and UVs from the built-in vertex_index.
	created, err := b.device.CreateRenderPipeline(&wgpu.RenderPipelineDescriptor{
		Label:  p.PipelineKey() + " Composition Pipeline",
		Layout: pipelineLayout,
		Vertex: wgpu.VertexState{
			Module:     vs,
			EntryPoint: vertexShader.EntryPoint(),
		},
		Fragment: &wgpu.FragmentState{
			Module:     fs,
			EntryPoint: fragmentShader.EntryPoint(),
			Targets: []wgpu.ColorTargetState{
				{
					Format:    *b.surfaceFormat,
					WriteMask: gputypes.ColorWriteMaskAll,
				},
			},
		},
		Primitive: wgpu.PrimitiveState{
			Topology:  gputypes.PrimitiveTopologyTriangleList,
			FrontFace: gputypes.FrontFaceCCW,
			CullMode:  gputypes.CullModeNone,
		},
		Multisample: wgpu.MultisampleState{
			Count: 1,
			Mask:  0xFFFFFFFF,
		},
	})
	if err != nil {
		return fmt.Errorf("composition: failed to create render pipeline: %w", err)
	}

	p.SetRenderPipeline(created)
	return nil
}

func (b *wgpuRendererBackendImpl) BeginCompositionFrame() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.compositionSurface != nil {
		return fmt.Errorf("previous composition frame surface not yet presented")
	}

	surfaceTexture, sub, err := b.surface.GetCurrentTexture()
	if err != nil {
		return err
	} else if sub {
		return fmt.Errorf("suboptimal surface texture acquired; presentation may fail")
	}

	view, err := surfaceTexture.CreateView(nil)
	if err != nil {
		return err
	}

	encoder, err := b.device.CreateCommandEncoder(nil)
	if err != nil {
		view.Release()
		return err
	}

	b.compositionFrameEncoder = encoder
	b.compositionSurface = surfaceTexture
	b.compositionView = view
	return nil
}

func (b *wgpuRendererBackendImpl) BeginCompositionPass() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.compositionFrameEncoder == nil {
		return
	}

	pass, err := b.compositionFrameEncoder.BeginRenderPass(&wgpu.RenderPassDescriptor{
		ColorAttachments: []wgpu.RenderPassColorAttachment{
			{
				View:       b.compositionView,
				LoadOp:     gputypes.LoadOpClear,
				StoreOp:    gputypes.StoreOpStore,
				ClearValue: wgpu.Color{R: 0.0, G: 0.0, B: 0.0, A: 1.0},
			},
		},
	})
	if err != nil {
		return
	}
	b.compositionPass = pass
}

func (b *wgpuRendererBackendImpl) CompositionDrawCall(
	p pipeline.Pipeline,
	bindGroups []bind_group_provider.BindGroupProvider,
) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.compositionPass == nil {
		return
	}

	renderPipeline := p.Pipeline().(*wgpu.RenderPipeline)
	b.compositionPass.SetPipeline(renderPipeline)

	for i, bg := range bindGroups {
		b.compositionPass.SetBindGroup(uint32(i), bg.BindGroup(), nil)
	}

	// Full-screen triangle: 3 vertices, 1 instance, no vertex/index buffers.
	b.compositionPass.Draw(3, 1, 0, 0)
}

func (b *wgpuRendererBackendImpl) EndCompositionPass() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.compositionPass == nil {
		return
	}

	b.compositionPass.End()
	b.compositionPass = nil
}

func (b *wgpuRendererBackendImpl) EndCompositionFrame() {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.compositionFrameEncoder == nil {
		return
	}

	commandBuffer, err := b.compositionFrameEncoder.Finish()
	if err != nil {
		b.compositionFrameEncoder = nil
		return
	}

	b.queue.Submit(commandBuffer)

	b.compositionFrameEncoder = nil
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
