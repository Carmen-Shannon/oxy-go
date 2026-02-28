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
	"github.com/cogentcore/webgpu/wgpu"
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
		r.backend = newWGPURendererBackend(window.SurfaceDescriptor(), r.forceFallbackAdapter, msaa)
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
					AddressModeU:  wgpu.AddressModeRepeat,
					AddressModeV:  wgpu.AddressModeRepeat,
					AddressModeW:  wgpu.AddressModeRepeat,
					MagFilter:     wgpu.FilterModeLinear,
					MinFilter:     wgpu.FilterModeLinear,
					MipmapFilter:  wgpu.MipmapFilterModeLinear,
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
			isTexture := entry.Texture.SampleType != wgpu.TextureSampleTypeUndefined
			isSampler := entry.Sampler.Type != wgpu.SamplerBindingTypeUndefined

			if isTexture && provider.TextureView(binding) == nil {
				role := gi.bindingRoles[binding]
				var pixel [4]byte
				switch role {
				case shader.AnnotationArgNormalTexture:
					pixel = [4]byte{128, 128, 255, 255}
				case shader.AnnotationArgMetallicRoughnessTexture:
					pixel = [4]byte{0, 255, 0, 255}
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
					AddressModeU:  wgpu.AddressModeRepeat,
					AddressModeV:  wgpu.AddressModeRepeat,
					AddressModeW:  wgpu.AddressModeRepeat,
					MagFilter:     wgpu.FilterModeLinear,
					MinFilter:     wgpu.FilterModeLinear,
					MipmapFilter:  wgpu.MipmapFilterModeLinear,
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
