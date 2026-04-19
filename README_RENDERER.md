# Renderer System

The `engine/renderer` package provides the high-level rendering API for the oxy-go engine. It manages a pipeline cache, owns the GPU backend, and exposes a frame-oriented interface for compute, shadow, and render passes. All GPU interaction flows through the `Renderer` interface, which delegates to a pluggable backend (currently WebGPU via `wgpu`).

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/renderer`

---

## Architecture

```
Renderer (public interface, embeds common.Delegate[Renderer])
 └─ renderer (unexported struct, embeds common.DelegateImpl[Renderer])
      ├─ pipelineCache (map[string]pipeline.Pipeline)
      └─ RendererBackend (interface)
           └─ wgpuRendererBackendImpl (WGPU backend)
```

The `Renderer` interface resolves pipeline keys, acquires the mutex, and forwards every call to the `RendererBackend`. The backend performs the actual WebGPU device/queue/surface operations. This separation keeps the public API backend-agnostic.

---

## Backend Types

| Constant          | Value | Description                                 |
| ----------------- | ----- | ------------------------------------------- |
| `BackendTypeWGPU` | `0`   | WebGPU backend via `cogentcore/webgpu/wgpu` |

Additional backends can be added by implementing the `RendererBackend` interface and extending `RendererBackendType`.

## Present Modes

| Constant              | Value | Description                                                     |
| --------------------- | ----- | --------------------------------------------------------------- |
| `PresentModeVSync`    | `0`   | Waits for vertical blank before presenting. Eliminates tearing. |
| `PresentModeUncapped` | `1`   | Presents immediately. Lowest latency, may tear.                 |

## MSAA Sample Counts

| Constant  | Value | Description                                                  |
| --------- | ----- | ------------------------------------------------------------ |
| `MSAAOff` | `1`   | Disables multisample anti-aliasing.                          |
| `MSAA4x`  | `4`   | 4× MSAA. Default. Guaranteed by WebGPU on all adapters.      |
| `MSAA8x`  | `8`   | 8× MSAA. Adapter-dependent; not all hardware supports this.  |
| `MSAA16x` | `16`  | 16× MSAA. Adapter-dependent; not all hardware supports this. |

When `MSAAOff` is selected, the render pass draws directly to the swapchain surface (no intermediate MSAA texture, no resolve step). When any multi-sample count is active, an MSAA texture is created and the swapchain view is used as the resolve target.

---

## Builder Options

The `NewRenderer` constructor accepts variadic `RendererBuilderOption` functions:

| Option                                | Description                                                                                                                               |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `WithPipeline(key, p)`                | Pre-registers a single Pipeline in the cache under `key`.                                                                                 |
| `WithPipelines(map)`                  | Replaces the pipeline cache with the provided map.                                                                                        |
| `WithPresentMode(mode)`               | Sets the surface present mode (VSync or Uncapped).                                                                                        |
| `WithMSAA(count)`                     | Sets the MSAA sample count (default `MSAA4x`). Use `MSAAOff` to disable.                                                                  |
| `WithForceSoftwareRenderer(force)`    | Forces a CPU/software fallback adapter (requires SwiftShader or lavapipe).                                                                |
| `WithGPUSerializedProfiling(enabled)` | Enables GPU serialized profiling mode — each phase submits and polls separately. For diagnostic use only; eliminates GPU/CPU parallelism. |

---

## Constructor

```go
func NewRenderer(
    backendType RendererBackendType,
    window window.Window,
    options ...RendererBuilderOption,
) Renderer
```

Creates a new `Renderer` with the specified backend type and window. The window provides the surface descriptor for WebGPU surface creation. Builder options are applied before backend initialization. The delegation target is set to itself (`r.Delegate = r`).

---

## Renderer Interface

The `Renderer` interface groups its methods into the following categories.

### Delegation

The `Renderer` interface embeds `common.Delegate[Renderer]`, exposing `SetDelegate(delegate Renderer)`. In production code the delegate is set to the instance itself during construction. In test code the delegate can be replaced with a mock.

### Pipeline Management

| Method                                                    | Description                                                                  |
| --------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `Pipeline(key) pipeline.Pipeline`                         | Retrieves a cached pipeline by key, or `nil`.                                |
| `Pipelines() map[string]pipeline.Pipeline`                | Returns the full pipeline cache.                                             |
| `RegisterPipelines(pipelines ...pipeline.Pipeline) error` | Creates GPU pipeline objects and caches them. Skips already-registered keys. |
| `SetPipeline(key, p)`                                     | Adds or updates a single pipeline in the cache.                              |
| `SetPipelines(map)`                                       | Replaces the entire pipeline cache.                                          |

### Resource Initialization

| Method                                                                                 | Description                                                            |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `InitMeshBuffers(provider, vertexData, indexData, indexCount) error`                   | Creates GPU vertex/index buffers and stores them on the provider.      |
| `InitBindGroup(provider, descriptor, bufferUsageOverrides, bufferSizeOverrides) error` | Creates a bind group with its layout, buffers, textures, and samplers. |
| `InitTextureView(provider, bindingKey, stagingData) error`                             | Uploads texture pixel data and creates a texture view.                 |
| `InitSampler(provider, bindingKey, samplerStagingData) error`                          | Creates a GPU sampler with the given parameters.                       |

### Buffer Operations

| Method                                                                                                      | Description                                                             |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `WriteBuffers(writes []bind_group_provider.BufferWrite)`                                                    | Batch-writes data to GPU buffers identified by provider and binding.    |
| `CreateBuffer(label, size, usage) (*wgpu.Buffer, error)`                                                    | Creates a GPU buffer with the specified label, size, and usage flags.   |
| `CopyBufferToBuffer(src, dst, srcOffset, dstOffset, size)`                                                  | Encodes a buffer-to-buffer copy on the current compute frame encoder.   |
| `ReadMappedBuffer(buf, offset, size) ([]byte, error)`                                                       | Synchronously maps a buffer for reading and returns a copy of the data. |
| `WriteRawBuffer(buf *wgpu.Buffer, offset uint64, data []byte)`                                              | Writes raw bytes to a GPU buffer at the given offset                    |
| `WriteTexture(tex *wgpu.Texture, data []byte, width, height, bytesPerRow uint32)`                           | Uploads pixel data to a GPU texture                                     |
| `CreateContactShadowTextures(width, height int) (csView *wgpu.TextureView, csTex *wgpu.Texture, err error)` | Allocates a contact shadow output texture                               |
| `MaxTextureDimension2D() uint32`                                                                            | Returns the device maximum 2D texture dimension                         |
| `SetInjections(injections map[string]string)`                                                               | Sets the shader injection map used by the pre-processor                 |

### Compute Frame

| Method                                               | Description                                                                         |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `BeginComputeFrame() error`                          | Creates a command encoder for compute work.                                         |
| `DispatchComputeBatch(dispatches []ComputeDispatch)` | Encodes one or more compute dispatches as a batch within the current compute frame. |
| `EndComputeFrame()`                                  | Finishes and submits the compute command buffer.                                    |

#### ComputeDispatch

`ComputeGroupProvider` pairs a WGSL group index with the `BindGroupProvider` that supplies the bind group for that slot.

```go
type ComputeGroupProvider struct {
    Group    uint32
    Provider bind_group_provider.BindGroupProvider
}
```

`ComputeDispatch` groups a compute pipeline key, an ordered list of group providers, and the workgroup dispatch dimensions for use with `DispatchComputeBatch`.

```go
type ComputeDispatch struct {
    PipelineKey    string
    Providers      []ComputeGroupProvider
    WorkGroupCount [3]uint32
}
```

### Render Frame

| Method                                                                          | Description                                              |
| ------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `BeginFrame() error`                                                            | Acquires the surface texture and begins the render pass. |
| `DrawCall(pipelineKey, meshProvider, instanceCount, bindGroups) error`          | Issues an indexed draw call.                             |
| `DrawCallIndirect(pipelineKey, meshProvider, indirectBuffer, bindGroups) error` | Issues an indirect indexed draw call.                    |
| `EndFrame()`                                                                    | Ends the render pass and submits the command buffer.     |
| `Present()`                                                                     | Presents the rendered frame to the surface.              |

### Shadow Frame

The shadow frame renders depth-only passes into a `Depth32Float` atlas for directional CSM and spot/point lights.

| Method                                                                                                                                                                                  | Description                                                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `RegisterShadowDepthPipeline(p pipeline.Pipeline) error`                                                                                                                                | Registers a depth-only pipeline for shadow rendering.                                                               |
| `CreateShadowDepthTexture(width, height int) (depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error)`                                                                          | Allocates a `Depth32Float` atlas texture.                                                                           |
| `CreateComparisonSampler() (*wgpu.Sampler, error)`                                                                                                                                      | Creates a hardware depth comparison sampler for PCF lookups.                                                        |
| `CreateLinearSampler() (*wgpu.Sampler, error)`                                                                                                                                          | Creates a standard linear filtering sampler for post-processing passes (SSAO, SSR, composition).                    |
| `BeginShadowFrame() error`                                                                                                                                                              | Creates a command encoder for batching shadow depth passes. Aliases the shared geometry encoder when one is active. |
| `ShadowDrawCall(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider) error`                | Issues an instanced shadow draw call within the current shadow pass.                                                |
| `ShadowDrawCallIndirect(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider) error` | Issues an indirect shadow draw call; instance count is read from `indirectBuffer` on the GPU.                       |
| `EndShadowFrame()`                                                                                                                                                                      | Finishes the shadow command encoder and submits to the GPU queue.                                                   |

#### Shadow Atlas Pass (single-pass, viewport switching)

All tiles for one atlas share a single render pass. The pass is opened once with `BeginShadowAtlasPass`, then `SetShadowViewport` is called before each tile's draw calls to constrain rasterization to that tile's region.

| Method                                              | Description                                                                                                                   |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `BeginShadowAtlasPass(depthView *wgpu.TextureView)` | Opens a single depth-only render pass for the full atlas, clearing depth to 1.0. Must be called after `BeginShadowFrame`.     |
| `SetShadowViewport(x, y, width, height uint32)`     | Sets the viewport and scissor rect on the open atlas pass, constraining rasterization to the specified tile region in texels. |
| `EndShadowAtlasPass()`                              | Closes the shadow atlas render pass after all tiles for this atlas have been drawn.                                           |

### G-Buffer Frame

| Method                                                                                                                                                                                                                | Description                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CreateGBufferTextures(width, height int) (normView *wgpu.TextureView, normTex *wgpu.Texture, albedoView *wgpu.TextureView, albedoTex *wgpu.Texture, depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error)` | Allocates Normal (RGBA16Float), Albedo (RGBA8Unorm), and Depth (Depth24Plus) G-Buffer textures                                                                          |
| `RegisterGBufferPipeline(p) error`                                                                                                                                                                                    | Creates a render pipeline for the G-Buffer MRT pre-pass.                                                                                                                |
| `BeginGeometryFrame() error`                                                                                                                                                                                          | Begins a batched geometry frame that encompasses both the shadow and G-Buffer passes; uses reference counting to flush to a single command buffer on `EndGeometryFrame` |
| `EndGeometryFrame()`                                                                                                                                                                                                  | Ends the batched geometry frame and submits the accumulated shadow + G-Buffer command buffer                                                                            |
| `BeginGBufferFrame() error`                                                                                                                                                                                           | Creates a command encoder for the G-Buffer pass.                                                                                                                        |
| `BeginGBufferPass(normView, albedoView, depthView *wgpu.TextureView)`                                                                                                                                                 | Begins a G-Buffer draw pass using the three MRT views                                                                                                                   |
| `GBufferDrawCall(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, instanceCount uint32, bindGroups []bind_group_provider.BindGroupProvider) error`                                             | Issues an instanced G-Buffer draw call.                                                                                                                                 |
| `GBufferDrawCallIndirect(pipelineKey string, meshProvider bind_group_provider.BindGroupProvider, indirectBuffer *wgpu.Buffer, bindGroups []bind_group_provider.BindGroupProvider) error`                              | Issues an indirect G-Buffer draw call; instance count is read from `indirectBuffer` on the GPU.                                                                         |
| `EndGBufferPass()`                                                                                                                                                                                                    | Ends the G-Buffer render pass.                                                                                                                                          |
| `EndGBufferFrame()`                                                                                                                                                                                                   | Finishes and submits the G-Buffer command buffer.                                                                                                                       |

### SSAO / Composition / SSR

| Method                                                                                                                                                                      | Description                                                                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CreateSSAOTextures(w, h) (rawView, rawTex, blurView, blurTex, scratchView, scratchTex, error)`                                                                             | Creates all SSAO textures (raw, blurred, scratch).                                                                                                                              |
| `CreateCompositionTextures(w, h, sampleCount) (hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex, error)`                                                             | Creates HDR, MSAA resolve, and depth textures for the composition pass.                                                                                                         |
| `RegisterCompositionPipeline(p) error`                                                                                                                                      | Creates the full-screen composition render pipeline.                                                                                                                            |
| `CreateSSRTextures(w, h) (ssrView, ssrTex, error)`                                                                                                                          | Creates the half-resolution RGBA16Float SSR result texture.                                                                                                                     |
| `CreateHiZTextures(w, h) (hizView *wgpu.TextureView, hizTex *wgpu.Texture, mipReadViews []*wgpu.TextureView, mipStorageViews []*wgpu.TextureView, mipCount int, err error)` | Creates the R32Float Hi-Z depth pyramid with per-mip views.                                                                                                                     |
| `CreateBloomTextures(w, h) (...views, mipCount, error)`                                                                                                                     | Creates two RGBA16Float mip chain textures (down chain + up chain) with per-mip read/storage views and a mip-0 up-chain view for the composition shader. Mip count capped at 6. |
| `BeginHDRFrame(colorView, resolveView, depthView *wgpu.TextureView, sampleCount uint32) error`                                                                              | Begins a render pass targeting the offscreen HDR texture.                                                                                                                       |
| `BeginCompositionFrame() error`                                                                                                                                             | Begins the composition tone-mapping frame                                                                                                                                       |
| `BeginCompositionPass()`                                                                                                                                                    | Begins the composition draw pass                                                                                                                                                |
| `CompositionDrawCall(pipelineKey string, bindGroups []bind_group_provider.BindGroupProvider) error`                                                                         | Issues a full-screen composition draw call                                                                                                                                      |
| `EndCompositionPass()`                                                                                                                                                      | Ends the composition draw pass                                                                                                                                                  |
| `EndCompositionFrame()`                                                                                                                                                     | Ends the composition frame and accumulates its command buffer                                                                                                                   |
| `FlushFrame() wgpu.SubmissionIndex`                                                                                                                                         | Submits all accumulated per-frame command buffers (geometry, compute, HDR, composition) in a single batched `Queue.Submit` call. Returns the submission index.                  |
| `WaitIdle()`                                                                                                                                                                | Blocks until all in-flight GPU work has completed. Must be called before releasing GPU resources (e.g., on window resize).                                                      |

### Configuration

| Method                                        | Description                                                                                                                                                                                                                                      |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `SetRenderTargetFormat(format)`               | Overrides the color attachment format (e.g. RGBA16Float for HDR passes).                                                                                                                                                                         |
| `SetInjections(injections map[string]string)` | Sets the WGSL injection map used by the shader pre-processor for `@oxy:inject` annotations.                                                                                                                                                      |
| `SampleCount() uint32`                        | Returns the current MSAA sample count.                                                                                                                                                                                                           |
| `MaxTextureDimension2D() uint32`              | Returns the device maximum 2D texture dimension, raised to the adapter's reported limit when above the WebGPU default.                                                                                                                           |
| `CurrentFrameSlot() int`                      | Returns the index of the frame slot currently being encoded (0 or 1).                                                                                                                                                                            |
| `GPUTimings() map[string]float64`             | Always returns `nil`. GPU timestamp readback via `MapAsync` is permanently disabled due to a bug in the underlying wgpu fork. `WriteTimestamp` and `ResolveQuerySet` calls are retained (harmless) but the readback step is permanently omitted. |
| `SyncGPUTimestamps()`                         | Retained for interface compatibility. Performs a frames-in-flight fence via `Device.Poll`; GPU timestamp readback via `MapAsync` is permanently disabled.                                                                                        |

### Display

| Method                  | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| `Resize(width, height)` | Reconfigures the surface, MSAA texture, and depth texture for a new size. |
| `SetPresentMode(mode)`  | Changes the present mode at runtime.                                      |

---

## Frame Lifecycle

A typical frame follows this order:

```
1.  WriteBuffers(...)                    — upload per-frame uniform data
2.  BeginComputeFrame()                  — pre-geometry compute passes (Hi-Z downsample, light culling)
    DispatchComputeBatch(...)
    EndComputeFrame()
3.  BeginGeometryFrame()                 — opens shared geometry command encoder
      BeginShadowFrame()                 — shadow depth passes
        BeginShadowAtlasPass(depthView)  — one pass per atlas; clears depth to 1.0
          SetShadowViewport(x, y, w, h)  — constrain to each cascade / light tile
          ShadowDrawCall(...)            — depth-only draw per cascade / light face
        EndShadowAtlasPass()
      EndShadowFrame()
      BeginGBufferFrame()                — G-Buffer MRT pre-pass
        BeginGBufferPass(norm, alb, dep)
        GBufferDrawCall(...) / GBufferDrawCallIndirect(...)
        EndGBufferPass()
      EndGBufferFrame()
    EndGeometryFrame()                   — submits merged shadow + G-Buffer command buffer
4.  BeginComputeFrame()                  — post-geometry compute (SSAO, contact shadows)
    DispatchComputeBatch(...)
    EndComputeFrame()
5.  BeginHDRFrame(colorView, resolveView, depthView, sampleCount)
    DrawCall(...) / DrawCallIndirect(...)  — main lit render pass (to offscreen HDR texture)
    EndFrame()
6.  BeginComputeFrame()                  — post-HDR compute (SSR ray march, luminance, bloom)
    DispatchComputeBatch(...)
    EndComputeFrame()
7.  BeginCompositionFrame()              — full-screen HDR → LDR tone-mapping + SSR blend
    BeginCompositionPass()
    CompositionDrawCall(...)
    EndCompositionPass()
    EndCompositionFrame()
8.  FlushFrame()                         — single batched Queue.Submit for all accumulated buffers
9.  Present()                            — flip to display
```

Steps 3–8 are only executed when lighting/GI sub-handlers are active. For unlit scenes, only steps 1, 2, 5, and 9 run.

---

## Sub-Packages

The renderer package contains the following sub-packages, each documented separately:

| Sub-Package            | Description                                                                               | Documentation                                      |
| ---------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `animator/`            | GPU compute animation backends (simple transform and skeletal)                            | [README_ANIMATOR.md](README_ANIMATOR.md)           |
| `bind_group_provider/` | Bind group creation, buffer and texture storage per draw entity                           | [README_BGP.md](README_BGP.md)                     |
| `gbuffer/`             | G-Buffer MRT handler for geometry pre-pass (normals, albedo, depth)                       | [GBuffer Package](#gbuffer-package)                |
| `material/`            | Material GPU types, overlay modes, and effect parameters                                  | [README_MATERIAL.md](README_MATERIAL.md)           |
| `pipeline/`            | Render and compute pipeline configuration and GPU object management                       | [README_PIPELINE.md](README_PIPELINE.md)           |
| `postprocessing/`      | Screen-space post-processing handlers (SSAO, Composition, SSR, TAA)                       | [README_POSTPROCESSOR.md](README_POSTPROCESSOR.md) |
| `shader/`              | Shader loading, WGSL parsing, annotation pre-processing, and bind group layout generation | [README_SHADER.md](README_SHADER.md)               |

---

## GBuffer Package

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/renderer/gbuffer`

The `gbuffer` package manages the multiple render target (MRT) textures written during the geometry pre-pass. Downstream screen-space effects (SSAO, SSR, contact shadows) read from these textures.

**Constructor:** `NewGBufferHandler(opts ...GBufferHandlerOption) GBufferHandler`

### Builder Options

| Option           | Parameters          | Default | Description                                          |
| ---------------- | ------------------- | ------- | ---------------------------------------------------- |
| `WithScreenSize` | `width, height int` | 0, 0    | Initial screen dimensions for G-Buffer texture alloc |

### GBufferHandler Interface (22 methods)

| Method                                            | Description                                    |
| ------------------------------------------------- | ---------------------------------------------- |
| `Enabled() bool` / `SetEnabled(bool)`             | Whether GPU resources are initialized          |
| `SetSlot(slot int)`                               | Sets the active double-buffer slot             |
| `ScreenWidth() int` / `ScreenHeight() int`        | Current screen dimensions                      |
| `NormalTexture()` / `SetNormalTexture`            | Normals + roughness MRT texture (RGBA16Float)  |
| `NormalTextureView()` / `SetNormalTextureView`    | View for normal texture                        |
| `AlbedoTexture()` / `SetAlbedoTexture`            | Albedo + metallic MRT texture (RGBA8Unorm)     |
| `AlbedoTextureView()` / `SetAlbedoTextureView`    | View for albedo texture                        |
| `DepthTexture()` / `SetDepthTexture`              | Shared depth texture (Depth24Plus)             |
| `DepthTextureView()` / `SetDepthTextureView`      | View for depth texture                         |
| `PipelineKey(name)` / `SetPipelineKey(name, key)` | Pipeline key storage                           |
| `PipelineKeys() map[string]string`                | Returns all pipeline keys                      |
| `Resize(width, height int)`                       | Updates stored screen dimensions (no GPU work) |

All texture getters/setters are double-buffered (2-slot arrays indexed by the active slot set via `SetSlot`).

### MRT Textures

| Texture | Format      | Contents                                         |
| ------- | ----------- | ------------------------------------------------ |
| Normal  | RGBA16Float | World normal XYZ (packed [0,1]) + roughness in W |
| Albedo  | RGBA8Unorm  | Albedo RGB + metallic in A                       |
| Depth   | Depth24Plus | Shared depth for the pre-pass                    |

### GPU Type: `GPUGBufferOutput` (48 bytes)

| Field      | Type         | Offset | Description          |
| ---------- | ------------ | ------ | -------------------- |
| `Position` | `[4]float32` | 0      | World-space position |
| `Normal`   | `[4]float32` | 16     | World-space normal   |
| `Albedo`   | `[4]float32` | 32     | Albedo color         |

Methods: `Size() int`, `Marshal() []byte`

Embedded WGSL source: `GPUGBufferOutputSource` (`assets/gbuffer-output.wgsl`)

### Files

| File                         | Contents                                                           |
| ---------------------------- | ------------------------------------------------------------------ |
| `gbuffer_handler.go`         | `GBufferHandler` interface                                         |
| `gbuffer_handler_impl.go`    | Unexported `gBufferHandlerImpl` struct                             |
| `gbuffer_handler_builder.go` | `GBufferHandlerOption` type, `WithScreenSize`, `NewGBufferHandler` |
| `gpu_types.go`               | `GPUGBufferOutput` struct and embedded WGSL source                 |

---

## Files

| File                       | Contents                                                                           |
| -------------------------- | ---------------------------------------------------------------------------------- |
| `renderer.go`              | `Renderer` interface definition                                                    |
| `renderer_impl.go`         | Unexported `renderer` struct definition                                            |
| `renderer_builder.go`      | `RendererBuilderOption` type, `With*` builder functions, `NewRenderer` constructor |
| `renderer_backend.go`      | `RendererBackend` interface definition                                             |
| `wgpu_renderer_backend.go` | `wgpuRendererBackend` implementation of `RendererBackend` using `cogentcore/wgpu`  |
