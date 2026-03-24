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

| Option                             | Description                                                                |
| ---------------------------------- | -------------------------------------------------------------------------- |
| `WithPipeline(key, p)`             | Pre-registers a single Pipeline in the cache under `key`.                  |
| `WithPipelines(map)`               | Replaces the pipeline cache with the provided map.                         |
| `WithPresentMode(mode)`            | Sets the surface present mode (VSync or Uncapped).                         |
| `WithMSAA(count)`                  | Sets the MSAA sample count (default `MSAA4x`). Use `MSAAOff` to disable.   |
| `WithForceSoftwareRenderer(force)` | Forces a CPU/software fallback adapter (requires SwiftShader or lavapipe). |

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

`ComputeDispatch` groups a compute pipeline key, its bind group provider, and the workgroup dispatch dimensions for use with `DispatchComputeBatch`.

```go
type ComputeDispatch struct {
    PipelineKey    string
    Provider       bind_group_provider.BindGroupProvider
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

| Method                                                                                                                                                                                                             | Description                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| `RegisterShadowDepthPipeline(p pipeline.Pipeline) error`                                                                                                                                                           | Registers a depth-only pipeline for shadow rendering             |
| `CreateShadowDepthTexture(width, height int) (depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error)`                                                                                                     | Allocates a `Depth32Float` atlas texture                         |
| `CreateComparisonSampler() (*wgpu.Sampler, error)`                                                                                                                                                                 | Creates a hardware depth comparison sampler                      |
| `BeginShadowDepthPass(depthView *wgpu.TextureView, x, y, width, height uint32, clear bool)`                                                                                                                        | Begins a single shadow atlas tile depth pass                     |
| `ShadowDrawCall(pipelineKey string, bindGroups []bind_group_provider.BindGroupProvider, vertexBuffers []*wgpu.Buffer, indexBuffer *wgpu.Buffer, indexCount uint32) error`                                          | Issues a shadow draw call                                        |
| `ShadowDrawCallIndirect(pipelineKey string, bindGroups []bind_group_provider.BindGroupProvider, vertexBuffers []*wgpu.Buffer, indexBuffer *wgpu.Buffer, indirectBuffer *wgpu.Buffer, indirectOffset uint64) error` | Issues an indirect shadow draw call                              |
| `EndShadowFrame()`                                                                                                                                                                                                 | Ends the shadow frame and submits its command buffer immediately |

### G-Buffer Frame

| Method                                                                                                                                                                                                                | Description                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CreateGBufferTextures(width, height int) (normView *wgpu.TextureView, normTex *wgpu.Texture, albedoView *wgpu.TextureView, albedoTex *wgpu.Texture, depthView *wgpu.TextureView, depthTex *wgpu.Texture, err error)` | Allocates Normal (RGBA16Float), Albedo (RGBA8Unorm), and Depth (Depth24Plus) G-Buffer textures                                                                          |
| `RegisterGBufferPipeline(p) error`                                                                                                                                                                                    | Creates a render pipeline for the G-Buffer MRT pre-pass.                                                                                                                |
| `BeginGeometryFrame() error`                                                                                                                                                                                          | Begins a batched geometry frame that encompasses both the shadow and G-Buffer passes; uses reference counting to flush to a single command buffer on `EndGeometryFrame` |
| `EndGeometryFrame()`                                                                                                                                                                                                  | Ends the batched geometry frame and submits the accumulated shadow + G-Buffer command buffer                                                                            |
| `BeginGBufferFrame() error`                                                                                                                                                                                           | Creates a command encoder for the G-Buffer pass.                                                                                                                        |
| `BeginGBufferPass(normView, albedoView, depthView *wgpu.TextureView)`                                                                                                                                                 | Begins a G-Buffer draw pass using the three MRT views                                                                                                                   |
| `GBufferDrawCall(pipelineKey string, bindGroups []bind_group_provider.BindGroupProvider, vertexBuffers []*wgpu.Buffer, indexBuffer *wgpu.Buffer, indexCount uint32) error`                                            | Issues a G-Buffer draw call                                                                                                                                             |
| `GBufferDrawCallIndirect(pipelineKey string, bindGroups []bind_group_provider.BindGroupProvider, vertexBuffers []*wgpu.Buffer, indexBuffer *wgpu.Buffer, indirectBuffer *wgpu.Buffer, indirectOffset uint64) error`   | Issues an indirect G-Buffer draw call                                                                                                                                   |
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
| `FlushFrame()`                                                                                                                                                              | Submits all accumulated per-frame command buffers (geometry, compute, HDR, composition) in a single batched `Queue.Submit` call                                                 |

### Configuration

| Method                          | Description                                                              |
| ------------------------------- | ------------------------------------------------------------------------ |
| `SetRenderTargetFormat(format)` | Overrides the color attachment format (e.g. RGBA16Float for HDR passes). |
| `SampleCount() uint32`          | Returns the current MSAA sample count.                                   |

### Display

| Method                  | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| `Resize(width, height)` | Reconfigures the surface, MSAA texture, and depth texture for a new size. |
| `SetPresentMode(mode)`  | Changes the present mode at runtime.                                      |

---

## Frame Lifecycle

A typical frame follows this order:

```
1. WriteBuffers(...)                 — upload per-frame uniform data
2. BeginComputeFrame()               — compute passes (e.g., light culling, SSAO, SSR, Hi-Z)
   DispatchComputeBatch(...)
   EndComputeFrame()
3. **Shadow pass** — `BeginShadowDepthPass(depthView, x, y, width, height, clear)`: renders depth-only into the shadow atlas for each cascade/light.
4. BeginGBufferFrame()               — G-Buffer MRT pre-pass
   BeginGBufferPass(posView, normView, albView, depthView)
   DrawCall(...)
   EndGBufferPass()
   EndGBufferFrame()
5. (compute) SSAO + bilateral blur   — dispatched via DispatchComputeBatch
6. BeginFrame() / BeginHDRFrame()    — main lit render pass (to swapchain or HDR texture)
   DrawCall(...) / DrawCallIndirect(...)
   EndFrame()
7. (compute) SSR Hi-Z + ray march    — dispatched via DispatchComputeBatch
8. Composition pass (BeginFrame)      — full-screen HDR → LDR tone mapping + SSR blend
   DrawCall(...)
   EndFrame()
9. Present()                          — flip to display
```

Steps 3–8 are only executed when lighting/GI sub-handlers are active. For unlit scenes, only steps 1, 2, 6, and 9 run.

---

## Sub-Packages

The renderer package contains the following sub-packages, each documented separately:

| Sub-Package            | Description                                                                               | Documentation                            |
| ---------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------- |
| `animator/`            | GPU compute animation backends (simple transform and skeletal)                            | [README_ANIMATOR.md](README_ANIMATOR.md) |
| `bind_group_provider/` | Bind group creation, buffer and texture storage per draw entity                           | [README_BGP.md](README_BGP.md)           |
| `material/`            | Material GPU types, overlay modes, and effect parameters                                  | [README_MATERIAL.md](README_MATERIAL.md) |
| `pipeline/`            | Render and compute pipeline configuration and GPU object management                       | [README_PIPELINE.md](README_PIPELINE.md) |
| `shader/`              | Shader loading, WGSL parsing, annotation pre-processing, and bind group layout generation | [README_SHADER.md](README_SHADER.md)     |

---

---

## Files

| File                       | Contents                                                                           |
| -------------------------- | ---------------------------------------------------------------------------------- |
| `renderer.go`              | `Renderer` interface definition                                                    |
| `renderer_impl.go`         | Unexported `renderer` struct definition                                            |
| `renderer_builder.go`      | `RendererBuilderOption` type, `With*` builder functions, `NewRenderer` constructor |
| `renderer_backend.go`      | `RendererBackend` interface definition                                             |
| `wgpu_renderer_backend.go` | `wgpuRendererBackend` implementation of `RendererBackend` using `cogentcore/wgpu`  |
