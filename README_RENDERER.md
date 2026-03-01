# Renderer System

The `engine/renderer` package provides the high-level rendering API for the oxy-go engine. It manages a pipeline cache, owns the GPU backend, and exposes a frame-oriented interface for compute, shadow, and render passes. All GPU interaction flows through the `Renderer` interface, which delegates to a pluggable backend (currently WebGPU via `wgpu`).

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/renderer`

---

## Architecture

```
Renderer (public interface, embeds common.Delegate[Renderer])
 └─ renderer (unexported struct, embeds common.DelegateImpl[Renderer])
      ├─ pipelineCache (map[string]pipeline.Pipeline)
      ├─ materialCache (map[string]material.Material)
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

| Method                                                     | Description                                                             |
| ---------------------------------------------------------- | ----------------------------------------------------------------------- |
| `WriteBuffers(writes []bind_group_provider.BufferWrite)`   | Batch-writes data to GPU buffers identified by provider and binding.    |
| `CreateBuffer(label, size, usage) (*wgpu.Buffer, error)`   | Creates a GPU buffer with the specified label, size, and usage flags.   |
| `CopyBufferToBuffer(src, dst, srcOffset, dstOffset, size)` | Encodes a buffer-to-buffer copy on the current compute frame encoder.   |
| `ReadMappedBuffer(buf, offset, size) ([]byte, error)`      | Synchronously maps a buffer for reading and returns a copy of the data. |

### Material Management

| Method                                                                                                      | Description                                                                          |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `RegisterMaterial(mat material.Material, key string, pipelineOpts ...pipeline.PipelineBuilderOption) error` | Creates GPU resources for a material and optionally registers a new render pipeline. |
| `Material(name string) material.Material`                                                                   | Returns a cached material by name, or `nil` if not found.                            |

### Compute Frame

| Method                                                          | Description                                                       |
| --------------------------------------------------------------- | ----------------------------------------------------------------- |
| `BeginComputeFrame() error`                                     | Creates a command encoder for compute work.                       |
| `DispatchCompute(pipelineKey, computeProvider, workGroupCount)` | Dispatches a compute shader with the given work group dimensions. |
| `EndComputeFrame()`                                             | Finishes and submits the compute command buffer.                  |

### Render Frame

| Method                                                                          | Description                                              |
| ------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `BeginFrame() error`                                                            | Acquires the surface texture and begins the render pass. |
| `DrawCall(pipelineKey, meshProvider, instanceCount, bindGroups) error`          | Issues an indexed draw call.                             |
| `DrawCallIndirect(pipelineKey, meshProvider, indirectBuffer, bindGroups) error` | Issues an indirect indexed draw call.                    |
| `EndFrame()`                                                                    | Ends the render pass and submits the command buffer.     |
| `Present()`                                                                     | Presents the rendered frame to the surface.              |

### Shadow Frame (VSM)

| Method                                                                                           | Description                                                                          |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `RegisterShadowPipeline(p) error`                                                                | Creates a depth-moments render pipeline for VSM shadow mapping.                      |
| `RegisterVSMShadowPipeline(p) error`                                                             | Creates a compute pipeline for VSM blur or SAT passes.                               |
| `CreateVSMTextures(w, h) (view, tex, scratchView, scratchTex, auxDepthView, auxDepthTex, error)` | Creates the RG32Float moments texture, scratch texture, and auxiliary depth texture. |
| `CreateLinearSampler() (*Sampler, error)`                                                        | Creates a linear-filtering sampler for VSM/SSAO/SSR texture lookups.                 |
| `CreateSATTextures(w, h) (viewA, texA, viewB, texB, error)`                                      | Creates two RGBA32Float ping-pong textures for SAT generation.                       |
| `BeginShadowFrame() error`                                                                       | Creates a command encoder for shadow passes.                                         |
| `BeginVSMShadowPass(depthView, colorView)`                                                       | Begins a VSM depth-moments render pass targeting the given depth and color views.    |
| `ShadowDrawCall(pipelineKey, meshProvider, instanceCount, bindGroups) error`                     | Issues an indexed draw call into the shadow pass.                                    |
| `ShadowDrawCallIndirect(pipelineKey, meshProvider, indirectBuffer, bindGroups) error`            | Issues an indirect indexed draw into the shadow pass.                                |
| `EndShadowPass()`                                                                                | Ends the current shadow render pass.                                                 |
| `EndShadowFrame()`                                                                               | Finishes and submits the shadow command buffer.                                      |

### G-Buffer Frame

| Method                                                                                                          | Description                                                                                                                |
| --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `CreateGBufferTextures(w, h) (posView, posTex, normView, normTex, albView, albTex, depthView, depthTex, error)` | Creates the RGBA16Float position/normal textures, RGBA8Unorm albedo texture, and Depth24Plus texture for the MRT pre-pass. |
| `RegisterGBufferPipeline(p) error`                                                                              | Creates a render pipeline for the G-Buffer MRT pre-pass.                                                                   |
| `BeginGBufferFrame() error`                                                                                     | Creates a command encoder for the G-Buffer pass.                                                                           |
| `BeginGBufferPass(posView, normView, albView, depthView)`                                                       | Begins a multi-target render pass for the G-Buffer.                                                                        |
| `EndGBufferPass()`                                                                                              | Ends the G-Buffer render pass.                                                                                             |
| `EndGBufferFrame()`                                                                                             | Finishes and submits the G-Buffer command buffer.                                                                          |

### SSAO / Composition / SSR

| Method                                                                                                               | Description                                                             |
| -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `CreateSSAOTextures(w, h) (rawView, rawTex, blurView, blurTex, scratchView, scratchTex, noiseView, noiseTex, error)` | Creates all SSAO textures (raw, blurred, scratch, 4×4 noise).           |
| `CreateCompositionTextures(w, h, sampleCount) (hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex, error)`      | Creates HDR, MSAA resolve, and depth textures for the composition pass. |
| `RegisterCompositionPipeline(p) error`                                                                               | Creates the full-screen composition render pipeline.                    |
| `CreateSSRTextures(w, h) (ssrView, ssrTex, error)`                                                                   | Creates the half-resolution RGBA16Float SSR result texture.             |
| `CreateHiZTextures(w, h) (fullView, tex, mipCount, mipReadViews, storageViews, error)`                               | Creates the R32Float Hi-Z depth pyramid with per-mip views.             |
| `BeginHDRFrame(colorView, resolveView, depthView) error`                                                             | Begins a render pass targeting the offscreen HDR texture.               |

### Probe Baking

| Method                                                                           | Description                                                           |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `CreateProbeBakeTextures(res) (colorView, colorTex, depthView, depthTex, error)` | Creates per-face cubemap bake render targets at the given resolution. |
| `RegisterProbeBakePipeline(p) error`                                             | Creates the render pipeline used for probe cubemap face baking.       |
| `BeginProbeBakeFrame() error`                                                    | Creates a command encoder for probe bake passes.                      |
| `BeginProbeBakePass(colorView, depthView)`                                       | Begins a render pass for a single cubemap face.                       |

### Configuration

| Method                          | Description                                                              |
| ------------------------------- | ------------------------------------------------------------------------ |
| `SetRenderTargetFormat(format)` | Overrides the color attachment format (e.g. RGBA16Float for HDR passes). |
| `SampleCount() int`             | Returns the current MSAA sample count.                                   |

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
   DispatchCompute(...)
   EndComputeFrame()
3. BeginShadowFrame()                — VSM depth-moments pass
   BeginVSMShadowPass(depthView, colorView)
   ShadowDrawCall(...) / ShadowDrawCallIndirect(...)
   EndShadowPass()
   EndShadowFrame()
   (compute) VSM blur or SAT generation passes
4. BeginGBufferFrame()               — G-Buffer MRT pre-pass
   BeginGBufferPass(posView, normView, albView, depthView)
   DrawCall(...)
   EndGBufferPass()
   EndGBufferFrame()
5. (compute) SSAO + bilateral blur   — dispatched via DispatchCompute
6. BeginFrame() / BeginHDRFrame()    — main lit render pass (to swapchain or HDR texture)
   DrawCall(...) / DrawCallIndirect(...)
   EndFrame()
7. (compute) SSR Hi-Z + ray march    — dispatched via DispatchCompute
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

## Files

| File                       | Purpose                                                                       |
| -------------------------- | ----------------------------------------------------------------------------- |
| `renderer.go`              | `Renderer` interface, unexported `renderer` struct, `NewRenderer` constructor |
| `renderer_backend.go`      | `RendererBackendType` enum, `PresentMode` enum, `RendererBackend` interface   |
| `renderer_builder.go`      | `RendererBuilderOption` type and builder functions                            |
| `wgpu_renderer_backend.go` | Full WebGPU backend implementation (`wgpuRendererBackendImpl`)                |
