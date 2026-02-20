# Renderer System

The `engine/renderer` package provides the high-level rendering API for the oxy-go engine. It manages a pipeline cache, owns the GPU backend, and exposes a frame-oriented interface for compute, shadow, and render passes. All GPU interaction flows through the `Renderer` interface, which delegates to a pluggable backend (currently WebGPU via `wgpu`).

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/renderer`

---

## Architecture

```
Renderer (public interface)
 └─ renderer (unexported struct)
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
    windowHwnd, windowHinstance unsafe.Pointer,
    options ...RendererBuilderOption,
) Renderer
```

Creates a new `Renderer` with the specified backend type and native window handles. Builder options are applied before backend initialization.

---

## Renderer Interface

The `Renderer` interface groups its methods into the following categories.

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

### Buffer Writes

| Method                               | Description                                                          |
| ------------------------------------ | -------------------------------------------------------------------- |
| `WriteBuffers(writes []BufferWrite)` | Batch-writes data to GPU buffers identified by provider and binding. |

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

### Shadow Frame

| Method                                                                                | Description                                                     |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `RegisterShadowPipeline(p) error`                                                     | Creates a depth-only render pipeline for shadow mapping.        |
| `CreateShadowDepthTexture(width, height) (*TextureView, *Texture, error)`             | Creates a Depth32Float texture for shadow map rendering.        |
| `CreateComparisonSampler() (*Sampler, error)`                                         | Creates a comparison sampler for shadow map sampling.           |
| `BeginShadowFrame() error`                                                            | Creates a command encoder for shadow passes.                    |
| `BeginShadowPass(depthView)`                                                          | Begins a depth-only render pass targeting the given depth view. |
| `ShadowDrawCall(pipelineKey, meshProvider, instanceCount, bindGroups) error`          | Issues an indexed draw call into the shadow pass.               |
| `ShadowDrawCallIndirect(pipelineKey, meshProvider, indirectBuffer, bindGroups) error` | Issues an indirect indexed draw into the shadow pass.           |
| `EndShadowPass()`                                                                     | Ends the current shadow render pass.                            |
| `EndShadowFrame()`                                                                    | Finishes and submits the shadow command buffer.                 |

### Display

| Method                  | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| `Resize(width, height)` | Reconfigures the surface, MSAA texture, and depth texture for a new size. |
| `SetPresentMode(mode)`  | Changes the present mode at runtime.                                      |

---

## Frame Lifecycle

A typical frame follows this order:

```
1. WriteBuffers(...)               — upload per-frame uniform data
2. BeginComputeFrame()             — compute passes (e.g., light culling)
   DispatchCompute(...)
   EndComputeFrame()
3. BeginShadowFrame()              — shadow map passes
   BeginShadowPass(depthView)
   ShadowDrawCall(...) / ShadowDrawCallIndirect(...)
   EndShadowPass()
   EndShadowFrame()
4. BeginFrame()                    — main render pass (MSAA per config)
   DrawCall(...) / DrawCallIndirect(...)
   EndFrame()
5. Present()                       — flip to display
```

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
