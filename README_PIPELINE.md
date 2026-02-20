# Pipeline

The `engine/renderer/pipeline` package manages render and compute pipeline configuration for the oxy-go engine. A pipeline pairs shader references with GPU state (depth, blend, cull, topology) and holds the underlying WebGPU pipeline object after registration with the Renderer. Pipelines are created with a unique key, cached by the Renderer, and looked up during draw and compute dispatch calls.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline`

---

## Architecture

```
Pipeline (public interface)
 └─ pipeline (unexported struct)
```

The package follows the standard oxy-go interface-first pattern: a single public `Pipeline` interface backed by an unexported `pipeline` struct with a compile-time implementation check.

---

## Pipeline Types

| Constant              | Value | Description                                               |
| --------------------- | ----- | --------------------------------------------------------- |
| `PipelineTypeCompute` | `0`   | Compute pipeline with a single compute shader entry point |
| `PipelineTypeRender`  | `1`   | Render pipeline with vertex and fragment shader stages    |

---

## Pipeline Interface

### Identity & Shader Access

| Method                             | Description                                                              |
| ---------------------------------- | ------------------------------------------------------------------------ |
| `Type() PipelineType`              | Returns the pipeline type (render or compute)                            |
| `PipelineKey() string`             | Returns the unique cache key                                             |
| `Shader(shaderType) shader.Shader` | Returns the shader for a given stage, or nil                             |
| `Pipeline() any`                   | Returns the underlying `*wgpu.RenderPipeline` or `*wgpu.ComputePipeline` |

### Render State (render pipelines only)

| Method                              | Description                                    |
| ----------------------------------- | ---------------------------------------------- |
| `DepthTestEnabled() bool`           | Whether depth testing is on (default `true`)   |
| `DepthWriteEnabled() bool`          | Whether depth writes are on (default `true`)   |
| `DepthBias() int32`                 | Constant depth bias value (default `0`)        |
| `DepthBiasSlopeScale() float32`     | Slope-based depth bias (default `0`)           |
| `BlendEnabled() bool`               | Whether blending is on (default `false`)       |
| `CullMode() wgpu.CullMode`          | Face culling mode (default `CullModeNone`)     |
| `Topology() wgpu.PrimitiveTopology` | Primitive topology (default `TriangleList`)    |
| `FrontFace() wgpu.FrontFace`        | Winding order (default `FrontFaceCCW`)         |
| `WriteMask() wgpu.ColorWriteMask`   | Color write mask (default `ColorWriteMaskAll`) |
| `BlendState() *wgpu.BlendState`     | Blend factors/operations, or nil               |

### Mutators

| Method                                        | Description                   |
| --------------------------------------------- | ----------------------------- |
| `SetRenderPipeline(p *wgpu.RenderPipeline)`   | Sets the GPU render pipeline  |
| `SetComputePipeline(p *wgpu.ComputePipeline)` | Sets the GPU compute pipeline |

---

## Builder Options

The `NewPipeline` constructor accepts variadic `PipelineBuilderOption` functions:

| Option                           | Description                                   |
| -------------------------------- | --------------------------------------------- |
| `WithVertexShader(s)`            | Sets the vertex shader                        |
| `WithFragmentShader(s)`          | Sets the fragment shader                      |
| `WithComputeShader(s)`           | Sets the compute shader                       |
| `WithDepthTestEnabled(enabled)`  | Toggles depth testing                         |
| `WithDepthWriteEnabled(enabled)` | Toggles depth writing                         |
| `WithDepthBias(bias, slope)`     | Sets depth bias constant and slope scale      |
| `WithBlendEnabled(enabled)`      | Toggles blending                              |
| `WithCullMode(mode)`             | Sets the face culling mode                    |
| `WithTopology(topology)`         | Sets the primitive topology                   |
| `WithFrontFace(frontFace)`       | Sets the front face winding order             |
| `WithWriteMask(mask)`            | Sets the color write mask                     |
| `WithBlendState(state)`          | Sets the blend state (factors and operations) |

---

## Constructor

```go
func NewPipeline(pipelineKey string, pipelineType PipelineType, opts ...PipelineBuilderOption) Pipeline
```

Creates a new `Pipeline` with the specified key and type. Render state defaults are:

- Depth test: **enabled**, depth write: **enabled**
- Blend: **disabled**
- Cull mode: **None**, topology: **TriangleList**, front face: **CCW**
- Write mask: **All**
- Blend state: **SrcAlpha / OneMinusSrcAlpha** (pre-configured but inactive until blend is enabled)

Builder options are applied after defaults.

---

## Files

| File                  | Purpose                                                     |
| --------------------- | ----------------------------------------------------------- |
| `pipeline.go`         | `Pipeline` interface, `pipeline` struct, constructor, impls |
| `pipeline_builder.go` | `PipelineBuilderOption` type and 12 builder functions       |
