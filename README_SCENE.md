# Scene System

The `engine/scene` package is the central orchestrator of the oxy-go engine. A Scene owns a Camera, a Renderer, and a pool of Animators. It manages GameObjects, lights, shadows, Forward+ light culling, and all per-frame GPU work — from compute dispatch through draw calls. Scenes can be hot-swapped via the `Active` flag to switch between views or levels.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/scene`

---

## Architecture

```
Scene (public interface)
 └─ scene (unexported struct)
      ├── Camera           — view/projection, frustum planes, GPU uniform
      ├── Renderer         — pipeline cache, GPU resource init, frame submission
      ├── animatorPool     — map[Model][]Animator (compute + render per model)
      ├── registry         — non-ephemeral GameObjects by ID
      ├── lights / lightsBGP            — light list + GPU storage buffer
      ├── shadow*                       — shadow depth texture, pipelines, BGPs
      ├── lightCull* / tileLit*         — Forward+ tile culling state
      └── computePool      — DynamicWorkerPool for parallel CPU prep
```

The Scene performs all GPU wiring automatically. When a GameObject is `Add`-ed, the Scene:

1. Creates (or reuses) an Animator for the object's Model
2. Registers compute and render pipelines on the Renderer
3. Initializes GPU bind groups, mesh buffers, and material textures
4. Adds an instance to the Animator with the object's initial transform

---

## Constructor

```go
func NewScene(
    name string,
    cam camera.Camera,
    r renderer.Renderer,
    vertexShader shader.Shader,
    options ...SceneBuilderOption,
) Scene
```

Creates a new Scene. All three required arguments (camera, renderer, vertex shader) must be non-nil — panics otherwise. The vertex shader is scanned for a bind group containing `"camera"` and its layout is used to initialize the camera's GPU bind group.

---

## Builder Options

The `NewScene` constructor accepts variadic `SceneBuilderOption` functions:

| Option                                | Description                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------------- |
| `WithActive(active)`                  | Sets whether the scene starts active for rendering. Default: `false`.           |
| `WithObjects(objects...)`             | Adds initial GameObjects. Assigns IDs and persists non-ephemeral objects.       |
| `WithComputeWorkers(n)`               | Sets the number of parallel CPU prep goroutines. Default: `runtime.NumCPU()-1`. |
| `WithCullingDisabled(disabled)`       | Disables GPU frustum culling. Default: `false` (culling enabled).               |
| `WithShadowHalfExtent(halfExtent)`    | Orthographic half-extent of the shadow frustum in world units. Default: `40.0`. |
| `WithShadowNearFar(near, far)`        | Near/far planes for the shadow projection. Default: `0.1`, `200.0`.             |
| `WithShadowBias(bias)`                | Depth comparison bias for shadow sampling. Default: `0.001`.                    |
| `WithShadowNormalBiasScale(scale)`    | Normal-offset bias multiplier on per-texel world size. Default: `3.0`.          |
| `WithShadowMapResolution(resolution)` | Shadow depth texture width/height in texels. Default: `2048`.                   |

---

## Scene Interface

### Object Management

| Method                                                                          | Description                                                                                                            |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `Add(obj, computeShader, vertexShader, fragmentShader, pipelineOpts...) uint64` | Adds a GameObject, auto-creates/reuses an Animator, registers pipelines, inits GPU resources, returns the assigned ID. |
| `Get(id) GameObject`                                                            | Retrieves a non-ephemeral object by ID, or `nil`.                                                                      |
| `Remove(id)`                                                                    | Removes a non-ephemeral object and swap-removes its instance from the animator.                                        |
| `Clear()`                                                                       | Removes all objects and animators. Does not release GPU resources.                                                     |
| `Count() int`                                                                   | Number of persisted (non-ephemeral) objects.                                                                           |
| `CountEphemeral() int`                                                          | Total instance count across all animators.                                                                             |

### Scene State

| Method                         | Description                                |
| ------------------------------ | ------------------------------------------ |
| `Name() string`                | Returns the scene's identifier.            |
| `SetName(name)`                | Sets the scene's identifier.               |
| `Active() bool`                | Whether the scene is active for rendering. |
| `SetActive(active)`            | Enables or disables the scene.             |
| `Camera() Camera`              | Returns the attached camera.               |
| `SetCamera(cam)`               | Replaces the camera.                       |
| `Renderer() Renderer`          | Returns the attached renderer.             |
| `SetRenderer(r)`               | Replaces the renderer.                     |
| `CullingDisabled() bool`       | Whether GPU frustum culling is disabled.   |
| `SetCullingDisabled(disabled)` | Enables or disables frustum culling.       |

### Lighting

| Method                                       | Description                                                                 |
| -------------------------------------------- | --------------------------------------------------------------------------- |
| `AddLight(l)`                                | Adds a light source to the scene.                                           |
| `RemoveLight(l)`                             | Removes a light by reference.                                               |
| `DetachLight(obj)`                           | Detaches an object's auto-registered light. Required for ephemeral objects. |
| `Lights() []Light`                           | Returns a copy of all registered lights.                                    |
| `AmbientColor() [3]float32`                  | Returns the scene's ambient RGB color.                                      |
| `SetAmbientColor(color)`                     | Sets the ambient RGB color.                                                 |
| `LightBindGroupProvider() BindGroupProvider` | Returns the GPU light buffer BGP, or `nil`.                                 |
| `InitLightBindGroup(fragmentShader)`         | Initializes the light storage buffer from a lit fragment shader's layout.   |

### Shadow Mapping

| Method                                                     | Description                                                                                                                                   |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `InitShadowMap(shadowVertShader, shadowSkinnedVertShader)` | Creates shadow depth texture, comparison sampler, shadow data BGP, and registers shadow pipelines.                                            |
| `InitShadowLitBindGroup(litFragmentShader)`                | Creates the fragment-side BGP for shadow map sampling (texture + sampler + uniform). Must be called after `InitShadowMap`.                    |
| `PrepareShadows()`                                         | Computes light VP, uploads shadow uniform, renders the depth-only shadow pass. Must be called after `PrepareCompute` and before `BeginFrame`. |
| `ShadowDepthTextureView() *TextureView`                    | Returns the shadow depth texture view, or `nil`.                                                                                              |
| `ShadowDataBindGroupProvider() BindGroupProvider`          | Returns the shadow data BGP (depth pass), or `nil`.                                                                                           |
| `ShadowLitBindGroupProvider() BindGroupProvider`           | Returns the shadow lit BGP (fragment sampling), or `nil`.                                                                                     |

### Forward+ Light Culling

| Method                                                                                | Description                                                                                                             |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `InitLightCullResources(cullComputeShader, litFragShader, screenWidth, screenHeight)` | Creates tile culling pipeline, compute BGP, and fragment-side tile BGP. Must be called after `InitLightBindGroup`.      |
| `PrepareLightCulling()`                                                               | Uploads cull uniforms and dispatches the light culling compute shader. Call after `PrepareCompute`, before `DrawCalls`. |

### Convenience

| Method                                                                                                                 | Description                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `InitLighting(litFragShader, shadowVertShader, shadowSkinnedVertShader, cullComputeShader, screenWidth, screenHeight)` | Initializes the full lighting pipeline in the correct order: light bind group → shadow map → shadow lit bind group → light cull resources → camera BGP re-init. |

### Frame Methods

| Method                      | Description                                                                                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PrepareCompute(deltaTime)` | Updates camera, syncs light positions, advances animations, uploads buffers, dispatches compute shaders. Must be called within `BeginComputeFrame`/`EndComputeFrame`. |
| `DrawCalls() error`         | Issues instanced draw calls for all animators. Must be called within `BeginFrame`/`EndFrame`. Uses indirect draw when frustum culling is active.                      |

---

## Frame Lifecycle

A typical lit scene frame follows this order:

```
1. renderer.BeginComputeFrame()
   scene.PrepareCompute(dt)          — camera update, animation, buffer writes, compute dispatch
   renderer.EndComputeFrame()

2. scene.PrepareLightCulling()       — Forward+ tile culling (own compute frame)

3. scene.PrepareShadows()            — shadow depth pass (own shadow frame)

4. renderer.BeginFrame()
   scene.DrawCalls()                 — instanced draw calls (regular or indirect)
   renderer.EndFrame()

5. renderer.Present()
```

For unlit scenes, steps 2 and 3 are skipped and `InitLighting` is not called.

---

## Animator Pool

The Scene maintains an `animatorPool` mapping each unique `Model` to a slice of `Animator` instances. When `Add` is called:

- If an Animator for the Model exists with capacity, the object is added as a new instance.
- If all Animators for the Model are full, a new Animator is created with fresh pipelines and GPU resources.
- Each Animator owns its compute BGP, output BGP, and staged write data.

This design allows instanced rendering — hundreds of objects sharing the same Model are drawn in a single GPU draw call.

---

## GPU Resource Wiring

The Scene uses shader annotation declarations to automatically wire bind groups during `DrawCalls`. For each render pipeline, vertex and fragment shader declarations are inspected and matched to providers:

| Provider Annotation      | Source                       |
| ------------------------ | ---------------------------- |
| `@oxy:provider camera`   | Camera's BindGroupProvider   |
| `@oxy:provider material` | Material's BindGroupProvider |
| `@oxy:provider lights`   | Scene's light BGP            |
| `@oxy:provider shadow`   | Scene's shadow lit BGP       |
| `@oxy:provider tiles`    | Scene's tile lit BGP         |
| `@oxy:provider effect`   | Model's effect provider      |
| `@oxy:provider animator` | Animator's output BGP        |

Bind group types (`@oxy:group`) are also matched by their declared data type (e.g., `InstanceData`, `Camera`, `Light`, `ShadowData`, `TileUniforms`, etc.).

---

## Parallel Compute Prep

`PrepareCompute` uses a persistent `DynamicWorkerPool` to parallelize the CPU-intensive animation prep phase:

1. **Pre-pass** (serial): reserved for future GPU rebuild steps
2. **Phase 1** (parallel): each animator's `PrepareFrame` + `Flush` runs concurrently across worker goroutines
3. **Phase 2** (serial): all staged buffer writes are coalesced into a single `WriteBuffers` call, then compute shaders are dispatched sequentially

The worker count defaults to `runtime.NumCPU()-1` and can be overridden with `WithComputeWorkers`.

---

## Files

| File               | Purpose                                                                               |
| ------------------ | ------------------------------------------------------------------------------------- |
| `scene.go`         | `Scene` interface, `scene` struct, `NewScene` constructor, all method implementations |
| `scene_builder.go` | `SceneBuilderOption` type and builder functions                                       |
