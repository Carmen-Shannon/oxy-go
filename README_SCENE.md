# Scene System

The `engine/scene` package is the central orchestrator of the oxy-go engine. A Scene owns a Camera, a Renderer, and a pool of Animators. It manages GameObjects, lights, shadows, Forward+ light culling, and all per-frame GPU work — from compute dispatch through draw calls. Scenes can be hot-swapped via the `Active` flag to switch between views or levels.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/scene`

---

## Architecture

```
Scene (public interface)
 ├─ common.Delegate[Scene]   ← mock / test delegation
 └─ scene (unexported struct)
      ├─ common.DelegateImpl[Scene]
      ├── Camera           — view/projection, frustum planes, GPU uniform
      ├── Renderer         — pipeline cache, GPU resource init, frame submission
      ├── animatorPool     — map[Model][]Animator (compute + render per model)
      ├── registry         — non-ephemeral GameObjects by ID
      ├── lightHandler     — LightingHandler (light list, shadows, Forward+ culling)
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
    options ...SceneBuilderOption,
) Scene
```

Creates a new Scene. Both required arguments (camera, renderer) must be non-nil — panics otherwise. The vertex shader is loaded internally from the engine's standard shader assets. Sets `s.Delegate = s` so delegation routes to itself by default.

---

## Builder Options

The `NewScene` constructor accepts variadic `SceneBuilderOption` functions:

| Option                          | Description                                                                                         |
| ------------------------------- | --------------------------------------------------------------------------------------------------- |
| `WithActive(active)`            | Sets whether the scene starts active for rendering. Default: `false`.                               |
| `WithObjects(objects...)`       | Adds initial GameObjects. Assigns IDs and persists non-ephemeral objects.                           |
| `WithComputeWorkers(n)`         | Sets the number of parallel CPU prep goroutines. Default: `max(runtime.NumCPU()-1, 1)`.             |
| `WithCullingDisabled(disabled)` | Disables GPU frustum culling. Default: `false` (culling enabled).                                   |
| `WithLighting(handler)`         | Sets the `light.LightingHandler` for the scene. Enables lighting, shadows, and Forward+ culling.    |
| `WithPhysics(ph)`               | Sets the `physics.Physics` handler for the scene. Enables GPU-driven physics simulation.            |
| `WithScreenSize(width, height)` | Sets the initial screen dimensions for light culling tile calculations and shadow map setup.         |

---

## Scene Interface

### Object Management

| Method                                           | Description                                                                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `Add(obj, pipelineOpts...) uint64`               | Adds a GameObject, auto-creates/reuses an Animator, registers pipelines, inits GPU resources, returns the assigned ID. |
| `Get(id) GameObject`                             | Retrieves a non-ephemeral object by ID, or `nil`.                                                                      |
| `Remove(id)`                                     | Removes a non-ephemeral object and swap-removes its instance from the animator.                                        |
| `Count() int`                                    | Number of persisted (non-ephemeral) objects.                                                                           |
| `CountEphemeral() int`                           | Total instance count across all animators.                                                                             |

Shaders are resolved automatically inside `Add` based on whether the model is skinned and whether lighting is enabled — no shader parameters are needed.

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

### Physics

| Method                                   | Description                                   |
| ---------------------------------------- | --------------------------------------------- |
| `SetPhysicsHandler(ph physics.Physics)`  | Sets the physics handler for GPU simulation.  |

### Resize

| Method                        | Description                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `Resize(width, height int)`   | Reconfigures the renderer’s surface size after a window resize.                    |

### Lighting

| Method                       | Description                                                                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `AddLight(l)`                | Adds a light source. On the first call, lazily initializes the full lighting pipeline (bind groups, shadow map, tile culling).    |
| `RemoveLight(l)`             | Removes a light by reference.                                                                                                    |
| `DetachLight(obj)`           | Detaches an object’s auto-registered light. Required for ephemeral objects.                                                       |
| `Lights() []Light`           | Returns a copy of all registered lights.                                                                                         |
| `AmbientColor() [3]float32`  | Returns the scene’s ambient RGB color.                                                                                            |
| `SetAmbientColor(color)`     | Sets the ambient RGB color.                                                                                                      |

All lighting/shadow/culling initialization methods (`initLightBindGroup`, `initShadowMap`, `initShadowLitBindGroup`, `initLightCullResources`, `initLighting`) are **unexported** and called automatically by `AddLight` on the first light addition. Shadow configuration is handled by the `light.LightingHandler` passed via `WithLighting`.

### Frame Methods

| Method                      | Description                                                                                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PrepareCompute(deltaTime)` | Updates camera, syncs light positions, advances animations, uploads buffers, dispatches compute shaders. Must be called within `BeginComputeFrame`/`EndComputeFrame`. |
| `DrawCalls() error`         | Issues instanced draw calls for all animators. Must be called within `BeginFrame`/`EndFrame`. Uses indirect draw when frustum culling is active.                      |

---

## Frame Lifecycle

A typical frame follows this order:

```
1. renderer.BeginComputeFrame()
   scene.PrepareCompute(dt)          — camera update, animation, buffer writes, compute dispatch
   renderer.EndComputeFrame()

2. (internal) prepareLightCulling    — Forward+ tile culling (automatic if lighting enabled)

3. (internal) prepareShadows         — shadow depth pass (automatic if lighting enabled)

4. renderer.BeginFrame()
   scene.DrawCalls()                 — instanced draw calls (regular or indirect)
   renderer.EndFrame()

5. renderer.Present()
```

Steps 2 and 3 are handled internally when lighting is enabled via `WithLighting` and lights have been added. For unlit scenes these steps are skipped automatically.

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
