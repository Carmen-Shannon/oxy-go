# Scene System

The `engine/scene` package is the central orchestrator of the oxy-go engine. A Scene owns a Camera, a Renderer, and a pool of Animators. It manages GameObjects, lights, shadows, Forward+ light culling, and all per-frame GPU work — from compute dispatch through draw calls. Scenes can be hot-swapped via the `Active` flag to switch between views or levels.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/scene`

---

## Architecture

```
Scene (public interface)
 └─ scene (unexported struct in scene_impl.go)
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

Creates a new Scene. Both required arguments (camera, renderer) must be non-nil — panics otherwise. The vertex shader is loaded internally from the engine's standard shader assets.

---

## Builder Options

The `NewScene` constructor accepts variadic `SceneBuilderOption` functions:

| Option                                              | Description                                                                                                |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `WithActive(active)`                                | Sets whether the scene starts active for rendering. Default: `false`.                                      |
| `WithObjects(objects...)`                           | Adds initial GameObjects. Assigns IDs and persists non-ephemeral objects.                                  |
| `WithComputeWorkers(n)`                             | Sets the number of parallel CPU prep goroutines. Default: `max(runtime.NumCPU()-1, 1)`.                    |
| `WithCullingDisabled(disabled)`                     | Disables GPU frustum culling. Default: `false` (culling enabled).                                          |
| `WithLighting(handler)`                             | Sets the `light.LightingHandler` for the scene. Enables lighting, shadows, and Forward+ culling.           |
| `WithPhysics(opts ...physics.PhysicsBuilderOption)` | Configures the physics subsystem with the given builder options; constructs the Physics handler internally |
| `WithScreenSize(width, height)`                     | Sets the initial screen dimensions for light culling tile calculations and shadow map setup.               |
| `WithMaxBonesGPU(n uint64)`                         | Sets the maximum number of bones per skinned model instance for GPU buffer allocation. Default: `64`.      |
| `WithLODEnabled(enabled bool)`                      | Enables per-frame distance-based LOD mesh selection.                                                       |
| `WithLODDistances(lod1, lod2 float32)`              | Camera distance thresholds: objects beyond `lod1` use LOD1; objects beyond `lod2` use LOD2.                |
| `WithLODShadowBias(bias int)`                       | Extra LOD levels applied to shadow rendering. Default: `1` (shadow passes use one coarser LOD level).      |

---

## Scene Interface

### Object Management

| Method                                       | Description                                                                                                            |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `AddGameObject(obj, pipelineOpts...) uint64` | Adds a GameObject, auto-creates/reuses an Animator, registers pipelines, inits GPU resources, returns the assigned ID. |
| `Get(id) GameObject`                         | Retrieves a non-ephemeral object by ID, or `nil`.                                                                      |
| `RemoveGameObject(id)`                       | Removes a non-ephemeral object and swap-removes its instance from the animator.                                        |
| `Count() int`                                | Number of persisted (non-ephemeral) objects.                                                                           |
| `CountEphemeral() int`                       | Total instance count across all animators.                                                                             |

Shaders are resolved automatically inside `AddGameObject` based on whether the model is skinned and whether lighting is enabled — no shader parameters are needed.

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

| Method                                  | Description                                  |
| --------------------------------------- | -------------------------------------------- |
| `SetPhysicsHandler(ph physics.Physics)` | Sets the physics handler for GPU simulation. |

### Resize

| Method                      | Description                                                     |
| --------------------------- | --------------------------------------------------------------- |
| `Resize(width, height int)` | Reconfigures the renderer’s surface size after a window resize. |

### Lighting

| Method                      | Description                                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `AddLight(l)`               | Adds a light source. On the first call, lazily initializes the full lighting pipeline (bind groups, shadow map, tile culling). |
| `RemoveLight(l)`            | Removes a light by reference.                                                                                                  |
| `DetachLight(obj)`          | Detaches an object’s auto-registered light. Required for ephemeral objects.                                                    |
| `Lights() []Light`          | Returns a copy of all registered lights.                                                                                       |
| `AmbientColor() [3]float32` | Returns the scene’s ambient RGB color.                                                                                         |
| `SetAmbientColor(color)`    | Sets the ambient RGB color.                                                                                                    |

All lighting/shadow/culling initialization methods (`initLighting`, `initShadowMap`, `initCSMShadowLitBindGroup`, `initGBuffer`, `initSSAO`, `initSSAOLitBindGroup`, `initContactShadows`, `initLightCullResources`, `initComposition`, `initSSR`, `initLightBindGroup`, `initPhysics`) are **unexported** and called automatically by `AddLight` on the first light addition. Shadow configuration is handled by the `light.LightingHandler` and its sub-handlers passed via `WithLighting`.

### Frame Methods

| Method                            | Description                                                                                                                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SyncFrameSlot(slot int)`         | Switches all dual-slot GPU resources to the given frame slot. Call once per frame after `SyncGPUTimestamps` and before any `Prepare*` calls.                                      |
| `PrepareCompute(deltaTime)`       | Updates camera matrices, advances animation state, uploads staged buffer writes, and dispatches all compute shaders. Must be called within `BeginComputeFrame`/`EndComputeFrame`. |
| `DrawCalls() error`               | Issues instanced draw calls for all animators. Must be called within `BeginFrame`/`EndFrame`. Uses indirect draw when frustum culling is active.                                  |
| `PrepareShadows()`                | Renders all shadow depth passes: dual-cascade CSM + spot/point cube shadow atlas with viewport switching.                                                                         |
| `PrepareLights()`                 | Marshals the current light list into the GPU light buffer. Must be called after `PrepareShadows` so shadow index assignments are up to date.                                      |
| `PrepareLightCulling()`           | Updates the light cull uniform buffer and dispatches the Forward+ tile assignment compute shader.                                                                                 |
| `PrepareGBuffer()`                | Renders the G-Buffer MRT pre-pass (normals, albedo, depth)                                                                                                                        |
| `PrepareSSAO()`                   | Dispatches the SSAO compute pass and bilateral blur                                                                                                                               |
| `PrepareContactShadows()`         | Dispatches the screen-space contact shadow ray march compute pass                                                                                                                 |
| `PrepareSSR()`                    | Dispatches the Hi-Z SSR compute pass                                                                                                                                              |
| `PrepareLuminance(dt float32)`    | Dispatches the GPU luminance compute pass to update the adapted exposure buffer. Must be called after `DrawCalls` and after `PrepareSSR`. No-op when auto-exposure is disabled.   |
| `PrepareBloom()`                  | Dispatches the bloom downsample and upsample compute passes. Must be called after `PrepareLuminance` and before `PrepareTAA`.                                                     |
| `PrepareTAA()`                    | Dispatches TAA resolve, blending current HDR with history using motion-vector reprojection and neighborhood variance clamping. Must run after `PrepareBloom` and before `PrepareComposition`. No-op if TAA is uninitialized or camera is nil. |
| `AcquireCompositionFrame() error` | Acquires the swapchain image for the composition pass. Must be called immediately before `PrepareComposition` each frame.                                                         |
| `PrepareComposition()`            | Runs the fullscreen composition pass: ACES tone mapping and gamma correction, writes the final LDR result to the swapchain.                                                       |
| `BeginHDRFrame() error`           | Opens the HDR render pass targeting the composition handler's textures. Must be called before `DrawCalls` each frame.                                                             |

### Per-Frame Render Order

The following sequence matches the engine's `handleRender` call order each frame:

1. `SyncFrameSlot` — switch dual-slot GPU resources to the current frame slot
2. `PrepareCompute` — camera update, animation advance, buffer writes, compute dispatch
3. `PrepareShadows` — shadow depth passes (CSM + spot/point atlas)
4. `PrepareLights` — marshal updated light list into the GPU light buffer
5. `PrepareGBuffer` — G-Buffer MRT pre-pass (normals, albedo, depth)
6. `PrepareLightCulling` — Forward+ tile assignment compute shader
7. `PrepareSSAO` — SSAO hemisphere sampling and bilateral blur
8. `PrepareContactShadows` — contact shadow screen-space ray march
9. `BeginHDRFrame` — open the HDR render pass
10. `DrawCalls` — instanced draw calls (regular or indirect)
11. `PrepareSSR` — SSR Hi-Z ray march compute shader
12. `PrepareLuminance` — luminance compute for auto-exposure adaptation
13. `PrepareBloom` — bloom downsample/upsample compute chain
14. `PrepareTAA` — temporal resolve compute pass (history reprojection + clamping)
15. `AcquireCompositionFrame` — acquire swapchain image
16. `PrepareComposition` — ACES tone mapping, write LDR to swapchain

---

## Frame Lifecycle

A typical frame follows this order:

```
1. renderer.BeginComputeFrame()
   scene.PrepareCompute(dt)          — camera update, animation, buffer writes, compute dispatch
   renderer.EndComputeFrame()

2. (internal) prepareLightCulling    — Forward+ tile culling (automatic if lighting enabled)

3. **Shadow pass** (`PrepareShadows`) — renders depth-only passes for the directional dual-cascade CSM atlas and any shadow-casting spot/point lights (16-tap Poisson PCF, no blur pass).

4. (internal) G-Buffer pre-pass      — MRT geometry pass (position, normal, albedo) when GI is active

5. (internal) SSAO compute           — hemisphere sampling + bilateral blur when SSAO handler present

6. **Lit draw calls** (`DrawCalls()`) — instanced draw calls (regular or indirect) with SSAO bind groups wired from the lighting handler.

7. (internal) SSR compute            — Hi-Z ray march when SSR handler present

7.5 (optional) Luminance compute     — `PrepareLuminance(dt)` dispatches the luminance compute shader (single 16×16 workgroup) to compute the log-average luminance of the HDR frame and temporally adapt the GPU-side exposure buffer. Only runs when auto-exposure is enabled on the `CompositionHandler`.

7.6 (optional) Bloom compute         — `PrepareBloom()` dispatches progressive downsample/upsample blur chain (13-tap box-tent down, 9-tap tent up) on bright HDR pixels. Runs its own compute frame. No-op when bloom is disabled.

7.7 (optional) TAA compute           — `PrepareTAA()` dispatches the TAA resolve shader, blending the current HDR frame with temporal history using motion-vector reprojection and neighborhood variance clamping. Runs after bloom and before composition. No-op when TAA is not initialized or the camera is nil.

8. (internal) Composition pass       — full-screen HDR→LDR tone mapping + SSR blend + bloom blend to swapchain

9. renderer.Present()
```

Steps 2–8 are handled internally when lighting is enabled via `WithLighting` and lights have been added. For unlit scenes these steps are skipped automatically. GI steps (4–5, 7–7.7, 8) are only executed when the corresponding sub-handlers are present on the `LightingHandler`.

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

| Provider Annotation         | Source                                   |
| --------------------------- | ---------------------------------------- |
| `@oxy:provider camera`      | Camera's BindGroupProvider               |
| `@oxy:provider material`    | Material's BindGroupProvider             |
| `@oxy:provider lights`      | Scene's light BGP                        |
| `@oxy:provider shadow`      | Scene's shadow lit BGP                   |
| `@oxy:provider tiles`       | Scene's tile lit BGP                     |
| `@oxy:provider ssao`        | LightingHandler's `"ssao_lit"` BGP       |
| `@oxy:provider composition` | CompositionHandler's `"composition"` BGP |
| `@oxy:provider ssr`         | SSRHandler's `"ssr_compute"` BGP         |
| `@oxy:provider effect`      | Model's effect provider                  |
| `@oxy:provider animator`    | Animator's output BGP                    |

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

| File               | Purpose                                                     |
| ------------------ | ----------------------------------------------------------- |
| `scene.go`         | `Scene` interface and exported method implementations       |
| `scene_impl.go`    | Unexported `scene` struct and all internal helper functions |
| `scene_builder.go` | `SceneBuilderOption` type and builder functions             |
