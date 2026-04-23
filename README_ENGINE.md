# Engine

The `engine` package is the **main entrypoint** of the oxy-go package. It represents the highest-level instance of the engine itself, containing all render and game logic. The Engine owns the window, manages scenes by z-index, and drives two concurrent loops — a fixed-rate **tick loop** for game logic and an uncapped (or frame-limited) **render loop** that executes the full GPU frame lifecycle across all active scenes.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine`

---

## Architecture

```
Engine (public interface)
 └─ engine (unexported struct)
      ├── window              — GLFW window, message loop, input callbacks
      ├── scenes              — map[int]Scene keyed by z-index (render order)
      ├── tickCallback        — fixed-rate game logic callback
      ├── renderCallback      — per-frame render callback
      ├── Profiler            — optional frame timing profiler
      └── goroutines
           ├── handleEngine   — fixed-rate tick loop
           ├── handleRender   — uncapped render loop (compute → shadow → cull → draw)
           └── handleQuit     — quit signal listener
```

The Engine spawns three goroutines when `Run()` is called:

1. **handleEngine** — Fires the tick callback at the configured rate (default 60 Hz). Supports dynamic rate changes at runtime via `SetTickRate`.
2. **handleRender** — Iterates active scenes in ascending z-index order and executes the full frame lifecycle: compute dispatch, shadow pass, light culling, draw calls, and present. Recovers from panics to avoid crashing the process.
3. **handleQuit** — Blocks on the quit channel and decrements the WaitGroup when shutdown is signalled.

The window's `ProcessMessages()` blocks on the main thread (required by GLFW/OS), while the engine and render loops run concurrently.

---

## Constructor

```go
func NewEngine(options ...EngineBuilderOption) Engine
```

Creates a new Engine with sensible defaults and applies each option in order. When a window is provided via `WithWindow()`, wires the window's resize callback to call `Scene.Resize(width, height)` on all registered scenes.

**Defaults:**

| Property           | Default        |
| ------------------ | -------------- |
| Tick Rate          | 60 Hz          |
| Render Frame Limit | Uncapped (`0`) |
| Profiling          | Disabled       |
| Scenes             | Empty map      |

---

## Builder Options

| Option                      | Description                                                        |
| --------------------------- | ------------------------------------------------------------------ |
| `WithProfiling(enabled)`    | Enables or disables performance profiling output.                  |
| `WithTickRate(fps)`         | Sets the engine tick rate in Hz. Values ≤ 0 default to 60.         |
| `WithWindow(w)`             | Sets a pre-configured Window instead of creating one internally.   |
| `WithScene(key, s)`         | Registers a scene at the given z-index during construction.        |
| `WithRenderFrameLimit(fps)` | Sets an optional render frame rate cap. Pass 0 to uncap (default). |

---

## Engine Interface

### Lifecycle

| Method   | Description                                                                                                        |
| -------- | ------------------------------------------------------------------------------------------------------------------ |
| `Run()`  | Starts the engine, render, and quit goroutines, then blocks on the window message loop until the window is closed. |
| `Quit()` | Signals all goroutines to stop. Safe to call multiple times (uses `sync.Once`).                                    |

### Window

| Method                   | Description                             |
| ------------------------ | --------------------------------------- |
| `Window() window.Window` | Returns the underlying window instance. |

### Tick & Render

| Method                        | Description                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `SetTickRate(hz)`             | Sets the engine tick rate in Hz. Takes effect immediately if the engine is running (sent via channel). |
| `SetTickCallback(callback)`   | Registers the function called each engine tick. Receives `deltaTime float32` in seconds.               |
| `SetRenderFrameLimit(fps)`    | Sets an optional render frame rate cap. Pass 0 to uncap.                                               |

### Profiling

| Method              | Description                                      |
| ------------------- | ------------------------------------------------ |
| `EnableProfiler()`  | Enables performance profiling output to the log. |
| `DisableProfiler()` | Disables performance profiling output.           |

### Scene Management

| Method                         | Description                                                      |
| ------------------------------ | ---------------------------------------------------------------- |
| `AddScene(key, s)`             | Registers a scene at the given z-index. Lower keys render first. |
| `RemoveScene(key)`             | Removes the scene at the given z-index.                          |
| `Scene(key) scene.Scene`       | Retrieves the scene at the given z-index, or `nil`.              |
| `Scenes() map[int]scene.Scene` | Returns a copy of all registered scenes keyed by z-index.        |

---

## Render Loop Frame Lifecycle

Each iteration of `handleRender`, for all active scenes sorted by ascending z-index:

```
Phase A — Compute (animation, physics):
  renderer.SyncGPUTimestamps()
  scene.SyncFrameSlot(renderer.CurrentFrameSlot())    for each active scene
  renderer.BeginComputeFrame()
  ── scene.PrepareCompute(dt)                          for each active scene
  renderer.EndComputeFrame()

Phase B — Geometry (shadows, lights, G-Buffer):
  renderer.BeginGeometryFrame()
  ── scene.PrepareShadows()                            for each active scene
  ── scene.PrepareLights()                             for each active scene
  ── scene.PrepareGBuffer()                            for each active scene
  renderer.EndGeometryFrame()

Phase C — Compute (light culling, SSAO, contact shadows):
  renderer.BeginComputeFrame()
  ── scene.PrepareLightCulling()                       for each active scene
  ── scene.PrepareSSAO()                               for each active scene
  ── scene.PrepareContactShadows()                     for each active scene
  renderer.EndComputeFrame()

Phase D — HDR lit draw:
  scene.BeginHDRFrame()
  ── scene.DrawCalls()                                 for each active scene
  renderer.EndFrame()

Phase E — Post-process compute (SSR, luminance, bloom, TAA):
  renderer.BeginComputeFrame()
  ── scene.PrepareSSR()                                for each active scene
  ── scene.PrepareLuminance(dt)                        for each active scene
  ── scene.PrepareBloom()                              for each active scene
  ── scene.PrepareTAA()                                for each active scene
  renderer.EndComputeFrame()

Phase F — Composition and present:
  scene.AcquireCompositionFrame()
  ── scene.PrepareComposition()                        for each active scene (if acquire succeeded)
  renderer.FlushFrame()                                batched GPU submit for all phases A–F
  renderer.Present()

Post-frame:
  renderCallback(dt)                                   user render callback (if set)
  profiler.Tick()                                      profiling sample (if enabled)
  frame rate limiting sleep                            (if renderFrameLimit > 0)
```

**Ordering constraints:**

- All command buffers from phases A–F are accumulated and submitted in a single `FlushFrame()` call to minimise `vkQueueSubmit` overhead.
- `PrepareLights()` must be called **after** `PrepareShadows()` because shadow slot assignments must be populated before the light buffer is marshalled.
- `PrepareSSR()` must be called **after** `DrawCalls()` because SSR reads the populated HDR texture.
- `PrepareTAA()` must be called **after** `PrepareBloom()` and before composition (`AcquireCompositionFrame()` / `PrepareComposition()`).

All active scenes sharing the same renderer are rendered within a single render pass, enabling layered compositing by z-index order.

---

## Tick Rate

The engine tick rate controls how frequently the tick callback fires. It defaults to 60 Hz and can be changed at construction time or at runtime:

- **At construction:** `WithTickRate(120)` sets 120 Hz before the engine starts.
- **At runtime:** `SetTickRate(120)` sends the new rate to the engine goroutine via a buffered channel, taking effect on the next tick.

Values ≤ 0 are clamped to the default of 60 Hz.

---

## Shutdown

Shutdown can be triggered in two ways:

1. **Window close** — Closing the GLFW window (or pressing Escape) ends `ProcessMessages()`, which returns from `Run()`.
2. **Programmatic** — Calling `Quit()` closes the quit channel via `sync.Once`, signalling all goroutines to exit.

The `handleRender` goroutine includes a `recover()` guard — if a panic occurs during rendering, it logs the error and calls `Quit()` to shut down gracefully.

---

## Files

| File                | Purpose                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `engine.go`         | `Engine` interface definition and thin method delegation implementations                                     |
| `engine_impl.go`    | `engine` unexported struct, goroutine loops (`handleEngine`, `handleRender`, `handleQuit`), internal helpers |
| `engine_builder.go` | `EngineBuilderOption` type, 5 `With*` builder functions, and `NewEngine` constructor                         |
