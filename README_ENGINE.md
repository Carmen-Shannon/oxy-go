# Engine

The `engine` package is the **main entrypoint** of the oxy-go package. It represents the highest-level instance of the engine itself, containing all render and game logic. The Engine owns the window, manages scenes by z-index, and drives two concurrent loops — a fixed-rate **tick loop** for game logic and an uncapped (or frame-limited) **render loop** that executes the full GPU frame lifecycle across all active scenes.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine`

---

## Architecture

```
Engine (public interface)
 └─ engine (unexported struct)
      ├── Window              — GLFW window, message loop, input callbacks
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

Creates a new Engine with sensible defaults, applies each option in order, and wires the window's resize callback to propagate dimension changes to all scenes' renderers and cameras.

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

| Method            | Description                             |
| ----------------- | --------------------------------------- |
| `Window() Window` | Returns the underlying window instance. |

### Tick & Render

| Method                        | Description                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `SetTickRate(fps)`            | Sets the engine tick rate in Hz. Takes effect immediately if the engine is running (sent via channel). |
| `SetTickCallback(callback)`   | Registers the function called each engine tick. Receives `deltaTime float32` in seconds.               |
| `SetRenderCallback(callback)` | Registers the function called each render frame. Receives `deltaTime float32` in seconds.              |
| `SetRenderFrameLimit(fps)`    | Sets an optional render frame rate cap. Pass 0 to uncap.                                               |

### Profiling

| Method              | Description                                      |
| ------------------- | ------------------------------------------------ |
| `EnableProfiler()`  | Enables performance profiling output to the log. |
| `DisableProfiler()` | Disables performance profiling output.           |

### Scene Management

| Method                   | Description                                                      |
| ------------------------ | ---------------------------------------------------------------- |
| `AddScene(key, s)`       | Registers a scene at the given z-index. Lower keys render first. |
| `RemoveScene(key)`       | Removes the scene at the given z-index.                          |
| `Scene(key) Scene`       | Retrieves the scene at the given z-index, or `nil`.              |
| `Scenes() map[int]Scene` | Returns a copy of all registered scenes keyed by z-index.        |

---

## Render Loop Frame Lifecycle

Each iteration of `handleRender`, for all active scenes sorted by ascending z-index:

```
1. renderer.BeginComputeFrame()
   ── scene.PrepareCompute(dt)       for each active scene
   renderer.EndComputeFrame()

2. scene.PrepareShadows()             for each active scene

3. scene.PrepareLightCulling()        for each active scene

4. renderer.BeginFrame()
   ── scene.DrawCalls()              for each active scene
   renderer.EndFrame()
   renderer.Present()

5. renderCallback(dt)                 user render callback (if set)
6. profiler.Tick()                    profiling sample (if enabled)
7. frame rate limiting sleep          (if renderFrameLimit > 0)
```

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

| File                | Purpose                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `engine.go`         | `Engine` interface, `engine` struct, `NewEngine` constructor, goroutine loops, all method implementations |
| `engine_builder.go` | `EngineBuilderOption` type and 5 builder functions                                                        |
