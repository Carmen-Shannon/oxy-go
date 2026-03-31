# Window System

The `engine/window` package provides a platform-abstracted windowing layer built on GLFW. It handles window creation, input callbacks (keyboard, mouse, scroll), resize events, and exposes a `wgpu.SurfaceDescriptor` for WebGPU surface creation. The window runs the main message loop and drives the engine's update cycle.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/window`

---

## Architecture

```
Window (public interface, embeds common.Delegate[Window])
 └─ window (unexported struct in window_impl.go)
      └── glfwWindow (platform-specific GLFW state)
```

The public `Window` interface is implemented by `window`. The `window` struct holds a `backend platformBackend` field (defined in `window_impl.go`). The `platformBackend` interface exposes four methods — `isRunning()`, `processMessages()`, `surfaceDescriptor()`, and `close()` — that abstract all platform-specific operations. `window_glfw.go` provides the `glfwWindow` concrete implementation of this interface.

---

## Constructor

```go
func NewWindow(options ...WindowBuilderOption) Window
```

Creates a new Window with default configuration, applies each option in order, initializes the underlying GLFW window, and sets the delegation target to itself (`w.Delegate = w`). Panics if the platform window fails to create.

**Defaults:**

| Property   | Default                  |
| ---------- | ------------------------ |
| Title      | `"Default Window Title"` |
| Width      | `1280`                   |
| Height     | `720`                    |
| Min Width  | `600`                    |
| Min Height | `200`                    |
| Max Width  | `1600`                   |
| Max Height | `1200`                   |

---

## Builder Options

| Option                     | Description                                       |
| -------------------------- | ------------------------------------------------- |
| `WithTitle(title)`         | Sets the window title displayed in the title bar. |
| `WithWidth(width)`         | Sets the initial window width in pixels.          |
| `WithHeight(height)`       | Sets the initial window height in pixels.         |
| `WithMinWidth(minWidth)`   | Sets the minimum allowed window width.            |
| `WithMinHeight(minHeight)` | Sets the minimum allowed window height.           |
| `WithMaxWidth(maxWidth)`   | Sets the maximum allowed window width.            |
| `WithMaxHeight(maxHeight)` | Sets the maximum allowed window height.           |

---

## Window Interface

### Delegation

The `Window` interface embeds `common.Delegate[Window]`, exposing `SetDelegate(delegate Window)`. In production code the delegate is set to the instance itself during construction. In test code the delegate can be replaced with a mock.

### Lifecycle

| Method              | Description                                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ProcessMessages()` | Runs the blocking message loop. Polls GLFW events, calls the update callback, and yields with `runtime.Gosched()` each iteration until the window is closed. |
| `IsRunning() bool`  | Returns `true` if the window is still active (not closed).                                                                                                   |
| `Close() error`     | Destroys the GLFW window and terminates the GLFW library.                                                                                                    |

### Dimensions

| Method         | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `Width() int`  | Current window client area width in pixels (framebuffer size on high-DPI).  |
| `Height() int` | Current window client area height in pixels (framebuffer size on high-DPI). |

### Surface

| Method                                        | Description                                                                                                                                   |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `SurfaceDescriptor() *wgpu.SurfaceDescriptor` | Returns a platform-appropriate WebGPU surface descriptor from the GLFW window via `wgpuglfw`. Returns `nil` if the window is not initialized. |

### Input Callbacks

All callbacks accept a function (or `nil` to disable):

| Method                       | Signature                 | Description                                        |
| ---------------------------- | ------------------------- | -------------------------------------------------- |
| `SetUpdateCallback`          | `func()`                  | Called each message loop iteration.                |
| `SetResizeCallback`          | `func(width, height int)` | Called on window resize with new pixel dimensions. |
| `SetScrollCallback`          | `func(delta float32)`     | Mouse scroll wheel (positive = up/zoom in).        |
| `SetKeyDownCallback`         | `func(keyCode uint32)`    | Key press or repeat.                               |
| `SetKeyUpCallback`           | `func(keyCode uint32)`    | Key release.                                       |
| `SetMiddleMouseDownCallback` | `func(x, y int32)`        | Middle mouse button press with cursor position.    |
| `SetMiddleMouseUpCallback`   | `func(x, y int32)`        | Middle mouse button release with cursor position.  |
| `SetMouseMoveCallback`       | `func(x, y int32)`        | Mouse cursor movement.                             |

---

## GLFW Platform Layer

The platform layer lives in `window_glfw.go` and handles all GLFW-specific logic:

- **Window creation** — Disables OpenGL context (`glfw.NoAPI`) since WebGPU provides its own graphics API, creates the GLFW window, and registers all input callbacks.
- **High-DPI handling** — Uses `SetFramebufferSizeCallback` and `GetFramebufferSize` for pixel-accurate dimensions. On Retina/HiDPI displays, framebuffer size may differ from the requested window size.
- **Surface descriptor** — Uses the `wgpuglfw` bridge package for per-platform surface creation (Windows HWND, X11, Wayland, macOS Metal).
- **Message loop** — `glfw.PollEvents()` dispatches pending events without blocking.
- **Escape key** — Hardcoded to close the window via `SetShouldClose(true)`.

### Dependencies

| Package                                                                                  | Purpose                            |
| ---------------------------------------------------------------------------------------- | ---------------------------------- |
| [`go-gl/glfw/v3.3/glfw`](https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw)             | GLFW bindings for Go               |
| [`cogentcore/webgpu/wgpu`](https://pkg.go.dev/github.com/cogentcore/webgpu/wgpu)         | WebGPU types (`SurfaceDescriptor`) |
| [`cogentcore/webgpu/wgpuglfw`](https://pkg.go.dev/github.com/cogentcore/webgpu/wgpuglfw) | GLFW → WebGPU surface bridge       |

---

## Usage

```go
win := window.NewWindow(
    window.WithTitle("Oxy Engine"),
    window.WithWidth(1280),
    window.WithHeight(720),
)

win.SetResizeCallback(func(w, h int) {
    renderer.Resize(w, h)
})

win.SetUpdateCallback(func() {
    // per-frame engine tick + render
})

win.ProcessMessages() // blocks until window is closed
```

The Engine constructor typically creates the window and wires it into the engine loop — see `engine.go` and the examples for full integration.

---

## Files

| File                | Purpose                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------- |
| `window.go`         | `Window` interface and exported method implementations (callback setters, `ProcessMessages`) |
| `window_impl.go`    | `platformBackend` interface, unexported `window` struct definition                           |
| `window_builder.go` | `WindowBuilderOption` type and 7 builder functions                                           |
| `window_glfw.go`    | GLFW platform layer: window creation, input callbacks, surface descriptor, message polling   |
