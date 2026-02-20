package window

import (
	"fmt"
	"runtime"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/cogentcore/webgpu/wgpuglfw"
	"github.com/go-gl/glfw/v3.3/glfw"
)

// glfwWindow holds the GLFW-specific window state.
type glfwWindow struct {
	parent  *engineWindow
	window  *glfw.Window
	running bool
}

// newPlatformWindow creates the GLFW window with input callbacks and stores it as the internal window.
//
// GLFW reference: https://www.glfw.org/docs/latest/window_guide.html
// go-gl/glfw: https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw
func newPlatformWindow(w *engineWindow) error {
	runtime.LockOSThread()

	if err := glfw.Init(); err != nil {
		return fmt.Errorf("failed to initialize GLFW: %v", err)
	}

	// WebGPU provides its own graphics API, so disable OpenGL context creation.
	// Reference: https://www.glfw.org/docs/latest/window_guide.html#window_hints_ctx
	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)

	win, err := glfw.CreateWindow(w.width, w.height, w.title, nil, nil)
	if err != nil {
		glfw.Terminate()
		return fmt.Errorf("failed to create GLFW window: %v", err)
	}

	gw := &glfwWindow{
		parent:  w,
		window:  win,
		running: true,
	}
	w.internalWindow = gw

	// Register GLFW callbacks for input and window events.
	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw#Window.SetKeyCallback
	win.SetKeyCallback(func(_ *glfw.Window, key glfw.Key, scancode int, action glfw.Action, mods glfw.ModifierKey) {
		if key == glfw.KeyEscape && action == glfw.Press {
			gw.running = false
			win.SetShouldClose(true)
			return
		}
		switch action {
		case glfw.Press, glfw.Repeat:
			if w.onKeyDown != nil {
				w.onKeyDown(uint32(key))
			}
		case glfw.Release:
			if w.onKeyUp != nil {
				w.onKeyUp(uint32(key))
			}
		}
	})

	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw#Window.SetScrollCallback
	win.SetScrollCallback(func(_ *glfw.Window, xoff, yoff float64) {
		if w.onScroll != nil {
			w.onScroll(float32(yoff))
		}
	})

	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw#Window.SetMouseButtonCallback
	win.SetMouseButtonCallback(func(_ *glfw.Window, button glfw.MouseButton, action glfw.Action, mods glfw.ModifierKey) {
		if button == glfw.MouseButtonMiddle {
			xpos, ypos := win.GetCursorPos()
			switch action {
			case glfw.Press:
				if w.onMiddleMouseDown != nil {
					w.onMiddleMouseDown(int32(xpos), int32(ypos))
				}
			case glfw.Release:
				if w.onMiddleMouseUp != nil {
					w.onMiddleMouseUp(int32(xpos), int32(ypos))
				}
			}
		}
	})

	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw#Window.SetCursorPosCallback
	win.SetCursorPosCallback(func(_ *glfw.Window, xpos, ypos float64) {
		if w.onMouseMove != nil {
			w.onMouseMove(int32(xpos), int32(ypos))
		}
	})

	// Use framebuffer size callback for pixel-accurate resize events.
	// On high-DPI displays (e.g., macOS Retina), framebuffer size differs from window size.
	// The renderer requires pixel dimensions for correct surface configuration.
	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw#Window.SetFramebufferSizeCallback
	win.SetFramebufferSizeCallback(func(_ *glfw.Window, width, height int) {
		w.width = width
		w.height = height
		if w.onResize != nil {
			w.onResize(width, height)
		}
	})

	// Update stored dimensions to reflect actual framebuffer size (may differ from requested on high-DPI).
	fbWidth, fbHeight := win.GetFramebufferSize()
	w.width = fbWidth
	w.height = fbHeight

	return nil
}

// platformGetSurfaceDescriptor creates a platform-appropriate wgpu.SurfaceDescriptor from the GLFW window.
// Uses the wgpuglfw bridge package which has per-platform implementations (Windows, X11, Wayland, macOS).
//
// Reference: https://pkg.go.dev/github.com/cogentcore/webgpu/wgpuglfw#GetSurfaceDescriptor
func platformGetSurfaceDescriptor(w *engineWindow) *wgpu.SurfaceDescriptor {
	if w.internalWindow == nil {
		return nil
	}
	gw := w.internalWindow.(*glfwWindow)
	return wgpuglfw.GetSurfaceDescriptor(gw.window)
}

// platformIsRunningCheck returns whether the GLFW window is still active.
// Returns false if the internal window is nil, the running flag is cleared, or GLFW reports ShouldClose.
//
// Parameters:
//   - w: the engineWindow to check
//
// Returns:
//   - bool: true if the window is still running
func platformIsRunningCheck(w *engineWindow) bool {
	if w.internalWindow == nil {
		return false
	}
	gw := w.internalWindow.(*glfwWindow)
	return gw.running && !gw.window.ShouldClose()
}

// platformCloseWindow destroys the GLFW window and terminates the GLFW library.
// Returns an error if the internal window has not been initialized.
//
// Parameters:
//   - w: the engineWindow to close
//
// Returns:
//   - error: error if the window is not initialized
func platformCloseWindow(w *engineWindow) error {
	if w.internalWindow == nil {
		return fmt.Errorf("window is not initialized")
	}
	gw := w.internalWindow.(*glfwWindow)
	gw.running = false
	gw.window.SetShouldClose(true)
	gw.window.Destroy()
	glfw.Terminate()
	return nil
}

// platformProcessMessages polls GLFW for pending events without blocking.
// This is the GLFW equivalent of the Win32 PeekMessage loop.
//
// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw#PollEvents
func platformProcessMessages(w *engineWindow) bool {
	glfw.PollEvents()
	return platformIsRunningCheck(w)
}
