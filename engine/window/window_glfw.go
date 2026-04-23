package window

import (
	"fmt"
	"runtime"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/cogentcore/webgpu/wgpuglfw"
	"github.com/go-gl/glfw/v3.4/glfw"
)

// glfwWindow holds the GLFW-specific window state.
type glfwWindow struct {
	parent  *window
	window  *glfw.Window
	running bool
}

// newPlatformWindow creates the GLFW window with input callbacks and returns it as a platformBackend.
//
// GLFW reference: https://www.glfw.org/docs/latest/window_guide.html
// go-gl/glfw: https://pkg.go.dev/github.com/go-gl/glfw/v3.4/glfw
func newPlatformWindow(w *window) (platformBackend, error) {
	runtime.LockOSThread()

	if err := glfw.Init(); err != nil {
		return nil, fmt.Errorf("failed to initialize GLFW: %v", err)
	}

	// WebGPU provides its own graphics API, so disable OpenGL context creation.
	// Reference: https://www.glfw.org/docs/latest/window_guide.html#window_hints_ctx
	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)

	win, err := glfw.CreateWindow(w.width, w.height, w.title, nil, nil)
	if err != nil {
		glfw.Terminate()
		return nil, fmt.Errorf("failed to create GLFW window: %v", err)
	}

	gw := &glfwWindow{
		parent:  w,
		window:  win,
		running: true,
	}

	// Register GLFW callbacks for input and window events.
	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.4/glfw#Window.SetKeyCallback
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

	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.4/glfw#Window.SetScrollCallback
	win.SetScrollCallback(func(_ *glfw.Window, xoff, yoff float64) {
		if w.onScroll != nil {
			w.onScroll(float32(yoff))
		}
	})

	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.4/glfw#Window.SetMouseButtonCallback
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

	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.4/glfw#Window.SetCursorPosCallback
	win.SetCursorPosCallback(func(_ *glfw.Window, xpos, ypos float64) {
		if w.onMouseMove != nil {
			w.onMouseMove(int32(xpos), int32(ypos))
		}
	})

	// Use framebuffer size callback for pixel-accurate resize events.
	// On high-DPI displays (e.g., macOS Retina), framebuffer size differs from window size.
	// The renderer requires pixel dimensions for correct surface configuration.
	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.4/glfw#Window.SetFramebufferSizeCallback
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

	return gw, nil
}

func (gw *glfwWindow) isRunning() bool {
	return gw.running && !gw.window.ShouldClose()
}

func (gw *glfwWindow) processMessages() bool {
	glfw.PollEvents()
	return gw.isRunning()
}

func (gw *glfwWindow) surfaceDescriptor() *wgpu.SurfaceDescriptor {
	return wgpuglfw.GetSurfaceDescriptor(gw.window)
}

func (gw *glfwWindow) close() error {
	gw.running = false
	gw.window.SetShouldClose(true)
	gw.window.Destroy()
	glfw.Terminate()
	return nil
}
