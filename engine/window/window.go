// Package window provides platform windowing, input event handling, and WebGPU surface creation.
//
// The [Window] interface abstracts GLFW behind a common API that exposes resize,
// keyboard, mouse, and scroll callbacks. Instances are created via [NewWindow]
// using the option-builder pattern.
package window

import (
	"fmt"
	"runtime"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/cogentcore/webgpu/wgpu"
)

// Window provides platform windowing and input event handling.
// Wraps platform-specific window implementations with a common interface.
type Window interface {
	common.Delegate[Window]

	// SetUpdateCallback sets the function called each message loop iteration.
	//
	// Parameters:
	//   - callback: function to call (or nil to disable)
	SetUpdateCallback(callback func())

	// SetResizeCallback sets the function called when the window is resized.
	//
	// Parameters:
	//   - callback: function receiving new width and height in pixels
	SetResizeCallback(callback func(width, height int))

	// SetScrollCallback sets the callback for mouse scroll wheel events.
	//
	// Parameters:
	//   - callback: function receiving scroll delta (positive = up/zoom in, negative = down/zoom out)
	SetScrollCallback(callback func(delta float32))

	// SetKeyDownCallback sets the callback for key press events.
	//
	// Parameters:
	//   - callback: function receiving the virtual key code
	SetKeyDownCallback(callback func(keyCode uint32))

	// SetKeyUpCallback sets the callback for key release events.
	//
	// Parameters:
	//   - callback: function receiving the virtual key code
	SetKeyUpCallback(callback func(keyCode uint32))

	// SetMiddleMouseDownCallback sets the callback for middle mouse button press.
	//
	// Parameters:
	//   - callback: function receiving mouse x, y position
	SetMiddleMouseDownCallback(callback func(x, y int32))

	// SetMiddleMouseUpCallback sets the callback for middle mouse button release.
	//
	// Parameters:
	//   - callback: function receiving mouse x, y position
	SetMiddleMouseUpCallback(callback func(x, y int32))

	// SetMouseMoveCallback sets the callback for mouse movement.
	//
	// Parameters:
	//   - callback: function receiving mouse x, y position
	SetMouseMoveCallback(callback func(x, y int32))

	// SurfaceDescriptor returns a wgpu.SurfaceDescriptor suitable for creating a WebGPU surface.
	// The descriptor is platform-appropriate (Windows HWND, X11 Xlib, Wayland, macOS Metal, etc.)
	// and is created by the wgpuglfw bridge from the underlying GLFW window.
	//
	// Returns:
	//   - *wgpu.SurfaceDescriptor: the platform-specific surface descriptor, or nil if window is not initialized
	SurfaceDescriptor() *wgpu.SurfaceDescriptor

	// IsRunning returns true if the window is still active.
	//
	// Returns:
	//   - bool: true if window is running, false if closed
	IsRunning() bool

	// Close closes the window and releases platform resources.
	//
	// Returns:
	//   - error: error if close operation fails
	Close() error

	// ProcessMessages runs the window message loop.
	// Blocks until the window is closed. Calls OnUpdate callback each iteration.
	ProcessMessages()

	// Width returns the current window client area width in pixels.
	//
	// Returns:
	//   - int: width in pixels
	Width() int

	// Height returns the current window client area height in pixels.
	//
	// Returns:
	//   - int: height in pixels
	Height() int
}

var _ Window = &window{}

func (w *window) SetUpdateCallback(callback func())                  { w.onUpdate = callback }
func (w *window) SetResizeCallback(callback func(width, height int)) { w.onResize = callback }
func (w *window) SetScrollCallback(callback func(delta float32))     { w.onScroll = callback }
func (w *window) SetKeyDownCallback(callback func(keyCode uint32))   { w.onKeyDown = callback }
func (w *window) SetKeyUpCallback(callback func(keyCode uint32))     { w.onKeyUp = callback }
func (w *window) SetMiddleMouseUpCallback(callback func(x, y int32)) { w.onMiddleMouseUp = callback }
func (w *window) SetMouseMoveCallback(callback func(x, y int32))     { w.onMouseMove = callback }
func (w *window) SurfaceDescriptor() *wgpu.SurfaceDescriptor {
	if w.backend == nil {
		return nil
	}
	return w.backend.surfaceDescriptor()
}

func (w *window) IsRunning() bool {
	if w.backend == nil {
		return false
	}
	return w.backend.isRunning()
}

func (w *window) Close() error {
	if w.backend == nil {
		return fmt.Errorf("window is not initialized")
	}
	return w.backend.close()
}
func (w *window) Width() int  { return w.width }
func (w *window) Height() int { return w.height }

func (w *window) SetMiddleMouseDownCallback(callback func(x, y int32)) {
	w.onMiddleMouseDown = callback
}

func (w *window) ProcessMessages() {
	for w.Delegate.IsRunning() {
		if !w.backend.processMessages() {
			break
		}
		if w.onUpdate != nil {
			w.onUpdate()
		}
		runtime.Gosched()
	}
}
