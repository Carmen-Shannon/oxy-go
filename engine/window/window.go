package window

import (
	"fmt"
	"runtime"

	"github.com/cogentcore/webgpu/wgpu"
)

// Window provides platform windowing and input event handling.
// Wraps platform-specific window implementations with a common interface.
type Window interface {
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

// engineWindow is the implementation of the Window interface.
// Holds window configuration, GLFW state, and event callbacks.
type engineWindow struct {
	// title is the window title displayed in the title bar.
	title string

	// maxWidth is the maximum allowed window width during resize.
	maxWidth int

	// maxHeight is the maximum allowed window height during resize.
	maxHeight int

	// minWidth is the minimum allowed window width during resize.
	minWidth int

	// minHeight is the minimum allowed window height during resize.
	minHeight int

	// width is the current window client area width in pixels.
	width int

	// height is the current window client area height in pixels.
	height int

	// internalWindow holds the platform-specific window data (glfwWindow).
	internalWindow any

	// onUpdate is called each iteration of the message loop (if set).
	onUpdate func()

	// onResize is called when the window is resized.
	onResize func(width, height int)

	// onScroll is called for mouse wheel events.
	// Positive delta = scroll up (zoom in), negative = scroll down (zoom out).
	onScroll func(delta float32)

	// onKeyDown is called when a key is pressed.
	onKeyDown func(keyCode uint32)

	// onKeyUp is called when a key is released.
	onKeyUp func(keyCode uint32)

	// onMiddleMouseDown is called when the middle mouse button is pressed.
	onMiddleMouseDown func(x, y int32)

	// onMiddleMouseUp is called when the middle mouse button is released.
	onMiddleMouseUp func(x, y int32)

	// onMouseMove is called when the mouse moves within the window.
	onMouseMove func(x, y int32)
}

var _ Window = &engineWindow{}

// NewWindow creates a new Window with the specified options.
// Applies default values first, then each option in order.
//
// Parameters:
//   - options: functional options to configure the window
//
// Returns:
//   - Window: the configured window (not yet spawned)
func NewWindow(options ...WindowBuilderOption) Window {
	w := &engineWindow{
		title:     "Default Window Title",
		maxWidth:  1600,
		maxHeight: 1200,
		minWidth:  600,
		minHeight: 200,
		width:     1280,
		height:    720,
	}
	for _, opt := range options {
		opt(w)
	}
	if err := newPlatformWindow(w); err != nil {
		panic(fmt.Sprintf("failed to create platform window: %v", err))
	}
	return w
}

func (w *engineWindow) SetUpdateCallback(callback func()) {
	w.onUpdate = callback
}

func (w *engineWindow) SetResizeCallback(callback func(width, height int)) {
	w.onResize = callback
}

func (w *engineWindow) SetScrollCallback(callback func(delta float32)) {
	w.onScroll = callback
}

func (w *engineWindow) SetKeyDownCallback(callback func(keyCode uint32)) {
	w.onKeyDown = callback
}

func (w *engineWindow) SetKeyUpCallback(callback func(keyCode uint32)) {
	w.onKeyUp = callback
}

func (w *engineWindow) SetMiddleMouseDownCallback(callback func(x, y int32)) {
	w.onMiddleMouseDown = callback
}

func (w *engineWindow) SetMiddleMouseUpCallback(callback func(x, y int32)) {
	w.onMiddleMouseUp = callback
}

func (w *engineWindow) SetMouseMoveCallback(callback func(x, y int32)) {
	w.onMouseMove = callback
}

func (w *engineWindow) SurfaceDescriptor() *wgpu.SurfaceDescriptor {
	return platformGetSurfaceDescriptor(w)
}

func (w *engineWindow) IsRunning() bool {
	return platformIsRunningCheck(w)
}

func (w *engineWindow) Close() error {
	return platformCloseWindow(w)
}

func (w *engineWindow) ProcessMessages() {
	for w.IsRunning() {
		if succ := platformProcessMessages(w); !succ {
			break
		}

		if w.onUpdate != nil {
			w.onUpdate()
		}

		runtime.Gosched()
	}
}

func (w *engineWindow) Width() int {
	return w.width
}

func (w *engineWindow) Height() int {
	return w.height
}
