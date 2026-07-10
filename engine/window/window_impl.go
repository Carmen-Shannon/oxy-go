package window

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/oliverbestmann/webgpu/wgpu"
)

// platformBackend abstracts the platform-specific windowing operations.
// Implemented by glfwWindow for production; replaced by a fake in tests.
type platformBackend interface {
	isRunning() bool
	processMessages() bool
	surfaceDescriptor() *wgpu.SurfaceDescriptor
	close() error
}

// window is the implementation of the Window interface.
// Holds window configuration, GLFW state, and event callbacks.
type window struct {
	common.DelegateImpl[Window]

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

	// backend is the platform-specific windowing backend.
	backend platformBackend

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
