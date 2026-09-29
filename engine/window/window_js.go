//go:build js

package window

import (
	"sync"
	"syscall/js"

	"github.com/Carmen-Shannon/webgpu/wgpu"

	"github.com/Carmen-Shannon/oxy-go/common"
)

// jsWindow holds the browser-specific window state for GOOS=js GOARCH=wasm builds.
// The HTML canvas element doubles as both the window surface and the WebGPU surface.
// See https://pkg.go.dev/syscall/js for the browser interop package used here.
type jsWindow struct {
	parent  *window
	canvas  js.Value
	running bool
	// listeners records each registered event listener (target, type, and the
	// js.Func wrapping its callback) so close can detach and release all of them.
	listeners []listenerRecord
}

// listenerRecord pairs a registered js.Func with the DOM event it was attached to.
type listenerRecord struct {
	target    js.Value
	eventType string
	fn        js.Func
}

// compile-time check that jsWindow satisfies the platformBackend contract.
var _ platformBackend = &jsWindow{}

// once guards the lazy initialization of the browserKeyCodes table.
var once sync.Once

// browserKeyCodes maps DOM KeyboardEvent.code strings to the engine's GLFW-compatible
// virtual key codes defined in the common package.
// See https://developer.mozilla.org/docs/Web/API/KeyboardEvent/code
var browserKeyCodes map[string]uint32

// initKeyCodes builds the code-to-keycode mapping table on first use.
func initKeyCodes() {
	browserKeyCodes = map[string]uint32{
		"KeyW":          common.KeyW,
		"KeyA":          common.KeyA,
		"KeyS":          common.KeyS,
		"KeyD":          common.KeyD,
		"KeyQ":          common.KeyQ,
		"KeyE":          common.KeyE,
		"KeyB":          common.KeyB,
		"KeyC":          common.KeyC,
		"KeyF":          common.KeyF,
		"KeyG":          common.KeyG,
		"KeyH":          common.KeyH,
		"KeyI":          common.KeyI,
		"KeyJ":          common.KeyJ,
		"KeyK":          common.KeyK,
		"KeyL":          common.KeyL,
		"KeyM":          common.KeyM,
		"KeyN":          common.KeyN,
		"KeyO":          common.KeyO,
		"KeyP":          common.KeyP,
		"KeyR":          common.KeyR,
		"KeyT":          common.KeyT,
		"KeyU":          common.KeyU,
		"KeyV":          common.KeyV,
		"KeyX":          common.KeyX,
		"KeyY":          common.KeyY,
		"KeyZ":          common.KeyZ,
		"KeySpace":      common.KeySpace,
		"KeyEsc":        common.KeyEsc,
		"KeyTab":        common.KeyTab,
		"KeyBackspace":  common.KeyBackspace,
		"KeyRight":      common.KeyRight,
		"KeyLeft":       common.KeyLeft,
		"KeyDown":       common.KeyDown,
		"KeyUp":         common.KeyUp,
		"KeyLeftShift":  common.KeyLeftShift,
		"KeyLeftCtrl":   common.KeyLeftCtrl,
		"KeyLeftAlt":    common.KeyLeftAlt,
		"KeyRightShift": common.KeyRightShift,
		"KeyRightCtrl":  common.KeyRightCtrl,
		"KeyRightAlt":   common.KeyRightAlt,
		"Digit0":        common.Key0,
		"Digit1":        common.Key1,
		"Digit2":        common.Key2,
		"Digit3":        common.Key3,
		"Digit4":        common.Key4,
		"Digit5":        common.Key5,
		"Digit6":        common.Key6,
		"Digit7":        common.Key7,
		"Digit8":        common.Key8,
		"Digit9":        common.Key9,
		"F1":            common.KeyF1,
		"F2":            common.KeyF2,
		"F3":            common.KeyF3,
		"F4":            common.KeyF4,
		"F5":            common.KeyF5,
		"F6":            common.KeyF6,
		"F7":            common.KeyF7,
		"F8":            common.KeyF8,
		"F9":            common.KeyF9,
		"F10":           common.KeyF10,
		"F11":           common.KeyF11,
		"F12":           common.KeyF12,
	}
}

// newPlatformWindow creates the browser canvas window with input callbacks and
// returns it as a platformBackend.
//
// Parameters:
//   - w: *window the window implementation receiving event callbacks
//
// Returns:
//   - platformBackend: the browser backend driving the canvas surface
//   - error: error if one occurs (always nil on the browser)
func newPlatformWindow(w *window) (platformBackend, error) {
	once.Do(initKeyCodes)

	document := js.Global().Get("document")
	canvas := document.Call("getElementById", "oxy")
	if !canvas.Truthy() {
		canvas = document.Call("createElement", "canvas")
		document.Get("body").Call("appendChild", canvas)
	}

	canvas.Set("width", w.width)
	canvas.Set("height", w.height)

	jw := &jsWindow{
		parent:  w,
		canvas:  canvas,
		running: true,
	}

	addListener := func(target js.Value, eventType string, fn js.Func) {
		target.Call("addEventListener", eventType, fn)
		jw.listeners = append(jw.listeners, listenerRecord{
			target:    target,
			eventType: eventType,
			fn:        fn,
		})
	}

	addListener(canvas, "keydown", js.FuncOf(func(_ js.Value, args []js.Value) any {
		event := args[0]
		code := event.Get("code").String()
		key := keyCodeFor(code)

		if key == common.KeyEsc {
			jw.running = false
		}
		if w.onKeyDown != nil {
			w.onKeyDown(key)
		}
		if keyNeedsPreventDefault(code) {
			event.Call("preventDefault")
		}
		return nil
	}))

	addListener(canvas, "keyup", js.FuncOf(func(_ js.Value, args []js.Value) any {
		event := args[0]
		code := event.Get("code").String()
		if w.onKeyUp != nil {
			w.onKeyUp(keyCodeFor(code))
		}
		if keyNeedsPreventDefault(code) {
			event.Call("preventDefault")
		}
		return nil
	}))

	addListener(canvas, "wheel", js.FuncOf(func(_ js.Value, args []js.Value) any {
		event := args[0]
		// Browser deltaY is negative when scrolling up; the engine's onScroll
		// contract expects positive to mean scroll up (GLFW semantics), so negate.
		if w.onScroll != nil {
			w.onScroll(float32(-event.Get("deltaY").Float()))
		}
		event.Call("preventDefault")
		return nil
	}))

	addListener(canvas, "mousedown", js.FuncOf(func(_ js.Value, args []js.Value) any {
		event := args[0]
		if event.Get("button").Int() == 1 {
			x, y := canvasRelativeCoords(canvas, event)
			if w.onMiddleMouseDown != nil {
				w.onMiddleMouseDown(x, y)
			}
			// Block the browser's middle-click autoscroll behavior.
			event.Call("preventDefault")
		}
		return nil
	}))

	addListener(canvas, "mouseup", js.FuncOf(func(_ js.Value, args []js.Value) any {
		event := args[0]
		if event.Get("button").Int() == 1 {
			x, y := canvasRelativeCoords(canvas, event)
			if w.onMiddleMouseUp != nil {
				w.onMiddleMouseUp(x, y)
			}
			event.Call("preventDefault")
		}
		return nil
	}))

	addListener(canvas, "mousemove", js.FuncOf(func(_ js.Value, args []js.Value) any {
		event := args[0]
		if w.onMouseMove != nil {
			x, y := canvasRelativeCoords(canvas, event)
			w.onMouseMove(x, y)
		}
		return nil
	}))

	addListener(js.Global(), "resize", js.FuncOf(func(_ js.Value, args []js.Value) any {
		width := js.Global().Get("innerWidth").Int()
		height := js.Global().Get("innerHeight").Int()

		canvas.Set("width", width)
		canvas.Set("height", height)

		w.width = width
		w.height = height
		if w.onResize != nil {
			w.onResize(width, height)
		}
		return nil
	}))

	return jw, nil
}

// isRunning reports whether the browser window is still running.
//
// Returns:
//   - bool: true while the window has not been closed
func (jw *jsWindow) isRunning() bool {
	return jw.running
}

// processMessages returns false because the browser drives its own event loop.
// There are no queued window messages to poll on the web platform; returning
// false makes ProcessMessages return immediately instead of spinning. The
// requestAnimationFrame-driven loop is a separate blocker and out of scope here.
//
// Returns:
//   - bool: always false on the browser
func (jw *jsWindow) processMessages() bool {
	return false
}

// surfaceDescriptor returns the WebGPU surface descriptor for the browser canvas.
// See https://pkg.go.dev/github.com/Carmen-Shannon/webgpu/wgpu#SurfaceDescriptor
//
// Returns:
//   - *wgpu.SurfaceDescriptor: descriptor carrying the HTML canvas element
func (jw *jsWindow) surfaceDescriptor() *wgpu.SurfaceDescriptor {
	return &wgpu.SurfaceDescriptor{
		Canvas: jw.canvas,
	}
}

// close tears down the browser window by detaching every registered event
// listener and releasing all retained js.Func values.
//
// Returns:
//   - error: error if one occurs (always nil on the browser)
func (jw *jsWindow) close() error {
	jw.running = false

	for _, rec := range jw.listeners {
		rec.target.Call("removeEventListener", rec.eventType, rec.fn)
	}
	for _, rec := range jw.listeners {
		rec.fn.Release()
	}
	jw.listeners = nil

	return nil
}

// canvasRelativeCoords converts an event's client-space position into
// canvas-relative integer coordinates for the engine callbacks.
//
// Parameters:
//   - canvas: js.Value the canvas element to compute coordinates against
//   - event: js.Value the mouse event carrying clientX and clientY
//
// Returns:
//   - int32: x coordinate relative to the canvas left edge
//   - int32: y coordinate relative to the canvas top edge
func canvasRelativeCoords(canvas js.Value, event js.Value) (int32, int32) {
	rect := canvas.Call("getBoundingClientRect")
	x := event.Get("clientX").Float() - rect.Get("left").Float()
	y := event.Get("clientY").Float() - rect.Get("top").Float()
	return int32(x), int32(y)
}

// keyCodeFor looks up the engine virtual key code for a DOM KeyboardEvent.code
// string, returning 0 for codes the engine does not consume.
//
// Parameters:
//   - code: string the KeyboardEvent.code value to map
//
// Returns:
//   - uint32: the GLFW-compatible virtual key code, or 0 if unmapped
func keyCodeFor(code string) uint32 {
	if key, ok := browserKeyCodes[code]; ok {
		return key
	}
	return 0
}

// keyNeedsPreventDefault reports whether a key event should suppress the
// browser's default action (page scroll, tab focus, fullscreen, back navigation).
//
// Parameters:
//   - code: string the KeyboardEvent.code value to check
//
// Returns:
//   - bool: true when the default browser action must be prevented
func keyNeedsPreventDefault(code string) bool {
	switch code {
	case "KeySpace", "KeyTab", "KeyEsc",
		"KeyRight", "KeyLeft", "KeyDown", "KeyUp",
		"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12":
		return true
	}
	return false
}
