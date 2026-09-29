//go:build js

package engine

import (
	"syscall/js"
	"time"
)

// frameCallback retains the requestAnimationFrame callback so the browser
// never collects it while it is registered.
var frameCallback js.Func

// frameCallbackSet guards Release of frameCallback on shutdown, avoiding any
// need to compare zero-value js.Func instances.
var frameCallbackSet bool

// startRenderProducer registers the requestAnimationFrame frame-request loop.
// Each browser frame submits one non-blocking frame request to the render
// consumer and re-registers for the next frame. When a renderFrameLimit below
// the display refresh rate is configured, the next registration is deferred
// via setTimeout for the remaining frame budget instead of registering the
// next rAF immediately. If the window is no longer running, the engine is
// signaled to quit and no further frame is registered.
func (e *engine) startRenderProducer() {
	lastFrame := time.Now()

	frameCallback = js.FuncOf(func(_ js.Value, _ []js.Value) any {
		if e.window == nil || !e.window.IsRunning() {
			e.signalQuit()
			return nil
		}

		now := time.Now()

		select {
		case e.frameRequestCh <- struct{}{}:
		default:
		}

		elapsed := now.Sub(lastFrame)
		lastFrame = now

		if e.renderFrameLimit > 0 && elapsed < e.renderFrameLimit {
			remaining := e.renderFrameLimit - elapsed
			js.Global().Call("setTimeout", frameCallback, remaining.Milliseconds())
			return nil
		}

		js.Global().Call("requestAnimationFrame", frameCallback)
		return nil
	})
	frameCallbackSet = true
	js.Global().Call("requestAnimationFrame", frameCallback)
}

// runWindowLoop blocks until the engine quits, then closes the window to
// release browser event listeners and releases the retained frame callback.
// Blocking here is what keeps the Go wasm program alive after main starts it.
func (e *engine) runWindowLoop() {
	<-e.quitChannel

	if e.window != nil {
		_ = e.window.Close()
	}
	if frameCallbackSet {
		frameCallback.Release()
		frameCallbackSet = false
	}
}
