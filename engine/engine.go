// Package engine provides the main [Engine] interface and orchestration layer for the
// game engine's core loops and window management.
//
// [Engine] coordinates the fixed-rate tick loop, the render loop, and scene lifecycle.
// Scenes are iterated in z-index order each frame through compute, geometry, lighting,
// and composition phases. Instances are created via [NewEngine] using functional options.
package engine

import (
	"time"

	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

// Engine is the main entry point for the engine.
// It orchestrates the engine loop, render loop, and window management.
type Engine interface {
	// Window returns the underlying window.
	//
	// Returns:
	//   - window.Window: the window instance
	Window() window.Window

	// EnableProfiler enables performance profiling output to the log.
	EnableProfiler()

	// DisableProfiler disables performance profiling output.
	DisableProfiler()

	// SetTickRate sets the engine tick rate in frames per second.
	// The tick callback will be called at this rate for game logic updates.
	//
	// Parameters:
	//   - fps: target frames per second (defaults to 60 if <= 0)
	SetTickRate(fps float64)

	// SetTickCallback registers the function called each engine tick.
	// Use this for game logic, physics, input processing, and animation updates.
	//
	// Parameters:
	//   - callback: function to call at the configured tick rate, receiving the delta time in seconds
	SetTickCallback(callback func(deltaTime float32))

	// SetRenderCallback registers the function called each render frame.
	// Use this for GPU buffer updates and scene rendering.
	//
	// Parameters:
	//   - callback: function to call each render frame, receiving the delta time in seconds
	SetRenderCallback(callback func(deltaTime float32))

	// SetRenderFrameLimit sets an optional render frame rate cap in frames per second.
	// Pass 0 to uncap the render loop (default).
	//
	// Parameters:
	//   - fps: maximum render frames per second (0 = uncapped)
	SetRenderFrameLimit(fps float64)

	// AddScene registers a scene at the given z-index key.
	// Scenes are rendered in ascending key order during the render loop.
	//
	// Parameters:
	//   - key: the z-index determining render order (lower renders first)
	//   - s: the Scene to register
	AddScene(key int, s scene.Scene)

	// RemoveScene removes the scene at the given z-index key.
	//
	// Parameters:
	//   - key: the z-index of the scene to remove
	RemoveScene(key int)

	// Scene retrieves the scene registered at the given z-index key.
	// Returns nil if no scene exists at that key.
	//
	// Parameters:
	//   - key: the z-index of the scene to retrieve
	//
	// Returns:
	//   - scene.Scene: the scene at the key, or nil if not found
	Scene(key int) scene.Scene

	// Scenes returns a copy of all registered scenes keyed by z-index.
	//
	// Returns:
	//   - map[int]scene.Scene: a copy of the scenes map
	Scenes() map[int]scene.Scene

	// Run starts the main engine loop (blocks until window closes).
	Run()

	// Quit signals all engine goroutines to stop and shuts down the engine.
	// This is an alternative to submitting a MessageShutdown message.
	// Safe to call multiple times; subsequent calls are no-ops.
	Quit()
}

var _ Engine = &engine{}

func (e *engine) Quit()                                              { e.signalQuit() }
func (e *engine) EnableProfiler()                                    { e.profilingEnabled = true }
func (e *engine) DisableProfiler()                                   { e.profilingEnabled = false }
func (e *engine) SetTickCallback(callback func(deltaTime float32))   { e.tickCallback = callback }
func (e *engine) SetRenderCallback(callback func(deltaTime float32)) { e.renderCallback = callback }
func (e *engine) AddScene(key int, s scene.Scene)                    { e.scenes[key] = s }
func (e *engine) RemoveScene(key int)                                { delete(e.scenes, key) }
func (e *engine) Scene(key int) scene.Scene                          { return e.scenes[key] }
func (e *engine) Scenes() map[int]scene.Scene                        { return e.scenes }
func (e *engine) Window() window.Window                              { return e.window }

func (e *engine) Run() {
	e.handle()
	e.window.ProcessMessages()
}

func (e *engine) SetTickRate(fps float64) {
	if fps <= 0 {
		fps = 60
	}
	newRate := time.Second / time.Duration(fps)

	if e.running {
		// Send to channel for immediate update in running engine loop
		// Non-blocking send - if channel is full, replace the pending value
		select {
		case e.tickRateChannel <- newRate:
		default:
			// Channel has a pending update, drain and send new value
			select {
			case <-e.tickRateChannel:
			default:
			}
			e.tickRateChannel <- newRate
		}
	} else {
		// Engine not running, just update the field
		e.engineTickRate = newRate
	}
}

func (e *engine) SetRenderFrameLimit(fps float64) {
	if fps <= 0 {
		e.renderFrameLimit = 0
		return
	}
	e.renderFrameLimit = time.Second / time.Duration(fps)
}
