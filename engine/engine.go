package engine

import (
	"log"
	"sort"
	"sync"
	"time"

	"github.com/Carmen-Shannon/oxy-go/engine/profiler"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

// engine implements the Engine interface.
// Coordinates engine, render, and window threads.
type engine struct {
	tickRateChannel chan time.Duration // Channel for dynamic tick rate updates

	running bool
	wg      sync.WaitGroup

	quitChannel chan struct{}
	quitOnce    sync.Once // Ensures quitChannel is only closed once

	window window.Window

	profiler         *profiler.Profiler
	profilingEnabled bool

	engineTickRate time.Duration
	tickCallback   func(deltaTime float32)
	renderCallback func(deltaTime float32)

	scenes map[int]scene.Scene

	renderFrameLimit time.Duration // minimum frame duration; 0 = uncapped
}

// Engine is the main entry point for the engine.
// It orchestrates the engine loop, render loop, and window management.
type Engine interface {
	// Init initializes the window with the provided options.
	//
	// Parameters:
	//   - options: functional options for window configuration
	//
	// Returns:
	//   - error: error if initialization fails
	// Init(options ...window.WindowBuilderOption) error

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

// NewEngine creates a new Engine instance with the provided options.
// Initializes message channels and profiler with sensible defaults.
// Options are applied directly to the engine struct via the option-builder pattern.
//
// Parameters:
//   - options: functional options for engine configuration (profiling, tick rate, etc.)
//
// Returns:
//   - Engine: the newly created engine
func NewEngine(options ...EngineBuilderOption) Engine {
	e := &engine{
		tickRateChannel:  make(chan time.Duration, 1),
		quitChannel:      make(chan struct{}),
		scenes:           make(map[int]scene.Scene),
		running:          false,
		wg:               sync.WaitGroup{},
		profiler:         profiler.NewProfiler(),
		profilingEnabled: false,
		engineTickRate:   time.Second / 60,
	}

	for _, opt := range options {
		opt(e)
	}

	if e.window != nil {
		e.window.SetResizeCallback(func(width, height int) {
			for _, s := range e.scenes {
				if r := s.Renderer(); r != nil {
					r.Resize(width, height)
				}
				if c := s.Camera(); c != nil {
					c.SetAspect(float32(width) / float32(height))
				}
			}
		})
	}

	return e
}

func (e *engine) Window() window.Window {
	return e.window
}

func (e *engine) Run() {
	e.handle()
	e.window.ProcessMessages()
}

// Quit signals all engine goroutines to stop and shuts down the engine.
// Safe to call multiple times; subsequent calls are no-ops due to sync.Once.
func (e *engine) Quit() {
	e.signalQuit()
}

// signalQuit closes the quit channel to signal all goroutines to exit.
// Uses sync.Once to ensure the channel is only closed once.
func (e *engine) signalQuit() {
	e.quitOnce.Do(func() {
		e.running = false
		close(e.quitChannel)
	})
}

// handle launches the engine, render, and quit goroutines.
// Each goroutine is tracked by the engine's WaitGroup.
func (e *engine) handle() {
	e.wg.Add(3)
	go e.handleEngine()
	go e.handleRender()
	go e.handleQuit()
}

// handleEngine runs the fixed-rate engine tick loop in its own goroutine.
// Fires the tick callback at the configured tick rate and listens for dynamic rate changes
// via tickRateChannel. Exits when the quit channel is closed.
func (e *engine) handleEngine() {
	defer e.wg.Done()

	ticker := time.NewTicker(e.engineTickRate)
	defer ticker.Stop()

	lastTick := time.Now()

	for {
		select {
		case <-e.quitChannel:
			return
		case <-ticker.C:
			now := time.Now()
			dt := float32(now.Sub(lastTick).Seconds())
			lastTick = now

			if e.tickCallback != nil {
				e.tickCallback(dt)
			}
		case newRate := <-e.tickRateChannel:
			ticker.Reset(newRate)
			e.engineTickRate = newRate
		}
	}
}

// handleRender runs the uncapped (or frame-limited) render loop in its own goroutine.
// Iterates active scenes in ascending z-index order, executing the full frame lifecycle:
// compute dispatch, shadow pass, light culling, and draw calls.
// Recovers from panics to avoid crashing the process and signals quit on recovery.
func (e *engine) handleRender() {
	defer e.wg.Done()
	// Recover from panics inside the render goroutine to avoid crashing the whole process.
	defer func() {
		if r := recover(); r != nil {
			log.Printf("render goroutine recovered from panic: %v", r)
			e.signalQuit()
		}
	}()

	lastRender := time.Now()

	for {
		select {
		case <-e.quitChannel:
			return
		default:
			now := time.Now()
			dt := float32(now.Sub(lastRender).Seconds())
			lastRender = now

			// Draw all active scenes in ascending z-index order.
			// The engine owns the frame lifecycle: BeginFrame once, Render each scene, EndFrame + Present once.
			// All scenes sharing the same renderer are rendered within a single render pass, enabling layered compositing.
			keys := make([]int, 0, len(e.scenes))
			for k := range e.scenes {
				keys = append(keys, k)
			}
			sort.Ints(keys)

			// Collect active scenes and find the renderer for the frame
			var activeScenes []scene.Scene
			for _, k := range keys {
				s := e.scenes[k]
				if s.Active() {
					activeScenes = append(activeScenes, s)
				}
			}

			if len(activeScenes) > 0 {
				// Use the first active scene's renderer to manage the frame
				frameRenderer := activeScenes[0].Renderer()
				if frameRenderer != nil {
					// Phase 1 — Compute: batch all compute dispatches into a single GPU submission
					if err := frameRenderer.BeginComputeFrame(); err == nil {
						for _, s := range activeScenes {
							s.PrepareCompute(dt)
						}
						frameRenderer.EndComputeFrame()
					}

					// Phase 1b — Shadows: render depth-only shadow passes for directional lights.
					for _, s := range activeScenes {
						s.PrepareShadows()
					}

					// Phase 1c — Light culling: dispatch the Forward+ tile culling compute shader.
					for _, s := range activeScenes {
						s.PrepareLightCulling()
					}

					// Phase 2 — Render: batch all draw calls into a single render pass
					if err := frameRenderer.BeginFrame(); err == nil {
						for _, s := range activeScenes {
							_ = s.DrawCalls()
						}
						frameRenderer.EndFrame()
						frameRenderer.Present()
					}
				}
			}

			if e.renderCallback != nil {
				e.renderCallback(dt)
			}

			if e.profilingEnabled && e.profiler != nil {
				e.profiler.Tick()
			}

			// Frame rate limiting
			if e.renderFrameLimit > 0 {
				elapsed := time.Since(lastRender)
				if remaining := e.renderFrameLimit - elapsed; remaining > 0 {
					time.Sleep(remaining)
				}
			}
		}
	}
}

// handleQuit blocks until the quit channel is closed, then decrements the WaitGroup.
func (e *engine) handleQuit() {
	defer e.wg.Done()
	<-e.quitChannel
}

// EnableProfiler enables performance profiling output to the log.
func (e *engine) EnableProfiler() {
	e.profilingEnabled = true
}

// DisableProfiler disables performance profiling output.
func (e *engine) DisableProfiler() {
	e.profilingEnabled = false
}

// SetTickRate sets the engine tick rate in frames per second.
// If the engine is running, the change takes effect immediately.
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

// SetTickCallback registers the function called each engine tick.
func (e *engine) SetTickCallback(callback func(deltaTime float32)) {
	e.tickCallback = callback
}

// SetRenderCallback registers the function called each render frame.
func (e *engine) SetRenderCallback(callback func(deltaTime float32)) {
	e.renderCallback = callback
}

// SetRenderFrameLimit sets an optional render frame rate cap.
// Pass 0 to uncap the render loop.
func (e *engine) SetRenderFrameLimit(fps float64) {
	if fps <= 0 {
		e.renderFrameLimit = 0
		return
	}
	e.renderFrameLimit = time.Second / time.Duration(fps)
}

func (e *engine) AddScene(key int, s scene.Scene) {
	e.scenes[key] = s
}

func (e *engine) RemoveScene(key int) {
	delete(e.scenes, key)
}

func (e *engine) Scene(key int) scene.Scene {
	return e.scenes[key]
}

func (e *engine) Scenes() map[int]scene.Scene {
	cp := make(map[int]scene.Scene, len(e.scenes))
	for k, v := range e.scenes {
		cp[k] = v
	}
	return cp
}
