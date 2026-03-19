package engine

import (
	"sync"
	"time"

	"github.com/Carmen-Shannon/oxy-go/engine/profiler"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

// EngineBuilderOption is a functional option for configuring an Engine.
// Use the With* functions to create options that are applied directly to the engine instance.
type EngineBuilderOption func(*engine)

// WithProfiling enables or disables performance profiling output.
//
// Parameters:
//   - enabled: if true, enables performance profiling
//
// Returns:
//   - EngineBuilderOption: option function to apply
func WithProfiling(enabled bool) EngineBuilderOption {
	return func(e *engine) { e.profilingEnabled = enabled }
}

// WithTickRate sets the engine tick rate in frames per second.
// The tick callback will be called at this rate for game logic updates.
// Values <= 0 will be treated as the default (60Hz).
//
// Parameters:
//   - fps: target ticks per second (default 60)
//
// Returns:
//   - EngineBuilderOption: option function to apply
func WithTickRate(fps float64) EngineBuilderOption {
	return func(e *engine) {
		if fps <= 0 {
			fps = 60.0
		}
		e.engineTickRate = time.Second / time.Duration(fps)
	}
}

// WithWindow sets a custom configured window for the engine to use rather than allowing the engine
// to create and manage one internally.
//
// Parameters:
//   - w: a pre-configured Window instance
//
// Returns:
//   - EngineBuilderOption: option function to apply
func WithWindow(w window.Window) EngineBuilderOption {
	return func(e *engine) { e.window = w }
}

// WithScene registers a scene at the given z-index key during engine construction.
// Scenes are rendered in ascending key order during the render loop.
//
// Parameters:
//   - key: the z-index determining render order (lower renders first)
//   - s: the Scene to register
//
// Returns:
//   - EngineBuilderOption: option function to apply
func WithScene(key int, s scene.Scene) EngineBuilderOption {
	return func(e *engine) { e.scenes[key] = s }
}

// WithRenderFrameLimit sets an optional render frame rate cap in frames per second.
// Pass 0 to uncap the render loop (default).
//
// Parameters:
//   - fps: maximum render frames per second (0 = uncapped)
//
// Returns:
//   - EngineBuilderOption: option function to apply
func WithRenderFrameLimit(fps float64) EngineBuilderOption {
	return func(e *engine) {
		if fps <= 0 {
			e.renderFrameLimit = 0
			return
		}
		e.renderFrameLimit = time.Second / time.Duration(fps)
	}
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
				s.Resize(width, height)
			}
		})
	}
	e.Delegate = e
	return e
}
