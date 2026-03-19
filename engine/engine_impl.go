package engine

import (
	"log"
	"runtime/debug"
	"sort"
	"sync"
	"time"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/profiler"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

// engine implements the Engine interface.
// Coordinates engine, render, and window threads.
type engine struct {
	common.DelegateImpl[Engine]

	tickRateChannel chan time.Duration

	running bool
	wg      sync.WaitGroup

	quitChannel chan struct{}
	quitOnce    sync.Once

	window window.Window

	profiler         *profiler.Profiler
	profilingEnabled bool

	engineTickRate               time.Duration
	tickCallback, renderCallback func(deltaTime float32)

	scenes map[int]scene.Scene

	renderFrameLimit time.Duration
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
	e.running = true
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
	defer func() {
		if r := recover(); r != nil {
			log.Printf("render goroutine recovered from panic: %v\n%s", r, debug.Stack())
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

			keys := make([]int, 0, len(e.scenes))
			for k := range e.scenes {
				keys = append(keys, k)
			}
			sort.Ints(keys)

			var activeScenes []scene.Scene
			for _, k := range keys {
				s := e.scenes[k]
				if s.Active() {
					activeScenes = append(activeScenes, s)
				}
			}

			if len(activeScenes) > 0 {
				frameRenderer := activeScenes[0].Renderer()
				if frameRenderer != nil {
					if err := frameRenderer.BeginComputeFrame(); err == nil {
						for _, s := range activeScenes {
							s.PrepareCompute(dt)
						}
						frameRenderer.EndComputeFrame()
					}

					if err := frameRenderer.BeginGeometryFrame(); err == nil {
						for _, s := range activeScenes {
							s.PrepareShadows()
						}
						for _, s := range activeScenes {
							s.PrepareGBuffer()
						}
						frameRenderer.EndGeometryFrame()
					}

					if err := frameRenderer.BeginComputeFrame(); err == nil {
						for _, s := range activeScenes {
							s.PrepareLightCulling()
						}
						for _, s := range activeScenes {
							s.PrepareSSAO()
						}
						for _, s := range activeScenes {
							s.PrepareContactShadows()
						}
						frameRenderer.EndComputeFrame()
					}

					if err := activeScenes[0].BeginHDRFrame(); err == nil {
						for _, s := range activeScenes {
							_ = s.DrawCalls()
						}
						frameRenderer.EndFrame()

						if err := frameRenderer.BeginComputeFrame(); err == nil {
							for _, s := range activeScenes {
								s.PrepareSSR()
							}
							frameRenderer.EndComputeFrame()
						}

						for _, s := range activeScenes {
							s.PrepareComposition()
						}

						frameRenderer.FlushFrame()
						frameRenderer.Present()
					} else if err := frameRenderer.BeginFrame(); err == nil {
						for _, s := range activeScenes {
							_ = s.DrawCalls()
						}
						frameRenderer.EndFrame()
						frameRenderer.FlushFrame()
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
