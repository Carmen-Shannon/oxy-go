package engine

import (
	"log"
	"runtime/debug"
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
	tickRateChannel chan time.Duration
	resizeEvents    chan [2]int

	running bool
	wg      sync.WaitGroup

	quitChannel chan struct{}
	quitOnce    sync.Once

	window window.Window

	profiler         profiler.Profiler
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

			select {
			case dims := <-e.resizeEvents:
				for _, s := range e.scenes {
					s.Resize(dims[0], dims[1])
				}
			default:
			}

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
					stopGPUSync := func() {}
					if e.profilingEnabled && e.profiler != nil {
						stopGPUSync = e.profiler.Section("GPU_Sync")
					}
					frameRenderer.SyncGPUTimestamps()
					stopGPUSync()

					currentSlot := frameRenderer.CurrentFrameSlot()
					for _, s := range activeScenes {
						s.SyncFrameSlot(currentSlot)
					}

					if err := frameRenderer.BeginComputeFrame(); err == nil {
						stopPrepareCompute := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopPrepareCompute = e.profiler.Section("PrepareCompute")
						}
						for _, s := range activeScenes {
							s.PrepareCompute(dt)
						}
						frameRenderer.EndComputeFrame()
						stopPrepareCompute()
					}

					if err := frameRenderer.BeginGeometryFrame(); err == nil {
						stopPrepareShadows := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopPrepareShadows = e.profiler.Section("PrepareShadows")
						}
						for _, s := range activeScenes {
							s.PrepareShadows()
						}
						stopPrepareShadows()

						stopPrepareLights := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopPrepareLights = e.profiler.Section("PrepareLights")
						}
						for _, s := range activeScenes {
							s.PrepareLights()
						}
						stopPrepareLights()

						stopPrepareGBuffer := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopPrepareGBuffer = e.profiler.Section("PrepareGBuffer")
						}
						for _, s := range activeScenes {
							s.PrepareGBuffer()
						}
						frameRenderer.EndGeometryFrame()
						stopPrepareGBuffer()
					}

					if err := frameRenderer.BeginComputeFrame(); err == nil {
						stopPrepareLightCulling := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopPrepareLightCulling = e.profiler.Section("PrepareLightCulling")
						}
						for _, s := range activeScenes {
							s.PrepareLightCulling()
						}
						stopPrepareLightCulling()

						stopPrepareSSAO := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopPrepareSSAO = e.profiler.Section("PrepareSSAO")
						}
						for _, s := range activeScenes {
							s.PrepareSSAO()
						}
						stopPrepareSSAO()

						stopPrepareContactShadows := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopPrepareContactShadows = e.profiler.Section("PrepareContactShadows")
						}
						for _, s := range activeScenes {
							s.PrepareContactShadows()
						}
						stopPrepareContactShadows()

						frameRenderer.EndComputeFrame()
					}

					if err := activeScenes[0].BeginHDRFrame(); err == nil {
						stopDrawCalls := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopDrawCalls = e.profiler.Section("DrawCalls")
						}
						for _, s := range activeScenes {
							_ = s.DrawCalls()
						}
						frameRenderer.EndFrame()
						stopDrawCalls()

						if err := frameRenderer.BeginComputeFrame(); err == nil {
							stopPrepareSSR := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareSSR = e.profiler.Section("PrepareSSR")
							}
							for _, s := range activeScenes {
								s.PrepareSSR()
							}
							stopPrepareSSR()

							stopPrepareLuminance := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareLuminance = e.profiler.Section("PrepareLuminance")
							}
							for _, s := range activeScenes {
								s.PrepareLuminance(dt)
							}
							stopPrepareLuminance()

							stopPrepareBloom := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareBloom = e.profiler.Section("PrepareBloom")
							}
							for _, s := range activeScenes {
								s.PrepareBloom()
							}
							stopPrepareBloom()
							stopPrepareTAA := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareTAA = e.profiler.Section("PrepareTAA")
							}
							for _, s := range activeScenes {
								s.PrepareTAA()
							}
							stopPrepareTAA()
							frameRenderer.EndComputeFrame()
						}

						stopAcquireFrame := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopAcquireFrame = e.profiler.Section("AcquireFrame")
						}
						acquireErr := activeScenes[0].AcquireCompositionFrame()
						stopAcquireFrame()

						if acquireErr == nil {
							stopPrepareComposition := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareComposition = e.profiler.Section("PrepareComposition")
							}
							for _, s := range activeScenes {
								s.PrepareComposition()
							}
							stopPrepareComposition()
						}

						stopFlushFrame := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopFlushFrame = e.profiler.Section("FlushFrame")
						}
						frameRenderer.FlushFrame()
						stopFlushFrame()

						stopPresent := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopPresent = e.profiler.Section("Present")
						}
						frameRenderer.Present()
						stopPresent()
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
					timer := time.NewTimer(remaining)
					select {
					case <-e.quitChannel:
						timer.Stop()
						return
					case <-timer.C:
					}
					timer.Stop()
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

func (e *engine) resizeCallback(width, height int) {
	// Always deliver the latest resize dimensions.
	// Drain any stale pending event first so the channel always holds the most recent value.
	select {
	case <-e.resizeEvents:
	default:
	}
	e.resizeEvents <- [2]int{width, height}
}
