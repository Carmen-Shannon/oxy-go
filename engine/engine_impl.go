package engine

import (
	"log"
	"runtime/debug"
	"sort"
	"sync"
	"time"

	"github.com/Carmen-Shannon/oxy-go/engine/command"
	"github.com/Carmen-Shannon/oxy-go/engine/context"
	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
	"github.com/Carmen-Shannon/oxy-go/engine/profiler"
	"github.com/Carmen-Shannon/oxy-go/engine/queue"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

// engine implements the Engine interface.
// Coordinates engine, render, and window threads.
type engine struct {
	tickRateChannel chan time.Duration
	resizeEvents    chan [2]int
	cmdCtx          context.Context
	cmdQueue        queue.Queue[command.Command]

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

// startSceneLifecycle advances a scene lifecycle through startup states when needed.
//
// Parameters:
//   - s: the scene to start
func (e *engine) startSceneLifecycle(s scene.Scene) {
	if s == nil || s.Lifecycle() == nil {
		return
	}

	lc := s.Lifecycle()
	sceneName := s.Name()

	switch lc.State() {
	case lifecycle.LifecycleStateRegistered:
		if err := lc.SetState(lifecycle.LifecycleStateStarting); err != nil {
			log.Printf("engine: failed to transition scene %q to Starting: %v", sceneName, err)
			return
		}
		if lc.State() == lifecycle.LifecycleStateStarting {
			if err := lc.SetState(lifecycle.LifecycleStateRunning); err != nil {
				log.Printf("engine: failed to transition scene %q to Running: %v", sceneName, err)
			}
		}

	case lifecycle.LifecycleStateStarting:
		if err := lc.SetState(lifecycle.LifecycleStateRunning); err != nil {
			log.Printf("engine: failed to transition scene %q to Running: %v", sceneName, err)
		}
	}
}

// shutdownSceneLifecycle advances a scene lifecycle through shutdown states.
//
// Parameters:
//   - s: the scene to shut down
//
// Returns:
//   - bool: true if the scene can be deleted from the engine scene map
func (e *engine) shutdownSceneLifecycle(s scene.Scene) bool {
	if s == nil {
		return false
	}
	if s.Lifecycle() == nil {
		log.Printf("engine: scene %q has nil lifecycle and cannot be removed safely", s.Name())
		return false
	}

	lc := s.Lifecycle()
	state := lc.State()
	if state == lifecycle.LifecycleStateRemoved {
		return true
	}

	var path []lifecycle.LifecycleState
	switch state {
	case lifecycle.LifecycleStateRegistered:
		path = []lifecycle.LifecycleState{lifecycle.LifecycleStateStopped, lifecycle.LifecycleStateRemoved}
	case lifecycle.LifecycleStateRunning:
		path = []lifecycle.LifecycleState{lifecycle.LifecycleStateDraining, lifecycle.LifecycleStateStopped, lifecycle.LifecycleStateRemoved}
	case lifecycle.LifecycleStatePaused:
		path = []lifecycle.LifecycleState{lifecycle.LifecycleStateStopped, lifecycle.LifecycleStateRemoved}
	case lifecycle.LifecycleStateDraining:
		path = []lifecycle.LifecycleState{lifecycle.LifecycleStateStopped, lifecycle.LifecycleStateRemoved}
	case lifecycle.LifecycleStateErrored:
		path = []lifecycle.LifecycleState{lifecycle.LifecycleStateDraining, lifecycle.LifecycleStateStopped, lifecycle.LifecycleStateRemoved}
	case lifecycle.LifecycleStateStopped:
		path = []lifecycle.LifecycleState{lifecycle.LifecycleStateRemoved}
	default:
		log.Printf("engine: scene %q in state %v has no supported shutdown path", s.Name(), state)
		return false
	}

	for _, target := range path {
		current := lc.State()
		if current == target {
			continue
		}
		if err := lc.SetState(target); err != nil {
			log.Printf("engine: failed to transition scene %q from %v to %v during removal: %v", s.Name(), current, target, err)
			return false
		}
	}

	if lc.State() != lifecycle.LifecycleStateRemoved {
		log.Printf("engine: scene %q did not reach Removed state during removal", s.Name())
		return false
	}

	return true
}

// handle launches the engine, render, and quit goroutines.
// Each goroutine is tracked by the engine's WaitGroup.
func (e *engine) handle() {
	e.running = true
	e.cmdQueue.Start(e.cmdCtx, e.quitChannel)
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
// Iterates scenes in ascending z-index order using lifecycle state filters:
// Running scenes execute the full frame pipeline, and Paused scenes execute compute-only phases.
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

			var computeScenes []scene.Scene
			var frameScenes []scene.Scene
			for _, k := range keys {
				s := e.scenes[k]
				if s == nil || s.Lifecycle() == nil {
					continue
				}

				switch s.Lifecycle().State() {
				case lifecycle.LifecycleStateRunning:
					computeScenes = append(computeScenes, s)
					frameScenes = append(frameScenes, s)
				case lifecycle.LifecycleStatePaused:
					computeScenes = append(computeScenes, s)
				}
			}

			var driverScene scene.Scene
			if len(frameScenes) > 0 {
				driverScene = frameScenes[0]
			} else if len(computeScenes) > 0 {
				driverScene = computeScenes[0]
			}

			if driverScene != nil {
				frameRenderer := driverScene.Renderer()
				if frameRenderer != nil {
					computePrepared := false

					if len(computeScenes) > 0 {
						stopGPUSync := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopGPUSync = e.profiler.Section("GPU_Sync")
						}
						frameRenderer.SyncGPUTimestamps()
						stopGPUSync()

						currentSlot := frameRenderer.CurrentFrameSlot()
						for _, s := range computeScenes {
							s.SyncFrameSlot(currentSlot)
						}

						if err := frameRenderer.BeginComputeFrame(); err == nil {
							stopPrepareCompute := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareCompute = e.profiler.Section("PrepareCompute")
							}
							for _, s := range computeScenes {
								s.PrepareCompute(dt)
							}
							frameRenderer.EndComputeFrame()
							stopPrepareCompute()
							computePrepared = true
						}
					}

					if len(frameScenes) > 0 {
						if err := frameRenderer.BeginGeometryFrame(); err == nil {
							stopPrepareShadows := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareShadows = e.profiler.Section("PrepareShadows")
							}
							for _, s := range frameScenes {
								s.PrepareShadows()
							}
							stopPrepareShadows()

							stopPrepareLights := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareLights = e.profiler.Section("PrepareLights")
							}
							for _, s := range frameScenes {
								s.PrepareLights()
							}
							stopPrepareLights()

							stopPrepareGBuffer := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareGBuffer = e.profiler.Section("PrepareGBuffer")
							}
							for _, s := range frameScenes {
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
							for _, s := range frameScenes {
								s.PrepareLightCulling()
							}
							stopPrepareLightCulling()

							stopPrepareSSAO := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareSSAO = e.profiler.Section("PrepareSSAO")
							}
							for _, s := range frameScenes {
								s.PrepareSSAO()
							}
							stopPrepareSSAO()

							stopPrepareContactShadows := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopPrepareContactShadows = e.profiler.Section("PrepareContactShadows")
							}
							for _, s := range frameScenes {
								s.PrepareContactShadows()
							}
							stopPrepareContactShadows()

							frameRenderer.EndComputeFrame()
						}

						if err := frameScenes[0].BeginHDRFrame(); err == nil {
							stopDrawCalls := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopDrawCalls = e.profiler.Section("DrawCalls")
							}
							for _, s := range frameScenes {
								_ = s.DrawCalls()
							}
							frameRenderer.EndFrame()
							stopDrawCalls()

							if err := frameRenderer.BeginComputeFrame(); err == nil {
								stopPrepareSSR := func() {}
								if e.profilingEnabled && e.profiler != nil {
									stopPrepareSSR = e.profiler.Section("PrepareSSR")
								}
								for _, s := range frameScenes {
									s.PrepareSSR()
								}
								stopPrepareSSR()

								stopPrepareLuminance := func() {}
								if e.profilingEnabled && e.profiler != nil {
									stopPrepareLuminance = e.profiler.Section("PrepareLuminance")
								}
								for _, s := range frameScenes {
									s.PrepareLuminance(dt)
								}
								stopPrepareLuminance()

								stopPrepareBloom := func() {}
								if e.profilingEnabled && e.profiler != nil {
									stopPrepareBloom = e.profiler.Section("PrepareBloom")
								}
								for _, s := range frameScenes {
									s.PrepareBloom()
								}
								stopPrepareBloom()
								stopPrepareTAA := func() {}
								if e.profilingEnabled && e.profiler != nil {
									stopPrepareTAA = e.profiler.Section("PrepareTAA")
								}
								for _, s := range frameScenes {
									s.PrepareTAA()
								}
								stopPrepareTAA()
								frameRenderer.EndComputeFrame()
							}

							stopAcquireFrame := func() {}
							if e.profilingEnabled && e.profiler != nil {
								stopAcquireFrame = e.profiler.Section("AcquireFrame")
							}
							acquireErr := frameScenes[0].AcquireCompositionFrame()
							stopAcquireFrame()

							if acquireErr == nil {
								stopPrepareComposition := func() {}
								if e.profilingEnabled && e.profiler != nil {
									stopPrepareComposition = e.profiler.Section("PrepareComposition")
								}
								for _, s := range frameScenes {
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
							for _, s := range frameScenes {
								_ = s.DrawCalls()
							}
							frameRenderer.EndFrame()
							frameRenderer.FlushFrame()
							frameRenderer.Present()
						}
					} else if computePrepared {
						stopFlushFrame := func() {}
						if e.profilingEnabled && e.profiler != nil {
							stopFlushFrame = e.profiler.Section("FlushFrame")
						}
						frameRenderer.FlushFrame()
						stopFlushFrame()
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
