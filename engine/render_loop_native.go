//go:build !js

package engine

import "time"

// startRenderProducer launches the native frame-request producer goroutine.
// It paces frame requests to the render consumer, replicating the legacy
// renderFrameLimit semantics: when a frame limit is set, the producer measures
// the elapsed time from the start of each frame and sleeps the remainder
// before requesting the next frame. With no limit (0) it pumps frames as fast
// as the consumer can render them, matching the previous uncapped busy loop.
func (e *engine) startRenderProducer() {
	e.wg.Add(1)
	go func() {
		defer e.wg.Done()

		for {
			select {
			case <-e.quitChannel:
				return
			default:
			}

			frameStart := time.Now()

			select {
			case e.frameRequestCh <- struct{}{}:
			case <-e.quitChannel:
				return
			}

			if e.renderFrameLimit > 0 {
				elapsed := time.Since(frameStart)
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
	}()
}

// runWindowLoop blocks on the native platform message loop until the window
// closes, matching the pre-existing native Run behavior exactly.
func (e *engine) runWindowLoop() {
	e.window.ProcessMessages()
}
