package engine_test

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Carmen-Shannon/oxy-go/engine"
	cameramocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/camera"
	renderermocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/renderer"
	scenemocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/scene"
	windowmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/window"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

type engineTest struct {
	suite.Suite
}

func TestEngine(t *testing.T) {
	suite.Run(t, new(engineTest))
}

func (suite *engineTest) TestNewEngine() {
	suite.Run("default construction returns non-nil engine", func() {
		eng := engine.NewEngine()
		suite.NotNil(eng)
	})

	suite.Run("window is nil when none provided", func() {
		eng := engine.NewEngine()
		suite.Nil(eng.Window())
	})

	suite.Run("with window option stores the window", func() {
		w := newBasicMockWindow()
		eng := engine.NewEngine(engine.WithWindow(w))
		suite.Equal(w, eng.Window())
	})

	suite.Run("with profiling option enables profiler without panic", func() {
		suite.NotPanics(func() {
			_ = engine.NewEngine(engine.WithProfiling(true))
		})
	})

	suite.Run("with tick rate option does not panic", func() {
		suite.NotPanics(func() {
			_ = engine.NewEngine(engine.WithTickRate(120))
		})
	})

	suite.Run("with tick rate zero defaults without panic", func() {
		suite.NotPanics(func() {
			_ = engine.NewEngine(engine.WithTickRate(0))
		})
	})

	suite.Run("with tick rate negative defaults without panic", func() {
		suite.NotPanics(func() {
			_ = engine.NewEngine(engine.WithTickRate(-10))
		})
	})

	suite.Run("with scene option registers scene at key", func() {
		s := &scenemocks.MockScene{}
		eng := engine.NewEngine(engine.WithScene(0, s))
		suite.Equal(s, eng.Scene(0))
	})

	suite.Run("with render frame limit option does not panic", func() {
		suite.NotPanics(func() {
			_ = engine.NewEngine(engine.WithRenderFrameLimit(60))
		})
	})

	suite.Run("with render frame limit zero defaults to uncapped", func() {
		suite.NotPanics(func() {
			_ = engine.NewEngine(engine.WithRenderFrameLimit(0))
		})
	})

	suite.Run("with render frame limit negative defaults to uncapped", func() {
		suite.NotPanics(func() {
			_ = engine.NewEngine(engine.WithRenderFrameLimit(-5))
		})
	})

	suite.Run("multiple options combined apply without panic", func() {
		w := newBasicMockWindow()
		s := &scenemocks.MockScene{}
		suite.NotPanics(func() {
			_ = engine.NewEngine(
				engine.WithWindow(w),
				engine.WithProfiling(true),
				engine.WithTickRate(30),
				engine.WithScene(1, s),
				engine.WithRenderFrameLimit(144),
			)
		})
	})

	suite.Run("resize callback invokes renderer resize and camera set aspect", func() {
		var capturedCallback func(int, int)
		w := &windowmocks.MockWindow{}
		w.EXPECT().SetResizeCallback(mock.Anything).Run(func(cb func(int, int)) {
			capturedCallback = cb
		}).Once()

		r := &renderermocks.MockRenderer{}
		r.EXPECT().Resize(800, 600).Once()

		c := &cameramocks.MockCamera{}
		c.EXPECT().SetAspect(float32(800) / float32(600)).Once()

		s := &scenemocks.MockScene{}
		s.EXPECT().Renderer().Return(r).Maybe()
		s.EXPECT().Camera().Return(c).Maybe()

		_ = engine.NewEngine(
			engine.WithWindow(w),
			engine.WithScene(0, s),
		)

		suite.NotNil(capturedCallback)
		capturedCallback(800, 600)
		w.AssertExpectations(suite.T())
		r.AssertExpectations(suite.T())
		c.AssertExpectations(suite.T())
	})

	suite.Run("resize callback skips nil renderer and camera", func() {
		var capturedCallback func(int, int)
		w := &windowmocks.MockWindow{}
		w.EXPECT().SetResizeCallback(mock.Anything).Run(func(cb func(int, int)) {
			capturedCallback = cb
		}).Once()

		s := &scenemocks.MockScene{}
		s.EXPECT().Renderer().Return(nil).Maybe()
		s.EXPECT().Camera().Return(nil).Maybe()

		_ = engine.NewEngine(
			engine.WithWindow(w),
			engine.WithScene(0, s),
		)

		suite.NotNil(capturedCallback)
		suite.NotPanics(func() {
			capturedCallback(1024, 768)
		})
	})
}

func (suite *engineTest) TestSceneManagement() {
	suite.Run("add scene and retrieve by key", func() {
		eng := engine.NewEngine()
		s := &scenemocks.MockScene{}
		eng.AddScene(5, s)
		suite.Equal(s, eng.Scene(5))
	})

	suite.Run("retrieve non-existent key returns nil", func() {
		eng := engine.NewEngine()
		suite.Nil(eng.Scene(99))
	})

	suite.Run("remove scene deletes from map", func() {
		eng := engine.NewEngine()
		s := &scenemocks.MockScene{}
		eng.AddScene(1, s)
		eng.RemoveScene(1)
		suite.Nil(eng.Scene(1))
	})

	suite.Run("scenes returns a copy not a reference", func() {
		eng := engine.NewEngine()
		s := &scenemocks.MockScene{}
		eng.AddScene(0, s)
		cp := eng.Scenes()
		delete(cp, 0)
		suite.NotNil(eng.Scene(0))
	})

	suite.Run("overwrite scene at existing key", func() {
		eng := engine.NewEngine()
		s1 := &scenemocks.MockScene{}
		s2 := &scenemocks.MockScene{}
		eng.AddScene(0, s1)
		eng.AddScene(0, s2)
		suite.Equal(s2, eng.Scene(0))
	})

	suite.Run("scenes returns correct count", func() {
		eng := engine.NewEngine()
		eng.AddScene(0, &scenemocks.MockScene{})
		eng.AddScene(1, &scenemocks.MockScene{})
		eng.AddScene(2, &scenemocks.MockScene{})
		suite.Len(eng.Scenes(), 3)
	})

	suite.Run("remove non-existent key does not panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.RemoveScene(42)
		})
	})
}

func (suite *engineTest) TestWindow() {
	suite.Run("engine with no window returns nil", func() {
		eng := engine.NewEngine()
		suite.Nil(eng.Window())
	})

	suite.Run("engine with mock window returns that window", func() {
		w := newBasicMockWindow()
		eng := engine.NewEngine(engine.WithWindow(w))
		suite.Equal(w, eng.Window())
	})
}

func (suite *engineTest) TestSetTickRate() {
	suite.Run("positive value does not panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.SetTickRate(120)
		})
	})

	suite.Run("zero defaults to 60fps without panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.SetTickRate(0)
		})
	})

	suite.Run("negative defaults to 60fps without panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.SetTickRate(-10)
		})
	})
}

func (suite *engineTest) TestSetTickCallback() {
	suite.Run("set callback does not panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.SetTickCallback(func(dt float32) {})
		})
	})

	suite.Run("set nil callback does not panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.SetTickCallback(nil)
		})
	})
}

func (suite *engineTest) TestSetRenderCallback() {
	suite.Run("set callback does not panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.SetRenderCallback(func(dt float32) {})
		})
	})

	suite.Run("set nil callback does not panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.SetRenderCallback(nil)
		})
	})
}

func (suite *engineTest) TestSetRenderFrameLimit() {
	suite.Run("zero means uncapped without panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.SetRenderFrameLimit(0)
		})
	})

	suite.Run("negative means uncapped without panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.SetRenderFrameLimit(-5)
		})
	})

	suite.Run("positive value sets limit without panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.SetRenderFrameLimit(144)
		})
	})
}

func (suite *engineTest) TestEnableDisableProfiler() {
	suite.Run("enable profiler does not panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.EnableProfiler()
		})
	})

	suite.Run("disable profiler does not panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.DisableProfiler()
		})
	})

	suite.Run("toggle profiler on and off does not panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.EnableProfiler()
			eng.DisableProfiler()
			eng.EnableProfiler()
		})
	})

	suite.Run("with profiling option at construction", func() {
		suite.NotPanics(func() {
			eng := engine.NewEngine(engine.WithProfiling(true))
			eng.DisableProfiler()
		})
	})
}

func (suite *engineTest) TestQuit() {
	suite.Run("quit before run does not panic", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.Quit()
		})
	})

	suite.Run("double quit is safe", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.Quit()
			eng.Quit()
		})
	})

	suite.Run("triple quit is safe", func() {
		eng := engine.NewEngine()
		suite.NotPanics(func() {
			eng.Quit()
			eng.Quit()
			eng.Quit()
		})
	})
}

func (suite *engineTest) TestRunIntegration() {
	suite.Run("run with mock window exits cleanly", func() {
		w := newBasicMockWindow()
		w.EXPECT().ProcessMessages().RunAndReturn(func() {
			time.Sleep(3 * time.Millisecond)
		}).Once()
		eng := engine.NewEngine(engine.WithWindow(w))

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		select {
		case <-done:
			eng.Quit()
		case <-time.After(2 * time.Second):
			eng.Quit()
			suite.Fail("engine.Run did not return within timeout")
		}
	})

	suite.Run("tick callback fires during run", func() {
		w, stop := newBlockingMockWindow()
		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithTickRate(1000),
		)

		var tickCount atomic.Int32
		eng.SetTickCallback(func(dt float32) {
			tickCount.Add(1)
		})

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(100 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		suite.Greater(tickCount.Load(), int32(0))
	})

	suite.Run("render callback fires during run", func() {
		w, stop := newBlockingMockWindow()
		eng := engine.NewEngine(engine.WithWindow(w))

		var renderCount atomic.Int32
		eng.SetRenderCallback(func(dt float32) {
			renderCount.Add(1)
		})

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(100 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		suite.Greater(renderCount.Load(), int32(0))
	})

	suite.Run("active scene active method is queried during run", func() {
		w, stop := newBlockingMockWindow()

		var activeCount atomic.Int32
		s := &scenemocks.MockScene{}
		s.EXPECT().Active().RunAndReturn(func() bool {
			activeCount.Add(1)
			return true
		}).Maybe()
		s.EXPECT().Renderer().Return(nil).Maybe()

		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithScene(0, s),
		)

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(100 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		suite.Greater(activeCount.Load(), int32(0))
	})

	suite.Run("quit stops the engine run loop", func() {
		w, stop := newBlockingMockWindow()
		eng := engine.NewEngine(engine.WithWindow(w))

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(50 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not exit after Quit")
		}
	})
}

func (suite *engineTest) TestRenderPipeline() {
	suite.Run("active scene full render pipeline is executed", func() {
		w, stop := newBlockingMockWindow()

		r := &renderermocks.MockRenderer{}
		var computeCount, frameCount atomic.Int32

		r.EXPECT().BeginComputeFrame().RunAndReturn(func() error {
			computeCount.Add(1)
			return nil
		}).Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()
		r.EXPECT().BeginFrame().RunAndReturn(func() error {
			frameCount.Add(1)
			return nil
		}).Maybe()
		r.EXPECT().EndFrame().Return().Maybe()
		r.EXPECT().Present().Return().Maybe()

		s := &scenemocks.MockScene{}
		s.EXPECT().Active().Return(true).Maybe()
		s.EXPECT().Renderer().Return(r).Maybe()
		s.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Return().Maybe()
		s.EXPECT().PrepareShadows().Return().Maybe()
		s.EXPECT().PrepareLightCulling().Return().Maybe()
		s.EXPECT().DrawCalls().Return(nil).Maybe()

		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithScene(0, s),
		)

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(150 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		suite.Greater(computeCount.Load(), int32(0))
		suite.Greater(frameCount.Load(), int32(0))
	})

	suite.Run("inactive scene is skipped during render", func() {
		w, stop := newBlockingMockWindow()

		r := &renderermocks.MockRenderer{}

		s := &scenemocks.MockScene{}
		s.EXPECT().Active().Return(false).Maybe()
		s.EXPECT().Renderer().Return(r).Maybe()

		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithScene(0, s),
		)

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(100 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		// Since the scene is inactive, none of the render pipeline methods should be called.
		r.AssertNotCalled(suite.T(), "BeginComputeFrame")
		r.AssertNotCalled(suite.T(), "BeginFrame")
	})

	suite.Run("multiple active scenes share the same frame", func() {
		w, stop := newBlockingMockWindow()

		r := &renderermocks.MockRenderer{}
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()
		r.EXPECT().BeginFrame().Return(nil).Maybe()
		r.EXPECT().EndFrame().Return().Maybe()
		r.EXPECT().Present().Return().Maybe()

		var s1PrepCount, s2PrepCount atomic.Int32

		s1 := &scenemocks.MockScene{}
		s1.EXPECT().Active().Return(true).Maybe()
		s1.EXPECT().Renderer().Return(r).Maybe()
		s1.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Run(func(_ float32) {
			s1PrepCount.Add(1)
		}).Return().Maybe()
		s1.EXPECT().PrepareShadows().Return().Maybe()
		s1.EXPECT().PrepareLightCulling().Return().Maybe()
		s1.EXPECT().DrawCalls().Return(nil).Maybe()

		s2 := &scenemocks.MockScene{}
		s2.EXPECT().Active().Return(true).Maybe()
		s2.EXPECT().Renderer().Return(r).Maybe()
		s2.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Run(func(_ float32) {
			s2PrepCount.Add(1)
		}).Return().Maybe()
		s2.EXPECT().PrepareShadows().Return().Maybe()
		s2.EXPECT().PrepareLightCulling().Return().Maybe()
		s2.EXPECT().DrawCalls().Return(nil).Maybe()

		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithScene(0, s1),
			engine.WithScene(1, s2),
		)

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(150 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		suite.Greater(s1PrepCount.Load(), int32(0))
		suite.Greater(s2PrepCount.Load(), int32(0))
	})

	suite.Run("render frame limit throttles render loop", func() {
		w, stop := newBlockingMockWindow()

		var renderCount atomic.Int32
		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithRenderFrameLimit(30),
		)
		eng.SetRenderCallback(func(dt float32) {
			renderCount.Add(1)
		})

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(200 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		// At 30fps over 200ms we expect roughly 6 frames. Allow generous bounds,
		// but it should be significantly less than an uncapped loop.
		suite.Greater(renderCount.Load(), int32(0))
		suite.Less(renderCount.Load(), int32(30))
	})

	suite.Run("profiler ticks during render loop when enabled", func() {
		w, stop := newBlockingMockWindow()

		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithProfiling(true),
		)

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(100 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}
	})

	suite.Run("BeginComputeFrame error skips compute phase", func() {
		w, stop := newBlockingMockWindow()

		r := &renderermocks.MockRenderer{}
		r.EXPECT().BeginComputeFrame().Return(fmt.Errorf("compute error")).Maybe()
		r.EXPECT().BeginFrame().Return(nil).Maybe()
		r.EXPECT().EndFrame().Return().Maybe()
		r.EXPECT().Present().Return().Maybe()

		s := &scenemocks.MockScene{}
		s.EXPECT().Active().Return(true).Maybe()
		s.EXPECT().Renderer().Return(r).Maybe()
		s.EXPECT().PrepareShadows().Return().Maybe()
		s.EXPECT().PrepareLightCulling().Return().Maybe()
		s.EXPECT().DrawCalls().Return(nil).Maybe()

		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithScene(0, s),
		)

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(100 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		// PrepareCompute and EndComputeFrame should NOT have been called
		s.AssertNotCalled(suite.T(), "PrepareCompute", mock.Anything)
		r.AssertNotCalled(suite.T(), "EndComputeFrame")
	})

	suite.Run("BeginFrame error skips draw phase", func() {
		w, stop := newBlockingMockWindow()

		r := &renderermocks.MockRenderer{}
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()
		r.EXPECT().BeginFrame().Return(fmt.Errorf("frame error")).Maybe()

		s := &scenemocks.MockScene{}
		s.EXPECT().Active().Return(true).Maybe()
		s.EXPECT().Renderer().Return(r).Maybe()
		s.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Return().Maybe()
		s.EXPECT().PrepareShadows().Return().Maybe()
		s.EXPECT().PrepareLightCulling().Return().Maybe()

		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithScene(0, s),
		)

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(100 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		// DrawCalls, EndFrame, and Present should NOT have been called
		s.AssertNotCalled(suite.T(), "DrawCalls")
		r.AssertNotCalled(suite.T(), "EndFrame")
		r.AssertNotCalled(suite.T(), "Present")
	})
}

func (suite *engineTest) TestDynamicTickRate() {
	suite.Run("set tick rate while engine is running", func() {
		w, stop := newBlockingMockWindow()
		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithTickRate(60),
		)

		var tickCount atomic.Int32
		eng.SetTickCallback(func(dt float32) {
			tickCount.Add(1)
		})

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		// Wait for the engine loop to start
		time.Sleep(50 * time.Millisecond)

		// Change tick rate while running — exercises the tickRateChannel path
		eng.SetTickRate(500)

		time.Sleep(100 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		suite.Greater(tickCount.Load(), int32(0))
	})

	suite.Run("set tick rate to zero while running defaults to 60", func() {
		w, stop := newBlockingMockWindow()
		eng := engine.NewEngine(
			engine.WithWindow(w),
			engine.WithTickRate(120),
		)

		var tickCount atomic.Int32
		eng.SetTickCallback(func(dt float32) {
			tickCount.Add(1)
		})

		done := make(chan struct{})
		go func() {
			eng.Run()
			close(done)
		}()

		time.Sleep(50 * time.Millisecond)
		eng.SetTickRate(0)
		time.Sleep(100 * time.Millisecond)
		stop()
		eng.Quit()

		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("engine.Run did not return within timeout")
		}

		suite.Greater(tickCount.Load(), int32(0))
	})
}

// newBasicMockWindow creates a MockWindow with the minimum expectations needed
// to pass it to engine.NewEngine without triggering unexpected call panics.
func newBasicMockWindow() *windowmocks.MockWindow {
	w := &windowmocks.MockWindow{}
	w.EXPECT().SetResizeCallback(mock.Anything).Maybe()
	return w
}

// newBlockingMockWindow creates a MockWindow whose ProcessMessages blocks
// until the returned stop function is called. The mock is pre-configured
// with the expectations needed for engine.NewEngine and engine.Run.
//
// Returns:
//   - *windowmocks.MockWindow: the configured mock window
//   - func(): stop function that unblocks ProcessMessages
func newBlockingMockWindow() (*windowmocks.MockWindow, func()) {
	w := newBasicMockWindow()
	running := &atomic.Bool{}
	running.Store(true)
	w.EXPECT().ProcessMessages().RunAndReturn(func() {
		for running.Load() {
			time.Sleep(1 * time.Millisecond)
		}
	}).Once()
	return w, func() { running.Store(false) }
}
