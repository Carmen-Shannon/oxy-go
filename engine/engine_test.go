package engine

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/Carmen-Shannon/oxy-go/engine/command"
	oxycontext "github.com/Carmen-Shannon/oxy-go/engine/context"
	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
	"github.com/Carmen-Shannon/oxy-go/engine/profiler"
	renderer_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/mocks"
	scene_mocks "github.com/Carmen-Shannon/oxy-go/engine/scene/mocks"
	window_mocks "github.com/Carmen-Shannon/oxy-go/engine/window/mocks"
	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

func TestRunEngineTests(t *testing.T) {
	suite.Run(t, new(engineTest))
}

type engineTest struct {
	suite.Suite
	windowMock     *window_mocks.MockWindow
	sceneMock      *scene_mocks.MockScene
	sceneLifecycle lifecycle.Lifecycle
	engine         Engine
}

type lifecycleStub struct {
	state      lifecycle.LifecycleState
	setStateFn func(target lifecycle.LifecycleState) error
	onToFn     func(to lifecycle.LifecycleState, hook lifecycle.Hook)
	onFromFn   func(from lifecycle.LifecycleState, hook lifecycle.Hook)
}

func (stub *lifecycleStub) State() lifecycle.LifecycleState {
	return stub.state
}

func (stub *lifecycleStub) SetState(state lifecycle.LifecycleState) error {
	if stub.setStateFn != nil {
		return stub.setStateFn(state)
	}
	stub.state = state
	return nil
}

func (stub *lifecycleStub) OnTransitionTo(to lifecycle.LifecycleState, hook lifecycle.Hook) func() {
	if stub.onToFn != nil {
		stub.onToFn(to, hook)
	}
	return func() {}
}

func (stub *lifecycleStub) OnTransitionFrom(from lifecycle.LifecycleState, hook lifecycle.Hook) func() {
	if stub.onFromFn != nil {
		stub.onFromFn(from, hook)
	}
	return func() {}
}

func newSceneMockWithLifecycle(t *testing.T, name string, state lifecycle.LifecycleState) (*scene_mocks.MockScene, lifecycle.Lifecycle) {
	sceneMock := scene_mocks.NewMockScene(t)
	lc := lifecycle.NewLifecycle(lifecycle.WithState(state))
	sceneMock.EXPECT().Lifecycle().Return(lc).Maybe()
	sceneMock.EXPECT().Name().Return(name).Maybe()
	return sceneMock, lc
}

func (suite *engineTest) configurePrimaryScene(name string, state lifecycle.LifecycleState) (*scene_mocks.MockScene, lifecycle.Lifecycle) {
	sceneMock, lc := newSceneMockWithLifecycle(suite.T(), name, state)
	eImpl := suite.engine.(*engine)
	for key := range eImpl.scenes {
		delete(eImpl.scenes, key)
	}
	eImpl.scenes[0] = sceneMock
	suite.sceneMock = sceneMock
	suite.sceneLifecycle = lc
	return sceneMock, lc
}

func (suite *engineTest) setSceneAtKey(key int, name string, state lifecycle.LifecycleState) (*scene_mocks.MockScene, lifecycle.Lifecycle) {
	sceneMock, lc := newSceneMockWithLifecycle(suite.T(), name, state)
	eImpl := suite.engine.(*engine)
	eImpl.scenes[key] = sceneMock
	return sceneMock, lc
}

func (suite *engineTest) SetupSubTest() {
	suite.windowMock = window_mocks.NewMockWindow(suite.T())
	suite.windowMock.EXPECT().SetResizeCallback(mock.Anything).Return().Maybe()
	suite.sceneMock, suite.sceneLifecycle = newSceneMockWithLifecycle(suite.T(), "setup-scene", lifecycle.LifecycleStateStopped)
	suite.engine = NewEngine(WithWindow(suite.windowMock), WithScene(0, suite.sceneMock))
}

func (suite *engineTest) TearDownSubTest() {
	eImpl := suite.engine.(*engine)
	eImpl.signalQuit()
	eImpl.wg.Wait()
}

func (suite *engineTest) TestNewEngine() {
	suite.Run("should apply any provided options", func() {
		dummyEngine := NewEngine(
			WithProfiling(true),
			WithTickRate(0),
			WithWindow(suite.windowMock),
			WithScene(0, suite.sceneMock),
			WithRenderFrameLimit(30), // allowing the conditional check to run
			WithRenderFrameLimit(0),
		)
		suite.NotNil(dummyEngine)
	})
}

func (suite *engineTest) TestWindow() {
	suite.Run("should return the underlying window", func() {
		window := suite.engine.Window()
		suite.Equal(suite.windowMock, window)
	})
}

func (suite *engineTest) TestEnableProfiler() {
	suite.Run("should enable profiling output", func() {
		suite.engine.EnableProfiler()
		eImpl := suite.engine.(*engine)
		suite.NotNil(eImpl.profiler)
		suite.True(eImpl.profilingEnabled)
	})
}

func (suite *engineTest) TestDisableProfiler() {
	suite.Run("should disable profiling output", func() {
		suite.engine.DisableProfiler()
		eImpl := suite.engine.(*engine)
		suite.False(eImpl.profilingEnabled)
	})
}

func (suite *engineTest) TestSetTickRate() {
	suite.Run("should update the tick rate when the engine is running", func() {
		eImpl := suite.engine.(*engine)
		eImpl.running = true

		suite.engine.SetTickRate(120)
		select {
		case received := <-eImpl.tickRateChannel:
			suite.Equal(time.Second/time.Duration(120), received)
		default:
			suite.Fail("expected a value on tickRateChannel but got none")
		}
	})

	suite.Run("should drain and replace a pending tick rate update when channel is full", func() {
		eImpl := suite.engine.(*engine)
		eImpl.running = true

		eImpl.tickRateChannel <- time.Second / time.Duration(60)

		suite.engine.SetTickRate(120)

		select {
		case received := <-eImpl.tickRateChannel:
			suite.Equal(time.Second/time.Duration(120), received)
		default:
			suite.Fail("expected updated tick rate on tickRateChannel but got none")
		}
	})

	suite.Run("should update the tick rate when the engine is not running", func() {
		eImpl := suite.engine.(*engine)
		eImpl.running = false

		suite.engine.SetTickRate(120)
		suite.Equal(time.Second/time.Duration(120), eImpl.engineTickRate)
	})

	suite.Run("should default to a 60hz tick rate when the provided value is 0 or less", func() {
		eImpl := suite.engine.(*engine)
		eImpl.running = false

		suite.engine.SetTickRate(0)
		suite.Equal(time.Second/time.Duration(60), eImpl.engineTickRate)
	})
}

func (suite *engineTest) TestSetTickCallback() {
	suite.Run("should set the tick callback function", func() {
		called := false
		dummyCallback := func(deltaTime float32) { called = true }
		suite.engine.SetTickCallback(dummyCallback)
		eImpl := suite.engine.(*engine)
		suite.NotNil(eImpl.tickCallback)
		eImpl.tickCallback(0)
		suite.True(called)
	})
}

func (suite *engineTest) TestSetRenderFrameLimit() {
	suite.Run("should update the render frame limit", func() {
		suite.engine.SetRenderFrameLimit(30)
		eImpl := suite.engine.(*engine)
		suite.Equal(time.Second/time.Duration(30), eImpl.renderFrameLimit)
	})

	suite.Run("should uncap the render frame limit when the provided value is 0 or less", func() {
		suite.engine.SetRenderFrameLimit(0)
		eImpl := suite.engine.(*engine)
		suite.Equal(time.Duration(0), eImpl.renderFrameLimit)
	})
}

func (suite *engineTest) TestAddScene() {
	suite.Run("should add a scene to the engine", func() {
		newSceneMock, _ := newSceneMockWithLifecycle(suite.T(), "added-scene", lifecycle.LifecycleStateStopped)
		suite.engine.AddScene(1, newSceneMock)
		scene := suite.engine.Scene(1)
		suite.Equal(newSceneMock, scene)
	})

	suite.Run("should transition registered scenes to running", func() {
		newSceneMock, lc := newSceneMockWithLifecycle(suite.T(), "registered-scene", lifecycle.LifecycleStateRegistered)
		suite.engine.AddScene(2, newSceneMock)
		suite.Equal(lifecycle.LifecycleStateRunning, lc.State())
	})

	suite.Run("should transition starting scenes to running", func() {
		newSceneMock, lc := newSceneMockWithLifecycle(suite.T(), "starting-scene", lifecycle.LifecycleStateStarting)
		suite.engine.AddScene(3, newSceneMock)
		suite.Equal(lifecycle.LifecycleStateRunning, lc.State())
	})

	suite.Run("should overwrite an existing scene with the same key", func() {
		newSceneMock, _ := newSceneMockWithLifecycle(suite.T(), "replacement-scene", lifecycle.LifecycleStateStopped)
		suite.engine.AddScene(0, newSceneMock)
		scene := suite.engine.Scene(0)
		suite.Equal(newSceneMock, scene)
	})
}

func (suite *engineTest) TestRemoveScene() {
	suite.Run("should remove a stopped scene from the engine", func() {
		suite.engine.RemoveScene(0)
		scene := suite.engine.Scene(0)
		suite.Nil(scene)
		suite.Equal(lifecycle.LifecycleStateRemoved, suite.sceneLifecycle.State())
	})

	suite.Run("should transition registered scene through stopped to removed", func() {
		_, lc := suite.setSceneAtKey(1, "registered-remove", lifecycle.LifecycleStateRegistered)
		suite.engine.RemoveScene(1)
		suite.Nil(suite.engine.Scene(1))
		suite.Equal(lifecycle.LifecycleStateRemoved, lc.State())
	})

	suite.Run("should transition running scene through draining and stopped to removed", func() {
		_, lc := suite.setSceneAtKey(2, "running-remove", lifecycle.LifecycleStateRunning)
		suite.engine.RemoveScene(2)
		suite.Nil(suite.engine.Scene(2))
		suite.Equal(lifecycle.LifecycleStateRemoved, lc.State())
	})

	suite.Run("should transition paused scene through stopped to removed", func() {
		_, lc := suite.setSceneAtKey(3, "paused-remove", lifecycle.LifecycleStatePaused)
		suite.engine.RemoveScene(3)
		suite.Nil(suite.engine.Scene(3))
		suite.Equal(lifecycle.LifecycleStateRemoved, lc.State())
	})

	suite.Run("should transition draining scene through stopped to removed", func() {
		_, lc := suite.setSceneAtKey(4, "draining-remove", lifecycle.LifecycleStateDraining)
		suite.engine.RemoveScene(4)
		suite.Nil(suite.engine.Scene(4))
		suite.Equal(lifecycle.LifecycleStateRemoved, lc.State())
	})

	suite.Run("should transition errored scene through draining and stopped to removed", func() {
		_, lc := suite.setSceneAtKey(5, "errored-remove", lifecycle.LifecycleStateErrored)
		suite.engine.RemoveScene(5)
		suite.Nil(suite.engine.Scene(5))
		suite.Equal(lifecycle.LifecycleStateRemoved, lc.State())
	})
}

func (suite *engineTest) TestScene() {
	suite.Run("should return the scene for a given z-index key", func() {
		scene := suite.engine.Scene(0)
		suite.Equal(suite.sceneMock, scene)
	})

	suite.Run("should return nil for a non-existent key", func() {
		scene := suite.engine.Scene(999)
		suite.Nil(scene)
	})
}

func (suite *engineTest) TestScenes() {
	suite.Run("should return all registered scenes", func() {
		scenes := suite.engine.Scenes()
		suite.Equal(1, len(scenes))
		suite.Equal(suite.sceneMock, scenes[0])
	})
}

func (suite *engineTest) TestRun() {
	suite.Run("should start the engine loop and process window messages", func() {
		suite.windowMock.EXPECT().ProcessMessages().Return().Once()
		suite.engine.Run()
		engineImpl := suite.engine.(*engine)
		suite.True(engineImpl.running)
	})
}

func (suite *engineTest) TestQuit() {
	suite.Run("should signal all engine goroutines to stop", func() {
		engineImpl := suite.engine.(*engine)
		engineImpl.running = true
		suite.engine.Quit()
		suite.False(engineImpl.running)
	})
}

func (suite *engineTest) TestHandleEngine() {
	suite.Run("should exit cleanly when quit channel is closed", func() {
		eImpl := suite.engine.(*engine)
		eImpl.wg.Add(1)
		go eImpl.handleEngine()

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleEngine did not exit within timeout")
		}
	})

	suite.Run("should invoke the tick callback on each tick", func() {
		eImpl := suite.engine.(*engine)
		eImpl.engineTickRate = 10 * time.Millisecond

		called := make(chan float32, 1)
		eImpl.tickCallback = func(dt float32) {
			select {
			case called <- dt:
			default:
			}
		}

		eImpl.wg.Add(1)
		go eImpl.handleEngine()

		select {
		case dt := <-called:
			suite.GreaterOrEqual(dt, float32(0))
		case <-time.After(2 * time.Second):
			suite.Fail("tick callback was not called within timeout")
		}

		eImpl.signalQuit()
		eImpl.wg.Wait()
	})

	suite.Run("should update engineTickRate when a new rate is received", func() {
		eImpl := suite.engine.(*engine)
		eImpl.engineTickRate = 10 * time.Millisecond
		newRate := 50 * time.Millisecond

		eImpl.tickRateChannel <- newRate // pre-fill before goroutine starts

		eImpl.wg.Add(1)
		go eImpl.handleEngine()

		time.Sleep(20 * time.Millisecond) // give goroutine time to drain the channel
		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
			suite.Equal(newRate, eImpl.engineTickRate)
		case <-time.After(2 * time.Second):
			suite.Fail("handleEngine did not exit within timeout")
		}
	})
}

func (suite *engineTest) TestHandleRender() {
	suite.Run("should exit cleanly when quit channel is closed", func() {
		eImpl := suite.engine.(*engine)
		suite.configurePrimaryScene("stopped-scene", lifecycle.LifecycleStateStopped)

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		time.Sleep(10 * time.Millisecond)
		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}
	})

	suite.Run("should execute the full HDR render path when renderer and scenes are available", func() {
		eImpl := suite.engine.(*engine)
		rendererMock := renderer_mocks.NewMockRenderer(suite.T())
		sceneMock, _ := suite.configurePrimaryScene("running-scene", lifecycle.LifecycleStateRunning)

		rendered := make(chan struct{}, 1)
		var once sync.Once
		rendererMock.EXPECT().Present().RunAndReturn(func() {
			once.Do(func() { rendered <- struct{}{} })
		}).Maybe()

		sceneMock.EXPECT().Renderer().Return(rendererMock).Maybe()
		rendererMock.EXPECT().BeginComputeFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Return().Maybe()
		rendererMock.EXPECT().EndComputeFrame().Return().Maybe()
		rendererMock.EXPECT().BeginGeometryFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareShadows().Return().Maybe()
		sceneMock.EXPECT().PrepareLights().Return().Maybe()
		sceneMock.EXPECT().PrepareGBuffer().Return().Maybe()
		rendererMock.EXPECT().EndGeometryFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareLightCulling().Return().Maybe()
		sceneMock.EXPECT().PrepareSSAO().Return().Maybe()
		sceneMock.EXPECT().PrepareContactShadows().Return().Maybe()
		sceneMock.EXPECT().BeginHDRFrame().Return(nil).Maybe()
		sceneMock.EXPECT().DrawCalls().Return(nil).Maybe()
		rendererMock.EXPECT().EndFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareSSR().Return().Maybe()
		sceneMock.EXPECT().PrepareLuminance(mock.AnythingOfType("float32")).Return().Maybe()
		sceneMock.EXPECT().PrepareBloom().Return().Maybe()
		sceneMock.EXPECT().PrepareTAA().Return().Maybe()
		rendererMock.EXPECT().SyncGPUTimestamps().Return().Maybe()
		rendererMock.EXPECT().CurrentFrameSlot().Return(0).Maybe()
		sceneMock.EXPECT().SyncFrameSlot(mock.Anything).Maybe()
		sceneMock.EXPECT().AcquireCompositionFrame().Return(nil).Maybe()
		sceneMock.EXPECT().PrepareComposition().Return().Maybe()
		rendererMock.EXPECT().FlushFrame().Return(wgpu.SubmissionIndex(0)).Maybe()

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-rendered:
		case <-time.After(2 * time.Second):
			suite.Fail("HDR render path was not executed within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}
	})

	suite.Run("should execute compute-only path and flush for paused scenes", func() {
		eImpl := suite.engine.(*engine)
		rendererMock := renderer_mocks.NewMockRenderer(suite.T())
		sceneMock, _ := suite.configurePrimaryScene("paused-scene", lifecycle.LifecycleStatePaused)

		flushed := make(chan struct{}, 1)
		var once sync.Once
		rendererMock.EXPECT().FlushFrame().RunAndReturn(func() wgpu.SubmissionIndex {
			once.Do(func() { flushed <- struct{}{} })
			return wgpu.SubmissionIndex(0)
		}).Maybe()

		sceneMock.EXPECT().Renderer().Return(rendererMock).Maybe()
		rendererMock.EXPECT().SyncGPUTimestamps().Return().Maybe()
		rendererMock.EXPECT().CurrentFrameSlot().Return(0).Maybe()
		sceneMock.EXPECT().SyncFrameSlot(mock.Anything).Maybe()
		rendererMock.EXPECT().BeginComputeFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Return().Maybe()
		rendererMock.EXPECT().EndComputeFrame().Return().Maybe()

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-flushed:
		case <-time.After(2 * time.Second):
			suite.Fail("paused compute-only render path was not executed within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}

		sceneMock.AssertNotCalled(suite.T(), "PrepareShadows")
		sceneMock.AssertNotCalled(suite.T(), "PrepareLights")
		sceneMock.AssertNotCalled(suite.T(), "PrepareGBuffer")
		sceneMock.AssertNotCalled(suite.T(), "PrepareLightCulling")
		sceneMock.AssertNotCalled(suite.T(), "PrepareSSAO")
		sceneMock.AssertNotCalled(suite.T(), "PrepareContactShadows")
		sceneMock.AssertNotCalled(suite.T(), "BeginHDRFrame")
		sceneMock.AssertNotCalled(suite.T(), "DrawCalls")
		sceneMock.AssertNotCalled(suite.T(), "PrepareComposition")
		rendererMock.AssertNotCalled(suite.T(), "Present")
	})

	suite.Run("should skip the renderer pipeline when the running scene has no renderer", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, _ := suite.configurePrimaryScene("running-no-renderer-scene", lifecycle.LifecycleStateRunning)
		sceneMock.EXPECT().Renderer().Return(nil).Maybe()

		called := make(chan struct{}, 1)
		var once sync.Once
		eImpl.renderCallback = func(dt float32) {
			once.Do(func() { called <- struct{}{} })
		}

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-called:
		case <-time.After(2 * time.Second):
			suite.Fail("render callback was not called within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}
	})

	suite.Run("should invoke the render callback each frame", func() {
		eImpl := suite.engine.(*engine)
		suite.configurePrimaryScene("stopped-callback-scene", lifecycle.LifecycleStateStopped)

		called := make(chan struct{}, 1)
		var once sync.Once
		eImpl.renderCallback = func(dt float32) {
			once.Do(func() { called <- struct{}{} })
		}

		suite.NotNil(eImpl.renderCallback)

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-called:
		case <-time.After(2 * time.Second):
			suite.Fail("render callback was not called within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}
	})

	suite.Run("should recover from a panic in the render goroutine and signal quit", func() {
		eImpl := suite.engine.(*engine)
		rendererMock := renderer_mocks.NewMockRenderer(suite.T())
		sceneMock, _ := suite.configurePrimaryScene("running-panic-scene", lifecycle.LifecycleStateRunning)

		sceneMock.EXPECT().Renderer().Return(rendererMock).Maybe()
		rendererMock.EXPECT().BeginComputeFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Return().Maybe()
		rendererMock.EXPECT().EndComputeFrame().Return().Maybe()
		rendererMock.EXPECT().BeginGeometryFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareShadows().Return().Maybe()
		sceneMock.EXPECT().PrepareLights().Return().Maybe()
		sceneMock.EXPECT().PrepareGBuffer().Return().Maybe()
		rendererMock.EXPECT().EndGeometryFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareLightCulling().Return().Maybe()
		sceneMock.EXPECT().PrepareSSAO().Return().Maybe()
		sceneMock.EXPECT().PrepareContactShadows().Return().Maybe()
		rendererMock.EXPECT().SyncGPUTimestamps().Return().Maybe()
		rendererMock.EXPECT().CurrentFrameSlot().Return(0).Maybe()
		sceneMock.EXPECT().SyncFrameSlot(mock.Anything).Maybe()
		sceneMock.EXPECT().BeginHDRFrame().Return(nil).Maybe()
		sceneMock.EXPECT().DrawCalls().RunAndReturn(func() error {
			panic("test render panic")
		}).Maybe()

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}

		suite.False(eImpl.running)
	})

	suite.Run("should tick the profiler each frame when profiling is enabled", func() {
		eImpl := suite.engine.(*engine)
		suite.configurePrimaryScene("stopped-profiler-scene", lifecycle.LifecycleStateStopped)

		eImpl.profilingEnabled = true
		eImpl.profiler = profiler.NewProfiler()

		called := make(chan struct{}, 1)
		var once sync.Once
		eImpl.renderCallback = func(dt float32) {
			once.Do(func() { called <- struct{}{} })
		}

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-called:
		case <-time.After(2 * time.Second):
			suite.Fail("render callback was not called within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}
	})

	suite.Run("should sleep to enforce the render frame limit when set", func() {
		eImpl := suite.engine.(*engine)
		suite.configurePrimaryScene("stopped-frame-limit-scene", lifecycle.LifecycleStateStopped)
		eImpl.renderFrameLimit = 1 * time.Millisecond

		called := make(chan struct{}, 2)
		var count sync.Once
		second := make(chan struct{}, 1)
		eImpl.renderCallback = func(dt float32) {
			select {
			case called <- struct{}{}:
			default:
			}
			if len(called) >= 2 {
				count.Do(func() { second <- struct{}{} })
			}
		}

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-second:
		case <-time.After(2 * time.Second):
			suite.Fail("render frame limit path was not exercised within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}
	})

	suite.Run("should invoke profiler sections through the full HDR render path", func() {
		eImpl := suite.engine.(*engine)
		rendererMock := renderer_mocks.NewMockRenderer(suite.T())
		sceneMock, _ := suite.configurePrimaryScene("running-profiled-scene", lifecycle.LifecycleStateRunning)

		eImpl.profilingEnabled = true
		eImpl.profiler = profiler.NewProfiler()

		rendered := make(chan struct{}, 1)
		var once sync.Once
		rendererMock.EXPECT().Present().RunAndReturn(func() {
			once.Do(func() { rendered <- struct{}{} })
		}).Maybe()

		sceneMock.EXPECT().Renderer().Return(rendererMock).Maybe()
		rendererMock.EXPECT().SyncGPUTimestamps().Return().Maybe()
		rendererMock.EXPECT().CurrentFrameSlot().Return(0).Maybe()
		sceneMock.EXPECT().SyncFrameSlot(mock.Anything).Maybe()
		rendererMock.EXPECT().BeginComputeFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Return().Maybe()
		rendererMock.EXPECT().EndComputeFrame().Return().Maybe()
		rendererMock.EXPECT().BeginGeometryFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareShadows().Return().Maybe()
		sceneMock.EXPECT().PrepareLights().Return().Maybe()
		sceneMock.EXPECT().PrepareGBuffer().Return().Maybe()
		rendererMock.EXPECT().EndGeometryFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareLightCulling().Return().Maybe()
		sceneMock.EXPECT().PrepareSSAO().Return().Maybe()
		sceneMock.EXPECT().PrepareContactShadows().Return().Maybe()
		sceneMock.EXPECT().BeginHDRFrame().Return(nil).Maybe()
		sceneMock.EXPECT().DrawCalls().Return(nil).Maybe()
		rendererMock.EXPECT().EndFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareSSR().Return().Maybe()
		sceneMock.EXPECT().PrepareLuminance(mock.AnythingOfType("float32")).Return().Maybe()
		sceneMock.EXPECT().PrepareBloom().Return().Maybe()
		sceneMock.EXPECT().PrepareTAA().Return().Maybe()
		sceneMock.EXPECT().AcquireCompositionFrame().Return(nil).Maybe()
		sceneMock.EXPECT().PrepareComposition().Return().Maybe()
		rendererMock.EXPECT().FlushFrame().Return(wgpu.SubmissionIndex(0)).Maybe()

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-rendered:
		case <-time.After(2 * time.Second):
			suite.Fail("HDR render path with profiling was not executed within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}
	})

	suite.Run("should deliver resize event to all scenes within the render loop", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, _ := suite.configurePrimaryScene("stopped-resize-scene", lifecycle.LifecycleStateStopped)
		sceneMock.EXPECT().Resize(800, 600).Return().Once()

		eImpl.resizeEvents <- [2]int{800, 600}

		called := make(chan struct{}, 1)
		var once sync.Once
		eImpl.renderCallback = func(dt float32) {
			once.Do(func() { called <- struct{}{} })
		}

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-called:
		case <-time.After(2 * time.Second):
			suite.Fail("render callback was not called within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}
	})

	suite.Run("should skip PrepareComposition when AcquireCompositionFrame fails but still flush and present", func() {
		eImpl := suite.engine.(*engine)
		rendererMock := renderer_mocks.NewMockRenderer(suite.T())
		sceneMock, _ := suite.configurePrimaryScene("running-acquire-fail-scene", lifecycle.LifecycleStateRunning)

		rendered := make(chan struct{}, 1)
		var once sync.Once
		rendererMock.EXPECT().Present().RunAndReturn(func() {
			once.Do(func() { rendered <- struct{}{} })
		}).Maybe()

		sceneMock.EXPECT().Renderer().Return(rendererMock).Maybe()
		rendererMock.EXPECT().SyncGPUTimestamps().Return().Maybe()
		rendererMock.EXPECT().CurrentFrameSlot().Return(0).Maybe()
		sceneMock.EXPECT().SyncFrameSlot(mock.Anything).Maybe()
		rendererMock.EXPECT().BeginComputeFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Return().Maybe()
		rendererMock.EXPECT().EndComputeFrame().Return().Maybe()
		rendererMock.EXPECT().BeginGeometryFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareShadows().Return().Maybe()
		sceneMock.EXPECT().PrepareLights().Return().Maybe()
		sceneMock.EXPECT().PrepareGBuffer().Return().Maybe()
		rendererMock.EXPECT().EndGeometryFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareLightCulling().Return().Maybe()
		sceneMock.EXPECT().PrepareSSAO().Return().Maybe()
		sceneMock.EXPECT().PrepareContactShadows().Return().Maybe()
		sceneMock.EXPECT().BeginHDRFrame().Return(nil).Maybe()
		sceneMock.EXPECT().DrawCalls().Return(nil).Maybe()
		rendererMock.EXPECT().EndFrame().Return().Maybe()
		sceneMock.EXPECT().PrepareSSR().Return().Maybe()
		sceneMock.EXPECT().PrepareLuminance(mock.AnythingOfType("float32")).Return().Maybe()
		sceneMock.EXPECT().PrepareBloom().Return().Maybe()
		sceneMock.EXPECT().PrepareTAA().Return().Maybe()
		sceneMock.EXPECT().AcquireCompositionFrame().Return(fmt.Errorf("surface lost")).Maybe()
		rendererMock.EXPECT().FlushFrame().Return(wgpu.SubmissionIndex(0)).Maybe()

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-rendered:
		case <-time.After(2 * time.Second):
			suite.Fail("HDR render path was not executed within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}

		sceneMock.AssertNotCalled(suite.T(), "PrepareComposition")
	})

	suite.Run("should skip the rate-limit sleep when frame already exceeded the limit", func() {
		eImpl := suite.engine.(*engine)
		suite.configurePrimaryScene("stopped-over-limit-scene", lifecycle.LifecycleStateStopped)
		eImpl.renderFrameLimit = 1 * time.Nanosecond

		called := make(chan struct{}, 1)
		var once sync.Once
		eImpl.renderCallback = func(dt float32) {
			once.Do(func() { called <- struct{}{} })
		}

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-called:
		case <-time.After(2 * time.Second):
			suite.Fail("render callback was not called within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}
	})

	suite.Run("should exit from the rate-limit timer select when quit is signaled", func() {
		eImpl := suite.engine.(*engine)
		suite.configurePrimaryScene("stopped-rate-limit-scene", lifecycle.LifecycleStateStopped)
		eImpl.renderFrameLimit = 1 * time.Second

		firstFrame := make(chan struct{}, 1)
		var once sync.Once
		eImpl.renderCallback = func(dt float32) {
			once.Do(func() { firstFrame <- struct{}{} })
		}

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-firstFrame:
		case <-time.After(2 * time.Second):
			suite.Fail("first frame callback not received within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit from rate-limit sleep within timeout")
		}
	})
}

func (suite *engineTest) TestResizeCallback() {
	suite.Run("should send resize event to channel without calling Resize directly", func() {
		eImpl := suite.engine.(*engine)
		eImpl.resizeCallback(800, 600)
		suite.Len(eImpl.resizeEvents, 1)
	})

	suite.Run("should replace the stale event with the latest dimensions", func() {
		eImpl := suite.engine.(*engine)
		eImpl.resizeCallback(800, 600)
		eImpl.resizeCallback(1920, 1080)
		suite.Len(eImpl.resizeEvents, 1)
		dims := <-eImpl.resizeEvents
		suite.Equal([2]int{1920, 1080}, dims)
	})
}

func (suite *engineTest) TestStartSceneLifecycle() {
	suite.Run("nil scene is ignored", func() {
		eImpl := suite.engine.(*engine)
		suite.NotPanics(func() { eImpl.startSceneLifecycle(nil) })
	})

	suite.Run("scene with nil lifecycle is ignored", func() {
		eImpl := suite.engine.(*engine)
		sceneMock := scene_mocks.NewMockScene(suite.T())
		sceneMock.EXPECT().Lifecycle().Return(nil).Maybe()

		eImpl.startSceneLifecycle(sceneMock)

		sceneMock.AssertNotCalled(suite.T(), "Name")
	})

	suite.Run("non-startable state remains unchanged", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, lc := newSceneMockWithLifecycle(suite.T(), "paused-scene", lifecycle.LifecycleStatePaused)

		eImpl.startSceneLifecycle(sceneMock)

		suite.Equal(lifecycle.LifecycleStatePaused, lc.State())
	})

	suite.Run("registered startup failure stops before running", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, lc := newSceneMockWithLifecycle(suite.T(), "registered-fail-scene", lifecycle.LifecycleStateRegistered)
		lc.OnTransitionTo(lifecycle.LifecycleStateStarting, lifecycle.Hook(func() error {
			return fmt.Errorf("start failed")
		}))

		eImpl.startSceneLifecycle(sceneMock)

		suite.Equal(lifecycle.LifecycleStateStarting, lc.State())
	})

	suite.Run("starting to running hook failure still reaches running state", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, lc := newSceneMockWithLifecycle(suite.T(), "starting-fail-scene", lifecycle.LifecycleStateStarting)
		lc.OnTransitionTo(lifecycle.LifecycleStateRunning, lifecycle.Hook(func() error {
			return fmt.Errorf("running failed")
		}))

		eImpl.startSceneLifecycle(sceneMock)

		suite.Equal(lifecycle.LifecycleStateRunning, lc.State())
	})

	suite.Run("registered scene running transition failure after starting branch", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, lc := newSceneMockWithLifecycle(suite.T(), "registered-running-fail-scene", lifecycle.LifecycleStateRegistered)
		lc.OnTransitionTo(lifecycle.LifecycleStateRunning, lifecycle.Hook(func() error {
			return fmt.Errorf("running failed")
		}))

		eImpl.startSceneLifecycle(sceneMock)

		suite.Equal(lifecycle.LifecycleStateRunning, lc.State())
	})
}

func (suite *engineTest) TestShutdownSceneLifecycle() {
	suite.Run("nil scene returns false", func() {
		eImpl := suite.engine.(*engine)
		suite.False(eImpl.shutdownSceneLifecycle(nil))
	})

	suite.Run("scene with nil lifecycle returns false", func() {
		eImpl := suite.engine.(*engine)
		sceneMock := scene_mocks.NewMockScene(suite.T())
		sceneMock.EXPECT().Lifecycle().Return(nil).Maybe()
		sceneMock.EXPECT().Name().Return("nil-lifecycle-scene").Maybe()

		suite.False(eImpl.shutdownSceneLifecycle(sceneMock))
	})

	suite.Run("already removed scene returns true", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, _ := newSceneMockWithLifecycle(suite.T(), "removed-scene", lifecycle.LifecycleStateRemoved)

		suite.True(eImpl.shutdownSceneLifecycle(sceneMock))
	})

	suite.Run("unsupported shutdown state returns false", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, lc := newSceneMockWithLifecycle(suite.T(), "starting-scene", lifecycle.LifecycleStateStarting)

		suite.False(eImpl.shutdownSceneLifecycle(sceneMock))
		suite.Equal(lifecycle.LifecycleStateStarting, lc.State())
	})

	suite.Run("transition failure returns false", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, lc := newSceneMockWithLifecycle(suite.T(), "running-fail-scene", lifecycle.LifecycleStateRunning)
		lc.OnTransitionTo(lifecycle.LifecycleStateDraining, lifecycle.Hook(func() error {
			return fmt.Errorf("draining failed")
		}))

		suite.False(eImpl.shutdownSceneLifecycle(sceneMock))
		suite.Equal(lifecycle.LifecycleStateDraining, lc.State())
	})

	suite.Run("running scene transitions to removed on success", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, lc := newSceneMockWithLifecycle(suite.T(), "running-scene", lifecycle.LifecycleStateRunning)

		suite.True(eImpl.shutdownSceneLifecycle(sceneMock))
		suite.Equal(lifecycle.LifecycleStateRemoved, lc.State())
	})

	suite.Run("continues when current state already equals target", func() {
		eImpl := suite.engine.(*engine)
		calls := 0
		lcStub := &lifecycleStub{state: lifecycle.LifecycleStateRunning}
		lcStub.setStateFn = func(target lifecycle.LifecycleState) error {
			calls++
			if calls == 1 {
				lcStub.state = lifecycle.LifecycleStateStopped
				return nil
			}
			lcStub.state = target
			return nil
		}

		sceneMock := scene_mocks.NewMockScene(suite.T())
		sceneMock.EXPECT().Lifecycle().Return(lcStub).Maybe()
		sceneMock.EXPECT().Name().Return("continue-branch-scene").Maybe()

		suite.True(eImpl.shutdownSceneLifecycle(sceneMock))
		suite.Equal(lifecycle.LifecycleStateRemoved, lcStub.State())
	})

	suite.Run("returns false when lifecycle does not reach removed", func() {
		eImpl := suite.engine.(*engine)
		lcStub := &lifecycleStub{state: lifecycle.LifecycleStateStopped}
		lcStub.setStateFn = func(target lifecycle.LifecycleState) error {
			return nil
		}

		sceneMock := scene_mocks.NewMockScene(suite.T())
		sceneMock.EXPECT().Lifecycle().Return(lcStub).Maybe()
		sceneMock.EXPECT().Name().Return("not-removed-scene").Maybe()

		suite.False(eImpl.shutdownSceneLifecycle(sceneMock))
		suite.Equal(lifecycle.LifecycleStateStopped, lcStub.State())
	})
}

func (suite *engineTest) TestRemoveSceneNoOpBranches() {
	suite.Run("missing scene key is ignored", func() {
		eImpl := suite.engine.(*engine)
		initialCount := len(eImpl.scenes)

		suite.engine.RemoveScene(999)

		suite.Equal(initialCount, len(eImpl.scenes))
	})

	suite.Run("nil scene map entry is ignored without deletion", func() {
		eImpl := suite.engine.(*engine)
		eImpl.scenes[77] = nil

		suite.engine.RemoveScene(77)

		_, exists := eImpl.scenes[77]
		suite.True(exists)
		suite.Nil(eImpl.scenes[77])
	})

	suite.Run("scene that cannot complete shutdown is not deleted", func() {
		eImpl := suite.engine.(*engine)
		sceneMock := scene_mocks.NewMockScene(suite.T())
		sceneMock.EXPECT().Lifecycle().Return(nil).Maybe()
		sceneMock.EXPECT().Name().Return("shutdown-false-scene").Maybe()
		eImpl.scenes[88] = sceneMock

		suite.engine.RemoveScene(88)

		_, exists := eImpl.scenes[88]
		suite.True(exists)
		suite.Equal(sceneMock, eImpl.scenes[88])
	})
}

func (suite *engineTest) TestHandleRenderLifecycleFilteringBranches() {
	suite.Run("skips nil scene entries and nil lifecycle scenes", func() {
		eImpl := suite.engine.(*engine)
		for key := range eImpl.scenes {
			delete(eImpl.scenes, key)
		}

		nilLifecycleScene := scene_mocks.NewMockScene(suite.T())
		nilLifecycleScene.EXPECT().Lifecycle().Return(nil).Maybe()
		eImpl.scenes[0] = nil
		eImpl.scenes[1] = nilLifecycleScene

		called := make(chan struct{}, 1)
		var once sync.Once
		eImpl.renderCallback = func(dt float32) {
			once.Do(func() { called <- struct{}{} })
		}

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-called:
		case <-time.After(2 * time.Second):
			suite.Fail("render callback was not called within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}
	})

	suite.Run("paused driver scene with nil renderer skips render pipeline", func() {
		eImpl := suite.engine.(*engine)
		sceneMock, _ := suite.configurePrimaryScene("paused-no-renderer-scene", lifecycle.LifecycleStatePaused)
		sceneMock.EXPECT().Renderer().Return(nil).Maybe()

		called := make(chan struct{}, 1)
		var once sync.Once
		eImpl.renderCallback = func(dt float32) {
			once.Do(func() { called <- struct{}{} })
		}

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-called:
		case <-time.After(2 * time.Second):
			suite.Fail("render callback was not called within timeout")
		}

		eImpl.signalQuit()

		done := make(chan struct{})
		go func() { eImpl.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			suite.Fail("handleRender did not exit within timeout")
		}

		sceneMock.AssertNotCalled(suite.T(), "PrepareCompute")
	})
}

func (suite *engineTest) TestSubmitCommand() {
	suite.Run("should delegate to the internal command queue without panicking", func() {
		cmd := command.NewCommand(command.CommandTypeAsync, command.WithCommandFunc(func(_ oxycontext.Context) error {
			return nil
		}))
		suite.NotPanics(func() {
			suite.engine.SubmitCommand(cmd)
		})
	})

	suite.Run("should accept a linear command", func() {
		cmd := command.NewCommand(command.CommandTypeLinear, command.WithCommandFunc(func(_ oxycontext.Context) error {
			return nil
		}))
		suite.NotPanics(func() {
			suite.engine.SubmitCommand(cmd)
		})
	})
}

func (suite *engineTest) TestCommandQueue() {
	suite.Run("should return the engine's command queue", func() {
		q := suite.engine.CommandQueue()
		suite.NotNil(q)
	})
}
