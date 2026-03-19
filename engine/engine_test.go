package engine

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/Carmen-Shannon/oxy-go/engine/profiler"
	renderer_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/mocks"
	scene_mocks "github.com/Carmen-Shannon/oxy-go/engine/scene/mocks"
	window_mocks "github.com/Carmen-Shannon/oxy-go/engine/window/mocks"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

func TestRunEngineTests(t *testing.T) {
	suite.Run(t, new(engineTest))
}

type engineTest struct {
	suite.Suite
	windowMock *window_mocks.MockWindow
	sceneMock  *scene_mocks.MockScene
	engine     Engine
}

func (suite *engineTest) SetupSubTest() {
	suite.windowMock = window_mocks.NewMockWindow(suite.T())
	suite.windowMock.EXPECT().SetResizeCallback(mock.Anything).Return().Maybe()
	suite.sceneMock = scene_mocks.NewMockScene(suite.T())
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

func (suite *engineTest) TestSetRenderCallback() {
	suite.Run("should set the render callback function", func() {
		called := false
		dummyCallback := func(deltaTime float32) { called = true }
		suite.engine.SetRenderCallback(dummyCallback)
		eImpl := suite.engine.(*engine)
		suite.NotNil(eImpl.renderCallback)
		eImpl.renderCallback(0)
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
		suite.engine.AddScene(1, suite.sceneMock)
		scene := suite.engine.Scene(1)
		suite.Equal(suite.sceneMock, scene)
	})

	suite.Run("should overwrite an existing scene with the same key", func() {
		newSceneMock := scene_mocks.NewMockScene(suite.T())
		suite.engine.AddScene(0, newSceneMock)
		scene := suite.engine.Scene(0)
		suite.Equal(newSceneMock, scene)
	})
}

func (suite *engineTest) TestRemoveScene() {
	suite.Run("should remove a scene from the engine", func() {
		suite.engine.RemoveScene(0)
		scene := suite.engine.Scene(0)
		suite.Nil(scene)
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
		suite.sceneMock.EXPECT().Active().Return(false).Maybe()
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
		suite.sceneMock.EXPECT().Active().Return(false).Maybe()

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

		rendered := make(chan struct{}, 1)
		var once sync.Once
		rendererMock.EXPECT().Present().RunAndReturn(func() {
			once.Do(func() { rendered <- struct{}{} })
		}).Maybe()

		suite.sceneMock.EXPECT().Active().Return(true).Maybe()
		suite.sceneMock.EXPECT().Renderer().Return(rendererMock).Maybe()
		rendererMock.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		suite.sceneMock.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Return().Maybe()
		rendererMock.EXPECT().EndComputeFrame().Return().Maybe()
		rendererMock.EXPECT().BeginGeometryFrame().Return(nil).Maybe()
		suite.sceneMock.EXPECT().PrepareShadows().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareGBuffer().Return().Maybe()
		rendererMock.EXPECT().EndGeometryFrame().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareLightCulling().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareSSAO().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareContactShadows().Return().Maybe()
		suite.sceneMock.EXPECT().BeginHDRFrame().Return(nil).Maybe()
		suite.sceneMock.EXPECT().DrawCalls().Return(nil).Maybe()
		rendererMock.EXPECT().EndFrame().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareSSR().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareComposition().Return().Maybe()
		rendererMock.EXPECT().FlushFrame().Return().Maybe()

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

	suite.Run("should fall back to the basic frame path when HDR frame initialization fails", func() {
		eImpl := suite.engine.(*engine)
		rendererMock := renderer_mocks.NewMockRenderer(suite.T())

		rendered := make(chan struct{}, 1)
		var once sync.Once
		rendererMock.EXPECT().Present().RunAndReturn(func() {
			once.Do(func() { rendered <- struct{}{} })
		}).Maybe()

		suite.sceneMock.EXPECT().Active().Return(true).Maybe()
		suite.sceneMock.EXPECT().Renderer().Return(rendererMock).Maybe()
		rendererMock.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		suite.sceneMock.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Return().Maybe()
		rendererMock.EXPECT().EndComputeFrame().Return().Maybe()
		rendererMock.EXPECT().BeginGeometryFrame().Return(nil).Maybe()
		suite.sceneMock.EXPECT().PrepareShadows().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareGBuffer().Return().Maybe()
		rendererMock.EXPECT().EndGeometryFrame().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareLightCulling().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareSSAO().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareContactShadows().Return().Maybe()
		suite.sceneMock.EXPECT().BeginHDRFrame().Return(fmt.Errorf("hdr unavailable")).Maybe()
		rendererMock.EXPECT().BeginFrame().Return(nil).Maybe()
		suite.sceneMock.EXPECT().DrawCalls().Return(nil).Maybe()
		rendererMock.EXPECT().EndFrame().Return().Maybe()
		rendererMock.EXPECT().FlushFrame().Return().Maybe()

		eImpl.wg.Add(1)
		go eImpl.handleRender()

		select {
		case <-rendered:
		case <-time.After(2 * time.Second):
			suite.Fail("basic frame path was not executed within timeout")
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

	suite.Run("should skip the renderer pipeline when the active scene has no renderer", func() {
		eImpl := suite.engine.(*engine)
		suite.sceneMock.EXPECT().Active().Return(true).Maybe()
		suite.sceneMock.EXPECT().Renderer().Return(nil).Maybe()

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
		suite.sceneMock.EXPECT().Active().Return(false).Maybe()

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

		suite.sceneMock.EXPECT().Active().Return(true).Maybe()
		suite.sceneMock.EXPECT().Renderer().Return(rendererMock).Maybe()
		rendererMock.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		suite.sceneMock.EXPECT().PrepareCompute(mock.AnythingOfType("float32")).Return().Maybe()
		rendererMock.EXPECT().EndComputeFrame().Return().Maybe()
		rendererMock.EXPECT().BeginGeometryFrame().Return(nil).Maybe()
		suite.sceneMock.EXPECT().PrepareShadows().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareGBuffer().Return().Maybe()
		rendererMock.EXPECT().EndGeometryFrame().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareLightCulling().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareSSAO().Return().Maybe()
		suite.sceneMock.EXPECT().PrepareContactShadows().Return().Maybe()
		suite.sceneMock.EXPECT().BeginHDRFrame().Return(nil).Maybe()
		suite.sceneMock.EXPECT().DrawCalls().RunAndReturn(func() error {
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
		suite.sceneMock.EXPECT().Active().Return(false).Maybe()

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
		suite.sceneMock.EXPECT().Active().Return(false).Maybe()
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
}

func (suite *engineTest) TestResizeCallback() {
	suite.Run("should call Resize on all registered scenes", func() {
		suite.sceneMock.EXPECT().Resize(800, 600).Return().Once()
		suite.engine.(*engine).resizeCallback(800, 600)
	})
}
