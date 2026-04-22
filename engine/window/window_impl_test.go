package window

import (
	"fmt"
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

// fakeBackend is a test-only implementation of platformBackend.
type fakeBackend struct {
	isRunningCallCount int
	maxIsRunning       int
	processMsgReturn   bool
	descriptorReturn   *wgpu.SurfaceDescriptor
	closeReturn        error
}

func (f *fakeBackend) isRunning() bool {
	f.isRunningCallCount++
	return f.isRunningCallCount <= f.maxIsRunning
}

func (f *fakeBackend) processMessages() bool {
	return f.processMsgReturn
}

func (f *fakeBackend) surfaceDescriptor() *wgpu.SurfaceDescriptor {
	return f.descriptorReturn
}

func (f *fakeBackend) close() error {
	return f.closeReturn
}

type windowTest struct {
	suite.Suite
}

func TestWindow(t *testing.T) {
	suite.Run(t, new(windowTest))
}

func (suite *windowTest) TestWithTitle() {
	suite.Run("sets title on window", func() {
		w := &window{}
		WithTitle("hello")(w)
		suite.Equal("hello", w.title)
	})
}

func (suite *windowTest) TestWithMaxWidth() {
	suite.Run("sets maxWidth on window", func() {
		w := &window{}
		WithMaxWidth(1920)(w)
		suite.Equal(1920, w.maxWidth)
	})
}

func (suite *windowTest) TestWithMaxHeight() {
	suite.Run("sets maxHeight on window", func() {
		w := &window{}
		WithMaxHeight(1080)(w)
		suite.Equal(1080, w.maxHeight)
	})
}

func (suite *windowTest) TestWithMinWidth() {
	suite.Run("sets minWidth on window", func() {
		w := &window{}
		WithMinWidth(320)(w)
		suite.Equal(320, w.minWidth)
	})
}

func (suite *windowTest) TestWithMinHeight() {
	suite.Run("sets minHeight on window", func() {
		w := &window{}
		WithMinHeight(240)(w)
		suite.Equal(240, w.minHeight)
	})
}

func (suite *windowTest) TestWithWidth() {
	suite.Run("sets width on window", func() {
		w := &window{}
		WithWidth(1280)(w)
		suite.Equal(1280, w.width)
	})
}

func (suite *windowTest) TestWithHeight() {
	suite.Run("sets height on window", func() {
		w := &window{}
		WithHeight(720)(w)
		suite.Equal(720, w.height)
	})
}

func (suite *windowTest) TestSetCallbacks() {
	suite.Run("SetUpdateCallback sets onUpdate to non-nil", func() {
		w := &window{}
		w.SetUpdateCallback(func() {})
		suite.NotNil(w.onUpdate)
	})

	suite.Run("SetUpdateCallback sets onUpdate to nil", func() {
		w := &window{}
		w.SetUpdateCallback(func() {})
		w.SetUpdateCallback(nil)
		suite.Nil(w.onUpdate)
	})

	suite.Run("SetResizeCallback sets onResize to non-nil", func() {
		w := &window{}
		w.SetResizeCallback(func(width, height int) {})
		suite.NotNil(w.onResize)
	})

	suite.Run("SetScrollCallback sets onScroll to non-nil", func() {
		w := &window{}
		w.SetScrollCallback(func(delta float32) {})
		suite.NotNil(w.onScroll)
	})

	suite.Run("SetKeyDownCallback sets onKeyDown to non-nil", func() {
		w := &window{}
		w.SetKeyDownCallback(func(keyCode uint32) {})
		suite.NotNil(w.onKeyDown)
	})

	suite.Run("SetKeyUpCallback sets onKeyUp to non-nil", func() {
		w := &window{}
		w.SetKeyUpCallback(func(keyCode uint32) {})
		suite.NotNil(w.onKeyUp)
	})

	suite.Run("SetMiddleMouseDownCallback sets onMiddleMouseDown to non-nil", func() {
		w := &window{}
		w.SetMiddleMouseDownCallback(func(x, y int32) {})
		suite.NotNil(w.onMiddleMouseDown)
	})

	suite.Run("SetMiddleMouseUpCallback sets onMiddleMouseUp to non-nil", func() {
		w := &window{}
		w.SetMiddleMouseUpCallback(func(x, y int32) {})
		suite.NotNil(w.onMiddleMouseUp)
	})

	suite.Run("SetMouseMoveCallback sets onMouseMove to non-nil", func() {
		w := &window{}
		w.SetMouseMoveCallback(func(x, y int32) {})
		suite.NotNil(w.onMouseMove)
	})
}

func (suite *windowTest) TestWidth() {
	suite.Run("returns window width", func() {
		w := &window{width: 800}
		suite.Equal(800, w.Width())
	})
}

func (suite *windowTest) TestHeight() {
	suite.Run("returns window height", func() {
		w := &window{height: 600}
		suite.Equal(600, w.Height())
	})
}

func (suite *windowTest) TestSurfaceDescriptor() {
	suite.Run("returns nil when backend is nil", func() {
		w := &window{}
		suite.Nil(w.SurfaceDescriptor())
	})

	suite.Run("returns descriptor from backend", func() {
		w := &window{
			backend: &fakeBackend{descriptorReturn: &wgpu.SurfaceDescriptor{}},
		}
		suite.NotNil(w.SurfaceDescriptor())
	})
}

func (suite *windowTest) TestIsRunning() {
	suite.Run("returns false when backend is nil", func() {
		w := &window{}
		suite.False(w.IsRunning())
	})

	suite.Run("returns true when backend reports running", func() {
		w := &window{
			backend: &fakeBackend{maxIsRunning: 1},
		}
		suite.True(w.IsRunning())
	})

	suite.Run("returns false when backend reports not running", func() {
		w := &window{
			backend: &fakeBackend{maxIsRunning: 0},
		}
		suite.False(w.IsRunning())
	})
}

func (suite *windowTest) TestGLFWWindowIsRunning() {
	suite.Run("returns false without panic when not running and window is nil", func() {
		gw := &glfwWindow{
			running: false,
			window:  nil,
		}

		result := true
		suite.NotPanics(func() {
			result = gw.isRunning()
		})

		suite.False(result)
	})
}

func (suite *windowTest) TestClose() {
	suite.Run("returns error when backend is nil", func() {
		w := &window{}
		suite.Error(w.Close())
	})

	suite.Run("returns nil when backend closes successfully", func() {
		w := &window{
			backend: &fakeBackend{closeReturn: nil},
		}
		suite.NoError(w.Close())
	})

	suite.Run("returns error from backend", func() {
		w := &window{
			backend: &fakeBackend{closeReturn: fmt.Errorf("fail")},
		}
		suite.Error(w.Close())
	})
}

func (suite *windowTest) TestProcessMessages() {
	suite.Run("loop never executes when backend not running", func() {
		w := &window{backend: &fakeBackend{maxIsRunning: 0}}
		w.Delegate = w
		called := false
		w.onUpdate = func() { called = true }
		w.ProcessMessages()
		suite.False(called)
	})

	suite.Run("loop breaks when processMessages returns false", func() {
		w := &window{backend: &fakeBackend{maxIsRunning: 1, processMsgReturn: false}}
		w.Delegate = w
		called := false
		w.onUpdate = func() { called = true }
		w.ProcessMessages()
		suite.False(called)
	})

	suite.Run("loop executes onUpdate when set", func() {
		w := &window{backend: &fakeBackend{maxIsRunning: 1, processMsgReturn: true}}
		w.Delegate = w
		called := false
		w.onUpdate = func() { called = true }
		w.ProcessMessages()
		suite.True(called)
	})

	suite.Run("loop does not panic with nil onUpdate", func() {
		w := &window{backend: &fakeBackend{maxIsRunning: 1, processMsgReturn: true}}
		w.Delegate = w
		w.onUpdate = nil
		suite.NotPanics(func() { w.ProcessMessages() })
	})
}
