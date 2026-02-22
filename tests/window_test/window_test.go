package window_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/window"
	"github.com/stretchr/testify/suite"
)

type windowTest struct {
	suite.Suite
}

func TestWindow(t *testing.T) {
	suite.Run(t, new(windowTest))
}

func (suite *windowTest) TestNewWindow() {
	suite.Run("creates window with default values", func() {
		w := window.NewWindow()
		suite.NotNil(w)
		suite.Greater(w.Width(), 0)
		suite.Greater(w.Height(), 0)
		suite.NoError(w.Close())
	})

	suite.Run("applies all builder options", func() {
		w := window.NewWindow(
			window.WithTitle("Test Window"),
			window.WithWidth(800),
			window.WithHeight(600),
			window.WithMinWidth(400),
			window.WithMinHeight(300),
			window.WithMaxWidth(1920),
			window.WithMaxHeight(1080),
		)
		suite.NotNil(w)
		suite.Greater(w.Width(), 0)
		suite.Greater(w.Height(), 0)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestWithTitle() {
	suite.Run("applies title option to window", func() {
		w := window.NewWindow(window.WithTitle("Custom Title"))
		suite.NotNil(w)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestWithMaxWidth() {
	suite.Run("applies max width option to window", func() {
		w := window.NewWindow(window.WithMaxWidth(1920))
		suite.NotNil(w)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestWithMaxHeight() {
	suite.Run("applies max height option to window", func() {
		w := window.NewWindow(window.WithMaxHeight(1080))
		suite.NotNil(w)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestWithMinWidth() {
	suite.Run("applies min width option to window", func() {
		w := window.NewWindow(window.WithMinWidth(400))
		suite.NotNil(w)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestWithMinHeight() {
	suite.Run("applies min height option to window", func() {
		w := window.NewWindow(window.WithMinHeight(300))
		suite.NotNil(w)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestWithWidth() {
	suite.Run("applies width option to window", func() {
		w := window.NewWindow(window.WithWidth(800))
		suite.Greater(w.Width(), 0)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestWithHeight() {
	suite.Run("applies height option to window", func() {
		w := window.NewWindow(window.WithHeight(600))
		suite.Greater(w.Height(), 0)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestIsRunning() {
	suite.Run("returns true for a newly created window", func() {
		w := window.NewWindow()
		suite.True(w.IsRunning())
		suite.NoError(w.Close())
	})

	suite.Run("returns false after close", func() {
		w := window.NewWindow()
		w.Close()
		suite.False(w.IsRunning())
	})
}

func (suite *windowTest) TestClose() {
	suite.Run("returns nil error on success", func() {
		w := window.NewWindow()
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestWidth() {
	suite.Run("returns positive value for default window", func() {
		w := window.NewWindow()
		suite.Greater(w.Width(), 0)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestHeight() {
	suite.Run("returns positive value for default window", func() {
		w := window.NewWindow()
		suite.Greater(w.Height(), 0)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestSurfaceDescriptor() {
	suite.Run("returns non-nil descriptor for active window", func() {
		w := window.NewWindow()
		suite.NotNil(w.SurfaceDescriptor())
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestSetUpdateCallback() {
	suite.Run("accepts non-nil and nil callbacks", func() {
		w := window.NewWindow()
		w.SetUpdateCallback(func() {})
		w.SetUpdateCallback(nil)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestSetResizeCallback() {
	suite.Run("accepts non-nil and nil callbacks", func() {
		w := window.NewWindow()
		w.SetResizeCallback(func(width, height int) {})
		w.SetResizeCallback(nil)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestSetScrollCallback() {
	suite.Run("accepts non-nil and nil callbacks", func() {
		w := window.NewWindow()
		w.SetScrollCallback(func(delta float32) {})
		w.SetScrollCallback(nil)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestSetKeyDownCallback() {
	suite.Run("accepts non-nil and nil callbacks", func() {
		w := window.NewWindow()
		w.SetKeyDownCallback(func(keyCode uint32) {})
		w.SetKeyDownCallback(nil)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestSetKeyUpCallback() {
	suite.Run("accepts non-nil and nil callbacks", func() {
		w := window.NewWindow()
		w.SetKeyUpCallback(func(keyCode uint32) {})
		w.SetKeyUpCallback(nil)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestSetMiddleMouseDownCallback() {
	suite.Run("accepts non-nil and nil callbacks", func() {
		w := window.NewWindow()
		w.SetMiddleMouseDownCallback(func(x, y int32) {})
		w.SetMiddleMouseDownCallback(nil)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestSetMiddleMouseUpCallback() {
	suite.Run("accepts non-nil and nil callbacks", func() {
		w := window.NewWindow()
		w.SetMiddleMouseUpCallback(func(x, y int32) {})
		w.SetMiddleMouseUpCallback(nil)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestSetMouseMoveCallback() {
	suite.Run("accepts non-nil and nil callbacks", func() {
		w := window.NewWindow()
		w.SetMouseMoveCallback(func(x, y int32) {})
		w.SetMouseMoveCallback(nil)
		suite.NoError(w.Close())
	})
}

func (suite *windowTest) TestProcessMessages() {
	suite.Run("exits when window is closed from update callback", func() {
		w := window.NewWindow()
		iterations := 0
		w.SetUpdateCallback(func() {
			iterations++
			w.Close()
		})
		w.ProcessMessages()
		suite.GreaterOrEqual(iterations, 1)
	})

	suite.Run("returns immediately when window is already closed", func() {
		w := window.NewWindow()
		w.Close()
		w.ProcessMessages()
		suite.False(w.IsRunning())
	})
}
