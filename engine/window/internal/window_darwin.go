//go:build darwin
// +build darwin

package internal

import (
	"fmt"

	"github.com/go-gl/glfw/v3.3/glfw"
)

func PlatformGetHandles(w *glfw.Window) (displayHandle, windowHandle uintptr, err error) {
	if w == nil {
		return 0, 0, fmt.Errorf("window is not initialized")
	}
	displayHandle = 0
	windowHandle = uintptr(w.GetCocoaWindow())
	return displayHandle, windowHandle, nil
}
