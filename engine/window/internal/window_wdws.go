//go:build windows
// +build windows

package internal

import (
	"fmt"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
)

func PlatformGetHandles(w *glfw.Window) (displayHandle, windowHandle uintptr, err error) {
	if w == nil {
		return 0, 0, fmt.Errorf("window is not initialized")
	}

	displayHandle = 0
	windowHandle = uintptr(unsafe.Pointer(w.GetWin32Window()))
	return displayHandle, windowHandle, nil
}
