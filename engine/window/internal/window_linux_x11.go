//go:build linux && !wayland
// +build linux,!wayland

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
	displayHandle = uintptr(unsafe.Pointer(glfw.GetX11Display()))
	windowHandle = uintptr(w.GetX11Window())
	return displayHandle, windowHandle, nil
}
