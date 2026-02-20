package common

// Virtual key codes for cross-platform input handling.
// These values match GLFW key codes which use ASCII values for printable keys.
// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw#Key
const (
	KeyW         = 87  // W key (ASCII)
	KeyA         = 65  // A key (ASCII)
	KeyS         = 83  // S key (ASCII)
	KeyD         = 68  // D key (ASCII)
	KeyQ         = 81  // Q key (ASCII)
	KeyE         = 69  // E key (ASCII)
	KeyB         = 66  // B key (ASCII)
	KeyC         = 67  // C key (ASCII)
	KeyF         = 70  // F key (ASCII)
	KeyG         = 71  // G key (ASCII)
	KeyL         = 76  // L key (ASCII)
	KeyM         = 77  // M key (ASCII)
	KeyT         = 84  // T key (ASCII)
	KeyV         = 86  // V key (ASCII)
	KeyX         = 88  // X key (ASCII)
	KeySpace     = 32  // Spacebar (ASCII)
	KeyBackspace = 259 // Backspace key (GLFW)
	KeyEsc       = 256 // Escape key (GLFW)

	Key0 = 48 // 0 key (ASCII)
	Key1 = 49 // 1 key (ASCII)
	Key2 = 50 // 2 key (ASCII)
	Key3 = 51 // 3 key (ASCII)
	Key4 = 52 // 4 key (ASCII)
	Key5 = 53 // 5 key (ASCII)
	Key6 = 54 // 6 key (ASCII)
	Key7 = 55 // 7 key (ASCII)
	Key8 = 56 // 8 key (ASCII)
	Key9 = 57 // 9 key (ASCII)
)

// Additional non-printable keys
const (
	KeyLeftShift  = 340 // Left Shift (GLFW)
	KeyRightShift = 344 // Right Shift (GLFW)
)
