package window

import "fmt"

// WindowBuilderOption is a functional option for configuring an engineWindow.
// Use the With* functions to create options.
type WindowBuilderOption func(w *window)

// WithTitle sets the window title displayed in the title bar.
//
// Parameters:
//   - title: the window title text
//
// Returns:
//   - WindowBuilderOption: option function to apply
func WithTitle(title string) WindowBuilderOption {
	return func(w *window) {
		w.title = title
	}
}

// WithMaxWidth sets the maximum allowed window width.
//
// Parameters:
//   - maxWidth: maximum width in pixels
//
// Returns:
//   - WindowBuilderOption: option function to apply
func WithMaxWidth(maxWidth int) WindowBuilderOption {
	return func(w *window) {
		w.maxWidth = maxWidth
	}
}

// WithMaxHeight sets the maximum allowed window height.
//
// Parameters:
//   - maxHeight: maximum height in pixels
//
// Returns:
//   - WindowBuilderOption: option function to apply
func WithMaxHeight(maxHeight int) WindowBuilderOption {
	return func(w *window) {
		w.maxHeight = maxHeight
	}
}

// WithMinWidth sets the minimum allowed window width.
//
// Parameters:
//   - minWidth: minimum width in pixels
//
// Returns:
//   - WindowBuilderOption: option function to apply
func WithMinWidth(minWidth int) WindowBuilderOption {
	return func(w *window) {
		w.minWidth = minWidth
	}
}

// WithMinHeight sets the minimum allowed window height.
//
// Parameters:
//   - minHeight: minimum height in pixels
//
// Returns:
//   - WindowBuilderOption: option function to apply
func WithMinHeight(minHeight int) WindowBuilderOption {
	return func(w *window) {
		w.minHeight = minHeight
	}
}

// WithWidth sets the initial window width.
//
// Parameters:
//   - width: initial width in pixels
//
// Returns:
//   - WindowBuilderOption: option function to apply
func WithWidth(width int) WindowBuilderOption {
	return func(w *window) {
		w.width = width
	}
}

// WithHeight sets the initial window height.
//
// Parameters:
//   - height: initial height in pixels
//
// Returns:
//   - WindowBuilderOption: option function to apply
func WithHeight(height int) WindowBuilderOption {
	return func(w *window) {
		w.height = height
	}
}

// NewWindow creates a new Window with the specified options.
// Applies default values first, then each option in order.
//
// Parameters:
//   - options: functional options to configure the window
//
// Returns:
//   - Window: the configured window (not yet spawned)
func NewWindow(options ...WindowBuilderOption) Window {
	w := &window{
		title:     "Default Window Title",
		maxWidth:  1600,
		maxHeight: 1200,
		minWidth:  600,
		minHeight: 200,
		width:     1280,
		height:    720,
	}
	for _, opt := range options {
		opt(w)
	}
	if err := newPlatformWindow(w); err != nil {
		panic(fmt.Sprintf("failed to create platform window: %v", err))
	}
	w.Delegate = w
	return w
}
