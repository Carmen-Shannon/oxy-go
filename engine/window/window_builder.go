package window

// WindowBuilderOption is a functional option for configuring an engineWindow.
// Use the With* functions to create options.
type WindowBuilderOption func(w *engineWindow)

// WithTitle sets the window title displayed in the title bar.
//
// Parameters:
//   - title: the window title text
//
// Returns:
//   - WindowBuilderOption: option function to apply
func WithTitle(title string) WindowBuilderOption {
	return func(w *engineWindow) {
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
	return func(w *engineWindow) {
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
	return func(w *engineWindow) {
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
	return func(w *engineWindow) {
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
	return func(w *engineWindow) {
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
	return func(w *engineWindow) {
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
	return func(w *engineWindow) {
		w.height = height
	}
}
