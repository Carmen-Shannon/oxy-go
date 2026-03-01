package light

// GBufferHandlerOption is a functional option for configuring a GBufferHandler
// during construction via NewGBufferHandler.
type GBufferHandlerOption func(*gBufferHandlerImpl)

// WithGBufferScreenSize sets the initial screen dimensions used for G-Buffer
// texture allocation. These should match the surface dimensions at the time
// of initialization.
//
// Parameters:
//   - width: the screen width in pixels
//   - height: the screen height in pixels
//
// Returns:
//   - GBufferHandlerOption: a function that applies the screen size option to a gBufferHandlerImpl
func WithGBufferScreenSize(width, height int) GBufferHandlerOption {
	return func(h *gBufferHandlerImpl) {
		h.screenWidth = width
		h.screenHeight = height
	}
}

// NewGBufferHandler creates a new GBufferHandler with sensible defaults and any
// provided options applied. GPU resources are not allocated until the owning
// scene calls the appropriate initialization methods.
//
// Parameters:
//   - opts: variadic list of GBufferHandlerOption functions to configure the handler
//
// Returns:
//   - GBufferHandler: a new handler instance ready to be attached to a scene
func NewGBufferHandler(opts ...GBufferHandlerOption) GBufferHandler {
	h := &gBufferHandlerImpl{
		enabled:      false,
		pipelineKeys: make(map[string]string),
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}
