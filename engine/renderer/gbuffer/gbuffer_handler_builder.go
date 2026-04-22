package gbuffer

// GBufferHandlerOption is a functional option for configuring a GBufferHandler
// during construction via NewGBufferHandler.
type GBufferHandlerOption func(*gBufferHandlerImpl)

// WithScreenSize sets the initial screen dimensions used for G-Buffer
// texture allocation.
func WithScreenSize(width, height int) GBufferHandlerOption {
	return func(h *gBufferHandlerImpl) {
		h.screenWidth = width
		h.screenHeight = height
	}
}

// NewGBufferHandler creates a new GBufferHandler with sensible defaults and any
// provided options applied.
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
