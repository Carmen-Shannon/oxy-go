package gbuffer

// HandlerOption is a functional option for configuring a Handler during
// construction via NewHandler.
type HandlerOption func(*handlerImpl)

// GBufferHandler is a backward-compatible alias for Handler.
type GBufferHandler = Handler

// GBufferHandlerOption is a backward-compatible alias for HandlerOption.
type GBufferHandlerOption = HandlerOption

// WithScreenSize sets the initial screen dimensions used for G-Buffer
// texture allocation.
func WithScreenSize(width, height int) HandlerOption {
	return func(h *handlerImpl) {
		h.screenWidth = width
		h.screenHeight = height
	}
}

// NewHandler creates a new Handler with sensible defaults and any provided
// options applied.
func NewHandler(opts ...HandlerOption) Handler {
	h := &handlerImpl{
		enabled:      false,
		pipelineKeys: make(map[string]string),
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

// NewGBufferHandler creates a new Handler using the legacy constructor name.
func NewGBufferHandler(opts ...GBufferHandlerOption) GBufferHandler {
	return NewHandler(opts...)
}
