package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// CompositionHandlerOption is a functional option for configuring a CompositionHandler
// during construction via NewCompositionHandler.
type CompositionHandlerOption func(*compositionHandlerImpl)

// WithCompositionScreenSize sets the initial screen dimensions used for HDR texture
// allocation. These should match the surface dimensions at the time of initialization.
//
// Parameters:
//   - width: the screen width in pixels
//   - height: the screen height in pixels
//
// Returns:
//   - CompositionHandlerOption: a function that applies the screen size option to a compositionHandlerImpl
func WithCompositionScreenSize(width, height int) CompositionHandlerOption {
	return func(h *compositionHandlerImpl) {
		h.screenWidth = width
		h.screenHeight = height
	}
}

// WithToneMappingEnabled enables or disables ACES tone mapping during the
// composition pass. When disabled, the HDR texture is passed through with
// only gamma correction applied.
//
// Parameters:
//   - enabled: true to enable ACES tone mapping
//
// Returns:
//   - CompositionHandlerOption: a function that applies the tone mapping option to a compositionHandlerImpl
func WithToneMappingEnabled(enabled bool) CompositionHandlerOption {
	return func(h *compositionHandlerImpl) {
		h.toneMappingEnabled = enabled
	}
}

// WithExposure sets the exposure multiplier applied to HDR values before
// tone mapping. Higher values brighten the final image. A value of 1.0 is neutral.
//
// Parameters:
//   - exposure: the exposure multiplier
//
// Returns:
//   - CompositionHandlerOption: a function that applies the exposure option to a compositionHandlerImpl
func WithExposure(exposure float32) CompositionHandlerOption {
	return func(h *compositionHandlerImpl) {
		h.exposure = exposure
	}
}

// NewCompositionHandler creates a new CompositionHandler with sensible defaults and any
// provided options applied. GPU resources are not allocated until the owning scene
// calls the appropriate initialization methods.
//
// Default values:
//   - ToneMappingEnabled: true
//   - Exposure: 1.0
//
// Parameters:
//   - opts: variadic list of CompositionHandlerOption functions to configure the handler
//
// Returns:
//   - CompositionHandler: a new handler instance ready to be attached to a scene
func NewCompositionHandler(opts ...CompositionHandlerOption) CompositionHandler {
	h := &compositionHandlerImpl{
		enabled:            false,
		toneMappingEnabled: true,
		exposure:           1.0,
		pipelineKeys:       make(map[string]string),
		bgps: map[string]bind_group_provider.BindGroupProvider{
			"composition": bind_group_provider.NewBindGroupProvider("composition"),
		},
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}
