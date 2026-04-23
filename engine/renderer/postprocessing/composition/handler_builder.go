package composition

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// HandlerOption is a functional option for configuring a Handler
// during construction via NewHandler.
type HandlerOption func(*handlerImpl)

// WithCompositionScreenSize sets the initial screen dimensions used for HDR texture
// allocation. These should match the surface dimensions at the time of initialization.
//
// Parameters:
//   - width: the screen width in pixels
//   - height: the screen height in pixels
//
// Returns:
//   - HandlerOption: a function that applies the screen size option to a handlerImpl
func WithCompositionScreenSize(width, height int) HandlerOption {
	return func(h *handlerImpl) {
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
//   - HandlerOption: a function that applies the tone mapping option to a handlerImpl
func WithToneMappingEnabled(enabled bool) HandlerOption {
	return func(h *handlerImpl) {
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
//   - HandlerOption: a function that applies the exposure option to a handlerImpl
func WithExposure(exposure float32) HandlerOption {
	return func(h *handlerImpl) {
		h.exposure = exposure
	}
}

// WithAutoExposure enables or disables the eye-adaptation (auto-exposure) system.
// When enabled, the luminance compute shader drives exposure each frame.
//
// Parameters:
//   - enabled: true to enable auto-exposure
//
// Returns:
//   - HandlerOption: a function that applies the auto-exposure option
func WithAutoExposure(enabled bool) HandlerOption {
	return func(h *handlerImpl) {
		h.autoExposureEnabled = enabled
	}
}

// WithAdaptSpeed sets the eye-adaptation rate (exposure change per second).
// Higher values cause faster adaptation to scene luminance changes.
//
// Parameters:
//   - speed: the adaptation speed (exposures per second)
//
// Returns:
//   - HandlerOption: a function that applies the adapt speed option
func WithAdaptSpeed(speed float32) HandlerOption {
	return func(h *handlerImpl) {
		h.adaptSpeed = speed
	}
}

// WithMinExposure sets the lower clamp boundary for the auto-exposure system.
//
// Parameters:
//   - min: the minimum allowed exposure value
//
// Returns:
//   - HandlerOption: a function that applies the min exposure option
func WithMinExposure(min float32) HandlerOption {
	return func(h *handlerImpl) {
		h.minExposure = min
	}
}

// WithMaxExposure sets the upper clamp boundary for the auto-exposure system.
//
// Parameters:
//   - max: the maximum allowed exposure value
//
// Returns:
//   - HandlerOption: a function that applies the max exposure option
func WithMaxExposure(max float32) HandlerOption {
	return func(h *handlerImpl) {
		h.maxExposure = max
	}
}

// WithLuminanceWorkgroupSize sets the tile dimension of the luminance compute
// shader workgroup. The shader dispatches a single (size × size) workgroup
// where each thread samples one HDR texel to compute log-average luminance.
// Defaults to 16 (256 threads total).
//
// Parameters:
//   - size: the workgroup tile dimension (number of threads per axis)
//
// Returns:
//   - HandlerOption: a function that applies the workgroup size option
func WithLuminanceWorkgroupSize(size int) HandlerOption {
	return func(h *handlerImpl) {
		h.luminanceWorkgroupSize = size
	}
}

// WithBloomEnabled enables or disables the bloom post-processing effect.
//
// Parameters:
//   - enabled: true to enable bloom
func WithBloomEnabled(enabled bool) HandlerOption {
	return func(h *handlerImpl) {
		h.bloomEnabled = enabled
	}
}

// WithBloomThreshold sets the brightness threshold for bloom extraction.
// Pixels below this brightness will not contribute to the bloom effect.
//
// Parameters:
//   - threshold: the brightness threshold (typical range 0.5–2.0, default 1.0)
func WithBloomThreshold(threshold float32) HandlerOption {
	return func(h *handlerImpl) {
		h.bloomThreshold = threshold
	}
}

// WithBloomIntensity sets the intensity multiplier applied to the bloom
// contribution when blended into the final composition output.
//
// Parameters:
//   - intensity: the bloom intensity (typical range 0.1–1.0, default 0.5)
func WithBloomIntensity(intensity float32) HandlerOption {
	return func(h *handlerImpl) {
		h.bloomIntensity = intensity
	}
}

// NewHandler creates a new Handler with sensible defaults and any
// provided options applied. GPU resources are not allocated until the owning scene
// calls the appropriate initialization methods.
//
// Default values:
//   - ToneMappingEnabled: true
//   - Exposure: 1.0
//
// Parameters:
//   - opts: variadic list of HandlerOption functions to configure the handler
//
// Returns:
//   - Handler: a new handler instance ready to be attached to a scene
func NewHandler(opts ...HandlerOption) Handler {
	h := &handlerImpl{
		enabled:                false,
		toneMappingEnabled:     true,
		exposure:               1.0,
		autoExposureEnabled:    false,
		adaptSpeed:             1.0,
		minExposure:            0.1,
		maxExposure:            10.0,
		luminanceWorkgroupSize: 16,
		bloomEnabled:           false,
		bloomThreshold:         1.0,
		bloomIntensity:         0.5,
		pipelineKeys:           make(map[string]string),
		bgps: map[string]bind_group_provider.BindGroupProvider{
			"composition":       bind_group_provider.NewBindGroupProvider("composition"),
			"luminance_compute": bind_group_provider.NewBindGroupProvider("luminance_compute"),
		},
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}
