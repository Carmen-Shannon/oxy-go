package ssao

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// HandlerOption is a functional option for configuring a Handler
// during construction via NewHandler.
type HandlerOption func(*handlerImpl)

// WithSSAOScreenSize sets the initial screen dimensions used for SSAO texture
// allocation. These should match the surface dimensions at the time of
// initialization.
//
// Parameters:
//   - width: the screen width in pixels
//   - height: the screen height in pixels
//
// Returns:
//   - HandlerOption: a function that applies the screen size option to a handlerImpl
func WithSSAOScreenSize(width, height int) HandlerOption {
	return func(h *handlerImpl) {
		h.screenWidth = width
		h.screenHeight = height
	}
}

// WithSSAOSampleCount sets the number of hemisphere samples per pixel.
// Higher values produce smoother AO at the cost of more texture fetches.
// Clamped to [1, 32] during compute dispatch.
//
// Parameters:
//   - count: the number of samples (recommended: 16)
//
// Returns:
//   - HandlerOption: a function that applies the sample count option to a handlerImpl
func WithSSAOSampleCount(count int) HandlerOption {
	return func(h *handlerImpl) {
		h.sampleCount = count
	}
}

// WithSSAOMaxSamples sets the GPU compile-time upper bound for the SSAO kernel
// sample array.
//
// Parameters:
//   - max: the maximum number of samples (recommended: 32)
//
// Returns:
//   - HandlerOption: a function that applies the max samples option to a handlerImpl
func WithSSAOMaxSamples(max int) HandlerOption {
	return func(h *handlerImpl) {
		h.maxSamples = max
	}
}

// WithSSAOScreenRadius sets the desired SSAO sampling radius in screen pixels.
// The engine auto-computes the world-space radius each frame from this value,
// the camera distance, FOV, and screen height, producing consistent visual
// results regardless of zoom level.
//
// Parameters:
//   - pixels: the screen-space radius in pixels (recommended: 24.0)
//
// Returns:
//   - HandlerOption: a function that applies the screen radius option to a handlerImpl
func WithSSAOScreenRadius(pixels float32) HandlerOption {
	return func(h *handlerImpl) {
		h.screenRadius = pixels
	}
}

// WithSSAOBias sets the depth comparison bias used to prevent self-occlusion
// artifacts on flat surfaces.
//
// Parameters:
//   - bias: the depth bias (recommended: 0.025)
//
// Returns:
//   - HandlerOption: a function that applies the bias option to a handlerImpl
func WithSSAOBias(bias float32) HandlerOption {
	return func(h *handlerImpl) {
		h.bias = bias
	}
}

// WithSSAOPower sets the exponent applied to the final AO value. Higher
// values produce darker, more contrasty occlusion.
//
// Parameters:
//   - power: the power exponent (recommended: 2.0)
//
// Returns:
//   - HandlerOption: a function that applies the power option to a handlerImpl
func WithSSAOPower(power float32) HandlerOption {
	return func(h *handlerImpl) {
		h.power = power
	}
}

// WithSSAOBlurRadius sets the half-width of the bilateral blur kernel in
// texels. A radius of 4 produces a 9-texel kernel (2×4+1).
//
// Parameters:
//   - radius: the blur kernel half-width in texels (recommended: 4)
//
// Returns:
//   - HandlerOption: a function that applies the blur radius option to a handlerImpl
func WithSSAOBlurRadius(radius int) HandlerOption {
	return func(h *handlerImpl) {
		h.blurRadius = radius
	}
}

// WithSSAOHalfResolution enables or disables half-resolution SSAO. When
// enabled, SSAO textures are allocated at half the screen resolution in each
// dimension (quarter pixel count), significantly reducing compute cost at the
// expense of slightly softer ambient occlusion. The bilateral blur and lit
// shader linear sampler provide adequate upsampling quality.
//
// Parameters:
//   - enabled: true to enable half-resolution SSAO
//
// Returns:
//   - HandlerOption: a function that applies the half-resolution option to a handlerImpl
func WithSSAOHalfResolution(enabled bool) HandlerOption {
	return func(h *handlerImpl) {
		h.halfResolution = enabled
	}
}

// NewHandler creates a new Handler with sensible defaults and any
// provided options applied. GPU resources are not allocated until the owning
// scene calls the appropriate initialization methods.
//
// Default values:
//   - SampleCount: 16
//   - ScreenRadius: 24.0
//   - Bias: 0.025
//   - Power: 2.0
//   - BlurRadius: 4
//
// Parameters:
//   - opts: variadic list of HandlerOption functions to configure the handler
//
// Returns:
//   - Handler: a new handler instance ready to be attached to a scene
func NewHandler(opts ...HandlerOption) Handler {
	h := &handlerImpl{
		enabled:      false,
		sampleCount:  16,
		maxSamples:   32,
		screenRadius: 24.0,
		bias:         0.025,
		power:        2.0,
		blurRadius:   4,
		pipelineKeys: make(map[string]string),
		bgps: map[string]bind_group_provider.BindGroupProvider{
			"ssao_compute": bind_group_provider.NewBindGroupProvider("ssao_compute"),
			"ssao_blur_h":  bind_group_provider.NewBindGroupProvider("ssao_blur_h"),
			"ssao_blur_v":  bind_group_provider.NewBindGroupProvider("ssao_blur_v"),
		},
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}
