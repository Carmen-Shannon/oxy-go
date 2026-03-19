package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// SSAOHandlerOption is a functional option for configuring an SSAOHandler
// during construction via NewSSAOHandler.
type SSAOHandlerOption func(*ssaoHandlerImpl)

// WithSSAOScreenSize sets the initial screen dimensions used for SSAO texture
// allocation. These should match the surface dimensions at the time of
// initialization.
//
// Parameters:
//   - width: the screen width in pixels
//   - height: the screen height in pixels
//
// Returns:
//   - SSAOHandlerOption: a function that applies the screen size option to an ssaoHandlerImpl
func WithSSAOScreenSize(width, height int) SSAOHandlerOption {
	return func(h *ssaoHandlerImpl) {
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
//   - SSAOHandlerOption: a function that applies the sample count option to an ssaoHandlerImpl
func WithSSAOSampleCount(count int) SSAOHandlerOption {
	return func(h *ssaoHandlerImpl) {
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
//   - SSAOHandlerOption: a function that applies the max samples option to an ssaoHandlerImpl
func WithSSAOMaxSamples(max int) SSAOHandlerOption {
	return func(h *ssaoHandlerImpl) {
		h.maxSamples = max
	}
}

// WithSSAORadius sets the hemisphere sampling radius in world-space units.
// Larger radii detect occlusion over longer distances but may introduce
// halo artifacts.
//
// Parameters:
//   - radius: the sampling radius in world units (recommended: 0.5)
//
// Returns:
//   - SSAOHandlerOption: a function that applies the radius option to an ssaoHandlerImpl
func WithSSAORadius(radius float32) SSAOHandlerOption {
	return func(h *ssaoHandlerImpl) {
		h.radius = radius
	}
}

// WithSSAOBias sets the depth comparison bias used to prevent self-occlusion
// artifacts on flat surfaces.
//
// Parameters:
//   - bias: the depth bias (recommended: 0.025)
//
// Returns:
//   - SSAOHandlerOption: a function that applies the bias option to an ssaoHandlerImpl
func WithSSAOBias(bias float32) SSAOHandlerOption {
	return func(h *ssaoHandlerImpl) {
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
//   - SSAOHandlerOption: a function that applies the power option to an ssaoHandlerImpl
func WithSSAOPower(power float32) SSAOHandlerOption {
	return func(h *ssaoHandlerImpl) {
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
//   - SSAOHandlerOption: a function that applies the blur radius option to an ssaoHandlerImpl
func WithSSAOBlurRadius(radius int) SSAOHandlerOption {
	return func(h *ssaoHandlerImpl) {
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
//   - SSAOHandlerOption: a function that applies the half-resolution option to an ssaoHandlerImpl
func WithSSAOHalfResolution(enabled bool) SSAOHandlerOption {
	return func(h *ssaoHandlerImpl) {
		h.halfResolution = enabled
	}
}

// NewSSAOHandler creates a new SSAOHandler with sensible defaults and any
// provided options applied. GPU resources are not allocated until the owning
// scene calls the appropriate initialization methods.
//
// Default values:
//   - SampleCount: 16
//   - Radius: 0.5
//   - Bias: 0.025
//   - Power: 2.0
//   - BlurRadius: 4
//
// Parameters:
//   - opts: variadic list of SSAOHandlerOption functions to configure the handler
//
// Returns:
//   - SSAOHandler: a new handler instance ready to be attached to a scene
func NewSSAOHandler(opts ...SSAOHandlerOption) SSAOHandler {
	h := &ssaoHandlerImpl{
		enabled:      false,
		sampleCount:  16,
		maxSamples:   32,
		radius:       0.5,
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
