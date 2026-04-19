package postprocessing

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// SSRHandlerOption is a functional option for configuring an SSRHandler
// during construction via NewSSRHandler.
type SSRHandlerOption func(*ssrHandlerImpl)

// WithSSRScreenSize sets the initial screen dimensions used for SSR texture
// allocation. These should match the surface dimensions at the time of
// initialization.
//
// Parameters:
//   - width: the screen width in pixels
//   - height: the screen height in pixels
//
// Returns:
//   - SSRHandlerOption: a function that applies the screen size option to an ssrHandlerImpl
func WithSSRScreenSize(width, height int) SSRHandlerOption {
	return func(h *ssrHandlerImpl) {
		h.screenWidth = width
		h.screenHeight = height
	}
}

// WithSSRMaxSteps sets the maximum number of ray march steps per pixel.
// Higher values find more reflections but cost more compute.
//
// Parameters:
//   - steps: the maximum step count (recommended: 64)
//
// Returns:
//   - SSRHandlerOption: a function that applies the max steps option to an ssrHandlerImpl
func WithSSRMaxSteps(steps int) SSRHandlerOption {
	return func(h *ssrHandlerImpl) {
		h.maxSteps = steps
	}
}

// WithSSRMaxDistance sets the maximum ray march distance in view-space units.
// Reflections beyond this distance are discarded.
//
// Parameters:
//   - distance: the maximum distance (recommended: 50.0)
//
// Returns:
//   - SSRHandlerOption: a function that applies the max distance option to an ssrHandlerImpl
func WithSSRMaxDistance(distance float32) SSRHandlerOption {
	return func(h *ssrHandlerImpl) {
		h.maxDistance = distance
	}
}

// WithSSRThickness sets the depth thickness tolerance used for hit detection.
// Thinner values are more precise but may miss thin geometry.
//
// Parameters:
//   - thickness: the depth tolerance (recommended: 0.1)
//
// Returns:
//   - SSRHandlerOption: a function that applies the thickness option to an ssrHandlerImpl
func WithSSRThickness(thickness float32) SSRHandlerOption {
	return func(h *ssrHandlerImpl) {
		h.thickness = thickness
	}
}

// WithSSRStride sets the step stride multiplier for the ray march. A stride
// of 1.0 uses uniform stepping; larger strides trade accuracy for speed.
//
// Parameters:
//   - stride: the stride multiplier (recommended: 1.0)
//
// Returns:
//   - SSRHandlerOption: a function that applies the stride option to an ssrHandlerImpl
func WithSSRStride(stride float32) SSRHandlerOption {
	return func(h *ssrHandlerImpl) {
		h.stride = stride
	}
}

// WithSSRRoughnessCutoff sets the roughness value above which SSR is skipped.
// Surfaces rougher than this threshold receive no screen-space reflections.
//
// Parameters:
//   - cutoff: the roughness cutoff (recommended: 0.5)
//
// Returns:
//   - SSRHandlerOption: a function that applies the roughness cutoff option to an ssrHandlerImpl
func WithSSRRoughnessCutoff(cutoff float32) SSRHandlerOption {
	return func(h *ssrHandlerImpl) {
		h.roughnessCutoff = cutoff
	}
}

// NewSSRHandler creates a new SSRHandler with sensible defaults and any
// provided options applied. GPU resources are not allocated until the owning
// scene calls the appropriate initialization methods.
//
// Default values:
//   - MaxSteps: 64
//   - MaxDistance: 50.0
//   - Thickness: 0.1
//   - Stride: 1.0
//   - RoughnessCutoff: 0.5
//
// Parameters:
//   - opts: variadic list of SSRHandlerOption functions to configure the handler
//
// Returns:
//   - SSRHandler: a new handler instance ready to be attached to a scene
func NewSSRHandler(opts ...SSRHandlerOption) SSRHandler {
	h := &ssrHandlerImpl{
		enabled:         false,
		maxSteps:        64,
		maxDistance:     50.0,
		thickness:       0.1,
		stride:          1.0,
		roughnessCutoff: 0.5,
		pipelineKeys:    make(map[string]string),
		bgps: map[string]bind_group_provider.BindGroupProvider{
			"ssr_compute": bind_group_provider.NewBindGroupProvider("ssr_compute"),
		},
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}
