package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// ShadowHandlerOption is a functional option for configuring a ShadowHandler
// during construction via NewShadowHandler.
type ShadowHandlerOption func(*shadowHandlerImpl)

// WithShadowNearFar sets the near and far planes for the directional shadow projection.
//
// Parameters:
//   - near: near plane distance
//   - far: far plane distance
//
// Returns:
//   - ShadowHandlerOption: a function that applies the near/far option to a shadowHandlerImpl
func WithShadowNearFar(near, far float32) ShadowHandlerOption {
	return func(h *shadowHandlerImpl) {
		h.shadowNear = near
		h.shadowFar = far
	}
}

// WithShadowNormalBiasScale sets the multiplier applied to the shadow map texel
// world-size to derive the normal-offset bias.
//
// Parameters:
//   - scale: multiplier on per-texel world size
//
// Returns:
//   - ShadowHandlerOption: a function that applies the normal bias scale option to a shadowHandlerImpl
func WithShadowNormalBiasScale(scale float32) ShadowHandlerOption {
	return func(h *shadowHandlerImpl) {
		h.shadowNormalBiasScale = scale
	}
}

// WithShadowMapResolution sets the width and height in texels of the shadow depth texture.
//
// Parameters:
//   - resolution: shadow map width and height in texels
//
// Returns:
//   - ShadowHandlerOption: a function that applies the resolution option to a shadowHandlerImpl
func WithShadowMapResolution(resolution int) ShadowHandlerOption {
	return func(h *shadowHandlerImpl) {
		h.shadowMapResolution = resolution
	}
}

// WithPCFRadius sets the Poisson disk PCF kernel radius in texels. Higher
// values produce softer shadow edges. Typical range: 0.5–3.0.
//
// Parameters:
//   - radius: PCF kernel radius in texels
//
// Returns:
//   - ShadowHandlerOption: a function that applies the PCF radius option
func WithPCFRadius(radius float32) ShadowHandlerOption {
	return func(h *shadowHandlerImpl) {
		h.pcfRadius = radius
	}
}

// WithPCFSamples sets the Poisson disk tap count used for PCF shadow filtering.
//
// Parameters:
//   - samples: the number of PCF samples
//
// Returns:
//   - ShadowHandlerOption: a function that applies the PCF samples option
func WithPCFSamples(samples uint32) ShadowHandlerOption {
	return func(h *shadowHandlerImpl) {
		h.pcfSamples = samples
	}
}

// WithShadowInnerRadius sets the world-space radius of the high-fidelity inner
// shadow cascade. The inner cascade is a camera-centered sphere that maintains
// constant texel density regardless of zoom level. Fragments beyond this radius
// fall back to the outer frustum-fit cascade.
//
// Parameters:
//   - radius: the inner cascade sphere radius in world units
//
// Returns:
//   - ShadowHandlerOption: a function that applies the inner radius option to a shadowHandlerImpl
func WithShadowInnerRadius(radius float32) ShadowHandlerOption {
	return func(h *shadowHandlerImpl) {
		h.shadowInnerRadius = radius
	}
}

// WithLightShadowTileSize sets the width and height in texels of each tile in
// the per-light shadow atlas.
//
// Parameters:
//   - size: the tile width/height in texels
//
// Returns:
//   - ShadowHandlerOption: a function that applies the tile size option to a shadowHandlerImpl
func WithLightShadowTileSize(size int) ShadowHandlerOption {
	return func(h *shadowHandlerImpl) {
		h.lightShadowTileSize = size
	}
}

// NewShadowHandler creates a new ShadowHandler with sensible defaults and any
// provided options applied. GPU resources are not allocated until the owning
// scene calls the appropriate initialization methods.
//
// Parameters:
//   - opts: variadic list of ShadowHandlerOption functions to configure the handler
//
// Returns:
//   - ShadowHandler: a new handler instance ready to be attached to a scene
func NewShadowHandler(opts ...ShadowHandlerOption) ShadowHandler {
	h := &shadowHandlerImpl{
		shadowNear:            0.1,
		shadowFar:             200.0,
		shadowNormalBiasScale: 3.0,
		shadowMapResolution:   2048,
		pcfRadius:             1.0,
		pcfSamples:            16,
		shadowInnerRadius:     100.0,
		lightShadowTileSize:   1024,
		pipelineKeys:          make(map[string]string),
		bgps:                  make(map[string]bind_group_provider.BindGroupProvider),
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}
