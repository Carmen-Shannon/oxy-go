package light

import "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"

// LightingHandlerOption is a functional option for configuring a LightingHandler
// during construction via NewLightingHandler.
type LightingHandlerOption func(*lightingHandlerImpl)

// WithAmbientColor sets the initial ambient light color for the scene.
// Default is black (no ambient contribution).
//
// Parameters:
//   - color: the ambient RGB color
//
// Returns:
//   - LightingHandlerOption: a function that applies the ambient color option to a lightingHandlerImpl
func WithAmbientColor(color [3]float32) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.ambientColor = color
	}
}

// WithGBufferHandler attaches a pre-configured GBufferHandler to the lighting
// subsystem, overriding the default that is auto-created by NewLightingHandler.
// GPU resources are initialized lazily during the first lighting initialization.
//
// Parameters:
//   - handler: the pre-configured GBufferHandler
//
// Returns:
//   - LightingHandlerOption: a function that applies the G-Buffer handler option to a lightingHandlerImpl
func WithGBufferHandler(handler GBufferHandler) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.gBufferHandler = handler
	}
}

// WithSSAOHandler attaches a pre-configured SSAOHandler to the lighting
// subsystem, overriding the default that is auto-created by NewLightingHandler.
// GPU resources are initialized lazily during the first lighting initialization.
//
// Parameters:
//   - handler: the pre-configured SSAOHandler
//
// Returns:
//   - LightingHandlerOption: a function that applies the SSAO handler option to a lightingHandlerImpl
func WithSSAOHandler(handler SSAOHandler) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.ssaoHandler = handler
	}
}

// WithCompositionHandler attaches a pre-configured CompositionHandler to the
// lighting subsystem, overriding the default that is auto-created by
// NewLightingHandler. The composition handler manages the offscreen HDR render
// target and the full-screen tone mapping pass. GPU resources are initialized
// lazily during the first lighting initialization.
//
// Parameters:
//   - handler: the pre-configured CompositionHandler
//
// Returns:
//   - LightingHandlerOption: a function that applies the composition handler option to a lightingHandlerImpl
func WithCompositionHandler(handler CompositionHandler) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.compositionHandler = handler
	}
}

// WithSSRHandler attaches a pre-configured SSRHandler to the lighting
// subsystem, overriding the default that is auto-created by NewLightingHandler.
// The SSR handler reads the G-Buffer and writes to a texture sampled by the
// composition pass. GPU resources are initialized lazily during the first
// lighting initialization.
//
// Parameters:
//   - handler: the pre-configured SSRHandler
//
// Returns:
//   - LightingHandlerOption: a function that applies the SSR handler option to a lightingHandlerImpl
func WithSSRHandler(handler SSRHandler) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.ssrHandler = handler
	}
}

// WithShadowHandler sets the ShadowHandler for the LightingHandler.
// If not provided, NewLightingHandler creates a default ShadowHandler.
//
// Parameters:
//   - h: the ShadowHandler to use
//
// Returns:
//   - LightingHandlerOption: a function that applies the shadow handler option
func WithShadowHandler(h ShadowHandler) LightingHandlerOption {
	return func(impl *lightingHandlerImpl) {
		impl.shadowHandler = h
	}
}

// WithContactShadowHandler attaches a pre-configured ContactShadowHandler to
// the lighting subsystem, overriding the default that is auto-created by
// NewLightingHandler. GPU resources are initialized lazily during the first
// lighting initialization.
//
// Parameters:
//   - handler: the pre-configured ContactShadowHandler
//
// Returns:
//   - LightingHandlerOption: a function that applies the contact shadow handler option to a lightingHandlerImpl
func WithContactShadowHandler(handler ContactShadowHandler) LightingHandlerOption {
	return func(impl *lightingHandlerImpl) {
		impl.contactShadowHandler = handler
	}
}

// WithTileSize sets the Forward+ tile size in pixels.
//
// Parameters:
//   - size: tile width and height in pixels (default 16)
//
// Returns:
//   - LightingHandlerOption: option function to apply
func WithTileSize(size int) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.tileSize = size
	}
}

// WithMaxLightsPerTile sets the maximum number of light indices stored per tile.
//
// Parameters:
//   - max: maximum lights per tile (default 256)
//
// Returns:
//   - LightingHandlerOption: option function to apply
func WithMaxLightsPerTile(max int) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.maxLightsPerTile = max
	}
}

// WithMaxGPULights sets the maximum number of lights that can be marshaled into
// the GPU storage buffer per frame. The CPU-side light list is unbounded; this
// cap controls only how many lights the GPU evaluates. When the active light
// count exceeds this budget, the scene's light priority system selects the most
// impactful lights.
//
// Parameters:
//   - max: maximum GPU lights (default 1024)
//
// Returns:
//   - LightingHandlerOption: option function to apply
func WithMaxGPULights(max int) LightingHandlerOption {
	return func(h *lightingHandlerImpl) {
		h.maxGPULights = max
	}
}

// NewLightingHandler creates a new LightingHandler with sensible defaults and any
// provided options applied. Pre-creates named BindGroupProviders for each lighting
// subsystem stage. GPU resources are not allocated until the owning scene calls
// the appropriate initialization methods.
//
// Parameters:
//   - opts: variadic list of LightingHandlerOption functions to configure the handler
//
// Returns:
//   - LightingHandler: a new handler instance ready to be attached to a scene
func NewLightingHandler(opts ...LightingHandlerOption) LightingHandler {
	h := &lightingHandlerImpl{
		enabled:          false,
		lights:           make([]Light, 0),
		tileSize:         16,
		maxLightsPerTile: 256,
		maxGPULights:     1024,
		bgps: map[string]bind_group_provider.BindGroupProvider{
			"lights":          bind_group_provider.NewBindGroupProvider("lights"),
			"light_cull":      bind_group_provider.NewBindGroupProvider("light_cull"),
			"tile_lit":        bind_group_provider.NewBindGroupProvider("tile_lit"),
			"ssao_lit":        bind_group_provider.NewBindGroupProvider("ssao_lit"),
			"probe_lit":       bind_group_provider.NewBindGroupProvider("probe_lit"),
			"composition_lit": bind_group_provider.NewBindGroupProvider("composition_lit"),
			"ssr_lit":         bind_group_provider.NewBindGroupProvider("ssr_lit"),
		},
		pipelineKeys: make(map[string]string),
	}
	for _, opt := range opts {
		opt(h)
	}

	// Always create the GI subsystems (GBuffer, SSAO, Composition, SSR) with
	// sensible defaults if they were not explicitly provided via options. The
	// full GI pipeline is mandatory for lit scenes.
	if h.gBufferHandler == nil {
		h.gBufferHandler = NewGBufferHandler()
	}
	if h.ssaoHandler == nil {
		h.ssaoHandler = NewSSAOHandler()
	}
	if h.compositionHandler == nil {
		h.compositionHandler = NewCompositionHandler(
			WithToneMappingEnabled(true),
			WithExposure(1.0),
		)
	}
	if h.ssrHandler == nil {
		h.ssrHandler = NewSSRHandler()
	}
	if h.shadowHandler == nil {
		h.shadowHandler = NewShadowHandler()
	}
	if h.contactShadowHandler == nil {
		h.contactShadowHandler = NewContactShadowHandler()
	}

	return h
}
