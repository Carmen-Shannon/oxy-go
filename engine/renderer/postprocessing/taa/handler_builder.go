package taa

import "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"

// HandlerOption is a functional option for configuring a Handler.
type HandlerOption func(*handlerImpl)

// WithTAAScreenSize sets the initial screen dimensions for TAA texture allocation.
func WithTAAScreenSize(width, height int) HandlerOption {
	return func(h *handlerImpl) {
		h.screenWidth = width
		h.screenHeight = height
	}
}

// WithTAABlendFactor sets the weight given to the current frame during temporal blending.
// Lower values = more smoothing but more ghosting. Recommended range: 0.05-0.2.
// Default: 0.1.
func WithTAABlendFactor(f float32) HandlerOption {
	return func(h *handlerImpl) { h.blendFactor = f }
}

// WithTAAHistoryRectificationScale sets the diagnostic expansion scale applied to
// the YCoCg history clamp box around the 3x3 neighborhood mean.
// Default: 1.0.
func WithTAAHistoryRectificationScale(scale float32) HandlerOption {
	return func(h *handlerImpl) { h.historyRectificationScale = scale }
}

// WithTAAJitterScale sets the multiplier applied to the Halton jitter offsets.
// Higher values increase sub-pixel jitter amplitude; lower values reduce it.
// Default: 1.0.
func WithTAAJitterScale(scale float32) HandlerOption {
	return func(h *handlerImpl) { h.jitterScale = scale }
}

// NewHandler creates a new Handler with sensible defaults.
// Default blend factor: 0.1 (10% current frame, 90% history).
// Default history rectification scale: 1.0.
func NewHandler(opts ...HandlerOption) Handler {
	h := &handlerImpl{
		enabled:                   false,
		blendFactor:               0.1,
		historyRectificationScale: 1.0,
		rawHistoryOnly:            false,
		jitterScale:               1.0,
		pipelineKeys:              make(map[string]string),
		bgps: map[string]bind_group_provider.BindGroupProvider{
			"taa_resolve_0": bind_group_provider.NewBindGroupProvider("taa_resolve_0"),
			"taa_resolve_1": bind_group_provider.NewBindGroupProvider("taa_resolve_1"),
			"taa_sharpen_0": bind_group_provider.NewBindGroupProvider("taa_sharpen_0"),
			"taa_sharpen_1": bind_group_provider.NewBindGroupProvider("taa_sharpen_1"),
		},
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}
