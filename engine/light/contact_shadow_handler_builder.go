package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// ContactShadowHandlerOption is a functional option for configuring a
// ContactShadowHandler during construction via NewContactShadowHandler.
type ContactShadowHandlerOption func(*contactShadowHandlerImpl)

// WithContactShadowsEnabled sets whether contact shadows are enabled.
// When false, the contact shadow compute shader is not dispatched and the
// lit shader receives a 1×1 white fallback texture (no contact darkening).
//
// Parameters:
//   - enabled: true to enable contact shadows
//
// Returns:
//   - ContactShadowHandlerOption: a function that applies the enabled option to a contactShadowHandlerImpl
func WithContactShadowsEnabled(enabled bool) ContactShadowHandlerOption {
	return func(h *contactShadowHandlerImpl) {
		h.enabled = enabled
	}
}

// WithContactShadowStepCount sets the number of ray march steps per pixel.
// Higher values produce more accurate contact shadows at greater compute cost.
//
// Parameters:
//   - count: the number of ray march steps (recommended: 16)
//
// Returns:
//   - ContactShadowHandlerOption: a function that applies the step count option to a contactShadowHandlerImpl
func WithContactShadowStepCount(count int) ContactShadowHandlerOption {
	return func(h *contactShadowHandlerImpl) {
		h.stepCount = count
	}
}

// WithContactShadowMaxDistance sets the maximum ray march distance in
// world-space units. Larger distances detect contact shadows further from
// surfaces but may introduce artifacts.
//
// Parameters:
//   - dist: the maximum ray march distance (recommended: 1.0)
//
// Returns:
//   - ContactShadowHandlerOption: a function that applies the max distance option to a contactShadowHandlerImpl
func WithContactShadowMaxDistance(dist float32) ContactShadowHandlerOption {
	return func(h *contactShadowHandlerImpl) {
		h.maxDistance = dist
	}
}

// WithContactShadowThickness sets the depth thickness tolerance for hit
// detection in NDC depth space. Thinner values produce sharper contact
// shadows but may miss thin geometry.
//
// Parameters:
//   - thickness: the depth thickness tolerance (recommended: 0.05)
//
// Returns:
//   - ContactShadowHandlerOption: a function that applies the thickness option to a contactShadowHandlerImpl
func WithContactShadowThickness(thickness float32) ContactShadowHandlerOption {
	return func(h *contactShadowHandlerImpl) {
		h.thickness = thickness
	}
}

// NewContactShadowHandler creates a new ContactShadowHandler with sensible
// defaults and any provided options applied. GPU resources are not allocated
// until the owning scene calls the appropriate initialization methods.
//
// Default values:
//   - enabled: true
//   - StepCount: 16
//   - MaxDistance: 1.0
//   - Thickness: 0.05
//
// Parameters:
//   - opts: variadic list of ContactShadowHandlerOption functions to configure the handler
//
// Returns:
//   - ContactShadowHandler: a new handler instance ready to be attached to a scene
func NewContactShadowHandler(opts ...ContactShadowHandlerOption) ContactShadowHandler {
	h := &contactShadowHandlerImpl{
		enabled:      true,
		stepCount:    16,
		maxDistance:  1.0,
		thickness:    0.05,
		pipelineKeys: make(map[string]string),
		bgps: map[string]bind_group_provider.BindGroupProvider{
			"contact_shadow_compute": bind_group_provider.NewBindGroupProvider("contact_shadow_compute"),
		},
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}
