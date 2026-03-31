package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// ContactShadowHandler defines the interface for the scene's contact shadow
// subsystem.
//
// The ContactShadowHandler manages the ray march configuration, the contact
// shadow output texture, compute pipeline keys, and bind group providers
// needed by the contact shadow compute shader. It is created via
// NewContactShadowHandler with builder options and attached to a scene's
// lighting handler. GPU resources are initialized lazily by the owning scene
// when contact shadows are first enabled.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type ContactShadowHandler interface {
	// Enabled returns whether the contact shadow subsystem has been
	// GPU-initialized and is ready for rendering.
	//
	// Returns:
	//   - bool: true if contact shadow GPU resources have been initialized
	Enabled() bool

	// SetEnabled sets whether the contact shadow subsystem is GPU-initialized.
	//
	// Parameters:
	//   - enabled: true to mark as initialized
	SetEnabled(enabled bool)

	// SetSlot selects the active texture slot. Texture and view getters and
	// setters read and write the [slot] index of the underlying arrays.
	//
	// Parameters:
	//   - slot: the slot index (0 or 1)
	SetSlot(slot int)

	// StepCount returns the number of ray march steps per pixel.
	//
	// Returns:
	//   - int: the step count
	StepCount() int

	// MaxDistance returns the maximum ray march distance in world-space units.
	//
	// Returns:
	//   - float32: the max distance
	MaxDistance() float32

	// Thickness returns the depth thickness tolerance for hit detection in
	// NDC depth space.
	//
	// Returns:
	//   - float32: the thickness value
	Thickness() float32

	// PipelineKey retrieves the pipeline key associated with the given name.
	// Returns an empty string if the name does not exist.
	//
	// Parameters:
	//   - name: the pipeline name
	//
	// Returns:
	//   - string: the pipeline key, or empty if not found
	PipelineKey(name string) string

	// PipelineKeys returns the full map of pipeline keys.
	//
	// Returns:
	//   - map[string]string: all registered pipeline name-to-key mappings
	PipelineKeys() map[string]string

	// SetPipelineKey stores a pipeline key under the given name.
	//
	// Parameters:
	//   - name: the pipeline name
	//   - key: the pipeline key
	SetPipelineKey(name, key string)

	// Bgp retrieves the bind group provider associated with the given key.
	// Returns nil if the key does not exist.
	//
	// Parameters:
	//   - key: the bind group provider name
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the provider, or nil if not found
	Bgp(key string) bind_group_provider.BindGroupProvider

	// Bgps returns the full map of bind group providers.
	//
	// Returns:
	//   - map[string]bind_group_provider.BindGroupProvider: all registered providers
	Bgps() map[string]bind_group_provider.BindGroupProvider

	// SetBgp stores a bind group provider under the given key.
	//
	// Parameters:
	//   - key: the bind group provider name
	//   - bgp: the bind group provider
	SetBgp(key string, bgp bind_group_provider.BindGroupProvider)

	// Texture returns the R32Float texture storing the contact shadow output.
	//
	// Returns:
	//   - *wgpu.Texture: the contact shadow texture, or nil if not initialized
	Texture() *wgpu.Texture

	// SetTexture sets the contact shadow output texture.
	//
	// Parameters:
	//   - t: the contact shadow texture
	SetTexture(t *wgpu.Texture)

	// TextureView returns the texture view for the contact shadow output texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the contact shadow texture view, or nil if not initialized
	TextureView() *wgpu.TextureView

	// SetTextureView sets the texture view for the contact shadow output texture.
	//
	// Parameters:
	//   - tv: the contact shadow texture view
	SetTextureView(tv *wgpu.TextureView)

	// LinearSampler returns the linear sampler used when the lit shader samples
	// the contact shadow texture.
	//
	// Returns:
	//   - *wgpu.Sampler: the linear sampler, or nil if not initialized
	LinearSampler() *wgpu.Sampler

	// SetLinearSampler sets the linear sampler for contact shadow texture sampling.
	//
	// Parameters:
	//   - s: the linear sampler
	SetLinearSampler(s *wgpu.Sampler)
}

var _ ContactShadowHandler = &contactShadowHandlerImpl{}

func (h *contactShadowHandlerImpl) Enabled() bool                   { return h.enabled }
func (h *contactShadowHandlerImpl) SetEnabled(enabled bool)         { h.enabled = enabled }
func (h *contactShadowHandlerImpl) StepCount() int                  { return h.stepCount }
func (h *contactShadowHandlerImpl) MaxDistance() float32            { return h.maxDistance }
func (h *contactShadowHandlerImpl) Thickness() float32              { return h.thickness }
func (h *contactShadowHandlerImpl) PipelineKey(name string) string  { return h.pipelineKeys[name] }
func (h *contactShadowHandlerImpl) PipelineKeys() map[string]string { return h.pipelineKeys }
func (h *contactShadowHandlerImpl) SetPipelineKey(name, key string) { h.pipelineKeys[name] = key }
func (h *contactShadowHandlerImpl) SetSlot(slot int)                { h.activeSlot = slot }
func (h *contactShadowHandlerImpl) Texture() *wgpu.Texture          { return h.textures[h.activeSlot] }
func (h *contactShadowHandlerImpl) SetTexture(t *wgpu.Texture)      { h.textures[h.activeSlot] = t }
func (h *contactShadowHandlerImpl) TextureView() *wgpu.TextureView {
	return h.textureViews[h.activeSlot]
}
func (h *contactShadowHandlerImpl) SetTextureView(tv *wgpu.TextureView) {
	h.textureViews[h.activeSlot] = tv
}
func (h *contactShadowHandlerImpl) LinearSampler() *wgpu.Sampler     { return h.linearSampler }
func (h *contactShadowHandlerImpl) SetLinearSampler(s *wgpu.Sampler) { h.linearSampler = s }

func (h *contactShadowHandlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return h.bgps[key]
}

func (h *contactShadowHandlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}

func (h *contactShadowHandlerImpl) SetBgp(key string, bgp bind_group_provider.BindGroupProvider) {
	h.bgps[key] = bgp
}
