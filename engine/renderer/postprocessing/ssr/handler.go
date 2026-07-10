// Package ssr provides the screen-space reflections postprocessing subsystem.
package ssr

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/oliverbestmann/webgpu/wgpu"
)

// Handler defines the interface for the scene's screen-space reflections subsystem.
//
// The Handler manages the ray march configuration, the SSR result texture, compute
// pipeline keys, and bind group providers needed by the SSR compute shader. It is
// created via NewHandler with builder options and attached to a scene's lighting
// handler. GPU resources are initialized lazily by the owning scene when SSR is first
// enabled.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type Handler interface {
	// Enabled returns whether the SSR subsystem has been GPU-initialized
	// and is ready for rendering.
	//
	// Returns:
	//   - bool: true if SSR GPU resources have been initialized
	Enabled() bool

	// SetEnabled sets whether the SSR subsystem is GPU-initialized.
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

	// ScreenWidth returns the current screen width in pixels used for texture sizing.
	//
	// Returns:
	//   - int: the screen width
	ScreenWidth() int

	// ScreenHeight returns the current screen height in pixels used for texture sizing.
	//
	// Returns:
	//   - int: the screen height
	ScreenHeight() int

	// MaxSteps returns the maximum number of ray march steps per pixel.
	//
	// Returns:
	//   - int: the max step count
	MaxSteps() int

	// MaxDistance returns the maximum ray march distance in view-space units.
	//
	// Returns:
	//   - float32: the max distance
	MaxDistance() float32

	// Thickness returns the depth thickness tolerance for hit detection.
	//
	// Returns:
	//   - float32: the thickness value
	Thickness() float32

	// Stride returns the step stride multiplier for the ray march.
	//
	// Returns:
	//   - float32: the stride multiplier
	Stride() float32

	// RoughnessCutoff returns the roughness value above which SSR is skipped.
	//
	// Returns:
	//   - float32: the roughness cutoff
	RoughnessCutoff() float32

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

	// SSRTexture returns the RGBA16Float texture storing the SSR ray march result
	// at half resolution. RGB = reflected color, A = confidence.
	//
	// Returns:
	//   - *wgpu.Texture: the SSR result texture, or nil if not initialized
	SSRTexture() *wgpu.Texture

	// SetSSRTexture sets the SSR result texture.
	//
	// Parameters:
	//   - t: the SSR result texture
	SetSSRTexture(t *wgpu.Texture)

	// SSRTextureView returns the texture view for the SSR result texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the SSR texture view, or nil if not initialized
	SSRTextureView() *wgpu.TextureView

	// SetSSRTextureView sets the texture view for the SSR result texture.
	//
	// Parameters:
	//   - tv: the SSR texture view
	SetSSRTextureView(tv *wgpu.TextureView)

	// LinearSampler returns the linear sampler used when the composition shader
	// samples the SSR result texture.
	//
	// Returns:
	//   - *wgpu.Sampler: the linear sampler, or nil if not initialized
	LinearSampler() *wgpu.Sampler

	// SetLinearSampler sets the linear sampler for SSR texture sampling.
	//
	// Parameters:
	//   - s: the linear sampler
	SetLinearSampler(s *wgpu.Sampler)

	// Resize updates the screen dimensions for texture sizing. Existing
	// textures are not automatically recreated — call the appropriate texture
	// creation method if the dimensions change while the handler is enabled.
	//
	// Parameters:
	//   - width: the new screen width in pixels
	//   - height: the new screen height in pixels
	Resize(width, height int)

	// HiZTexture returns the R32Float Hi-Z depth pyramid texture.
	//
	// Returns:
	//   - *wgpu.Texture: the Hi-Z texture, or nil if not initialized
	HiZTexture() *wgpu.Texture

	// SetHiZTexture sets the Hi-Z depth pyramid texture.
	//
	// Parameters:
	//   - t: the Hi-Z texture
	SetHiZTexture(t *wgpu.Texture)

	// HiZTextureView returns the full mip chain texture view for the Hi-Z pyramid,
	// used by the SSR compute shader to sample at arbitrary mip levels.
	//
	// Returns:
	//   - *wgpu.TextureView: the full mip chain view, or nil if not initialized
	HiZTextureView() *wgpu.TextureView

	// SetHiZTextureView sets the full mip chain texture view for the Hi-Z pyramid.
	//
	// Parameters:
	//   - tv: the full mip chain texture view
	SetHiZTextureView(tv *wgpu.TextureView)

	// HiZMipCount returns the number of mip levels in the Hi-Z depth pyramid.
	//
	// Returns:
	//   - int: the mip level count
	HiZMipCount() int

	// SetHiZMipCount sets the number of mip levels in the Hi-Z depth pyramid.
	//
	// Parameters:
	//   - count: the mip level count
	SetHiZMipCount(count int)

	// HiZMipReadViews returns the per-mip-level texture read views used as
	// input for Hi-Z downsample passes.
	//
	// Returns:
	//   - []*wgpu.TextureView: per-mip read views
	HiZMipReadViews() []*wgpu.TextureView

	// SetHiZMipReadViews sets the per-mip-level texture read views.
	//
	// Parameters:
	//   - views: per-mip read views
	SetHiZMipReadViews(views []*wgpu.TextureView)

	// HiZStorageViews returns the per-mip-level storage texture views used as
	// output for Hi-Z downsample passes.
	//
	// Returns:
	//   - []*wgpu.TextureView: per-mip storage write views
	HiZStorageViews() []*wgpu.TextureView

	// SetHiZStorageViews sets the per-mip-level storage texture views.
	//
	// Parameters:
	//   - views: per-mip storage write views
	SetHiZStorageViews(views []*wgpu.TextureView)

	// HiZMaxTexture returns the R32Float MAX Hi-Z depth pyramid texture used for occlusion culling.
	//
	// Returns:
	//   - *wgpu.Texture: the MAX Hi-Z texture, or nil if not initialized
	HiZMaxTexture() *wgpu.Texture

	// SetHiZMaxTexture sets the MAX Hi-Z depth pyramid texture.
	//
	// Parameters:
	//   - t: the MAX Hi-Z texture
	SetHiZMaxTexture(t *wgpu.Texture)

	// HiZMaxTextureView returns the full mip chain texture view for the MAX Hi-Z pyramid.
	//
	// Returns:
	//   - *wgpu.TextureView: the full mip chain view, or nil if not initialized
	HiZMaxTextureView() *wgpu.TextureView

	// SetHiZMaxTextureView sets the full mip chain texture view for the MAX Hi-Z pyramid.
	//
	// Parameters:
	//   - tv: the full mip chain texture view
	SetHiZMaxTextureView(tv *wgpu.TextureView)

	// HiZMaxMipReadViews returns the per-mip-level texture read views for MAX Hi-Z downsample passes.
	//
	// Returns:
	//   - []*wgpu.TextureView: per-mip read views
	HiZMaxMipReadViews() []*wgpu.TextureView

	// SetHiZMaxMipReadViews sets the per-mip-level texture read views for the MAX Hi-Z pyramid.
	//
	// Parameters:
	//   - views: per-mip read views
	SetHiZMaxMipReadViews(views []*wgpu.TextureView)

	// HiZMaxStorageViews returns the per-mip-level storage texture views for MAX Hi-Z downsample passes.
	//
	// Returns:
	//   - []*wgpu.TextureView: per-mip storage write views
	HiZMaxStorageViews() []*wgpu.TextureView

	// SetHiZMaxStorageViews sets the per-mip-level storage texture views for the MAX Hi-Z pyramid.
	//
	// Parameters:
	//   - views: per-mip storage write views
	SetHiZMaxStorageViews(views []*wgpu.TextureView)
}

var _ Handler = &handlerImpl{}

func (h *handlerImpl) Enabled() bool                     { return h.enabled }
func (h *handlerImpl) SetEnabled(enabled bool)           { h.enabled = enabled }
func (h *handlerImpl) ScreenWidth() int                  { return h.screenWidth }
func (h *handlerImpl) ScreenHeight() int                 { return h.screenHeight }
func (h *handlerImpl) MaxSteps() int                     { return h.maxSteps }
func (h *handlerImpl) MaxDistance() float32              { return h.maxDistance }
func (h *handlerImpl) Thickness() float32                { return h.thickness }
func (h *handlerImpl) Stride() float32                   { return h.stride }
func (h *handlerImpl) RoughnessCutoff() float32          { return h.roughnessCutoff }
func (h *handlerImpl) PipelineKey(name string) string    { return h.pipelineKeys[name] }
func (h *handlerImpl) PipelineKeys() map[string]string   { return h.pipelineKeys }
func (h *handlerImpl) SetPipelineKey(name, key string)   { h.pipelineKeys[name] = key }
func (h *handlerImpl) SetSlot(slot int)                  { h.activeSlot = slot }
func (h *handlerImpl) SSRTexture() *wgpu.Texture         { return h.ssrTextures[h.activeSlot] }
func (h *handlerImpl) SetSSRTexture(t *wgpu.Texture)     { h.ssrTextures[h.activeSlot] = t }
func (h *handlerImpl) SSRTextureView() *wgpu.TextureView { return h.ssrTextureViews[h.activeSlot] }
func (h *handlerImpl) SetSSRTextureView(tv *wgpu.TextureView) {
	h.ssrTextureViews[h.activeSlot] = tv
}
func (h *handlerImpl) LinearSampler() *wgpu.Sampler      { return h.linearSampler }
func (h *handlerImpl) SetLinearSampler(s *wgpu.Sampler)  { h.linearSampler = s }
func (h *handlerImpl) HiZTexture() *wgpu.Texture         { return h.hizTextures[h.activeSlot] }
func (h *handlerImpl) SetHiZTexture(t *wgpu.Texture)     { h.hizTextures[h.activeSlot] = t }
func (h *handlerImpl) HiZTextureView() *wgpu.TextureView { return h.hizTextureViews[h.activeSlot] }
func (h *handlerImpl) SetHiZTextureView(tv *wgpu.TextureView) {
	h.hizTextureViews[h.activeSlot] = tv
}
func (h *handlerImpl) HiZMipCount() int         { return h.hizMipCount }
func (h *handlerImpl) SetHiZMipCount(count int) { h.hizMipCount = count }
func (h *handlerImpl) HiZMipReadViews() []*wgpu.TextureView {
	return h.hizMipReadViewsArr[h.activeSlot]
}
func (h *handlerImpl) SetHiZMipReadViews(views []*wgpu.TextureView) {
	h.hizMipReadViewsArr[h.activeSlot] = views
}
func (h *handlerImpl) HiZStorageViews() []*wgpu.TextureView {
	return h.hizStorageViewsArr[h.activeSlot]
}
func (h *handlerImpl) SetHiZStorageViews(views []*wgpu.TextureView) {
	h.hizStorageViewsArr[h.activeSlot] = views
}
func (h *handlerImpl) HiZMaxTexture() *wgpu.Texture     { return h.maxHizTextures[h.activeSlot] }
func (h *handlerImpl) SetHiZMaxTexture(t *wgpu.Texture) { h.maxHizTextures[h.activeSlot] = t }
func (h *handlerImpl) HiZMaxTextureView() *wgpu.TextureView {
	return h.maxHizTextureViews[h.activeSlot]
}
func (h *handlerImpl) SetHiZMaxTextureView(tv *wgpu.TextureView) {
	h.maxHizTextureViews[h.activeSlot] = tv
}
func (h *handlerImpl) HiZMaxMipReadViews() []*wgpu.TextureView {
	return h.maxHizMipReadViewsArr[h.activeSlot]
}
func (h *handlerImpl) SetHiZMaxMipReadViews(views []*wgpu.TextureView) {
	h.maxHizMipReadViewsArr[h.activeSlot] = views
}
func (h *handlerImpl) HiZMaxStorageViews() []*wgpu.TextureView {
	return h.maxHizStorageViewsArr[h.activeSlot]
}
func (h *handlerImpl) SetHiZMaxStorageViews(views []*wgpu.TextureView) {
	h.maxHizStorageViewsArr[h.activeSlot] = views
}

func (h *handlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return h.bgps[key]
}

func (h *handlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}

func (h *handlerImpl) SetBgp(key string, bgp bind_group_provider.BindGroupProvider) {
	h.bgps[key] = bgp
}

func (h *handlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
}
