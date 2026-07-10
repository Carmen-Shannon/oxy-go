// Package composition provides the postprocessing composition and tone-mapping subsystem.
package composition

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/oliverbestmann/webgpu/wgpu"
)

// Handler defines the interface for the scene's composition and tone mapping subsystem.
//
// The Handler manages the offscreen HDR render target, the full-screen
// composition pipeline, and tone mapping configuration. When composition is active,
// the lit pass renders to an RGBA16Float offscreen texture instead of the swapchain.
// A final full-screen pass then samples the HDR result along with any SSR texture,
// applies ACES tone mapping and gamma correction, and writes to the swapchain.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type Handler interface {
	// Enabled returns whether the composition subsystem has been GPU-initialized
	// and is ready for rendering.
	//
	// Returns:
	//   - bool: true if composition GPU resources have been initialized
	Enabled() bool

	// SetEnabled sets whether the composition subsystem is GPU-initialized.
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

	// ToneMappingEnabled returns whether ACES tone mapping is applied during composition.
	//
	// Returns:
	//   - bool: true if tone mapping is enabled
	ToneMappingEnabled() bool

	// Exposure returns the exposure multiplier applied to HDR values before tone mapping.
	//
	// Returns:
	//   - float32: the current exposure value
	Exposure() float32

	// SetExposure sets the exposure multiplier applied to HDR values before tone mapping.
	//
	// Parameters:
	//   - exposure: the exposure multiplier (1.0 = neutral)
	SetExposure(exposure float32)

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

	// HDRTexture returns the RGBA16Float offscreen render target that the lit
	// pass writes to when composition is active.
	//
	// Returns:
	//   - *wgpu.Texture: the HDR texture, or nil if not initialized
	HDRTexture() *wgpu.Texture

	// SetHDRTexture sets the HDR offscreen render target texture.
	//
	// Parameters:
	//   - t: the HDR texture
	SetHDRTexture(t *wgpu.Texture)

	// HDRTextureView returns the texture view for the HDR render target.
	//
	// Returns:
	//   - *wgpu.TextureView: the HDR texture view, or nil if not initialized
	HDRTextureView() *wgpu.TextureView

	// SetHDRTextureView sets the texture view for the HDR render target.
	//
	// Parameters:
	//   - tv: the HDR texture view
	SetHDRTextureView(tv *wgpu.TextureView)

	// MSAATexture returns the multi-sampled RGBA16Float texture used as the
	// lit pass color attachment when MSAA is enabled. Resolves into the HDR texture.
	//
	// Returns:
	//   - *wgpu.Texture: the MSAA texture, or nil if MSAA is off or not initialized
	MSAATexture() *wgpu.Texture

	// SetMSAATexture sets the MSAA resolve source texture.
	//
	// Parameters:
	//   - t: the MSAA texture
	SetMSAATexture(t *wgpu.Texture)

	// MSAATextureView returns the texture view for the MSAA texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the MSAA texture view, or nil if MSAA is off or not initialized
	MSAATextureView() *wgpu.TextureView

	// SetMSAATextureView sets the texture view for the MSAA texture.
	//
	// Parameters:
	//   - tv: the MSAA texture view
	SetMSAATextureView(tv *wgpu.TextureView)

	// DepthTexture returns the depth texture for the offscreen HDR render pass.
	//
	// Returns:
	//   - *wgpu.Texture: the depth texture, or nil if not initialized
	DepthTexture() *wgpu.Texture

	// SetDepthTexture sets the depth texture for the offscreen HDR render pass.
	//
	// Parameters:
	//   - t: the depth texture
	SetDepthTexture(t *wgpu.Texture)

	// DepthTextureView returns the texture view for the offscreen depth texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the depth texture view, or nil if not initialized
	DepthTextureView() *wgpu.TextureView

	// SetDepthTextureView sets the texture view for the offscreen depth texture.
	//
	// Parameters:
	//   - tv: the depth texture view
	SetDepthTextureView(tv *wgpu.TextureView)

	// LinearSampler returns the linear sampler used for sampling the HDR and SSR
	// textures in the composition fragment shader.
	//
	// Returns:
	//   - *wgpu.Sampler: the linear sampler, or nil if not initialized
	LinearSampler() *wgpu.Sampler

	// SetLinearSampler sets the linear sampler for composition texture sampling.
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

	// AutoExposureEnabled returns whether the eye-adaptation / auto-exposure system
	// is active. When true, the luminance compute shader drives exposure each frame
	// instead of the static Exposure value.
	//
	// Returns:
	//   - bool: true if auto-exposure is enabled
	AutoExposureEnabled() bool

	// SetAutoExposureEnabled enables or disables the auto-exposure system.
	//
	// Parameters:
	//   - enabled: true to enable eye adaptation
	SetAutoExposureEnabled(enabled bool)

	// AdaptSpeed returns the eye-adaptation rate (exposure change per second).
	//
	// Returns:
	//   - float32: the adaptation speed
	AdaptSpeed() float32

	// SetAdaptSpeed sets the eye-adaptation rate (exposure change per second).
	//
	// Parameters:
	//   - speed: the adaptation speed
	SetAdaptSpeed(speed float32)

	// MinExposure returns the lower clamp boundary for auto-exposure.
	//
	// Returns:
	//   - float32: the minimum exposure value
	MinExposure() float32

	// SetMinExposure sets the lower clamp boundary for auto-exposure.
	//
	// Parameters:
	//   - min: the minimum exposure value
	SetMinExposure(min float32)

	// MaxExposure returns the upper clamp boundary for auto-exposure.
	//
	// Returns:
	//   - float32: the maximum exposure value
	MaxExposure() float32

	// SetMaxExposure sets the upper clamp boundary for auto-exposure.
	//
	// Parameters:
	//   - max: the maximum exposure value
	SetMaxExposure(max float32)

	// LuminanceWorkgroupSize returns the tile dimension of the luminance compute
	// shader workgroup. The shader uses a single (size × size) workgroup where each
	// thread samples one texel of the HDR image to compute log-average luminance.
	//
	// Returns:
	//   - int: the workgroup tile dimension (number of threads per axis)
	LuminanceWorkgroupSize() int

	// ExposureBuffer returns the persistent GPU storage buffer holding the current
	// adapted exposure value written each frame by the luminance compute shader.
	//
	// Returns:
	//   - *wgpu.Buffer: the exposure storage buffer, or nil if not initialized
	ExposureBuffer() *wgpu.Buffer

	// SetExposureBuffer stores the persistent GPU exposure storage buffer.
	//
	// Parameters:
	//   - b: the exposure storage buffer
	SetExposureBuffer(b *wgpu.Buffer)

	// BloomEnabled returns whether the bloom post-processing effect is active.
	//
	// Returns:
	//   - bool: true if bloom is enabled
	BloomEnabled() bool

	// SetBloomEnabled enables or disables the bloom post-processing effect.
	//
	// Parameters:
	//   - enabled: true to enable bloom
	SetBloomEnabled(enabled bool)

	// BloomThreshold returns the brightness threshold for bloom extraction.
	// Pixels below this brightness are excluded from the bloom contribution.
	//
	// Returns:
	//   - float32: the brightness threshold
	BloomThreshold() float32

	// SetBloomThreshold sets the brightness threshold for bloom extraction.
	//
	// Parameters:
	//   - threshold: the brightness threshold (typical range 0.5–2.0)
	SetBloomThreshold(threshold float32)

	// BloomIntensity returns the intensity multiplier applied to the bloom
	// contribution when blended into the final composition.
	//
	// Returns:
	//   - float32: the bloom intensity
	BloomIntensity() float32

	// SetBloomIntensity sets the bloom intensity multiplier.
	//
	// Parameters:
	//   - intensity: the bloom intensity (typical range 0.1–1.0)
	SetBloomIntensity(intensity float32)

	// BloomMipCount returns the number of mip levels in the bloom mip chain.
	//
	// Returns:
	//   - int: the mip level count
	BloomMipCount() int

	// SetBloomMipCount stores the number of mip levels in the bloom mip chain.
	//
	// Parameters:
	//   - count: the mip level count
	SetBloomMipCount(count int)

	// BloomDownTexture returns the bloom downsample chain texture.
	//
	// Returns:
	//   - *wgpu.Texture: the downsample chain texture, or nil if not initialized
	BloomDownTexture() *wgpu.Texture

	// SetBloomDownTexture stores the bloom downsample chain texture.
	//
	// Parameters:
	//   - t: the downsample chain texture
	SetBloomDownTexture(t *wgpu.Texture)

	// BloomDownReadViews returns the per-mip read views for the downsample chain.
	//
	// Returns:
	//   - []*wgpu.TextureView: the per-mip read views
	BloomDownReadViews() []*wgpu.TextureView

	// SetBloomDownReadViews stores the per-mip read views for the downsample chain.
	//
	// Parameters:
	//   - views: the per-mip read views
	SetBloomDownReadViews(views []*wgpu.TextureView)

	// BloomDownStorageViews returns the per-mip storage views for the downsample chain.
	//
	// Returns:
	//   - []*wgpu.TextureView: the per-mip storage views
	BloomDownStorageViews() []*wgpu.TextureView

	// SetBloomDownStorageViews stores the per-mip storage views for the downsample chain.
	//
	// Parameters:
	//   - views: the per-mip storage views
	SetBloomDownStorageViews(views []*wgpu.TextureView)

	// BloomUpTexture returns the bloom upsample chain texture.
	//
	// Returns:
	//   - *wgpu.Texture: the upsample chain texture, or nil if not initialized
	BloomUpTexture() *wgpu.Texture

	// SetBloomUpTexture stores the bloom upsample chain texture.
	//
	// Parameters:
	//   - t: the upsample chain texture
	SetBloomUpTexture(t *wgpu.Texture)

	// BloomUpReadViews returns the per-mip read views for the upsample chain.
	//
	// Returns:
	//   - []*wgpu.TextureView: the per-mip read views
	BloomUpReadViews() []*wgpu.TextureView

	// SetBloomUpReadViews stores the per-mip read views for the upsample chain.
	//
	// Parameters:
	//   - views: the per-mip read views
	SetBloomUpReadViews(views []*wgpu.TextureView)

	// BloomUpStorageViews returns the per-mip storage views for the upsample chain.
	//
	// Returns:
	//   - []*wgpu.TextureView: the per-mip storage views
	BloomUpStorageViews() []*wgpu.TextureView

	// SetBloomUpStorageViews stores the per-mip storage views for the upsample chain.
	//
	// Parameters:
	//   - views: the per-mip storage views
	SetBloomUpStorageViews(views []*wgpu.TextureView)

	// BloomUpMip0View returns the mip 0 read view of the upsample chain texture.
	// This is the final bloom result texture view sampled in the composition shader.
	//
	// Returns:
	//   - *wgpu.TextureView: the mip 0 read view
	BloomUpMip0View() *wgpu.TextureView

	// SetBloomUpMip0View stores the mip 0 read view of the upsample chain.
	//
	// Parameters:
	//   - tv: the mip 0 read view
	SetBloomUpMip0View(tv *wgpu.TextureView)
}

var _ Handler = &handlerImpl{}

func (h *handlerImpl) Enabled() bool                   { return h.enabled }
func (h *handlerImpl) SetEnabled(enabled bool)         { h.enabled = enabled }
func (h *handlerImpl) ScreenWidth() int                { return h.screenWidth }
func (h *handlerImpl) ScreenHeight() int               { return h.screenHeight }
func (h *handlerImpl) ToneMappingEnabled() bool        { return h.toneMappingEnabled }
func (h *handlerImpl) Exposure() float32               { return h.exposure }
func (h *handlerImpl) SetExposure(exposure float32)    { h.exposure = exposure }
func (h *handlerImpl) PipelineKey(name string) string  { return h.pipelineKeys[name] }
func (h *handlerImpl) PipelineKeys() map[string]string { return h.pipelineKeys }
func (h *handlerImpl) SetPipelineKey(name, key string) { h.pipelineKeys[name] = key }
func (h *handlerImpl) SetSlot(slot int)                { h.activeSlot = slot }
func (h *handlerImpl) HDRTexture() *wgpu.Texture       { return h.hdrTextures[h.activeSlot] }
func (h *handlerImpl) SetHDRTexture(t *wgpu.Texture)   { h.hdrTextures[h.activeSlot] = t }
func (h *handlerImpl) HDRTextureView() *wgpu.TextureView {
	return h.hdrTextureViews[h.activeSlot]
}
func (h *handlerImpl) SetHDRTextureView(tv *wgpu.TextureView) {
	h.hdrTextureViews[h.activeSlot] = tv
}
func (h *handlerImpl) MSAATexture() *wgpu.Texture     { return h.msaaTextures[h.activeSlot] }
func (h *handlerImpl) SetMSAATexture(t *wgpu.Texture) { h.msaaTextures[h.activeSlot] = t }
func (h *handlerImpl) MSAATextureView() *wgpu.TextureView {
	return h.msaaTextureViews[h.activeSlot]
}
func (h *handlerImpl) SetMSAATextureView(tv *wgpu.TextureView) {
	h.msaaTextureViews[h.activeSlot] = tv
}
func (h *handlerImpl) DepthTexture() *wgpu.Texture     { return h.depthTextures[h.activeSlot] }
func (h *handlerImpl) SetDepthTexture(t *wgpu.Texture) { h.depthTextures[h.activeSlot] = t }
func (h *handlerImpl) DepthTextureView() *wgpu.TextureView {
	return h.depthTextureViews[h.activeSlot]
}
func (h *handlerImpl) SetDepthTextureView(tv *wgpu.TextureView) {
	h.depthTextureViews[h.activeSlot] = tv
}
func (h *handlerImpl) LinearSampler() *wgpu.Sampler     { return h.linearSampler }
func (h *handlerImpl) SetLinearSampler(s *wgpu.Sampler) { h.linearSampler = s }

func (h *handlerImpl) AutoExposureEnabled() bool { return h.autoExposureEnabled }
func (h *handlerImpl) SetAutoExposureEnabled(enabled bool) {
	h.autoExposureEnabled = enabled
}
func (h *handlerImpl) AdaptSpeed() float32              { return h.adaptSpeed }
func (h *handlerImpl) SetAdaptSpeed(speed float32)      { h.adaptSpeed = speed }
func (h *handlerImpl) MinExposure() float32             { return h.minExposure }
func (h *handlerImpl) SetMinExposure(min float32)       { h.minExposure = min }
func (h *handlerImpl) MaxExposure() float32             { return h.maxExposure }
func (h *handlerImpl) SetMaxExposure(max float32)       { h.maxExposure = max }
func (h *handlerImpl) LuminanceWorkgroupSize() int      { return h.luminanceWorkgroupSize }
func (h *handlerImpl) ExposureBuffer() *wgpu.Buffer     { return h.exposureBuffer }
func (h *handlerImpl) SetExposureBuffer(b *wgpu.Buffer) { h.exposureBuffer = b }

func (h *handlerImpl) BloomEnabled() bool                  { return h.bloomEnabled }
func (h *handlerImpl) SetBloomEnabled(enabled bool)        { h.bloomEnabled = enabled }
func (h *handlerImpl) BloomThreshold() float32             { return h.bloomThreshold }
func (h *handlerImpl) SetBloomThreshold(threshold float32) { h.bloomThreshold = threshold }
func (h *handlerImpl) BloomIntensity() float32             { return h.bloomIntensity }
func (h *handlerImpl) SetBloomIntensity(intensity float32) { h.bloomIntensity = intensity }
func (h *handlerImpl) BloomMipCount() int                  { return h.bloomMipCount }
func (h *handlerImpl) SetBloomMipCount(count int)          { h.bloomMipCount = count }
func (h *handlerImpl) BloomDownTexture() *wgpu.Texture {
	return h.bloomDownTextures[h.activeSlot]
}
func (h *handlerImpl) SetBloomDownTexture(t *wgpu.Texture) {
	h.bloomDownTextures[h.activeSlot] = t
}
func (h *handlerImpl) BloomDownReadViews() []*wgpu.TextureView {
	return h.bloomDownReadViewsArr[h.activeSlot]
}
func (h *handlerImpl) SetBloomDownReadViews(views []*wgpu.TextureView) {
	h.bloomDownReadViewsArr[h.activeSlot] = views
}
func (h *handlerImpl) BloomDownStorageViews() []*wgpu.TextureView {
	return h.bloomDownStorageViewsArr[h.activeSlot]
}
func (h *handlerImpl) SetBloomDownStorageViews(views []*wgpu.TextureView) {
	h.bloomDownStorageViewsArr[h.activeSlot] = views
}
func (h *handlerImpl) BloomUpTexture() *wgpu.Texture {
	return h.bloomUpTextures[h.activeSlot]
}
func (h *handlerImpl) SetBloomUpTexture(t *wgpu.Texture) {
	h.bloomUpTextures[h.activeSlot] = t
}
func (h *handlerImpl) BloomUpReadViews() []*wgpu.TextureView {
	return h.bloomUpReadViewsArr[h.activeSlot]
}
func (h *handlerImpl) SetBloomUpReadViews(views []*wgpu.TextureView) {
	h.bloomUpReadViewsArr[h.activeSlot] = views
}
func (h *handlerImpl) BloomUpStorageViews() []*wgpu.TextureView {
	return h.bloomUpStorageViewsArr[h.activeSlot]
}
func (h *handlerImpl) SetBloomUpStorageViews(views []*wgpu.TextureView) {
	h.bloomUpStorageViewsArr[h.activeSlot] = views
}
func (h *handlerImpl) BloomUpMip0View() *wgpu.TextureView {
	return h.bloomUpMip0Views[h.activeSlot]
}
func (h *handlerImpl) SetBloomUpMip0View(tv *wgpu.TextureView) {
	h.bloomUpMip0Views[h.activeSlot] = tv
}

func (h *handlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
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
