package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/gogpu/wgpu"
)

// compositionHandlerImpl is the implementation of the CompositionHandler interface.
type compositionHandlerImpl struct {
	enabled bool

	screenWidth  int
	screenHeight int

	toneMappingEnabled bool
	exposure           float32

	pipelineKeys map[string]string
	bgps         map[string]bind_group_provider.BindGroupProvider

	// Offscreen HDR render target (RGBA16Float) that the lit pass writes to
	// instead of the swapchain when composition is active.
	hdrTexture     *wgpu.Texture
	hdrTextureView *wgpu.TextureView

	// MSAA resolve target for the HDR texture when MSAA is enabled.
	// The lit pass renders to this multi-sampled texture, which resolves
	// into hdrTexture at the end of the render pass.
	msaaTexture     *wgpu.Texture
	msaaTextureView *wgpu.TextureView

	// Depth texture for the offscreen HDR render pass.
	depthTexture     *wgpu.Texture
	depthTextureView *wgpu.TextureView

	// Linear sampler for sampling the HDR and SSR textures in the composition shader.
	linearSampler *wgpu.Sampler
}

// CompositionHandler defines the interface for the scene's composition and tone mapping subsystem.
//
// The CompositionHandler manages the offscreen HDR render target, the full-screen
// composition pipeline, and tone mapping configuration. When composition is active,
// the lit pass renders to an RGBA16Float offscreen texture instead of the swapchain.
// A final full-screen pass then samples the HDR result along with any SSR texture,
// applies ACES tone mapping and gamma correction, and writes to the swapchain.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type CompositionHandler interface {
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
}

var _ CompositionHandler = &compositionHandlerImpl{}

func (h *compositionHandlerImpl) Enabled() bool {
	return h.enabled
}

func (h *compositionHandlerImpl) SetEnabled(enabled bool) {
	h.enabled = enabled
}

func (h *compositionHandlerImpl) ScreenWidth() int {
	return h.screenWidth
}

func (h *compositionHandlerImpl) ScreenHeight() int {
	return h.screenHeight
}

func (h *compositionHandlerImpl) ToneMappingEnabled() bool {
	return h.toneMappingEnabled
}

func (h *compositionHandlerImpl) Exposure() float32 {
	return h.exposure
}

func (h *compositionHandlerImpl) SetExposure(exposure float32) {
	h.exposure = exposure
}

func (h *compositionHandlerImpl) PipelineKey(name string) string {
	return h.pipelineKeys[name]
}

func (h *compositionHandlerImpl) PipelineKeys() map[string]string {
	return h.pipelineKeys
}

func (h *compositionHandlerImpl) SetPipelineKey(name, key string) {
	h.pipelineKeys[name] = key
}

func (h *compositionHandlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return h.bgps[key]
}

func (h *compositionHandlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}

func (h *compositionHandlerImpl) SetBgp(key string, bgp bind_group_provider.BindGroupProvider) {
	h.bgps[key] = bgp
}

func (h *compositionHandlerImpl) HDRTexture() *wgpu.Texture {
	return h.hdrTexture
}

func (h *compositionHandlerImpl) SetHDRTexture(t *wgpu.Texture) {
	h.hdrTexture = t
}

func (h *compositionHandlerImpl) HDRTextureView() *wgpu.TextureView {
	return h.hdrTextureView
}

func (h *compositionHandlerImpl) SetHDRTextureView(tv *wgpu.TextureView) {
	h.hdrTextureView = tv
}

func (h *compositionHandlerImpl) MSAATexture() *wgpu.Texture {
	return h.msaaTexture
}

func (h *compositionHandlerImpl) SetMSAATexture(t *wgpu.Texture) {
	h.msaaTexture = t
}

func (h *compositionHandlerImpl) MSAATextureView() *wgpu.TextureView {
	return h.msaaTextureView
}

func (h *compositionHandlerImpl) SetMSAATextureView(tv *wgpu.TextureView) {
	h.msaaTextureView = tv
}

func (h *compositionHandlerImpl) DepthTexture() *wgpu.Texture {
	return h.depthTexture
}

func (h *compositionHandlerImpl) SetDepthTexture(t *wgpu.Texture) {
	h.depthTexture = t
}

func (h *compositionHandlerImpl) DepthTextureView() *wgpu.TextureView {
	return h.depthTextureView
}

func (h *compositionHandlerImpl) SetDepthTextureView(tv *wgpu.TextureView) {
	h.depthTextureView = tv
}

func (h *compositionHandlerImpl) LinearSampler() *wgpu.Sampler {
	return h.linearSampler
}

func (h *compositionHandlerImpl) SetLinearSampler(s *wgpu.Sampler) {
	h.linearSampler = s
}

func (h *compositionHandlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
}
