package gbuffer

import (
	"github.com/cogentcore/webgpu/wgpu"
)

// GBufferHandler defines the interface for the scene's G-Buffer subsystem.
//
// The GBufferHandler manages the multiple render target (MRT) textures that
// store per-pixel geometric and material data written during the G-Buffer
// pre-pass. These textures are consumed by downstream screen-space effect
// passes such as SSAO and SSR. GPU resources are initialized lazily by the
// owning scene when GI features are first enabled.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type GBufferHandler interface {
	// Enabled returns whether the G-Buffer subsystem has been GPU-initialized
	// and is ready for rendering.
	Enabled() bool

	// SetEnabled sets whether the G-Buffer subsystem is GPU-initialized.
	SetEnabled(enabled bool)

	// SetSlot selects the active texture slot.
	SetSlot(slot int)

	// ScreenWidth returns the current screen width in pixels used for texture sizing.
	ScreenWidth() int

	// ScreenHeight returns the current screen height in pixels used for texture sizing.
	ScreenHeight() int

	// PipelineKey retrieves the pipeline key associated with the given name.
	PipelineKey(name string) string

	// PipelineKeys returns the full map of pipeline keys.
	PipelineKeys() map[string]string

	// SetPipelineKey stores a pipeline key under the given name.
	SetPipelineKey(name, key string)

	// NormalTexture returns the RGBA16Float texture storing world-space normals and roughness.
	NormalTexture() *wgpu.Texture

	// SetNormalTexture sets the normal MRT texture.
	SetNormalTexture(t *wgpu.Texture)

	// NormalTextureView returns the texture view for the normal texture.
	NormalTextureView() *wgpu.TextureView

	// SetNormalTextureView sets the texture view for the normal texture.
	SetNormalTextureView(tv *wgpu.TextureView)

	// AlbedoTexture returns the RGBA8Unorm texture storing albedo and metallic.
	AlbedoTexture() *wgpu.Texture

	// SetAlbedoTexture sets the albedo MRT texture.
	SetAlbedoTexture(t *wgpu.Texture)

	// AlbedoTextureView returns the texture view for the albedo texture.
	AlbedoTextureView() *wgpu.TextureView

	// SetAlbedoTextureView sets the texture view for the albedo texture.
	SetAlbedoTextureView(tv *wgpu.TextureView)

	// DepthTexture returns the Depth24Plus texture used for depth testing.
	DepthTexture() *wgpu.Texture

	// SetDepthTexture sets the depth texture for the G-Buffer pass.
	SetDepthTexture(t *wgpu.Texture)

	// DepthTextureView returns the texture view for the G-Buffer depth texture.
	DepthTextureView() *wgpu.TextureView

	// SetDepthTextureView sets the texture view for the G-Buffer depth texture.
	SetDepthTextureView(tv *wgpu.TextureView)

	// Resize updates the screen dimensions for texture sizing.
	Resize(width, height int)
}

var _ GBufferHandler = &gBufferHandlerImpl{}

func (h *gBufferHandlerImpl) Enabled() bool                    { return h.enabled }
func (h *gBufferHandlerImpl) SetEnabled(enabled bool)          { h.enabled = enabled }
func (h *gBufferHandlerImpl) ScreenWidth() int                 { return h.screenWidth }
func (h *gBufferHandlerImpl) ScreenHeight() int                { return h.screenHeight }
func (h *gBufferHandlerImpl) PipelineKey(name string) string   { return h.pipelineKeys[name] }
func (h *gBufferHandlerImpl) PipelineKeys() map[string]string  { return h.pipelineKeys }
func (h *gBufferHandlerImpl) SetPipelineKey(name, key string)  { h.pipelineKeys[name] = key }
func (h *gBufferHandlerImpl) SetSlot(slot int)                 { h.activeSlot = slot }
func (h *gBufferHandlerImpl) NormalTexture() *wgpu.Texture     { return h.normalTextures[h.activeSlot] }
func (h *gBufferHandlerImpl) SetNormalTexture(t *wgpu.Texture) { h.normalTextures[h.activeSlot] = t }
func (h *gBufferHandlerImpl) NormalTextureView() *wgpu.TextureView {
	return h.normalTextureViews[h.activeSlot]
}
func (h *gBufferHandlerImpl) SetNormalTextureView(tv *wgpu.TextureView) {
	h.normalTextureViews[h.activeSlot] = tv
}
func (h *gBufferHandlerImpl) AlbedoTexture() *wgpu.Texture     { return h.albedoTextures[h.activeSlot] }
func (h *gBufferHandlerImpl) SetAlbedoTexture(t *wgpu.Texture) { h.albedoTextures[h.activeSlot] = t }
func (h *gBufferHandlerImpl) AlbedoTextureView() *wgpu.TextureView {
	return h.albedoTextureViews[h.activeSlot]
}
func (h *gBufferHandlerImpl) SetAlbedoTextureView(tv *wgpu.TextureView) {
	h.albedoTextureViews[h.activeSlot] = tv
}
func (h *gBufferHandlerImpl) DepthTexture() *wgpu.Texture     { return h.depthTextures[h.activeSlot] }
func (h *gBufferHandlerImpl) SetDepthTexture(t *wgpu.Texture) { h.depthTextures[h.activeSlot] = t }
func (h *gBufferHandlerImpl) DepthTextureView() *wgpu.TextureView {
	return h.depthTextureViews[h.activeSlot]
}
func (h *gBufferHandlerImpl) SetDepthTextureView(tv *wgpu.TextureView) {
	h.depthTextureViews[h.activeSlot] = tv
}

func (h *gBufferHandlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
}
