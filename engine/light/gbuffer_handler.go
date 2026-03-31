package light

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
	//
	// Returns:
	//   - bool: true if G-Buffer GPU resources have been initialized
	Enabled() bool

	// SetEnabled sets whether the G-Buffer subsystem is GPU-initialized.
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

	// NormalTexture returns the RGBA16Float texture storing world-space
	// normals (XYZ packed to [0,1]) and roughness (W).
	//
	// Returns:
	//   - *wgpu.Texture: the normal texture, or nil if not initialized
	NormalTexture() *wgpu.Texture

	// SetNormalTexture sets the normal MRT texture.
	//
	// Parameters:
	//   - t: the normal texture
	SetNormalTexture(t *wgpu.Texture)

	// NormalTextureView returns the texture view for the normal texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the normal texture view, or nil if not initialized
	NormalTextureView() *wgpu.TextureView

	// SetNormalTextureView sets the texture view for the normal texture.
	//
	// Parameters:
	//   - tv: the normal texture view
	SetNormalTextureView(tv *wgpu.TextureView)

	// AlbedoTexture returns the RGBA8Unorm texture storing albedo (RGB)
	// and metallic (A).
	//
	// Returns:
	//   - *wgpu.Texture: the albedo texture, or nil if not initialized
	AlbedoTexture() *wgpu.Texture

	// SetAlbedoTexture sets the albedo MRT texture.
	//
	// Parameters:
	//   - t: the albedo texture
	SetAlbedoTexture(t *wgpu.Texture)

	// AlbedoTextureView returns the texture view for the albedo texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the albedo texture view, or nil if not initialized
	AlbedoTextureView() *wgpu.TextureView

	// SetAlbedoTextureView sets the texture view for the albedo texture.
	//
	// Parameters:
	//   - tv: the albedo texture view
	SetAlbedoTextureView(tv *wgpu.TextureView)

	// DepthTexture returns the Depth24Plus texture used for depth testing
	// during the G-Buffer pass.
	//
	// Returns:
	//   - *wgpu.Texture: the depth texture, or nil if not initialized
	DepthTexture() *wgpu.Texture

	// SetDepthTexture sets the depth texture for the G-Buffer pass.
	//
	// Parameters:
	//   - t: the depth texture
	SetDepthTexture(t *wgpu.Texture)

	// DepthTextureView returns the texture view for the G-Buffer depth texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the depth texture view, or nil if not initialized
	DepthTextureView() *wgpu.TextureView

	// SetDepthTextureView sets the texture view for the G-Buffer depth texture.
	//
	// Parameters:
	//   - tv: the depth texture view
	SetDepthTextureView(tv *wgpu.TextureView)

	// Resize updates the screen dimensions for texture sizing. Existing
	// textures are not automatically recreated — call the appropriate texture
	// creation method if the dimensions change while the handler is enabled.
	//
	// Parameters:
	//   - width: the new screen width in pixels
	//   - height: the new screen height in pixels
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
