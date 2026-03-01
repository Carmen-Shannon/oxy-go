package light

import (
	"github.com/cogentcore/webgpu/wgpu"
)

// gBufferHandlerImpl is the implementation of the GBufferHandler interface.
type gBufferHandlerImpl struct {
	enabled bool

	screenWidth  int
	screenHeight int

	pipelineKeys map[string]string

	// G-Buffer MRT textures and views.
	positionTexture     *wgpu.Texture
	positionTextureView *wgpu.TextureView
	normalTexture       *wgpu.Texture
	normalTextureView   *wgpu.TextureView
	albedoTexture       *wgpu.Texture
	albedoTextureView   *wgpu.TextureView

	// Shared depth texture for the G-Buffer pass. When nil, the G-Buffer
	// pass creates its own depth texture; otherwise it reuses the depth
	// texture from the main render pass.
	depthTexture     *wgpu.Texture
	depthTextureView *wgpu.TextureView
}

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

	// PositionTexture returns the RGBA16Float texture storing world-space
	// position (XYZ) and linear depth (W).
	//
	// Returns:
	//   - *wgpu.Texture: the position texture, or nil if not initialized
	PositionTexture() *wgpu.Texture

	// SetPositionTexture sets the position MRT texture.
	//
	// Parameters:
	//   - t: the position texture
	SetPositionTexture(t *wgpu.Texture)

	// PositionTextureView returns the texture view for the position texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the position texture view, or nil if not initialized
	PositionTextureView() *wgpu.TextureView

	// SetPositionTextureView sets the texture view for the position texture.
	//
	// Parameters:
	//   - tv: the position texture view
	SetPositionTextureView(tv *wgpu.TextureView)

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

func (h *gBufferHandlerImpl) Enabled() bool {
	return h.enabled
}

func (h *gBufferHandlerImpl) SetEnabled(enabled bool) {
	h.enabled = enabled
}

func (h *gBufferHandlerImpl) ScreenWidth() int {
	return h.screenWidth
}

func (h *gBufferHandlerImpl) ScreenHeight() int {
	return h.screenHeight
}

func (h *gBufferHandlerImpl) PipelineKey(name string) string {
	return h.pipelineKeys[name]
}

func (h *gBufferHandlerImpl) PipelineKeys() map[string]string {
	return h.pipelineKeys
}

func (h *gBufferHandlerImpl) SetPipelineKey(name, key string) {
	h.pipelineKeys[name] = key
}

func (h *gBufferHandlerImpl) PositionTexture() *wgpu.Texture {
	return h.positionTexture
}

func (h *gBufferHandlerImpl) SetPositionTexture(t *wgpu.Texture) {
	h.positionTexture = t
}

func (h *gBufferHandlerImpl) PositionTextureView() *wgpu.TextureView {
	return h.positionTextureView
}

func (h *gBufferHandlerImpl) SetPositionTextureView(tv *wgpu.TextureView) {
	h.positionTextureView = tv
}

func (h *gBufferHandlerImpl) NormalTexture() *wgpu.Texture {
	return h.normalTexture
}

func (h *gBufferHandlerImpl) SetNormalTexture(t *wgpu.Texture) {
	h.normalTexture = t
}

func (h *gBufferHandlerImpl) NormalTextureView() *wgpu.TextureView {
	return h.normalTextureView
}

func (h *gBufferHandlerImpl) SetNormalTextureView(tv *wgpu.TextureView) {
	h.normalTextureView = tv
}

func (h *gBufferHandlerImpl) AlbedoTexture() *wgpu.Texture {
	return h.albedoTexture
}

func (h *gBufferHandlerImpl) SetAlbedoTexture(t *wgpu.Texture) {
	h.albedoTexture = t
}

func (h *gBufferHandlerImpl) AlbedoTextureView() *wgpu.TextureView {
	return h.albedoTextureView
}

func (h *gBufferHandlerImpl) SetAlbedoTextureView(tv *wgpu.TextureView) {
	h.albedoTextureView = tv
}

func (h *gBufferHandlerImpl) DepthTexture() *wgpu.Texture {
	return h.depthTexture
}

func (h *gBufferHandlerImpl) SetDepthTexture(t *wgpu.Texture) {
	h.depthTexture = t
}

func (h *gBufferHandlerImpl) DepthTextureView() *wgpu.TextureView {
	return h.depthTextureView
}

func (h *gBufferHandlerImpl) SetDepthTextureView(tv *wgpu.TextureView) {
	h.depthTextureView = tv
}

func (h *gBufferHandlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
}
