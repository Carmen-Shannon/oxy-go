package light

import "github.com/cogentcore/webgpu/wgpu"

// gBufferHandlerImpl is the implementation of the GBufferHandler interface.
type gBufferHandlerImpl struct {
	enabled bool

	screenWidth  int
	screenHeight int

	pipelineKeys map[string]string

	activeSlot int

	// G-Buffer MRT textures and views.
	normalTextures     [2]*wgpu.Texture
	normalTextureViews [2]*wgpu.TextureView
	albedoTextures     [2]*wgpu.Texture
	albedoTextureViews [2]*wgpu.TextureView

	// Shared depth texture for the G-Buffer pass. When nil, the G-Buffer
	// pass creates its own depth texture; otherwise it reuses the depth
	// texture from the main render pass.
	depthTextures     [2]*wgpu.Texture
	depthTextureViews [2]*wgpu.TextureView
}
