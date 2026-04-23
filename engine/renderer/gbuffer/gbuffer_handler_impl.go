package gbuffer

import "github.com/cogentcore/webgpu/wgpu"

// handlerImpl is the implementation of the Handler interface.
type handlerImpl struct {
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

	// Shared depth texture for the G-Buffer pass.
	depthTextures     [2]*wgpu.Texture
	depthTextureViews [2]*wgpu.TextureView
}
