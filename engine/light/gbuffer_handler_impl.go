package light

import "github.com/cogentcore/webgpu/wgpu"

// gBufferHandlerImpl is the implementation of the GBufferHandler interface.
type gBufferHandlerImpl struct {
	enabled bool

	screenWidth  int
	screenHeight int

	pipelineKeys map[string]string

	// G-Buffer MRT textures and views.
	normalTexture     *wgpu.Texture
	normalTextureView *wgpu.TextureView
	albedoTexture     *wgpu.Texture
	albedoTextureView *wgpu.TextureView

	// Shared depth texture for the G-Buffer pass. When nil, the G-Buffer
	// pass creates its own depth texture; otherwise it reuses the depth
	// texture from the main render pass.
	depthTexture     *wgpu.Texture
	depthTextureView *wgpu.TextureView
}
