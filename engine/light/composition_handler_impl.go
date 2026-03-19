package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
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
