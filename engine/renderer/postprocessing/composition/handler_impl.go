package composition

import (
	"github.com/oliverbestmann/webgpu/wgpu"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// handlerImpl is the implementation of the Handler interface.
type handlerImpl struct {
	enabled bool

	screenWidth  int
	screenHeight int

	toneMappingEnabled     bool
	exposure               float32
	autoExposureEnabled    bool
	adaptSpeed             float32
	minExposure            float32
	maxExposure            float32
	luminanceWorkgroupSize int

	pipelineKeys map[string]string
	bgps         map[string]bind_group_provider.BindGroupProvider

	activeSlot int

	// Offscreen HDR render target (RGBA16Float) that the lit pass writes to
	// instead of the swapchain when composition is active.
	hdrTextures     [2]*wgpu.Texture
	hdrTextureViews [2]*wgpu.TextureView

	// MSAA resolve target for the HDR texture when MSAA is enabled.
	// The lit pass renders to this multi-sampled texture, which resolves
	// into hdrTexture at the end of the render pass.
	msaaTextures     [2]*wgpu.Texture
	msaaTextureViews [2]*wgpu.TextureView

	// Depth texture for the offscreen HDR render pass.
	depthTextures     [2]*wgpu.Texture
	depthTextureViews [2]*wgpu.TextureView

	// Linear sampler for sampling the HDR and SSR textures in the composition shader.
	linearSampler *wgpu.Sampler

	// Persistent GPU storage buffer holding the current adapted exposure value.
	// Written each frame by the luminance compute shader when auto-exposure is active.
	exposureBuffer *wgpu.Buffer

	// Bloom configuration: whether bloom is enabled, threshold for bright pixel
	// extraction, and intensity multiplier for the final composition blend.
	bloomEnabled   bool
	bloomThreshold float32
	bloomIntensity float32

	// Number of mip levels in the bloom mip chain textures.
	bloomMipCount int

	// Bloom downsample chain texture and per-mip views. Each mip stores the
	// progressively downsampled bright-pass result.
	bloomDownTextures        [2]*wgpu.Texture
	bloomDownReadViewsArr    [2][]*wgpu.TextureView
	bloomDownStorageViewsArr [2][]*wgpu.TextureView

	// Bloom upsample chain texture and per-mip views. Each mip stores the
	// progressively upsampled and blended bloom result. Mip 0 is the final
	// bloom output sampled by the composition shader.
	bloomUpTextures        [2]*wgpu.Texture
	bloomUpReadViewsArr    [2][]*wgpu.TextureView
	bloomUpStorageViewsArr [2][]*wgpu.TextureView
	bloomUpMip0Views       [2]*wgpu.TextureView
}
