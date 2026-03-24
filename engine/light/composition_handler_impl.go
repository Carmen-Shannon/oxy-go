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

	toneMappingEnabled     bool
	exposure               float32
	autoExposureEnabled    bool
	adaptSpeed             float32
	minExposure            float32
	maxExposure            float32
	luminanceWorkgroupSize int

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
	bloomDownTexture      *wgpu.Texture
	bloomDownReadViews    []*wgpu.TextureView
	bloomDownStorageViews []*wgpu.TextureView

	// Bloom upsample chain texture and per-mip views. Each mip stores the
	// progressively upsampled and blended bloom result. Mip 0 is the final
	// bloom output sampled by the composition shader.
	bloomUpTexture      *wgpu.Texture
	bloomUpReadViews    []*wgpu.TextureView
	bloomUpStorageViews []*wgpu.TextureView
	bloomUpMip0View     *wgpu.TextureView
}
