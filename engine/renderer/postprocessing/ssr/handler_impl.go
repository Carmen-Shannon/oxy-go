package ssr

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// handlerImpl is the implementation of the Handler interface.
type handlerImpl struct {
	enabled bool

	screenWidth  int
	screenHeight int

	// Ray march quality parameters.
	maxSteps        int
	maxDistance     float32
	thickness       float32
	stride          float32
	roughnessCutoff float32

	pipelineKeys map[string]string
	bgps         map[string]bind_group_provider.BindGroupProvider

	activeSlot int

	// SSR result texture (RGBA16Float, half-resolution).
	// RGB = reflected color, A = confidence/hit mask.
	ssrTextures     [2]*wgpu.Texture
	ssrTextureViews [2]*wgpu.TextureView

	// Linear sampler for the composition shader to sample the SSR result.
	linearSampler *wgpu.Sampler

	// Hi-Z depth pyramid (R32Float, full mip chain, min-depth per cell).
	// Used by the Hi-Z SSR ray march to skip empty screen space in large steps.
	hizTextures        [2]*wgpu.Texture
	hizTextureViews    [2]*wgpu.TextureView   // Full mip chain view for SSR reads.
	hizMipCount        int                    // Number of mip levels in the Hi-Z pyramid.
	hizMipReadViewsArr [2][]*wgpu.TextureView // Per-mip read views for downsample input.
	hizStorageViewsArr [2][]*wgpu.TextureView // Per-mip storage views (write) for downsample output.

	// MAX Hi-Z depth pyramid (R32Float, full mip chain, max-depth per cell).
	// Used by occlusion culling: min_ndc_z > max_hiz_sample means fully occluded.
	maxHizTextures        [2]*wgpu.Texture
	maxHizTextureViews    [2]*wgpu.TextureView
	maxHizMipReadViewsArr [2][]*wgpu.TextureView
	maxHizStorageViewsArr [2][]*wgpu.TextureView
}
