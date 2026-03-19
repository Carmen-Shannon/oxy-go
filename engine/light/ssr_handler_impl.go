package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// ssrHandlerImpl is the implementation of the SSRHandler interface.
type ssrHandlerImpl struct {
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

	// SSR result texture (RGBA16Float, half-resolution).
	// RGB = reflected color, A = confidence/hit mask.
	ssrTexture     *wgpu.Texture
	ssrTextureView *wgpu.TextureView

	// Linear sampler for the composition shader to sample the SSR result.
	linearSampler *wgpu.Sampler

	// Hi-Z depth pyramid (R32Float, full mip chain, min-depth per cell).
	// Used by the Hi-Z SSR ray march to skip empty screen space in large steps.
	hizTexture      *wgpu.Texture
	hizTextureView  *wgpu.TextureView   // Full mip chain view for SSR reads.
	hizMipCount     int                 // Number of mip levels in the Hi-Z pyramid.
	hizMipReadViews []*wgpu.TextureView // Per-mip read views for downsample input.
	hizStorageViews []*wgpu.TextureView // Per-mip storage views (write) for downsample output.
}
