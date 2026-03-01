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

// SSRHandler defines the interface for the scene's screen-space reflections subsystem.
//
// The SSRHandler manages the ray march configuration, the SSR result texture, compute
// pipeline keys, and bind group providers needed by the SSR compute shader. It is
// created via NewSSRHandler with builder options and attached to a scene's lighting
// handler. GPU resources are initialized lazily by the owning scene when SSR is first
// enabled.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type SSRHandler interface {
	// Enabled returns whether the SSR subsystem has been GPU-initialized
	// and is ready for rendering.
	//
	// Returns:
	//   - bool: true if SSR GPU resources have been initialized
	Enabled() bool

	// SetEnabled sets whether the SSR subsystem is GPU-initialized.
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

	// MaxSteps returns the maximum number of ray march steps per pixel.
	//
	// Returns:
	//   - int: the max step count
	MaxSteps() int

	// MaxDistance returns the maximum ray march distance in view-space units.
	//
	// Returns:
	//   - float32: the max distance
	MaxDistance() float32

	// Thickness returns the depth thickness tolerance for hit detection.
	//
	// Returns:
	//   - float32: the thickness value
	Thickness() float32

	// Stride returns the step stride multiplier for the ray march.
	//
	// Returns:
	//   - float32: the stride multiplier
	Stride() float32

	// RoughnessCutoff returns the roughness value above which SSR is skipped.
	//
	// Returns:
	//   - float32: the roughness cutoff
	RoughnessCutoff() float32

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

	// Bgp retrieves the bind group provider associated with the given key.
	// Returns nil if the key does not exist.
	//
	// Parameters:
	//   - key: the bind group provider name
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the provider, or nil if not found
	Bgp(key string) bind_group_provider.BindGroupProvider

	// Bgps returns the full map of bind group providers.
	//
	// Returns:
	//   - map[string]bind_group_provider.BindGroupProvider: all registered providers
	Bgps() map[string]bind_group_provider.BindGroupProvider

	// SetBgp stores a bind group provider under the given key.
	//
	// Parameters:
	//   - key: the bind group provider name
	//   - bgp: the bind group provider
	SetBgp(key string, bgp bind_group_provider.BindGroupProvider)

	// SSRTexture returns the RGBA16Float texture storing the SSR ray march result
	// at half resolution. RGB = reflected color, A = confidence.
	//
	// Returns:
	//   - *wgpu.Texture: the SSR result texture, or nil if not initialized
	SSRTexture() *wgpu.Texture

	// SetSSRTexture sets the SSR result texture.
	//
	// Parameters:
	//   - t: the SSR result texture
	SetSSRTexture(t *wgpu.Texture)

	// SSRTextureView returns the texture view for the SSR result texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the SSR texture view, or nil if not initialized
	SSRTextureView() *wgpu.TextureView

	// SetSSRTextureView sets the texture view for the SSR result texture.
	//
	// Parameters:
	//   - tv: the SSR texture view
	SetSSRTextureView(tv *wgpu.TextureView)

	// LinearSampler returns the linear sampler used when the composition shader
	// samples the SSR result texture.
	//
	// Returns:
	//   - *wgpu.Sampler: the linear sampler, or nil if not initialized
	LinearSampler() *wgpu.Sampler

	// SetLinearSampler sets the linear sampler for SSR texture sampling.
	//
	// Parameters:
	//   - s: the linear sampler
	SetLinearSampler(s *wgpu.Sampler)

	// Resize updates the screen dimensions for texture sizing. Existing
	// textures are not automatically recreated — call the appropriate texture
	// creation method if the dimensions change while the handler is enabled.
	//
	// Parameters:
	//   - width: the new screen width in pixels
	//   - height: the new screen height in pixels
	Resize(width, height int)

	// HiZTexture returns the R32Float Hi-Z depth pyramid texture.
	//
	// Returns:
	//   - *wgpu.Texture: the Hi-Z texture, or nil if not initialized
	HiZTexture() *wgpu.Texture

	// SetHiZTexture sets the Hi-Z depth pyramid texture.
	//
	// Parameters:
	//   - t: the Hi-Z texture
	SetHiZTexture(t *wgpu.Texture)

	// HiZTextureView returns the full mip chain texture view for the Hi-Z pyramid,
	// used by the SSR compute shader to sample at arbitrary mip levels.
	//
	// Returns:
	//   - *wgpu.TextureView: the full mip chain view, or nil if not initialized
	HiZTextureView() *wgpu.TextureView

	// SetHiZTextureView sets the full mip chain texture view for the Hi-Z pyramid.
	//
	// Parameters:
	//   - tv: the full mip chain texture view
	SetHiZTextureView(tv *wgpu.TextureView)

	// HiZMipCount returns the number of mip levels in the Hi-Z depth pyramid.
	//
	// Returns:
	//   - int: the mip level count
	HiZMipCount() int

	// SetHiZMipCount sets the number of mip levels in the Hi-Z depth pyramid.
	//
	// Parameters:
	//   - count: the mip level count
	SetHiZMipCount(count int)

	// HiZMipReadViews returns the per-mip-level texture read views used as
	// input for Hi-Z downsample passes.
	//
	// Returns:
	//   - []*wgpu.TextureView: per-mip read views
	HiZMipReadViews() []*wgpu.TextureView

	// SetHiZMipReadViews sets the per-mip-level texture read views.
	//
	// Parameters:
	//   - views: per-mip read views
	SetHiZMipReadViews(views []*wgpu.TextureView)

	// HiZStorageViews returns the per-mip-level storage texture views used as
	// output for Hi-Z downsample passes.
	//
	// Returns:
	//   - []*wgpu.TextureView: per-mip storage write views
	HiZStorageViews() []*wgpu.TextureView

	// SetHiZStorageViews sets the per-mip-level storage texture views.
	//
	// Parameters:
	//   - views: per-mip storage write views
	SetHiZStorageViews(views []*wgpu.TextureView)
}

var _ SSRHandler = &ssrHandlerImpl{}

func (h *ssrHandlerImpl) Enabled() bool {
	return h.enabled
}

func (h *ssrHandlerImpl) SetEnabled(enabled bool) {
	h.enabled = enabled
}

func (h *ssrHandlerImpl) ScreenWidth() int {
	return h.screenWidth
}

func (h *ssrHandlerImpl) ScreenHeight() int {
	return h.screenHeight
}

func (h *ssrHandlerImpl) MaxSteps() int {
	return h.maxSteps
}

func (h *ssrHandlerImpl) MaxDistance() float32 {
	return h.maxDistance
}

func (h *ssrHandlerImpl) Thickness() float32 {
	return h.thickness
}

func (h *ssrHandlerImpl) Stride() float32 {
	return h.stride
}

func (h *ssrHandlerImpl) RoughnessCutoff() float32 {
	return h.roughnessCutoff
}

func (h *ssrHandlerImpl) PipelineKey(name string) string {
	return h.pipelineKeys[name]
}

func (h *ssrHandlerImpl) PipelineKeys() map[string]string {
	return h.pipelineKeys
}

func (h *ssrHandlerImpl) SetPipelineKey(name, key string) {
	h.pipelineKeys[name] = key
}

func (h *ssrHandlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return h.bgps[key]
}

func (h *ssrHandlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}

func (h *ssrHandlerImpl) SetBgp(key string, bgp bind_group_provider.BindGroupProvider) {
	h.bgps[key] = bgp
}

func (h *ssrHandlerImpl) SSRTexture() *wgpu.Texture {
	return h.ssrTexture
}

func (h *ssrHandlerImpl) SetSSRTexture(t *wgpu.Texture) {
	h.ssrTexture = t
}

func (h *ssrHandlerImpl) SSRTextureView() *wgpu.TextureView {
	return h.ssrTextureView
}

func (h *ssrHandlerImpl) SetSSRTextureView(tv *wgpu.TextureView) {
	h.ssrTextureView = tv
}

func (h *ssrHandlerImpl) LinearSampler() *wgpu.Sampler {
	return h.linearSampler
}

func (h *ssrHandlerImpl) SetLinearSampler(s *wgpu.Sampler) {
	h.linearSampler = s
}

func (h *ssrHandlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
}

func (h *ssrHandlerImpl) HiZTexture() *wgpu.Texture {
	return h.hizTexture
}

func (h *ssrHandlerImpl) SetHiZTexture(t *wgpu.Texture) {
	h.hizTexture = t
}

func (h *ssrHandlerImpl) HiZTextureView() *wgpu.TextureView {
	return h.hizTextureView
}

func (h *ssrHandlerImpl) SetHiZTextureView(tv *wgpu.TextureView) {
	h.hizTextureView = tv
}

func (h *ssrHandlerImpl) HiZMipCount() int {
	return h.hizMipCount
}

func (h *ssrHandlerImpl) SetHiZMipCount(count int) {
	h.hizMipCount = count
}

func (h *ssrHandlerImpl) HiZMipReadViews() []*wgpu.TextureView {
	return h.hizMipReadViews
}

func (h *ssrHandlerImpl) SetHiZMipReadViews(views []*wgpu.TextureView) {
	h.hizMipReadViews = views
}

func (h *ssrHandlerImpl) HiZStorageViews() []*wgpu.TextureView {
	return h.hizStorageViews
}

func (h *ssrHandlerImpl) SetHiZStorageViews(views []*wgpu.TextureView) {
	h.hizStorageViews = views
}
