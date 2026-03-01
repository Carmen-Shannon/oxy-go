package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// ssaoHandlerImpl is the implementation of the SSAOHandler interface.
type ssaoHandlerImpl struct {
	enabled bool

	screenWidth  int
	screenHeight int

	// Quality parameters.
	sampleCount    int
	radius         float32
	bias           float32
	power          float32
	blurRadius     int
	halfResolution bool

	pipelineKeys map[string]string
	bgps         map[string]bind_group_provider.BindGroupProvider

	// Raw SSAO output (R8Unorm, screen resolution).
	rawTexture     *wgpu.Texture
	rawTextureView *wgpu.TextureView

	// Blurred SSAO output (R8Unorm, screen resolution).
	blurredTexture     *wgpu.Texture
	blurredTextureView *wgpu.TextureView

	// Intermediate scratch texture for the separable bilateral blur
	// (horizontal pass writes here, vertical pass reads from here).
	scratchTexture     *wgpu.Texture
	scratchTextureView *wgpu.TextureView

	// 4×4 noise texture (RGBA16Float) for kernel rotation.
	noiseTexture     *wgpu.Texture
	noiseTextureView *wgpu.TextureView

	// Linear sampler used for the final SSAO texture bound to the lit shader.
	linearSampler *wgpu.Sampler
}

// SSAOHandler defines the interface for the scene's SSAO subsystem.
//
// The SSAOHandler manages the hemisphere sample kernel, noise texture, raw and
// blurred occlusion textures, compute pipeline keys, and bind group providers
// needed by the SSAO compute and bilateral blur shaders. It is created via
// NewSSAOHandler with builder options and attached to a scene via
// the GI configuration. GPU resources are initialized lazily by the owning
// scene when SSAO is first enabled.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type SSAOHandler interface {
	// Enabled returns whether the SSAO subsystem has been GPU-initialized
	// and is ready for rendering.
	//
	// Returns:
	//   - bool: true if SSAO GPU resources have been initialized
	Enabled() bool

	// SetEnabled sets whether the SSAO subsystem is GPU-initialized.
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

	// SampleCount returns the number of hemisphere samples per pixel.
	//
	// Returns:
	//   - int: the sample count (1–32)
	SampleCount() int

	// Radius returns the hemisphere sampling radius in world-space units.
	//
	// Returns:
	//   - float32: the sampling radius
	Radius() float32

	// Bias returns the depth comparison bias used to prevent self-occlusion.
	//
	// Returns:
	//   - float32: the depth bias
	Bias() float32

	// Power returns the exponent applied to the final AO value.
	//
	// Returns:
	//   - float32: the power exponent
	Power() float32

	// BlurRadius returns the half-width of the bilateral blur kernel in texels.
	//
	// Returns:
	//   - int: the blur kernel half-width
	BlurRadius() int

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

	// RawTexture returns the R8Unorm texture storing the raw (pre-blur) SSAO
	// occlusion values.
	//
	// Returns:
	//   - *wgpu.Texture: the raw SSAO texture, or nil if not initialized
	RawTexture() *wgpu.Texture

	// SetRawTexture sets the raw SSAO texture.
	//
	// Parameters:
	//   - t: the raw SSAO texture
	SetRawTexture(t *wgpu.Texture)

	// RawTextureView returns the texture view for the raw SSAO texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the raw SSAO texture view, or nil if not initialized
	RawTextureView() *wgpu.TextureView

	// SetRawTextureView sets the texture view for the raw SSAO texture.
	//
	// Parameters:
	//   - tv: the raw SSAO texture view
	SetRawTextureView(tv *wgpu.TextureView)

	// BlurredTexture returns the R8Unorm texture storing the blurred (final)
	// SSAO occlusion values bound to the lit shader.
	//
	// Returns:
	//   - *wgpu.Texture: the blurred SSAO texture, or nil if not initialized
	BlurredTexture() *wgpu.Texture

	// SetBlurredTexture sets the blurred SSAO texture.
	//
	// Parameters:
	//   - t: the blurred SSAO texture
	SetBlurredTexture(t *wgpu.Texture)

	// BlurredTextureView returns the texture view for the blurred SSAO texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the blurred SSAO texture view, or nil if not initialized
	BlurredTextureView() *wgpu.TextureView

	// SetBlurredTextureView sets the texture view for the blurred SSAO texture.
	//
	// Parameters:
	//   - tv: the blurred SSAO texture view
	SetBlurredTextureView(tv *wgpu.TextureView)

	// ScratchTexture returns the R8Unorm scratch texture used as an intermediate
	// target between the horizontal and vertical blur passes.
	//
	// Returns:
	//   - *wgpu.Texture: the scratch texture, or nil if not initialized
	ScratchTexture() *wgpu.Texture

	// SetScratchTexture sets the scratch texture for the blur passes.
	//
	// Parameters:
	//   - t: the scratch texture
	SetScratchTexture(t *wgpu.Texture)

	// ScratchTextureView returns the texture view for the scratch texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the scratch texture view, or nil if not initialized
	ScratchTextureView() *wgpu.TextureView

	// SetScratchTextureView sets the texture view for the scratch texture.
	//
	// Parameters:
	//   - tv: the scratch texture view
	SetScratchTextureView(tv *wgpu.TextureView)

	// NoiseTexture returns the 4×4 RGBA16Float texture storing random rotation
	// vectors for the SSAO sample kernel.
	//
	// Returns:
	//   - *wgpu.Texture: the noise texture, or nil if not initialized
	NoiseTexture() *wgpu.Texture

	// SetNoiseTexture sets the SSAO noise texture.
	//
	// Parameters:
	//   - t: the noise texture
	SetNoiseTexture(t *wgpu.Texture)

	// NoiseTextureView returns the texture view for the noise texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the noise texture view, or nil if not initialized
	NoiseTextureView() *wgpu.TextureView

	// SetNoiseTextureView sets the texture view for the noise texture.
	//
	// Parameters:
	//   - tv: the noise texture view
	SetNoiseTextureView(tv *wgpu.TextureView)

	// LinearSampler returns the linear sampler used when binding the SSAO
	// blurred texture to the lit fragment shader.
	//
	// Returns:
	//   - *wgpu.Sampler: the linear sampler, or nil if not initialized
	LinearSampler() *wgpu.Sampler

	// SetLinearSampler sets the linear sampler for SSAO texture sampling.
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

	// HalfResolution returns whether SSAO textures are allocated at half the
	// screen resolution for improved performance.
	//
	// Returns:
	//   - bool: true if half-resolution mode is enabled
	HalfResolution() bool

	// SetHalfResolution sets whether SSAO textures should be allocated at half
	// the screen resolution.
	//
	// Parameters:
	//   - enabled: true to enable half-resolution mode
	SetHalfResolution(enabled bool)
}

var _ SSAOHandler = &ssaoHandlerImpl{}

func (h *ssaoHandlerImpl) Enabled() bool {
	return h.enabled
}

func (h *ssaoHandlerImpl) SetEnabled(enabled bool) {
	h.enabled = enabled
}

func (h *ssaoHandlerImpl) ScreenWidth() int {
	return h.screenWidth
}

func (h *ssaoHandlerImpl) ScreenHeight() int {
	return h.screenHeight
}

func (h *ssaoHandlerImpl) SampleCount() int {
	return h.sampleCount
}

func (h *ssaoHandlerImpl) Radius() float32 {
	return h.radius
}

func (h *ssaoHandlerImpl) Bias() float32 {
	return h.bias
}

func (h *ssaoHandlerImpl) Power() float32 {
	return h.power
}

func (h *ssaoHandlerImpl) BlurRadius() int {
	return h.blurRadius
}

func (h *ssaoHandlerImpl) PipelineKey(name string) string {
	return h.pipelineKeys[name]
}

func (h *ssaoHandlerImpl) PipelineKeys() map[string]string {
	return h.pipelineKeys
}

func (h *ssaoHandlerImpl) SetPipelineKey(name, key string) {
	h.pipelineKeys[name] = key
}

func (h *ssaoHandlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return h.bgps[key]
}

func (h *ssaoHandlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}

func (h *ssaoHandlerImpl) SetBgp(key string, bgp bind_group_provider.BindGroupProvider) {
	h.bgps[key] = bgp
}

func (h *ssaoHandlerImpl) RawTexture() *wgpu.Texture {
	return h.rawTexture
}

func (h *ssaoHandlerImpl) SetRawTexture(t *wgpu.Texture) {
	h.rawTexture = t
}

func (h *ssaoHandlerImpl) RawTextureView() *wgpu.TextureView {
	return h.rawTextureView
}

func (h *ssaoHandlerImpl) SetRawTextureView(tv *wgpu.TextureView) {
	h.rawTextureView = tv
}

func (h *ssaoHandlerImpl) BlurredTexture() *wgpu.Texture {
	return h.blurredTexture
}

func (h *ssaoHandlerImpl) SetBlurredTexture(t *wgpu.Texture) {
	h.blurredTexture = t
}

func (h *ssaoHandlerImpl) BlurredTextureView() *wgpu.TextureView {
	return h.blurredTextureView
}

func (h *ssaoHandlerImpl) SetBlurredTextureView(tv *wgpu.TextureView) {
	h.blurredTextureView = tv
}

func (h *ssaoHandlerImpl) ScratchTexture() *wgpu.Texture {
	return h.scratchTexture
}

func (h *ssaoHandlerImpl) SetScratchTexture(t *wgpu.Texture) {
	h.scratchTexture = t
}

func (h *ssaoHandlerImpl) ScratchTextureView() *wgpu.TextureView {
	return h.scratchTextureView
}

func (h *ssaoHandlerImpl) SetScratchTextureView(tv *wgpu.TextureView) {
	h.scratchTextureView = tv
}

func (h *ssaoHandlerImpl) NoiseTexture() *wgpu.Texture {
	return h.noiseTexture
}

func (h *ssaoHandlerImpl) SetNoiseTexture(t *wgpu.Texture) {
	h.noiseTexture = t
}

func (h *ssaoHandlerImpl) NoiseTextureView() *wgpu.TextureView {
	return h.noiseTextureView
}

func (h *ssaoHandlerImpl) SetNoiseTextureView(tv *wgpu.TextureView) {
	h.noiseTextureView = tv
}

func (h *ssaoHandlerImpl) LinearSampler() *wgpu.Sampler {
	return h.linearSampler
}

func (h *ssaoHandlerImpl) SetLinearSampler(s *wgpu.Sampler) {
	h.linearSampler = s
}

func (h *ssaoHandlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
}

func (h *ssaoHandlerImpl) HalfResolution() bool {
	return h.halfResolution
}

func (h *ssaoHandlerImpl) SetHalfResolution(enabled bool) {
	h.halfResolution = enabled
}
