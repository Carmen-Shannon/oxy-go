// Package ssao provides the screen-space ambient occlusion postprocessing subsystem.
package ssao

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/oliverbestmann/webgpu/wgpu"
)

// Handler defines the interface for the scene's SSAO subsystem.
//
// The Handler manages the hemisphere sample kernel, noise texture, raw and
// blurred occlusion textures, compute pipeline keys, and bind group providers
// needed by the SSAO compute and bilateral blur shaders. It is created via
// NewHandler with builder options and attached to a scene via
// the GI configuration. GPU resources are initialized lazily by the owning
// scene when SSAO is first enabled.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type Handler interface {
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

	// SetSlot selects the active texture slot. Texture and view getters and
	// setters read and write the [slot] index of the underlying arrays.
	//
	// Parameters:
	//   - slot: the slot index (0 or 1)
	SetSlot(slot int)

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

	// MaxSamples returns the GPU compile-time upper bound for the SSAO kernel
	// sample array.
	//
	// Returns:
	//   - int: the maximum number of samples
	MaxSamples() int

	// ScreenRadius returns the desired SSAO sampling radius in screen pixels.
	// The engine auto-computes the world-space radius each frame from this value,
	// the camera distance, FOV, and screen height.
	//
	// Returns:
	//   - float32: the screen-space radius in pixels
	ScreenRadius() float32

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

var _ Handler = &handlerImpl{}

func (h *handlerImpl) Enabled() bool                     { return h.enabled }
func (h *handlerImpl) SetEnabled(enabled bool)           { h.enabled = enabled }
func (h *handlerImpl) ScreenWidth() int                  { return h.screenWidth }
func (h *handlerImpl) ScreenHeight() int                 { return h.screenHeight }
func (h *handlerImpl) SampleCount() int                  { return h.sampleCount }
func (h *handlerImpl) MaxSamples() int                   { return h.maxSamples }
func (h *handlerImpl) ScreenRadius() float32             { return h.screenRadius }
func (h *handlerImpl) Bias() float32                     { return h.bias }
func (h *handlerImpl) Power() float32                    { return h.power }
func (h *handlerImpl) BlurRadius() int                   { return h.blurRadius }
func (h *handlerImpl) PipelineKey(name string) string    { return h.pipelineKeys[name] }
func (h *handlerImpl) PipelineKeys() map[string]string   { return h.pipelineKeys }
func (h *handlerImpl) SetPipelineKey(name, key string)   { h.pipelineKeys[name] = key }
func (h *handlerImpl) SetSlot(slot int)                  { h.activeSlot = slot }
func (h *handlerImpl) RawTexture() *wgpu.Texture         { return h.rawTextures[h.activeSlot] }
func (h *handlerImpl) SetRawTexture(t *wgpu.Texture)     { h.rawTextures[h.activeSlot] = t }
func (h *handlerImpl) RawTextureView() *wgpu.TextureView { return h.rawTextureViews[h.activeSlot] }
func (h *handlerImpl) SetRawTextureView(tv *wgpu.TextureView) {
	h.rawTextureViews[h.activeSlot] = tv
}
func (h *handlerImpl) BlurredTexture() *wgpu.Texture     { return h.blurredTextures[h.activeSlot] }
func (h *handlerImpl) SetBlurredTexture(t *wgpu.Texture) { h.blurredTextures[h.activeSlot] = t }
func (h *handlerImpl) BlurredTextureView() *wgpu.TextureView {
	return h.blurredTextureViews[h.activeSlot]
}
func (h *handlerImpl) SetBlurredTextureView(tv *wgpu.TextureView) {
	h.blurredTextureViews[h.activeSlot] = tv
}
func (h *handlerImpl) ScratchTexture() *wgpu.Texture     { return h.scratchTextures[h.activeSlot] }
func (h *handlerImpl) SetScratchTexture(t *wgpu.Texture) { h.scratchTextures[h.activeSlot] = t }
func (h *handlerImpl) ScratchTextureView() *wgpu.TextureView {
	return h.scratchTextureViews[h.activeSlot]
}
func (h *handlerImpl) SetScratchTextureView(tv *wgpu.TextureView) {
	h.scratchTextureViews[h.activeSlot] = tv
}
func (h *handlerImpl) LinearSampler() *wgpu.Sampler     { return h.linearSampler }
func (h *handlerImpl) SetLinearSampler(s *wgpu.Sampler) { h.linearSampler = s }
func (h *handlerImpl) HalfResolution() bool             { return h.halfResolution }
func (h *handlerImpl) SetHalfResolution(enabled bool)   { h.halfResolution = enabled }

func (h *handlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return h.bgps[key]
}

func (h *handlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}

func (h *handlerImpl) SetBgp(key string, bgp bind_group_provider.BindGroupProvider) {
	h.bgps[key] = bgp
}

func (h *handlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
}
