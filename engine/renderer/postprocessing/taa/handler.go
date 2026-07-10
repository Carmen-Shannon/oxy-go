// Package taa provides the temporal anti-aliasing postprocessing subsystem.
package taa

import (
	"github.com/oliverbestmann/webgpu/wgpu"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// Handler manages GPU resources and state for the Temporal Anti-Aliasing
// resolve pass. It owns two ping-pong RGBA16Float textures (history and resolve
// output), a linear sampler, the taa_resolve compute pipeline key, and the two
// slot-indexed BGPs that wire those resources into the compute shader.
type Handler interface {
	// Enabled reports whether TAA is active.
	Enabled() bool

	// SetEnabled enables or disables the TAA resolve pass.
	SetEnabled(enabled bool)

	// SetSlot switches the active ping-pong slot (0 or 1). Called by SyncFrameSlot
	// at the start of each frame before any Prepare method runs.
	SetSlot(slot int)

	// ScreenWidth returns the current render width in texels.
	ScreenWidth() int

	// ScreenHeight returns the current render height in texels.
	ScreenHeight() int

	// BlendFactor returns the weight given to the current frame during temporal
	// blending. Typical value: 0.1 (10% current frame, 90% history).
	BlendFactor() float32

	// SetBlendFactor sets the current-frame blend weight.
	SetBlendFactor(f float32)

	// HistoryRectificationScale returns the diagnostic scale applied to the YCoCg
	// history clamp box around the 3x3 neighborhood mean. A value of 1.0 keeps
	// the production clamp unchanged.
	HistoryRectificationScale() float32

	// SetHistoryRectificationScale sets the diagnostic history clamp expansion scale.
	SetHistoryRectificationScale(scale float32)

	// RawHistoryOnly reports whether the diagnostic mode that outputs raw reprojected
	// history before neighborhood clamping is enabled.
	RawHistoryOnly() bool

	// SetRawHistoryOnly enables or disables the raw-history-only diagnostic mode.
	SetRawHistoryOnly(enabled bool)

	// JitterScale returns the multiplier applied to Halton jitter before it is
	// written into the handler and camera projection state.
	JitterScale() float32

	// SetJitterScale sets the multiplier applied to Halton jitter.
	SetJitterScale(scale float32)

	// JitterX returns the NDC X jitter applied to the current frame's projection.
	JitterX() float32

	// JitterY returns the NDC Y jitter applied to the current frame's projection.
	JitterY() float32

	// PrevJitterX returns the NDC X jitter that was applied in the previous frame.
	PrevJitterX() float32

	// PrevJitterY returns the NDC Y jitter that was applied in the previous frame.
	PrevJitterY() float32

	// FrameIndex returns the monotonically increasing frame counter used to index
	// into the Halton sequence for jitter generation.
	FrameIndex() uint64

	// AdvanceFrame records the new jitter values for the upcoming frame, saves the
	// current jitter as "previous", and increments the frame counter.
	AdvanceFrame(jitterX, jitterY float32)

	// PipelineKey returns the registered wgpu pipeline key for the named pipeline.
	// The only keys used by TAA are "taa_resolve" and "taa_sharpen".
	PipelineKey(name string) string

	// PipelineKeys returns all registered pipeline keys.
	PipelineKeys() map[string]string

	// SetPipelineKey stores a pipeline key under the given name.
	SetPipelineKey(name, key string)

	// Bgp returns the BindGroupProvider for the given key.
	// Known keys: "taa_resolve_0", "taa_resolve_1", "taa_sharpen_0", "taa_sharpen_1".
	Bgp(key string) bind_group_provider.BindGroupProvider

	// Bgps returns all BindGroupProviders.
	Bgps() map[string]bind_group_provider.BindGroupProvider

	// SetBgp stores a BindGroupProvider under the given key.
	SetBgp(key string, bgp bind_group_provider.BindGroupProvider)

	// TAATexture returns the active slot's ping-pong texture.
	TAATexture() *wgpu.Texture

	// SetTAATexture stores a texture in the active slot.
	SetTAATexture(t *wgpu.Texture)

	// TAATextureView returns the active slot's ping-pong texture view.
	TAATextureView() *wgpu.TextureView

	// SetTAATextureView stores a texture view in the active slot.
	SetTAATextureView(tv *wgpu.TextureView)

	// LinearSampler returns the shared linear sampler used for bilinear history sampling.
	LinearSampler() *wgpu.Sampler

	// SetLinearSampler stores the linear sampler.
	SetLinearSampler(s *wgpu.Sampler)

	// SharpenTexture returns the RGBA16Float CAS output texture read by the composition pass.
	// Returns nil before initTAA() runs.
	SharpenTexture() *wgpu.Texture

	// SetSharpenTexture stores the CAS output texture.
	SetSharpenTexture(t *wgpu.Texture)

	// SharpenTextureView returns the texture view for the CAS output texture.
	SharpenTextureView() *wgpu.TextureView

	// SetSharpenTextureView stores the CAS output texture view.
	SetSharpenTextureView(tv *wgpu.TextureView)

	// Resize updates the stored screen dimensions. Does not touch GPU resources;
	// GPU recreation is handled by initTAA() called from resizePostProcessing().
	Resize(width, height int)
}

var _ Handler = &handlerImpl{}
