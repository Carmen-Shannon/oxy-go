package postprocessing

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

type taaHandlerImpl struct {
	enabled bool

	screenWidth  int
	screenHeight int

	blendFactor               float32
	historyRectificationScale float32
	rawHistoryOnly            bool
	jitterScale               float32

	// Halton jitter state.
	jitterX     float32
	jitterY     float32
	prevJitterX float32
	prevJitterY float32
	frameIndex  uint64

	pipelineKeys map[string]string
	bgps         map[string]bind_group_provider.BindGroupProvider

	activeSlot int

	// Ping-pong RGBA16Float textures.
	// Slot 0: taaTextures[0] is the resolved output; taaTextures[1] is the history read.
	// Slot 1: taaTextures[1] is the resolved output; taaTextures[0] is the history read.
	taaTextures     [2]*wgpu.Texture
	taaTextureViews [2]*wgpu.TextureView

	linearSampler *wgpu.Sampler

	sharpenTexture     *wgpu.Texture
	sharpenTextureView *wgpu.TextureView
}

func (h *taaHandlerImpl) Enabled() bool            { return h.enabled }
func (h *taaHandlerImpl) SetEnabled(b bool)        { h.enabled = b }
func (h *taaHandlerImpl) SetSlot(slot int)         { h.activeSlot = slot }
func (h *taaHandlerImpl) ScreenWidth() int         { return h.screenWidth }
func (h *taaHandlerImpl) ScreenHeight() int        { return h.screenHeight }
func (h *taaHandlerImpl) BlendFactor() float32     { return h.blendFactor }
func (h *taaHandlerImpl) SetBlendFactor(f float32) { h.blendFactor = f }
func (h *taaHandlerImpl) HistoryRectificationScale() float32 {
	return h.historyRectificationScale
}
func (h *taaHandlerImpl) SetHistoryRectificationScale(scale float32) {
	h.historyRectificationScale = scale
}
func (h *taaHandlerImpl) RawHistoryOnly() bool { return h.rawHistoryOnly }
func (h *taaHandlerImpl) SetRawHistoryOnly(enabled bool) {
	h.rawHistoryOnly = enabled
}
func (h *taaHandlerImpl) JitterScale() float32 { return h.jitterScale }
func (h *taaHandlerImpl) SetJitterScale(scale float32) {
	h.jitterScale = scale
}
func (h *taaHandlerImpl) JitterX() float32     { return h.jitterX }
func (h *taaHandlerImpl) JitterY() float32     { return h.jitterY }
func (h *taaHandlerImpl) PrevJitterX() float32 { return h.prevJitterX }
func (h *taaHandlerImpl) PrevJitterY() float32 { return h.prevJitterY }
func (h *taaHandlerImpl) FrameIndex() uint64   { return h.frameIndex }

func (h *taaHandlerImpl) AdvanceFrame(jitterX, jitterY float32) {
	h.prevJitterX = h.jitterX
	h.prevJitterY = h.jitterY
	h.jitterX = jitterX
	h.jitterY = jitterY
	h.frameIndex++
}

func (h *taaHandlerImpl) PipelineKey(name string) string  { return h.pipelineKeys[name] }
func (h *taaHandlerImpl) PipelineKeys() map[string]string { return h.pipelineKeys }
func (h *taaHandlerImpl) SetPipelineKey(name, key string) { h.pipelineKeys[name] = key }

func (h *taaHandlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider { return h.bgps[key] }
func (h *taaHandlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}
func (h *taaHandlerImpl) SetBgp(key string, bgp bind_group_provider.BindGroupProvider) {
	h.bgps[key] = bgp
}

func (h *taaHandlerImpl) TAATexture() *wgpu.Texture         { return h.taaTextures[h.activeSlot] }
func (h *taaHandlerImpl) SetTAATexture(t *wgpu.Texture)     { h.taaTextures[h.activeSlot] = t }
func (h *taaHandlerImpl) TAATextureView() *wgpu.TextureView { return h.taaTextureViews[h.activeSlot] }
func (h *taaHandlerImpl) SetTAATextureView(tv *wgpu.TextureView) {
	h.taaTextureViews[h.activeSlot] = tv
}
func (h *taaHandlerImpl) LinearSampler() *wgpu.Sampler     { return h.linearSampler }
func (h *taaHandlerImpl) SetLinearSampler(s *wgpu.Sampler) { h.linearSampler = s }

func (h *taaHandlerImpl) SharpenTexture() *wgpu.Texture              { return h.sharpenTexture }
func (h *taaHandlerImpl) SetSharpenTexture(t *wgpu.Texture)          { h.sharpenTexture = t }
func (h *taaHandlerImpl) SharpenTextureView() *wgpu.TextureView      { return h.sharpenTextureView }
func (h *taaHandlerImpl) SetSharpenTextureView(tv *wgpu.TextureView) { h.sharpenTextureView = tv }

func (h *taaHandlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
}
