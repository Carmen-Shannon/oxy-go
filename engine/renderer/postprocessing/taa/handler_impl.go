package taa

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/oliverbestmann/webgpu/wgpu"
)

type handlerImpl struct {
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

func (h *handlerImpl) Enabled() bool            { return h.enabled }
func (h *handlerImpl) SetEnabled(b bool)        { h.enabled = b }
func (h *handlerImpl) SetSlot(slot int)         { h.activeSlot = slot }
func (h *handlerImpl) ScreenWidth() int         { return h.screenWidth }
func (h *handlerImpl) ScreenHeight() int        { return h.screenHeight }
func (h *handlerImpl) BlendFactor() float32     { return h.blendFactor }
func (h *handlerImpl) SetBlendFactor(f float32) { h.blendFactor = f }
func (h *handlerImpl) HistoryRectificationScale() float32 {
	return h.historyRectificationScale
}
func (h *handlerImpl) SetHistoryRectificationScale(scale float32) {
	h.historyRectificationScale = scale
}
func (h *handlerImpl) RawHistoryOnly() bool { return h.rawHistoryOnly }
func (h *handlerImpl) SetRawHistoryOnly(enabled bool) {
	h.rawHistoryOnly = enabled
}
func (h *handlerImpl) JitterScale() float32 { return h.jitterScale }
func (h *handlerImpl) SetJitterScale(scale float32) {
	h.jitterScale = scale
}
func (h *handlerImpl) JitterX() float32     { return h.jitterX }
func (h *handlerImpl) JitterY() float32     { return h.jitterY }
func (h *handlerImpl) PrevJitterX() float32 { return h.prevJitterX }
func (h *handlerImpl) PrevJitterY() float32 { return h.prevJitterY }
func (h *handlerImpl) FrameIndex() uint64   { return h.frameIndex }

func (h *handlerImpl) AdvanceFrame(jitterX, jitterY float32) {
	h.prevJitterX = h.jitterX
	h.prevJitterY = h.jitterY
	h.jitterX = jitterX
	h.jitterY = jitterY
	h.frameIndex++
}

func (h *handlerImpl) PipelineKey(name string) string  { return h.pipelineKeys[name] }
func (h *handlerImpl) PipelineKeys() map[string]string { return h.pipelineKeys }
func (h *handlerImpl) SetPipelineKey(name, key string) { h.pipelineKeys[name] = key }

func (h *handlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider { return h.bgps[key] }
func (h *handlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}
func (h *handlerImpl) SetBgp(key string, bgp bind_group_provider.BindGroupProvider) {
	h.bgps[key] = bgp
}

func (h *handlerImpl) TAATexture() *wgpu.Texture         { return h.taaTextures[h.activeSlot] }
func (h *handlerImpl) SetTAATexture(t *wgpu.Texture)     { h.taaTextures[h.activeSlot] = t }
func (h *handlerImpl) TAATextureView() *wgpu.TextureView { return h.taaTextureViews[h.activeSlot] }
func (h *handlerImpl) SetTAATextureView(tv *wgpu.TextureView) {
	h.taaTextureViews[h.activeSlot] = tv
}
func (h *handlerImpl) LinearSampler() *wgpu.Sampler     { return h.linearSampler }
func (h *handlerImpl) SetLinearSampler(s *wgpu.Sampler) { h.linearSampler = s }

func (h *handlerImpl) SharpenTexture() *wgpu.Texture              { return h.sharpenTexture }
func (h *handlerImpl) SetSharpenTexture(t *wgpu.Texture)          { h.sharpenTexture = t }
func (h *handlerImpl) SharpenTextureView() *wgpu.TextureView      { return h.sharpenTextureView }
func (h *handlerImpl) SetSharpenTextureView(tv *wgpu.TextureView) { h.sharpenTextureView = tv }

func (h *handlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
}
