package light

import (
	"fmt"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/gogpu/wgpu"
)

// lightingHandlerImpl is the implementation of the LightingHandler interface.
type lightingHandlerImpl struct {
	enabled bool

	lights       []Light
	ambientColor [3]float32

	bgps         map[string]bind_group_provider.BindGroupProvider
	pipelineKeys map[string]string

	shadowHalfExtent      float32
	shadowNear            float32
	shadowFar             float32
	shadowBias            float32
	shadowNormalBiasScale float32
	shadowMapResolution   int

	// VSM-specific resources and configuration.
	vsmTexture             *wgpu.Texture
	vsmTextureView         *wgpu.TextureView
	vsmScratchTexture      *wgpu.Texture
	vsmScratchTextureView  *wgpu.TextureView
	vsmAuxDepthTexture     *wgpu.Texture
	vsmAuxDepthTextureView *wgpu.TextureView
	vsmLinearSampler       *wgpu.Sampler
	vsmBlurRadius          int
	vsmMinVariance         float32
	vsmLightBleedReduction float32
	vsmLightSize           float32

	// PCSS (Percentage-Closer Soft Shadows) resources and configuration.
	pcssEnabled     bool
	satTextureA     *wgpu.Texture
	satTextureAView *wgpu.TextureView
	satTextureB     *wgpu.Texture
	satTextureBView *wgpu.TextureView

	screenWidth  int
	screenHeight int
	tileCountX   uint32
	tileCountY   uint32

	// GI sub-handlers — owned by the lighting system and lazily initialized
	// by the scene when the first light is added.
	gBufferHandler     GBufferHandler
	ssaoHandler        SSAOHandler
	probeGrid          IrradianceProbeGrid
	compositionHandler CompositionHandler
	ssrHandler         SSRHandler
}

// LightingHandler defines the interface for the scene's lighting subsystem.
//
// The LightingHandler manages the light list, ambient color, shadow mapping
// configuration, Forward+ tile culling state, and all associated GPU resources
// (bind group providers, pipeline keys, shadow textures). It is created via
// NewLightingHandler with builder options and attached to a scene via
// WithLighting. GPU resources are initialized lazily by the scene when the
// first light is added.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type LightingHandler interface {
	// Enabled returns whether the lighting subsystem has been GPU-initialized and
	// is ready for rendering.
	//
	// Returns:
	//   - bool: true if lighting GPU resources have been initialized
	Enabled() bool

	// SetEnabled sets whether the lighting subsystem is GPU-initialized.
	//
	// Parameters:
	//   - enabled: true to mark as initialized
	SetEnabled(enabled bool)

	// Lights returns a copy of the current light list.
	//
	// Returns:
	//   - []Light: a copy of the lights registered with the handler
	Lights() []Light

	// AddLight appends a light to the handler's light list.
	//
	// Parameters:
	//   - l: the Light to add
	AddLight(l Light)

	// RemoveLight removes a light from the handler's list by reference equality.
	//
	// Parameters:
	//   - l: the Light to remove
	RemoveLight(l Light)

	// AmbientColor returns the scene's ambient light color.
	//
	// Returns:
	//   - [3]float32: the ambient RGB color
	AmbientColor() [3]float32

	// SetAmbientColor sets the scene's ambient light color.
	//
	// Parameters:
	//   - color: the ambient RGB color
	SetAmbientColor(color [3]float32)

	// Bgp retrieves the BindGroupProvider associated with the given key.
	// Returns nil if the key does not exist.
	//
	// Known keys:
	//   - "lights": light storage buffer BGP
	//   - "shadow_data": shadow depth pass BGP (light VP uniform)
	//   - "shadow_lit": lit pass shadow sampling BGP (texture + sampler + uniform)
	//   - "light_cull": compute shader cull BGP
	//   - "tile_lit": fragment shader tile data BGP
	//   - "ssao_lit": lit pass SSAO occlusion texture + sampler BGP (real or fallback white)
	//
	// Parameters:
	//   - key: the bind group provider identifier
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the provider, or nil if not found
	Bgp(key string) bind_group_provider.BindGroupProvider

	// Bgps returns the full map of bind group providers.
	//
	// Returns:
	//   - map[string]bind_group_provider.BindGroupProvider: all registered providers
	Bgps() map[string]bind_group_provider.BindGroupProvider

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

	// ShadowHalfExtent returns the orthographic half-extent of the directional shadow frustum
	// in world units.
	//
	// Returns:
	//   - float32: the shadow frustum half-extent
	ShadowHalfExtent() float32

	// ShadowNear returns the near plane distance for the directional shadow projection.
	//
	// Returns:
	//   - float32: the shadow near plane
	ShadowNear() float32

	// ShadowFar returns the far plane distance for the directional shadow projection.
	//
	// Returns:
	//   - float32: the shadow far plane
	ShadowFar() float32

	// ShadowBias returns the constant depth bias applied to shadow comparisons.
	//
	// Returns:
	//   - float32: the shadow bias
	ShadowBias() float32

	// ShadowNormalBiasScale returns the multiplier applied to the shadow map texel
	// world-size to derive the normal-offset bias.
	//
	// Returns:
	//   - float32: the normal bias scale
	ShadowNormalBiasScale() float32

	// ShadowMapResolution returns the width and height in texels of the shadow depth texture.
	//
	// Returns:
	//   - int: the shadow map resolution
	ShadowMapResolution() int

	// VSMTexture returns the RG32Float variance shadow map texture.
	//
	// Returns:
	//   - *wgpu.Texture: the VSM texture, or nil if not initialized
	VSMTexture() *wgpu.Texture

	// SetVSMTexture sets the RG32Float variance shadow map texture.
	//
	// Parameters:
	//   - t: the VSM texture
	SetVSMTexture(t *wgpu.Texture)

	// VSMTextureView returns the texture view for the VSM texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the VSM texture view, or nil if not initialized
	VSMTextureView() *wgpu.TextureView

	// SetVSMTextureView sets the texture view for the VSM texture.
	//
	// Parameters:
	//   - tv: the VSM texture view
	SetVSMTextureView(tv *wgpu.TextureView)

	// VSMScratchTexture returns the scratch RG32Float texture used as intermediate
	// storage during the separable blur pass.
	//
	// Returns:
	//   - *wgpu.Texture: the scratch texture, or nil if not initialized
	VSMScratchTexture() *wgpu.Texture

	// SetVSMScratchTexture sets the scratch texture used during the separable blur pass.
	//
	// Parameters:
	//   - t: the scratch texture
	SetVSMScratchTexture(t *wgpu.Texture)

	// VSMScratchTextureView returns the texture view for the scratch texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the scratch texture view, or nil if not initialized
	VSMScratchTextureView() *wgpu.TextureView

	// SetVSMScratchTextureView sets the texture view for the scratch texture.
	//
	// Parameters:
	//   - tv: the scratch texture view
	SetVSMScratchTextureView(tv *wgpu.TextureView)

	// VSMAuxDepthTexture returns the auxiliary Depth32Float texture used for hardware
	// z-testing during the VSM shadow render pass.
	//
	// Returns:
	//   - *wgpu.Texture: the auxiliary depth texture, or nil if not initialized
	VSMAuxDepthTexture() *wgpu.Texture

	// SetVSMAuxDepthTexture sets the auxiliary depth texture for the VSM shadow pass.
	//
	// Parameters:
	//   - t: the auxiliary depth texture
	SetVSMAuxDepthTexture(t *wgpu.Texture)

	// VSMAuxDepthTextureView returns the texture view for the auxiliary depth texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the auxiliary depth texture view, or nil if not initialized
	VSMAuxDepthTextureView() *wgpu.TextureView

	// SetVSMAuxDepthTextureView sets the texture view for the auxiliary depth texture.
	//
	// Parameters:
	//   - tv: the auxiliary depth texture view
	SetVSMAuxDepthTextureView(tv *wgpu.TextureView)

	// VSMLinearSampler returns the linear sampler used for VSM texture lookups in
	// the lit fragment shader (replaces the comparison sampler used by PCF).
	//
	// Returns:
	//   - *wgpu.Sampler: the linear sampler, or nil if not initialized
	VSMLinearSampler() *wgpu.Sampler

	// SetVSMLinearSampler sets the linear sampler used for VSM texture lookups.
	//
	// Parameters:
	//   - s: the linear sampler
	SetVSMLinearSampler(s *wgpu.Sampler)

	// VSMBlurRadius returns the half-width (in texels) of the separable blur
	// applied to the variance shadow map.
	//
	// Returns:
	//   - int: the blur radius
	VSMBlurRadius() int

	// VSMMinVariance returns the minimum variance clamp for Chebyshev's inequality.
	//
	// Returns:
	//   - float32: the minimum variance
	VSMMinVariance() float32

	// VSMLightBleedReduction returns the exponent applied to the raw Chebyshev shadow
	// probability to reduce light-bleeding artifacts.
	//
	// Returns:
	//   - float32: the light bleed reduction exponent
	VSMLightBleedReduction() float32

	// VSMLightSize returns the world-space area light size used for PCSS penumbra estimation.
	//
	// Returns:
	//   - float32: the light size
	VSMLightSize() float32

	// PCSSEnabled returns whether PCSS contact-hardening soft shadows are enabled.
	// PCSS requires VSM to also be enabled.
	//
	// Returns:
	//   - bool: true if PCSS is enabled
	PCSSEnabled() bool

	// SetPCSSEnabled sets whether PCSS contact-hardening soft shadows are enabled.
	//
	// Parameters:
	//   - enabled: true to enable PCSS
	SetPCSSEnabled(enabled bool)

	// SATTextureA returns the first RGBA32Float SAT ping-pong texture.
	//
	// Returns:
	//   - *wgpu.Texture: SAT texture A, or nil if not initialized
	SATTextureA() *wgpu.Texture

	// SetSATTextureA sets the first SAT ping-pong texture.
	//
	// Parameters:
	//   - t: SAT texture A
	SetSATTextureA(t *wgpu.Texture)

	// SATTextureAView returns the texture view for SAT texture A.
	//
	// Returns:
	//   - *wgpu.TextureView: SAT texture A view, or nil if not initialized
	SATTextureAView() *wgpu.TextureView

	// SetSATTextureAView sets the texture view for SAT texture A.
	//
	// Parameters:
	//   - tv: SAT texture A view
	SetSATTextureAView(tv *wgpu.TextureView)

	// SATTextureB returns the second RGBA32Float SAT ping-pong texture.
	//
	// Returns:
	//   - *wgpu.Texture: SAT texture B, or nil if not initialized
	SATTextureB() *wgpu.Texture

	// SetSATTextureB sets the second SAT ping-pong texture.
	//
	// Parameters:
	//   - t: SAT texture B
	SetSATTextureB(t *wgpu.Texture)

	// SATTextureBView returns the texture view for SAT texture B.
	//
	// Returns:
	//   - *wgpu.TextureView: SAT texture B view, or nil if not initialized
	SATTextureBView() *wgpu.TextureView

	// SetSATTextureBView sets the texture view for SAT texture B.
	//
	// Parameters:
	//   - tv: SAT texture B view
	SetSATTextureBView(tv *wgpu.TextureView)

	// ScreenWidth returns the current screen width in pixels used for tile calculations.
	//
	// Returns:
	//   - int: the screen width
	ScreenWidth() int

	// ScreenHeight returns the current screen height in pixels used for tile calculations.
	//
	// Returns:
	//   - int: the screen height
	ScreenHeight() int

	// TileCountX returns the number of Forward+ tile columns for the current screen width.
	//
	// Returns:
	//   - uint32: the tile column count
	TileCountX() uint32

	// TileCountY returns the number of Forward+ tile rows for the current screen height.
	//
	// Returns:
	//   - uint32: the tile row count
	TileCountY() uint32

	// Resize updates the screen dimensions and recalculates tile counts for Forward+
	// light culling. This should be called when the window is resized.
	//
	// Parameters:
	//   - width: the new screen width in pixels
	//   - height: the new screen height in pixels
	Resize(width, height int)

	// GBufferHandler returns the GBufferHandler attached to this lighting
	// subsystem, or nil if no G-Buffer pre-pass is configured.
	//
	// Returns:
	//   - GBufferHandler: the G-Buffer handler, or nil
	GBufferHandler() GBufferHandler

	// SSAOHandler returns the SSAOHandler attached to this lighting
	// subsystem, or nil if SSAO is not configured.
	//
	// Returns:
	//   - SSAOHandler: the SSAO handler, or nil
	SSAOHandler() SSAOHandler

	// ProbeGrid returns the IrradianceProbeGrid attached to this lighting
	// subsystem, or nil if probe-based GI is not configured.
	//
	// Returns:
	//   - IrradianceProbeGrid: the probe grid handler, or nil
	ProbeGrid() IrradianceProbeGrid

	// CompositionHandler returns the CompositionHandler attached to this lighting
	// subsystem, or nil if composition/tone mapping is not configured.
	//
	// Returns:
	//   - CompositionHandler: the composition handler, or nil
	CompositionHandler() CompositionHandler

	// SSRHandler returns the SSRHandler attached to this lighting
	// subsystem, or nil if screen-space reflections are not configured.
	//
	// Returns:
	//   - SSRHandler: the SSR handler, or nil
	SSRHandler() SSRHandler
}

var _ LightingHandler = &lightingHandlerImpl{}

// NewLightingHandler creates a new LightingHandler with sensible defaults and any
// provided options applied. Pre-creates named BindGroupProviders for each lighting
// subsystem stage. GPU resources are not allocated until the owning scene calls
// the appropriate initialization methods.
//
// Parameters:
//   - opts: variadic list of LightingHandlerOption functions to configure the handler
//
// Returns:
//   - LightingHandler: a new handler instance ready to be attached to a scene
func NewLightingHandler(opts ...LightingHandlerOption) LightingHandler {
	h := &lightingHandlerImpl{
		enabled: false,
		lights:  make([]Light, 0),
		bgps: map[string]bind_group_provider.BindGroupProvider{
			"lights":          bind_group_provider.NewBindGroupProvider("lights"),
			"shadow_data":     bind_group_provider.NewBindGroupProvider("shadow_data"),
			"shadow_lit":      bind_group_provider.NewBindGroupProvider("shadow_lit"),
			"light_cull":      bind_group_provider.NewBindGroupProvider("light_cull"),
			"tile_lit":        bind_group_provider.NewBindGroupProvider("tile_lit"),
			"vsm_blur_h":      bind_group_provider.NewBindGroupProvider("vsm_blur_h"),
			"vsm_blur_v":      bind_group_provider.NewBindGroupProvider("vsm_blur_v"),
			"sat_prepare":     bind_group_provider.NewBindGroupProvider("sat_prepare"),
			"ssao_lit":        bind_group_provider.NewBindGroupProvider("ssao_lit"),
			"probe_lit":       bind_group_provider.NewBindGroupProvider("probe_lit"),
			"composition_lit": bind_group_provider.NewBindGroupProvider("composition_lit"),
			"ssr_lit":         bind_group_provider.NewBindGroupProvider("ssr_lit"),
		},
		pipelineKeys:           make(map[string]string),
		shadowHalfExtent:       DefaultShadowHalfExtent,
		shadowNear:             DefaultShadowNear,
		shadowFar:              DefaultShadowFar,
		shadowBias:             DefaultShadowBias,
		shadowNormalBiasScale:  DefaultShadowNormalBiasScale,
		shadowMapResolution:    ShadowMapResolution,
		vsmBlurRadius:          DefaultVSMBlurRadius,
		vsmMinVariance:         DefaultVSMMinVariance,
		vsmLightBleedReduction: DefaultVSMLightBleedReduction,
		vsmLightSize:           DefaultVSMLightSize,
		pcssEnabled:            false,
	}
	for _, opt := range opts {
		opt(h)
	}

	// Always create the GI subsystems (GBuffer, SSAO, Composition, SSR) with
	// sensible defaults if they were not explicitly provided via options. The
	// full GI pipeline is mandatory for lit scenes.
	if h.gBufferHandler == nil {
		h.gBufferHandler = NewGBufferHandler()
	}
	if h.ssaoHandler == nil {
		h.ssaoHandler = NewSSAOHandler()
	}
	if h.compositionHandler == nil {
		h.compositionHandler = NewCompositionHandler(
			WithToneMappingEnabled(true),
			WithExposure(1.0),
		)
	}
	if h.ssrHandler == nil {
		h.ssrHandler = NewSSRHandler()
	}

	// Pre-create per-pass SAT bind group providers. Each prefix-sum pass gets
	// its own BGP (and thus its own uniform buffer) so that all params can be
	// written upfront and dispatches batched into a single GPU submission.
	if h.pcssEnabled {
		numPasses := 0
		for v := h.shadowMapResolution; v > 1; v >>= 1 {
			numPasses++
		}
		for i := 0; i < 2*numPasses; i++ {
			name := fmt.Sprintf("sat_pass_%d", i)
			h.bgps[name] = bind_group_provider.NewBindGroupProvider(name)
		}
	}

	return h
}

func (h *lightingHandlerImpl) Enabled() bool {
	return h.enabled
}

func (h *lightingHandlerImpl) SetEnabled(enabled bool) {
	h.enabled = enabled
}

func (h *lightingHandlerImpl) Lights() []Light {
	out := make([]Light, len(h.lights))
	copy(out, h.lights)
	return out
}

func (h *lightingHandlerImpl) AddLight(l Light) {
	h.lights = append(h.lights, l)
}

func (h *lightingHandlerImpl) RemoveLight(l Light) {
	for i, existing := range h.lights {
		if existing == l {
			h.lights = append(h.lights[:i], h.lights[i+1:]...)
			return
		}
	}
}

func (h *lightingHandlerImpl) AmbientColor() [3]float32 {
	return h.ambientColor
}

func (h *lightingHandlerImpl) SetAmbientColor(color [3]float32) {
	h.ambientColor = color
}

func (h *lightingHandlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return h.bgps[key]
}

func (h *lightingHandlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}

func (h *lightingHandlerImpl) PipelineKey(name string) string {
	return h.pipelineKeys[name]
}

func (h *lightingHandlerImpl) PipelineKeys() map[string]string {
	return h.pipelineKeys
}

func (h *lightingHandlerImpl) SetPipelineKey(name, key string) {
	h.pipelineKeys[name] = key
}

func (h *lightingHandlerImpl) ShadowHalfExtent() float32 {
	return h.shadowHalfExtent
}

func (h *lightingHandlerImpl) ShadowNear() float32 {
	return h.shadowNear
}

func (h *lightingHandlerImpl) ShadowFar() float32 {
	return h.shadowFar
}

func (h *lightingHandlerImpl) ShadowBias() float32 {
	return h.shadowBias
}

func (h *lightingHandlerImpl) ShadowNormalBiasScale() float32 {
	return h.shadowNormalBiasScale
}

func (h *lightingHandlerImpl) ShadowMapResolution() int {
	return h.shadowMapResolution
}

func (h *lightingHandlerImpl) VSMTexture() *wgpu.Texture {
	return h.vsmTexture
}

func (h *lightingHandlerImpl) SetVSMTexture(t *wgpu.Texture) {
	h.vsmTexture = t
}

func (h *lightingHandlerImpl) VSMTextureView() *wgpu.TextureView {
	return h.vsmTextureView
}

func (h *lightingHandlerImpl) SetVSMTextureView(tv *wgpu.TextureView) {
	h.vsmTextureView = tv
}

func (h *lightingHandlerImpl) VSMScratchTexture() *wgpu.Texture {
	return h.vsmScratchTexture
}

func (h *lightingHandlerImpl) SetVSMScratchTexture(t *wgpu.Texture) {
	h.vsmScratchTexture = t
}

func (h *lightingHandlerImpl) VSMScratchTextureView() *wgpu.TextureView {
	return h.vsmScratchTextureView
}

func (h *lightingHandlerImpl) SetVSMScratchTextureView(tv *wgpu.TextureView) {
	h.vsmScratchTextureView = tv
}

func (h *lightingHandlerImpl) VSMAuxDepthTexture() *wgpu.Texture {
	return h.vsmAuxDepthTexture
}

func (h *lightingHandlerImpl) SetVSMAuxDepthTexture(t *wgpu.Texture) {
	h.vsmAuxDepthTexture = t
}

func (h *lightingHandlerImpl) VSMAuxDepthTextureView() *wgpu.TextureView {
	return h.vsmAuxDepthTextureView
}

func (h *lightingHandlerImpl) SetVSMAuxDepthTextureView(tv *wgpu.TextureView) {
	h.vsmAuxDepthTextureView = tv
}

func (h *lightingHandlerImpl) VSMLinearSampler() *wgpu.Sampler {
	return h.vsmLinearSampler
}

func (h *lightingHandlerImpl) SetVSMLinearSampler(s *wgpu.Sampler) {
	h.vsmLinearSampler = s
}

func (h *lightingHandlerImpl) VSMBlurRadius() int {
	return h.vsmBlurRadius
}

func (h *lightingHandlerImpl) VSMMinVariance() float32 {
	return h.vsmMinVariance
}

func (h *lightingHandlerImpl) VSMLightBleedReduction() float32 {
	return h.vsmLightBleedReduction
}

func (h *lightingHandlerImpl) VSMLightSize() float32 {
	return h.vsmLightSize
}

func (h *lightingHandlerImpl) PCSSEnabled() bool {
	return h.pcssEnabled
}

func (h *lightingHandlerImpl) SetPCSSEnabled(enabled bool) {
	h.pcssEnabled = enabled
}

func (h *lightingHandlerImpl) SATTextureA() *wgpu.Texture {
	return h.satTextureA
}

func (h *lightingHandlerImpl) SetSATTextureA(t *wgpu.Texture) {
	h.satTextureA = t
}

func (h *lightingHandlerImpl) SATTextureAView() *wgpu.TextureView {
	return h.satTextureAView
}

func (h *lightingHandlerImpl) SetSATTextureAView(tv *wgpu.TextureView) {
	h.satTextureAView = tv
}

func (h *lightingHandlerImpl) SATTextureB() *wgpu.Texture {
	return h.satTextureB
}

func (h *lightingHandlerImpl) SetSATTextureB(t *wgpu.Texture) {
	h.satTextureB = t
}

func (h *lightingHandlerImpl) SATTextureBView() *wgpu.TextureView {
	return h.satTextureBView
}

func (h *lightingHandlerImpl) SetSATTextureBView(tv *wgpu.TextureView) {
	h.satTextureBView = tv
}

func (h *lightingHandlerImpl) ScreenWidth() int {
	return h.screenWidth
}

func (h *lightingHandlerImpl) ScreenHeight() int {
	return h.screenHeight
}

func (h *lightingHandlerImpl) TileCountX() uint32 {
	return h.tileCountX
}

func (h *lightingHandlerImpl) TileCountY() uint32 {
	return h.tileCountY
}

func (h *lightingHandlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
	h.tileCountX, h.tileCountY = TileCounts(width, height)
}

func (h *lightingHandlerImpl) GBufferHandler() GBufferHandler {
	return h.gBufferHandler
}

func (h *lightingHandlerImpl) SSAOHandler() SSAOHandler {
	return h.ssaoHandler
}

func (h *lightingHandlerImpl) ProbeGrid() IrradianceProbeGrid {
	return h.probeGrid
}

func (h *lightingHandlerImpl) CompositionHandler() CompositionHandler {
	return h.compositionHandler
}

func (h *lightingHandlerImpl) SSRHandler() SSRHandler {
	return h.ssrHandler
}
