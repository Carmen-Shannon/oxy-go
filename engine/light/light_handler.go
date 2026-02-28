package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// lightingHandlerImpl is the implementation of the LightingHandler interface.
type lightingHandlerImpl struct {
	enabled bool

	lights       []Light
	ambientColor [3]float32

	bgps         map[string]bind_group_provider.BindGroupProvider
	pipelineKeys map[string]string

	shadowDepthTexture     *wgpu.Texture
	shadowDepthTextureView *wgpu.TextureView
	shadowComparisonSamp   *wgpu.Sampler

	shadowHalfExtent      float32
	shadowNear            float32
	shadowFar             float32
	shadowBias            float32
	shadowNormalBiasScale float32
	shadowMapResolution   int

	screenWidth  int
	screenHeight int
	tileCountX   uint32
	tileCountY   uint32
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

	// ShadowDepthTexture returns the depth texture used for shadow mapping.
	//
	// Returns:
	//   - *wgpu.Texture: the shadow depth texture, or nil if not initialized
	ShadowDepthTexture() *wgpu.Texture

	// SetShadowDepthTexture sets the depth texture used for shadow mapping.
	//
	// Parameters:
	//   - t: the shadow depth texture
	SetShadowDepthTexture(t *wgpu.Texture)

	// ShadowDepthTextureView returns the texture view into the shadow depth texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the shadow depth texture view, or nil if not initialized
	ShadowDepthTextureView() *wgpu.TextureView

	// SetShadowDepthTextureView sets the texture view into the shadow depth texture.
	//
	// Parameters:
	//   - tv: the shadow depth texture view
	SetShadowDepthTextureView(tv *wgpu.TextureView)

	// ShadowComparisonSampler returns the comparison sampler used for PCF shadow sampling.
	//
	// Returns:
	//   - *wgpu.Sampler: the comparison sampler, or nil if not initialized
	ShadowComparisonSampler() *wgpu.Sampler

	// SetShadowComparisonSampler sets the comparison sampler used for PCF shadow sampling.
	//
	// Parameters:
	//   - s: the comparison sampler
	SetShadowComparisonSampler(s *wgpu.Sampler)

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
			"lights":      bind_group_provider.NewBindGroupProvider("lights"),
			"shadow_data": bind_group_provider.NewBindGroupProvider("shadow_data"),
			"shadow_lit":  bind_group_provider.NewBindGroupProvider("shadow_lit"),
			"light_cull":  bind_group_provider.NewBindGroupProvider("light_cull"),
			"tile_lit":    bind_group_provider.NewBindGroupProvider("tile_lit"),
		},
		pipelineKeys:          make(map[string]string),
		shadowHalfExtent:      DefaultShadowHalfExtent,
		shadowNear:            DefaultShadowNear,
		shadowFar:             DefaultShadowFar,
		shadowBias:            DefaultShadowBias,
		shadowNormalBiasScale: DefaultShadowNormalBiasScale,
		shadowMapResolution:   ShadowMapResolution,
	}
	for _, opt := range opts {
		opt(h)
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

func (h *lightingHandlerImpl) ShadowDepthTexture() *wgpu.Texture {
	return h.shadowDepthTexture
}

func (h *lightingHandlerImpl) SetShadowDepthTexture(t *wgpu.Texture) {
	h.shadowDepthTexture = t
}

func (h *lightingHandlerImpl) ShadowDepthTextureView() *wgpu.TextureView {
	return h.shadowDepthTextureView
}

func (h *lightingHandlerImpl) SetShadowDepthTextureView(tv *wgpu.TextureView) {
	h.shadowDepthTextureView = tv
}

func (h *lightingHandlerImpl) ShadowComparisonSampler() *wgpu.Sampler {
	return h.shadowComparisonSamp
}

func (h *lightingHandlerImpl) SetShadowComparisonSampler(s *wgpu.Sampler) {
	h.shadowComparisonSamp = s
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
