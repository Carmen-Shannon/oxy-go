package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// shadowHandlerImpl is the implementation of the ShadowHandler interface.
type shadowHandlerImpl struct {
	// Shadow frustum configuration.
	shadowNear            float32
	shadowFar             float32
	shadowNormalBiasScale float32
	shadowMapResolution   int

	// PCF quality parameters.
	pcfRadius  float32
	pcfSamples uint32

	// Comparison sampler for depth shadow maps.
	comparisonSampler *wgpu.Sampler

	// CSM configuration.
	shadowInnerRadius float32

	// CSM GPU resources.
	csmAtlasTexture     *wgpu.Texture
	csmAtlasTextureView *wgpu.TextureView

	// Per-light (spot/point) shadow atlas resources.
	lightShadowAtlasSlots int
	lightShadowAtlasCols  int
	lightShadowAtlas      *wgpu.Texture
	lightShadowAtlasView  *wgpu.TextureView
	lightShadowTileSize   int

	bgps         map[string]bind_group_provider.BindGroupProvider
	pipelineKeys map[string]string
}
