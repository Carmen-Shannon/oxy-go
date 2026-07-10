package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/oliverbestmann/webgpu/wgpu"
)

// lightSnapshot captures the shadow-relevant fields of a Light at a point in
// time. It is compared against the live light state to detect changes that
// require a depth re-render.
type lightSnapshot struct {
	position     [3]float32
	direction    [3]float32
	lightRange   float32
	shadowBias   float32
	enabled      bool
	castsShadows bool
	innerCone    float32
	outerCone    float32
}

// snapshotOf builds a lightSnapshot from the current state of the given light.
func snapshotOf(l Light) lightSnapshot {
	p := l.Position()
	d := l.Direction()
	return lightSnapshot{
		position:     p,
		direction:    d,
		lightRange:   l.Range(),
		shadowBias:   l.ShadowBias(),
		enabled:      l.Enabled(),
		castsShadows: l.CastsShadows(),
		innerCone:    l.InnerCone(),
		outerCone:    l.OuterCone(),
	}
}

// shadowHandlerImpl is the implementation of the ShadowHandler interface.
type shadowHandlerImpl struct {
	// Shadow frustum configuration.
	shadowNear            float32
	shadowFar             float32
	shadowNormalBiasScale float32
	shadowMapResolution   int

	// PCF quality parameters.
	pcfRadius      float32
	pcfSamples     uint32
	pcfSamplesSpot uint32

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

	// Shadow-caching dirty-flag state. snapshots stores the last-committed
	// field values per light; dirtyFlags tracks whether a re-render is needed.
	snapshots  map[Light]lightSnapshot
	dirtyFlags map[Light]bool

	bgps         map[string]bind_group_provider.BindGroupProvider
	pipelineKeys map[string]string
}

func (h *shadowHandlerImpl) CheckAndMarkDirty(l Light) bool {
	live := snapshotOf(l)
	stored, exists := h.snapshots[l]
	if !exists || live != stored {
		h.dirtyFlags[l] = true
	}
	return h.dirtyFlags[l]
}

func (h *shadowHandlerImpl) MarkAllDirty() {
	for l := range h.snapshots {
		h.dirtyFlags[l] = true
	}
}

func (h *shadowHandlerImpl) CommitSnapshot(l Light) {
	h.snapshots[l] = snapshotOf(l)
	h.dirtyFlags[l] = false
}

func (h *shadowHandlerImpl) OnLightRemoved(l Light) {
	delete(h.snapshots, l)
	delete(h.dirtyFlags, l)
}

func (h *shadowHandlerImpl) ForceMarkDirty(l Light) {
	h.dirtyFlags[l] = true
}
