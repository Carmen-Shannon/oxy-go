package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// ShadowType identifies the kind of shadow map entry.
type ShadowType uint32

const (
	// ShadowTypeSpot identifies a spot light shadow entry.
	ShadowTypeSpot ShadowType = iota
	// ShadowTypeCubeFace identifies a cube shadow map face entry for point light shadows.
	ShadowTypeCubeFace
)

// ShadowHandler defines the interface for the scene's shadow-mapping subsystem.
//
// The ShadowHandler manages shadow frustum configuration, PCF quality parameters,
// CSM cascade settings, and all associated GPU resources (atlas textures,
// auxiliary depth views, bind group providers, pipeline keys). It is
// created via NewShadowHandler with builder options and attached to a scene's
// LightingHandler via NewLightingHandler or WithShadowHandler. GPU resources are
// initialized lazily by the owning scene when shadow mapping is first enabled.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type ShadowHandler interface {
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

	// PCFRadius returns the Poisson disk PCF kernel radius in texels.
	//
	// Returns:
	//   - float32: the PCF radius
	PCFRadius() float32

	// PCFSamples returns the Poisson disk tap count used for PCF shadow filtering.
	//
	// Returns:
	//   - uint32: the PCF sample count
	PCFSamples() uint32

	// PCFSamplesSpot returns the Poisson disk tap count used for PCF shadow
	// filtering of spot and point lights.
	//
	// Returns:
	//   - uint32: the spot/point PCF sample count
	PCFSamplesSpot() uint32

	// ComparisonSampler returns the comparison sampler used for depth shadow
	// map lookups in the lit fragment shader.
	//
	// Returns:
	//   - *wgpu.Sampler: the comparison sampler, or nil if not initialized
	ComparisonSampler() *wgpu.Sampler

	// SetComparisonSampler sets the comparison sampler for depth shadow maps.
	//
	// Parameters:
	//   - s: the comparison sampler
	SetComparisonSampler(s *wgpu.Sampler)

	// CascadeCount returns the fixed number of shadow cascades (always 2).
	//
	// Returns:
	//   - int: the cascade count (always 2)
	CascadeCount() int

	// ShadowInnerRadius returns the world-space radius of the high-fidelity inner
	// shadow cascade centered on the camera.
	//
	// Returns:
	//   - float32: the inner cascade radius in world units
	ShadowInnerRadius() float32

	// CSMAtlasTexture returns the CSM atlas texture.
	//
	// Returns:
	//   - *wgpu.Texture: the atlas texture, or nil if not initialized
	CSMAtlasTexture() *wgpu.Texture

	// SetCSMAtlasTexture sets the CSM atlas texture.
	//
	// Parameters:
	//   - t: the atlas texture
	SetCSMAtlasTexture(t *wgpu.Texture)

	// CSMAtlasTextureView returns the texture view for the CSM atlas texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the atlas texture view, or nil if not initialized
	CSMAtlasTextureView() *wgpu.TextureView

	// SetCSMAtlasTextureView sets the texture view for the CSM atlas texture.
	//
	// Parameters:
	//   - tv: the atlas texture view
	SetCSMAtlasTextureView(tv *wgpu.TextureView)

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

	// LightShadowAtlasSlots returns the computed number of atlas slots allocated
	// for per-light (spot/point) shadow entries.
	//
	// Returns:
	//   - int: the number of allocated atlas slots
	LightShadowAtlasSlots() int

	// SetLightShadowAtlasSlots stores the computed atlas slot count.
	//
	// Parameters:
	//   - n: the number of atlas slots
	SetLightShadowAtlasSlots(n int)

	// LightShadowAtlasCols returns the number of tile columns in the spot/point shadow atlas grid.
	//
	// Returns:
	//   - int: the number of tile columns
	LightShadowAtlasCols() int

	// SetLightShadowAtlasCols stores the number of tile columns in the shadow atlas grid.
	//
	// Parameters:
	//   - n: the number of tile columns
	SetLightShadowAtlasCols(n int)

	// LightShadowAtlas returns the shadow atlas texture for spot/point lights.
	//
	// Returns:
	//   - *wgpu.Texture: the atlas texture, or nil if not initialized
	LightShadowAtlas() *wgpu.Texture

	// SetLightShadowAtlas sets the shadow atlas texture for spot/point lights.
	//
	// Parameters:
	//   - t: the atlas texture
	SetLightShadowAtlas(t *wgpu.Texture)

	// LightShadowAtlasView returns the texture view for the spot/point shadow atlas.
	//
	// Returns:
	//   - *wgpu.TextureView: the atlas texture view, or nil if not initialized
	LightShadowAtlasView() *wgpu.TextureView

	// SetLightShadowAtlasView sets the texture view for the spot/point shadow atlas.
	//
	// Parameters:
	//   - tv: the atlas texture view
	SetLightShadowAtlasView(tv *wgpu.TextureView)

	// LightShadowTileSize returns the width/height in texels of each tile in the
	// per-light shadow atlas.
	//
	// Returns:
	//   - int: the tile size in texels
	LightShadowTileSize() int

	// CheckAndMarkDirty compares the light's current fields to the stored snapshot.
	// Returns true if the light is dirty (no snapshot, fields changed, or externally
	// marked dirty). The dirty flag is set internally; call CommitSnapshot after
	// rendering to clear it.
	//
	// Parameters:
	//   - l: the light to check
	//
	// Returns:
	//   - bool: true if the light requires a depth re-render this frame
	CheckAndMarkDirty(l Light) bool

	// MarkAllDirty marks all currently-tracked lights as dirty. Called when any
	// skeletal shadow-caster is active in the scene, ensuring animated geometry
	// changes are reflected in all spot/point shadow maps.
	MarkAllDirty()

	// CommitSnapshot stores the light's current field values as the accepted
	// snapshot and clears the dirty flag. Call after a successful depth render.
	//
	// Parameters:
	//   - l: the light whose snapshot to commit
	CommitSnapshot(l Light)

	// OnLightRemoved cleans up all snapshot and dirty-flag state for a removed
	// light, preventing stale entries from accumulating over time.
	//
	// Parameters:
	//   - l: the light that was removed from the scene
	OnLightRemoved(l Light)

	// ForceMarkDirty unconditionally marks the light as requiring a depth re-render
	// this frame. Called by the scene when a light's atlas slot index migrates between
	// frames due to a neighbour light entering or leaving the shadow slot list.
	//
	// Parameters:
	//   - l: the light to force-dirty
	ForceMarkDirty(l Light)
}

var _ ShadowHandler = &shadowHandlerImpl{}

func (h *shadowHandlerImpl) ShadowNear() float32                         { return h.shadowNear }
func (h *shadowHandlerImpl) ShadowFar() float32                          { return h.shadowFar }
func (h *shadowHandlerImpl) ShadowNormalBiasScale() float32              { return h.shadowNormalBiasScale }
func (h *shadowHandlerImpl) ShadowMapResolution() int                    { return h.shadowMapResolution }
func (h *shadowHandlerImpl) PCFRadius() float32                          { return h.pcfRadius }
func (h *shadowHandlerImpl) PCFSamples() uint32                          { return h.pcfSamples }
func (h *shadowHandlerImpl) PCFSamplesSpot() uint32                      { return h.pcfSamplesSpot }
func (h *shadowHandlerImpl) ComparisonSampler() *wgpu.Sampler            { return h.comparisonSampler }
func (h *shadowHandlerImpl) SetComparisonSampler(s *wgpu.Sampler)        { h.comparisonSampler = s }
func (h *shadowHandlerImpl) CascadeCount() int                           { return 2 }
func (h *shadowHandlerImpl) ShadowInnerRadius() float32                  { return h.shadowInnerRadius }
func (h *shadowHandlerImpl) CSMAtlasTexture() *wgpu.Texture              { return h.csmAtlasTexture }
func (h *shadowHandlerImpl) SetCSMAtlasTexture(t *wgpu.Texture)          { h.csmAtlasTexture = t }
func (h *shadowHandlerImpl) CSMAtlasTextureView() *wgpu.TextureView      { return h.csmAtlasTextureView }
func (h *shadowHandlerImpl) PipelineKey(name string) string              { return h.pipelineKeys[name] }
func (h *shadowHandlerImpl) PipelineKeys() map[string]string             { return h.pipelineKeys }
func (h *shadowHandlerImpl) SetPipelineKey(name, key string)             { h.pipelineKeys[name] = key }
func (h *shadowHandlerImpl) LightShadowAtlasSlots() int                  { return h.lightShadowAtlasSlots }
func (h *shadowHandlerImpl) SetLightShadowAtlasSlots(n int)              { h.lightShadowAtlasSlots = n }
func (h *shadowHandlerImpl) LightShadowAtlasCols() int                   { return h.lightShadowAtlasCols }
func (h *shadowHandlerImpl) SetLightShadowAtlasCols(n int)               { h.lightShadowAtlasCols = n }
func (h *shadowHandlerImpl) LightShadowAtlas() *wgpu.Texture             { return h.lightShadowAtlas }
func (h *shadowHandlerImpl) SetLightShadowAtlas(t *wgpu.Texture)         { h.lightShadowAtlas = t }
func (h *shadowHandlerImpl) LightShadowAtlasView() *wgpu.TextureView     { return h.lightShadowAtlasView }
func (h *shadowHandlerImpl) LightShadowTileSize() int                    { return h.lightShadowTileSize }
func (h *shadowHandlerImpl) SetCSMAtlasTextureView(tv *wgpu.TextureView) { h.csmAtlasTextureView = tv }

func (h *shadowHandlerImpl) SetLightShadowAtlasView(tv *wgpu.TextureView) {
	h.lightShadowAtlasView = tv
}

func (h *shadowHandlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return h.bgps[key]
}

func (h *shadowHandlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}

func (h *shadowHandlerImpl) SetBgp(key string, bgp bind_group_provider.BindGroupProvider) {
	h.bgps[key] = bgp
}
