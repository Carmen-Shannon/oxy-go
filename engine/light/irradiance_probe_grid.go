package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// irradianceProbeGridImpl is the implementation of the IrradianceProbeGrid interface.
type irradianceProbeGridImpl struct {
	enabled bool

	// Grid layout.
	countX, countY, countZ int
	gridMin                [3]float32
	gridMax                [3]float32
	spacing                [3]float32

	// Cubemap baking resolution (pixels per face edge).
	bakeResolution int

	// CPU-side probe data. Every probe stores its world-space position and
	// L2 SH coefficients in the format expected by GPUIrradianceProbe.
	probes []GPUIrradianceProbe

	// Indices of probes that need re-baking. This list is consumed by the
	// scene's incremental bake loop and cleared after each bake pass.
	dirtyProbes []int

	// GPU-side resources.
	probeBuffer      *wgpu.Buffer // storage buffer holding the full probe array
	gridParamsBuffer *wgpu.Buffer // uniform buffer holding GPUProbeGridParams

	pipelineKeys map[string]string
	bgps         map[string]bind_group_provider.BindGroupProvider

	// Cubemap bake render targets (reused for every probe/face).
	bakeColorTexture     *wgpu.Texture
	bakeColorTextureView *wgpu.TextureView
	bakeDepthTexture     *wgpu.Texture
	bakeDepthTextureView *wgpu.TextureView
}

// IrradianceProbeGrid defines the interface for the scene's irradiance probe
// grid subsystem.
//
// The probe grid stores a regular 3-D grid of irradiance probes, each
// containing L2 spherical harmonic coefficients that encode low-frequency
// indirect illumination sampled from the surrounding scene. During probe
// baking, the scene is rendered from each probe position into a tiny cubemap
// (6 faces at the configured resolution), then the cubemap is projected into
// SH coefficients via a compute shader. The resulting SH data is uploaded to
// a GPU storage buffer and sampled in the lit fragment shader for per-pixel
// diffuse indirect lighting via trilinear probe interpolation.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type IrradianceProbeGrid interface {
	// Enabled returns whether the probe grid subsystem has been GPU-initialized
	// and is ready for rendering.
	//
	// Returns:
	//   - bool: true if probe grid GPU resources have been initialized
	Enabled() bool

	// SetEnabled sets whether the probe grid subsystem is GPU-initialized.
	//
	// Parameters:
	//   - enabled: true to mark as initialized
	SetEnabled(enabled bool)

	// CountX returns the number of probes along the X axis.
	//
	// Returns:
	//   - int: the probe count along X
	CountX() int

	// CountY returns the number of probes along the Y axis.
	//
	// Returns:
	//   - int: the probe count along Y
	CountY() int

	// CountZ returns the number of probes along the Z axis.
	//
	// Returns:
	//   - int: the probe count along Z
	CountZ() int

	// TotalProbes returns the total number of probes in the grid (X × Y × Z).
	//
	// Returns:
	//   - int: the total probe count
	TotalProbes() int

	// GridMin returns the world-space minimum corner of the probe grid.
	//
	// Returns:
	//   - [3]float32: the minimum corner (x, y, z)
	GridMin() [3]float32

	// GridMax returns the world-space maximum corner of the probe grid.
	//
	// Returns:
	//   - [3]float32: the maximum corner (x, y, z)
	GridMax() [3]float32

	// Spacing returns the world-space distance between adjacent probes per axis.
	//
	// Returns:
	//   - [3]float32: the spacing (x, y, z)
	Spacing() [3]float32

	// BakeResolution returns the cubemap face resolution in pixels per edge.
	//
	// Returns:
	//   - int: the cubemap face resolution
	BakeResolution() int

	// ProbeIndex computes the flat array index for the probe at grid position
	// (x, y, z). The layout is: index = x + y*countX + z*countX*countY.
	//
	// Parameters:
	//   - x: the X grid coordinate
	//   - y: the Y grid coordinate
	//   - z: the Z grid coordinate
	//
	// Returns:
	//   - int: the flat array index
	ProbeIndex(x, y, z int) int

	// Probe returns a copy of the probe at the given flat index.
	//
	// Parameters:
	//   - index: the flat probe index
	//
	// Returns:
	//   - GPUIrradianceProbe: the probe data
	Probe(index int) GPUIrradianceProbe

	// Probes returns a reference to the full slice of CPU-side probe data.
	//
	// Returns:
	//   - []GPUIrradianceProbe: all probes in the grid
	Probes() []GPUIrradianceProbe

	// SetProbe writes the given probe data at the specified flat index and
	// marks the probe as dirty for re-baking.
	//
	// Parameters:
	//   - index: the flat probe index
	//   - p: the probe data to set
	SetProbe(index int, p GPUIrradianceProbe)

	// SetProbes replaces the entire CPU-side probe slice.
	//
	// Parameters:
	//   - probes: the new probe data
	SetProbes(probes []GPUIrradianceProbe)

	// DirtyProbes returns the indices of probes that need re-baking.
	//
	// Returns:
	//   - []int: dirty probe indices
	DirtyProbes() []int

	// SetDirtyProbes replaces the dirty probe index list.
	//
	// Parameters:
	//   - indices: the new list of dirty probe indices
	SetDirtyProbes(indices []int)

	// MarkAllDirty adds every probe index to the dirty list.
	MarkAllDirty()

	// ClearDirtyProbes empties the dirty probe list.
	ClearDirtyProbes()

	// ProbeBuffer returns the GPU storage buffer holding the serialized probe array.
	//
	// Returns:
	//   - *wgpu.Buffer: the probe storage buffer, or nil if not initialized
	ProbeBuffer() *wgpu.Buffer

	// SetProbeBuffer sets the GPU storage buffer for probes.
	//
	// Parameters:
	//   - buf: the probe storage buffer
	SetProbeBuffer(buf *wgpu.Buffer)

	// GridParamsBuffer returns the GPU uniform buffer holding the serialized
	// GPUProbeGridParams data.
	//
	// Returns:
	//   - *wgpu.Buffer: the grid params uniform buffer, or nil if not initialized
	GridParamsBuffer() *wgpu.Buffer

	// SetGridParamsBuffer sets the GPU uniform buffer for grid parameters.
	//
	// Parameters:
	//   - buf: the grid params uniform buffer
	SetGridParamsBuffer(buf *wgpu.Buffer)

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
	// Valid keys:
	//   - "probe_grid": lit shader bind group (probe buffer + grid params)
	//   - "probe_sh_project": SH projection compute shader bind group
	//   - "probe_bake_camera": bake pass camera uniform bind group
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

	// BakeColorTexture returns the RGBA8Unorm texture used as the color
	// attachment during cubemap face baking.
	//
	// Returns:
	//   - *wgpu.Texture: the bake color texture, or nil if not initialized
	BakeColorTexture() *wgpu.Texture

	// SetBakeColorTexture sets the cubemap bake color texture.
	//
	// Parameters:
	//   - t: the bake color texture
	SetBakeColorTexture(t *wgpu.Texture)

	// BakeColorTextureView returns the texture view for the bake color texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the bake color texture view, or nil if not initialized
	BakeColorTextureView() *wgpu.TextureView

	// SetBakeColorTextureView sets the texture view for the bake color texture.
	//
	// Parameters:
	//   - tv: the bake color texture view
	SetBakeColorTextureView(tv *wgpu.TextureView)

	// BakeDepthTexture returns the depth texture used during cubemap face baking.
	//
	// Returns:
	//   - *wgpu.Texture: the bake depth texture, or nil if not initialized
	BakeDepthTexture() *wgpu.Texture

	// SetBakeDepthTexture sets the cubemap bake depth texture.
	//
	// Parameters:
	//   - t: the bake depth texture
	SetBakeDepthTexture(t *wgpu.Texture)

	// BakeDepthTextureView returns the texture view for the bake depth texture.
	//
	// Returns:
	//   - *wgpu.TextureView: the bake depth texture view, or nil if not initialized
	BakeDepthTextureView() *wgpu.TextureView

	// SetBakeDepthTextureView sets the texture view for the bake depth texture.
	//
	// Parameters:
	//   - tv: the bake depth texture view
	SetBakeDepthTextureView(tv *wgpu.TextureView)

	// BuildGPUGridParams computes and returns the GPUProbeGridParams uniform
	// data matching the current grid configuration.
	//
	// Returns:
	//   - GPUProbeGridParams: the serializable grid parameter block
	BuildGPUGridParams() GPUProbeGridParams
}

var _ IrradianceProbeGrid = &irradianceProbeGridImpl{}

func (h *irradianceProbeGridImpl) Enabled() bool {
	return h.enabled
}

func (h *irradianceProbeGridImpl) SetEnabled(enabled bool) {
	h.enabled = enabled
}

func (h *irradianceProbeGridImpl) CountX() int {
	return h.countX
}

func (h *irradianceProbeGridImpl) CountY() int {
	return h.countY
}

func (h *irradianceProbeGridImpl) CountZ() int {
	return h.countZ
}

func (h *irradianceProbeGridImpl) TotalProbes() int {
	return h.countX * h.countY * h.countZ
}

func (h *irradianceProbeGridImpl) GridMin() [3]float32 {
	return h.gridMin
}

func (h *irradianceProbeGridImpl) GridMax() [3]float32 {
	return h.gridMax
}

func (h *irradianceProbeGridImpl) Spacing() [3]float32 {
	return h.spacing
}

func (h *irradianceProbeGridImpl) BakeResolution() int {
	return h.bakeResolution
}

func (h *irradianceProbeGridImpl) ProbeIndex(x, y, z int) int {
	return x + y*h.countX + z*h.countX*h.countY
}

func (h *irradianceProbeGridImpl) Probe(index int) GPUIrradianceProbe {
	return h.probes[index]
}

func (h *irradianceProbeGridImpl) Probes() []GPUIrradianceProbe {
	return h.probes
}

func (h *irradianceProbeGridImpl) SetProbe(index int, p GPUIrradianceProbe) {
	h.probes[index] = p
	h.dirtyProbes = append(h.dirtyProbes, index)
}

func (h *irradianceProbeGridImpl) SetProbes(probes []GPUIrradianceProbe) {
	h.probes = probes
}

func (h *irradianceProbeGridImpl) DirtyProbes() []int {
	return h.dirtyProbes
}

func (h *irradianceProbeGridImpl) SetDirtyProbes(indices []int) {
	h.dirtyProbes = indices
}

func (h *irradianceProbeGridImpl) MarkAllDirty() {
	total := h.TotalProbes()
	h.dirtyProbes = make([]int, total)
	for i := 0; i < total; i++ {
		h.dirtyProbes[i] = i
	}
}

func (h *irradianceProbeGridImpl) ClearDirtyProbes() {
	h.dirtyProbes = h.dirtyProbes[:0]
}

func (h *irradianceProbeGridImpl) ProbeBuffer() *wgpu.Buffer {
	return h.probeBuffer
}

func (h *irradianceProbeGridImpl) SetProbeBuffer(buf *wgpu.Buffer) {
	h.probeBuffer = buf
}

func (h *irradianceProbeGridImpl) GridParamsBuffer() *wgpu.Buffer {
	return h.gridParamsBuffer
}

func (h *irradianceProbeGridImpl) SetGridParamsBuffer(buf *wgpu.Buffer) {
	h.gridParamsBuffer = buf
}

func (h *irradianceProbeGridImpl) PipelineKey(name string) string {
	return h.pipelineKeys[name]
}

func (h *irradianceProbeGridImpl) PipelineKeys() map[string]string {
	return h.pipelineKeys
}

func (h *irradianceProbeGridImpl) SetPipelineKey(name, key string) {
	h.pipelineKeys[name] = key
}

func (h *irradianceProbeGridImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return h.bgps[key]
}

func (h *irradianceProbeGridImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return h.bgps
}

func (h *irradianceProbeGridImpl) SetBgp(key string, bgp bind_group_provider.BindGroupProvider) {
	h.bgps[key] = bgp
}

func (h *irradianceProbeGridImpl) BakeColorTexture() *wgpu.Texture {
	return h.bakeColorTexture
}

func (h *irradianceProbeGridImpl) SetBakeColorTexture(t *wgpu.Texture) {
	h.bakeColorTexture = t
}

func (h *irradianceProbeGridImpl) BakeColorTextureView() *wgpu.TextureView {
	return h.bakeColorTextureView
}

func (h *irradianceProbeGridImpl) SetBakeColorTextureView(tv *wgpu.TextureView) {
	h.bakeColorTextureView = tv
}

func (h *irradianceProbeGridImpl) BakeDepthTexture() *wgpu.Texture {
	return h.bakeDepthTexture
}

func (h *irradianceProbeGridImpl) SetBakeDepthTexture(t *wgpu.Texture) {
	h.bakeDepthTexture = t
}

func (h *irradianceProbeGridImpl) BakeDepthTextureView() *wgpu.TextureView {
	return h.bakeDepthTextureView
}

func (h *irradianceProbeGridImpl) SetBakeDepthTextureView(tv *wgpu.TextureView) {
	h.bakeDepthTextureView = tv
}

func (h *irradianceProbeGridImpl) BuildGPUGridParams() GPUProbeGridParams {
	return GPUProbeGridParams{
		GridMin:     h.gridMin,
		ProbeCountX: uint32(h.countX),
		GridMax:     h.gridMax,
		ProbeCountY: uint32(h.countY),
		Spacing:     h.spacing,
		ProbeCountZ: uint32(h.countZ),
		TotalProbes: uint32(h.TotalProbes()),
	}
}
