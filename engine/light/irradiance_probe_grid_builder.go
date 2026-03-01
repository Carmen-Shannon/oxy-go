package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// IrradianceProbeGridOption is a functional option for configuring an
// IrradianceProbeGrid during construction via NewIrradianceProbeGrid.
type IrradianceProbeGridOption func(*irradianceProbeGridImpl)

// WithProbeGridCounts sets the number of probes in each axis.
//
// Parameters:
//   - x: the number of probes along the X axis
//   - y: the number of probes along the Y axis
//   - z: the number of probes along the Z axis
//
// Returns:
//   - IrradianceProbeGridOption: a function that applies the count option to an irradianceProbeGridImpl
func WithProbeGridCounts(x, y, z int) IrradianceProbeGridOption {
	return func(h *irradianceProbeGridImpl) {
		h.countX = x
		h.countY = y
		h.countZ = z
	}
}

// WithProbeGridBounds sets the world-space axis-aligned bounding box of the
// probe grid. Probes are placed uniformly between min and max on each axis.
//
// Parameters:
//   - min: the minimum corner (x, y, z)
//   - max: the maximum corner (x, y, z)
//
// Returns:
//   - IrradianceProbeGridOption: a function that applies the bounds option to an irradianceProbeGridImpl
func WithProbeGridBounds(min, max [3]float32) IrradianceProbeGridOption {
	return func(h *irradianceProbeGridImpl) {
		h.gridMin = min
		h.gridMax = max
	}
}

// WithProbeBakeResolution sets the cubemap face resolution used during probe
// baking. Each face is rendered at resolution × resolution pixels.
//
// Parameters:
//   - resolution: the cubemap face edge length in pixels (e.g. 32)
//
// Returns:
//   - IrradianceProbeGridOption: a function that applies the resolution option to an irradianceProbeGridImpl
func WithProbeBakeResolution(resolution int) IrradianceProbeGridOption {
	return func(h *irradianceProbeGridImpl) {
		h.bakeResolution = resolution
	}
}

// NewIrradianceProbeGrid creates a new IrradianceProbeGrid with sensible
// defaults and any provided options applied. After options are applied the
// per-axis spacing is derived from the grid bounds and counts, and the
// CPU-side probe array is pre-allocated with world-space positions set to
// each grid cell centre. GPU resources are not allocated until the owning
// scene calls the appropriate initialization methods.
//
// Defaults:
//   - Grid counts: 8 × 4 × 8
//   - Bounds: (-10, -2, -10) to (10, 6, 10)
//   - Bake resolution: 32 pixels per cubemap face edge
//
// Parameters:
//   - opts: variadic list of IrradianceProbeGridOption functions to configure the handler
//
// Returns:
//   - IrradianceProbeGrid: a new handler instance ready to be attached to a scene
func NewIrradianceProbeGrid(opts ...IrradianceProbeGridOption) IrradianceProbeGrid {
	h := &irradianceProbeGridImpl{
		enabled:        false,
		countX:         8,
		countY:         4,
		countZ:         8,
		gridMin:        [3]float32{-10, -2, -10},
		gridMax:        [3]float32{10, 6, 10},
		bakeResolution: 32,
		pipelineKeys:   make(map[string]string),
		bgps: map[string]bind_group_provider.BindGroupProvider{
			"probe_grid":        bind_group_provider.NewBindGroupProvider("probe_grid"),
			"probe_sh_project":  bind_group_provider.NewBindGroupProvider("probe_sh_project"),
			"probe_bake_camera": bind_group_provider.NewBindGroupProvider("probe_bake_camera"),
		},
	}
	for _, opt := range opts {
		opt(h)
	}

	// Derive per-axis spacing from bounds and counts.
	for i := 0; i < 3; i++ {
		counts := [3]int{h.countX, h.countY, h.countZ}
		if counts[i] > 1 {
			h.spacing[i] = (h.gridMax[i] - h.gridMin[i]) / float32(counts[i]-1)
		} else {
			h.spacing[i] = 0
		}
	}

	// Pre-allocate probe array with positions at grid cell centres.
	total := h.countX * h.countY * h.countZ
	h.probes = make([]GPUIrradianceProbe, total)
	for z := 0; z < h.countZ; z++ {
		for y := 0; y < h.countY; y++ {
			for x := 0; x < h.countX; x++ {
				idx := x + y*h.countX + z*h.countX*h.countY
				h.probes[idx].Position = [4]float32{
					h.gridMin[0] + float32(x)*h.spacing[0],
					h.gridMin[1] + float32(y)*h.spacing[1],
					h.gridMin[2] + float32(z)*h.spacing[2],
					1.0, // active status
				}
			}
		}
	}

	// All probes start dirty (need initial bake).
	h.dirtyProbes = make([]int, total)
	for i := 0; i < total; i++ {
		h.dirtyProbes[i] = i
	}

	return h
}
