package camera

import (
	"sync"
	"sync/atomic"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// cameraCount is an atomic counter used to generate unique bind group provider names for each camera instance.
var cameraCount atomic.Uint64

type camera struct {
	common.DelegateImpl[Camera]

	mu *sync.Mutex

	up [3]float32

	fov    float32
	aspect float32
	near   float32
	far    float32

	viewMatrix              [16]float32
	projectionMatrix        [16]float32
	viewProjectionMatrix    [16]float32
	inverseProjectionMatrix [16]float32

	jitterX, jitterY         float32
	prevViewProjectionMatrix [16]float32

	controller        CameraController
	bindGroupProvider bind_group_provider.BindGroupProvider
}

// updateMatrices recalculates the view, projection, view-projection, and inverse projection matrices.
// It reads position and target from the attached controller. This is a no-op when the controller is nil.
// Caller must hold the mutex.
func (c *camera) updateMatrices() {
	if c.controller == nil {
		return
	}

	// Save the current jittered VP as "previous" before overwriting.
	copy(c.prevViewProjectionMatrix[:], c.viewProjectionMatrix[:])

	px, py, pz := c.controller.Position()
	tx, ty, tz := c.controller.Target()

	common.LookAt(c.viewMatrix[:],
		px, py, pz,
		tx, ty, tz,
		c.up[0], c.up[1], c.up[2],
	)

	common.Perspective(c.projectionMatrix[:],
		c.fov, c.aspect, c.near, c.far,
	)

	// Apply sub-pixel jitter to clip-space translation elements.
	// projectionMatrix[8] and [9] are the X and Y translation elements
	// (column 2, rows 0 and 1 in column-major storage).
	if c.jitterX != 0 || c.jitterY != 0 {
		c.projectionMatrix[8] += c.jitterX
		c.projectionMatrix[9] += c.jitterY
	}

	common.Mul4(c.viewProjectionMatrix[:], c.projectionMatrix[:], c.viewMatrix[:])
	common.Invert4(c.inverseProjectionMatrix[:], c.projectionMatrix[:])
}
