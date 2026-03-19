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

	common.Mul4(c.viewProjectionMatrix[:], c.projectionMatrix[:], c.viewMatrix[:])
	common.Invert4(c.inverseProjectionMatrix[:], c.projectionMatrix[:])
}
