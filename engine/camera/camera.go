package camera

import (
	"math"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// cameraCount is an atomic counter used to generate unique bind group provider names for each camera instance.
var cameraCount atomic.Uint64

type cameraImpl struct {
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

// Camera defines the interface for the camera system.
// The camera holds perspective settings and computes view/projection matrices
// from an attached CameraController each frame via Update().
type Camera interface {
	// Up returns the camera's up vector.
	//
	// Returns:
	//   - x, y, z: up vector components
	Up() (x, y, z float32)

	// Fov returns the field of view in radians.
	//
	// Returns:
	//   - float32: field of view in radians
	Fov() float32

	// Aspect returns the aspect ratio (width / height).
	//
	// Returns:
	//   - float32: the aspect ratio
	Aspect() float32

	// Near returns the near clipping plane distance.
	//
	// Returns:
	//   - float32: near plane distance
	Near() float32

	// Far returns the far clipping plane distance.
	//
	// Returns:
	//   - float32: far plane distance
	Far() float32

	// ViewMatrix returns the current 4x4 view matrix as 16 floats (column-major).
	//
	// Returns:
	//   - [16]float32: the view matrix
	ViewMatrix() [16]float32

	// ProjectionMatrix returns the current 4x4 projection matrix as 16 floats (column-major).
	//
	// Returns:
	//   - [16]float32: the projection matrix
	ProjectionMatrix() [16]float32

	// ViewProjectionMatrix returns the current combined view-projection matrix as 16 floats (column-major).
	//
	// Returns:
	//   - [16]float32: the combined view-projection matrix
	ViewProjectionMatrix() [16]float32

	// InverseProjectionMatrix returns the inverse of the current projection matrix
	// as 16 floats (column-major). Used by the Forward+ light culling compute shader
	// to reconstruct per-tile view-space frustum planes from screen coordinates.
	//
	// Returns:
	//   - [16]float32: the inverse projection matrix
	InverseProjectionMatrix() [16]float32

	// Controller returns the attached CameraController.
	// Returns nil if no controller is attached.
	//
	// Returns:
	//   - CameraController: the attached controller or nil
	Controller() CameraController

	// BindGroupProvider returns the camera's bind group provider for GPU resources.
	// Returns nil if not set.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the bind group provider or nil
	BindGroupProvider() bind_group_provider.BindGroupProvider

	// Update reads position/target from controller and recomputes matrices.
	// Should be called once per frame (typically in the tick callback).
	// If no controller is attached, this method does nothing.
	Update()

	// SetUp sets the camera's up vector.
	//
	// Parameters:
	//   - x, y, z: up vector components
	SetUp(x, y, z float32)

	// SetFov sets the field of view in radians and recomputes matrices.
	//
	// Parameters:
	//   - fov: field of view in radians
	SetFov(fov float32)

	// SetAspect sets the aspect ratio (width / height) and recomputes matrices.
	//
	// Parameters:
	//   - aspect: the aspect ratio
	SetAspect(aspect float32)

	// SetNear sets the near clipping plane distance and recomputes matrices.
	//
	// Parameters:
	//   - near: near plane distance
	SetNear(near float32)

	// SetFar sets the far clipping plane distance and recomputes matrices.
	//
	// Parameters:
	//   - far: far plane distance
	SetFar(far float32)

	// SetController attaches a CameraController to the camera.
	//
	// Parameters:
	//   - ctrl: the controller to attach
	SetController(ctrl CameraController)

	// SetBindGroupProvider sets the camera's bind group provider.
	//
	// Parameters:
	//   - provider: the bind group provider to set
	SetBindGroupProvider(provider bind_group_provider.BindGroupProvider)
}

var _ Camera = &cameraImpl{}

// NewCamera creates a new Camera with default perspective settings.
// A controller must be attached via SetController or WithController option
// before position/target data is available.
//
// Parameters:
//   - options: functional options to configure the camera
//
// Returns:
//   - Camera: the newly created camera
func NewCamera(options ...CameraBuilderOption) Camera {
	c := &cameraImpl{
		mu:                   &sync.Mutex{},
		up:                   [3]float32{0, 1, 0},
		fov:                  45.0 * (math.Pi / 180.0), // radians
		aspect:               1.0,
		near:                 0.1,
		far:                  100.0,
		viewMatrix:           [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
		projectionMatrix:     [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
		viewProjectionMatrix: [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
		bindGroupProvider: bind_group_provider.NewBindGroupProvider(
			"camera_" + strconv.FormatUint(cameraCount.Load(), 10),
		),
	}
	for _, option := range options {
		option(c)
	}
	if c.controller != nil {
		c.updateMatrices()
	}
	cameraCount.Add(1)
	return c
}

func (c *cameraImpl) Up() (x, y, z float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.up[0], c.up[1], c.up[2]
}

func (c *cameraImpl) Fov() float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.fov
}

func (c *cameraImpl) Aspect() float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.aspect
}

func (c *cameraImpl) Near() float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.near
}

func (c *cameraImpl) Far() float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.far
}

func (c *cameraImpl) ViewMatrix() [16]float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.viewMatrix
}

func (c *cameraImpl) ProjectionMatrix() [16]float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.projectionMatrix
}

func (c *cameraImpl) ViewProjectionMatrix() [16]float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.viewProjectionMatrix
}

func (c *cameraImpl) InverseProjectionMatrix() [16]float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.inverseProjectionMatrix
}

func (c *cameraImpl) SetUp(x, y, z float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.up = [3]float32{x, y, z}
	c.updateMatrices()
}

func (c *cameraImpl) SetFov(fov float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.fov = fov
	c.updateMatrices()
}

func (c *cameraImpl) SetAspect(aspect float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.aspect = aspect
	c.updateMatrices()
}

func (c *cameraImpl) SetNear(near float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.near = near
	c.updateMatrices()
}

func (c *cameraImpl) SetFar(far float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.far = far
	c.updateMatrices()
}

func (c *cameraImpl) Controller() CameraController {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.controller
}

func (c *cameraImpl) Update() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.controller == nil {
		return
	}
	c.updateMatrices()
}

func (c *cameraImpl) SetController(ctrl CameraController) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.controller = ctrl
}

func (c *cameraImpl) BindGroupProvider() bind_group_provider.BindGroupProvider {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.bindGroupProvider
}

func (c *cameraImpl) SetBindGroupProvider(provider bind_group_provider.BindGroupProvider) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.bindGroupProvider = provider
}

// updateMatrices recalculates the view, projection, view-projection, and inverse projection matrices.
// It reads position and target from the attached controller. This is a no-op when the controller is nil.
// Caller must hold the mutex.
func (c *cameraImpl) updateMatrices() {
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
