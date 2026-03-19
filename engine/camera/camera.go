// Package camera provides the [Camera] and [CameraController] interfaces for
// managing perspective projection and input-driven camera movement.
//
// A [Camera] holds perspective settings (field of view, aspect ratio, near/far
// planes) and recomputes view and projection matrices each frame by reading
// position and target from its attached [CameraController]. The controller
// manages spatial state and supports multiple input modes such as orbit and
// planar movement. Instances are created via [NewCamera] and
// [NewCameraController] using functional options.
package camera

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// Camera defines the interface for the camera system.
// The camera holds perspective settings and computes view/projection matrices
// from an attached CameraController each frame via Update().
type Camera interface {
	common.Delegate[Camera]

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

var _ Camera = &camera{}

func (c *camera) Up() (x, y, z float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.up[0], c.up[1], c.up[2]
}

func (c *camera) Fov() float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.fov
}

func (c *camera) Aspect() float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.aspect
}

func (c *camera) Near() float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.near
}

func (c *camera) Far() float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.far
}

func (c *camera) ViewMatrix() [16]float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.viewMatrix
}

func (c *camera) ProjectionMatrix() [16]float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.projectionMatrix
}

func (c *camera) ViewProjectionMatrix() [16]float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.viewProjectionMatrix
}

func (c *camera) InverseProjectionMatrix() [16]float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.inverseProjectionMatrix
}

func (c *camera) SetUp(x, y, z float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.up = [3]float32{x, y, z}
	c.updateMatrices()
}

func (c *camera) SetFov(fov float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.fov = fov
	c.updateMatrices()
}

func (c *camera) SetAspect(aspect float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.aspect = aspect
	c.updateMatrices()
}

func (c *camera) SetNear(near float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.near = near
	c.updateMatrices()
}

func (c *camera) SetFar(far float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.far = far
	c.updateMatrices()
}

func (c *camera) Controller() CameraController {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.controller
}

func (c *camera) Update() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.controller == nil {
		return
	}
	c.updateMatrices()
}

func (c *camera) SetController(ctrl CameraController) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.controller = ctrl
}

func (c *camera) BindGroupProvider() bind_group_provider.BindGroupProvider {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.bindGroupProvider
}

func (c *camera) SetBindGroupProvider(provider bind_group_provider.BindGroupProvider) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.bindGroupProvider = provider
}
