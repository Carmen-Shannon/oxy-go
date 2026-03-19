package camera

import (
	"math"
	"strconv"
	"sync"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

type CameraBuilderOption func(*camera)

// WithUp sets the camera's up vector.
//
// Parameters:
//   - x, y, z: up vector components
//
// Returns:
//   - CameraBuilderOption: a function that sets the camera's up vector
func WithUp(x, y, z float32) CameraBuilderOption {
	return func(c *camera) {
		c.up = [3]float32{x, y, z}
		c.updateMatrices()
	}
}

// WithFov sets the camera's field of view in radians.
//
// Parameters:
//   - fov: field of view in radians
//
// Returns:
//   - CameraBuilderOption: a function that sets the camera's field of view
func WithFov(fov float32) CameraBuilderOption {
	return func(c *camera) {
		c.fov = fov
		c.updateMatrices()
	}
}

// WithAspect sets the camera's aspect ratio (width / height).
//
// Parameters:
//   - aspect: the aspect ratio to set
//
// Returns:
//   - CameraBuilderOption: a function that sets the camera's aspect ratio
func WithAspect(aspect float32) CameraBuilderOption {
	return func(c *camera) {
		c.aspect = aspect
		c.updateMatrices()
	}
}

// WithNear sets the near clipping plane distance.
//
// Parameters:
//   - near: near plane distance
//
// Returns:
//   - CameraBuilderOption: a function that sets the near plane
func WithNear(near float32) CameraBuilderOption {
	return func(c *camera) {
		c.near = near
		c.updateMatrices()
	}
}

// WithFar sets the far clipping plane distance.
//
// Parameters:
//   - far: far plane distance
//
// Returns:
//   - CameraBuilderOption: functional option to set the far plane
func WithFar(far float32) CameraBuilderOption {
	return func(c *camera) {
		c.far = far
		c.updateMatrices()
	}
}

// WithController attaches a controller to the camera.
// After all options are applied, the camera recomputes its matrices from the controller's state.
//
// Parameters:
//   - ctrl: the controller to attach
//
// Returns:
//   - CameraBuilderOption: functional option to set the controller
func WithController(ctrl CameraController) CameraBuilderOption {
	return func(c *camera) {
		c.controller = ctrl
	}
}

// WithBindGroupProvider attaches a bind group provider to the camera.
// The provider describes the GPU binding requirements for camera uniforms.
//
// Parameters:
//   - provider: the bind group provider to attach
//
// Returns:
//   - CameraBuilderOption: functional option to set the bind group provider
func WithBindGroupProvider(provider bind_group_provider.BindGroupProvider) CameraBuilderOption {
	return func(c *camera) {
		c.bindGroupProvider = provider
	}
}

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
	c := &camera{
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
	c.Delegate = c
	return c
}
