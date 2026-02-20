package camera

// CameraControllerOption is a functional option for configuring a CameraController.
type CameraControllerOption func(*cameraControllerImpl)

// WithRadius sets the initial orbit radius (distance from target).
//
// Parameters:
//   - radius: distance from the orbit target
//
// Returns:
//   - CameraControllerOption: functional option to set the radius
func WithRadius(radius float32) CameraControllerOption {
	return func(cc *cameraControllerImpl) {
		cc.radius = radius
	}
}

// WithAzimuth sets the initial horizontal angle around the Y axis.
//
// Parameters:
//   - azimuth: horizontal angle in radians (0 = +Z axis)
//
// Returns:
//   - CameraControllerOption: functional option to set the azimuth
func WithAzimuth(azimuth float32) CameraControllerOption {
	return func(cc *cameraControllerImpl) {
		cc.azimuth = azimuth
	}
}

// WithElevation sets the initial vertical angle from the horizontal plane.
//
// Parameters:
//   - elevation: vertical angle in radians (0 = horizontal)
//
// Returns:
//   - CameraControllerOption: functional option to set the elevation
func WithElevation(elevation float32) CameraControllerOption {
	return func(cc *cameraControllerImpl) {
		cc.elevation = elevation
	}
}

// WithTarget sets the look-at/pivot point.
//
// Parameters:
//   - x: X coordinate of the target
//   - y: Y coordinate of the target
//   - z: Z coordinate of the target
//
// Returns:
//   - CameraControllerOption: functional option to set the target position
func WithTarget(x, y, z float32) CameraControllerOption {
	return func(cc *cameraControllerImpl) {
		cc.target[0] = x
		cc.target[1] = y
		cc.target[2] = z
	}
}

// WithRadiusBounds sets the minimum and maximum orbit radius.
//
// Parameters:
//   - min: minimum zoom distance
//   - max: maximum zoom distance
//
// Returns:
//   - CameraControllerOption: functional option to set radius bounds
func WithRadiusBounds(min, max float32) CameraControllerOption {
	return func(cc *cameraControllerImpl) {
		cc.minRadius = min
		cc.maxRadius = max
	}
}

// WithElevationBounds sets the minimum and maximum elevation angles.
//
// Parameters:
//   - min: minimum vertical angle in radians (prevents looking straight down)
//   - max: maximum vertical angle in radians (prevents flipping over)
//
// Returns:
//   - CameraControllerOption: functional option to set elevation bounds
func WithElevationBounds(min, max float32) CameraControllerOption {
	return func(cc *cameraControllerImpl) {
		cc.minElevation = min
		cc.maxElevation = max
	}
}

// WithOrbitSpeed sets the keyboard orbit speed.
//
// Parameters:
//   - speed: radians per orbit call
//
// Returns:
//   - CameraControllerOption: functional option to set orbit speed
func WithOrbitSpeed(speed float32) CameraControllerOption {
	return func(cc *cameraControllerImpl) {
		cc.orbitSpeed = speed
	}
}

// WithMouseSensitivity sets the mouse drag sensitivity.
//
// Parameters:
//   - sensitivity: multiplier for mouse movement
//
// Returns:
//   - CameraControllerOption: functional option to set mouse sensitivity
func WithMouseSensitivity(sensitivity float32) CameraControllerOption {
	return func(cc *cameraControllerImpl) {
		cc.mouseSensitivity = sensitivity
	}
}

// WithZoomSpeed sets the zoom speed multiplier.
//
// Parameters:
//   - speed: multiplier for zoom input
//
// Returns:
//   - CameraControllerOption: functional option to set zoom speed
func WithZoomSpeed(speed float32) CameraControllerOption {
	return func(cc *cameraControllerImpl) {
		cc.zoomSpeed = speed
	}
}

// WithPanSpeed sets the planar pan speed multiplier.
//
// Parameters:
//   - speed: multiplier for pan input
//
// Returns:
//   - CameraControllerOption: functional option to set pan speed
func WithPanSpeed(speed float32) CameraControllerOption {
	return func(cc *cameraControllerImpl) {
		cc.panSpeed = speed
	}
}
