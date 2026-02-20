package camera

// CameraController defines the union interface for camera control systems.
// Controllers own positional state (position, target). Camera reads from controller
// and computes view/projection matrices. Embeds both orbitCameraController and
// planarCameraController, enabling orbit and planar controls to work simultaneously
// from a single controller instance.
type CameraController interface {
	orbitCameraController
	planarCameraController

	// Position returns the camera's world-space position.
	//
	// Returns:
	//   - x, y, z: world-space camera position
	Position() (x, y, z float32)

	// Target returns the look-at point.
	//
	// Returns:
	//   - x, y, z: world-space target position
	Target() (x, y, z float32)

	// SetTarget sets the look-at/pivot point and recomputes position from spherical coordinates.
	//
	// Parameters:
	//   - x, y, z: world-space coordinates
	SetTarget(x, y, z float32)

	// SetPosition sets the camera's world-space position directly.
	//
	// Parameters:
	//   - x, y, z: world-space coordinates
	SetPosition(x, y, z float32)

	// Zoom adjusts the camera's distance by modifying orbit radius.
	// Positive delta zooms in (closer to target).
	//
	// Parameters:
	//   - delta: zoom amount scaled by ZoomSpeed
	Zoom(delta float32)
}

// orbitCameraController defines orbit-specific control methods.
// Provides third-person orbit controls using spherical coordinates (radius, azimuth, elevation)
// relative to the target/pivot point.
type orbitCameraController interface {
	// OrbitLeft rotates the camera left around the target by one orbit speed step.
	OrbitLeft()

	// OrbitRight rotates the camera right around the target by one orbit speed step.
	OrbitRight()

	// OrbitUp tilts the camera upward by one orbit speed step, clamped to max elevation.
	OrbitUp()

	// OrbitDown tilts the camera downward by one orbit speed step, clamped to min elevation.
	OrbitDown()

	// Radius returns the current orbit radius (distance from target).
	//
	// Returns:
	//   - float32: current distance from target
	Radius() float32

	// SetRadius sets the orbit radius directly, clamped to min/max bounds.
	//
	// Parameters:
	//   - radius: new distance from target
	SetRadius(radius float32)

	// MinRadius returns the minimum allowed orbit radius.
	//
	// Returns:
	//   - float32: minimum zoom distance
	MinRadius() float32

	// MaxRadius returns the maximum allowed orbit radius.
	//
	// Returns:
	//   - float32: maximum zoom distance
	MaxRadius() float32

	// Azimuth returns the current horizontal angle around the Y axis.
	//
	// Returns:
	//   - float32: azimuth in radians
	Azimuth() float32

	// SetAzimuth sets the horizontal angle directly and recomputes position.
	//
	// Parameters:
	//   - azimuth: new horizontal angle in radians
	SetAzimuth(azimuth float32)

	// Elevation returns the current vertical angle from the horizontal plane.
	//
	// Returns:
	//   - float32: elevation in radians
	Elevation() float32

	// SetElevation sets the vertical angle directly, clamped to min/max bounds.
	//
	// Parameters:
	//   - elevation: new vertical angle in radians
	SetElevation(elevation float32)

	// MinElevation returns the minimum allowed elevation angle.
	//
	// Returns:
	//   - float32: minimum elevation in radians
	MinElevation() float32

	// MaxElevation returns the maximum allowed elevation angle.
	//
	// Returns:
	//   - float32: maximum elevation in radians
	MaxElevation() float32

	// OrbitSpeed returns the keyboard orbit speed in radians per step.
	//
	// Returns:
	//   - float32: radians per orbit call
	OrbitSpeed() float32

	// MouseSensitivity returns the mouse drag sensitivity multiplier.
	//
	// Returns:
	//   - float32: multiplier for mouse movement
	MouseSensitivity() float32

	// ZoomSpeed returns the zoom speed multiplier.
	//
	// Returns:
	//   - float32: multiplier for zoom input
	ZoomSpeed() float32
}

// planarCameraController defines planar translation control methods.
// Provides first-person-style panning along the camera's local axes without
// changing orbit angles. Panning shifts both position and target by the same
// offset, preserving the orbit relationship.
type planarCameraController interface {
	// PanRight translates the camera along its local right axis.
	// Positive delta moves right, negative moves left.
	//
	// Parameters:
	//   - delta: pan amount scaled by PanSpeed
	PanRight(delta float32)

	// PanUp translates the camera along its local up axis.
	// Positive delta moves up, negative moves down.
	//
	// Parameters:
	//   - delta: pan amount scaled by PanSpeed
	PanUp(delta float32)

	// PanForward translates the camera along its local forward axis (dolly).
	// Positive delta moves toward the target, negative moves away.
	//
	// Parameters:
	//   - delta: pan amount scaled by PanSpeed
	PanForward(delta float32)

	// PanSpeed returns the pan speed multiplier.
	//
	// Returns:
	//   - float32: multiplier for pan input
	PanSpeed() float32
}
