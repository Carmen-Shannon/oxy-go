package camera

import (
	"math"
	"sync"
)

// cameraControllerImpl is the single implementation of CameraController.
// Supports both orbit and planar controls simultaneously. Orbit methods modify
// spherical coordinates and recompute position; planar methods translate both
// position and target along local camera axes, preserving the orbit relationship.
type cameraControllerImpl struct {
	mu *sync.Mutex

	// Camera position (computed from target + spherical coords)
	position [3]float32
	target   [3]float32

	// Spherical coordinates (offset from target)
	radius    float32
	azimuth   float32 // Horizontal angle around Y axis
	elevation float32 // Vertical angle from horizontal plane

	// Orbit constraints
	minRadius    float32
	maxRadius    float32
	minElevation float32
	maxElevation float32

	// Orbit speed settings
	orbitSpeed       float32
	mouseSensitivity float32
	zoomSpeed        float32

	// Planar speed
	panSpeed float32
}

// Compile-time interface compliance check
var _ CameraController = &cameraControllerImpl{}

// NewCameraController creates a new camera controller with sensible defaults.
// The returned controller supports both orbit and planar controls simultaneously.
//
// Parameters:
//   - options: functional options to configure the controller
//
// Returns:
//   - CameraController: the newly created controller
func NewCameraController(options ...CameraControllerOption) CameraController {
	cc := &cameraControllerImpl{
		mu:     &sync.Mutex{},
		target: [3]float32{0, 0, 0},

		radius:    250.0,
		azimuth:   0.0,
		elevation: float32(math.Pi / 6),

		minRadius:    20.0,
		maxRadius:    2000.0,
		minElevation: 0.05,
		maxElevation: float32(math.Pi/2 - 0.1),

		orbitSpeed:       0.03,
		mouseSensitivity: 0.005,
		zoomSpeed:        15.0,

		panSpeed: 1.0,
	}

	for _, option := range options {
		option(cc)
	}

	cc.updatePosition()
	return cc
}

// NewOrbitController creates a new camera controller configured for orbit-style control.
// This is a convenience wrapper around NewCameraController for backward compatibility.
//
// Parameters:
//   - options: functional options to configure the controller
//
// Returns:
//   - CameraController: the newly created controller
func NewOrbitController(options ...CameraControllerOption) CameraController {
	return NewCameraController(options...)
}

// --- internal helpers ---

// updatePosition recomputes the camera position from spherical coordinates.
// Must be called whenever radius, azimuth, elevation, or target changes.
// Caller must hold the mutex.
func (cc *cameraControllerImpl) updatePosition() {
	cosElev := float32(math.Cos(float64(cc.elevation)))
	sinElev := float32(math.Sin(float64(cc.elevation)))
	cosAzim := float32(math.Cos(float64(cc.azimuth)))
	sinAzim := float32(math.Sin(float64(cc.azimuth)))

	cc.position[0] = cc.target[0] + cc.radius*cosElev*sinAzim
	cc.position[1] = cc.target[1] + cc.radius*sinElev
	cc.position[2] = cc.target[2] + cc.radius*cosElev*cosAzim
}

// localAxes computes the camera's local coordinate axes consistent with the LookAt matrix.
// Returns right (rx,ry,rz), up (ux,uy,uz), and forward (fx,fy,fz) vectors.
// If position and target coincide, all returned components are zero.
// Caller must hold the mutex.
func (cc *cameraControllerImpl) localAxes() (rx, ry, rz, ux, uy, uz, fx, fy, fz float32) {
	// backward = normalize(position - target), matching LookAt's z-axis
	bx := cc.position[0] - cc.target[0]
	by := cc.position[1] - cc.target[1]
	bz := cc.position[2] - cc.target[2]
	bLen := float32(math.Sqrt(float64(bx*bx + by*by + bz*bz)))
	if bLen < 1e-8 {
		return
	}
	bx /= bLen
	by /= bLen
	bz /= bLen

	// right = normalize(cross(worldUp, backward)) where worldUp = (0, 1, 0)
	// cross((0,1,0), (bx,by,bz)) = (1*bz - 0*by, 0*bx - 0*bz, 0*by - 1*bx) = (bz, 0, -bx)
	rx = bz
	rz = -bx
	rLen := float32(math.Sqrt(float64(rx*rx + rz*rz)))
	if rLen < 1e-8 {
		return
	}
	rx /= rLen
	rz /= rLen

	// up = cross(backward, right), matching LookAt's y-axis
	ux = by*rz - bz*ry
	uy = bz*rx - bx*rz
	uz = bx*ry - by*rx

	// forward = -backward
	fx = -bx
	fy = -by
	fz = -bz
	return
}

// --- CameraController shared methods ---

func (cc *cameraControllerImpl) Position() (x, y, z float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.position[0], cc.position[1], cc.position[2]
}

func (cc *cameraControllerImpl) SetPosition(x, y, z float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.position[0] = x
	cc.position[1] = y
	cc.position[2] = z
}

func (cc *cameraControllerImpl) Target() (x, y, z float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.target[0], cc.target[1], cc.target[2]
}

func (cc *cameraControllerImpl) SetTarget(x, y, z float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.target[0] = x
	cc.target[1] = y
	cc.target[2] = z
	cc.updatePosition()
}

func (cc *cameraControllerImpl) Zoom(delta float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.radius -= delta * cc.zoomSpeed
	if cc.radius < cc.minRadius {
		cc.radius = cc.minRadius
	}
	if cc.radius > cc.maxRadius {
		cc.radius = cc.maxRadius
	}
	cc.updatePosition()
}

// --- orbitCameraController implementation ---

func (cc *cameraControllerImpl) OrbitLeft() {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.azimuth -= cc.orbitSpeed
	cc.updatePosition()
}

func (cc *cameraControllerImpl) OrbitRight() {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.azimuth += cc.orbitSpeed
	cc.updatePosition()
}

func (cc *cameraControllerImpl) OrbitUp() {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.elevation += cc.orbitSpeed
	if cc.elevation > cc.maxElevation {
		cc.elevation = cc.maxElevation
	}
	cc.updatePosition()
}

func (cc *cameraControllerImpl) OrbitDown() {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.elevation -= cc.orbitSpeed
	if cc.elevation < cc.minElevation {
		cc.elevation = cc.minElevation
	}
	cc.updatePosition()
}

func (cc *cameraControllerImpl) Radius() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.radius
}

func (cc *cameraControllerImpl) SetRadius(radius float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.radius = radius
	if cc.radius < cc.minRadius {
		cc.radius = cc.minRadius
	}
	if cc.radius > cc.maxRadius {
		cc.radius = cc.maxRadius
	}
	cc.updatePosition()
}

func (cc *cameraControllerImpl) MinRadius() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.minRadius
}

func (cc *cameraControllerImpl) MaxRadius() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.maxRadius
}

func (cc *cameraControllerImpl) Azimuth() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.azimuth
}

func (cc *cameraControllerImpl) SetAzimuth(azimuth float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.azimuth = azimuth
	cc.updatePosition()
}

func (cc *cameraControllerImpl) Elevation() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.elevation
}

func (cc *cameraControllerImpl) SetElevation(elevation float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.elevation = elevation
	if cc.elevation < cc.minElevation {
		cc.elevation = cc.minElevation
	}
	if cc.elevation > cc.maxElevation {
		cc.elevation = cc.maxElevation
	}
	cc.updatePosition()
}

func (cc *cameraControllerImpl) MinElevation() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.minElevation
}

func (cc *cameraControllerImpl) MaxElevation() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.maxElevation
}

func (cc *cameraControllerImpl) OrbitSpeed() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.orbitSpeed
}

func (cc *cameraControllerImpl) MouseSensitivity() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.mouseSensitivity
}

func (cc *cameraControllerImpl) ZoomSpeed() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.zoomSpeed
}

// --- planarCameraController implementation ---

func (cc *cameraControllerImpl) PanRight(delta float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	rx, _, rz, _, _, _, _, _, _ := cc.localAxes()
	offset := delta * cc.panSpeed

	cc.target[0] += rx * offset
	cc.target[1] += 0 // ry is always 0 for right vector with worldUp=(0,1,0)
	cc.target[2] += rz * offset
	cc.position[0] += rx * offset
	cc.position[1] += 0
	cc.position[2] += rz * offset
}

func (cc *cameraControllerImpl) PanUp(delta float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	_, _, _, ux, uy, uz, _, _, _ := cc.localAxes()
	offset := delta * cc.panSpeed

	cc.target[0] += ux * offset
	cc.target[1] += uy * offset
	cc.target[2] += uz * offset
	cc.position[0] += ux * offset
	cc.position[1] += uy * offset
	cc.position[2] += uz * offset
}

func (cc *cameraControllerImpl) PanForward(delta float32) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	_, _, _, _, _, _, fx, fy, fz := cc.localAxes()
	offset := delta * cc.panSpeed

	cc.target[0] += fx * offset
	cc.target[1] += fy * offset
	cc.target[2] += fz * offset
	cc.position[0] += fx * offset
	cc.position[1] += fy * offset
	cc.position[2] += fz * offset
}

func (cc *cameraControllerImpl) PanSpeed() float32 {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.panSpeed
}
