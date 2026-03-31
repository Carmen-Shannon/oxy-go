package physics

import (
	"sync"

	"github.com/Carmen-Shannon/oxy-go/common"
)

// rigidBody is the implementation of the RigidBody interface. It represents a physical object in space with
// properties that define it's physical behavior and interactions with other rigid body objects.
type rigidBody struct {
	common.DelegateImpl[RigidBody]

	// position is the world-space position of the rigid body, updated from GPU readback.
	position [3]float32

	// quaternion is the orientation of the rigid body (xyzw), updated from GPU readback.
	quaternion [4]float32

	// velocity and angularVelocity represent the linear and angular velocity of the rigid body, respectively.
	velocity, angularVelocity [3]float32

	// mass, inverseMass, bounce, and friction are physical properties that affect how the rigid body interacts with forces and other rigid bodies.
	mass, inverseMass, bounce, friction float32

	// active, static, and kinematic are state flags that determine how the rigid body interacts with other rigid bodies.
	active, static, kinematic bool

	// particles is the set of spherical collision elements that compose the rigid body.
	particles []Particle

	// particleRadius is the radius of each spherical particle generated for the Rigid Body.
	particleRadius float32

	// surfaceOnly indicates whether only surface particles are generated during voxelization (this is true by default)
	surfaceOnly bool

	// these properties are populated when creating the Rigid Body, and should not be manually set.

	// inverseInertiaTensorBody is the inverse of the inertia tensor of the rigid body, which is used in rotational dynamics calculations.
	inverseInertiaTensorBody [9]float32

	// mu protects pendingForce and pendingTorque for concurrent access from the tick goroutine.
	mu sync.Mutex

	// pendingForce accumulates externally-applied forces (XYZ) until drained by the physics controller.
	pendingForce [3]float32

	// pendingTorque accumulates externally-applied torques (XYZ) until drained by the physics controller.
	pendingTorque [3]float32
}

// drainForce returns the accumulated pending force and resets it to zero.
//
// Returns:
//   - [3]float32: the accumulated force vector since the last drain.
func (r *rigidBody) drainForce() [3]float32 {
	r.mu.Lock()
	f := r.pendingForce
	r.pendingForce = [3]float32{}
	r.mu.Unlock()
	return f
}

// drainTorque returns the accumulated pending torque and resets it to zero.
//
// Returns:
//   - [3]float32: the accumulated torque vector since the last drain.
func (r *rigidBody) drainTorque() [3]float32 {
	r.mu.Lock()
	t := r.pendingTorque
	r.pendingTorque = [3]float32{}
	r.mu.Unlock()
	return t
}

// computeInverseInertiaTensor derives the body-space inverse inertia tensor from
// the current particle layout and mass. Each particle is treated as a point mass
// m_i = mass / len(particles). The resulting 3x3 symmetric tensor is inverted and
// stored on the rigid body.
//
// For single-particle bodies where the particle sits at the body origin (the
// common case for DEM droplets), the point-mass formula yields a zero tensor
// which is non-invertible. In this case we fall back to the analytical inertia
// of a solid sphere: I = (2/5) * m * r², where r is the particle radius. This
// gives the body a physically meaningful rotational response to off-center
// contact forces.
func (r *rigidBody) computeInverseInertiaTensor() {
	n := len(r.particles)
	if n == 0 || r.mass == 0 {
		r.inverseInertiaTensorBody = [9]float32{}
		return
	}

	mi := r.mass / float32(n)
	var ixx, iyy, izz, ixy, ixz, iyz float32

	for _, p := range r.particles {
		x, y, z := p.LocalPosition[0], p.LocalPosition[1], p.LocalPosition[2]
		ixx += mi * (y*y + z*z)
		iyy += mi * (x*x + z*z)
		izz += mi * (x*x + y*y)
		ixy -= mi * x * y
		ixz -= mi * x * z
		iyz -= mi * y * z
	}

	// Check if the tensor is degenerate (all diagonal elements near zero).
	// This happens when all particles sit at the body origin (e.g. single-
	// particle droplets). Fall back to solid-sphere inertia: I = (2/5)mr².
	diag := ixx + iyy + izz
	if diag < 1e-12 {
		sphereI := 0.4 * r.mass * r.particleRadius * r.particleRadius
		if sphereI > 0 {
			invI := 1.0 / sphereI
			r.inverseInertiaTensorBody = [9]float32{
				invI, 0, 0,
				0, invI, 0,
				0, 0, invI,
			}
			return
		}
		r.inverseInertiaTensorBody = [9]float32{}
		return
	}

	tensor := [9]float32{
		ixx, ixy, ixz,
		ixy, iyy, iyz,
		ixz, iyz, izz,
	}

	inv, ok := common.Invert3x3(tensor)
	if !ok {
		r.inverseInertiaTensorBody = [9]float32{}
		return
	}
	r.inverseInertiaTensorBody = inv
}
