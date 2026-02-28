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

// RigidBody is the interface that defines the behavior of a rigid body in a physics simulation.
type RigidBody interface {
	common.Delegate[RigidBody]

	// Position returns the world-space position of the rigid body.
	// This value is updated from GPU readback after each physics frame.
	//
	// Returns:
	//   - [3]float32: A 3D vector representing the world-space position.
	Position() [3]float32

	// Quaternion returns the orientation quaternion (xyzw) of the rigid body.
	// This value is updated from GPU readback after each physics frame.
	//
	// Returns:
	//   - [4]float32: The orientation quaternion in xyzw order.
	Quaternion() [4]float32

	// Velocity returns the linear velocity of the rigid body as a 3D vector.
	//
	// Returns:
	//   - [3]float32: A 3D vector representing the linear velocity of the rigid body.
	Velocity() [3]float32

	// AngularVelocity returns the angular velocity of the rigid body as a 3D vector.
	//
	// Returns:
	//   - [3]float32: A 3D vector representing the angular velocity of the rigid body.
	AngularVelocity() [3]float32

	// Mass returns the mass of the rigid body.
	//
	// Returns:
	//   - float32: The mass of the rigid body.
	Mass() float32

	// InverseMass returns the inverse of the mass of the rigid body.
	//
	// Returns:
	//   - float32: The inverse of the mass of the rigid body.
	InverseMass() float32

	// Bounce returns the bounce factor of the rigid body, which determines how much it bounces when colliding with other objects.
	//
	// Returns:
	//   - float32: The bounce factor of the rigid body.
	Bounce() float32

	// Friction returns the friction factor of the rigid body, which determines how much it resists sliding against other objects.
	//
	// Returns:
	//   - float32: The friction factor of the rigid body.
	Friction() float32

	// InverseInertiaTensorBody returns the inverse of the inertia tensor of the rigid body in body-local space.
	//
	// Returns:
	//   - [9]float32: A 3x3 matrix (flattened to a 1D array) representing the inverse inertia tensor of the rigid body in body-local space.
	InverseInertiaTensorBody() [9]float32

	// Active returns whether the rigid body is active in the simulation.
	//
	// Returns:
	//   - bool: True if the rigid body is active, false otherwise.
	Active() bool

	// Static returns whether the rigid body is static (immovable) in the simulation.
	//
	// Returns:
	//   - bool: True if the rigid body is static, false otherwise.
	Static() bool

	// Kinematic returns whether the rigid body is kinematic (moves according to animation or script) in the simulation.
	//
	// Returns:
	//   - bool: True if the rigid body is kinematic, false otherwise.
	Kinematic() bool

	// Particles returns the set of spherical collision elements that compose the rigid body.
	//
	// Returns:
	//   - []Particle: A slice of Particle structs representing the collision elements of the rigid body.
	Particles() []Particle

	// ParticleRadius returns the radius of each spherical particle generated for the Rigid Body.
	//
	// Returns:
	//   - float32: The radius of each spherical particle generated for the Rigid Body.
	ParticleRadius() float32

	// SurfaceOnly returns whether only surface particles are generated during voxelization.
	//
	// Returns:
	//   - bool: True if only surface particles are generated during voxelization, false otherwise.
	SurfaceOnly() bool

	// SetVelocity sets the linear velocity of the rigid body.
	//
	// Parameters:
	//   - velocity: A 3D vector (XYZ) representing the linear velocity to set for the rigid body.
	SetVelocity(velocity [3]float32)

	// SetAngularVelocity sets the angular velocity of the rigid body.
	//
	// Parameters:
	//   - angularVelocity: A 3D vector (XYZ) representing the angular velocity to set for the rigid body.
	SetAngularVelocity(angularVelocity [3]float32)

	// SetMass sets the mass of the rigid body.
	//
	// Parameters:
	//   - mass: The mass to set for the rigid body.
	SetMass(mass float32)

	// SetInverseMass sets the inverse of the mass of the rigid body.
	//
	// Parameters:
	//   - inverseMass: The inverse of the mass to set for the rigid body.
	SetInverseMass(inverseMass float32)

	// SetBounce sets the bounce factor of the rigid body.
	//
	// Parameters:
	//   - bounce: The bounce factor to set for the rigid body.
	SetBounce(bounce float32)

	// SetFriction sets the friction factor of the rigid body.
	//
	// Parameters:
	//   - friction: The friction factor to set for the rigid body.
	SetFriction(friction float32)

	// SetActive sets whether the rigid body is active in the simulation.
	//
	// Parameters:
	//   - active: True to set the rigid body as active, false to set it as inactive.
	SetActive(active bool)

	// SetStatic sets whether the rigid body is static (immovable) in the simulation.
	//
	// Parameters:
	//   - static: True to set the rigid body as static, false to set it as non-static.
	SetStatic(static bool)

	// SetKinematic sets whether the rigid body is kinematic (moves according to animation or script) in the simulation.
	//
	// Parameters:
	//   - kinematic: True to set the rigid body as kinematic, false to set it as non-kinematic.
	SetKinematic(kinematic bool)

	// SetParticles sets the collision particles of the rigid body.
	//
	// Parameters:
	//   - particles: A slice of Particle structs representing the collision elements to set for the rigid body.
	SetParticles(particles []Particle)

	// SetParticleRadius sets the radius of each spherical particle generated for the Rigid Body.
	//
	// Parameters:
	//   - radius: The radius to set for each spherical particle generated for the Rigid Body.
	SetParticleRadius(radius float32)

	// SetSurfaceOnly sets whether only surface particles are generated during voxelization.
	//
	// Parameters:
	//   - surfaceOnly: True to set only surface particles to be generated during voxelization, false to allow internal particles as well.
	SetSurfaceOnly(surfaceOnly bool)

	// SetPosition sets the world-space position of the rigid body.
	// This is primarily called during GPU readback processing.
	//
	// Parameters:
	//   - position: A 3D vector (XYZ) representing the world-space position to set.
	SetPosition(position [3]float32)

	// SetQuaternion sets the orientation quaternion of the rigid body.
	// This is primarily called during GPU readback processing.
	//
	// Parameters:
	//   - quaternion: A quaternion (XYZW) representing the orientation to set.
	SetQuaternion(quaternion [4]float32)

	// ApplyForce adds a force vector to the pending force accumulator.
	// This is safe to call from any goroutine (e.g. the tick callback).
	// The accumulated force is drained by the physics controller each frame
	// and uploaded as GPUBody.ExternalForce.
	//
	// Parameters:
	//   - force: the force vector (XYZ) to accumulate.
	ApplyForce(force [3]float32)

	// ApplyTorque adds a torque vector to the pending torque accumulator.
	// This is safe to call from any goroutine (e.g. the tick callback).
	// The accumulated torque is drained by the physics controller each frame
	// and uploaded as GPUBody.ExternalTorque.
	//
	// Parameters:
	//   - torque: the torque vector (XYZ) to accumulate.
	ApplyTorque(torque [3]float32)
}

var _ RigidBody = &rigidBody{}

// NewRigidBody creates a new RigidBody with default physical properties. Defaults are:
// mass=1, bounce=0.5, friction=0.5, surfaceOnly=true. If mass is zero and the body
// is not kinematic it is automatically marked static. The inverse inertia tensor is
// computed from the particle layout when particles are provided.
//
// Parameters:
//   - options: variadic list of RigidBodyOption functions to configure the rigid body
//
// Returns:
//   - RigidBody: a new RigidBody instance
func NewRigidBody(options ...RigidBodyOption) RigidBody {
	r := &rigidBody{
		surfaceOnly: true,
		mass:        1.0,
		bounce:      0.5,
		friction:    0.5,
	}
	for _, option := range options {
		option(r)
	}

	if r.inverseMass == 0 && r.mass > 0 {
		r.inverseMass = 1.0 / r.mass
	}

	if r.mass == 0 && !r.kinematic {
		r.static = true
	}

	if len(r.particles) > 0 {
		r.computeInverseInertiaTensor()
	}

	r.Delegate = r
	return r
}

func (r *rigidBody) Position() [3]float32 {
	return r.position
}

func (r *rigidBody) Quaternion() [4]float32 {
	return r.quaternion
}

func (r *rigidBody) Velocity() [3]float32 {
	return r.velocity
}

func (r *rigidBody) AngularVelocity() [3]float32 {
	return r.angularVelocity
}

func (r *rigidBody) Mass() float32 {
	return r.mass
}

func (r *rigidBody) InverseMass() float32 {
	return r.inverseMass
}

func (r *rigidBody) Bounce() float32 {
	return r.bounce
}

func (r *rigidBody) Friction() float32 {
	return r.friction
}

func (r *rigidBody) InverseInertiaTensorBody() [9]float32 {
	return r.inverseInertiaTensorBody
}

func (r *rigidBody) Active() bool {
	return r.active
}

func (r *rigidBody) Static() bool {
	return r.static
}

func (r *rigidBody) Kinematic() bool {
	return r.kinematic
}

func (r *rigidBody) Particles() []Particle {
	return r.particles
}

func (r *rigidBody) ParticleRadius() float32 {
	return r.particleRadius
}

func (r *rigidBody) SurfaceOnly() bool {
	return r.surfaceOnly
}

func (r *rigidBody) SetVelocity(velocity [3]float32) {
	r.velocity = velocity
}

func (r *rigidBody) SetAngularVelocity(angularVelocity [3]float32) {
	r.angularVelocity = angularVelocity
}

func (r *rigidBody) SetMass(mass float32) {
	r.mass = mass
}

func (r *rigidBody) SetInverseMass(inverseMass float32) {
	r.inverseMass = inverseMass
}

func (r *rigidBody) SetBounce(bounce float32) {
	r.bounce = bounce
}

func (r *rigidBody) SetFriction(friction float32) {
	r.friction = friction
}

func (r *rigidBody) SetActive(active bool) {
	r.active = active
}

func (r *rigidBody) SetStatic(static bool) {
	r.static = static
}

func (r *rigidBody) SetKinematic(kinematic bool) {
	r.kinematic = kinematic
}

func (r *rigidBody) SetParticles(particles []Particle) {
	r.particles = particles
	r.computeInverseInertiaTensor()
}

func (r *rigidBody) SetParticleRadius(radius float32) {
	r.particleRadius = radius
}

func (r *rigidBody) SetSurfaceOnly(surfaceOnly bool) {
	r.surfaceOnly = surfaceOnly
}

func (r *rigidBody) SetPosition(position [3]float32) {
	r.position = position
}

func (r *rigidBody) SetQuaternion(quaternion [4]float32) {
	r.quaternion = quaternion
}

func (r *rigidBody) ApplyForce(force [3]float32) {
	r.mu.Lock()
	r.pendingForce[0] += force[0]
	r.pendingForce[1] += force[1]
	r.pendingForce[2] += force[2]
	r.mu.Unlock()
}

func (r *rigidBody) ApplyTorque(torque [3]float32) {
	r.mu.Lock()
	r.pendingTorque[0] += torque[0]
	r.pendingTorque[1] += torque[1]
	r.pendingTorque[2] += torque[2]
	r.mu.Unlock()
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
