package physics

import (
	"github.com/Carmen-Shannon/oxy-go/common"
)

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

func (r *rigidBody) Position() [3]float32                 { return r.position }
func (r *rigidBody) Quaternion() [4]float32               { return r.quaternion }
func (r *rigidBody) Velocity() [3]float32                 { return r.velocity }
func (r *rigidBody) AngularVelocity() [3]float32          { return r.angularVelocity }
func (r *rigidBody) Mass() float32                        { return r.mass }
func (r *rigidBody) InverseMass() float32                 { return r.inverseMass }
func (r *rigidBody) Bounce() float32                      { return r.bounce }
func (r *rigidBody) Friction() float32                    { return r.friction }
func (r *rigidBody) InverseInertiaTensorBody() [9]float32 { return r.inverseInertiaTensorBody }
func (r *rigidBody) Active() bool                         { return r.active }
func (r *rigidBody) Static() bool                         { return r.static }
func (r *rigidBody) Kinematic() bool                      { return r.kinematic }
func (r *rigidBody) Particles() []Particle                { return r.particles }
func (r *rigidBody) ParticleRadius() float32              { return r.particleRadius }
func (r *rigidBody) SurfaceOnly() bool                    { return r.surfaceOnly }
func (r *rigidBody) SetVelocity(velocity [3]float32)      { r.velocity = velocity }
func (r *rigidBody) SetMass(mass float32)                 { r.mass = mass }
func (r *rigidBody) SetInverseMass(inverseMass float32)   { r.inverseMass = inverseMass }
func (r *rigidBody) SetBounce(bounce float32)             { r.bounce = bounce }
func (r *rigidBody) SetFriction(friction float32)         { r.friction = friction }
func (r *rigidBody) SetActive(active bool)                { r.active = active }
func (r *rigidBody) SetStatic(static bool)                { r.static = static }
func (r *rigidBody) SetKinematic(kinematic bool)          { r.kinematic = kinematic }
func (r *rigidBody) SetParticleRadius(radius float32)     { r.particleRadius = radius }
func (r *rigidBody) SetSurfaceOnly(surfaceOnly bool)      { r.surfaceOnly = surfaceOnly }
func (r *rigidBody) SetPosition(position [3]float32)      { r.position = position }
func (r *rigidBody) SetQuaternion(quaternion [4]float32)  { r.quaternion = quaternion }

func (r *rigidBody) SetAngularVelocity(angularVelocity [3]float32) {
	r.angularVelocity = angularVelocity
}

func (r *rigidBody) SetParticles(particles []Particle) {
	r.particles = particles
	r.computeInverseInertiaTensor()
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
