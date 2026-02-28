package physics

// RigidBodyOption is a functional option for configuring a rigidBody during construction.
type RigidBodyOption func(*rigidBody)

// WithVelocity sets the linear velocity of the rigid body.
//
// Parameters:
//   - velocity: A 3D vector representing the linear velocity of the rigid body.
//
// Returns:
//   - RigidBodyOption: A function that can be used to set the velocity of a rigid body.
func WithVelocity(velocity [3]float32) RigidBodyOption {
	return func(r *rigidBody) {
		r.velocity = velocity
	}
}

// WithAngularVelocity sets the angular velocity of the rigid body.
//
// Parameters:
//   - angularVelocity: A 3D vector representing the angular velocity of the rigid body.
//
// Returns:
//   - RigidBodyOption: A function that can be used to set the angular velocity of a rigid body.
func WithAngularVelocity(angularVelocity [3]float32) RigidBodyOption {
	return func(r *rigidBody) {
		r.angularVelocity = angularVelocity
	}
}

// WithMass sets the mass of the rigid body.
//
// Parameters:
//   - mass: The mass of the rigid body.
//
// Returns:
//   - RigidBodyOption: A function that can be used to set the mass of a rigid body.
func WithMass(mass float32) RigidBodyOption {
	return func(r *rigidBody) {
		r.mass = mass
	}
}

// WithInverseMass sets the inverse of the mass of the rigid body.
//
// Parameters:
//   - inverseMass: The inverse of the mass of the rigid body.
//
// Returns:
//   - RigidBodyOption: A function that can be used to set the inverse mass of a rigid body.
func WithInverseMass(inverseMass float32) RigidBodyOption {
	return func(r *rigidBody) {
		r.inverseMass = inverseMass
	}
}

// WithBounce sets the bounce factor of the rigid body.
//
// Parameters:
//   - bounce: The bounce factor of the rigid body.
//
// Returns:
//   - RigidBodyOption: A function that can be used to set the bounce factor of a rigid body.
func WithBounce(bounce float32) RigidBodyOption {
	return func(r *rigidBody) {
		r.bounce = bounce
	}
}

// WithFriction sets the friction factor of the rigid body.
//
// Parameters:
//   - friction: The friction factor of the rigid body.
//
// Returns:
//   - RigidBodyOption: A function that can be used to set the friction factor of a rigid body.
func WithFriction(friction float32) RigidBodyOption {
	return func(r *rigidBody) {
		r.friction = friction
	}
}

// WithActive sets whether the rigid body is active in the simulation.
//
// Parameters:
//   - active: True if the rigid body is active, false otherwise.
//
// Returns:
//   - RigidBodyOption: A function that can be used to set the active state of a rigid body.
func WithActive(active bool) RigidBodyOption {
	return func(r *rigidBody) {
		r.active = active
	}
}

// WithStatic sets whether the rigid body is static (immovable) in the simulation.
//
// Parameters:
//   - static: True if the rigid body is static, false otherwise.
//
// Returns:
//   - RigidBodyOption: A function that can be used to set the static state of a rigid body.
func WithStatic(static bool) RigidBodyOption {
	return func(r *rigidBody) {
		r.static = static
	}
}

// WithKinematic sets whether the rigid body is kinematic (moves according to animation or script) in the simulation.
//
// Parameters:
//   - kinematic: True if the rigid body is kinematic, false otherwise.
//
// Returns:
//   - RigidBodyOption: A function that can be used to set the kinematic state of a rigid body.
func WithKinematic(kinematic bool) RigidBodyOption {
	return func(r *rigidBody) {
		r.kinematic = kinematic
	}
}

// WithParticles sets the collision particles of the rigid body.
//
// Parameters:
//   - particles: A slice of Particle structs representing the collision elements of the rigid body.
//
// Returns:
//   - RigidBodyOption: A function that can be used to set the collision particles of a rigid body.
func WithParticles(particles []Particle) RigidBodyOption {
	return func(r *rigidBody) {
		r.particles = particles
	}
}

// WithParticleRadius sets the radius of each spherical collision particle for the rigid body.
// This value is used by the physics controller to derive the global particle diameter.
//
// Parameters:
//   - radius: The radius of each particle.
//
// Returns:
//   - RigidBodyOption: A function that applies the particle radius option to a rigidBody.
func WithParticleRadius(radius float32) RigidBodyOption {
	return func(r *rigidBody) {
		r.particleRadius = radius
	}
}
