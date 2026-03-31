package physics

import "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"

// PhysicsBuilderOption is a functional option for configuring a physicsImpl during construction.
type PhysicsBuilderOption func(*physicsImpl)

// WithFixedDt sets the fixed timestep used for each physics substep.
//
// Parameters:
//   - fixedDt: simulation timestep in seconds (e.g. 1/60 for 60 Hz)
//
// Returns:
//   - PhysicsBuilderOption: a function that applies the fixed timestep option to a physicsImpl
func WithFixedDt(fixedDt float32) PhysicsBuilderOption {
	return func(p *physicsImpl) {
		p.fixedDt = fixedDt
	}
}

// WithMaxSubsteps sets the maximum number of fixed-timestep substeps allowed per frame.
// Capping substeps prevents spiral-of-death when frame time significantly exceeds the
// fixed timestep.
//
// Parameters:
//   - maxSubsteps: maximum substep count per frame
//
// Returns:
//   - PhysicsBuilderOption: a function that applies the max substeps option to a physicsImpl
func WithMaxSubsteps(maxSubsteps int) PhysicsBuilderOption {
	return func(p *physicsImpl) {
		p.maxSubsteps = maxSubsteps
	}
}

// WithMaxBodies sets the maximum number of rigid bodies the physics system can hold.
// This determines the size of the GPU body storage buffer.
//
// Parameters:
//   - maxBodies: maximum body count
//
// Returns:
//   - PhysicsBuilderOption: a function that applies the max bodies option to a physicsImpl
func WithMaxBodies(maxBodies int) PhysicsBuilderOption {
	return func(p *physicsImpl) {
		p.maxBodies = maxBodies
	}
}

// WithMaxParticles sets the maximum number of collision particles across all rigid bodies.
// This determines the size of the GPU particle storage buffer.
//
// Parameters:
//   - maxParticles: maximum particle count
//
// Returns:
//   - PhysicsBuilderOption: a function that applies the max particles option to a physicsImpl
func WithMaxParticles(maxParticles int) PhysicsBuilderOption {
	return func(p *physicsImpl) {
		p.maxParticles = maxParticles
	}
}

// WithMaxGridCells sets the maximum total number of spatial grid cells used for
// collision broad-phase. This caps the dimensions of the uniform grid (x*y*z).
//
// Parameters:
//   - maxGridCells: maximum grid cell count
//
// Returns:
//   - PhysicsBuilderOption: a function that applies the max grid cells option to a physicsImpl
func WithMaxGridCells(maxGridCells int) PhysicsBuilderOption {
	return func(p *physicsImpl) {
		p.maxGridCells = maxGridCells
	}
}

// WithSpringCoeff sets the DEM spring coefficient used in particle collision response.
// Higher values produce stiffer collisions.
//
// Parameters:
//   - springCoeff: DEM spring coefficient k
//
// Returns:
//   - PhysicsBuilderOption: a function that applies the spring coefficient option to a physicsImpl
func WithSpringCoeff(springCoeff float32) PhysicsBuilderOption {
	return func(p *physicsImpl) {
		p.springCoeff = springCoeff
	}
}

// WithDampingCoeff sets the DEM damping coefficient used in particle collision response.
// Higher values dissipate more kinetic energy on contact.
//
// Parameters:
//   - dampingCoeff: DEM damping coefficient η
//
// Returns:
//   - PhysicsBuilderOption: a function that applies the damping coefficient option to a physicsImpl
func WithDampingCoeff(dampingCoeff float32) PhysicsBuilderOption {
	return func(p *physicsImpl) {
		p.dampingCoeff = dampingCoeff
	}
}

// WithShearCoeff sets the DEM tangential friction coefficient used in particle collision response.
// This controls the tangential (shear) force applied during contact.
//
// Parameters:
//   - shearCoeff: DEM tangential friction coefficient μ_t
//
// Returns:
//   - PhysicsBuilderOption: a function that applies the shear coefficient option to a physicsImpl
func WithShearCoeff(shearCoeff float32) PhysicsBuilderOption {
	return func(p *physicsImpl) {
		p.shearCoeff = shearCoeff
	}
}

// WithGravity sets the gravitational acceleration vector applied to all dynamic bodies
// every substep. This is a field force built into the physics globals, not an external
// per-body force — it persists correctly across multiple substeps.
//
// Parameters:
//   - gravity: gravitational acceleration in m/s² (e.g. [3]float32{0, -9.81, 0})
//
// Returns:
//   - PhysicsBuilderOption: a function that applies the gravity option to a physicsImpl
func WithGravity(gravity [3]float32) PhysicsBuilderOption {
	return func(p *physicsImpl) {
		p.gravity = gravity
	}
}

// WithBoundaryPlanes sets the analytical containment half-planes used for
// wall collision in the collision shader. Each plane is defined as
// [nx, ny, nz, d, y_min, y_max] where the normal points INTO the containment volume,
// dot(n, p) + d >= 0 indicates p is inside, and the plane is only active when the
// particle’s Y coordinate is within [y_min, y_max]. Up to 6 planes are supported.
//
// Parameters:
//   - planes: slice of [6]float32 plane definitions (max 6)
//
// Returns:
//   - PhysicsBuilderOption: a function that applies the boundary plane option to a physicsImpl
func WithBoundaryPlanes(planes [][6]float32) PhysicsBuilderOption {
	return func(p *physicsImpl) {
		n := min(len(planes), 6)
		p.boundaryCount = n
		for i := range n {
			p.boundaryPlanes[i] = [4]float32{planes[i][0], planes[i][1], planes[i][2], planes[i][3]}
			p.boundaryYRanges[i] = [4]float32{planes[i][4], planes[i][5], 0, 0}
		}
	}
}

// NewPhysics creates a new Physics instance with sensible defaults. The returned
// system starts disabled and becomes active upon the first RegisterBody call.
// Use PhysicsBuilderOption values to override default configuration.
//
// Parameters:
//   - options: variadic list of PhysicsBuilderOption functions to configure the physics system
//
// Returns:
//   - Physics: a new Physics instance
func NewPhysics(options ...PhysicsBuilderOption) Physics {
	r := &physicsImpl{
		enabled:         false,
		bodiesMap:       make(map[uint64]int),
		stagedWriteData: make([]bind_group_provider.BufferWrite, 0, 16),
		buffers:         bind_group_provider.NewBindGroupProvider("physics_buffers"),
		bgps: map[string]bind_group_provider.BindGroupProvider{
			"particle_values":   bind_group_provider.NewBindGroupProvider("particle_values"),
			"aabb_reduce":       bind_group_provider.NewBindGroupProvider("aabb_reduce"),
			"grid_build_params": bind_group_provider.NewBindGroupProvider("grid_build_params"),
			"grid_clear":        bind_group_provider.NewBindGroupProvider("grid_clear"),
			"grid_insert":       bind_group_provider.NewBindGroupProvider("grid_insert"),
			"collision":         bind_group_provider.NewBindGroupProvider("collision"),
			"momenta":           bind_group_provider.NewBindGroupProvider("momenta"),
			"integrate":         bind_group_provider.NewBindGroupProvider("integrate"),
			"sync":              bind_group_provider.NewBindGroupProvider("sync"),
		},
		pipelineKeys: make(map[string]string),
		fixedDt:      1.0 / 60.0,
		maxSubsteps:  4,
		maxBodies:    256,
		maxParticles: 2048,
		maxGridCells: 128 * 128 * 128,
		springCoeff:  1.0,
		dampingCoeff: 0.1,
		shearCoeff:   0.5,
		slotsPerCell: 16,
		bodyIdxMask:  0xFFFFFF,
	}
	for _, opt := range options {
		opt(r)
	}
	return r
}
