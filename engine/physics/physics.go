package physics

import (
	"encoding/binary"
	"math"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/gogpu/wgpu"
)

// physicsImpl is the implementation of the Physics interface. It manages GPU-side
// rigid body simulation state including body/particle registration, fixed-timestep
// accumulation, staged buffer writes, and asynchronous GPU readback.
type physicsImpl struct {
	enabled bool

	bodies        []RigidBody
	bodiesMap     map[uint64]int
	bodiesCount   int
	particleCount int
	syncMap       []uint32

	freeBodySlots    []int       // stack of reusable body indices
	bodyParticleInfo [][2]uint32 // [particleStart, particleCount] per body slot

	buffers bind_group_provider.BindGroupProvider
	bgps    map[string]bind_group_provider.BindGroupProvider

	pipelineKeys map[string]string

	accumulator      float32
	fixedDt          float32
	maxSubsteps      int
	maxBodies        uint32
	maxParticles     uint32
	maxGridCells     uint32
	particleDiameter float32

	springCoeff, dampingCoeff, shearCoeff float32

	gravity [3]float32

	boundaryPlanes  [6][4]float32
	boundaryYRanges [6][4]float32
	boundaryCount   uint32

	readbackRequested bool
	readbackPending   bool
	stagingBuffer     *wgpu.Buffer
	stagedWriteData   []bind_group_provider.BufferWrite
}

// Physics defines the interface for the GPU-accelerated rigid body physics system.
// It manages body/particle registration, per-frame simulation stepping, GPU buffer
// staging, and asynchronous readback of simulation results. All physics compute
// pipeline dispatches and buffer bindings are driven through this interface.
type Physics interface {
	// Enabled returns whether the physics system is active. It becomes true when
	// at least one body is registered and false when all bodies are removed.
	//
	// Returns:
	//   - bool: true if physics simulation is active
	Enabled() bool

	// BodiesCount returns the number of body slots currently allocated (including
	// slots on the free list that have been deactivated but not yet reclaimed).
	//
	// Returns:
	//   - int: the total number of allocated body slots
	BodiesCount() int

	// ParticleCount returns the total number of particles across all registered bodies.
	//
	// Returns:
	//   - int: the total particle count
	ParticleCount() int

	// Buffers returns the primary BindGroupProvider that owns the physics storage
	// buffers (bodies at binding 0, particles at binding 1, etc.).
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the physics buffer provider
	Buffers() bind_group_provider.BindGroupProvider

	// Bgp returns the BindGroupProvider associated with the given compute stage key
	// (e.g. "collision", "integrate", "grid_insert").
	//
	// Parameters:
	//   - key: the stage identifier
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the provider for the requested stage, or nil if not found
	Bgp(key string) bind_group_provider.BindGroupProvider

	// Bgps returns the full map of compute-stage BindGroupProviders keyed by stage name.
	//
	// Returns:
	//   - map[string]bind_group_provider.BindGroupProvider: all compute-stage providers
	Bgps() map[string]bind_group_provider.BindGroupProvider

	// PipelineKey returns the compute pipeline cache key for the given stage name.
	//
	// Parameters:
	//   - key: the stage identifier
	//
	// Returns:
	//   - string: the pipeline cache key associated with the stage
	PipelineKey(key string) string

	// MaxBodies returns the upper limit of rigid bodies the system can hold.
	//
	// Returns:
	//   - uint32: the maximum body count
	MaxBodies() uint32

	// MaxParticles returns the upper limit of particles across all bodies.
	//
	// Returns:
	//   - uint32: the maximum particle count
	MaxParticles() uint32

	// MaxGridCells returns the upper limit of spatial grid cells (x*y*z cap).
	//
	// Returns:
	//   - uint32: the maximum grid cell count
	MaxGridCells() uint32

	// SetPipelineKey associates a compute pipeline cache key with the given stage name.
	//
	// Parameters:
	//   - name: the stage identifier
	//   - key: the pipeline cache key to store
	SetPipelineKey(name, key string)

	// RegisterBody registers a rigid body in the physics system, uploading its GPU
	// data (body struct, particles, sync mapping) via staged buffer writes. If a
	// previously freed slot exists it is reused; otherwise a new slot is allocated.
	//
	// Parameters:
	//   - objID: the unique game-object identifier used for lookup
	//   - position: world-space spawn position (XYZ)
	//   - rotation: Euler rotation angles in radians (XYZ)
	//   - rb: the RigidBody carrying mass, particles, and physics properties
	//   - instanceID: the Animator instance ID written to the sync map
	//
	// Returns:
	//   - int: the body slot index assigned to this registration
	RegisterBody(objID uint64, position, rotation [3]float32, rb RigidBody, instanceID uint32) int

	// RemoveBody deactivates a body by zeroing its GPU flags and inverse mass,
	// and returns the body slot to the free list for reuse by future registrations.
	//
	// Parameters:
	//   - objID: the unique game-object identifier to remove
	RemoveBody(objID uint64)

	// BodyIndex returns the GPU body buffer slot index for the given object ID.
	//
	// Parameters:
	//   - objID: the unique object identifier
	//
	// Returns:
	//   - int: the zero-based body slot index
	//   - bool: true if the object was found
	BodyIndex(objID uint64) (int, bool)

	// BodyParticleInfo returns the particle start index and count for a given body slot.
	//
	// Parameters:
	//   - bodyIndex: the body slot index returned by RegisterBody
	//
	// Returns:
	//   - start: the first particle index in the global particle buffer
	//   - count: the number of particles belonging to this body
	BodyParticleInfo(bodyIndex int) (start, count uint32)

	// StagedWriteData drains and returns all pending GPU buffer writes that have been
	// queued since the last call. The internal queue is reset after draining.
	//
	// Returns:
	//   - []bind_group_provider.BufferWrite: the accumulated buffer writes
	StagedWriteData() []bind_group_provider.BufferWrite

	// PrepareStep advances the physics accumulator by dt and determines how many
	// fixed-timestep substeps to execute this frame. It also drains pending
	// per-body forces/torques into staged buffer writes and builds the
	// GPUPhysicsGlobals uniform data.
	//
	// Parameters:
	//   - dt: wall-clock frame delta time in seconds
	//
	// Returns:
	//   - substeps: the number of fixed-timestep substeps to dispatch (0 if none)
	//   - globalsData: serialized GPUPhysicsGlobals bytes for GPU upload, or nil if substeps is 0
	PrepareStep(dt float32) (substeps int, globalsData []byte)

	// RequestReadback flags that a GPU-to-CPU readback of the body buffer should
	// be initiated on the next command submission.
	RequestReadback()

	// ConsumeReadbackRequest checks whether a readback was requested, clears the
	// request flag, and sets readbackPending so the scene knows a copy is in-flight.
	//
	// Returns:
	//   - bool: true if a readback was requested and consumed
	ConsumeReadbackRequest() bool

	// ReadbackPending returns whether a GPU readback copy is currently in-flight.
	//
	// Returns:
	//   - bool: true if a readback is pending
	ReadbackPending() bool

	// ClearReadbackPending resets the pending readback flag after the mapping
	// callback has completed.
	ClearReadbackPending()

	// StagingBuffer returns the GPU staging buffer used for readback mapping.
	//
	// Returns:
	//   - *wgpu.Buffer: the staging buffer, or nil if not yet created
	StagingBuffer() *wgpu.Buffer

	// SetStagingBuffer sets the GPU staging buffer used for readback mapping.
	//
	// Parameters:
	//   - buf: the staging buffer to store
	SetStagingBuffer(buf *wgpu.Buffer)

	// ProcessReadback unmarshals body positions and quaternions from the raw GPU
	// body buffer readback data and updates the corresponding CPU-side RigidBody
	// state. Each GPUBody is 160 bytes; only the Position (offset 0, 16 bytes)
	// and Quaternion (offset 16, 16 bytes) are extracted.
	//
	// Parameters:
	//   - data: the raw byte slice containing bodiesCount × 160 bytes of GPUBody data
	ProcessReadback(data []byte)
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
	}
	for _, opt := range options {
		opt(r)
	}
	return r
}

func (p *physicsImpl) Enabled() bool {
	return p.enabled
}

func (p *physicsImpl) BodiesCount() int {
	return p.bodiesCount
}

func (p *physicsImpl) ParticleCount() int {
	return p.particleCount
}

func (p *physicsImpl) Buffers() bind_group_provider.BindGroupProvider {
	return p.buffers
}

func (p *physicsImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return p.bgps[key]
}

func (p *physicsImpl) Bgps() map[string]bind_group_provider.BindGroupProvider {
	return p.bgps
}

func (p *physicsImpl) PipelineKey(key string) string {
	return p.pipelineKeys[key]
}

// PipelineKeys returns the full map of compute pipeline cache keys keyed by stage name.
//
// Returns:
//   - map[string]string: all pipeline key mappings
func (p *physicsImpl) PipelineKeys() map[string]string {
	return p.pipelineKeys
}

func (p *physicsImpl) MaxBodies() uint32 {
	return p.maxBodies
}

func (p *physicsImpl) MaxParticles() uint32 {
	return p.maxParticles
}

func (p *physicsImpl) MaxGridCells() uint32 {
	return p.maxGridCells
}

func (p *physicsImpl) SetPipelineKey(name, key string) {
	p.pipelineKeys[name] = key
}

func (p *physicsImpl) RequestReadback() {
	p.readbackRequested = true
}

func (p *physicsImpl) ConsumeReadbackRequest() bool {
	if !p.readbackRequested {
		return false
	}
	p.readbackRequested = false
	p.readbackPending = true
	return true
}

func (p *physicsImpl) ReadbackPending() bool {
	return p.readbackPending
}

func (p *physicsImpl) ClearReadbackPending() {
	p.readbackPending = false
}

func (p *physicsImpl) StagingBuffer() *wgpu.Buffer {
	return p.stagingBuffer
}

func (p *physicsImpl) SetStagingBuffer(buf *wgpu.Buffer) {
	p.stagingBuffer = buf
}

func (p *physicsImpl) ProcessReadback(data []byte) {
	bodySize := (&GPUBody{}).Size()
	count := p.bodiesCount
	if len(data) < count*bodySize {
		count = len(data) / bodySize
	}

	for i := 0; i < count; i++ {
		base := i * bodySize

		// Position: offset 0, 4 × float32 (xyz, w unused)
		var pos [3]float32
		pos[0] = math.Float32frombits(binary.LittleEndian.Uint32(data[base : base+4]))
		pos[1] = math.Float32frombits(binary.LittleEndian.Uint32(data[base+4 : base+8]))
		pos[2] = math.Float32frombits(binary.LittleEndian.Uint32(data[base+8 : base+12]))

		// Quaternion: offset 16, 4 × float32 (xyzw)
		var quat [4]float32
		quat[0] = math.Float32frombits(binary.LittleEndian.Uint32(data[base+16 : base+20]))
		quat[1] = math.Float32frombits(binary.LittleEndian.Uint32(data[base+20 : base+24]))
		quat[2] = math.Float32frombits(binary.LittleEndian.Uint32(data[base+24 : base+28]))
		quat[3] = math.Float32frombits(binary.LittleEndian.Uint32(data[base+28 : base+32]))

		if i < len(p.bodies) && p.bodies[i] != nil {
			p.bodies[i].SetPosition(pos)
			p.bodies[i].SetQuaternion(quat)
		}
	}
}

func (p *physicsImpl) RegisterBody(objID uint64, position, rotation [3]float32, rb RigidBody, instanceID uint32) int {
	particleCountU := uint32(len(rb.Particles()))

	var bodyIndex int
	var particleStart uint32

	if len(p.freeBodySlots) > 0 {
		// Reuse a freed slot — keeps bodiesCount/particleCount bounded.
		bodyIndex = p.freeBodySlots[len(p.freeBodySlots)-1]
		p.freeBodySlots = p.freeBodySlots[:len(p.freeBodySlots)-1]
		particleStart = p.bodyParticleInfo[bodyIndex][0]
		p.bodies[bodyIndex] = rb
		p.syncMap[bodyIndex] = instanceID
	} else {
		// Allocate a new slot.
		bodyIndex = p.bodiesCount
		particleStart = uint32(p.particleCount)
		p.bodies = append(p.bodies, rb)
		p.syncMap = append(p.syncMap, instanceID)
		p.bodyParticleInfo = append(p.bodyParticleInfo, [2]uint32{})
		p.bodiesCount++
		p.particleCount += int(particleCountU)
	}

	p.bodiesMap[objID] = bodyIndex
	p.bodyParticleInfo[bodyIndex] = [2]uint32{particleStart, particleCountU}

	quat := eulerToQuaternion(rotation)

	// linear momentum: P = mass * velocity (§29.1.1 Eq. 1)
	vel := rb.Velocity()
	mass := rb.Mass()
	linearMomentum := [4]float32{vel[0] * mass, vel[1] * mass, vel[2] * mass, 0}

	// angular momentum: L = I_body * ω (§29.1.2 Eq. 4)
	// recover I_body from the stored inverse via common.Invert3x3
	invI := rb.InverseInertiaTensorBody()
	inertiaTensor, ok := common.Invert3x3(invI)
	angVel := rb.AngularVelocity()
	var angularMomentum [4]float32
	if ok {
		angularMomentum = [4]float32{
			inertiaTensor[0]*angVel[0] + inertiaTensor[1]*angVel[1] + inertiaTensor[2]*angVel[2],
			inertiaTensor[3]*angVel[0] + inertiaTensor[4]*angVel[1] + inertiaTensor[5]*angVel[2],
			inertiaTensor[6]*angVel[0] + inertiaTensor[7]*angVel[1] + inertiaTensor[8]*angVel[2],
			0,
		}
	}

	var flags uint32
	if rb.Active() {
		flags |= 1
	}
	if rb.Static() {
		flags |= 2
	}
	if rb.Kinematic() {
		flags |= 4
	}

	// convert row-major [9]float32 inverse inertia tensor to column-major padded [12]float32
	// for WGSL mat3x3<f32> storage layout (3 columns × vec4 stride)
	invIPadded := [12]float32{
		invI[0], invI[3], invI[6], 0,
		invI[1], invI[4], invI[7], 0,
		invI[2], invI[5], invI[8], 0,
	}

	particleStart = p.bodyParticleInfo[bodyIndex][0]
	particleCount := particleCountU

	gpuBody := GPUBody{
		Position:        [4]float32{position[0], position[1], position[2], 0},
		Quaternion:      quat,
		LinearMomentum:  linearMomentum,
		AngularMomentum: angularMomentum,
		InvInertiaTBody: invIPadded,
		InverseMass:     rb.InverseMass(),
		ParticleStart:   particleStart,
		ParticleCount:   particleCount,
		Flags:           flags,
	}

	// stage body data write at the correct slot offset
	p.stagedWriteData = append(p.stagedWriteData, bind_group_provider.BufferWrite{
		Provider: p.buffers,
		Binding:  0,
		Offset:   uint64(bodyIndex) * 160,
		Data:     gpuBody.Marshal(),
	})

	// batch all particles into a single write at the correct start offset
	particles := rb.Particles()
	particleSize := (&GPUParticle{}).Size()
	isStatic := rb.Static()
	particleData := make([]byte, 0, len(particles)*particleSize)
	for _, part := range particles {
		// Pack bodyIndex in the low 24 bits and boneIndex in the upper 8 bits
		// of local_position.w. For non-kinematic particles boneIndex is 0 so
		// the packed value equals bodyIndex, preserving backward compatibility.
		packed := uint32(bodyIndex) | (part.BoneIndex << 24)

		// Surface normal validity flag: w = 1.0 for static body particles
		// that have a computed surface normal (enables Fix A gating in the
		// collision shader). Kinematic/dynamic particles get w = 0.0 so the
		// gating check is skipped (their normals aren't in world space).
		var snW float32
		if isStatic {
			snW = 1.0
		}

		gpuPart := GPUParticle{
			LocalPosition: [4]float32{
				part.LocalPosition[0],
				part.LocalPosition[1],
				part.LocalPosition[2],
				math.Float32frombits(packed),
			},
			SurfaceNormal: [4]float32{
				part.SurfaceNormal[0],
				part.SurfaceNormal[1],
				part.SurfaceNormal[2],
				snW,
			},
		}
		particleData = append(particleData, gpuPart.Marshal()...)
	}

	p.stagedWriteData = append(p.stagedWriteData, bind_group_provider.BufferWrite{
		Provider: p.buffers,
		Binding:  1,
		Offset:   uint64(particleStart) * uint64(particleSize),
		Data:     particleData,
	})

	// stage sync mapping entry (body index → Animator instance ID)
	syncData := make([]byte, 4)
	binary.LittleEndian.PutUint32(syncData, instanceID)
	p.stagedWriteData = append(p.stagedWriteData, bind_group_provider.BufferWrite{
		Provider: p.buffers,
		Binding:  7,
		Offset:   uint64(bodyIndex) * 4,
		Data:     syncData,
	})

	// derive particle diameter from first registered body
	if p.particleDiameter == 0 && rb.ParticleRadius() > 0 {
		p.particleDiameter = rb.ParticleRadius() * 2
	}

	p.enabled = true
	return bodyIndex
}

func (p *physicsImpl) RemoveBody(objID uint64) {
	idx, ok := p.bodiesMap[objID]
	if !ok {
		return
	}

	delete(p.bodiesMap, objID)
	p.bodies[idx] = nil
	p.freeBodySlots = append(p.freeBodySlots, idx)

	// zero flags (offset 124 within GPUBody) to mark inactive
	flagsData := make([]byte, 4)
	p.stagedWriteData = append(p.stagedWriteData, bind_group_provider.BufferWrite{
		Provider: p.buffers,
		Binding:  0,
		Offset:   uint64(idx)*160 + 124,
		Data:     flagsData,
	})

	// zero inverse mass (offset 112) to prevent integration
	invMassData := make([]byte, 4)
	p.stagedWriteData = append(p.stagedWriteData, bind_group_provider.BufferWrite{
		Provider: p.buffers,
		Binding:  0,
		Offset:   uint64(idx)*160 + 112,
		Data:     invMassData,
	})

	if len(p.bodiesMap) == 0 {
		p.enabled = false
	}
}

func (p *physicsImpl) BodyIndex(objID uint64) (int, bool) {
	idx, ok := p.bodiesMap[objID]
	return idx, ok
}

func (p *physicsImpl) BodyParticleInfo(bodyIndex int) (start, count uint32) {
	if bodyIndex < 0 || bodyIndex >= len(p.bodyParticleInfo) {
		return 0, 0
	}
	info := p.bodyParticleInfo[bodyIndex]
	return info[0], info[1]
}

func (p *physicsImpl) StagedWriteData() []bind_group_provider.BufferWrite {
	w := p.stagedWriteData
	p.stagedWriteData = p.stagedWriteData[:0]
	return w
}

func (p *physicsImpl) PrepareStep(dt float32) (substeps int, globalsData []byte) {
	if !p.enabled || p.bodiesCount == 0 {
		return 0, nil
	}

	// drain pending forces/torques from all registered RigidBodies and stage
	// buffer writes targeting GPUBody.ExternalForce (offset 128) and GPUBody.ExternalTorque (offset 144)
	for i, rb := range p.bodies {
		if rb == nil {
			continue
		}
		rbImpl, ok := rb.(*rigidBody)
		if !ok {
			continue
		}

		force := rbImpl.drainForce()
		torque := rbImpl.drainTorque()

		hasForce := force[0] != 0 || force[1] != 0 || force[2] != 0
		hasTorque := torque[0] != 0 || torque[1] != 0 || torque[2] != 0

		if hasForce {
			data := make([]byte, 16)
			for j := 0; j < 3; j++ {
				binary.LittleEndian.PutUint32(data[j*4:], math.Float32bits(force[j]))
			}
			p.stagedWriteData = append(p.stagedWriteData, bind_group_provider.BufferWrite{
				Provider: p.buffers,
				Binding:  0,
				Offset:   uint64(i)*160 + 128,
				Data:     data,
			})
		}

		if hasTorque {
			data := make([]byte, 16)
			for j := 0; j < 3; j++ {
				binary.LittleEndian.PutUint32(data[j*4:], math.Float32bits(torque[j]))
			}
			p.stagedWriteData = append(p.stagedWriteData, bind_group_provider.BufferWrite{
				Provider: p.buffers,
				Binding:  0,
				Offset:   uint64(i)*160 + 144,
				Data:     data,
			})
		}
	}

	p.accumulator += dt
	substeps = 0
	for p.accumulator >= p.fixedDt && substeps < p.maxSubsteps {
		p.accumulator -= p.fixedDt
		substeps++
	}

	// Clamp leftover accumulator so unprocessed time never snowballs across
	// frames. Without this cap, sustained low frame-rates accumulate a growing
	// time debt that the fixed-timestep loop can never repay, causing the
	// simulation to fall permanently behind wall-clock time.
	if p.accumulator > p.fixedDt {
		p.accumulator = p.fixedDt
	}

	if substeps == 0 {
		return 0, nil
	}

	globals := GPUPhysicsGlobals{
		DeltaTime:        p.fixedDt,
		ParticleDiameter: p.particleDiameter,
		SpringCoeff:      p.springCoeff,
		DampingCoeff:     p.dampingCoeff,
		ShearCoeff:       p.shearCoeff,
		BodyCount:        uint32(p.bodiesCount),
		ParticleCount:    uint32(p.particleCount),
		MaxGridCells:     p.maxGridCells,
		BoundaryCount:    p.boundaryCount,
		GravityX:         p.gravity[0],
		GravityY:         p.gravity[1],
		GravityZ:         p.gravity[2],
		BoundaryPlanes:   p.boundaryPlanes,
		BoundaryYRanges:  p.boundaryYRanges,
	}

	return substeps, globals.Marshal()
}

// eulerToQuaternion converts XYZ Euler angles (in radians) to a unit quaternion
// stored as [x, y, z, w].
//
// Parameters:
//   - euler: rotation angles as [3]float32 in XYZ order (radians)
//
// Returns:
//   - [4]float32: unit quaternion as [x, y, z, w]
func eulerToQuaternion(euler [3]float32) [4]float32 {
	cx := float32(math.Cos(float64(euler[0] * 0.5)))
	sx := float32(math.Sin(float64(euler[0] * 0.5)))
	cy := float32(math.Cos(float64(euler[1] * 0.5)))
	sy := float32(math.Sin(float64(euler[1] * 0.5)))
	cz := float32(math.Cos(float64(euler[2] * 0.5)))
	sz := float32(math.Sin(float64(euler[2] * 0.5)))

	return [4]float32{
		sx*cy*cz - cx*sy*sz,
		cx*sy*cz + sx*cy*sz,
		cx*cy*sz - sx*sy*cz,
		cx*cy*cz + sx*sy*sz,
	}
}
