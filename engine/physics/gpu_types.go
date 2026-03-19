package physics

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"
)

// GPUBodySource is the canonical WGSL definition of the Body struct.
// Matches GPUBody layout exactly (160 bytes, std430 aligned).
//
//go:embed assets/body.wgsl
var GPUBodySource string

// GPUBody is the GPU-aligned representation of a single rigid body.
// One element per registered body in the physics storage buffer. Maps to the
// paper's "textures for linear and angular momenta, position, and the
// orientation quaternion" (GPU Gems 3, §29.2.2), packed into a single storage
// buffer instead of separate textures.
// Matches the WGSL Body struct layout exactly (see GPUBodySource).
// Size: 160 bytes (std430 / WGSL aligned).
type GPUBody struct {
	Position        [4]float32  // offset   0: world-space center of mass (xyz), w unused       — §29.1.1 Eq. 3
	Quaternion      [4]float32  // offset  16: orientation quaternion (xyzw)                     — §29.1.2 Eq. 8
	LinearMomentum  [4]float32  // offset  32: linear momentum (xyz), w unused                  — §29.1.1 Eq. 1
	AngularMomentum [4]float32  // offset  48: angular momentum (xyz), w unused                 — §29.1.2 Eq. 4
	InvInertiaTBody [12]float32 // offset  64: body-space inverse inertia tensor (mat3x3<f32>)  — §29.1.2 Eq. 6
	InverseMass     float32     // offset 112: 1/mass (0 for static bodies)                    — §29.1.1 Eq. 2
	ParticleStart   uint32      // offset 116: start index into the global particle buffer      — §29.2.2
	ParticleCount   uint32      // offset 120: number of particles belonging to this body       — §29.2.2
	Flags           uint32      // offset 124: bit-packed: active (bit 0), static (1), kinematic (2)
	ExternalForce   [4]float32  // offset 128: CPU-uploaded force for this frame (xyz), w unused
	ExternalTorque  [4]float32  // offset 144: CPU-uploaded torque for this frame (xyz), w unused
}

// Size returns the size of the GPUBody struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (160)
func (g *GPUBody) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUBody struct into a byte buffer suitable for GPU upload.
// The mat3x3<f32> field is marshaled as 3 columns of vec3<f32>, each padded to
// 16 bytes (vec4 stride) per WGSL storage buffer layout rules.
//
// Returns:
//   - []byte: 160-byte buffer ready for GPU upload
func (g *GPUBody) Marshal() []byte {
	buf := make([]byte, g.Size())
	off := 0

	// Position — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.Position[i]))
		off += 4
	}

	// Quaternion — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.Quaternion[i]))
		off += 4
	}

	// LinearMomentum — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.LinearMomentum[i]))
		off += 4
	}

	// AngularMomentum — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.AngularMomentum[i]))
		off += 4
	}

	// InvInertiaTBody — mat3x3<f32> as 3 × padded vec4 (48 bytes)
	// Go layout: [col0.x, col0.y, col0.z, _pad0, col1.x, col1.y, col1.z, _pad1, col2.x, col2.y, col2.z, _pad2]
	for col := range 3 {
		base := col * 4
		for row := range 3 {
			binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.InvInertiaTBody[base+row]))
			off += 4
		}
		// column padding to vec4 stride
		binary.LittleEndian.PutUint32(buf[off:], 0)
		off += 4
	}

	// InverseMass — f32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.InverseMass))
	off += 4

	// ParticleStart — u32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], g.ParticleStart)
	off += 4

	// ParticleCount — u32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], g.ParticleCount)
	off += 4

	// Flags — u32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], g.Flags)
	off += 4

	// ExternalForce — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.ExternalForce[i]))
		off += 4
	}

	// ExternalTorque — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.ExternalTorque[i]))
		off += 4
	}

	return buf
}

// GPUParticleSource is the canonical WGSL definition of the Particle struct.
// Matches GPUParticle layout exactly (96 bytes, std430 aligned).
//
//go:embed assets/particle.wgsl
var GPUParticleSource string

// GPUParticle is the GPU-aligned representation of a single particle across all
// rigid bodies. One element per particle in the global particle storage buffer.
// Maps to the paper's "textures for the particle's position, velocity, and
// relative position to the center of mass" (GPU Gems 3, §29.2.2).
// Matches the WGSL Particle struct layout exactly (see GPUParticleSource).
// Size: 96 bytes (std430 / WGSL aligned).
type GPUParticle struct {
	WorldPosition [4]float32 // offset  0: current world-space position (xyz), w unused              — §29.2.3 Eq. 19
	Velocity      [4]float32 // offset 16: current velocity (xyz), w unused                          — §29.2.3 Eq. 20
	RelPosition   [4]float32 // offset 32: current relative position to body center (xyz), w unused  — §29.2.3 Eq. 18
	Force         [4]float32 // offset 48: accumulated collision force (xyz), w unused                — §29.2.5 Eqs. 10–12
	LocalPosition [4]float32 // offset 64: body-local rest position (xyz), w = bitcast<f32>(bodyIndex) — §29.2.3
	SurfaceNormal [4]float32 // offset 80: outward surface normal (xyz), w = 1.0 if valid (static)  — Fix A
}

// Size returns the size of the GPUParticle struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (96)
func (g *GPUParticle) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUParticle struct into a byte buffer suitable for GPU upload.
// LocalPosition.w carries the owning body index as bitcast<f32>(u32); all other .w
// components are unused and marshaled as-is.
//
// Returns:
//   - []byte: 96-byte buffer ready for GPU upload
func (g *GPUParticle) Marshal() []byte {
	buf := make([]byte, g.Size())
	off := 0

	// WorldPosition — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.WorldPosition[i]))
		off += 4
	}

	// Velocity — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.Velocity[i]))
		off += 4
	}

	// RelPosition — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.RelPosition[i]))
		off += 4
	}

	// Force — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.Force[i]))
		off += 4
	}

	// LocalPosition — vec4<f32> (16 bytes)
	// xyz = body-local rest position, w = body index stored via bitcast<f32>(u32)
	for i := range 3 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.LocalPosition[i]))
		off += 4
	}
	// w component: already a float32 whose bits represent the body index (set by
	// the caller via math.Float32frombits(bodyIndex)), so marshal directly
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.LocalPosition[3]))
	off += 4

	// SurfaceNormal — vec4<f32> (16 bytes)
	// xyz = outward surface normal, w = 1.0 if valid (static wall particle)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.SurfaceNormal[i]))
		off += 4
	}

	return buf
}

// GPUGridCellSource is the canonical WGSL definition of the GridCell struct.
// Matches GPUGridCell layout exactly (32 bytes, std430 aligned).
//
//go:embed assets/grid-cell.wgsl
var GPUGridCellSource string

// GPUGridCell is the GPU-aligned representation of a single cell in the uniform
// spatial grid used for collision broad-phase. Maps to the paper's "flat 3D
// texture" grid (GPU Gems 3, §29.2.2, §29.2.4) adapted for compute shader
// storage buffers. Each cell stores up to 16 particle indices; empty slots use
// the sentinel value 0xFFFFFFFF.
// Matches the WGSL GridCell struct layout exactly (see GPUGridCellSource).
// Size: 64 bytes (std430 / WGSL aligned).
type GPUGridCell struct {
	Indices0 [4]uint32 // offset  0: slots  0-3  particle indices (0xFFFFFFFF = empty) — §29.2.4
	Indices1 [4]uint32 // offset 16: slots  4-7  particle indices (0xFFFFFFFF = empty)
	Indices2 [4]uint32 // offset 32: slots  8-11 particle indices (0xFFFFFFFF = empty)
	Indices3 [4]uint32 // offset 48: slots 12-15 particle indices (0xFFFFFFFF = empty)
}

// Size returns the size of the GPUGridCell struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (64)
func (g *GPUGridCell) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUGridCell struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 64-byte buffer ready for GPU upload
func (g *GPUGridCell) Marshal() []byte {
	buf := make([]byte, g.Size())
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[i*4:], g.Indices0[i])
	}
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[16+i*4:], g.Indices1[i])
	}
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[32+i*4:], g.Indices2[i])
	}
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[48+i*4:], g.Indices3[i])
	}
	return buf
}

// GPUPhysicsGlobalsSource is the canonical WGSL definition of the PhysicsGlobals struct.
// Matches GPUPhysicsGlobals layout exactly (240 bytes, std140/WGSL uniform aligned).
//
//go:embed assets/physics-globals.wgsl
var GPUPhysicsGlobalsSource string

// GPUPhysicsGlobals is the GPU-aligned representation of the per-frame physics
// uniform buffer. A single instance is uploaded each frame, shared across all
// physics compute stages. Contains simulation constants (timestep, DEM coefficients),
// body/particle counts, and up to 6 analytical boundary half-planes with Y-height ranges.
// Grid-related fields (origin, dims) are stored separately in GPUGridParams so
// the GPU AABB reduction shader can write them directly without a CPU round-trip.
// Matches the WGSL PhysicsGlobals struct layout exactly (see GPUPhysicsGlobalsSource).
// Size: 240 bytes (std140 / WGSL uniform aligned).
type GPUPhysicsGlobals struct {
	DeltaTime        float32       // offset  0: simulation timestep in seconds                         — Eqs. 1, 3, 4, 7–8
	ParticleDiameter float32       // offset  4: diameter of all particles (2 × radius)                 — §29.1.4
	SpringCoeff      float32       // offset  8: DEM spring coefficient k                               — §29.1.5 Eq. 10
	DampingCoeff     float32       // offset 12: DEM damping coefficient η                              — §29.1.5 Eq. 11
	ShearCoeff       float32       // offset 16: DEM tangential friction coefficient μ_t                — §29.1.5 Eq. 12
	BodyCount        uint32        // offset 20: number of active rigid bodies
	ParticleCount    uint32        // offset 24: total particle count across all bodies
	MaxGridCells     uint32        // offset 28: maximum total grid cells (x*y*z cap for AABB reduction)
	BoundaryCount    uint32        // offset 32: number of active boundary planes (0–6)
	GravityX         float32       // offset 36: gravitational acceleration X component (m/s²)
	GravityY         float32       // offset 40: gravitational acceleration Y component (m/s²)
	GravityZ         float32       // offset 44: gravitational acceleration Z component (m/s²)
	BoundaryPlanes   [6][4]float32 // offset  48: up to 6 half-plane definitions (normal.xyz, d)
	BoundaryYRanges  [6][4]float32 // offset 144: per-plane Y activation range (y_min, y_max, 0, 0)
}

// Size returns the size of the GPUPhysicsGlobals struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (240)
func (g *GPUPhysicsGlobals) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUPhysicsGlobals struct into a byte buffer suitable for
// GPU upload. Explicit padding bytes are written as zero to match WGSL alignment.
//
// Returns:
//   - []byte: 240-byte buffer ready for GPU upload
func (g *GPUPhysicsGlobals) Marshal() []byte {
	buf := make([]byte, g.Size())
	off := 0

	// DeltaTime — f32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.DeltaTime))
	off += 4

	// ParticleDiameter — f32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.ParticleDiameter))
	off += 4

	// SpringCoeff — f32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.SpringCoeff))
	off += 4

	// DampingCoeff — f32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.DampingCoeff))
	off += 4

	// ShearCoeff — f32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.ShearCoeff))
	off += 4

	// BodyCount — u32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], g.BodyCount)
	off += 4

	// ParticleCount — u32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], g.ParticleCount)
	off += 4

	// MaxGridCells — u32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], g.MaxGridCells)
	off += 4

	// BoundaryCount — u32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], g.BoundaryCount)
	off += 4

	// GravityX — f32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.GravityX))
	off += 4

	// GravityY — f32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.GravityY))
	off += 4

	// GravityZ — f32 (4 bytes)
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.GravityZ))
	off += 4

	// BoundaryPlanes — 6 × vec4<f32> (96 bytes)
	for i := range 6 {
		for j := range 4 {
			binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.BoundaryPlanes[i][j]))
			off += 4
		}
	}

	// BoundaryYRanges — 6 × vec4<f32> (96 bytes)
	for i := range 6 {
		for j := range 4 {
			binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.BoundaryYRanges[i][j]))
			off += 4
		}
	}

	return buf
}

// GPUGridParamsSource is the canonical WGSL definition of the GridParams struct.
// Matches GPUGridParams layout exactly (32 bytes, std430 aligned).
//
//go:embed assets/grid-params.wgsl
var GPUGridParamsSource string

// GPUGridParams is the GPU-aligned representation of the spatial grid parameters.
// Written by the GPU AABB reduction shader (Stage 1.5) rather than uploaded from
// the CPU, which is why these fields live in a separate storage buffer instead of
// inside GPUPhysicsGlobals.
// Matches the WGSL GridParams struct layout exactly (see GPUGridParamsSource).
// Size: 32 bytes (std430 / WGSL aligned).
type GPUGridParams struct {
	GridOrigin [4]float32 // offset  0: world-space grid origin (xyz), w unused  — §29.1.4 Eq. 9
	GridDims   [4]uint32  // offset 16: grid dimensions (x, y, z), w = total cell count — §29.2.2
}

// Size returns the size of the GPUGridParams struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (32)
func (g *GPUGridParams) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUGridParams struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 32-byte buffer ready for GPU upload
func (g *GPUGridParams) Marshal() []byte {
	buf := make([]byte, g.Size())
	off := 0

	// GridOrigin — vec4<f32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(g.GridOrigin[i]))
		off += 4
	}

	// GridDims — vec4<u32> (16 bytes)
	for i := range 4 {
		binary.LittleEndian.PutUint32(buf[off:], g.GridDims[i])
		off += 4
	}

	return buf
}
