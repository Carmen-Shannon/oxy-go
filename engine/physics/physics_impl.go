package physics

import (
	"math"

	"github.com/oliverbestmann/webgpu/wgpu"

	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// physicsImpl is the implementation of the Physics interface. It manages GPU-side
// rigid body simulation state including body/particle registration, fixed-timestep
// accumulation, staged buffer writes, and asynchronous GPU readback.
type physicsImpl struct {
	bodies        []RigidBody
	bodiesMap     map[uint64]int
	bodiesCount   int
	particleCount int
	syncMap       []int

	freeBodySlots    []int    // stack of reusable body indices
	bodyParticleInfo [][2]int // [particleStart, particleCount] per body slot

	buffers bind_group_provider.BindGroupProvider
	bgps    map[string]bind_group_provider.BindGroupProvider

	pipelineKeys map[string]string

	accumulator      float32
	fixedDt          float32
	maxSubsteps      int
	maxBodies        int
	maxParticles     int
	maxGridCells     int
	particleDiameter float32

	springCoeff, dampingCoeff, shearCoeff float32

	slotsPerCell uint32
	bodyIdxMask  uint32

	gravity [3]float32

	boundaryPlanes  [6][4]float32
	boundaryYRanges [6][4]float32
	boundaryCount   int

	readbackRequested bool
	readbackPending   bool
	stagingBuffer     *wgpu.Buffer
	stagedWriteData   []bind_group_provider.BufferWrite

	lc lifecycle.Lifecycle
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
