package physics_test

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/physics"
)

func TestRunGPUTypesTests(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

type gpuTypesTest struct {
	suite.Suite
}

func (suite *gpuTypesTest) TestGPUBody() {
	suite.Run("Size should return 160 bytes", func() {
		g := &physics.GPUBody{}
		suite.Equal(160, g.Size())
		suite.Equal(160, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 160-byte buffer with correct encoding", func() {
		g := &physics.GPUBody{
			Position:        [4]float32{1.0, 2.0, 3.0, 0.0},
			Quaternion:      [4]float32{0.0, 0.0, 0.0, 1.0},
			LinearMomentum:  [4]float32{0.1, 0.2, 0.3, 0.0},
			AngularMomentum: [4]float32{0.4, 0.5, 0.6, 0.0},
			InvInertiaTBody: [12]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			InverseMass:     0.5,
			ParticleStart:   10,
			ParticleCount:   20,
			Flags:           1,
			ExternalForce:   [4]float32{0.0, -9.8, 0.0, 0.0},
			ExternalTorque:  [4]float32{0.1, 0.2, 0.3, 0.4},
		}
		buf := g.Marshal()
		suite.Equal(160, len(buf))
		// offset 0: Position[0]
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		// offset 16: Quaternion[0]
		suite.Equal(math.Float32bits(0.0), binary.LittleEndian.Uint32(buf[16:20]))
		// offset 64: InvInertiaTBody[0] (col0.x)
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[64:68]))
		// offset 76: zero padding after col0.z (64 + 3*4 = 76)
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[76:80]))
		// offset 112: InverseMass
		suite.Equal(math.Float32bits(0.5), binary.LittleEndian.Uint32(buf[112:116]))
		// offset 124: Flags
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[124:128]))
		// offset 128: ExternalForce[0]
		suite.Equal(math.Float32bits(0.0), binary.LittleEndian.Uint32(buf[128:132]))
		// offset 156: ExternalTorque[3]
		suite.Equal(math.Float32bits(0.4), binary.LittleEndian.Uint32(buf[156:160]))
	})
}

func (suite *gpuTypesTest) TestGPUParticle() {
	suite.Run("Size should return 96 bytes", func() {
		g := &physics.GPUParticle{}
		suite.Equal(96, g.Size())
		suite.Equal(96, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 96-byte buffer with correct encoding", func() {
		g := &physics.GPUParticle{
			WorldPosition: [4]float32{1.0, 2.0, 3.0, 0.0},
			Velocity:      [4]float32{0.1, 0.2, 0.3, 0.0},
			RelPosition:   [4]float32{0.4, 0.5, 0.6, 0.0},
			Force:         [4]float32{0.0, -9.8, 0.0, 0.0},
			LocalPosition: [4]float32{1.1, 2.2, 3.3, math.Float32frombits(7)},
			SurfaceNormal: [4]float32{0.0, 1.0, 0.0, 1.0},
		}
		buf := g.Marshal()
		suite.Equal(96, len(buf))
		// offset 0: WorldPosition[0]
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		// offset 16: Velocity[0]
		suite.Equal(math.Float32bits(0.1), binary.LittleEndian.Uint32(buf[16:20]))
		// offset 48: Force[0]
		suite.Equal(math.Float32bits(0.0), binary.LittleEndian.Uint32(buf[48:52]))
		// offset 64: LocalPosition[0]
		suite.Equal(math.Float32bits(1.1), binary.LittleEndian.Uint32(buf[64:68]))
		// offset 80: SurfaceNormal[0]
		suite.Equal(math.Float32bits(0.0), binary.LittleEndian.Uint32(buf[80:84]))
		// offset 92: SurfaceNormal[3]
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[92:96]))
	})
}

func (suite *gpuTypesTest) TestGPUGridCell() {
	suite.Run("Size should return 64 bytes", func() {
		g := &physics.GPUGridCell{}
		suite.Equal(64, g.Size())
		suite.Equal(64, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 64-byte buffer with correct encoding", func() {
		g := &physics.GPUGridCell{
			Indices0: [4]uint32{1, 2, 3, 4},
			Indices1: [4]uint32{5, 6, 7, 8},
			Indices2: [4]uint32{9, 10, 11, 12},
			Indices3: [4]uint32{13, 14, 15, 16},
		}
		buf := g.Marshal()
		suite.Equal(64, len(buf))
		// offset 0: Indices0[0] = 1
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[0:4]))
		// offset 16: Indices1[0] = 5
		suite.Equal(uint32(5), binary.LittleEndian.Uint32(buf[16:20]))
		// offset 32: Indices2[0] = 9
		suite.Equal(uint32(9), binary.LittleEndian.Uint32(buf[32:36]))
		// offset 48: Indices3[0] = 13
		suite.Equal(uint32(13), binary.LittleEndian.Uint32(buf[48:52]))
		// offset 60: Indices3[3] = 16
		suite.Equal(uint32(16), binary.LittleEndian.Uint32(buf[60:64]))
	})
}

func (suite *gpuTypesTest) TestGPUPhysicsGlobals() {
	suite.Run("Size should return 240 bytes", func() {
		g := &physics.GPUPhysicsGlobals{}
		suite.Equal(240, g.Size())
		suite.Equal(240, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 240-byte buffer with correct encoding", func() {
		g := &physics.GPUPhysicsGlobals{
			DeltaTime:      0.016,
			BodyCount:      10,
			BoundaryCount:  2,
			GravityY:       -9.8,
			BoundaryPlanes: [6][4]float32{{1, 0, 0, 5}},
		}
		buf := g.Marshal()
		suite.Equal(240, len(buf))
		// offset 0: DeltaTime
		suite.InDelta(0.016, float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[0:4]))), 1e-6)
		// offset 20: BodyCount = 10
		suite.Equal(uint32(10), binary.LittleEndian.Uint32(buf[20:24]))
		// offset 32: BoundaryCount = 2
		suite.Equal(uint32(2), binary.LittleEndian.Uint32(buf[32:36]))
		// offset 40: GravityY = -9.8
		suite.InDelta(-9.8, float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[40:44]))), 1e-5)
		// offset 48: BoundaryPlanes[0][0] = 1.0
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[48:52]))
		// offset 144: BoundaryYRanges[0][0] = 0.0 (zero value)
		suite.Equal(math.Float32bits(0.0), binary.LittleEndian.Uint32(buf[144:148]))
	})
}

func (suite *gpuTypesTest) TestGPUGridParams() {
	suite.Run("Size should return 32 bytes", func() {
		g := &physics.GPUGridParams{}
		suite.Equal(32, g.Size())
		suite.Equal(32, int(unsafe.Sizeof(*g)))
	})

	suite.Run("Marshal should return a 32-byte buffer with correct encoding", func() {
		g := &physics.GPUGridParams{
			GridOrigin: [4]float32{1.0, 2.0, 3.0, 0.0},
			GridDims:   [4]uint32{4, 5, 6, 120},
		}
		buf := g.Marshal()
		suite.Equal(32, len(buf))
		// offset 0: GridOrigin[0] = 1.0
		suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(buf[0:4]))
		// offset 12: GridOrigin[3] = 0.0
		suite.Equal(math.Float32bits(0.0), binary.LittleEndian.Uint32(buf[12:16]))
		// offset 16: GridDims[0] = 4
		suite.Equal(uint32(4), binary.LittleEndian.Uint32(buf[16:20]))
		// offset 28: GridDims[3] = 120
		suite.Equal(uint32(120), binary.LittleEndian.Uint32(buf[28:32]))
	})
}
