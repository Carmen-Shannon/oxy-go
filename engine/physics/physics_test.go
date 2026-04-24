package physics_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/stretchr/testify/suite"
)

type physicsTest struct {
	suite.Suite
	p physics.Physics
}

func TestRunPhysicsTests(t *testing.T) {
	suite.Run(t, new(physicsTest))
}

func (suite *physicsTest) SetupSubTest() {
	suite.p = physics.NewPhysics()
}

func makeProcessReadbackBuf(pos [3]float32, quat [4]float32) []byte {
	buf := make([]byte, 160)
	binary.LittleEndian.PutUint32(buf[0:], math.Float32bits(pos[0]))
	binary.LittleEndian.PutUint32(buf[4:], math.Float32bits(pos[1]))
	binary.LittleEndian.PutUint32(buf[8:], math.Float32bits(pos[2]))
	// offset 12: unused w
	binary.LittleEndian.PutUint32(buf[16:], math.Float32bits(quat[0]))
	binary.LittleEndian.PutUint32(buf[20:], math.Float32bits(quat[1]))
	binary.LittleEndian.PutUint32(buf[24:], math.Float32bits(quat[2]))
	binary.LittleEndian.PutUint32(buf[28:], math.Float32bits(quat[3]))
	return buf
}

func (suite *physicsTest) TestNewPhysics() {
	suite.Run("MaxBodies defaults to 256", func() {
		suite.Equal(256, suite.p.MaxBodies())
	})

	suite.Run("MaxParticles defaults to 2048", func() {
		suite.Equal(2048, suite.p.MaxParticles())
	})

	suite.Run("MaxGridCells defaults to 128*128*128", func() {
		suite.Equal(128*128*128, suite.p.MaxGridCells())
	})

	suite.Run("SlotsPerCell defaults to 16", func() {
		suite.Equal(uint32(16), suite.p.SlotsPerCell())
	})

	suite.Run("BodyIdxMask defaults to 0xFFFFFF", func() {
		suite.Equal(uint32(0xFFFFFF), suite.p.BodyIdxMask())
	})

	suite.Run("Lifecycle defaults to registered state", func() {
		suite.NotNil(suite.p.Lifecycle())
		suite.Equal(lifecycle.LifecycleStateRegistered, suite.p.Lifecycle().State())
	})

	suite.Run("BodiesCount defaults to 0", func() {
		suite.Equal(0, suite.p.BodiesCount())
	})

	suite.Run("ParticleCount defaults to 0", func() {
		suite.Equal(0, suite.p.ParticleCount())
	})

	suite.Run("Buffers is non-nil", func() {
		suite.NotNil(suite.p.Buffers())
	})
}

func (suite *physicsTest) TestLifecycle() {
	suite.Run("registered initially", func() {
		suite.NotNil(suite.p.Lifecycle())
		suite.Equal(lifecycle.LifecycleStateRegistered, suite.p.Lifecycle().State())
	})
}

func (suite *physicsTest) TestBodiesCount() {
	suite.Run("0 initially", func() {
		suite.Equal(0, suite.p.BodiesCount())
	})
}

func (suite *physicsTest) TestParticleCount() {
	suite.Run("0 initially", func() {
		suite.Equal(0, suite.p.ParticleCount())
	})
}

func (suite *physicsTest) TestBuffers() {
	suite.Run("non-nil", func() {
		suite.NotNil(suite.p.Buffers())
	})
}

func (suite *physicsTest) TestPipelineKey() {
	suite.Run("empty string for unknown key", func() {
		suite.Equal("", suite.p.PipelineKey("unknown"))
	})

	suite.Run("returns value after SetPipelineKey", func() {
		suite.p.SetPipelineKey("integrate", "my_pipeline_key")
		suite.Equal("my_pipeline_key", suite.p.PipelineKey("integrate"))
	})
}

func (suite *physicsTest) TestMaxBodies() {
	suite.Run("default 256", func() {
		suite.Equal(256, suite.p.MaxBodies())
	})
}

func (suite *physicsTest) TestMaxParticles() {
	suite.Run("default 2048", func() {
		suite.Equal(2048, suite.p.MaxParticles())
	})
}

func (suite *physicsTest) TestMaxGridCells() {
	suite.Run("default 128*128*128", func() {
		suite.Equal(128*128*128, suite.p.MaxGridCells())
	})
}

func (suite *physicsTest) TestSlotsPerCell() {
	suite.Run("default 16", func() {
		suite.Equal(uint32(16), suite.p.SlotsPerCell())
	})
}

func (suite *physicsTest) TestBodyIdxMask() {
	suite.Run("default 0xFFFFFF", func() {
		suite.Equal(uint32(0xFFFFFF), suite.p.BodyIdxMask())
	})
}

func (suite *physicsTest) TestBgp() {
	suite.Run("non-nil for collision", func() {
		suite.NotNil(suite.p.Bgp("collision"))
	})

	suite.Run("nil for unknown key", func() {
		suite.Nil(suite.p.Bgp("unknown_key_xyz"))
	})
}

func (suite *physicsTest) TestBgps() {
	suite.Run("map contains all 9 expected keys", func() {
		bgps := suite.p.Bgps()
		for _, key := range []string{
			"particle_values", "aabb_reduce", "grid_build_params",
			"grid_clear", "grid_insert", "collision",
			"momenta", "integrate", "sync",
		} {
			suite.Contains(bgps, key)
		}
	})
}

func (suite *physicsTest) TestRequestReadback() {
	suite.Run("ReadbackPending is still false immediately after RequestReadback", func() {
		suite.p.RequestReadback()
		suite.False(suite.p.ReadbackPending())
	})
}

func (suite *physicsTest) TestReadbackPending() {
	suite.Run("false initially", func() {
		suite.False(suite.p.ReadbackPending())
	})
}

func (suite *physicsTest) TestClearReadbackPending() {
	suite.Run("safe to call when already false", func() {
		suite.p.ClearReadbackPending()
		suite.False(suite.p.ReadbackPending())
	})
}

func (suite *physicsTest) TestStagingBuffer() {
	suite.Run("nil initially", func() {
		suite.Nil(suite.p.StagingBuffer())
	})
}

func (suite *physicsTest) TestSetStagingBuffer() {
	suite.Run("nil stays nil after SetStagingBuffer(nil)", func() {
		suite.p.SetStagingBuffer(nil)
		suite.Nil(suite.p.StagingBuffer())
	})
}

func (suite *physicsTest) TestConsumeReadbackRequest() {
	suite.Run("returns false when readback not requested", func() {
		suite.False(suite.p.ConsumeReadbackRequest())
	})

	suite.Run("returns true when readback requested and sets ReadbackPending", func() {
		suite.p.RequestReadback()
		suite.True(suite.p.ConsumeReadbackRequest())
		suite.True(suite.p.ReadbackPending())
	})

	suite.Run("clears the readback request flag so second consume returns false", func() {
		suite.p.RequestReadback()
		suite.p.ConsumeReadbackRequest()
		suite.False(suite.p.ConsumeReadbackRequest())
	})
}

func (suite *physicsTest) TestProcessReadback() {
	suite.Run("short data clamps count to 0 and does not update position", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.p.StagedWriteData()
		// 100 bytes < 160 (one GPUBody), so count clamps to 0 and no body is updated
		suite.p.ProcessReadback(make([]byte, 100))
		suite.Equal([3]float32{}, rb.Position())
	})

	suite.Run("nil body slot does not panic", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.p.RemoveBody(1)
		suite.p.StagedWriteData()
		// bodiesCount=1, bodies[0]=nil: ProcessReadback must skip without panicking
		suite.p.ProcessReadback(make([]byte, 160))
	})

	suite.Run("normal: updates position and quaternion from buffer", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.p.StagedWriteData()

		pos := [3]float32{1.0, 2.0, 3.0}
		quat := [4]float32{0.1, 0.2, 0.3, 0.9}
		suite.p.ProcessReadback(makeProcessReadbackBuf(pos, quat))

		suite.InDelta(float64(pos[0]), float64(rb.Position()[0]), 1e-6)
		suite.InDelta(float64(pos[1]), float64(rb.Position()[1]), 1e-6)
		suite.InDelta(float64(pos[2]), float64(rb.Position()[2]), 1e-6)
		suite.InDelta(float64(quat[0]), float64(rb.Quaternion()[0]), 1e-6)
		suite.InDelta(float64(quat[1]), float64(rb.Quaternion()[1]), 1e-6)
		suite.InDelta(float64(quat[2]), float64(rb.Quaternion()[2]), 1e-6)
		suite.InDelta(float64(quat[3]), float64(rb.Quaternion()[3]), 1e-6)
	})
}

func (suite *physicsTest) TestRegisterBody() {
	suite.Run("first body returns index 0 and bodiesCount becomes 1", func() {
		rb := physics.NewRigidBody()
		idx := suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.Equal(0, idx)
		suite.Equal(1, suite.p.BodiesCount())
	})

	suite.Run("lifecycle remains registered after first registration", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.NotNil(suite.p.Lifecycle())
		suite.Equal(lifecycle.LifecycleStateRegistered, suite.p.Lifecycle().State())
	})

	suite.Run("slot reuse: second body after remove is assigned index 0", func() {
		rb1 := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb1, 0)
		suite.p.RemoveBody(1)
		rb2 := physics.NewRigidBody()
		idx := suite.p.RegisterBody(2, [3]float32{}, [3]float32{}, rb2, 0)
		suite.Equal(0, idx)
	})

	suite.Run("particle diameter set on first registration; second does not change it", func() {
		suite.p = physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithMaxSubsteps(4))
		rb1 := physics.NewRigidBody(physics.WithParticleRadius(0.1))
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb1, 0)
		rb2 := physics.NewRigidBody(physics.WithParticleRadius(0.5))
		suite.p.RegisterBody(2, [3]float32{}, [3]float32{}, rb2, 0)
		_, globalsData := suite.p.PrepareStep(0.05)
		// ParticleDiameter is at offset 4 in GPUPhysicsGlobals
		diameter := math.Float32frombits(binary.LittleEndian.Uint32(globalsData[4:8]))
		// First body set diameter = 0.1*2 = 0.2; second body's 0.5 must not overwrite it
		suite.InDelta(float64(float32(0.2)), float64(diameter), 1e-6)
	})

	suite.Run("Active body has PhysicsStateActive flag set in body write data", func() {
		rb := physics.NewRigidBody(physics.WithActive(true))
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		writes := suite.p.StagedWriteData()
		flags := binary.LittleEndian.Uint32(writes[0].Data[124:128])
		suite.True(flags&uint32(physics.PhysicsStateActive) != 0)
	})

	suite.Run("Static body has PhysicsStateStatic flag set in body write data", func() {
		rb := physics.NewRigidBody(physics.WithStatic(true))
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		writes := suite.p.StagedWriteData()
		flags := binary.LittleEndian.Uint32(writes[0].Data[124:128])
		suite.True(flags&uint32(physics.PhysicsStateStatic) != 0)
	})

	suite.Run("Kinematic body has PhysicsStateKinematic flag set in body write data", func() {
		rb := physics.NewRigidBody(physics.WithKinematic(true))
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		writes := suite.p.StagedWriteData()
		flags := binary.LittleEndian.Uint32(writes[0].Data[124:128])
		suite.True(flags&uint32(physics.PhysicsStateKinematic) != 0)
	})

	suite.Run("static body particle has snW=1.0 at SurfaceNormal.w (offset 92 per particle)", func() {
		part := physics.Particle{LocalPosition: [3]float32{1, 0, 0}}
		rb := physics.NewRigidBody(
			physics.WithStatic(true),
			physics.WithParticles([]physics.Particle{part}),
		)
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		writes := suite.p.StagedWriteData()
		// writes[1] is the particle batch write; each particle is 96 bytes
		// SurfaceNormal starts at offset 80; its w component (snW) is at 80+12=92
		particleData := writes[1].Data
		snW := math.Float32frombits(binary.LittleEndian.Uint32(particleData[92:96]))
		suite.InDelta(float64(1.0), float64(snW), 1e-6)
	})

	suite.Run("non-static body particle has snW=0.0 at SurfaceNormal.w", func() {
		part := physics.Particle{LocalPosition: [3]float32{1, 0, 0}}
		rb := physics.NewRigidBody(
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{part}),
		)
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		writes := suite.p.StagedWriteData()
		particleData := writes[1].Data
		snW := math.Float32frombits(binary.LittleEndian.Uint32(particleData[92:96]))
		suite.InDelta(float64(0.0), float64(snW), 1e-6)
	})

	suite.Run("angular momentum ok=true: invertible inertia tensor produces non-zero L", func() {
		rb := physics.NewRigidBody(
			physics.WithMass(1.0),
			physics.WithAngularVelocity([3]float32{1, 0, 0}),
			physics.WithParticles([]physics.Particle{
				{LocalPosition: [3]float32{1, 0, 0}},
				{LocalPosition: [3]float32{0, 1, 0}},
			}),
		)
		idx := suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.Equal(0, idx)
		writes := suite.p.StagedWriteData()
		// AngularMomentum is at offset 48 in the 160-byte GPUBody
		lx := math.Float32frombits(binary.LittleEndian.Uint32(writes[0].Data[48:52]))
		suite.NotEqual(float32(0), lx)
	})

	suite.Run("angular momentum ok=false: no particles yields zero angular momentum", func() {
		rb := physics.NewRigidBody(
			physics.WithMass(1.0),
			physics.WithAngularVelocity([3]float32{1, 0, 0}),
		)
		idx := suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.Equal(0, idx)
		writes := suite.p.StagedWriteData()
		// AngularMomentum at offset 48: inertia tensor is zero (no particles) → ok=false → L stays zero
		lx := math.Float32frombits(binary.LittleEndian.Uint32(writes[0].Data[48:52]))
		suite.Equal(float32(0), lx)
	})
}

func (suite *physicsTest) TestRemoveBody() {
	suite.Run("unknown objID is a no-op with no staged writes", func() {
		suite.p.RemoveBody(999)
		suite.Empty(suite.p.StagedWriteData())
	})

	suite.Run("known objID stages two writes: zero flags and zero inverseMass", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.p.StagedWriteData()
		suite.p.RemoveBody(1)
		writes := suite.p.StagedWriteData()
		suite.Len(writes, 2)
		// flags write: binding 0, offset 0*160+124 = 124
		suite.Equal(uint64(124), writes[0].Offset)
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(writes[0].Data))
		// inverseMass write: binding 0, offset 0*160+112 = 112
		suite.Equal(uint64(112), writes[1].Offset)
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(writes[1].Data))
	})

	suite.Run("lifecycle remains registered when last body is removed", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.p.RemoveBody(1)
		suite.NotNil(suite.p.Lifecycle())
		suite.Equal(lifecycle.LifecycleStateRegistered, suite.p.Lifecycle().State())
	})

	suite.Run("lifecycle remains registered when other bodies remain after removal", func() {
		rb1 := physics.NewRigidBody()
		rb2 := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb1, 0)
		suite.p.RegisterBody(2, [3]float32{}, [3]float32{}, rb2, 0)
		suite.p.RemoveBody(1)
		suite.NotNil(suite.p.Lifecycle())
		suite.Equal(lifecycle.LifecycleStateRegistered, suite.p.Lifecycle().State())
	})
}

func (suite *physicsTest) TestBodyParticleInfo() {
	suite.Run("negative body index returns (0, 0)", func() {
		start, count := suite.p.BodyParticleInfo(-1)
		suite.Equal(uint32(0), start)
		suite.Equal(uint32(0), count)
	})

	suite.Run("body index >= len returns (0, 0)", func() {
		start, count := suite.p.BodyParticleInfo(999)
		suite.Equal(uint32(0), start)
		suite.Equal(uint32(0), count)
	})

	suite.Run("valid index returns correct particle info", func() {
		parts := []physics.Particle{
			{LocalPosition: [3]float32{1, 0, 0}},
			{LocalPosition: [3]float32{0, 1, 0}},
		}
		rb := physics.NewRigidBody(physics.WithParticles(parts))
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		start, count := suite.p.BodyParticleInfo(0)
		suite.Equal(uint32(0), start)
		suite.Equal(uint32(2), count)
	})
}

func (suite *physicsTest) TestStagedWriteData() {
	suite.Run("non-empty after RegisterBody", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		writes := suite.p.StagedWriteData()
		suite.NotEmpty(writes)
	})

	suite.Run("drained on first call", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		first := suite.p.StagedWriteData()
		suite.Len(first, 3)
	})

	suite.Run("empty on second call after drain", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.p.StagedWriteData()
		second := suite.p.StagedWriteData()
		suite.Empty(second)
	})
}

func (suite *physicsTest) TestBodyIndex() {
	suite.Run("returns false for unknown object ID", func() {
		_, ok := suite.p.BodyIndex(42)
		suite.False(ok)
	})

	suite.Run("returns correct index and true after registration", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(7, [3]float32{}, [3]float32{}, rb, 0)
		idx, ok := suite.p.BodyIndex(7)
		suite.True(ok)
		suite.Equal(0, idx)
	})

	suite.Run("returns false after body is removed", func() {
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(7, [3]float32{}, [3]float32{}, rb, 0)
		suite.p.RemoveBody(7)
		_, ok := suite.p.BodyIndex(7)
		suite.False(ok)
	})
}

func (suite *physicsTest) TestBuilderOptions() {
	suite.Run("WithMaxBodies overrides default", func() {
		p := physics.NewPhysics(physics.WithMaxBodies(512))
		suite.Equal(512, p.MaxBodies())
	})

	suite.Run("WithMaxParticles overrides default", func() {
		p := physics.NewPhysics(physics.WithMaxParticles(1024))
		suite.Equal(1024, p.MaxParticles())
	})

	suite.Run("WithMaxGridCells overrides default", func() {
		p := physics.NewPhysics(physics.WithMaxGridCells(4096))
		suite.Equal(4096, p.MaxGridCells())
	})

	suite.Run("WithMaxSubsteps affects substep count", func() {
		p := physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithMaxSubsteps(2))
		rb := physics.NewRigidBody()
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		substeps, _ := p.PrepareStep(0.05)
		suite.Equal(2, substeps)
	})

	suite.Run("WithSpringCoeff is reflected in PrepareStep globals", func() {
		p := physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithSpringCoeff(2.5))
		rb := physics.NewRigidBody()
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		_, globalsData := p.PrepareStep(0.05)
		// SpringCoeff is at offset 8 in GPUPhysicsGlobals
		v := math.Float32frombits(binary.LittleEndian.Uint32(globalsData[8:12]))
		suite.InDelta(float64(2.5), float64(v), 1e-6)
	})

	suite.Run("WithDampingCoeff is reflected in PrepareStep globals", func() {
		p := physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithDampingCoeff(0.3))
		rb := physics.NewRigidBody()
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		_, globalsData := p.PrepareStep(0.05)
		// DampingCoeff is at offset 12 in GPUPhysicsGlobals
		v := math.Float32frombits(binary.LittleEndian.Uint32(globalsData[12:16]))
		suite.InDelta(float64(0.3), float64(v), 1e-6)
	})

	suite.Run("WithShearCoeff is reflected in PrepareStep globals", func() {
		p := physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithShearCoeff(0.8))
		rb := physics.NewRigidBody()
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		_, globalsData := p.PrepareStep(0.05)
		// ShearCoeff is at offset 16 in GPUPhysicsGlobals
		v := math.Float32frombits(binary.LittleEndian.Uint32(globalsData[16:20]))
		suite.InDelta(float64(0.8), float64(v), 1e-6)
	})

	suite.Run("WithGravity is reflected in PrepareStep globals", func() {
		p := physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithGravity([3]float32{0, -9.81, 0}))
		rb := physics.NewRigidBody()
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		_, globalsData := p.PrepareStep(0.05)
		// GravityY is at offset 40 in GPUPhysicsGlobals
		gy := math.Float32frombits(binary.LittleEndian.Uint32(globalsData[40:44]))
		suite.InDelta(float64(-9.81), float64(gy), 1e-4)
	})

	suite.Run("WithBoundaryPlanes sets boundary count in PrepareStep globals", func() {
		planes := [][6]float32{
			{0, 1, 0, 0, -100, 100},
			{0, -1, 0, 10, -100, 100},
		}
		p := physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithBoundaryPlanes(planes))
		rb := physics.NewRigidBody()
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		_, globalsData := p.PrepareStep(0.05)
		// BoundaryCount is at offset 32 in GPUPhysicsGlobals
		bc := binary.LittleEndian.Uint32(globalsData[32:36])
		suite.Equal(uint32(2), bc)
	})
}

func (suite *physicsTest) TestPrepareStep() {
	suite.Run("with no bodies registered substeps are still computed", func() {
		suite.p = physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithMaxSubsteps(4))
		substeps, globalsData := suite.p.PrepareStep(0.05)
		suite.Equal(4, substeps)
		suite.NotNil(globalsData)
	})

	suite.Run("substeps=0 when dt is too small for fixedDt", func() {
		suite.p = physics.NewPhysics(physics.WithFixedDt(1.0), physics.WithMaxSubsteps(4))
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		substeps, globalsData := suite.p.PrepareStep(0.001)
		suite.Equal(0, substeps)
		suite.Nil(globalsData)
	})

	suite.Run("substeps=4 and globalsData non-nil when dt covers 5 steps but capped at maxSubsteps", func() {
		suite.p = physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithMaxSubsteps(4))
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		substeps, globalsData := suite.p.PrepareStep(0.05)
		suite.Equal(4, substeps)
		suite.NotNil(globalsData)
	})

	suite.Run("force drain: force write staged at ExternalForce offset 128", func() {
		suite.p = physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithMaxSubsteps(4))
		rb := physics.NewRigidBody(physics.WithActive(true))
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.p.StagedWriteData()
		rb.ApplyForce([3]float32{1, 0, 0})
		suite.p.PrepareStep(0.05)
		writes := suite.p.StagedWriteData()
		found := false
		for _, w := range writes {
			if w.Binding == 0 && w.Offset == 128 {
				found = true
				break
			}
		}
		suite.True(found)
	})

	suite.Run("torque drain: torque write staged at ExternalTorque offset 144", func() {
		suite.p = physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithMaxSubsteps(4))
		rb := physics.NewRigidBody(physics.WithActive(true))
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		suite.p.StagedWriteData()
		rb.ApplyTorque([3]float32{0, 1, 0})
		suite.p.PrepareStep(0.05)
		writes := suite.p.StagedWriteData()
		found := false
		for _, w := range writes {
			if w.Binding == 0 && w.Offset == 144 {
				found = true
				break
			}
		}
		suite.True(found)
	})

	suite.Run("accumulator clamped when remaining time exceeds fixedDt after maxSubsteps", func() {
		// dt=0.06, fixedDt=0.01, maxSubsteps=4: 4 iters leave accumulator=0.02 > 0.01 → clamp
		suite.p = physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithMaxSubsteps(4))
		rb := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		substeps, globalsData := suite.p.PrepareStep(0.06)
		suite.Equal(4, substeps)
		suite.NotNil(globalsData)
	})

	suite.Run("nil body slot in loop is skipped without panic", func() {
		suite.p = physics.NewPhysics(physics.WithFixedDt(0.01), physics.WithMaxSubsteps(4))
		rb1 := physics.NewRigidBody()
		rb2 := physics.NewRigidBody()
		suite.p.RegisterBody(1, [3]float32{}, [3]float32{}, rb1, 0)
		suite.p.RegisterBody(2, [3]float32{}, [3]float32{}, rb2, 0)
		suite.p.RemoveBody(1) // bodies[0]=nil, bodies[1]=rb2
		suite.p.StagedWriteData()
		substeps, _ := suite.p.PrepareStep(0.05)
		suite.Equal(4, substeps)
	})
}
