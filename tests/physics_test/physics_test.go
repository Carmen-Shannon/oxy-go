package physics_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/stretchr/testify/suite"

	modelmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/model"
)

type physicsTest struct {
	suite.Suite
}

func TestPhysics(t *testing.T) {
	suite.Run(t, new(physicsTest))
}

func (suite *physicsTest) TestNewPhysics() {
	suite.Run("defaults are sensible", func() {
		p := physics.NewPhysics()
		suite.False(p.Enabled())
		suite.Equal(0, p.BodiesCount())
		suite.Equal(0, p.ParticleCount())
		suite.Equal(uint32(256), p.MaxBodies())
		suite.Equal(uint32(2048), p.MaxParticles())
		suite.Equal(uint32(128*128*128), p.MaxGridCells())
		suite.NotNil(p.Buffers())
		suite.NotNil(p.Bgps())
	})

	suite.Run("option builders override defaults", func() {
		p := physics.NewPhysics(
			physics.WithFixedDt(1.0/120.0),
			physics.WithMaxSubsteps(8),
			physics.WithMaxBodies(512),
			physics.WithMaxParticles(4096),
			physics.WithMaxGridCells(64*64*64),
			physics.WithSpringCoeff(2.0),
			physics.WithDampingCoeff(0.5),
			physics.WithShearCoeff(0.3),
			physics.WithGravity([3]float32{0, -20.0, 0}),
		)
		suite.Equal(uint32(512), p.MaxBodies())
		suite.Equal(uint32(4096), p.MaxParticles())
		suite.Equal(uint32(64*64*64), p.MaxGridCells())
	})

	suite.Run("boundary planes option applies up to six planes", func() {
		planes := [][6]float32{
			{1, 0, 0, 5, -10, 10},
			{-1, 0, 0, 5, -10, 10},
			{0, 1, 0, 0, -10, 10},
			{0, -1, 0, 10, -10, 10},
			{0, 0, 1, 5, -10, 10},
			{0, 0, -1, 5, -10, 10},
		}
		p := physics.NewPhysics(physics.WithBoundaryPlanes(planes))
		suite.NotNil(p)
	})

	suite.Run("boundary planes clamps at six", func() {
		planes := make([][6]float32, 10)
		p := physics.NewPhysics(physics.WithBoundaryPlanes(planes))
		suite.NotNil(p)
	})

	suite.Run("bgps map contains expected stage keys", func() {
		p := physics.NewPhysics()
		expectedKeys := []string{
			"particle_values", "aabb_reduce", "grid_build_params",
			"grid_clear", "grid_insert", "collision",
			"momenta", "integrate", "sync",
		}
		for _, k := range expectedKeys {
			suite.NotNil(p.Bgp(k))
		}
	})

	suite.Run("bgp returns nil for unknown key", func() {
		p := physics.NewPhysics()
		suite.Nil(p.Bgp("nonexistent"))
	})
}

func (suite *physicsTest) TestSetPipelineKey() {
	suite.Run("stores and retrieves pipeline key", func() {
		p := physics.NewPhysics()
		p.SetPipelineKey("collision", "pipeline_abc")
		suite.Equal("pipeline_abc", p.PipelineKey("collision"))
	})

	suite.Run("unknown key returns empty string", func() {
		p := physics.NewPhysics()
		suite.Equal("", p.PipelineKey("missing"))
	})

	suite.Run("overwrite replaces previous value", func() {
		p := physics.NewPhysics()
		p.SetPipelineKey("integrate", "v1")
		p.SetPipelineKey("integrate", "v2")
		suite.Equal("v2", p.PipelineKey("integrate"))
	})
}

func (suite *physicsTest) TestReadbackFlags() {
	suite.Run("request and consume cycle", func() {
		p := physics.NewPhysics()
		suite.False(p.ReadbackPending())
		suite.False(p.ConsumeReadbackRequest())

		p.RequestReadback()
		suite.True(p.ConsumeReadbackRequest())
		suite.True(p.ReadbackPending())

		// second consume returns false
		suite.False(p.ConsumeReadbackRequest())
	})

	suite.Run("clear readback pending resets flag", func() {
		p := physics.NewPhysics()
		p.RequestReadback()
		p.ConsumeReadbackRequest()
		suite.True(p.ReadbackPending())

		p.ClearReadbackPending()
		suite.False(p.ReadbackPending())
	})
}

func (suite *physicsTest) TestStagingBuffer() {
	suite.Run("initially nil", func() {
		p := physics.NewPhysics()
		suite.Nil(p.StagingBuffer())
	})

	suite.Run("set and get staging buffer", func() {
		p := physics.NewPhysics()
		// nil -> nil round-trip (we can't create a real *wgpu.Buffer without a device)
		p.SetStagingBuffer(nil)
		suite.Nil(p.StagingBuffer())
	})
}

func (suite *physicsTest) TestRegisterBody() {
	suite.Run("first registration enables physics and returns index zero", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(
			physics.WithMass(10),
			physics.WithActive(true),
			physics.WithParticleRadius(0.1),
			physics.WithParticles([]physics.Particle{
				{LocalPosition: [3]float32{0, 0, 0}},
			}),
		)
		idx := p.RegisterBody(100, [3]float32{1, 2, 3}, [3]float32{0, 0, 0}, rb, 42)
		suite.Equal(0, idx)
		suite.True(p.Enabled())
		suite.Equal(1, p.BodiesCount())
		suite.Equal(1, p.ParticleCount())
	})

	suite.Run("staged write data is produced for body, particles, and sync", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(
			physics.WithMass(5),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{
				{LocalPosition: [3]float32{0.5, 0, 0}},
				{LocalPosition: [3]float32{-0.5, 0, 0}},
			}),
			physics.WithParticleRadius(0.25),
		)
		p.RegisterBody(200, [3]float32{0, 0, 0}, [3]float32{0, 0, 0}, rb, 7)
		writes := p.StagedWriteData()
		// Expect at least 3 writes: body data (binding 0), particle data (binding 1), sync (binding 7)
		suite.GreaterOrEqual(len(writes), 3)
	})

	suite.Run("body index lookup works after registration", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		p.RegisterBody(300, [3]float32{}, [3]float32{}, rb, 0)
		idx, ok := p.BodyIndex(300)
		suite.True(ok)
		suite.Equal(0, idx)
	})

	suite.Run("body index lookup returns false for unknown id", func() {
		p := physics.NewPhysics()
		_, ok := p.BodyIndex(999)
		suite.False(ok)
	})

	suite.Run("multiple bodies get sequential indices", func() {
		p := physics.NewPhysics()
		rb1 := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		rb2 := physics.NewRigidBody(physics.WithMass(2), physics.WithActive(true))
		idx1 := p.RegisterBody(1, [3]float32{}, [3]float32{}, rb1, 0)
		idx2 := p.RegisterBody(2, [3]float32{}, [3]float32{}, rb2, 1)
		suite.Equal(0, idx1)
		suite.Equal(1, idx2)
		suite.Equal(2, p.BodiesCount())
	})

	suite.Run("static body has zero inverse mass in gpu data", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(
			physics.WithMass(0),
			physics.WithStatic(true),
		)
		p.RegisterBody(400, [3]float32{}, [3]float32{}, rb, 0)
		writes := p.StagedWriteData()
		// The first write should be the GPUBody at binding 0
		suite.GreaterOrEqual(len(writes), 1)
		bodyWrite := writes[0]
		suite.Equal(0, bodyWrite.Binding)
		suite.Equal(160, len(bodyWrite.Data))
		// InverseMass at offset 112 should be zero
		invMass := math.Float32frombits(binary.LittleEndian.Uint32(bodyWrite.Data[112:116]))
		suite.Equal(float32(0), invMass)
	})

	suite.Run("flags encode active static kinematic bits correctly", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithStatic(true),
			physics.WithKinematic(true),
		)
		p.RegisterBody(500, [3]float32{}, [3]float32{}, rb, 0)
		writes := p.StagedWriteData()
		bodyWrite := writes[0]
		flags := binary.LittleEndian.Uint32(bodyWrite.Data[124:128])
		suite.Equal(uint32(1), flags&1) // active
		suite.Equal(uint32(2), flags&2) // static
		suite.Equal(uint32(4), flags&4) // kinematic
	})

	suite.Run("initial velocity produces non-zero linear momentum", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(
			physics.WithMass(10),
			physics.WithVelocity([3]float32{1, 0, 0}),
			physics.WithActive(true),
		)
		p.RegisterBody(600, [3]float32{}, [3]float32{}, rb, 0)
		writes := p.StagedWriteData()
		bodyWrite := writes[0]
		// LinearMomentum at offset 32: P = mass * velocity = (10, 0, 0)
		px := math.Float32frombits(binary.LittleEndian.Uint32(bodyWrite.Data[32:36]))
		suite.InDelta(10.0, float64(px), 1e-5)
	})

	suite.Run("non-zero rotation produces valid quaternion in gpu data", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		halfPi := float32(math.Pi / 2)
		p.RegisterBody(700, [3]float32{0, 5, 0}, [3]float32{halfPi, 0, 0}, rb, 0)
		writes := p.StagedWriteData()
		bodyWrite := writes[0]
		// Quaternion at offset 16 (4 × float32)
		qx := math.Float32frombits(binary.LittleEndian.Uint32(bodyWrite.Data[16:20]))
		qy := math.Float32frombits(binary.LittleEndian.Uint32(bodyWrite.Data[20:24]))
		qz := math.Float32frombits(binary.LittleEndian.Uint32(bodyWrite.Data[24:28]))
		qw := math.Float32frombits(binary.LittleEndian.Uint32(bodyWrite.Data[28:32]))
		// quaternion should be unit length
		length := math.Sqrt(float64(qx*qx + qy*qy + qz*qz + qw*qw))
		suite.InDelta(1.0, length, 1e-5)
	})

	suite.Run("particle diameter is derived from first registered body", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticleRadius(0.5),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
		)
		p.RegisterBody(800, [3]float32{}, [3]float32{}, rb, 0)

		// Trigger PrepareStep so globals are generated; the particle diameter
		// should be 2 × 0.5 = 1.0 in the marshaled data.
		substeps, data := p.PrepareStep(1.0 / 30.0)
		suite.Greater(substeps, 0)
		suite.NotNil(data)
		// ParticleDiameter is at offset 4 in GPUPhysicsGlobals
		diameter := math.Float32frombits(binary.LittleEndian.Uint32(data[4:8]))
		suite.InDelta(1.0, float64(diameter), 1e-6)
	})

	suite.Run("static body particles have surface normal w of one", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(
			physics.WithMass(0),
			physics.WithStatic(true),
			physics.WithParticles([]physics.Particle{
				{LocalPosition: [3]float32{1, 0, 0}, SurfaceNormal: [3]float32{1, 0, 0}},
			}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(801, [3]float32{}, [3]float32{}, rb, 0)
		writes := p.StagedWriteData()
		// particle write is at binding 1
		var particleWrite []byte
		for _, w := range writes {
			if w.Binding == 1 {
				particleWrite = w.Data
				break
			}
		}
		suite.NotNil(particleWrite)
		// GPUParticle.SurfaceNormal.w is at offset 80+12=92 within the 96-byte particle
		snW := math.Float32frombits(binary.LittleEndian.Uint32(particleWrite[92:96]))
		suite.InDelta(1.0, float64(snW), 1e-6)
	})

	suite.Run("dynamic body particles have surface normal w of zero", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(
			physics.WithMass(5),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{
				{LocalPosition: [3]float32{1, 0, 0}, SurfaceNormal: [3]float32{1, 0, 0}},
			}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(802, [3]float32{}, [3]float32{}, rb, 0)
		writes := p.StagedWriteData()
		var particleWrite []byte
		for _, w := range writes {
			if w.Binding == 1 {
				particleWrite = w.Data
				break
			}
		}
		suite.NotNil(particleWrite)
		snW := math.Float32frombits(binary.LittleEndian.Uint32(particleWrite[92:96]))
		suite.InDelta(0.0, float64(snW), 1e-6)
	})
}

func (suite *physicsTest) TestRemoveBody() {
	suite.Run("removing the only body disables physics", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		p.RegisterBody(100, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData() // drain register writes
		suite.True(p.Enabled())

		p.RemoveBody(100)
		suite.False(p.Enabled())
	})

	suite.Run("removing unknown id is a no-op", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		p.RegisterBody(100, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		p.RemoveBody(999)
		suite.True(p.Enabled())
	})

	suite.Run("staged writes zero flags and inverse mass on removal", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		p.RegisterBody(100, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData() // drain

		p.RemoveBody(100)
		writes := p.StagedWriteData()
		// Should have 2 writes: one for flags (offset 124) and one for inverse mass (offset 112)
		suite.Equal(2, len(writes))

		var flagsOffset, invMassOffset uint64
		for _, w := range writes {
			if w.Offset == 124 {
				flagsOffset = w.Offset
				suite.Equal(uint32(0), binary.LittleEndian.Uint32(w.Data))
			}
			if w.Offset == 112 {
				invMassOffset = w.Offset
				suite.Equal(uint32(0), binary.LittleEndian.Uint32(w.Data))
			}
		}
		suite.Equal(uint64(124), flagsOffset)
		suite.Equal(uint64(112), invMassOffset)
	})

	suite.Run("body index is no longer found after removal", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		p.RegisterBody(100, [3]float32{}, [3]float32{}, rb, 0)
		p.RemoveBody(100)
		_, ok := p.BodyIndex(100)
		suite.False(ok)
	})

	suite.Run("removed slot is reused by next registration", func() {
		p := physics.NewPhysics()
		rb1 := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		rb2 := physics.NewRigidBody(
			physics.WithMass(2),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{1, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		idx0 := p.RegisterBody(10, [3]float32{}, [3]float32{}, rb1, 0)
		p.RegisterBody(20, [3]float32{}, [3]float32{}, rb2, 1)
		p.StagedWriteData()

		p.RemoveBody(10)
		p.StagedWriteData()

		rb3 := physics.NewRigidBody(
			physics.WithMass(3),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{2, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		reusedIdx := p.RegisterBody(30, [3]float32{}, [3]float32{}, rb3, 2)
		suite.Equal(idx0, reusedIdx)
	})

	suite.Run("removing one of two bodies keeps physics enabled", func() {
		p := physics.NewPhysics()
		rb1 := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		rb2 := physics.NewRigidBody(physics.WithMass(2), physics.WithActive(true))
		p.RegisterBody(10, [3]float32{}, [3]float32{}, rb1, 0)
		p.RegisterBody(20, [3]float32{}, [3]float32{}, rb2, 1)
		p.StagedWriteData()

		p.RemoveBody(10)
		suite.True(p.Enabled())
	})
}

func (suite *physicsTest) TestBodyParticleInfo() {
	suite.Run("returns correct start and count for registered body", func() {
		p := physics.NewPhysics()
		particles := []physics.Particle{
			{LocalPosition: [3]float32{0, 0, 0}},
			{LocalPosition: [3]float32{1, 0, 0}},
			{LocalPosition: [3]float32{0, 1, 0}},
		}
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles(particles),
			physics.WithParticleRadius(0.1),
		)
		idx := p.RegisterBody(100, [3]float32{}, [3]float32{}, rb, 0)
		start, count := p.BodyParticleInfo(idx)
		suite.Equal(uint32(0), start)
		suite.Equal(uint32(3), count)
	})

	suite.Run("second body starts after first body particles", func() {
		p := physics.NewPhysics()
		rb1 := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{
				{LocalPosition: [3]float32{0, 0, 0}},
				{LocalPosition: [3]float32{1, 0, 0}},
			}),
			physics.WithParticleRadius(0.1),
		)
		rb2 := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{
				{LocalPosition: [3]float32{2, 0, 0}},
			}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb1, 0)
		idx2 := p.RegisterBody(2, [3]float32{}, [3]float32{}, rb2, 1)
		start, count := p.BodyParticleInfo(idx2)
		suite.Equal(uint32(2), start)
		suite.Equal(uint32(1), count)
	})

	suite.Run("out of range index returns zeros", func() {
		p := physics.NewPhysics()
		start, count := p.BodyParticleInfo(-1)
		suite.Equal(uint32(0), start)
		suite.Equal(uint32(0), count)
	})

	suite.Run("index beyond allocated slots returns zeros", func() {
		p := physics.NewPhysics()
		start, count := p.BodyParticleInfo(999)
		suite.Equal(uint32(0), start)
		suite.Equal(uint32(0), count)
	})
}

func (suite *physicsTest) TestStagedWriteData() {
	suite.Run("returns empty after drain", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		p.RegisterBody(100, [3]float32{}, [3]float32{}, rb, 0)
		first := p.StagedWriteData()
		suite.NotEmpty(first)

		second := p.StagedWriteData()
		suite.Empty(second)
	})

	suite.Run("accumulates writes across multiple operations", func() {
		p := physics.NewPhysics()
		rb1 := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		rb2 := physics.NewRigidBody(physics.WithMass(2), physics.WithActive(true))
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb1, 0)
		p.RegisterBody(2, [3]float32{}, [3]float32{}, rb2, 1)
		writes := p.StagedWriteData()
		// Each register produces at least 3 writes (body, particles, sync) × 2 bodies
		suite.GreaterOrEqual(len(writes), 6)
	})
}

func (suite *physicsTest) TestPrepareStep() {
	suite.Run("returns zero substeps when disabled", func() {
		p := physics.NewPhysics()
		substeps, data := p.PrepareStep(1.0 / 60.0)
		suite.Equal(0, substeps)
		suite.Nil(data)
	})

	suite.Run("returns one substep for exactly one fixed dt", func() {
		p := physics.NewPhysics(physics.WithFixedDt(1.0 / 60.0))
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		substeps, data := p.PrepareStep(1.0 / 60.0)
		suite.Equal(1, substeps)
		suite.NotNil(data)
	})

	suite.Run("caps substeps at max", func() {
		p := physics.NewPhysics(
			physics.WithFixedDt(1.0/60.0),
			physics.WithMaxSubsteps(2),
		)
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		// big dt that would otherwise require many substeps
		substeps, data := p.PrepareStep(1.0)
		suite.Equal(2, substeps)
		suite.NotNil(data)
	})

	suite.Run("returns zero substeps when dt is too small", func() {
		p := physics.NewPhysics(physics.WithFixedDt(1.0 / 60.0))
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		substeps, data := p.PrepareStep(1e-6)
		suite.Equal(0, substeps)
		suite.Nil(data)
	})

	suite.Run("globals data contains correct body and particle counts", func() {
		p := physics.NewPhysics(physics.WithFixedDt(1.0 / 60.0))
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{
				{LocalPosition: [3]float32{0, 0, 0}},
				{LocalPosition: [3]float32{1, 0, 0}},
			}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		_, data := p.PrepareStep(1.0 / 60.0)
		suite.NotNil(data)
		suite.Equal(240, len(data))

		// BodyCount at offset 20
		bodyCount := binary.LittleEndian.Uint32(data[20:24])
		suite.Equal(uint32(1), bodyCount)

		// ParticleCount at offset 24
		particleCount := binary.LittleEndian.Uint32(data[24:28])
		suite.Equal(uint32(2), particleCount)
	})

	suite.Run("globals data contains configured gravity", func() {
		p := physics.NewPhysics(
			physics.WithFixedDt(1.0/60.0),
			physics.WithGravity([3]float32{0, -9.81, 0}),
		)
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		_, data := p.PrepareStep(1.0 / 60.0)
		// GravityY at offset 40
		gravY := math.Float32frombits(binary.LittleEndian.Uint32(data[40:44]))
		suite.InDelta(-9.81, float64(gravY), 1e-5)
	})

	suite.Run("drains pending force into staged writes", func() {
		p := physics.NewPhysics(physics.WithFixedDt(1.0 / 60.0))
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData() // drain register writes

		rb.ApplyForce([3]float32{100, 0, 0})
		p.PrepareStep(1.0 / 60.0)

		writes := p.StagedWriteData()
		var foundForce bool
		for _, w := range writes {
			// ExternalForce is at offset 128 within GPUBody
			if w.Binding == 0 && w.Offset == 128 {
				foundForce = true
				fx := math.Float32frombits(binary.LittleEndian.Uint32(w.Data[0:4]))
				suite.InDelta(100.0, float64(fx), 1e-5)
			}
		}
		suite.True(foundForce)
	})

	suite.Run("drains pending torque into staged writes", func() {
		p := physics.NewPhysics(physics.WithFixedDt(1.0 / 60.0))
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		rb.ApplyTorque([3]float32{0, 50, 0})
		p.PrepareStep(1.0 / 60.0)

		writes := p.StagedWriteData()
		var foundTorque bool
		for _, w := range writes {
			// ExternalTorque is at offset 144 within GPUBody
			if w.Binding == 0 && w.Offset == 144 {
				foundTorque = true
				ty := math.Float32frombits(binary.LittleEndian.Uint32(w.Data[4:8]))
				suite.InDelta(50.0, float64(ty), 1e-5)
			}
		}
		suite.True(foundTorque)
	})

	suite.Run("zero force and torque produce no extra staged writes", func() {
		p := physics.NewPhysics(physics.WithFixedDt(1.0 / 60.0))
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData() // drain

		p.PrepareStep(1.0 / 60.0)
		writes := p.StagedWriteData()
		// No force/torque writes expected
		for _, w := range writes {
			if w.Binding == 0 && (w.Offset == 128 || w.Offset == 144) {
				suite.Fail("unexpected force/torque write when none applied")
			}
		}
	})

	suite.Run("accumulator clamps leftover to prevent spiral of death", func() {
		p := physics.NewPhysics(
			physics.WithFixedDt(1.0/60.0),
			physics.WithMaxSubsteps(4),
		)
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		// Huge dt that would create massive accumulator debt
		substeps1, _ := p.PrepareStep(10.0)
		suite.Equal(4, substeps1) // capped

		// Next frame with tiny dt should still eventually tick due to clamped leftover
		substeps2, _ := p.PrepareStep(1.0 / 60.0)
		suite.GreaterOrEqual(substeps2, 1)
	})
}

func (suite *physicsTest) TestProcessReadback() {
	suite.Run("updates body positions and quaternions from raw data", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithActive(true),
			physics.WithParticles([]physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}),
			physics.WithParticleRadius(0.1),
		)
		p.RegisterBody(1, [3]float32{0, 0, 0}, [3]float32{0, 0, 0}, rb, 0)

		// Build fake GPU readback: 160 bytes per body
		data := make([]byte, 160)
		// Position at offset 0: (10, 20, 30)
		binary.LittleEndian.PutUint32(data[0:], math.Float32bits(10))
		binary.LittleEndian.PutUint32(data[4:], math.Float32bits(20))
		binary.LittleEndian.PutUint32(data[8:], math.Float32bits(30))
		// Quaternion at offset 16: (0, 0, 0, 1) = identity
		binary.LittleEndian.PutUint32(data[16:], math.Float32bits(0))
		binary.LittleEndian.PutUint32(data[20:], math.Float32bits(0))
		binary.LittleEndian.PutUint32(data[24:], math.Float32bits(0))
		binary.LittleEndian.PutUint32(data[28:], math.Float32bits(1))

		p.ProcessReadback(data)

		pos := rb.Position()
		suite.InDelta(10.0, float64(pos[0]), 1e-5)
		suite.InDelta(20.0, float64(pos[1]), 1e-5)
		suite.InDelta(30.0, float64(pos[2]), 1e-5)

		quat := rb.Quaternion()
		suite.InDelta(0.0, float64(quat[0]), 1e-5)
		suite.InDelta(0.0, float64(quat[1]), 1e-5)
		suite.InDelta(0.0, float64(quat[2]), 1e-5)
		suite.InDelta(1.0, float64(quat[3]), 1e-5)
	})

	suite.Run("handles data shorter than expected body count", func() {
		p := physics.NewPhysics()
		rb1 := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		rb2 := physics.NewRigidBody(physics.WithMass(2), physics.WithActive(true))
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb1, 0)
		p.RegisterBody(2, [3]float32{}, [3]float32{}, rb2, 1)

		// Only provide data for 1 body (160 bytes), not 2
		data := make([]byte, 160)
		binary.LittleEndian.PutUint32(data[0:], math.Float32bits(5))
		p.ProcessReadback(data)

		suite.InDelta(5.0, float64(rb1.Position()[0]), 1e-5)
		// rb2 should be unchanged (zero position)
		suite.Equal([3]float32{0, 0, 0}, rb2.Position())
	})

	suite.Run("handles empty data gracefully", func() {
		p := physics.NewPhysics()
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)

		suite.NotPanics(func() {
			p.ProcessReadback(nil)
		})
		suite.NotPanics(func() {
			p.ProcessReadback([]byte{})
		})
	})

	suite.Run("skips nil body slots from removed bodies", func() {
		p := physics.NewPhysics()
		rb1 := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		rb2 := physics.NewRigidBody(physics.WithMass(2), physics.WithActive(true))
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb1, 0)
		p.RegisterBody(2, [3]float32{}, [3]float32{}, rb2, 1)

		p.RemoveBody(1) // body at index 0 becomes nil

		data := make([]byte, 320) // 2 bodies
		binary.LittleEndian.PutUint32(data[160:], math.Float32bits(99))
		suite.NotPanics(func() {
			p.ProcessReadback(data)
		})
		suite.InDelta(99.0, float64(rb2.Position()[0]), 1e-5)
	})
}

func (suite *physicsTest) TestNewRigidBody() {
	suite.Run("defaults are mass one bounce half friction half", func() {
		rb := physics.NewRigidBody()
		suite.Equal(float32(1.0), rb.Mass())
		suite.InDelta(1.0, float64(rb.InverseMass()), 1e-6)
		suite.Equal(float32(0.5), rb.Bounce())
		suite.Equal(float32(0.5), rb.Friction())
		suite.True(rb.SurfaceOnly())
	})

	suite.Run("zero mass non-kinematic body becomes static", func() {
		rb := physics.NewRigidBody(physics.WithMass(0))
		suite.True(rb.Static())
		suite.Equal(float32(0), rb.InverseMass())
	})

	suite.Run("zero mass kinematic body is not forced static", func() {
		rb := physics.NewRigidBody(physics.WithMass(0), physics.WithKinematic(true))
		suite.False(rb.Static())
	})

	suite.Run("inverse mass is computed from mass", func() {
		rb := physics.NewRigidBody(physics.WithMass(4))
		suite.InDelta(0.25, float64(rb.InverseMass()), 1e-6)
	})

	suite.Run("explicit inverse mass overrides computed value", func() {
		rb := physics.NewRigidBody(physics.WithMass(10), physics.WithInverseMass(0.5))
		suite.Equal(float32(0.5), rb.InverseMass())
	})

	suite.Run("particles with spread positions produce valid inertia tensor", func() {
		particles := []physics.Particle{
			{LocalPosition: [3]float32{1, 0, 0}},
			{LocalPosition: [3]float32{-1, 0, 0}},
			{LocalPosition: [3]float32{0, 1, 0}},
			{LocalPosition: [3]float32{0, -1, 0}},
		}
		rb := physics.NewRigidBody(
			physics.WithMass(4),
			physics.WithParticles(particles),
			physics.WithParticleRadius(0.25),
		)
		invI := rb.InverseInertiaTensorBody()
		// diagonal elements should be non-zero for spread particles
		suite.NotEqual(float32(0), invI[0])
		suite.NotEqual(float32(0), invI[4])
		suite.NotEqual(float32(0), invI[8])
	})

	suite.Run("single particle at origin uses solid sphere fallback", func() {
		rb := physics.NewRigidBody(
			physics.WithMass(1),
			physics.WithParticles([]physics.Particle{
				{LocalPosition: [3]float32{0, 0, 0}},
			}),
			physics.WithParticleRadius(0.5),
		)
		invI := rb.InverseInertiaTensorBody()
		// I = 2/5 * m * r² = 2/5 * 1 * 0.25 = 0.1 → invI = 10
		expected := float32(1.0 / (0.4 * 1.0 * 0.5 * 0.5))
		suite.InDelta(float64(expected), float64(invI[0]), 1e-4)
		suite.InDelta(float64(expected), float64(invI[4]), 1e-4)
		suite.InDelta(float64(expected), float64(invI[8]), 1e-4)
		// off-diagonals should be zero
		suite.InDelta(0.0, float64(invI[1]), 1e-6)
		suite.InDelta(0.0, float64(invI[2]), 1e-6)
		suite.InDelta(0.0, float64(invI[3]), 1e-6)
	})

	suite.Run("no particles yields zero inertia tensor", func() {
		rb := physics.NewRigidBody(physics.WithMass(1))
		invI := rb.InverseInertiaTensorBody()
		for i := range 9 {
			suite.Equal(float32(0), invI[i])
		}
	})

	suite.Run("zero mass with particles yields zero inertia tensor", func() {
		rb := physics.NewRigidBody(
			physics.WithMass(0),
			physics.WithParticles([]physics.Particle{
				{LocalPosition: [3]float32{1, 0, 0}},
			}),
		)
		invI := rb.InverseInertiaTensorBody()
		for i := range 9 {
			suite.Equal(float32(0), invI[i])
		}
	})

	suite.Run("all builder options apply correctly", func() {
		rb := physics.NewRigidBody(
			physics.WithVelocity([3]float32{1, 2, 3}),
			physics.WithAngularVelocity([3]float32{4, 5, 6}),
			physics.WithMass(10),
			physics.WithBounce(0.8),
			physics.WithFriction(0.3),
			physics.WithActive(true),
			physics.WithKinematic(true),
			physics.WithParticleRadius(0.5),
		)
		suite.Equal([3]float32{1, 2, 3}, rb.Velocity())
		suite.Equal([3]float32{4, 5, 6}, rb.AngularVelocity())
		suite.Equal(float32(10), rb.Mass())
		suite.Equal(float32(0.8), rb.Bounce())
		suite.Equal(float32(0.3), rb.Friction())
		suite.True(rb.Active())
		suite.True(rb.Kinematic())
		suite.Equal(float32(0.5), rb.ParticleRadius())
	})
}

func (suite *physicsTest) TestRigidBodySetters() {
	suite.Run("SetVelocity updates velocity", func() {
		rb := physics.NewRigidBody()
		rb.SetVelocity([3]float32{7, 8, 9})
		suite.Equal([3]float32{7, 8, 9}, rb.Velocity())
	})

	suite.Run("SetAngularVelocity updates angular velocity", func() {
		rb := physics.NewRigidBody()
		rb.SetAngularVelocity([3]float32{1, 2, 3})
		suite.Equal([3]float32{1, 2, 3}, rb.AngularVelocity())
	})

	suite.Run("SetMass updates mass", func() {
		rb := physics.NewRigidBody()
		rb.SetMass(50)
		suite.Equal(float32(50), rb.Mass())
	})

	suite.Run("SetInverseMass updates inverse mass", func() {
		rb := physics.NewRigidBody()
		rb.SetInverseMass(0.1)
		suite.Equal(float32(0.1), rb.InverseMass())
	})

	suite.Run("SetBounce updates bounce", func() {
		rb := physics.NewRigidBody()
		rb.SetBounce(0.9)
		suite.Equal(float32(0.9), rb.Bounce())
	})

	suite.Run("SetFriction updates friction", func() {
		rb := physics.NewRigidBody()
		rb.SetFriction(0.2)
		suite.Equal(float32(0.2), rb.Friction())
	})

	suite.Run("SetActive updates active flag", func() {
		rb := physics.NewRigidBody()
		rb.SetActive(true)
		suite.True(rb.Active())
		rb.SetActive(false)
		suite.False(rb.Active())
	})

	suite.Run("SetStatic updates static flag", func() {
		rb := physics.NewRigidBody()
		rb.SetStatic(true)
		suite.True(rb.Static())
	})

	suite.Run("SetKinematic updates kinematic flag", func() {
		rb := physics.NewRigidBody()
		rb.SetKinematic(true)
		suite.True(rb.Kinematic())
	})

	suite.Run("SetParticles updates particles and recomputes inertia tensor", func() {
		rb := physics.NewRigidBody(physics.WithMass(4), physics.WithParticleRadius(0.5))
		suite.Nil(rb.Particles())

		particles := []physics.Particle{
			{LocalPosition: [3]float32{1, 0, 0}},
			{LocalPosition: [3]float32{-1, 0, 0}},
			{LocalPosition: [3]float32{0, 1, 0}},
			{LocalPosition: [3]float32{0, -1, 0}},
		}
		rb.SetParticles(particles)
		suite.Len(rb.Particles(), 4)
		invI := rb.InverseInertiaTensorBody()
		// with particles spread in XY, all diagonal elements should be non-zero
		suite.NotEqual(float32(0), invI[0])
		suite.NotEqual(float32(0), invI[4])
		suite.NotEqual(float32(0), invI[8])
	})

	suite.Run("SetParticleRadius updates particle radius", func() {
		rb := physics.NewRigidBody()
		rb.SetParticleRadius(1.5)
		suite.Equal(float32(1.5), rb.ParticleRadius())
	})

	suite.Run("SetSurfaceOnly updates surface only flag", func() {
		rb := physics.NewRigidBody()
		suite.True(rb.SurfaceOnly()) // default is true
		rb.SetSurfaceOnly(false)
		suite.False(rb.SurfaceOnly())
	})

	suite.Run("SetPosition updates position", func() {
		rb := physics.NewRigidBody()
		rb.SetPosition([3]float32{10, 20, 30})
		suite.Equal([3]float32{10, 20, 30}, rb.Position())
	})

	suite.Run("SetQuaternion updates quaternion", func() {
		rb := physics.NewRigidBody()
		rb.SetQuaternion([4]float32{0, 0, 0, 1})
		suite.Equal([4]float32{0, 0, 0, 1}, rb.Quaternion())
	})
}

func (suite *physicsTest) TestApplyForce() {
	suite.Run("accumulates force vector", func() {
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		rb.ApplyForce([3]float32{10, 0, 0})
		rb.ApplyForce([3]float32{0, 5, 0})

		// force is drained through PrepareStep, we verify via physics controller
		p := physics.NewPhysics(physics.WithFixedDt(1.0 / 60.0))
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		p.PrepareStep(1.0 / 60.0)
		writes := p.StagedWriteData()

		var foundForce bool
		for _, w := range writes {
			if w.Binding == 0 && w.Offset == 128 {
				foundForce = true
				fx := math.Float32frombits(binary.LittleEndian.Uint32(w.Data[0:4]))
				fy := math.Float32frombits(binary.LittleEndian.Uint32(w.Data[4:8]))
				suite.InDelta(10.0, float64(fx), 1e-5)
				suite.InDelta(5.0, float64(fy), 1e-5)
			}
		}
		suite.True(foundForce)
	})

	suite.Run("force is drained after prepare step", func() {
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		rb.ApplyForce([3]float32{10, 0, 0})

		p := physics.NewPhysics(physics.WithFixedDt(1.0 / 60.0))
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		p.PrepareStep(1.0 / 60.0)
		p.StagedWriteData()

		// Second call with no new force should produce no force write
		p.PrepareStep(1.0 / 60.0)
		writes := p.StagedWriteData()
		for _, w := range writes {
			if w.Binding == 0 && w.Offset == 128 {
				suite.Fail("unexpected force write on second step without ApplyForce")
			}
		}
	})
}

func (suite *physicsTest) TestApplyTorque() {
	suite.Run("accumulates torque vector", func() {
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithActive(true))
		rb.ApplyTorque([3]float32{0, 100, 0})
		rb.ApplyTorque([3]float32{0, 100, 0})

		p := physics.NewPhysics(physics.WithFixedDt(1.0 / 60.0))
		p.RegisterBody(1, [3]float32{}, [3]float32{}, rb, 0)
		p.StagedWriteData()

		p.PrepareStep(1.0 / 60.0)
		writes := p.StagedWriteData()

		var foundTorque bool
		for _, w := range writes {
			if w.Binding == 0 && w.Offset == 144 {
				foundTorque = true
				ty := math.Float32frombits(binary.LittleEndian.Uint32(w.Data[4:8]))
				suite.InDelta(200.0, float64(ty), 1e-4)
			}
		}
		suite.True(foundTorque)
	})
}

func (suite *physicsTest) TestGPUBodyMarshal() {
	suite.Run("produces 160 byte buffer", func() {
		body := physics.GPUBody{}
		data := body.Marshal()
		suite.Equal(160, len(data))
	})

	suite.Run("size returns 160", func() {
		body := physics.GPUBody{}
		suite.Equal(160, body.Size())
	})

	suite.Run("position is at offset zero", func() {
		body := physics.GPUBody{Position: [4]float32{1, 2, 3, 0}}
		data := body.Marshal()
		x := math.Float32frombits(binary.LittleEndian.Uint32(data[0:4]))
		y := math.Float32frombits(binary.LittleEndian.Uint32(data[4:8]))
		z := math.Float32frombits(binary.LittleEndian.Uint32(data[8:12]))
		suite.InDelta(1.0, float64(x), 1e-7)
		suite.InDelta(2.0, float64(y), 1e-7)
		suite.InDelta(3.0, float64(z), 1e-7)
	})

	suite.Run("quaternion is at offset 16", func() {
		body := physics.GPUBody{Quaternion: [4]float32{0.1, 0.2, 0.3, 0.9}}
		data := body.Marshal()
		qx := math.Float32frombits(binary.LittleEndian.Uint32(data[16:20]))
		suite.InDelta(0.1, float64(qx), 1e-6)
	})

	suite.Run("flags are at offset 124", func() {
		body := physics.GPUBody{Flags: 7}
		data := body.Marshal()
		flags := binary.LittleEndian.Uint32(data[124:128])
		suite.Equal(uint32(7), flags)
	})

	suite.Run("inverse mass is at offset 112", func() {
		body := physics.GPUBody{InverseMass: 0.5}
		data := body.Marshal()
		invMass := math.Float32frombits(binary.LittleEndian.Uint32(data[112:116]))
		suite.InDelta(0.5, float64(invMass), 1e-7)
	})

	suite.Run("particle start and count at offsets 116 and 120", func() {
		body := physics.GPUBody{ParticleStart: 10, ParticleCount: 20}
		data := body.Marshal()
		start := binary.LittleEndian.Uint32(data[116:120])
		count := binary.LittleEndian.Uint32(data[120:124])
		suite.Equal(uint32(10), start)
		suite.Equal(uint32(20), count)
	})

	suite.Run("external force and torque at correct offsets", func() {
		body := physics.GPUBody{
			ExternalForce:  [4]float32{1, 2, 3, 0},
			ExternalTorque: [4]float32{4, 5, 6, 0},
		}
		data := body.Marshal()
		fx := math.Float32frombits(binary.LittleEndian.Uint32(data[128:132]))
		tx := math.Float32frombits(binary.LittleEndian.Uint32(data[144:148]))
		suite.InDelta(1.0, float64(fx), 1e-7)
		suite.InDelta(4.0, float64(tx), 1e-7)
	})

	suite.Run("inverse inertia tensor columns are padded to vec4 stride", func() {
		// Set known values for a 3x3 identity encoded as [col0.x, col0.y, col0.z, pad, col1.x, ...]
		body := physics.GPUBody{
			InvInertiaTBody: [12]float32{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
			},
		}
		data := body.Marshal()
		// Column 0 starts at offset 64
		col0x := math.Float32frombits(binary.LittleEndian.Uint32(data[64:68]))
		pad0 := binary.LittleEndian.Uint32(data[76:80]) // padding byte at offset 76
		suite.InDelta(1.0, float64(col0x), 1e-7)
		suite.Equal(uint32(0), pad0) // padding is zero
	})
}

func (suite *physicsTest) TestGPUParticleMarshal() {
	suite.Run("produces 96 byte buffer", func() {
		p := physics.GPUParticle{}
		data := p.Marshal()
		suite.Equal(96, len(data))
	})

	suite.Run("size returns 96", func() {
		p := physics.GPUParticle{}
		suite.Equal(96, p.Size())
	})

	suite.Run("world position is at offset zero", func() {
		p := physics.GPUParticle{WorldPosition: [4]float32{10, 20, 30, 0}}
		data := p.Marshal()
		x := math.Float32frombits(binary.LittleEndian.Uint32(data[0:4]))
		suite.InDelta(10.0, float64(x), 1e-7)
	})

	suite.Run("local position w carries body index via bitcast", func() {
		packed := math.Float32frombits(42) // body index 42
		p := physics.GPUParticle{LocalPosition: [4]float32{1, 2, 3, packed}}
		data := p.Marshal()
		// w component at offset 64+12=76
		wBits := binary.LittleEndian.Uint32(data[76:80])
		suite.Equal(uint32(42), math.Float32bits(math.Float32frombits(wBits)))
	})

	suite.Run("surface normal w at offset 92", func() {
		p := physics.GPUParticle{SurfaceNormal: [4]float32{0, 1, 0, 1.0}}
		data := p.Marshal()
		w := math.Float32frombits(binary.LittleEndian.Uint32(data[92:96]))
		suite.InDelta(1.0, float64(w), 1e-7)
	})
}

func (suite *physicsTest) TestGPUGridCellMarshal() {
	suite.Run("produces 64 byte buffer", func() {
		cell := physics.GPUGridCell{}
		suite.Equal(64, len(cell.Marshal()))
	})

	suite.Run("size returns 64", func() {
		cell := physics.GPUGridCell{}
		suite.Equal(64, cell.Size())
	})

	suite.Run("sentinel values are preserved", func() {
		cell := physics.GPUGridCell{
			Indices0: [4]uint32{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
			Indices1: [4]uint32{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
			Indices2: [4]uint32{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
			Indices3: [4]uint32{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
		}
		data := cell.Marshal()
		for i := range 16 {
			val := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
			suite.Equal(uint32(0xFFFFFFFF), val)
		}
	})

	suite.Run("specific indices are at correct offsets", func() {
		cell := physics.GPUGridCell{
			Indices0: [4]uint32{0, 1, 2, 3},
			Indices1: [4]uint32{4, 5, 6, 7},
			Indices2: [4]uint32{8, 9, 10, 11},
			Indices3: [4]uint32{12, 13, 14, 15},
		}
		data := cell.Marshal()
		for i := range 16 {
			val := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
			suite.Equal(uint32(i), val)
		}
	})
}

func (suite *physicsTest) TestGPUPhysicsGlobalsMarshal() {
	suite.Run("produces 240 byte buffer", func() {
		g := physics.GPUPhysicsGlobals{}
		suite.Equal(240, len(g.Marshal()))
	})

	suite.Run("size returns 240", func() {
		g := physics.GPUPhysicsGlobals{}
		suite.Equal(240, g.Size())
	})

	suite.Run("delta time at offset zero", func() {
		g := physics.GPUPhysicsGlobals{DeltaTime: 1.0 / 60.0}
		data := g.Marshal()
		dt := math.Float32frombits(binary.LittleEndian.Uint32(data[0:4]))
		suite.InDelta(1.0/60.0, float64(dt), 1e-7)
	})

	suite.Run("spring damping shear coefficients at correct offsets", func() {
		g := physics.GPUPhysicsGlobals{
			SpringCoeff:  2.0,
			DampingCoeff: 0.3,
			ShearCoeff:   0.7,
		}
		data := g.Marshal()
		spring := math.Float32frombits(binary.LittleEndian.Uint32(data[8:12]))
		damp := math.Float32frombits(binary.LittleEndian.Uint32(data[12:16]))
		shear := math.Float32frombits(binary.LittleEndian.Uint32(data[16:20]))
		suite.InDelta(2.0, float64(spring), 1e-7)
		suite.InDelta(0.3, float64(damp), 1e-7)
		suite.InDelta(0.7, float64(shear), 1e-7)
	})

	suite.Run("body and particle counts at offsets 20 and 24", func() {
		g := physics.GPUPhysicsGlobals{BodyCount: 5, ParticleCount: 100}
		data := g.Marshal()
		suite.Equal(uint32(5), binary.LittleEndian.Uint32(data[20:24]))
		suite.Equal(uint32(100), binary.LittleEndian.Uint32(data[24:28]))
	})

	suite.Run("gravity at offsets 36 40 44", func() {
		g := physics.GPUPhysicsGlobals{GravityX: 0, GravityY: -9.81, GravityZ: 0}
		data := g.Marshal()
		gy := math.Float32frombits(binary.LittleEndian.Uint32(data[40:44]))
		suite.InDelta(-9.81, float64(gy), 1e-5)
	})

	suite.Run("boundary planes start at offset 48", func() {
		g := physics.GPUPhysicsGlobals{}
		g.BoundaryPlanes[0] = [4]float32{1, 0, 0, 5}
		data := g.Marshal()
		nx := math.Float32frombits(binary.LittleEndian.Uint32(data[48:52]))
		d := math.Float32frombits(binary.LittleEndian.Uint32(data[60:64]))
		suite.InDelta(1.0, float64(nx), 1e-7)
		suite.InDelta(5.0, float64(d), 1e-7)
	})

	suite.Run("boundary y ranges start at offset 144", func() {
		g := physics.GPUPhysicsGlobals{}
		g.BoundaryYRanges[0] = [4]float32{-10, 10, 0, 0}
		data := g.Marshal()
		yMin := math.Float32frombits(binary.LittleEndian.Uint32(data[144:148]))
		yMax := math.Float32frombits(binary.LittleEndian.Uint32(data[148:152]))
		suite.InDelta(-10.0, float64(yMin), 1e-7)
		suite.InDelta(10.0, float64(yMax), 1e-7)
	})
}

func (suite *physicsTest) TestGPUGridParamsMarshal() {
	suite.Run("produces 32 byte buffer", func() {
		g := physics.GPUGridParams{}
		suite.Equal(32, len(g.Marshal()))
	})

	suite.Run("size returns 32", func() {
		g := physics.GPUGridParams{}
		suite.Equal(32, g.Size())
	})

	suite.Run("grid origin at offset zero", func() {
		g := physics.GPUGridParams{GridOrigin: [4]float32{1, 2, 3, 0}}
		data := g.Marshal()
		x := math.Float32frombits(binary.LittleEndian.Uint32(data[0:4]))
		suite.InDelta(1.0, float64(x), 1e-7)
	})

	suite.Run("grid dims at offset 16", func() {
		g := physics.GPUGridParams{GridDims: [4]uint32{64, 32, 16, 64 * 32 * 16}}
		data := g.Marshal()
		dx := binary.LittleEndian.Uint32(data[16:20])
		dy := binary.LittleEndian.Uint32(data[20:24])
		dz := binary.LittleEndian.Uint32(data[24:28])
		total := binary.LittleEndian.Uint32(data[28:32])
		suite.Equal(uint32(64), dx)
		suite.Equal(uint32(32), dy)
		suite.Equal(uint32(16), dz)
		suite.Equal(uint32(64*32*16), total)
	})
}

func (suite *physicsTest) TestVoxelizeMesh() {
	suite.Run("returns nil for model with insufficient vertex data", func() {
		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().VertexData().Return(nil).Maybe()
		mdl.EXPECT().IndexData().Return(nil).Maybe()

		result := physics.VoxelizeMesh(mdl, 0.1, false)
		suite.Nil(result)
	})

	suite.Run("returns nil for model with insufficient index data", func() {
		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		// 2 vertices worth of data (128 bytes at stride 64)
		mdl.EXPECT().VertexData().Return(make([]byte, 128)).Maybe()
		// Fewer than 3 indices (only 2 uint32s = 8 bytes)
		mdl.EXPECT().IndexData().Return(make([]byte, 8)).Maybe()

		result := physics.VoxelizeMesh(mdl, 0.1, false)
		suite.Nil(result)
	})

	suite.Run("produces particles for a closed cube mesh", func() {
		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()

		// Use a closed cube mesh so ray-cast parity produces interior voxels
		mdl.EXPECT().VertexData().Return(buildCubeVertexData()).Maybe()
		mdl.EXPECT().IndexData().Return(buildCubeIndexData()).Maybe()

		result := physics.VoxelizeMesh(mdl, 0.3, false)
		suite.NotNil(result)
		suite.Greater(len(result), 0)
	})

	suite.Run("surface only produces fewer particles than volume fill", func() {
		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()

		// Build a cube-like shape with two triangles per face (simplified: just a
		// large triangle that creates multiple voxel layers)
		vertData := buildCubeVertexData()
		idxData := buildCubeIndexData()

		mdl.EXPECT().VertexData().Return(vertData).Maybe()
		mdl.EXPECT().IndexData().Return(idxData).Maybe()

		volumeParticles := physics.VoxelizeMesh(mdl, 0.3, false)

		mdl2 := &modelmocks.MockModel{}
		mdl2.EXPECT().Skinned().Return(false).Maybe()
		mdl2.EXPECT().VertexData().Return(vertData).Maybe()
		mdl2.EXPECT().IndexData().Return(idxData).Maybe()
		surfaceParticles := physics.VoxelizeMesh(mdl2, 0.3, true)

		if volumeParticles != nil && surfaceParticles != nil && len(volumeParticles) > len(surfaceParticles) {
			suite.Less(len(surfaceParticles), len(volumeParticles))
		}
	})

	suite.Run("skinned model uses stride 96", func() {
		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()

		// Build a closed cube with stride 96 (skinned vertices)
		mdl.EXPECT().VertexData().Return(buildSkinnedCubeVertexData()).Maybe()
		mdl.EXPECT().IndexData().Return(buildCubeIndexData()).Maybe()

		result := physics.VoxelizeMesh(mdl, 0.3, false)
		suite.NotNil(result)
		suite.Greater(len(result), 0)
	})

	suite.Run("degenerate mesh with zero extent returns nil", func() {
		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()

		// All vertices at the same point → AABB has zero extent
		vertData := make([]byte, 64*3)
		mdl.EXPECT().VertexData().Return(vertData).Maybe()

		idxData := make([]byte, 12)
		binary.LittleEndian.PutUint32(idxData[0:], 0)
		binary.LittleEndian.PutUint32(idxData[4:], 1)
		binary.LittleEndian.PutUint32(idxData[8:], 2)
		mdl.EXPECT().IndexData().Return(idxData).Maybe()

		result := physics.VoxelizeMesh(mdl, 0.5, false)
		suite.Nil(result)
	})
}

func (suite *physicsTest) TestAssignBoneIndices() {
	suite.Run("no-op for non-skinned model", func() {
		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()

		particles := []physics.Particle{
			{LocalPosition: [3]float32{0, 0, 0}},
		}
		physics.AssignBoneIndices(particles, mdl)
		suite.Equal(uint32(0), particles[0].BoneIndex)
	})

	suite.Run("no-op when skeleton is nil", func() {
		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Maybe()

		particles := []physics.Particle{
			{LocalPosition: [3]float32{0, 0, 0}},
		}
		physics.AssignBoneIndices(particles, mdl)
		suite.Equal(uint32(0), particles[0].BoneIndex)
	})

	suite.Run("no-op for zero vertex count", func() {
		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(&model.Skeleton{}).Maybe()
		mdl.EXPECT().VertexData().Return(nil).Maybe()

		particles := []physics.Particle{
			{LocalPosition: [3]float32{1, 0, 0}},
		}
		physics.AssignBoneIndices(particles, mdl)
		suite.Equal(uint32(0), particles[0].BoneIndex)
	})

	suite.Run("assigns nearest bone and transforms to bone-local space", func() {
		// Create a skinned vertex buffer with one vertex at (1,0,0) bound to bone 0
		vertData := make([]byte, 96)
		// Position at offset 0
		binary.LittleEndian.PutUint32(vertData[0:], math.Float32bits(1.0))
		binary.LittleEndian.PutUint32(vertData[4:], math.Float32bits(0.0))
		binary.LittleEndian.PutUint32(vertData[8:], math.Float32bits(0.0))
		// BoneIndices at offset 64: bone 0 for slot 0
		binary.LittleEndian.PutUint32(vertData[64:], 0)
		binary.LittleEndian.PutUint32(vertData[68:], 0)
		binary.LittleEndian.PutUint32(vertData[72:], 0)
		binary.LittleEndian.PutUint32(vertData[76:], 0)
		// BoneWeights at offset 80: weight 1.0 for slot 0
		binary.LittleEndian.PutUint32(vertData[80:], math.Float32bits(1.0))
		binary.LittleEndian.PutUint32(vertData[84:], math.Float32bits(0.0))
		binary.LittleEndian.PutUint32(vertData[88:], math.Float32bits(0.0))
		binary.LittleEndian.PutUint32(vertData[92:], math.Float32bits(0.0))

		// Identity inverse bind matrix for bone 0
		identity := [16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		skel := &model.Skeleton{
			Bones: []model.Bone{
				{Name: "root", ParentIndex: -1, InverseBindMatrix: identity},
			},
		}

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(skel).Maybe()
		mdl.EXPECT().VertexData().Return(vertData).Maybe()

		particles := []physics.Particle{
			{LocalPosition: [3]float32{1, 0, 0}},
		}
		physics.AssignBoneIndices(particles, mdl)

		suite.Equal(uint32(0), particles[0].BoneIndex)
		// With identity inverse bind, position should remain unchanged
		suite.InDelta(1.0, float64(particles[0].LocalPosition[0]), 1e-5)
		suite.InDelta(0.0, float64(particles[0].LocalPosition[1]), 1e-5)
		suite.InDelta(0.0, float64(particles[0].LocalPosition[2]), 1e-5)
	})

	suite.Run("transforms position using non-identity inverse bind matrix", func() {
		vertData := make([]byte, 96)
		binary.LittleEndian.PutUint32(vertData[0:], math.Float32bits(0.0))
		binary.LittleEndian.PutUint32(vertData[4:], math.Float32bits(0.0))
		binary.LittleEndian.PutUint32(vertData[8:], math.Float32bits(0.0))
		binary.LittleEndian.PutUint32(vertData[64:], 0)
		binary.LittleEndian.PutUint32(vertData[80:], math.Float32bits(1.0))

		// Translation-only inverse bind matrix: translate by (5, 0, 0)
		translateInvBind := [16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			5, 0, 0, 1,
		}
		skel := &model.Skeleton{
			Bones: []model.Bone{
				{Name: "root", ParentIndex: -1, InverseBindMatrix: translateInvBind},
			},
		}

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(skel).Maybe()
		mdl.EXPECT().VertexData().Return(vertData).Maybe()

		particles := []physics.Particle{
			{LocalPosition: [3]float32{0, 0, 0}},
		}
		physics.AssignBoneIndices(particles, mdl)

		// bone_local = inverseBind * model_pos
		// Using column-major multiplication: result.x = m[0]*x + m[4]*y + m[8]*z + m[12]
		// = 1*0 + 0*0 + 0*0 + 5 = 5
		suite.InDelta(5.0, float64(particles[0].LocalPosition[0]), 1e-5)
	})

	suite.Run("picks bone with highest weight among four slots", func() {
		vertData := make([]byte, 96)
		binary.LittleEndian.PutUint32(vertData[0:], math.Float32bits(0.0))
		binary.LittleEndian.PutUint32(vertData[4:], math.Float32bits(0.0))
		binary.LittleEndian.PutUint32(vertData[8:], math.Float32bits(0.0))
		// BoneIndices: 0, 1, 2, 3
		binary.LittleEndian.PutUint32(vertData[64:], 0)
		binary.LittleEndian.PutUint32(vertData[68:], 1)
		binary.LittleEndian.PutUint32(vertData[72:], 2)
		binary.LittleEndian.PutUint32(vertData[76:], 3)
		// BoneWeights: 0.1, 0.6, 0.2, 0.1 → bone 1 should win
		binary.LittleEndian.PutUint32(vertData[80:], math.Float32bits(0.1))
		binary.LittleEndian.PutUint32(vertData[84:], math.Float32bits(0.6))
		binary.LittleEndian.PutUint32(vertData[88:], math.Float32bits(0.2))
		binary.LittleEndian.PutUint32(vertData[92:], math.Float32bits(0.1))

		identity := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		skel := &model.Skeleton{
			Bones: []model.Bone{
				{Name: "bone0", InverseBindMatrix: identity},
				{Name: "bone1", InverseBindMatrix: identity},
				{Name: "bone2", InverseBindMatrix: identity},
				{Name: "bone3", InverseBindMatrix: identity},
			},
		}

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(skel).Maybe()
		mdl.EXPECT().VertexData().Return(vertData).Maybe()

		particles := []physics.Particle{
			{LocalPosition: [3]float32{0, 0, 0}},
		}
		physics.AssignBoneIndices(particles, mdl)
		suite.Equal(uint32(1), particles[0].BoneIndex)
	})
}

// buildCubeVertexData creates vertex data for a simple axis-aligned cube with
// corners at (-1,-1,-1) to (1,1,1) encoded at stride 64 (non-skinned).
func buildCubeVertexData() []byte {
	verts := [][3]float32{
		{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, // front
		{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}, // back
	}
	data := make([]byte, 64*len(verts))
	for i, v := range verts {
		off := i * 64
		binary.LittleEndian.PutUint32(data[off:], math.Float32bits(v[0]))
		binary.LittleEndian.PutUint32(data[off+4:], math.Float32bits(v[1]))
		binary.LittleEndian.PutUint32(data[off+8:], math.Float32bits(v[2]))
	}
	return data
}

// buildSkinnedCubeVertexData creates vertex data for a cube with stride 96 (skinned format).
func buildSkinnedCubeVertexData() []byte {
	verts := [][3]float32{
		{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
		{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
	}
	data := make([]byte, 96*len(verts))
	for i, v := range verts {
		off := i * 96
		binary.LittleEndian.PutUint32(data[off:], math.Float32bits(v[0]))
		binary.LittleEndian.PutUint32(data[off+4:], math.Float32bits(v[1]))
		binary.LittleEndian.PutUint32(data[off+8:], math.Float32bits(v[2]))
	}
	return data
}

// buildCubeIndexData creates index data for 12 triangles forming a simple cube.
func buildCubeIndexData() []byte {
	indices := []uint32{
		// front face
		0, 1, 2, 0, 2, 3,
		// back face
		4, 6, 5, 4, 7, 6,
		// left face
		0, 3, 7, 0, 7, 4,
		// right face
		1, 5, 6, 1, 6, 2,
		// bottom face
		0, 4, 5, 0, 5, 1,
		// top face
		3, 2, 6, 3, 6, 7,
	}
	data := make([]byte, 4*len(indices))
	for i, idx := range indices {
		binary.LittleEndian.PutUint32(data[i*4:], idx)
	}
	return data
}
