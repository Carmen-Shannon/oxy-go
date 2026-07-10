package physics_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/physics"
)

type rigidBodyTest struct {
	suite.Suite
}

func TestRunRigidBodyTests(t *testing.T) {
	suite.Run(t, new(rigidBodyTest))
}

func (suite *rigidBodyTest) SetupSubTest() {}

func (suite *rigidBodyTest) TestNewRigidBody() {
	suite.Run("WithVelocity sets velocity", func() {
		rb := physics.NewRigidBody(physics.WithVelocity([3]float32{1, 2, 3}))
		suite.Equal([3]float32{1, 2, 3}, rb.Velocity())
	})

	suite.Run("WithInverseMass skips auto-set branch", func() {
		rb := physics.NewRigidBody(physics.WithInverseMass(0.5))
		suite.InDelta(float32(0.5), rb.InverseMass(), 1e-6)
	})

	suite.Run("WithBounce sets bounce", func() {
		rb := physics.NewRigidBody(physics.WithBounce(0.8))
		suite.InDelta(float32(0.8), rb.Bounce(), 1e-6)
	})

	suite.Run("WithFriction sets friction", func() {
		rb := physics.NewRigidBody(physics.WithFriction(0.3))
		suite.InDelta(float32(0.3), rb.Friction(), 1e-6)
	})

	suite.Run("zero mass not kinematic becomes static", func() {
		rb := physics.NewRigidBody(physics.WithMass(0))
		suite.True(rb.Static())
	})

	suite.Run("zero mass kinematic does not become static", func() {
		rb := physics.NewRigidBody(physics.WithMass(0), physics.WithKinematic(true))
		suite.False(rb.Static())
	})
}

func (suite *rigidBodyTest) TestBounce() {
	suite.Run("returns configured bounce value", func() {
		rb := physics.NewRigidBody(physics.WithBounce(0.7))
		suite.InDelta(float32(0.7), rb.Bounce(), 1e-6)
	})
}

func (suite *rigidBodyTest) TestFriction() {
	suite.Run("returns configured friction value", func() {
		rb := physics.NewRigidBody(physics.WithFriction(0.2))
		suite.InDelta(float32(0.2), rb.Friction(), 1e-6)
	})
}

func (suite *rigidBodyTest) TestSurfaceOnly() {
	suite.Run("defaults to true", func() {
		rb := physics.NewRigidBody()
		suite.True(rb.SurfaceOnly())
	})
}

func (suite *rigidBodyTest) TestSetVelocity() {
	suite.Run("set then get", func() {
		rb := physics.NewRigidBody()
		rb.SetVelocity([3]float32{4, 5, 6})
		suite.Equal([3]float32{4, 5, 6}, rb.Velocity())
	})
}

func (suite *rigidBodyTest) TestSetAngularVelocity() {
	suite.Run("set then get", func() {
		rb := physics.NewRigidBody()
		rb.SetAngularVelocity([3]float32{7, 8, 9})
		suite.Equal([3]float32{7, 8, 9}, rb.AngularVelocity())
	})
}

func (suite *rigidBodyTest) TestSetMass() {
	suite.Run("set then get", func() {
		rb := physics.NewRigidBody()
		rb.SetMass(5.0)
		suite.InDelta(float32(5.0), rb.Mass(), 1e-6)
	})
}

func (suite *rigidBodyTest) TestSetInverseMass() {
	suite.Run("set then get", func() {
		rb := physics.NewRigidBody()
		rb.SetInverseMass(0.25)
		suite.InDelta(float32(0.25), rb.InverseMass(), 1e-6)
	})
}

func (suite *rigidBodyTest) TestSetBounce() {
	suite.Run("set then get", func() {
		rb := physics.NewRigidBody()
		rb.SetBounce(0.9)
		suite.InDelta(float32(0.9), rb.Bounce(), 1e-6)
	})
}

func (suite *rigidBodyTest) TestSetFriction() {
	suite.Run("set then get", func() {
		rb := physics.NewRigidBody()
		rb.SetFriction(0.1)
		suite.InDelta(float32(0.1), rb.Friction(), 1e-6)
	})
}

func (suite *rigidBodyTest) TestSetActive() {
	suite.Run("false to true", func() {
		rb := physics.NewRigidBody()
		rb.SetActive(true)
		suite.True(rb.Active())
	})

	suite.Run("true to false", func() {
		rb := physics.NewRigidBody(physics.WithActive(true))
		rb.SetActive(false)
		suite.False(rb.Active())
	})
}

func (suite *rigidBodyTest) TestSetStatic() {
	suite.Run("false to true", func() {
		rb := physics.NewRigidBody()
		rb.SetStatic(true)
		suite.True(rb.Static())
	})
}

func (suite *rigidBodyTest) TestSetKinematic() {
	suite.Run("false to true", func() {
		rb := physics.NewRigidBody()
		rb.SetKinematic(true)
		suite.True(rb.Kinematic())
	})
}

func (suite *rigidBodyTest) TestSetParticleRadius() {
	suite.Run("set then get", func() {
		rb := physics.NewRigidBody()
		rb.SetParticleRadius(0.3)
		suite.InDelta(float32(0.3), rb.ParticleRadius(), 1e-6)
	})
}

func (suite *rigidBodyTest) TestSetSurfaceOnly() {
	suite.Run("true to false", func() {
		rb := physics.NewRigidBody()
		rb.SetSurfaceOnly(false)
		suite.False(rb.SurfaceOnly())
	})
}

func (suite *rigidBodyTest) TestSetParticles() {
	suite.Run("nil particles clears and zeroes tensor", func() {
		rb := physics.NewRigidBody()
		rb.SetParticles(nil)
		suite.Nil(rb.Particles())
		suite.Equal([9]float32{}, rb.InverseInertiaTensorBody())
	})

	suite.Run("non-nil particles at non-zero positions produces non-zero tensor", func() {
		rb := physics.NewRigidBody()
		rb.SetParticles([]physics.Particle{
			{LocalPosition: [3]float32{1, 0, 0}},
			{LocalPosition: [3]float32{0, 1, 0}},
			{LocalPosition: [3]float32{0, 0, 1}},
		})
		tensor := rb.InverseInertiaTensorBody()
		suite.True(tensor[0] != 0 || tensor[4] != 0 || tensor[8] != 0)
	})
}

func (suite *rigidBodyTest) TestWithAngularVelocity() {
	suite.Run("sets angular velocity via builder", func() {
		rb := physics.NewRigidBody(physics.WithAngularVelocity([3]float32{1, 2, 3}))
		suite.Equal([3]float32{1, 2, 3}, rb.AngularVelocity())
	})
}

func (suite *rigidBodyTest) TestWithStatic() {
	suite.Run("sets static flag via builder", func() {
		rb := physics.NewRigidBody(physics.WithMass(1), physics.WithStatic(true))
		suite.True(rb.Static())
	})
}

func (suite *rigidBodyTest) TestWithParticles() {
	suite.Run("builder particles trigger tensor computation in constructor", func() {
		rb := physics.NewRigidBody(physics.WithParticles([]physics.Particle{
			{LocalPosition: [3]float32{1, 0, 0}},
			{LocalPosition: [3]float32{0, 1, 0}},
			{LocalPosition: [3]float32{0, 0, 1}},
		}))
		suite.Len(rb.Particles(), 3)
	})
}

func (suite *rigidBodyTest) TestWithParticleRadius() {
	suite.Run("sets particle radius via builder", func() {
		rb := physics.NewRigidBody(physics.WithParticleRadius(0.5))
		suite.InDelta(float32(0.5), rb.ParticleRadius(), 1e-6)
	})
}

func (suite *rigidBodyTest) TestSetPosition() {
	suite.Run("set then get", func() {
		rb := physics.NewRigidBody()
		rb.SetPosition([3]float32{1, 2, 3})
		suite.Equal([3]float32{1, 2, 3}, rb.Position())
	})
}

func (suite *rigidBodyTest) TestSetQuaternion() {
	suite.Run("set then get", func() {
		rb := physics.NewRigidBody()
		rb.SetQuaternion([4]float32{0, 0, 0, 1})
		suite.Equal([4]float32{0, 0, 0, 1}, rb.Quaternion())
	})
}
