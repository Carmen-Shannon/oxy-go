package physics

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type rigidBodyImplTest struct {
	suite.Suite
}

func TestRunRigidBodyImplTests(t *testing.T) {
	suite.Run(t, new(rigidBodyImplTest))
}

func (suite *rigidBodyImplTest) SetupSubTest() {}

func (suite *rigidBodyImplTest) TestComputeInverseInertiaTensor() {
	suite.Run("zero particles returns zero tensor", func() {
		r := &rigidBody{mass: 1.0}
		r.computeInverseInertiaTensor()
		suite.Equal([9]float32{}, r.inverseInertiaTensorBody)
	})

	suite.Run("zero mass returns zero tensor", func() {
		r := &rigidBody{
			mass: 0,
			particles: []Particle{
				{LocalPosition: [3]float32{1, 0, 0}},
			},
		}
		r.computeInverseInertiaTensor()
		suite.Equal([9]float32{}, r.inverseInertiaTensorBody)
	})

	suite.Run("single particle at origin with radius uses sphere fallback", func() {
		r := &rigidBody{
			mass:           1.0,
			particleRadius: 0.5,
			particles:      []Particle{{LocalPosition: [3]float32{0, 0, 0}}},
		}
		r.computeInverseInertiaTensor()
		invI := float32(1.0 / (0.4 * 1.0 * 0.25))
		suite.InDelta(invI, r.inverseInertiaTensorBody[0], 1e-6)
		suite.InDelta(invI, r.inverseInertiaTensorBody[4], 1e-6)
		suite.InDelta(invI, r.inverseInertiaTensorBody[8], 1e-6)
		suite.InDelta(float32(0), r.inverseInertiaTensorBody[1], 1e-6)
		suite.InDelta(float32(0), r.inverseInertiaTensorBody[2], 1e-6)
		suite.InDelta(float32(0), r.inverseInertiaTensorBody[3], 1e-6)
		suite.InDelta(float32(0), r.inverseInertiaTensorBody[5], 1e-6)
		suite.InDelta(float32(0), r.inverseInertiaTensorBody[6], 1e-6)
		suite.InDelta(float32(0), r.inverseInertiaTensorBody[7], 1e-6)
	})

	suite.Run("single particle at origin with zero radius returns zero tensor", func() {
		r := &rigidBody{
			mass:           1.0,
			particleRadius: 0.0,
			particles:      []Particle{{LocalPosition: [3]float32{0, 0, 0}}},
		}
		r.computeInverseInertiaTensor()
		suite.Equal([9]float32{}, r.inverseInertiaTensorBody)
	})

	suite.Run("collinear particles on X axis produce degenerate tensor and fall back to zero", func() {
		r := &rigidBody{
			mass: 1.0,
			particles: []Particle{
				{LocalPosition: [3]float32{1, 0, 0}},
				{LocalPosition: [3]float32{2, 0, 0}},
			},
		}
		r.computeInverseInertiaTensor()
		suite.Equal([9]float32{}, r.inverseInertiaTensorBody)
	})

	suite.Run("non-collinear particles produce valid inverse tensor", func() {
		r := &rigidBody{
			mass: 1.0,
			particles: []Particle{
				{LocalPosition: [3]float32{1, 0, 0}},
				{LocalPosition: [3]float32{0, 1, 0}},
				{LocalPosition: [3]float32{0, 0, 1}},
			},
		}
		r.computeInverseInertiaTensor()
		tensor := r.inverseInertiaTensorBody
		suite.True(tensor[0] != 0 || tensor[4] != 0 || tensor[8] != 0)
	})
}

func (suite *rigidBodyImplTest) TestDrainForce() {
	suite.Run("accumulated force is returned and reset", func() {
		r := &rigidBody{}
		r.ApplyForce([3]float32{1, 2, 3})
		r.ApplyForce([3]float32{4, 5, 6})
		got := r.drainForce()
		suite.Equal([3]float32{5, 7, 9}, got)
		again := r.drainForce()
		suite.Equal([3]float32{}, again)
	})
}

func (suite *rigidBodyImplTest) TestDrainTorque() {
	suite.Run("accumulated torque is returned and reset", func() {
		r := &rigidBody{}
		r.ApplyTorque([3]float32{1, 0, 0})
		r.ApplyTorque([3]float32{0, 2, 0})
		got := r.drainTorque()
		suite.Equal([3]float32{1, 2, 0}, got)
		again := r.drainTorque()
		suite.Equal([3]float32{}, again)
	})
}
