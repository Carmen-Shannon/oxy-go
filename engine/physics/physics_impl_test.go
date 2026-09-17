package physics

import (
	"math"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/common"
)

type physicsImplTest struct {
	suite.Suite
}

func TestRunPhysicsImplTests(t *testing.T) {
	suite.Run(t, new(physicsImplTest))
}

func (suite *physicsImplTest) SetupSubTest() {}

type stubRigidBody struct {
	common.DelegateImpl[RigidBody]
}

func (s *stubRigidBody) Position() [3]float32                 { return [3]float32{} }
func (s *stubRigidBody) Quaternion() [4]float32               { return [4]float32{} }
func (s *stubRigidBody) Velocity() [3]float32                 { return [3]float32{} }
func (s *stubRigidBody) AngularVelocity() [3]float32          { return [3]float32{} }
func (s *stubRigidBody) Mass() float32                        { return 0 }
func (s *stubRigidBody) InverseMass() float32                 { return 0 }
func (s *stubRigidBody) Bounce() float32                      { return 0 }
func (s *stubRigidBody) Friction() float32                    { return 0 }
func (s *stubRigidBody) InverseInertiaTensorBody() [9]float32 { return [9]float32{} }
func (s *stubRigidBody) Active() bool                         { return false }
func (s *stubRigidBody) Static() bool                         { return false }
func (s *stubRigidBody) Kinematic() bool                      { return false }
func (s *stubRigidBody) Particles() []Particle                { return nil }
func (s *stubRigidBody) ParticleRadius() float32              { return 0 }
func (s *stubRigidBody) SurfaceOnly() bool                    { return false }
func (s *stubRigidBody) SetVelocity(_ [3]float32)             {}
func (s *stubRigidBody) SetAngularVelocity(_ [3]float32)      {}
func (s *stubRigidBody) SetMass(_ float32)                    {}
func (s *stubRigidBody) SetInverseMass(_ float32)             {}
func (s *stubRigidBody) SetBounce(_ float32)                  {}
func (s *stubRigidBody) SetFriction(_ float32)                {}
func (s *stubRigidBody) SetActive(_ bool)                     {}
func (s *stubRigidBody) SetStatic(_ bool)                     {}
func (s *stubRigidBody) SetKinematic(_ bool)                  {}
func (s *stubRigidBody) SetParticles(_ []Particle)            {}
func (s *stubRigidBody) SetParticleRadius(_ float32)          {}
func (s *stubRigidBody) SetSurfaceOnly(_ bool)                {}
func (s *stubRigidBody) SetPosition(_ [3]float32)             {}
func (s *stubRigidBody) SetQuaternion(_ [4]float32)           {}
func (s *stubRigidBody) ApplyForce(_ [3]float32)              {}
func (s *stubRigidBody) ApplyTorque(_ [3]float32)             {}

func (suite *physicsImplTest) TestEulerToQuaternion() {
	suite.Run("zero angles returns identity quaternion", func() {
		got := eulerToQuaternion([3]float32{0, 0, 0})
		suite.InDelta(0.0, got[0], 1e-6)
		suite.InDelta(0.0, got[1], 1e-6)
		suite.InDelta(0.0, got[2], 1e-6)
		suite.InDelta(1.0, got[3], 1e-6)
	})

	suite.Run("90 degrees around X", func() {
		got := eulerToQuaternion([3]float32{math.Pi / 2, 0, 0})
		suite.InDelta(math.Sqrt2/2, got[0], 1e-6)
		suite.InDelta(0.0, got[1], 1e-6)
		suite.InDelta(0.0, got[2], 1e-6)
		suite.InDelta(math.Sqrt2/2, got[3], 1e-6)
	})

	suite.Run("90 degrees around Y", func() {
		got := eulerToQuaternion([3]float32{0, math.Pi / 2, 0})
		suite.InDelta(0.0, got[0], 1e-6)
		suite.InDelta(math.Sqrt2/2, got[1], 1e-6)
		suite.InDelta(0.0, got[2], 1e-6)
		suite.InDelta(math.Sqrt2/2, got[3], 1e-6)
	})

	suite.Run("90 degrees around Z", func() {
		got := eulerToQuaternion([3]float32{0, 0, math.Pi / 2})
		suite.InDelta(0.0, got[0], 1e-6)
		suite.InDelta(0.0, got[1], 1e-6)
		suite.InDelta(math.Sqrt2/2, got[2], 1e-6)
		suite.InDelta(math.Sqrt2/2, got[3], 1e-6)
	})
}

func (suite *physicsImplTest) TestPrepareStepNonRigidBody() {
	suite.Run("stub body skipped and substeps still computed correctly", func() {
		p := NewPhysics(WithFixedDt(0.01), WithMaxSubsteps(4))
		pImpl := p.(*physicsImpl)
		pImpl.bodiesCount = 1
		pImpl.bodies = append(pImpl.bodies, &stubRigidBody{})

		substeps, data := p.PrepareStep(0.05)
		suite.Equal(4, substeps)
		suite.NotNil(data)
	})
}

func (suite *physicsImplTest) TestPipelineKeys() {
	suite.Run("PipelineKeys returns non-nil map", func() {
		p := NewPhysics()
		pImpl := p.(*physicsImpl)
		suite.NotNil(pImpl.PipelineKeys())
	})

	suite.Run("Lifecycle returns initialized lifecycle", func() {
		p := NewPhysics()
		pImpl := p.(*physicsImpl)
		suite.NotNil(pImpl.Lifecycle())
	})

	suite.Run("SetPipelineKey is visible in PipelineKeys", func() {
		p := NewPhysics()
		pImpl := p.(*physicsImpl)
		pImpl.SetPipelineKey("test", "test-pipeline")
		suite.Equal("test-pipeline", pImpl.PipelineKeys()["test"])
	})
}
