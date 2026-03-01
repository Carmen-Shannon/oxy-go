package game_object_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	animatormocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/animator"
	lightmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/light"
	modelmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/model"
	physicsmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/physics"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

type gameObjectTest struct {
	suite.Suite
}

func TestGameObject(t *testing.T) {
	suite.Run(t, new(gameObjectTest))
}

func (suite *gameObjectTest) TestNewGameObject() {
	suite.Run("id defaults to zero", func() {
		obj := game_object.NewGameObject()
		suite.Equal(uint64(0), obj.ID())
	})

	suite.Run("enabled defaults to false", func() {
		obj := game_object.NewGameObject()
		suite.False(obj.Enabled())
	})

	suite.Run("ephemeral defaults to false", func() {
		obj := game_object.NewGameObject()
		suite.False(obj.Ephemeral())
	})

	suite.Run("model defaults to nil", func() {
		obj := game_object.NewGameObject()
		suite.Nil(obj.Model())
	})

	suite.Run("animator defaults to nil", func() {
		obj := game_object.NewGameObject()
		suite.Nil(obj.Animator())
	})

	suite.Run("animator instance id defaults to zero", func() {
		obj := game_object.NewGameObject()
		suite.Equal(0, obj.AnimatorInstanceID())
	})

	suite.Run("light defaults to nil", func() {
		obj := game_object.NewGameObject()
		suite.Nil(obj.Light())
	})

	suite.Run("rigid body defaults to nil", func() {
		obj := game_object.NewGameObject()
		suite.Nil(obj.RigidBody())
	})

	suite.Run("position defaults to zero", func() {
		obj := game_object.NewGameObject()
		x, y, z := obj.Position()
		suite.Equal(float32(0), x)
		suite.Equal(float32(0), y)
		suite.Equal(float32(0), z)
	})

	suite.Run("scale defaults to one", func() {
		obj := game_object.NewGameObject()
		sx, sy, sz := obj.Scale()
		suite.Equal(float32(1), sx)
		suite.Equal(float32(1), sy)
		suite.Equal(float32(1), sz)
	})

	suite.Run("rotation defaults to zero", func() {
		obj := game_object.NewGameObject()
		rx, ry, rz := obj.Rotation()
		suite.Equal(float32(0), rx)
		suite.Equal(float32(0), ry)
		suite.Equal(float32(0), rz)
	})

	suite.Run("rotation speed defaults to zero", func() {
		obj := game_object.NewGameObject()
		rx, ry, rz := obj.RotationSpeed()
		suite.Equal(float32(0), rx)
		suite.Equal(float32(0), ry)
		suite.Equal(float32(0), rz)
	})
}

func (suite *gameObjectTest) TestNewGameObjectWithOptions() {
	suite.Run("WithID sets the id", func() {
		obj := game_object.NewGameObject(game_object.WithID(42))
		suite.Equal(uint64(42), obj.ID())
	})

	suite.Run("WithEnabled sets enabled to true", func() {
		obj := game_object.NewGameObject(game_object.WithEnabled(true))
		suite.True(obj.Enabled())
	})

	suite.Run("WithEphemeral sets ephemeral to true", func() {
		obj := game_object.NewGameObject(game_object.WithEphemeral(true))
		suite.True(obj.Ephemeral())
	})

	suite.Run("WithModel sets the model", func() {
		m := &modelmocks.MockModel{}
		obj := game_object.NewGameObject(game_object.WithModel(m))
		suite.Equal(m, obj.Model())
	})

	suite.Run("WithPosition sets the initial position", func() {
		obj := game_object.NewGameObject(game_object.WithPosition(1, 2, 3))
		x, y, z := obj.Position()
		suite.Equal(float32(1), x)
		suite.Equal(float32(2), y)
		suite.Equal(float32(3), z)
	})

	suite.Run("WithScale sets the initial scale", func() {
		obj := game_object.NewGameObject(game_object.WithScale(2, 3, 4))
		sx, sy, sz := obj.Scale()
		suite.Equal(float32(2), sx)
		suite.Equal(float32(3), sy)
		suite.Equal(float32(4), sz)
	})

	suite.Run("WithRotation sets the initial rotation", func() {
		obj := game_object.NewGameObject(game_object.WithRotation(0.1, 0.2, 0.3))
		rx, ry, rz := obj.Rotation()
		suite.Equal(float32(0.1), rx)
		suite.Equal(float32(0.2), ry)
		suite.Equal(float32(0.3), rz)
	})

	suite.Run("WithRotationSpeed sets the initial rotation speed", func() {
		obj := game_object.NewGameObject(game_object.WithRotationSpeed(0.5, 0.6, 0.7))
		rx, ry, rz := obj.RotationSpeed()
		suite.Equal(float32(0.5), rx)
		suite.Equal(float32(0.6), ry)
		suite.Equal(float32(0.7), rz)
	})

	suite.Run("WithLight sets the attached light", func() {
		l := &lightmocks.MockLight{}
		obj := game_object.NewGameObject(game_object.WithLight(l))
		suite.Equal(l, obj.Light())
	})

	suite.Run("WithRigidBody sets the attached rigid body", func() {
		rb := &physicsmocks.MockRigidBody{}
		obj := game_object.NewGameObject(game_object.WithRigidBody(rb))
		suite.Equal(rb, obj.RigidBody())
	})

	suite.Run("combined options apply correctly", func() {
		m := &modelmocks.MockModel{}
		l := &lightmocks.MockLight{}
		obj := game_object.NewGameObject(
			game_object.WithID(99),
			game_object.WithEnabled(true),
			game_object.WithEphemeral(true),
			game_object.WithModel(m),
			game_object.WithLight(l),
			game_object.WithPosition(10, 20, 30),
			game_object.WithScale(2, 2, 2),
			game_object.WithRotation(0.1, 0.2, 0.3),
			game_object.WithRotationSpeed(1, 2, 3),
		)
		suite.Equal(uint64(99), obj.ID())
		suite.True(obj.Enabled())
		suite.True(obj.Ephemeral())
		suite.Equal(m, obj.Model())
		suite.Equal(l, obj.Light())

		x, y, z := obj.Position()
		suite.Equal(float32(10), x)
		suite.Equal(float32(20), y)
		suite.Equal(float32(30), z)

		sx, sy, sz := obj.Scale()
		suite.Equal(float32(2), sx)
		suite.Equal(float32(2), sy)
		suite.Equal(float32(2), sz)

		rx, ry, rz := obj.Rotation()
		suite.Equal(float32(0.1), rx)
		suite.Equal(float32(0.2), ry)
		suite.Equal(float32(0.3), rz)

		rsx, rsy, rsz := obj.RotationSpeed()
		suite.Equal(float32(1), rsx)
		suite.Equal(float32(2), rsy)
		suite.Equal(float32(3), rsz)
	})
}

func (suite *gameObjectTest) TestSettersAndGetters() {
	suite.Run("SetID and ID", func() {
		obj := game_object.NewGameObject()
		obj.SetID(123)
		suite.Equal(uint64(123), obj.ID())
	})

	suite.Run("SetEnabled and Enabled", func() {
		obj := game_object.NewGameObject()
		obj.SetEnabled(true)
		suite.True(obj.Enabled())
		obj.SetEnabled(false)
		suite.False(obj.Enabled())
	})

	suite.Run("SetModel and Model", func() {
		obj := game_object.NewGameObject()
		m := &modelmocks.MockModel{}
		obj.SetModel(m)
		suite.Equal(m, obj.Model())
	})

	suite.Run("SetModel to nil clears model", func() {
		m := &modelmocks.MockModel{}
		obj := game_object.NewGameObject(game_object.WithModel(m))
		obj.SetModel(nil)
		suite.Nil(obj.Model())
	})

	suite.Run("SetAnimator and Animator", func() {
		obj := game_object.NewGameObject()
		a := &animatormocks.MockAnimator{}
		obj.SetAnimator(a)
		suite.Equal(a, obj.Animator())
	})

	suite.Run("SetAnimatorInstanceID and AnimatorInstanceID", func() {
		obj := game_object.NewGameObject()
		obj.SetAnimatorInstanceID(7)
		suite.Equal(7, obj.AnimatorInstanceID())
	})

	suite.Run("SetLight and Light", func() {
		obj := game_object.NewGameObject()
		l := &lightmocks.MockLight{}
		obj.SetLight(l)
		suite.Equal(l, obj.Light())
	})

	suite.Run("SetLight to nil detaches light", func() {
		l := &lightmocks.MockLight{}
		obj := game_object.NewGameObject(game_object.WithLight(l))
		obj.SetLight(nil)
		suite.Nil(obj.Light())
	})

	suite.Run("SetRigidBody and RigidBody", func() {
		obj := game_object.NewGameObject()
		rb := &physicsmocks.MockRigidBody{}
		obj.SetRigidBody(rb)
		suite.Equal(rb, obj.RigidBody())
	})

	suite.Run("SetRigidBody to nil detaches rigid body", func() {
		rb := &physicsmocks.MockRigidBody{}
		obj := game_object.NewGameObject(game_object.WithRigidBody(rb))
		obj.SetRigidBody(nil)
		suite.Nil(obj.RigidBody())
	})
}

func (suite *gameObjectTest) TestTransformWithoutAnimator() {
	suite.Run("SetPosition updates initial position", func() {
		obj := game_object.NewGameObject()
		obj.SetPosition(5, 6, 7)
		x, y, z := obj.Position()
		suite.Equal(float32(5), x)
		suite.Equal(float32(6), y)
		suite.Equal(float32(7), z)
	})

	suite.Run("SetScale updates initial scale", func() {
		obj := game_object.NewGameObject()
		obj.SetScale(3, 4, 5)
		sx, sy, sz := obj.Scale()
		suite.Equal(float32(3), sx)
		suite.Equal(float32(4), sy)
		suite.Equal(float32(5), sz)
	})

	suite.Run("SetRotation updates initial rotation", func() {
		obj := game_object.NewGameObject()
		obj.SetRotation(0.4, 0.5, 0.6)
		rx, ry, rz := obj.Rotation()
		suite.Equal(float32(0.4), rx)
		suite.Equal(float32(0.5), ry)
		suite.Equal(float32(0.6), rz)
	})

	suite.Run("SetRotationSpeed updates initial rotation speed", func() {
		obj := game_object.NewGameObject()
		obj.SetRotationSpeed(1.1, 2.2, 3.3)
		rx, ry, rz := obj.RotationSpeed()
		suite.Equal(float32(1.1), rx)
		suite.Equal(float32(2.2), ry)
		suite.Equal(float32(3.3), rz)
	})

	suite.Run("TransformData returns all initial values", func() {
		obj := game_object.NewGameObject(
			game_object.WithPosition(1, 2, 3),
			game_object.WithScale(4, 5, 6),
			game_object.WithRotation(0.1, 0.2, 0.3),
			game_object.WithRotationSpeed(0.4, 0.5, 0.6),
		)
		pos, scale, rot, rotSpeed := obj.TransformData()
		suite.Equal([3]float32{1, 2, 3}, pos)
		suite.Equal([3]float32{4, 5, 6}, scale)
		suite.Equal([3]float32{0.1, 0.2, 0.3}, rot)
		suite.Equal([3]float32{0.4, 0.5, 0.6}, rotSpeed)
	})

	suite.Run("TransformData defaults match individual getters", func() {
		obj := game_object.NewGameObject()
		pos, scale, rot, rotSpeed := obj.TransformData()
		suite.Equal([3]float32{0, 0, 0}, pos)
		suite.Equal([3]float32{1, 1, 1}, scale)
		suite.Equal([3]float32{0, 0, 0}, rot)
		suite.Equal([3]float32{0, 0, 0}, rotSpeed)
	})
}

func (suite *gameObjectTest) TestTransformWithAnimator() {
	suite.Run("Position delegates to animator InstanceTransform", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceTransform(uint32(0)).Return(
			[3]float32{10, 20, 30},
			[3]float32{1, 1, 1},
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		x, y, z := obj.Position()
		suite.Equal(float32(10), x)
		suite.Equal(float32(20), y)
		suite.Equal(float32(30), z)
		a.AssertExpectations(suite.T())
	})

	suite.Run("Scale delegates to animator InstanceTransform", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceTransform(uint32(0)).Return(
			[3]float32{0, 0, 0},
			[3]float32{2, 3, 4},
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		sx, sy, sz := obj.Scale()
		suite.Equal(float32(2), sx)
		suite.Equal(float32(3), sy)
		suite.Equal(float32(4), sz)
		a.AssertExpectations(suite.T())
	})

	suite.Run("Rotation delegates to animator InstanceRotation", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceRotation(uint32(0)).Return(
			[3]float32{0, 0, 0},       // rotSpeed
			[3]float32{0.1, 0.2, 0.3}, // rot
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		rx, ry, rz := obj.Rotation()
		suite.Equal(float32(0.1), rx)
		suite.Equal(float32(0.2), ry)
		suite.Equal(float32(0.3), rz)
		a.AssertExpectations(suite.T())
	})

	suite.Run("RotationSpeed delegates to animator InstanceRotation", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceRotation(uint32(0)).Return(
			[3]float32{1.1, 2.2, 3.3}, // rotSpeed
			[3]float32{0, 0, 0},       // rot
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		rx, ry, rz := obj.RotationSpeed()
		suite.Equal(float32(1.1), rx)
		suite.Equal(float32(2.2), ry)
		suite.Equal(float32(3.3), rz)
		a.AssertExpectations(suite.T())
	})

	suite.Run("TransformData delegates to both InstanceTransform and InstanceRotation", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceTransform(uint32(3)).Return(
			[3]float32{10, 20, 30},
			[3]float32{2, 2, 2},
		).Once()
		a.EXPECT().InstanceRotation(uint32(3)).Return(
			[3]float32{0.5, 0.6, 0.7}, // rotSpeed
			[3]float32{0.1, 0.2, 0.3}, // rot
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		obj.SetAnimatorInstanceID(3)
		pos, scale, rot, rotSpeed := obj.TransformData()
		suite.Equal([3]float32{10, 20, 30}, pos)
		suite.Equal([3]float32{2, 2, 2}, scale)
		suite.Equal([3]float32{0.1, 0.2, 0.3}, rot)
		suite.Equal([3]float32{0.5, 0.6, 0.7}, rotSpeed)
		a.AssertExpectations(suite.T())
	})

	suite.Run("uses correct animator instance id", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceTransform(uint32(5)).Return(
			[3]float32{99, 88, 77},
			[3]float32{1, 1, 1},
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		obj.SetAnimatorInstanceID(5)
		x, y, z := obj.Position()
		suite.Equal(float32(99), x)
		suite.Equal(float32(88), y)
		suite.Equal(float32(77), z)
		a.AssertExpectations(suite.T())
	})
}

func (suite *gameObjectTest) TestSetTransformWithAnimator() {
	suite.Run("SetPosition preserves current scale", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceTransform(uint32(0)).Return(
			[3]float32{0, 0, 0},
			[3]float32{2, 3, 4},
		).Once()
		a.EXPECT().SetInstanceTransform(
			uint32(0),
			[3]float32{10, 20, 30},
			[3]float32{2, 3, 4},
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		obj.SetPosition(10, 20, 30)
		a.AssertExpectations(suite.T())
	})

	suite.Run("SetScale preserves current position", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceTransform(uint32(0)).Return(
			[3]float32{5, 6, 7},
			[3]float32{1, 1, 1},
		).Once()
		a.EXPECT().SetInstanceTransform(
			uint32(0),
			[3]float32{5, 6, 7},
			[3]float32{3, 3, 3},
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		obj.SetScale(3, 3, 3)
		a.AssertExpectations(suite.T())
	})

	suite.Run("SetRotation preserves current rotation speed", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceRotation(uint32(0)).Return(
			[3]float32{1.0, 2.0, 3.0}, // rotSpeed
			[3]float32{0, 0, 0},       // rot (old)
		).Once()
		a.EXPECT().SetInstanceRotation(
			uint32(0),
			[3]float32{1.0, 2.0, 3.0},
			[3]float32{0.5, 0.6, 0.7},
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		obj.SetRotation(0.5, 0.6, 0.7)
		a.AssertExpectations(suite.T())
	})

	suite.Run("SetRotationSpeed preserves current rotation", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceRotation(uint32(0)).Return(
			[3]float32{0, 0, 0},       // rotSpeed (old)
			[3]float32{0.1, 0.2, 0.3}, // rot
		).Once()
		a.EXPECT().SetInstanceRotation(
			uint32(0),
			[3]float32{9, 8, 7},
			[3]float32{0.1, 0.2, 0.3},
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		obj.SetRotationSpeed(9, 8, 7)
		a.AssertExpectations(suite.T())
	})

	suite.Run("SetPosition uses correct instance id", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceTransform(uint32(4)).Return(
			[3]float32{0, 0, 0},
			[3]float32{1, 1, 1},
		).Once()
		a.EXPECT().SetInstanceTransform(
			uint32(4),
			[3]float32{1, 2, 3},
			[3]float32{1, 1, 1},
		).Once()

		obj := game_object.NewGameObject()
		obj.SetAnimator(a)
		obj.SetAnimatorInstanceID(4)
		obj.SetPosition(1, 2, 3)
		a.AssertExpectations(suite.T())
	})
}

func (suite *gameObjectTest) TestAnimatorTransition() {
	suite.Run("initial values are used before animator is set", func() {
		obj := game_object.NewGameObject(
			game_object.WithPosition(1, 2, 3),
			game_object.WithScale(4, 5, 6),
		)
		x, y, z := obj.Position()
		suite.Equal(float32(1), x)
		suite.Equal(float32(2), y)
		suite.Equal(float32(3), z)
	})

	suite.Run("animator values are used after animator is set", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceTransform(mock.Anything).Return(
			[3]float32{99, 88, 77},
			[3]float32{5, 5, 5},
		).Maybe()

		obj := game_object.NewGameObject(game_object.WithPosition(1, 2, 3))
		obj.SetAnimator(a)
		x, y, z := obj.Position()
		suite.Equal(float32(99), x)
		suite.Equal(float32(88), y)
		suite.Equal(float32(77), z)
	})

	suite.Run("clearing animator falls back to initial values", func() {
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceTransform(mock.Anything).Return(
			[3]float32{99, 88, 77},
			[3]float32{1, 1, 1},
		).Maybe()

		obj := game_object.NewGameObject(game_object.WithPosition(1, 2, 3))
		obj.SetAnimator(a)

		// Now clear the animator
		obj.SetAnimator(nil)
		x, y, z := obj.Position()
		suite.Equal(float32(1), x)
		suite.Equal(float32(2), y)
		suite.Equal(float32(3), z)
	})

	suite.Run("SetPosition without animator stores to initial, then animator overrides", func() {
		obj := game_object.NewGameObject()
		obj.SetPosition(10, 20, 30)

		// Confirm initial stored
		x, y, z := obj.Position()
		suite.Equal(float32(10), x)
		suite.Equal(float32(20), y)
		suite.Equal(float32(30), z)

		// Now attach animator — position comes from animator
		a := &animatormocks.MockAnimator{}
		a.EXPECT().InstanceTransform(mock.Anything).Return(
			[3]float32{0, 0, 0},
			[3]float32{1, 1, 1},
		).Maybe()
		obj.SetAnimator(a)
		x, y, z = obj.Position()
		suite.Equal(float32(0), x)
		suite.Equal(float32(0), y)
		suite.Equal(float32(0), z)
	})
}
