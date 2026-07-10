package game_object_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	light_mocks "github.com/Carmen-Shannon/oxy-go/engine/light/mocks"
	model_mocks "github.com/Carmen-Shannon/oxy-go/engine/model/mocks"
	physics_mocks "github.com/Carmen-Shannon/oxy-go/engine/physics/mocks"
	animator_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/animator/mocks"
)

func TestRunGameObjectTests(t *testing.T) {
	suite.Run(t, new(gameObjectTest))
}

type gameObjectTest struct {
	suite.Suite
	animatorMock  *animator_mocks.MockAnimator
	modelMock     *model_mocks.MockModel
	lightMock     *light_mocks.MockLight
	rigidBodyMock *physics_mocks.MockRigidBody
	obj           game_object.GameObject
}

func (suite *gameObjectTest) SetupSubTest() {
	suite.animatorMock = animator_mocks.NewMockAnimator(suite.T())
	suite.modelMock = model_mocks.NewMockModel(suite.T())
	suite.lightMock = light_mocks.NewMockLight(suite.T())
	suite.rigidBodyMock = physics_mocks.NewMockRigidBody(suite.T())
	suite.obj = game_object.NewGameObject()
}

func (suite *gameObjectTest) TestNewGameObject() {
	suite.Run("should create a new game object with provided options", func() {
		obj := game_object.NewGameObject(
			game_object.WithID(1),
			game_object.WithEnabled(true),
			game_object.WithEphemeral(true),
			game_object.WithModel(suite.modelMock),
			game_object.WithPosition(1, 2, 3),
			game_object.WithScale(1, 1, 1),
			game_object.WithRotation(0, 0, 0),
			game_object.WithRotationSpeed(0, 0, 0),
			game_object.WithLight(suite.lightMock),
			game_object.WithRigidBody(suite.rigidBodyMock),
		)
		suite.NotNil(obj)
	})
	suite.Run("should apply the contact-shadow exclusion option and keep the default animator instance id", func() {
		obj := game_object.NewGameObject(game_object.WithContactShadowExcluded(true))
		suite.True(obj.ContactShadowExcluded())
		suite.Equal(0, obj.AnimatorInstanceID())
	})
}

func (suite *gameObjectTest) TestID() {
	suite.Run("should return the object ID", func() {
		suite.Equal(uint64(0), suite.obj.ID())
	})
}

func (suite *gameObjectTest) TestSetID() {
	suite.Run("should update the object ID", func() {
		suite.obj.SetID(42)
		suite.Equal(uint64(42), suite.obj.ID())
	})
}

func (suite *gameObjectTest) TestEnabled() {
	suite.Run("should return whether the object is enabled", func() {
		suite.Equal(false, suite.obj.Enabled())
	})
}

func (suite *gameObjectTest) TestSetEnabled() {
	suite.Run("should update the enabled state", func() {
		suite.obj.SetEnabled(true)
		suite.Equal(true, suite.obj.Enabled())
	})
}

func (suite *gameObjectTest) TestEphemeral() {
	suite.Run("should return false by default", func() {
		suite.Equal(false, suite.obj.Ephemeral())
	})
}

func (suite *gameObjectTest) TestModel() {
	suite.Run("should return nil when no model is set", func() {
		suite.Nil(suite.obj.Model())
	})
}

func (suite *gameObjectTest) TestSetModel() {
	suite.Run("should update the model", func() {
		suite.obj.SetModel(suite.modelMock)
		suite.Equal(suite.modelMock, suite.obj.Model())
	})
}

func (suite *gameObjectTest) TestAnimator() {
	suite.Run("should return nil when no animator is set", func() {
		suite.Nil(suite.obj.Animator())
	})
}

func (suite *gameObjectTest) TestSetAnimator() {
	suite.Run("should update the animator", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.Equal(suite.animatorMock, suite.obj.Animator())
	})
}

func (suite *gameObjectTest) TestAnimatorInstanceID() {
	suite.Run("should return the animator instance ID", func() {
		suite.Equal(0, suite.obj.AnimatorInstanceID())
	})
}

func (suite *gameObjectTest) TestSetAnimatorInstanceID() {
	suite.Run("should update the animator instance ID", func() {
		suite.obj.SetAnimatorInstanceID(5)
		suite.Equal(5, suite.obj.AnimatorInstanceID())
	})
}

func (suite *gameObjectTest) TestLight() {
	suite.Run("should return nil when no light is attached", func() {
		suite.Nil(suite.obj.Light())
	})
}

func (suite *gameObjectTest) TestSetLight() {
	suite.Run("should update the attached light", func() {
		suite.obj.SetLight(suite.lightMock)
		suite.Equal(suite.lightMock, suite.obj.Light())
	})
}

func (suite *gameObjectTest) TestRigidBody() {
	suite.Run("should return nil when no rigid body is attached", func() {
		suite.Nil(suite.obj.RigidBody())
	})
}

func (suite *gameObjectTest) TestSetRigidBody() {
	suite.Run("should update the attached rigid body", func() {
		suite.obj.SetRigidBody(suite.rigidBodyMock)
		suite.Equal(suite.rigidBodyMock, suite.obj.RigidBody())
	})
}

func (suite *gameObjectTest) TestContactShadowExcluded() {
	suite.Run("should return false by default", func() {
		suite.False(suite.obj.ContactShadowExcluded())
	})

	suite.Run("should update the contact-shadow exclusion flag", func() {
		suite.obj.SetContactShadowExcluded(true)
		suite.True(suite.obj.ContactShadowExcluded())

		suite.obj.SetContactShadowExcluded(false)
		suite.False(suite.obj.ContactShadowExcluded())
	})

	suite.Run("should update the flag without querying the animator", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.obj.SetAnimatorInstanceID(7)

		suite.obj.SetContactShadowExcluded(true)
		suite.True(suite.obj.ContactShadowExcluded())

		suite.obj.SetContactShadowExcluded(false)
		suite.False(suite.obj.ContactShadowExcluded())
	})
}

func (suite *gameObjectTest) TestPosition() {
	suite.Run("should return initial position when no animator is set", func() {
		obj := game_object.NewGameObject(game_object.WithPosition(1, 2, 3))
		x, y, z := obj.Position()
		suite.Equal(float32(1), x)
		suite.Equal(float32(2), y)
		suite.Equal(float32(3), z)
	})
	suite.Run("should return position from animator when animator is set", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.obj.SetAnimatorInstanceID(0)
		suite.animatorMock.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{4, 5, 6}, [3]float32{1, 1, 1}).Once()
		x, y, z := suite.obj.Position()
		suite.Equal(float32(4), x)
		suite.Equal(float32(5), y)
		suite.Equal(float32(6), z)
	})
}

func (suite *gameObjectTest) TestRotation() {
	suite.Run("should return initial rotation when no animator is set", func() {
		obj := game_object.NewGameObject(game_object.WithRotation(1, 2, 3))
		rx, ry, rz := obj.Rotation()
		suite.Equal(float32(1), rx)
		suite.Equal(float32(2), ry)
		suite.Equal(float32(3), rz)
	})
	suite.Run("should return rotation from animator when animator is set", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.obj.SetAnimatorInstanceID(0)
		suite.animatorMock.EXPECT().InstanceRotation(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{7, 8, 9}).Once()
		rx, ry, rz := suite.obj.Rotation()
		suite.Equal(float32(7), rx)
		suite.Equal(float32(8), ry)
		suite.Equal(float32(9), rz)
	})
}

func (suite *gameObjectTest) TestRotationSpeed() {
	suite.Run("should return initial rotation speed when no animator is set", func() {
		obj := game_object.NewGameObject(game_object.WithRotationSpeed(1, 2, 3))
		rx, ry, rz := obj.RotationSpeed()
		suite.Equal(float32(1), rx)
		suite.Equal(float32(2), ry)
		suite.Equal(float32(3), rz)
	})
	suite.Run("should return rotation speed from animator when animator is set", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.obj.SetAnimatorInstanceID(0)
		suite.animatorMock.EXPECT().InstanceRotation(uint32(0)).Return([3]float32{4, 5, 6}, [3]float32{0, 0, 0}).Once()
		rx, ry, rz := suite.obj.RotationSpeed()
		suite.Equal(float32(4), rx)
		suite.Equal(float32(5), ry)
		suite.Equal(float32(6), rz)
	})
}

func (suite *gameObjectTest) TestScale() {
	suite.Run("should return initial scale when no animator is set", func() {
		sx, sy, sz := suite.obj.Scale()
		suite.Equal(float32(1), sx)
		suite.Equal(float32(1), sy)
		suite.Equal(float32(1), sz)
	})
	suite.Run("should return scale from animator when animator is set", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.obj.SetAnimatorInstanceID(0)
		suite.animatorMock.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{2, 3, 4}).Once()
		sx, sy, sz := suite.obj.Scale()
		suite.Equal(float32(2), sx)
		suite.Equal(float32(3), sy)
		suite.Equal(float32(4), sz)
	})
}

func (suite *gameObjectTest) TestTransformData() {
	suite.Run("should return initial transform data when no animator is set", func() {
		obj := game_object.NewGameObject(
			game_object.WithPosition(1, 2, 3),
			game_object.WithScale(4, 5, 6),
			game_object.WithRotation(7, 8, 9),
			game_object.WithRotationSpeed(10, 11, 12),
		)
		pos, scale, rot, rotSpeed := obj.TransformData()
		suite.Equal([3]float32{1, 2, 3}, pos)
		suite.Equal([3]float32{4, 5, 6}, scale)
		suite.Equal([3]float32{7, 8, 9}, rot)
		suite.Equal([3]float32{10, 11, 12}, rotSpeed)
	})
	suite.Run("should return transform data from animator when animator is set", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.obj.SetAnimatorInstanceID(0)
		suite.animatorMock.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{1, 2, 3}, [3]float32{4, 5, 6}).Once()
		suite.animatorMock.EXPECT().InstanceRotation(uint32(0)).Return([3]float32{7, 8, 9}, [3]float32{10, 11, 12}).Once()
		pos, scale, rot, rotSpeed := suite.obj.TransformData()
		suite.Equal([3]float32{1, 2, 3}, pos)
		suite.Equal([3]float32{4, 5, 6}, scale)
		suite.Equal([3]float32{10, 11, 12}, rot)
		suite.Equal([3]float32{7, 8, 9}, rotSpeed)
	})
}

func (suite *gameObjectTest) TestSetPosition() {
	suite.Run("should update initial position when no animator is set", func() {
		suite.obj.SetPosition(1, 2, 3)
		x, y, z := suite.obj.Position()
		suite.Equal(float32(1), x)
		suite.Equal(float32(2), y)
		suite.Equal(float32(3), z)
	})
	suite.Run("should set position via animator when animator is set", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.obj.SetAnimatorInstanceID(0)
		suite.animatorMock.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{1, 1, 1}).Once()
		suite.animatorMock.EXPECT().SetInstanceTransform(uint32(0), [3]float32{4, 5, 6}, [3]float32{1, 1, 1}).Return().Once()
		suite.obj.SetPosition(4, 5, 6)
	})
}

func (suite *gameObjectTest) TestSetRotation() {
	suite.Run("should update initial rotation when no animator is set", func() {
		suite.obj.SetRotation(1, 2, 3)
		rx, ry, rz := suite.obj.Rotation()
		suite.Equal(float32(1), rx)
		suite.Equal(float32(2), ry)
		suite.Equal(float32(3), rz)
	})
	suite.Run("should set rotation via animator when animator is set", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.obj.SetAnimatorInstanceID(0)
		suite.animatorMock.EXPECT().InstanceRotation(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{0, 0, 0}).Once()
		suite.animatorMock.EXPECT().SetInstanceRotation(uint32(0), [3]float32{0, 0, 0}, [3]float32{1, 2, 3}).Return().Once()
		suite.obj.SetRotation(1, 2, 3)
	})
}

func (suite *gameObjectTest) TestSetRotationSpeed() {
	suite.Run("should update initial rotation speed when no animator is set", func() {
		suite.obj.SetRotationSpeed(1, 2, 3)
		rx, ry, rz := suite.obj.RotationSpeed()
		suite.Equal(float32(1), rx)
		suite.Equal(float32(2), ry)
		suite.Equal(float32(3), rz)
	})
	suite.Run("should set rotation speed via animator when animator is set", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.obj.SetAnimatorInstanceID(0)
		suite.animatorMock.EXPECT().InstanceRotation(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{0, 0, 0}).Once()
		suite.animatorMock.EXPECT().SetInstanceRotation(uint32(0), [3]float32{1, 2, 3}, [3]float32{0, 0, 0}).Return().Once()
		suite.obj.SetRotationSpeed(1, 2, 3)
	})
}

func (suite *gameObjectTest) TestSetScale() {
	suite.Run("should update initial scale when no animator is set", func() {
		suite.obj.SetScale(2, 3, 4)
		sx, sy, sz := suite.obj.Scale()
		suite.Equal(float32(2), sx)
		suite.Equal(float32(3), sy)
		suite.Equal(float32(4), sz)
	})
	suite.Run("should set scale via animator when animator is set", func() {
		suite.obj.SetAnimator(suite.animatorMock)
		suite.obj.SetAnimatorInstanceID(0)
		suite.animatorMock.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{1, 1, 1}).Once()
		suite.animatorMock.EXPECT().SetInstanceTransform(uint32(0), [3]float32{0, 0, 0}, [3]float32{2, 3, 4}).Return().Once()
		suite.obj.SetScale(2, 3, 4)
	})
}
