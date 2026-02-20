package game_object

import (
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// GameObjectBuilderOption is a functional option for configuring a GameObject during construction.
type GameObjectBuilderOption func(*gameObject)

// WithID sets the ID of the GameObject.
//
// Parameters:
//   - id: unique identifier for the GameObject
//
// Returns:
//   - GameObjectBuilderOption: functional option to set the ID
func WithID(id uint64) GameObjectBuilderOption {
	return func(obj *gameObject) {
		obj.id = id
	}
}

// WithEnabled sets whether the GameObject is enabled for rendering.
//
// Parameters:
//   - enabled: true to render the object, false to skip it
//
// Returns:
//   - GameObjectBuilderOption: functional option to set the Enabled state
func WithEnabled(enabled bool) GameObjectBuilderOption {
	return func(obj *gameObject) {
		obj.enabled.Store(enabled)
	}
}

// WithEphemeral marks the GameObject as ephemeral. Ephemeral objects are not
// persisted in the scene's registry when added via Scene.Add. The scene only
// ensures the object's animator is registered for rendering.
//
// Parameters:
//   - ephemeral: true to mark as ephemeral
//
// Returns:
//   - GameObjectBuilderOption: functional option to set the Ephemeral flag
func WithEphemeral(ephemeral bool) GameObjectBuilderOption {
	return func(obj *gameObject) {
		obj.ephemeral = ephemeral
	}
}

// WithModel sets the Model for this GameObject.
//
// Parameters:
//   - m: the Model to associate
//
// Returns:
//   - GameObjectBuilderOption: functional option to set the Model
func WithModel(m model.Model) GameObjectBuilderOption {
	return func(obj *gameObject) {
		obj.mdl = m
	}
}

// WithPosition sets the initial position of the GameObject before it is added to a Scene.
//
// Parameters:
//   - x: the x position
//   - y: the y position
//   - z: the z position
//
// Returns:
//   - GameObjectBuilderOption: functional option to set the initial position
func WithPosition(x, y, z float32) GameObjectBuilderOption {
	return func(obj *gameObject) {
		obj.initialPosition = [3]float32{x, y, z}
	}
}

// WithScale sets the initial scale of the GameObject before it is added to a Scene.
//
// Parameters:
//   - sx: the x scale factor
//   - sy: the y scale factor
//   - sz: the z scale factor
//
// Returns:
//   - GameObjectBuilderOption: functional option to set the initial scale
func WithScale(sx, sy, sz float32) GameObjectBuilderOption {
	return func(obj *gameObject) {
		obj.initialScale = [3]float32{sx, sy, sz}
	}
}

// WithRotation sets the initial rotation of the GameObject before it is added to a Scene.
//
// Parameters:
//   - rx: the x rotation angle
//   - ry: the y rotation angle
//   - rz: the z rotation angle
//
// Returns:
//   - GameObjectBuilderOption: functional option to set the initial rotation
func WithRotation(rx, ry, rz float32) GameObjectBuilderOption {
	return func(obj *gameObject) {
		obj.initialRotation = [3]float32{rx, ry, rz}
	}
}

// WithRotationSpeed sets the initial rotation speed of the GameObject before it is added to a Scene.
//
// Parameters:
//   - rx: the x rotation speed
//   - ry: the y rotation speed
//   - rz: the z rotation speed
//
// Returns:
//   - GameObjectBuilderOption: functional option to set the initial rotation speed
func WithRotationSpeed(rx, ry, rz float32) GameObjectBuilderOption {
	return func(obj *gameObject) {
		obj.initialRotationSpeed = [3]float32{rx, ry, rz}
	}
}

// WithLight attaches a Light to the GameObject. When added to a scene, the
// scene will automatically sync the light's position from the object's
// transform each frame.
//
// Parameters:
//   - l: the Light to attach
//
// Returns:
//   - GameObjectBuilderOption: functional option to set the attached light
func WithLight(l light.Light) GameObjectBuilderOption {
	return func(obj *gameObject) {
		obj.attachedLight = l
	}
}
