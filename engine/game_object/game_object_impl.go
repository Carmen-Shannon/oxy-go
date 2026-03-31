package game_object

import (
	"sync/atomic"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
)

type gameObject struct {
	id                 uint64
	enabled            atomic.Bool
	ephemeral          bool
	mdl                model.Model
	animator           animator.Animator
	animatorInstanceID int
	attachedLight      light.Light
	rigidBody          physics.RigidBody

	// initial transform state used before the object is added to a Scene
	initialPosition      [3]float32
	initialScale         [3]float32
	initialRotation      [3]float32
	initialRotationSpeed [3]float32
}
