package game_object

import (
	"sync/atomic"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
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

	// initial transform state used before the object is added to a Scene
	initialPosition      [3]float32
	initialScale         [3]float32
	initialRotation      [3]float32
	initialRotationSpeed [3]float32
}

// GameObject defines the interface for a scene entity bound to an Animator instance.
// Position, rotation, and scale are derived from the Animator's internal arrays
// via the animatorInstanceID, eliminating per-object data duplication.
type GameObject interface {
	// ID returns the object's unique identifier.
	//
	// Returns:
	//   - uint64: the object ID
	ID() uint64

	// Enabled returns whether this object is enabled for rendering.
	//
	// Returns:
	//   - bool: true if enabled
	Enabled() bool

	// Ephemeral returns whether this object is ephemeral.
	// Ephemeral objects are not persisted in the scene's registry when added.
	//
	// Returns:
	//   - bool: true if ephemeral
	Ephemeral() bool

	// Model returns the Model associated with this object, or nil if not set.
	//
	// Returns:
	//   - model.Model: the associated model or nil
	Model() model.Model

	// Animator returns the Animator associated with this object.
	//
	// Returns:
	//   - animator.Animator: the associated Animator, or nil
	Animator() animator.Animator

	// AnimatorInstanceID returns the instance index within the Animator.
	//
	// Returns:
	//   - int: the instance index, or -1 if unset
	AnimatorInstanceID() int

	// Position derives the instance's current position from the Animator.
	// Returns zeros if no Animator is set.
	//
	// Returns:
	//   - x, y, z: position components
	Position() (x, y, z float32)

	// Rotation derives the instance's current rotation from the Animator.
	// Returns zeros if no Animator is set or on skeletal backends.
	//
	// Returns:
	//   - rx, ry, rz: rotation angles
	Rotation() (rx, ry, rz float32)

	// RotationSpeed derives the instance's current rotation speed from the Animator.
	// Returns zeros if no Animator is set or on skeletal backends.
	//
	// Returns:
	//   - rx, ry, rz: rotation speed values
	RotationSpeed() (rx, ry, rz float32)

	// Scale derives the instance's current scale from the Animator.
	// Returns zeros if no Animator is set.
	//
	// Returns:
	//   - sx, sy, sz: scale components
	Scale() (sx, sy, sz float32)

	// TransformData reads all transform data from the Animator in a single pair of calls.
	// Returns zeros if no Animator is set.
	//
	// Returns:
	//   - pos: position as [3]float32 (x, y, z)
	//   - scale: scale as [3]float32 (x, y, z)
	//   - rot: rotation as [3]float32 (rx, ry, rz)
	//   - rotSpeed: rotation speed as [3]float32 (rx, ry, rz)
	TransformData() (pos, scale, rot, rotSpeed [3]float32)

	// SetID sets the object's unique identifier.
	//
	// Parameters:
	//   - id: the ID to assign
	SetID(id uint64)

	// SetEnabled sets whether the object is enabled for rendering.
	//
	// Parameters:
	//   - enabled: true to enable
	SetEnabled(enabled bool)

	// SetModel assigns a Model to this object.
	//
	// Parameters:
	//   - m: the Model to associate
	SetModel(m model.Model)

	// SetAnimator sets the Animator associated with this object.
	//
	// Parameters:
	//   - anim: the Animator to associate
	SetAnimator(anim animator.Animator)

	// SetAnimatorInstanceID sets the instance index within the Animator.
	//
	// Parameters:
	//   - instanceID: the instance index
	SetAnimatorInstanceID(instanceID int)

	// SetPosition updates the instance's position via the Animator, preserving current scale.
	//
	// Parameters:
	//   - x, y, z: new position components
	SetPosition(x, y, z float32)

	// SetRotation updates the instance's rotation via the Animator, preserving current rotation speed.
	//
	// Parameters:
	//   - rx, ry, rz: new rotation angles
	SetRotation(rx, ry, rz float32)

	// SetRotationSpeed updates the instance's rotation speed via the Animator, preserving current rotation.
	//
	// Parameters:
	//   - rx, ry, rz: new rotation speed values
	SetRotationSpeed(rx, ry, rz float32)

	// SetScale updates the instance's scale via the Animator, preserving current position.
	//
	// Parameters:
	//   - sx, sy, sz: new scale factors
	SetScale(sx, sy, sz float32)

	// Light returns the Light attached to this object, or nil if none is set.
	//
	// Returns:
	//   - light.Light: the attached light or nil
	Light() light.Light

	// SetLight attaches a Light to this object. When the object is added to a
	// scene, the scene will automatically sync the light's position from the
	// object's transform each frame. Pass nil to detach.
	//
	// Parameters:
	//   - l: the Light to attach, or nil to detach
	SetLight(l light.Light)
}

var _ GameObject = &gameObject{}

// NewGameObject creates a new GameObject configured with the given options.
//
// Parameters:
//   - options: functional options to configure the object
//
// Returns:
//   - GameObject: the newly created object
func NewGameObject(options ...GameObjectBuilderOption) GameObject {
	obj := &gameObject{
		initialScale: [3]float32{1, 1, 1},
	}
	for _, option := range options {
		option(obj)
	}
	return obj
}

func (g *gameObject) ID() uint64 {
	return g.id
}

func (g *gameObject) Enabled() bool {
	return g.enabled.Load()
}

func (g *gameObject) Ephemeral() bool {
	return g.ephemeral
}

func (g *gameObject) Model() model.Model {
	return g.mdl
}

func (g *gameObject) Animator() animator.Animator {
	return g.animator
}

func (g *gameObject) AnimatorInstanceID() int {
	return g.animatorInstanceID
}

func (g *gameObject) Position() (x, y, z float32) {
	if g.animator == nil {
		return g.initialPosition[0], g.initialPosition[1], g.initialPosition[2]
	}
	pos, _ := g.animator.InstanceTransform(uint32(g.animatorInstanceID))
	return pos[0], pos[1], pos[2]
}

func (g *gameObject) Rotation() (rx, ry, rz float32) {
	if g.animator == nil {
		return g.initialRotation[0], g.initialRotation[1], g.initialRotation[2]
	}
	_, rot := g.animator.InstanceRotation(uint32(g.animatorInstanceID))
	return rot[0], rot[1], rot[2]
}

func (g *gameObject) RotationSpeed() (rx, ry, rz float32) {
	if g.animator == nil {
		return g.initialRotationSpeed[0], g.initialRotationSpeed[1], g.initialRotationSpeed[2]
	}
	rotSpeed, _ := g.animator.InstanceRotation(uint32(g.animatorInstanceID))
	return rotSpeed[0], rotSpeed[1], rotSpeed[2]
}

func (g *gameObject) Scale() (sx, sy, sz float32) {
	if g.animator == nil {
		return g.initialScale[0], g.initialScale[1], g.initialScale[2]
	}
	_, scale := g.animator.InstanceTransform(uint32(g.animatorInstanceID))
	return scale[0], scale[1], scale[2]
}

func (g *gameObject) TransformData() (pos, scale, rot, rotSpeed [3]float32) {
	if g.animator == nil {
		return g.initialPosition, g.initialScale, g.initialRotation, g.initialRotationSpeed
	}
	pos, scale = g.animator.InstanceTransform(uint32(g.animatorInstanceID))
	rotSpeed, rot = g.animator.InstanceRotation(uint32(g.animatorInstanceID))
	return
}

func (g *gameObject) SetID(id uint64) {
	g.id = id
}

func (g *gameObject) SetEnabled(enabled bool) {
	g.enabled.Store(enabled)
}

func (g *gameObject) SetModel(m model.Model) {
	g.mdl = m
}

func (g *gameObject) SetAnimator(anim animator.Animator) {
	g.animator = anim
}

func (g *gameObject) SetAnimatorInstanceID(instanceID int) {
	g.animatorInstanceID = instanceID
}

func (g *gameObject) SetPosition(x, y, z float32) {
	if g.animator == nil {
		g.initialPosition = [3]float32{x, y, z}
		return
	}
	_, scale := g.animator.InstanceTransform(uint32(g.animatorInstanceID))
	g.animator.SetInstanceTransform(uint32(g.animatorInstanceID), [3]float32{x, y, z}, scale)
}

func (g *gameObject) SetRotation(rx, ry, rz float32) {
	if g.animator == nil {
		g.initialRotation = [3]float32{rx, ry, rz}
		return
	}
	rotSpeed, _ := g.animator.InstanceRotation(uint32(g.animatorInstanceID))
	g.animator.SetInstanceRotation(uint32(g.animatorInstanceID), rotSpeed, [3]float32{rx, ry, rz})
}

func (g *gameObject) SetRotationSpeed(rx, ry, rz float32) {
	if g.animator == nil {
		g.initialRotationSpeed = [3]float32{rx, ry, rz}
		return
	}
	_, rot := g.animator.InstanceRotation(uint32(g.animatorInstanceID))
	g.animator.SetInstanceRotation(uint32(g.animatorInstanceID), [3]float32{rx, ry, rz}, rot)
}

func (g *gameObject) SetScale(sx, sy, sz float32) {
	if g.animator == nil {
		g.initialScale = [3]float32{sx, sy, sz}
		return
	}
	pos, _ := g.animator.InstanceTransform(uint32(g.animatorInstanceID))
	g.animator.SetInstanceTransform(uint32(g.animatorInstanceID), pos, [3]float32{sx, sy, sz})
}

func (g *gameObject) Light() light.Light {
	return g.attachedLight
}

func (g *gameObject) SetLight(l light.Light) {
	g.attachedLight = l
}
