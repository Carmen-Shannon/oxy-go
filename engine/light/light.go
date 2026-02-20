package light

// LightType identifies the kind of light source.
type LightType int

const (
	// LightTypeDirectional represents a light with no position, only direction.
	// Used for large distant sources like the sun or moon. Affects all fragments
	// uniformly with no distance attenuation.
	LightTypeDirectional LightType = iota

	// LightTypePoint represents a light that emits in all directions from a position.
	// Used for bare bulbs, lanterns, candle flames, and particle-emitted lights.
	// Attenuates with distance up to a configurable range.
	LightTypePoint

	// LightTypeSpot represents a light that emits in a cone from a position along a direction.
	// Used for flashlights, desk lamps, and wall sconces. Attenuates with both
	// distance and angle from the cone axis, controlled by inner and outer cone angles.
	LightTypeSpot
)

// lightImpl is the implementation of the Light interface.
type lightImpl struct {
	lightType    LightType
	position     [3]float32
	direction    [3]float32
	color        [3]float32
	intensity    float32
	lightRange   float32
	innerCone    float32 // stored as cos(angle in radians)
	outerCone    float32 // stored as cos(angle in radians)
	enabled      bool
	ephemeral    bool
	castsShadows bool
}

// Light defines the interface for a light source in the scene.
//
// Lights are scene-level entities that contribute to the final pixel color
// during the lit forward rendering pass. All light types (directional, point,
// spot) share this interface; type-specific properties (e.g. cone angles for
// spot lights) return zero values when not applicable.
//
// Lights are managed by the scene and marshaled into a GPU storage buffer
// each frame via the gpu_types helpers.
type Light interface {
	// Type returns the kind of light source.
	//
	// Returns:
	//   - LightType: the light type (directional, point, or spot)
	Type() LightType

	// Position returns the world-space position of the light.
	// Meaningless for directional lights.
	//
	// Returns:
	//   - [3]float32: position as (x, y, z)
	Position() [3]float32

	// Direction returns the normalized direction of the light.
	// For directional lights this is the light direction. For spot lights this
	// is the cone axis. Meaningless for point lights.
	//
	// Returns:
	//   - [3]float32: normalized direction as (x, y, z)
	Direction() [3]float32

	// Color returns the RGB color of the light.
	//
	// Returns:
	//   - [3]float32: color as (r, g, b)
	Color() [3]float32

	// Intensity returns the scalar intensity multiplier for the light.
	//
	// Returns:
	//   - float32: the intensity value
	Intensity() float32

	// Range returns the maximum attenuation distance for point and spot lights.
	// Beyond this distance the light contributes zero energy. Meaningless for
	// directional lights.
	//
	// Returns:
	//   - float32: the range value
	Range() float32

	// InnerCone returns the cosine of the inner cone half-angle for spot lights.
	// Fragments within this angle receive full intensity. Meaningless for
	// directional and point lights.
	//
	// Returns:
	//   - float32: cos(inner half-angle)
	InnerCone() float32

	// OuterCone returns the cosine of the outer cone half-angle for spot lights.
	// Fragments outside this angle receive zero intensity from the spot cone
	// falloff. Meaningless for directional and point lights.
	//
	// Returns:
	//   - float32: cos(outer half-angle)
	OuterCone() float32

	// Enabled returns whether this light is active for rendering.
	// Disabled lights are skipped during GPU buffer marshaling.
	//
	// Returns:
	//   - bool: true if the light is enabled
	Enabled() bool

	// Ephemeral returns whether this light is a short-lived particle-emitted light.
	// Ephemeral lights are not persisted in the scene's light registry and are
	// managed by their owning particle system.
	//
	// Returns:
	//   - bool: true if ephemeral
	Ephemeral() bool

	// CastsShadows returns whether this light is eligible for shadow map generation.
	// Shadow-casting lights have their depth pass rendered each frame, which is
	// expensive. Most ephemeral and distant lights should have this disabled.
	//
	// Returns:
	//   - bool: true if the light casts shadows
	CastsShadows() bool

	// SetPosition sets the world-space position of the light.
	//
	// Parameters:
	//   - x, y, z: position components
	SetPosition(x, y, z float32)

	// SetDirection sets the direction of the light and normalizes it.
	//
	// Parameters:
	//   - x, y, z: direction components (will be normalized)
	SetDirection(x, y, z float32)

	// SetColor sets the RGB color of the light.
	//
	// Parameters:
	//   - r, g, b: color components
	SetColor(r, g, b float32)

	// SetIntensity sets the scalar intensity multiplier.
	//
	// Parameters:
	//   - intensity: the intensity value
	SetIntensity(intensity float32)

	// SetRange sets the maximum attenuation distance.
	//
	// Parameters:
	//   - lightRange: the range value
	SetRange(lightRange float32)

	// SetSpotCone sets the inner and outer cone half-angles for spot lights.
	// Angles are specified in degrees and stored internally as cosines.
	//
	// Parameters:
	//   - innerDeg: inner cone half-angle in degrees
	//   - outerDeg: outer cone half-angle in degrees
	SetSpotCone(innerDeg, outerDeg float32)

	// SetEnabled enables or disables the light for rendering.
	//
	// Parameters:
	//   - enabled: true to enable
	SetEnabled(enabled bool)

	// SetEphemeral marks the light as ephemeral (particle-emitted).
	//
	// Parameters:
	//   - ephemeral: true if ephemeral
	SetEphemeral(ephemeral bool)

	// SetCastsShadows sets whether the light is eligible for shadow mapping.
	//
	// Parameters:
	//   - castsShadows: true to enable shadow casting
	SetCastsShadows(castsShadows bool)
}

var _ Light = &lightImpl{}

// NewLight creates a new Light of the specified type with sensible defaults and
// any provided options applied.
//
// Parameters:
//   - lightType: the kind of light to create (directional, point, or spot)
//   - opts: variadic list of LightBuilderOption functions to configure the light
//
// Returns:
//   - Light: a new Light instance
func NewLight(lightType LightType, opts ...LightBuilderOption) Light {
	l := &lightImpl{
		lightType:    lightType,
		position:     [3]float32{0, 0, 0},
		direction:    [3]float32{0, -1, 0},
		color:        [3]float32{1, 1, 1},
		intensity:    1.0,
		lightRange:   10.0,
		innerCone:    0.9063, // cos(25°)
		outerCone:    0.8192, // cos(35°)
		enabled:      true,
		ephemeral:    false,
		castsShadows: false,
	}
	for _, opt := range opts {
		opt(l)
	}
	return l
}

func (l *lightImpl) Type() LightType {
	return l.lightType
}

func (l *lightImpl) Position() [3]float32 {
	return l.position
}

func (l *lightImpl) Direction() [3]float32 {
	return l.direction
}

func (l *lightImpl) Color() [3]float32 {
	return l.color
}

func (l *lightImpl) Intensity() float32 {
	return l.intensity
}

func (l *lightImpl) Range() float32 {
	return l.lightRange
}

func (l *lightImpl) InnerCone() float32 {
	return l.innerCone
}

func (l *lightImpl) OuterCone() float32 {
	return l.outerCone
}

func (l *lightImpl) Enabled() bool {
	return l.enabled
}

func (l *lightImpl) Ephemeral() bool {
	return l.ephemeral
}

func (l *lightImpl) CastsShadows() bool {
	return l.castsShadows
}

func (l *lightImpl) SetPosition(x, y, z float32) {
	l.position = [3]float32{x, y, z}
}

func (l *lightImpl) SetDirection(x, y, z float32) {
	l.direction = normalize3(x, y, z)
}

func (l *lightImpl) SetColor(r, g, b float32) {
	l.color = [3]float32{r, g, b}
}

func (l *lightImpl) SetIntensity(intensity float32) {
	l.intensity = intensity
}

func (l *lightImpl) SetRange(lightRange float32) {
	l.lightRange = lightRange
}

func (l *lightImpl) SetSpotCone(innerDeg, outerDeg float32) {
	l.innerCone = cosDeg(innerDeg)
	l.outerCone = cosDeg(outerDeg)
}

func (l *lightImpl) SetEnabled(enabled bool) {
	l.enabled = enabled
}

func (l *lightImpl) SetEphemeral(ephemeral bool) {
	l.ephemeral = ephemeral
}

func (l *lightImpl) SetCastsShadows(castsShadows bool) {
	l.castsShadows = castsShadows
}
