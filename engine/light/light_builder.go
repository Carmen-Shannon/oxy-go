package light

// LightBuilderOption is a function that configures a Light instance during construction.
type LightBuilderOption func(*light)

// WithPosition is an option builder that sets the world-space position of the light.
//
// Parameters:
//   - x: the x position component
//   - y: the y position component
//   - z: the z position component
//
// Returns:
//   - LightBuilderOption: a function that applies the position option to a lightImpl
func WithPosition(x, y, z float32) LightBuilderOption {
	return func(l *light) {
		l.position = [3]float32{x, y, z}
	}
}

// WithDirection is an option builder that sets the direction of the light.
// The direction is normalized before storing.
//
// Parameters:
//   - x: the x direction component
//   - y: the y direction component
//   - z: the z direction component
//
// Returns:
//   - LightBuilderOption: a function that applies the direction option to a lightImpl
func WithDirection(x, y, z float32) LightBuilderOption {
	return func(l *light) {
		l.direction = normalize3(x, y, z)
	}
}

// WithColor is an option builder that sets the RGB color of the light.
//
// Parameters:
//   - r: the red color component
//   - g: the green color component
//   - b: the blue color component
//
// Returns:
//   - LightBuilderOption: a function that applies the color option to a lightImpl
func WithColor(r, g, b float32) LightBuilderOption {
	return func(l *light) {
		l.color = [3]float32{r, g, b}
	}
}

// WithIntensity is an option builder that sets the scalar intensity multiplier.
//
// Parameters:
//   - intensity: the intensity value
//
// Returns:
//   - LightBuilderOption: a function that applies the intensity option to a lightImpl
func WithIntensity(intensity float32) LightBuilderOption {
	return func(l *light) {
		l.intensity = intensity
	}
}

// WithRange is an option builder that sets the maximum attenuation distance for
// point and spot lights.
//
// Parameters:
//   - lightRange: the range value
//
// Returns:
//   - LightBuilderOption: a function that applies the range option to a lightImpl
func WithRange(lightRange float32) LightBuilderOption {
	return func(l *light) {
		l.lightRange = lightRange
	}
}

// WithSpotCone is an option builder that sets the inner and outer cone half-angles
// for spot lights. Angles are specified in degrees and converted to cosines internally,
// which is the format required by the GPU shader.
//
// Parameters:
//   - innerDeg: inner cone half-angle in degrees
//   - outerDeg: outer cone half-angle in degrees
//
// Returns:
//   - LightBuilderOption: a function that applies the spot cone option to a lightImpl
func WithSpotCone(innerDeg, outerDeg float32) LightBuilderOption {
	return func(l *light) {
		l.innerCone = cosDeg(innerDeg)
		l.outerCone = cosDeg(outerDeg)
	}
}

// WithEnabled is an option builder that sets whether the light is active for rendering.
//
// Parameters:
//   - enabled: true to enable the light
//
// Returns:
//   - LightBuilderOption: a function that applies the enabled option to a lightImpl
func WithEnabled(enabled bool) LightBuilderOption {
	return func(l *light) {
		l.enabled = enabled
	}
}

// WithEphemeral is an option builder that marks the light as ephemeral, meaning it
// is a short-lived particle-emitted light that is not persisted in the scene registry.
//
// Parameters:
//   - ephemeral: true if the light is ephemeral
//
// Returns:
//   - LightBuilderOption: a function that applies the ephemeral option to a lightImpl
func WithEphemeral(ephemeral bool) LightBuilderOption {
	return func(l *light) {
		l.ephemeral = ephemeral
	}
}

// WithCastsShadows is an option builder that sets whether the light is eligible for
// shadow map generation.
//
// Parameters:
//   - castsShadows: true to enable shadow casting
//
// Returns:
//   - LightBuilderOption: a function that applies the shadow casting option to a lightImpl
func WithCastsShadows(castsShadows bool) LightBuilderOption {
	return func(l *light) {
		l.castsShadows = castsShadows
	}
}

// WithShadowBias sets the depth comparison bias for this light's shadow map.
// Default is 0.003.
//
// Parameters:
//   - bias: the depth bias value
//
// Returns:
//   - LightBuilderOption: a function that applies the shadow bias option to a lightImpl
func WithShadowBias(bias float32) LightBuilderOption {
	return func(l *light) {
		l.shadowBias = bias
	}
}

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
	l := &light{
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
		shadowBias:   0.0002,
	}
	for _, opt := range opts {
		opt(l)
	}
	return l
}
