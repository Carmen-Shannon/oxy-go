package light

import "math"

// LightBuilderOption is a function that configures a Light instance during construction.
type LightBuilderOption func(*lightImpl)

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
	return func(l *lightImpl) {
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
	return func(l *lightImpl) {
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
	return func(l *lightImpl) {
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
	return func(l *lightImpl) {
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
	return func(l *lightImpl) {
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
	return func(l *lightImpl) {
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
	return func(l *lightImpl) {
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
	return func(l *lightImpl) {
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
	return func(l *lightImpl) {
		l.castsShadows = castsShadows
	}
}

// normalize3 normalizes a 3-component vector. Returns a zero vector if the input
// has zero length.
func normalize3(x, y, z float32) [3]float32 {
	length := float32(math.Sqrt(float64(x*x + y*y + z*z)))
	if length == 0 {
		return [3]float32{0, 0, 0}
	}
	inv := 1.0 / length
	return [3]float32{x * inv, y * inv, z * inv}
}

// cosDeg converts an angle in degrees to the cosine of that angle in radians.
func cosDeg(deg float32) float32 {
	return float32(math.Cos(float64(deg) * math.Pi / 180.0))
}
