package light

import (
	"math"
)

// light is the implementation of the Light interface.
type light struct {
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
	shadowBias   float32
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
