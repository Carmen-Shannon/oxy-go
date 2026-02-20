package material

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"
)

// GPUOverlayParamsSource is the canonical WGSL definition of the OverlayParams struct.
// Matches GPUOverlayParams layout exactly (16 bytes, std430 aligned).
//
//go:embed assets/overlay_params.wgsl
var GPUOverlayParamsSource string

// GPUOverlayParams is the GPU-aligned uniform for the overlay fragment shader.
// Matches the WGSL OverlayParams struct layout exactly (see GPUOverlayParamsSource).
// Size: 16 bytes (one vec4<f32>, std430 aligned).
type GPUOverlayParams struct {
	OverlayColor [4]float32 // offset 0: RGBA overlay color written to all fragments (16 bytes)
}

// Size returns the size of the GPUOverlayParams struct in bytes.
//
// Returns:
//   - int: the size of the struct in bytes.
func (g *GPUOverlayParams) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUOverlayParams struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 16-byte buffer ready for GPU upload.
func (g *GPUOverlayParams) Marshal() []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(g.OverlayColor[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.OverlayColor[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.OverlayColor[2]))
	binary.LittleEndian.PutUint32(buf[12:16], math.Float32bits(g.OverlayColor[3]))
	return buf
}

// GPUEffectParamsSource is the canonical WGSL definition of the EffectParams struct.
// Matches GPUEffectParams layout exactly (16 bytes, std430 aligned).
//
//go:embed assets/effect_params.wgsl
var GPUEffectParamsSource string

// GPUEffectParams is the GPU-aligned uniform for the textured and skinned-rainbow fragment shaders.
// The RGB channels set the tint color; the alpha channel controls the tint blend intensity
// (0.0 = no tint, 1.0 = fully tinted).
// Matches the WGSL EffectParams struct layout exactly (see GPUEffectParamsSource).
// Size: 16 bytes (one vec4<f32>, std430 aligned).
type GPUEffectParams struct {
	TintColor [4]float32 // offset 0: RGB tint color + alpha blend intensity (16 bytes)
}

// Size returns the size of the GPUEffectParams struct in bytes.
//
// Returns:
//   - int: the size of the struct in bytes.
func (g *GPUEffectParams) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUEffectParams struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 16-byte buffer ready for GPU upload.
func (g *GPUEffectParams) Marshal() []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(g.TintColor[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.TintColor[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.TintColor[2]))
	binary.LittleEndian.PutUint32(buf[12:16], math.Float32bits(g.TintColor[3]))
	return buf
}
