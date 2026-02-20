package camera

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"
)

// GPUCameraUniformSource is the canonical WGSL definition of the CameraUniform struct.
// Matches GPUCameraUniform layout exactly (80 bytes, std430 aligned).
//
//go:embed assets/camera_uniform.wgsl
var GPUCameraUniformSource string

// GPUCameraUniform is the GPU-aligned representation of the camera uniform buffer.
// Matches the WGSL CameraUniform struct layout exactly (see GPUCameraUniformSource).
// Size: 80 bytes (std430 / WGSL aligned).
type GPUCameraUniform struct {
	ViewProj       [16]float32 // offset  0: combined view-projection matrix (mat4x4<f32>)
	CameraPosition [3]float32  // offset 64: world-space camera position (vec3<f32>)
	_pad           float32     // offset 76: padding to 80 bytes
}

// Size returns the size of the GPUCameraUniform struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (80)
func (g *GPUCameraUniform) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUCameraUniform struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: the serialized byte buffer
func (g *GPUCameraUniform) Marshal() []byte {
	buf := make([]byte, g.Size())
	for i := range 16 {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(g.ViewProj[i]))
	}
	for i := range 3 {
		binary.LittleEndian.PutUint32(buf[64+i*4:], math.Float32bits(g.CameraPosition[i]))
	}
	binary.LittleEndian.PutUint32(buf[76:], 0) // _pad
	return buf
}
