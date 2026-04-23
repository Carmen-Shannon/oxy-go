package ssr

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"
)

// GPUSSRParamsSource is the canonical WGSL definition of the SSRParams struct.
// Matches GPUSSRParams layout exactly (64 bytes, std140 aligned).
//
//go:embed assets/ssr-params.wgsl
var GPUSSRParamsSource string

// GPUSSRParams is the GPU-aligned uniform data for the screen-space reflections
// compute shader. Contains the projection, inverse projection, and view matrices
// along with ray march configuration parameters.
// Matches the WGSL SSRParams struct layout exactly (see GPUSSRParamsSource).
// Size: 224 bytes.
//
// Layout:
//
//	mat4x4<f32> projection      (64 bytes, offset   0)
//	mat4x4<f32> inv_projection  (64 bytes, offset  64)
//	mat4x4<f32> view            (64 bytes, offset 128)
//	f32         max_distance    ( 4 bytes, offset 192)
//	f32         thickness       ( 4 bytes, offset 196)
//	f32         stride          ( 4 bytes, offset 200)
//	u32         max_steps       ( 4 bytes, offset 204)
//	f32         screen_width    ( 4 bytes, offset 208)
//	f32         screen_height   ( 4 bytes, offset 212)
//	f32         roughness_cutoff( 4 bytes, offset 216)
//	u32         hiz_mip_count   ( 4 bytes, offset 220)
type GPUSSRParams struct {
	Projection      [16]float32 // offset   0: projection matrix (column-major)
	InvProjection   [16]float32 // offset  64: inverse projection matrix (column-major)
	View            [16]float32 // offset 128: view matrix (column-major)
	MaxDistance     float32     // offset 192: maximum ray march distance in view space
	Thickness       float32     // offset 196: depth thickness tolerance for hit detection
	Stride          float32     // offset 200: step stride multiplier for ray march (unused in Hi-Z mode)
	MaxSteps        uint32      // offset 204: maximum number of ray march steps
	ScreenWidth     float32     // offset 208: screen width in pixels
	ScreenHeight    float32     // offset 212: screen height in pixels
	RoughnessCutoff float32     // offset 216: roughness above which SSR is skipped
	HiZMipCount     uint32      // offset 220: number of Hi-Z mip levels in the depth pyramid
}

// Size returns the size of the GPUSSRParams struct in bytes.
//
// Returns:
//   - int: the size in bytes (224)
func (p *GPUSSRParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUSSRParams struct into a byte buffer suitable for
// GPU upload.
//
// Returns:
//   - []byte: 224-byte buffer ready for GPU upload
func (p *GPUSSRParams) Marshal() []byte {
	buf := make([]byte, p.Size())
	off := 0
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.Projection[i]))
		off += 4
	}
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.InvProjection[i]))
		off += 4
	}
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.View[i]))
		off += 4
	}
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.MaxDistance))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.Thickness))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.Stride))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], p.MaxSteps)
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.ScreenWidth))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.ScreenHeight))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.RoughnessCutoff))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], p.HiZMipCount)
	return buf
}
