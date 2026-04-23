package ssao

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"
)

// GPUBlurParamsSource is the canonical WGSL definition of the BlurParams struct.
// Matches GPUBlurParams layout exactly (24 bytes, std430 aligned).
//
//go:embed assets/blur-params.wgsl
var GPUBlurParamsSource string

// GPUBlurParams is the GPU-aligned representation of the separable blur
// compute shader uniform. Controls the blur direction, kernel half-width,
// an optional G-Buffer coordinate scale for half-resolution SSAO, and an
// optional per-cascade column width for atlas-aware shadow blurring.
// Matches the WGSL BlurParams struct layout exactly (see GPUBlurParamsSource).
// Size: 24 bytes (vec2<i32> + i32 + i32 + i32 + i32 pad).
//
// Layout:
//
//	vec2<i32> direction     (8 bytes, offset  0)
//	i32       radius        (4 bytes, offset  8)
//	i32       gbuffer_scale (4 bytes, offset 12)
//	i32       cascade_width (4 bytes, offset 16)
//	i32       _pad          (4 bytes, offset 20)
type GPUBlurParams struct {
	Direction    [2]int32 // (1,0) for horizontal, (0,1) for vertical
	Radius       int32    // half-width of the box filter kernel in texels
	GBufferScale int32    // coordinate multiplier for depth texture lookups (1 = full-res, 2 = half-res SSAO)
	CascadeWidth int32    // per-cascade atlas column width in texels; 0 disables column clamping
	//nolint:unused
	_pad int32 // explicit padding for 8-byte struct alignment
}

// Size returns the size of the GPUBlurParams struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (24)
func (p *GPUBlurParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUBlurParams struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 24-byte buffer ready for GPU upload
func (p *GPUBlurParams) Marshal() []byte {
	buf := make([]byte, p.Size())
	binary.LittleEndian.PutUint32(buf[0:4], uint32(p.Direction[0]))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(p.Direction[1]))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(p.Radius))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(p.GBufferScale))
	binary.LittleEndian.PutUint32(buf[16:20], uint32(p.CascadeWidth))
	binary.LittleEndian.PutUint32(buf[20:24], uint32(p._pad))
	return buf
}

// GPUSSAOParamsSource is the canonical WGSL definition of the SSAOParams struct.
// Matches GPUSSAOParams layout exactly (176 bytes, std430 aligned).
//
//go:embed assets/ssao-params.wgsl
var GPUSSAOParamsSource string

// GPUSSAOParams is the GPU-aligned uniform data for the SSAO compute shader.
// Contains the view-projection and inverse view-projection matrices needed to
// project world-space samples and reconstruct world positions from depth, plus
// SSAO quality parameters, screen dimensions, and camera position.
// Matches the WGSL SSAOParams struct layout exactly (see GPUSSAOParamsSource).
// Size: 176 bytes (std430 / WGSL aligned).
//
// Layout:
//
//	mat4x4<f32> projection        (64 bytes, offset   0)
//	mat4x4<f32> inv_view_proj     (64 bytes, offset  64)
//	f32         radius            ( 4 bytes, offset 128)
//	f32         bias              ( 4 bytes, offset 132)
//	f32         power             ( 4 bytes, offset 136)
//	u32         sample_count      ( 4 bytes, offset 140)
//	f32         screen_width      ( 4 bytes, offset 144)
//	f32         screen_height     ( 4 bytes, offset 148)
//	f32         gbuffer_scale     ( 4 bytes, offset 152)
//	f32         _pad              ( 4 bytes, offset 156)
//	vec3<f32>   camera_position   (12 bytes, offset 160)
//	f32         _pad2             ( 4 bytes, offset 172)
type GPUSSAOParams struct {
	Projection     [16]float32 // offset   0: view-projection matrix (column-major)
	InvViewProj    [16]float32 // offset  64: inverse view-projection matrix (column-major)
	Radius         float32     // offset 128: hemisphere sample radius in world units
	Bias           float32     // offset 132: depth comparison bias to prevent self-occlusion
	Power          float32     // offset 136: exponent applied to the final AO value
	SampleCount    uint32      // offset 140: number of hemisphere samples (max 32)
	ScreenWidth    float32     // offset 144: screen width in pixels (SSAO output resolution)
	ScreenHeight   float32     // offset 148: screen height in pixels (SSAO output resolution)
	GBufferScale   float32     // offset 152: coordinate multiplier for G-Buffer texture lookups (1.0 = full-res, 2.0 = half-res)
	_pad           float32     // offset 156: padding to 160-byte alignment
	CameraPosition [3]float32  // offset 160: world-space camera position for linear depth computation
	_pad2          float32     // offset 172: padding to 176-byte alignment
}

// Size returns the size of the GPUSSAOParams struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (176)
func (p *GPUSSAOParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUSSAOParams struct into a byte buffer suitable for
// GPU upload.
//
// Returns:
//   - []byte: 176-byte buffer ready for GPU upload
func (p *GPUSSAOParams) Marshal() []byte {
	buf := make([]byte, p.Size())
	off := 0
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.Projection[i]))
		off += 4
	}
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.InvViewProj[i]))
		off += 4
	}
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.Radius))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.Bias))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.Power))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], p.SampleCount)
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.ScreenWidth))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.ScreenHeight))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.GBufferScale))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], 0) // _pad
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.CameraPosition[0]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.CameraPosition[1]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.CameraPosition[2]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], 0) // _pad2
	return buf
}
