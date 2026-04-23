package composition

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"
)

// GPUCompositionParamsSource is the canonical WGSL definition of the CompositionParams struct.
// Matches GPUCompositionParams layout exactly (32 bytes, std140 aligned).
//
//go:embed assets/composition-params.wgsl
var GPUCompositionParamsSource string

// GPUCompositionParams is the GPU-aligned uniform data for the composition
// fragment shader. Contains tone mapping, exposure, auto-exposure, and bloom
// configuration. When auto-exposure is enabled the fragment shader reads the
// adapted exposure from the luminance storage buffer instead of the static value.
// Matches the WGSL CompositionParams struct layout exactly (see GPUCompositionParamsSource).
// Size: 32 bytes.
//
// Layout:
//
//	u32 tone_mapping_enabled  (4 bytes, offset  0)
//	f32 exposure              (4 bytes, offset  4)
//	u32 auto_exposure_enabled (4 bytes, offset  8)
//	u32 bloom_enabled         (4 bytes, offset 12)
//	f32 bloom_intensity       (4 bytes, offset 16)
//	u32 _pad5                 (4 bytes, offset 20)
//	u32 _pad6                 (4 bytes, offset 24)
//	u32 _pad7                 (4 bytes, offset 28)
type GPUCompositionParams struct {
	ToneMappingEnabled  uint32
	Exposure            float32
	AutoExposureEnabled uint32
	BloomEnabled        uint32
	BloomIntensity      float32
	_pad5               uint32
	_pad6               uint32
	_pad7               uint32
}

// Size returns the size of the GPUCompositionParams struct in bytes.
//
// Returns:
//   - int: the size in bytes (32)
func (p *GPUCompositionParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUCompositionParams struct into a byte buffer
// suitable for GPU uniform buffer upload.
//
// Returns:
//   - []byte: the marshaled buffer ready for GPU upload
func (p *GPUCompositionParams) Marshal() []byte {
	buf := make([]byte, p.Size())
	binary.LittleEndian.PutUint32(buf[0:4], p.ToneMappingEnabled)
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(p.Exposure))
	binary.LittleEndian.PutUint32(buf[8:12], p.AutoExposureEnabled)
	binary.LittleEndian.PutUint32(buf[12:16], p.BloomEnabled)
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(p.BloomIntensity))
	binary.LittleEndian.PutUint32(buf[20:24], 0) // _pad5
	binary.LittleEndian.PutUint32(buf[24:28], 0) // _pad6
	binary.LittleEndian.PutUint32(buf[28:32], 0) // _pad7
	return buf
}

// GPULuminanceParamsSource is the canonical WGSL definition of the LuminanceParams struct.
// Matches GPULuminanceParams layout exactly (32 bytes, std430 aligned).
//
//go:embed assets/luminance-params.wgsl
var GPULuminanceParamsSource string

// GPULuminanceParams is the GPU-aligned uniform data for the luminance compute shader.
// Each frame, the compute shader samples a 16x16 grid of HDR texels, computes the
// log-average luminance, derives a target exposure via the key-value formula, and
// smoothly adapts the persistent exposure storage buffer.
// Matches the WGSL LuminanceParams struct layout exactly (see GPULuminanceParamsSource).
// Size: 32 bytes.
//
// Layout:
//
//	u32 screen_width          (4 bytes, offset  0)
//	u32 screen_height         (4 bytes, offset  4)
//	f32 adapt_speed           (4 bytes, offset  8)
//	f32 delta_time            (4 bytes, offset 12)
//	f32 min_exposure          (4 bytes, offset 16)
//	f32 max_exposure          (4 bytes, offset 20)
//	f32 key_value             (4 bytes, offset 24)
//	u32 auto_exposure_enabled (4 bytes, offset 28)
type GPULuminanceParams struct {
	ScreenWidth         uint32
	ScreenHeight        uint32
	AdaptSpeed          float32
	DeltaTime           float32
	MinExposure         float32
	MaxExposure         float32
	KeyValue            float32
	AutoExposureEnabled uint32
}

// Size returns the size of the GPULuminanceParams struct in bytes.
//
// Returns:
//   - uint64: the struct size in bytes (32)
func (p *GPULuminanceParams) Size() uint64 {
	return 32
}

// Marshal serializes the GPULuminanceParams struct into a byte buffer suitable
// for GPU uniform buffer upload.
//
// Returns:
//   - []byte: 32-byte buffer ready for GPU upload
func (p *GPULuminanceParams) Marshal() []byte {
	buf := make([]byte, p.Size())
	binary.LittleEndian.PutUint32(buf[0:4], p.ScreenWidth)
	binary.LittleEndian.PutUint32(buf[4:8], p.ScreenHeight)
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(p.AdaptSpeed))
	binary.LittleEndian.PutUint32(buf[12:16], math.Float32bits(p.DeltaTime))
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(p.MinExposure))
	binary.LittleEndian.PutUint32(buf[20:24], math.Float32bits(p.MaxExposure))
	binary.LittleEndian.PutUint32(buf[24:28], math.Float32bits(p.KeyValue))
	binary.LittleEndian.PutUint32(buf[28:32], p.AutoExposureEnabled)
	return buf
}

// GPUBloomParamsSource is the canonical WGSL definition of the BloomParams struct.
// Matches GPUBloomParams layout exactly (16 bytes, std140 aligned).
//
//go:embed assets/bloom-params.wgsl
var GPUBloomParamsSource string

// GPUBloomParams is the GPU-aligned uniform data for the bloom downsample
// compute shader. Contains the brightness threshold that controls which
// pixels contribute to the bloom effect. Set to 0 to disable threshold
// filtering (used for downsample passes after the first).
// Matches the WGSL BloomParams struct layout exactly (see GPUBloomParamsSource).
// Size: 16 bytes.
//
// Layout:
//
//	f32 threshold (4 bytes, offset  0)
//	u32 _pad0     (4 bytes, offset  4)
//	u32 _pad1     (4 bytes, offset  8)
//	u32 _pad2     (4 bytes, offset 12)
type GPUBloomParams struct {
	Threshold float32
	_pad0     uint32
	_pad1     uint32
	_pad2     uint32
}

// Size returns the size of the GPUBloomParams struct in bytes.
//
// Returns:
//   - int: the size in bytes (16)
func (p *GPUBloomParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUBloomParams struct into a byte buffer
// suitable for GPU uniform buffer upload.
//
// Returns:
//   - []byte: the marshaled buffer ready for GPU upload
func (p *GPUBloomParams) Marshal() []byte {
	buf := make([]byte, p.Size())
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(p.Threshold))
	binary.LittleEndian.PutUint32(buf[4:8], 0)   // _pad0
	binary.LittleEndian.PutUint32(buf[8:12], 0)  // _pad1
	binary.LittleEndian.PutUint32(buf[12:16], 0) // _pad2
	return buf
}
