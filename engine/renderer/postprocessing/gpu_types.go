package postprocessing

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
// Each frame, the compute shader samples a 16×16 grid of HDR texels, computes the
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

// GPUTAAParamsSource is the canonical WGSL definition of the TAAParams struct.
// Matches GPUTAAParams layout exactly (160 bytes, std430 aligned).
//
//go:embed assets/taa-params.wgsl
var GPUTAAParamsSource string

// GPUTAAParams is the GPU-aligned representation of the TAA resolve compute
// shader uniform buffer. Holds the inverse current VP for depth reconstruction,
// the previous VP for historical reprojection, per-frame jitter offsets, screen
// dimensions, the temporal blend weight, the diagnostic history clamp
// rectification scale, and the raw-history-only diagnostic flag.
// Matches the WGSL TAAParams struct layout exactly (see GPUTAAParamsSource).
// Size: 176 bytes (std430 / WGSL aligned).
type GPUTAAParams struct {
	InvCurrViewProj           [16]float32 // offset   0: inverse of jittered current view-proj matrix
	PrevViewProj              [16]float32 // offset  64: previous frame's jittered view-proj matrix
	JitterCurr                [2]float32  // offset 128: current NDC jitter (x, y)
	JitterPrev                [2]float32  // offset 136: previous NDC jitter (x, y)
	ScreenWidth               float32     // offset 144
	ScreenHeight              float32     // offset 148
	BlendFactor               float32     // offset 152: weight for the current frame (e.g. 0.1)
	HistoryRectificationScale float32     // offset 156: YCoCg clamp expansion scale (1.0 = baseline)
	RawHistoryOnly            float32     // offset 160: 1.0 = output raw reprojected history
	_pad                      [3]float32  // offset 164: uniform padding to 16-byte struct alignment
}

// Size returns the size of the GPUTAAParams struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (176)
func (p *GPUTAAParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUTAAParams struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 176-byte buffer ready for GPU upload
func (p *GPUTAAParams) Marshal() []byte {
	buf := make([]byte, p.Size())
	off := 0
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.InvCurrViewProj[i]))
		off += 4
	}
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.PrevViewProj[i]))
		off += 4
	}
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.JitterCurr[0]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.JitterCurr[1]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.JitterPrev[0]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.JitterPrev[1]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.ScreenWidth))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.ScreenHeight))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.BlendFactor))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.HistoryRectificationScale))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.RawHistoryOnly))
	return buf
}
