package light

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"

	"github.com/Carmen-Shannon/oxy-go/common"
)

// MaxGPULights is the maximum number of lights that can be marshaled into the
// GPU storage buffer per frame. The CPU-side light list is unbounded; this cap
// controls only how many lights the GPU evaluates. When the active light count
// exceeds this budget, the scene's light priority system selects the most
// impactful lights.
const MaxGPULights = 1024

// GPULightSource is the canonical WGSL definition of the Light struct.
// Matches GPULight layout exactly (64 bytes, std430 aligned).
//
//go:embed assets/light.wgsl
var GPULightSource string

// GPULight is the GPU-aligned representation of a single light source.
// Matches the WGSL Light struct layout exactly (see GPULightSource).
// Size: 64 bytes (std430 / WGSL aligned).
type GPULight struct {
	Position     [3]float32 // offset  0: world-space position (point/spot) or unused (directional)
	LightType    uint32     // offset 12: 0 = directional, 1 = point, 2 = spot
	Color        [3]float32 // offset 16: RGB color
	Intensity    float32    // offset 28: scalar multiplier
	Direction    [3]float32 // offset 32: normalized direction (directional/spot) or unused (point)
	LightRange   float32    // offset 44: attenuation cutoff distance
	InnerCone    float32    // offset 48: cos(inner half-angle) for spot
	OuterCone    float32    // offset 52: cos(outer half-angle) for spot
	CastsShadows uint32     // offset 56: 1 = casts shadows, 0 = does not
	_pad         uint32     // offset 60: padding to 64-byte alignment
}

// Size returns the size of the GPULight struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (64)
func (g *GPULight) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPULight struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 64-byte buffer ready for GPU upload
func (g *GPULight) Marshal() []byte {
	buf := make([]byte, 64)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(g.Position[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.Position[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.Position[2]))
	binary.LittleEndian.PutUint32(buf[12:16], g.LightType)
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(g.Color[0]))
	binary.LittleEndian.PutUint32(buf[20:24], math.Float32bits(g.Color[1]))
	binary.LittleEndian.PutUint32(buf[24:28], math.Float32bits(g.Color[2]))
	binary.LittleEndian.PutUint32(buf[28:32], math.Float32bits(g.Intensity))
	binary.LittleEndian.PutUint32(buf[32:36], math.Float32bits(g.Direction[0]))
	binary.LittleEndian.PutUint32(buf[36:40], math.Float32bits(g.Direction[1]))
	binary.LittleEndian.PutUint32(buf[40:44], math.Float32bits(g.Direction[2]))
	binary.LittleEndian.PutUint32(buf[44:48], math.Float32bits(g.LightRange))
	binary.LittleEndian.PutUint32(buf[48:52], math.Float32bits(g.InnerCone))
	binary.LittleEndian.PutUint32(buf[52:56], math.Float32bits(g.OuterCone))
	binary.LittleEndian.PutUint32(buf[56:60], g.CastsShadows)
	binary.LittleEndian.PutUint32(buf[60:64], 0) // padding
	return buf
}

// GPULightHeaderSource is the canonical WGSL definition of the LightHeader struct.
// Matches GPULightHeader layout exactly (16 bytes, std430 aligned).
//
//go:embed assets/light-header.wgsl
var GPULightHeaderSource string

// GPULightHeader is the header prepended to the light storage buffer.
// Contains the ambient color and the active light count.
// Matches the WGSL LightHeader struct layout exactly (see GPULightHeaderSource).
// Size: 16 bytes (vec3 + u32, std430 aligned).
type GPULightHeader struct {
	AmbientColor [3]float32 // offset 0: scene ambient RGB
	LightCount   uint32     // offset 12: number of active lights following the header
}

// Size returns the size of the GPULightHeader struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (16)
func (h *GPULightHeader) Size() int {
	return int(unsafe.Sizeof(*h))
}

// Marshal serializes the GPULightHeader struct into a byte buffer suitable for
// GPU upload.
//
// Returns:
//   - []byte: 16-byte buffer ready for GPU upload
func (h *GPULightHeader) Marshal() []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(h.AmbientColor[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(h.AmbientColor[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(h.AmbientColor[2]))
	binary.LittleEndian.PutUint32(buf[12:16], h.LightCount)
	return buf
}

// GPUShadowDataSource is the canonical WGSL definition of the ShadowData struct.
// Matches GPUShadowData layout exactly (176 bytes, std430 aligned).
//
//go:embed assets/shadow-data.wgsl
var GPUShadowDataSource string

// GPUShadowData is the GPU-aligned representation of directional shadow data.
// Matches the WGSL ShadowData struct layout exactly (see GPUShadowDataSource).
// Size: 176 bytes (std430 / WGSL aligned).
//
// Layout:
//
//	mat4x4<f32> light_vp              (64 bytes, offset   0)
//	mat4x4<f32> light_view            (64 bytes, offset  64)
//	vec2<f32>   texel_size            ( 8 bytes, offset 128)
//	f32         bias                  ( 4 bytes, offset 136)
//	f32         normal_bias           ( 4 bytes, offset 140)
//	f32         shadow_near           ( 4 bytes, offset 144)
//	f32         shadow_far            ( 4 bytes, offset 148)
//	f32         min_variance          ( 4 bytes, offset 152)
//	f32         light_bleed_reduction ( 4 bytes, offset 156)
//	f32         light_size            ( 4 bytes, offset 160)
//	f32         shadow_half_extent    ( 4 bytes, offset 164)
//	vec2<f32>   _pad                  ( 8 bytes, offset 168)
type GPUShadowData struct {
	LightVP             [16]float32 // orthographic view-projection from light's perspective
	LightView           [16]float32 // view-only matrix (no projection) for linear depth
	TexelSize           [2]float32  // 1.0 / shadow_map_resolution for VSM texel calculations
	Bias                float32     // depth comparison bias to reduce shadow acne
	NormalBias          float32     // world-space normal-offset distance for shadow lookup
	ShadowNear          float32     // near plane for linear depth normalization
	ShadowFar           float32     // far plane for linear depth normalization
	MinVariance         float32     // minimum variance clamp for Chebyshev's inequality (VSM)
	LightBleedReduction float32     // exponent to reduce light-bleeding artifacts (VSM)
	LightSize           float32     // world-space light size for PCSS penumbra estimation
	ShadowHalfExtent    float32     // orthographic frustum half-size for PCSS world-to-texel conversion
	_pad                [2]float32  // padding to 176 bytes (struct must be 16-byte aligned)
}

// Size returns the size of the GPUShadowData struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (176)
func (s *GPUShadowData) Size() int {
	return int(unsafe.Sizeof(*s))
}

// ComputeDirectionalLightVP builds an orthographic view-projection matrix for a
// directional light's shadow pass and stores it in the receiver's LightVP field.
// Also stores the view-only matrix in LightView and the near/far planes in
// ShadowNear/ShadowFar for VSM linear depth normalization.
// The frustum is centered on the provided center position (typically the camera
// position) and aligned to look along the light's direction.
//
// Parameters:
//   - lightDir: normalized direction the light points (from light toward scene)
//   - centerX, centerY, centerZ: world-space center of the shadow frustum
//   - halfExtent: half-size of the orthographic frustum in world units
//   - near: near plane distance
//   - far: far plane distance
func (s *GPUShadowData) ComputeDirectionalLightVP(lightDir [3]float32, centerX, centerY, centerZ, halfExtent, near, far float32) {
	// Position the "eye" behind the center, opposite the light direction,
	// so we look from behind the scene toward the lit area.
	eyeX := centerX - lightDir[0]*far*0.5
	eyeY := centerY - lightDir[1]*far*0.5
	eyeZ := centerZ - lightDir[2]*far*0.5

	// Choose a stable up vector that isn't parallel to the light direction.
	// If the light points nearly straight up or down, use X-axis as up.
	upX, upY, upZ := float32(0), float32(1), float32(0)
	if absF32(lightDir[1]) > 0.99 {
		upX, upY, upZ = 1, 0, 0
	}

	var view [16]float32
	common.LookAt(view[:],
		eyeX, eyeY, eyeZ,
		centerX, centerY, centerZ,
		upX, upY, upZ,
	)

	// Store view-only matrix for VSM linear depth computation.
	copy(s.LightView[:], view[:])
	s.ShadowNear = near
	s.ShadowFar = far

	var proj [16]float32
	ortho(proj[:], -halfExtent, halfExtent, -halfExtent, halfExtent, near, far)

	common.Mul4(s.LightVP[:], proj[:], view[:])
}

// ComputeNormalBias derives the world-space normal-offset bias from the shadow
// map parameters and stores it in the receiver's NormalBias field. The result is
// the distance (in world units) that fragment positions are shifted along their
// surface normal before projecting into light clip space. This prevents
// self-shadowing on concave geometry.
//
// Parameters:
//   - halfExtent: orthographic frustum half-size in world units
//   - scale: multiplier on the per-texel world size (typically 2.0–4.0)
//   - resolution: shadow map resolution in texels (width and height)
func (s *GPUShadowData) ComputeNormalBias(halfExtent, scale float32, resolution int) {
	texelWorldSize := 2.0 * halfExtent / float32(resolution)
	s.NormalBias = texelWorldSize * scale
}

// Marshal serializes the GPUShadowData struct into a byte buffer suitable for
// GPU uniform upload.
//
// Returns:
//   - []byte: 176-byte buffer ready for GPU upload
func (s *GPUShadowData) Marshal() []byte {
	buf := make([]byte, 176)
	off := 0

	// light_vp (64 bytes)
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.LightVP[i]))
		off += 4
	}

	// light_view (64 bytes)
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.LightView[i]))
		off += 4
	}

	// texel_size (8 bytes)
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.TexelSize[0]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.TexelSize[1]))
	off += 4

	// bias, normal_bias (8 bytes)
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.Bias))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.NormalBias))
	off += 4

	// shadow_near, shadow_far (8 bytes)
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.ShadowNear))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.ShadowFar))
	off += 4

	// min_variance, light_bleed_reduction (8 bytes)
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.MinVariance))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.LightBleedReduction))
	off += 4

	// light_size, shadow_half_extent (8 bytes)
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.LightSize))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(s.ShadowHalfExtent))
	off += 4

	// _pad (8 bytes, zeroed)
	// buf[off:off+8] is already zeroed from make()

	return buf
}

// GPUShadowUniformSource is the canonical WGSL definition of the ShadowUniform struct.
// Matches GPUShadowUniform layout exactly (136 bytes, std430 aligned).
//
//go:embed assets/shadow-uniform.wgsl
var GPUShadowUniformSource string

// GPUShadowUniform is the GPU-aligned representation of the shadow vertex
// shader uniform containing the light view-projection matrix, the view-only matrix,
// and near/far planes for linear depth normalization in VSM mode.
// Matches the WGSL ShadowUniform struct layout exactly (see GPUShadowUniformSource).
// Size: 136 bytes (mat4x4 + mat4x4 + f32 + f32).
//
// Layout:
//
//	mat4x4<f32> light_vp    (64 bytes, offset  0)
//	mat4x4<f32> light_view  (64 bytes, offset 64)
//	f32         shadow_near ( 4 bytes, offset 128)
//	f32         shadow_far  ( 4 bytes, offset 132)
type GPUShadowUniform struct {
	LightVP    [16]float32 // orthographic view-projection from light's perspective
	LightView  [16]float32 // view-only matrix (no projection) for linear depth in VSM
	ShadowNear float32     // near plane for linear depth normalization
	ShadowFar  float32     // far plane for linear depth normalization
}

// Size returns the size of the GPUShadowUniform struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (136)
func (u *GPUShadowUniform) Size() int {
	return int(unsafe.Sizeof(*u))
}

// Marshal serializes the GPUShadowUniform struct into a byte buffer suitable for
// GPU uniform upload.
//
// Returns:
//   - []byte: 136-byte buffer ready for GPU upload
func (u *GPUShadowUniform) Marshal() []byte {
	buf := make([]byte, 136)
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(u.LightVP[i]))
	}
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[64+i*4:64+(i+1)*4], math.Float32bits(u.LightView[i]))
	}
	binary.LittleEndian.PutUint32(buf[128:132], math.Float32bits(u.ShadowNear))
	binary.LittleEndian.PutUint32(buf[132:136], math.Float32bits(u.ShadowFar))
	return buf
}

// GPUBlurParamsSource is the canonical WGSL definition of the BlurParams struct.
// Matches GPUBlurParams layout exactly (16 bytes, std430 aligned).
//
//go:embed assets/blur-params.wgsl
var GPUBlurParamsSource string

// GPUBlurParams is the GPU-aligned representation of the separable blur
// compute shader uniform. Controls the blur direction, kernel half-width,
// and an optional G-Buffer coordinate scale for half-resolution SSAO.
// Matches the WGSL BlurParams struct layout exactly (see GPUBlurParamsSource).
// Size: 16 bytes (vec2<i32> + i32 + i32).
//
// Layout:
//
//	vec2<i32> direction     (8 bytes, offset  0)
//	i32       radius        (4 bytes, offset  8)
//	i32       gbuffer_scale (4 bytes, offset 12)
type GPUBlurParams struct {
	Direction    [2]int32 // (1,0) for horizontal, (0,1) for vertical
	Radius       int32    // half-width of the box filter kernel in texels
	GBufferScale int32    // coordinate multiplier for depth texture lookups (1 = full-res, 2 = half-res SSAO)
}

// Size returns the size of the GPUBlurParams struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (16)
func (p *GPUBlurParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUBlurParams struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 16-byte buffer ready for GPU upload
func (p *GPUBlurParams) Marshal() []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(p.Direction[0]))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(p.Direction[1]))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(p.Radius))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(p.GBufferScale))
	return buf
}

// GPUSATParamsSource is the canonical WGSL definition of the SATParams struct.
// Matches GPUSATParams layout exactly (16 bytes, std430 aligned).
//
//go:embed assets/sat-params.wgsl
var GPUSATParamsSource string

// GPUSATParams is the GPU-aligned representation of the SAT (Summed-Area Table)
// compute shader uniform. Controls the prefix-sum direction and step offset.
// When Offset is 0, the shader performs precision distribution from the RG32Float
// moments texture into RGBA32Float; when Offset > 0, it performs a standard
// recursive-doubling prefix-sum step.
// Matches the WGSL SATParams struct layout exactly (see GPUSATParamsSource).
// Size: 16 bytes (vec2<i32> + i32 + i32 padding).
//
// Layout:
//
//	vec2<i32> direction  (8 bytes, offset  0)
//	i32       offset     (4 bytes, offset  8)
//	i32       _pad       (4 bytes, offset 12)
type GPUSATParams struct {
	Direction [2]int32 // (1,0) for horizontal, (0,1) for vertical
	Offset    int32    // 2^k step offset for recursive doubling; 0 = precision distribution pass
	_pad      int32    // padding to 16-byte alignment
}

// Size returns the size of the GPUSATParams struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (16)
func (p *GPUSATParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUSATParams struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 16-byte buffer ready for GPU upload
func (p *GPUSATParams) Marshal() []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(p.Direction[0]))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(p.Direction[1]))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(p.Offset))
	binary.LittleEndian.PutUint32(buf[12:16], 0) // padding
	return buf
}

// GPULightCullUniformsSource is the canonical WGSL definition of the LightCullUniforms struct.
// Matches GPULightCullUniforms layout exactly (160 bytes, std430 aligned).
//
//go:embed assets/light-cull-uniforms.wgsl
var GPULightCullUniformsSource string

// GPULightCullUniforms is the GPU-aligned uniform data for the light culling
// compute shader. Contains the inverse projection and view matrices needed
// to reconstruct per-tile frustum planes, plus tile/screen dimensions and
// the active light count.
// Matches the WGSL LightCullUniforms struct layout exactly (see GPULightCullUniformsSource).
// Size: 160 bytes (std430 / WGSL aligned).
//
// Layout:
//
//	mat4x4<f32> inv_proj       (64 bytes, offset  0)
//	mat4x4<f32> view_matrix    (64 bytes, offset 64)
//	u32         tile_count_x   ( 4 bytes, offset 128)
//	u32         tile_count_y   ( 4 bytes, offset 132)
//	u32         screen_width   ( 4 bytes, offset 136)
//	u32         screen_height  ( 4 bytes, offset 140)
//	u32         light_count    ( 4 bytes, offset 144)
//	f32         near           ( 4 bytes, offset 148)
//	f32         far            ( 4 bytes, offset 152)
//	u32         _pad           ( 4 bytes, offset 156)
type GPULightCullUniforms struct {
	InvProj      [16]float32 // inverse projection matrix
	ViewMatrix   [16]float32 // camera view matrix
	TileCountX   uint32
	TileCountY   uint32
	ScreenWidth  uint32
	ScreenHeight uint32
	LightCount   uint32
	Near         float32
	Far          float32
	_pad         uint32
}

// Size returns the size of the GPULightCullUniforms struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (160)
func (u *GPULightCullUniforms) Size() int {
	return int(unsafe.Sizeof(*u))
}

// Marshal serializes GPULightCullUniforms into a 160-byte little-endian buffer
// suitable for GPU upload.
//
// Returns:
//   - []byte: 160-byte buffer ready for GPU upload
func (u *GPULightCullUniforms) Marshal() []byte {
	buf := make([]byte, 160)
	off := 0

	// inv_proj (64 bytes)
	for i := range 16 {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(u.InvProj[i]))
		off += 4
	}
	// view_matrix (64 bytes)
	for i := range 16 {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(u.ViewMatrix[i]))
		off += 4
	}
	// tile_count_x, tile_count_y
	binary.LittleEndian.PutUint32(buf[off:off+4], u.TileCountX)
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], u.TileCountY)
	off += 4
	// screen_width, screen_height
	binary.LittleEndian.PutUint32(buf[off:off+4], u.ScreenWidth)
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], u.ScreenHeight)
	off += 4
	// light_count
	binary.LittleEndian.PutUint32(buf[off:off+4], u.LightCount)
	off += 4
	// near, far
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(u.Near))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(u.Far))
	off += 4
	// _pad
	binary.LittleEndian.PutUint32(buf[off:off+4], 0)

	return buf
}

// GPUTileUniformsSource is the canonical WGSL definition of the TileUniforms struct.
// Matches GPUTileUniforms layout exactly (8 bytes).
//
//go:embed assets/tile-uniforms.wgsl
var GPUTileUniformsSource string

// GPUTileUniforms is the GPU-aligned uniform data read by the lit fragment
// shader to compute which tile a fragment belongs to and index into the
// per-tile light list buffer.
// Matches the WGSL TileUniforms struct layout exactly (see GPUTileUniformsSource).
// Size: 8 bytes.
//
// Layout:
//
//	u32 tile_count_x       (4 bytes, offset 0)
//	u32 max_lights_per_tile (4 bytes, offset 4)
type GPUTileUniforms struct {
	TileCountX       uint32
	MaxLightsPerTile uint32
}

// Size returns the size of the GPUTileUniforms struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (8)
func (u *GPUTileUniforms) Size() int {
	return int(unsafe.Sizeof(*u))
}

// Marshal serializes GPUTileUniforms into an 8-byte little-endian buffer suitable
// for GPU upload.
//
// Returns:
//   - []byte: 8-byte buffer ready for GPU upload
func (u *GPUTileUniforms) Marshal() []byte {
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint32(buf[0:4], u.TileCountX)
	binary.LittleEndian.PutUint32(buf[4:8], u.MaxLightsPerTile)
	return buf
}

// GPUGBufferOutputSource is the canonical WGSL definition of the GBufferOutput struct.
// Matches GPUGBufferOutput layout exactly (48 bytes, std430 aligned).
//
//go:embed assets/gbuffer-output.wgsl
var GPUGBufferOutputSource string

// GPUGBufferOutput is the GPU-aligned representation of a single G-Buffer fragment
// output. Matches the WGSL GBufferOutput struct layout exactly (see GPUGBufferOutputSource).
// This struct is written by the G-Buffer MRT fragment shader and is not typically
// uploaded from the CPU, but Marshal/Size are provided for readback and testing.
// Size: 48 bytes (3 × vec4<f32>).
type GPUGBufferOutput struct {
	Position [4]float32 // offset  0: world XYZ + linear depth in W (16 bytes)
	Normal   [4]float32 // offset 16: world normal XYZ (packed [0,1]) + roughness in W (16 bytes)
	Albedo   [4]float32 // offset 32: albedo RGB + metallic in A (16 bytes)
}

// Size returns the size of the GPUGBufferOutput struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (48)
func (g *GPUGBufferOutput) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUGBufferOutput struct into a byte buffer suitable for
// GPU readback validation or testing.
//
// Returns:
//   - []byte: 48-byte buffer matching the GPU layout
func (g *GPUGBufferOutput) Marshal() []byte {
	buf := make([]byte, 48)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(g.Position[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.Position[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.Position[2]))
	binary.LittleEndian.PutUint32(buf[12:16], math.Float32bits(g.Position[3]))
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(g.Normal[0]))
	binary.LittleEndian.PutUint32(buf[20:24], math.Float32bits(g.Normal[1]))
	binary.LittleEndian.PutUint32(buf[24:28], math.Float32bits(g.Normal[2]))
	binary.LittleEndian.PutUint32(buf[28:32], math.Float32bits(g.Normal[3]))
	binary.LittleEndian.PutUint32(buf[32:36], math.Float32bits(g.Albedo[0]))
	binary.LittleEndian.PutUint32(buf[36:40], math.Float32bits(g.Albedo[1]))
	binary.LittleEndian.PutUint32(buf[40:44], math.Float32bits(g.Albedo[2]))
	binary.LittleEndian.PutUint32(buf[44:48], math.Float32bits(g.Albedo[3]))
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
	buf := make([]byte, 176)
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

// ToGPULight converts a Light interface value into the GPU-aligned GPULight struct
// suitable for writing into the light storage buffer.
//
// Parameters:
//   - l: the Light to convert
//
// Returns:
//   - GPULight: the GPU-aligned representation
func ToGPULight(l Light) GPULight {
	shadowVal := uint32(0)
	if l.CastsShadows() {
		shadowVal = 1
	}
	return GPULight{
		Position:     l.Position(),
		LightType:    uint32(l.Type()),
		Color:        l.Color(),
		Intensity:    l.Intensity(),
		Direction:    l.Direction(),
		LightRange:   l.Range(),
		InnerCone:    l.InnerCone(),
		OuterCone:    l.OuterCone(),
		CastsShadows: shadowVal,
	}
}

// MarshalLightBuffer marshals a slice of enabled lights into a byte buffer
// suitable for GPU upload. The buffer layout is:
//
//	[GPULightHeader (16 bytes)] [GPULight × count (64 bytes each)]
//
// Only enabled lights are included, up to MaxGPULights. Lights beyond the
// budget are silently dropped; callers should pre-sort by priority if truncation
// is expected.
//
// Parameters:
//   - lights: the full slice of lights to marshal (only enabled lights are included)
//   - ambient: the scene ambient color as RGB
//
// Returns:
//   - []byte: the marshaled buffer ready for GPU upload
func MarshalLightBuffer(lights []Light, ambient [3]float32) []byte {
	headerSize := (&GPULightHeader{}).Size()
	lightSize := (&GPULight{}).Size()

	// Pre-count enabled lights to size the buffer.
	enabledCount := 0
	for _, l := range lights {
		if l.Enabled() {
			enabledCount++
			if enabledCount >= MaxGPULights {
				break
			}
		}
	}

	buf := make([]byte, headerSize+enabledCount*lightSize)

	// Write header.
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(ambient[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(ambient[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(ambient[2]))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(enabledCount))

	// Write each enabled light.
	offset := headerSize
	written := 0
	for _, l := range lights {
		if !l.Enabled() {
			continue
		}
		if written >= MaxGPULights {
			break
		}
		gpu := ToGPULight(l)
		copy(buf[offset:offset+lightSize], gpu.Marshal())
		offset += lightSize
		written++
	}

	return buf
}

// ortho builds an orthographic projection matrix compatible with WebGPU's
// clip-space convention: X/Y in [-1, 1], Z in [0, 1].
// Output is column-major.
func ortho(out []float32, left, right, bottom, top, near, far float32) {
	common.Identity(out)
	rl := right - left
	tb := top - bottom
	fn := far - near

	out[0] = 2.0 / rl
	out[5] = 2.0 / tb
	out[10] = -1.0 / fn // WebGPU Z: [0, 1]
	out[12] = -(right + left) / rl
	out[13] = -(top + bottom) / tb
	out[14] = -near / fn
}

// absF32 returns the absolute value of a float32.
func absF32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// GPUIrradianceProbeSource is the canonical WGSL definition of the IrradianceProbe struct.
// Matches GPUIrradianceProbe layout exactly (160 bytes, std430 aligned).
//
//go:embed assets/irradiance-probe.wgsl
var GPUIrradianceProbeSource string

// GPUIrradianceProbe is the GPU-aligned representation of a single irradiance
// probe storing L2 spherical harmonics (9 coefficients per RGB channel).
// Each channel is stored as an array<f32, 12> — the first 9 elements are the
// SH coefficients and the last 3 are padding to maintain 16-byte alignment.
// Matches the WGSL IrradianceProbe struct layout exactly (see GPUIrradianceProbeSource).
// Size: 160 bytes (vec4 + 3 × array<f32, 12>).
//
// Layout:
//
//	vec4<f32>       position  (16 bytes, offset   0) — world xyz + status in w
//	array<f32, 12>  sh_r      (48 bytes, offset  16) — L2 SH red (9 used + 3 pad)
//	array<f32, 12>  sh_g      (48 bytes, offset  64) — L2 SH green (9 used + 3 pad)
//	array<f32, 12>  sh_b      (48 bytes, offset 112) — L2 SH blue (9 used + 3 pad)
type GPUIrradianceProbe struct {
	Position [4]float32  // offset   0: world xyz + status flags in w (0=inactive, 1=active)
	SH_R     [12]float32 // offset  16: L2 SH coefficients for red (indices 0-8 used, 9-11 padding)
	SH_G     [12]float32 // offset  64: L2 SH coefficients for green (indices 0-8 used, 9-11 padding)
	SH_B     [12]float32 // offset 112: L2 SH coefficients for blue (indices 0-8 used, 9-11 padding)
}

// Size returns the size of the GPUIrradianceProbe struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (160)
func (p *GPUIrradianceProbe) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUIrradianceProbe struct into a byte buffer suitable
// for GPU storage buffer upload.
//
// Returns:
//   - []byte: 160-byte buffer ready for GPU upload
func (p *GPUIrradianceProbe) Marshal() []byte {
	buf := make([]byte, 160)
	off := 0
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.Position[i]))
		off += 4
	}
	for i := 0; i < 12; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.SH_R[i]))
		off += 4
	}
	for i := 0; i < 12; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.SH_G[i]))
		off += 4
	}
	for i := 0; i < 12; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.SH_B[i]))
		off += 4
	}
	return buf
}

// GPUProbeGridParamsSource is the canonical WGSL definition of the ProbeGridParams struct.
// Matches GPUProbeGridParams layout exactly (64 bytes, std430 aligned).
//
//go:embed assets/probe-grid-params.wgsl
var GPUProbeGridParamsSource string

// GPUProbeGridParams is the GPU-aligned uniform data describing the layout of
// the irradiance probe grid. The lit fragment shader uses these parameters to
// compute which probes surround a given world-space fragment position and to
// perform trilinear interpolation of SH data.
// Matches the WGSL ProbeGridParams struct layout exactly (see GPUProbeGridParamsSource).
// Size: 80 bytes (WGSL-aligned). In WGSL, vec3<u32> has alignment 16, so
// after total_probes at offset 52, the _pad field is aligned to offset 64.
// The struct's overall alignment is 16 so total size rounds to 80.
//
// Layout (WGSL byte offsets):
//
//	vec3<f32> grid_min       (12 bytes, offset  0)
//	u32       probe_count_x  ( 4 bytes, offset 12)
//	vec3<f32> grid_max       (12 bytes, offset 16)
//	u32       probe_count_y  ( 4 bytes, offset 28)
//	vec3<f32> spacing        (12 bytes, offset 32)
//	u32       probe_count_z  ( 4 bytes, offset 44)
//	u32       total_probes   ( 4 bytes, offset 48)
//	           implicit pad  (12 bytes, offset 52) — align _pad to 16
//	vec3<u32> _pad           (12 bytes, offset 64)
//	           trailing pad  ( 4 bytes, offset 76) — align struct to 16
type GPUProbeGridParams struct {
	GridMin     [3]float32 // offset  0: world-space minimum corner of the probe grid
	ProbeCountX uint32     // offset 12: number of probes along the X axis
	GridMax     [3]float32 // offset 16: world-space maximum corner of the probe grid
	ProbeCountY uint32     // offset 28: number of probes along the Y axis
	Spacing     [3]float32 // offset 32: world-space distance between adjacent probes per axis
	ProbeCountZ uint32     // offset 44: number of probes along the Z axis
	TotalProbes uint32     // offset 48: total number of probes (X × Y × Z)
	_pad        [7]uint32  // offset 52: padding to match 80-byte WGSL-aligned size
}

// Size returns the size of the GPUProbeGridParams struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (80)
func (p *GPUProbeGridParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUProbeGridParams struct into a byte buffer suitable
// for GPU uniform upload.
//
// Returns:
//   - []byte: 80-byte buffer ready for GPU upload
func (p *GPUProbeGridParams) Marshal() []byte {
	buf := make([]byte, 80)
	off := 0
	for i := 0; i < 3; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.GridMin[i]))
		off += 4
	}
	binary.LittleEndian.PutUint32(buf[off:off+4], p.ProbeCountX)
	off += 4
	for i := 0; i < 3; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.GridMax[i]))
		off += 4
	}
	binary.LittleEndian.PutUint32(buf[off:off+4], p.ProbeCountY)
	off += 4
	for i := 0; i < 3; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.Spacing[i]))
		off += 4
	}
	binary.LittleEndian.PutUint32(buf[off:off+4], p.ProbeCountZ)
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], p.TotalProbes)
	// remaining bytes (52-80) are zeroed from make
	return buf
}

// GPUProbeBakeCameraSource is the canonical WGSL definition of the ProbeBakeCamera struct.
// Matches GPUProbeBakeCamera layout exactly (80 bytes, std430 aligned).
//
//go:embed assets/probe-bake-camera.wgsl
var GPUProbeBakeCameraSource string

// GPUProbeBakeCamera is the GPU-aligned representation of the camera uniform
// used during probe cubemap baking. It contains a combined view-projection
// matrix and the probe's world-space position (used as the camera origin for
// specular calculations in the bake shader).
// Matches the WGSL ProbeBakeCamera struct layout exactly (see GPUProbeBakeCameraSource).
// Size: 80 bytes (mat4x4 + vec3 + pad).
//
// Layout:
//
//	mat4x4<f32> view_proj        (64 bytes, offset  0)
//	vec3<f32>   camera_position  (12 bytes, offset 64)
//	f32         _pad             ( 4 bytes, offset 76)
type GPUProbeBakeCamera struct {
	ViewProj       [16]float32 // offset  0: cubemap face view-projection matrix (column-major)
	CameraPosition [3]float32  // offset 64: probe world-space position
	_pad           float32     // offset 76: padding to 80-byte alignment
}

// Size returns the size of the GPUProbeBakeCamera struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (80)
func (c *GPUProbeBakeCamera) Size() int {
	return int(unsafe.Sizeof(*c))
}

// Marshal serializes the GPUProbeBakeCamera struct into a byte buffer suitable
// for GPU uniform upload.
//
// Returns:
//   - []byte: 80-byte buffer ready for GPU upload
func (c *GPUProbeBakeCamera) Marshal() []byte {
	buf := make([]byte, 80)
	off := 0
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(c.ViewProj[i]))
		off += 4
	}
	for i := 0; i < 3; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(c.CameraPosition[i]))
		off += 4
	}
	binary.LittleEndian.PutUint32(buf[off:off+4], 0) // _pad
	return buf
}

// MarshalProbeBuffer marshals a slice of irradiance probes into a byte buffer
// suitable for GPU storage buffer upload. The layout is a tightly-packed
// array of GPUIrradianceProbe structs.
//
// Parameters:
//   - probes: the slice of probes to marshal
//
// Returns:
//   - []byte: the marshaled buffer ready for GPU upload
func MarshalProbeBuffer(probes []GPUIrradianceProbe) []byte {
	probeSize := (&GPUIrradianceProbe{}).Size()
	buf := make([]byte, len(probes)*probeSize)
	for i, p := range probes {
		copy(buf[i*probeSize:(i+1)*probeSize], p.Marshal())
	}
	return buf
}

// GPUSHProjectParamsSource is the canonical WGSL definition of the SHProjectParams struct.
// Matches GPUSHProjectParams layout exactly (16 bytes, std140 aligned).
//
//go:embed assets/sh-project-params.wgsl
var GPUSHProjectParamsSource string

// GPUSHProjectParams is the GPU-aligned representation of the SH projection
// compute shader uniform parameters. It identifies the target probe, the
// cubemap face being projected, and the bake resolution.
// Matches the WGSL SHProjectParams struct layout exactly (see GPUSHProjectParamsSource).
// Size: 16 bytes.
//
// Layout:
//
//	u32 probe_index  (4 bytes, offset  0)
//	u32 face_index   (4 bytes, offset  4)
//	u32 resolution   (4 bytes, offset  8)
//	u32 _pad         (4 bytes, offset 12)
type GPUSHProjectParams struct {
	ProbeIndex uint32
	FaceIndex  uint32
	Resolution uint32
	_pad       uint32
}

// Size returns the size of the GPUSHProjectParams struct in bytes.
//
// Returns:
//   - int: the size in bytes (16)
func (p *GPUSHProjectParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUSHProjectParams struct into a byte buffer
// suitable for GPU uniform buffer upload.
//
// Returns:
//   - []byte: the marshaled buffer ready for GPU upload
func (p *GPUSHProjectParams) Marshal() []byte {
	buf := make([]byte, p.Size())
	binary.LittleEndian.PutUint32(buf[0:4], p.ProbeIndex)
	binary.LittleEndian.PutUint32(buf[4:8], p.FaceIndex)
	binary.LittleEndian.PutUint32(buf[8:12], p.Resolution)
	binary.LittleEndian.PutUint32(buf[12:16], 0) // _pad
	return buf
}

// GPUCompositionParamsSource is the canonical WGSL definition of the CompositionParams struct.
// Matches GPUCompositionParams layout exactly (16 bytes, std140 aligned).
//
//go:embed assets/composition-params.wgsl
var GPUCompositionParamsSource string

// GPUCompositionParams is the GPU-aligned uniform data for the composition
// fragment shader. Contains a flag controlling whether ACES tone mapping is
// applied and an exposure multiplier that scales HDR values before tone mapping.
// Matches the WGSL CompositionParams struct layout exactly (see GPUCompositionParamsSource).
// Size: 16 bytes.
//
// Layout:
//
//	u32 tone_mapping_enabled  (4 bytes, offset  0)
//	f32 exposure              (4 bytes, offset  4)
//	u32 _pad1                 (4 bytes, offset  8)
//	u32 _pad2                 (4 bytes, offset 12)
type GPUCompositionParams struct {
	ToneMappingEnabled uint32
	Exposure           float32
	_pad1              uint32
	_pad2              uint32
}

// Size returns the size of the GPUCompositionParams struct in bytes.
//
// Returns:
//   - int: the size in bytes (16)
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
	binary.LittleEndian.PutUint32(buf[8:12], 0)  // _pad1
	binary.LittleEndian.PutUint32(buf[12:16], 0) // _pad2
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
	buf := make([]byte, 224)
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
