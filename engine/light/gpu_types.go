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
//go:embed assets/light_header.wgsl
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
// Matches GPUShadowData layout exactly (80 bytes, std430 aligned).
//
//go:embed assets/shadow_data.wgsl
var GPUShadowDataSource string

// GPUShadowData is the GPU-aligned representation of directional shadow data.
// Matches the WGSL ShadowData struct layout exactly (see GPUShadowDataSource).
// Size: 80 bytes (std430 / WGSL aligned).
//
// Layout:
//
//	mat4x4<f32> light_vp       (64 bytes, offset 0)
//	vec2<f32>   texel_size     ( 8 bytes, offset 64)
//	f32         bias           ( 4 bytes, offset 72)
//	f32         normal_bias    ( 4 bytes, offset 76)
type GPUShadowData struct {
	LightVP    [16]float32 // orthographic view-projection from light's perspective
	TexelSize  [2]float32  // 1.0 / shadow_map_resolution for PCF offset calculations
	Bias       float32     // depth comparison bias to reduce shadow acne
	NormalBias float32     // world-space normal-offset distance for shadow lookup
}

// Size returns the size of the GPUShadowData struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (80)
func (s *GPUShadowData) Size() int {
	return int(unsafe.Sizeof(*s))
}

// ComputeDirectionalLightVP builds an orthographic view-projection matrix for a
// directional light's shadow pass and stores it in the receiver's LightVP field.
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
//   - []byte: 80-byte buffer ready for GPU upload
func (s *GPUShadowData) Marshal() []byte {
	buf := make([]byte, 80)
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(s.LightVP[i]))
	}
	binary.LittleEndian.PutUint32(buf[64:68], math.Float32bits(s.TexelSize[0]))
	binary.LittleEndian.PutUint32(buf[68:72], math.Float32bits(s.TexelSize[1]))
	binary.LittleEndian.PutUint32(buf[72:76], math.Float32bits(s.Bias))
	binary.LittleEndian.PutUint32(buf[76:80], math.Float32bits(s.NormalBias))
	return buf
}

// GPUShadowUniformSource is the canonical WGSL definition of the ShadowUniform struct.
// Matches GPUShadowUniform layout exactly (64 bytes, std430 aligned).
//
//go:embed assets/shadow_uniform.wgsl
var GPUShadowUniformSource string

// GPUShadowUniform is the GPU-aligned representation of the shadow vertex
// shader uniform containing only the light view-projection matrix.
// Matches the WGSL ShadowUniform struct layout exactly (see GPUShadowUniformSource).
// Size: 64 bytes (mat4x4<f32>).
type GPUShadowUniform struct {
	LightVP [16]float32 // orthographic view-projection from light's perspective
}

// Size returns the size of the GPUShadowUniform struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (64)
func (u *GPUShadowUniform) Size() int {
	return int(unsafe.Sizeof(*u))
}

// Marshal serializes the GPUShadowUniform struct into a byte buffer suitable for
// GPU uniform upload.
//
// Returns:
//   - []byte: 64-byte buffer ready for GPU upload
func (u *GPUShadowUniform) Marshal() []byte {
	buf := make([]byte, 64)
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(u.LightVP[i]))
	}
	return buf
}

// GPULightCullUniformsSource is the canonical WGSL definition of the LightCullUniforms struct.
// Matches GPULightCullUniforms layout exactly (160 bytes, std430 aligned).
//
//go:embed assets/light_cull_uniforms.wgsl
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
//go:embed assets/tile_uniforms.wgsl
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
