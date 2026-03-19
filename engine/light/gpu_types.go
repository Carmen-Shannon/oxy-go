package light

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"

	"github.com/Carmen-Shannon/oxy-go/common"
)

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
	ShadowIndex  uint32     // offset 60: index into light shadow data buffer, 0xFFFFFFFF = no shadow
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
	binary.LittleEndian.PutUint32(buf[60:64], g.ShadowIndex)
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

// GPUShadowUniformSource is the canonical WGSL definition of the ShadowUniform struct.
// Matches GPUShadowUniform layout exactly (64 bytes, std430 aligned).
//
//go:embed assets/shadow-uniform.wgsl
var GPUShadowUniformSource string

// GPUShadowUniform is the GPU-aligned representation of the shadow vertex
// shader uniform containing the light view-projection matrix.
// Matches the WGSL ShadowUniform struct layout exactly (see GPUShadowUniformSource).
// Size: 64 bytes (mat4x4).
//
// Layout:
//
//	mat4x4<f32> light_vp (64 bytes, offset 0)
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

// GPUCSMDataSource is the canonical WGSL definition of the CSMData and CSMCascade structs.
// Matches GPUCSMData layout exactly (32-byte header + 80 bytes per cascade).
//
//go:embed assets/csm-data.wgsl
var GPUCSMDataSource string

// GPUCSMCascade holds per-cascade GPU data for Cascaded Shadow Maps.
// Layout: 80 bytes (std430).
//
//	Offset  0: LightVP    [16]float32  (64 bytes)
//	Offset 64: ShadowNear float32      ( 4 bytes)
//	Offset 68: ShadowFar  float32      ( 4 bytes)
//	Offset 72: CamFar     float32      ( 4 bytes)
//	Offset 76: NormalBias float32      ( 4 bytes)
type GPUCSMCascade struct {
	LightVP    [16]float32
	ShadowNear float32
	ShadowFar  float32
	CamFar     float32
	NormalBias float32
}

// Size returns the byte size of GPUCSMCascade (80 bytes).
func (c *GPUCSMCascade) Size() int {
	return int(unsafe.Sizeof(*c))
}

// Marshal serialises GPUCSMCascade into an 80-byte little-endian buffer.
func (c *GPUCSMCascade) Marshal() []byte {
	buf := make([]byte, 80)
	off := 0
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(c.LightVP[i]))
		off += 4
	}
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(c.ShadowNear))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(c.ShadowFar))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(c.CamFar))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(c.NormalBias))
	return buf
}

// GPUCSMData is the lit-pass uniform buffer for the dual-cascade shadow system.
// It contains a 32-byte header followed by exactly 2 GPUCSMCascade blocks.
// Layout: 32 + 2×80 = 192 bytes.
//
//	Header (32 bytes, offset 0):
//	  Offset  0: TexelSize         [2]float32  ( 8 bytes)
//	  Offset  8: Bias              float32     ( 4 bytes)
//	  Offset 12: InnerRadius       float32     ( 4 bytes)
//	  Offset 16: PCFRadius         float32     ( 4 bytes)
//	  Offset 20: ShadowMaxDistance float32     ( 4 bytes)
//	  Offset 24: _pad0             float32     ( 4 bytes)
//	  Offset 28: _pad1             float32     ( 4 bytes)
//
//	Per-cascade blocks follow at offset 32 + i*80.
type GPUCSMData struct {
	TexelSize         [2]float32
	Bias              float32
	InnerRadius       float32
	PCFRadius         float32
	ShadowMaxDistance float32
	_pad0             float32
	_pad1             float32
	Cascades          [2]GPUCSMCascade
}

// Size returns the byte size of the marshalled GPUCSMData buffer (always 192 bytes).
func (g *GPUCSMData) Size() int {
	return 32 + 2*(&GPUCSMCascade{}).Size()
}

// Marshal serialises GPUCSMData into a little-endian byte buffer of g.Size() bytes.
// The header fields are written first, followed by each GPUCSMCascade block in order.
func (g *GPUCSMData) Marshal() []byte {
	buf := make([]byte, g.Size())
	off := 0

	// Header
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(g.TexelSize[0]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(g.TexelSize[1]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(g.Bias))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(g.InnerRadius))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(g.PCFRadius))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(g.ShadowMaxDistance))
	off += 4
	// 2 padding floats (zeroed from make)
	off += 8

	// Per-cascade blocks
	for i := range g.Cascades {
		cb := g.Cascades[i].Marshal()
		copy(buf[off:off+len(cb)], cb)
		off += len(cb)
	}

	return buf
}

// ComputeCascades builds the two per-cascade light-space matrices and depth
// ranges for the dual-cascade shadow system:
//   - Cascade 0 (inner): camera-centered sphere with radius innerRadius,
//     providing constant texel density regardless of zoom level.
//   - Cascade 1 (outer): frustum-fit covering the full depth range
//     [camNear, camFar] at lower resolution.
//
// Parameters:
//   - lightDir:         normalised direction toward the light source
//   - camNear/camFar:   camera view depth range for the outer cascade
//   - camFov:           camera vertical field of view in radians
//   - camAspect:        camera aspect ratio (width / height)
//   - camViewMatrix:    column-major world-to-camera view matrix
//   - cameraPosition:   camera world-space position
//   - innerRadius:      world-space radius for the high-fidelity inner cascade
//   - normalBiasScale:  multiplier applied to per-texel world size for normal-offset bias
//   - resolution:       shadow map texel resolution (width == height of one cascade slice)
func (g *GPUCSMData) ComputeCascades(
	lightDir [3]float32,
	camNear, camFar float32,
	camFov, camAspect float32,
	camViewMatrix [16]float32,
	cameraPosition [3]float32,
	innerRadius float32,
	normalBiasScale float32,
	resolution int,
) {
	upX, upY, upZ := float32(0), float32(1), float32(0)
	if absF32(lightDir[1]) > 0.99 {
		upX, upY, upZ = 1, 0, 0
	}
	nearPad := float32(1.0)

	// ── Cascade 0: camera-centred sphere ───────────────────────────────
	{
		cx, cy, cz := cameraPosition[0], cameraPosition[1], cameraPosition[2]
		sphereRadius := innerRadius

		sceneDepthPadding := sphereRadius * 0.5
		if sceneDepthPadding < 5.0 {
			sceneDepthPadding = 5.0
		}

		eyeOffset := nearPad + sphereRadius + sceneDepthPadding
		eyeX := cx - lightDir[0]*eyeOffset
		eyeY := cy - lightDir[1]*eyeOffset
		eyeZ := cz - lightDir[2]*eyeOffset

		shadowNear := nearPad
		shadowFar := nearPad + 2.0*sphereRadius + sceneDepthPadding

		var lightView [16]float32
		common.LookAt(lightView[:], eyeX, eyeY, eyeZ, cx, cy, cz, upX, upY, upZ)

		worldTexelSize := (2.0 * sphereRadius) / float32(resolution)

		centR := lightView[0]*cx + lightView[4]*cy + lightView[8]*cz
		centU := lightView[1]*cx + lightView[5]*cy + lightView[9]*cz
		snapR, snapU := centR, centU
		if worldTexelSize > 0 {
			snapR = float32(math.Round(float64(centR/worldTexelSize))) * worldTexelSize
			snapU = float32(math.Round(float64(centU/worldTexelSize))) * worldTexelSize
		}
		dr := snapR - centR
		du := snapU - centU

		lightRx, lightRy, lightRz := lightView[0], lightView[4], lightView[8]
		lightUx, lightUy, lightUz := lightView[1], lightView[5], lightView[9]
		snapCx := cx + dr*lightRx + du*lightUx
		snapCy := cy + dr*lightRy + du*lightUy
		snapCz := cz + dr*lightRz + du*lightUz
		snapEyeX := eyeX + dr*lightRx + du*lightUx
		snapEyeY := eyeY + dr*lightRy + du*lightUy
		snapEyeZ := eyeZ + dr*lightRz + du*lightUz

		common.LookAt(lightView[:], snapEyeX, snapEyeY, snapEyeZ, snapCx, snapCy, snapCz, upX, upY, upZ)

		var proj [16]float32
		ortho(proj[:], -sphereRadius, sphereRadius, -sphereRadius, sphereRadius, shadowNear, shadowFar)
		common.Mul4(g.Cascades[0].LightVP[:], proj[:], lightView[:])

		g.Cascades[0].ShadowNear = shadowNear
		g.Cascades[0].ShadowFar = shadowFar
		g.Cascades[0].CamFar = innerRadius
		g.Cascades[0].NormalBias = worldTexelSize * normalBiasScale
	}

	// ── Cascade 1: frustum-fit outer cascade ───────────────────────────
	{
		var invView [16]float32
		if !common.Invert4(invView[:], camViewMatrix[:]) {
			common.Identity(invView[:])
		}

		tanHalfFovY := float32(math.Tan(float64(camFov * 0.5)))
		tanHalfFovX := tanHalfFovY * camAspect

		near := camNear
		far := camFar

		nW := near * tanHalfFovX
		nH := near * tanHalfFovY
		fW := far * tanHalfFovX
		fH := far * tanHalfFovY

		viewCorners := [8][3]float32{
			{-nW, -nH, -near}, {nW, -nH, -near}, {-nW, nH, -near}, {nW, nH, -near},
			{-fW, -fH, -far}, {fW, -fH, -far}, {-fW, fH, -far}, {fW, fH, -far},
		}

		var worldCorners [8][3]float32
		for j, vc := range viewCorners {
			worldCorners[j] = transformPoint(invView[:], vc)
		}

		var cx, cy, cz float32
		for _, wc := range worldCorners {
			cx += wc[0]
			cy += wc[1]
			cz += wc[2]
		}
		cx /= 8
		cy /= 8
		cz /= 8

		minProj := float32(math.Inf(1))
		maxProj := float32(math.Inf(-1))
		for _, wc := range worldCorners {
			p := (wc[0]-cx)*lightDir[0] + (wc[1]-cy)*lightDir[1] + (wc[2]-cz)*lightDir[2]
			if p < minProj {
				minProj = p
			}
			if p > maxProj {
				maxProj = p
			}
		}

		sceneDepthPadding := (maxProj - minProj) * 0.5
		if sceneDepthPadding < 5.0 {
			sceneDepthPadding = 5.0
		}

		eyeOffset := nearPad - minProj + sceneDepthPadding
		eyeX := cx - lightDir[0]*eyeOffset
		eyeY := cy - lightDir[1]*eyeOffset
		eyeZ := cz - lightDir[2]*eyeOffset

		shadowNear := nearPad
		shadowFar := nearPad + (maxProj - minProj) + sceneDepthPadding

		var lightView [16]float32
		common.LookAt(lightView[:], eyeX, eyeY, eyeZ, cx, cy, cz, upX, upY, upZ)

		var sphereRadius float32
		for _, wc := range worldCorners {
			dx := wc[0] - cx
			dy := wc[1] - cy
			dz := wc[2] - cz
			r := float32(math.Sqrt(float64(dx*dx + dy*dy + dz*dz)))
			if r > sphereRadius {
				sphereRadius = r
			}
		}
		if sphereRadius < 0.5 {
			sphereRadius = 0.5
		}

		worldTexelSize := (2.0 * sphereRadius) / float32(resolution)

		centR := lightView[0]*cx + lightView[4]*cy + lightView[8]*cz
		centU := lightView[1]*cx + lightView[5]*cy + lightView[9]*cz
		snapR, snapU := centR, centU
		if worldTexelSize > 0 {
			snapR = float32(math.Round(float64(centR/worldTexelSize))) * worldTexelSize
			snapU = float32(math.Round(float64(centU/worldTexelSize))) * worldTexelSize
		}
		dr := snapR - centR
		du := snapU - centU

		lightRx, lightRy, lightRz := lightView[0], lightView[4], lightView[8]
		lightUx, lightUy, lightUz := lightView[1], lightView[5], lightView[9]
		snapCx := cx + dr*lightRx + du*lightUx
		snapCy := cy + dr*lightRy + du*lightUy
		snapCz := cz + dr*lightRz + du*lightUz
		snapEyeX := eyeX + dr*lightRx + du*lightUx
		snapEyeY := eyeY + dr*lightRy + du*lightUy
		snapEyeZ := eyeZ + dr*lightRz + du*lightUz

		common.LookAt(lightView[:], snapEyeX, snapEyeY, snapEyeZ, snapCx, snapCy, snapCz, upX, upY, upZ)

		var proj [16]float32
		ortho(proj[:], -sphereRadius, sphereRadius, -sphereRadius, sphereRadius, shadowNear, shadowFar)
		common.Mul4(g.Cascades[1].LightVP[:], proj[:], lightView[:])

		g.Cascades[1].ShadowNear = shadowNear
		g.Cascades[1].ShadowFar = shadowFar
		g.Cascades[1].CamFar = camFar
		g.Cascades[1].NormalBias = worldTexelSize * normalBiasScale
	}
}

//go:embed assets/light-shadow-entry.wgsl
var GPULightShadowEntrySource string

// GPULightShadowEntry holds per-light shadow map data for spot and point lights.
// Spot lights use 1 entry; point lights use 6 consecutive entries (one per cube face).
// Size: 96 bytes (16-byte aligned).
type GPULightShadowEntry struct {
	LightVP    [16]float32 // offset  0: light view-projection matrix (64 bytes)
	AtlasRect  [4]float32  // offset 64: (u_offset, v_offset, u_scale, v_scale) in atlas UV space (16 bytes)
	Bias       float32     // offset 80: depth comparison bias
	Near       float32     // offset 84: shadow near plane
	Far        float32     // offset 88: shadow far plane
	ShadowType ShadowType  // offset 92: 0=spot, 1=cube_face
}

// Size returns the size of the GPULightShadowEntry struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (96)
func (g *GPULightShadowEntry) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPULightShadowEntry struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 96-byte buffer ready for GPU upload
func (g *GPULightShadowEntry) Marshal() []byte {
	buf := make([]byte, 96)
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(g.LightVP[i]))
	}
	binary.LittleEndian.PutUint32(buf[64:68], math.Float32bits(g.AtlasRect[0]))
	binary.LittleEndian.PutUint32(buf[68:72], math.Float32bits(g.AtlasRect[1]))
	binary.LittleEndian.PutUint32(buf[72:76], math.Float32bits(g.AtlasRect[2]))
	binary.LittleEndian.PutUint32(buf[76:80], math.Float32bits(g.AtlasRect[3]))
	binary.LittleEndian.PutUint32(buf[80:84], math.Float32bits(g.Bias))
	binary.LittleEndian.PutUint32(buf[84:88], math.Float32bits(g.Near))
	binary.LittleEndian.PutUint32(buf[88:92], math.Float32bits(g.Far))
	binary.LittleEndian.PutUint32(buf[92:96], uint32(g.ShadowType))
	return buf
}

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
	buf := make([]byte, 24)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(p.Direction[0]))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(p.Direction[1]))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(p.Radius))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(p.GBufferScale))
	binary.LittleEndian.PutUint32(buf[16:20], uint32(p.CascadeWidth))
	binary.LittleEndian.PutUint32(buf[20:24], uint32(p._pad))
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
// Matches GPUTileUniforms layout exactly (16 bytes).
//
//go:embed assets/tile-uniforms.wgsl
var GPUTileUniformsSource string

// GPUTileUniforms is the GPU-aligned uniform data read by the lit fragment
// shader to compute which tile a fragment belongs to and index into the
// per-tile light list buffer.
// Matches the WGSL TileUniforms struct layout exactly (see GPUTileUniformsSource).
// Size: 16 bytes.
//
// Layout:
//
//	u32 tile_count_x        (4 bytes, offset  0)
//	u32 max_lights_per_tile (4 bytes, offset  4)
//	u32 screen_width        (4 bytes, offset  8)
//	u32 screen_height       (4 bytes, offset 12)
type GPUTileUniforms struct {
	TileCountX       uint32
	MaxLightsPerTile uint32
	ScreenWidth      uint32
	ScreenHeight     uint32
}

// Size returns the size of the GPUTileUniforms struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (16)
func (u *GPUTileUniforms) Size() int {
	return int(unsafe.Sizeof(*u))
}

// Marshal serializes GPUTileUniforms into a 16-byte little-endian buffer suitable
// for GPU upload.
//
// Returns:
//   - []byte: 16-byte buffer ready for GPU upload
func (u *GPUTileUniforms) Marshal() []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint32(buf[0:4], u.TileCountX)
	binary.LittleEndian.PutUint32(buf[4:8], u.MaxLightsPerTile)
	binary.LittleEndian.PutUint32(buf[8:12], u.ScreenWidth)
	binary.LittleEndian.PutUint32(buf[12:16], u.ScreenHeight)
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
		ShadowIndex:  0xFFFFFFFF,
	}
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

// transformPoint multiplies a column-major 4×4 matrix m by a 3D point p (w=1)
// and returns the transformed xyz components.
func transformPoint(m []float32, p [3]float32) [3]float32 {
	x := m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12]
	y := m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13]
	z := m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14]
	return [3]float32{x, y, z}
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

// GPUContactShadowParamsSource is the canonical WGSL definition of the ContactShadowParams struct.
// Matches GPUContactShadowParams layout exactly (176 bytes, std430 aligned).
//
//go:embed assets/contact-shadow-params.wgsl
var GPUContactShadowParamsSource string

// GPUContactShadowParams is the GPU-aligned uniform data for the contact
// shadow compute shader. Contains the view-projection and inverse
// view-projection matrices, the directional light direction, ray march
// parameters, screen dimensions, and camera position.
// Matches the WGSL ContactShadowParams struct layout exactly (see GPUContactShadowParamsSource).
// Size: 176 bytes (std430 / WGSL aligned).
//
// Layout:
//
//	mat4x4<f32> view_proj         (64 bytes, offset   0)
//	mat4x4<f32> inv_view_proj     (64 bytes, offset  64)
//	vec3<f32>   light_direction   (12 bytes, offset 128)
//	u32         step_count        ( 4 bytes, offset 140)
//	f32         max_distance      ( 4 bytes, offset 144)
//	f32         thickness         ( 4 bytes, offset 148)
//	f32         screen_width      ( 4 bytes, offset 152)
//	f32         screen_height     ( 4 bytes, offset 156)
//	vec3<f32>   camera_position   (12 bytes, offset 160)
//	f32         _pad              ( 4 bytes, offset 172)
type GPUContactShadowParams struct {
	ViewProj       [16]float32 // offset   0: view-projection matrix (column-major)
	InvViewProj    [16]float32 // offset  64: inverse view-projection matrix (column-major)
	LightDirection [3]float32  // offset 128: directional light direction (world space)
	StepCount      uint32      // offset 140: number of ray march steps
	MaxDistance    float32     // offset 144: max ray march distance in world units
	Thickness      float32     // offset 148: depth thickness tolerance in NDC depth space
	ScreenWidth    float32     // offset 152: output texture width in pixels
	ScreenHeight   float32     // offset 156: output texture height in pixels
	CameraPosition [3]float32  // offset 160: world-space camera position
	Pad            float32     // offset 172: padding to 176-byte alignment
}

// Size returns the size of the GPUContactShadowParams struct in bytes.
//
// Returns:
//   - int: the struct size in bytes (176)
func (p *GPUContactShadowParams) Size() int {
	return int(unsafe.Sizeof(*p))
}

// Marshal serializes the GPUContactShadowParams struct into a byte buffer
// suitable for GPU upload.
//
// Returns:
//   - []byte: 176-byte buffer ready for GPU upload
func (p *GPUContactShadowParams) Marshal() []byte {
	buf := make([]byte, 176)
	off := 0
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.ViewProj[i]))
		off += 4
	}
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.InvViewProj[i]))
		off += 4
	}
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.LightDirection[0]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.LightDirection[1]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.LightDirection[2]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], p.StepCount)
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.MaxDistance))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.Thickness))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.ScreenWidth))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.ScreenHeight))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.CameraPosition[0]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.CameraPosition[1]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(p.CameraPosition[2]))
	off += 4
	binary.LittleEndian.PutUint32(buf[off:off+4], 0) // _pad
	return buf
}
