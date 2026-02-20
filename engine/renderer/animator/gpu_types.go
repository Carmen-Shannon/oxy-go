package animator

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"
)

// GPUInstanceDataSource is the canonical WGSL definition of the InstanceData struct.
// Matches GPUInstanceData layout exactly (64 bytes, std430 aligned).
//
//go:embed assets/instance_data.wgsl
var GPUInstanceDataSource string

// GPUInstanceData is the GPU-aligned representation of per-instance data for models.
// Size: 64 bytes (std430 aligned).
type GPUInstanceData struct {
	Model [16]float32 // offset 0, size 64 (mat4x4<f32>)
}

// Size returns the size of the GPUInstanceData struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUInstanceData) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUInstanceData struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 64-byte buffer ready for GPU upload.
func (g *GPUInstanceData) Marshal() []byte {
	buf := make([]byte, 64)
	for i := range 16 {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(g.Model[i]))
	}
	return buf
}

// GPUAnimationDataSource is the canonical WGSL definition of the AnimationData struct.
// Matches GPUAnimationData layout exactly (64 bytes, std430 aligned).
//
//go:embed assets/animation_data.wgsl
var GPUAnimationDataSource string

// GPUAnimationData is the GPU-aligned representation of per-instance animation data for the simple animator.
// Matches the WGSL AnimationData struct layout exactly (see GPUAnimationDataSource).
// Size: 64 bytes (16 floats = 4 × vec4, std430 aligned).
type GPUAnimationData struct {
	RotSpeed [3]float32 // offset 0: rotation speed around X, Y, Z axes (radians per frame)
	_pad0    float32    // offset 12: implicit vec3 pad
	Rot      [3]float32 // offset 16: current rotation angles around X, Y, Z axes
	_pad1    float32    // offset 28: implicit vec3 pad
	Pos      [3]float32 // offset 32: position X, Y, Z
	_pad2    float32    // offset 44: implicit vec3 pad
	Scale    [3]float32 // offset 48: scale X, Y, Z
	_pad3    float32    // offset 60: implicit vec3 pad
}

// Size returns the size of the GPUAnimationData struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUAnimationData) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUAnimationData struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 64-byte buffer ready for GPU upload.
func (g *GPUAnimationData) Marshal() []byte {
	buf := make([]byte, 64)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(g.RotSpeed[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.RotSpeed[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.RotSpeed[2]))
	binary.LittleEndian.PutUint32(buf[12:16], 0) // _pad0
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(g.Rot[0]))
	binary.LittleEndian.PutUint32(buf[20:24], math.Float32bits(g.Rot[1]))
	binary.LittleEndian.PutUint32(buf[24:28], math.Float32bits(g.Rot[2]))
	binary.LittleEndian.PutUint32(buf[28:32], 0) // _pad1
	binary.LittleEndian.PutUint32(buf[32:36], math.Float32bits(g.Pos[0]))
	binary.LittleEndian.PutUint32(buf[36:40], math.Float32bits(g.Pos[1]))
	binary.LittleEndian.PutUint32(buf[40:44], math.Float32bits(g.Pos[2]))
	binary.LittleEndian.PutUint32(buf[44:48], 0) // _pad2
	binary.LittleEndian.PutUint32(buf[48:52], math.Float32bits(g.Scale[0]))
	binary.LittleEndian.PutUint32(buf[52:56], math.Float32bits(g.Scale[1]))
	binary.LittleEndian.PutUint32(buf[56:60], math.Float32bits(g.Scale[2]))
	binary.LittleEndian.PutUint32(buf[60:64], 0) // _pad3
	return buf
}

// GPUFrustumPlaneSource is the canonical WGSL definition of the FrustumPlane struct.
// Matches GPUFrustumPlane layout exactly (16 bytes, std430 aligned).
//
//go:embed assets/frustum_plane.wgsl
var GPUFrustumPlaneSource string

// GPUFrustumPlane is the GPU-aligned representation of a single view-frustum plane.
// Matches the WGSL FrustumPlane struct layout exactly (see GPUFrustumPlaneSource).
// Size: 16 bytes (vec3 normal + f32 distance, std430 aligned).
type GPUFrustumPlane struct {
	Normal   [3]float32 // offset 0: plane normal (x, y, z)
	Distance float32    // offset 12: signed distance from origin
}

// Size returns the size of the GPUFrustumPlane struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUFrustumPlane) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUFrustumPlane struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 16-byte buffer ready for GPU upload.
func (g *GPUFrustumPlane) Marshal() []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(g.Normal[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.Normal[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.Normal[2]))
	binary.LittleEndian.PutUint32(buf[12:16], math.Float32bits(g.Distance))
	return buf
}

// GPUGlobalDataSource is the canonical WGSL definition of the GlobalData struct.
// Matches GPUGlobalData layout exactly (112 bytes, std430 aligned).
//
//go:embed assets/simple_globals.wgsl
var GPUGlobalDataSource string

// GPUGlobalData is the GPU-aligned per-frame uniform for the frustum-culled simple compute shader.
// Matches the WGSL GlobalData struct layout exactly (see GPUGlobalDataSource).
// Size: 112 bytes (instance_count u32 + delta_time f32 + bounding_radius f32 + pad + 6 × GPUFrustumPlane).
type GPUGlobalData struct {
	InstanceCount  uint32             // offset 0
	DeltaTime      float32            // offset 4
	BoundingRadius float32            // offset 8
	_padding       float32            // offset 12: pad to 16 bytes before planes array
	Planes         [6]GPUFrustumPlane // offset 16: 6 × 16 bytes = 96 bytes
}

// Size returns the size of the GPUGlobalData struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUGlobalData) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUGlobalData struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 112-byte buffer ready for GPU upload.
func (g *GPUGlobalData) Marshal() []byte {
	buf := make([]byte, 112)
	binary.LittleEndian.PutUint32(buf[0:4], g.InstanceCount)
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.DeltaTime))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.BoundingRadius))
	binary.LittleEndian.PutUint32(buf[12:16], 0) // _padding
	off := 16
	for i := range 6 {
		p := g.Planes[i]
		binary.LittleEndian.PutUint32(buf[off+0:off+4], math.Float32bits(p.Normal[0]))
		binary.LittleEndian.PutUint32(buf[off+4:off+8], math.Float32bits(p.Normal[1]))
		binary.LittleEndian.PutUint32(buf[off+8:off+12], math.Float32bits(p.Normal[2]))
		binary.LittleEndian.PutUint32(buf[off+12:off+16], math.Float32bits(p.Distance))
		off += 16
	}
	return buf
}

// GPUAnimationGlobalsSource is the canonical WGSL definition of the AnimationGlobals struct.
// Matches GPUAnimationGlobals layout exactly (128 bytes, std430 aligned).
//
//go:embed assets/animation_globals.wgsl
var GPUAnimationGlobalsSource string

// GPUAnimationGlobals is the GPU-aligned per-frame uniform for the frustum-culled skeletal compute shader.
// Matches the WGSL AnimationGlobals struct layout exactly (see GPUAnimationGlobalsSource).
// Size: 128 bytes (8 u32/f32 fields + 6 × GPUFrustumPlane).
type GPUAnimationGlobals struct {
	InstanceCount      uint32             // offset 0
	BoneCount          uint32             // offset 4
	BoundingRadius     float32            // offset 8
	ChannelDataOffset  uint32             // offset 12: u32 index into anim_packed where channel headers start
	KeyframeDataOffset uint32             // offset 16: u32 index into anim_packed where keyframes start
	_pad1              uint32             // offset 20
	_pad2              uint32             // offset 24
	_pad3              uint32             // offset 28
	Planes             [6]GPUFrustumPlane // offset 32: 6 × 16 bytes = 96 bytes
}

// Size returns the size of the GPUAnimationGlobals struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUAnimationGlobals) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUAnimationGlobals struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 128-byte buffer ready for GPU upload.
func (g *GPUAnimationGlobals) Marshal() []byte {
	buf := make([]byte, 128)
	binary.LittleEndian.PutUint32(buf[0:4], g.InstanceCount)
	binary.LittleEndian.PutUint32(buf[4:8], g.BoneCount)
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.BoundingRadius))
	binary.LittleEndian.PutUint32(buf[12:16], g.ChannelDataOffset)
	binary.LittleEndian.PutUint32(buf[16:20], g.KeyframeDataOffset)
	binary.LittleEndian.PutUint32(buf[20:24], 0) // _pad1
	binary.LittleEndian.PutUint32(buf[24:28], 0) // _pad2
	binary.LittleEndian.PutUint32(buf[28:32], 0) // _pad3
	off := 32
	for i := range 6 {
		p := g.Planes[i]
		binary.LittleEndian.PutUint32(buf[off+0:off+4], math.Float32bits(p.Normal[0]))
		binary.LittleEndian.PutUint32(buf[off+4:off+8], math.Float32bits(p.Normal[1]))
		binary.LittleEndian.PutUint32(buf[off+8:off+12], math.Float32bits(p.Normal[2]))
		binary.LittleEndian.PutUint32(buf[off+12:off+16], math.Float32bits(p.Distance))
		off += 16
	}
	return buf
}

// GPUIndirectArgsSource is the canonical WGSL definition of the IndirectArgs struct.
// Matches GPUIndirectArgs layout exactly (20 bytes).
//
//go:embed assets/indirect_args.wgsl
var GPUIndirectArgsSource string

// GPUIndirectArgs is the GPU-aligned DrawIndexedIndirect arguments written by the compute shader.
// Matches the WGSL IndirectArgs struct layout exactly (see GPUIndirectArgsSource).
// Size: 20 bytes (5 × u32).
type GPUIndirectArgs struct {
	IndexCount    uint32 // offset 0: number of indices per instance
	InstanceCount uint32 // offset 4: number of visible instances (written by compute shader)
	FirstIndex    uint32 // offset 8: offset into the index buffer
	BaseVertex    int32  // offset 12: added to each index value (signed)
	FirstInstance uint32 // offset 16: first instance ID
}

// Size returns the size of the GPUIndirectArgs struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUIndirectArgs) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUIndirectArgs struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 20-byte buffer ready for GPU upload.
func (g *GPUIndirectArgs) Marshal() []byte {
	buf := make([]byte, 20)
	binary.LittleEndian.PutUint32(buf[0:4], g.IndexCount)
	binary.LittleEndian.PutUint32(buf[4:8], g.InstanceCount)
	binary.LittleEndian.PutUint32(buf[8:12], g.FirstIndex)
	binary.LittleEndian.PutUint32(buf[12:16], uint32(g.BaseVertex))
	binary.LittleEndian.PutUint32(buf[16:20], g.FirstInstance)
	return buf
}

// GPUBoneInfoSource is the canonical WGSL definition of the BoneInfo struct.
// Matches GPUBoneInfo layout exactly (112 bytes, std430 aligned).
//
//go:embed assets/bone_info.wgsl
var GPUBoneInfoSource string

// GPUBoneInfo is the GPU-aligned representation of a single bone in the skeleton.
// Matches the WGSL BoneInfo struct layout exactly (see GPUBoneInfoSource).
// Size: 112 bytes (std430 aligned).
type GPUBoneInfo struct {
	InverseBindMatrix [16]float32 // offset 0, size 64 (mat4x4<f32>)
	LocalTranslation  [3]float32  // offset 64, size 12 (vec3<f32>)
	ParentIndex       int32       // offset 76, size 4 (fills vec3 gap)
	LocalScale        [3]float32  // offset 80, size 12 (vec3<f32>)
	_padScale         float32     // offset 92, size 4 (align vec4 to 16)
	LocalRotation     [4]float32  // offset 96, size 16 (vec4<f32>, quaternion)
}

// Size returns the size of the GPUBoneInfo struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUBoneInfo) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUBoneInfo struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 112-byte buffer ready for GPU upload.
func (g *GPUBoneInfo) Marshal() []byte {
	buf := make([]byte, 112)
	for i := range 16 {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(g.InverseBindMatrix[i]))
	}
	binary.LittleEndian.PutUint32(buf[64:68], math.Float32bits(g.LocalTranslation[0]))
	binary.LittleEndian.PutUint32(buf[68:72], math.Float32bits(g.LocalTranslation[1]))
	binary.LittleEndian.PutUint32(buf[72:76], math.Float32bits(g.LocalTranslation[2]))
	binary.LittleEndian.PutUint32(buf[76:80], uint32(g.ParentIndex))
	binary.LittleEndian.PutUint32(buf[80:84], math.Float32bits(g.LocalScale[0]))
	binary.LittleEndian.PutUint32(buf[84:88], math.Float32bits(g.LocalScale[1]))
	binary.LittleEndian.PutUint32(buf[88:92], math.Float32bits(g.LocalScale[2]))
	binary.LittleEndian.PutUint32(buf[92:96], 0) // _padScale
	binary.LittleEndian.PutUint32(buf[96:100], math.Float32bits(g.LocalRotation[0]))
	binary.LittleEndian.PutUint32(buf[100:104], math.Float32bits(g.LocalRotation[1]))
	binary.LittleEndian.PutUint32(buf[104:108], math.Float32bits(g.LocalRotation[2]))
	binary.LittleEndian.PutUint32(buf[108:112], math.Float32bits(g.LocalRotation[3]))
	return buf
}

// GPUKeyFrame is the GPU-aligned representation of a single animation keyframe.
// Size: 64 bytes (std430 aligned).
//
// Layout:
//
//	f32         time            ( 4 bytes, offset  0)
//	vec3<f32>   _pad0           (12 bytes, offset  4)
//	vec3<f32>   translation     (12 bytes, offset 16)
//	f32         _pad1           ( 4 bytes, offset 28)
//	vec4<f32>   rotation        (16 bytes, offset 32)
//	vec3<f32>   scale           (12 bytes, offset 48)
//	f32         _pad2           ( 4 bytes, offset 60)
type GPUKeyFrame struct {
	Time        float32    // offset 0, size 4 (f32)
	_pad0       [3]float32 // offset 4, size 12 (vec3 pad)
	Translation [3]float32 // offset 16, size 12 (vec3<f32>)
	_pad1       float32    // offset 28, size 4 (align vec4 to 16)
	Rotation    [4]float32 // offset 32, size 16 (vec4<f32>, quaternion)
	Scale       [3]float32 // offset 48, size 12 (vec3<f32>)
	_pad2       float32    // offset 60, size 4 (align vec4 to 16)
}

// Size returns the size of the GPUKeyFrame struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUKeyFrame) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUKeyFrame struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 64-byte buffer ready for GPU upload.
func (g *GPUKeyFrame) Marshal() []byte {
	buf := make([]byte, 64)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(g.Time))
	binary.LittleEndian.PutUint32(buf[4:8], 0)   // _pad0[0]
	binary.LittleEndian.PutUint32(buf[8:12], 0)  // _pad0[1]
	binary.LittleEndian.PutUint32(buf[12:16], 0) // _pad0[2]
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(g.Translation[0]))
	binary.LittleEndian.PutUint32(buf[20:24], math.Float32bits(g.Translation[1]))
	binary.LittleEndian.PutUint32(buf[24:28], math.Float32bits(g.Translation[2]))
	binary.LittleEndian.PutUint32(buf[28:32], 0) // _pad1
	binary.LittleEndian.PutUint32(buf[32:36], math.Float32bits(g.Rotation[0]))
	binary.LittleEndian.PutUint32(buf[36:40], math.Float32bits(g.Rotation[1]))
	binary.LittleEndian.PutUint32(buf[40:44], math.Float32bits(g.Rotation[2]))
	binary.LittleEndian.PutUint32(buf[44:48], math.Float32bits(g.Rotation[3]))
	binary.LittleEndian.PutUint32(buf[48:52], math.Float32bits(g.Scale[0]))
	binary.LittleEndian.PutUint32(buf[52:56], math.Float32bits(g.Scale[1]))
	binary.LittleEndian.PutUint32(buf[56:60], math.Float32bits(g.Scale[2]))
	binary.LittleEndian.PutUint32(buf[60:64], 0) // _pad2
	return buf
}

// GPUChannelHeader is the GPU-aligned representation of a channel header in the packed animation buffer.
// Describes which bone an animation channel targets and where its keyframes are stored.
// Size: 32 bytes (8 × u32).
type GPUChannelHeader struct {
	BoneIndex         uint32 // offset 0: index of the bone this channel animates
	PositionKeyOffset uint32 // offset 4: start index into the keyframe array for position keys
	PositionKeyCount  uint32 // offset 8: number of position keyframes
	RotationKeyOffset uint32 // offset 12: start index into the keyframe array for rotation keys
	RotationKeyCount  uint32 // offset 16: number of rotation keyframes
	ScaleKeyOffset    uint32 // offset 20: start index into the keyframe array for scale keys
	ScaleKeyCount     uint32 // offset 24: number of scale keyframes
	_pad              uint32 // offset 28: pad to 32 bytes
}

// Size returns the size of the GPUChannelHeader struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUChannelHeader) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUChannelHeader struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 32-byte buffer ready for GPU upload.
func (g *GPUChannelHeader) Marshal() []byte {
	buf := make([]byte, 32)
	binary.LittleEndian.PutUint32(buf[0:4], g.BoneIndex)
	binary.LittleEndian.PutUint32(buf[4:8], g.PositionKeyOffset)
	binary.LittleEndian.PutUint32(buf[8:12], g.PositionKeyCount)
	binary.LittleEndian.PutUint32(buf[12:16], g.RotationKeyOffset)
	binary.LittleEndian.PutUint32(buf[16:20], g.RotationKeyCount)
	binary.LittleEndian.PutUint32(buf[20:24], g.ScaleKeyOffset)
	binary.LittleEndian.PutUint32(buf[24:28], g.ScaleKeyCount)
	binary.LittleEndian.PutUint32(buf[28:32], 0) // _pad
	return buf
}

// GPUClipHeader is the GPU-aligned representation of an animation clip header.
// Describes the duration, playback rate, and channel range for an animation clip.
// Size: 16 bytes (4 × f32/u32).
type GPUClipHeader struct {
	Duration       float32 // offset 0: total clip duration in seconds
	TicksPerSecond float32 // offset 4: playback rate (used for time conversion)
	ChannelOffset  uint32  // offset 8: start index into the channel headers array
	ChannelCount   uint32  // offset 12: number of channels in this clip
}

// Size returns the size of the GPUClipHeader struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUClipHeader) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUClipHeader struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 16-byte buffer ready for GPU upload.
func (g *GPUClipHeader) Marshal() []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(g.Duration))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.TicksPerSecond))
	binary.LittleEndian.PutUint32(buf[8:12], g.ChannelOffset)
	binary.LittleEndian.PutUint32(buf[12:16], g.ChannelCount)
	return buf
}

// GPUSkeletalAnimationDataSource is the canonical WGSL definition of the SkeletalAnimationData struct.
// Matches GPUSkeletalAnimationData layout exactly (48 bytes, std430 aligned).
//
//go:embed assets/skeletal_animation_data.wgsl
var GPUSkeletalAnimationDataSource string

// GPUSkeletalAnimationData is the GPU-aligned per-instance animation state for the skeletal compute shader.
// Matches the WGSL SkeletalAnimationData struct layout exactly (see GPUSkeletalAnimationDataSource).
//
// WGSL layout (storage buffer alignment rules):
//
//	animation_index:      u32       offset  0
//	animation_time:       f32       offset  4
//	blend_weight:         f32       offset  8
//	secondary_anim_index: u32       offset 12
//	secondary_anim_time:  f32       offset 16
//	_pad:                 vec3<f32> offset 32 (align 16 → gap 20..31)
//	struct align = 16, struct size = roundUp(16, 44) = 48
//
// Size: 48 bytes.
type GPUSkeletalAnimationData struct {
	AnimationIndex     uint32     // offset 0: index of the primary animation clip
	AnimationTime      float32    // offset 4: current playback time of the primary clip
	BlendWeight        float32    // offset 8: blend weight between primary and secondary (0.0 = primary, 1.0 = secondary)
	SecondaryAnimIndex uint32     // offset 12: index of the secondary animation clip for blending
	SecondaryAnimTime  float32    // offset 16: current playback time of the secondary clip
	_pad               [7]float32 // offset 20: padding to reach WGSL struct size of 48 bytes
}

// Size returns the size of the GPUSkeletalAnimationData struct in bytes.
//
// Returns:
//   - int: The size of the struct in bytes.
func (g *GPUSkeletalAnimationData) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUSkeletalAnimationData struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 48-byte buffer ready for GPU upload.
func (g *GPUSkeletalAnimationData) Marshal() []byte {
	buf := make([]byte, 48)
	binary.LittleEndian.PutUint32(buf[0:4], g.AnimationIndex)
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.AnimationTime))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.BlendWeight))
	binary.LittleEndian.PutUint32(buf[12:16], g.SecondaryAnimIndex)
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(g.SecondaryAnimTime))
	for i := range 7 {
		binary.LittleEndian.PutUint32(buf[20+i*4:24+i*4], 0) // _pad
	}
	return buf
}
