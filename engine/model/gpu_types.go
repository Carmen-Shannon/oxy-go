package model

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"
)

// GPUVertexSource is the canonical WGSL definition of the VertexInput struct for static mesh pipelines.
// Matches GPUVertex layout exactly (64 bytes, std430 aligned).
//
//go:embed assets/vertex.wgsl
var GPUVertexSource string

// GPUVertex is the GPU-aligned representation of a single mesh vertex for static (non-skinned) models.
// Matches the WGSL VertexInput struct layout exactly (see GPUVertexSource).
// Size: 64 bytes (std430 aligned, no padding required).
type GPUVertex struct {
	Position [3]float32 // offset  0: vertex position in model space (12 bytes)
	Normal   [3]float32 // offset 12: vertex normal for lighting (12 bytes)
	TexCoord [2]float32 // offset 24: UV texture coordinate (8 bytes)
	Color    [4]float32 // offset 32: per-vertex RGBA color (16 bytes)
	Tangent  [4]float32 // offset 48: tangent vector (xyz) + handedness (w) for normal mapping (16 bytes)
}

// Size returns the size of the GPUVertex struct in bytes.
//
// Returns:
//   - int: the size of the struct in bytes.
func (g *GPUVertex) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUVertex struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 64-byte buffer ready for GPU upload.
func (g *GPUVertex) Marshal() []byte {
	buf := make([]byte, 64)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(g.Position[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.Position[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.Position[2]))
	binary.LittleEndian.PutUint32(buf[12:16], math.Float32bits(g.Normal[0]))
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(g.Normal[1]))
	binary.LittleEndian.PutUint32(buf[20:24], math.Float32bits(g.Normal[2]))
	binary.LittleEndian.PutUint32(buf[24:28], math.Float32bits(g.TexCoord[0]))
	binary.LittleEndian.PutUint32(buf[28:32], math.Float32bits(g.TexCoord[1]))
	binary.LittleEndian.PutUint32(buf[32:36], math.Float32bits(g.Color[0]))
	binary.LittleEndian.PutUint32(buf[36:40], math.Float32bits(g.Color[1]))
	binary.LittleEndian.PutUint32(buf[40:44], math.Float32bits(g.Color[2]))
	binary.LittleEndian.PutUint32(buf[44:48], math.Float32bits(g.Color[3]))
	binary.LittleEndian.PutUint32(buf[48:52], math.Float32bits(g.Tangent[0]))
	binary.LittleEndian.PutUint32(buf[52:56], math.Float32bits(g.Tangent[1]))
	binary.LittleEndian.PutUint32(buf[56:60], math.Float32bits(g.Tangent[2]))
	binary.LittleEndian.PutUint32(buf[60:64], math.Float32bits(g.Tangent[3]))
	return buf
}

// GPUSkinnedVertexSource is the canonical WGSL definition of the VertexInput struct for skinned mesh pipelines.
// Matches GPUSkinnedVertex layout exactly (96 bytes, std430 aligned).
//
//go:embed assets/skinned_vertex.wgsl
var GPUSkinnedVertexSource string

// GPUSkinnedVertex is the GPU-aligned representation of a single mesh vertex for skinned (bone-animated) models.
// It extends GPUVertex with per-vertex bone skinning data.
// Matches the WGSL VertexInput struct layout for skinned pipelines (see GPUSkinnedVertexSource).
// Size: 96 bytes (64 base vertex + 32 skinning data, std430 aligned, no padding required).
type GPUSkinnedVertex struct {
	GPUVertex              // offset  0: base vertex data (position, normal, uv, color, tangent) — 64 bytes
	BoneIndices [4]uint32  // offset 64: indices of up to 4 influencing bones (16 bytes)
	BoneWeights [4]float32 // offset 80: blend weights for each bone (must sum to 1.0) (16 bytes)
}

// Size returns the size of the GPUSkinnedVertex struct in bytes.
//
// Returns:
//   - int: the size of the struct in bytes.
func (g *GPUSkinnedVertex) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUSkinnedVertex struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 96-byte buffer ready for GPU upload.
func (g *GPUSkinnedVertex) Marshal() []byte {
	buf := make([]byte, 96)
	// Base vertex fields (64 bytes)
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(g.Position[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(g.Position[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(g.Position[2]))
	binary.LittleEndian.PutUint32(buf[12:16], math.Float32bits(g.Normal[0]))
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(g.Normal[1]))
	binary.LittleEndian.PutUint32(buf[20:24], math.Float32bits(g.Normal[2]))
	binary.LittleEndian.PutUint32(buf[24:28], math.Float32bits(g.TexCoord[0]))
	binary.LittleEndian.PutUint32(buf[28:32], math.Float32bits(g.TexCoord[1]))
	binary.LittleEndian.PutUint32(buf[32:36], math.Float32bits(g.Color[0]))
	binary.LittleEndian.PutUint32(buf[36:40], math.Float32bits(g.Color[1]))
	binary.LittleEndian.PutUint32(buf[40:44], math.Float32bits(g.Color[2]))
	binary.LittleEndian.PutUint32(buf[44:48], math.Float32bits(g.Color[3]))
	binary.LittleEndian.PutUint32(buf[48:52], math.Float32bits(g.Tangent[0]))
	binary.LittleEndian.PutUint32(buf[52:56], math.Float32bits(g.Tangent[1]))
	binary.LittleEndian.PutUint32(buf[56:60], math.Float32bits(g.Tangent[2]))
	binary.LittleEndian.PutUint32(buf[60:64], math.Float32bits(g.Tangent[3]))
	// Bone skinning fields (32 bytes)
	binary.LittleEndian.PutUint32(buf[64:68], g.BoneIndices[0])
	binary.LittleEndian.PutUint32(buf[68:72], g.BoneIndices[1])
	binary.LittleEndian.PutUint32(buf[72:76], g.BoneIndices[2])
	binary.LittleEndian.PutUint32(buf[76:80], g.BoneIndices[3])
	binary.LittleEndian.PutUint32(buf[80:84], math.Float32bits(g.BoneWeights[0]))
	binary.LittleEndian.PutUint32(buf[84:88], math.Float32bits(g.BoneWeights[1]))
	binary.LittleEndian.PutUint32(buf[88:92], math.Float32bits(g.BoneWeights[2]))
	binary.LittleEndian.PutUint32(buf[92:96], math.Float32bits(g.BoneWeights[3]))
	return buf
}

// ComputeBoundingRadius calculates the bounding sphere radius from a slice of
// GPUSkinnedVertex positions. The radius is the maximum distance from the origin
// across all vertices in the slice.
//
// Parameters:
//   - vertices: the vertex data to compute the bounding radius from
//
// Returns:
//   - float32: the maximum distance from the origin
func ComputeBoundingRadius(vertices []GPUSkinnedVertex) float32 {
	var maxDistSq float32
	for _, v := range vertices {
		p := v.Position
		distSq := p[0]*p[0] + p[1]*p[1] + p[2]*p[2]
		if distSq > maxDistSq {
			maxDistSq = distSq
		}
	}
	return float32(math.Sqrt(float64(maxDistSq)))
}

// GPUModelDataSource is the canonical WGSL definition of the ModelData struct for per-instance model matrices.
// Matches GPUModelData layout exactly (64 bytes, std430 aligned).
//
//go:embed assets/model_data.wgsl
var GPUModelDataSource string

// GPUModelData is the GPU-aligned representation of a single per-instance model matrix.
// Matches the WGSL ModelData struct layout exactly (see GPUModelDataSource).
// Size: 64 bytes (mat4x4<f32> = 16 × float32, std430 aligned, no padding required).
type GPUModelData struct {
	Model [16]float32 // offset 0: 4×4 model-to-world transform matrix (64 bytes)
}

// Size returns the size of the GPUModelData struct in bytes.
//
// Returns:
//   - int: the size of the struct in bytes.
func (g *GPUModelData) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUModelData struct into a byte buffer suitable for GPU upload.
//
// Returns:
//   - []byte: 64-byte buffer ready for GPU upload.
func (g *GPUModelData) Marshal() []byte {
	buf := make([]byte, 64)
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(g.Model[i]))
	}
	return buf
}
