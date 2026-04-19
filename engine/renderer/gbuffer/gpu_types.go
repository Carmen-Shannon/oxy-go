package gbuffer

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"
)

// GPUGBufferOutputSource is the canonical WGSL definition of the GBufferOutput struct.
// Matches GPUGBufferOutput layout exactly (48 bytes, std430 aligned).
//
//go:embed assets/gbuffer-output.wgsl
var GPUGBufferOutputSource string

// GPUGBufferOutput is the GPU-aligned representation of a single G-Buffer fragment
// output. Size: 48 bytes (3 × vec4<f32>).
type GPUGBufferOutput struct {
	Position [4]float32
	Normal   [4]float32
	Albedo   [4]float32
}

// Size returns the size of the GPUGBufferOutput struct in bytes.
func (g *GPUGBufferOutput) Size() int {
	return int(unsafe.Sizeof(*g))
}

// Marshal serializes the GPUGBufferOutput struct into a byte buffer.
func (g *GPUGBufferOutput) Marshal() []byte {
	buf := make([]byte, g.Size())
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
