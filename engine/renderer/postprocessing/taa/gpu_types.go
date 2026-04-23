package taa

import (
	_ "embed"
	"encoding/binary"
	"math"
	"unsafe"
)

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
