package shader

import "github.com/cogentcore/webgpu/wgpu"

// vertexFormatInfo holds the wgpu vertex format and its byte size for offset calculation
type vertexFormatInfo struct {
	format wgpu.VertexFormat
	size   uint64
}

// sampledTextureInfo holds the view dimension and multisampled flag for a sampled texture type
type sampledTextureInfo struct {
	viewDimension wgpu.TextureViewDimension
	multisampled  bool
}

// wgslTypeLayout holds the byte size and alignment for a WGSL type per the WGSL specification.
// Used to compute MinBindingSize for buffer bindings.
type wgslTypeLayout struct {
	size  uint64
	align uint64
}

// parsedField represents a single field extracted from a WGSL struct during parsing
type parsedField struct {
	name      string
	typeName  string
	location  int
	isBuiltin bool
}

// parsedStruct represents a WGSL struct block extracted during parsing
type parsedStruct struct {
	name   string
	fields []parsedField
}
