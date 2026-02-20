package shader

import (
	"strconv"
	"strings"

	"github.com/cogentcore/webgpu/wgpu"
)

// wgslPrimitiveLayoutMap maps WGSL primitive, vector, matrix, and atomic type names
// to their byte size and alignment per the WGSL specification.
//
// Reference: https://www.w3.org/TR/WGSL/#alignment-and-size
var wgslPrimitiveLayoutMap = map[string]wgslTypeLayout{
	// Scalars
	"f32":  {4, 4},
	"i32":  {4, 4},
	"u32":  {4, 4},
	"f16":  {2, 2},
	"bool": {4, 4},

	// Vectors – f32
	"vec2<f32>": {8, 8},
	"vec2f":     {8, 8},
	"vec3<f32>": {12, 16},
	"vec3f":     {12, 16},
	"vec4<f32>": {16, 16},
	"vec4f":     {16, 16},

	// Vectors – i32
	"vec2<i32>": {8, 8},
	"vec2i":     {8, 8},
	"vec3<i32>": {12, 16},
	"vec3i":     {12, 16},
	"vec4<i32>": {16, 16},
	"vec4i":     {16, 16},

	// Vectors – u32
	"vec2<u32>": {8, 8},
	"vec2u":     {8, 8},
	"vec3<u32>": {12, 16},
	"vec3u":     {12, 16},
	"vec4<u32>": {16, 16},
	"vec4u":     {16, 16},

	// Vectors – f16
	"vec2<f16>": {4, 4},
	"vec2h":     {4, 4},
	"vec4<f16>": {8, 8},
	"vec4h":     {8, 8},

	// Matrices – matCxR<f32>: C columns of vecR<f32>, stride = roundUp(align(vecR), size(vecR))
	"mat2x2<f32>": {16, 8},
	"mat2x3<f32>": {32, 16},
	"mat2x4<f32>": {32, 16},
	"mat3x2<f32>": {24, 8},
	"mat3x3<f32>": {48, 16},
	"mat3x4<f32>": {48, 16},
	"mat4x2<f32>": {32, 8},
	"mat4x3<f32>": {64, 16},
	"mat4x4<f32>": {64, 16},

	// Atomic types
	"atomic<u32>": {4, 4},
	"atomic<i32>": {4, 4},
}

// roundUpAlign rounds value up to the next multiple of alignment.
// Alignment must be a power of two.
//
// Parameters:
//   - alignment: the required alignment (must be a power of two)
//   - value: the value to align
//
// Returns:
//   - uint64: value rounded up to the next multiple of alignment
func roundUpAlign(alignment, value uint64) uint64 {
	if alignment == 0 {
		return value
	}
	return (value + alignment - 1) &^ (alignment - 1)
}

// resolveTypeLayout resolves a WGSL type name to its size and alignment using primitives
// and previously-computed struct layouts. Handles fixed-size arrays (array<T, N>) and returns
// false for runtime-sized arrays or unknown types.
//
// Parameters:
//   - typeName: the WGSL type name to resolve, e.g. "f32", "CameraUniform", "array<FrustumPlane, 6>"
//   - knownTypes: a map of already-resolved type names to their layouts
//
// Returns:
//   - wgslTypeLayout: the resolved layout
//   - bool: true if the type could be resolved, false for runtime-sized arrays or unknown types
func resolveTypeLayout(typeName string, knownTypes map[string]wgslTypeLayout) (wgslTypeLayout, bool) {
	// Check primitives first
	if layout, ok := wgslPrimitiveLayoutMap[typeName]; ok {
		return layout, true
	}

	// Check already-computed structs
	if layout, ok := knownTypes[typeName]; ok {
		return layout, true
	}

	// Handle array<ElementType, Count> (fixed-size) and array<ElementType> (runtime-sized)
	if strings.HasPrefix(typeName, "array<") && strings.HasSuffix(typeName, ">") {
		inner := typeName[6 : len(typeName)-1]
		parts := strings.SplitN(inner, ",", 2)
		elemType := strings.TrimSpace(parts[0])

		elemLayout, ok := resolveTypeLayout(elemType, knownTypes)
		if !ok {
			return wgslTypeLayout{}, false
		}

		if len(parts) == 2 {
			// Fixed-size array: array<T, N>
			countStr := strings.TrimSpace(parts[1])
			count, err := strconv.ParseUint(countStr, 10, 64)
			if err != nil {
				return wgslTypeLayout{}, false
			}
			stride := roundUpAlign(elemLayout.align, elemLayout.size)
			return wgslTypeLayout{count * stride, elemLayout.align}, true
		}

		// Runtime-sized array — return element stride as MinBindingSize.
		// This is the minimum useful binding size (one element) and allows callers
		// to scale by instance count when computing actual buffer sizes.
		stride := roundUpAlign(elemLayout.align, elemLayout.size)
		return wgslTypeLayout{stride, elemLayout.align}, true
	}

	return wgslTypeLayout{}, false
}

// computeStructLayout computes the byte size and alignment of a single WGSL struct using
// WGSL struct layout rules: each field is placed at the next aligned offset, and the total
// size is rounded up to the struct's alignment (max alignment of all fields).
//
// If the struct contains a runtime-sized array as its last field, the returned size is the
// offset of that array (the fixed-size prefix). Fields with @builtin attributes are skipped
// as they are not part of the buffer layout.
//
// Parameters:
//   - ps: the parsed struct whose layout to compute
//   - knownTypes: a map of already-resolved type names to their layouts
//
// Returns:
//   - wgslTypeLayout: the computed layout
//   - bool: true if all fields could be resolved
func computeStructLayout(ps parsedStruct, knownTypes map[string]wgslTypeLayout) (wgslTypeLayout, bool) {
	offset := uint64(0)
	maxAlign := uint64(1)

	for _, field := range ps.fields {
		if field.isBuiltin {
			continue
		}

		fieldLayout, ok := resolveTypeLayout(field.typeName, knownTypes)
		if !ok {
			// If the last field is a runtime-sized array, the struct size is the offset so far
			if strings.HasPrefix(field.typeName, "array<") && !strings.Contains(field.typeName, ",") {
				// Runtime-sized array as last member — struct size is the fixed-prefix offset
				offset = roundUpAlign(maxAlign, offset)
				if offset == 0 {
					// Struct has only a runtime-sized array; use element size as minimum
					inner := field.typeName[6 : len(field.typeName)-1]
					elemType := strings.TrimSpace(inner)
					if elemLayout, elemOk := resolveTypeLayout(elemType, knownTypes); elemOk {
						return wgslTypeLayout{roundUpAlign(elemLayout.align, elemLayout.size), elemLayout.align}, true
					}
				}
				return wgslTypeLayout{offset, maxAlign}, true
			}
			return wgslTypeLayout{}, false
		}

		offset = roundUpAlign(fieldLayout.align, offset)
		offset += fieldLayout.size

		if fieldLayout.align > maxAlign {
			maxAlign = fieldLayout.align
		}
	}

	size := roundUpAlign(maxAlign, offset)
	return wgslTypeLayout{size, maxAlign}, true
}

// computeStructSizes computes the byte size and alignment of all parsed WGSL structs.
// It resolves dependencies between structs iteratively, handling cases where one struct
// contains fields typed as another struct. Returns a map from struct name to layout.
//
// Parameters:
//   - structs: all parsed struct blocks from the WGSL source
//
// Returns:
//   - map[string]wgslTypeLayout: a map from struct name to computed layout
func computeStructSizes(structs []parsedStruct) map[string]wgslTypeLayout {
	resolved := make(map[string]wgslTypeLayout, len(structs))
	remaining := make([]parsedStruct, len(structs))
	copy(remaining, structs)

	for {
		progress := false
		next := remaining[:0]

		for _, ps := range remaining {
			if layout, ok := computeStructLayout(ps, resolved); ok {
				resolved[ps.name] = layout
				progress = true
			} else {
				next = append(next, ps)
			}
		}

		remaining = next
		if !progress || len(remaining) == 0 {
			break
		}
	}

	return resolved
}

// classifyResource creates a wgpu.BindGroupLayoutEntry from a parsed WGSL resource declaration.
// It determines the resource category (buffer, texture, sampler, storage texture) from the
// address space qualifier and type name, and populates the corresponding layout fields.
//
// Parameters:
//   - binding: the binding index from @binding(N)
//   - visibility: the shader stage visibility flag
//   - addressSpace: the address space qualifier (e.g. "uniform", "storage, read_write"), empty for handle types
//   - typeName: the WGSL type string (e.g. "CameraUniform", "texture_2d<f32>", "sampler")
//
// Returns:
//   - wgpu.BindGroupLayoutEntry: a fully populated layout entry for the resource
func classifyResource(binding uint32, visibility wgpu.ShaderStage, addressSpace, typeName string) wgpu.BindGroupLayoutEntry {
	entry := wgpu.BindGroupLayoutEntry{
		Binding:    binding,
		Visibility: visibility,
	}

	if addressSpace != "" {
		switch {
		case addressSpace == "uniform":
			entry.Buffer.Type = wgpu.BufferBindingTypeUniform
		case strings.HasPrefix(addressSpace, "storage"):
			if strings.Contains(addressSpace, "read_write") {
				entry.Buffer.Type = wgpu.BufferBindingTypeStorage
			} else {
				entry.Buffer.Type = wgpu.BufferBindingTypeReadOnlyStorage
			}
		}
		return entry
	}

	switch {
	case typeName == "sampler":
		entry.Sampler.Type = wgpu.SamplerBindingTypeFiltering
	case typeName == "sampler_comparison":
		entry.Sampler.Type = wgpu.SamplerBindingTypeComparison
	case strings.HasPrefix(typeName, "texture_storage_"):
		classifyStorageTexture(typeName, &entry)
	case strings.HasPrefix(typeName, "texture_depth_"):
		classifyDepthTexture(typeName, &entry)
	case strings.HasPrefix(typeName, "texture_"):
		classifySampledTexture(typeName, &entry)
	}

	return entry
}

// classifySampledTexture parses a sampled texture type (e.g. "texture_2d<f32>") and populates
// the texture layout fields on the entry
//
// Parameters:
//   - typeName: the full WGSL texture type string
//   - entry: the bind group layout entry to populate
func classifySampledTexture(typeName string, entry *wgpu.BindGroupLayoutEntry) {
	base, param := splitTypeParams(typeName)

	if info, ok := wgslSampledTextureMap[base]; ok {
		entry.Texture.ViewDimension = info.viewDimension
		entry.Texture.Multisampled = info.multisampled
	}
	if st, ok := wgslSampleTypeMap[param]; ok {
		entry.Texture.SampleType = st
	}
}

// classifyDepthTexture parses a depth texture type (e.g. "texture_depth_2d") and populates
// the texture layout fields on the entry
//
// Parameters:
//   - typeName: the full WGSL depth texture type string
//   - entry: the bind group layout entry to populate
func classifyDepthTexture(typeName string, entry *wgpu.BindGroupLayoutEntry) {
	entry.Texture.SampleType = wgpu.TextureSampleTypeDepth
	if info, ok := wgslSampledTextureMap[typeName]; ok {
		entry.Texture.ViewDimension = info.viewDimension
		entry.Texture.Multisampled = info.multisampled
	}
}

// classifyStorageTexture parses a storage texture type (e.g. "texture_storage_2d<rgba8unorm, write>")
// and populates the storage texture layout fields on the entry
//
// Parameters:
//   - typeName: the full WGSL storage texture type string
//   - entry: the bind group layout entry to populate
func classifyStorageTexture(typeName string, entry *wgpu.BindGroupLayoutEntry) {
	base, params := splitTypeParams(typeName)

	if dim, ok := wgslStorageTextureDimMap[base]; ok {
		entry.StorageTexture.ViewDimension = dim
	}

	parts := strings.SplitN(params, ",", 2)
	if len(parts) >= 1 {
		formatStr := strings.TrimSpace(parts[0])
		if format, ok := wgslTexelFormatMap[formatStr]; ok {
			entry.StorageTexture.Format = format
		}
	}
	if len(parts) >= 2 {
		accessStr := strings.TrimSpace(parts[1])
		if access, ok := wgslStorageAccessMap[accessStr]; ok {
			entry.StorageTexture.Access = access
		}
	}
}

// splitTypeParams splits a WGSL parameterized type into its base name and parameter string.
// For "texture_2d<f32>" returns ("texture_2d", "f32").
// For "texture_depth_2d" (no params) returns ("texture_depth_2d", "").
//
// Parameters:
//   - typeName: the WGSL type string to split
//
// Returns:
//   - base: the type name before the first angle bracket
//   - params: the content between angle brackets, or empty if none
func splitTypeParams(typeName string) (base string, params string) {
	before, after, ok := strings.Cut(typeName, "<")
	if !ok {
		return typeName, ""
	}
	base = before
	params = strings.TrimSuffix(after, ">")
	params = strings.TrimSpace(params)
	return base, params
}

// stripComments removes both single-line (//) and block (/* */) comments from WGSL source.
// Block comments may be nested per the WGSL specification.
//
// Parameters:
//   - source: raw WGSL source string
//
// Returns:
//   - string: source with all comments removed
func stripComments(source string) string {
	return stripLineComments(stripBlockComments(source))
}

// stripLineComments removes single-line // comments from WGSL source so they
// do not interfere with struct and field parsing
//
// Parameters:
//   - source: raw WGSL source string
//
// Returns:
//   - string: source with line comments removed
func stripLineComments(source string) string {
	var sb strings.Builder
	lines := strings.SplitSeq(source, "\n")
	for line := range lines {
		if idx := strings.Index(line, "//"); idx >= 0 {
			line = line[:idx]
		}
		sb.WriteString(line)
		sb.WriteByte('\n')
	}
	return sb.String()
}

// stripBlockComments removes block comments (/* ... */) from WGSL source,
// handling nested block comments per the WGSL specification
//
// Parameters:
//   - source: raw WGSL source string
//
// Returns:
//   - string: source with block comments removed
func stripBlockComments(source string) string {
	var sb strings.Builder
	sb.Grow(len(source))
	depth := 0
	i := 0
	for i < len(source) {
		if i+1 < len(source) {
			if source[i] == '/' && source[i+1] == '*' {
				depth++
				i += 2
				continue
			}
			if source[i] == '*' && source[i+1] == '/' {
				if depth > 0 {
					depth--
				}
				i += 2
				continue
			}
		}
		if depth == 0 {
			sb.WriteByte(source[i])
		}
		i++
	}
	return sb.String()
}

// isVertexInputStruct returns true if the struct is a pure vertex input, meaning
// it has at least one @location field and zero @builtin fields. This distinguishes
// vertex input structs from vertex output structs which mix @location with @builtin(position).
//
// Parameters:
//   - ps: the parsed struct to check
//
// Returns:
//   - bool: true if this is a vertex input struct
func isVertexInputStruct(ps parsedStruct) bool {
	hasLocation := false
	for _, f := range ps.fields {
		if f.isBuiltin {
			return false
		}
		if f.location >= 0 {
			hasLocation = true
		}
	}
	return hasLocation
}

// buildVertexBufferLayout converts a parsed vertex input struct into a wgpu.VertexBufferLayout.
// It maps each field's WGSL type to a wgpu.VertexFormat using wgslVertexFormatMap, calculates
// sequential byte offsets, and sets the total array stride. Returns false if any field has
// an unrecognized type.
//
// Parameters:
//   - ps: the parsed struct containing vertex input fields
//
// Returns:
//   - wgpu.VertexBufferLayout: the constructed vertex buffer layout
//   - bool: false if a field type could not be mapped to a vertex format
func buildVertexBufferLayout(ps parsedStruct) (wgpu.VertexBufferLayout, bool) {
	attrs := make([]wgpu.VertexAttribute, 0, len(ps.fields))
	var offset uint64

	for _, f := range ps.fields {
		info, ok := wgslVertexFormatMap[f.typeName]
		if !ok {
			return wgpu.VertexBufferLayout{}, false
		}

		attrs = append(attrs, wgpu.VertexAttribute{
			Format:         info.format,
			Offset:         offset,
			ShaderLocation: uint32(f.location),
		})
		offset += info.size
	}

	return wgpu.VertexBufferLayout{
		ArrayStride: offset,
		StepMode:    wgpu.VertexStepModeVertex,
		Attributes:  attrs,
	}, true
}

// splitAtTopLevelCommas splits a string at commas that are not nested inside angle brackets.
// This correctly handles WGSL types like array<FrustumPlane, 6> where the comma is part of
// the type syntax rather than a field separator.
//
// Parameters:
//   - s: the string to split (typically the body of a WGSL struct)
//
// Returns:
//   - []string: substrings between top-level commas
func splitAtTopLevelCommas(s string) []string {
	var parts []string
	depth := 0
	start := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '<':
			depth++
		case '>':
			if depth > 0 {
				depth--
			}
		case ',':
			if depth == 0 {
				parts = append(parts, s[start:i])
				start = i + 1
			}
		}
	}
	parts = append(parts, s[start:])
	return parts
}
