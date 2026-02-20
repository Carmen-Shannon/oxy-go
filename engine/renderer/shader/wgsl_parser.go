package shader

import (
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/cogentcore/webgpu/wgpu"
)

// wgslVertexFormatMap maps WGSL type names to their corresponding wgpu vertex format and byte size
var wgslVertexFormatMap = map[string]vertexFormatInfo{
	"f32":       {wgpu.VertexFormatFloat32, 4},
	"vec2f":     {wgpu.VertexFormatFloat32x2, 8},
	"vec2<f32>": {wgpu.VertexFormatFloat32x2, 8},
	"vec3f":     {wgpu.VertexFormatFloat32x3, 12},
	"vec3<f32>": {wgpu.VertexFormatFloat32x3, 12},
	"vec4f":     {wgpu.VertexFormatFloat32x4, 16},
	"vec4<f32>": {wgpu.VertexFormatFloat32x4, 16},
	"i32":       {wgpu.VertexFormatSint32, 4},
	"vec2i":     {wgpu.VertexFormatSint32x2, 8},
	"vec2<i32>": {wgpu.VertexFormatSint32x2, 8},
	"vec3i":     {wgpu.VertexFormatSint32x3, 12},
	"vec3<i32>": {wgpu.VertexFormatSint32x3, 12},
	"vec4i":     {wgpu.VertexFormatSint32x4, 16},
	"vec4<i32>": {wgpu.VertexFormatSint32x4, 16},
	"u32":       {wgpu.VertexFormatUint32, 4},
	"vec2u":     {wgpu.VertexFormatUint32x2, 8},
	"vec2<u32>": {wgpu.VertexFormatUint32x2, 8},
	"vec3u":     {wgpu.VertexFormatUint32x3, 12},
	"vec3<u32>": {wgpu.VertexFormatUint32x3, 12},
	"vec4u":     {wgpu.VertexFormatUint32x4, 16},
	"vec4<u32>": {wgpu.VertexFormatUint32x4, 16},
	"vec2<f16>": {wgpu.VertexFormatFloat16x2, 4},
	"vec2h":     {wgpu.VertexFormatFloat16x2, 4},
	"vec4<f16>": {wgpu.VertexFormatFloat16x4, 8},
	"vec4h":     {wgpu.VertexFormatFloat16x4, 8},
}

// wgslSampledTextureMap maps WGSL sampled texture base names to their view dimension and multisampled flag
var wgslSampledTextureMap = map[string]sampledTextureInfo{
	"texture_1d":                    {wgpu.TextureViewDimension1D, false},
	"texture_2d":                    {wgpu.TextureViewDimension2D, false},
	"texture_2d_array":              {wgpu.TextureViewDimension2DArray, false},
	"texture_3d":                    {wgpu.TextureViewDimension3D, false},
	"texture_cube":                  {wgpu.TextureViewDimensionCube, false},
	"texture_cube_array":            {wgpu.TextureViewDimensionCubeArray, false},
	"texture_multisampled_2d":       {wgpu.TextureViewDimension2D, true},
	"texture_depth_2d":              {wgpu.TextureViewDimension2D, false},
	"texture_depth_2d_array":        {wgpu.TextureViewDimension2DArray, false},
	"texture_depth_cube":            {wgpu.TextureViewDimensionCube, false},
	"texture_depth_cube_array":      {wgpu.TextureViewDimensionCubeArray, false},
	"texture_depth_multisampled_2d": {wgpu.TextureViewDimension2D, true},
}

// wgslStorageTextureDimMap maps WGSL storage texture base names to their view dimension
var wgslStorageTextureDimMap = map[string]wgpu.TextureViewDimension{
	"texture_storage_1d":       wgpu.TextureViewDimension1D,
	"texture_storage_2d":       wgpu.TextureViewDimension2D,
	"texture_storage_2d_array": wgpu.TextureViewDimension2DArray,
	"texture_storage_3d":       wgpu.TextureViewDimension3D,
}

// wgslSampleTypeMap maps WGSL scalar type parameters to their wgpu texture sample type
var wgslSampleTypeMap = map[string]wgpu.TextureSampleType{
	"f32": wgpu.TextureSampleTypeFloat,
	"i32": wgpu.TextureSampleTypeSint,
	"u32": wgpu.TextureSampleTypeUint,
}

// wgslStorageAccessMap maps WGSL access mode keywords to their wgpu storage texture access
var wgslStorageAccessMap = map[string]wgpu.StorageTextureAccess{
	"write":      wgpu.StorageTextureAccessWriteOnly,
	"read":       wgpu.StorageTextureAccessReadOnly,
	"read_write": wgpu.StorageTextureAccessReadWrite,
}

// wgslTexelFormatMap maps WGSL texel format strings to their corresponding wgpu texture formats.
// These are the formats valid for storage textures per the WGSL specification.
var wgslTexelFormatMap = map[string]wgpu.TextureFormat{
	"rgba8unorm":  wgpu.TextureFormatRGBA8Unorm,
	"rgba8snorm":  wgpu.TextureFormatRGBA8Snorm,
	"rgba8uint":   wgpu.TextureFormatRGBA8Uint,
	"rgba8sint":   wgpu.TextureFormatRGBA8Sint,
	"rgba16uint":  wgpu.TextureFormatRGBA16Uint,
	"rgba16sint":  wgpu.TextureFormatRGBA16Sint,
	"rgba16float": wgpu.TextureFormatRGBA16Float,
	"r32uint":     wgpu.TextureFormatR32Uint,
	"r32sint":     wgpu.TextureFormatR32Sint,
	"r32float":    wgpu.TextureFormatR32Float,
	"rg32uint":    wgpu.TextureFormatRG32Uint,
	"rg32sint":    wgpu.TextureFormatRG32Sint,
	"rg32float":   wgpu.TextureFormatRG32Float,
	"rgba32uint":  wgpu.TextureFormatRGBA32Uint,
	"rgba32sint":  wgpu.TextureFormatRGBA32Sint,
	"rgba32float": wgpu.TextureFormatRGBA32Float,
	"bgra8unorm":  wgpu.TextureFormatBGRA8Unorm,
}

var (
	// structBlockRegex matches struct declarations and captures the name and body
	structBlockRegex = regexp.MustCompile(`struct\s+(\w+)\s*\{([^}]*)\}`)

	// locationRegex matches @location(N) attributes
	locationRegex = regexp.MustCompile(`@location\((\d+)\)`)

	// builtinRegex matches @builtin(...) attributes
	builtinRegex = regexp.MustCompile(`@builtin\(\w+\)`)

	// fieldRegex matches a struct field line: optional attributes, name, colon, type.
	// The type capture (.+) is greedy to handle parameterized types like array<T, N>.
	fieldRegex = regexp.MustCompile(`(?:(?:@\w+\([^)]*\)\s*)*)*\s*(\w+)\s*:\s*(.+)`)

	// vertexEntryRegex matches @vertex functions and captures the entry point name
	vertexEntryRegex = regexp.MustCompile(`(?s)@vertex\b.*?\bfn\s+(\w+)`)

	// fragmentEntryRegex matches @fragment functions and captures the entry point name
	fragmentEntryRegex = regexp.MustCompile(`(?s)@fragment\b.*?\bfn\s+(\w+)`)

	// computeEntryRegex matches @compute functions and captures the entry point name
	computeEntryRegex = regexp.MustCompile(`(?s)@compute\b.*?\bfn\s+(\w+)`)

	// workgroupSizeRegex captures 1-3 integer dimensions from @workgroup_size(x[, y[, z]])
	workgroupSizeRegex = regexp.MustCompile(`@workgroup_size\(\s*(\d+)\s*(?:,\s*(\d+)\s*(?:,\s*(\d+)\s*)?)?\)`)

	// bindGroupDeclRegex captures group, binding, optional address space, variable name, and type
	// from declarations like: @group(0) @binding(0) var<uniform> camera: CameraUniform;
	// or handle types: @group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
	bindGroupDeclRegex = regexp.MustCompile(`@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<([^>]*)>)?\s+(\w+)\s*:\s*([^;]+?)\s*;`)
)

// parseVertexLayouts extracts vertex buffer layouts from WGSL source code.
// It finds all structs that are pure vertex inputs (have @location attributes but no @builtin fields)
// and converts them into wgpu.VertexBufferLayout entries. Compute shaders or shaders with no
// vertex input structs return an empty map. Structs containing unrecognized WGSL types are skipped.
//
// Parameters:
//   - source: the raw WGSL source code string
//
// Returns:
//   - map[int][]wgpu.VertexBufferLayout: vertex layouts keyed by sequential index
func parseVertexLayouts(source string) map[int][]wgpu.VertexBufferLayout {
	result := make(map[int][]wgpu.VertexBufferLayout)
	cleaned := stripLineComments(source)
	structs := parseStructBlocks(cleaned)

	layoutIndex := 0
	for _, ps := range structs {
		if !isVertexInputStruct(ps) {
			continue
		}
		layout, ok := buildVertexBufferLayout(ps)
		if !ok {
			continue
		}
		result[layoutIndex] = []wgpu.VertexBufferLayout{layout}
		layoutIndex++
	}

	return result
}

// parseBindGroupLayouts extracts all @group(N) @binding(M) resource declarations from WGSL
// source and returns them as wgpu.BindGroupLayoutDescriptor values grouped by group index.
// Each descriptor's entries are sorted by binding index. The provided visibility flag is
// applied to all entries, corresponding to the shader stage that declared them.
//
// Parameters:
//   - source: the raw WGSL source code string
//   - visibility: the shader stage visibility flag to set on each entry
//
// Returns:
//   - map[int]wgpu.BindGroupLayoutDescriptor: layout descriptors keyed by group index
//   - map[int]map[int]string: variable names keyed by group and binding index for resource tracking
func parseBindGroupLayouts(source string, visibility wgpu.ShaderStage) (map[int]wgpu.BindGroupLayoutDescriptor, map[int]map[int]string) {
	groups := make(map[int][]wgpu.BindGroupLayoutEntry)
	varNames := make(map[int]map[int]string)
	cleaned := stripComments(source)

	// Parse all struct definitions and compute their sizes so we can set MinBindingSize
	// on buffer layout entries. This enables InitBindGroup to create correctly-sized GPU buffers.
	structs := parseStructBlocks(cleaned)
	structSizes := computeStructSizes(structs)

	matches := bindGroupDeclRegex.FindAllStringSubmatch(cleaned, -1)
	for _, match := range matches {
		group, _ := strconv.Atoi(match[1])
		binding, _ := strconv.Atoi(match[2])
		addressSpace := strings.TrimSpace(match[3])
		varName := strings.TrimSpace(match[4])
		typeName := strings.TrimSpace(match[5])

		entry := classifyResource(uint32(binding), visibility, addressSpace, typeName)

		// Set MinBindingSize for buffer bindings by resolving the bound type's size.
		if entry.Buffer.Type != wgpu.BufferBindingTypeUndefined {
			if layout, ok := resolveTypeLayout(typeName, structSizes); ok && layout.size > 0 {
				entry.Buffer.MinBindingSize = layout.size
			}
		}

		groups[group] = append(groups[group], entry)

		if varNames[group] == nil {
			varNames[group] = make(map[int]string)
		}
		varNames[group][binding] = varName
	}

	result := make(map[int]wgpu.BindGroupLayoutDescriptor, len(groups))
	for g, entries := range groups {
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].Binding < entries[j].Binding
		})
		result[g] = wgpu.BindGroupLayoutDescriptor{
			Entries: entries,
		}
	}

	return result, varNames
}

// parseWorkgroupSize extracts the @workgroup_size(x, y, z) dimensions from WGSL source.
// Omitted dimensions default to 1 per the WGSL specification.
// Returns [1, 1, 1] if no @workgroup_size annotation is found.
//
// Parameters:
//   - source: the raw WGSL source code string
//
// Returns:
//   - [3]uint32: the workgroup size as [x, y, z]
func parseWorkgroupSize(source string) [3]uint32 {
	cleaned := stripComments(source)
	result := [3]uint32{1, 1, 1}

	match := workgroupSizeRegex.FindStringSubmatch(cleaned)
	if match == nil {
		return result
	}

	if match[1] != "" {
		if v, err := strconv.ParseUint(match[1], 10, 32); err == nil {
			result[0] = uint32(v)
		}
	}
	if match[2] != "" {
		if v, err := strconv.ParseUint(match[2], 10, 32); err == nil {
			result[1] = uint32(v)
		}
	}
	if match[3] != "" {
		if v, err := strconv.ParseUint(match[3], 10, 32); err == nil {
			result[2] = uint32(v)
		}
	}

	return result
}

// parseEntryPoint extracts the entry point function name for the given shader type
// from WGSL source. Returns an empty string if no matching entry point annotation is found.
//
// Parameters:
//   - source: the raw WGSL source code string
//   - shaderType: the shader type to search for (ShaderTypeVertex, ShaderTypeFragment, or ShaderTypeCompute)
//
// Returns:
//   - string: the entry point function name, or empty string if not found
func parseEntryPoint(source string, shaderType ShaderType) string {
	cleaned := stripComments(source)

	var re *regexp.Regexp
	switch shaderType {
	case ShaderTypeVertex:
		re = vertexEntryRegex
	case ShaderTypeFragment:
		re = fragmentEntryRegex
	case ShaderTypeCompute:
		re = computeEntryRegex
	default:
		return ""
	}

	if match := re.FindStringSubmatch(cleaned); match != nil {
		return match[1]
	}
	return ""
}

// parseStructBlocks finds all struct { ... } blocks in the cleaned WGSL source
// and parses their fields including @location and @builtin attributes
//
// Parameters:
//   - source: WGSL source with comments already stripped
//
// Returns:
//   - []parsedStruct: all struct blocks found in the source
func parseStructBlocks(source string) []parsedStruct {
	matches := structBlockRegex.FindAllStringSubmatch(source, -1)
	structs := make([]parsedStruct, 0, len(matches))

	for _, match := range matches {
		name := match[1]
		body := match[2]

		fields := parseStructFields(body)
		structs = append(structs, parsedStruct{
			name:   name,
			fields: fields,
		})
	}

	return structs
}

// parseStructFields parses the body of a struct block into individual fields,
// extracting @location and @builtin attributes along with the field name and type
//
// Parameters:
//   - body: the content between { and } of a struct declaration
//
// Returns:
//   - []parsedField: all fields found in the struct body
func parseStructFields(body string) []parsedField {
	lines := splitAtTopLevelCommas(body)
	fields := make([]parsedField, 0, len(lines))

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var field parsedField

		// check for @builtin
		if builtinRegex.MatchString(line) {
			field.isBuiltin = true
		}

		// check for @location(N)
		if locMatch := locationRegex.FindStringSubmatch(line); locMatch != nil {
			loc, err := strconv.Atoi(locMatch[1])
			if err == nil {
				field.location = loc
			}
		} else {
			field.location = -1
		}

		// extract field name and type
		if fm := fieldRegex.FindStringSubmatch(line); fm != nil {
			field.name = fm[1]
			field.typeName = strings.TrimSpace(fm[2])
		} else {
			continue
		}

		fields = append(fields, field)
	}

	return fields
}
