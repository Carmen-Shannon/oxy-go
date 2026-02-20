package shader

import (
	"fmt"
	"os"

	"github.com/cogentcore/webgpu/wgpu"
)

// ShaderType identifies whether a shader is a render shader or a compute shader.
type ShaderType int

const (
	// ShaderTypeCompute indicates a shader containing a @compute entry point.
	ShaderTypeCompute ShaderType = iota

	// ShaderTypeVertex is the vertex shader type, used for vertex processing in render pipelines.
	ShaderTypeVertex

	// ShaderTypeFragment is the fragment shader type, used for fragment processing in pair with a vertex shader.
	ShaderTypeFragment
)

// shader is the implementation of the Shader interface.
// It holds all of the persistent shader data required for pipeline creation and material binding.
type shader struct {
	key                        string
	source                     string
	shaderType                 ShaderType
	bindGroupLayoutDescriptors map[int]wgpu.BindGroupLayoutDescriptor
	bindingVarNames            map[int]map[int]string
	vertexLayouts              map[int][]wgpu.VertexBufferLayout
	workGroupSize              [3]uint32
	entryPoint                 string
	module                     *wgpu.ShaderModuleDescriptor

	pp PreProcessor
}

// Shader defines the interface for a loaded and parsed WGSL shader. It exposes the shader's
// unique key, source code, entry point, bind group layout descriptors, vertex buffer layouts,
// workgroup size, and pre-processor declarations needed for pipeline creation and resource wiring.
type Shader interface {
	// Key retrieves the unique identifier for this shader, used for caching and lookups.
	//
	// Returns:
	//   - string: the shader's unique key
	Key() string

	// Source retrieves the WGSL shader source code.
	//
	// Returns:
	//   - string: the WGSL source code of the shader
	Source() string

	// BindGroupLayoutDescriptor retrieves the bind group layout descriptor for a specific binding key.
	//
	// Parameters:
	//   - bindingKey: the integer key identifying the bind group layout descriptor
	//
	// Returns:
	//   - wgpu.BindGroupLayoutDescriptor: the bind group layout descriptor associated with the key, or an empty descriptor if not set
	BindGroupLayoutDescriptor(bindingKey int) wgpu.BindGroupLayoutDescriptor

	// BindGroupLayoutDescriptors retrieves all parsed bind group layout descriptors.
	// These are the CPU-side descriptors extracted from the shader source which can be
	// used by the renderer to create the actual wgpu.BindGroupLayout GPU objects.
	//
	// Returns:
	//   - map[int]wgpu.BindGroupLayoutDescriptor: descriptors keyed by group index
	BindGroupLayoutDescriptors() map[int]wgpu.BindGroupLayoutDescriptor

	// BindGroupVarName retrieves the variable name for a given group and binding index, if it exists.
	// This is used for tracking resource usage and debugging.
	//
	// Parameters:
	//   - group: the bind group index
	//   - binding: the binding index within the group
	//
	// Returns:
	//   - string: the variable name associated with the group and binding, or an empty string if not found
	BindGroupVarName(group, binding int) string

	// BindGroupFromVarName retrieves the binding index for a given group and variable name, if it exists.
	//
	// Parameters:
	//   - group: the bind group index
	//   - varName: the variable name within the group
	//
	// Returns:
	//   - int: the binding index associated with the variable name, or -1 if not found
	//   - bool: true if the variable name was found, false otherwise
	BindGroupFromVarName(group int, varName string) (int, bool)

	// BindGroupVarNames retrieves all variable names for all bind groups.
	//
	// Returns:
	//   - map[int]map[int]string: variable names keyed by group and binding index
	BindGroupVarNames() map[int]map[int]string

	// VertexLayout retrieves the vertex buffer layout for a specific key.
	//
	// Parameters:
	//   - key: the integer key identifying the vertex layout
	//
	// Returns:
	//   - []wgpu.VertexBufferLayout: the vertex buffer layout associated with the key, or nil if not set
	VertexLayout(key int) []wgpu.VertexBufferLayout

	// VertexLayouts retrieves all vertex buffer layouts associated with this shader.
	//
	// Returns:
	//   - map[int][]wgpu.VertexBufferLayout: a map of keys to their corresponding vertex buffer layouts
	VertexLayouts() map[int][]wgpu.VertexBufferLayout

	// EntryPoint returns the entry point name for this shader.
	//
	// Returns:
	//   - string: the entry point name (e.g. "main")
	EntryPoint() string

	// WorkgroupSize returns the workgroup size dimensions for compute shaders.
	// Returns [0, 0, 0] for non-compute shaders and [1, 1, 1] as the default when
	// @workgroup_size is not specified.
	//
	// Returns:
	//   - [3]uint32: the workgroup size as [x, y, z]
	WorkgroupSize() [3]uint32

	// Module returns the wgpu.ShaderModuleDescriptor for this shader, which is built from the NewShader function.
	//
	// Returns:
	//   - *wgpu.ShaderModuleDescriptor: the shader module descriptor containing the WGSL code and label
	Module() *wgpu.ShaderModuleDescriptor

	// ShaderType returns the type of the shader (vertex, fragment, or compute).
	//
	// Returns:
	//   - ShaderType: ShaderTypeVertex, ShaderTypeFragment, or ShaderTypeCompute
	ShaderType() ShaderType

	// Declarations returns the list of parsed annotations from the shader source that represent resource bindings and providers.
	// This is used by the scene to match shaders with game objects and their associated resource providers.
	//
	// Returns:
	//   - []Annotation: a slice of annotations representing bind group declarations and providers parsed from the shader source
	Declarations() []Annotation
}

var _ Shader = &shader{}

// NewShader creates a new Shader instance with all specified options applied.
// The VertexLayouts are automatically parsed from the source code if WithSource is used.
// Additionally, the VertexLayouts will be automatically parsed when setting the source via SetSource.
//
// Parameters:
//   - key: a unique identifier for the shader, used for caching and lookups
//   - shaderType: the type of shader (vertex, fragment or compute), used for validation and pipeline setup
//   - sourcePath: the file path to read WGSL source from
//
// Returns:
//   - Shader: a new Shader instance with the provided configuration
func NewShader(key string, shaderType ShaderType, sourcePath string) Shader {
	if sourcePath == "" {
		panic(fmt.Sprintf("shader: %s must have a valid source provided via WithSourceFromPath", key))
	}
	s := &shader{
		key:                        key,
		shaderType:                 shaderType,
		bindGroupLayoutDescriptors: make(map[int]wgpu.BindGroupLayoutDescriptor),
		bindingVarNames:            make(map[int]map[int]string),
		vertexLayouts:              make(map[int][]wgpu.VertexBufferLayout),
		workGroupSize:              [3]uint32{0, 0, 0},
		pp:                         NewPreProcessor(),
	}
	s.parseSourceFromPath(sourcePath)
	return s
}

func (s *shader) Key() string {
	return s.key
}

func (s *shader) Source() string {
	return s.source
}

func (s *shader) VertexLayout(key int) []wgpu.VertexBufferLayout {
	return s.vertexLayouts[key]
}

func (s *shader) VertexLayouts() map[int][]wgpu.VertexBufferLayout {
	return s.vertexLayouts
}

func (s *shader) EntryPoint() string {
	return s.entryPoint
}

func (s *shader) WorkgroupSize() [3]uint32 {
	return s.workGroupSize
}

func (s *shader) BindGroupLayoutDescriptor(bindingKey int) wgpu.BindGroupLayoutDescriptor {
	return s.bindGroupLayoutDescriptors[bindingKey]
}

func (s *shader) BindGroupLayoutDescriptors() map[int]wgpu.BindGroupLayoutDescriptor {
	return s.bindGroupLayoutDescriptors
}

func (s *shader) BindGroupVarName(group, binding int) string {
	if s.bindingVarNames[group] == nil {
		return ""
	}
	return s.bindingVarNames[group][binding]
}

func (s *shader) BindGroupFromVarName(group int, varName string) (int, bool) {
	if s.bindingVarNames[group] == nil {
		return -1, false
	}
	for binding, name := range s.bindingVarNames[group] {
		if name == varName {
			return binding, true
		}
	}
	return -1, false
}

func (s *shader) BindGroupVarNames() map[int]map[int]string {
	return s.bindingVarNames
}

func (s *shader) Module() *wgpu.ShaderModuleDescriptor {
	return s.module
}

func (s *shader) SetVertexLayout(key int, layout []wgpu.VertexBufferLayout) {
	s.vertexLayouts[key] = layout
}

func (s *shader) SetVertexLayouts(layouts map[int][]wgpu.VertexBufferLayout) {
	s.vertexLayouts = layouts
}

func (s *shader) ShaderType() ShaderType {
	return s.shaderType
}

func (s *shader) Declarations() []Annotation {
	return s.pp.Declarations()
}

// parseSource sets the WGSL source, builds the shader module descriptor, parses the
// entry point name, and extracts layout metadata appropriate for the shader type.
// Vertex shaders get vertex buffer layouts parsed. Compute shaders get workgroup size
// parsed. All shader types get bind group layout descriptors parsed.
func (s *shader) parseSourceFromPath(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(fmt.Sprintf("shader: failed to read source file %q: %v", path, err))
	}
	s.source, err = s.pp.Process(string(data))
	if err != nil {
		panic(fmt.Sprintf("shader: failed to pre-process shader source %q: %v", path, err))
	}
	s.module = &wgpu.ShaderModuleDescriptor{
		Label: s.key,
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{
			Code: s.source,
		},
	}
	s.entryPoint = parseEntryPoint(s.source, s.shaderType)
	if s.shaderType == ShaderTypeVertex {
		s.vertexLayouts = parseVertexLayouts(s.source)
	}
	if s.shaderType == ShaderTypeCompute {
		s.workGroupSize = parseWorkgroupSize(s.source)
	}
	var visibility wgpu.ShaderStage
	switch s.shaderType {
	case ShaderTypeVertex:
		visibility = wgpu.ShaderStageVertex
	case ShaderTypeFragment:
		visibility = wgpu.ShaderStageFragment
	case ShaderTypeCompute:
		visibility = wgpu.ShaderStageCompute
	default:
		visibility = wgpu.ShaderStageNone
	}
	s.bindGroupLayoutDescriptors, s.bindingVarNames = parseBindGroupLayouts(s.source, visibility)
}
