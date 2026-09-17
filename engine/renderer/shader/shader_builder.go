package shader

import (
	"fmt"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// ShaderBuilderOption is a functional option for configuring a Shader during construction.
// Use the With* functions to create options that are applied directly to the shader instance.
type ShaderBuilderOption func(*shader)

// WithInjections sets the injection map forwarded to the pre-processor for @oxy:inject resolution.
//
// Parameters:
//   - injections: a map of injection key-value pairs to substitute into the shader source
//
// Returns:
//   - ShaderBuilderOption: option function to apply
func WithInjections(injections map[string]string) ShaderBuilderOption {
	return func(s *shader) { s.injections = injections }
}

// NewShader creates a new Shader instance with all specified options applied.
// The VertexLayouts are automatically parsed from the source code if WithSource is used.
// Additionally, the VertexLayouts will be automatically parsed when setting the source via SetSource.
//
// Parameters:
//   - key: a unique identifier for the shader, used for caching and lookups
//   - shaderType: the type of shader (vertex, fragment or compute), used for validation and pipeline setup
//   - sourcePath: the file path to read WGSL source from
//   - opts: optional builder options (e.g. [WithInjections]) applied before source parsing
//
// Returns:
//   - Shader: a new Shader instance with the provided configuration
func NewShader(key string, shaderType ShaderType, sourcePath string, opts ...ShaderBuilderOption) Shader {
	if sourcePath == "" {
		panic(fmt.Sprintf("shader: %s must have a valid source provided", key))
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
	for _, opt := range opts {
		opt(s)
	}
	s.parseSourceFromPath(sourcePath, s.injections)
	s.Delegate = s
	return s
}
