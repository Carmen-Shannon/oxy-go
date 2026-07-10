package shader

import (
	"fmt"
	"os"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/oliverbestmann/webgpu/wgpu"
)

// shader is the implementation of the Shader interface.
// It holds all of the persistent shader data required for pipeline creation and material binding.
type shader struct {
	common.DelegateImpl[Shader]

	key                        string
	source                     string
	shaderType                 ShaderType
	bindGroupLayoutDescriptors map[int]wgpu.BindGroupLayoutDescriptor
	bindingVarNames            map[int]map[int]string
	vertexLayouts              map[int][]wgpu.VertexBufferLayout
	workGroupSize              [3]uint32
	entryPoint                 string
	module                     *wgpu.ShaderModuleDescriptor
	injections                 map[string]string

	pp PreProcessor
}

// parseSourceFromPath sets the WGSL source, builds the shader module descriptor, parses the
// entry point name, and extracts layout metadata appropriate for the shader type.
// Vertex shaders get vertex buffer layouts parsed. Compute shaders get workgroup size
// parsed. All shader types get bind group layout descriptors parsed.
func (s *shader) parseSourceFromPath(path string, injections map[string]string) {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(fmt.Sprintf("shader: failed to read source file %q: %v", path, err))
	}
	s.source, err = s.pp.Process(string(data), injections)
	if err != nil {
		panic(fmt.Sprintf("shader: failed to pre-process shader source %q: %v", path, err))
	}
	s.module = &wgpu.ShaderModuleDescriptor{
		Label: s.key,
		WGSLSource: &wgpu.ShaderSourceWGSL{
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
