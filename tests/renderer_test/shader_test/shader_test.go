package shader_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/suite"
)

type shaderTest struct {
	suite.Suite
}

func TestShader(t *testing.T) {
	suite.Run(t, new(shaderTest))
}

func (suite *shaderTest) TestShaderTypeConstants() {
	suite.Run("ShaderTypeCompute is zero", func() {
		suite.Equal(shader.ShaderType(0), shader.ShaderTypeCompute)
	})

	suite.Run("ShaderTypeVertex is one", func() {
		suite.Equal(shader.ShaderType(1), shader.ShaderTypeVertex)
	})

	suite.Run("ShaderTypeFragment is two", func() {
		suite.Equal(shader.ShaderType(2), shader.ShaderTypeFragment)
	})
}

func (suite *shaderTest) TestNewShaderPanicsWithEmptyPath() {
	suite.Run("panics when source path is empty", func() {
		suite.Panics(func() {
			shader.NewShader("test", shader.ShaderTypeVertex, "")
		})
	})
}

func (suite *shaderTest) TestNewShaderPanicsWithInvalidPath() {
	suite.Run("panics when source path does not exist", func() {
		suite.Panics(func() {
			shader.NewShader("test", shader.ShaderTypeVertex, "nonexistent/path.wgsl")
		})
	})
}

func (suite *shaderTest) TestNewVertexShader() {
	suite.Run("key is preserved", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		suite.Equal("test-vert", s.Key())
	})

	suite.Run("shader type is vertex", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		suite.Equal(shader.ShaderTypeVertex, s.ShaderType())
	})

	suite.Run("source is non-empty", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		suite.NotEmpty(s.Source())
	})

	suite.Run("entry point is vs_main", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		suite.Equal("vs_main", s.EntryPoint())
	})

	suite.Run("module descriptor is populated", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		suite.NotNil(s.Module())
		suite.Equal("test-vert", s.Module().Label)
		suite.NotNil(s.Module().WGSLDescriptor)
		suite.NotEmpty(s.Module().WGSLDescriptor.Code)
	})

	suite.Run("vertex layouts are parsed", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		layouts := s.VertexLayouts()
		suite.NotEmpty(layouts)
		suite.NotNil(s.VertexLayout(0))
	})

	suite.Run("vertex layout has correct attributes for VertexInput", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		layout := s.VertexLayout(0)
		suite.NotNil(layout)
		suite.Len(layout, 1)
		suite.Equal(wgpu.VertexStepModeVertex, layout[0].StepMode)
		// VertexInput has position: vec3<f32> => 12 bytes
		suite.Equal(uint64(12), layout[0].ArrayStride)
		suite.Len(layout[0].Attributes, 1)
		suite.Equal(wgpu.VertexFormatFloat32x3, layout[0].Attributes[0].Format)
	})

	suite.Run("bind group layouts are parsed", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		descriptors := s.BindGroupLayoutDescriptors()
		suite.NotEmpty(descriptors)
		// group 0 has camera uniform, group 1 has instance buffer
		suite.Contains(descriptors, 0)
		suite.Contains(descriptors, 1)
	})

	suite.Run("bind group 0 has uniform buffer type", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		suite.Len(desc.Entries, 1)
		suite.Equal(wgpu.BufferBindingTypeUniform, desc.Entries[0].Buffer.Type)
		suite.Equal(wgpu.ShaderStageVertex, desc.Entries[0].Visibility)
	})

	suite.Run("bind group 1 has read-only storage buffer type", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		desc := s.BindGroupLayoutDescriptor(1)
		suite.Len(desc.Entries, 1)
		suite.Equal(wgpu.BufferBindingTypeReadOnlyStorage, desc.Entries[0].Buffer.Type)
	})

	suite.Run("workgroup size is zero for vertex shaders", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		suite.Equal([3]uint32{0, 0, 0}, s.WorkgroupSize())
	})

	suite.Run("bind group var names are tracked", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		suite.Equal("camera", s.BindGroupVarName(0, 0))
		suite.Equal("instance_buffer", s.BindGroupVarName(1, 0))
	})

	suite.Run("bind group from var name returns correct binding", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		binding, ok := s.BindGroupFromVarName(0, "camera")
		suite.True(ok)
		suite.Equal(0, binding)
	})

	suite.Run("bind group from var name returns false for unknown", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		binding, ok := s.BindGroupFromVarName(0, "nonexistent")
		suite.False(ok)
		suite.Equal(-1, binding)
	})

	suite.Run("bind group var name returns empty for nonexistent group", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		suite.Equal("", s.BindGroupVarName(99, 0))
	})

	suite.Run("bind group from var name returns false for nonexistent group", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		binding, ok := s.BindGroupFromVarName(99, "camera")
		suite.False(ok)
		suite.Equal(-1, binding)
	})

	suite.Run("bind group var names returns full map", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		names := s.BindGroupVarNames()
		suite.NotEmpty(names)
		suite.Contains(names, 0)
		suite.Contains(names, 1)
	})
}

func (suite *shaderTest) TestNewFragmentShader() {
	suite.Run("shader type is fragment", func() {
		s := shader.NewShader("test-frag", shader.ShaderTypeFragment, "../../assets/shaders/test_frag.wgsl")
		suite.Equal(shader.ShaderTypeFragment, s.ShaderType())
	})

	suite.Run("entry point is fs_main", func() {
		s := shader.NewShader("test-frag", shader.ShaderTypeFragment, "../../assets/shaders/test_frag.wgsl")
		suite.Equal("fs_main", s.EntryPoint())
	})

	suite.Run("vertex layouts are empty for fragment shaders", func() {
		s := shader.NewShader("test-frag", shader.ShaderTypeFragment, "../../assets/shaders/test_frag.wgsl")
		suite.Empty(s.VertexLayouts())
	})

	suite.Run("workgroup size is zero for fragment shaders", func() {
		s := shader.NewShader("test-frag", shader.ShaderTypeFragment, "../../assets/shaders/test_frag.wgsl")
		suite.Equal([3]uint32{0, 0, 0}, s.WorkgroupSize())
	})
}

func (suite *shaderTest) TestNewComputeShader() {
	suite.Run("shader type is compute", func() {
		s := shader.NewShader("test-compute", shader.ShaderTypeCompute, "../../assets/shaders/test_compute.wgsl")
		suite.Equal(shader.ShaderTypeCompute, s.ShaderType())
	})

	suite.Run("entry point is cs_main", func() {
		s := shader.NewShader("test-compute", shader.ShaderTypeCompute, "../../assets/shaders/test_compute.wgsl")
		suite.Equal("cs_main", s.EntryPoint())
	})

	suite.Run("workgroup size is 64,1,1 for single-dimension compute", func() {
		s := shader.NewShader("test-compute", shader.ShaderTypeCompute, "../../assets/shaders/test_compute.wgsl")
		suite.Equal([3]uint32{64, 1, 1}, s.WorkgroupSize())
	})

	suite.Run("vertex layouts are empty for compute shaders", func() {
		s := shader.NewShader("test-compute", shader.ShaderTypeCompute, "../../assets/shaders/test_compute.wgsl")
		suite.Empty(s.VertexLayouts())
	})

	suite.Run("bind groups include uniform and storage bindings", func() {
		s := shader.NewShader("test-compute", shader.ShaderTypeCompute, "../../assets/shaders/test_compute.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		suite.Len(desc.Entries, 2)
		suite.Equal(wgpu.BufferBindingTypeUniform, desc.Entries[0].Buffer.Type)
		suite.Equal(wgpu.BufferBindingTypeStorage, desc.Entries[1].Buffer.Type)
		suite.Equal(wgpu.ShaderStageCompute, desc.Entries[0].Visibility)
	})
}

func (suite *shaderTest) TestComputeShader3DWorkgroupSize() {
	suite.Run("workgroup size is 8,4,2 for 3D compute", func() {
		s := shader.NewShader("test-3d", shader.ShaderTypeCompute, "../../assets/shaders/test_compute_3d.wgsl")
		suite.Equal([3]uint32{8, 4, 2}, s.WorkgroupSize())
	})

	suite.Run("has three bind group entries in group 0", func() {
		s := shader.NewShader("test-3d", shader.ShaderTypeCompute, "../../assets/shaders/test_compute_3d.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		suite.Len(desc.Entries, 3)
		suite.Equal(wgpu.BufferBindingTypeUniform, desc.Entries[0].Buffer.Type)
		suite.Equal(wgpu.BufferBindingTypeStorage, desc.Entries[1].Buffer.Type)
		suite.Equal(wgpu.BufferBindingTypeReadOnlyStorage, desc.Entries[2].Buffer.Type)
	})
}

func (suite *shaderTest) TestAnnotatedVertexShader() {
	suite.Run("preprocessor injects struct definitions and generates bindings", func() {
		s := shader.NewShader("annotated-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_annotated_vert.wgsl")
		source := s.Source()
		suite.Contains(source, "CameraUniform")
		suite.Contains(source, "InstanceData")
		suite.Contains(source, "ModelData")
		suite.Contains(source, "VertexInput")
		suite.Contains(source, "@group(0) @binding(0) var<uniform> camera: CameraUniform;")
		suite.Contains(source, "@group(1) @binding(0) var<storage, read> instance_buffer: array<InstanceData>;")
		suite.Contains(source, "@group(1) @binding(1) var<storage, read> model_buffer: array<ModelData>;")
	})

	suite.Run("entry point is vs_main", func() {
		s := shader.NewShader("annotated-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_annotated_vert.wgsl")
		suite.Equal("vs_main", s.EntryPoint())
	})

	suite.Run("declarations include groups and providers", func() {
		s := shader.NewShader("annotated-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_annotated_vert.wgsl")
		decls := s.Declarations()
		suite.NotEmpty(decls)

		groupCount := 0
		providerCount := 0
		for _, d := range decls {
			switch d.Type {
			case shader.AnnotationTypeBindingGroup:
				groupCount++
			case shader.AnnotationTypeProvider:
				providerCount++
			}
		}
		suite.Equal(3, groupCount)
		suite.Equal(2, providerCount)
	})

	suite.Run("vertex layouts are parsed from injected VertexInput", func() {
		s := shader.NewShader("annotated-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_annotated_vert.wgsl")
		layouts := s.VertexLayouts()
		suite.NotEmpty(layouts)
		// VertexInput has position(vec3f), normal(vec3f), uv(vec2f), color(vec4f), tangent(vec4f)
		layout := s.VertexLayout(0)
		suite.NotNil(layout)
		suite.Len(layout, 1)
		suite.Len(layout[0].Attributes, 5)
		// total stride: 12 + 12 + 8 + 16 + 16 = 64
		suite.Equal(uint64(64), layout[0].ArrayStride)
	})
}

func (suite *shaderTest) TestAnnotatedFragmentShader() {
	suite.Run("preprocessor injects structs and generates group bindings", func() {
		s := shader.NewShader("annotated-frag", shader.ShaderTypeFragment, "../../assets/shaders/test_annotated_frag.wgsl")
		source := s.Source()
		suite.Contains(source, "OverlayParams")
		suite.Contains(source, "EffectParams")
		suite.Contains(source, "@group(0) @binding(0) var<uniform> overlay_uniform: OverlayParams;")
		suite.Contains(source, "@group(0) @binding(1) var<uniform> effect_uniform: EffectParams;")
	})

	suite.Run("entry point is fs_main", func() {
		s := shader.NewShader("annotated-frag", shader.ShaderTypeFragment, "../../assets/shaders/test_annotated_frag.wgsl")
		suite.Equal("fs_main", s.EntryPoint())
	})

	suite.Run("bind group layouts include texture and sampler types", func() {
		s := shader.NewShader("annotated-frag", shader.ShaderTypeFragment, "../../assets/shaders/test_annotated_frag.wgsl")
		desc := s.BindGroupLayoutDescriptor(1)
		suite.Len(desc.Entries, 2)
		// binding 0 is a texture_2d<f32>
		suite.Equal(wgpu.TextureViewDimension2D, desc.Entries[0].Texture.ViewDimension)
		suite.Equal(wgpu.TextureSampleTypeFloat, desc.Entries[0].Texture.SampleType)
		// binding 1 is a sampler
		suite.Equal(wgpu.SamplerBindingTypeFiltering, desc.Entries[1].Sampler.Type)
	})
}

func (suite *shaderTest) TestTextureTypesFragmentShader() {
	suite.Run("depth texture is classified correctly", func() {
		s := shader.NewShader("tex-types", shader.ShaderTypeFragment, "../../assets/shaders/test_texture_types_frag.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		suite.True(len(desc.Entries) >= 6)

		// binding 0: texture_depth_2d
		suite.Equal(wgpu.TextureSampleTypeDepth, desc.Entries[0].Texture.SampleType)
		suite.Equal(wgpu.TextureViewDimension2D, desc.Entries[0].Texture.ViewDimension)
	})

	suite.Run("comparison sampler is classified correctly", func() {
		s := shader.NewShader("tex-types", shader.ShaderTypeFragment, "../../assets/shaders/test_texture_types_frag.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// binding 1: sampler_comparison
		suite.Equal(wgpu.SamplerBindingTypeComparison, desc.Entries[1].Sampler.Type)
	})

	suite.Run("cube texture is classified correctly", func() {
		s := shader.NewShader("tex-types", shader.ShaderTypeFragment, "../../assets/shaders/test_texture_types_frag.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// binding 2: texture_cube<f32>
		suite.Equal(wgpu.TextureViewDimensionCube, desc.Entries[2].Texture.ViewDimension)
		suite.Equal(wgpu.TextureSampleTypeFloat, desc.Entries[2].Texture.SampleType)
	})

	suite.Run("2d array texture is classified correctly", func() {
		s := shader.NewShader("tex-types", shader.ShaderTypeFragment, "../../assets/shaders/test_texture_types_frag.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// binding 3: texture_2d_array<f32>
		suite.Equal(wgpu.TextureViewDimension2DArray, desc.Entries[3].Texture.ViewDimension)
	})

	suite.Run("storage texture is classified correctly", func() {
		s := shader.NewShader("tex-types", shader.ShaderTypeFragment, "../../assets/shaders/test_texture_types_frag.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// binding 4: texture_storage_2d<rgba8unorm, write>
		suite.Equal(wgpu.TextureViewDimension2D, desc.Entries[4].StorageTexture.ViewDimension)
		suite.Equal(wgpu.TextureFormatRGBA8Unorm, desc.Entries[4].StorageTexture.Format)
		suite.Equal(wgpu.StorageTextureAccessWriteOnly, desc.Entries[4].StorageTexture.Access)
	})

	suite.Run("multisampled texture is classified correctly", func() {
		s := shader.NewShader("tex-types", shader.ShaderTypeFragment, "../../assets/shaders/test_texture_types_frag.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// binding 5: texture_multisampled_2d<f32>
		suite.Equal(wgpu.TextureViewDimension2D, desc.Entries[5].Texture.ViewDimension)
		suite.True(desc.Entries[5].Texture.Multisampled)
	})
}

func (suite *shaderTest) TestShadowVertexShader() {
	suite.Run("shadow vertex shader parses correctly", func() {
		s := shader.NewShader("shadow", shader.ShaderTypeVertex, "../../assets/shaders/test_shadow_vert.wgsl")
		suite.Equal("vs_main", s.EntryPoint())
		suite.Equal(shader.ShaderTypeVertex, s.ShaderType())
	})

	suite.Run("shadow uniform has correct MinBindingSize", func() {
		s := shader.NewShader("shadow", shader.ShaderTypeVertex, "../../assets/shaders/test_shadow_vert.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		suite.Len(desc.Entries, 1)
		// ShadowUniform has a single mat4x4<f32> = 64 bytes
		suite.Equal(uint64(64), desc.Entries[0].Buffer.MinBindingSize)
	})
}

func (suite *shaderTest) TestBlockCommentShader() {
	suite.Run("block comments are stripped and shader parses correctly", func() {
		s := shader.NewShader("block-comments", shader.ShaderTypeCompute, "../../assets/shaders/test_block_comments.wgsl")
		suite.Equal("cs_main", s.EntryPoint())
		suite.Equal([3]uint32{16, 8, 1}, s.WorkgroupSize())
	})

	suite.Run("source preserves block comments but parsing still succeeds", func() {
		s := shader.NewShader("block-comments", shader.ShaderTypeCompute, "../../assets/shaders/test_block_comments.wgsl")
		source := s.Source()
		// Pre-processor keeps block comments; the parser strips them internally
		suite.Contains(source, "/*")
	})

	suite.Run("bind group layout contains uniform params", func() {
		s := shader.NewShader("block-comments", shader.ShaderTypeCompute, "../../assets/shaders/test_block_comments.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		suite.Len(desc.Entries, 1)
		suite.Equal(wgpu.BufferBindingTypeUniform, desc.Entries[0].Buffer.Type)
	})
}

func (suite *shaderTest) TestRuntimeSizedArrayShader() {
	suite.Run("shader with runtime-sized array in struct parses bind groups", func() {
		s := shader.NewShader("runtime-array", shader.ShaderTypeCompute, "../../assets/shaders/test_runtime_array.wgsl")
		suite.Equal("cs_main", s.EntryPoint())
		suite.Equal([3]uint32{64, 1, 1}, s.WorkgroupSize())
	})

	suite.Run("bind group has two entries for data and output", func() {
		s := shader.NewShader("runtime-array", shader.ShaderTypeCompute, "../../assets/shaders/test_runtime_array.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		suite.Len(desc.Entries, 2)
		suite.Equal(wgpu.BufferBindingTypeReadOnlyStorage, desc.Entries[0].Buffer.Type)
		suite.Equal(wgpu.BufferBindingTypeStorage, desc.Entries[1].Buffer.Type)
	})

	suite.Run("DataBuffer MinBindingSize includes runtime array element stride", func() {
		s := shader.NewShader("runtime-array", shader.ShaderTypeCompute, "../../assets/shaders/test_runtime_array.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// DataBuffer: Header(u32+f32=8,align4) + array<Element>(vec4<f32>=16,align16)
		// offset: align(4,0)=0+8=8, align(16,8)=16+16=32, maxAlign=16, roundUp(16,32)=32
		suite.Equal(uint64(32), desc.Entries[0].Buffer.MinBindingSize)
	})

	suite.Run("output runtime array MinBindingSize is element stride", func() {
		s := shader.NewShader("runtime-array", shader.ShaderTypeCompute, "../../assets/shaders/test_runtime_array.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// output: array<vec4<f32>> => element size=16, align=16, stride=16
		suite.Equal(uint64(16), desc.Entries[1].Buffer.MinBindingSize)
	})

	suite.Run("var names are tracked correctly", func() {
		s := shader.NewShader("runtime-array", shader.ShaderTypeCompute, "../../assets/shaders/test_runtime_array.wgsl")
		suite.Equal("data", s.BindGroupVarName(0, 0))
		suite.Equal("output", s.BindGroupVarName(0, 1))
	})
}

func (suite *shaderTest) TestOnlyRuntimeArrayShader() {
	suite.Run("shader with struct containing only runtime array parses correctly", func() {
		s := shader.NewShader("only-runtime", shader.ShaderTypeCompute, "../../assets/shaders/test_only_runtime_array.wgsl")
		suite.Equal("cs_main", s.EntryPoint())
		suite.Equal([3]uint32{32, 1, 1}, s.WorkgroupSize())
	})

	suite.Run("OnlyArray MinBindingSize equals element stride", func() {
		s := shader.NewShader("only-runtime", shader.ShaderTypeCompute, "../../assets/shaders/test_only_runtime_array.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		suite.Len(desc.Entries, 1)
		// Item { pos: vec3<f32> (12 bytes, align 16), weight: f32 (4 bytes, align 4) }
		// offset 0: vec3<f32> size=12
		// offset 12: align(4,12)=12, + 4 = 16
		// struct align=16, roundUp(16,16) = 16
		// OnlyArray has only runtime array -> offset==0 -> uses element size = 16
		suite.Equal(uint64(16), desc.Entries[0].Buffer.MinBindingSize)
	})
}

func (suite *shaderTest) TestBadVertexTypeShader() {
	suite.Run("vertex shader with unrecognized type produces empty vertex layouts", func() {
		s := shader.NewShader("bad-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_bad_vertex_type.wgsl")
		suite.Equal("vs_main", s.EntryPoint())
		layouts := s.VertexLayouts()
		suite.Empty(layouts)
	})

	suite.Run("bind group layout still parses despite bad vertex type", func() {
		s := shader.NewShader("bad-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_bad_vertex_type.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		suite.Len(desc.Entries, 1)
		suite.Equal(wgpu.BufferBindingTypeUniform, desc.Entries[0].Buffer.Type)
	})
}

func (suite *shaderTest) TestParserCoverageShader() {
	suite.Run("reversed struct dependency order resolves correctly", func() {
		s := shader.NewShader("parser-cov", shader.ShaderTypeCompute, "../../assets/shaders/test_parser_coverage.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// Outer depends on Inner (defined after), Inner={x:f32,y:f32}={8,4}
		// Outer: Inner(8)+ f32(4) = 12, align 4 => 12 bytes
		suite.Equal(uint64(12), desc.Entries[0].Buffer.MinBindingSize)
	})

	suite.Run("fixed-size array struct has correct MinBindingSize", func() {
		s := shader.NewShader("parser-cov", shader.ShaderTypeCompute, "../../assets/shaders/test_parser_coverage.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// WithFixedArray: array<f32, 4> => stride=4, count=4, total=16, align 4
		suite.Equal(uint64(16), desc.Entries[1].Buffer.MinBindingSize)
	})

	suite.Run("runtime array with unknown element uses fixed prefix size", func() {
		s := shader.NewShader("parser-cov", shader.ShaderTypeCompute, "../../assets/shaders/test_parser_coverage.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// WithUnknownRuntime: u32(4) + array<NeverDefinedType>(fails) => fallback prefix=4
		suite.Equal(uint64(4), desc.Entries[2].Buffer.MinBindingSize)
	})

	suite.Run("struct with only unknown runtime array has zero MinBindingSize", func() {
		s := shader.NewShader("parser-cov", shader.ShaderTypeCompute, "../../assets/shaders/test_parser_coverage.wgsl")
		desc := s.BindGroupLayoutDescriptor(0)
		// OnlyUnknownArray: only field is array<AlsoNeverDefined> => offset=0, inner resolve fails
		// struct resolves as {0, 1} => MinBindingSize condition layout.size>0 is false => 0
		suite.Equal(uint64(0), desc.Entries[3].Buffer.MinBindingSize)
	})

	suite.Run("entry point and workgroup size parse correctly", func() {
		s := shader.NewShader("parser-cov", shader.ShaderTypeCompute, "../../assets/shaders/test_parser_coverage.wgsl")
		suite.Equal("cs_main", s.EntryPoint())
		suite.Equal([3]uint32{16, 1, 1}, s.WorkgroupSize())
	})
}

func (suite *shaderTest) TestSetDelegate() {
	suite.Run("set delegate does not panic", func() {
		s := shader.NewShader("test-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_vert.wgsl")
		suite.NotPanics(func() {
			s.SetDelegate(s)
		})
	})
}

func (suite *shaderTest) TestNoEntryPointShader() {
	suite.Run("compute shader without entry point returns empty entry point", func() {
		s := shader.NewShader("no-entry", shader.ShaderTypeCompute, "../../assets/shaders/test_no_entry.wgsl")
		suite.Equal("", s.EntryPoint())
	})

	suite.Run("compute shader without workgroup_size defaults to 1,1,1", func() {
		s := shader.NewShader("no-entry", shader.ShaderTypeCompute, "../../assets/shaders/test_no_entry.wgsl")
		suite.Equal([3]uint32{1, 1, 1}, s.WorkgroupSize())
	})

	suite.Run("fragment shader without entry point returns empty entry point", func() {
		s := shader.NewShader("no-entry-frag", shader.ShaderTypeFragment, "../../assets/shaders/test_no_entry.wgsl")
		suite.Equal("", s.EntryPoint())
	})

	suite.Run("vertex shader without entry point returns empty entry point", func() {
		s := shader.NewShader("no-entry-vert", shader.ShaderTypeVertex, "../../assets/shaders/test_no_entry.wgsl")
		suite.Equal("", s.EntryPoint())
	})
}
