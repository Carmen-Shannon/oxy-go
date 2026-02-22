package shader_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/stretchr/testify/suite"
)

type annotationsTest struct {
	suite.Suite
	pp shader.PreProcessor
}

func TestAnnotations(t *testing.T) {
	suite.Run(t, new(annotationsTest))
}

func (suite *annotationsTest) SetupTest() {
	suite.pp = shader.NewPreProcessor()
}

func (suite *annotationsTest) TestIncludeAnnotation() {
	suite.Run("camera include injects CameraUniform struct", func() {
		source := "//@oxy:include camera\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "CameraUniform")
		suite.Contains(out, "view_proj")
	})

	suite.Run("overlay_params include injects OverlayParams struct", func() {
		source := "//@oxy:include overlay_params\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "OverlayParams")
	})

	suite.Run("effect_params include injects EffectParams struct", func() {
		source := "//@oxy:include effect_params\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "EffectParams")
	})

	suite.Run("light include injects Light struct", func() {
		source := "//@oxy:include light\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "Light")
	})

	suite.Run("light_header include injects LightHeader struct", func() {
		source := "//@oxy:include light_header\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "LightHeader")
	})

	suite.Run("shadow_data include injects ShadowData struct", func() {
		source := "//@oxy:include shadow_data\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "ShadowData")
	})

	suite.Run("shadow_uniform include injects ShadowUniform struct", func() {
		source := "//@oxy:include shadow_uniform\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "ShadowUniform")
	})

	suite.Run("tile_uniforms include injects TileUniforms struct", func() {
		source := "//@oxy:include tile_uniforms\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "TileUniforms")
	})

	suite.Run("animation_data include injects AnimationData struct", func() {
		source := "//@oxy:include animation_data\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "AnimationData")
	})

	suite.Run("skeletal_animation_data include injects SkeletalAnimationData struct", func() {
		source := "//@oxy:include skeletal_animation_data\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "SkeletalAnimationData")
	})

	suite.Run("animation_globals include injects AnimationGlobals struct", func() {
		source := "//@oxy:include animation_globals\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "AnimationGlobals")
	})

	suite.Run("global_data include injects GlobalData struct", func() {
		source := "//@oxy:include global_data\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "GlobalData")
	})

	suite.Run("indirect_args include injects IndirectArgs struct", func() {
		source := "//@oxy:include indirect_args\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "IndirectArgs")
	})

	suite.Run("bone_info include injects BoneInfo struct", func() {
		source := "//@oxy:include bone_info\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "BoneInfo")
	})

	suite.Run("instance_data include injects InstanceData struct", func() {
		source := "//@oxy:include instance_data\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "InstanceData")
	})

	suite.Run("model_data include injects ModelData struct", func() {
		source := "//@oxy:include model_data\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "ModelData")
	})

	suite.Run("vertex include injects VertexInput struct", func() {
		source := "//@oxy:include vertex\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "VertexInput")
	})

	suite.Run("skinned_vertex include injects VertexInput struct", func() {
		source := "//@oxy:include skinned_vertex\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "VertexInput")
	})

	suite.Run("frustum_plane include injects FrustumPlane struct", func() {
		source := "//@oxy:include frustum_plane\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "FrustumPlane")
	})

	suite.Run("light_cull_uniforms include injects LightCullUniforms struct", func() {
		source := "//@oxy:include light_cull_uniforms\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "LightCullUniforms")
	})

	suite.Run("include does not produce declarations", func() {
		source := "//@oxy:include camera\n"
		_, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Empty(suite.pp.Declarations())
	})
}

func (suite *annotationsTest) TestIncludeAnnotationErrors() {
	suite.Run("empty annotation returns error", func() {
		source := "//@oxy:\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "empty @oxy annotation")
	})

	suite.Run("include with no argument returns error", func() {
		source := "//@oxy:include\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "requires exactly one argument")
	})

	suite.Run("include with too many arguments returns error", func() {
		source := "//@oxy:include camera extra\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "requires exactly one argument")
	})

	suite.Run("include with unknown struct type returns error", func() {
		source := "//@oxy:include unknown_type\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "unknown struct type")
	})

	suite.Run("unknown annotation type returns error", func() {
		source := "//@oxy:foobar something\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "unknown @oxy annotation type")
	})
}

func (suite *annotationsTest) TestGroupAnnotation() {
	suite.Run("group annotation generates binding declaration", func() {
		source := "//@oxy:group 0 0 storage_uniform camera camera\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "@group(0) @binding(0) var<uniform> camera: CameraUniform;")
	})

	suite.Run("group annotation with storage_read generates read binding", func() {
		source := "//@oxy:group 1 0 storage_read buffer instance_data\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "@group(1) @binding(0) var<storage, read> buffer: InstanceData;")
	})

	suite.Run("group annotation with storage_read_write generates read_write binding", func() {
		source := "//@oxy:group 2 1 storage_read_write output instance_data\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "@group(2) @binding(1) var<storage, read_write> output: InstanceData;")
	})

	suite.Run("group annotation with array type generates array binding", func() {
		source := "//@oxy:group 1 0 storage_read buffer array<instance_data>\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "@group(1) @binding(0) var<storage, read> buffer: array<InstanceData>;")
	})

	suite.Run("group annotation adds to declarations", func() {
		source := "//@oxy:group 0 0 storage_uniform cam camera\n"
		_, err := suite.pp.Process(source)
		suite.NoError(err)
		decls := suite.pp.Declarations()
		suite.Len(decls, 1)
		suite.Equal(shader.AnnotationTypeBindingGroup, decls[0].Type)
		suite.NotNil(decls[0].Group)
		suite.Equal(0, *decls[0].Group)
		suite.NotNil(decls[0].Binding)
		suite.Equal(0, *decls[0].Binding)
	})
}

func (suite *annotationsTest) TestGroupAnnotationErrors() {
	suite.Run("group with too few arguments returns error", func() {
		source := "//@oxy:group 0 0 storage_uniform camera\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "requires exactly four arguments")
	})

	suite.Run("group with non-integer group number returns error", func() {
		source := "//@oxy:group abc 0 storage_uniform cam camera\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "invalid group number")
	})

	suite.Run("group with non-integer binding number returns error", func() {
		source := "//@oxy:group 0 xyz storage_uniform cam camera\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "invalid binding number")
	})

	suite.Run("group with unknown address space returns error", func() {
		source := "//@oxy:group 0 0 storage_banana cam camera\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "unknown address space")
	})

	suite.Run("group with unknown struct type returns error", func() {
		source := "//@oxy:group 0 0 storage_uniform cam unknown_struct\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "unknown struct type")
	})

	suite.Run("group with unknown array element type returns error", func() {
		source := "//@oxy:group 0 0 storage_read buf array<unknown_element>\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "unknown array element type")
	})
}

func (suite *annotationsTest) TestProviderAnnotation() {
	suite.Run("provider annotation with identity only", func() {
		source := "//@oxy:provider 2 0 material\n"
		_, err := suite.pp.Process(source)
		suite.NoError(err)
		decls := suite.pp.Declarations()
		suite.Len(decls, 1)
		suite.Equal(shader.AnnotationTypeProvider, decls[0].Type)
		suite.NotNil(decls[0].Group)
		suite.Equal(2, *decls[0].Group)
		suite.NotNil(decls[0].Binding)
		suite.Equal(0, *decls[0].Binding)
		suite.Len(decls[0].Args, 1)
		suite.Equal(shader.AnnotationArgMaterial, decls[0].Args[0])
	})

	suite.Run("provider annotation with binding role", func() {
		source := "//@oxy:provider 2 0 material diffuse_texture\n"
		_, err := suite.pp.Process(source)
		suite.NoError(err)
		decls := suite.pp.Declarations()
		suite.Len(decls, 1)
		suite.Len(decls[0].Args, 2)
		suite.Equal(shader.AnnotationArgMaterial, decls[0].Args[0])
		suite.Equal(shader.AnnotationArgDiffuseTexture, decls[0].Args[1])
	})

	suite.Run("provider annotation does not produce WGSL output", func() {
		source := "line before\n//@oxy:provider 0 0 material\nline after\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "line before")
		suite.Contains(out, "line after")
		suite.NotContains(out, "@oxy:provider")
	})

	suite.Run("all provider identities are accepted", func() {
		identities := []shader.AnnotationArg{
			shader.AnnotationArgCamera,
			shader.AnnotationArgMaterial,
			shader.AnnotationArgLights,
			shader.AnnotationArgShadow,
			shader.AnnotationArgTiles,
			shader.AnnotationArgEffect,
			shader.AnnotationArgAnimator,
			shader.AnnotationArgAnimatorOutput,
			shader.AnnotationArgAnimatorPacked,
			shader.AnnotationArgAnimatorScratch,
		}
		for _, id := range identities {
			source := "//@oxy:provider 0 0 " + string(id) + "\n"
			_, err := suite.pp.Process(source)
			suite.NoError(err)
		}
	})

	suite.Run("all binding roles are accepted", func() {
		roles := []shader.AnnotationArg{
			shader.AnnotationArgDiffuseTexture,
			shader.AnnotationArgDiffuseSampler,
			shader.AnnotationArgNormalTexture,
			shader.AnnotationArgNormalSampler,
			shader.AnnotationArgMetallicRoughnessTexture,
			shader.AnnotationArgMetallicRoughnessSampler,
		}
		for _, role := range roles {
			source := "//@oxy:provider 0 0 material " + string(role) + "\n"
			_, err := suite.pp.Process(source)
			suite.NoError(err)
		}
	})
}

func (suite *annotationsTest) TestProviderAnnotationErrors() {
	suite.Run("provider with too few arguments returns error", func() {
		source := "//@oxy:provider 0 0\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "requires three or four arguments")
	})

	suite.Run("provider with too many arguments returns error", func() {
		source := "//@oxy:provider 0 0 material diffuse_texture extra\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "requires three or four arguments")
	})

	suite.Run("provider with non-integer group returns error", func() {
		source := "//@oxy:provider abc 0 material\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "invalid group number")
	})

	suite.Run("provider with non-integer binding returns error", func() {
		source := "//@oxy:provider 0 abc material\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "invalid binding number")
	})

	suite.Run("provider with unknown identity returns error", func() {
		source := "//@oxy:provider 0 0 unknown_provider\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "unknown provider identity")
	})

	suite.Run("provider with unknown binding role returns error", func() {
		source := "//@oxy:provider 0 0 material unknown_role\n"
		_, err := suite.pp.Process(source)
		suite.Error(err)
		suite.Contains(err.Error(), "unknown binding role")
	})
}

func (suite *annotationsTest) TestNonAnnotationLinesPassThrough() {
	suite.Run("regular WGSL code passes through unchanged", func() {
		source := "struct Foo {\n    bar: f32,\n};\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Equal(source, out)
	})

	suite.Run("regular comment without @oxy prefix passes through", func() {
		source := "// This is a regular comment\n"
		out, err := suite.pp.Process(source)
		suite.NoError(err)
		suite.Contains(out, "// This is a regular comment")
	})

	suite.Run("empty source produces empty output", func() {
		out, err := suite.pp.Process("")
		suite.NoError(err)
		suite.Equal("", out)
	})
}

func (suite *annotationsTest) TestDeclarationsResetOnEachProcess() {
	suite.Run("declarations are reset between process calls", func() {
		source1 := "//@oxy:provider 0 0 material\n"
		_, err := suite.pp.Process(source1)
		suite.NoError(err)
		suite.Len(suite.pp.Declarations(), 1)

		source2 := "// no annotations\n"
		_, err = suite.pp.Process(source2)
		suite.NoError(err)
		suite.Empty(suite.pp.Declarations())
	})
}

func (suite *annotationsTest) TestMultipleAnnotationsInSingleSource() {
	suite.Run("multiple group and provider annotations in order", func() {
		source := "//@oxy:group 0 0 storage_uniform cam camera\n" +
			"//@oxy:group 1 0 storage_read buf instance_data\n" +
			"//@oxy:provider 2 0 material diffuse_texture\n"
		_, err := suite.pp.Process(source)
		suite.NoError(err)
		decls := suite.pp.Declarations()
		suite.Len(decls, 3)
		suite.Equal(shader.AnnotationTypeBindingGroup, decls[0].Type)
		suite.Equal(shader.AnnotationTypeBindingGroup, decls[1].Type)
		suite.Equal(shader.AnnotationTypeProvider, decls[2].Type)
	})
}
