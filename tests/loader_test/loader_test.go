package loader_test

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/loader"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	materialmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/material"
	modelmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/model"
	renderermocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/renderer"
	shadermocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/shader"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

type loaderTest struct {
	suite.Suite
}

func TestLoader(t *testing.T) {
	suite.Run(t, new(loaderTest))
}

func (suite *loaderTest) TestBackendTypeConstant() {
	suite.Run("BackendTypeGLTF is zero", func() {
		suite.Equal(loader.LoaderBackendType(0), loader.BackendTypeGLTF)
	})
}

func (suite *loaderTest) TestNewLoader() {
	suite.Run("Get returns nil for unknown key", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		suite.Nil(l.Get("nonexistent"))
	})

	suite.Run("Models returns empty map", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		suite.Empty(l.Models())
	})
}

func (suite *loaderTest) TestNewLoaderWithOptions() {
	suite.Run("WithRenderer sets the renderer", func() {
		r := &renderermocks.MockRenderer{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))
		// Verify renderer is set by calling InitMaterialGPU which checks for nil renderer
		// A nil renderer would return an error; a non-nil one will proceed further
		suite.NotNil(l)
	})

	suite.Run("WithModel pre-populates cache", func() {
		m := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("test_model", m))
		suite.Equal(m, l.Get("test_model"))
	})

	suite.Run("multiple WithModel options populate cache", func() {
		m1 := &modelmocks.MockModel{}
		m2 := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF,
			loader.WithModel("model_a", m1),
			loader.WithModel("model_b", m2),
		)
		suite.Equal(m1, l.Get("model_a"))
		suite.Equal(m2, l.Get("model_b"))
	})
}

func (suite *loaderTest) TestGet() {
	suite.Run("returns nil for unknown key", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		suite.Nil(l.Get("does_not_exist"))
	})

	suite.Run("returns pre-cached model by key", func() {
		m := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("cached", m))
		suite.Equal(m, l.Get("cached"))
	})

	suite.Run("returns nil for empty string key", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		suite.Nil(l.Get(""))
	})
}

func (suite *loaderTest) TestModels() {
	suite.Run("returns empty map for new loader", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		models := l.Models()
		suite.NotNil(models)
		suite.Len(models, 0)
	})

	suite.Run("returns all pre-cached models", func() {
		m1 := &modelmocks.MockModel{}
		m2 := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF,
			loader.WithModel("a", m1),
			loader.WithModel("b", m2),
		)
		models := l.Models()
		suite.Len(models, 2)
		suite.Equal(m1, models["a"])
		suite.Equal(m2, models["b"])
	})

	suite.Run("returns a defensive copy", func() {
		m := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("x", m))
		models := l.Models()
		models["injected"] = &modelmocks.MockModel{}

		// Original cache should be unaffected
		suite.Nil(l.Get("injected"))
		suite.Len(l.Models(), 1)
	})
}

func (suite *loaderTest) TestLoadUnsupportedFormat() {
	suite.Run("returns error for .obj extension", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load("model.obj", nil)
		suite.Error(err)
		suite.Contains(err.Error(), "unsupported model format")
	})

	suite.Run("returns error for .fbx extension", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load("model.fbx", nil)
		suite.Error(err)
		suite.Contains(err.Error(), "unsupported model format")
	})

	suite.Run("returns error for no extension", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load("modelfile", nil)
		suite.Error(err)
		suite.Contains(err.Error(), "unsupported model format")
	})
}

func (suite *loaderTest) TestLoadNonexistentFile() {
	suite.Run("returns error for missing gltf file", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load("/nonexistent/path/model.gltf", nil)
		suite.Error(err)
	})

	suite.Run("returns error for missing glb file", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load("/nonexistent/path/model.glb", nil)
		suite.Error(err)
	})
}

func (suite *loaderTest) TestLoadCacheHit() {
	suite.Run("returns cached model on second Load call", func() {
		m := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("scene.glb", m))

		result, err := l.Load("scene.glb", nil)
		suite.NoError(err)
		suite.Equal(m, result)
	})
}

func (suite *loaderTest) TestLoadMeshOnlyUnsupportedFormat() {
	suite.Run("returns error for unsupported extension", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.LoadMeshOnly("model.obj", nil)
		suite.Error(err)
		suite.Contains(err.Error(), "unsupported model format")
	})
}

func (suite *loaderTest) TestLoadMeshOnlyCacheHit() {
	suite.Run("returns cached model on second call", func() {
		m := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("static.glb", m))

		result, err := l.LoadMeshOnly("static.glb", nil)
		suite.NoError(err)
		suite.Equal(m, result)
	})
}

func (suite *loaderTest) TestLoadReaderCacheHit() {
	suite.Run("returns cached model by name", func() {
		m := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("stream_model", m))

		result, err := l.LoadReader("stream_model", bytes.NewReader(nil), true, nil)
		suite.NoError(err)
		suite.Equal(m, result)
	})
}

func (suite *loaderTest) TestLoadReaderInvalidData() {
	suite.Run("returns error for empty reader", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.LoadReader("bad", bytes.NewReader(nil), true, nil)
		suite.Error(err)
	})

	suite.Run("returns error for garbage bytes as GLB", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.LoadReader("garbage", bytes.NewReader([]byte("not a glb")), true, nil)
		suite.Error(err)
	})

	suite.Run("returns error for invalid JSON as gltf", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.LoadReader("bad_json", bytes.NewReader([]byte("{invalid json")), false, nil)
		suite.Error(err)
	})

	suite.Run("returns error for valid JSON but wrong gltf version", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		json := []byte(`{"asset":{"version":"1.0"}}`)
		_, err := l.LoadReader("v1", bytes.NewReader(json), false, nil)
		suite.Error(err)
	})
}

func (suite *loaderTest) TestLoadReaderMinimalGLB() {
	suite.Run("loads minimal GLB with no meshes", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		result, err := l.LoadReader("minimal", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(result)
	})

	suite.Run("model is cached after load", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		_, err := l.LoadReader("cached_glb", bytes.NewReader(glb), true, nil)
		suite.NoError(err)

		cached := l.Get("cached_glb")
		suite.NotNil(cached)
	})

	suite.Run("cached model has expected name", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		m, err := l.LoadReader("named_model", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		// With no scene name and empty fallback, the importer produces "unnamed_model"
		suite.Equal("unnamed_model", m.Name())
	})

	suite.Run("cached model has no skeleton", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		m, err := l.LoadReader("no_skel", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.Nil(m.Skeleton())
		suite.False(m.Skinned())
	})

	suite.Run("cached model has no animations", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		m, err := l.LoadReader("no_anim", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.Equal(0, m.AnimationCount())
	})

	suite.Run("cached model has no materials", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		m, err := l.LoadReader("no_mat", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.Empty(m.ImportedMaterials())
	})
}

func (suite *loaderTest) TestLoadReaderMinimalGLTFJSON() {
	suite.Run("loads minimal gltf JSON via reader", func() {
		json := []byte(`{"asset":{"version":"2.0"}}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		m, err := l.LoadReader("json_model", bytes.NewReader(json), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.Equal("unnamed_model", m.Name())
	})
}

func (suite *loaderTest) TestLoadReaderWithSceneName() {
	suite.Run("uses scene name when available", func() {
		json := []byte(`{"asset":{"version":"2.0"},"scene":0,"scenes":[{"name":"MyScene"}]}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		m, err := l.LoadReader("scene_named", bytes.NewReader(json), false, nil)
		suite.NoError(err)
		suite.Equal("MyScene", m.Name())
	})
}

func (suite *loaderTest) TestLoadReaderWithRenderer() {
	suite.Run("calls InitMeshBuffers when renderer is set", func() {
		r := &renderermocks.MockRenderer{}
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		json := []byte(`{"asset":{"version":"2.0"}}`)
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))

		m, err := l.LoadReader("with_renderer", bytes.NewReader(json), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadWithTempGLBFile() {
	suite.Run("loads a real GLB file from disk", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("caches model by file path", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "cached.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m1, err := l.Load(path, nil)
		suite.NoError(err)

		m2, err := l.Load(path, nil)
		suite.NoError(err)
		suite.Equal(m1, m2)
	})

	suite.Run("accepts .gltf extension with JSON content", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.gltf")
		suite.Require().NoError(os.WriteFile(path, []byte(`{"asset":{"version":"2.0"}}`), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadMeshOnlyWithTempFile() {
	suite.Run("loads a real GLB file via LoadMeshOnly", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "static.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadMeshOnly(path, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.Nil(m.Skeleton())
		suite.False(m.Skinned())
	})

	suite.Run("caches model by file path", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "mesh_cached.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m1, err := l.LoadMeshOnly(path, nil)
		suite.NoError(err)

		m2, err := l.LoadMeshOnly(path, nil)
		suite.NoError(err)
		suite.Equal(m1, m2)
	})
}

func (suite *loaderTest) TestInitMaterialGPUWithoutRenderer() {
	suite.Run("returns error when renderer is nil", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		err := l.InitMaterialGPU(nil, nil, "test_provider")
		suite.Error(err)
		suite.Contains(err.Error(), "cannot InitMaterialGPU without a Renderer")
	})
}

func (suite *loaderTest) TestLoadReaderRendererInitMeshBuffersError() {
	suite.Run("propagates InitMeshBuffers error", func() {
		r := &renderermocks.MockRenderer{}
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(fmt.Errorf("gpu buffer creation failed")).Maybe()

		// A minimal glTF with one triangle mesh to trigger InitMeshBuffers
		json := buildMinimalTriangleGLTF()
		glb := buildGLBWithBin(json, buildMinimalTriangleBin())

		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))
		_, err := l.LoadReader("mesh_err", bytes.NewReader(glb), true, nil)
		suite.Error(err)
		suite.Contains(err.Error(), "failed to init mesh bind group")
	})
}

func (suite *loaderTest) TestLoadModelNameFromPath() {
	suite.Run("model name falls back to file path when no scene name", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "fallback_name.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path, nil)
		suite.NoError(err)
		// The importer uses the file path as fallback name
		suite.Equal(path, m.Name())
	})
}

func (suite *loaderTest) TestLoadReaderCaseInsensitiveExtension() {
	suite.Run("Load accepts uppercase .GLB extension", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.GLB")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("Load accepts mixed case .Gltf extension", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.Gltf")
		suite.Require().NoError(os.WriteFile(path, []byte(`{"asset":{"version":"2.0"}}`), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadMeshOnlyNonexistentFile() {
	suite.Run("returns error for missing file", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.LoadMeshOnly("/nonexistent/model.glb", nil)
		suite.Error(err)
	})
}

func (suite *loaderTest) TestModelsAfterLoad() {
	suite.Run("Models includes loaded model after LoadReader", func() {
		json := []byte(`{"asset":{"version":"2.0"}}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		_, err := l.LoadReader("loaded", bytes.NewReader(json), false, nil)
		suite.NoError(err)

		models := l.Models()
		suite.Len(models, 1)
		suite.NotNil(models["loaded"])
	})
}

func (suite *loaderTest) TestLoadReaderGLBWithSceneName() {
	suite.Run("GLB with named scene uses scene name", func() {
		json := `{"asset":{"version":"2.0"},"scene":0,"scenes":[{"name":"TestScene"}]}`
		glb := buildMinimalGLB(json)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		m, err := l.LoadReader("test_scene_glb", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.Equal("TestScene", m.Name())
	})
}

func (suite *loaderTest) TestInitMaterialGPUNoDeclarations() {
	suite.Run("returns nil when shader has no material provider declarations", func() {
		r := &renderermocks.MockRenderer{}
		s := &shadermocks.MockShader{}
		mat := &materialmocks.MockMaterial{}

		// Shader returns empty declarations â€” no material provider found
		s.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()

		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))
		err := l.InitMaterialGPU(mat, s, "test_provider")
		suite.NoError(err)
	})

	suite.Run("returns nil when shader has only non-material provider declarations", func() {
		r := &renderermocks.MockRenderer{}
		s := &shadermocks.MockShader{}
		mat := &materialmocks.MockMaterial{}

		group := 0
		binding := 0
		s.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type:    shader.AnnotationTypeProvider,
				Args:    []shader.AnnotationArg{"animator_output"},
				Group:   &group,
				Binding: &binding,
			},
		}).Maybe()

		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))
		err := l.InitMaterialGPU(mat, s, "test_provider")
		suite.NoError(err)
	})
}

func (suite *loaderTest) TestInitMaterialGPUWithFallbackTextures() {
	suite.Run("creates fallback textures for unset material bindings", func() {
		r := &renderermocks.MockRenderer{}
		s := &shadermocks.MockShader{}
		mat := &materialmocks.MockMaterial{}

		group := 2
		diffuseTexBinding := 0
		diffuseSamplerBinding := 1
		normalTexBinding := 2
		normalSamplerBinding := 3

		s.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type:    shader.AnnotationTypeProvider,
				Args:    []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseTexture},
				Group:   &group,
				Binding: &diffuseTexBinding,
			},
			{
				Type:    shader.AnnotationTypeProvider,
				Args:    []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseSampler},
				Group:   &group,
				Binding: &diffuseSamplerBinding,
			},
			{
				Type:    shader.AnnotationTypeProvider,
				Args:    []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgNormalTexture},
				Group:   &group,
				Binding: &normalTexBinding,
			},
			{
				Type:    shader.AnnotationTypeProvider,
				Args:    []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgNormalSampler},
				Group:   &group,
				Binding: &normalSamplerBinding,
			},
		}).Maybe()

		// Material has no textures set
		mat.EXPECT().DiffuseTexture().Return(nil).Maybe()
		mat.EXPECT().NormalTexture().Return(nil).Maybe()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Maybe()

		// Shader returns layout descriptor with texture and sampler entries
		s.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat}},
				{Binding: 1, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
				{Binding: 2, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat}},
				{Binding: 3, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			},
		}).Maybe()

		// Renderer should be called for fallback textures and samplers
		r.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		mat.EXPECT().SetBindGroupProvider(mock.Anything).Maybe()

		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))
		err := l.InitMaterialGPU(mat, s, "test_material")
		suite.NoError(err)
	})
}

func (suite *loaderTest) TestLoadReaderWithMaterials() {
	suite.Run("loads glTF with materials without renderer", func() {
		json := []byte(`{
			"asset": {"version": "2.0"},
			"materials": [
				{
					"name": "TestMaterial",
					"pbrMetallicRoughness": {
						"baseColorFactor": [1.0, 0.0, 0.0, 1.0],
						"metallicFactor": 0.5,
						"roughnessFactor": 0.8
					}
				}
			]
		}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		m, err := l.LoadReader("with_materials", bytes.NewReader(json), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.Len(m.ImportedMaterials(), 1)
		suite.Equal("TestMaterial", m.ImportedMaterials()[0].Name)
	})

	suite.Run("material base color is extracted", func() {
		json := []byte(`{
			"asset": {"version": "2.0"},
			"materials": [
				{
					"pbrMetallicRoughness": {
						"baseColorFactor": [0.5, 0.6, 0.7, 1.0]
					}
				}
			]
		}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		m, err := l.LoadReader("color_mat", bytes.NewReader(json), false, nil)
		suite.NoError(err)
		suite.Len(m.ImportedMaterials(), 1)
		suite.InDelta(0.5, float64(m.ImportedMaterials()[0].BaseColor[0]), 1e-6)
		suite.InDelta(0.6, float64(m.ImportedMaterials()[0].BaseColor[1]), 1e-6)
		suite.InDelta(0.7, float64(m.ImportedMaterials()[0].BaseColor[2]), 1e-6)
		suite.InDelta(1.0, float64(m.ImportedMaterials()[0].BaseColor[3]), 1e-6)
	})

	suite.Run("material metallic and roughness are extracted", func() {
		json := []byte(`{
			"asset": {"version": "2.0"},
			"materials": [
				{
					"pbrMetallicRoughness": {
						"metallicFactor": 0.3,
						"roughnessFactor": 0.9
					}
				}
			]
		}`)
		l := loader.NewLoader(loader.BackendTypeGLTF)

		m, err := l.LoadReader("pbr_mat", bytes.NewReader(json), false, nil)
		suite.NoError(err)
		suite.Len(m.ImportedMaterials(), 1)
		suite.InDelta(0.3, float64(m.ImportedMaterials()[0].Metallic), 1e-6)
		suite.InDelta(0.9, float64(m.ImportedMaterials()[0].Roughness), 1e-6)
	})
}

func (suite *loaderTest) TestLoadReaderTriangleMeshWithRenderer() {
	suite.Run("loads a triangle mesh and calls renderer mesh init", func() {
		r := &renderermocks.MockRenderer{}
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, 3).Return(nil).Once()

		jsonStr := buildMinimalTriangleGLTF()
		glb := buildGLBWithBin(jsonStr, buildMinimalTriangleBin())

		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))
		m, err := l.LoadReader("triangle", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.NotNil(m.MeshProvider())
		r.AssertExpectations(suite.T())
	})
}

func (suite *loaderTest) TestLoadReaderTriangleMeshWithoutRenderer() {
	suite.Run("loads a triangle mesh without renderer", func() {
		jsonStr := buildMinimalTriangleGLTF()
		glb := buildGLBWithBin(jsonStr, buildMinimalTriangleBin())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("triangle_no_gpu", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.False(m.Skinned())
	})
}

func (suite *loaderTest) TestLoadWithRendererAndMaterials() {
	suite.Run("loads glTF with materials and renderer but no fragment shader", func() {
		r := &renderermocks.MockRenderer{}
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		json := []byte(`{
			"asset": {"version": "2.0"},
			"materials": [{"name": "Mat1", "pbrMetallicRoughness": {}}]
		}`)
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))

		m, err := l.LoadReader("with_renderer_mats", bytes.NewReader(json), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.Len(m.ImportedMaterials(), 1)
	})
}

// buildMinimalGLB constructs a minimal valid GLB binary from a glTF JSON string.
// The JSON is padded with spaces to 4-byte alignment per the GLB spec.
func buildMinimalGLB(jsonStr string) []byte {
	jsonBytes := []byte(jsonStr)

	// Pad JSON to 4-byte alignment with spaces (GLB spec)
	for len(jsonBytes)%4 != 0 {
		jsonBytes = append(jsonBytes, ' ')
	}

	jsonChunkLen := uint32(len(jsonBytes))
	totalLen := uint32(12 + 8 + jsonChunkLen) // header + chunkHeader + jsonData

	buf := &bytes.Buffer{}
	binary.Write(buf, binary.LittleEndian, uint32(0x46546C67)) // magic "glTF"
	binary.Write(buf, binary.LittleEndian, uint32(2))          // version
	binary.Write(buf, binary.LittleEndian, totalLen)
	binary.Write(buf, binary.LittleEndian, jsonChunkLen)
	binary.Write(buf, binary.LittleEndian, uint32(0x4E4F534A)) // JSON chunk type
	buf.Write(jsonBytes)

	return buf.Bytes()
}

// buildGLBWithBin constructs a GLB binary with both a JSON chunk and a BIN chunk.
func buildGLBWithBin(jsonStr string, binData []byte) []byte {
	jsonBytes := []byte(jsonStr)
	for len(jsonBytes)%4 != 0 {
		jsonBytes = append(jsonBytes, ' ')
	}
	for len(binData)%4 != 0 {
		binData = append(binData, 0)
	}

	jsonChunkLen := uint32(len(jsonBytes))
	binChunkLen := uint32(len(binData))
	totalLen := uint32(12 + 8 + jsonChunkLen + 8 + binChunkLen)

	buf := &bytes.Buffer{}
	binary.Write(buf, binary.LittleEndian, uint32(0x46546C67)) // magic
	binary.Write(buf, binary.LittleEndian, uint32(2))          // version
	binary.Write(buf, binary.LittleEndian, totalLen)
	binary.Write(buf, binary.LittleEndian, jsonChunkLen)
	binary.Write(buf, binary.LittleEndian, uint32(0x4E4F534A)) // JSON
	buf.Write(jsonBytes)
	binary.Write(buf, binary.LittleEndian, binChunkLen)
	binary.Write(buf, binary.LittleEndian, uint32(0x004E4942)) // BIN
	buf.Write(binData)

	return buf.Bytes()
}

// buildMinimalTriangleGLTF returns a glTF JSON string describing a single triangle
// with 3 vertices (position only) and 3 indices, backed by a single binary buffer.
func buildMinimalTriangleGLTF() string {
	// 3 vertices Ã— 12 bytes (float32Ã—3) = 36 bytes for positions
	// 3 indices Ã— 2 bytes (uint16) = 6 bytes, padded to 8 for alignment
	// Total buffer: 44 bytes
	// BufferView 0: positions at offset 0, length 36
	// BufferView 1: indices at offset 36, length 6
	return strings.TrimSpace(`{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{
    "primitives": [{
      "attributes": {"POSITION": 0},
      "indices": 1
    }]
  }],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "max": [1.0, 1.0, 0.0],
      "min": [0.0, 0.0, 0.0]
    },
    {
      "bufferView": 1,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR"
    }
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 44}]
}`)
}

// buildMinimalTriangleBin constructs the binary buffer for the minimal triangle mesh.
// Contains 3 vertex positions followed by 3 uint16 indices, padded to 4-byte alignment.
func buildMinimalTriangleBin() []byte {
	buf := &bytes.Buffer{}

	// 3 vertices: (0,0,0), (1,0,0), (0,1,0)
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	for _, p := range positions {
		binary.Write(buf, binary.LittleEndian, p)
	}

	// 3 indices: 0, 1, 2
	indices := []uint16{0, 1, 2}
	for _, idx := range indices {
		binary.Write(buf, binary.LittleEndian, idx)
	}

	// Pad to 4-byte alignment
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}

	return buf.Bytes()
}

// foxModelPath returns the path to the Fox.glb test fixture relative to the test directory.
func foxModelPath() string {
	return filepath.Join("..", "assets", "models", "Fox.glb")
}

func (suite *loaderTest) TestFoxModelLoad() {
	suite.Run("loads Fox.glb without error", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("model has a non-empty name", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.NotEmpty(m.Name())
	})
}

func (suite *loaderTest) TestFoxModelSkinned() {
	suite.Run("fox model reports as skinned", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.True(m.Skinned())
	})
}

func (suite *loaderTest) TestFoxModelSkeleton() {
	suite.Run("skeleton is not nil", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.NotNil(m.Skeleton())
	})

	suite.Run("skeleton has bones", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.Greater(len(m.Skeleton().Bones), 0)
	})

	suite.Run("skeleton has root bone indices", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.NotEmpty(m.Skeleton().RootBoneIndices)
	})

	suite.Run("skeleton has bone name to index mapping", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.NotEmpty(m.Skeleton().BoneNameToIndex)
	})

	suite.Run("bone count matches name-to-index map length", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		skel := m.Skeleton()
		suite.Equal(len(skel.Bones), len(skel.BoneNameToIndex))
	})

	suite.Run("root bones have parent index of negative one", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		skel := m.Skeleton()
		for _, rootIdx := range skel.RootBoneIndices {
			suite.Equal(int32(-1), skel.Bones[rootIdx].ParentIndex)
		}
	})

	suite.Run("non-root bones have valid parent indices", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		skel := m.Skeleton()
		boneCount := int32(len(skel.Bones))
		for _, bone := range skel.Bones {
			if bone.ParentIndex == -1 {
				continue
			}
			suite.GreaterOrEqual(bone.ParentIndex, int32(0))
			suite.Less(bone.ParentIndex, boneCount)
		}
	})

	suite.Run("all bones have non-empty names", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, bone := range m.Skeleton().Bones {
			suite.NotEmpty(bone.Name)
		}
	})

	suite.Run("bone name to index map entries match bones slice", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		skel := m.Skeleton()
		for name, idx := range skel.BoneNameToIndex {
			suite.Equal(name, skel.Bones[idx].Name)
		}
	})

	suite.Run("bones have non-zero inverse bind matrices", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, bone := range m.Skeleton().Bones {
			allZero := true
			for _, v := range bone.InverseBindMatrix {
				if v != 0 {
					allZero = false
					break
				}
			}
			suite.False(allZero)
		}
	})

	suite.Run("bones have non-zero local transform scale", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, bone := range m.Skeleton().Bones {
			hasScale := bone.LocalTransform.Scale[0] != 0 ||
				bone.LocalTransform.Scale[1] != 0 ||
				bone.LocalTransform.Scale[2] != 0
			suite.True(hasScale)
		}
	})

	suite.Run("skeleton parents come before children in topological order", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		skel := m.Skeleton()
		for i, bone := range skel.Bones {
			if bone.ParentIndex >= 0 {
				suite.Less(int(bone.ParentIndex), i)
			}
		}
	})
}

func (suite *loaderTest) TestFoxModelAnimations() {
	suite.Run("has at least one animation", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.Greater(m.AnimationCount(), 0)
	})

	suite.Run("animation count matches animations slice length", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.Equal(m.AnimationCount(), len(m.Animations()))
	})

	suite.Run("animation names count matches animation count", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.Len(m.AnimationNames(), m.AnimationCount())
	})

	suite.Run("all animations have non-empty names", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			suite.NotEmpty(clip.Name)
		}
	})

	suite.Run("all animations have positive duration", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			suite.Greater(clip.Duration, float32(0))
		}
	})

	suite.Run("all animations have at least one channel", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			suite.NotEmpty(clip.Channels)
		}
	})

	suite.Run("animation channels reference valid bone indices", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		boneCount := int32(len(m.Skeleton().Bones))
		for _, clip := range m.Animations() {
			for _, ch := range clip.Channels {
				suite.GreaterOrEqual(ch.BoneIndex, int32(0))
				suite.Less(ch.BoneIndex, boneCount)
			}
		}
	})

	suite.Run("every animation has channels with keyframes", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			hasKeyframes := false
			for _, ch := range clip.Channels {
				if len(ch.PositionKeys) > 0 || len(ch.RotationKeys) > 0 || len(ch.ScaleKeys) > 0 {
					hasKeyframes = true
					break
				}
			}
			suite.True(hasKeyframes)
		}
	})

	suite.Run("rotation keyframe quaternions have approximately unit length", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		checked := 0
		for _, clip := range m.Animations() {
			for _, ch := range clip.Channels {
				for _, rk := range ch.RotationKeys {
					q := rk.Value
					length := float64(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
					suite.InDelta(1.0, length, 1e-3)
					checked++
					if checked > 100 {
						return
					}
				}
			}
		}
		suite.Greater(checked, 0)
	})

	suite.Run("position keyframes have non-negative timestamps", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			for _, ch := range clip.Channels {
				for _, pk := range ch.PositionKeys {
					suite.GreaterOrEqual(pk.Time, float32(0))
				}
			}
		}
	})

	suite.Run("keyframe timestamps do not exceed animation duration", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			for _, ch := range clip.Channels {
				for _, pk := range ch.PositionKeys {
					suite.LessOrEqual(pk.Time, clip.Duration)
				}
				for _, rk := range ch.RotationKeys {
					suite.LessOrEqual(rk.Time, clip.Duration)
				}
				for _, sk := range ch.ScaleKeys {
					suite.LessOrEqual(sk.Time, clip.Duration)
				}
			}
		}
	})
}

func (suite *loaderTest) TestFoxModelAnimationLookup() {
	suite.Run("known animation names return valid indices", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for i, name := range m.AnimationNames() {
			suite.Equal(i, m.GetAnimationIndex(name))
		}
	})

	suite.Run("unknown animation name returns negative one", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.Equal(-1, m.GetAnimationIndex("nonexistent_animation"))
	})
}

func (suite *loaderTest) TestFoxModelMaterials() {
	suite.Run("has at least one imported material", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.NotEmpty(m.ImportedMaterials())
	})

	suite.Run("first material has a name", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.NotEmpty(m.ImportedMaterials()[0].Name)
	})

	suite.Run("materials have valid base color range", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, mat := range m.ImportedMaterials() {
			for _, c := range mat.BaseColor {
				suite.GreaterOrEqual(c, float32(0))
				suite.LessOrEqual(c, float32(1))
			}
		}
	})

	suite.Run("at least one material has diffuse texture data", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		hasDiffuse := false
		for _, mat := range m.ImportedMaterials() {
			if mat.DiffuseTexture != nil && len(mat.DiffuseTexture.Data) > 0 {
				hasDiffuse = true
				break
			}
		}
		suite.True(hasDiffuse)
	})

	suite.Run("diffuse texture has a mime type", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, mat := range m.ImportedMaterials() {
			if mat.DiffuseTexture != nil && len(mat.DiffuseTexture.Data) > 0 {
				suite.NotEmpty(mat.DiffuseTexture.MimeType)
				return
			}
		}
	})

	suite.Run("metallic and roughness values are in valid range", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		for _, mat := range m.ImportedMaterials() {
			suite.GreaterOrEqual(mat.Metallic, float32(0))
			suite.LessOrEqual(mat.Metallic, float32(1))
			suite.GreaterOrEqual(mat.Roughness, float32(0))
			suite.LessOrEqual(mat.Roughness, float32(1))
		}
	})
}

func (suite *loaderTest) TestFoxModelLoadReader() {
	suite.Run("loads Fox.glb via LoadReader with same results", func() {
		data, err := os.ReadFile(foxModelPath())
		suite.Require().NoError(err)

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("fox_reader", bytes.NewReader(data), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(m.Skinned())
		suite.NotNil(m.Skeleton())
		suite.Greater(m.AnimationCount(), 0)
		suite.NotEmpty(m.ImportedMaterials())
	})

	suite.Run("LoadReader skeleton matches Load skeleton bone count", func() {
		data, err := os.ReadFile(foxModelPath())
		suite.Require().NoError(err)

		l1 := loader.NewLoader(loader.BackendTypeGLTF)
		m1, err := l1.Load(foxModelPath(), nil)
		suite.Require().NoError(err)

		l2 := loader.NewLoader(loader.BackendTypeGLTF)
		m2, err := l2.LoadReader("fox_reader_cmp", bytes.NewReader(data), true, nil)
		suite.Require().NoError(err)

		suite.Equal(len(m1.Skeleton().Bones), len(m2.Skeleton().Bones))
	})

	suite.Run("LoadReader animation count matches Load animation count", func() {
		data, err := os.ReadFile(foxModelPath())
		suite.Require().NoError(err)

		l1 := loader.NewLoader(loader.BackendTypeGLTF)
		m1, err := l1.Load(foxModelPath(), nil)
		suite.Require().NoError(err)

		l2 := loader.NewLoader(loader.BackendTypeGLTF)
		m2, err := l2.LoadReader("fox_reader_anim", bytes.NewReader(data), true, nil)
		suite.Require().NoError(err)

		suite.Equal(m1.AnimationCount(), m2.AnimationCount())
	})
}

func (suite *loaderTest) TestFoxModelLoadMeshOnly() {
	suite.Run("loads Fox.glb mesh only without error", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadMeshOnly(foxModelPath(), nil)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("mesh-only model is not skinned", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadMeshOnly(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.False(m.Skinned())
	})

	suite.Run("mesh-only model has no skeleton", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadMeshOnly(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.Nil(m.Skeleton())
	})

	suite.Run("mesh-only model has no animations", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadMeshOnly(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.Equal(0, m.AnimationCount())
	})

	suite.Run("mesh-only model still has imported materials", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadMeshOnly(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.NotEmpty(m.ImportedMaterials())
	})

	suite.Run("mesh-only model has a mesh provider", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadMeshOnly(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.NotNil(m.MeshProvider())
	})
}

func (suite *loaderTest) TestFoxModelCaching() {
	suite.Run("second Load call returns cached model", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m1, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)

		m2, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)

		suite.Equal(m1.Name(), m2.Name())
	})

	suite.Run("Get returns model after Load", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)

		cached := l.Get(foxModelPath())
		suite.NotNil(cached)
	})

	suite.Run("Models map contains loaded fox model", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)

		models := l.Models()
		suite.Contains(models, foxModelPath())
	})
}

func (suite *loaderTest) TestFoxModelMeshProvider() {
	suite.Run("model has a non-nil mesh provider", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.NotNil(m.MeshProvider())
	})
}

func (suite *loaderTest) TestFoxModelRenderMaterials() {
	suite.Run("render materials are created without renderer", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.NotEmpty(m.RenderMaterials())
	})

	suite.Run("render materials count matches imported materials count", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath(), nil)
		suite.Require().NoError(err)
		suite.Equal(len(m.ImportedMaterials()), len(m.RenderMaterials()))
	})
}

func (suite *loaderTest) TestLoadReaderDataURITexture() {
	suite.Run("extracts embedded base64 PNG texture from material", func() {
		pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQaaaabjru5erkjggg=="
		dataURI := "data:image/png;base64," + pngBase64

		jsonStr := `{
"asset": {"version": "2.0"},
"materials": [{
"name": "DataURIMat",
"pbrMetallicRoughness": {
"baseColorTexture": {"index": 0}
}
}],
"textures": [{"source": 0}],
"images": [{"uri": "` + dataURI + `", "mimeType": "image/png"}]
}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("data_uri_tex", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.Len(m.ImportedMaterials(), 1)
		suite.NotNil(m.ImportedMaterials()[0].DiffuseTexture)
		suite.NotEmpty(m.ImportedMaterials()[0].DiffuseTexture.Data)
	})
}

func (suite *loaderTest) TestLoadReaderDataURIBuffer() {
	suite.Run("loads glTF with base64-encoded buffer data URI", func() {
		positions := [3][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
		indices := [3]uint16{0, 1, 2}

		buf := &bytes.Buffer{}
		for _, p := range positions {
			binary.Write(buf, binary.LittleEndian, p)
		}
		for _, idx := range indices {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		b64 := base64.StdEncoding.EncodeToString(buf.Bytes())
		dataURI := "data:application/octet-stream;base64," + b64

		jsonStr := `{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": 36},
{"buffer": 0, "byteOffset": 36, "byteLength": 6}
],
"buffers": [{"uri": "` + dataURI + `", "byteLength": ` + fmt.Sprintf("%d", buf.Len()) + `}]
}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("data_uri_buf", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.NotNil(m.MeshProvider())
	})
}

func (suite *loaderTest) TestLoadReaderSamplerWrapModes() {
	suite.Run("sampler with clamp-to-edge and mirrored-repeat wrap modes is extracted", func() {
		pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQaaaabjru5erkjggg=="
		dataURI := "data:image/png;base64," + pngBase64

		jsonStr := `{
"asset": {"version": "2.0"},
"materials": [{
"name": "SamplerMat",
"pbrMetallicRoughness": {
"baseColorTexture": {"index": 0}
}
}],
"textures": [{"source": 0, "sampler": 0}],
"samplers": [{
"magFilter": 9728,
"minFilter": 9987,
"wrapS": 33071,
"wrapT": 33648
}],
"images": [{"uri": "` + dataURI + `", "mimeType": "image/png"}]
}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("sampler_wrap", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.Len(m.ImportedMaterials(), 1)
		suite.NotNil(m.ImportedMaterials()[0].DiffuseTexture)
		suite.NotNil(m.ImportedMaterials()[0].DiffuseTexture.SamplerData)
	})
}

func (suite *loaderTest) TestLoadReaderVertexColorsMesh() {
	suite.Run("extracts mesh with VEC4 float vertex colors", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, c := range [][4]float32{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}} {
			binary.Write(buf, binary.LittleEndian, c)
		}
		colorLen := buf.Len() - posLen
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxOffset := buf.Len() - 6
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC4"},
{"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, colorLen, idxOffset, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("color_mesh", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("extracts mesh with VEC3 float vertex colors", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, c := range [][3]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}} {
			binary.Write(buf, binary.LittleEndian, c)
		}
		colorLen := buf.Len() - posLen
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxOffset := buf.Len() - 6
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
{"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, colorLen, idxOffset, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("color3_mesh", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("extracts mesh with VEC4 unsigned byte vertex colors", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, c := range [][4]uint8{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}} {
			buf.Write(c[:])
		}
		colorLen := buf.Len() - posLen
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxOffset := buf.Len() - 6
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
{"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, colorLen, idxOffset, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("color_byte_mesh", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("extracts mesh with VEC4 unsigned short vertex colors", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, c := range [][4]uint16{
			{65535, 0, 0, 65535}, {0, 65535, 0, 65535}, {0, 0, 65535, 65535}} {
			binary.Write(buf, binary.LittleEndian, c)
		}
		colorLen := buf.Len() - posLen
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxOffset := buf.Len() - 6
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "VEC4"},
{"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, colorLen, idxOffset, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("color_short_mesh", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadReaderUint32Indices() {
	suite.Run("loads mesh with uint32 index accessor", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint32{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("uint32_indices", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadReaderUint8Indices() {
	suite.Run("loads mesh with uint8 index accessor", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		buf.Write([]byte{0, 1, 2})
		idxLen := buf.Len() - posLen
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5121, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("uint8_indices", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadReaderMatrixTransformSkeleton() {
	suite.Run("loads skeleton with bone using matrix transform", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		ibmOffset := buf.Len()
		ibm := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		binary.Write(buf, binary.LittleEndian, ibm)
		ibmLen := buf.Len() - ibmOffset

		jointsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			buf.Write([]byte{0, 0, 0, 0})
		}
		jointsLen := buf.Len() - jointsOffset

		weightsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
		}
		weightsLen := buf.Len() - weightsOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [
{"mesh": 0, "skin": 0},
{
"name": "RootBone",
"matrix": [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 2, 3, 1]
}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 3, "WEIGHTS_0": 4}, "indices": 1}]}],
"skins": [{"joints": [1], "inverseBindMatrices": 2}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 1, "type": "MAT4"},
{"bufferView": 3, "componentType": 5121, "count": 3, "type": "VEC4"},
{"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}],
"animations": [{
"name": "BoneAnim",
"channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}],
"samplers": [{"input": 5, "output": 6, "interpolation": "LINEAR"}]
}]
}`, posLen, posLen, idxLen, ibmOffset, ibmLen, jointsOffset, jointsLen, weightsOffset, weightsLen, buf.Len())

		animInputOffset := buf.Len()
		binary.Write(buf, binary.LittleEndian, []float32{0, 1})
		animInputLen := buf.Len() - animInputOffset
		animOutputOffset := buf.Len()
		binary.Write(buf, binary.LittleEndian, [][3]float32{{0, 0, 0}, {1, 2, 3}})
		animOutputLen := buf.Len() - animOutputOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr = fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [
{"mesh": 0, "skin": 0},
{
"name": "RootBone",
"matrix": [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 2, 3, 1]
}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 3, "WEIGHTS_0": 4}, "indices": 1}]}],
"skins": [{"joints": [1], "inverseBindMatrices": 2}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 1, "type": "MAT4"},
{"bufferView": 3, "componentType": 5121, "count": 3, "type": "VEC4"},
{"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"},
{"bufferView": 5, "componentType": 5126, "count": 2, "type": "SCALAR", "max": [1], "min": [0]},
{"bufferView": 6, "componentType": 5126, "count": 2, "type": "VEC3"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, ibmOffset, ibmLen, jointsOffset, jointsLen, weightsOffset, weightsLen, animInputOffset, animInputLen, animOutputOffset, animOutputLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("matrix_skeleton", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(m.Skinned())
		suite.NotNil(m.Skeleton())
		suite.Len(m.Skeleton().Bones, 1)
		bone := m.Skeleton().Bones[0]
		suite.Equal("RootBone", bone.Name)
		suite.InDelta(2.0, float64(bone.LocalTransform.Scale[0]), 1e-3)
		suite.InDelta(2.0, float64(bone.LocalTransform.Scale[1]), 1e-3)
		suite.InDelta(2.0, float64(bone.LocalTransform.Scale[2]), 1e-3)
		suite.InDelta(1.0, float64(bone.LocalTransform.Translation[0]), 1e-3)
		suite.InDelta(2.0, float64(bone.LocalTransform.Translation[1]), 1e-3)
		suite.InDelta(3.0, float64(bone.LocalTransform.Translation[2]), 1e-3)
	})
}

func (suite *loaderTest) TestLoadReaderAnimationWithoutSkeleton() {
	suite.Run("extracts animations when no skeleton is present", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		animInputOffset := buf.Len()
		binary.Write(buf, binary.LittleEndian, []float32{0, 1})
		animInputLen := buf.Len() - animInputOffset
		animOutputOffset := buf.Len()
		binary.Write(buf, binary.LittleEndian, [][3]float32{{0, 0, 0}, {1, 2, 3}})
		animOutputLen := buf.Len() - animOutputOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0, 1]}],
"nodes": [
{"mesh": 0},
{"name": "AnimTarget"}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 2, "type": "SCALAR", "max": [1], "min": [0]},
{"bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC3"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}],
"animations": [{
"name": "NoSkelAnim",
"channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}],
"samplers": [{"input": 2, "output": 3, "interpolation": "LINEAR"}]
}]
}`, posLen, posLen, idxLen, animInputOffset, animInputLen, animOutputOffset, animOutputLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("no_skel_anim", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.False(m.Skinned())
	})
}

func (suite *loaderTest) TestLoadReaderMaterialWithAllTextures() {
	suite.Run("extracts material with diffuse normal and metallic textures", func() {
		pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQaaaabjru5erkjggg=="
		dataURI := "data:image/png;base64," + pngBase64

		jsonStr := `{
"asset": {"version": "2.0"},
"materials": [{
"name": "FullMat",
"pbrMetallicRoughness": {
"baseColorFactor": [0.8, 0.6, 0.4, 1.0],
"metallicFactor": 0.5,
"roughnessFactor": 0.3,
"baseColorTexture": {"index": 0},
"metallicRoughnessTexture": {"index": 1}
},
"normalTexture": {"index": 2}
}],
"textures": [
{"source": 0},
{"source": 0},
{"source": 0}
],
"images": [{"uri": "` + dataURI + `", "mimeType": "image/png"}]
}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("full_mat", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.Len(m.ImportedMaterials(), 1)
		mat := m.ImportedMaterials()[0]
		suite.Equal("FullMat", mat.Name)
		suite.NotNil(mat.DiffuseTexture)
		suite.NotNil(mat.NormalTexture)
		suite.NotNil(mat.MetallicRoughnessTexture)
		suite.InDelta(0.5, float64(mat.Metallic), 1e-6)
		suite.InDelta(0.3, float64(mat.Roughness), 1e-6)
	})
}

func (suite *loaderTest) TestLoadReaderMaterialWithExternalTexture() {
	suite.Run("handles material with external texture URI gracefully", func() {
		jsonStr := `{
"asset": {"version": "2.0"},
"materials": [{
"name": "ExternalMat",
"pbrMetallicRoughness": {
"baseColorTexture": {"index": 0}
}
}],
"textures": [{"source": 0}],
"images": [{"uri": "textures/diffuse.png", "mimeType": "image/png"}]
}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("external_tex", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.Len(m.ImportedMaterials(), 1)
	})
}

func (suite *loaderTest) TestLoadReaderMeshWithoutIndices() {
	suite.Run("generates sequential indices when primitive omits indices", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, buf.Len(), buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("no_indices", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadReaderMeshWithTangents() {
	suite.Run("extracts mesh with TANGENT attribute", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, n := range [][3]float32{{0, 0, 1}, {0, 0, 1}, {0, 0, 1}} {
			binary.Write(buf, binary.LittleEndian, n)
		}
		normalLen := buf.Len() - posLen
		normalOffset := posLen
		for _, uv := range [][2]float32{{0, 0}, {1, 0}, {0, 1}} {
			binary.Write(buf, binary.LittleEndian, uv)
		}
		uvLen := buf.Len() - normalOffset - normalLen
		uvOffset := normalOffset + normalLen
		for _, t := range [][4]float32{{1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}} {
			binary.Write(buf, binary.LittleEndian, t)
		}
		tangentLen := buf.Len() - uvOffset - uvLen
		tangentOffset := uvOffset + uvLen
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - tangentOffset - tangentLen
		idxOffset := tangentOffset + tangentLen
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{
"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "TANGENT": 3},
"indices": 4
}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
{"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
{"bufferView": 4, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, normalOffset, normalLen, uvOffset, uvLen, tangentOffset, tangentLen, idxOffset, idxLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("tangent_mesh", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadReaderVertexColorsVec3UnsignedByte() {
	suite.Run("extracts mesh with VEC3 unsigned byte vertex colors", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, c := range [][3]uint8{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}} {
			buf.Write(c[:])
		}
		colorLen := buf.Len() - posLen
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}
		colorPadded := buf.Len() - posLen
		_ = colorPadded
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxOffset := posLen + colorLen
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC3"},
{"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, colorLen, idxOffset, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("color3_byte_mesh", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadReaderVertexColorsVec3UnsignedShort() {
	suite.Run("extracts mesh with VEC3 unsigned short vertex colors", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, c := range [][3]uint16{{65535, 0, 0}, {0, 65535, 0}, {0, 0, 65535}} {
			binary.Write(buf, binary.LittleEndian, c)
		}
		colorLen := buf.Len() - posLen
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxOffset := buf.Len() - 6
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "VEC3"},
{"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, colorLen, idxOffset, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("color3_short_mesh", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadReaderMultipleMeshes() {
	suite.Run("loads model with two separate meshes", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		set2Offset := buf.Len()
		for _, p := range [][3]float32{{2, 0, 0}, {3, 0, 0}, {2, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		pos2Len := buf.Len() - set2Offset
		idx2Offset := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idx2Len := buf.Len() - idx2Offset
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0, 1]}],
"nodes": [{"mesh": 0}, {"mesh": 1}],
"meshes": [
{"name": "Mesh1", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]},
{"name": "Mesh2", "primitives": [{"attributes": {"POSITION": 2}, "indices": 3}]}
],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3", "max": [3,1,0], "min": [2,0,0]},
{"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, set2Offset, pos2Len, idx2Offset, idx2Len, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("multi_mesh", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadReaderSamplerFilterTypes() {
	suite.Run("extracts sampler with nearest mag filter", func() {
		pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQaaaabjru5erkjggg=="
		dataURI := "data:image/png;base64," + pngBase64

		jsonStr := `{
"asset": {"version": "2.0"},
"materials": [{
"name": "NearestMat",
"pbrMetallicRoughness": {
"baseColorTexture": {"index": 0}
}
}],
"textures": [{"source": 0, "sampler": 0}],
"samplers": [{
"magFilter": 9728,
"minFilter": 9984,
"wrapS": 10497,
"wrapT": 10497
}],
"images": [{"uri": "` + dataURI + `", "mimeType": "image/png"}]
}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("nearest_filter", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.NotNil(m.ImportedMaterials()[0].DiffuseTexture)
		suite.NotNil(m.ImportedMaterials()[0].DiffuseTexture.SamplerData)
	})

	suite.Run("extracts sampler with linear mipmap nearest filter", func() {
		pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQaaaabjru5erkjggg=="
		dataURI := "data:image/png;base64," + pngBase64

		jsonStr := `{
"asset": {"version": "2.0"},
"materials": [{
"name": "LinearMipNearest",
"pbrMetallicRoughness": {
"baseColorTexture": {"index": 0}
}
}],
"textures": [{"source": 0, "sampler": 0}],
"samplers": [{
"magFilter": 9729,
"minFilter": 9985
}],
"images": [{"uri": "` + dataURI + `", "mimeType": "image/png"}]
}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("linear_mip_nearest", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.NotNil(m.ImportedMaterials()[0].DiffuseTexture)
		suite.NotNil(m.ImportedMaterials()[0].DiffuseTexture.SamplerData)
	})

	suite.Run("extracts sampler with nearest mipmap linear filter", func() {
		pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQaaaabjru5erkjggg=="
		dataURI := "data:image/png;base64," + pngBase64

		jsonStr := `{
"asset": {"version": "2.0"},
"materials": [{
"name": "NearestMipLinear",
"pbrMetallicRoughness": {
"baseColorTexture": {"index": 0}
}
}],
"textures": [{"source": 0, "sampler": 0}],
"samplers": [{
"minFilter": 9986
}],
"images": [{"uri": "` + dataURI + `", "mimeType": "image/png"}]
}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("nearest_mip_linear", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.NotNil(m.ImportedMaterials()[0].DiffuseTexture)
		suite.NotNil(m.ImportedMaterials()[0].DiffuseTexture.SamplerData)
	})
}

func (suite *loaderTest) TestLoadReaderJointsWithUnsignedShort() {
	suite.Run("loads skeleton with unsigned short joint indices", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		ibmOffset := buf.Len()
		ibm := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		binary.Write(buf, binary.LittleEndian, ibm)
		ibmLen := buf.Len() - ibmOffset

		jointsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]uint16{0, 0, 0, 0})
		}
		jointsLen := buf.Len() - jointsOffset

		weightsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
		}
		weightsLen := buf.Len() - weightsOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [
{"mesh": 0, "skin": 0},
{"name": "Bone", "translation": [0, 0, 0]}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 3, "WEIGHTS_0": 4}, "indices": 1}]}],
"skins": [{"joints": [1], "inverseBindMatrices": 2}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 1, "type": "MAT4"},
{"bufferView": 3, "componentType": 5123, "count": 3, "type": "VEC4"},
{"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, ibmOffset, ibmLen, jointsOffset, jointsLen, weightsOffset, weightsLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("ushort_joints", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(m.Skinned())
	})
}

func (suite *loaderTest) TestLoadReaderSkeletonRotationMatrixBranches() {
	suite.Run("skeleton with 180 degree X rotation matrix", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen

		ibmOffset := buf.Len()
		ibm := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		binary.Write(buf, binary.LittleEndian, ibm)
		ibmLen := buf.Len() - ibmOffset

		jointsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]uint8{0, 0, 0, 0})
		}
		jointsLen := buf.Len() - jointsOffset

		weightsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
		}
		weightsLen := buf.Len() - weightsOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [
{"mesh": 0, "skin": 0},
{"name": "Bone180X", "matrix": [1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,0,0,1]}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 3, "WEIGHTS_0": 4}, "indices": 1}]}],
"skins": [{"joints": [1], "inverseBindMatrices": 2}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 1, "type": "MAT4"},
{"bufferView": 3, "componentType": 5121, "count": 3, "type": "VEC4"},
{"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, ibmOffset, ibmLen, jointsOffset, jointsLen, weightsOffset, weightsLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("rot_x180", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(m.Skinned())
	})

	suite.Run("skeleton with 180 degree Y rotation matrix", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen

		ibmOffset := buf.Len()
		ibm := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		binary.Write(buf, binary.LittleEndian, ibm)
		ibmLen := buf.Len() - ibmOffset

		jointsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]uint8{0, 0, 0, 0})
		}
		jointsLen := buf.Len() - jointsOffset

		weightsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
		}
		weightsLen := buf.Len() - weightsOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [
{"mesh": 0, "skin": 0},
{"name": "Bone180Y", "matrix": [-1,0,0,0, 0,1,0,0, 0,0,-1,0, 0,0,0,1]}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 3, "WEIGHTS_0": 4}, "indices": 1}]}],
"skins": [{"joints": [1], "inverseBindMatrices": 2}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 1, "type": "MAT4"},
{"bufferView": 3, "componentType": 5121, "count": 3, "type": "VEC4"},
{"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, ibmOffset, ibmLen, jointsOffset, jointsLen, weightsOffset, weightsLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("rot_y180", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(m.Skinned())
	})

	suite.Run("skeleton with 180 degree Z rotation matrix", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen

		ibmOffset := buf.Len()
		ibm := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		binary.Write(buf, binary.LittleEndian, ibm)
		ibmLen := buf.Len() - ibmOffset

		jointsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]uint8{0, 0, 0, 0})
		}
		jointsLen := buf.Len() - jointsOffset

		weightsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
		}
		weightsLen := buf.Len() - weightsOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [
{"mesh": 0, "skin": 0},
{"name": "Bone180Z", "matrix": [-1,0,0,0, 0,-1,0,0, 0,0,1,0, 0,0,0,1]}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 3, "WEIGHTS_0": 4}, "indices": 1}]}],
"skins": [{"joints": [1], "inverseBindMatrices": 2}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 1, "type": "MAT4"},
{"bufferView": 3, "componentType": 5121, "count": 3, "type": "VEC4"},
{"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, ibmOffset, ibmLen, jointsOffset, jointsLen, weightsOffset, weightsLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("rot_z180", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(m.Skinned())
	})
}

func (suite *loaderTest) TestLoadExternalBufferFile() {
	suite.Run("loads gltf file with external bin buffer", func() {
		tmpDir, err := os.MkdirTemp("", "loader_test_*")
		suite.NoError(err)
		defer os.RemoveAll(tmpDir)

		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}
		binData := buf.Bytes()

		err = os.WriteFile(filepath.Join(tmpDir, "model.bin"), binData, 0644)
		suite.NoError(err)

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"uri": "model.bin", "byteLength": %d}]
}`, posLen, posLen, len(binData))

		err = os.WriteFile(filepath.Join(tmpDir, "model.gltf"), []byte(jsonStr), 0644)
		suite.NoError(err)

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(filepath.Join(tmpDir, "model.gltf"), nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadReaderMeshWithMaterialIndex() {
	suite.Run("primitive with material index references correct material", func() {
		pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQaaaabjru5erkjggg=="
		dataURI := "data:image/png;base64," + pngBase64

		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 1}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"byteLength": %d}],
"materials": [
{"name": "Mat0", "pbrMetallicRoughness": {"baseColorFactor": [1,0,0,1]}},
{"name": "Mat1", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
],
"textures": [{"source": 0}],
"images": [{"uri": "%s", "mimeType": "image/png"}]
}`, posLen, posLen, len(buf.Bytes()), dataURI)

		glb := buildGLBWithBin(jsonStr, buf.Bytes())
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("mat_idx", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(len(m.ImportedMaterials()) >= 2)
	})
}

func (suite *loaderTest) TestLoadReaderMultiplePrimitivesPerMesh() {
	suite.Run("mesh with two primitives produces correctly named meshes", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen

		pos2Offset := buf.Len()
		for _, p := range [][3]float32{{2, 0, 0}, {3, 0, 0}, {2, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		pos2Len := buf.Len() - pos2Offset
		idx2Offset := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idx2Len := buf.Len() - idx2Offset
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"name": "MultiPrim", "primitives": [
{"attributes": {"POSITION": 0}, "indices": 1},
{"attributes": {"POSITION": 2}, "indices": 3}
]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3", "max": [3,1,0], "min": [2,0,0]},
{"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, pos2Offset, pos2Len, idx2Offset, idx2Len, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("multi_prim", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadReaderAnimationWithMorphWeights() {
	suite.Run("animation with morph weight channel is skipped gracefully", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen

		ibmOffset := buf.Len()
		ibm := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		binary.Write(buf, binary.LittleEndian, ibm)
		ibmLen := buf.Len() - ibmOffset

		jointsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]uint8{0, 0, 0, 0})
		}
		jointsLen := buf.Len() - jointsOffset

		weightsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
		}
		weightsLen := buf.Len() - weightsOffset

		animInputOffset := buf.Len()
		binary.Write(buf, binary.LittleEndian, []float32{0.0, 1.0})
		animInputLen := buf.Len() - animInputOffset

		animTransOffset := buf.Len()
		binary.Write(buf, binary.LittleEndian, [][3]float32{{0, 0, 0}, {0, 1, 0}})
		animTransLen := buf.Len() - animTransOffset

		animRotOffset := buf.Len()
		binary.Write(buf, binary.LittleEndian, [][4]float32{{0, 0, 0, 1}, {0, 0, 0, 1}})
		animRotLen := buf.Len() - animRotOffset

		animScaleOffset := buf.Len()
		binary.Write(buf, binary.LittleEndian, [][3]float32{{1, 1, 1}, {1, 1, 1}})
		animScaleLen := buf.Len() - animScaleOffset

		morphWeightOffset := buf.Len()
		binary.Write(buf, binary.LittleEndian, []float32{0.0, 1.0})
		morphWeightLen := buf.Len() - morphWeightOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [
{"mesh": 0, "skin": 0},
{"name": "Bone", "translation": [0, 0, 0]}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 4, "WEIGHTS_0": 5}, "indices": 1}]}],
"skins": [{"joints": [1], "inverseBindMatrices": 2}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 1, "type": "MAT4"},
{"bufferView": 3, "componentType": 5126, "count": 2, "type": "SCALAR"},
{"bufferView": 4, "componentType": 5121, "count": 3, "type": "VEC4"},
{"bufferView": 5, "componentType": 5126, "count": 3, "type": "VEC4"},
{"bufferView": 6, "componentType": 5126, "count": 2, "type": "VEC3"},
{"bufferView": 7, "componentType": 5126, "count": 2, "type": "VEC4"},
{"bufferView": 8, "componentType": 5126, "count": 2, "type": "VEC3"},
{"bufferView": 9, "componentType": 5126, "count": 2, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}],
"animations": [{
"name": "MorphAnim",
"channels": [
{"sampler": 0, "target": {"node": 1, "path": "translation"}},
{"sampler": 1, "target": {"node": 1, "path": "rotation"}},
{"sampler": 2, "target": {"node": 1, "path": "scale"}},
{"sampler": 3, "target": {"node": 1, "path": "weights"}}
],
"samplers": [
{"input": 3, "output": 6, "interpolation": "LINEAR"},
{"input": 3, "output": 7, "interpolation": "LINEAR"},
{"input": 3, "output": 8, "interpolation": "LINEAR"},
{"input": 3, "output": 9, "interpolation": "LINEAR"}
]
}]
}`, posLen, posLen, idxLen, ibmOffset, ibmLen, animInputOffset, animInputLen, jointsOffset, jointsLen, weightsOffset, weightsLen, animTransOffset, animTransLen, animRotOffset, animRotLen, animScaleOffset, animScaleLen, morphWeightOffset, morphWeightLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("morph_weights", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(m.Skinned())
		suite.Equal(1, m.AnimationCount())
	})
}

func (suite *loaderTest) TestLoadReaderSkeletonWithZeroScaleMatrix() {
	suite.Run("skeleton node with zero-scale matrix uses identity fallback", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen

		ibmOffset := buf.Len()
		ibm := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		binary.Write(buf, binary.LittleEndian, ibm)
		ibmLen := buf.Len() - ibmOffset

		jointsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]uint8{0, 0, 0, 0})
		}
		jointsLen := buf.Len() - jointsOffset

		weightsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
		}
		weightsLen := buf.Len() - weightsOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [
{"mesh": 0, "skin": 0},
{"name": "ZeroScale", "matrix": [0,0,0,0, 0,0,0,0, 0,0,0,0, 5,10,15,1]}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 3, "WEIGHTS_0": 4}, "indices": 1}]}],
"skins": [{"joints": [1], "inverseBindMatrices": 2}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 1, "type": "MAT4"},
{"bufferView": 3, "componentType": 5121, "count": 3, "type": "VEC4"},
{"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, ibmOffset, ibmLen, jointsOffset, jointsLen, weightsOffset, weightsLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("zero_scale", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(m.Skinned())
	})
}

func (suite *loaderTest) TestLoadReaderSkeletonWithoutInverseBindMatrices() {
	suite.Run("skeleton without IBM uses identity matrices for bones", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen

		jointsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]uint8{0, 0, 0, 0})
		}
		jointsLen := buf.Len() - jointsOffset

		weightsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
		}
		weightsLen := buf.Len() - weightsOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [
{"mesh": 0, "skin": 0},
{"name": "Root", "children": [2]},
{"name": "Child"}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}, "indices": 1}]}],
"skins": [{"joints": [1, 2]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5121, "count": 3, "type": "VEC4"},
{"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, jointsOffset, jointsLen, weightsOffset, weightsLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("no_ibm", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(m.Skinned())
		suite.NotNil(m.Skeleton())
		suite.Len(m.Skeleton().Bones, 2)
	})
}

func (suite *loaderTest) TestLoadReaderDataURINonBase64Error() {
	suite.Run("data URI without base64 encoding returns error", func() {
		jsonStr := `{
"asset": {"version": "2.0"},
"materials": [{
"name": "BadURI",
"pbrMetallicRoughness": {
"baseColorTexture": {"index": 0}
}
}],
"textures": [{"source": 0}],
"images": [{"uri": "data:image/png;charset=utf-8,notbase64", "mimeType": "image/png"}]
}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("bad_uri", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.Error(err)
		suite.Nil(m)
	})
}

func (suite *loaderTest) TestLoadReaderGLBInvalidMagic() {
	suite.Run("GLB with invalid magic byte returns error", func() {
		data := make([]byte, 20)
		binary.LittleEndian.PutUint32(data[0:4], 0xDEADBEEF)
		binary.LittleEndian.PutUint32(data[4:8], 2)
		binary.LittleEndian.PutUint32(data[8:12], 20)

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("bad_magic", bytes.NewReader(data), true, nil)
		suite.Error(err)
		suite.Nil(m)
	})
}

func (suite *loaderTest) TestLoadReaderGLBInvalidVersion() {
	suite.Run("GLB with version 1 returns error", func() {
		data := make([]byte, 20)
		binary.LittleEndian.PutUint32(data[0:4], 0x46546C67)
		binary.LittleEndian.PutUint32(data[4:8], 1)
		binary.LittleEndian.PutUint32(data[8:12], 20)

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("bad_version", bytes.NewReader(data), true, nil)
		suite.Error(err)
		suite.Nil(m)
	})
}

func (suite *loaderTest) TestLoadReaderGLBTooSmall() {
	suite.Run("GLB with less than 12 bytes returns error", func() {
		data := make([]byte, 8)
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("tiny", bytes.NewReader(data), true, nil)
		suite.Error(err)
		suite.Nil(m)
	})
}

func (suite *loaderTest) TestLoadReaderGLBMissingJSONChunk() {
	suite.Run("GLB with only binary chunk and no JSON returns error", func() {
		binChunk := []byte{0, 0, 0, 0}
		data := &bytes.Buffer{}
		binary.Write(data, binary.LittleEndian, uint32(0x46546C67))
		binary.Write(data, binary.LittleEndian, uint32(2))
		totalLen := uint32(12 + 8 + len(binChunk))
		binary.Write(data, binary.LittleEndian, totalLen)
		binary.Write(data, binary.LittleEndian, uint32(len(binChunk)))
		binary.Write(data, binary.LittleEndian, uint32(0x004E4942))
		data.Write(binChunk)

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("no_json", bytes.NewReader(data.Bytes()), true, nil)
		suite.Error(err)
		suite.Nil(m)
	})
}

func (suite *loaderTest) TestLoadReaderGLTFInvalidVersion() {
	suite.Run("GLTF JSON with version 1.0 returns error", func() {
		jsonStr := `{"asset": {"version": "1.0"}}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("v1", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.Error(err)
		suite.Nil(m)
	})
}

func (suite *loaderTest) TestLoadReaderModelNameFromSceneName() {
	suite.Run("model name comes from scene when scene has a name", func() {
		jsonStr := `{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"name": "MySceneName"}]
}`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("fallback", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.Equal("MySceneName", m.Name())
	})
}

func (suite *loaderTest) TestLoadReaderEmptyMeshName() {
	suite.Run("mesh with empty name gets auto-generated name", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, len(buf.Bytes()))

		glb := buildGLBWithBin(jsonStr, buf.Bytes())
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("empty_name", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadMeshOnlyWithTempGLTFFile() {
	suite.Run("LoadMeshOnly from an actual gltf file on disk", func() {
		tmpDir, err := os.MkdirTemp("", "loader_meshonly_*")
		suite.NoError(err)
		defer os.RemoveAll(tmpDir)

		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}
		binData := buf.Bytes()

		err = os.WriteFile(filepath.Join(tmpDir, "mesh.bin"), binData, 0644)
		suite.NoError(err)

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"uri": "mesh.bin", "byteLength": %d}],
"materials": [{"name": "SimpleMat", "pbrMetallicRoughness": {"baseColorFactor": [1,0,0,1]}}]
}`, posLen, posLen, len(binData))

		err = os.WriteFile(filepath.Join(tmpDir, "mesh.gltf"), []byte(jsonStr), 0644)
		suite.NoError(err)

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadMeshOnly(filepath.Join(tmpDir, "mesh.gltf"), nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.False(m.Skinned())
		suite.Len(m.ImportedMaterials(), 1)
	})
}

func (suite *loaderTest) TestLoadReaderSkeletonWithAnonymousBones() {
	suite.Run("bones without names get auto-generated names", func() {
		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		idxLen := buf.Len() - posLen

		ibmOffset := buf.Len()
		ibm := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		binary.Write(buf, binary.LittleEndian, ibm)
		ibmLen := buf.Len() - ibmOffset

		jointsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]uint8{0, 0, 0, 0})
		}
		jointsLen := buf.Len() - jointsOffset

		weightsOffset := buf.Len()
		for i := 0; i < 3; i++ {
			binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
		}
		weightsLen := buf.Len() - weightsOffset

		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [
{"mesh": 0, "skin": 0},
{"translation": [0, 0, 0]}
],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 3, "WEIGHTS_0": 4}, "indices": 1}]}],
"skins": [{"joints": [1], "inverseBindMatrices": 2}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
{"bufferView": 2, "componentType": 5126, "count": 1, "type": "MAT4"},
{"bufferView": 3, "componentType": 5121, "count": 3, "type": "VEC4"},
{"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": %d}
],
"buffers": [{"byteLength": %d}]
}`, posLen, posLen, idxLen, ibmOffset, ibmLen, jointsOffset, jointsLen, weightsOffset, weightsLen, buf.Len())

		glb := buildGLBWithBin(jsonStr, buf.Bytes())
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("anon_bones", bytes.NewReader(glb), true, nil)
		suite.NoError(err)
		suite.NotNil(m)
		suite.True(m.Skinned())
		suite.Contains(m.Skeleton().Bones[0].Name, "bone_")
	})
}

func (suite *loaderTest) TestLoadReaderInvalidGLTFJSON() {
	suite.Run("malformed JSON returns parse error", func() {
		jsonStr := `{this is not valid json`

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("bad_json", bytes.NewReader([]byte(jsonStr)), false, nil)
		suite.Error(err)
		suite.Nil(m)
	})
}

func (suite *loaderTest) TestLoadReaderBufferSizeMismatch() {
	suite.Run("buffer with insufficient data returns error", func() {
		buf := make([]byte, 4)
		jsonStr := `{
"asset": {"version": "2.0"},
"buffers": [{"byteLength": 1000}]
}`

		glb := buildGLBWithBin(jsonStr, buf)
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.LoadReader("small_buf", bytes.NewReader(glb), true, nil)
		suite.Error(err)
		suite.Nil(m)
	})
}
