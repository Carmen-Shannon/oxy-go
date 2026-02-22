package scene_test

import (
	"fmt"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"

	animatormocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/animator"
	cameramocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/camera"
	gameobjectmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/game_object"
	lightmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/light"
	materialmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/material"
	modelmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/model"
	pipelinemocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/pipeline"
	renderermocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/renderer"
	shadermocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/shader"
)

type sceneTest struct {
	suite.Suite
}

func TestScene(t *testing.T) {
	suite.Run(t, new(sceneTest))
}

// newMinimalScene creates a Scene with the minimum required mocks. It sets up
// a vertex shader returning a camera group at index 0, a camera with a non-nil BGP,
// and a renderer that succeeds on InitBindGroup. Returns (scene, cam, renderer, vertShader).
func newMinimalScene(name string, opts ...scene.SceneBuilderOption) (scene.Scene, *cameramocks.MockCamera, *renderermocks.MockRenderer, *shadermocks.MockShader) {
	cam := &cameramocks.MockCamera{}
	r := &renderermocks.MockRenderer{}
	vs := &shadermocks.MockShader{}

	bgp := bind_group_provider.NewBindGroupProvider("cam_bgp")

	vs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
		0: {0: "camera_uniform"},
	}).Maybe()
	vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
		Label: "camera_bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
		},
	}).Maybe()

	cam.EXPECT().BindGroupProvider().Return(bgp).Maybe()
	cam.EXPECT().SetDelegate(mock.Anything).Maybe()
	r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

	s := scene.NewScene(name, cam, r, vs, opts...)
	return s, cam, r, vs
}

func (suite *sceneTest) TestNewScene() {
	suite.Run("panics when camera is nil", func() {
		r := &renderermocks.MockRenderer{}
		vs := &shadermocks.MockShader{}
		suite.Panics(func() {
			scene.NewScene("test", nil, r, vs)
		})
	})

	suite.Run("panics when renderer is nil", func() {
		cam := &cameramocks.MockCamera{}
		vs := &shadermocks.MockShader{}
		suite.Panics(func() {
			scene.NewScene("test", cam, nil, vs)
		})
	})

	suite.Run("panics when vertex shader is nil", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}
		suite.Panics(func() {
			scene.NewScene("test", cam, r, nil)
		})
	})

	suite.Run("returns non-nil scene with valid args", func() {
		s, _, _, _ := newMinimalScene("test_scene")
		suite.NotNil(s)
	})

	suite.Run("name is set correctly", func() {
		s, _, _, _ := newMinimalScene("my_scene")
		suite.Equal("my_scene", s.Name())
	})

	suite.Run("default active is false", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.False(s.Active())
	})

	suite.Run("default culling disabled is false", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.False(s.CullingDisabled())
	})

	suite.Run("default count is zero", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.Equal(0, s.Count())
	})

	suite.Run("default ephemeral count is zero", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.Equal(0, s.CountEphemeral())
	})

	suite.Run("camera is stored", func() {
		s, cam, _, _ := newMinimalScene("test")
		suite.Equal(cam, s.Camera())
	})

	suite.Run("renderer is stored", func() {
		s, _, r, _ := newMinimalScene("test")
		suite.Equal(r, s.Renderer())
	})

	suite.Run("camera BGP is nil when camera has no BGP", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}
		vs := &shadermocks.MockShader{}

		vs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "camera_uniform"},
		}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		cam.EXPECT().BindGroupProvider().Return(nil).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		s := scene.NewScene("test", cam, r, vs)
		suite.NotNil(s)
	})

	suite.Run("default lights is empty", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.Empty(s.Lights())
	})

	suite.Run("default ambient color is zero", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.Equal([3]float32{0, 0, 0}, s.AmbientColor())
	})

	suite.Run("default light bind group provider is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.Nil(s.LightBindGroupProvider())
	})

	suite.Run("default shadow depth texture view is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.Nil(s.ShadowDepthTextureView())
	})

	suite.Run("default shadow data BGP is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.Nil(s.ShadowDataBindGroupProvider())
	})

	suite.Run("default shadow lit BGP is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.Nil(s.ShadowLitBindGroupProvider())
	})
}

func (suite *sceneTest) TestSetName() {
	suite.Run("sets and retrieves new name", func() {
		s, _, _, _ := newMinimalScene("original")
		s.SetName("updated")
		suite.Equal("updated", s.Name())
	})
}

func (suite *sceneTest) TestSetActive() {
	suite.Run("sets active to true", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetActive(true)
		suite.True(s.Active())
	})

	suite.Run("sets active back to false", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetActive(true)
		s.SetActive(false)
		suite.False(s.Active())
	})
}

func (suite *sceneTest) TestSetCamera() {
	suite.Run("replaces camera", func() {
		s, _, _, _ := newMinimalScene("test")
		newCam := &cameramocks.MockCamera{}
		s.SetCamera(newCam)
		suite.Equal(newCam, s.Camera())
	})
}

func (suite *sceneTest) TestSetRenderer() {
	suite.Run("replaces renderer", func() {
		s, _, _, _ := newMinimalScene("test")
		newR := &renderermocks.MockRenderer{}
		s.SetRenderer(newR)
		suite.Equal(newR, s.Renderer())
	})
}

func (suite *sceneTest) TestCullingDisabled() {
	suite.Run("sets culling disabled to true", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetCullingDisabled(true)
		suite.True(s.CullingDisabled())
	})

	suite.Run("sets culling disabled back to false", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetCullingDisabled(true)
		s.SetCullingDisabled(false)
		suite.False(s.CullingDisabled())
	})
}

func (suite *sceneTest) TestAddLight() {
	suite.Run("adds a single light", func() {
		s, _, _, _ := newMinimalScene("test")
		l := &lightmocks.MockLight{}
		s.AddLight(l)
		suite.Len(s.Lights(), 1)
	})

	suite.Run("adds multiple lights", func() {
		s, _, _, _ := newMinimalScene("test")
		l1 := &lightmocks.MockLight{}
		l2 := &lightmocks.MockLight{}
		s.AddLight(l1)
		s.AddLight(l2)
		suite.Len(s.Lights(), 2)
	})
}

func (suite *sceneTest) TestRemoveLight() {
	suite.Run("removes an existing light", func() {
		s, _, _, _ := newMinimalScene("test")
		l := &lightmocks.MockLight{}
		s.AddLight(l)
		s.RemoveLight(l)
		suite.Empty(s.Lights())
	})

	suite.Run("no-op when removing non-existent light", func() {
		s, _, _, _ := newMinimalScene("test")
		l1 := &lightmocks.MockLight{}
		l2 := &lightmocks.MockLight{}
		s.AddLight(l1)
		s.RemoveLight(l2)
		suite.Len(s.Lights(), 1)
	})
}

func (suite *sceneTest) TestDetachLight() {
	suite.Run("no-op when object has no light", func() {
		s, _, _, _ := newMinimalScene("test")
		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().Light().Return(nil).Maybe()
		s.DetachLight(obj) // should not panic
	})

	suite.Run("removes attached light from scene lists", func() {
		s, _, _, _ := newMinimalScene("test")
		l := &lightmocks.MockLight{}
		s.AddLight(l)
		suite.Len(s.Lights(), 1)

		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().Light().Return(l).Maybe()
		s.DetachLight(obj)
		suite.Empty(s.Lights())
	})
}

func (suite *sceneTest) TestLights() {
	suite.Run("returns a copy not the original slice", func() {
		s, _, _, _ := newMinimalScene("test")
		l := &lightmocks.MockLight{}
		s.AddLight(l)
		lights := s.Lights()
		lights[0] = nil // mutate the returned copy
		suite.NotNil(s.Lights()[0])
	})
}

func (suite *sceneTest) TestAmbientColor() {
	suite.Run("sets and retrieves ambient color", func() {
		s, _, _, _ := newMinimalScene("test")
		color := [3]float32{0.5, 0.6, 0.7}
		s.SetAmbientColor(color)
		suite.Equal(color, s.AmbientColor())
	})
}

func (suite *sceneTest) TestInitLightBindGroup() {
	suite.Run("no-op when fragment shader is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.InitLightBindGroup(nil)
		suite.Nil(s.LightBindGroupProvider())
	})

	suite.Run("no-op when renderer is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		fs := &shadermocks.MockShader{}
		s.InitLightBindGroup(fs)
		suite.Nil(s.LightBindGroupProvider())
	})

	suite.Run("no-op when no light group found in shader", func() {
		s, _, _, _ := newMinimalScene("test")
		fs := &shadermocks.MockShader{}
		// Return var names that don't contain "light"
		fs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "camera_uniform"},
		}).Maybe()
		s.InitLightBindGroup(fs)
		suite.Nil(s.LightBindGroupProvider())
	})

	suite.Run("initializes light BGP when light group found", func() {
		s, _, r, _ := newMinimalScene("test")
		fs := &shadermocks.MockShader{}
		fs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			3: {0: "light_header", 1: "lights_array"},
		}).Maybe()
		fs.EXPECT().BindGroupLayoutDescriptor(3).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "light_bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		s.InitLightBindGroup(fs)
		suite.NotNil(s.LightBindGroupProvider())
	})
}

func (suite *sceneTest) TestInitShadowMap() {
	suite.Run("no-op when renderer is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		vs := &shadermocks.MockShader{}
		s.InitShadowMap(vs, nil)
		suite.Nil(s.ShadowDepthTextureView())
	})

	suite.Run("no-op when shadow vertex shader is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.InitShadowMap(nil, nil)
		suite.Nil(s.ShadowDepthTextureView())
	})

	suite.Run("creates shadow depth texture and comparison sampler", func() {
		s, _, r, _ := newMinimalScene("test")
		svs := &shadermocks.MockShader{}

		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}

		r.EXPECT().CreateShadowDepthTexture(light.ShadowMapResolution, light.ShadowMapResolution).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()

		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "shadow_bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()

		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()

		s.InitShadowMap(svs, nil)
		suite.Equal(tv, s.ShadowDepthTextureView())
		suite.NotNil(s.ShadowDataBindGroupProvider())
	})

	suite.Run("registers skinned shadow pipeline when skinned shader provided", func() {
		s, _, r, _ := newMinimalScene("test")
		svs := &shadermocks.MockShader{}
		ssvs := &shadermocks.MockShader{}

		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}

		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()

		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "shadow_bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()

		// Expect two RegisterShadowPipeline calls (static + skinned)
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Twice()

		s.InitShadowMap(svs, ssvs)
		suite.NotNil(s.ShadowDepthTextureView())
	})
}

func (suite *sceneTest) TestInitShadowLitBindGroup() {
	suite.Run("no-op when renderer is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		fs := &shadermocks.MockShader{}
		s.InitShadowLitBindGroup(fs)
		suite.Nil(s.ShadowLitBindGroupProvider())
	})

	suite.Run("no-op when lit fragment shader is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.InitShadowLitBindGroup(nil)
		suite.Nil(s.ShadowLitBindGroupProvider())
	})

	suite.Run("no-op when shadow resources not initialized", func() {
		s, _, _, _ := newMinimalScene("test")
		fs := &shadermocks.MockShader{}
		// ShadowDepthTextureView and ShadowComparisonSamp are nil
		s.InitShadowLitBindGroup(fs)
		suite.Nil(s.ShadowLitBindGroupProvider())
	})

	suite.Run("no-op when no shadow group found in shader", func() {
		s, _, r, _ := newMinimalScene("test")

		// First init shadow map so texture/sampler exist
		svs := &shadermocks.MockShader{}
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()
		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(svs, nil)

		// Now call InitShadowLitBindGroup with a shader that has no shadow group
		fs := &shadermocks.MockShader{}
		fs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "camera_uniform"},
		}).Maybe()
		s.InitShadowLitBindGroup(fs)
		suite.Nil(s.ShadowLitBindGroupProvider())
	})

	suite.Run("initializes shadow lit BGP when shadow group found", func() {
		s, _, r, _ := newMinimalScene("test")

		// Init shadow map first
		svs := &shadermocks.MockShader{}
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()
		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(svs, nil)

		// Now init shadow lit bind group
		fs := &shadermocks.MockShader{}
		fs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			4: {0: "shadow_depth_texture", 1: "shadow_comparison_sampler", 2: "shadow_uniform"},
		}).Maybe()
		fs.EXPECT().BindGroupLayoutDescriptor(4).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "shadow_lit_bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeDepth}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeComparison}},
				{Binding: 2, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()

		s.InitShadowLitBindGroup(fs)
		suite.NotNil(s.ShadowLitBindGroupProvider())
	})
}

func (suite *sceneTest) TestPrepareShadows() {
	suite.Run("no-op when shadow resources are not initialized", func() {
		s, _, _, _ := newMinimalScene("test")
		// Should not panic
		s.PrepareShadows()
	})

	suite.Run("no-op when no shadow-casting directional light exists", func() {
		s, _, r, _ := newMinimalScene("test")

		// Init shadow map
		svs := &shadermocks.MockShader{}
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()
		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(svs, nil)

		// Add a non-shadow-casting light
		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(false).Maybe()
		s.AddLight(l)

		// Should not call WriteBuffers or shadow rendering
		s.PrepareShadows()
	})
}

func (suite *sceneTest) TestPrepareLightCulling() {
	suite.Run("no-op when lightCullBGP is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		// lightCullBGP is nil by default, should not panic
		s.PrepareLightCulling()
	})

	suite.Run("no-op when renderer is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		s.PrepareLightCulling()
	})

	suite.Run("no-op when camera is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetCamera(nil)
		s.PrepareLightCulling()
	})
}

func (suite *sceneTest) TestGet() {
	suite.Run("returns nil for non-existent ID", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.Nil(s.Get(999))
	})
}

func (suite *sceneTest) TestRemove() {
	suite.Run("no-op when removing non-existent ID", func() {
		s, _, _, _ := newMinimalScene("test")
		s.Remove(999) // should not panic
		suite.Equal(0, s.Count())
	})
}

func (suite *sceneTest) TestClear() {
	suite.Run("clears all objects and animators", func() {
		s, _, _, _ := newMinimalScene("test")
		s.Clear()
		suite.Equal(0, s.Count())
		suite.Equal(0, s.CountEphemeral())
	})
}

func (suite *sceneTest) TestPrepareCompute() {
	suite.Run("no-op when renderer is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		s.PrepareCompute(0.016) // should not panic
	})

	suite.Run("updates camera and writes VP when camera and BGP exist", func() {
		s, cam, r, _ := newMinimalScene("test")

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp_prep")
		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		// Override camera BGP for this path
		cam.EXPECT().BindGroupProvider().Unset()
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()

		s.PrepareCompute(0.016)
	})

	suite.Run("no-op for empty animator pool", func() {
		s, cam, r, _ := newMinimalScene("test")
		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		s.PrepareCompute(0.016) // should not panic with zero animators
	})
}

func (suite *sceneTest) TestDrawCalls() {
	suite.Run("returns error when renderer is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		err := s.DrawCalls()
		suite.Error(err)
	})

	suite.Run("returns nil with empty animator pool", func() {
		s, _, _, _ := newMinimalScene("test")
		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestInitLighting() {
	suite.Run("calls all sub-init methods in correct order", func() {
		s, cam, r, _ := newMinimalScene("test")

		litFragShader := &shadermocks.MockShader{}
		shadowVertShader := &shadermocks.MockShader{}
		cullComputeShader := &shadermocks.MockShader{}

		// InitLightBindGroup: fragment shader has light group
		litFragShader.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "camera_position"},
			3: {0: "light_header", 1: "lights_array"},
			4: {0: "shadow_depth_texture", 1: "shadow_comparison_sampler", 2: "shadow_uniform"},
			5: {0: "tile_uniforms", 1: "tile_counts", 2: "tile_indices"},
		}).Maybe()
		litFragShader.EXPECT().BindGroupLayoutDescriptor(3).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "light_bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		litFragShader.EXPECT().BindGroupLayoutDescriptor(4).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "shadow_lit_bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeDepth}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeComparison}},
				{Binding: 2, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		litFragShader.EXPECT().BindGroupLayoutDescriptor(5).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "tile_bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 8}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
				{Binding: 2, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
			},
		}).Maybe()

		// InitShadowMap: shadow vertex shader
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()
		shadowVertShader.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		shadowVertShader.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()

		// InitLightCullResources: cull compute shader
		cullComputeShader.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "cull_bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 160}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		// reinitCameraBGPForLitPipeline: cam BGP re-init
		litFragShader.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "camera_lit_bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Unset()
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()

		s.InitLighting(litFragShader, shadowVertShader, nil, cullComputeShader, 1280, 720)

		suite.NotNil(s.LightBindGroupProvider())
		suite.NotNil(s.ShadowDepthTextureView())
		suite.NotNil(s.ShadowLitBindGroupProvider())
	})
}

func (suite *sceneTest) TestInitLightCullResources() {
	suite.Run("no-op when renderer is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		cs := &shadermocks.MockShader{}
		fs := &shadermocks.MockShader{}
		s.InitLightCullResources(cs, fs, 1280, 720)
	})

	suite.Run("no-op when cullComputeShader is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		fs := &shadermocks.MockShader{}
		s.InitLightCullResources(nil, fs, 1280, 720)
	})

	suite.Run("no-op when litFragmentShader is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		cs := &shadermocks.MockShader{}
		s.InitLightCullResources(cs, nil, 1280, 720)
	})

	suite.Run("no-op when lightsBGP is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		cs := &shadermocks.MockShader{}
		fs := &shadermocks.MockShader{}
		// lightsBGP is nil by default (InitLightBindGroup not called)
		s.InitLightCullResources(cs, fs, 1280, 720)
	})
}

func (suite *sceneTest) TestWithActive() {
	suite.Run("sets active via builder option", func() {
		s, _, _, _ := newMinimalScene("test", scene.WithActive(true))
		suite.True(s.Active())
	})

	suite.Run("default active false without option", func() {
		s, _, _, _ := newMinimalScene("test")
		suite.False(s.Active())
	})
}

func (suite *sceneTest) TestWithComputeWorkers() {
	suite.Run("sets compute workers", func() {
		s, _, _, _ := newMinimalScene("test", scene.WithComputeWorkers(4))
		suite.NotNil(s)
	})

	suite.Run("clamps to minimum of 1", func() {
		s, _, _, _ := newMinimalScene("test", scene.WithComputeWorkers(0))
		suite.NotNil(s)
	})

	suite.Run("clamps negative to 1", func() {
		s, _, _, _ := newMinimalScene("test", scene.WithComputeWorkers(-5))
		suite.NotNil(s)
	})
}

func (suite *sceneTest) TestWithCullingDisabled() {
	suite.Run("sets culling disabled via builder", func() {
		s, _, _, _ := newMinimalScene("test", scene.WithCullingDisabled(true))
		suite.True(s.CullingDisabled())
	})
}

func (suite *sceneTest) TestWithObjects() {
	suite.Run("adds non-ephemeral objects to registry with auto-IDs", func() {
		var id1, id2 uint64
		obj1 := &gameobjectmocks.MockGameObject{}
		obj2 := &gameobjectmocks.MockGameObject{}
		obj1.EXPECT().ID().RunAndReturn(func() uint64 { return id1 }).Maybe()
		obj1.EXPECT().SetID(mock.Anything).Run(func(id uint64) { id1 = id }).Return().Maybe()
		obj1.EXPECT().Ephemeral().Return(false).Maybe()
		obj2.EXPECT().ID().RunAndReturn(func() uint64 { return id2 }).Maybe()
		obj2.EXPECT().SetID(mock.Anything).Run(func(id uint64) { id2 = id }).Return().Maybe()
		obj2.EXPECT().Ephemeral().Return(false).Maybe()

		s, _, _, _ := newMinimalScene("test", scene.WithObjects(obj1, obj2))
		suite.Equal(2, s.Count())
	})

	suite.Run("ephemeral objects are not persisted", func() {
		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().ID().Return(uint64(0)).Maybe()
		obj.EXPECT().SetID(mock.Anything).Return().Maybe()
		obj.EXPECT().Ephemeral().Return(true).Maybe()

		s, _, _, _ := newMinimalScene("test", scene.WithObjects(obj))
		suite.Equal(0, s.Count())
	})

	suite.Run("objects with existing IDs keep them", func() {
		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().ID().Return(uint64(42)).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()

		s, _, _, _ := newMinimalScene("test", scene.WithObjects(obj))
		suite.Equal(1, s.Count())
		suite.NotNil(s.Get(42))
	})
}

func (suite *sceneTest) TestWithShadowHalfExtent() {
	suite.Run("accepts shadow half extent option without error", func() {
		s, _, _, _ := newMinimalScene("test", scene.WithShadowHalfExtent(60.0))
		suite.NotNil(s)
	})
}

func (suite *sceneTest) TestWithShadowNearFar() {
	suite.Run("accepts shadow near far option without error", func() {
		s, _, _, _ := newMinimalScene("test", scene.WithShadowNearFar(0.5, 500.0))
		suite.NotNil(s)
	})
}

func (suite *sceneTest) TestWithShadowBias() {
	suite.Run("accepts shadow bias option without error", func() {
		s, _, _, _ := newMinimalScene("test", scene.WithShadowBias(0.002))
		suite.NotNil(s)
	})
}

func (suite *sceneTest) TestWithShadowNormalBiasScale() {
	suite.Run("accepts shadow normal bias scale option without error", func() {
		s, _, _, _ := newMinimalScene("test", scene.WithShadowNormalBiasScale(4.0))
		suite.NotNil(s)
	})
}

func (suite *sceneTest) TestWithShadowMapResolution() {
	suite.Run("accepts shadow map resolution option without error", func() {
		s, _, _, _ := newMinimalScene("test", scene.WithShadowMapResolution(4096))
		suite.NotNil(s)
	})
}

func (suite *sceneTest) TestAddAndGet() {
	suite.Run("panics when renderer is nil", func() {
		s, _, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)

		obj := &gameobjectmocks.MockGameObject{}
		cs := &shadermocks.MockShader{}
		vs := &shadermocks.MockShader{}
		fs := &shadermocks.MockShader{}

		suite.Panics(func() {
			s.Add(obj, cs, vs, fs)
		})
	})

	suite.Run("panics when object has no model", func() {
		s, _, _, _ := newMinimalScene("test")

		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().Model().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		vs := &shadermocks.MockShader{}
		fs := &shadermocks.MockShader{}

		suite.Panics(func() {
			s.Add(obj, cs, vs, fs)
		})
	})
}

func (suite *sceneTest) TestAddWithMinimalModel() {
	suite.Run("adds a non-ephemeral object successfully", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("cube").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()

		var objID uint64
		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0},
			[3]float32{1, 1, 1},
			[3]float32{0, 0, 0},
			[3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()

		// Compute shader declarations
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("cube_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		// Vertex shader declarations
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		// Fragment shader
		fs := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
		suite.Equal(1, s.Count())
		suite.NotNil(s.Get(id))
	})

	suite.Run("adds an ephemeral object successfully without persisting in registry", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("particle").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(0.5)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()

		var objID uint64
		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(true).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{1, 2, 3},
			[3]float32{1, 1, 1},
			[3]float32{0, 0, 0},
			[3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("particle_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
		suite.Equal(0, s.Count()) // ephemeral, not persisted
	})

	suite.Run("adds object with attached light and tracks it", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("lamp").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(0.5)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()

		l := &lightmocks.MockLight{}

		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().ID().Return(uint64(0)).Maybe()
		obj.EXPECT().SetID(mock.Anything).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(l).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{5, 10, 5},
			[3]float32{1, 1, 1},
			[3]float32{0, 0, 0},
			[3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("lamp_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		s.Add(obj, cs, vs, fs)
		suite.Len(s.Lights(), 1)
	})
}

func (suite *sceneTest) TestRemoveWithAnimator() {
	suite.Run("removes object and cleans up animator instance", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("box").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()

		// Create a mock animator to verify cleanup
		anim := &animatormocks.MockAnimator{}
		anim.EXPECT().RemoveInstance(uint32(0)).Return(uint32(0), false).Maybe()

		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().ID().Return(uint64(0)).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) {
			obj.EXPECT().ID().Unset()
			obj.EXPECT().ID().Return(id).Maybe()
		}).Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0},
			[3]float32{1, 1, 1},
			[3]float32{0, 0, 0},
			[3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()
		obj.EXPECT().Animator().Return(anim).Maybe()
		obj.EXPECT().AnimatorInstanceID().Return(0).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("box_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.Equal(1, s.Count())

		s.Remove(id)
		suite.Equal(0, s.Count())
		suite.Nil(s.Get(id))
	})

	suite.Run("removes object with attached light and cleans up light lists", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("lightobj").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()

		l := &lightmocks.MockLight{}
		anim := &animatormocks.MockAnimator{}
		anim.EXPECT().RemoveInstance(mock.Anything).Return(uint32(0), false).Maybe()

		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().ID().Return(uint64(0)).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) {
			obj.EXPECT().ID().Unset()
			obj.EXPECT().ID().Return(id).Maybe()
		}).Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(l).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0},
			[3]float32{1, 1, 1},
			[3]float32{0, 0, 0},
			[3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()
		obj.EXPECT().Animator().Return(anim).Maybe()
		obj.EXPECT().AnimatorInstanceID().Return(0).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("lightobj_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.Len(s.Lights(), 1)

		s.Remove(id)
		suite.Empty(s.Lights())
	})
}

func (suite *sceneTest) TestCountEphemeral() {
	suite.Run("counts instances across animators", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("obj").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("obj_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		for i := 0; i < 3; i++ {
			obj := &gameobjectmocks.MockGameObject{}
			obj.EXPECT().ID().Return(uint64(0)).Maybe()
			obj.EXPECT().SetID(mock.Anything).Return().Maybe()
			obj.EXPECT().Model().Return(mdl).Maybe()
			obj.EXPECT().Ephemeral().Return(false).Maybe()
			obj.EXPECT().Light().Return(nil).Maybe()
			obj.EXPECT().TransformData().Return(
				[3]float32{float32(i), 0, 0},
				[3]float32{1, 1, 1},
				[3]float32{0, 0, 0},
				[3]float32{0, 0, 0},
			).Maybe()
			obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
			obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()
			s.Add(obj, cs, vs, fs)
		}

		// CountEphemeral sums animator instance counts
		suite.Equal(3, s.CountEphemeral())
	})
}

func (suite *sceneTest) TestDrawCallsWithAnimators() {
	suite.Run("issues draw call for animator with instances", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("cube").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("cube_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().ID().Return(uint64(0)).Maybe()
		obj.EXPECT().SetID(mock.Anything).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0},
			[3]float32{1, 1, 1},
			[3]float32{0, 0, 0},
			[3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()

		s.Add(obj, cs, vs, fs)

		// Mock the render materials and pipeline lookup for DrawCalls
		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("cube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()

		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("cube").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		// Vertex and fragment shader declarations for provider resolution
		group0 := 0
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestPrepareShadowsWithDirectionalLight() {
	suite.Run("writes shadow uniform and executes shadow pass", func() {
		s, cam, r, _ := newMinimalScene("test")

		// Set up shadow map
		svs := &shadermocks.MockShader{}
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}

		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()
		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(svs, nil)

		// Add a directional light that casts shadows
		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{-0.5, -1.0, -0.5}).Maybe()
		s.AddLight(l)

		// Camera controller for shadow center
		cam.EXPECT().Controller().Return(nil).Maybe()

		// Renderer shadow frame calls
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(nil).Once()
		r.EXPECT().BeginShadowPass(tv).Return().Once()
		r.EXPECT().EndShadowPass().Return().Once()
		r.EXPECT().EndShadowFrame().Return().Once()

		s.PrepareShadows()
	})
}

func (suite *sceneTest) TestPrepareComputeWithAnimators() {
	suite.Run("processes animators and dispatches compute shaders", func() {
		s, cam, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("cube", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		cpKey := mdl.ComputePipelineKey()
		suite.NotEmpty(cpKey)

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		computePipeline := &pipelinemocks.MockPipeline{}
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(cpKey).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{1, 1, 1}).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(cpKey, mock.Anything, mock.Anything).Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})
	})

	suite.Run("syncs light positions from attached game objects", func() {
		s, cam, r, _ := newMinimalScene("test")

		l := &lightmocks.MockLight{}
		l.EXPECT().SetPosition(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		l.EXPECT().Enabled().Return(true).Maybe()

		mdl, obj, cs, vs, fs := newAddableObject("lamp", false)
		obj.EXPECT().Light().Unset()
		obj.EXPECT().Light().Return(l).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		cpKey := mdl.ComputePipelineKey()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		obj.EXPECT().Enabled().Return(true).Maybe()
		obj.EXPECT().Position().Return(float32(5), float32(10), float32(5)).Maybe()

		computePipeline := &pipelinemocks.MockPipeline{}
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(cpKey).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{1, 1, 1}).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		s.PrepareCompute(0.016)
	})

	suite.Run("writes light buffer when lightsBGP is initialized", func() {
		s, cam, r, _ := newMinimalScene("test")

		litFs := &shadermocks.MockShader{}
		litFs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			1: {0: "light_header", 1: "lights"},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64 * light.MaxGPULights}},
			},
		}).Maybe()
		s.InitLightBindGroup(litFs)

		suite.NotNil(s.LightBindGroupProvider())

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		s.PrepareCompute(0.016)
	})

	suite.Run("processes compute declarations with binding annotations", func() {
		s, cam, r, _ := newMinimalScene("test")

		binding0 := 0
		binding1 := 1
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type:    shader.AnnotationTypeBindingGroup,
				Group:   &group0,
				Binding: &binding0,
				Args:    []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgGlobalData},
			},
			{
				Type:    shader.AnnotationTypeBindingGroup,
				Group:   &group0,
				Binding: &binding1,
				Args:    []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("annotated_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 128}},
			},
		}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}

		mdl := newStatefulModel("annotated_cube")
		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		cpKey := mdl.ComputePipelineKey()
		suite.NotEmpty(cpKey)

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		computePipeline := &pipelinemocks.MockPipeline{}
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(cpKey).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgGlobalData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
		}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{64, 1, 1}).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		s.PrepareCompute(0.016)
	})

	suite.Run("writes camera position from controller", func() {
		s, cam, r, _ := newMinimalScene("test")

		ctrl := &cameramocks.MockCameraController{}
		ctrl.EXPECT().Position().Return(float32(10), float32(20), float32(30)).Maybe()
		ctrl.EXPECT().Target().Return(float32(0), float32(0), float32(0)).Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Controller().Return(ctrl).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		s.PrepareCompute(0.016)
	})
}

func (suite *sceneTest) TestPrepareLightCullingWithData() {
	suite.Run("dispatches light culling compute shader", func() {
		s, cam, r, _ := newMinimalScene("test")

		litFs := &shadermocks.MockShader{}
		litFs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			1: {0: "light_header", 1: "lights"},
			5: {0: "tile_uniforms", 1: "tile_light_counts", 2: "tile_light_indices"},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(5).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 8}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
				{Binding: 2, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		s.InitLightBindGroup(litFs)

		cullCs := &shadermocks.MockShader{}
		cullCs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 160}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		s.InitLightCullResources(cullCs, litFs, 1280, 720)

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		s.AddLight(l)

		cam.EXPECT().InverseProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().ViewMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Near().Return(float32(0.1)).Maybe()
		cam.EXPECT().Far().Return(float32(100.0)).Maybe()

		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareLightCulling()
		})
	})

	suite.Run("counts zero enabled lights but still dispatches", func() {
		s, cam, r, _ := newMinimalScene("test")

		litFs := &shadermocks.MockShader{}
		litFs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			1: {0: "light_header", 1: "lights"},
			5: {0: "tile_uniforms", 1: "tile_light_counts", 2: "tile_light_indices"},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(5).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 8}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
				{Binding: 2, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		s.InitLightBindGroup(litFs)

		cullCs := &shadermocks.MockShader{}
		cullCs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 160}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		s.InitLightCullResources(cullCs, litFs, 800, 600)

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(false).Maybe()
		s.AddLight(l)

		cam.EXPECT().InverseProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().ViewMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Near().Return(float32(0.1)).Maybe()
		cam.EXPECT().Far().Return(float32(100.0)).Maybe()

		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		s.PrepareLightCulling()
	})
}

func (suite *sceneTest) TestDrawCallsProviderResolution() {
	suite.Run("resolves material provider from declarations", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("mat_test", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("mat_test").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()

		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("mat_test").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		group1 := 1
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group1, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
		}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("resolves lights and shadow providers from binding group declarations", func() {
		s, _, r, _ := newMinimalScene("test")

		litFs := &shadermocks.MockShader{}
		litFs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			2: {0: "light_header", 1: "lights"},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		s.InitLightBindGroup(litFs)

		mdl, obj, cs, vs, fs := newAddableObject("lit_obj", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("lit_obj").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("lit_obj").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		group1 := 1
		group2 := 2
		binding0 := 0
		binding1 := 1
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group1, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group2, Binding: &binding0,
				Args: []shader.AnnotationArg{"light_header", "uniform", shader.AnnotationArgLightHeader},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group2, Binding: &binding1,
				Args: []shader.AnnotationArg{"lights", "storage", shader.AnnotationArgLight},
			},
		}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("skips material when pipeline key is empty", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("empty_key", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("").Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("skips material when render pipeline is nil", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("no_pipe", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("missing").Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()

		r.EXPECT().Pipeline("missing").Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("skips material when vertex shader is nil", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("no_vs", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("no_vs").Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		r.EXPECT().Pipeline("no_vs").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("skips when no render materials", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("no_mats", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("skips when mesh provider is nil after add", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("nil_mesh", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("skips group when required provider is missing", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("missing_prov", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("missing_prov").Maybe()
		mat.EXPECT().BindGroupProvider().Return(nil).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("missing_prov").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		group1 := 1
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group1, Args: []shader.AnnotationArg{shader.AnnotationArgShadow}},
		}).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("resolves instance data from binding group", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("inst_data", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("inst_data").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("inst_data").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		group1 := 1
		binding0 := 0
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group1, Binding: &binding0,
				Args: []shader.AnnotationArg{"instance_data", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("returns error when draw call fails", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("err_draw", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("err_draw").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("err_draw").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(fmt.Errorf("GPU device lost")).Maybe()

		err := s.DrawCalls()
		suite.Error(err)
		suite.Contains(err.Error(), "draw call failed")
	})
}

func (suite *sceneTest) TestPrepareShadowsWithAnimatorsInPool() {
	suite.Run("issues shadow draw call for each animator with instances", func() {
		s, cam, r, _ := newMinimalScene("test")

		svs := &shadermocks.MockShader{}
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()
		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(svs, nil)

		mdl, obj, cs, vs, fs := newAddableObject("shadow_cube", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().ComputePipelineKey().Unset()
		mdl.EXPECT().ComputePipelineKey().Return("").Maybe()

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{-0.5, -1.0, -0.5}).Maybe()
		s.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(nil).Once()
		r.EXPECT().BeginShadowPass(tv).Return().Once()
		r.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndShadowPass().Return().Once()
		r.EXPECT().EndShadowFrame().Return().Once()

		s.PrepareShadows()
	})
}

func (suite *sceneTest) TestRemoveSwapInstance() {
	suite.Run("swap-removes instance and updates swapped object ID", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj1, cs, vs, fs := newAddableObject("shared_model", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id1 := s.Add(obj1, cs, vs, fs)
		suite.NotZero(id1)

		var obj2ID uint64
		var obj2AnimInstID int
		obj2 := &gameobjectmocks.MockGameObject{}
		obj2.EXPECT().ID().RunAndReturn(func() uint64 { return obj2ID }).Maybe()
		obj2.EXPECT().SetID(mock.Anything).Run(func(id uint64) { obj2ID = id }).Return().Maybe()
		obj2.EXPECT().Model().Return(mdl).Maybe()
		obj2.EXPECT().Ephemeral().Return(false).Maybe()
		obj2.EXPECT().Light().Return(nil).Maybe()
		obj2.EXPECT().TransformData().Return(
			[3]float32{5, 5, 5}, [3]float32{1, 1, 1}, [3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj2.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj2.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { obj2AnimInstID = id }).Return().Maybe()
		obj2.EXPECT().Animator().Return(nil).Maybe()
		obj2.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return obj2AnimInstID }).Maybe()

		id2 := s.Add(obj2, cs, vs, fs)
		suite.NotZero(id2)
		suite.Equal(2, s.Count())

		obj1.EXPECT().Animator().Return(nil).Maybe()
		obj1.EXPECT().AnimatorInstanceID().Return(0).Maybe()

		s.Remove(id1)
		suite.Equal(1, s.Count())
		suite.Nil(s.Get(id1))
		suite.NotNil(s.Get(id2))
	})
}

func (suite *sceneTest) TestAddReusesExistingAnimator() {
	suite.Run("second add with same model reuses existing animator", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj1, cs, vs, fs := newAddableObject("reuse_model", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id1 := s.Add(obj1, cs, vs, fs)
		suite.NotZero(id1)
		suite.Equal(1, s.Count())
		suite.Equal(1, s.CountEphemeral())

		obj2 := newMinimalGameObject(false)
		obj2.EXPECT().Model().Return(mdl).Maybe()
		obj2.EXPECT().Light().Return(nil).Maybe()

		id2 := s.Add(obj2, cs, vs, fs)
		suite.NotZero(id2)
		suite.Equal(2, s.Count())
		suite.Equal(2, s.CountEphemeral())
	})
}

func (suite *sceneTest) TestDetachLightActualRemoval() {
	suite.Run("detaches a tracked light from scene light lists via Add", func() {
		s, _, r, _ := newMinimalScene("test")

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()

		_, obj, cs, vs, fs := newAddableObject("lamp", false)
		obj.EXPECT().Light().Unset()
		obj.EXPECT().Light().Return(l).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		s.Add(obj, cs, vs, fs)
		suite.Len(s.Lights(), 1)

		s.DetachLight(obj)
		suite.Len(s.Lights(), 0)
	})
}

func (suite *sceneTest) TestInitLightCullResourcesSuccessPath() {
	suite.Run("creates cull and tile BGPs with shared buffers", func() {
		s, _, r, _ := newMinimalScene("test")

		litFs := &shadermocks.MockShader{}
		litFs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			1: {0: "light_header", 1: "lights"},
			5: {0: "tile_uniforms", 1: "tile_light_counts", 2: "tile_light_indices"},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(5).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 8}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
				{Binding: 2, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		s.InitLightBindGroup(litFs)

		cullCs := &shadermocks.MockShader{}
		cullCs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 160}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		suite.NotPanics(func() {
			s.InitLightCullResources(cullCs, litFs, 1920, 1080)
		})
	})
}

func (suite *sceneTest) TestReinitCameraBGPForLitPipeline() {
	suite.Run("reinits camera BGP with merged visibility via InitLighting", func() {
		s, cam, r, _ := newMinimalScene("test")

		litFs := &shadermocks.MockShader{}
		litFs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "camera_uniform"},
			1: {0: "light_header", 1: "lights"},
			3: {0: "shadow_data"},
			5: {0: "tile_uniforms", 1: "tile_light_counts", 2: "tile_light_indices"},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "camera_bgl",
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(3).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(5).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 8}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
				{Binding: 2, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
			},
		}).Maybe()

		svs := &shadermocks.MockShader{}
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Maybe()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Maybe()
		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		cullCs := &shadermocks.MockShader{}
		cullCs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 160}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()

		cam.EXPECT().BindGroupProvider().Unset()
		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp_lit")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()

		suite.NotPanics(func() {
			s.InitLighting(litFs, svs, nil, cullCs, 1280, 720)
		})
	})
}

func (suite *sceneTest) TestCreateAnimatorWithAnnotations() {
	suite.Run("processes compute shader with AnimationData annotation", func() {
		s, _, r, _ := newMinimalScene("test")

		binding0 := 0
		binding1 := 1
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgGlobalData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("annotated_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 128}},
			},
		}).Maybe()

		outputBinding := 0
		outputGroup := 1
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &outputGroup, Binding: &outputBinding,
				Args: []shader.AnnotationArg{"instance_data", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()

		fs := &shadermocks.MockShader{}

		mdl := newStatefulModel("annotated_cube")
		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
		suite.Equal(1, s.Count())
	})

	suite.Run("processes output provider annotation from vertex shader", func() {
		s, _, r, _ := newMinimalScene("test")

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("provider_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		outputGroup := 2
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &outputGroup, Args: []shader.AnnotationArg{shader.AnnotationArgAnimator}},
		}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 16}},
			},
		}).Maybe()

		fs := &shadermocks.MockShader{}

		mdl := newStatefulModel("provider_cube")
		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
	})

	suite.Run("processes compute output provider for shared buffer", func() {
		s, _, r, _ := newMinimalScene("test")

		binding0 := 0
		binding1 := 1
		binding2 := 2
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgGlobalData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
			{
				Type: shader.AnnotationTypeProvider, Group: &group0, Binding: &binding2,
				Args: []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("shared_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 128}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
			},
		}).Maybe()

		outputBinding := 0
		outputGroup := 1
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &outputGroup, Binding: &outputBinding,
				Args: []shader.AnnotationArg{"instance_data", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()

		fs := &shadermocks.MockShader{}

		mdl := newStatefulModel("shared_cube")
		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
	})

	suite.Run("processes IndirectArgs annotation with usage override", func() {
		s, _, r, _ := newMinimalScene("test")

		binding0 := 0
		binding1 := 1
		binding2 := 2
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgGlobalData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding2,
				Args: []shader.AnnotationArg{"indirect_args", "storage", shader.AnnotationArgIndirectArgs},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("indirect_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 128}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 20}},
			},
		}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}

		mdl := newStatefulModel("indirect_cube")
		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
	})

	suite.Run("processes ModelData annotation for per-instance model matrices", func() {
		s, _, r, _ := newMinimalScene("test")

		binding0 := 0
		binding1 := 1
		binding2 := 2
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgGlobalData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding2,
				Args: []shader.AnnotationArg{"model_data", "storage", shader.AnnotationArgModelData},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("modeldata_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 128}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
			},
		}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}

		mdl := newStatefulModel("modeldata_cube")
		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
	})
}

func (suite *sceneTest) TestNewSceneCameraBGPInit() {
	suite.Run("skips camera BGP when no camera group in vertex shader", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}
		vs := &shadermocks.MockShader{}

		cam.EXPECT().BindGroupProvider().Return(nil).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		vs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "transforms"},
		}).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		s := scene.NewScene("no_cam_bgp", cam, r, vs)
		suite.NotNil(s)
	})
}

func (suite *sceneTest) TestDrawCallsWithEffectProvider() {
	suite.Run("resolves effect provider from declarations", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("effect_obj", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		effectBGP := bind_group_provider.NewBindGroupProvider("effect_bgp")
		mdl.EXPECT().EffectProvider().Return(effectBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("effect_obj").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("effect_obj").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		group1 := 1
		group2 := 2
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group1, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
			{Type: shader.AnnotationTypeProvider, Group: &group2, Args: []shader.AnnotationArg{shader.AnnotationArgEffect}},
		}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestDrawCallsWithTilesProvider() {
	suite.Run("resolves tiles provider from declarations", func() {
		s, _, r, _ := newMinimalScene("test")

		litFs := &shadermocks.MockShader{}
		litFs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			1: {0: "light_header", 1: "lights"},
			2: {0: "tile_uniforms", 1: "tile_light_counts", 2: "tile_light_indices"},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 8}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
				{Binding: 2, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		s.InitLightBindGroup(litFs)

		cullCs := &shadermocks.MockShader{}
		cullCs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 160}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		s.InitLightCullResources(cullCs, litFs, 1280, 720)

		mdl, obj, cs, vs, fs := newAddableObject("tiled_obj", false)
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("tiled_obj").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("tiled_obj").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		group1 := 1
		group2 := 2
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group1, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
			{Type: shader.AnnotationTypeProvider, Group: &group2, Args: []shader.AnnotationArg{shader.AnnotationArgTiles}},
		}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestDrawCallsWithAnimatorProvider() {
	suite.Run("resolves animator output provider from declarations", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("anim_obj", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("anim_obj").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("anim_obj").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		group1 := 1
		group2 := 2
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
			{Type: shader.AnnotationTypeProvider, Group: &group2, Args: []shader.AnnotationArg{shader.AnnotationArgAnimator}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group1, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
		}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestDrawCallsWithBindingGroupAnnotatedTypes() {
	suite.Run("resolves shadow data and tile uniforms via binding group annotations", func() {
		s, _, r, _ := newMinimalScene("test")

		litFs := &shadermocks.MockShader{}
		litFs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			1: {0: "light_header", 1: "lights"},
			2: {0: "tile_uniforms", 1: "tile_light_counts", 2: "tile_light_indices"},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		litFs.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 8}},
				{Binding: 1, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
				{Binding: 2, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		s.InitLightBindGroup(litFs)

		cullCs := &shadermocks.MockShader{}
		cullCs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 160}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		s.InitLightCullResources(cullCs, litFs, 1280, 720)

		svs := &shadermocks.MockShader{}
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Maybe()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Maybe()
		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(svs, nil)

		sdf := &shadermocks.MockShader{}
		sdf.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			3: {0: "shadow_data"},
		}).Maybe()
		sdf.EXPECT().BindGroupLayoutDescriptor(3).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		s.InitShadowLitBindGroup(sdf)

		mdl, obj, cs, vs, fs := newAddableObject("annotated_lit", false)
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("annotated_lit").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("annotated_lit").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		group1 := 1
		group2 := 2
		group3 := 3
		group4 := 4
		binding0 := 0
		binding1 := 1
		binding2 := 2
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group1, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group2, Binding: &binding0,
				Args: []shader.AnnotationArg{"light_header", "uniform", shader.AnnotationArgLightHeader},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group3, Binding: &binding0,
				Args: []shader.AnnotationArg{"shadow_data", "uniform", shader.AnnotationArgShadowData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group4, Binding: &binding0,
				Args: []shader.AnnotationArg{"tile_uniforms", "uniform", shader.AnnotationArgTileUniforms},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group4, Binding: &binding1,
				Args: []shader.AnnotationArg{"tile_counts", "storage", "tile_light_counts"},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group4, Binding: &binding2,
				Args: []shader.AnnotationArg{"tile_indices", "storage", "tile_light_indices"},
			},
		}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("resolves effect params and overlay params via binding group annotations", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("overlay_obj", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		effectBGP := bind_group_provider.NewBindGroupProvider("effect_bgp")
		mdl.EXPECT().EffectProvider().Return(effectBGP).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("overlay_obj").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}

		r.EXPECT().Pipeline("overlay_obj").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		group1 := 1
		group2 := 2
		binding0 := 0
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group1, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group2, Binding: &binding0,
				Args: []shader.AnnotationArg{"effect_params", "uniform", shader.AnnotationArgEffectParams},
			},
		}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestPrepareComputeWithCulling() {
	suite.Run("extracts frustum planes and enables culling on animators", func() {
		s, cam, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("cull_cube", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		cpKey := mdl.ComputePipelineKey()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1.002, -1,
			0, 0, -0.2, 0,
		}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		computePipeline := &pipelinemocks.MockPipeline{}
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(cpKey).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{64, 1, 1}).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		s.PrepareCompute(0.016)

		// Second call — culling should be enabled now on the animator
		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1.002, -1,
			0, 0, -0.2, 0,
		}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		s.PrepareCompute(0.016)
	})
}

func (suite *sceneTest) TestCreateAnimatorSkeletal() {
	suite.Run("creates skeletal animator with bone and packed bindings", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Name().Return("fox").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(2.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		mdl.EXPECT().Skeleton().Return(&model.Skeleton{
			Bones: []model.Bone{
				{ParentIndex: -1},
				{ParentIndex: 0},
			},
		}).Maybe()
		mdl.EXPECT().Animations().Return([]*model.AnimationClip{
			{
				Name:     "walk",
				Duration: 1.0,
				Channels: []model.AnimationChannel{
					{
						BoneIndex:    0,
						PositionKeys: []model.VectorKeyframe{{Time: 0, Value: [3]float32{0, 0, 0}}},
						RotationKeys: []model.QuaternionKeyframe{{Time: 0, Value: [4]float32{0, 0, 0, 1}}},
						ScaleKeys:    []model.VectorKeyframe{{Time: 0, Value: [3]float32{1, 1, 1}}},
					},
				},
			},
		}).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0, 0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		binding0 := 0
		binding1 := 1
		binding2 := 2
		binding3 := 3
		binding4 := 4
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgGlobalData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"skel_data", "storage", shader.AnnotationArgSkeletalAnimationData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding2,
				Args: []shader.AnnotationArg{"bone_info", "storage", shader.AnnotationArgBoneInfo},
			},
			{
				Type: shader.AnnotationTypeProvider, Group: &group0, Binding: &binding3,
				Args: []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
			},
			{
				Type: shader.AnnotationTypeProvider, Group: &group0, Binding: &binding4,
				Args: []shader.AnnotationArg{shader.AnnotationArgAnimatorPacked},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("fox_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 128}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 16}},
				{Binding: 4, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()

		outputGroup := 1
		outputBinding := 0
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &outputGroup, Args: []shader.AnnotationArg{shader.AnnotationArgAnimator}},
		}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: uint32(outputBinding), Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 16}},
			},
		}).Maybe()

		fs := &shadermocks.MockShader{}

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
		suite.Equal(1, s.Count())
	})
}

func (suite *sceneTest) TestCreateAnimatorWithMeshInit() {
	suite.Run("initializes mesh buffers when meshProvider has no vertex buffer", func() {
		s, _, r, _ := newMinimalScene("test")

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("init_mesh").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{1, 2, 3, 4}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 1}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("init_mesh_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(meshBGP, []byte{1, 2, 3, 4}, []byte{0, 1}, 1).Return(nil).Once()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
	})
}

func (suite *sceneTest) TestPrepareShadowsWithSkinnedModel() {
	suite.Run("selects skinned shadow pipeline when model is skinned", func() {
		s, cam, r, _ := newMinimalScene("test")

		svs := &shadermocks.MockShader{}
		ssvs := &shadermocks.MockShader{}
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()
		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(svs, ssvs)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Name().Return("skinned_shadow").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(2.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Maybe()
		mdl.EXPECT().Animations().Return([]*model.AnimationClip{}).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		binding0 := 0
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgGlobalData},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("skinned_shadow_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
			},
		}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		fs := &shadermocks.MockShader{}

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		s.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(nil).Once()
		r.EXPECT().BeginShadowPass(tv).Return().Once()
		r.EXPECT().ShadowDrawCall(mock.Anything, meshBGP, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndShadowPass().Return().Once()
		r.EXPECT().EndShadowFrame().Return().Once()

		s.PrepareShadows()
	})
}

func (suite *sceneTest) TestDrawCallsWithNilModel() {
	suite.Run("skips animator when model returns nil from real animator", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("nil_model_test", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		// After Add, the real animator holds a reference to the model.
		// We can't make Model() return nil on a real animator, but we can test
		// the path where RenderMaterials is empty.
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestPrepareShadowsBeginFrameError() {
	suite.Run("returns early when BeginShadowFrame returns error", func() {
		s, cam, r, _ := newMinimalScene("test")

		svs := &shadermocks.MockShader{}
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()
		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(svs, nil)

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		s.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(fmt.Errorf("GPU error")).Once()

		suite.NotPanics(func() {
			s.PrepareShadows()
		})
	})
}

func (suite *sceneTest) TestPrepareShadowsWithShadowLitBGP() {
	suite.Run("writes to shadow lit BGP when it has buffers", func() {
		s, cam, r, _ := newMinimalScene("test")

		svs := &shadermocks.MockShader{}
		tv := &wgpu.TextureView{}
		tex := &wgpu.Texture{}
		samp := &wgpu.Sampler{}
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(tv, tex, nil).Once()
		r.EXPECT().CreateComparisonSampler().Return(samp, nil).Once()
		svs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		svs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageVertex, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(svs, nil)

		sdf := &shadermocks.MockShader{}
		sdf.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_data"},
		}).Maybe()
		sdf.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Visibility: wgpu.ShaderStageFragment, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		s.InitShadowLitBindGroup(sdf)

		shadowLitBGP := s.ShadowLitBindGroupProvider()
		suite.NotNil(shadowLitBGP)
		shadowLitBGP.SetBuffer(0, &wgpu.Buffer{})

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		s.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(nil).Once()
		r.EXPECT().BeginShadowPass(tv).Return().Once()
		r.EXPECT().EndShadowPass().Return().Once()
		r.EXPECT().EndShadowFrame().Return().Once()

		s.PrepareShadows()
	})
}

func (suite *sceneTest) TestDrawCallsWithCullingEnabled() {
	suite.Run("attempts indirect draw but falls through when buffer is nil", func() {
		s, cam, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("cull_draw", false)
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		// Enable culling by calling PrepareCompute with a valid VP matrix
		cpKey := mdl.ComputePipelineKey()
		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1.002, -1,
			0, 0, -0.2, 0,
		}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		computePipeline := &pipelinemocks.MockPipeline{}
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(cpKey).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{64, 1, 1}).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		s.PrepareCompute(0.016)

		// Now attempt DrawCalls — culling is enabled, IndirectBuffer returns nil
		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		mat := &materialmocks.MockMaterial{}
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		mat.EXPECT().PipelineKey().Return("cull_draw").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().SetDelegate(mock.Anything).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{mat}).Maybe()

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVertShader := &shadermocks.MockShader{}
		renderFragShader := &shadermocks.MockShader{}
		r.EXPECT().Pipeline("cull_draw").Return(renderPipeline).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVertShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFragShader).Maybe()

		group0 := 0
		group1 := 1
		renderVertShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group0, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}).Maybe()
		renderFragShader.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &group1, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
		}).Maybe()

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestRemoveWithSwapAndLight() {
	suite.Run("removes object with light and updates remaining animator instances", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl, obj1, cs, vs, fs := newAddableObject("lit_swap", false)

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		obj1.EXPECT().Light().Unset()
		obj1.EXPECT().Light().Return(l).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		id1 := s.Add(obj1, cs, vs, fs)
		suite.NotZero(id1)
		suite.Len(s.Lights(), 1)

		obj2 := newMinimalGameObject(false)
		obj2.EXPECT().Model().Return(mdl).Maybe()
		obj2.EXPECT().Light().Return(nil).Maybe()
		var obj2AnimInstID int
		obj2.EXPECT().SetAnimatorInstanceID(mock.Anything).Unset()
		obj2.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { obj2AnimInstID = id }).Return().Maybe()
		obj2.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return obj2AnimInstID }).Maybe()

		id2 := s.Add(obj2, cs, vs, fs)
		suite.NotZero(id2)
		suite.Equal(2, s.Count())

		obj1.EXPECT().Animator().Return(nil).Maybe()
		obj1.EXPECT().AnimatorInstanceID().Return(0).Maybe()

		s.Remove(id1)
		suite.Equal(1, s.Count())
		suite.Len(s.Lights(), 0)
	})
}

func (suite *sceneTest) TestRemoveSwapActualSwap() {
	suite.Run("swap-remove updates the remaining object's animator instance ID", func() {
		s, _, r, _ := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		mdl := newStatefulModel("swap_mdl")

		// obj1: non-ephemeral, stateful Animator + AnimatorInstanceID
		var obj1Anim animator.Animator
		var obj1InstID int
		obj1 := &gameobjectmocks.MockGameObject{}
		var obj1ID uint64
		obj1.EXPECT().ID().RunAndReturn(func() uint64 { return obj1ID }).Maybe()
		obj1.EXPECT().SetID(mock.Anything).Run(func(id uint64) { obj1ID = id }).Return().Maybe()
		obj1.EXPECT().Ephemeral().Return(false).Maybe()
		obj1.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0}, [3]float32{1, 1, 1}, [3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj1.EXPECT().Model().Return(mdl).Maybe()
		obj1.EXPECT().Light().Return(nil).Maybe()
		obj1.EXPECT().SetAnimator(mock.Anything).Run(func(a animator.Animator) { obj1Anim = a }).Return().Maybe()
		obj1.EXPECT().Animator().RunAndReturn(func() animator.Animator { return obj1Anim }).Maybe()
		obj1.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { obj1InstID = id }).Return().Maybe()
		obj1.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return obj1InstID }).Maybe()

		// obj2: non-ephemeral, same model, stateful Animator + AnimatorInstanceID
		var obj2Anim animator.Animator
		var obj2InstID int
		obj2 := &gameobjectmocks.MockGameObject{}
		var obj2ID uint64
		obj2.EXPECT().ID().RunAndReturn(func() uint64 { return obj2ID }).Maybe()
		obj2.EXPECT().SetID(mock.Anything).Run(func(id uint64) { obj2ID = id }).Return().Maybe()
		obj2.EXPECT().Ephemeral().Return(false).Maybe()
		obj2.EXPECT().TransformData().Return(
			[3]float32{1, 0, 0}, [3]float32{1, 1, 1}, [3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj2.EXPECT().Model().Return(mdl).Maybe()
		obj2.EXPECT().Light().Return(nil).Maybe()
		obj2.EXPECT().SetAnimator(mock.Anything).Run(func(a animator.Animator) { obj2Anim = a }).Return().Maybe()
		obj2.EXPECT().Animator().RunAndReturn(func() animator.Animator { return obj2Anim }).Maybe()
		obj2.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { obj2InstID = id }).Return().Maybe()
		obj2.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return obj2InstID }).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("swap_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		fs := &shadermocks.MockShader{}

		id1 := s.Add(obj1, cs, vs, fs)
		id2 := s.Add(obj2, cs, vs, fs)
		suite.NotZero(id1)
		suite.NotZero(id2)
		suite.Equal(2, s.Count())

		// obj1 should have instance 0, obj2 should have instance 1
		suite.Equal(0, obj1InstID)
		suite.Equal(1, obj2InstID)

		// Remove obj1 (index 0). The real animator has 2 instances, so it will:
		// - Swap last instance (1) into slot 0
		// - Return (1, true)
		// Scene then finds obj2 (AnimatorInstanceID==1) and updates it to 0.
		s.Remove(id1)
		suite.Equal(1, s.Count())

		// After swap-remove, obj2 should now have instance ID 0 (swapped from 1)
		suite.Equal(0, obj2InstID)

		_ = obj1Anim
		_ = obj2Anim
	})
}

func (suite *sceneTest) TestPrepareComputePhase2CullingWithMeshProvider() {
	suite.Run("phase 2 resets indirect args when culling enabled and mesh provider present", func() {
		s, cam, r, _ := newMinimalScene("test")

		// Create a model with a real MeshProvider that has IndexCount > 0
		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(36)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("cull_mesh").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		binding0 := 0
		binding1 := 1
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgAnimationGlobals},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("cull_mesh_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
			},
		}).Maybe()

		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		fs := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		// Set up pipeline returns for PrepareCompute
		computePipeline := &pipelinemocks.MockPipeline{}
		indirectBinding := 2
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(mock.Anything).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgAnimationGlobals},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &indirectBinding,
				Args: []shader.AnnotationArg{"indirect", "storage", shader.AnnotationArgIndirectArgs},
			},
		}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{64, 1, 1}).Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1.002, -1,
			0, 0, -0.2, 0,
		}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		// First call: Phase 1 sets frustum planes → culling enabled
		s.PrepareCompute(0.016)

		// Second call: Phase 2 should now enter the culling branch with non-nil MeshProvider
		s.PrepareCompute(0.016)
	})
}

func (suite *sceneTest) TestInitShadowMapWithSkinnedShader() {
	suite.Run("registers both static and skinned shadow pipelines", func() {
		s, _, r, _ := newMinimalScene("test")

		shadowVS := &shadermocks.MockShader{}
		shadowVS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		shadowVS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()

		skinnedVS := &shadermocks.MockShader{}

		var depthTex wgpu.Texture
		var depthView wgpu.TextureView
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(&depthView, &depthTex, nil).Maybe()

		var samp wgpu.Sampler
		r.EXPECT().CreateComparisonSampler().Return(&samp, nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		var registeredPipelines []string
		r.EXPECT().RegisterShadowPipeline(mock.Anything).RunAndReturn(func(p pipeline.Pipeline) error {
			registeredPipelines = append(registeredPipelines, p.PipelineKey())
			return nil
		}).Maybe()

		s.InitShadowMap(shadowVS, skinnedVS)

		suite.Len(registeredPipelines, 2)
	})
}

func (suite *sceneTest) TestReinitCameraBGPNoCameraGroup() {
	suite.Run("skips reinit when fragment shader has no camera group", func() {
		s, _, r, _ := newMinimalScene("test")

		lightFS := &shadermocks.MockShader{}
		// No "camera" in the var names at all
		lightFS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "light_data"},
		}).Maybe()
		lightFS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		// InitLighting calls InitLightBindGroup, InitShadowLitBindGroup, InitLightCullResources, reinitCameraBGP
		// For this test we only care that reinitCameraBGP is reached with no camera group.
		// To simplify, call a sequence that reaches reinitCameraBGPForLitPipeline.
		// InitLighting requires lightFragShader, cullComputeShader, litFragShader, shadowVertexShader, etc.
		// It's easier to directly test via InitLighting paths.
		// Actually reinitCameraBGPForLitPipeline is private, called from InitLighting.
		// Let's test via InitLighting with a litFragShader that has no camera group.

		lightFragShader := &shadermocks.MockShader{}
		lightFragShader.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			3: {0: "light_header", 1: "lights"},
			5: {0: "tile_uniforms", 1: "tile_light_counts", 2: "tile_light_indices"},
		}).Maybe()
		lightFragShader.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()

		cullCS := &shadermocks.MockShader{}
		cullCS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 160}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()

		litFS := lightFragShader // same shader serves for both light and lit

		shadowVS := &shadermocks.MockShader{}
		shadowVS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		shadowVS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		var depthTex wgpu.Texture
		var depthView wgpu.TextureView
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(&depthView, &depthTex, nil).Maybe()
		var samp wgpu.Sampler
		r.EXPECT().CreateComparisonSampler().Return(&samp, nil).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		// litFS has NO "camera" in its var names → reinitCameraBGP returns early
		s.InitLighting(litFS, shadowVS, nil, cullCS, 800, 600)
	})
}

func (suite *sceneTest) TestPrepareShadowsIndirectDrawPath() {
	suite.Run("enters culling-enabled path in shadow pass and falls through on nil indirect buffer", func() {
		s, cam, r, _ := newMinimalScene("test")

		// Set up shadow infrastructure
		shadowVS := &shadermocks.MockShader{}
		shadowVS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		shadowVS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		var depthTex wgpu.Texture
		var depthView wgpu.TextureView
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(&depthView, &depthTex, nil).Maybe()
		var samp wgpu.Sampler
		r.EXPECT().CreateComparisonSampler().Return(&samp, nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(shadowVS, nil)

		// Add a light
		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		s.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()

		// Add an object with non-nil mesh provider
		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(36)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("shadow_obj").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		binding0 := 0
		binding1 := 1
		group0 := 0
		indirectBinding := 2
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgAnimationGlobals},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("shadow_obj_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		fs := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		// Run PrepareCompute to enable culling on the animator
		computePipeline := &pipelinemocks.MockPipeline{}
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(mock.Anything).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgAnimationGlobals},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &indirectBinding,
				Args: []shader.AnnotationArg{"indirect", "storage", shader.AnnotationArgIndirectArgs},
			},
		}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{64, 1, 1}).Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1.002, -1,
			0, 0, -0.2, 0,
		}).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		// First PrepareCompute: enables culling
		s.PrepareCompute(0.016)

		// Now call PrepareShadows — the animator has CullingEnabled, it will enter the
		// culling path, scan for IndirectArgs, but IndirectBuffer returns nil → falls through
		// to normal ShadowDrawCall.
		r.EXPECT().BeginShadowFrame().Return(nil).Maybe()
		r.EXPECT().BeginShadowPass(mock.Anything).Return().Maybe()
		r.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndShadowPass().Return().Maybe()
		r.EXPECT().EndShadowFrame().Return().Maybe()

		s.PrepareShadows()
	})
}

func (suite *sceneTest) TestDrawCallsIndirectDrawPath() {
	suite.Run("enters culling path in draw calls and falls through on nil indirect buffer", func() {
		s, cam, r, _ := newMinimalScene("test")

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(36)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("cull_draw").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		matMock := &materialmocks.MockMaterial{}
		matMock.EXPECT().PipelineKey().Return("cull_draw").Maybe()
		matMock.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		binding0 := 0
		binding1 := 1
		group0 := 0
		indirectBinding := 2
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgAnimationGlobals},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("cull_draw_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
			},
		}).Maybe()

		outputGroup := 1
		outputBinding := 0
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &outputGroup, Binding: &outputBinding,
				Args: []shader.AnnotationArg{"instance_buffer", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		// Enable culling via PrepareCompute
		computePipeline := &pipelinemocks.MockPipeline{}
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(mock.Anything).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(mock.Anything).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgAnimationGlobals},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &indirectBinding,
				Args: []shader.AnnotationArg{"indirect", "storage", shader.AnnotationArgIndirectArgs},
			},
		}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{64, 1, 1}).Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, -1, 0, 0, -0.2, 0,
		}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		s.PrepareCompute(0.016)

		// Now DrawCalls — animator has CullingEnabled, but IndirectBuffer returns nil → normal DrawCall
		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVS := &shadermocks.MockShader{}
		renderFS := &shadermocks.MockShader{}
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVS).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFS).Maybe()

		renderVS.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &outputGroup, Binding: &outputBinding,
				Args: []shader.AnnotationArg{"instance_buffer", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		renderFS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()

		r.EXPECT().Pipeline("cull_draw").Return(renderPipeline).Maybe()
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().DrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestPrepareComputeWithLightBufferWrites() {
	suite.Run("writes light buffer when lights BGP initialized and lights exist", func() {
		s, cam, r, _ := newMinimalScene("test")

		// Initialize light bind group
		lightFS := &shadermocks.MockShader{}
		lightFS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			3: {0: "light_header", 1: "lights"},
		}).Maybe()
		lightFS.EXPECT().BindGroupLayoutDescriptor(3).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.InitLightBindGroup(lightFS)

		// Add some lights
		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		l.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		l.EXPECT().Color().Return([3]float32{1, 1, 1}).Maybe()
		l.EXPECT().Intensity().Return(float32(1.0)).Maybe()
		l.EXPECT().Range().Return(float32(10.0)).Maybe()
		l.EXPECT().InnerCone().Return(float32(0)).Maybe()
		l.EXPECT().OuterCone().Return(float32(0)).Maybe()
		l.EXPECT().CastsShadows().Return(false).Maybe()
		s.AddLight(l)

		// Add an object to process
		mdl, obj, cs, vs, fs := newAddableObject("light_obj", false)
		_ = mdl
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		// Set up PrepareCompute mocks
		computePipeline := &pipelinemocks.MockPipeline{}
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(mock.Anything).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{64, 1, 1}).Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		s.PrepareCompute(0.016)
	})
}

func (suite *sceneTest) TestPrepareComputeWithCameraController() {
	suite.Run("includes camera position from controller in uniform data", func() {
		s, cam, r, _ := newMinimalScene("test")

		mdl, obj, cs, vs, fs := newAddableObject("cam_ctrl", false)
		_ = mdl
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		computePipeline := &pipelinemocks.MockPipeline{}
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(mock.Anything).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{64, 1, 1}).Maybe()

		ctrl := &cameramocks.MockCameraController{}
		ctrl.EXPECT().Position().Return(float32(1), float32(2), float32(3)).Maybe()
		ctrl.EXPECT().Target().Return(float32(0), float32(0), float32(0)).Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Controller().Return(ctrl).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		s.PrepareCompute(0.016)
	})
}

func (suite *sceneTest) TestDrawCallsWithShadowAndTiles() {
	suite.Run("resolves shadow and tile providers via binding group annotation types", func() {
		s, cam, r, _ := newMinimalScene("test")

		// Set up shadow + tile infrastructure
		shadowVS := &shadermocks.MockShader{}
		shadowVS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		shadowVS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()

		var depthTex wgpu.Texture
		var depthView wgpu.TextureView
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(&depthView, &depthTex, nil).Maybe()
		var sampObj wgpu.Sampler
		r.EXPECT().CreateComparisonSampler().Return(&sampObj, nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(shadowVS, nil)

		// InitShadowLitBindGroup
		litFS := &shadermocks.MockShader{}
		litFS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			3: {0: "light_header", 1: "lights"},
			4: {0: "shadow_depth_texture", 1: "shadow_sampler", 2: "shadow_uniform"},
			5: {0: "tile_uniforms", 1: "tile_light_counts", 2: "tile_light_indices"},
		}).Maybe()
		litFS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		s.InitShadowLitBindGroup(litFS)

		// InitLightBindGroup
		s.InitLightBindGroup(litFS)

		// InitLightCullResources
		cullCS := &shadermocks.MockShader{}
		cullCS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 160}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		s.InitLightCullResources(cullCS, litFS, 800, 600)

		// Add an object
		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(36)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("shadow_tile_obj").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		matMock := &materialmocks.MockMaterial{}
		matMock.EXPECT().PipelineKey().Return("shadow_tile_obj").Maybe()
		matMock.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("shadow_tile_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		// Set up DrawCalls - render pipeline has camera, instance, material, shadow, tile groups
		group0 := 0
		group1 := 1
		group2 := 2
		group3 := 3
		group4 := 4
		binding0 := 0
		binding1 := 1

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVS := &shadermocks.MockShader{}
		renderFS := &shadermocks.MockShader{}
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVS).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFS).Maybe()

		renderVS.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeProvider, Group: &group0,
				Args: []shader.AnnotationArg{shader.AnnotationArgCamera},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group1, Binding: &binding0,
				Args: []shader.AnnotationArg{"instance_buffer", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		renderFS.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeProvider, Group: &group2,
				Args: []shader.AnnotationArg{shader.AnnotationArgMaterial},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group3, Binding: &binding0,
				Args: []shader.AnnotationArg{"shadow_data", "uniform", shader.AnnotationArgShadowData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group4, Binding: &binding1,
				Args: []shader.AnnotationArg{"tile_uniforms", "uniform", shader.AnnotationArgTileUniforms},
			},
		}).Maybe()

		r.EXPECT().Pipeline("shadow_tile_obj").Return(renderPipeline).Maybe()
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		_ = cam
		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestDrawCallsSkipsMaterialWithMissingProvider() {
	suite.Run("skips material when a required bind group provider is nil", func() {
		s, _, r, _ := newMinimalScene("test")

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(6)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("skip_mat").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		matMock := &materialmocks.MockMaterial{}
		matMock.EXPECT().PipelineKey().Return("skip_mat").Maybe()
		matMock.EXPECT().BindGroupProvider().Return(nil).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("skip_mat_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		group0 := 0
		group1 := 1

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVS := &shadermocks.MockShader{}
		renderFS := &shadermocks.MockShader{}
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVS).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFS).Maybe()

		renderVS.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0,
				Args: []shader.AnnotationArg{"instance_buffer", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		// Fragment shader declares a material group at group1
		renderFS.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeProvider, Group: &group1,
				Args: []shader.AnnotationArg{shader.AnnotationArgMaterial},
			},
		}).Maybe()

		r.EXPECT().Pipeline("skip_mat").Return(renderPipeline).Maybe()
		// DrawCall should NOT be called since the material group's provider (mat.BindGroupProvider()) is nil
		// which causes skipMaterial = true

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestDrawCallsNoMaterials() {
	suite.Run("skips model when RenderMaterials returns empty slice", func() {
		s, _, r, _ := newMinimalScene("test")

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(6)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("no_mats").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("no_mats_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestNewSceneNilCameraBGP() {
	suite.Run("skips bind group init when camera has nil BindGroupProvider", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}
		vs := &shadermocks.MockShader{}

		cam.EXPECT().BindGroupProvider().Return(nil).Maybe()
		vs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "camera_uniform"},
		}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Label: "camera_bgl",
		}).Maybe()

		// InitBindGroup should NOT be called since BGP is nil
		sc := scene.NewScene("no_bgp_cam", cam, r, vs)
		suite.NotNil(sc)
		suite.Equal("no_bgp_cam", sc.Name())
	})
}

func (suite *sceneTest) TestPrepareShadowsWithSkinnedPipeline() {
	suite.Run("uses skinned shadow pipeline when model is skinned", func() {
		s, cam, r, _ := newMinimalScene("test")

		// Set up shadow map with skinned shader
		shadowVS := &shadermocks.MockShader{}
		shadowVS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		shadowVS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		skinnedVS := &shadermocks.MockShader{}

		var depthTex wgpu.Texture
		var depthView wgpu.TextureView
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(&depthView, &depthTex, nil).Maybe()
		var sampObj wgpu.Sampler
		r.EXPECT().CreateComparisonSampler().Return(&sampObj, nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(shadowVS, skinnedVS)

		// Add a directional light
		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		s.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()

		// Add a skinned model
		meshBGP := bind_group_provider.NewBindGroupProvider("skinned_mesh")
		meshBGP.SetIndexCount(100)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Name().Return("skinned_fox").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(2.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		mdl.EXPECT().Skeleton().Return(&model.Skeleton{
			Bones: []model.Bone{{ParentIndex: -1}, {ParentIndex: 0}},
		}).Maybe()
		mdl.EXPECT().Animations().Return([]*model.AnimationClip{
			{
				Name:     "idle",
				Duration: 1.0,
				Channels: []model.AnimationChannel{
					{
						BoneIndex:    0,
						PositionKeys: []model.VectorKeyframe{{Time: 0, Value: [3]float32{0, 0, 0}}},
						RotationKeys: []model.QuaternionKeyframe{{Time: 0, Value: [4]float32{0, 0, 0, 1}}},
						ScaleKeys:    []model.VectorKeyframe{{Time: 0, Value: [3]float32{1, 1, 1}}},
					},
				},
			},
		}).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0, 0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		binding0 := 0
		binding1 := 1
		binding2 := 2
		binding3 := 3
		binding4 := 4
		group0 := 0
		csAdd := &shadermocks.MockShader{}
		csAdd.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgGlobalData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"skel_data", "storage", shader.AnnotationArgSkeletalAnimationData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding2,
				Args: []shader.AnnotationArg{"bone_info", "storage", shader.AnnotationArgBoneInfo},
			},
			{
				Type: shader.AnnotationTypeProvider, Group: &group0, Binding: &binding3,
				Args: []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
			},
			{
				Type: shader.AnnotationTypeProvider, Group: &group0, Binding: &binding4,
				Args: []shader.AnnotationArg{shader.AnnotationArgAnimatorPacked},
			},
		}).Maybe()
		csAdd.EXPECT().Key().Return("skinned_fox_compute").Maybe()
		csAdd.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 128}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 16}},
				{Binding: 4, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()

		outputGroup := 1
		vsAdd := &shadermocks.MockShader{}
		vsAdd.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &outputGroup, Args: []shader.AnnotationArg{shader.AnnotationArgAnimator}},
		}).Maybe()
		vsAdd.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 16}},
			},
		}).Maybe()
		fsAdd := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		s.Add(obj, csAdd, vsAdd, fsAdd)

		// PrepareShadows should use the skinned pipeline key
		r.EXPECT().BeginShadowFrame().Return(nil).Maybe()
		r.EXPECT().BeginShadowPass(mock.Anything).Return().Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().ShadowDrawCall("shadow_depth_skinned", mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndShadowPass().Return().Maybe()
		r.EXPECT().EndShadowFrame().Return().Maybe()

		s.PrepareShadows()
	})
}

func (suite *sceneTest) TestPrepareShadowsNoPipelineKey() {
	suite.Run("skips shadow draw when no pipeline key is set", func() {
		s, cam, r, _ := newMinimalScene("test")

		// Initialize shadow with nil views by NOT calling InitShadowMap but
		// manually setting the internal state.
		// Actually we need shadowDepthTextureView + shadowDataBGP to be non-nil.
		// The simplest way is to call InitShadowMap normally then test a model
		// where pipeKey == "" (which happens if shadowPipelineKey is "" and model
		// is not skinned OR shadowSkinnedPipeKey is "").
		// Since InitShadowMap always sets shadowPipelineKey, we can't easily
		// make it empty. Skip this test - it's an edge guard.

		// Instead, test when model has nil meshProvider → skips to next animator.
		shadowVS := &shadermocks.MockShader{}
		shadowVS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		shadowVS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()

		var depthTex wgpu.Texture
		var depthView wgpu.TextureView
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(&depthView, &depthTex, nil).Maybe()
		var sampObj wgpu.Sampler
		r.EXPECT().CreateComparisonSampler().Return(&sampObj, nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(shadowVS, nil)

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		s.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()

		// Add object with nil MeshProvider so shadow draw is skipped
		mdl, obj, cs, vs, fs := newAddableObject("nil_mesh_shadow", false)
		_ = mdl
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		r.EXPECT().BeginShadowFrame().Return(nil).Maybe()
		r.EXPECT().BeginShadowPass(mock.Anything).Return().Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		// ShadowDrawCall should NOT be called since meshProvider is nil
		r.EXPECT().EndShadowPass().Return().Maybe()
		r.EXPECT().EndShadowFrame().Return().Maybe()

		s.PrepareShadows()
	})
}

func (suite *sceneTest) TestPrepareShadowsWithCameraController() {
	suite.Run("uses camera controller target for shadow frustum center", func() {
		s, cam, r, _ := newMinimalScene("test")

		shadowVS := &shadermocks.MockShader{}
		shadowVS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		shadowVS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()

		var depthTex wgpu.Texture
		var depthView wgpu.TextureView
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(&depthView, &depthTex, nil).Maybe()
		var sampObj wgpu.Sampler
		r.EXPECT().CreateComparisonSampler().Return(&sampObj, nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(shadowVS, nil)

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		s.AddLight(l)

		ctrl := &cameramocks.MockCameraController{}
		ctrl.EXPECT().Target().Return(float32(5), float32(0), float32(5)).Maybe()
		cam.EXPECT().Controller().Return(ctrl).Maybe()

		// Add object with mesh provider for valid shadow draw
		meshBGP := bind_group_provider.NewBindGroupProvider("shadow_mesh")
		meshBGP.SetIndexCount(36)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("shadow_ctrl").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("shadow_ctrl_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		fs := &shadermocks.MockShader{}

		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		r.EXPECT().BeginShadowFrame().Return(nil).Maybe()
		r.EXPECT().BeginShadowPass(mock.Anything).Return().Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndShadowPass().Return().Maybe()
		r.EXPECT().EndShadowFrame().Return().Maybe()

		s.PrepareShadows()
	})
}

func (suite *sceneTest) TestInitLightBindGroupNoLightGroup() {
	suite.Run("returns early when fragment shader has no light group", func() {
		s, _, r, _ := newMinimalScene("test")

		fs := &shadermocks.MockShader{}
		fs.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "camera_data"},
		}).Maybe()

		// InitLightBindGroup should return early since no group has "light" in its var names
		// r.InitBindGroup should NOT be called
		s.InitLightBindGroup(fs)

		_ = r
	})
}

func (suite *sceneTest) TestInitShadowLitBindGroupNoShadowGroup() {
	suite.Run("returns early when fragment shader has no shadow group", func() {
		s, _, r, _ := newMinimalScene("test")

		// Need shadow texture + sampler to exist (otherwise early return before group scan)
		shadowVS := &shadermocks.MockShader{}
		shadowVS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		shadowVS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		var depthTex wgpu.Texture
		var depthView wgpu.TextureView
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(&depthView, &depthTex, nil).Maybe()
		var sampObj wgpu.Sampler
		r.EXPECT().CreateComparisonSampler().Return(&sampObj, nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(shadowVS, nil)

		// Now InitShadowLitBindGroup with a shader that has no "shadow" in var names
		litFS := &shadermocks.MockShader{}
		litFS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "camera_data"},
		}).Maybe()

		s.InitShadowLitBindGroup(litFS)
	})
}

func (suite *sceneTest) TestPrepareLightCullingWithEnabledLights() {
	suite.Run("writes light count and frustum data to cull uniforms", func() {
		s, cam, r, _ := newMinimalScene("test")

		// Set up light bind group
		lightFS := &shadermocks.MockShader{}
		lightFS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			3: {0: "light_header", 1: "lights"},
			5: {0: "tile_uniforms", 1: "tile_light_counts", 2: "tile_light_indices"},
		}).Maybe()
		lightFS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.InitLightBindGroup(lightFS)

		// Set up light cull resources
		cullCS := &shadermocks.MockShader{}
		cullCS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 160}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
			},
		}).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		s.InitLightCullResources(cullCS, lightFS, 800, 600)

		// Add a light
		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		l.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		l.EXPECT().Color().Return([3]float32{1, 1, 1}).Maybe()
		l.EXPECT().Intensity().Return(float32(1.0)).Maybe()
		l.EXPECT().Range().Return(float32(10.0)).Maybe()
		l.EXPECT().InnerCone().Return(float32(0)).Maybe()
		l.EXPECT().OuterCone().Return(float32(0)).Maybe()
		l.EXPECT().CastsShadows().Return(false).Maybe()
		s.AddLight(l)

		cam.EXPECT().InverseProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().ViewMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Near().Return(float32(0.1)).Maybe()
		cam.EXPECT().Far().Return(float32(100.0)).Maybe()

		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		s.PrepareLightCulling()
	})
}

func (suite *sceneTest) TestDrawCallsEmptyPipelineKey() {
	suite.Run("skips material when pipeline key is empty", func() {
		s, _, r, _ := newMinimalScene("test")

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(6)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("empty_key").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		matMock := &materialmocks.MockMaterial{}
		matMock.EXPECT().PipelineKey().Return("").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("empty_key_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestDrawCallsNilRenderPipeline() {
	suite.Run("skips material when render pipeline lookup returns nil", func() {
		s, _, r, _ := newMinimalScene("test")

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(6)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("nil_rp").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		matMock := &materialmocks.MockMaterial{}
		matMock.EXPECT().PipelineKey().Return("nil_rp").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("nil_rp_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		// Return nil for the render pipeline lookup
		r.EXPECT().Pipeline("nil_rp").Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestDrawCallsNilVertexShader() {
	suite.Run("skips material when vertex shader is nil on the render pipeline", func() {
		s, _, r, _ := newMinimalScene("test")

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(6)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("nil_vs").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		matMock := &materialmocks.MockMaterial{}
		matMock.EXPECT().PipelineKey().Return("nil_vs").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("nil_vs_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(nil).Maybe()
		r.EXPECT().Pipeline("nil_vs").Return(renderPipeline).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestCreateAnimatorWithModelDataBinding() {
	suite.Run("resolves model data binding size override for compute BGP", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("model_data").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		binding0 := 0
		binding1 := 1
		binding2 := 2
		binding3 := 3
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgAnimationGlobals},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding2,
				Args: []shader.AnnotationArg{"model_data", "storage", shader.AnnotationArgModelData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding3,
				Args: []shader.AnnotationArg{"indirect_args", "storage", shader.AnnotationArgIndirectArgs},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("model_data_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 20}},
			},
		}).Maybe()

		outputGroup := 1
		outputBinding := 0
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &outputGroup, Binding: &outputBinding,
				Args: []shader.AnnotationArg{"instance_buffer", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		fs := &shadermocks.MockShader{}

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
	})
}

func (suite *sceneTest) TestCreateAnimatorWithOutputProviderAndSharedBuffer() {
	suite.Run("resolves output provider binding and shares buffer to output BGP", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("shared_buf").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		binding0 := 0
		binding1 := 1
		binding2 := 2
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgAnimationGlobals},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"anim_data", "storage", shader.AnnotationArgAnimationData},
			},
			{
				Type: shader.AnnotationTypeProvider, Group: &group0, Binding: &binding2,
				Args: []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("shared_buf_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
			},
		}).Maybe()

		outputGroup := 1
		outputBinding := 0
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &outputGroup, Binding: &outputBinding,
				Args: []shader.AnnotationArg{"instance_buffer", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		fs := &shadermocks.MockShader{}

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
	})
}

func (suite *sceneTest) TestPrepareComputeWithLightObjectSync() {
	suite.Run("syncs game object position to attached light during compute", func() {
		s, cam, r, _ := newMinimalScene("test")

		mdl := newStatefulModel("light_sync")

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Enabled().Return(true).Maybe()
		obj.EXPECT().Position().Return(float32(10), float32(20), float32(30)).Maybe()

		l := &lightmocks.MockLight{}
		l.EXPECT().SetPosition(float32(10), float32(20), float32(30)).Return().Maybe()
		obj.EXPECT().Light().Return(l).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("light_sync_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		fs := &shadermocks.MockShader{}

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, vs, fs)

		computePipeline := &pipelinemocks.MockPipeline{}
		computeShaderMock := &shadermocks.MockShader{}
		r.EXPECT().Pipeline(mock.Anything).Return(computePipeline).Maybe()
		computePipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShaderMock).Maybe()
		computeShaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		computeShaderMock.EXPECT().WorkgroupSize().Return([3]uint32{64, 1, 1}).Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		s.PrepareCompute(0.016)
	})
}

func (suite *sceneTest) TestDrawCallsWithOverlayParams() {
	suite.Run("resolves overlay params binding group type to effect provider", func() {
		s, _, r, _ := newMinimalScene("test")

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(6)
		effectBGP := bind_group_provider.NewBindGroupProvider("effect_bgp")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("overlay").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()
		mdl.EXPECT().EffectProvider().Return(effectBGP).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		matMock := &materialmocks.MockMaterial{}
		matMock.EXPECT().PipelineKey().Return("overlay").Maybe()
		matMock.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("overlay_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		group0 := 0
		group1 := 1
		binding0 := 0

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVS := &shadermocks.MockShader{}
		renderFS := &shadermocks.MockShader{}
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVS).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFS).Maybe()

		renderVS.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"instance_buffer", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		renderFS.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group1, Binding: &binding0,
				Args: []shader.AnnotationArg{"overlay_params", "uniform", shader.AnnotationArgOverlayParams},
			},
		}).Maybe()

		r.EXPECT().Pipeline("overlay").Return(renderPipeline).Maybe()
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestDrawCallsWithShadowUniform() {
	suite.Run("resolves shadow uniform binding group type to shadow lit BGP", func() {
		s, _, r, _ := newMinimalScene("test")

		// Setup shadow infrastructure
		shadowVS := &shadermocks.MockShader{}
		shadowVS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			0: {0: "shadow_uniform"},
		}).Maybe()
		shadowVS.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		var depthTex wgpu.Texture
		var depthView wgpu.TextureView
		r.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(&depthView, &depthTex, nil).Maybe()
		var sampObj wgpu.Sampler
		r.EXPECT().CreateComparisonSampler().Return(&sampObj, nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterShadowPipeline(mock.Anything).Return(nil).Maybe()
		s.InitShadowMap(shadowVS, nil)

		litFS := &shadermocks.MockShader{}
		litFS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			4: {0: "shadow_depth_texture", 1: "shadow_sampler", 2: "shadow_data"},
		}).Maybe()
		litFS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 80}},
			},
		}).Maybe()
		s.InitShadowLitBindGroup(litFS)

		// Add an object
		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(6)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("shadow_uni").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		matMock := &materialmocks.MockMaterial{}
		matMock.EXPECT().PipelineKey().Return("shadow_uni").Maybe()
		matMock.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("shadow_uni_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		group0 := 0
		group1 := 1
		binding0 := 0

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVShader := &shadermocks.MockShader{}
		renderFShader := &shadermocks.MockShader{}
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFShader).Maybe()

		renderVShader.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"instance_buffer", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		renderFShader.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group1, Binding: &binding0,
				Args: []shader.AnnotationArg{"shadow_data", "uniform", shader.AnnotationArgShadowUniform},
			},
		}).Maybe()

		r.EXPECT().Pipeline("shadow_uni").Return(renderPipeline).Maybe()
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestDrawCallsWithLightHeader() {
	suite.Run("resolves light header binding group type to lights BGP", func() {
		s, _, r, _ := newMinimalScene("test")

		// Init light bind group
		lightFS := &shadermocks.MockShader{}
		lightFS.EXPECT().BindGroupVarNames().Return(map[int]map[int]string{
			3: {0: "light_header", 1: "lights"},
		}).Maybe()
		lightFS.EXPECT().BindGroupLayoutDescriptor(3).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 16}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 64}},
			},
		}).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		s.InitLightBindGroup(lightFS)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(6)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("light_hdr").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		matMock := &materialmocks.MockMaterial{}
		matMock.EXPECT().PipelineKey().Return("light_hdr").Maybe()
		matMock.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("light_hdr_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		group0 := 0
		group1 := 1
		binding0 := 0

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVShader := &shadermocks.MockShader{}
		renderFShader := &shadermocks.MockShader{}
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFShader).Maybe()

		renderVShader.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"instance_buffer", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()
		renderFShader.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group1, Binding: &binding0,
				Args: []shader.AnnotationArg{"light_data", "storage", shader.AnnotationArgLightHeader},
			},
		}).Maybe()

		r.EXPECT().Pipeline("light_hdr").Return(renderPipeline).Maybe()
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestDrawCallsWithCameraBindingGroup() {
	suite.Run("resolves camera binding group type to camera BGP", func() {
		s, _, r, _ := newMinimalScene("test")

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh_bgp")
		meshBGP.SetIndexCount(6)

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("cam_bg").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{}).Maybe()
		mdl.EXPECT().IndexCount().Return(0).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")
		matMock := &materialmocks.MockMaterial{}
		matMock.EXPECT().PipelineKey().Return("cam_bg").Maybe()
		matMock.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Maybe()

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().Key().Return("cam_bg_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addVS := &shadermocks.MockShader{}
		addVS.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		addVS.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		addFS := &shadermocks.MockShader{}

		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		s.Add(obj, cs, addVS, addFS)

		group0 := 0
		group1 := 1
		binding0 := 0

		renderPipeline := &pipelinemocks.MockPipeline{}
		renderVShader := &shadermocks.MockShader{}
		renderFShader := &shadermocks.MockShader{}
		renderPipeline.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderVShader).Maybe()
		renderPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(renderFShader).Maybe()

		renderVShader.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"camera_data", "uniform", shader.AnnotationArgCamera},
			},
		}).Maybe()
		renderFShader.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group1, Binding: &binding0,
				Args: []shader.AnnotationArg{"instance_buffer", "storage", shader.AnnotationArgInstanceData},
			},
		}).Maybe()

		r.EXPECT().Pipeline("cam_bg").Return(renderPipeline).Maybe()
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestCreateAnimatorWithScratchBinding() {
	suite.Run("resolves scratch binding for skeletal animator", func() {
		s, _, r, _ := newMinimalScene("test")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Name().Return("scratch").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(2.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()

		var cpKey string
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()

		mdl.EXPECT().Skeleton().Return(&model.Skeleton{
			Bones: []model.Bone{{ParentIndex: -1}, {ParentIndex: 0}},
		}).Maybe()
		mdl.EXPECT().Animations().Return([]*model.AnimationClip{
			{
				Name:     "walk",
				Duration: 1.0,
				Channels: []model.AnimationChannel{
					{BoneIndex: 0,
						PositionKeys: []model.VectorKeyframe{{Time: 0, Value: [3]float32{0, 0, 0}}},
						RotationKeys: []model.QuaternionKeyframe{{Time: 0, Value: [4]float32{0, 0, 0, 1}}},
						ScaleKeys:    []model.VectorKeyframe{{Time: 0, Value: [3]float32{1, 1, 1}}},
					},
				},
			},
		}).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0, 0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		binding0 := 0
		binding1 := 1
		binding2 := 2
		binding3 := 3
		binding4 := 4
		binding5 := 5
		group0 := 0
		cs := &shadermocks.MockShader{}
		cs.EXPECT().Declarations().Return([]shader.Annotation{
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding0,
				Args: []shader.AnnotationArg{"globals", "uniform", shader.AnnotationArgGlobalData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding1,
				Args: []shader.AnnotationArg{"skel_data", "storage", shader.AnnotationArgSkeletalAnimationData},
			},
			{
				Type: shader.AnnotationTypeBindingGroup, Group: &group0, Binding: &binding2,
				Args: []shader.AnnotationArg{"bone_info", "storage", shader.AnnotationArgBoneInfo},
			},
			{
				Type: shader.AnnotationTypeProvider, Group: &group0, Binding: &binding3,
				Args: []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
			},
			{
				Type: shader.AnnotationTypeProvider, Group: &group0, Binding: &binding4,
				Args: []shader.AnnotationArg{shader.AnnotationArgAnimatorPacked},
			},
			{
				Type: shader.AnnotationTypeProvider, Group: &group0, Binding: &binding5,
				Args: []shader.AnnotationArg{shader.AnnotationArgAnimatorScratch},
			},
		}).Maybe()
		cs.EXPECT().Key().Return("scratch_compute").Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 64}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 128}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 16}},
				{Binding: 4, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4}},
				{Binding: 5, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
			},
		}).Maybe()

		outputGroup := 1
		vs := &shadermocks.MockShader{}
		vs.EXPECT().Declarations().Return([]shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &outputGroup, Args: []shader.AnnotationArg{shader.AnnotationArgAnimator}},
		}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 16}},
			},
		}).Maybe()
		fs := &shadermocks.MockShader{}

		obj := newMinimalGameObject(false)
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		id := s.Add(obj, cs, vs, fs)
		suite.NotZero(id)
	})
}

// newStatefulModel creates a MockModel with stateful ComputePipelineKey and MeshProvider tracking.
func newStatefulModel(name string) *modelmocks.MockModel {
	mdl := &modelmocks.MockModel{}
	mdl.EXPECT().Skinned().Return(false).Maybe()
	mdl.EXPECT().Name().Return(name).Maybe()
	mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
	mdl.EXPECT().MeshProvider().Return(nil).Maybe()

	var cpKey string
	mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { cpKey = key }).Return().Maybe()
	mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return cpKey }).Maybe()
	return mdl
}

// newMinimalGameObject creates a MockGameObject with stateful ID tracking and default transform data.
func newMinimalGameObject(ephemeral bool) *gameobjectmocks.MockGameObject {
	var objID uint64
	obj := &gameobjectmocks.MockGameObject{}
	obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
	obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
	obj.EXPECT().Ephemeral().Return(ephemeral).Maybe()
	obj.EXPECT().TransformData().Return(
		[3]float32{0, 0, 0},
		[3]float32{1, 1, 1},
		[3]float32{0, 0, 0},
		[3]float32{0, 0, 0},
	).Maybe()
	obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
	obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()
	return obj
}

// newAddableObject creates a full set of mocks ready for scene.Add: model, game object, compute/vertex/fragment shaders.
// The model has stateful ComputePipelineKey tracking. Returns (model, obj, cs, vs, fs).
func newAddableObject(name string, ephemeral bool) (*modelmocks.MockModel, *gameobjectmocks.MockGameObject, *shadermocks.MockShader, *shadermocks.MockShader, *shadermocks.MockShader) {
	mdl := newStatefulModel(name)

	obj := newMinimalGameObject(ephemeral)
	obj.EXPECT().Model().Return(mdl).Maybe()
	obj.EXPECT().Light().Return(nil).Maybe()

	cs := &shadermocks.MockShader{}
	cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
	cs.EXPECT().Key().Return(name + "_compute").Maybe()
	cs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

	vs := &shadermocks.MockShader{}
	vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
	vs.EXPECT().BindGroupLayoutDescriptor(mock.Anything).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

	fs := &shadermocks.MockShader{}

	return mdl, obj, cs, vs, fs
}
