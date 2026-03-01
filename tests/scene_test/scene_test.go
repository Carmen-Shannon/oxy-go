package scene_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/cogentcore/webgpu/wgpu"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"

	cameramocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/camera"
	gameobjectmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/game_object"
	lightmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/light"
	materialmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/material"
	modelmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/model"
	physicsmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/physics"
	renderermocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/renderer"
)

type sceneTest struct {
	suite.Suite
	origDir string
}

func TestScene(t *testing.T) {
	suite.Run(t, new(sceneTest))
}

func (suite *sceneTest) SetupSuite() {
	dir, err := os.Getwd()
	suite.Require().NoError(err)
	suite.origDir = dir

	// Change to the project root so that relative asset paths used by the engine
	// (e.g. "engine/model/assets/simple-vert.wgsl") resolve correctly.
	suite.Require().NoError(os.Chdir("../.."))
}

func (suite *sceneTest) TearDownSuite() {
	_ = os.Chdir(suite.origDir)
}

// newMinimalScene creates a Scene with the minimum required mocks. It sets up
// a camera with a non-nil BindGroupProvider and a renderer that succeeds on
// InitBindGroup. Returns (scene, cameraMock, rendererMock).
func newMinimalScene(name string, opts ...scene.SceneBuilderOption) (scene.Scene, *cameramocks.MockCamera, *renderermocks.MockRenderer) {
	cam := &cameramocks.MockCamera{}
	r := &renderermocks.MockRenderer{}

	bgp := bind_group_provider.NewBindGroupProvider("cam_bgp")

	cam.EXPECT().BindGroupProvider().Return(bgp).Maybe()
	cam.EXPECT().SetDelegate(mock.Anything).Maybe()
	r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

	s := scene.NewScene(name, cam, r, opts...)
	return s, cam, r
}

// newLitScene creates a Scene with a pre-enabled LightingHandler injected via
// WithLighting. This bypasses the initLighting GPU init path, allowing light
// list manipulation methods to be tested without shader file loading.
func newLitScene(name string, opts ...scene.SceneBuilderOption) (scene.Scene, *cameramocks.MockCamera, *renderermocks.MockRenderer) {
	handler := light.NewLightingHandler()
	handler.SetEnabled(true)
	allOpts := append([]scene.SceneBuilderOption{scene.WithLighting(handler)}, opts...)
	return newMinimalScene(name, allOpts...)
}

// newAddableObject creates a mock GameObject with a mock Model suitable for
// passing to scene.Add(). The model is non-skinned with no materials and no
// mesh provider. The renderer mock must be configured to accept InitBindGroup
// and RegisterPipelines calls (newMinimalScene already does this).
func newAddableObject(mdlName string, ephemeral bool) (*gameobjectmocks.MockGameObject, *modelmocks.MockModel) {
	mdl := &modelmocks.MockModel{}
	mdl.EXPECT().Skinned().Return(false).Maybe()
	mdl.EXPECT().Name().Return(mdlName).Maybe()
	mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
	mdl.EXPECT().MeshProvider().Return(nil).Maybe()
	mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
	mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()

	obj := &gameobjectmocks.MockGameObject{}
	var objID uint64
	obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
	obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
	obj.EXPECT().Model().Return(mdl).Maybe()
	obj.EXPECT().Ephemeral().Return(ephemeral).Maybe()
	obj.EXPECT().Light().Return(nil).Maybe()
	obj.EXPECT().RigidBody().Return(nil).Maybe()
	obj.EXPECT().TransformData().Return(
		[3]float32{0, 0, 0},
		[3]float32{1, 1, 1},
		[3]float32{0, 0, 0},
		[3]float32{0, 0, 0},
	).Maybe()
	obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
	obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()

	return obj, mdl
}

// newWiredObject creates a mock pair with pipeline key tracking for PrepareCompute and
// DrawCalls testing. Unlike newAddableObject, it wires ComputePipelineKey to properly track
// the key set by the scene during Add. Accepts optional materials for the model.
//
// Parameters:
//   - mdlName: the name to give the model mock
//   - ephemeral: whether the game object should be ephemeral
//   - mats: materials the model should return from RenderMaterials
func newWiredObject(mdlName string, ephemeral bool, mats []material.Material) (*gameobjectmocks.MockGameObject, *modelmocks.MockModel) {
	mdl := &modelmocks.MockModel{}
	mdl.EXPECT().Skinned().Return(false).Maybe()
	mdl.EXPECT().Name().Return(mdlName).Maybe()
	mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
	mdl.EXPECT().MeshProvider().Return(nil).Maybe()
	mdl.EXPECT().RenderMaterials().Return(mats).Maybe()
	mdl.EXPECT().EffectProvider().Return(nil).Maybe()

	var computeKey string
	mdl.EXPECT().SetComputePipelineKey(mock.Anything).Run(func(key string) { computeKey = key }).Return().Maybe()
	mdl.EXPECT().ComputePipelineKey().RunAndReturn(func() string { return computeKey }).Maybe()

	obj := &gameobjectmocks.MockGameObject{}
	var objID uint64
	obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
	obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
	obj.EXPECT().Model().Return(mdl).Maybe()
	obj.EXPECT().Ephemeral().Return(ephemeral).Maybe()
	obj.EXPECT().Light().Return(nil).Maybe()
	obj.EXPECT().RigidBody().Return(nil).Maybe()
	obj.EXPECT().TransformData().Return(
		[3]float32{0, 0, 0},
		[3]float32{1, 1, 1},
		[3]float32{0, 0, 0},
		[3]float32{0, 0, 0},
	).Maybe()
	obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
	obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()

	return obj, mdl
}

// capturePipelines configures the renderer mock to capture pipelines registered via
// RegisterPipelines and serve them from Pipeline(key) calls. Returns the capture map.
func capturePipelines(r *renderermocks.MockRenderer) map[string]pipeline.Pipeline {
	captured := make(map[string]pipeline.Pipeline)
	r.EXPECT().RegisterPipelines(mock.Anything).RunAndReturn(func(pipelines ...pipeline.Pipeline) error {
		for _, p := range pipelines {
			captured[p.PipelineKey()] = p
		}
		return nil
	}).Maybe()
	r.EXPECT().Pipeline(mock.Anything).RunAndReturn(func(key string) pipeline.Pipeline {
		return captured[key]
	}).Maybe()
	return captured
}

// wirePipelineLookup configures the renderer mock to return captured pipelines from
// Pipeline(key) calls using the provided capture map.
func wirePipelineLookup(r *renderermocks.MockRenderer, captured map[string]pipeline.Pipeline) {
	r.EXPECT().Pipeline(mock.Anything).RunAndReturn(func(key string) pipeline.Pipeline {
		return captured[key]
	}).Maybe()
}

func (suite *sceneTest) TestNewScene() {
	suite.Run("panics when camera is nil", func() {
		r := &renderermocks.MockRenderer{}
		suite.Panics(func() {
			scene.NewScene("test", nil, r)
		})
	})

	suite.Run("panics when renderer is nil", func() {
		cam := &cameramocks.MockCamera{}
		suite.Panics(func() {
			scene.NewScene("test", cam, nil)
		})
	})

	suite.Run("returns non-nil scene with valid args", func() {
		s, _, _ := newMinimalScene("test_scene")
		suite.NotNil(s)
	})

	suite.Run("name is set correctly", func() {
		s, _, _ := newMinimalScene("my_scene")
		suite.Equal("my_scene", s.Name())
	})

	suite.Run("default active is false", func() {
		s, _, _ := newMinimalScene("test")
		suite.False(s.Active())
	})

	suite.Run("default culling disabled is false", func() {
		s, _, _ := newMinimalScene("test")
		suite.False(s.CullingDisabled())
	})

	suite.Run("default count is zero", func() {
		s, _, _ := newMinimalScene("test")
		suite.Equal(0, s.Count())
	})

	suite.Run("default ephemeral count is zero", func() {
		s, _, _ := newMinimalScene("test")
		suite.Equal(0, s.CountEphemeral())
	})

	suite.Run("camera is stored", func() {
		s, cam, _ := newMinimalScene("test")
		suite.Equal(cam, s.Camera())
	})

	suite.Run("renderer is stored", func() {
		s, _, r := newMinimalScene("test")
		suite.Equal(r, s.Renderer())
	})

	suite.Run("camera with nil BGP does not panic", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		cam.EXPECT().BindGroupProvider().Return(nil).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		s := scene.NewScene("test", cam, r)
		suite.NotNil(s)
	})

	suite.Run("panics when camera bind group init fails", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		cam.EXPECT().BindGroupProvider().Return(bind_group_provider.NewBindGroupProvider("cam")).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(fmt.Errorf("GPU init failed")).Maybe()

		suite.Panics(func() {
			scene.NewScene("test", cam, r)
		})
	})

	suite.Run("default lights is empty", func() {
		s, _, _ := newMinimalScene("test")
		suite.Empty(s.Lights())
	})

	suite.Run("default ambient color is zero", func() {
		s, _, _ := newMinimalScene("test")
		suite.Equal([3]float32{0, 0, 0}, s.AmbientColor())
	})
}

func (suite *sceneTest) TestSetName() {
	suite.Run("sets and retrieves new name", func() {
		s, _, _ := newMinimalScene("original")
		s.SetName("updated")
		suite.Equal("updated", s.Name())
	})
}

func (suite *sceneTest) TestSetActive() {
	suite.Run("sets active to true", func() {
		s, _, _ := newMinimalScene("test")
		s.SetActive(true)
		suite.True(s.Active())
	})

	suite.Run("sets active back to false", func() {
		s, _, _ := newMinimalScene("test")
		s.SetActive(true)
		s.SetActive(false)
		suite.False(s.Active())
	})
}

func (suite *sceneTest) TestSetCamera() {
	suite.Run("replaces camera", func() {
		s, _, _ := newMinimalScene("test")
		newCam := &cameramocks.MockCamera{}
		s.SetCamera(newCam)
		suite.Equal(newCam, s.Camera())
	})
}

func (suite *sceneTest) TestSetRenderer() {
	suite.Run("replaces renderer", func() {
		s, _, _ := newMinimalScene("test")
		newR := &renderermocks.MockRenderer{}
		s.SetRenderer(newR)
		suite.Equal(newR, s.Renderer())
	})
}

func (suite *sceneTest) TestSetPhysicsHandler() {
	suite.Run("sets and replaces physics handler", func() {
		s, _, _ := newMinimalScene("test")
		ph := &physicsmocks.MockPhysics{}
		s.SetPhysicsHandler(ph)
		// No getter on Scene interface — just verify no panic
		suite.NotNil(s)
	})
}

func (suite *sceneTest) TestSetCullingDisabled() {
	suite.Run("sets culling disabled to true", func() {
		s, _, _ := newMinimalScene("test")
		s.SetCullingDisabled(true)
		suite.True(s.CullingDisabled())
	})

	suite.Run("sets culling disabled back to false", func() {
		s, _, _ := newMinimalScene("test")
		s.SetCullingDisabled(true)
		s.SetCullingDisabled(false)
		suite.False(s.CullingDisabled())
	})
}

func (suite *sceneTest) TestCount() {
	suite.Run("returns zero for empty scene", func() {
		s, _, _ := newMinimalScene("test")
		suite.Equal(0, s.Count())
	})
}

func (suite *sceneTest) TestCountEphemeral() {
	suite.Run("returns zero for empty scene", func() {
		s, _, _ := newMinimalScene("test")
		suite.Equal(0, s.CountEphemeral())
	})
}

func (suite *sceneTest) TestAdd() {
	suite.Run("panics when renderer is nil", func() {
		s, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		obj, _ := newAddableObject("cube", false)
		suite.Panics(func() {
			s.Add(obj)
		})
	})

	suite.Run("panics when object has no model", func() {
		s, _, _ := newMinimalScene("test")
		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().Model().Return(nil).Maybe()
		suite.Panics(func() {
			s.Add(obj)
		})
	})

	suite.Run("adds a non-ephemeral object and returns assigned ID", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		obj, _ := newAddableObject("cube", false)

		id := s.Add(obj)
		suite.NotZero(id)
		suite.Equal(1, s.Count())
		suite.NotNil(s.Get(id))
	})

	suite.Run("adds an ephemeral object without persisting in registry", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		obj, _ := newAddableObject("particle", true)

		id := s.Add(obj)
		suite.NotZero(id)
		suite.Equal(0, s.Count())
		suite.Nil(s.Get(id))
	})

	suite.Run("increments ephemeral count after adding objects", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		obj1, _ := newAddableObject("cube", false)
		obj2, _ := newAddableObject("cube", false)
		s.Add(obj1)
		s.Add(obj2)
		suite.Equal(2, s.CountEphemeral())
	})

	suite.Run("object with attached light is tracked", func() {
		s, _, r := newLitScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		l := &lightmocks.MockLight{}
		obj, _ := newAddableObject("lamp", false)
		obj.EXPECT().Light().Unset()
		obj.EXPECT().Light().Return(l).Maybe()

		s.Add(obj)
		suite.Len(s.Lights(), 1)
	})

	suite.Run("reuses existing animator for same model", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("shared").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()

		for i := 0; i < 3; i++ {
			obj := &gameobjectmocks.MockGameObject{}
			var objID uint64
			obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
			obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
			obj.EXPECT().Model().Return(mdl).Maybe()
			obj.EXPECT().Ephemeral().Return(false).Maybe()
			obj.EXPECT().Light().Return(nil).Maybe()
			obj.EXPECT().RigidBody().Return(nil).Maybe()
			obj.EXPECT().TransformData().Return(
				[3]float32{float32(i), 0, 0},
				[3]float32{1, 1, 1},
				[3]float32{0, 0, 0},
				[3]float32{0, 0, 0},
			).Maybe()
			obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
			obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()
			s.Add(obj)
		}

		suite.Equal(3, s.Count())
		suite.Equal(3, s.CountEphemeral())
	})

	suite.Run("registers material GPU resources when material BGP is nil", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterMaterial(mock.Anything, mock.Anything).Return(nil).Maybe()

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().BindGroupProvider().Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, _ := newWiredObject("matcube", false, []material.Material{mat})
		id := s.Add(obj)
		suite.NotZero(id)
	})

	suite.Run("uses lit shaders when lighting is enabled", func() {
		s, _, r := newLitScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		obj, _ := newWiredObject("litcube", false, []material.Material{})
		id := s.Add(obj)
		suite.NotZero(id)
		suite.Equal(1, s.CountEphemeral())
	})

	suite.Run("resolves skinned model shaders when model is skinned", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Name().Return("fox").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(2.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Maybe()
		mdl.EXPECT().Animations().Return(nil).Maybe()

		obj := &gameobjectmocks.MockGameObject{}
		var objID uint64
		obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().RigidBody().Return(nil).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0}, [3]float32{1, 1, 1},
			[3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()

		id := s.Add(obj)
		suite.NotZero(id)
		suite.Equal(1, s.Count())
	})

	suite.Run("creates skeletal animator with bone and animation data", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		skeleton := &model.Skeleton{
			Bones: []model.Bone{
				{Name: "root", ParentIndex: -1},
				{Name: "spine", ParentIndex: 0},
			},
			RootBoneIndices: []int32{0},
			BoneNameToIndex: map[string]int32{"root": 0, "spine": 1},
		}
		animations := []*model.AnimationClip{
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
		}

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Name().Return("skel_fox").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(2.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().Return("").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()
		mdl.EXPECT().Skeleton().Return(skeleton).Maybe()
		mdl.EXPECT().Animations().Return(animations).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		obj := &gameobjectmocks.MockGameObject{}
		var objID uint64
		obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().RigidBody().Return(nil).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0}, [3]float32{1, 1, 1},
			[3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()

		id := s.Add(obj)
		suite.NotZero(id)
		suite.Equal(1, s.Count())
	})

	suite.Run("uses custom fragment shader path from material", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().BindGroupProvider().Return(bind_group_provider.NewBindGroupProvider("mat")).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("engine/model/assets/textured-frag.wgsl").Maybe()

		obj, _ := newWiredObject("custom_frag", false, []material.Material{mat})
		id := s.Add(obj)
		suite.NotZero(id)
	})

	suite.Run("initializes mesh GPU buffers when vertex buffer is nil", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")

		obj, mdl := newWiredObject("mesh_cube", false, []material.Material{})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 1, 2, 3}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 1}).Maybe()
		mdl.EXPECT().IndexCount().Return(3).Maybe()

		id := s.Add(obj)
		suite.NotZero(id)
		r.AssertCalled(suite.T(), "InitMeshBuffers", mock.Anything, mock.Anything, mock.Anything, mock.Anything)
	})
}

func (suite *sceneTest) TestGet() {
	suite.Run("returns nil for non-existent ID", func() {
		s, _, _ := newMinimalScene("test")
		suite.Nil(s.Get(999))
	})

	suite.Run("returns object after adding", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		obj, _ := newAddableObject("cube", false)
		id := s.Add(obj)
		suite.NotNil(s.Get(id))
	})
}

func (suite *sceneTest) TestRemove() {
	suite.Run("no-op for non-existent ID", func() {
		s, _, _ := newMinimalScene("test")
		s.Remove(999)
		suite.Equal(0, s.Count())
	})

	suite.Run("removes added object from registry", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		obj, _ := newAddableObject("cube", false)
		obj.EXPECT().Animator().Return(nil).Maybe()
		obj.EXPECT().AnimatorInstanceID().Return(-1).Maybe()

		id := s.Add(obj)
		suite.Equal(1, s.Count())

		s.Remove(id)
		suite.Equal(0, s.Count())
		suite.Nil(s.Get(id))
	})

	suite.Run("removes attached light when object is removed", func() {
		s, _, r := newLitScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		l := &lightmocks.MockLight{}
		obj, _ := newAddableObject("lamp", false)
		obj.EXPECT().Light().Unset()
		obj.EXPECT().Light().Return(l).Maybe()
		obj.EXPECT().Animator().Return(nil).Maybe()
		obj.EXPECT().AnimatorInstanceID().Return(-1).Maybe()

		id := s.Add(obj)
		suite.Len(s.Lights(), 1)

		s.Remove(id)
		suite.Empty(s.Lights())
	})

	suite.Run("swap-remove updates swapped object after first object removed", func() {
		s, _, r := newMinimalScene("test")
		captured := capturePipelines(r)
		_ = captured

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("shared").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()

		obj1 := &gameobjectmocks.MockGameObject{}
		var obj1ID uint64
		var capturedAnim animator.Animator
		var inst1 int
		obj1.EXPECT().ID().RunAndReturn(func() uint64 { return obj1ID }).Maybe()
		obj1.EXPECT().SetID(mock.Anything).Run(func(id uint64) { obj1ID = id }).Return().Maybe()
		obj1.EXPECT().Model().Return(mdl).Maybe()
		obj1.EXPECT().Ephemeral().Return(false).Maybe()
		obj1.EXPECT().Light().Return(nil).Maybe()
		obj1.EXPECT().RigidBody().Return(nil).Maybe()
		obj1.EXPECT().TransformData().Return([3]float32{0, 0, 0}, [3]float32{1, 1, 1}, [3]float32{0, 0, 0}, [3]float32{0, 0, 0}).Maybe()
		obj1.EXPECT().SetAnimator(mock.Anything).Run(func(a animator.Animator) { capturedAnim = a }).Return().Maybe()
		obj1.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { inst1 = id }).Return().Maybe()
		obj1.EXPECT().Animator().RunAndReturn(func() animator.Animator { return capturedAnim }).Maybe()
		obj1.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return inst1 }).Maybe()

		obj2 := &gameobjectmocks.MockGameObject{}
		var obj2ID uint64
		var inst2 int
		obj2.EXPECT().ID().RunAndReturn(func() uint64 { return obj2ID }).Maybe()
		obj2.EXPECT().SetID(mock.Anything).Run(func(id uint64) { obj2ID = id }).Return().Maybe()
		obj2.EXPECT().Model().Return(mdl).Maybe()
		obj2.EXPECT().Ephemeral().Return(false).Maybe()
		obj2.EXPECT().Light().Return(nil).Maybe()
		obj2.EXPECT().RigidBody().Return(nil).Maybe()
		obj2.EXPECT().TransformData().Return([3]float32{1, 0, 0}, [3]float32{1, 1, 1}, [3]float32{0, 0, 0}, [3]float32{0, 0, 0}).Maybe()
		obj2.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj2.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { inst2 = id }).Return().Maybe()

		id1 := s.Add(obj1)
		id2 := s.Add(obj2)
		suite.Equal(2, s.Count())

		s.Remove(id1)
		suite.Equal(1, s.Count())
		suite.Nil(s.Get(id1))
		suite.NotNil(s.Get(id2))
		suite.Equal(0, inst2)
	})

	suite.Run("cleans up physics body when physics handler is set", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		ph := &physicsmocks.MockPhysics{}
		ph.EXPECT().BodyIndex(mock.Anything).Return(0, false).Maybe()
		ph.EXPECT().RemoveBody(mock.Anything).Return().Maybe()
		s.SetPhysicsHandler(ph)

		obj, _ := newAddableObject("cube", false)
		obj.EXPECT().Animator().Return(nil).Maybe()
		obj.EXPECT().AnimatorInstanceID().Return(-1).Maybe()

		id := s.Add(obj)
		s.Remove(id)
		ph.AssertCalled(suite.T(), "RemoveBody", id)
	})
}

func (suite *sceneTest) TestPrepareCompute() {
	suite.Run("no-op when renderer is nil", func() {
		s, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		s.PrepareCompute(0.016)
	})

	suite.Run("no-op for empty animator pool", func() {
		s, cam, r := newMinimalScene("test")
		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		s.PrepareCompute(0.016)
	})

	suite.Run("dispatches compute for scene with one object", func() {
		s, cam, r := newMinimalScene("test")
		captured := capturePipelines(r)

		obj, _ := newWiredObject("cube", false, []material.Material{})
		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})
	})

	suite.Run("syncs attached light positions during compute", func() {
		s, cam, r := newLitScene("test")
		captured := capturePipelines(r)

		l := &lightmocks.MockLight{}
		l.EXPECT().SetPosition(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		l.EXPECT().Position().Return([3]float32{1, 2, 3}).Maybe()
		l.EXPECT().Color().Return([3]float32{1, 1, 1}).Maybe()
		l.EXPECT().Intensity().Return(float32(1.0)).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		l.EXPECT().Range().Return(float32(10.0)).Maybe()
		l.EXPECT().InnerCone().Return(float32(0)).Maybe()
		l.EXPECT().OuterCone().Return(float32(0)).Maybe()
		l.EXPECT().CastsShadows().Return(false).Maybe()

		obj, _ := newWiredObject("lamp", false, []material.Material{})
		obj.EXPECT().Light().Unset()
		obj.EXPECT().Light().Return(l).Maybe()
		obj.EXPECT().Enabled().Return(true).Maybe()
		obj.EXPECT().Position().Return(float32(1), float32(2), float32(3)).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		s.PrepareCompute(0.016)
		l.AssertCalled(suite.T(), "SetPosition", float32(1), float32(2), float32(3))
	})

	suite.Run("extracts camera controller position for GPU uniform", func() {
		s, cam, r := newMinimalScene("test")
		captured := capturePipelines(r)

		obj, _ := newWiredObject("cube", false, []material.Material{})
		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		ctrl := &cameramocks.MockCameraController{}
		ctrl.EXPECT().Position().Return(float32(5), float32(10), float32(15)).Maybe()
		ctrl.EXPECT().Target().Return(float32(0), float32(0), float32(0)).Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(ctrl).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})
	})

	suite.Run("handles phase 2 culling reset for animator with mesh provider", func() {
		s, cam, r := newMinimalScene("test")
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")

		obj, mdl := newWiredObject("cull_cube", false, []material.Material{})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(6).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})
	})
}

func (suite *sceneTest) TestDrawCalls() {
	suite.Run("returns error when renderer is nil", func() {
		s, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		err := s.DrawCalls()
		suite.Error(err)
	})

	suite.Run("returns nil with empty animator pool", func() {
		s, _, _ := newMinimalScene("test")
		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("issues draw call for object with material and mesh", func() {
		s, _, r := newMinimalScene("test")
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bgp")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("cube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, mdl := newWiredObject("cube", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("skips material when pipeline key is empty", func() {
		s, _, r := newMinimalScene("test")
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterMaterial(mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("").Maybe()
		mat.EXPECT().BindGroupProvider().Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, mdl := newWiredObject("cube", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("resolves lit shader bindings for lights shadow and tiles providers", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_lit")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("litcube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, mdl := newWiredObject("litcube", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
		r.AssertCalled(suite.T(), "DrawCall", "litcube", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("resolves material via Provider when available", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matProviderBGP := bind_group_provider.NewBindGroupProvider("mat_provider")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("provmat").Maybe()
		mat.EXPECT().BindGroupProvider().Return(bind_group_provider.NewBindGroupProvider("mat_fallback")).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(matProviderBGP).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, mdl := newWiredObject("provmat", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
		r.AssertCalled(suite.T(), "DrawCall", "provmat", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("skips draw when model has no render materials", func() {
		s, _, r := newMinimalScene("test")
		captured := capturePipelines(r)

		obj, _ := newWiredObject("empty_mat", false, []material.Material{})
		s.Add(obj)

		wirePipelineLookup(r, captured)
		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("indirect draw after PrepareCompute enables culling", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		camBGP.SetBuffer(0, &wgpu.Buffer{})
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()
		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		// InitBindGroup that creates buffers for ALL descriptor entries
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		handler := light.NewLightingHandler()
		handler.SetEnabled(true)

		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().DrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_indirect")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("indcube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler))

		obj, mdl := newWiredObject("indcube", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(3).Maybe()

		s.Add(obj)
		wirePipelineLookup(r, captured)

		// PrepareCompute sets frustum planes → activates culling
		s.PrepareCompute(0.016)

		err := s.DrawCalls()
		suite.NoError(err)
		r.AssertCalled(suite.T(), "DrawCallIndirect", "indcube", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("returns error when DrawCall fails", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_err")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("errcube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, mdl := newWiredObject("errcube", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(fmt.Errorf("draw failed")).Maybe()

		err := s.DrawCalls()
		suite.Error(err)
		suite.Contains(err.Error(), "draw call failed")
	})

	suite.Run("returns error when DrawCallIndirect fails", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		camBGP.SetBuffer(0, &wgpu.Buffer{})
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()
		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
		}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		handler := light.NewLightingHandler()
		handler.SetEnabled(true)

		capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().DrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(fmt.Errorf("indirect draw failed")).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_inderr")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("inderrcube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler))

		obj, mdl := newWiredObject("inderrcube", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(3).Maybe()

		s.Add(obj)

		s.PrepareCompute(0.016)

		err := s.DrawCalls()
		suite.Error(err)
		suite.Contains(err.Error(), "indirect draw call failed")
	})

	suite.Run("resolves binding group annotation types from lit shader", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, r := newLitScene("test", scene.WithLighting(handler))
		capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_litbg")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("litbg").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, mdl := newWiredObject("litbg", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
		r.AssertCalled(suite.T(), "DrawCall", "litbg", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("resolves effect provider for annotated shader", func() {
		s, _, r := newMinimalScene("test")
		capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_eff")
		effectBGP := bind_group_provider.NewBindGroupProvider("effect")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("effcube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, mdl := newWiredObject("effcube", false, []material.Material{mat})
		mdl.EXPECT().EffectProvider().Unset()
		mdl.EXPECT().EffectProvider().Return(effectBGP).Maybe()
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})

	suite.Run("resolves provider annotations for camera lights tiles and animator", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_prov")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("provcube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("tests/assets/shaders/test_dc_provider_frag.wgsl").Maybe()

		obj, mdl := newWiredObject("provcube", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)
		wirePipelineLookup(r, captured)
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
		r.AssertCalled(suite.T(), "DrawCall", "provcube", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("resolves bindgroup annotations for light shadow overlay and effect", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_bg")
		effectBGP := bind_group_provider.NewBindGroupProvider("effect_bg")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("bgcube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("tests/assets/shaders/test_dc_bindgroup_frag.wgsl").Maybe()

		obj, mdl := newWiredObject("bgcube", false, []material.Material{mat})
		mdl.EXPECT().EffectProvider().Unset()
		mdl.EXPECT().EffectProvider().Return(effectBGP).Maybe()
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)
		wirePipelineLookup(r, captured)
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
		r.AssertCalled(suite.T(), "DrawCall", "bgcube", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("resolves effect provider and shadow_uniform bindgroup annotations", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_eff2")
		effectBGP := bind_group_provider.NewBindGroupProvider("effect_prov")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("effprov").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("tests/assets/shaders/test_dc_effect_frag.wgsl").Maybe()

		obj, mdl := newWiredObject("effprov", false, []material.Material{mat})
		mdl.EXPECT().EffectProvider().Unset()
		mdl.EXPECT().EffectProvider().Return(effectBGP).Maybe()
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)
		wirePipelineLookup(r, captured)
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
		r.AssertCalled(suite.T(), "DrawCall", "effprov", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("skips material when provider group is nil for disabled lighting", func() {
		s, _, r := newMinimalScene("test")
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_skip")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("skipcube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("tests/assets/shaders/test_dc_provider_frag.wgsl").Maybe()

		obj, mdl := newWiredObject("skipcube", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)
		wirePipelineLookup(r, captured)

		// lighting is disabled so lights/tiles providers return nil → skipMaterial
		err := s.DrawCalls()
		suite.NoError(err)
		r.AssertNotCalled(suite.T(), "DrawCall")
	})

	suite.Run("skips draw when render pipeline is nil", func() {
		s, _, r := newMinimalScene("test")
		capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_nil_pipe")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("nonexistent_pipe").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, mdl := newWiredObject("nilpipe", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		// Override Pipeline to return nil for the render key
		r.EXPECT().Pipeline(mock.Anything).Unset()
		r.EXPECT().Pipeline(mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
		r.AssertNotCalled(suite.T(), "DrawCall")
	})

	suite.Run("resolves effect_params via material Provider fallback", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat_eff_fb")
		matProvBGP := bind_group_provider.NewBindGroupProvider("mat_prov_fb")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("efffb").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(matProvBGP).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("tests/assets/shaders/test_dc_bindgroup_frag.wgsl").Maybe()

		obj, mdl := newWiredObject("efffb", false, []material.Material{mat})
		// EffectProvider returns nil so effect_params falls back to mat.Provider(g)
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)
		wirePipelineLookup(r, captured)
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
		r.AssertCalled(suite.T(), "DrawCall", "efffb", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("resolves overlay_params via effect provider fallback", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterMaterial(mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		effectBGP := bind_group_provider.NewBindGroupProvider("eff_overlay")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("ovfb").Maybe()
		mat.EXPECT().BindGroupProvider().Return(nil).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("tests/assets/shaders/test_dc_bindgroup_frag.wgsl").Maybe()

		obj, mdl := newWiredObject("ovfb", false, []material.Material{mat})
		mdl.EXPECT().EffectProvider().Unset()
		mdl.EXPECT().EffectProvider().Return(effectBGP).Maybe()
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)
		wirePipelineLookup(r, captured)
		r.EXPECT().DrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		err := s.DrawCalls()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestAddLight() {
	suite.Run("adds a single light", func() {
		s, _, _ := newLitScene("test")
		l := &lightmocks.MockLight{}
		s.AddLight(l)
		suite.Len(s.Lights(), 1)
	})

	suite.Run("adds multiple lights", func() {
		s, _, _ := newLitScene("test")
		l1 := &lightmocks.MockLight{}
		l2 := &lightmocks.MockLight{}
		s.AddLight(l1)
		s.AddLight(l2)
		suite.Len(s.Lights(), 2)
	})

	suite.Run("triggers initLighting on first call without pre-enabled handler", func() {
		s, cam, r := newMinimalScene("test", scene.WithScreenSize(1280, 720))
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()
		r.EXPECT().RegisterVSMShadowPipeline(mock.Anything).Return(nil).Maybe()
		cam.EXPECT().BindGroupProvider().Unset()
		cam.EXPECT().BindGroupProvider().Return(bind_group_provider.NewBindGroupProvider("cam_bgp")).Maybe()

		// GBuffer (initLighting step 6)
		r.EXPECT().CreateGBufferTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(nil).Maybe()

		// SSAO (initLighting step 7)
		r.EXPECT().CreateSSAOTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().WriteTexture(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		// SSAO lit fallback (initLighting step 8)
		r.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		// Composition (initLighting step 9)
		r.EXPECT().SampleCount().Return(uint32(1)).Maybe()
		r.EXPECT().CreateCompositionTextures(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().SetRenderTargetFormat(mock.Anything).Return().Maybe()
		r.EXPECT().RegisterCompositionPipeline(mock.Anything).Return(nil).Maybe()

		// SSR (initLighting step 10)
		r.EXPECT().CreateSSRTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateHiZTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, []*wgpu.TextureView{{}}, []*wgpu.TextureView{{}}, 1, nil).Maybe()

		// Probes lit fallback (initLighting step 12)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		l := &lightmocks.MockLight{}
		s.AddLight(l)
		suite.Len(s.Lights(), 1)
	})

	suite.Run("fully initializes lighting with buffer-creating InitBindGroup", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for binding := range sizeOverrides {
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()
		r.EXPECT().RegisterVSMShadowPipeline(mock.Anything).Return(nil).Maybe()
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		// GBuffer (initLighting step 6)
		r.EXPECT().CreateGBufferTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(nil).Maybe()

		// SSAO (initLighting step 7)
		r.EXPECT().CreateSSAOTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().WriteTexture(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		// SSAO lit fallback (initLighting step 8)
		r.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		// Composition (initLighting step 9)
		r.EXPECT().SampleCount().Return(uint32(1)).Maybe()
		r.EXPECT().CreateCompositionTextures(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().SetRenderTargetFormat(mock.Anything).Return().Maybe()
		r.EXPECT().RegisterCompositionPipeline(mock.Anything).Return(nil).Maybe()

		// SSR (initLighting step 10)
		r.EXPECT().CreateSSRTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateHiZTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, []*wgpu.TextureView{{}}, []*wgpu.TextureView{{}}, 1, nil).Maybe()

		// Probes lit fallback (initLighting step 12)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		s := scene.NewScene("test", cam, r, scene.WithScreenSize(1280, 720))

		l := &lightmocks.MockLight{}
		s.AddLight(l)
		suite.Len(s.Lights(), 1)
	})
}

func (suite *sceneTest) TestRemoveLight() {
	suite.Run("removes an existing light", func() {
		s, _, _ := newLitScene("test")
		l := &lightmocks.MockLight{}
		s.AddLight(l)
		s.RemoveLight(l)
		suite.Empty(s.Lights())
	})

	suite.Run("no-op when removing non-existent light", func() {
		s, _, _ := newLitScene("test")
		l1 := &lightmocks.MockLight{}
		l2 := &lightmocks.MockLight{}
		s.AddLight(l1)
		s.RemoveLight(l2)
		suite.Len(s.Lights(), 1)
	})
}

func (suite *sceneTest) TestDetachLight() {
	suite.Run("no-op when object has no light", func() {
		s, _, _ := newLitScene("test")
		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().Light().Return(nil).Maybe()
		s.DetachLight(obj)
	})

	suite.Run("removes attached light from scene", func() {
		s, _, _ := newLitScene("test")
		l := &lightmocks.MockLight{}
		s.AddLight(l)
		suite.Len(s.Lights(), 1)

		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().Light().Return(l).Maybe()
		s.DetachLight(obj)
		suite.Empty(s.Lights())
	})

	suite.Run("removes light tracking when object was added via Add", func() {
		s, _, r := newLitScene("test")
		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()

		l := &lightmocks.MockLight{}
		obj, _ := newAddableObject("lamp", false)
		obj.EXPECT().Light().Unset()
		obj.EXPECT().Light().Return(l).Maybe()

		s.Add(obj)
		suite.Len(s.Lights(), 1)

		s.DetachLight(obj)
		suite.Empty(s.Lights())
	})
}

func (suite *sceneTest) TestLights() {
	suite.Run("returns empty list for fresh scene", func() {
		s, _, _ := newLitScene("test")
		suite.Empty(s.Lights())
	})

	suite.Run("returns copy not original slice", func() {
		s, _, _ := newLitScene("test")
		l := &lightmocks.MockLight{}
		s.AddLight(l)
		lights := s.Lights()
		lights[0] = nil
		suite.NotNil(s.Lights()[0])
	})
}

func (suite *sceneTest) TestSetAmbientColor() {
	suite.Run("sets and retrieves ambient color", func() {
		s, _, _ := newMinimalScene("test")
		color := [3]float32{0.5, 0.6, 0.7}
		s.SetAmbientColor(color)
		suite.Equal(color, s.AmbientColor())
	})
}

func (suite *sceneTest) TestResize() {
	suite.Run("propagates to renderer", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().Resize(1920, 1080).Return().Once()
		cam := &cameramocks.MockCamera{}
		cam.EXPECT().SetAspect(float32(1920) / float32(1080)).Return().Once()
		s.SetCamera(cam)
		s.Resize(1920, 1080)
	})

	suite.Run("sets camera aspect when height is positive", func() {
		s, cam, r := newMinimalScene("test")
		r.EXPECT().Resize(800, 600).Return().Once()
		cam.EXPECT().SetAspect(float32(800) / float32(600)).Return().Once()
		s.Resize(800, 600)
	})

	suite.Run("skips camera aspect when height is zero", func() {
		s, _, r := newMinimalScene("test")
		r.EXPECT().Resize(800, 0).Return().Once()
		// SetAspect should NOT be called
		s.Resize(800, 0)
	})

	suite.Run("propagates to enabled light handler", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))
		r.EXPECT().Resize(1280, 720).Return().Once()
		cam.EXPECT().SetAspect(float32(1280) / float32(720)).Return().Once()
		s.Resize(1280, 720)
		suite.Equal(1280, handler.ScreenWidth())
		suite.Equal(720, handler.ScreenHeight())
	})

	suite.Run("no-op on renderer when renderer is nil", func() {
		s, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		s.SetCamera(nil)
		s.Resize(100, 100)
	})
}

func (suite *sceneTest) TestPrepareShadows() {
	suite.Run("no-op when lighting not initialized", func() {
		s, _, _ := newMinimalScene("test")
		s.PrepareShadows()
	})

	suite.Run("no-op when no shadow-casting directional light exists", func() {
		s, _, _ := newLitScene("test")
		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(false).Maybe()
		l.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		s.AddLight(l)
		s.PrepareShadows()
	})

	suite.Run("dispatches shadow depth pass for shadow-casting directional light", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SetVSMTextureView(&wgpu.TextureView{})
		handler.SetVSMAuxDepthTextureView(&wgpu.TextureView{})
		handler.SetPipelineKey("shadow_static_back", "shadow_pipe")

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		// Add shadow-casting directional light
		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		l.EXPECT().Position().Return([3]float32{0, 10, 0}).Maybe()
		l.EXPECT().Color().Return([3]float32{1, 1, 1}).Maybe()
		l.EXPECT().Intensity().Return(float32(1.0)).Maybe()
		l.EXPECT().Range().Return(float32(100)).Maybe()
		l.EXPECT().InnerCone().Return(float32(0)).Maybe()
		l.EXPECT().OuterCone().Return(float32(0)).Maybe()
		handler.AddLight(l)

		// Add camera controller so shadow VP uses target center
		ctrl := &cameramocks.MockCameraController{}
		ctrl.EXPECT().Target().Return(float32(0), float32(0), float32(0)).Maybe()
		ctrl.EXPECT().Position().Return(float32(0), float32(5), float32(10)).Maybe()
		cam.EXPECT().Controller().Return(ctrl).Maybe()

		// Create object with shadow-casting model and mesh
		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		obj, mdl := newWiredObject("shadow_cube", false, []material.Material{})
		mdl.EXPECT().CastsShadows().Return(true).Maybe()
		mdl.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Maybe()
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(nil).Maybe()
		r.EXPECT().BeginVSMShadowPass(mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndShadowPass().Return().Maybe()
		r.EXPECT().EndShadowFrame().Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareShadows()
		})
		r.AssertCalled(suite.T(), "ShadowDrawCall", "shadow_pipe", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("writes to shadow lit BGP when buffer exists", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SetVSMTextureView(&wgpu.TextureView{})
		handler.SetVSMAuxDepthTextureView(&wgpu.TextureView{})
		handler.SetPipelineKey("shadow_static_front", "shadow_front_pipe")
		handler.Bgp("shadow_lit").SetBuffer(0, &wgpu.Buffer{})

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{1, -1, 0}).Maybe()
		l.EXPECT().Position().Return([3]float32{0, 0, 0}).Maybe()
		l.EXPECT().Color().Return([3]float32{1, 1, 1}).Maybe()
		l.EXPECT().Intensity().Return(float32(1.0)).Maybe()
		l.EXPECT().Range().Return(float32(50)).Maybe()
		l.EXPECT().InnerCone().Return(float32(0)).Maybe()
		l.EXPECT().OuterCone().Return(float32(0)).Maybe()
		handler.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		obj, mdl := newWiredObject("front_cull", false, []material.Material{})
		mdl.EXPECT().CastsShadows().Return(true).Maybe()
		mdl.EXPECT().ShadowCullMode().Return(model.ShadowCullModeFront).Maybe()
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(nil).Maybe()
		r.EXPECT().BeginVSMShadowPass(mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndShadowPass().Return().Maybe()
		r.EXPECT().EndShadowFrame().Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareShadows()
		})
	})

	suite.Run("handles shadow cull mode none", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SetVSMTextureView(&wgpu.TextureView{})
		handler.SetVSMAuxDepthTextureView(&wgpu.TextureView{})
		handler.SetPipelineKey("shadow_static_none", "shadow_none_pipe")

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		l.EXPECT().Position().Return([3]float32{0, 0, 0}).Maybe()
		l.EXPECT().Color().Return([3]float32{1, 1, 1}).Maybe()
		l.EXPECT().Intensity().Return(float32(1.0)).Maybe()
		l.EXPECT().Range().Return(float32(50)).Maybe()
		l.EXPECT().InnerCone().Return(float32(0)).Maybe()
		l.EXPECT().OuterCone().Return(float32(0)).Maybe()
		handler.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		obj, mdl := newWiredObject("no_cull", false, []material.Material{})
		mdl.EXPECT().CastsShadows().Return(true).Maybe()
		mdl.EXPECT().ShadowCullMode().Return(model.ShadowCullModeNone).Maybe()
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(nil).Maybe()
		r.EXPECT().BeginVSMShadowPass(mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndShadowPass().Return().Maybe()
		r.EXPECT().EndShadowFrame().Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareShadows()
		})
	})

	suite.Run("uses skinned shadow pipeline key for skinned model", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SetVSMTextureView(&wgpu.TextureView{})
		handler.SetVSMAuxDepthTextureView(&wgpu.TextureView{})
		handler.SetPipelineKey("shadow_skinned_back", "shadow_skinned_pipe")

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		l.EXPECT().Position().Return([3]float32{0, 0, 0}).Maybe()
		l.EXPECT().Color().Return([3]float32{1, 1, 1}).Maybe()
		l.EXPECT().Intensity().Return(float32(1.0)).Maybe()
		l.EXPECT().Range().Return(float32(50)).Maybe()
		l.EXPECT().InnerCone().Return(float32(0)).Maybe()
		l.EXPECT().OuterCone().Return(float32(0)).Maybe()
		handler.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Name().Return("skinfox").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(2.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().Return("").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Maybe()
		mdl.EXPECT().Animations().Return(nil).Maybe()
		mdl.EXPECT().CastsShadows().Return(true).Maybe()
		mdl.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		obj := &gameobjectmocks.MockGameObject{}
		var objID uint64
		obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().RigidBody().Return(nil).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0}, [3]float32{1, 1, 1},
			[3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Return().Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(nil).Maybe()
		r.EXPECT().BeginVSMShadowPass(mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndShadowPass().Return().Maybe()
		r.EXPECT().EndShadowFrame().Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareShadows()
		})
		r.AssertCalled(suite.T(), "ShadowDrawCall", "shadow_skinned_pipe", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("skips model that does not cast shadows", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SetVSMTextureView(&wgpu.TextureView{})
		handler.SetVSMAuxDepthTextureView(&wgpu.TextureView{})
		handler.SetPipelineKey("shadow_static_back", "shadow_pipe")

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		l.EXPECT().Position().Return([3]float32{0, 0, 0}).Maybe()
		l.EXPECT().Color().Return([3]float32{1, 1, 1}).Maybe()
		l.EXPECT().Intensity().Return(float32(1.0)).Maybe()
		l.EXPECT().Range().Return(float32(50)).Maybe()
		l.EXPECT().InnerCone().Return(float32(0)).Maybe()
		l.EXPECT().OuterCone().Return(float32(0)).Maybe()
		handler.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		obj, mdl := newWiredObject("no_shadow", false, []material.Material{})
		mdl.EXPECT().CastsShadows().Return(false).Maybe()
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(nil).Maybe()
		r.EXPECT().BeginVSMShadowPass(mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndShadowPass().Return().Maybe()
		r.EXPECT().EndShadowFrame().Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareShadows()
		})
	})

	suite.Run("skips shadow pass when BeginShadowFrame returns error", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SetVSMTextureView(&wgpu.TextureView{})
		handler.SetVSMAuxDepthTextureView(&wgpu.TextureView{})
		handler.SetPipelineKey("shadow_static_back", "shadow_pipe")

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		l.EXPECT().Position().Return([3]float32{0, 0, 0}).Maybe()
		l.EXPECT().Color().Return([3]float32{1, 1, 1}).Maybe()
		l.EXPECT().Intensity().Return(float32(1.0)).Maybe()
		l.EXPECT().Range().Return(float32(50)).Maybe()
		l.EXPECT().InnerCone().Return(float32(0)).Maybe()
		l.EXPECT().OuterCone().Return(float32(0)).Maybe()
		handler.AddLight(l)

		cam.EXPECT().Controller().Return(nil).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(fmt.Errorf("no encoder")).Maybe()

		suite.NotPanics(func() {
			s.PrepareShadows()
		})
	})

	suite.Run("indirect shadow draw after PrepareCompute enables culling", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		camBGP.SetBuffer(0, &wgpu.Buffer{})
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()
		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
		}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().BeginShadowFrame().Return(nil).Maybe()
		r.EXPECT().BeginVSMShadowPass(mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndShadowPass().Return().Maybe()
		r.EXPECT().EndShadowFrame().Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SetVSMTextureView(&wgpu.TextureView{})
		handler.SetVSMAuxDepthTextureView(&wgpu.TextureView{})
		handler.SetPipelineKey("shadow_static_back", "shadow_indirect_pipe")

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler))

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		l.EXPECT().Position().Return([3]float32{0, 0, 0}).Maybe()
		l.EXPECT().Color().Return([3]float32{1, 1, 1}).Maybe()
		l.EXPECT().Intensity().Return(float32(1.0)).Maybe()
		l.EXPECT().Range().Return(float32(50)).Maybe()
		l.EXPECT().InnerCone().Return(float32(0)).Maybe()
		l.EXPECT().OuterCone().Return(float32(0)).Maybe()
		handler.AddLight(l)

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("shadow_ind").Maybe()
		mat.EXPECT().BindGroupProvider().Return(bind_group_provider.NewBindGroupProvider("mat_shadow")).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, mdl := newWiredObject("shadow_ind", false, []material.Material{mat})
		mdl.EXPECT().CastsShadows().Return(true).Maybe()
		mdl.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Maybe()
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(3).Maybe()

		s.Add(obj)
		wirePipelineLookup(r, captured)

		// PrepareCompute sets frustum planes → activates culling
		s.PrepareCompute(0.016)

		s.PrepareShadows()
		r.AssertCalled(suite.T(), "ShadowDrawCallIndirect", "shadow_indirect_pipe", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("no-op when renderer is nil", func() {
		s, _, _ := newLitScene("test")
		s.SetRenderer(nil)
		s.PrepareShadows()
	})
}

func (suite *sceneTest) TestPrepareLightCulling() {
	suite.Run("no-op when lighting not initialized", func() {
		s, _, _ := newMinimalScene("test")
		s.PrepareLightCulling()
	})

	suite.Run("no-op when renderer is nil", func() {
		s, _, _ := newMinimalScene("test")
		s.SetRenderer(nil)
		s.PrepareLightCulling()
	})

	suite.Run("no-op when camera is nil", func() {
		s, _, _ := newMinimalScene("test")
		s.SetCamera(nil)
		s.PrepareLightCulling()
	})

	suite.Run("dispatches cull compute with initialized handler", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.Resize(1280, 720)
		handler.SetPipelineKey("light_cull", "light_cull_compute")

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		handler.AddLight(l)

		cam.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().ViewMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Near().Return(float32(0.1)).Maybe()
		cam.EXPECT().Far().Return(float32(100.0)).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareLightCulling()
		})
	})
}

func (suite *sceneTest) TestWithActive() {
	suite.Run("sets active via builder option", func() {
		s, _, _ := newMinimalScene("test", scene.WithActive(true))
		suite.True(s.Active())
	})

	suite.Run("default active false without option", func() {
		s, _, _ := newMinimalScene("test")
		suite.False(s.Active())
	})
}

func (suite *sceneTest) TestWithComputeWorkers() {
	suite.Run("does not panic with positive value", func() {
		s, _, _ := newMinimalScene("test", scene.WithComputeWorkers(4))
		suite.NotNil(s)
	})

	suite.Run("clamps zero to minimum of 1", func() {
		s, _, _ := newMinimalScene("test", scene.WithComputeWorkers(0))
		suite.NotNil(s)
	})

	suite.Run("clamps negative to 1", func() {
		s, _, _ := newMinimalScene("test", scene.WithComputeWorkers(-5))
		suite.NotNil(s)
	})
}

func (suite *sceneTest) TestWithCullingDisabled() {
	suite.Run("sets culling disabled via builder", func() {
		s, _, _ := newMinimalScene("test", scene.WithCullingDisabled(true))
		suite.True(s.CullingDisabled())
	})
}

func (suite *sceneTest) TestWithObjects() {
	suite.Run("adds non-ephemeral objects to registry with auto IDs", func() {
		var id1, id2 uint64
		obj1 := &gameobjectmocks.MockGameObject{}
		obj2 := &gameobjectmocks.MockGameObject{}
		obj1.EXPECT().ID().RunAndReturn(func() uint64 { return id1 }).Maybe()
		obj1.EXPECT().SetID(mock.Anything).Run(func(id uint64) { id1 = id }).Return().Maybe()
		obj1.EXPECT().Ephemeral().Return(false).Maybe()
		obj2.EXPECT().ID().RunAndReturn(func() uint64 { return id2 }).Maybe()
		obj2.EXPECT().SetID(mock.Anything).Run(func(id uint64) { id2 = id }).Return().Maybe()
		obj2.EXPECT().Ephemeral().Return(false).Maybe()

		s, _, _ := newMinimalScene("test", scene.WithObjects(obj1, obj2))
		suite.Equal(2, s.Count())
	})

	suite.Run("ephemeral objects are not persisted", func() {
		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().ID().Return(uint64(0)).Maybe()
		obj.EXPECT().SetID(mock.Anything).Return().Maybe()
		obj.EXPECT().Ephemeral().Return(true).Maybe()

		s, _, _ := newMinimalScene("test", scene.WithObjects(obj))
		suite.Equal(0, s.Count())
	})

	suite.Run("objects with existing IDs keep them", func() {
		obj := &gameobjectmocks.MockGameObject{}
		obj.EXPECT().ID().Return(uint64(42)).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()

		s, _, _ := newMinimalScene("test", scene.WithObjects(obj))
		suite.Equal(1, s.Count())
		suite.NotNil(s.Get(42))
	})
}

func (suite *sceneTest) TestWithLighting() {
	suite.Run("injects pre-configured lighting handler", func() {
		handler := light.NewLightingHandler(light.WithAmbientColor([3]float32{0.1, 0.2, 0.3}))
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		suite.Equal([3]float32{0.1, 0.2, 0.3}, s.AmbientColor())
	})
}

func (suite *sceneTest) TestWithPhysics() {
	suite.Run("sets physics handler via builder option", func() {
		s, _, _ := newMinimalScene("test", scene.WithPhysics())
		suite.NotNil(s)
	})
}

func (suite *sceneTest) TestWithScreenSize() {
	suite.Run("sets initial screen dimensions", func() {
		s, _, _ := newMinimalScene("test", scene.WithScreenSize(1920, 1080))
		suite.NotNil(s)
	})
}

// newPhysicsScene creates a scene with a buffer-creating renderer and a physics
// mock handler already wired. The renderer's InitBindGroup creates dummy buffers
// for each descriptor entry. Returns (scene, cameraMock, rendererMock, physicsMock).
func newPhysicsScene(name string, opts ...scene.SceneBuilderOption) (scene.Scene, *cameramocks.MockCamera, *renderermocks.MockRenderer, *physicsmocks.MockPhysics) {
	cam := &cameramocks.MockCamera{}
	r := &renderermocks.MockRenderer{}
	ph := &physicsmocks.MockPhysics{}

	bgp := bind_group_provider.NewBindGroupProvider("cam_bgp")
	cam.EXPECT().BindGroupProvider().Return(bgp).Maybe()
	cam.EXPECT().SetDelegate(mock.Anything).Maybe()

	r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
		provider bind_group_provider.BindGroupProvider,
		desc wgpu.BindGroupLayoutDescriptor,
		usageOverrides map[int]wgpu.BufferUsage,
		sizeOverrides map[int]uint64,
	) error {
		for _, entry := range desc.Entries {
			binding := int(entry.Binding)
			if provider.Buffer(binding) == nil {
				provider.SetBuffer(binding, &wgpu.Buffer{})
			}
		}
		return nil
	}).Maybe()

	allOpts := append([]scene.SceneBuilderOption{scene.WithPhysics()}, opts...)
	s := scene.NewScene(name, cam, r, allOpts...)
	s.SetPhysicsHandler(ph)

	return s, cam, r, ph
}

// newPhysicsAddableObject creates a mock GameObject with a RigidBody suitable
// for triggering initPhysicsGPU. The rb mock is configured for a standard dynamic body.
func newPhysicsAddableObject(mdlName string) (*gameobjectmocks.MockGameObject, *modelmocks.MockModel, *physicsmocks.MockRigidBody) {
	mdl := &modelmocks.MockModel{}
	mdl.EXPECT().Skinned().Return(false).Maybe()
	mdl.EXPECT().Name().Return(mdlName).Maybe()
	mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
	mdl.EXPECT().MeshProvider().Return(nil).Maybe()
	mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
	mdl.EXPECT().ComputePipelineKey().Return("").Maybe()
	mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()
	mdl.EXPECT().EffectProvider().Return(nil).Maybe()

	rb := &physicsmocks.MockRigidBody{}
	rb.EXPECT().Kinematic().Return(false).Maybe()

	obj := &gameobjectmocks.MockGameObject{}
	var objID uint64
	var instID int
	obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
	obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
	obj.EXPECT().Model().Return(mdl).Maybe()
	obj.EXPECT().Ephemeral().Return(false).Maybe()
	obj.EXPECT().Light().Return(nil).Maybe()
	obj.EXPECT().RigidBody().Return(rb).Maybe()
	obj.EXPECT().TransformData().Return(
		[3]float32{1, 2, 3},
		[3]float32{1, 1, 1},
		[3]float32{0, 0, 0},
		[3]float32{0, 0, 0},
	).Maybe()
	obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
	obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { instID = id }).Return().Maybe()
	obj.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return instID }).Maybe()

	return obj, mdl, rb
}

// wirePhysicsMockForInit configures the physics mock with the minimum expectations
// needed for initPhysicsGPU, createPhysicsSyncGroup, and RegisterBody to succeed.
func wirePhysicsMockForInit(ph *physicsmocks.MockPhysics) {
	// Buffers BGP for the unified physics buffer set
	bufBGP := bind_group_provider.NewBindGroupProvider("physics_buffers")
	ph.EXPECT().Buffers().Return(bufBGP).Maybe()

	// Per-stage bind group providers
	stageNames := []string{
		"particle_values", "aabb_reduce", "grid_build_params", "grid_clear",
		"grid_insert", "collision", "momenta", "integrate", "sync",
	}
	for _, name := range stageNames {
		bgp := bind_group_provider.NewBindGroupProvider("physics_" + name)
		ph.EXPECT().Bgp(name).Return(bgp).Maybe()
	}

	ph.EXPECT().MaxBodies().Return(uint32(256)).Maybe()
	ph.EXPECT().MaxParticles().Return(uint32(2048)).Maybe()
	ph.EXPECT().MaxGridCells().Return(uint32(128 * 128 * 128)).Maybe()

	// Track pipeline keys so PipelineKey returns the value set by SetPipelineKey
	pipelineKeys := make(map[string]string)
	ph.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Run(func(name, key string) {
		pipelineKeys[name] = key
	}).Return().Maybe()
	ph.EXPECT().PipelineKey(mock.Anything).RunAndReturn(func(key string) string {
		return pipelineKeys[key]
	}).Maybe()

	ph.EXPECT().SetStagingBuffer(mock.Anything).Return().Maybe()
	ph.EXPECT().RegisterBody(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(0).Maybe()
	ph.EXPECT().BodyIndex(mock.Anything).Return(0, true).Maybe()
	ph.EXPECT().BodyParticleInfo(mock.Anything).Return(uint32(0), uint32(0)).Maybe()
	ph.EXPECT().RemoveBody(mock.Anything).Return().Maybe()
}

func (suite *sceneTest) TestAddWithPhysics() {
	suite.Run("triggers initPhysicsGPU and registers body on first rigid body add", func() {
		s, _, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_cube")
		id := s.Add(obj)
		suite.NotZero(id)
		suite.Equal(1, s.Count())

		// Verify physics init happened: SetPipelineKey should have been called for all stages
		ph.AssertCalled(suite.T(), "SetPipelineKey", "bone_update", mock.Anything)
		ph.AssertCalled(suite.T(), "SetStagingBuffer", mock.Anything)
		ph.AssertCalled(suite.T(), "RegisterBody", mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything)

		wirePipelineLookup(r, captured)
	})

	suite.Run("second body skips initPhysicsGPU but still registers body", func() {
		s, _, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj1, _, _ := newPhysicsAddableObject("phys_cube")
		s.Add(obj1)

		// Second add should reuse animator (same model name not same mock - but using different mock)
		obj2, _, _ := newPhysicsAddableObject("phys_cube2")
		id2 := s.Add(obj2)
		suite.NotZero(id2)
		suite.Equal(2, s.Count())

		wirePipelineLookup(r, captured)
	})

	suite.Run("creates bone particle update group for kinematic skinned body", func() {
		s, _, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)
		// Override BodyParticleInfo for this test to return non-zero particles
		ph.EXPECT().BodyParticleInfo(mock.Anything).Unset()
		ph.EXPECT().BodyParticleInfo(mock.Anything).Return(uint32(0), uint32(10)).Maybe()

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		skeleton := &model.Skeleton{
			Bones: []model.Bone{
				{Name: "root", ParentIndex: -1},
				{Name: "spine", ParentIndex: 0},
			},
			RootBoneIndices: []int32{0},
			BoneNameToIndex: map[string]int32{"root": 0, "spine": 1},
		}

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Name().Return("skel_phys").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(2.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().Return("").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()
		mdl.EXPECT().Skeleton().Return(skeleton).Maybe()
		mdl.EXPECT().Animations().Return(nil).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		rb := &physicsmocks.MockRigidBody{}
		rb.EXPECT().Kinematic().Return(true).Maybe()

		obj := &gameobjectmocks.MockGameObject{}
		var objID uint64
		var instID int
		obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().RigidBody().Return(rb).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0}, [3]float32{1, 1, 1},
			[3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { instID = id }).Return().Maybe()
		obj.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return instID }).Maybe()

		id := s.Add(obj)
		suite.NotZero(id)

		wirePipelineLookup(r, captured)
	})

	suite.Run("skips bone particle update when particleCount is zero", func() {
		s, _, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		skeleton := &model.Skeleton{
			Bones: []model.Bone{
				{Name: "root", ParentIndex: -1},
			},
			RootBoneIndices: []int32{0},
			BoneNameToIndex: map[string]int32{"root": 0},
		}

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Name().Return("skel_nop").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().Return("").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()
		mdl.EXPECT().Skeleton().Return(skeleton).Maybe()
		mdl.EXPECT().Animations().Return(nil).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		rb := &physicsmocks.MockRigidBody{}
		rb.EXPECT().Kinematic().Return(true).Maybe()

		obj := &gameobjectmocks.MockGameObject{}
		var objID uint64
		var instID int
		obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().RigidBody().Return(rb).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0}, [3]float32{1, 1, 1},
			[3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { instID = id }).Return().Maybe()
		obj.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return instID }).Maybe()

		// BodyParticleInfo returns 0 particles → createBoneParticleUpdateGroup returns early
		id := s.Add(obj)
		suite.NotZero(id)
		suite.Equal(1, s.Count())

		wirePipelineLookup(r, captured)
	})
}

func (suite *sceneTest) TestRemoveWithPhysics() {
	suite.Run("patches sync map and removes body on removal", func() {
		s, _, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_rm")
		obj.EXPECT().Animator().RunAndReturn(func() animator.Animator { return nil }).Maybe()

		id := s.Add(obj)
		s.Remove(id)

		ph.AssertCalled(suite.T(), "RemoveBody", id)
		suite.Equal(0, s.Count())

		wirePipelineLookup(r, captured)
	})

	suite.Run("swap-remove patches sync map for swapped object", func() {
		s, _, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		bodyIdx := 0
		ph.EXPECT().RegisterBody(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Unset()
		ph.EXPECT().RegisterBody(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(objID uint64, pos [3]float32, rot [3]float32, rb physics.RigidBody, instID uint32) int {
			idx := bodyIdx
			bodyIdx++
			return idx
		}).Maybe()

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("shared_phys").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().Return("").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		rb := &physicsmocks.MockRigidBody{}
		rb.EXPECT().Kinematic().Return(false).Maybe()

		obj1 := &gameobjectmocks.MockGameObject{}
		var obj1ID uint64
		var inst1 int
		var capturedAnim animator.Animator
		obj1.EXPECT().ID().RunAndReturn(func() uint64 { return obj1ID }).Maybe()
		obj1.EXPECT().SetID(mock.Anything).Run(func(id uint64) { obj1ID = id }).Return().Maybe()
		obj1.EXPECT().Model().Return(mdl).Maybe()
		obj1.EXPECT().Ephemeral().Return(false).Maybe()
		obj1.EXPECT().Light().Return(nil).Maybe()
		obj1.EXPECT().RigidBody().Return(rb).Maybe()
		obj1.EXPECT().TransformData().Return([3]float32{0, 0, 0}, [3]float32{1, 1, 1}, [3]float32{0, 0, 0}, [3]float32{0, 0, 0}).Maybe()
		obj1.EXPECT().SetAnimator(mock.Anything).Run(func(a animator.Animator) { capturedAnim = a }).Return().Maybe()
		obj1.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { inst1 = id }).Return().Maybe()
		obj1.EXPECT().Animator().RunAndReturn(func() animator.Animator { return capturedAnim }).Maybe()
		obj1.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return inst1 }).Maybe()

		obj2 := &gameobjectmocks.MockGameObject{}
		var obj2ID uint64
		var inst2 int
		obj2.EXPECT().ID().RunAndReturn(func() uint64 { return obj2ID }).Maybe()
		obj2.EXPECT().SetID(mock.Anything).Run(func(id uint64) { obj2ID = id }).Return().Maybe()
		obj2.EXPECT().Model().Return(mdl).Maybe()
		obj2.EXPECT().Ephemeral().Return(false).Maybe()
		obj2.EXPECT().Light().Return(nil).Maybe()
		obj2.EXPECT().RigidBody().Return(rb).Maybe()
		obj2.EXPECT().TransformData().Return([3]float32{1, 0, 0}, [3]float32{1, 1, 1}, [3]float32{0, 0, 0}, [3]float32{0, 0, 0}).Maybe()
		obj2.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj2.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { inst2 = id }).Return().Maybe()
		obj2.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return inst2 }).Maybe()

		id1 := s.Add(obj1)
		id2 := s.Add(obj2)
		suite.Equal(2, s.Count())

		s.Remove(id1)
		suite.Equal(1, s.Count())
		suite.Nil(s.Get(id1))
		suite.NotNil(s.Get(id2))
		ph.AssertCalled(suite.T(), "RemoveBody", id1)

		wirePipelineLookup(r, captured)
	})
}

func (suite *sceneTest) TestPrepareComputeWithPhysics() {
	suite.Run("dispatches physics stages when handler is enabled with substeps", func() {
		s, cam, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_compute")
		s.Add(obj)

		wirePipelineLookup(r, captured)

		// Configure physics mock for PrepareCompute dispatch
		ph.EXPECT().Enabled().Return(true).Maybe()
		ph.EXPECT().ReadbackPending().Return(false).Maybe()
		ph.EXPECT().PrepareStep(mock.Anything).Return(1, make([]byte, 32)).Maybe()
		ph.EXPECT().StagedWriteData().Return(nil).Maybe()
		ph.EXPECT().ParticleCount().Return(10).Maybe()
		ph.EXPECT().BodiesCount().Return(1).Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).RunAndReturn(func(key string) string {
			return "physics_" + key + "_pipe"
		}).Maybe()
		ph.EXPECT().ConsumeReadbackRequest().Return(false).Maybe()

		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})

		// Should dispatch 8 stages per substep + sync group
		r.AssertCalled(suite.T(), "DispatchCompute", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("processes pending readback before physics step", func() {
		s, cam, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_readback")
		s.Add(obj)

		wirePipelineLookup(r, captured)

		stagingBuf := &wgpu.Buffer{}
		ph.EXPECT().Enabled().Return(true).Maybe()
		ph.EXPECT().ReadbackPending().Return(true).Maybe()
		ph.EXPECT().BodiesCount().Return(1).Maybe()
		ph.EXPECT().StagingBuffer().Return(stagingBuf).Maybe()
		r.EXPECT().ReadMappedBuffer(stagingBuf, mock.Anything, mock.Anything).Return(make([]byte, 128), nil).Maybe()
		ph.EXPECT().ProcessReadback(mock.Anything).Return().Maybe()
		ph.EXPECT().ClearReadbackPending().Return().Maybe()

		ph.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Maybe()
		ph.EXPECT().StagedWriteData().Return(nil).Maybe()
		ph.EXPECT().ParticleCount().Return(0).Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).Return("").Maybe()
		ph.EXPECT().ConsumeReadbackRequest().Return(false).Maybe()

		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})

		ph.AssertCalled(suite.T(), "ProcessReadback", mock.Anything)
		ph.AssertCalled(suite.T(), "ClearReadbackPending")
	})

	suite.Run("encodes buffer copy when readback is requested", func() {
		s, cam, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_copy")
		s.Add(obj)

		wirePipelineLookup(r, captured)

		stagingBuf := &wgpu.Buffer{}
		ph.EXPECT().Enabled().Return(true).Maybe()
		ph.EXPECT().ReadbackPending().Return(false).Maybe()
		ph.EXPECT().PrepareStep(mock.Anything).Return(1, make([]byte, 32)).Maybe()
		ph.EXPECT().StagedWriteData().Return(nil).Maybe()
		ph.EXPECT().ParticleCount().Return(5).Maybe()
		ph.EXPECT().BodiesCount().Return(1).Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).Return("phys_pipe").Maybe()
		ph.EXPECT().ConsumeReadbackRequest().Return(true).Maybe()
		ph.EXPECT().StagingBuffer().Return(stagingBuf).Maybe()

		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().CopyBufferToBuffer(mock.Anything, stagingBuf, mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})

		r.AssertCalled(suite.T(), "CopyBufferToBuffer", mock.Anything, stagingBuf, mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("flushes staged writes even when zero substeps", func() {
		s, cam, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_flush")
		s.Add(obj)

		wirePipelineLookup(r, captured)

		ph.EXPECT().Enabled().Return(true).Maybe()
		ph.EXPECT().ReadbackPending().Return(false).Maybe()
		ph.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Maybe()
		ph.EXPECT().StagedWriteData().Return([]bind_group_provider.BufferWrite{
			{Provider: bind_group_provider.NewBindGroupProvider("staged"), Binding: 0, Offset: 0, Data: []byte{1, 2, 3, 4}},
		}).Maybe()
		ph.EXPECT().ParticleCount().Return(0).Maybe()
		ph.EXPECT().BodiesCount().Return(0).Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).Return("").Maybe()
		ph.EXPECT().ConsumeReadbackRequest().Return(false).Maybe()

		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})
	})

	suite.Run("dispatches bone particle update after animator compute", func() {
		s, cam, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)
		ph.EXPECT().BodyParticleInfo(mock.Anything).Unset()
		ph.EXPECT().BodyParticleInfo(mock.Anything).Return(uint32(0), uint32(8)).Maybe()

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		skeleton := &model.Skeleton{
			Bones: []model.Bone{
				{Name: "root", ParentIndex: -1},
			},
			RootBoneIndices: []int32{0},
			BoneNameToIndex: map[string]int32{"root": 0},
		}

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Name().Return("bone_phys").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(2.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().Return("").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()
		mdl.EXPECT().Skeleton().Return(skeleton).Maybe()
		mdl.EXPECT().Animations().Return(nil).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		rb := &physicsmocks.MockRigidBody{}
		rb.EXPECT().Kinematic().Return(true).Maybe()

		obj := &gameobjectmocks.MockGameObject{}
		var objID uint64
		var instID int
		obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().RigidBody().Return(rb).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0}, [3]float32{1, 1, 1},
			[3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { instID = id }).Return().Maybe()
		obj.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return instID }).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)

		// Configure physics for compute dispatch
		ph.EXPECT().Enabled().Return(true).Maybe()
		ph.EXPECT().ReadbackPending().Return(false).Maybe()
		ph.EXPECT().PrepareStep(mock.Anything).Return(1, make([]byte, 32)).Maybe()
		ph.EXPECT().StagedWriteData().Return(nil).Maybe()
		ph.EXPECT().ParticleCount().Return(8).Maybe()
		ph.EXPECT().BodiesCount().Return(1).Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).RunAndReturn(func(key string) string {
			return "physics_" + key + "_pipe"
		}).Maybe()
		ph.EXPECT().ConsumeReadbackRequest().Return(false).Maybe()

		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})

		// bone_update dispatch should have been called
		r.AssertCalled(suite.T(), "DispatchCompute", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("physics disabled skips entire physics block", func() {
		s, cam, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_disabled")
		s.Add(obj)

		wirePipelineLookup(r, captured)

		ph.EXPECT().Enabled().Return(false).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})
	})

	suite.Run("readback with zero bodies does not call ReadMappedBuffer", func() {
		s, cam, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_zero")
		s.Add(obj)

		wirePipelineLookup(r, captured)

		ph.EXPECT().Enabled().Return(true).Maybe()
		ph.EXPECT().ReadbackPending().Return(true).Maybe()
		ph.EXPECT().BodiesCount().Return(0).Maybe()
		ph.EXPECT().ClearReadbackPending().Return().Maybe()

		ph.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Maybe()
		ph.EXPECT().StagedWriteData().Return(nil).Maybe()
		ph.EXPECT().ParticleCount().Return(0).Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).Return("").Maybe()
		ph.EXPECT().ConsumeReadbackRequest().Return(false).Maybe()

		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})

		ph.AssertCalled(suite.T(), "ClearReadbackPending")
	})

	suite.Run("readback error does not panic", func() {
		s, cam, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_err")
		s.Add(obj)

		wirePipelineLookup(r, captured)

		stagingBuf := &wgpu.Buffer{}
		ph.EXPECT().Enabled().Return(true).Maybe()
		ph.EXPECT().ReadbackPending().Return(true).Maybe()
		ph.EXPECT().BodiesCount().Return(1).Maybe()
		ph.EXPECT().StagingBuffer().Return(stagingBuf).Maybe()
		r.EXPECT().ReadMappedBuffer(stagingBuf, mock.Anything, mock.Anything).Return(nil, fmt.Errorf("map failed")).Maybe()
		ph.EXPECT().ClearReadbackPending().Return().Maybe()

		ph.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Maybe()
		ph.EXPECT().StagedWriteData().Return(nil).Maybe()
		ph.EXPECT().ParticleCount().Return(0).Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).Return("").Maybe()
		ph.EXPECT().ConsumeReadbackRequest().Return(false).Maybe()

		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})

		ph.AssertCalled(suite.T(), "ClearReadbackPending")
	})

	suite.Run("flushes sync writes staged during Add even with substeps", func() {
		s, cam, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()

		// Count WriteBuffers calls and check that sync writes are flushed
		var writeCallCount int
		r.EXPECT().WriteBuffers(mock.Anything).Run(func(writes []bind_group_provider.BufferWrite) {
			writeCallCount++
		}).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_sync_flush")
		s.Add(obj)

		wirePipelineLookup(r, captured)

		ph.EXPECT().Enabled().Return(true).Maybe()
		ph.EXPECT().ReadbackPending().Return(false).Maybe()
		ph.EXPECT().PrepareStep(mock.Anything).Return(1, make([]byte, 32)).Maybe()
		ph.EXPECT().StagedWriteData().Return(nil).Maybe()
		ph.EXPECT().ParticleCount().Return(1).Maybe()
		ph.EXPECT().BodiesCount().Return(1).Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).Return("phys_pipe").Maybe()
		ph.EXPECT().ConsumeReadbackRequest().Return(false).Maybe()

		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		s.PrepareCompute(0.016)

		// WriteBuffers should have been called for sync writes and stage writes
		suite.Greater(writeCallCount, 0)
	})
}

func (suite *sceneTest) TestRemoveWithPhysicsEdgeCases() {
	suite.Run("patchSyncMapEntry returns early when animator is nil", func() {
		s, _, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, mdl, _ := newPhysicsAddableObject("phys_nil_anim")
		mdl.EXPECT().BoundingRadius().Unset()
		mdl.EXPECT().BoundingRadius().Return(float32(0)).Maybe()
		obj.EXPECT().Animator().RunAndReturn(func() animator.Animator { return nil }).Maybe()

		id := s.Add(obj)

		// Remove calls patchSyncMapEntry with the object's animator (nil)
		s.Remove(id)
		suite.Equal(0, s.Count())
		_ = captured
	})

	suite.Run("patchSyncMapEntry returns early when BodyIndex is not found", func() {
		s, _, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("phys_no_idx").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().Return("").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		rb := &physicsmocks.MockRigidBody{}
		rb.EXPECT().Kinematic().Return(false).Maybe()

		obj := &gameobjectmocks.MockGameObject{}
		var objID uint64
		var instID int
		var capturedAnim animator.Animator
		obj.EXPECT().ID().RunAndReturn(func() uint64 { return objID }).Maybe()
		obj.EXPECT().SetID(mock.Anything).Run(func(id uint64) { objID = id }).Return().Maybe()
		obj.EXPECT().Model().Return(mdl).Maybe()
		obj.EXPECT().Ephemeral().Return(false).Maybe()
		obj.EXPECT().Light().Return(nil).Maybe()
		obj.EXPECT().RigidBody().Return(rb).Maybe()
		obj.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0}, [3]float32{1, 1, 1},
			[3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj.EXPECT().SetAnimator(mock.Anything).Run(func(a animator.Animator) { capturedAnim = a }).Return().Maybe()
		obj.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { instID = id }).Return().Maybe()
		obj.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return instID }).Maybe()
		obj.EXPECT().Animator().RunAndReturn(func() animator.Animator { return capturedAnim }).Maybe()

		id := s.Add(obj)

		// Override BodyIndex AFTER Add so initPhysicsGPU succeeds but Remove's patchSyncMapEntry
		// finds the anim non-nil, then BodyIndex returns false.
		ph.EXPECT().BodyIndex(mock.Anything).Unset()
		ph.EXPECT().BodyIndex(mock.Anything).Return(0, false).Maybe()

		s.Remove(id)
		suite.Equal(0, s.Count())
		_ = captured
	})
}

func (suite *sceneTest) TestInitPhysicsGPUErrors() {
	suite.Run("panics when RegisterPipelines fails", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}
		ph := &physicsmocks.MockPhysics{}

		bgp := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(bgp).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		// RegisterPipelines fails
		r.EXPECT().RegisterPipelines(mock.Anything).Return(fmt.Errorf("pipeline registration failed")).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithPhysics())
		s.SetPhysicsHandler(ph)

		// Set up physics mock with minimal expectations
		bufBGP := bind_group_provider.NewBindGroupProvider("physics_buffers")
		ph.EXPECT().Buffers().Return(bufBGP).Maybe()
		stageNames := []string{
			"particle_values", "aabb_reduce", "grid_build_params", "grid_clear",
			"grid_insert", "collision", "momenta", "integrate", "sync",
		}
		for _, name := range stageNames {
			ph.EXPECT().Bgp(name).Return(bind_group_provider.NewBindGroupProvider("physics_" + name)).Maybe()
		}
		ph.EXPECT().MaxBodies().Return(uint32(256)).Maybe()
		ph.EXPECT().MaxParticles().Return(uint32(2048)).Maybe()
		ph.EXPECT().MaxGridCells().Return(uint32(128 * 128 * 128)).Maybe()
		ph.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Return().Maybe()
		ph.EXPECT().SetStagingBuffer(mock.Anything).Return().Maybe()
		ph.EXPECT().RegisterBody(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(0).Maybe()
		ph.EXPECT().BodyIndex(mock.Anything).Return(0, true).Maybe()
		ph.EXPECT().BodyParticleInfo(mock.Anything).Return(uint32(0), uint32(0)).Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).Return("").Maybe()

		obj, _, _ := newPhysicsAddableObject("panic_pipe")
		suite.Panics(func() {
			s.Add(obj)
		})
	})

	suite.Run("panics when CreateBuffer for staging fails", func() {
		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}
		ph := &physicsmocks.MockPhysics{}

		bgp := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(bgp).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		r.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		r.EXPECT().Pipeline(mock.Anything).RunAndReturn(func(key string) pipeline.Pipeline { return nil }).Maybe()
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(nil, fmt.Errorf("staging buffer creation failed")).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		s := scene.NewScene("test", cam, r, scene.WithPhysics())
		s.SetPhysicsHandler(ph)

		bufBGP := bind_group_provider.NewBindGroupProvider("physics_buffers")
		ph.EXPECT().Buffers().Return(bufBGP).Maybe()
		stageNames := []string{
			"particle_values", "aabb_reduce", "grid_build_params", "grid_clear",
			"grid_insert", "collision", "momenta", "integrate", "sync",
		}
		for _, name := range stageNames {
			ph.EXPECT().Bgp(name).Return(bind_group_provider.NewBindGroupProvider("physics_" + name)).Maybe()
		}
		ph.EXPECT().MaxBodies().Return(uint32(256)).Maybe()
		ph.EXPECT().MaxParticles().Return(uint32(2048)).Maybe()
		ph.EXPECT().MaxGridCells().Return(uint32(128 * 128 * 128)).Maybe()
		pipelineKeys := make(map[string]string)
		ph.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Run(func(name, key string) {
			pipelineKeys[name] = key
		}).Return().Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).RunAndReturn(func(key string) string {
			return pipelineKeys[key]
		}).Maybe()
		ph.EXPECT().SetStagingBuffer(mock.Anything).Return().Maybe()
		ph.EXPECT().RegisterBody(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(0).Maybe()
		ph.EXPECT().BodyIndex(mock.Anything).Return(0, true).Maybe()
		ph.EXPECT().BodyParticleInfo(mock.Anything).Return(uint32(0), uint32(0)).Maybe()

		obj, _, _ := newPhysicsAddableObject("panic_buf")
		suite.Panics(func() {
			s.Add(obj)
		})
	})
}

func (suite *sceneTest) TestInitLightingErrors() {
	suite.Run("initShadowMap panics on failed CreateVSMTextures", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		// CreateVSMTextures fails
		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil, fmt.Errorf("vsm tex failed")).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler))

		// AddLight triggers initLighting which calls initShadowMap
		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()

		suite.Panics(func() {
			s.AddLight(l)
		})
		_ = captured
	})

	suite.Run("initLightCullResources panics when InitBindGroup fails for cull BGP", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		// First InitBindGroup succeeds (for camera), fail on subsequent ones after light setup
		initCallCount := 0
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			initCallCount++
			// Fail on the light cull bind group (6th call approximately:
			// 1=camera, 2=lights, 3=shadow_data, 4=shadow_lit, 5=light_cull)
			if initCallCount == 5 {
				return fmt.Errorf("cull bind group failed")
			}
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()
		r.EXPECT().RegisterVSMShadowPipeline(mock.Anything).Return(nil).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler), scene.WithScreenSize(800, 600))

		l := &lightmocks.MockLight{}
		l.EXPECT().Enabled().Return(true).Maybe()
		l.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		l.EXPECT().CastsShadows().Return(true).Maybe()

		suite.Panics(func() {
			s.AddLight(l)
		})
		_ = captured
	})

	suite.Run("initShadowMap panics on failed CreateLinearSampler", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		capturePipelines(r)
		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(nil, fmt.Errorf("sampler failed")).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler))

		l := &lightmocks.MockLight{}
		suite.Panics(func() {
			s.AddLight(l)
		})
	})

	suite.Run("initShadowMap panics on failed RegisterVSMShadowPipeline", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		capturePipelines(r)
		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()
		r.EXPECT().RegisterVSMShadowPipeline(mock.Anything).Return(fmt.Errorf("vsm shadow pipeline failed")).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler))

		l := &lightmocks.MockLight{}
		suite.Panics(func() {
			s.AddLight(l)
		})
	})

	suite.Run("initShadowMap panics on failed skinned RegisterVSMShadowPipeline", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		capturePipelines(r)
		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()

		// Let the 3 static VSM shadow pipeline registrations succeed, then fail on the first skinned one
		shadowCallCount := 0
		r.EXPECT().RegisterVSMShadowPipeline(mock.Anything).RunAndReturn(func(p pipeline.Pipeline) error {
			shadowCallCount++
			if shadowCallCount > 3 {
				return fmt.Errorf("skinned vsm shadow pipeline failed")
			}
			return nil
		}).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler))

		l := &lightmocks.MockLight{}
		suite.Panics(func() {
			s.AddLight(l)
		})
	})

	suite.Run("initLightBindGroup panics on failed InitBindGroup", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		// Call #1 = camera (NewScene), call #2 = initLightBindGroup → fail
		initCallCount := 0
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			initCallCount++
			if initCallCount == 2 {
				return fmt.Errorf("light bind group failed")
			}
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler))

		l := &lightmocks.MockLight{}
		suite.Panics(func() {
			s.AddLight(l)
		})
	})

	suite.Run("initShadowMap panics on failed shadow data InitBindGroup", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		// Call #1 = camera, call #2 = lights, call #3 = shadow_data → fail
		initCallCount := 0
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			initCallCount++
			if initCallCount == 3 {
				return fmt.Errorf("shadow data bind group failed")
			}
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler))

		l := &lightmocks.MockLight{}
		suite.Panics(func() {
			s.AddLight(l)
		})
	})

	suite.Run("initShadowLitBindGroup panics on failed InitBindGroup", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		// Call #1 = camera, #2 = lights, #3 = shadow_data, #4 = shadow_lit → fail
		initCallCount := 0
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			initCallCount++
			if initCallCount == 4 {
				return fmt.Errorf("shadow lit bind group failed")
			}
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		capturePipelines(r)
		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()
		r.EXPECT().RegisterVSMShadowPipeline(mock.Anything).Return(nil).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler))

		l := &lightmocks.MockLight{}
		suite.Panics(func() {
			s.AddLight(l)
		})
	})

	suite.Run("initLightCullResources panics on tile BGP InitBindGroup", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		// Call #1=camera, #2=lights, #3=shadow_data, #4=shadow_lit, #5=cull, #6=tile → fail
		initCallCount := 0
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			initCallCount++
			if initCallCount == 6 {
				return fmt.Errorf("tile bind group failed")
			}
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		captured := capturePipelines(r)
		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()
		r.EXPECT().RegisterVSMShadowPipeline(mock.Anything).Return(nil).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler), scene.WithScreenSize(800, 600))

		l := &lightmocks.MockLight{}
		suite.Panics(func() {
			s.AddLight(l)
		})
		_ = captured
	})

	suite.Run("initLightCullResources panics on failed RegisterPipelines", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()
		r.EXPECT().RegisterVSMShadowPipeline(mock.Anything).Return(nil).Maybe()

		// RegisterPipelines fails for the cull compute pipeline
		r.EXPECT().RegisterPipelines(mock.Anything).Return(fmt.Errorf("cull pipeline failed")).Maybe()
		r.EXPECT().Pipeline(mock.Anything).Return(nil).Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler), scene.WithScreenSize(800, 600))

		l := &lightmocks.MockLight{}
		suite.Panics(func() {
			s.AddLight(l)
		})
	})

	suite.Run("reinitCameraBGPForLitPipeline panics on failed InitBindGroup", func() {
		handler := light.NewLightingHandler()

		cam := &cameramocks.MockCamera{}
		r := &renderermocks.MockRenderer{}

		camBGP := bind_group_provider.NewBindGroupProvider("cam_bgp")
		cam.EXPECT().BindGroupProvider().Return(camBGP).Maybe()
		cam.EXPECT().SetDelegate(mock.Anything).Maybe()

		// Call #1=camera, #2=lights, #3=shadow_data, #4=shadow_lit, #5=cull, #6=tile, #7=reinitCamera → fail
		initCallCount := 0
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(
			provider bind_group_provider.BindGroupProvider,
			desc wgpu.BindGroupLayoutDescriptor,
			usageOverrides map[int]wgpu.BufferUsage,
			sizeOverrides map[int]uint64,
		) error {
			initCallCount++
			if initCallCount == 7 {
				return fmt.Errorf("reinit camera bind group failed")
			}
			for _, entry := range desc.Entries {
				binding := int(entry.Binding)
				if provider.Buffer(binding) == nil {
					provider.SetBuffer(binding, &wgpu.Buffer{})
				}
			}
			return nil
		}).Maybe()

		captured := capturePipelines(r)
		r.EXPECT().CreateVSMTextures(mock.Anything, mock.Anything).Return(&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		r.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()
		r.EXPECT().RegisterVSMShadowPipeline(mock.Anything).Return(nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		s := scene.NewScene("test", cam, r, scene.WithLighting(handler), scene.WithScreenSize(800, 600))

		l := &lightmocks.MockLight{}
		suite.Panics(func() {
			s.AddLight(l)
		})
		_ = captured
	})

	suite.Run("ReadMappedBuffer error skips ProcessReadback", func() {
		s, cam, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		obj, _, _ := newPhysicsAddableObject("phys_read_err")
		s.Add(obj)

		wirePipelineLookup(r, captured)

		stagingBuf := &wgpu.Buffer{}
		ph.EXPECT().Enabled().Return(true).Maybe()
		ph.EXPECT().ReadbackPending().Return(true).Maybe()
		ph.EXPECT().BodiesCount().Return(1).Maybe()
		ph.EXPECT().StagingBuffer().Return(stagingBuf).Maybe()
		r.EXPECT().ReadMappedBuffer(stagingBuf, mock.Anything, mock.Anything).Return(nil, fmt.Errorf("map failed")).Maybe()
		ph.EXPECT().ClearReadbackPending().Return().Maybe()

		ph.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Maybe()
		ph.EXPECT().StagedWriteData().Return(nil).Maybe()
		ph.EXPECT().ParticleCount().Return(0).Maybe()
		ph.EXPECT().PipelineKey(mock.Anything).Return("").Maybe()
		ph.EXPECT().ConsumeReadbackRequest().Return(false).Maybe()

		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()

		cam.EXPECT().Update().Return().Maybe()
		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		suite.NotPanics(func() {
			s.PrepareCompute(0.016)
		})

		// ProcessReadback should NOT have been called due to the error
		ph.AssertNotCalled(suite.T(), "ProcessReadback", mock.Anything)
		ph.AssertCalled(suite.T(), "ClearReadbackPending")
	})

	suite.Run("patchSyncMapEntry happy path writes sync data", func() {
		s, _, r, ph := newPhysicsScene("test")
		wirePhysicsMockForInit(ph)

		captured := capturePipelines(r)
		r.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(&wgpu.Buffer{}, nil).Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		mdl := &modelmocks.MockModel{}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().Name().Return("sync_happy").Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Maybe()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().ComputePipelineKey().Return("").Maybe()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{}).Maybe()
		mdl.EXPECT().EffectProvider().Return(nil).Maybe()

		rb := &physicsmocks.MockRigidBody{}
		rb.EXPECT().Kinematic().Return(false).Maybe()

		// Create two objects sharing same model to force swap-remove with a real animator
		obj1 := &gameobjectmocks.MockGameObject{}
		var obj1ID uint64
		var inst1 int
		var anim1 animator.Animator
		obj1.EXPECT().ID().RunAndReturn(func() uint64 { return obj1ID }).Maybe()
		obj1.EXPECT().SetID(mock.Anything).Run(func(id uint64) { obj1ID = id }).Return().Maybe()
		obj1.EXPECT().Model().Return(mdl).Maybe()
		obj1.EXPECT().Ephemeral().Return(false).Maybe()
		obj1.EXPECT().Light().Return(nil).Maybe()
		obj1.EXPECT().RigidBody().Return(rb).Maybe()
		obj1.EXPECT().TransformData().Return(
			[3]float32{0, 0, 0}, [3]float32{1, 1, 1},
			[3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj1.EXPECT().SetAnimator(mock.Anything).Run(func(a animator.Animator) { anim1 = a }).Return().Maybe()
		obj1.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { inst1 = id }).Return().Maybe()
		obj1.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return inst1 }).Maybe()
		obj1.EXPECT().Animator().RunAndReturn(func() animator.Animator { return anim1 }).Maybe()

		obj2 := &gameobjectmocks.MockGameObject{}
		var obj2ID uint64
		var inst2 int
		obj2.EXPECT().ID().RunAndReturn(func() uint64 { return obj2ID }).Maybe()
		obj2.EXPECT().SetID(mock.Anything).Run(func(id uint64) { obj2ID = id }).Return().Maybe()
		obj2.EXPECT().Model().Return(mdl).Maybe()
		obj2.EXPECT().Ephemeral().Return(false).Maybe()
		obj2.EXPECT().Light().Return(nil).Maybe()
		obj2.EXPECT().RigidBody().Return(rb).Maybe()
		obj2.EXPECT().TransformData().Return(
			[3]float32{1, 0, 0}, [3]float32{1, 1, 1},
			[3]float32{0, 0, 0}, [3]float32{0, 0, 0},
		).Maybe()
		obj2.EXPECT().SetAnimator(mock.Anything).Return().Maybe()
		obj2.EXPECT().SetAnimatorInstanceID(mock.Anything).Run(func(id int) { inst2 = id }).Return().Maybe()
		obj2.EXPECT().AnimatorInstanceID().RunAndReturn(func() int { return inst2 }).Maybe()
		obj2.EXPECT().Animator().RunAndReturn(func() animator.Animator { return anim1 }).Maybe()

		bodyIdx := 0
		ph.EXPECT().RegisterBody(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Unset()
		ph.EXPECT().RegisterBody(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).RunAndReturn(func(objID uint64, pos [3]float32, rot [3]float32, rb physics.RigidBody, instID uint32) int {
			idx := bodyIdx
			bodyIdx++
			return idx
		}).Maybe()

		id1 := s.Add(obj1)
		s.Add(obj2)
		suite.Equal(2, s.Count())

		// Remove first triggers swap-remove → patchSyncMapEntry(anim, swappedObj.ID(), removedIdx)
		// which takes the happy path: anim non-nil, BodyIndex found, syncAnimMap has entry
		s.Remove(id1)
		suite.Equal(1, s.Count())

		_ = captured
	})
}

func (suite *sceneTest) TestPrepareComposition() {
	suite.Run("no-op when lighting not initialized", func() {
		s, _, _ := newMinimalScene("test")
		s.PrepareComposition()
	})

	suite.Run("no-op when composition handler disabled", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.PrepareComposition()
	})

	suite.Run("no-op when renderer is nil", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.CompositionHandler().SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.SetRenderer(nil)
		s.PrepareComposition()
	})

	suite.Run("runs fullscreen composition pass", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.CompositionHandler().SetEnabled(true)
		handler.CompositionHandler().SetExposure(1.5)
		handler.CompositionHandler().SetPipelineKey("composition", "comp_pipe")

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginCompositionFrame().Return(nil).Maybe()
		r.EXPECT().BeginCompositionPass().Return().Maybe()
		r.EXPECT().CompositionDrawCall(mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndCompositionPass().Return().Maybe()
		r.EXPECT().EndCompositionFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareComposition()
		})
		r.AssertCalled(suite.T(), "CompositionDrawCall", "comp_pipe", mock.Anything)
	})

	suite.Run("returns early when BeginCompositionFrame fails", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.CompositionHandler().SetEnabled(true)

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginCompositionFrame().Return(fmt.Errorf("swapchain error")).Maybe()

		suite.NotPanics(func() {
			s.PrepareComposition()
		})
		r.AssertNotCalled(suite.T(), "BeginCompositionPass")
	})
}

func (suite *sceneTest) TestBeginHDRFrame() {
	suite.Run("returns error when composition not initialized", func() {
		s, _, _ := newMinimalScene("test")
		err := s.BeginHDRFrame()
		suite.Error(err)
	})

	suite.Run("returns error when composition handler disabled", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		err := s.BeginHDRFrame()
		suite.Error(err)
	})

	suite.Run("returns error when renderer is nil", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.CompositionHandler().SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.SetRenderer(nil)
		err := s.BeginHDRFrame()
		suite.Error(err)
	})

	suite.Run("renders to MSAA texture when sample count > 1 and MSAA view exists", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.CompositionHandler().SetEnabled(true)
		handler.CompositionHandler().SetMSAATextureView(&wgpu.TextureView{})
		handler.CompositionHandler().SetHDRTextureView(&wgpu.TextureView{})
		handler.CompositionHandler().SetDepthTextureView(&wgpu.TextureView{})

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().SampleCount().Return(uint32(4)).Maybe()
		r.EXPECT().BeginHDRFrame(mock.Anything, mock.Anything, mock.Anything, uint32(4)).Return(nil).Maybe()

		err := s.BeginHDRFrame()
		suite.NoError(err)
	})

	suite.Run("renders directly to HDR texture when sample count is 1", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.CompositionHandler().SetEnabled(true)
		handler.CompositionHandler().SetHDRTextureView(&wgpu.TextureView{})
		handler.CompositionHandler().SetDepthTextureView(&wgpu.TextureView{})

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().SampleCount().Return(uint32(1)).Maybe()
		r.EXPECT().BeginHDRFrame(mock.Anything, mock.Anything, mock.Anything, uint32(1)).Return(nil).Maybe()

		err := s.BeginHDRFrame()
		suite.NoError(err)
	})

	suite.Run("falls back to non-MSAA when MSAA view is nil", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.CompositionHandler().SetEnabled(true)
		handler.CompositionHandler().SetHDRTextureView(&wgpu.TextureView{})
		handler.CompositionHandler().SetDepthTextureView(&wgpu.TextureView{})

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().SampleCount().Return(uint32(4)).Maybe()
		r.EXPECT().BeginHDRFrame(mock.Anything, mock.Anything, mock.Anything, uint32(1)).Return(nil).Maybe()

		err := s.BeginHDRFrame()
		suite.NoError(err)
	})
}

func (suite *sceneTest) TestPrepareGBuffer() {
	suite.Run("no-op when lighting not initialized", func() {
		s, _, _ := newMinimalScene("test")
		s.PrepareGBuffer()
	})

	suite.Run("no-op when gbuffer handler disabled", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.PrepareGBuffer()
	})

	suite.Run("no-op when renderer is nil", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.GBufferHandler().SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.SetRenderer(nil)
		s.PrepareGBuffer()
	})

	suite.Run("returns early when BeginGBufferFrame fails", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.GBufferHandler().SetEnabled(true)

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().BeginGBufferFrame().Return(fmt.Errorf("device lost")).Maybe()

		suite.NotPanics(func() {
			s.PrepareGBuffer()
		})
	})

	suite.Run("completes gbuffer pass with no objects", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.GBufferHandler().SetEnabled(true)
		handler.GBufferHandler().SetNormalTextureView(&wgpu.TextureView{})
		handler.GBufferHandler().SetAlbedoTextureView(&wgpu.TextureView{})
		handler.GBufferHandler().SetDepthTextureView(&wgpu.TextureView{})

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().BeginGBufferFrame().Return(nil).Maybe()
		r.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndGBufferPass().Return().Maybe()
		r.EXPECT().EndGBufferFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareGBuffer()
		})
		r.AssertCalled(suite.T(), "EndGBufferFrame")
	})

	suite.Run("issues gbuffer draw call for object with mesh and material", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.GBufferHandler().SetEnabled(true)
		handler.GBufferHandler().SetPipelineKey("static", "gbuffer_static_pipe")
		handler.GBufferHandler().SetNormalTextureView(&wgpu.TextureView{})
		handler.GBufferHandler().SetAlbedoTextureView(&wgpu.TextureView{})
		handler.GBufferHandler().SetDepthTextureView(&wgpu.TextureView{})

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))
		captured := capturePipelines(r)
		r.EXPECT().InitMeshBuffers(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		meshBGP := bind_group_provider.NewBindGroupProvider("mesh")
		matBGP := bind_group_provider.NewBindGroupProvider("mat")

		mat := &materialmocks.MockMaterial{}
		mat.EXPECT().PipelineKey().Return("cube").Maybe()
		mat.EXPECT().BindGroupProvider().Return(matBGP).Maybe()
		mat.EXPECT().Provider(mock.Anything).Return(nil).Maybe()
		mat.EXPECT().FragmentShaderPath().Return("").Maybe()

		obj, mdl := newWiredObject("cube", false, []material.Material{mat})
		mdl.EXPECT().MeshProvider().Unset()
		mdl.EXPECT().MeshProvider().Return(meshBGP).Maybe()
		mdl.EXPECT().VertexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexData().Return([]byte{0, 0}).Maybe()
		mdl.EXPECT().IndexCount().Return(1).Maybe()

		s.Add(obj)

		wirePipelineLookup(r, captured)
		r.EXPECT().BeginGBufferFrame().Return(nil).Maybe()
		r.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().GBufferDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		r.EXPECT().EndGBufferPass().Return().Maybe()
		r.EXPECT().EndGBufferFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareGBuffer()
		})
		r.AssertCalled(suite.T(), "GBufferDrawCall", "gbuffer_static_pipe", mock.Anything, mock.Anything, mock.Anything)
	})
}

func (suite *sceneTest) TestPrepareSSAO() {
	suite.Run("no-op when lighting not initialized", func() {
		s, _, _ := newMinimalScene("test")
		s.PrepareSSAO()
	})

	suite.Run("no-op when ssao handler disabled", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.PrepareSSAO()
	})

	suite.Run("no-op when renderer is nil", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SSAOHandler().SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.SetRenderer(nil)
		s.PrepareSSAO()
	})

	suite.Run("no-op when camera is nil", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SSAOHandler().SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.SetCamera(nil)
		s.PrepareSSAO()
	})

	suite.Run("dispatches ssao compute and blur passes", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SSAOHandler().SetEnabled(true)
		handler.SSAOHandler().Resize(256, 256)
		handler.SSAOHandler().SetPipelineKey("ssao_compute", "ssao_comp_pipe")
		handler.SSAOHandler().SetPipelineKey("ssao_blur", "ssao_blur_pipe")

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))

		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareSSAO()
		})
		r.AssertCalled(suite.T(), "DispatchCompute", "ssao_comp_pipe", mock.Anything, mock.Anything)
		r.AssertCalled(suite.T(), "DispatchCompute", "ssao_blur_pipe", mock.Anything, mock.Anything)
	})

	suite.Run("returns early when BeginComputeFrame fails", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SSAOHandler().SetEnabled(true)
		handler.SSAOHandler().Resize(256, 256)

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))

		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().Controller().Return(nil).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(fmt.Errorf("device lost")).Maybe()

		suite.NotPanics(func() {
			s.PrepareSSAO()
		})
		r.AssertNotCalled(suite.T(), "DispatchCompute", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("uses half resolution gbuffer scale when enabled", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SSAOHandler().SetEnabled(true)
		handler.SSAOHandler().Resize(512, 512)
		handler.SSAOHandler().SetHalfResolution(true)
		handler.SSAOHandler().SetPipelineKey("ssao_compute", "ssao_half_pipe")
		handler.SSAOHandler().SetPipelineKey("ssao_blur", "ssao_blur_half_pipe")

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))

		cam.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Maybe()
		ctrl := &cameramocks.MockCameraController{}
		ctrl.EXPECT().Position().Return(float32(1), float32(2), float32(3)).Maybe()
		cam.EXPECT().Controller().Return(ctrl).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareSSAO()
		})
	})
}

func (suite *sceneTest) TestPrepareSSR() {
	suite.Run("no-op when lighting not initialized", func() {
		s, _, _ := newMinimalScene("test")
		s.PrepareSSR()
	})

	suite.Run("no-op when ssr handler disabled", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.PrepareSSR()
	})

	suite.Run("no-op when composition handler disabled", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SSRHandler().SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.PrepareSSR()
	})

	suite.Run("no-op when renderer is nil", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SSRHandler().SetEnabled(true)
		handler.CompositionHandler().SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.SetRenderer(nil)
		s.PrepareSSR()
	})

	suite.Run("no-op when camera is nil", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SSRHandler().SetEnabled(true)
		handler.CompositionHandler().SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.SetCamera(nil)
		s.PrepareSSR()
	})

	suite.Run("dispatches hiz pyramid and ssr compute", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SSRHandler().SetEnabled(true)
		handler.SSRHandler().Resize(256, 256)
		handler.SSRHandler().SetHiZMipCount(3)
		handler.SSRHandler().SetPipelineKey("hiz_init", "hiz_init_pipe")
		handler.SSRHandler().SetPipelineKey("hiz_downsample", "hiz_down_pipe")
		handler.SSRHandler().SetPipelineKey("ssr_compute", "ssr_comp_pipe")
		handler.CompositionHandler().SetEnabled(true)

		// Pre-create hiz_init and hiz_down BGPs expected by the code
		handler.SSRHandler().SetBgp("hiz_init", bind_group_provider.NewBindGroupProvider("hiz_init"))
		handler.SSRHandler().SetBgp("hiz_down_1", bind_group_provider.NewBindGroupProvider("hiz_down_1"))
		handler.SSRHandler().SetBgp("hiz_down_2", bind_group_provider.NewBindGroupProvider("hiz_down_2"))

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))

		cam.EXPECT().ProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().ViewMatrix().Return([16]float32{}).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareSSR()
		})
		r.AssertCalled(suite.T(), "DispatchCompute", "hiz_init_pipe", mock.Anything, mock.Anything)
		r.AssertCalled(suite.T(), "DispatchCompute", "ssr_comp_pipe", mock.Anything, mock.Anything)
	})

	suite.Run("returns early when BeginComputeFrame fails", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SSRHandler().SetEnabled(true)
		handler.SSRHandler().Resize(128, 128)
		handler.SSRHandler().SetHiZMipCount(2)
		handler.CompositionHandler().SetEnabled(true)

		s, cam, r := newMinimalScene("test", scene.WithLighting(handler))

		cam.EXPECT().ProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Maybe()
		cam.EXPECT().ViewMatrix().Return([16]float32{}).Maybe()

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(fmt.Errorf("device lost")).Maybe()

		suite.NotPanics(func() {
			s.PrepareSSR()
		})
		r.AssertNotCalled(suite.T(), "DispatchCompute", mock.Anything, mock.Anything, mock.Anything)
	})
}

func (suite *sceneTest) TestPrepareProbes() {
	suite.Run("no-op when probe grid is nil", func() {
		s, _, _ := newLitScene("test")
		s.PrepareProbes()
	})

	suite.Run("no-op when probe grid is disabled", func() {
		handler := light.NewLightingHandler(light.WithProbeGrid(light.NewIrradianceProbeGrid()))
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.PrepareProbes()
	})

	suite.Run("no-op when renderer is nil", func() {
		pg := light.NewIrradianceProbeGrid()
		pg.SetEnabled(true)
		handler := light.NewLightingHandler(light.WithProbeGrid(pg))
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.SetRenderer(nil)
		s.PrepareProbes()
	})

	suite.Run("no-op when no dirty probes", func() {
		pg := light.NewIrradianceProbeGrid()
		pg.SetEnabled(true)
		pg.ClearDirtyProbes()
		handler := light.NewLightingHandler(light.WithProbeGrid(pg))
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.PrepareProbes()
	})

	suite.Run("no-op when bake texture views are nil", func() {
		pg := light.NewIrradianceProbeGrid()
		pg.SetEnabled(true)
		handler := light.NewLightingHandler(light.WithProbeGrid(pg))
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.PrepareProbes()
	})

	suite.Run("no-op when bake camera bgp is nil", func() {
		pg := light.NewIrradianceProbeGrid(light.WithProbeGridCounts(2, 1, 1))
		pg.SetEnabled(true)
		pg.SetBakeColorTextureView(&wgpu.TextureView{})
		pg.SetBakeDepthTextureView(&wgpu.TextureView{})
		pg.SetBgp("probe_bake_camera", nil)
		handler := light.NewLightingHandler(light.WithProbeGrid(pg))
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.PrepareProbes()
	})

	suite.Run("bakes dirty probes up to rate limit", func() {
		pg := light.NewIrradianceProbeGrid(light.WithProbeGridCounts(2, 1, 1))
		pg.SetEnabled(true)
		pg.SetBakeColorTextureView(&wgpu.TextureView{})
		pg.SetBakeDepthTextureView(&wgpu.TextureView{})
		pg.SetProbeBuffer(&wgpu.Buffer{})
		pg.SetPipelineKey("static", "probe_bake_static")
		pg.SetPipelineKey("skinned", "probe_bake_skinned")
		pg.SetPipelineKey("sh_project", "probe_sh_project")
		handler := light.NewLightingHandler(light.WithProbeGrid(pg))
		handler.SetEnabled(true)

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginProbeBakeFrame().Return(nil).Maybe()
		r.EXPECT().BeginProbeBakePass(mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndProbeBakePass().Return().Maybe()
		r.EXPECT().EndProbeBakeFrame().Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareProbes()
		})
	})
}

func (suite *sceneTest) TestPrepareShadowBlur() {
	suite.Run("no-op when lighting disabled", func() {
		s, _, _ := newMinimalScene("test")
		s.PrepareShadowBlur()
	})

	suite.Run("no-op when renderer is nil", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		s, _, _ := newMinimalScene("test", scene.WithLighting(handler))
		s.SetRenderer(nil)
		s.PrepareShadowBlur()
	})

	suite.Run("dispatches standard vsm blur when pcss disabled", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SetPipelineKey("vsm_blur", "vsm_blur_pipe")

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareShadowBlur()
		})
		r.AssertCalled(suite.T(), "DispatchCompute", "vsm_blur_pipe", mock.Anything, mock.Anything)
	})

	suite.Run("returns early when BeginComputeFrame fails for vsm blur", func() {
		handler := light.NewLightingHandler()
		handler.SetEnabled(true)
		handler.SetPipelineKey("vsm_blur", "vsm_blur_pipe")

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(fmt.Errorf("device lost")).Maybe()

		suite.NotPanics(func() {
			s.PrepareShadowBlur()
		})
		r.AssertNotCalled(suite.T(), "DispatchCompute", mock.Anything, mock.Anything, mock.Anything)
	})

	suite.Run("dispatches SAT generation when pcss enabled", func() {
		handler := light.NewLightingHandler(light.WithPCSSEnabled(true))
		handler.SetEnabled(true)
		handler.SetPipelineKey("vsm_sat", "vsm_sat_pipe")

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(nil).Maybe()
		r.EXPECT().DispatchCompute(mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		r.EXPECT().EndComputeFrame().Return().Maybe()

		suite.NotPanics(func() {
			s.PrepareShadowBlur()
		})
		r.AssertCalled(suite.T(), "DispatchCompute", "vsm_sat_pipe", mock.Anything, mock.Anything)
	})

	suite.Run("returns early from SAT generation when BeginComputeFrame fails", func() {
		handler := light.NewLightingHandler(light.WithPCSSEnabled(true))
		handler.SetEnabled(true)
		handler.SetPipelineKey("vsm_sat", "vsm_sat_pipe")

		s, _, r := newMinimalScene("test", scene.WithLighting(handler))

		r.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		r.EXPECT().BeginComputeFrame().Return(fmt.Errorf("device lost")).Maybe()

		suite.NotPanics(func() {
			s.PrepareShadowBlur()
		})
		r.AssertNotCalled(suite.T(), "DispatchCompute", mock.Anything, mock.Anything, mock.Anything)
	})
}
