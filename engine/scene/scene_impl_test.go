package scene

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/cogentcore/webgpu/wgpu"

	"github.com/Carmen-Shannon/automation/tools/worker"
	"github.com/Carmen-Shannon/oxy-go/common"
	camera_mocks "github.com/Carmen-Shannon/oxy-go/engine/camera/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	game_object_mocks "github.com/Carmen-Shannon/oxy-go/engine/game_object/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	light_mocks "github.com/Carmen-Shannon/oxy-go/engine/light/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	model_mocks "github.com/Carmen-Shannon/oxy-go/engine/model/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	physics_mocks "github.com/Carmen-Shannon/oxy-go/engine/physics/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	animator_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/animator/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	bgp_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	material_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/material/mocks"
	renderer_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	pipeline_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	shader_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/shader/mocks"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

func TestRunSceneImplTests(t *testing.T) {
	suite.Run(t, new(sceneImplTest))
}

type sceneImplTest struct {
	suite.Suite
	rendererMock *renderer_mocks.MockRenderer
	scene        *scene
}

func (suite *sceneImplTest) SetupSuite() {
	for {
		if _, err := os.Stat("go.mod"); err == nil {
			break
		}
		if err := os.Chdir(".."); err != nil {
			suite.T().Fatalf("failed to locate module root: %v", err)
		}
	}
}

func (suite *sceneImplTest) SetupSubTest() {
	suite.rendererMock = renderer_mocks.NewMockRenderer(suite.T())
	suite.scene = &scene{
		mu:                     &sync.RWMutex{},
		r:                      suite.rendererMock,
		lightHandler:           light.NewLightingHandler(),
		maxBonesGPU:            64,
		drawGroupProvidersPool: make(map[int]bind_group_provider.BindGroupProvider),
	}
}

func (suite *sceneImplTest) TestGenerateSSAOKernel() {
	suite.Run("clamps sampleCount below 1 to 1", func() {
		buf := suite.scene.generateSSAOKernel(0)
		suite.NotNil(buf)
		maxSamples := suite.scene.lightHandler.SSAOHandler().MaxSamples()
		suite.Len(buf, maxSamples*16)
	})

	suite.Run("clamps sampleCount above MaxSamples to MaxSamples", func() {
		maxSamples := suite.scene.lightHandler.SSAOHandler().MaxSamples()
		buf := suite.scene.generateSSAOKernel(9999)
		suite.Len(buf, maxSamples*16)
	})

	suite.Run("returns correct buffer size for valid sampleCount", func() {
		maxSamples := suite.scene.lightHandler.SSAOHandler().MaxSamples()
		buf := suite.scene.generateSSAOKernel(8)
		suite.Len(buf, maxSamples*16)
	})
}

func (suite *sceneImplTest) TestShadowPipelineKey() {
	suite.Run("static back returns empty when no pipeline registered", func() {
		key := suite.scene.shadowPipelineKey(false, model.ShadowCullModeBack)
		suite.Equal("", key)
	})

	suite.Run("static front returns empty when no pipeline registered", func() {
		key := suite.scene.shadowPipelineKey(false, model.ShadowCullModeFront)
		suite.Equal("", key)
	})

	suite.Run("static none returns empty when no pipeline registered", func() {
		key := suite.scene.shadowPipelineKey(false, model.ShadowCullModeNone)
		suite.Equal("", key)
	})

	suite.Run("skinned back returns empty when no skinned pipeline registered", func() {
		key := suite.scene.shadowPipelineKey(true, model.ShadowCullModeBack)
		suite.Equal("", key)
	})

	suite.Run("skinned with registered skinned pipeline uses skinned prefix", func() {
		shadowMock := light_mocks.NewMockShadowHandler(suite.T())
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lhMock.EXPECT().ShadowHandler().Return(shadowMock).Times(2)
		shadowMock.EXPECT().PipelineKey("shadow_skinned_back").Return("shadow_skinned_back_key").Times(2)
		suite.scene.lightHandler = lhMock
		key := suite.scene.shadowPipelineKey(true, model.ShadowCullModeBack)
		suite.Equal("shadow_skinned_back_key", key)
	})
}

func (suite *sceneImplTest) TestBuildInjectionMap() {
	suite.Run("default scene populates standard keys", func() {
		suite.scene.buildInjectionMap()
		suite.NotNil(suite.scene.injections)
		suite.Equal("64u", suite.scene.injections["max_bones"])
		suite.Contains(suite.scene.injections, "tile_size")
		suite.Contains(suite.scene.injections, "max_lights_per_tile")
		suite.Contains(suite.scene.injections, "num_threads")
		suite.Contains(suite.scene.injections, "max_ssao_samples")
		suite.Contains(suite.scene.injections, "pcf_samples")
		_, hasSlots := suite.scene.injections["slots_per_cell"]
		suite.False(hasSlots)
		_, hasBodyIdx := suite.scene.injections["body_idx_mask"]
		suite.False(hasBodyIdx)
	})

	suite.Run("with physics handler slots_per_cell and body_idx_mask are set", func() {
		suite.scene.physicsHandler = physics.NewPhysics()
		suite.scene.buildInjectionMap()
		suite.NotEmpty(suite.scene.injections["slots_per_cell"])
		suite.NotEmpty(suite.scene.injections["body_idx_mask"])
	})
}

func (suite *sceneImplTest) TestComputeWorkgroupSize2D() {
	suite.Run("returns defaults when Pipeline returns nil", func() {
		suite.rendererMock.EXPECT().Pipeline("missing-key").Return(nil).Once()
		x, y := suite.scene.computeWorkgroupSize2D("missing-key", 8, 4)
		suite.Equal(uint32(8), x)
		suite.Equal(uint32(4), y)
	})

	suite.Run("returns defaults when Pipeline Shader returns nil", func() {
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("existing-key").Return(mockPipe).Once()
		x, y := suite.scene.computeWorkgroupSize2D("existing-key", 16, 16)
		suite.Equal(uint32(16), x)
		suite.Equal(uint32(16), y)
	})

	suite.Run("returns workgroup size from real pipeline", func() {
		suite.scene.buildInjectionMap()
		realShader := shader.NewShader(
			"test-compute",
			shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipeline := pipeline.NewPipeline("test-compute-key", pipeline.PipelineTypeCompute, pipeline.WithComputeShader(realShader))
		suite.rendererMock.EXPECT().Pipeline("test-compute-key").Return(realPipeline).Once()
		x, y := suite.scene.computeWorkgroupSize2D("test-compute-key", 1, 1)
		suite.Greater(x, uint32(0))
		_ = y
	})

	suite.Run("falls back defaultX when workgroup x is zero", func() {
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockShdr := shader_mocks.NewMockShader(suite.T())
		mockShdr.EXPECT().WorkgroupSize().Return([3]uint32{0, 4, 0}).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockShdr).Once()
		suite.rendererMock.EXPECT().Pipeline("key-x-zero").Return(mockPipe).Once()
		x, y := suite.scene.computeWorkgroupSize2D("key-x-zero", 8, 4)
		suite.Equal(uint32(8), x)
		suite.Equal(uint32(4), y)
	})

	suite.Run("falls back defaultY when workgroup y is zero", func() {
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockShdr := shader_mocks.NewMockShader(suite.T())
		mockShdr.EXPECT().WorkgroupSize().Return([3]uint32{4, 0, 0}).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockShdr).Once()
		suite.rendererMock.EXPECT().Pipeline("key-y-zero").Return(mockPipe).Once()
		x, y := suite.scene.computeWorkgroupSize2D("key-y-zero", 8, 16)
		suite.Equal(uint32(4), x)
		suite.Equal(uint32(16), y)
	})
}

func (suite *sceneImplTest) TestAddLight() {
	suite.Run("adds light to handler when lighting already enabled", func() {
		suite.scene.lightHandler.SetEnabled(true)

		l := light.NewLight(light.LightTypeDirectional)
		suite.scene.AddLight(l)

		suite.Len(suite.scene.lightHandler.Lights(), 1)
	})

	suite.Run("initializes lighting pipeline on first call and adds light", func() {
		suite.False(suite.scene.lightHandler.Enabled())

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(nil).Maybe()
		suite.scene.cam = camMock

		suite.scene.buildInjectionMap()

		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Maybe()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil, nil).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterShadowDepthPipeline(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		l := light.NewLight(light.LightTypeDirectional)
		suite.scene.AddLight(l)

		suite.True(suite.scene.lightHandler.Enabled())
		suite.Len(suite.scene.lightHandler.Lights(), 1)
	})
}

func (suite *sceneImplTest) TestResize() {
	suite.Run("propagates resize to all enabled lighting subsystems", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().SetAspect(float32(800) / float32(600)).Return().Once()
		suite.scene.cam = camMock

		suite.scene.lightHandler.SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.SSAOHandler().SetEnabled(true)
		suite.scene.lightHandler.CompositionHandler().SetEnabled(true)
		suite.scene.lightHandler.SSRHandler().SetEnabled(true)

		suite.rendererMock.EXPECT().Resize(800, 600).Return().Once()

		suite.scene.tileBufferCapacity = math.MaxInt32
		suite.scene.Resize(800, 600)

		suite.Equal(800, suite.scene.screenWidth)
		suite.Equal(600, suite.scene.screenHeight)
	})
}

func (suite *sceneImplTest) TestDetachLight() {
	suite.Run("removes light from handler when obj has a light and lightObjects is empty", func() {
		obj := game_object_mocks.NewMockGameObject(suite.T())
		l := light.NewLight(light.LightTypeDirectional)
		obj.EXPECT().Light().Return(l).Once()

		suite.scene.DetachLight(obj)

		suite.Len(suite.scene.lightObjects, 0)
	})

	suite.Run("removes obj from lightObjects when found", func() {
		obj := game_object_mocks.NewMockGameObject(suite.T())
		l := light.NewLight(light.LightTypeDirectional)
		obj.EXPECT().Light().Return(l).Once()

		suite.scene.lightObjects = []game_object.GameObject{obj}

		suite.scene.DetachLight(obj)

		suite.Len(suite.scene.lightObjects, 0)
	})
}

func (suite *sceneImplTest) TestPrepareGBuffer() {
	suite.Run("early return when BeginGBufferFrame errors", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(errors.New("fail")).Once()
		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("completes begin and end with empty animator pool", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator with zero instance count", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(0)).Once()
		mockMdl := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mockMdl: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator with nil model", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(nil).Once()
		mockMdl := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mockMdl: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator with nil mesh provider", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(nil).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator when pipeline key is empty", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		mockBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(mockBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator when camera BGP is nil", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock

		mockBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(mockBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator when model has no materials", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		mockMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(mockMeshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return(nil).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator when material BGP is nil", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		mockMat := material_mocks.NewMockMaterial(suite.T())
		mockMat.EXPECT().BindGroupProvider().Return(nil).Once()
		mockMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(mockMeshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("calls GBufferDrawCall when culling is disabled", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().GBufferDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockMat := material_mocks.NewMockMaterial(suite.T())
		mockMat.EXPECT().BindGroupProvider().Return(matBGP).Once()
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(false).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("culling enabled with empty compute key falls back to GBufferDrawCall", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().GBufferDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockMat := material_mocks.NewMockMaterial(suite.T())
		mockMat.EXPECT().BindGroupProvider().Return(matBGP).Once()
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("").Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("culling enabled with nil Pipeline falls back to GBufferDrawCall", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().Pipeline("compute-key").Return(nil).Once()
		suite.rendererMock.EXPECT().GBufferDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockMat := material_mocks.NewMockMaterial(suite.T())
		mockMat.EXPECT().BindGroupProvider().Return(matBGP).Once()
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("compute-key").Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("culling enabled with nil Shader falls back to GBufferDrawCall", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().GBufferDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("compute-key").Return(mockPipeline).Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockMat := material_mocks.NewMockMaterial(suite.T())
		mockMat.EXPECT().BindGroupProvider().Return(matBGP).Once()
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("compute-key").Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("culling: indirect buffer nil falls back to GBufferDrawCall", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().GBufferDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		suite.scene.buildInjectionMap()
		realShader := shader.NewShader("_gbuf_test_compute", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipeline := pipeline.NewPipeline("compute-key", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShader),
		)
		suite.rendererMock.EXPECT().Pipeline("compute-key").Return(realPipeline).Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockMat := material_mocks.NewMockMaterial(suite.T())
		mockMat.EXPECT().BindGroupProvider().Return(matBGP).Once()
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("compute-key").Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mockAnim.EXPECT().IndirectBuffer(3).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("calls GBufferDrawCallIndirect when indirect buffer available", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().GBufferDrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		suite.scene.buildInjectionMap()
		realShader := shader.NewShader("_gbuf_test_compute2", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipeline := pipeline.NewPipeline("compute-key2", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShader),
		)
		suite.rendererMock.EXPECT().Pipeline("compute-key2").Return(realPipeline).Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockMat := material_mocks.NewMockMaterial(suite.T())
		mockMat.EXPECT().BindGroupProvider().Return(matBGP).Once()
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("compute-key2").Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		indBuf := &wgpu.Buffer{}
		mockAnim.EXPECT().IndirectBuffer(3).Return(indBuf).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("culling: array<indirect_args> annotation triggers CutPrefix branch", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().GBufferDrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		const wgslArrayIndirect = `// Test fixture: minimal compute shader with array<indirect_args> binding.
//@oxy:include indirect_args
//@oxy:group 0 3 storage_read_write indirect_buf array<indirect_args>

@compute @workgroup_size(64)
fn main() {}`
		wgslPath := filepath.Join(suite.T().TempDir(), "test-compute-array-indirect.wgsl")
		suite.Require().NoError(os.WriteFile(wgslPath, []byte(wgslArrayIndirect), 0644))

		realShader := shader.NewShader("_gbuf_test_compute_array", shader.ShaderTypeCompute,
			wgslPath,
		)
		realPipeline := pipeline.NewPipeline("compute-key-array", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShader),
		)
		suite.rendererMock.EXPECT().Pipeline("compute-key-array").Return(realPipeline).Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockMat := material_mocks.NewMockMaterial(suite.T())
		mockMat.EXPECT().BindGroupProvider().Return(matBGP).Once()
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("compute-key-array").Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		indBuf := &wgpu.Buffer{}
		mockAnim.EXPECT().IndirectBuffer(3).Return(indBuf).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skinned model uses skinned pipeline key", func() {
		suite.scene.lightHandler.GBufferHandler().SetEnabled(true)
		suite.scene.lightHandler.GBufferHandler().SetPipelineKey("skinned", "gbuffer_skinned")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().GBufferDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockMat := material_mocks.NewMockMaterial(suite.T())
		mockMat.EXPECT().BindGroupProvider().Return(matBGP).Once()
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(true).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(false).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})
}

func (suite *sceneImplTest) TestPrepareSSAO() {
	suite.Run("early return when ssao handler is nil", func() {
		suite.scene.lightHandler = light.NewLightingHandler()
		suite.NotPanics(func() { suite.scene.PrepareSSAO() })
	})

	suite.Run("early return when ssao handler disabled", func() {
		suite.NotPanics(func() { suite.scene.PrepareSSAO() })
	})

	suite.Run("early return when renderer is nil", func() {
		suite.scene.lightHandler.SSAOHandler().SetEnabled(true)
		suite.scene.r = nil
		suite.NotPanics(func() { suite.scene.PrepareSSAO() })
	})

	suite.Run("early return when camera is nil", func() {
		suite.scene.lightHandler.SSAOHandler().SetEnabled(true)
		suite.NotPanics(func() { suite.scene.PrepareSSAO() })
	})

	suite.Run("full resolution nil controller both lookups dispatches succeed", func() {
		suite.scene.lightHandler.SSAOHandler().SetEnabled(true)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Twice()
		camMock.EXPECT().Controller().Return(nil).Twice()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(3)

		suite.NotPanics(func() { suite.scene.PrepareSSAO() })
	})

	suite.Run("full resolution nil controller second lookup dispatches succeed", func() {
		suite.scene.lightHandler.SSAOHandler().SetEnabled(true)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Twice()
		camMock.EXPECT().Controller().Return(nil).Twice()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(3)

		suite.NotPanics(func() { suite.scene.PrepareSSAO() })
	})

	suite.Run("half resolution with non-nil controller", func() {
		suite.scene.lightHandler.SSAOHandler().SetEnabled(true)
		suite.scene.lightHandler.SSAOHandler().SetHalfResolution(true)

		ctrl := camera_mocks.NewMockCameraController(suite.T())
		ctrl.EXPECT().Position().Return(float32(1), float32(2), float32(3)).Once()
		ctrl.EXPECT().Radius().Return(float32(5.0)).Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Twice()
		camMock.EXPECT().Controller().Return(ctrl).Twice()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(3)

		suite.NotPanics(func() { suite.scene.PrepareSSAO() })
	})
}

func (suite *sceneImplTest) TestPrepareContactShadows() {
	suite.Run("cam nil returns at first guard", func() {
		suite.NotPanics(func() { suite.scene.PrepareContactShadows() })
	})

	suite.Run("non-directional light no match returns", func() {
		suite.scene.lightHandler.ContactShadowHandler().SetTextureView(&wgpu.TextureView{})

		camMock := camera_mocks.NewMockCamera(suite.T())
		suite.scene.cam = camMock

		mockLight := light_mocks.NewMockLight(suite.T())
		mockLight.EXPECT().Enabled().Return(true).Once()
		mockLight.EXPECT().CastsShadows().Return(true).Once()
		mockLight.EXPECT().Type().Return(light.LightTypePoint).Once()
		suite.scene.lightHandler.AddLight(mockLight)

		suite.NotPanics(func() { suite.scene.PrepareContactShadows() })
	})

	suite.Run("directional light BeginComputeFrame error returns", func() {
		suite.scene.lightHandler.ContactShadowHandler().SetTextureView(&wgpu.TextureView{})

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		suite.scene.cam = camMock

		mockLight := light_mocks.NewMockLight(suite.T())
		mockLight.EXPECT().Enabled().Return(true).Once()
		mockLight.EXPECT().CastsShadows().Return(true).Once()
		mockLight.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		mockLight.EXPECT().Direction().Return([3]float32{0, -1, 0}).Once()
		suite.scene.lightHandler.AddLight(mockLight)

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return(errors.New("fail")).Once()

		suite.NotPanics(func() { suite.scene.PrepareContactShadows() })
	})

	suite.Run("directional light non-nil controller full dispatch", func() {
		suite.scene.lightHandler.ContactShadowHandler().SetTextureView(&wgpu.TextureView{})

		ctrl := camera_mocks.NewMockCameraController(suite.T())
		ctrl.EXPECT().Position().Return(float32(1), float32(2), float32(3)).Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().Controller().Return(ctrl).Once()
		suite.scene.cam = camMock

		mockLight := light_mocks.NewMockLight(suite.T())
		mockLight.EXPECT().Enabled().Return(true).Once()
		mockLight.EXPECT().CastsShadows().Return(true).Once()
		mockLight.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		mockLight.EXPECT().Direction().Return([3]float32{0, -1, 0}).Once()
		suite.scene.lightHandler.AddLight(mockLight)

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndComputeFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareContactShadows() })
	})
}

func (suite *sceneImplTest) TestPrepareSSR() {
	suite.Run("disabled returns early", func() {
		suite.NotPanics(func() { suite.scene.PrepareSSR() })
	})

	suite.Run("BeginComputeFrame error returns early", func() {
		suite.scene.lightHandler.SSRHandler().SetEnabled(true)
		suite.scene.lightHandler.CompositionHandler().SetEnabled(true)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return(errors.New("fail")).Once()

		suite.NotPanics(func() { suite.scene.PrepareSSR() })
	})

	suite.Run("default dimensions clamped to 1 no loop", func() {
		suite.scene.lightHandler.SSRHandler().SetEnabled(true)
		suite.scene.lightHandler.CompositionHandler().SetEnabled(true)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndComputeFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareSSR() })
	})

	suite.Run("mipCount 2 executes one loop iteration", func() {
		suite.scene.lightHandler.SSRHandler().SetEnabled(true)
		suite.scene.lightHandler.CompositionHandler().SetEnabled(true)
		suite.scene.lightHandler.SSRHandler().SetHiZMipCount(2)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(3)
		suite.rendererMock.EXPECT().EndComputeFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareSSR() })
	})
}

func (suite *sceneImplTest) TestPrepareComposition() {
	suite.Run("disabled returns early", func() {
		suite.NotPanics(func() { suite.scene.PrepareComposition() })
	})

	suite.Run("BeginCompositionFrame error returns early", func() {
		suite.scene.lightHandler.CompositionHandler().SetEnabled(true)

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginCompositionFrame().Return(errors.New("fail")).Once()

		suite.NotPanics(func() { suite.scene.PrepareComposition() })
	})

	suite.Run("full composition dispatch", func() {
		suite.scene.lightHandler.CompositionHandler().SetEnabled(true)

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginCompositionFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginCompositionPass().Return().Once()
		suite.rendererMock.EXPECT().CompositionDrawCall(mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().EndCompositionPass().Return().Once()
		suite.rendererMock.EXPECT().EndCompositionFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareComposition() })
	})
}

func (suite *sceneImplTest) TestBeginHDRFrame() {
	suite.Run("no-MSAA path", func() {
		suite.scene.lightHandler.CompositionHandler().SetEnabled(true)

		suite.rendererMock.EXPECT().SampleCount().Return(uint32(1)).Once()
		suite.rendererMock.EXPECT().BeginHDRFrame(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		suite.NoError(suite.scene.BeginHDRFrame())
	})

	suite.Run("MSAA path", func() {
		suite.scene.lightHandler.CompositionHandler().SetEnabled(true)
		suite.scene.lightHandler.CompositionHandler().SetMSAATextureView(&wgpu.TextureView{})

		suite.rendererMock.EXPECT().SampleCount().Return(uint32(4)).Once()
		suite.rendererMock.EXPECT().BeginHDRFrame(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		suite.NoError(suite.scene.BeginHDRFrame())
	})
}

func (suite *sceneImplTest) TestPrepareShadows() {
	suite.Run("disabled returns early", func() {
		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("no shadow work bails early", func() {
		suite.scene.lightHandler.SetEnabled(true)
		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("directional light nil controller buffer2 nil BeginShadowFrame error", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(nil).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(errors.New("shadow frame error")).Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("directional light non-nil controller buffer2 non-nil cascades no animators", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		ctrl := camera_mocks.NewMockCameraController(suite.T())
		ctrl.EXPECT().Position().Return(float32(1), float32(2), float32(3)).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(ctrl).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: skips zero instance count animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(0)).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: skips nil model animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(nil).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: skips animator with no cast shadows", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(false).Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: skips animator with nil mesh provider", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Twice()
		mockModel.EXPECT().MeshProvider().Return(nil).Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: skips animator with empty pipeline key", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Twice()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Twice()
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: ShadowDrawCall when culling disabled", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Twice()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Twice()
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Twice()
		mockAnim.EXPECT().CullingEnabled().Return(false).Twice()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: culling empty compute key falls back to ShadowDrawCall", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Twice()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Twice()
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mockModel.EXPECT().ComputePipelineKey().Return("").Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Twice()
		mockAnim.EXPECT().CullingEnabled().Return(true).Twice()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: culling nil Pipeline falls back to ShadowDrawCall", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Twice()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Twice()
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mockModel.EXPECT().ComputePipelineKey().Return("shadow-csm-nil-pipe").Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Twice()
		mockAnim.EXPECT().CullingEnabled().Return(true).Twice()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().Pipeline("shadow-csm-nil-pipe").Return(nil).Twice()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: culling nil Shader falls back to ShadowDrawCall", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mockPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipeline.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Twice()

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Twice()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Twice()
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mockModel.EXPECT().ComputePipelineKey().Return("shadow-csm-nil-shader").Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Twice()
		mockAnim.EXPECT().CullingEnabled().Return(true).Twice()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().Pipeline("shadow-csm-nil-shader").Return(mockPipeline).Twice()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: indirect buffer nil falls back to ShadowDrawCall", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.scene.buildInjectionMap()
		realShader := shader.NewShader("_shadow_csm_indirect_nil", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipeline := pipeline.NewPipeline("shadow-csm-indirect-nil", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShader),
		)

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Twice()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Twice()
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mockModel.EXPECT().ComputePipelineKey().Return("shadow-csm-indirect-nil").Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Twice()
		mockAnim.EXPECT().CullingEnabled().Return(true).Twice()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Twice()
		mockAnim.EXPECT().IndirectBuffer(3).Return(nil).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().Pipeline("shadow-csm-indirect-nil").Return(realPipeline).Twice()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: ShadowDrawCallIndirect when indirect available", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Once()
		dl.EXPECT().CastsShadows().Return(true).Once()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Once()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.scene.buildInjectionMap()
		realShader := shader.NewShader("_shadow_csm_indirect_ok", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipeline := pipeline.NewPipeline("shadow-csm-indirect-ok", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShader),
		)

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Twice()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Twice()
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mockModel.EXPECT().ComputePipelineKey().Return("shadow-csm-indirect-ok").Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Twice()
		mockAnim.EXPECT().CullingEnabled().Return(true).Twice()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Twice()
		mockAnim.EXPECT().IndirectBuffer(3).Return(&wgpu.Buffer{}).Twice()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().Pipeline("shadow-csm-indirect-ok").Return(realPipeline).Twice()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: skips zero instance count animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Twice()
		sl.EXPECT().CastsShadows().Return(true).Twice()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Twice()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Once()
		sl.EXPECT().Range().Return(float32(10)).Once()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Once()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(sl)

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(0)).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: skips nil model animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Twice()
		sl.EXPECT().CastsShadows().Return(true).Twice()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Twice()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Once()
		sl.EXPECT().Range().Return(float32(10)).Once()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Once()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(sl)

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(nil).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: skips animator with no cast shadows", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Twice()
		sl.EXPECT().CastsShadows().Return(true).Twice()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Twice()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Once()
		sl.EXPECT().Range().Return(float32(10)).Once()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Once()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(sl)

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: skips animator with nil mesh provider", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Twice()
		sl.EXPECT().CastsShadows().Return(true).Twice()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Twice()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Once()
		sl.EXPECT().Range().Return(float32(10)).Once()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Once()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(sl)

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Once()
		mockModel.EXPECT().MeshProvider().Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: skips animator with empty pipeline key", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Twice()
		sl.EXPECT().CastsShadows().Return(true).Twice()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Twice()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Once()
		sl.EXPECT().Range().Return(float32(10)).Once()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Once()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(sl)

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Once()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: ShadowDrawCall when culling disabled", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Twice()
		sl.EXPECT().CastsShadows().Return(true).Twice()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Twice()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Once()
		sl.EXPECT().Range().Return(float32(10)).Once()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Once()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(sl)

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Once()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(false).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: ShadowDrawCallIndirect when indirect available", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Twice()
		sl.EXPECT().CastsShadows().Return(true).Twice()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Twice()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Once()
		sl.EXPECT().Range().Return(float32(10)).Once()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Once()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(sl)

		suite.scene.buildInjectionMap()
		realShader := shader.NewShader("_shadow_spot_indirect_ok", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipeline := pipeline.NewPipeline("shadow-spot-indirect-ok", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShader),
		)

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Once()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("shadow-spot-indirect-ok").Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mockAnim.EXPECT().IndirectBuffer(3).Return(&wgpu.Buffer{}).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().Pipeline("shadow-spot-indirect-ok").Return(realPipeline).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().EndShadowPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face: skips zero instance count animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Times(3)
		pl.EXPECT().CastsShadows().Return(true).Times(3)
		pl.EXPECT().Type().Return(light.LightTypePoint).Times(3)
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		pl.EXPECT().Range().Return(float32(20)).Once()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Times(6)
		suite.scene.lightHandler.AddLight(pl)

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(0)).Times(6)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowPass().Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face: ShadowDrawCall when culling disabled", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Times(3)
		pl.EXPECT().CastsShadows().Return(true).Times(3)
		pl.EXPECT().Type().Return(light.LightTypePoint).Times(3)
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		pl.EXPECT().Range().Return(float32(20)).Once()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Times(6)
		suite.scene.lightHandler.AddLight(pl)

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(6)
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Times(6)
		mockModel.EXPECT().Skinned().Return(false).Times(6)
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(12)
		mockAnim.EXPECT().Model().Return(mockModel).Times(6)
		mockAnim.EXPECT().CullingEnabled().Return(false).Times(6)
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Times(6)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().ShadowDrawCall(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(6)
		suite.rendererMock.EXPECT().EndShadowPass().Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face: ShadowDrawCallIndirect when indirect available", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Times(3)
		pl.EXPECT().CastsShadows().Return(true).Times(3)
		pl.EXPECT().Type().Return(light.LightTypePoint).Times(3)
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		pl.EXPECT().Range().Return(float32(20)).Once()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Times(6)
		suite.scene.lightHandler.AddLight(pl)

		suite.scene.buildInjectionMap()
		realShader := shader.NewShader("_shadow_point_indirect_ok", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipeline := pipeline.NewPipeline("shadow-point-indirect-ok", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShader),
		)

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(6)
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Times(6)
		mockModel.EXPECT().Skinned().Return(false).Times(6)
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		mockModel.EXPECT().ComputePipelineKey().Return("shadow-point-indirect-ok").Times(6)
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(6)
		mockAnim.EXPECT().Model().Return(mockModel).Times(6)
		mockAnim.EXPECT().CullingEnabled().Return(true).Times(6)
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Times(6)
		mockAnim.EXPECT().IndirectBuffer(3).Return(&wgpu.Buffer{}).Times(6)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().Pipeline("shadow-point-indirect-ok").Return(realPipeline).Times(6)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(6)
		suite.rendererMock.EXPECT().EndShadowPass().Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face: break when slots exhausted", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl1 := light_mocks.NewMockLight(suite.T())
		pl1.EXPECT().Enabled().Return(true).Times(3)
		pl1.EXPECT().CastsShadows().Return(true).Times(3)
		pl1.EXPECT().Type().Return(light.LightTypePoint).Times(3)
		pl1.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		pl1.EXPECT().Range().Return(float32(20)).Once()
		pl1.EXPECT().ShadowBias().Return(float32(0.005)).Times(6)
		suite.scene.lightHandler.AddLight(pl1)

		pl2 := light_mocks.NewMockLight(suite.T())
		pl2.EXPECT().Enabled().Return(true).Twice()
		pl2.EXPECT().CastsShadows().Return(true).Twice()
		pl2.EXPECT().Type().Return(light.LightTypePoint).Twice()
		suite.scene.lightHandler.AddLight(pl2)

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowPass().Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("directional and spot: writes shadow entry buffer when Buffer4 non-nil", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(&wgpu.Buffer{}).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Twice()
		dl.EXPECT().CastsShadows().Return(true).Twice()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Twice()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Once()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(dl)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Once()
		sl.EXPECT().CastsShadows().Return(true).Once()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Once()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Once()
		sl.EXPECT().Range().Return(float32(10)).Once()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Once()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Once()
		suite.scene.lightHandler.AddLight(sl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(3)
		suite.rendererMock.EXPECT().EndShadowPass().Return().Times(3)
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point loop: non-point light continue before point light processes", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Times(3)
		sl.EXPECT().CastsShadows().Return(false).Times(3)
		suite.scene.lightHandler.AddLight(sl)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Times(3)
		pl.EXPECT().CastsShadows().Return(true).Times(3)
		pl.EXPECT().Type().Return(light.LightTypePoint).Times(3)
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		pl.EXPECT().Range().Return(float32(20)).Once()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Times(6)
		suite.scene.lightHandler.AddLight(pl)

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowPass().Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face depth pass: skips nil model animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Times(3)
		pl.EXPECT().CastsShadows().Return(true).Times(3)
		pl.EXPECT().Type().Return(light.LightTypePoint).Times(3)
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		pl.EXPECT().Range().Return(float32(20)).Once()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Times(6)
		suite.scene.lightHandler.AddLight(pl)

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(6)
		mockAnim.EXPECT().Model().Return(nil).Times(6)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowPass().Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face depth pass: skips animator with nil mesh provider", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Times(3)
		pl.EXPECT().CastsShadows().Return(true).Times(3)
		pl.EXPECT().Type().Return(light.LightTypePoint).Times(3)
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		pl.EXPECT().Range().Return(float32(20)).Once()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Times(6)
		suite.scene.lightHandler.AddLight(pl)

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(6)
		mockModel.EXPECT().MeshProvider().Return(nil).Times(6)
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(6)
		mockAnim.EXPECT().Model().Return(mockModel).Times(6)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowPass().Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face depth pass: skips animator with empty pipeline key", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Times(3)
		pl.EXPECT().CastsShadows().Return(true).Times(3)
		pl.EXPECT().Type().Return(light.LightTypePoint).Times(3)
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Once()
		pl.EXPECT().Range().Return(float32(20)).Once()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Times(6)
		suite.scene.lightHandler.AddLight(pl)

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(6)
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Times(6)
		mockModel.EXPECT().Skinned().Return(false).Times(6)
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(6)
		mockAnim.EXPECT().Model().Return(mockModel).Times(6)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().BeginShadowDepthPass(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowPass().Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})
}

func (suite *sceneImplTest) TestPrepareLightCulling() {
	suite.Run("disabled returns early", func() {
		suite.NotPanics(func() { suite.scene.PrepareLightCulling() })
	})

	suite.Run("nil cam returns early", func() {
		suite.scene.lightHandler.SetEnabled(true)
		suite.NotPanics(func() { suite.scene.PrepareLightCulling() })
	})

	suite.Run("begin compute error returns early", func() {
		suite.scene.lightHandler.SetEnabled(true)
		suite.scene.lightHandler.Resize(800, 600)

		mockLight := light_mocks.NewMockLight(suite.T())
		mockLight.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler.AddLight(mockLight)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Far().Return(float32(1000.0)).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return(errors.New("fail")).Once()

		suite.NotPanics(func() { suite.scene.PrepareLightCulling() })
	})

	suite.Run("full cull dispatch with enabled light", func() {
		suite.scene.lightHandler.SetEnabled(true)
		suite.scene.lightHandler.Resize(800, 600)

		mockLight := light_mocks.NewMockLight(suite.T())
		mockLight.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler.AddLight(mockLight)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Far().Return(float32(1000.0)).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndComputeFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareLightCulling() })
	})
}

func (suite *sceneImplTest) TestCountEphemeral() {
	suite.Run("nil pool returns zero", func() {
		suite.Equal(0, suite.scene.CountEphemeral())
	})

	suite.Run("pool with animator returns instance count", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(3)).Once()

		mapKey := model_mocks.NewMockModel(suite.T())

		suite.scene.animatorPool = map[model.Model][]animator.Animator{
			mapKey: {animMock},
		}

		suite.Equal(3, suite.scene.CountEphemeral())
	})

	suite.Run("pool with multiple animators sums counts", func() {
		animMock1 := animator_mocks.NewMockAnimator(suite.T())
		animMock1.EXPECT().InstanceCount().Return(uint32(2)).Once()
		animMock2 := animator_mocks.NewMockAnimator(suite.T())
		animMock2.EXPECT().InstanceCount().Return(uint32(5)).Once()

		mapKey := model_mocks.NewMockModel(suite.T())

		suite.scene.animatorPool = map[model.Model][]animator.Animator{
			mapKey: {animMock1, animMock2},
		}

		suite.Equal(7, suite.scene.CountEphemeral())
	})
}

func (suite *sceneImplTest) TestAddGameObject() {
	suite.Run("nil model panics", func() {
		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(nil).Once()
		suite.Panics(func() { suite.scene.AddGameObject(objMock) })
	})

	suite.Run("non-skinned non-lit new model createAnimator", func() {
		suite.scene.buildInjectionMap()
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		mdlMock := model_mocks.NewMockModel(suite.T())
		mdlMock.EXPECT().Skinned().Return(false).Maybe()
		mdlMock.EXPECT().Name().Return("testmodel").Maybe()
		mdlMock.EXPECT().BoundingRadius().Return(float32(0)).Once()
		mdlMock.EXPECT().MeshProvider().Return(nil).Once()
		mdlMock.EXPECT().RenderMaterials().Return(nil).Once()
		mdlMock.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()

		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(mdlMock).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(mock.Anything).Once()
		objMock.EXPECT().SetAnimatorInstanceID(mock.Anything).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(nil).Once()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
		suite.Len(suite.scene.animatorPool, 1)
	})

	suite.Run("non-skinned lit uses lit shaders", func() {
		suite.scene.lightHandler.SetEnabled(true)
		suite.scene.buildInjectionMap()
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(false).Once()
		poolKey.EXPECT().Name().Return("testlit").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()
		objMock.EXPECT().SetAnimatorInstanceID(0).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(nil).Once()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
	})

	suite.Run("skinned non-lit uses skeletal shaders", func() {
		suite.scene.buildInjectionMap()
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(true).Once()
		poolKey.EXPECT().Name().Return("testskeletal").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()
		objMock.EXPECT().SetAnimatorInstanceID(0).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(nil).Once()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
	})

	suite.Run("skinned lit uses lit-skinned shaders", func() {
		suite.scene.lightHandler.SetEnabled(true)
		suite.scene.buildInjectionMap()
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(true).Once()
		poolKey.EXPECT().Name().Return("testskllit").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()
		objMock.EXPECT().SetAnimatorInstanceID(0).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(nil).Once()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
	})

	suite.Run("full pool creates new animator", func() {
		suite.scene.buildInjectionMap()
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		fullPoolKey := model_mocks.NewMockModel(suite.T())
		fullPoolKey.EXPECT().Skinned().Return(false).Maybe()
		fullPoolKey.EXPECT().Name().Return("testfull").Maybe()
		fullPoolKey.EXPECT().BoundingRadius().Return(float32(0)).Once()
		fullPoolKey.EXPECT().MeshProvider().Return(nil).Once()
		fullPoolKey.EXPECT().RenderMaterials().Return(nil).Once()
		fullPoolKey.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()

		fullAnimMock := animator_mocks.NewMockAnimator(suite.T())
		fullAnimMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		fullAnimMock.EXPECT().MaxInstances().Return(uint32(1)).Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{fullPoolKey: {fullAnimMock}}

		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(fullPoolKey).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(mock.Anything).Once()
		objMock.EXPECT().SetAnimatorInstanceID(mock.Anything).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(nil).Once()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
	})

	suite.Run("AddInstance error panics", func() {
		suite.scene.buildInjectionMap()
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(false).Once()
		poolKey.EXPECT().Name().Return("testerr").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), errors.New("AddInstance failed")).Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Maybe()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()

		suite.Panics(func() { suite.scene.AddGameObject(objMock) })
	})

	suite.Run("ephemeral obj skips registry", func() {
		suite.scene.buildInjectionMap()
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(false).Once()
		poolKey.EXPECT().Name().Return("testephemeral").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()
		objMock.EXPECT().SetAnimatorInstanceID(0).Once()
		objMock.EXPECT().Ephemeral().Return(true).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(nil).Once()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
		suite.Empty(suite.scene.registry)
	})

	suite.Run("obj with light tracks it", func() {
		suite.scene.buildInjectionMap()
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(false).Once()
		poolKey.EXPECT().Name().Return("testlight").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		l := light.NewLight(light.LightTypePoint)

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()
		objMock.EXPECT().SetAnimatorInstanceID(0).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(l).Once()
		objMock.EXPECT().RigidBody().Return(nil).Once()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
		suite.Len(suite.scene.lightObjects, 1)
		suite.Len(suite.scene.lightHandler.Lights(), 1)
	})

	suite.Run("physics gpuReady=true pre-set sync maps", func() {
		suite.scene.buildInjectionMap()
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(false).Maybe()
		poolKey.EXPECT().Name().Return("testphys").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		syncBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		suite.scene.physicsHandler = physics.NewPhysics()
		suite.scene.physicsGPUReady = true
		suite.scene.physicsSyncAnimMap = map[animator.Animator]int{animMock: 0}
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: syncBGPMock}

		rb := physics.NewRigidBody()

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Maybe()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()
		objMock.EXPECT().SetAnimatorInstanceID(0).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(rb).Maybe()
		objMock.EXPECT().AnimatorInstanceID().Return(int(0)).Maybe()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
	})

	suite.Run("physics gpuReady=false nil sync maps triggers initPhysics", func() {
		suite.scene.buildInjectionMap()

		syncShader := shader.NewShader("_test_sync", shader.ShaderTypeCompute,
			"engine/physics/assets/physics-sync.wgsl", shader.WithInjections(suite.scene.injections))
		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(syncShader).Maybe()

		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(nil, nil).Maybe()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(pipeMock).Maybe()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(false).Maybe()
		poolKey.EXPECT().Name().Return("testphysinit").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		computeBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		computeBGPMock.EXPECT().Buffer(mock.Anything).Return(nil).Maybe()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGPMock).Maybe()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)
		suite.scene.physicsHandler = physics.NewPhysics()
		suite.scene.physicsGPUReady = false

		rb := physics.NewRigidBody()

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Maybe()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()
		objMock.EXPECT().SetAnimatorInstanceID(0).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(rb).Maybe()
		objMock.EXPECT().AnimatorInstanceID().Return(int(0)).Maybe()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
		suite.True(suite.scene.physicsGPUReady)
	})

	suite.Run("physics kinematic triggers bone particle group early return", func() {
		suite.scene.buildInjectionMap()
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(true).Maybe()
		poolKey.EXPECT().Name().Return("testkinematic").Maybe()
		poolKey.EXPECT().Skeleton().Return(&model.Skeleton{Bones: []model.Bone{{Name: "root"}}}).Once()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		syncBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.physicsHandler = physics.NewPhysics()
		suite.scene.physicsGPUReady = true
		suite.scene.physicsSyncAnimMap = map[animator.Animator]int{animMock: 0}
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: syncBGPMock}

		rb := physics.NewRigidBody()
		rb.SetKinematic(true)

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Maybe()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()
		objMock.EXPECT().SetAnimatorInstanceID(0).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(rb).Maybe()
		objMock.EXPECT().AnimatorInstanceID().Return(int(0)).Maybe()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
		suite.Len(suite.scene.boneParticleUpdateGroups, 0)
	})
}

func (suite *sceneImplTest) TestRemoveGameObject() {
	suite.Run("id not in registry returns early", func() {
		suite.NotPanics(func() { suite.scene.RemoveGameObject(1) })
	})

	suite.Run("id in registry no light no anim no physics", func() {
		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().Animator().Return(nil).Once()

		suite.scene.registry = map[uint64]game_object.GameObject{1: objMock}

		suite.scene.RemoveGameObject(1)
		suite.Empty(suite.scene.registry)
	})

	suite.Run("has light obj in lightObjects removes it", func() {
		l := light.NewLight(light.LightTypePoint)
		suite.scene.lightHandler.AddLight(l)

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Light().Return(l).Once()
		objMock.EXPECT().Animator().Return(nil).Once()

		suite.scene.registry = map[uint64]game_object.GameObject{1: objMock}
		suite.scene.lightObjects = []game_object.GameObject{objMock}

		suite.scene.RemoveGameObject(1)
		suite.Empty(suite.scene.lightObjects)
	})

	suite.Run("anim non-nil removedIdx negative skips swap", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().Animator().Return(animMock).Once()
		objMock.EXPECT().AnimatorInstanceID().Return(-1).Once()

		suite.scene.registry = map[uint64]game_object.GameObject{1: objMock}

		suite.NotPanics(func() { suite.scene.RemoveGameObject(1) })
	})

	suite.Run("anim non-nil removedIdx zero lut nil swapped false", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().RemoveInstance(uint32(0)).Return(uint32(0), false).Once()

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().Animator().Return(animMock).Once()
		objMock.EXPECT().AnimatorInstanceID().Return(0).Once()
		objMock.EXPECT().SetAnimatorInstanceID(-1).Once()

		suite.scene.registry = map[uint64]game_object.GameObject{1: objMock}

		suite.NotPanics(func() { suite.scene.RemoveGameObject(1) })
	})

	suite.Run("anim non-nil lut non-nil swapped false deletes from lut", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().RemoveInstance(uint32(0)).Return(uint32(0), false).Once()

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().Animator().Return(animMock).Once()
		objMock.EXPECT().AnimatorInstanceID().Return(0).Once()
		objMock.EXPECT().SetAnimatorInstanceID(-1).Once()

		suite.scene.registry = map[uint64]game_object.GameObject{1: objMock}
		suite.scene.instanceLookup = map[animator.Animator]map[uint32]uint64{
			animMock: {uint32(0): uint64(1)},
		}

		suite.scene.RemoveGameObject(1)
		suite.Empty(suite.scene.instanceLookup[animMock])
	})

	suite.Run("full swap path swappedObjID in registry updates instance id", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().RemoveInstance(uint32(0)).Return(uint32(1), true).Once()

		swappedObjMock := game_object_mocks.NewMockGameObject(suite.T())
		swappedObjMock.EXPECT().SetAnimatorInstanceID(0).Once()
		swappedObjMock.EXPECT().ID().Return(uint64(2)).Once()

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().Animator().Return(animMock).Once()
		objMock.EXPECT().AnimatorInstanceID().Return(0).Once()
		objMock.EXPECT().SetAnimatorInstanceID(-1).Once()

		suite.scene.registry = map[uint64]game_object.GameObject{1: objMock, 2: swappedObjMock}
		suite.scene.instanceLookup = map[animator.Animator]map[uint32]uint64{
			animMock: {uint32(0): uint64(1), uint32(1): uint64(2)},
		}

		suite.NotPanics(func() { suite.scene.RemoveGameObject(1) })
		suite.Equal(uint64(2), suite.scene.instanceLookup[animMock][uint32(0)])
	})

	suite.Run("physics non-nil calls RemoveBody", func() {
		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().Animator().Return(nil).Once()
		objMock.EXPECT().ID().Return(uint64(1)).Twice()

		suite.scene.registry = map[uint64]game_object.GameObject{1: objMock}
		suite.scene.physicsHandler = physics.NewPhysics()

		suite.NotPanics(func() { suite.scene.RemoveGameObject(1) })
	})
}

func (suite *sceneImplTest) TestPrepareCompute() {

	suite.Run("cam nil camBGP skips WriteBuffers", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("cam non-nil camBGP nil controller writes uniform", func() {
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("cam non-nil camBGP non-nil controller writes position", func() {
		ctrl := camera_mocks.NewMockCameraController(suite.T())
		ctrl.EXPECT().Position().Return(float32(1), float32(2), float32(3)).Once()
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().Controller().Return(ctrl).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("light object enabled syncs position", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		lMock := light_mocks.NewMockLight(suite.T())
		lMock.EXPECT().SetPosition(float32(1), float32(2), float32(3)).Return().Once()
		obj := game_object_mocks.NewMockGameObject(suite.T())
		obj.EXPECT().Light().Return(lMock).Once()
		obj.EXPECT().Enabled().Return(true).Once()
		obj.EXPECT().Position().Return(float32(1), float32(2), float32(3)).Once()
		suite.scene.lightObjects = []game_object.GameObject{obj}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("light object disabled skips sync", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		lMock := light_mocks.NewMockLight(suite.T())
		obj := game_object_mocks.NewMockGameObject(suite.T())
		obj.EXPECT().Light().Return(lMock).Once()
		obj.EXPECT().Enabled().Return(false).Once()
		suite.scene.lightObjects = []game_object.GameObject{obj}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("light object nil light skips", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		obj := game_object_mocks.NewMockGameObject(suite.T())
		obj.EXPECT().Light().Return(nil).Once()
		suite.scene.lightObjects = []game_object.GameObject{obj}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("light handler enabled single binding write", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		lightsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(lightsBGP).Once()
		mockLH.EXPECT().Lights().Return(nil).Once()
		mockLH.EXPECT().MaxGPULights().Return(100).Once()
		mockLH.EXPECT().MarshalLightBuffer(mock.Anything, mock.Anything).Return(make([]byte, 16)).Once()
		suite.scene.lightHandler = mockLH
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("light handler enabled two bindings when data over 16", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		lightsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(lightsBGP).Once()
		mockLH.EXPECT().Lights().Return(nil).Once()
		mockLH.EXPECT().MaxGPULights().Return(100).Once()
		mockLH.EXPECT().MarshalLightBuffer(mock.Anything, mock.Anything).Return(make([]byte, 32)).Once()
		suite.scene.lightHandler = mockLH
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("light sort triggers when rawLights exceeds MaxGPULights", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		l1 := light.NewLight(light.LightTypePoint)
		l2 := light.NewLight(light.LightTypePoint)
		lightsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(lightsBGP).Once()
		mockLH.EXPECT().Lights().Return([]light.Light{l1, l2}).Once()
		mockLH.EXPECT().MaxGPULights().Return(1).Once()
		mockLH.EXPECT().MarshalLightBuffer(mock.Anything, mock.Anything).Return(make([]byte, 16)).Once()
		suite.scene.lightHandler = mockLH
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("light sort non-nil controller uses camera position", func() {
		ctrl := camera_mocks.NewMockCameraController(suite.T())
		ctrl.EXPECT().Position().Return(float32(5), float32(5), float32(5)).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		camMock.EXPECT().Controller().Return(ctrl).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		l1 := light.NewLight(light.LightTypePoint)
		l2 := light.NewLight(light.LightTypePoint)
		lightsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(lightsBGP).Once()
		mockLH.EXPECT().Lights().Return([]light.Light{l1, l2}).Once()
		mockLH.EXPECT().MaxGPULights().Return(1).Once()
		mockLH.EXPECT().MarshalLightBuffer(mock.Anything, mock.Anything).Return(make([]byte, 16)).Once()
		suite.scene.lightHandler = mockLH
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 zero instance count skipped", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Times(4)
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 nil model skipped", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(nil).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 empty compute key skipped", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("").Times(2)
		mockModel.EXPECT().MeshProvider().Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 nil pipeline skipped", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k14").Times(3)
		mockModel.EXPECT().MeshProvider().Return(mockMeshBGP).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		suite.rendererMock.EXPECT().Pipeline("k14").Return(nil).Times(3)
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 nil shader skipped", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Times(3)
		mockMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k15").Times(3)
		mockModel.EXPECT().MeshProvider().Return(mockMeshBGP).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		suite.rendererMock.EXPECT().Pipeline("k15").Return(mockPipe).Times(3)
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 full path culling disabled no SetFrustumPlanes", func() {
		suite.scene.cullingDisabled = true
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)
		suite.scene.buildInjectionMap()
		realShdr := shader.NewShader("_pc16", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipe := pipeline.NewPipeline("pc16-key", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShdr),
		)
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("pc16-key").Times(2)
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(5)
		animMock.EXPECT().Model().Return(mockModel).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(1)).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("pc16-key").Return(realPipe).Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "pc16-key" && d[0].Provider == computeBGP
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 full path culling enabled calls SetFrustumPlanes and ResetIndirectArgs", func() {
		suite.scene.cullingDisabled = false
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)
		suite.scene.buildInjectionMap()
		realShdr := shader.NewShader("_pc17", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipe := pipeline.NewPipeline("pc17-key", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShdr),
		)
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGPMock.EXPECT().IndexCount().Return(10).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("pc17-key").Times(3)
		mockModel.EXPECT().MeshProvider().Return(meshBGPMock).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(5)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().SetFrustumPlanes(mock.Anything).Return().Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(1)).Once()
		animMock.EXPECT().ResetIndirectArgs(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("pc17-key").Return(realPipe).Times(3)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "pc17-key" && d[0].Provider == computeBGP
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 culling nil model falls through to StagedWriteData", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(nil).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 culling nil mesh provider falls through to StagedWriteData", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("").Times(2)
		mockModel.EXPECT().MeshProvider().Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 culling empty key continue skips StagedWriteData", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		meshBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("").Times(3)
		mockModel.EXPECT().MeshProvider().Return(meshBGPMock).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 culling nil pipeline continue skips StagedWriteData", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k21").Times(3)
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		suite.rendererMock.EXPECT().Pipeline("k21").Return(nil).Times(3)
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 culling nil shader continue skips StagedWriteData", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Times(3)
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k22").Times(3)
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		suite.rendererMock.EXPECT().Pipeline("k22").Return(mockPipe).Times(3)
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 culling valid calls ResetIndirectArgs StagedWriteData and triggers WriteBuffers", func() {
		suite.scene.cullingDisabled = false
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)
		suite.scene.buildInjectionMap()
		realShdr := shader.NewShader("_pc23", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipe := pipeline.NewPipeline("k23", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShdr),
		)
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP.EXPECT().IndexCount().Return(10).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k23").Times(3)
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(5)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().SetFrustumPlanes(mock.Anything).Return().Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(1)).Once()
		animMock.EXPECT().ResetIndirectArgs(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().StagedWriteData().Return([]bind_group_provider.BufferWrite{
			{Provider: computeBGP, Binding: 0, Data: []byte{1}},
		}).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k23").Return(realPipe).Times(3)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "k23" && d[0].Provider == computeBGP
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 non-empty StagedWriteData calls WriteBuffers", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(nil).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return([]bind_group_provider.BufferWrite{
			{Provider: bgpMock, Binding: 0, Offset: 0, Data: []byte{1}},
		}).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics nil skips block", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		suite.scene.physicsHandler = nil
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics Enabled false skips", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(false).Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics Enabled ReadbackPending false substeps 0 no physWrites", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics substeps 0 with physWrites calls WriteBuffers", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Once()
		phMock.EXPECT().StagedWriteData().Return([]bind_group_provider.BufferWrite{{Binding: 0}}).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics physicsSyncWrites appended on substeps 0", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		suite.scene.physicsSyncWrites = []bind_group_provider.BufferWrite{{Binding: 1}}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
		suite.Len(suite.scene.physicsSyncWrites, 0)
	})

	suite.Run("physics ReadbackPending true BodiesCount 0 no ReadMappedBuffer", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(true).Once()
		phMock.EXPECT().BodiesCount().Return(0).Once()
		phMock.EXPECT().ClearReadbackPending().Return().Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics ReadbackPending true ReadMappedBuffer error no ProcessReadback", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stagingBuf := &wgpu.Buffer{}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(true).Once()
		phMock.EXPECT().BodiesCount().Return(1).Once()
		phMock.EXPECT().StagingBuffer().Return(stagingBuf).Once()
		phMock.EXPECT().ClearReadbackPending().Return().Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		suite.rendererMock.EXPECT().ReadMappedBuffer(mock.Anything, uint64(0), mock.Anything).Return(nil, errors.New("fail")).Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics ReadbackPending true ReadMappedBuffer success calls ProcessReadback", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stagingBuf := &wgpu.Buffer{}
		readData := make([]byte, 16)
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(true).Once()
		phMock.EXPECT().BodiesCount().Return(1).Once()
		phMock.EXPECT().StagingBuffer().Return(stagingBuf).Once()
		phMock.EXPECT().ProcessReadback(mock.Anything).Return().Once()
		phMock.EXPECT().ClearReadbackPending().Return().Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		suite.rendererMock.EXPECT().ReadMappedBuffer(mock.Anything, uint64(0), mock.Anything).Return(readData, nil).Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics substeps 1 dispatches 8 stages", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(1, make([]byte, 16)).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		phMock.EXPECT().ParticleCount().Return(4).Once()
		phMock.EXPECT().BodiesCount().Return(2).Once()
		phMock.EXPECT().MaxGridCells().Return(100).Once()
		phMock.EXPECT().Buffers().Return(bufBGP).Times(2)
		phMock.EXPECT().PipelineKey(mock.Anything).Return("phys_key").Times(8)
		phMock.EXPECT().Bgp(mock.Anything).Return(stageBGP).Times(8)
		phMock.EXPECT().ConsumeReadbackRequest().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("phys_key").Return(nil).Times(8)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(8)
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics substeps 1 physDispatchGroups nil shader returns 1 1 1", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Times(8)
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(1, make([]byte, 16)).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		phMock.EXPECT().ParticleCount().Return(4).Once()
		phMock.EXPECT().BodiesCount().Return(2).Once()
		phMock.EXPECT().MaxGridCells().Return(100).Once()
		phMock.EXPECT().Buffers().Return(bufBGP).Times(2)
		phMock.EXPECT().PipelineKey(mock.Anything).Return("phys_key").Times(8)
		phMock.EXPECT().Bgp(mock.Anything).Return(stageBGP).Times(8)
		phMock.EXPECT().ConsumeReadbackRequest().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("phys_key").Return(mockPipe).Times(8)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(8)
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics physicsSyncGroup dispatches sync after substeps", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		syncBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: syncBGP}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(1, make([]byte, 16)).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		phMock.EXPECT().ParticleCount().Return(4).Once()
		phMock.EXPECT().BodiesCount().Return(2).Once()
		phMock.EXPECT().MaxGridCells().Return(100).Once()
		phMock.EXPECT().Buffers().Return(bufBGP).Times(2)
		phMock.EXPECT().PipelineKey(mock.Anything).Return("phys_key").Times(9)
		phMock.EXPECT().Bgp(mock.Anything).Return(stageBGP).Times(8)
		phMock.EXPECT().ConsumeReadbackRequest().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("phys_key").Return(nil).Times(9)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(9)
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics ConsumeReadbackRequest true StagingBuffer nil no CopyBufferToBuffer", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(1, make([]byte, 16)).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		phMock.EXPECT().ParticleCount().Return(4).Once()
		phMock.EXPECT().BodiesCount().Return(2).Once()
		phMock.EXPECT().MaxGridCells().Return(100).Once()
		phMock.EXPECT().Buffers().Return(bufBGP).Times(2)
		phMock.EXPECT().PipelineKey(mock.Anything).Return("phys_key").Times(8)
		phMock.EXPECT().Bgp(mock.Anything).Return(stageBGP).Times(8)
		phMock.EXPECT().ConsumeReadbackRequest().Return(true).Once()
		phMock.EXPECT().StagingBuffer().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("phys_key").Return(nil).Times(8)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(8)
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics ConsumeReadbackRequest true StagingBuffer non-nil calls CopyBufferToBuffer", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stagingBuf := &wgpu.Buffer{}
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP.EXPECT().Buffer(0).Return(nil).Once()
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(1, make([]byte, 16)).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		phMock.EXPECT().ParticleCount().Return(4).Once()
		phMock.EXPECT().BodiesCount().Return(2).Once()
		phMock.EXPECT().MaxGridCells().Return(100).Once()
		phMock.EXPECT().Buffers().Return(bufBGP).Times(3)
		phMock.EXPECT().PipelineKey(mock.Anything).Return("phys_key").Times(8)
		phMock.EXPECT().Bgp(mock.Anything).Return(stageBGP).Times(8)
		phMock.EXPECT().ConsumeReadbackRequest().Return(true).Once()
		phMock.EXPECT().StagingBuffer().Return(stagingBuf).Once()
		suite.rendererMock.EXPECT().Pipeline("phys_key").Return(nil).Times(8)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(8)
		suite.rendererMock.EXPECT().CopyBufferToBuffer(mock.Anything, mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("bone particle physicsHandler nil skips", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		suite.scene.physicsHandler = nil
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.boneParticleUpdateGroups = []*boneParticleUpdateGroup{
			{bgp: bgpMock, particleCount: 5},
		}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("bone particle empty groups skips", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		suite.scene.physicsHandler = physics.NewPhysics()
		suite.scene.boneParticleUpdateGroups = nil
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("bone particle nil boneUpdatePipe skips DispatchCompute", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.boneParticleUpdateGroups = []*boneParticleUpdateGroup{
			{bgp: bgpMock, particleCount: 5},
		}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(false).Once()
		phMock.EXPECT().PipelineKey("bone_update").Return("bone_key").Once()
		suite.scene.physicsHandler = phMock
		suite.rendererMock.EXPECT().Pipeline("bone_key").Return(nil).Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("bone particle nil boneUpdateShader skips", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.boneParticleUpdateGroups = []*boneParticleUpdateGroup{
			{bgp: bgpMock, particleCount: 5},
		}
		mockBonePipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockBonePipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Once()
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(false).Once()
		phMock.EXPECT().PipelineKey("bone_update").Return("bone_key").Once()
		suite.scene.physicsHandler = phMock
		suite.rendererMock.EXPECT().Pipeline("bone_key").Return(mockBonePipe).Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("bone particle valid shader dispatches per group", func() {
		suite.scene.buildInjectionMap()
		realBoneShader := shader.NewShader("_bone_pc42", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realBonePipe := pipeline.NewPipeline("bone_key42", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realBoneShader),
		)
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		bgp1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgp2 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.boneParticleUpdateGroups = []*boneParticleUpdateGroup{
			{bgp: bgp1, particleCount: 5},
			{bgp: bgp2, particleCount: 3},
		}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(false).Once()
		phMock.EXPECT().PipelineKey("bone_update").Return("bone_key42").Once()
		suite.scene.physicsHandler = phMock
		suite.rendererMock.EXPECT().Pipeline("bone_key42").Return(realBonePipe).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 2 &&
				d[0].PipelineKey == "bone_key42" && d[0].Provider == bgp1 &&
				d[1].PipelineKey == "bone_key42" && d[1].Provider == bgp2
		})).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physDispatchGroups xSize zero guard and groups zero guard", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockShdr := shader_mocks.NewMockShader(suite.T())
		mockShdr.EXPECT().WorkgroupSize().Return([3]uint32{0, 1, 1}).Times(8)
		mockPhysPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPhysPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockShdr).Times(8)
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(true).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(1, make([]byte, 16)).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		phMock.EXPECT().ParticleCount().Return(0).Once()
		phMock.EXPECT().BodiesCount().Return(0).Once()
		phMock.EXPECT().MaxGridCells().Return(8).Once()
		phMock.EXPECT().Buffers().Return(bufBGP).Times(2)
		phMock.EXPECT().PipelineKey(mock.Anything).Return("phys_key43").Times(8)
		phMock.EXPECT().Bgp(mock.Anything).Return(stageBGP).Times(8)
		phMock.EXPECT().ConsumeReadbackRequest().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("phys_key43").Return(mockPhysPipe).Times(8)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(8)
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("animator dispatch xSize zero guard", func() {
		suite.scene.cullingDisabled = true
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)
		mockShdr := shader_mocks.NewMockShader(suite.T())
		mockShdr.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockShdr.EXPECT().WorkgroupSize().Return([3]uint32{0, 1, 1}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockShdr).Times(2)
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k44").Times(2)
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(2)).Times(5)
		animMock.EXPECT().Model().Return(mockModel).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(2)).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k44").Return(mockPipe).Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "k44" && d[0].Provider == computeBGP &&
				d[0].WorkGroupCount == [3]uint32{2, 1, 1}
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("bone particle dispatch xSize zero guard and groups zero guard", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockBoneShdr := shader_mocks.NewMockShader(suite.T())
		mockBoneShdr.EXPECT().WorkgroupSize().Return([3]uint32{0, 1, 1}).Once()
		mockBonePipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockBonePipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockBoneShdr).Once()
		bgp1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgp2 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.boneParticleUpdateGroups = []*boneParticleUpdateGroup{
			{bgp: bgp1, particleCount: 0},
			{bgp: bgp2, particleCount: 5},
		}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Enabled().Return(false).Once()
		phMock.EXPECT().PipelineKey("bone_update").Return("bone_key45").Once()
		suite.scene.physicsHandler = phMock
		suite.rendererMock.EXPECT().Pipeline("bone_key45").Return(mockBonePipe).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 2 &&
				d[0].PipelineKey == "bone_key45" && d[0].Provider == bgp1 && d[0].WorkGroupCount == [3]uint32{1, 1, 1} &&
				d[1].PipelineKey == "bone_key45" && d[1].Provider == bgp2 && d[1].WorkGroupCount == [3]uint32{5, 1, 1}
		})).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("light sort directional light returns MaxFloat32 importance", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		dirLight := light.NewLight(light.LightTypeDirectional)
		pointLight := light.NewLight(light.LightTypePoint)
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(bgpMock).Once()
		mockLH.EXPECT().Lights().Return([]light.Light{dirLight, pointLight}).Once()
		mockLH.EXPECT().MaxGPULights().Return(1).Once()
		mockLH.EXPECT().MarshalLightBuffer(mock.Anything, mock.Anything).Return(make([]byte, 16)).Once()
		suite.scene.lightHandler = mockLH
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("light sort comparison branches impA greater and impA less than", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		l1 := light.NewLight(light.LightTypePoint)
		l1.SetPosition(0, 0, 1)
		l2 := light.NewLight(light.LightTypePoint)
		l2.SetPosition(0, 0, 5)
		l3 := light.NewLight(light.LightTypePoint)
		l3.SetPosition(0, 0, 10)
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(bgpMock).Once()
		mockLH.EXPECT().Lights().Return([]light.Light{l3, l1, l2}).Once()
		mockLH.EXPECT().MaxGPULights().Return(2).Once()
		mockLH.EXPECT().MarshalLightBuffer(mock.Anything, mock.Anything).Return(make([]byte, 16)).Once()
		suite.scene.lightHandler = mockLH
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("light sort equal importance returns zero", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		l1 := light.NewLight(light.LightTypePoint)
		l2 := light.NewLight(light.LightTypePoint)
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(bgpMock).Once()
		mockLH.EXPECT().Lights().Return([]light.Light{l1, l2}).Once()
		mockLH.EXPECT().MaxGPULights().Return(1).Once()
		mockLH.EXPECT().MarshalLightBuffer(mock.Anything, mock.Anything).Return(make([]byte, 16)).Once()
		suite.scene.lightHandler = mockLH
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 goroutine covers boneBinding and modelBinding annotation arms", func() {
		suite.scene.cullingDisabled = true
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)
		binding1 := 1
		binding2 := 2
		binding3 := 3
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeBindingGroup, Binding: &binding1, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgBoneInfo}},
			{Type: shader.AnnotationTypeBindingGroup, Binding: &binding2, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgModelData}},
			{Type: shader.AnnotationTypeBindingGroup, Binding: &binding3, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgGlobalData}},
		}
		mockShdr := shader_mocks.NewMockShader(suite.T())
		mockShdr.EXPECT().Declarations().Return(decls).Once()
		mockShdr.EXPECT().WorkgroupSize().Return([3]uint32{64, 1, 1}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockShdr).Times(2)
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("bone-model-key").Times(2)
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(2)).Times(5)
		animMock.EXPECT().Model().Return(mockModel).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(2)).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("bone-model-key").Return(mockPipe).Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "bone-model-key" && d[0].Provider == computeBGP &&
				d[0].WorkGroupCount == [3]uint32{1, 1, 1}
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})
}
func (suite *sceneImplTest) TestDrawCalls() {
	suite.Run("empty pool returns nil", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("zero instance count skipped", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("nil model skipped", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("nil mesh provider skipped", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("empty render materials skipped", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{}).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("empty pipeline key skipped", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("nil pipeline skipped", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k7").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		suite.rendererMock.EXPECT().Pipeline("k7").Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("nil vertex shader skipped", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k8").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("k8").Return(mockPipe).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("direct draw call empty decls no frag shader", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k9").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k9").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k9", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("direct draw call with frag shader empty decls", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k10").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		fragShdrMock := shader_mocks.NewMockShader(suite.T())
		fragShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(fragShdrMock).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k10").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k10", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("draw call error propagated", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k11").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k11").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k11", meshBGP, uint32(1), mock.Anything).Return(errors.New("draw failed")).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.Error(suite.scene.DrawCalls())
	})

	suite.Run("culling enabled empty compute key falls to DrawCall", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k12").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("").Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("k12").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k12", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("culling enabled nil compute pipeline skips material", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k13").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("ck13").Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		suite.rendererMock.EXPECT().Pipeline("k13").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().Pipeline("ck13").Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("culling enabled nil indirect buffer falls to DrawCall", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k14").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("ck14").Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		mockCSShdr := shader_mocks.NewMockShader(suite.T())
		mockCSShdr.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockComputePipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockComputePipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockCSShdr).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("k14").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().Pipeline("ck14").Return(mockComputePipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k14", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("culling enabled indirect buffer non-nil draws indirect", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k15").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("ck15").Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		mockCSShdr := shader_mocks.NewMockShader(suite.T())
		mockCSShdr.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockComputePipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockComputePipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockCSShdr).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().Pipeline("k15").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().Pipeline("ck15").Return(mockComputePipe).Once()
		suite.rendererMock.EXPECT().DrawCallIndirect("k15", meshBGP, mock.Anything, mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("indirect draw call error propagated", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k16").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("ck16").Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		mockCSShdr := shader_mocks.NewMockShader(suite.T())
		mockCSShdr.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockComputePipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockComputePipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockCSShdr).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().Pipeline("k16").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().Pipeline("ck16").Return(mockComputePipe).Once()
		suite.rendererMock.EXPECT().DrawCallIndirect("k16", meshBGP, mock.Anything, mock.Anything).Return(errors.New("indirect fail")).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.Error(suite.scene.DrawCalls())
	})

	suite.Run("bind group camera provider resolves", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgCamera},
		}
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k17").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k17").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k17", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("bind group camera AnnotationTypeBindingGroup resolves", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgCamera},
		}
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k18").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k18").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k18", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("bind group animator provider resolves", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgAnimator},
		}
		animBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k19").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().OutputBindGroupProvider().Return(animBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k19").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k19", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("bind group InstanceData AnnotationTypeBindingGroup resolves", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgInstanceData},
		}
		animBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k20").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().OutputBindGroupProvider().Return(animBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k20").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k20", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("skip material when provider nil for a group", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgSSAO},
		}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k21").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		suite.rendererMock.EXPECT().Pipeline("k21").Return(mockPipe).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("material provider non-nil returns mat.Provider(g)", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgMaterial},
		}
		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k22").Once()
		matMock.EXPECT().Provider(0).Return(matBGP).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k22").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k22", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("material provider nil falls to mat.BindGroupProvider", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgMaterial},
		}
		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k23").Once()
		matMock.EXPECT().Provider(0).Return(nil).Once()
		matMock.EXPECT().BindGroupProvider().Return(matBGP).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k23").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k23", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("lights provider enabled resolves lightsBGP", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgLights},
		}
		lightsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k24").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.lightHandler = mockLH
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(lightsBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k24").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k24", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("lights provider disabled resolves nil ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ skipMaterial", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgLights},
		}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k25").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.lightHandler = mockLH
		mockLH.EXPECT().Enabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k25").Return(mockPipe).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("shadow provider enabled resolves via ShadowHandler", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgShadow},
		}
		shadowBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k26").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		mockSH := light_mocks.NewMockShadowHandler(suite.T())
		mockSH.EXPECT().Bgp("csm_shadow_lit").Return(shadowBGP).Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.lightHandler = mockLH
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().ShadowHandler().Return(mockSH).Once()
		suite.rendererMock.EXPECT().Pipeline("k26").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k26", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("tiles provider enabled resolves tile_lit BGP", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgTiles},
		}
		tilesBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k27").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.lightHandler = mockLH
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(tilesBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k27").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k27", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("effect provider non-nil resolves", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgEffect},
		}
		effectBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k28").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(effectBGP).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k28").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k28", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("effect provider nil resolves nil ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ skipMaterial", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgEffect},
		}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k29").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(nil).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k29").Return(mockPipe).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("SSAO provider enabled resolves via ssao_lit", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgSSAO},
		}
		ssaoBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k30").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.lightHandler = mockLH
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("ssao_lit").Return(ssaoBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k30").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k30", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("Light binding group enabled resolves", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgLight},
		}
		lightsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k31").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.lightHandler = mockLH
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(lightsBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k31").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k31", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("ShadowUniform binding group enabled resolves", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgShadowUniform},
		}
		shadowBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k32").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		mockSH := light_mocks.NewMockShadowHandler(suite.T())
		mockSH.EXPECT().Bgp("csm_shadow_lit").Return(shadowBGP).Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.lightHandler = mockLH
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().ShadowHandler().Return(mockSH).Once()
		suite.rendererMock.EXPECT().Pipeline("k32").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k32", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("TileUniforms binding group enabled resolves", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgTileUniforms},
		}
		tilesBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k33").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.lightHandler = mockLH
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(tilesBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k33").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k33", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("OverlayParams binding group mat.BindGroupProvider non-nil", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgOverlayParams},
		}
		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k34").Once()
		matMock.EXPECT().BindGroupProvider().Return(matBGP).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k34").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k34", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("OverlayParams binding group nil mat BGP falls to mdl.EffectProvider", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgOverlayParams},
		}
		effectBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k35").Once()
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(effectBGP).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k35").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k35", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("EffectParams binding group mdl.EffectProvider non-nil", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgEffectParams},
		}
		effectBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k36").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(effectBGP).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k36").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k36", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("EffectParams binding group nil EffectProvider falls to mat.Provider(g)", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgEffectParams},
		}
		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k37").Once()
		matMock.EXPECT().Provider(0).Return(matBGP).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(nil).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k37").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k37", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("duplicate group declaration skips second decl", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl1 := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgCamera},
		}
		decl2 := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgAnimator},
		}
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k38").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl1, decl2}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k38").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k38", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("multiple groups highest g sets maxGroup", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		one := 1
		decl1 := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgCamera},
		}
		decl2 := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &one,
			Args:  []shader.AnnotationArg{shader.AnnotationArgAnimator},
		}
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock
		animBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k39").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl1, decl2}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().OutputBindGroupProvider().Return(animBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k39").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k39", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("AnnotationTypeBindingGroup array<camera> unwraps to camera", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		binding := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &binding,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArg("array<" + string(shader.AnnotationArgCamera) + ">")},
		}
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k40").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k40").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k40", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("culling cs.Declarations IndirectArgs non-array binding sets indirectBinding", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		csBinding := 3
		csDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Binding: &csBinding,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgIndirectArgs},
		}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k41").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("ck41").Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockRenderPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		mockCS := shader_mocks.NewMockShader(suite.T())
		mockCS.EXPECT().Declarations().Return([]shader.Annotation{csDecl}).Once()
		mockCPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockCPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockCS).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(3).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k41").Return(mockRenderPipe).Once()
		suite.rendererMock.EXPECT().Pipeline("ck41").Return(mockCPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k41", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("culling cs.Declarations IndirectArgs array-wrapped binding sets indirectBinding", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		csBinding := 3
		csDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Binding: &csBinding,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArg("array<" + string(shader.AnnotationArgIndirectArgs) + ">")},
		}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k42").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("ck42").Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockRenderPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		mockCS := shader_mocks.NewMockShader(suite.T())
		mockCS.EXPECT().Declarations().Return([]shader.Annotation{csDecl}).Once()
		mockCPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockCPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockCS).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(3).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k42").Return(mockRenderPipe).Once()
		suite.rendererMock.EXPECT().Pipeline("ck42").Return(mockCPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k42", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("culling cs.Declarations IndirectArgs non-nil buffer calls DrawCallIndirect with discovered binding", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		csBinding := 3
		csDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Binding: &csBinding,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgIndirectArgs},
		}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k43").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().ComputePipelineKey().Return("ck43").Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockRenderPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		mockCS := shader_mocks.NewMockShader(suite.T())
		mockCS.EXPECT().Declarations().Return([]shader.Annotation{csDecl}).Once()
		mockCPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockCPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(mockCS).Once()
		indBuf := &wgpu.Buffer{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(3).Return(indBuf).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k43").Return(mockRenderPipe).Once()
		suite.rendererMock.EXPECT().Pipeline("ck43").Return(mockCPipe).Once()
		suite.rendererMock.EXPECT().DrawCallIndirect("k43", meshBGP, mock.Anything, mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("decl with nil Group is skipped in provider scan", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		mapKey := model_mocks.NewMockModel(suite.T())
		animMock := animator_mocks.NewMockAnimator(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		zero := 0
		nilGroupDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: nil,
			Args:  []shader.AnnotationArg{shader.AnnotationArgCamera},
		}
		camGroupDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &zero,
			Args:  []shader.AnnotationArg{shader.AnnotationArgCamera},
		}
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{nilGroupDecl, camGroupDecl}).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		suite.scene.cam = camMock
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		mockModel.EXPECT().MeshProvider().Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		matMock.EXPECT().PipelineKey().Return("knil").Once()
		suite.rendererMock.EXPECT().Pipeline("knil").Return(mockPipe).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().DrawCall("knil", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})
}

func (suite *sceneImplTest) TestWithObjects() {
	suite.Run("empty list noop", func() {
		s := &scene{
			registry: make(map[uint64]game_object.GameObject),
			nextID:   1,
			mu:       &sync.RWMutex{},
		}
		opt := WithObjects()
		opt(s)
		suite.Equal(0, len(s.registry))
		suite.Equal(uint64(1), s.nextID)
	})

	suite.Run("id zero assigns nextID non-ephemeral goes to registry", func() {
		s := &scene{
			registry: make(map[uint64]game_object.GameObject),
			nextID:   1,
			mu:       &sync.RWMutex{},
		}
		mockObj := game_object_mocks.NewMockGameObject(suite.T())
		mockObj.EXPECT().ID().Return(uint64(0)).Once()
		mockObj.EXPECT().SetID(uint64(1)).Once()
		mockObj.EXPECT().Ephemeral().Return(false).Once()
		mockObj.EXPECT().ID().Return(uint64(1)).Once()
		opt := WithObjects(mockObj)
		opt(s)
		suite.Equal(uint64(2), s.nextID)
		suite.Equal(mockObj, s.registry[1])
	})

	suite.Run("id non-zero kept non-ephemeral goes to registry", func() {
		s := &scene{
			registry: make(map[uint64]game_object.GameObject),
			nextID:   1,
			mu:       &sync.RWMutex{},
		}
		mockObj := game_object_mocks.NewMockGameObject(suite.T())
		mockObj.EXPECT().ID().Return(uint64(5)).Once()
		mockObj.EXPECT().Ephemeral().Return(false).Once()
		mockObj.EXPECT().ID().Return(uint64(5)).Once()
		opt := WithObjects(mockObj)
		opt(s)
		suite.Equal(uint64(1), s.nextID)
		suite.Equal(mockObj, s.registry[5])
	})

	suite.Run("ephemeral skips registry", func() {
		s := &scene{
			registry: make(map[uint64]game_object.GameObject),
			nextID:   1,
			mu:       &sync.RWMutex{},
		}
		mockObj := game_object_mocks.NewMockGameObject(suite.T())
		mockObj.EXPECT().ID().Return(uint64(3)).Once()
		mockObj.EXPECT().Ephemeral().Return(true).Once()
		opt := WithObjects(mockObj)
		opt(s)
		suite.Equal(0, len(s.registry))
	})
}

func (suite *sceneImplTest) TestWithComputeWorkers() {
	suite.Run("n one or more sets computeWorkers", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithComputeWorkers(4)
		opt(s)
		suite.Equal(4, s.computeWorkers)
	})

	suite.Run("n zero clamps to one", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithComputeWorkers(0)
		opt(s)
		suite.Equal(1, s.computeWorkers)
	})

	suite.Run("n negative clamps to one", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithComputeWorkers(-5)
		opt(s)
		suite.Equal(1, s.computeWorkers)
	})
}

func (suite *sceneImplTest) TestWithLighting() {
	suite.Run("sets lightHandler", func() {
		s := &scene{mu: &sync.RWMutex{}}
		mock := light_mocks.NewMockLightingHandler(suite.T())
		opt := WithLighting(mock)
		opt(s)
		suite.Equal(mock, s.lightHandler)
	})
}

func (suite *sceneImplTest) TestWithPhysics() {
	suite.Run("sets physicsHandler", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithPhysics()
		opt(s)
		suite.NotNil(s.physicsHandler)
	})
}

func (suite *sceneImplTest) TestWithScreenSize() {
	suite.Run("sets screenWidth and screenHeight", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithScreenSize(1920, 1080)
		opt(s)
		suite.Equal(1920, s.screenWidth)
		suite.Equal(1080, s.screenHeight)
	})
}

func (suite *sceneImplTest) TestWithMaxBonesGPU() {
	suite.Run("valid value sets maxBonesGPU", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithMaxBonesGPU(128)
		opt(s)
		suite.Equal(uint64(128), s.maxBonesGPU)
	})

	suite.Run("zero clamps to one", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithMaxBonesGPU(0)
		opt(s)
		suite.Equal(uint64(1), s.maxBonesGPU)
	})
}

func (suite *sceneImplTest) TestNewScene() {
	suite.Run("bgp non-nil InitBindGroup success", func() {
		cam := camera_mocks.NewMockCamera(suite.T())
		r := renderer_mocks.NewMockRenderer(suite.T())
		bgp := bgp_mocks.NewMockBindGroupProvider(suite.T())

		cam.EXPECT().BindGroupProvider().Return(bgp).Once()
		r.EXPECT().SetInjections(mock.Anything).Return().Once()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		result := NewScene("test", cam, r)
		suite.NotNil(result)
	})

	suite.Run("bgp non-nil InitBindGroup error panics", func() {
		cam := camera_mocks.NewMockCamera(suite.T())
		r := renderer_mocks.NewMockRenderer(suite.T())
		bgp := bgp_mocks.NewMockBindGroupProvider(suite.T())

		cam.EXPECT().BindGroupProvider().Return(bgp).Once()
		r.EXPECT().SetInjections(mock.Anything).Return().Once()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("gpu err")).Once()

		suite.Panics(func() { NewScene("test", cam, r) })
	})
}

func (suite *sceneImplTest) TestInitSSAO() {
	suite.Run("nil SSAOHandler returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.initSSAO()
	})

	suite.Run("w zero returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 600
		suite.scene.initSSAO()
	})

	suite.Run("HalfResolution true CreateSSAOTextures error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		ssaoMock.EXPECT().HalfResolution().Return(true).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(400, 300).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), errors.New("tex err")).Once()
		suite.Panics(func() { suite.scene.initSSAO() })
	})

	suite.Run("CreateSSAOTextures error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), errors.New("tex err")).Once()
		suite.Panics(func() { suite.scene.initSSAO() })
	})

	suite.Run("CreateLinearSampler error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil), errors.New("samp err")).Once()
		suite.Panics(func() { suite.scene.initSSAO() })
	})

	suite.Run("RegisterPipelines ssao_compute error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil), nil).Once()
		ssaoMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("reg err")).Once()
		suite.Panics(func() { suite.scene.initSSAO() })
	})

	suite.Run("RegisterPipelines ssao_blur error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil), nil).Once()
		ssaoMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetPipelineKey("ssao_compute", "ssao_compute").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("blur reg err")).Once()
		suite.Panics(func() { suite.scene.initSSAO() })
	})

	suite.Run("InitBindGroup ssao_compute error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil), nil).Once()
		ssaoMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		ssaoBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssaoMock.EXPECT().Bgp("ssao_compute").Return(ssaoBGPMock).Once()
		ssaoBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		gbufMock.EXPECT().DepthTextureView().Return((*wgpu.TextureView)(nil)).Maybe()
		gbufMock.EXPECT().NormalTextureView().Return((*wgpu.TextureView)(nil)).Once()
		suite.rendererMock.EXPECT().InitBindGroup(ssaoBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("bgp err")).Once()
		suite.Panics(func() { suite.scene.initSSAO() })
	})

	suite.Run("InitBindGroup blur_h error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil), nil).Once()
		ssaoMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		ssaoBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		blurHBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssaoMock.EXPECT().Bgp("ssao_compute").Return(ssaoBGPMock).Once()
		ssaoMock.EXPECT().Bgp("ssao_blur_h").Return(blurHBGPMock).Once()
		ssaoBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		blurHBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		gbufMock.EXPECT().DepthTextureView().Return((*wgpu.TextureView)(nil)).Maybe()
		gbufMock.EXPECT().NormalTextureView().Return((*wgpu.TextureView)(nil)).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(ssaoBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(blurHBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("blurH err")).Once()
		suite.Panics(func() { suite.scene.initSSAO() })
	})

	suite.Run("InitBindGroup blur_v error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil), nil).Once()
		ssaoMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		ssaoBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		blurHBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		blurVBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssaoMock.EXPECT().Bgp("ssao_compute").Return(ssaoBGPMock).Once()
		ssaoMock.EXPECT().Bgp("ssao_blur_h").Return(blurHBGPMock).Once()
		ssaoMock.EXPECT().Bgp("ssao_blur_v").Return(blurVBGPMock).Once()
		ssaoBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		blurHBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		blurVBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		gbufMock.EXPECT().DepthTextureView().Return((*wgpu.TextureView)(nil)).Maybe()
		gbufMock.EXPECT().NormalTextureView().Return((*wgpu.TextureView)(nil)).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(ssaoBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(blurHBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(blurVBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("blurV err")).Once()
		suite.Panics(func() { suite.scene.initSSAO() })
	})

	suite.Run("full happy path HalfResolution false completes", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Once()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Once()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Once()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Once()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Once()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil), nil).Once()
		ssaoMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssaoMock.EXPECT().SetPipelineKey("ssao_compute", "ssao_compute").Once()
		ssaoMock.EXPECT().SetPipelineKey("ssao_blur", "ssao_blur_compute").Once()
		ssaoBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		blurHBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		blurVBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssaoMock.EXPECT().Bgp("ssao_compute").Return(ssaoBGPMock).Once()
		ssaoMock.EXPECT().Bgp("ssao_blur_h").Return(blurHBGPMock).Once()
		ssaoMock.EXPECT().Bgp("ssao_blur_v").Return(blurVBGPMock).Once()
		ssaoBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		blurHBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		blurVBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		gbufMock.EXPECT().DepthTextureView().Return((*wgpu.TextureView)(nil)).Maybe()
		gbufMock.EXPECT().NormalTextureView().Return((*wgpu.TextureView)(nil)).Once()
		suite.rendererMock.EXPECT().InitBindGroup(ssaoBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(blurHBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(blurVBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		ssaoMock.EXPECT().SampleCount().Return(8).Once()
		ssaoMock.EXPECT().MaxSamples().Return(32).Maybe()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Once()
		ssaoMock.EXPECT().Resize(800, 600).Once()
		ssaoMock.EXPECT().SetEnabled(true).Once()
		suite.scene.initSSAO()
	})
}

func (suite *sceneImplTest) TestInitSSAOLitBindGroup() {
	suite.Run("nil litFragmentShader returns early", func() {
		suite.scene.initSSAOLitBindGroup(nil)
	})

	suite.Run("no SSAO provider declaration returns early", func() {
		shaderMock := shader_mocks.NewMockShader(suite.T())
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		suite.scene.initSSAOLitBindGroup(shaderMock)
	})

	suite.Run("ssaoReady true binds texture and sampler", func() {
		ssaoGroupIdx := 6
		ssaoDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &ssaoGroupIdx,
			Args:  []shader.AnnotationArg{shader.AnnotationArgSSAO},
		}
		shaderMock := shader_mocks.NewMockShader(suite.T())
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{ssaoDecl}).Once()

		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat}},
				{Binding: 1, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			},
		}
		shaderMock.EXPECT().BindGroupLayoutDescriptor(ssaoGroupIdx).Return(desc).Once()

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		lhMock.EXPECT().Bgp("ssao_lit").Return(bgpMock).Once()
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()

		fakeView := new(wgpu.TextureView)
		fakeSampler := new(wgpu.Sampler)
		ssaoMock.EXPECT().Enabled().Return(true).Once()
		ssaoMock.EXPECT().BlurredTextureView().Return(fakeView).Maybe()
		ssaoMock.EXPECT().LinearSampler().Return(fakeSampler).Maybe()

		bgpMock.EXPECT().SetTextureView(0, fakeView).Once()
		bgpMock.EXPECT().SetSampler(1, fakeSampler).Once()

		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

		suite.scene.lightHandler = lhMock
		suite.scene.initSSAOLitBindGroup(shaderMock)
	})

	suite.Run("InitTextureView error panics", func() {
		ssaoGroupIdx := 6
		ssaoDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &ssaoGroupIdx,
			Args:  []shader.AnnotationArg{shader.AnnotationArgSSAO},
		}
		shaderMock := shader_mocks.NewMockShader(suite.T())
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{ssaoDecl}).Once()

		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat}},
			},
		}
		shaderMock.EXPECT().BindGroupLayoutDescriptor(ssaoGroupIdx).Return(desc).Once()

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		lhMock.EXPECT().Bgp("ssao_lit").Return(bgpMock).Once()
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		ssaoMock.EXPECT().Enabled().Return(false).Once()

		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 0, mock.Anything).Return(errors.New("tex err")).Once()

		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initSSAOLitBindGroup(shaderMock) })
	})

	suite.Run("InitSampler error panics", func() {
		ssaoGroupIdx := 6
		ssaoDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &ssaoGroupIdx,
			Args:  []shader.AnnotationArg{shader.AnnotationArgSSAO},
		}
		shaderMock := shader_mocks.NewMockShader(suite.T())
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{ssaoDecl}).Once()

		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat}},
				{Binding: 1, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			},
		}
		shaderMock.EXPECT().BindGroupLayoutDescriptor(ssaoGroupIdx).Return(desc).Once()

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		lhMock.EXPECT().Bgp("ssao_lit").Return(bgpMock).Once()
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		ssaoMock.EXPECT().Enabled().Return(false).Once()

		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 0, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 1, mock.Anything).Return(errors.New("samp err")).Once()

		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initSSAOLitBindGroup(shaderMock) })
	})

	suite.Run("InitBindGroup error panics", func() {
		ssaoGroupIdx := 6
		ssaoDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &ssaoGroupIdx,
			Args:  []shader.AnnotationArg{shader.AnnotationArgSSAO},
		}
		shaderMock := shader_mocks.NewMockShader(suite.T())
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{ssaoDecl}).Once()

		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat}},
				{Binding: 1, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			},
		}
		shaderMock.EXPECT().BindGroupLayoutDescriptor(ssaoGroupIdx).Return(desc).Once()

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		lhMock.EXPECT().Bgp("ssao_lit").Return(bgpMock).Once()
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		ssaoMock.EXPECT().Enabled().Return(false).Once()

		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 0, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 1, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("bgp err")).Once()

		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initSSAOLitBindGroup(shaderMock) })
	})
}

func (suite *sceneImplTest) TestInitGBuffer() {
	suite.Run("nil GBufferHandler returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lhMock.EXPECT().GBufferHandler().Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.initGBuffer()
	})

	suite.Run("zero screen width returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 480
		suite.scene.initGBuffer()
	})

	suite.Run("zero screen height returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 0
		suite.scene.initGBuffer()
	})

	suite.Run("CreateGBufferTextures error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateGBufferTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), errors.New("gpu err")).Once()
		suite.Panics(func() { suite.scene.initGBuffer() })
	})

	suite.Run("RegisterGBufferPipeline static error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_bones": "64u"}
		suite.rendererMock.EXPECT().CreateGBufferTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		gbufMock.EXPECT().SetNormalTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetNormalTextureView(mock.Anything).Maybe()
		gbufMock.EXPECT().SetAlbedoTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetAlbedoTextureView(mock.Anything).Maybe()
		gbufMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).
			Return(errors.New("pipe err")).Once()
		suite.Panics(func() { suite.scene.initGBuffer() })
	})

	suite.Run("RegisterGBufferPipeline skinned error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_bones": "64u"}
		suite.rendererMock.EXPECT().CreateGBufferTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		gbufMock.EXPECT().SetNormalTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetNormalTextureView(mock.Anything).Maybe()
		gbufMock.EXPECT().SetAlbedoTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetAlbedoTextureView(mock.Anything).Maybe()
		gbufMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(nil).Once()
		gbufMock.EXPECT().SetPipelineKey("static", "gbuffer_static").Once()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(errors.New("skinned pipe err")).Once()
		suite.Panics(func() { suite.scene.initGBuffer() })
	})

	suite.Run("happy path completes all steps", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_bones": "64u"}
		suite.rendererMock.EXPECT().CreateGBufferTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		gbufMock.EXPECT().SetNormalTexture(mock.Anything).Once()
		gbufMock.EXPECT().SetNormalTextureView(mock.Anything).Once()
		gbufMock.EXPECT().SetAlbedoTexture(mock.Anything).Once()
		gbufMock.EXPECT().SetAlbedoTextureView(mock.Anything).Once()
		gbufMock.EXPECT().SetDepthTexture(mock.Anything).Once()
		gbufMock.EXPECT().SetDepthTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(nil).Once()
		gbufMock.EXPECT().SetPipelineKey("static", "gbuffer_static").Once()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(nil).Once()
		gbufMock.EXPECT().SetPipelineKey("skinned", "gbuffer_skinned").Once()
		gbufMock.EXPECT().Resize(800, 600).Once()
		gbufMock.EXPECT().SetEnabled(true).Once()
		suite.scene.initGBuffer()
	})
}

func (suite *sceneImplTest) TestInitContactShadows() {
	suite.Run("nil csHandler returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.initContactShadows()
	})

	suite.Run("nil GBufferHandler returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		lhMock.EXPECT().GBufferHandler().Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.initContactShadows()
	})

	suite.Run("GBufferHandler not enabled returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(false).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.initContactShadows()
	})

	suite.Run("zero screen width returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 600
		suite.scene.initContactShadows()
	})

	suite.Run("zero screen height returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 0
		suite.scene.initContactShadows()
	})

	suite.Run("CreateContactShadowTextures error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateContactShadowTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), errors.New("tex err")).Once()
		suite.Panics(func() { suite.scene.initContactShadows() })
	})

	suite.Run("CreateLinearSampler error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateContactShadowTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		csMock.EXPECT().SetTexture(mock.Anything).Once()
		csMock.EXPECT().SetTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateLinearSampler().
			Return((*wgpu.Sampler)(nil), errors.New("samp err")).Once()
		suite.Panics(func() { suite.scene.initContactShadows() })
	})

	suite.Run("RegisterPipelines error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateContactShadowTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		csMock.EXPECT().SetTexture(mock.Anything).Once()
		csMock.EXPECT().SetTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil), nil).Once()
		csMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).
			Return(errors.New("pipe err")).Once()
		suite.Panics(func() { suite.scene.initContactShadows() })
	})

	suite.Run("InitBindGroup error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateContactShadowTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		csMock.EXPECT().SetTexture(mock.Anything).Once()
		csMock.EXPECT().SetTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil), nil).Once()
		csMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		csMock.EXPECT().SetPipelineKey("contact_shadow_compute", "contact_shadow_compute").Once()
		csBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		csMock.EXPECT().Bgp("contact_shadow_compute").Return(csBGPMock).Once()
		gbufMock.EXPECT().DepthTextureView().Return((*wgpu.TextureView)(nil)).Once()
		csBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(csBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("bgp err")).Once()
		suite.Panics(func() { suite.scene.initContactShadows() })
	})

	suite.Run("happy path completes", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		lhMock.EXPECT().GBufferHandler().Return(gbufMock).Maybe()
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateContactShadowTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil).Once()
		csMock.EXPECT().SetTexture(mock.Anything).Once()
		csMock.EXPECT().SetTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil), nil).Once()
		csMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		csMock.EXPECT().SetPipelineKey("contact_shadow_compute", "contact_shadow_compute").Once()
		csBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		csMock.EXPECT().Bgp("contact_shadow_compute").Return(csBGPMock).Once()
		gbufMock.EXPECT().DepthTextureView().Return((*wgpu.TextureView)(nil)).Once()
		csBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(csBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		csMock.EXPECT().SetEnabled(true).Once()
		suite.scene.initContactShadows()
	})
}
func (suite *sceneImplTest) TestInitLightBindGroup() {
	suite.Run("nil fragmentShader returns early", func() {
		suite.scene.initLightBindGroup(nil)
	})

	suite.Run("no declarations returns early", func() {
		shaderMock := shader_mocks.NewMockShader(suite.T())
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		suite.scene.initLightBindGroup(shaderMock)
	})

	suite.Run("declaration wrong type does not match returns early", func() {
		shaderMock := shader_mocks.NewMockShader(suite.T())
		lightGroupIdx := 3
		wrongDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &lightGroupIdx,
			Args:  []shader.AnnotationArg{"arg0", "arg1", shader.AnnotationArgLightHeader},
		}
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{wrongDecl}).Once()
		suite.scene.initLightBindGroup(shaderMock)
	})

	suite.Run("no storage buffer entries InitBindGroup succeeds", func() {
		shaderMock := shader_mocks.NewMockShader(suite.T())
		lightGroupIdx := 3
		matchingDecl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &lightGroupIdx,
			Args:  []shader.AnnotationArg{"arg0", "arg1", shader.AnnotationArgLightHeader},
		}
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{matchingDecl}).Once()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
			},
		}
		shaderMock.EXPECT().BindGroupLayoutDescriptor(lightGroupIdx).Return(desc).Once()

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lhMock.EXPECT().Bgp("lights").Return(bgpMock).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.initLightBindGroup(shaderMock)
	})

	suite.Run("storage buffer entry populates sizeOverrides InitBindGroup succeeds", func() {
		shaderMock := shader_mocks.NewMockShader(suite.T())
		lightGroupIdx := 3
		matchingDecl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &lightGroupIdx,
			Args:  []shader.AnnotationArg{"arg0", "arg1", shader.AnnotationArgLightHeader},
		}
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{matchingDecl}).Once()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage}},
			},
		}
		shaderMock.EXPECT().BindGroupLayoutDescriptor(lightGroupIdx).Return(desc).Once()

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lhMock.EXPECT().Bgp("lights").Return(bgpMock).Once()
		lhMock.EXPECT().MaxGPULights().Return(64).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.initLightBindGroup(shaderMock)
	})

	suite.Run("InitBindGroup error panics", func() {
		shaderMock := shader_mocks.NewMockShader(suite.T())
		lightGroupIdx := 3
		matchingDecl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &lightGroupIdx,
			Args:  []shader.AnnotationArg{"arg0", "arg1", shader.AnnotationArgLightHeader},
		}
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{matchingDecl}).Once()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{},
		}
		shaderMock.EXPECT().BindGroupLayoutDescriptor(lightGroupIdx).Return(desc).Once()

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lhMock.EXPECT().Bgp("lights").Return(bgpMock).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("init err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initLightBindGroup(shaderMock) })
	})
}

func (suite *sceneImplTest) TestInitShadowMap() {
	newBaseHandlers := func() (*light_mocks.MockLightingHandler, *light_mocks.MockShadowHandler) {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		lhMock.EXPECT().ShadowHandler().Return(shMock).Once()
		shMock.EXPECT().ShadowMapResolution().Return(1024).Once()
		shMock.EXPECT().CascadeCount().Return(2).Once()
		return lhMock, shMock
	}

	setupThroughCascades := func(lhMock *light_mocks.MockLightingHandler, shMock *light_mocks.MockShadowHandler, shaderMock *shader_mocks.MockShader) {
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil, nil).Once()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Once()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgp0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgp1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_data_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_0").Return(bgp0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		shMock.EXPECT().SetBgp("csm_data_1", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_1").Return(bgp1).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		lhMock.EXPECT().MaxGPULights().Return(1).Once()
		shMock.EXPECT().LightShadowTileSize().Return(256).Once()
	}

	setupThroughStaticPipelines := func(shMock *light_mocks.MockShadowHandler) {
		shMock.EXPECT().SetLightShadowAtlasSlots(6).Once()
		shMock.EXPECT().SetLightShadowAtlasCols(3).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(768, 512).Return(nil, nil, nil).Once()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Once()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Once()
		spotKeys := []string{"spot_shadow_0", "spot_shadow_1", "spot_shadow_2", "spot_shadow_3", "spot_shadow_4", "spot_shadow_5"}
		for _, key := range spotKeys {
			spotBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
			shMock.EXPECT().SetBgp(key, mock.Anything).Once()
			shMock.EXPECT().Bgp(key).Return(spotBGP).Once()
			suite.rendererMock.EXPECT().InitBindGroup(spotBGP, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		}
		suite.rendererMock.EXPECT().RegisterShadowDepthPipeline(mock.Anything).Return(nil).Times(3)
		shMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Times(3)
	}

	suite.Run("nil shadowVertShader returns early", func() {
		suite.scene.initShadowMap(nil, nil)
	})

	suite.Run("atlasW exceeds maxDim panics", func() {
		lhMock, _ := newBaseHandlers()
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(1024)).Once()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, nil) })
	})

	suite.Run("CreateShadowDepthTexture CSM error panics", func() {
		lhMock, _ := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil, errors.New("csm err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, nil) })
	})

	suite.Run("CreateComparisonSampler error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil, errors.New("sampler err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, nil) })
	})

	suite.Run("InitBindGroup CSM cascade error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil, nil).Once()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Once()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgp0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_data_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_0").Return(bgp0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("bgp err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, nil) })
	})

	suite.Run("uniform buffer entry populates sizeOverrides", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil, nil).Once()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Once()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding: 0,
					Buffer: wgpu.BufferBindingLayout{
						Type: wgpu.BufferBindingTypeUniform,
					},
				},
			},
		}
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(desc).Once()
		bgp0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgp1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_data_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_0").Return(bgp0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		shMock.EXPECT().SetBgp("csm_data_1", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_1").Return(bgp1).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		lhMock.EXPECT().MaxGPULights().Return(1).Once()
		shMock.EXPECT().LightShadowTileSize().Return(256).Once()
		setupThroughStaticPipelines(shMock)
		suite.scene.lightHandler = lhMock
		suite.scene.initShadowMap(shaderMock, nil)
	})

	suite.Run("maxDim greater than safeMaxTextureDim clamps effectiveMaxDim", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(99999)).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil, nil).Once()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Once()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgp0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgp1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_data_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_0").Return(bgp0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		shMock.EXPECT().SetBgp("csm_data_1", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_1").Return(bgp1).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		lhMock.EXPECT().MaxGPULights().Return(1).Once()
		shMock.EXPECT().LightShadowTileSize().Return(256).Once()
		setupThroughStaticPipelines(shMock)
		suite.scene.lightHandler = lhMock
		suite.scene.initShadowMap(shaderMock, nil)
	})

	suite.Run("large tileSize clamps maxTilesPerAxis to 1 and rows to maxTilesPerAxis", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil, nil).Once()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Once()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgp0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgp1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_data_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_0").Return(bgp0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		shMock.EXPECT().SetBgp("csm_data_1", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_1").Return(bgp1).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		lhMock.EXPECT().MaxGPULights().Return(1).Once()
		shMock.EXPECT().LightShadowTileSize().Return(16384).Once()
		shMock.EXPECT().SetLightShadowAtlasSlots(1).Once()
		shMock.EXPECT().SetLightShadowAtlasCols(1).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(16384, 16384).Return(nil, nil, nil).Once()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Once()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Once()
		spotBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("spot_shadow_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("spot_shadow_0").Return(spotBGP).Once()
		suite.rendererMock.EXPECT().InitBindGroup(spotBGP, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterShadowDepthPipeline(mock.Anything).Return(nil).Times(3)
		shMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Times(3)
		suite.scene.lightHandler = lhMock
		suite.scene.initShadowMap(shaderMock, nil)
	})

	suite.Run("CreateShadowDepthTexture spot error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		setupThroughCascades(lhMock, shMock, shaderMock)
		shMock.EXPECT().SetLightShadowAtlasSlots(6).Once()
		shMock.EXPECT().SetLightShadowAtlasCols(3).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(768, 512).Return(nil, nil, errors.New("spot err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, nil) })
	})

	suite.Run("InitBindGroup spot slot error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		setupThroughCascades(lhMock, shMock, shaderMock)
		shMock.EXPECT().SetLightShadowAtlasSlots(6).Once()
		shMock.EXPECT().SetLightShadowAtlasCols(3).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(768, 512).Return(nil, nil, nil).Once()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Once()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Once()
		spotBGP0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("spot_shadow_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("spot_shadow_0").Return(spotBGP0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(spotBGP0, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("spot bgp err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, nil) })
	})

	suite.Run("RegisterShadowDepthPipeline static error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		setupThroughCascades(lhMock, shMock, shaderMock)
		shMock.EXPECT().SetLightShadowAtlasSlots(6).Once()
		shMock.EXPECT().SetLightShadowAtlasCols(3).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(768, 512).Return(nil, nil, nil).Once()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Once()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Once()
		spotKeys := []string{"spot_shadow_0", "spot_shadow_1", "spot_shadow_2", "spot_shadow_3", "spot_shadow_4", "spot_shadow_5"}
		for _, key := range spotKeys {
			spotBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
			shMock.EXPECT().SetBgp(key, mock.Anything).Once()
			shMock.EXPECT().Bgp(key).Return(spotBGP).Once()
			suite.rendererMock.EXPECT().InitBindGroup(spotBGP, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		}
		suite.rendererMock.EXPECT().RegisterShadowDepthPipeline(mock.Anything).Return(errors.New("pipeline err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, nil) })
	})

	suite.Run("RegisterShadowDepthPipeline skinned error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		setupThroughCascades(lhMock, shMock, shaderMock)
		setupThroughStaticPipelines(shMock)
		skinnedMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().RegisterShadowDepthPipeline(mock.Anything).Return(errors.New("skinned err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, skinnedMock) })
	})

	suite.Run("full happy path nil skinned shader", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		setupThroughCascades(lhMock, shMock, shaderMock)
		setupThroughStaticPipelines(shMock)
		suite.scene.lightHandler = lhMock
		suite.scene.initShadowMap(shaderMock, nil)
	})

	suite.Run("full happy path with skinned shader", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		setupThroughCascades(lhMock, shMock, shaderMock)
		setupThroughStaticPipelines(shMock)
		skinnedMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().RegisterShadowDepthPipeline(mock.Anything).Return(nil).Times(3)
		shMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Times(3)
		suite.scene.lightHandler = lhMock
		suite.scene.initShadowMap(shaderMock, skinnedMock)
	})
}

func (suite *sceneImplTest) TestInitCSMShadowLitBindGroup() {
	makeReadyShader := func(decls []shader.Annotation) (*light_mocks.MockLightingHandler, *light_mocks.MockShadowHandler, *shader_mocks.MockShader) {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		shMock.EXPECT().CSMAtlasTextureView().Return(&wgpu.TextureView{}).Maybe()
		shMock.EXPECT().ComparisonSampler().Return(&wgpu.Sampler{}).Maybe()
		fragMock := shader_mocks.NewMockShader(suite.T())
		fragMock.EXPECT().Declarations().Return(decls).Maybe()
		return lhMock, shMock, fragMock
	}

	makeProviderDecl := func() shader.Annotation {
		grp := 4
		return shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &grp,
			Args:  []shader.AnnotationArg{shader.AnnotationArgShadow},
		}
	}

	setupBGP := func(shMock *light_mocks.MockShadowHandler, fragMock *shader_mocks.MockShader) *bgp_mocks.MockBindGroupProvider {
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_shadow_lit", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_shadow_lit").Return(bgpMock).Once()
		fragMock.EXPECT().BindGroupLayoutDescriptor(4).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgpMock.EXPECT().SetTextureView(0, mock.Anything).Maybe()
		bgpMock.EXPECT().SetSampler(1, mock.Anything).Maybe()
		return bgpMock
	}

	suite.Run("nil litFragmentShader returns early", func() {
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(nil) })
	})

	suite.Run("CSMAtlasTextureView nil returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		lhMock.EXPECT().ShadowHandler().Return(shMock).Once()
		shMock.EXPECT().CSMAtlasTextureView().Return(nil).Once()
		fragMock := shader_mocks.NewMockShader(suite.T())
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("ComparisonSampler nil returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		lhMock.EXPECT().ShadowHandler().Return(shMock).Once()
		shMock.EXPECT().CSMAtlasTextureView().Return(&wgpu.TextureView{}).Once()
		shMock.EXPECT().ComparisonSampler().Return(nil).Once()
		fragMock := shader_mocks.NewMockShader(suite.T())
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("no provider declaration shadowGroup negative returns early", func() {
		lhMock, _, fragMock := makeReadyShader([]shader.Annotation{})
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("wrong provider type and wrong arg shadowGroup negative returns early", func() {
		grp := 4
		wrongTypeDecl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &grp,
			Args:  []shader.AnnotationArg{shader.AnnotationArgShadow},
		}
		wrongArgDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &grp,
			Args:  []shader.AnnotationArg{shader.AnnotationArgCamera},
		}
		lhMock, _, fragMock := makeReadyShader([]shader.Annotation{wrongTypeDecl, wrongArgDecl})
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("LightShadowAtlasView nil contact shadow nil fallback succeeds", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("LightShadowAtlasView non-nil binding 3 set", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(&wgpu.TextureView{}).Maybe()
		bgpMock.EXPECT().SetTextureView(3, mock.Anything).Once()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("contact shadow handler nil InitTextureView error panics", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(errors.New("tex err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("contact shadow handler nil InitSampler error panics", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(errors.New("samp err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("contact shadow handler enabled false fallback path succeeds", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().Enabled().Return(false).Once()
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("contact shadow handler TextureView nil fallback path succeeds", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().Enabled().Return(true).Once()
		csMock.EXPECT().TextureView().Return(nil).Once()
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("contact shadow handler LinearSampler nil fallback path succeeds", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().Enabled().Return(true).Once()
		csMock.EXPECT().TextureView().Return(&wgpu.TextureView{}).Once()
		csMock.EXPECT().LinearSampler().Return(nil).Once()
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("contact shadow handler all OK bindings 5 and 6 set", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().Enabled().Return(true).Once()
		csMock.EXPECT().TextureView().Return(&wgpu.TextureView{}).Maybe()
		csMock.EXPECT().LinearSampler().Return(&wgpu.Sampler{}).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		bgpMock.EXPECT().SetTextureView(5, mock.Anything).Once()
		bgpMock.EXPECT().SetSampler(6, mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("sizeOverrides CSMData match sets override", func() {
		grp := 4
		binding := 2
		csmDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &grp,
			Binding: &binding,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgCSMData},
		}
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl(), csmDecl})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.MatchedBy(func(m map[int]uint64) bool {
			return len(m) == 1
		})).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("sizeOverrides array light_shadow_entry stripped and LightShadowAtlasSlots called", func() {
		grp := 4
		binding := 4
		entryDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &grp,
			Binding: &binding,
			Args:    []shader.AnnotationArg{"", "", "array<light_shadow_entry>"},
		}
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl(), entryDecl})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		shMock.EXPECT().LightShadowAtlasSlots().Return(64).Once()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.MatchedBy(func(m map[int]uint64) bool {
			return len(m) == 1
		})).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("sizeOverrides decl with nil Group is skipped", func() {
		binding := 2
		nilGroupDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   nil,
			Binding: &binding,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgCSMData},
		}
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl(), nilGroupDecl})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.MatchedBy(func(m map[int]uint64) bool {
			return len(m) == 0
		})).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("sizeOverrides decl with wrong group is skipped", func() {
		wrongGrp := 99
		binding := 2
		wrongGroupDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &wrongGrp,
			Binding: &binding,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgCSMData},
		}
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl(), wrongGroupDecl})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.MatchedBy(func(m map[int]uint64) bool {
			return len(m) == 0
		})).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("sizeOverrides decl with nil Binding is skipped", func() {
		grp := 4
		nilBindingDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &grp,
			Binding: nil,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgCSMData},
		}
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl(), nilBindingDecl})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.MatchedBy(func(m map[int]uint64) bool {
			return len(m) == 0
		})).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("InitBindGroup error panics", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("bind err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("full happy path all branches exercised", func() {
		grp := 4
		binding0 := 2
		binding1 := 4
		csmDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &grp,
			Binding: &binding0,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgCSMData},
		}
		entryDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &grp,
			Binding: &binding1,
			Args:    []shader.AnnotationArg{"", "", "array<light_shadow_entry>"},
		}
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl(), csmDecl, entryDecl})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(&wgpu.TextureView{}).Maybe()
		bgpMock.EXPECT().SetTextureView(3, mock.Anything).Once()
		shMock.EXPECT().LightShadowAtlasSlots().Return(64).Once()
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().Enabled().Return(true).Once()
		csMock.EXPECT().TextureView().Return(&wgpu.TextureView{}).Maybe()
		csMock.EXPECT().LinearSampler().Return(&wgpu.Sampler{}).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		bgpMock.EXPECT().SetTextureView(5, mock.Anything).Once()
		bgpMock.EXPECT().SetSampler(6, mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.MatchedBy(func(m map[int]uint64) bool {
			return len(m) == 2
		})).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})
}

func (suite *sceneImplTest) TestInitLightCullResources() {
	makeBase := func() (
		*light_mocks.MockLightingHandler,
		*bgp_mocks.MockBindGroupProvider,
		*bgp_mocks.MockBindGroupProvider,
		*bgp_mocks.MockBindGroupProvider,
		*shader_mocks.MockShader,
		*shader_mocks.MockShader,
	) {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lightsBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		cullBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		tileBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		cullShaderMock := shader_mocks.NewMockShader(suite.T())
		litShaderMock := shader_mocks.NewMockShader(suite.T())

		lightsBGPMock.EXPECT().Buffer(1).Return(&wgpu.Buffer{}).Maybe()
		cullBGPMock.EXPECT().SetBuffer(1, mock.Anything).Maybe()

		lhMock.EXPECT().Bgp("lights").Return(lightsBGPMock).Maybe()
		lhMock.EXPECT().Bgp("light_cull").Return(cullBGPMock).Maybe()
		lhMock.EXPECT().Bgp("tile_lit").Return(tileBGPMock).Maybe()
		lhMock.EXPECT().Resize(mock.Anything, mock.Anything).Maybe()
		lhMock.EXPECT().TileCountX().Return(4).Maybe()
		lhMock.EXPECT().TileCountY().Return(4).Maybe()
		lhMock.EXPECT().MaxLightsPerTile().Return(32).Maybe()
		lhMock.EXPECT().SetPipelineKey("light_cull", "light_cull_compute").Maybe()

		cullShaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		grp := 5
		tileDecl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &grp,
			Args:  []shader.AnnotationArg{"", "", shader.AnnotationArgTileUniforms},
		}
		litShaderMock.EXPECT().Declarations().Return([]shader.Annotation{tileDecl}).Maybe()
		litShaderMock.EXPECT().BindGroupLayoutDescriptor(5).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		cullBGPMock.EXPECT().Buffer(2).Return(nil).Maybe()
		cullBGPMock.EXPECT().Buffer(3).Return(nil).Maybe()

		return lhMock, lightsBGPMock, cullBGPMock, tileBGPMock, cullShaderMock, litShaderMock
	}

	suite.Run("cullComputeShader nil returns early", func() {
		litMock := shader_mocks.NewMockShader(suite.T())
		suite.scene.initLightCullResources(nil, litMock, 800, 600)
	})

	suite.Run("litFragmentShader nil returns early", func() {
		cullMock := shader_mocks.NewMockShader(suite.T())
		suite.scene.initLightCullResources(cullMock, nil, 800, 600)
	})

	suite.Run("lightsBGP Buffer(1) nil returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lightsBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		cullMock := shader_mocks.NewMockShader(suite.T())
		litMock := shader_mocks.NewMockShader(suite.T())
		lhMock.EXPECT().Bgp("lights").Return(lightsBGPMock).Once()
		lightsBGPMock.EXPECT().Buffer(1).Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initLightCullResources(cullMock, litMock, 800, 600) })
	})

	suite.Run("InitBindGroup for cullBGP error panics", func() {
		lhMock, _, cullBGPMock, _, cullShaderMock, litShaderMock := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("cull err")).Once()
		suite.Panics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("RegisterPipelines error panics", func() {
		lhMock, _, cullBGPMock, _, cullShaderMock, litShaderMock := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("reg err")).Once()
		suite.Panics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("no matching tile decl empty declarations panics", func() {
		lhMock, _, cullBGPMock, _, cullShaderMock, _ := makeBase()
		litShaderMock := shader_mocks.NewMockShader(suite.T())
		litShaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.Panics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("decl with wrong type is skipped panics", func() {
		lhMock, _, cullBGPMock, _, cullShaderMock, _ := makeBase()
		grp := 5
		wrongDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &grp,
			Args:  []shader.AnnotationArg{"", "", shader.AnnotationArgTileUniforms},
		}
		litShaderMock := shader_mocks.NewMockShader(suite.T())
		litShaderMock.EXPECT().Declarations().Return([]shader.Annotation{wrongDecl}).Maybe()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.Panics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("decl with nil Group is skipped panics", func() {
		lhMock, _, cullBGPMock, _, cullShaderMock, _ := makeBase()
		nilGrpDecl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: nil,
			Args:  []shader.AnnotationArg{"", "", shader.AnnotationArgTileUniforms},
		}
		litShaderMock := shader_mocks.NewMockShader(suite.T())
		litShaderMock.EXPECT().Declarations().Return([]shader.Annotation{nilGrpDecl}).Maybe()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.Panics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("InitBindGroup for tileBGP error panics", func() {
		lhMock, _, cullBGPMock, tileBGPMock, cullShaderMock, litShaderMock := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("tile err")).Once()
		suite.Panics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("cullBGP Buffer(2) non-nil tileBGP SetBuffer(1) called", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lightsBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		cullBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		tileBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		cullShaderMock := shader_mocks.NewMockShader(suite.T())
		litShaderMock := shader_mocks.NewMockShader(suite.T())

		lightsBGPMock.EXPECT().Buffer(1).Return(&wgpu.Buffer{}).Maybe()
		cullBGPMock.EXPECT().SetBuffer(1, mock.Anything).Maybe()

		lhMock.EXPECT().Bgp("lights").Return(lightsBGPMock).Maybe()
		lhMock.EXPECT().Bgp("light_cull").Return(cullBGPMock).Maybe()
		lhMock.EXPECT().Bgp("tile_lit").Return(tileBGPMock).Maybe()
		lhMock.EXPECT().Resize(mock.Anything, mock.Anything).Maybe()
		lhMock.EXPECT().TileCountX().Return(4).Maybe()
		lhMock.EXPECT().TileCountY().Return(4).Maybe()
		lhMock.EXPECT().MaxLightsPerTile().Return(32).Maybe()
		lhMock.EXPECT().SetPipelineKey("light_cull", "light_cull_compute").Maybe()

		cullShaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		grp := 5
		tileDecl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &grp,
			Args:  []shader.AnnotationArg{"", "", shader.AnnotationArgTileUniforms},
		}
		litShaderMock.EXPECT().Declarations().Return([]shader.Annotation{tileDecl}).Maybe()
		litShaderMock.EXPECT().BindGroupLayoutDescriptor(5).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		cullBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Maybe()
		cullBGPMock.EXPECT().Buffer(3).Return(nil).Maybe()
		tileBGPMock.EXPECT().SetBuffer(1, mock.Anything).Once()

		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("cullBGP Buffer(3) non-nil tileBGP SetBuffer(2) called", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lightsBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		cullBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		tileBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		cullShaderMock := shader_mocks.NewMockShader(suite.T())
		litShaderMock := shader_mocks.NewMockShader(suite.T())

		lightsBGPMock.EXPECT().Buffer(1).Return(&wgpu.Buffer{}).Maybe()
		cullBGPMock.EXPECT().SetBuffer(1, mock.Anything).Maybe()

		lhMock.EXPECT().Bgp("lights").Return(lightsBGPMock).Maybe()
		lhMock.EXPECT().Bgp("light_cull").Return(cullBGPMock).Maybe()
		lhMock.EXPECT().Bgp("tile_lit").Return(tileBGPMock).Maybe()
		lhMock.EXPECT().Resize(mock.Anything, mock.Anything).Maybe()
		lhMock.EXPECT().TileCountX().Return(4).Maybe()
		lhMock.EXPECT().TileCountY().Return(4).Maybe()
		lhMock.EXPECT().MaxLightsPerTile().Return(32).Maybe()
		lhMock.EXPECT().SetPipelineKey("light_cull", "light_cull_compute").Maybe()

		cullShaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		grp := 5
		tileDecl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &grp,
			Args:  []shader.AnnotationArg{"", "", shader.AnnotationArgTileUniforms},
		}
		litShaderMock.EXPECT().Declarations().Return([]shader.Annotation{tileDecl}).Maybe()
		litShaderMock.EXPECT().BindGroupLayoutDescriptor(5).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()

		cullBGPMock.EXPECT().Buffer(2).Return(nil).Maybe()
		cullBGPMock.EXPECT().Buffer(3).Return(&wgpu.Buffer{}).Maybe()
		tileBGPMock.EXPECT().SetBuffer(2, mock.Anything).Once()

		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("full happy path both Buffer(2) and Buffer(3) nil", func() {
		lhMock, _, cullBGPMock, tileBGPMock, cullShaderMock, litShaderMock := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})
}

func (suite *sceneImplTest) TestInitSSR() {
	makeReadyHandlers := func() (
		*light_mocks.MockLightingHandler,
		*light_mocks.MockSSRHandler,
		*light_mocks.MockGBufferHandler,
		*light_mocks.MockCompositionHandler,
	) {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := light_mocks.NewMockSSRHandler(suite.T())
		gbMock := light_mocks.NewMockGBufferHandler(suite.T())
		compMock := light_mocks.NewMockCompositionHandler(suite.T())
		lhMock.EXPECT().SSRHandler().Return(ssrMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbMock).Maybe()
		lhMock.EXPECT().CompositionHandler().Return(compMock).Maybe()
		gbMock.EXPECT().Enabled().Return(true).Maybe()
		compMock.EXPECT().Enabled().Return(true).Maybe()
		return lhMock, ssrMock, gbMock, compMock
	}

	makeFullBase := func(mipCount int) (
		*light_mocks.MockLightingHandler,
		*light_mocks.MockSSRHandler,
		*light_mocks.MockGBufferHandler,
		*light_mocks.MockCompositionHandler,
	) {
		lhMock, ssrMock, gbMock, compMock := makeReadyHandlers()
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		mipReadViews := make([]*wgpu.TextureView, mipCount)
		mipStorageViews := make([]*wgpu.TextureView, mipCount)
		for i := range mipReadViews {
			mipReadViews[i] = &wgpu.TextureView{}
		}
		for i := range mipStorageViews {
			mipStorageViews[i] = &wgpu.TextureView{}
		}
		suite.rendererMock.EXPECT().CreateSSRTextures(mock.Anything, mock.Anything).
			Return(&wgpu.TextureView{}, &wgpu.Texture{}, nil).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().
			Return(&wgpu.Sampler{}, nil).Maybe()
		suite.rendererMock.EXPECT().CreateHiZTextures(mock.Anything, mock.Anything).
			Return(&wgpu.TextureView{}, &wgpu.Texture{}, mipReadViews, mipStorageViews, mipCount, nil).Maybe()
		ssrMock.EXPECT().SetSSRTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetSSRTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipCount(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipReadViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZStorageViews(mock.Anything).Maybe()
		gbMock.EXPECT().DepthTextureView().Return(&wgpu.TextureView{}).Maybe()
		gbMock.EXPECT().NormalTextureView().Return(&wgpu.TextureView{}).Maybe()
		compMock.EXPECT().HDRTextureView().Return(&wgpu.TextureView{}).Maybe()
		return lhMock, ssrMock, gbMock, compMock
	}

	suite.Run("ssrHandler nil returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lhMock.EXPECT().SSRHandler().Return(nil)
		lhMock.EXPECT().GBufferHandler().Return(light_mocks.NewMockGBufferHandler(suite.T())).Maybe()
		lhMock.EXPECT().CompositionHandler().Return(light_mocks.NewMockCompositionHandler(suite.T())).Maybe()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("gbHandler nil returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := light_mocks.NewMockSSRHandler(suite.T())
		lhMock.EXPECT().SSRHandler().Return(ssrMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(nil).Maybe()
		lhMock.EXPECT().CompositionHandler().Return(light_mocks.NewMockCompositionHandler(suite.T())).Maybe()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("compHandler nil returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := light_mocks.NewMockSSRHandler(suite.T())
		gbMock := light_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().SSRHandler().Return(ssrMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbMock).Maybe()
		lhMock.EXPECT().CompositionHandler().Return(nil).Maybe()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("gbHandler not enabled returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := light_mocks.NewMockSSRHandler(suite.T())
		gbMock := light_mocks.NewMockGBufferHandler(suite.T())
		compMock := light_mocks.NewMockCompositionHandler(suite.T())
		lhMock.EXPECT().SSRHandler().Return(ssrMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbMock).Maybe()
		lhMock.EXPECT().CompositionHandler().Return(compMock).Maybe()
		gbMock.EXPECT().Enabled().Return(false).Maybe()
		compMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("compHandler not enabled returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := light_mocks.NewMockSSRHandler(suite.T())
		gbMock := light_mocks.NewMockGBufferHandler(suite.T())
		compMock := light_mocks.NewMockCompositionHandler(suite.T())
		lhMock.EXPECT().SSRHandler().Return(ssrMock).Maybe()
		lhMock.EXPECT().GBufferHandler().Return(gbMock).Maybe()
		lhMock.EXPECT().CompositionHandler().Return(compMock).Maybe()
		gbMock.EXPECT().Enabled().Return(true).Maybe()
		compMock.EXPECT().Enabled().Return(false).Maybe()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("zero screenWidth returns early", func() {
		lhMock, _, _, _ := makeReadyHandlers()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 600
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("zero screenHeight returns early", func() {
		lhMock, _, _, _ := makeReadyHandlers()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 0
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("halfW clamped to 1 when screenWidth=1", func() {
		lhMock, _, _, _ := makeReadyHandlers()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 1
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateSSRTextures(1, mock.Anything).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), errors.New("x")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("halfH clamped to 1 when screenHeight=1", func() {
		lhMock, _, _, _ := makeReadyHandlers()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 1
		suite.rendererMock.EXPECT().CreateSSRTextures(mock.Anything, 1).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), errors.New("x")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("CreateSSRTextures error panics", func() {
		lhMock, _, _, _ := makeReadyHandlers()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateSSRTextures(mock.Anything, mock.Anything).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), errors.New("ssr err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("CreateLinearSampler error panics", func() {
		lhMock, ssrMock, _, _ := makeReadyHandlers()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateSSRTextures(mock.Anything, mock.Anything).
			Return(&wgpu.TextureView{}, &wgpu.Texture{}, nil).Once()
		ssrMock.EXPECT().SetSSRTexture(mock.Anything).Once()
		ssrMock.EXPECT().SetSSRTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateLinearSampler().
			Return((*wgpu.Sampler)(nil), errors.New("samp err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("CreateHiZTextures error panics", func() {
		lhMock, ssrMock, _, _ := makeReadyHandlers()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateSSRTextures(mock.Anything, mock.Anything).
			Return(&wgpu.TextureView{}, &wgpu.Texture{}, nil).Once()
		ssrMock.EXPECT().SetSSRTexture(mock.Anything).Once()
		ssrMock.EXPECT().SetSSRTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateLinearSampler().
			Return(&wgpu.Sampler{}, nil).Once()
		ssrMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateHiZTextures(mock.Anything, mock.Anything).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil), nil, nil, 0, errors.New("hiz err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("RegisterPipelines hizInit error panics", func() {
		lhMock, _, _, _ := makeFullBase(1)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("reg err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("InitBindGroup hizInitBGP error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(1)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("init err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("RegisterPipelines hizDown error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(1)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("down err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("InitBindGroup in mip loop error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(2)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("mip err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("mip loop skipped when mipCount=1", func() {
		lhMock, ssrMock, _, _ := makeFullBase(1)
		ssrBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		ssrMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		ssrMock.EXPECT().SetBgp(mock.Anything, mock.Anything).Maybe()
		ssrMock.EXPECT().Bgp("ssr_compute").Return(ssrBGPMock).Once()
		ssrBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Maybe()
		ssrMock.EXPECT().Resize(800, 600).Once()
		ssrMock.EXPECT().SetEnabled(true).Once()
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("RegisterPipelines ssrCompute error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(1)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("ssr reg err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("InitBindGroup ssrBGP error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(1)
		ssrBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("ssr_compute", "ssr_compute").Once()
		ssrMock.EXPECT().Bgp("ssr_compute").Return(ssrBGPMock).Once()
		ssrBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(ssrBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("ssr bind err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("full happy path mipCount=1", func() {
		lhMock, ssrMock, _, _ := makeFullBase(1)
		ssrBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("ssr_compute", "ssr_compute").Once()
		ssrMock.EXPECT().Bgp("ssr_compute").Return(ssrBGPMock).Once()
		ssrBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(ssrBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().Resize(800, 600).Once()
		ssrMock.EXPECT().SetEnabled(true).Once()
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("full happy path mipCount=2", func() {
		lhMock, ssrMock, _, _ := makeFullBase(2)
		ssrBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_1", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("ssr_compute", "ssr_compute").Once()
		ssrMock.EXPECT().Bgp("ssr_compute").Return(ssrBGPMock).Once()
		ssrBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(ssrBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().Resize(800, 600).Once()
		ssrMock.EXPECT().SetEnabled(true).Once()
		suite.NotPanics(func() { suite.scene.initSSR() })
	})
}

func (suite *sceneImplTest) TestInitComposition() {
	makeBase := func() (
		*light_mocks.MockLightingHandler,
		*light_mocks.MockCompositionHandler,
		*light_mocks.MockSSRHandler,
		*bgp_mocks.MockBindGroupProvider,
	) {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := light_mocks.NewMockCompositionHandler(suite.T())
		ssrMock := light_mocks.NewMockSSRHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.buildInjectionMap()

		lhMock.EXPECT().CompositionHandler().Return(chMock).Maybe()
		lhMock.EXPECT().SSRHandler().Return(ssrMock).Maybe()

		suite.rendererMock.EXPECT().SampleCount().Return(uint32(1)).Maybe()
		suite.rendererMock.EXPECT().CreateCompositionTextures(800, 600, uint32(1)).
			Return(nil, nil, nil, nil, nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().SetRenderTargetFormat(wgpu.TextureFormatRGBA16Float).Maybe()

		chMock.EXPECT().SetHDRTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetHDRTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetMSAATexture(mock.Anything).Maybe()
		chMock.EXPECT().SetMSAATextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()

		return lhMock, chMock, ssrMock, bgpMock
	}

	makeFullBase := func() (
		*light_mocks.MockLightingHandler,
		*light_mocks.MockCompositionHandler,
		*light_mocks.MockSSRHandler,
		*bgp_mocks.MockBindGroupProvider,
	) {
		lhMock, chMock, ssrMock, bgpMock := makeBase()

		suite.rendererMock.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Maybe()
		chMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterCompositionPipeline(mock.Anything).Return(nil).Maybe()
		chMock.EXPECT().SetPipelineKey("composition", "composition").Maybe()
		chMock.EXPECT().Bgp("composition").Return(bgpMock).Maybe()
		bgpMock.EXPECT().SetTextureView(0, mock.Anything).Maybe()
		bgpMock.EXPECT().SetSampler(1, mock.Anything).Maybe()
		bgpMock.EXPECT().SetSampler(3, mock.Anything).Maybe()
		bgpMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()

		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return((*wgpu.Buffer)(nil), nil).Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		chMock.EXPECT().Exposure().Return(float32(1.0)).Maybe()
		chMock.EXPECT().SetExposureBuffer(mock.Anything).Maybe()
		chMock.EXPECT().HDRTextureView().Return(nil).Maybe()
		lumBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lumBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		lumBGPMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()
		chMock.EXPECT().Bgp("luminance_compute").Return(lumBGPMock).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(lumBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		chMock.EXPECT().BloomEnabled().Return(false).Maybe()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 6, mock.Anything).Return(nil).Maybe()

		return lhMock, chMock, ssrMock, bgpMock
	}

	suite.Run("nil CompositionHandler returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lhMock.EXPECT().CompositionHandler().Return(nil).Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initComposition() })
	})

	suite.Run("zero screenWidth returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := light_mocks.NewMockCompositionHandler(suite.T())
		lhMock.EXPECT().CompositionHandler().Return(chMock).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 600
		suite.NotPanics(func() { suite.scene.initComposition() })
	})

	suite.Run("zero screenHeight returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := light_mocks.NewMockCompositionHandler(suite.T())
		lhMock.EXPECT().CompositionHandler().Return(chMock).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 0
		suite.NotPanics(func() { suite.scene.initComposition() })
	})

	suite.Run("CreateCompositionTextures error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := light_mocks.NewMockCompositionHandler(suite.T())
		lhMock.EXPECT().CompositionHandler().Return(chMock).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().SampleCount().Return(uint32(1)).Once()
		suite.rendererMock.EXPECT().CreateCompositionTextures(800, 600, uint32(1)).
			Return(nil, nil, nil, nil, nil, nil, errors.New("tex err")).Once()
		suite.Panics(func() { suite.scene.initComposition() })
	})

	suite.Run("CreateLinearSampler error panics", func() {
		lhMock, _, _, _ := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().CreateLinearSampler().Return(nil, errors.New("samp err")).Once()
		suite.Panics(func() { suite.scene.initComposition() })
	})

	suite.Run("RegisterCompositionPipeline error panics", func() {
		lhMock, chMock, _, _ := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}, nil).Once()
		chMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterCompositionPipeline(mock.Anything).Return(errors.New("pipe err")).Once()
		suite.Panics(func() { suite.scene.initComposition() })
	})

	suite.Run("SSRTextureView non-nil calls SetTextureView at binding 2", func() {
		lhMock, chMock, ssrMock, bgpMock := makeFullBase()
		suite.scene.lightHandler = lhMock
		ssrMock.EXPECT().SSRTextureView().Return(&wgpu.TextureView{}).Times(2)
		bgpMock.EXPECT().SetTextureView(2, mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		chMock.EXPECT().Resize(800, 600).Once()
		chMock.EXPECT().SetEnabled(true).Once()
		suite.NotPanics(func() { suite.scene.initComposition() })
	})

	suite.Run("SSRTextureView nil InitTextureView error panics", func() {
		lhMock, _, ssrMock, bgpMock := makeFullBase()
		suite.scene.lightHandler = lhMock
		ssrMock.EXPECT().SSRTextureView().Return((*wgpu.TextureView)(nil)).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 2, mock.Anything).Return(errors.New("fallback err")).Once()
		suite.Panics(func() { suite.scene.initComposition() })
	})

	suite.Run("SSRTextureView nil InitTextureView fallback succeeds", func() {
		lhMock, chMock, ssrMock, bgpMock := makeFullBase()
		suite.scene.lightHandler = lhMock
		ssrMock.EXPECT().SSRTextureView().Return((*wgpu.TextureView)(nil)).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 2, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		chMock.EXPECT().Resize(800, 600).Once()
		chMock.EXPECT().SetEnabled(true).Once()
		suite.NotPanics(func() { suite.scene.initComposition() })
	})

	suite.Run("InitBindGroup error panics", func() {
		lhMock, _, ssrMock, bgpMock := makeFullBase()
		suite.scene.lightHandler = lhMock
		ssrMock.EXPECT().SSRTextureView().Return(&wgpu.TextureView{}).Times(2)
		bgpMock.EXPECT().SetTextureView(2, mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("bind err")).Once()
		suite.Panics(func() { suite.scene.initComposition() })
	})

	suite.Run("full happy path SSR view non-nil", func() {
		lhMock, chMock, ssrMock, bgpMock := makeFullBase()
		suite.scene.lightHandler = lhMock
		ssrMock.EXPECT().SSRTextureView().Return(&wgpu.TextureView{}).Times(2)
		bgpMock.EXPECT().SetTextureView(2, mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		chMock.EXPECT().Resize(800, 600).Once()
		chMock.EXPECT().SetEnabled(true).Once()
		suite.NotPanics(func() { suite.scene.initComposition() })
	})
}

func (suite *sceneImplTest) TestInitLighting() {
	makeMocks := func() (
		lhMock *light_mocks.MockLightingHandler,
		shMock *light_mocks.MockShadowHandler,
		compMock *light_mocks.MockCompositionHandler,
		ssrMock *light_mocks.MockSSRHandler,
		camMock *camera_mocks.MockCamera,
		camBGPMock *bgp_mocks.MockBindGroupProvider,
	) {
		lhMock = light_mocks.NewMockLightingHandler(suite.T())
		shMock = light_mocks.NewMockShadowHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbMock := light_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := light_mocks.NewMockSSAOHandler(suite.T())
		compMock = light_mocks.NewMockCompositionHandler(suite.T())
		ssrMock = light_mocks.NewMockSSRHandler(suite.T())
		camMock = camera_mocks.NewMockCamera(suite.T())
		camBGPMock = bgp_mocks.NewMockBindGroupProvider(suite.T())

		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		shMock.EXPECT().ShadowMapResolution().Return(1024).Maybe()
		shMock.EXPECT().CascadeCount().Return(2).Maybe()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Maybe()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Maybe()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Maybe()
		shMock.EXPECT().SetBgp(mock.Anything, mock.Anything).Maybe()
		shMock.EXPECT().Bgp(mock.Anything).Return(bgp_mocks.NewMockBindGroupProvider(suite.T())).Maybe()
		shMock.EXPECT().LightShadowTileSize().Return(256).Maybe()
		shMock.EXPECT().SetLightShadowAtlasSlots(mock.Anything).Maybe()
		shMock.EXPECT().SetLightShadowAtlasCols(mock.Anything).Maybe()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Maybe()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Maybe()
		shMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		shMock.EXPECT().CSMAtlasTextureView().Return(nil).Maybe()
		shMock.EXPECT().ComparisonSampler().Return(nil).Maybe()
		shMock.EXPECT().PCFSamples().Return(uint32(16)).Maybe()
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		shMock.EXPECT().LightShadowAtlasSlots().Return(6).Maybe()

		csBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		csBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Maybe()
		csMock.EXPECT().Enabled().Return(true).Maybe()
		csMock.EXPECT().TextureView().Return(nil).Maybe()
		csMock.EXPECT().LinearSampler().Return(nil).Maybe()
		csMock.EXPECT().SetTexture(mock.Anything).Maybe()
		csMock.EXPECT().SetTextureView(mock.Anything).Maybe()
		csMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		csMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		csMock.EXPECT().Bgp(mock.Anything).Return(csBGPMock).Maybe()
		csMock.EXPECT().SetEnabled(mock.Anything).Maybe()

		lhMock.EXPECT().GBufferHandler().Return(gbMock).Maybe()
		gbMock.EXPECT().Enabled().Return(true).Maybe()
		gbMock.EXPECT().SetNormalTexture(mock.Anything).Maybe()
		gbMock.EXPECT().SetNormalTextureView(mock.Anything).Maybe()
		gbMock.EXPECT().SetAlbedoTexture(mock.Anything).Maybe()
		gbMock.EXPECT().SetAlbedoTextureView(mock.Anything).Maybe()
		gbMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		gbMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		gbMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		gbMock.EXPECT().Resize(mock.Anything, mock.Anything).Maybe()
		gbMock.EXPECT().SetEnabled(mock.Anything).Maybe()
		gbMock.EXPECT().DepthTextureView().Return(nil).Maybe()
		gbMock.EXPECT().NormalTextureView().Return(nil).Maybe()

		ssaoBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssaoBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		lhMock.EXPECT().SSAOHandler().Return(ssaoMock).Maybe()
		ssaoMock.EXPECT().HalfResolution().Return(false).Maybe()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		ssaoMock.EXPECT().Bgp(mock.Anything).Return(ssaoBGPMock).Maybe()
		ssaoMock.EXPECT().SampleCount().Return(8).Maybe()
		ssaoMock.EXPECT().MaxSamples().Return(32).Maybe()
		ssaoMock.EXPECT().Resize(mock.Anything, mock.Anything).Maybe()
		ssaoMock.EXPECT().SetEnabled(mock.Anything).Maybe()
		ssaoMock.EXPECT().Enabled().Return(false).Maybe()
		ssaoMock.EXPECT().BlurredTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().LinearSampler().Return(nil).Maybe()

		lightsBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lightsBGPMock.EXPECT().Buffer(mock.Anything).Return(nil).Maybe()
		lightsBGPMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()
		lhMock.EXPECT().Bgp("lights").Return(lightsBGPMock).Maybe()
		lhMock.EXPECT().Bgp("light_cull").Return(bgp_mocks.NewMockBindGroupProvider(suite.T())).Maybe()
		lhMock.EXPECT().Bgp("tile_lit").Return(bgp_mocks.NewMockBindGroupProvider(suite.T())).Maybe()
		lhMock.EXPECT().Bgp("ssao_lit").Return(bgp_mocks.NewMockBindGroupProvider(suite.T())).Maybe()
		lhMock.EXPECT().MaxGPULights().Return(1).Maybe()
		lhMock.EXPECT().Resize(mock.Anything, mock.Anything).Maybe()
		lhMock.EXPECT().TileCountX().Return(4).Maybe()
		lhMock.EXPECT().TileCountY().Return(4).Maybe()
		lhMock.EXPECT().TileSize().Return(16).Maybe()
		lhMock.EXPECT().MaxLightsPerTile().Return(32).Maybe()
		lhMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()

		lhMock.EXPECT().CompositionHandler().Return(compMock).Maybe()
		compMock.EXPECT().SetHDRTexture(mock.Anything).Maybe()
		compMock.EXPECT().SetHDRTextureView(mock.Anything).Maybe()
		compMock.EXPECT().SetMSAATexture(mock.Anything).Maybe()
		compMock.EXPECT().SetMSAATextureView(mock.Anything).Maybe()
		compMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		compMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		compMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		compMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		compMock.EXPECT().Resize(mock.Anything, mock.Anything).Maybe()
		compMock.EXPECT().SetEnabled(mock.Anything).Maybe()
		compMock.EXPECT().HDRTextureView().Return(nil).Maybe()
		compMock.EXPECT().Enabled().Return(true).Maybe()
		compMock.EXPECT().Exposure().Return(float32(1.0)).Maybe()
		compMock.EXPECT().SetExposureBuffer(mock.Anything).Maybe()
		compMock.EXPECT().LuminanceWorkgroupSize().Return(16).Maybe()
		compMock.EXPECT().LuminanceWorkgroupSize().Return(16).Maybe()
		compMock.EXPECT().BloomEnabled().Return(false).Maybe()

		ssrInternalBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssrInternalBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		lhMock.EXPECT().SSRHandler().Return(ssrMock).Maybe()
		ssrMock.EXPECT().SetSSRTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetSSRTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipCount(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipReadViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZStorageViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		ssrMock.EXPECT().Bgp(mock.Anything).Return(ssrInternalBGPMock).Maybe()
		ssrMock.EXPECT().Resize(mock.Anything, mock.Anything).Maybe()
		ssrMock.EXPECT().SetEnabled(mock.Anything).Maybe()
		ssrMock.EXPECT().SetBgp(mock.Anything, mock.Anything).Maybe()

		camMock.EXPECT().BindGroupProvider().Return(camBGPMock).Maybe()
		camBGPMock.EXPECT().SetBindGroupLayout(mock.Anything).Maybe()

		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Maybe()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil, nil).Maybe()
		suite.rendererMock.EXPECT().RegisterShadowDepthPipeline(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateGBufferTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateSSAOTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return(nil, nil).Maybe()
		suite.rendererMock.EXPECT().CreateContactShadowTextures(mock.Anything, mock.Anything).Return(nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().SampleCount().Return(uint32(1)).Maybe()
		suite.rendererMock.EXPECT().CreateCompositionTextures(mock.Anything, mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().SetRenderTargetFormat(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateSSRTextures(mock.Anything, mock.Anything).Return(nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().CreateHiZTextures(mock.Anything, mock.Anything).Return(nil, nil, make([]*wgpu.TextureView, 1), make([]*wgpu.TextureView, 1), 1, nil).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterCompositionPipeline(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return((*wgpu.Buffer)(nil), nil).Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Maybe()

		return
	}

	setupSceneFields := func(lhMock *light_mocks.MockLightingHandler, camMock *camera_mocks.MockCamera) {
		suite.scene.cam = camMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.buildInjectionMap()
	}

	mkInitCompBGP := func(compMock *light_mocks.MockCompositionHandler, ssrMock *light_mocks.MockSSRHandler) *bgp_mocks.MockBindGroupProvider {
		bgpMockComp := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpMockComp.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		bgpMockComp.EXPECT().SetSampler(mock.Anything, mock.Anything).Maybe()
		bgpMockComp.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()
		compMock.EXPECT().Bgp("composition").Return(bgpMockComp).Once()
		ssrMock.EXPECT().SSRTextureView().Return(nil).Once()
		lumBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lumBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		lumBGPMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()
		compMock.EXPECT().Bgp("luminance_compute").Return(lumBGPMock).Maybe()
		return bgpMockComp
	}

	suite.Run("SSR disabled skips re-bind SetEnabled called", func() {
		lhMock, _, compMock, ssrMock, camMock, _ := makeMocks()
		setupSceneFields(lhMock, camMock)
		mkInitCompBGP(compMock, ssrMock)
		ssrMock.EXPECT().Enabled().Return(false).Once()
		lhMock.EXPECT().SetEnabled(true).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.NotPanics(func() { suite.scene.initLighting(800, 600) })
	})

	suite.Run("SSR comp enabled Bgp nil inner block skipped SetEnabled called", func() {
		lhMock, _, compMock, ssrMock, camMock, _ := makeMocks()
		setupSceneFields(lhMock, camMock)
		mkInitCompBGP(compMock, ssrMock)
		ssrMock.EXPECT().Enabled().Return(true).Once()
		compMock.EXPECT().Bgp("composition").Return(nil).Once()
		lhMock.EXPECT().SetEnabled(true).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.NotPanics(func() { suite.scene.initLighting(800, 600) })
	})

	suite.Run("SSR comp enabled Bgp non-nil SSRTextureView nil inner block skipped", func() {
		lhMock, _, compMock, ssrMock, camMock, _ := makeMocks()
		setupSceneFields(lhMock, camMock)
		mkInitCompBGP(compMock, ssrMock)
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssrMock.EXPECT().Enabled().Return(true).Once()
		compMock.EXPECT().Bgp("composition").Return(compBGPMock).Once()
		ssrMock.EXPECT().SSRTextureView().Return(nil).Once()
		lhMock.EXPECT().SetEnabled(true).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.NotPanics(func() { suite.scene.initLighting(800, 600) })
	})

	suite.Run("SSR re-bind InitBindGroup error panics", func() {
		lhMock, _, compMock, ssrMock, camMock, _ := makeMocks()
		setupSceneFields(lhMock, camMock)
		mkInitCompBGP(compMock, ssrMock)
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssrMock.EXPECT().Enabled().Return(true).Once()
		compMock.EXPECT().Bgp("composition").Return(compBGPMock).Once()
		ssrMock.EXPECT().SSRTextureView().Return(&wgpu.TextureView{}).Times(2)
		compBGPMock.EXPECT().SetTextureView(2, mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(compBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("rebind err")).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.Panics(func() { suite.scene.initLighting(800, 600) })
	})

	suite.Run("full happy path SSR re-bind succeeds SetEnabled called", func() {
		lhMock, _, compMock, ssrMock, camMock, _ := makeMocks()
		setupSceneFields(lhMock, camMock)
		mkInitCompBGP(compMock, ssrMock)
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssrMock.EXPECT().Enabled().Return(true).Once()
		compMock.EXPECT().Bgp("composition").Return(compBGPMock).Once()
		ssrMock.EXPECT().SSRTextureView().Return(&wgpu.TextureView{}).Times(2)
		compBGPMock.EXPECT().SetTextureView(2, mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(compBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		lhMock.EXPECT().SetEnabled(true).Once()
		suite.NotPanics(func() { suite.scene.initLighting(800, 600) })
	})
}

func (suite *sceneImplTest) TestInitPhysics() {
	makeBase := func() (phMock *physics_mocks.MockPhysics, buffersBGPMock *bgp_mocks.MockBindGroupProvider, stageBGPMock *bgp_mocks.MockBindGroupProvider) {
		phMock = physics_mocks.NewMockPhysics(suite.T())
		buffersBGPMock = bgp_mocks.NewMockBindGroupProvider(suite.T())
		stageBGPMock = bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.physicsHandler = phMock
		phMock.EXPECT().SlotsPerCell().Return(uint32(16)).Maybe()
		phMock.EXPECT().BodyIdxMask().Return(uint32(0xFFFFFF)).Maybe()
		phMock.EXPECT().MaxBodies().Return(10).Maybe()
		phMock.EXPECT().MaxParticles().Return(100).Maybe()
		phMock.EXPECT().MaxGridCells().Return(50).Maybe()
		phMock.EXPECT().Buffers().Return(buffersBGPMock).Maybe()
		phMock.EXPECT().Bgp(mock.Anything).Return(stageBGPMock).Maybe()
		phMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		phMock.EXPECT().SetStagingBuffer(mock.Anything).Maybe()
		buffersBGPMock.EXPECT().Buffer(mock.Anything).Return((*wgpu.Buffer)(nil)).Maybe()
		stageBGPMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()
		return
	}

	suite.Run("InitBindGroup on buffers panics", func() {
		_, buffersBGPMock, _ := makeBase()
		suite.rendererMock.EXPECT().InitBindGroup(buffersBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("buf err")).Once()
		suite.Panics(func() { suite.scene.initPhysics() })
	})

	suite.Run("InitBindGroup on stage BGP panics", func() {
		_, buffersBGPMock, stageBGPMock := makeBase()
		suite.rendererMock.EXPECT().InitBindGroup(buffersBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(stageBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("stage err")).Once()
		suite.Panics(func() { suite.scene.initPhysics() })
	})

	suite.Run("RegisterPipelines for stage panics", func() {
		makeBase()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("pipe err")).Once()
		suite.Panics(func() { suite.scene.initPhysics() })
	})

	suite.Run("RegisterPipelines for bone update panics", func() {
		makeBase()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(9)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("bone err")).Once()
		suite.Panics(func() { suite.scene.initPhysics() })
	})

	suite.Run("CreateBuffer panics", func() {
		makeBase()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return((*wgpu.Buffer)(nil), errors.New("buf err")).Once()
		suite.Panics(func() { suite.scene.initPhysics() })
	})

	suite.Run("full happy path completes", func() {
		makeBase()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return((*wgpu.Buffer)(nil), nil).Once()
		suite.NotPanics(func() { suite.scene.initPhysics() })
	})
}

func (suite *sceneImplTest) TestInitPhysicsSyncGroup() {
	makeBase := func() (phMock *physics_mocks.MockPhysics, animMock *animator_mocks.MockAnimator) {
		phMock = physics_mocks.NewMockPhysics(suite.T())
		buffersBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		animMock = animator_mocks.NewMockAnimator(suite.T())
		animBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		syncShaderMock := shader_mocks.NewMockShader(suite.T())
		suite.scene.physicsHandler = phMock
		suite.scene.physicsAnimBinding = 2
		suite.scene.physicsSyncGroup = nil
		suite.scene.physicsSyncAnimMap = make(map[animator.Animator]int)
		phMock.EXPECT().Buffers().Return(buffersBGPMock).Maybe()
		buffersBGPMock.EXPECT().Buffer(mock.Anything).Return((*wgpu.Buffer)(nil)).Maybe()
		phMock.EXPECT().PipelineKey("sync").Return("sync_pipeline_key").Maybe()
		phMock.EXPECT().MaxBodies().Return(4).Maybe()
		animMock.EXPECT().ComputeBindGroupProvider().Return(animBGPMock).Maybe()
		animBGPMock.EXPECT().Buffer(2).Return((*wgpu.Buffer)(nil)).Maybe()
		suite.rendererMock.EXPECT().Pipeline("sync_pipeline_key").Return(pipeMock).Maybe()
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(syncShaderMock).Maybe()
		syncShaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		return phMock, animMock
	}

	suite.Run("physicsAnimBinding discovered from real shader", func() {
		phMock := physics_mocks.NewMockPhysics(suite.T())
		buffersBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		syncShaderMock := shader_mocks.NewMockShader(suite.T())
		suite.scene.physicsHandler = phMock
		suite.scene.physicsAnimBinding = -1
		suite.scene.physicsSyncGroup = nil
		suite.scene.physicsSyncAnimMap = make(map[animator.Animator]int)
		phMock.EXPECT().Buffers().Return(buffersBGPMock).Maybe()
		buffersBGPMock.EXPECT().Buffer(mock.Anything).Return((*wgpu.Buffer)(nil)).Maybe()
		phMock.EXPECT().PipelineKey("sync").Return("sync_pipeline_key").Maybe()
		phMock.EXPECT().MaxBodies().Return(4).Maybe()
		animMock.EXPECT().ComputeBindGroupProvider().Return(animBGPMock).Maybe()
		animBGPMock.EXPECT().Buffer(mock.Anything).Return((*wgpu.Buffer)(nil)).Maybe()
		suite.rendererMock.EXPECT().Pipeline("sync_pipeline_key").Return(pipeMock).Maybe()
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(syncShaderMock).Maybe()
		syncShaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Once()
		suite.NotPanics(func() { suite.scene.initPhysicsSyncGroup(animMock) })
		suite.GreaterOrEqual(suite.scene.physicsAnimBinding, 0)
	})

	suite.Run("InitBindGroup error panics", func() {
		_, animMock := makeBase()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("init err")).Once()
		suite.Panics(func() { suite.scene.initPhysicsSyncGroup(animMock) })
	})

	suite.Run("physicsSyncGroup nil initialized on first call", func() {
		_, animMock := makeBase()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Once()
		id := suite.scene.initPhysicsSyncGroup(animMock)
		suite.Equal(0, id)
		suite.NotNil(suite.scene.physicsSyncGroup)
	})

	suite.Run("second call increments groupID", func() {
		_, animMock := makeBase()
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: bgp_mocks.NewMockBindGroupProvider(suite.T())}
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Once()
		id := suite.scene.initPhysicsSyncGroup(animMock)
		suite.Equal(1, id)
	})

	suite.Run("full happy path returns 0 and populates maps", func() {
		_, animMock := makeBase()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Once()
		id := suite.scene.initPhysicsSyncGroup(animMock)
		suite.Equal(0, id)
		suite.NotNil(suite.scene.physicsSyncGroup[0])
		suite.Equal(0, suite.scene.physicsSyncAnimMap[animMock])
	})
}

func (suite *sceneImplTest) TestReinitCameraBGPForLitPipeline() {
	grp := 1
	cameraDecl := shader.Annotation{
		Type:  shader.AnnotationTypeBindingGroup,
		Group: &grp,
		Args:  []shader.AnnotationArg{"", "", shader.AnnotationArgCamera},
	}

	makeBase := func() (*shader_mocks.MockShader, *camera_mocks.MockCamera, *bgp_mocks.MockBindGroupProvider) {
		shaderMock := shader_mocks.NewMockShader(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.cam = camMock
		return shaderMock, camMock, bgpMock
	}

	suite.Run("nil shader returns early", func() {
		suite.NotPanics(func() { suite.scene.reinitCameraBGPForLitPipeline(nil) })
	})

	suite.Run("no camera declaration returns early", func() {
		shaderMock, _, _ := makeBase()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		suite.NotPanics(func() { suite.scene.reinitCameraBGPForLitPipeline(shaderMock) })
	})

	suite.Run("camera BGP nil returns early", func() {
		shaderMock, camMock, _ := makeBase()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{cameraDecl}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.NotPanics(func() { suite.scene.reinitCameraBGPForLitPipeline(shaderMock) })
	})

	suite.Run("InitBindGroup error panics", func() {
		shaderMock, camMock, bgpMock := makeBase()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{cameraDecl}).Once()
		camMock.EXPECT().BindGroupProvider().Return(bgpMock).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgpMock.EXPECT().SetBindGroupLayout((*wgpu.BindGroupLayout)(nil)).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("init err")).Once()
		suite.Panics(func() { suite.scene.reinitCameraBGPForLitPipeline(shaderMock) })
	})

	suite.Run("full happy path succeeds", func() {
		shaderMock, camMock, bgpMock := makeBase()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{cameraDecl}).Once()
		camMock.EXPECT().BindGroupProvider().Return(bgpMock).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{{Binding: 0, Visibility: wgpu.ShaderStageFragment}},
		}).Once()
		bgpMock.EXPECT().SetBindGroupLayout((*wgpu.BindGroupLayout)(nil)).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.NotPanics(func() { suite.scene.reinitCameraBGPForLitPipeline(shaderMock) })
	})
}

func (suite *sceneImplTest) TestPatchSyncMapEntry() {
	makeBase := func() (*physics_mocks.MockPhysics, *animator_mocks.MockAnimator, *bgp_mocks.MockBindGroupProvider) {
		phMock := physics_mocks.NewMockPhysics(suite.T())
		animMock := animator_mocks.NewMockAnimator(suite.T())
		syncBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.physicsHandler = phMock
		suite.scene.physicsSyncAnimMap = map[animator.Animator]int{animMock: 0}
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: syncBGPMock}
		suite.scene.physicsSyncWrites = nil
		return phMock, animMock, syncBGPMock
	}

	suite.Run("nil physicsHandler returns early", func() {
		suite.scene.physicsHandler = nil
		suite.scene.patchSyncMapEntry(animator_mocks.NewMockAnimator(suite.T()), 1, 0)
		suite.Empty(suite.scene.physicsSyncWrites)
	})

	suite.Run("nil anim returns early", func() {
		_, _, _ = makeBase()
		suite.scene.patchSyncMapEntry(nil, 1, 0)
		suite.Empty(suite.scene.physicsSyncWrites)
	})

	suite.Run("BodyIndex not found returns early", func() {
		phMock, animMock, _ := makeBase()
		phMock.EXPECT().BodyIndex(uint64(42)).Return(0, false).Once()
		suite.scene.patchSyncMapEntry(animMock, 42, 0)
		suite.Empty(suite.scene.physicsSyncWrites)
	})

	suite.Run("anim not in syncAnimMap returns early", func() {
		phMock, _, _ := makeBase()
		suite.scene.physicsSyncAnimMap = map[animator.Animator]int{}
		phMock.EXPECT().BodyIndex(uint64(1)).Return(0, true).Once()
		unknownAnim := animator_mocks.NewMockAnimator(suite.T())
		suite.scene.patchSyncMapEntry(unknownAnim, 1, 0)
		suite.Empty(suite.scene.physicsSyncWrites)
	})

	suite.Run("full happy path appends BufferWrite", func() {
		phMock, animMock, syncBGPMock := makeBase()
		phMock.EXPECT().BodyIndex(uint64(10)).Return(3, true).Once()
		suite.scene.patchSyncMapEntry(animMock, 10, 7)
		suite.Len(suite.scene.physicsSyncWrites, 1)
		suite.Equal(syncBGPMock, suite.scene.physicsSyncWrites[0].Provider)
		suite.Equal(1, suite.scene.physicsSyncWrites[0].Binding)
		suite.Equal(uint64(12), suite.scene.physicsSyncWrites[0].Offset)
		suite.Equal([]byte{7, 0, 0, 0}, suite.scene.physicsSyncWrites[0].Data)
	})
}

func (suite *sceneImplTest) TestCreateBoneParticleUpdateGroup() {
	makeBase := func() (*physics_mocks.MockPhysics, *animator_mocks.MockAnimator, *model_mocks.MockModel) {
		phMock := physics_mocks.NewMockPhysics(suite.T())
		animMock := animator_mocks.NewMockAnimator(suite.T())
		mdlMock := model_mocks.NewMockModel(suite.T())
		suite.scene.physicsHandler = phMock
		suite.scene.boneParticleUpdateGroups = nil
		return phMock, animMock, mdlMock
	}

	makePhysBGPMock := func() *bgp_mocks.MockBindGroupProvider {
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpMock.EXPECT().Buffer(1).Return(nil).Maybe()
		bgpMock.EXPECT().Buffer(0).Return(nil).Maybe()
		return bgpMock
	}

	makeAnimBGPMock := func() *bgp_mocks.MockBindGroupProvider {
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpMock.EXPECT().Buffer(5).Return(nil).Maybe()
		bgpMock.EXPECT().Buffer(6).Return(nil).Maybe()
		return bgpMock
	}

	suite.Run("particleCount zero returns early", func() {
		phMock, animMock, mdlMock := makeBase()
		phMock.EXPECT().BodyParticleInfo(2).Return(uint32(0), uint32(0)).Once()
		suite.NotPanics(func() {
			suite.scene.createBoneParticleUpdateGroup(animMock, 2, mdlMock, 0)
		})
		suite.Len(suite.scene.boneParticleUpdateGroups, 0)
	})

	suite.Run("nil pipeline panics", func() {
		phMock, animMock, mdlMock := makeBase()
		phMock.EXPECT().BodyParticleInfo(0).Return(uint32(0), uint32(4)).Once()
		mdlMock.EXPECT().Skeleton().Return(&model.Skeleton{Bones: []model.Bone{}}).Once()
		physBGPMock := makePhysBGPMock()
		phMock.EXPECT().Buffers().Return(physBGPMock).Maybe()
		animBGPMock := makeAnimBGPMock()
		animMock.EXPECT().ComputeBindGroupProvider().Return(animBGPMock).Maybe()
		phMock.EXPECT().PipelineKey("bone_update").Return("bone_update_key").Once()
		suite.rendererMock.EXPECT().Pipeline("bone_update_key").Return(nil).Once()
		suite.Panics(func() {
			suite.scene.createBoneParticleUpdateGroup(animMock, 0, mdlMock, 0)
		})
	})

	suite.Run("InitBindGroup error panics", func() {
		phMock, animMock, mdlMock := makeBase()
		phMock.EXPECT().BodyParticleInfo(0).Return(uint32(0), uint32(4)).Once()
		mdlMock.EXPECT().Skeleton().Return(&model.Skeleton{Bones: []model.Bone{}}).Once()
		physBGPMock := makePhysBGPMock()
		phMock.EXPECT().Buffers().Return(physBGPMock).Maybe()
		animBGPMock := makeAnimBGPMock()
		animMock.EXPECT().ComputeBindGroupProvider().Return(animBGPMock).Maybe()
		phMock.EXPECT().PipelineKey("bone_update").Return("bone_update_key").Once()
		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		shaderMock := shader_mocks.NewMockShader(suite.T())
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(shaderMock).Once()
		suite.rendererMock.EXPECT().Pipeline("bone_update_key").Return(pipeMock).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("init fail")).Once()
		suite.Panics(func() {
			suite.scene.createBoneParticleUpdateGroup(animMock, 0, mdlMock, 0)
		})
	})

	suite.Run("happy path appends boneParticleUpdateGroup", func() {
		phMock, animMock, mdlMock := makeBase()
		phMock.EXPECT().BodyParticleInfo(1).Return(uint32(10), uint32(8)).Once()
		mdlMock.EXPECT().Skeleton().Return(&model.Skeleton{Bones: []model.Bone{{Name: "root"}, {Name: "spine"}}}).Once()
		physBGPMock := makePhysBGPMock()
		phMock.EXPECT().Buffers().Return(physBGPMock).Maybe()
		animBGPMock := makeAnimBGPMock()
		animMock.EXPECT().ComputeBindGroupProvider().Return(animBGPMock).Maybe()
		phMock.EXPECT().PipelineKey("bone_update").Return("bone_update_key").Once()
		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		shaderMock := shader_mocks.NewMockShader(suite.T())
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(shaderMock).Once()
		suite.rendererMock.EXPECT().Pipeline("bone_update_key").Return(pipeMock).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() {
			suite.scene.createBoneParticleUpdateGroup(animMock, 1, mdlMock, 7)
		})
		suite.Len(suite.scene.boneParticleUpdateGroups, 1)
		grp := suite.scene.boneParticleUpdateGroups[0]
		suite.Equal(uint32(10), grp.particleStart)
		suite.Equal(uint32(8), grp.particleCount)
		suite.Equal(uint32(2), grp.boneCount)
		suite.Equal(uint32(7), grp.instanceIndex)
	})
}

func (suite *sceneImplTest) TestInitMaterialGPU() {
	makeBase := func() (*material_mocks.MockMaterial, *shader_mocks.MockShader) {
		matMock := material_mocks.NewMockMaterial(suite.T())
		shaderMock := shader_mocks.NewMockShader(suite.T())
		return matMock, shaderMock
	}

	validTexPath := filepath.Join("examples", "assets", "textures", "wood.png")

	suite.Run("empty declarations returns nil", func() {
		mat, frag := makeBase()
		frag.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("non-provider and nil-group declarations are skipped and return nil", func() {
		mat, frag := makeBase()
		g := 2
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
			{Type: shader.AnnotationTypeProvider, Group: nil, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("provider declaration with non-material first arg is skipped and returns nil", func() {
		mat, frag := makeBase()
		g := 2
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Args: []shader.AnnotationArg{shader.AnnotationArgBoneInfo}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("provider single arg stores group with no binding role all textures nil", func() {
		mat, frag := makeBase()
		g := 2
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("provider two args with binding stores role all textures nil no entries", func() {
		mat, frag := makeBase()
		g := 2
		b := 0
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &b, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseTexture}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("non-nil texture with no registered binding is skipped", func() {
		mat, frag := makeBase()
		g := 2
		b := 1
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &b, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgNormalTexture}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		mat.EXPECT().DiffuseTexture().Return(&common.ImportedTexture{Path: validTexPath}).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("decode error returns error", func() {
		mat, frag := makeBase()
		g := 2
		b := 0
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &b, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseTexture}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		mat.EXPECT().DiffuseTexture().Return(&common.ImportedTexture{}).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.Error(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("InitTextureView error on user texture returns error", func() {
		mat, frag := makeBase()
		g := 2
		b := 0
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &b, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseTexture}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		mat.EXPECT().DiffuseTexture().Return(&common.ImportedTexture{Path: validTexPath}).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, 0, mock.Anything).Return(errors.New("tex error")).Once()
		suite.Error(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("user texture with no sampler binding skips sampler initialization", func() {
		mat, frag := makeBase()
		g := 2
		b := 0
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &b, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseTexture}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		mat.EXPECT().DiffuseTexture().Return(&common.ImportedTexture{Path: validTexPath}).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, 0, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("user texture with sampler binding and nil SamplerData uses default sampler", func() {
		mat, frag := makeBase()
		g := 2
		texB := 0
		samplerB := 1
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &texB, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseTexture}},
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &samplerB, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseSampler}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		mat.EXPECT().DiffuseTexture().Return(&common.ImportedTexture{Path: validTexPath}).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, 0, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, 1, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("user texture with sampler binding and non-nil SamplerData uses custom sampler", func() {
		mat, frag := makeBase()
		g := 2
		texB := 0
		samplerB := 1
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &texB, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseTexture}},
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &samplerB, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseSampler}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		customSampler := &common.SamplerStagingData{MagFilter: wgpu.FilterModeNearest}
		mat.EXPECT().DiffuseTexture().Return(&common.ImportedTexture{Path: validTexPath, SamplerData: customSampler}).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, 0, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, 1, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("InitSampler error on user texture sampler returns error", func() {
		mat, frag := makeBase()
		g := 2
		texB := 0
		samplerB := 1
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &texB, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseTexture}},
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &samplerB, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseSampler}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		mat.EXPECT().DiffuseTexture().Return(&common.ImportedTexture{Path: validTexPath}).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, 0, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, 1, mock.Anything).Return(errors.New("sampler err")).Once()
		suite.Error(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("fallback normal texture uses 128 128 255 255 pixel", func() {
		mat, frag := makeBase()
		g := 2
		b := 1
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &b, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgNormalTexture}},
		}
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 1, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat}},
			},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(descriptor).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, 1, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("fallback metallic roughness texture calls Roughness and Metallic", func() {
		mat, frag := makeBase()
		g := 2
		b := 2
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &b, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgMetallicRoughnessTexture}},
		}
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 2, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat}},
			},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(descriptor).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		mat.EXPECT().Roughness().Return(float32(0.8)).Once()
		mat.EXPECT().Metallic().Return(float32(0.2)).Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, 2, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("fallback default texture uses 255 255 255 255 pixel", func() {
		mat, frag := makeBase()
		g := 2
		b := 0
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &b, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseTexture}},
		}
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat}},
			},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(descriptor).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, 0, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("fallback InitTextureView error returns error", func() {
		mat, frag := makeBase()
		g := 2
		b := 0
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &b, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgDiffuseTexture}},
		}
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Texture: wgpu.TextureBindingLayout{SampleType: wgpu.TextureSampleTypeFloat}},
			},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(descriptor).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, 0, mock.Anything).Return(errors.New("fallback tex error")).Once()
		suite.Error(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("fallback sampler is initialized when descriptor has sampler entry", func() {
		mat, frag := makeBase()
		g := 2
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
		}
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 1, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(descriptor).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, 1, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("fallback InitSampler error returns error", func() {
		mat, frag := makeBase()
		g := 2
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
		}
		descriptor := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 1, Sampler: wgpu.SamplerBindingLayout{Type: wgpu.SamplerBindingTypeFiltering}},
			},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(descriptor).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, 1, mock.Anything).Return(errors.New("sampler fallback err")).Once()
		suite.Error(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("InitBindGroup error returns error", func() {
		mat, frag := makeBase()
		g := 2
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("bind group error")).Once()
		suite.Error(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("material params binding triggers size override and WriteBuffers", func() {
		mat, frag := makeBase()
		g := 2
		b := 3
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g, Binding: &b, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial, shader.AnnotationArgMaterialParams}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		mat.EXPECT().AlphaCutoff().Return(float32(0.5)).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})

	suite.Run("second group calls SetProvider but not SetBindGroupProvider", func() {
		mat, frag := makeBase()
		g1 := 2
		g2 := 3
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeProvider, Group: &g1, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
			{Type: shader.AnnotationTypeProvider, Group: &g2, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}},
		}
		frag.EXPECT().Declarations().Return(decls).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		frag.EXPECT().BindGroupLayoutDescriptor(3).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		mat.EXPECT().DiffuseTexture().Return(nil).Once()
		mat.EXPECT().NormalTexture().Return(nil).Once()
		mat.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		mat.EXPECT().SetBindGroupProvider(mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(2, mock.Anything).Return().Once()
		mat.EXPECT().SetProvider(3, mock.Anything).Return().Once()
		suite.NoError(suite.scene.initMaterialGPU(mat, frag, "test"))
	})
}

func (suite *sceneImplTest) TestCreateAnimator() {
	makeBase := func() (*model_mocks.MockModel, *shader_mocks.MockShader, *shader_mocks.MockShader, *shader_mocks.MockShader) {
		return model_mocks.NewMockModel(suite.T()),
			shader_mocks.NewMockShader(suite.T()),
			shader_mocks.NewMockShader(suite.T()),
			shader_mocks.NewMockShader(suite.T())
	}

	suite.Run("simple backend empty declarations returns non-nil animator", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		anim := suite.scene.createAnimator(mdl, cs, vs, fs)
		suite.NotNil(anim)
	})

	suite.Run("mesh provider with nil vertex buffer calls InitMeshBuffers", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpMock.EXPECT().VertexBuffer().Return(nil).Once()
		mdl.EXPECT().MeshProvider().Return(bgpMock).Once()
		mdl.EXPECT().VertexData().Return(nil).Once()
		mdl.EXPECT().IndexData().Return(nil).Once()
		mdl.EXPECT().IndexCount().Return(0).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitMeshBuffers(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("mesh provider with non-nil vertex buffer skips InitMeshBuffers", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpMock.EXPECT().VertexBuffer().Return(new(wgpu.Buffer)).Once()
		mdl.EXPECT().MeshProvider().Return(bgpMock).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("InitMeshBuffers error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = vs
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpMock.EXPECT().VertexBuffer().Return(nil).Once()
		mdl.EXPECT().MeshProvider().Return(bgpMock).Once()
		mdl.EXPECT().VertexData().Return(nil).Once()
		mdl.EXPECT().IndexData().Return(nil).Once()
		mdl.EXPECT().IndexCount().Return(0).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		suite.rendererMock.EXPECT().InitMeshBuffers(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("mesh err")).Once()
		suite.Panics(func() { suite.scene.createAnimator(mdl, cs, vs, fs) })
	})

	suite.Run("AnnotationArgAnimationData sets compute group", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		g := 3
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &g,
			Args:  []shader.AnnotationArg{"", "", shader.AnnotationArgAnimationData},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(3).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("AnnotationArgSkeletalAnimationData with array< prefix sets compute group", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		g := 2
		decl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &g,
			Args:  []shader.AnnotationArg{"", "", shader.AnnotationArg("array<" + string(shader.AnnotationArgSkeletalAnimationData) + ">")},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("compute group decl with nil Group is skipped", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b := 1
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   nil,
			Binding: &b,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgAnimationData},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("compute group decl non-BindingGroup type is skipped", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		g := 5
		b := 1
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Group:   &g,
			Binding: &b,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("output group BindingGroup InstanceData array< prefix sets group and binding", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		g := 2
		b := 1
		vsDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &g,
			Binding: &b,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArg("array<" + string(shader.AnnotationArgInstanceData) + ">")},
		}
		vs.EXPECT().Declarations().Return([]shader.Annotation{vsDecl}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("output group BindingGroup InstanceData with nil binding skips binding update", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		g := 2
		vsDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &g,
			Binding: nil,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgInstanceData},
		}
		vs.EXPECT().Declarations().Return([]shader.Annotation{vsDecl}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("output group set from Provider Animator", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		g := 3
		vsDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &g,
			Args:  []shader.AnnotationArg{shader.AnnotationArgAnimator},
		}
		vs.EXPECT().Declarations().Return([]shader.Annotation{vsDecl}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(3).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("output decl with nil Group is skipped", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vsDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: nil,
			Args:  []shader.AnnotationArg{shader.AnnotationArgAnimator},
		}
		vs.EXPECT().Declarations().Return([]shader.Annotation{vsDecl}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("output group Provider with non-Animator arg does not set outputGroup", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		g := 4
		vsDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &g,
			Args:  []shader.AnnotationArg{shader.AnnotationArgMaterial},
		}
		vs.EXPECT().Declarations().Return([]shader.Annotation{vsDecl}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("matching output descriptor entry overrides perInstanceOutputSize", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		g := 1
		zero := 0
		vsDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &g,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgInstanceData},
		}
		vs.EXPECT().Declarations().Return([]shader.Annotation{vsDecl}).Once()
		outputDesc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{MinBindingSize: 128}},
			},
		}
		vs.EXPECT().BindGroupLayoutDescriptor(1).Return(outputDesc).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("output descriptor entry zero MinBindingSize does not override perInstanceOutputSize", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		g := 1
		zero := 0
		vsDecl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &g,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgInstanceData},
		}
		vs.EXPECT().Declarations().Return([]shader.Annotation{vsDecl}).Once()
		outputDesc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{MinBindingSize: 0}},
			},
		}
		vs.EXPECT().BindGroupLayoutDescriptor(1).Return(outputDesc).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("typed compute entries IndirectArgs BoneInfo ModelData AnimGlobals GlobalData default storage", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b0, b1, b2, b3, b4, b5 := 0, 1, 2, 3, 4, 5
		g := 0
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &b0, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgIndirectArgs}},
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &b1, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgBoneInfo}},
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &b2, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgModelData}},
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &b3, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgAnimationGlobals}},
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &b4, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgGlobalData}},
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &b5, Args: []shader.AnnotationArg{"", "", "some_storage_type"}},
		}
		cs.EXPECT().Declarations().Return(decls).Maybe()
		computeDesc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{MinBindingSize: 16}},
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{MinBindingSize: 80}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{MinBindingSize: 64}},
				{Binding: 3, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 32}},
				{Binding: 4, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 32}},
				{Binding: 5, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 32}},
			},
		}
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(computeDesc).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("typed BoneInfo ModelData with zero MinBindingSize no size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b1, b2 := 1, 2
		g := 0
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &b1, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgBoneInfo}},
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &b2, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgModelData}},
		}
		cs.EXPECT().Declarations().Return(decls).Maybe()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{MinBindingSize: 0}},
				{Binding: 2, Buffer: wgpu.BufferBindingLayout{MinBindingSize: 0}},
			},
		}
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(desc).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("default ReadOnlyStorage binding with MinBindingSize adds size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b := 5
		g := 0
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &b, Args: []shader.AnnotationArg{"", "", "some_ro_type"}},
		}
		cs.EXPECT().Declarations().Return(decls).Maybe()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 5, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeReadOnlyStorage, MinBindingSize: 16}},
			},
		}
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(desc).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("default non-storage binding with MinBindingSize no size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b := 7
		g := 0
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &b, Args: []shader.AnnotationArg{"", "", "some_uniform_type"}},
		}
		cs.EXPECT().Declarations().Return(decls).Maybe()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 7, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeUniform, MinBindingSize: 32}},
			},
		}
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(desc).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("compute binding types loop skips Provider decls", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b := 3
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Binding: &b,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("compute binding types loop skips BindingGroup decls with nil Binding", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		g := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &g,
			Binding: nil,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgAnimationData},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("raw output binding sets computeOutputBinding and size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b := 6
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Binding: &b,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 6, Buffer: wgpu.BufferBindingLayout{MinBindingSize: 64}},
			},
		}
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(desc).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("raw packed binding adds size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b := 8
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Binding: &b,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorPacked},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 8, Buffer: wgpu.BufferBindingLayout{MinBindingSize: 4}},
			},
		}
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(desc).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("raw scratch binding adds size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b := 9
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Binding: &b,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorScratch},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		desc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 9, Buffer: wgpu.BufferBindingLayout{MinBindingSize: 64}},
			},
		}
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(desc).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("raw provider loop skips non-Provider decls", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b := 2
		g := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &g,
			Binding: &b,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgAnimationData},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("raw provider loop skips Provider decls with nil Binding", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Binding: nil,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("output size override for matching outputInstanceBinding storage entry", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		outputDesc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 0, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
			},
		}
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(outputDesc).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("InitBindGroup compute error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("compute bg err")).Once()
		suite.Panics(func() { suite.scene.createAnimator(mdl, cs, vs, fs) })
	})

	suite.Run("InitBindGroup output error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("output bg err")).Once()
		suite.Panics(func() { suite.scene.createAnimator(mdl, cs, vs, fs) })
	})

	suite.Run("RegisterPipelines compute error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("cp err")).Once()
		suite.Panics(func() { suite.scene.createAnimator(mdl, cs, vs, fs) })
	})

	suite.Run("RegisterPipelines render error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("rp err")).Once()
		suite.Panics(func() { suite.scene.createAnimator(mdl, cs, vs, fs) })
	})

	suite.Run("material with non-nil BGP is skipped", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		matMock := material_mocks.NewMockMaterial(suite.T())
		existingBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock.EXPECT().BindGroupProvider().Return(existingBGP).Once()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("material with empty pipeline key uses fragmentShader for initMaterialGPU", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		matMock.EXPECT().PipelineKey().Return("").Once()
		fs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("material pipeline key nil at Pipeline registers new pipeline second nil uses fragmentShader", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		matMock.EXPECT().PipelineKey().Return("mat-key").Once()
		matMock.EXPECT().PipelineOptions().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-key").Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-key").Return(nil).Once()
		fs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(3)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("material PipelineOptions non-empty non-PipelineBuilderOption items filtered", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		matMock.EXPECT().PipelineKey().Return("mat-key-opts").Once()
		matMock.EXPECT().PipelineOptions().Return([]any{"not-a-pipeline-option"}).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-key-opts").Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-key-opts").Return(nil).Once()
		fs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(3)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("material PipelineOptions with valid PipelineBuilderOption is included", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		matMock.EXPECT().PipelineKey().Return("mat-key-valid").Once()
		realOpt := pipeline.WithVertexShader(vs)
		matMock.EXPECT().PipelineOptions().Return([]any{realOpt}).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-key-valid").Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-key-valid").Return(nil).Once()
		fs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(3)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("material already registered Pipeline skips registration and uses Shader for frag", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		matMock.EXPECT().PipelineKey().Return("mat-existing").Once()
		existingPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		existingPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(fs).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-existing").Return(existingPipeline).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-existing").Return(existingPipeline).Once()
		fs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("material pipeline Shader returns nil falls back to fragmentShader", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		matMock.EXPECT().PipelineKey().Return("mat-nil-shader").Once()
		existingPipeline := pipeline_mocks.NewMockPipeline(suite.T())
		existingPipeline.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-nil-shader").Return(existingPipeline).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-nil-shader").Return(existingPipeline).Once()
		fs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("material RegisterPipelines error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		matMock.EXPECT().PipelineKey().Return("mat-err").Once()
		matMock.EXPECT().PipelineOptions().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("mat-err").Return(nil).Once()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("mat pipe err")).Once()
		suite.Panics(func() { suite.scene.createAnimator(mdl, cs, vs, fs) })
	})

	suite.Run("initMaterialGPU error panics", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		matMock.EXPECT().PipelineKey().Return("").Once()
		g := 2
		fragDecl := shader.Annotation{
			Type:  shader.AnnotationTypeProvider,
			Group: &g,
			Args:  []shader.AnnotationArg{shader.AnnotationArgMaterial},
		}
		fs.EXPECT().Declarations().Return([]shader.Annotation{fragDecl}).Once()
		fs.EXPECT().BindGroupLayoutDescriptor(2).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		matMock.EXPECT().DiffuseTexture().Return(nil).Once()
		matMock.EXPECT().NormalTexture().Return(nil).Once()
		matMock.EXPECT().MetallicRoughnessTexture().Return(nil).Once()
		mdl.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("mat bg err")).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.Panics(func() { suite.scene.createAnimator(mdl, cs, vs, fs) })
	})

	suite.Run("skeletal backend nil skeleton selects skeletal type", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		anim := suite.scene.createAnimator(mdl, cs, vs, fs)
		suite.NotNil(anim)
	})

	suite.Run("skeletal binding discovery nil Binding skips decl", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		g := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &g,
			Binding: nil,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgBoneInfo},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("skeletal binding discovery BoneInfo and AnimatorPacked both found", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		g := 0
		bBone, bPacked := 3, 4
		decls := []shader.Annotation{
			{Type: shader.AnnotationTypeBindingGroup, Group: &g, Binding: &bBone, Args: []shader.AnnotationArg{"", "", shader.AnnotationArgBoneInfo}},
			{Type: shader.AnnotationTypeProvider, Binding: &bPacked, Args: []shader.AnnotationArg{shader.AnnotationArgAnimatorPacked}},
		}
		cs.EXPECT().Declarations().Return(decls).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("skeletal binding discovery array< prefix stripped for BoneInfo", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		g := 0
		bBone := 5
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &g,
			Binding: &bBone,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArg("array<" + string(shader.AnnotationArgBoneInfo) + ">")},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("skeletal binding discovery Provider non-AnimatorPacked arg does not set packedBinding", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		b := 7
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Binding: &b,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("skeletal with skeleton and animations computes packed buffer size", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		skel := &model.Skeleton{
			Bones: []model.Bone{{}, {}, {}},
		}
		mdl.EXPECT().Skeleton().Return(skel).Maybe()
		clips := []*model.AnimationClip{
			{
				Name: "walk",
				Channels: []model.AnimationChannel{
					{
						BoneIndex:    0,
						PositionKeys: []model.VectorKeyframe{{}, {}, {}},
						RotationKeys: []model.QuaternionKeyframe{{}, {}},
						ScaleKeys:    []model.VectorKeyframe{{}},
					},
				},
			},
		}
		mdl.EXPECT().Animations().Return(clips).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		anim := suite.scene.createAnimator(mdl, cs, vs, fs)
		suite.NotNil(anim)
	})

	suite.Run("packed buffer size clamped to 4 when zero clips", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		skel := &model.Skeleton{
			Bones: []model.Bone{{}},
		}
		mdl.EXPECT().Skeleton().Return(skel).Maybe()
		mdl.EXPECT().Animations().Return([]*model.AnimationClip{}).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("compute output buffer shared to output BGP when annotation present", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()

		outBinding := 0
		outBindingPtr := &outBinding
		outDecl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Binding: outBindingPtr,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{outDecl}).Maybe()
		computeLayoutDesc := wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{
					Binding:    uint32(outBinding),
					Visibility: wgpu.ShaderStageCompute,
					Buffer: wgpu.BufferBindingLayout{
						Type:           wgpu.BufferBindingTypeStorage,
						MinBindingSize: 16,
					},
				},
			},
		}
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(computeLayoutDesc).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()

		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Run(func(provider bind_group_provider.BindGroupProvider, descriptor wgpu.BindGroupLayoutDescriptor, bufferUsageOverrides map[int]wgpu.BufferUsage, bufferSizeOverrides map[int]uint64) {
				provider.SetBuffer(outBinding, new(wgpu.Buffer))
			}).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)

		anim := suite.scene.createAnimator(mdl, cs, vs, fs)
		suite.NotNil(anim)
	})
}
