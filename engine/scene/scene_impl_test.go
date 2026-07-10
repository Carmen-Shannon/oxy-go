package scene

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"

	"github.com/Carmen-Shannon/automation/tools/worker"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/common"
	camera_mocks "github.com/Carmen-Shannon/oxy-go/engine/camera/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	game_object_mocks "github.com/Carmen-Shannon/oxy-go/engine/game_object/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
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
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/gbuffer"
	gbuffer_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/gbuffer/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	material_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/material/mocks"
	renderer_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	pipeline_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/composition"
	composition_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/composition/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/ssao"
	ssao_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/ssao/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/ssr"
	ssr_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/ssr/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/taa"
	taa_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/taa/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	shader_mocks "github.com/Carmen-Shannon/oxy-go/engine/renderer/shader/mocks"
)

func TestRunSceneImplTests(t *testing.T) {
	suite.Run(t, new(sceneImplTest))
}

type sceneImplTest struct {
	suite.Suite
	rendererMock *renderer_mocks.MockRenderer
	scene        *scene
}

func startAnimator(anim animator.Animator) animator.Animator {
	if anim == nil {
		return nil
	}
	if anim.Lifecycle().State() == lifecycle.LifecycleStateRegistered {
		anim.Lifecycle().SetState(lifecycle.LifecycleStateStarting)
	}
	return anim
}

func newSceneLifecycleHelper(r renderer.Renderer) *scene {
	s := &scene{
		mu:                     &sync.RWMutex{},
		DelegateImpl:           common.DelegateImpl[Scene]{},
		lc:                     lifecycle.NewLifecycle(),
		r:                      r,
		animatorPool:           make(map[model.Model][]animator.Animator),
		registry:               make(map[uint64]game_object.GameObject),
		physicsSyncGroup:       make(map[int]bind_group_provider.BindGroupProvider),
		physicsSyncAnimMap:     make(map[animator.Animator]int),
		lightHandler:           light.NewLightingHandler(),
		gBufferHandler:         gbuffer.NewGBufferHandler(),
		ssaoHandler:            ssao.NewHandler(),
		compositionHandler:     composition.NewHandler(),
		ssrHandler:             ssr.NewHandler(),
		taaHandler:             taa.NewHandler(),
		drawGroupProvidersPool: make(map[int]bind_group_provider.BindGroupProvider),
		shadowIndirectBuffers:  make(map[animator.Animator]*wgpu.Buffer),
		animIndirectBinding:    make(map[animator.Animator]int),
	}
	s.SetDelegate(s)
	return s
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
		DelegateImpl:           common.DelegateImpl[Scene]{},
		lc:                     lifecycle.NewLifecycle(),
		r:                      suite.rendererMock,
		lightHandler:           light.NewLightingHandler(),
		gBufferHandler:         gbuffer.NewGBufferHandler(),
		ssaoHandler:            ssao.NewHandler(),
		compositionHandler:     composition.NewHandler(),
		ssrHandler:             ssr.NewHandler(),
		taaHandler:             taa.NewHandler(),
		maxBonesGPU:            64,
		drawGroupProvidersPool: make(map[int]bind_group_provider.BindGroupProvider),
		shadowIndirectBuffers:  make(map[animator.Animator]*wgpu.Buffer),
		animIndirectBinding:    make(map[animator.Animator]int),
	}
	suite.scene.SetDelegate(suite.scene)
}

func (suite *sceneImplTest) TestGenerateSSAOKernel() {
	suite.Run("clamps sampleCount below 1 to 1", func() {
		buf := suite.scene.generateSSAOKernel(0)
		suite.NotNil(buf)
		maxSamples := suite.scene.ssaoHandler.MaxSamples()
		suite.Len(buf, maxSamples*16)
	})

	suite.Run("clamps sampleCount above MaxSamples to MaxSamples", func() {
		maxSamples := suite.scene.ssaoHandler.MaxSamples()
		buf := suite.scene.generateSSAOKernel(9999)
		suite.Len(buf, maxSamples*16)
	})

	suite.Run("returns correct buffer size for valid sampleCount", func() {
		maxSamples := suite.scene.ssaoHandler.MaxSamples()
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
		suite.Contains(suite.scene.injections, "pcf_samples_spot")
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
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(nil, nil).Maybe()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterShadowDepthPipeline(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateHiZTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, 0).Maybe()

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
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.ssaoHandler.SetEnabled(true)
		suite.scene.compositionHandler.SetEnabled(true)
		suite.scene.ssrHandler.SetEnabled(true)

		suite.rendererMock.EXPECT().Resize(800, 600).Return().Once()

		suite.scene.tileBufferCapacity = math.MaxInt32
		suite.scene.Resize(800, 600)

		suite.Equal(800, suite.scene.screenWidth)
		suite.Equal(600, suite.scene.screenHeight)
	})

	suite.Run("resizes TAA handler when enabled", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().SetAspect(float32(1024) / float32(768)).Return().Once()
		suite.scene.cam = camMock

		suite.scene.taaHandler.SetEnabled(true)

		suite.rendererMock.EXPECT().Resize(1024, 768).Return().Once()

		suite.scene.tileBufferCapacity = math.MaxInt32
		suite.scene.Resize(1024, 768)

		suite.Equal(1024, suite.scene.taaHandler.ScreenWidth())
		suite.Equal(768, suite.scene.taaHandler.ScreenHeight())
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
	suite.Run("completes begin and end with empty animator pool", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator with zero instance count", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(0)).Once()
		mockMdl := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mockMdl: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator with nil model", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(nil).Once()
		mockMdl := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mockMdl: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator with nil mesh provider", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(nil).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator when pipeline key is empty", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		mockBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(mockBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator when camera BGP is nil", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock

		mockBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(mockBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator when model has no materials", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()

		camMock := camera_mocks.NewMockCamera(suite.T())
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock

		mockMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(mockMeshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return(nil).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skips animator when material BGP is nil", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(mockMeshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("calls GBufferDrawCall when culling is disabled", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(false).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("culling enabled with empty compute key falls back to GBufferDrawCall", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().IndirectBuffer(0).Return(nil).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("culling enabled with nil Pipeline falls back to GBufferDrawCall", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().IndirectBuffer(0).Return(nil).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("culling enabled with nil Shader falls back to GBufferDrawCall", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().IndirectBuffer(0).Return(nil).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("culling: indirect buffer nil falls back to GBufferDrawCall", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		mockAnim.EXPECT().IndirectBuffer(3).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}
		suite.scene.animIndirectBinding = map[animator.Animator]int{mockAnim: 3}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("calls GBufferDrawCallIndirect when indirect buffer available", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().GBufferDrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		indBuf := &wgpu.Buffer{}
		mockAnim.EXPECT().IndirectBuffer(3).Return(indBuf).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}
		suite.scene.animIndirectBinding = map[animator.Animator]int{mockAnim: 3}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("culling: array<indirect_args> annotation triggers CutPrefix branch", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("static", "gbuffer_static")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginGBufferPass(mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndGBufferPass().Return().Once()
		suite.rendererMock.EXPECT().EndGBufferFrame().Return().Once()
		suite.rendererMock.EXPECT().GBufferDrawCallIndirect(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()

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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Once()
		mockAnim.EXPECT().Model().Return(mockModel).Once()
		mockAnim.EXPECT().CullingEnabled().Return(true).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		indBuf := &wgpu.Buffer{}
		mockAnim.EXPECT().IndirectBuffer(3).Return(indBuf).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}
		suite.scene.animIndirectBinding = map[animator.Animator]int{mockAnim: 3}

		suite.NotPanics(func() { suite.scene.PrepareGBuffer() })
	})

	suite.Run("skinned model uses skinned pipeline key", func() {
		suite.scene.gBufferHandler.SetEnabled(true)
		suite.scene.gBufferHandler.SetPipelineKey("skinned", "gbuffer_skinned")
		suite.rendererMock.EXPECT().BeginGBufferFrame().Return().Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().Skinned().Return(true).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{mockMat}).Once()
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		suite.scene.ssaoHandler.SetEnabled(true)
		suite.scene.r = nil
		suite.NotPanics(func() { suite.scene.PrepareSSAO() })
	})

	suite.Run("early return when camera is nil", func() {
		suite.scene.ssaoHandler.SetEnabled(true)
		suite.NotPanics(func() { suite.scene.PrepareSSAO() })
	})
	suite.Run("full resolution nil controller both lookups dispatches succeed", func() {
		suite.scene.ssaoHandler.SetEnabled(true)

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
		suite.scene.ssaoHandler.SetEnabled(true)

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
		suite.scene.ssaoHandler.SetEnabled(true)
		suite.scene.ssaoHandler.SetHalfResolution(true)

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

func (suite *sceneImplTest) TestPrepareTAA() {
	suite.Run("returns early when the taa handler is disabled", func() {
		suite.NotPanics(func() { suite.scene.PrepareTAA() })
	})

	suite.Run("writes taa params and dispatches resolve and sharpen passes", func() {
		taaHandler := taa.NewHandler(
			taa.WithTAAScreenSize(1280, 720),
			taa.WithTAABlendFactor(0.2),
		)
		taaHandler.SetEnabled(true)
		taaHandler.SetPipelineKey("taa_resolve", "taa_resolve_pipeline")
		taaHandler.SetPipelineKey("taa_sharpen", "taa_sharpen_pipeline")
		suite.scene.taaHandler = taaHandler

		camMock := camera_mocks.NewMockCamera(suite.T())
		currViewProjection := [16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		prevViewProjection := [16]float32{
			7, 0, 0, 0,
			0, 7, 0, 0,
			0, 0, 7, 0,
			0, 0, 0, 7,
		}
		expectedJitterY := float32(-1.0 / 2160.0)

		camMock.EXPECT().SetJitter(mock.Anything, mock.Anything).Run(func(x float32, y float32) {
			suite.InDelta(0.0, float64(x), 1e-6)
			suite.InDelta(float64(expectedJitterY), float64(y), 1e-6)
		}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return(currViewProjection).Once()
		camMock.EXPECT().PrevViewProjectionMatrix().Return(prevViewProjection).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(0).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Run(func(writes []bind_group_provider.BufferWrite) {
			suite.Len(writes, 1)
			suite.Equal(taaHandler.Bgp("taa_resolve_0"), writes[0].Provider)
			suite.Equal(0, writes[0].Binding)
			suite.Zero(writes[0].Offset)
			suite.Len(writes[0].Data, (&taa.GPUTAAParams{}).Size())
			suite.Equal(math.Float32bits(1.0), binary.LittleEndian.Uint32(writes[0].Data[0:4]))
			suite.Equal(math.Float32bits(prevViewProjection[0]), binary.LittleEndian.Uint32(writes[0].Data[64:68]))
			suite.Equal(math.Float32bits(0.0), binary.LittleEndian.Uint32(writes[0].Data[128:132]))
			suite.Equal(math.Float32bits(float32(-1.0/6.0)), binary.LittleEndian.Uint32(writes[0].Data[132:136]))
			suite.Equal(uint32(0), binary.LittleEndian.Uint32(writes[0].Data[136:140]))
			suite.Equal(uint32(0), binary.LittleEndian.Uint32(writes[0].Data[140:144]))
			suite.Equal(math.Float32bits(1280.0), binary.LittleEndian.Uint32(writes[0].Data[144:148]))
			suite.Equal(math.Float32bits(720.0), binary.LittleEndian.Uint32(writes[0].Data[148:152]))
			suite.Equal(math.Float32bits(0.2), binary.LittleEndian.Uint32(writes[0].Data[152:156]))
			suite.Equal(math.Float32bits(taaHandler.HistoryRectificationScale()), binary.LittleEndian.Uint32(writes[0].Data[156:160]))
		}).Once()

		dispatchCount := 0
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Run(func(dispatches []renderer.ComputeDispatch) {
			dispatchCount++
			suite.Len(dispatches, 1)
			suite.Equal([3]uint32{160, 90, 1}, dispatches[0].WorkGroupCount)

			switch dispatchCount {
			case 1:
				suite.Equal("taa_resolve_pipeline", dispatches[0].PipelineKey)
				suite.Len(dispatches[0].Providers, 1)
				suite.Equal(uint32(0), dispatches[0].Providers[0].Group)
				suite.Equal(taaHandler.Bgp("taa_resolve_0"), dispatches[0].Providers[0].Provider)
			case 2:
				suite.Equal("taa_sharpen_pipeline", dispatches[0].PipelineKey)
				suite.Len(dispatches[0].Providers, 1)
				suite.Equal(uint32(0), dispatches[0].Providers[0].Group)
				suite.Equal(taaHandler.Bgp("taa_sharpen_0"), dispatches[0].Providers[0].Provider)
			default:
				suite.Fail("unexpected dispatch count")
			}
		}).Twice()

		suite.scene.PrepareTAA()

		suite.Equal(uint64(1), taaHandler.FrameIndex())
		suite.InDelta(0.0, float64(taaHandler.JitterX()), 1e-6)
		suite.InDelta(float64(-1.0/6.0), float64(taaHandler.JitterY()), 1e-6)
		suite.InDelta(0.0, float64(taaHandler.PrevJitterX()), 1e-6)
		suite.InDelta(0.0, float64(taaHandler.PrevJitterY()), 1e-6)
	})

	suite.Run("returns early when lightHandler is nil", func() {
		suite.scene.lightHandler = nil
		suite.scene.taaHandler.SetEnabled(true)
		suite.NotPanics(func() { suite.scene.prepareTAA() })
	})

	suite.Run("returns early when camera is nil", func() {
		suite.scene.cam = nil
		suite.scene.taaHandler.SetEnabled(true)
		suite.NotPanics(func() { suite.scene.prepareTAA() })
	})

	suite.Run("skips NDC conversion when screen dimensions are zero", func() {
		taaHandler := taa.NewHandler(
			taa.WithTAAScreenSize(0, 0),
			taa.WithTAABlendFactor(0.1),
		)
		taaHandler.SetEnabled(true)
		taaHandler.SetPipelineKey("taa_resolve", "taa_resolve_pipeline")
		taaHandler.SetPipelineKey("taa_sharpen", "taa_sharpen_pipeline")
		suite.scene.taaHandler = taaHandler

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().SetJitter(float32(0), float32(0)).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
		}).Once()
		camMock.EXPECT().PrevViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
		}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(0).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Twice()

		suite.scene.prepareTAA()
	})

	suite.Run("uses slot 1 BGP keys when renderer returns frame slot 1", func() {
		taaHandler := taa.NewHandler(
			taa.WithTAAScreenSize(1280, 720),
			taa.WithTAABlendFactor(0.2),
		)
		taaHandler.SetEnabled(true)
		taaHandler.SetPipelineKey("taa_resolve", "taa_resolve_pipeline")
		taaHandler.SetPipelineKey("taa_sharpen", "taa_sharpen_pipeline")
		suite.scene.taaHandler = taaHandler

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().SetJitter(mock.Anything, mock.Anything).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
		}).Once()
		camMock.EXPECT().PrevViewProjectionMatrix().Return([16]float32{
			1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
		}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(1).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Run(func(writes []bind_group_provider.BufferWrite) {
			suite.Len(writes, 1)
			suite.Equal(taaHandler.Bgp("taa_resolve_1"), writes[0].Provider)
		}).Once()

		dispatchCount := 0
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Run(func(dispatches []renderer.ComputeDispatch) {
			dispatchCount++
			suite.Len(dispatches, 1)
			switch dispatchCount {
			case 1:
				suite.Equal("taa_resolve_pipeline", dispatches[0].PipelineKey)
				suite.Equal(taaHandler.Bgp("taa_resolve_1"), dispatches[0].Providers[0].Provider)
			case 2:
				suite.Equal("taa_sharpen_pipeline", dispatches[0].PipelineKey)
				suite.Equal(taaHandler.Bgp("taa_sharpen_1"), dispatches[0].Providers[0].Provider)
			}
		}).Twice()

		suite.scene.prepareTAA()
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
		suite.rendererMock.EXPECT().BeginComputeFrame().Return().Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndComputeFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareContactShadows() })
	})
}

func (suite *sceneImplTest) TestPrepareSSR() {
	suite.Run("disabled returns early", func() {
		suite.NotPanics(func() { suite.scene.PrepareSSR() })
	})

	suite.Run("default dimensions clamped to 1 no loop", func() {
		suite.scene.ssrHandler.SetEnabled(true)
		suite.scene.compositionHandler.SetEnabled(true)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(int(0)).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return().Once()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(nil).Times(4)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(3)
		suite.rendererMock.EXPECT().EndComputeFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareSSR() })
	})

	suite.Run("mipCount 2 executes one loop iteration", func() {
		suite.scene.ssrHandler.SetEnabled(true)
		suite.scene.compositionHandler.SetEnabled(true)
		suite.scene.ssrHandler.SetHiZMipCount(2)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(int(0)).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return().Once()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(nil).Times(4)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(5)
		suite.rendererMock.EXPECT().EndComputeFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareSSR() })
	})
}

func (suite *sceneImplTest) TestPrepareComposition() {
	suite.Run("disabled returns early", func() {
		suite.NotPanics(func() { suite.scene.PrepareComposition() })
	})

	suite.Run("nil renderer returns early", func() {
		suite.scene.r = nil
		suite.NotPanics(func() { suite.scene.PrepareComposition() })
	})

	suite.Run("full composition dispatch", func() {
		suite.scene.compositionHandler.SetEnabled(true)

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginCompositionPass().Return().Once()
		suite.rendererMock.EXPECT().CompositionDrawCall(mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().EndCompositionPass().Return().Once()
		suite.rendererMock.EXPECT().EndCompositionFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareComposition() })
	})
}

func (suite *sceneImplTest) TestBeginHDRFrame() {
	suite.Run("no-MSAA path", func() {
		suite.scene.compositionHandler.SetEnabled(true)

		suite.rendererMock.EXPECT().SampleCount().Return(uint32(1)).Once()
		suite.rendererMock.EXPECT().BeginHDRFrame(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.BeginHDRFrame()
	})

	suite.Run("MSAA path", func() {
		suite.scene.compositionHandler.SetEnabled(true)
		suite.scene.compositionHandler.SetMSAATextureView(&wgpu.TextureView{})

		suite.rendererMock.EXPECT().SampleCount().Return(uint32(4)).Once()
		suite.rendererMock.EXPECT().BeginHDRFrame(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.BeginHDRFrame()
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

	suite.Run("directional light non-nil controller buffer2 non-nil cascades no animators", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Maybe()
		dl.EXPECT().CastsShadows().Return(true).Maybe()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Maybe()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		suite.scene.lightHandler.AddLight(dl)

		ctrl := camera_mocks.NewMockCameraController(suite.T())
		ctrl.EXPECT().Position().Return(float32(1), float32(2), float32(3)).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(ctrl).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
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
		dl.EXPECT().Enabled().Return(true).Maybe()
		dl.EXPECT().CastsShadows().Return(true).Maybe()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Maybe()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(0)).Times(4)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
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
		dl.EXPECT().Enabled().Return(true).Maybe()
		dl.EXPECT().CastsShadows().Return(true).Maybe()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Maybe()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(nil).Times(4)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
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
		dl.EXPECT().Enabled().Return(true).Maybe()
		dl.EXPECT().CastsShadows().Return(true).Maybe()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Maybe()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(false).Times(4)
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Times(4)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
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
		dl.EXPECT().Enabled().Return(true).Maybe()
		dl.EXPECT().CastsShadows().Return(true).Maybe()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Maybe()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(4)
		mockModel.EXPECT().LODMeshProvider(0).Return(nil).Twice()
		mockModel.EXPECT().LODCount().Return(1).Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Times(6)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
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
		dl.EXPECT().Enabled().Return(true).Maybe()
		dl.EXPECT().CastsShadows().Return(true).Maybe()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Maybe()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(4)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Twice()
		mockModel.EXPECT().LODCount().Return(1).Twice()
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Times(6)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
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
		dl.EXPECT().Enabled().Return(true).Maybe()
		dl.EXPECT().CastsShadows().Return(true).Maybe()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Maybe()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mockBuf := &wgpu.Buffer{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(4)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Twice()
		mockModel.EXPECT().LODCount().Return(1).Times(3)
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mockModel.EXPECT().LODIndexCount(0).Return(6).Once()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Twice()
		mockModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(5)
		mockAnim.EXPECT().Model().Return(mockModel).Times(7)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Maybe()
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{mockAnim: mockBuf}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}
		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{mockAnim: outputBGP}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mockBuf, uint64(0), mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mockBuf, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: skips zero instance count animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(0)).Times(3)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: skips nil model animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		mockAnim.EXPECT().Model().Return(nil).Times(3)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: skips animator with no cast shadows", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(false).Times(3)
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		mockAnim.EXPECT().Model().Return(mockModel).Times(3)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: skips animator with nil mesh provider", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(3)
		mockModel.EXPECT().LODMeshProvider(0).Return(nil).Once()
		mockModel.EXPECT().LODCount().Return(1).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		mockAnim.EXPECT().Model().Return(mockModel).Times(4)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: skips animator with empty pipeline key", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(3)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().LODCount().Return(1).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		mockAnim.EXPECT().Model().Return(mockModel).Times(4)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
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
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		mockBuf := &wgpu.Buffer{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(3)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().LODCount().Return(1).Twice()
		mockModel.EXPECT().LODIndexCount(0).Return(6).Once()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Times(5)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Once()
		mockModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Once()
		mockModel.EXPECT().BoundingRadius().Return(float32(1000)).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Maybe()
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{mockAnim: mockBuf}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}
		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{mockAnim: outputBGP}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mockBuf, uint64(0), mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mockBuf, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face: skips zero instance count animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(0)).Times(8)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
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
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		mockBuf := &wgpu.Buffer{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(14)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Times(6)
		mockModel.EXPECT().LODCount().Return(1).Times(7)
		mockModel.EXPECT().Skinned().Return(false).Times(6)
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		mockModel.EXPECT().LODIndexCount(0).Return(6).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(15)
		mockAnim.EXPECT().Model().Return(mockModel).Times(21)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Times(12)
		mockModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Times(12)
		mockModel.EXPECT().BoundingRadius().Return(float32(1000)).Times(12)
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Maybe()
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{mockAnim: mockBuf}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}
		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{mockAnim: outputBGP}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mockBuf, uint64(0), mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mockBuf, mock.Anything).Return(nil).Times(6)
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face: break when slots exhausted", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl1 := light_mocks.NewMockLight(suite.T())
		pl1.EXPECT().Enabled().Return(true).Maybe()
		pl1.EXPECT().CastsShadows().Return(true).Maybe()
		pl1.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl1.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl1.EXPECT().Range().Return(float32(20)).Maybe()
		pl1.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl1.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl1.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl1.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl1)

		pl2 := light_mocks.NewMockLight(suite.T())
		pl2.EXPECT().Enabled().Return(true).Maybe()
		pl2.EXPECT().CastsShadows().Return(true).Maybe()
		pl2.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl2.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl2.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl2.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl2)

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
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
		dl.EXPECT().Enabled().Return(true).Maybe()
		dl.EXPECT().CastsShadows().Return(true).Maybe()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Maybe()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		suite.scene.lightHandler.AddLight(dl)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(3)
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Twice()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point loop: non-point light continue before point light processes", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(false).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{}).Maybe()
		sl.EXPECT().Range().Return(float32(0)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face depth pass: skips nil model animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(8)
		mockAnim.EXPECT().Model().Return(nil).Times(8)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face depth pass: skips animator with nil mesh provider", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(14)
		mockModel.EXPECT().LODMeshProvider(0).Return(nil).Times(6)
		mockModel.EXPECT().LODCount().Return(1).Times(6)
		mockModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Times(6)
		mockModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Times(6)
		mockModel.EXPECT().BoundingRadius().Return(float32(1000)).Times(6)
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(14)
		mockAnim.EXPECT().Model().Return(mockModel).Times(20)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face depth pass: skips animator with empty pipeline key", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(14)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Times(6)
		mockModel.EXPECT().LODCount().Return(1).Times(6)
		mockModel.EXPECT().Skinned().Return(false).Times(6)
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		mockModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Times(6)
		mockModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Times(6)
		mockModel.EXPECT().BoundingRadius().Return(float32(1000)).Times(6)
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(14)
		mockAnim.EXPECT().Model().Return(mockModel).Times(20)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("indirect pre-pass writes args and issues ShadowDrawCallIndirect", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Maybe()
		dl.EXPECT().CastsShadows().Return(true).Maybe()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Maybe()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(4)
		mockModel.EXPECT().LODIndexCount(0).Return(6).Once()
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Twice()
		mockModel.EXPECT().LODCount().Return(1).Times(3)
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Twice()
		mockModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Twice()

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(2)).Times(5)
		mockAnim.EXPECT().Model().Return(mockModel).Times(7)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		mockAnim.EXPECT().InstanceTransform(uint32(1)).Return([3]float32{}, [3]float32{1, 1, 1}).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Maybe()

		mockBuf := &wgpu.Buffer{}
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{mockAnim: mockBuf}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}
		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{mockAnim: outputBGP}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mockBuf, uint64(0), mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mockBuf, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("csm cascade: instance outside cascade frustum skips draw call", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")
		csmBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sh.SetBgp("csm_shadow_lit", csmBGPMock)
		csmBGPMock.EXPECT().Buffer(2).Return(&wgpu.Buffer{}).Once()
		csmBGPMock.EXPECT().Buffer(4).Return(nil).Once()

		dl := light_mocks.NewMockLight(suite.T())
		dl.EXPECT().Enabled().Return(true).Maybe()
		dl.EXPECT().CastsShadows().Return(true).Maybe()
		dl.EXPECT().Type().Return(light.LightTypeDirectional).Maybe()
		dl.EXPECT().Direction().Return([3]float32{0, 0, -1}).Maybe()
		dl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		suite.scene.lightHandler.AddLight(dl)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Controller().Return(nil).Once()
		camMock.EXPECT().Fov().Return(float32(1.0)).Once()
		camMock.EXPECT().Aspect().Return(float32(1.0)).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		suite.scene.cam = camMock

		mockBuf := &wgpu.Buffer{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(4)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Twice()
		mockModel.EXPECT().LODCount().Return(1).Times(3)
		mockModel.EXPECT().Skinned().Return(false).Twice()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Twice()
		mockModel.EXPECT().LODIndexCount(0).Return(6).Once()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-0.5, -0.5, -0.5}).Twice()
		mockModel.EXPECT().BoundingMax().Return([3]float32{0.5, 0.5, 0.5}).Twice()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(5)
		mockAnim.EXPECT().Model().Return(mockModel).Times(7)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{10000, 10000, 10000}, [3]float32{1, 1, 1}).Once()
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{mockAnim: mockBuf}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mockBuf, uint64(0), mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: instance outside spot frustum skips draw call", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		mockBuf := &wgpu.Buffer{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(3)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().LODCount().Return(1).Twice()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Once()
		mockModel.EXPECT().LODIndexCount(0).Return(6).Once()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-0.5, -0.5, -0.5}).Once()
		mockModel.EXPECT().BoundingMax().Return([3]float32{0.5, 0.5, 0.5}).Once()
		mockModel.EXPECT().BoundingRadius().Return(float32(0.5)).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Times(5)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{10000, 10000, 10000}, [3]float32{1, 1, 1}).Once()
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{mockAnim: mockBuf}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mockBuf, uint64(0), mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face: light outside camera frustum skips render passes", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{999, 999, 999}).Maybe()
		pl.EXPECT().Range().Return(float32(1)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		var view, proj, vp [16]float32
		common.LookAt(view[:], 0, 0, 0, 0, 0, -1, 0, 1, 0)
		common.Perspective(proj[:], float32(math.Pi/2), 1.0, 0.1, 100.0)
		common.Mul4(vp[:], proj[:], view[:])

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ViewProjectionMatrix().Return(vp).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: instance range exceeds light range skips draw", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 0, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(1)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		mockBuf := &wgpu.Buffer{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(3)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().LODCount().Return(1).Twice()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Once()
		mockModel.EXPECT().LODIndexCount(0).Return(6).Once()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-0.5, -0.5, -0.5}).Once()
		mockModel.EXPECT().BoundingMax().Return([3]float32{0.5, 0.5, 0.5}).Once()
		mockModel.EXPECT().BoundingRadius().Return(float32(0.5)).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Times(5)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{1000, 0, 0}, [3]float32{1, 1, 1}).Once()
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{mockAnim: mockBuf}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mockBuf, uint64(0), mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face: slot migration triggers ForceMarkDirty", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		suite.scene.cullingDisabled = true

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		suite.scene.lightPrevSlotMap = map[light.Light]uint32{pl: 999}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot slot migration triggers ForceMarkDirty", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		suite.scene.lightPrevSlotMap = map[light.Light]uint32{sl: 999}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot render loop: skips spot light not in lightShadowMap", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)

		sl1 := light_mocks.NewMockLight(suite.T())
		sl1.EXPECT().Enabled().Return(true).Maybe()
		sl1.EXPECT().CastsShadows().Return(true).Maybe()
		sl1.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl1.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl1.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl1.EXPECT().Range().Return(float32(10)).Maybe()
		sl1.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl1.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl1.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl1)

		sl2 := light_mocks.NewMockLight(suite.T())
		sl2.EXPECT().Enabled().Return(true).Maybe()
		sl2.EXPECT().CastsShadows().Return(true).Maybe()
		sl2.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		suite.scene.lightHandler.AddLight(sl2)

		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(0)).Times(3)
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: scale Y and Z dominate maxS calculation", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		mockBuf := &wgpu.Buffer{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(3)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().LODCount().Return(1).Twice()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Once()
		mockModel.EXPECT().LODIndexCount(0).Return(6).Once()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Once()
		mockModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Once()
		mockModel.EXPECT().BoundingRadius().Return(float32(1000)).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Times(5)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 3, 5}).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Maybe()
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{mockAnim: mockBuf}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}
		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{mockAnim: outputBGP}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mockBuf, uint64(0), mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mockBuf, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face depth pass: scale Y and Z dominate maxS calculation", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		mockBuf := &wgpu.Buffer{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(14)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Times(6)
		mockModel.EXPECT().LODCount().Return(1).Times(7)
		mockModel.EXPECT().Skinned().Return(false).Times(6)
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		mockModel.EXPECT().LODIndexCount(0).Return(6).Once()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Times(12)
		mockModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Times(12)
		mockModel.EXPECT().BoundingRadius().Return(float32(1000)).Times(12)
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(15)
		mockAnim.EXPECT().Model().Return(mockModel).Times(21)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{}, [3]float32{1, 3, 5}).Once()
		mockAnim.EXPECT().OutputBindGroupProvider().Return(outputBGP).Maybe()
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{mockAnim: mockBuf}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}
		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{mockAnim: outputBGP}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mockBuf, uint64(0), mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().ShadowDrawCallIndirect(mock.Anything, mock.Anything, mockBuf, mock.Anything).Return(nil).Times(6)
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("spot depth pass: instance within range but outside frustum visible=false continue", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(1)
		sh.SetLightShadowAtlasCols(1)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(50)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.99)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		mockBuf := &wgpu.Buffer{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Times(3)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().LODCount().Return(1).Twice()
		mockModel.EXPECT().Skinned().Return(false).Once()
		mockModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Once()
		mockModel.EXPECT().LODIndexCount(0).Return(6).Once()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-0.5, -0.5, -0.5}).Once()
		mockModel.EXPECT().BoundingMax().Return([3]float32{0.5, 0.5, 0.5}).Once()
		mockModel.EXPECT().BoundingRadius().Return(float32(0.5)).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		mockAnim.EXPECT().Model().Return(mockModel).Times(5)
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{20, 0, 0}, [3]float32{1, 1, 1}).Once()
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{mockAnim: mockBuf}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mockBuf, uint64(0), mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
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

	suite.Run("nil renderer returns early", func() {
		suite.scene.r = nil
		suite.scene.lightHandler.SetEnabled(true)
		suite.scene.cam = camera_mocks.NewMockCamera(suite.T())
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
		suite.rendererMock.EXPECT().BeginComputeFrame().Return().Once()
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

func (suite *sceneImplTest) TestSceneWrapperMethods() {
	suite.Run("SetPhysicsHandler stores replacement handler", func() {
		replacement := physics.NewPhysics()
		suite.scene.SetPhysicsHandler(replacement)
		suite.Equal(replacement, suite.scene.physicsHandler)
	})

	suite.Run("Get returns object for present id and nil for missing id", func() {
		objMock := game_object_mocks.NewMockGameObject(suite.T())
		suite.scene.registry = map[uint64]game_object.GameObject{
			42: objMock,
		}

		suite.Equal(objMock, suite.scene.Get(42))
		suite.Nil(suite.scene.Get(999))
	})

	suite.Run("Count returns registry size when populated", func() {
		objMockA := game_object_mocks.NewMockGameObject(suite.T())
		objMockB := game_object_mocks.NewMockGameObject(suite.T())
		suite.scene.registry = map[uint64]game_object.GameObject{
			1: objMockA,
			2: objMockB,
		}

		suite.Equal(2, suite.scene.Count())
	})

	suite.Run("RemoveLight deletes removed light from previous slot map", func() {
		removed := light.NewLight(light.LightTypePoint)
		kept := light.NewLight(light.LightTypeSpot)
		suite.scene.lightHandler.AddLight(removed)
		suite.scene.lightHandler.AddLight(kept)
		suite.scene.lightPrevSlotMap = map[light.Light]uint32{
			removed: 3,
			kept:    4,
		}

		suite.scene.RemoveLight(removed)

		_, removedStillTracked := suite.scene.lightPrevSlotMap[removed]
		_, keptStillTracked := suite.scene.lightPrevSlotMap[kept]
		suite.False(removedStillTracked)
		suite.True(keptStillTracked)
	})

	suite.Run("Lights returns values from light handler", func() {
		suite.scene.lightHandler = light.NewLightingHandler()
		l := light.NewLight(light.LightTypeDirectional)
		suite.scene.lightHandler.AddLight(l)

		lights := suite.scene.Lights()
		suite.Len(lights, 1)
		suite.Equal(l, lights[0])
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
		mdlMock.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdlMock.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdlMock.EXPECT().MeshProvider().Return(nil).Once()
		mdlMock.EXPECT().LODCount().Return(1).Once()
		mdlMock.EXPECT().RenderMaterials().Return(nil).Once()
		mdlMock.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()

		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Run(func(provider bind_group_provider.BindGroupProvider, descriptor wgpu.BindGroupLayoutDescriptor, bufferUsageOverrides map[int]wgpu.BufferUsage, bufferSizeOverrides map[int]uint64) {
				if provider.Buffer(1) == nil {
					provider.SetBuffer(1, &wgpu.Buffer{})
				}
			}).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(0).Maybe()

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(mdlMock).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(mock.Anything).Once()
		objMock.EXPECT().SetAnimatorInstanceID(mock.Anything).Once()
		objMock.EXPECT().ContactShadowExcluded().Return(false).Once()
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
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		objMock.EXPECT().ContactShadowExcluded().Return(false).Once()
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
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		objMock.EXPECT().ContactShadowExcluded().Return(false).Once()
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
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		objMock.EXPECT().ContactShadowExcluded().Return(false).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(nil).Once()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
	})

	suite.Run("contact-shadow excluded obj sets the instance flag", func() {
		suite.scene.buildInjectionMap()
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(false).Once()
		poolKey.EXPECT().Name().Return("testcontactshadow").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().SetInstanceFlags(uint32(0), animator.InstanceFlagContactShadowExcluded).Return().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()
		objMock.EXPECT().SetAnimatorInstanceID(0).Once()
		objMock.EXPECT().ContactShadowExcluded().Return(true).Once()
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
		fullPoolKey.EXPECT().BoundingMin().Return([3]float32{}).Once()
		fullPoolKey.EXPECT().BoundingMax().Return([3]float32{}).Once()
		fullPoolKey.EXPECT().MeshProvider().Return(nil).Once()
		fullPoolKey.EXPECT().LODCount().Return(1).Once()
		fullPoolKey.EXPECT().RenderMaterials().Return(nil).Once()
		fullPoolKey.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()

		fullAnimMock := animator_mocks.NewMockAnimator(suite.T())
		fullAnimMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		fullAnimMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		fullAnimMock.EXPECT().MaxInstances().Return(uint32(1)).Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{fullPoolKey: {fullAnimMock}}

		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Run(func(provider bind_group_provider.BindGroupProvider, descriptor wgpu.BindGroupLayoutDescriptor, bufferUsageOverrides map[int]wgpu.BufferUsage, bufferSizeOverrides map[int]uint64) {
				if provider.Buffer(1) == nil {
					provider.SetBuffer(1, &wgpu.Buffer{})
				}
			}).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(0).Maybe()

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(fullPoolKey).Once()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(mock.Anything).Once()
		objMock.EXPECT().SetAnimatorInstanceID(mock.Anything).Once()
		objMock.EXPECT().ContactShadowExcluded().Return(false).Once()
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
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		objMock.EXPECT().ContactShadowExcluded().Return(false).Once()
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
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		objMock.EXPECT().ContactShadowExcluded().Return(false).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(l).Once()
		objMock.EXPECT().RigidBody().Return(nil).Once()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
		suite.Len(suite.scene.lightObjects, 1)
		suite.Len(suite.scene.lightHandler.Lights(), 1)
	})

	suite.Run("physics pre-set sync maps", func() {
		suite.scene.buildInjectionMap()
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(false).Maybe()
		poolKey.EXPECT().Name().Return("testphys").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		syncBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		suite.scene.physicsHandler = physics.NewPhysics()
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
		objMock.EXPECT().ContactShadowExcluded().Return(false).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(rb).Maybe()
		objMock.EXPECT().AnimatorInstanceID().Return(int(0)).Maybe()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
	})

	suite.Run("physics lifecycle starting with nil sync maps initializes", func() {
		suite.scene.buildInjectionMap()

		syncShader := shader.NewShader("_test_sync", shader.ShaderTypeCompute,
			"engine/physics/assets/physics-sync.wgsl", shader.WithInjections(suite.scene.injections))
		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(syncShader).Maybe()

		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().Pipeline(mock.Anything).Return(pipeMock).Maybe()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()

		poolKey := model_mocks.NewMockModel(suite.T())
		poolKey.EXPECT().Skinned().Return(false).Maybe()
		poolKey.EXPECT().Name().Return("testphysinit").Maybe()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		computeBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		computeBGPMock.EXPECT().Buffer(mock.Anything).Return(nil).Maybe()
		computeBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGPMock).Maybe()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		suite.scene.registry = make(map[uint64]game_object.GameObject)
		suite.scene.SetPhysicsHandler(physics.NewPhysics())
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(0).Once()

		rb := physics.NewRigidBody()

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Model().Return(poolKey).Maybe()
		objMock.EXPECT().ID().Return(uint64(0)).Maybe()
		objMock.EXPECT().SetID(mock.Anything).Once()
		objMock.EXPECT().TransformData().Return([3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{}).Once()
		objMock.EXPECT().SetAnimator(animMock).Once()
		objMock.EXPECT().SetAnimatorInstanceID(0).Once()
		objMock.EXPECT().ContactShadowExcluded().Return(false).Once()
		objMock.EXPECT().Ephemeral().Return(false).Once()
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().RigidBody().Return(rb).Maybe()
		objMock.EXPECT().AnimatorInstanceID().Return(int(0)).Maybe()

		suite.NotPanics(func() { suite.scene.AddGameObject(objMock) })
		suite.Equal(lifecycle.LifecycleStateRunning, suite.scene.physicsHandler.Lifecycle().State())
		suite.NotNil(suite.scene.physicsSyncAnimMap)
		suite.NotNil(suite.scene.physicsSyncGroup)
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
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().MaxInstances().Return(uint32(1)).Once()
		animMock.EXPECT().AddInstance().Return(uint32(0), nil).Once()
		animMock.EXPECT().SetInstanceData(uint32(0), mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{poolKey: {animMock}}

		syncBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.physicsHandler = physics.NewPhysics()
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
		objMock.EXPECT().ContactShadowExcluded().Return(false).Once()
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
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()

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
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()

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
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()

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
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()

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

	suite.Run("prunes empty animator after last instance removed", func() {
		mockModel := model_mocks.NewMockModel(suite.T())
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animLC := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		animLC.OnTransitionTo(lifecycle.LifecycleStateRemoved, lifecycle.Hook(func() error {
			suite.scene.pruneAnimator(animMock)
			return nil
		}))
		animMock.EXPECT().RemoveInstance(uint32(0)).Return(uint32(0), false).Once()
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		animMock.EXPECT().Lifecycle().Return(animLC).Maybe()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().Release().Once()

		objMock := game_object_mocks.NewMockGameObject(suite.T())
		objMock.EXPECT().Light().Return(nil).Once()
		objMock.EXPECT().Animator().Return(animMock).Once()
		objMock.EXPECT().AnimatorInstanceID().Return(0).Once()
		objMock.EXPECT().SetAnimatorInstanceID(-1).Once()

		suite.scene.registry = map[uint64]game_object.GameObject{1: objMock}
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mockModel: {animMock}}
		suite.scene.shadowIndirectBuffers[animMock] = nil
		suite.scene.instanceLookup = map[animator.Animator]map[uint32]uint64{animMock: {}}

		suite.scene.RemoveGameObject(1)

		_, animPoolHasModel := suite.scene.animatorPool[mockModel]
		suite.False(animPoolHasModel)
		_, shadowHasAnim := suite.scene.shadowIndirectBuffers[animMock]
		suite.False(shadowHasAnim)
		_, lookupHasAnim := suite.scene.instanceLookup[animMock]
		suite.False(lookupHasAnim)
	})
}

func (suite *sceneImplTest) TestPruneAnimator() {
	suite.Run("single animator in pool — model key deleted", func() {
		mockModel := model_mocks.NewMockModel(suite.T())
		mockAnimator := animator_mocks.NewMockAnimator(suite.T())
		mockAnimator.EXPECT().Model().Return(mockModel).Once()
		mockAnimator.EXPECT().Release().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{mockModel: {mockAnimator}}

		suite.scene.pruneAnimator(mockAnimator)

		_, exists := suite.scene.animatorPool[mockModel]
		suite.False(exists)
	})

	suite.Run("multiple animators in pool — key retained animator removed", func() {
		mockModel := model_mocks.NewMockModel(suite.T())
		mockAnimatorA := animator_mocks.NewMockAnimator(suite.T())
		mockAnimatorB := animator_mocks.NewMockAnimator(suite.T())
		mockAnimatorA.EXPECT().Model().Return(mockModel).Once()
		mockAnimatorA.EXPECT().Release().Once()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{mockModel: {mockAnimatorA, mockAnimatorB}}

		suite.scene.pruneAnimator(mockAnimatorA)

		pool := suite.scene.animatorPool[mockModel]
		suite.Len(pool, 1)
		suite.Equal(mockAnimatorB, pool[0])
	})

	suite.Run("shadow indirect buffer deleted from map", func() {
		mockAnimator := animator_mocks.NewMockAnimator(suite.T())
		mockAnimator.EXPECT().Model().Return(nil).Once()
		mockAnimator.EXPECT().Release().Once()

		suite.scene.shadowIndirectBuffers[mockAnimator] = nil

		suite.scene.pruneAnimator(mockAnimator)

		_, exists := suite.scene.shadowIndirectBuffers[mockAnimator]
		suite.False(exists)
	})

	suite.Run("animator Release called", func() {
		mockAnimator := animator_mocks.NewMockAnimator(suite.T())
		mockAnimator.EXPECT().Model().Return(nil).Once()
		mockAnimator.EXPECT().Release().Once()

		suite.scene.pruneAnimator(mockAnimator)
	})

	suite.Run("instanceLookup entry deleted", func() {
		mockAnimator := animator_mocks.NewMockAnimator(suite.T())
		mockAnimator.EXPECT().Model().Return(nil).Once()
		mockAnimator.EXPECT().Release().Once()

		suite.scene.instanceLookup = map[animator.Animator]map[uint32]uint64{mockAnimator: {0: 1}}

		suite.scene.pruneAnimator(mockAnimator)

		_, exists := suite.scene.instanceLookup[mockAnimator]
		suite.False(exists)
	})

	suite.Run("nil model does not panic", func() {
		mockAnimator := animator_mocks.NewMockAnimator(suite.T())
		mockAnimator.EXPECT().Model().Return(nil).Once()
		mockAnimator.EXPECT().Release().Once()

		suite.NotPanics(func() { suite.scene.pruneAnimator(mockAnimator) })
	})
}

func (suite *sceneImplTest) TestPrepareCompute() {

	suite.Run("cam nil camBGP skips WriteBuffers", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		obj := game_object_mocks.NewMockGameObject(suite.T())
		obj.EXPECT().Light().Return(nil).Once()
		suite.scene.lightObjects = []game_object.GameObject{obj}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 zero instance count skipped", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Times(4)
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 nil model skipped", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("").Times(2)
		mockModel.EXPECT().LODMeshProvider(0).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k14").Times(2)
		mockModel.EXPECT().LODMeshProvider(0).Return(mockMeshBGP).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().ResetIndirectArgs(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("k14").Return(nil).Times(2)
		mockMeshBGP.EXPECT().IndexCount().Return(0).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase1 nil shader skipped", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Times(2)
		mockMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k15").Times(2)
		mockModel.EXPECT().LODMeshProvider(0).Return(mockMeshBGP).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().ResetIndirectArgs(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("k15").Return(mockPipe).Times(2)
		mockMeshBGP.EXPECT().IndexCount().Return(0).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("pc16-key").Times(2)
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(5)
		animMock.EXPECT().Model().Return(mockModel).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().SetScreenSize(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().SetProjectionX(mock.Anything).Return().Once()
		animMock.EXPECT().SetHiZMipCount(mock.Anything).Return().Once()
		animMock.EXPECT().SetViewProj(mock.Anything).Return().Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(1)).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("pc16-key").Return(realPipe).Times(2)
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "pc16-key" && d[0].Providers[0].Provider == computeBGP
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("pc17-key").Times(2)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGPMock).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(5)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().SetFrustumPlanes(mock.Anything).Return().Once()
		animMock.EXPECT().SetScreenSize(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().SetProjectionX(mock.Anything).Return().Once()
		animMock.EXPECT().SetHiZMipCount(mock.Anything).Return().Once()
		animMock.EXPECT().SetViewProj(mock.Anything).Return().Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(1)).Once()
		animMock.EXPECT().ResetIndirectArgs(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("pc17-key").Return(realPipe).Times(2)
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "pc17-key" && d[0].Providers[0].Provider == computeBGP
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 culling nil model falls through to StagedWriteData", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("").Times(2)
		mockModel.EXPECT().LODMeshProvider(0).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		meshBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("").Times(2)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGPMock).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().ResetIndirectArgs(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		meshBGPMock.EXPECT().IndexCount().Return(0).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 culling nil pipeline continue skips StagedWriteData", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k21").Times(2)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().ResetIndirectArgs(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("k21").Return(nil).Times(2)
		meshBGP.EXPECT().IndexCount().Return(0).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 culling nil shader continue skips StagedWriteData", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Times(2)
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k22").Times(2)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(4)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().ResetIndirectArgs(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("k22").Return(mockPipe).Times(2)
		meshBGP.EXPECT().IndexCount().Return(0).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k23").Times(2)
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(5)
		animMock.EXPECT().Model().Return(mockModel).Times(3)
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().SetFrustumPlanes(mock.Anything).Return().Once()
		animMock.EXPECT().SetScreenSize(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().SetProjectionX(mock.Anything).Return().Once()
		animMock.EXPECT().SetHiZMipCount(mock.Anything).Return().Once()
		animMock.EXPECT().SetViewProj(mock.Anything).Return().Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(1)).Once()
		animMock.EXPECT().ResetIndirectArgs(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().StagedWriteData().Return([]bind_group_provider.BufferWrite{
			{Provider: computeBGP, Binding: 0, Data: []byte{1}},
		}).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k23").Return(realPipe).Times(2)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "k23" && d[0].Providers[0].Provider == computeBGP
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("phase2 non-empty StagedWriteData calls WriteBuffers", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		suite.scene.physicsHandler = nil
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics lifecycle not running skips", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle()).Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics running ReadbackPending false substeps 0 no physWrites", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Once()
		phMock.EXPECT().BodiesCount().Return(0).Once()
		phMock.EXPECT().StagedWriteData().Return(nil).Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics substeps 0 with physWrites calls WriteBuffers", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Once()
		phMock.EXPECT().BodiesCount().Return(0).Once()
		phMock.EXPECT().StagedWriteData().Return([]bind_group_provider.BufferWrite{{Binding: 0}}).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physics physicsSyncWrites appended on substeps 0", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		suite.scene.physicsSyncWrites = []bind_group_provider.BufferWrite{{Binding: 1}}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
		phMock.EXPECT().ReadbackPending().Return(false).Once()
		phMock.EXPECT().PrepareStep(mock.Anything).Return(0, nil).Once()
		phMock.EXPECT().BodiesCount().Return(0).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
		phMock.EXPECT().ReadbackPending().Return(true).Once()
		phMock.EXPECT().BodiesCount().Return(0).Twice()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stagingBuf := &wgpu.Buffer{}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
		phMock.EXPECT().ReadbackPending().Return(true).Once()
		phMock.EXPECT().BodiesCount().Return(1).Twice()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stagingBuf := &wgpu.Buffer{}
		readData := make([]byte, 16)
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
		phMock.EXPECT().ReadbackPending().Return(true).Once()
		phMock.EXPECT().BodiesCount().Return(1).Twice()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Times(8)
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		syncBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: syncBGP}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
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

	suite.Run("physics paused flushes writes and dispatches sync without stepping", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}

		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		syncBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: syncBGP}
		suite.scene.physicsSyncWrites = []bind_group_provider.BufferWrite{{Binding: 1}}

		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStatePaused))).Once()
		phMock.EXPECT().StagedWriteData().Return([]bind_group_provider.BufferWrite{{Binding: 7}}).Once()
		phMock.EXPECT().BodiesCount().Return(2).Once()
		phMock.EXPECT().Buffers().Return(bufBGP).Once()
		phMock.EXPECT().PipelineKey("sync").Return("sync_key").Once()

		suite.rendererMock.EXPECT().WriteBuffers(mock.MatchedBy(func(writes []bind_group_provider.BufferWrite) bool {
			if len(writes) != 3 {
				return false
			}

			foundStaged := false
			foundSyncMap := false
			foundBodyCount := false

			for _, write := range writes {
				switch {
				case write.Binding == 7:
					foundStaged = true
				case write.Binding == 1:
					foundSyncMap = true
				case write.Provider == bufBGP && write.Binding == 3 && write.Offset == 20 && len(write.Data) == 4:
					foundBodyCount = binary.LittleEndian.Uint32(write.Data) == 2
				}
			}

			return foundStaged && foundSyncMap && foundBodyCount
		})).Return().Once()
		suite.rendererMock.EXPECT().Pipeline("sync_key").Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(dispatches []renderer.ComputeDispatch) bool {
			return len(dispatches) == 1 &&
				dispatches[0].PipelineKey == "sync_key" &&
				len(dispatches[0].Providers) == 1 &&
				dispatches[0].Providers[0].Group == 0 &&
				dispatches[0].Providers[0].Provider == syncBGP &&
				dispatches[0].WorkGroupCount == [3]uint32{1, 1, 1}
		})).Return().Once()

		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
		suite.Len(suite.scene.physicsSyncWrites, 0)
	})

	suite.Run("physics paused with zero bodies writes body count and skips sync dispatch", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}

		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		syncBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: syncBGP}
		suite.scene.physicsSyncWrites = []bind_group_provider.BufferWrite{{Binding: 1}}

		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStatePaused))).Once()
		phMock.EXPECT().StagedWriteData().Return([]bind_group_provider.BufferWrite{{Binding: 7}}).Once()
		phMock.EXPECT().BodiesCount().Return(0).Once()
		phMock.EXPECT().Buffers().Return(bufBGP).Once()

		suite.rendererMock.EXPECT().WriteBuffers(mock.MatchedBy(func(writes []bind_group_provider.BufferWrite) bool {
			if len(writes) != 3 {
				return false
			}

			foundStaged := false
			foundSyncMap := false
			foundBodyCount := false

			for _, write := range writes {
				switch {
				case write.Binding == 7:
					foundStaged = true
				case write.Binding == 1:
					foundSyncMap = true
				case write.Provider == bufBGP && write.Binding == 3 && write.Offset == 20 && len(write.Data) == 4:
					foundBodyCount = binary.LittleEndian.Uint32(write.Data) == 0
				}
			}

			return foundStaged && foundSyncMap && foundBodyCount
		})).Return().Once()

		suite.scene.physicsHandler = phMock
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
		suite.Len(suite.scene.physicsSyncWrites, 0)
	})

	suite.Run("physics ConsumeReadbackRequest true StagingBuffer nil no CopyBufferToBuffer", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		stagingBuf := &wgpu.Buffer{}
		stageBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bufBGP.EXPECT().Buffer(0).Return(nil).Once()
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.boneParticleUpdateGroups = []*boneParticleUpdateGroup{
			{bgp: bgpMock, particleCount: 5},
		}
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle()).Once()
		phMock.EXPECT().PipelineKey("bone_update").Return("bone_key").Once()
		suite.scene.physicsHandler = phMock
		suite.rendererMock.EXPECT().Pipeline("bone_key").Return(nil).Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("bone particle nil boneUpdateShader skips", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle()).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle()).Once()
		phMock.EXPECT().PipelineKey("bone_update").Return("bone_key42").Once()
		suite.scene.physicsHandler = phMock
		suite.rendererMock.EXPECT().Pipeline("bone_key42").Return(realBonePipe).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 2 &&
				d[0].PipelineKey == "bone_key42" && d[0].Providers[0].Provider == bgp1 &&
				d[1].PipelineKey == "bone_key42" && d[1].Providers[0].Provider == bgp2
		})).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})

	suite.Run("physDispatchGroups xSize zero guard and groups zero guard", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("k44").Times(2)
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(2)).Times(5)
		animMock.EXPECT().Model().Return(mockModel).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().SetScreenSize(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().SetProjectionX(mock.Anything).Return().Once()
		animMock.EXPECT().SetHiZMipCount(mock.Anything).Return().Once()
		animMock.EXPECT().SetViewProj(mock.Anything).Return().Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(2)).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("k44").Return(mockPipe).Times(2)
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "k44" && d[0].Providers[0].Provider == computeBGP &&
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
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
		phMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle()).Once()
		phMock.EXPECT().PipelineKey("bone_update").Return("bone_key45").Once()
		suite.scene.physicsHandler = phMock
		suite.rendererMock.EXPECT().Pipeline("bone_key45").Return(mockBonePipe).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 2 &&
				d[0].PipelineKey == "bone_key45" && d[0].Providers[0].Provider == bgp1 && d[0].WorkGroupCount == [3]uint32{1, 1, 1} &&
				d[1].PipelineKey == "bone_key45" && d[1].Providers[0].Provider == bgp2 && d[1].WorkGroupCount == [3]uint32{5, 1, 1}
		})).Return().Once()
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
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("bone-model-key").Times(2)
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(2)).Times(5)
		animMock.EXPECT().Model().Return(mockModel).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().SetScreenSize(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().SetProjectionX(mock.Anything).Return().Once()
		animMock.EXPECT().SetHiZMipCount(mock.Anything).Return().Once()
		animMock.EXPECT().SetViewProj(mock.Anything).Return().Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(2)).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		suite.rendererMock.EXPECT().Pipeline("bone-model-key").Return(mockPipe).Times(2)
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "bone-model-key" && d[0].Providers[0].Provider == computeBGP &&
				d[0].WorkGroupCount == [3]uint32{1, 1, 1}
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
	})
}

func (suite *sceneImplTest) TestPrepareLights() {
	suite.Run("light handler disabled returns early", func() {
		suite.NotPanics(func() { suite.scene.PrepareLights() })
	})

	suite.Run("light handler enabled single binding write", func() {
		lightsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(lightsBGP).Once()
		mockLH.EXPECT().Lights().Return(nil).Once()
		mockLH.EXPECT().MaxGPULights().Return(100).Once()
		mockLH.EXPECT().MarshalLightBuffer(mock.Anything, mock.Anything).Return(make([]byte, 16)).Once()
		suite.scene.lightHandler = mockLH
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareLights() })
	})

	suite.Run("light handler enabled two bindings when data over 16", func() {
		lightsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(lightsBGP).Once()
		mockLH.EXPECT().Lights().Return(nil).Once()
		mockLH.EXPECT().MaxGPULights().Return(100).Once()
		mockLH.EXPECT().MarshalLightBuffer(mock.Anything, mock.Anything).Return(make([]byte, 32)).Once()
		suite.scene.lightHandler = mockLH
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareLights() })
	})

	suite.Run("light sort triggers when rawLights exceeds MaxGPULights", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Controller().Return(nil).Once()
		suite.scene.cam = camMock
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
		suite.NotPanics(func() { suite.scene.PrepareLights() })
	})

	suite.Run("light sort non-nil controller uses camera position", func() {
		ctrl := camera_mocks.NewMockCameraController(suite.T())
		ctrl.EXPECT().Position().Return(float32(5), float32(5), float32(5)).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Controller().Return(ctrl).Once()
		suite.scene.cam = camMock
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
		suite.NotPanics(func() { suite.scene.PrepareLights() })
	})

	suite.Run("light sort directional light returns MaxFloat32 importance", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Controller().Return(nil).Once()
		suite.scene.cam = camMock
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
		suite.NotPanics(func() { suite.scene.PrepareLights() })
	})

	suite.Run("light sort comparison branches impA greater and impA less than", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Controller().Return(nil).Once()
		suite.scene.cam = camMock
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
		suite.NotPanics(func() { suite.scene.PrepareLights() })
	})

	suite.Run("light sort equal importance returns zero", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Controller().Return(nil).Once()
		suite.scene.cam = camMock
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
		suite.NotPanics(func() { suite.scene.PrepareLights() })
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
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(0)).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("nil model skipped", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("nil mesh provider skipped", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{}).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		fragShdrMock := shader_mocks.NewMockShader(suite.T())
		fragShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(fragShdrMock).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("k13").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k13", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("k14").Return(mockPipe).Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().Pipeline("k15").Return(mockPipe).Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().Pipeline("k16").Return(mockPipe).Once()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(effectBGP).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(nil).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(effectBGP).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(effectBGP).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(nil).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl1, decl2}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl1, decl2}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
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
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k41").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockRenderPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k41").Return(mockRenderPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k41", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("culling cs.Declarations IndirectArgs array-wrapped binding sets indirectBinding", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k42").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockRenderPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k42").Return(mockRenderPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k42", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("culling cs.Declarations IndirectArgs non-nil buffer calls DrawCallIndirect with discovered binding", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k43").Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockRenderPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockRenderPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		indBuf := &wgpu.Buffer{}
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		animMock.EXPECT().CullingEnabled().Return(true).Once()
		animMock.EXPECT().IndirectBuffer(0).Return(indBuf).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("k43").Return(mockRenderPipe).Once()
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
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Once()
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
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
		opt := WithPhysicsHandler(physics.NewPhysics())
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
		bgp.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		r.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		r.EXPECT().CreateHiZTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, 0).Once()

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
		suite.scene.lightHandler = lhMock
		suite.scene.initSSAO()
	})

	suite.Run("w zero returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.ssaoHandler = ssaoMock
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 600
		suite.scene.initSSAO()
	})

	suite.Run("RegisterPipelines ssao_compute error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.ssaoHandler = ssaoMock
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil)).Once()
		ssaoMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("reg err")).Once()
		suite.Panics(func() { suite.scene.initSSAO() })
	})

	suite.Run("RegisterPipelines ssao_blur error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.ssaoHandler = ssaoMock
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil)).Once()
		ssaoMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetPipelineKey("ssao_compute", "ssao_compute").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("blur reg err")).Once()
		suite.Panics(func() { suite.scene.initSSAO() })
	})

	suite.Run("InitBindGroup ssao_compute error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.ssaoHandler = ssaoMock
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil)).Once()
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
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.ssaoHandler = ssaoMock
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil)).Once()
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
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.ssaoHandler = ssaoMock
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Once()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil)).Once()
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
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.ssaoHandler = ssaoMock
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_ssao_samples": "32u"}
		ssaoMock.EXPECT().HalfResolution().Return(false).Once()
		suite.rendererMock.EXPECT().CreateSSAOTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Times(2)
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Times(2)
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Times(2)
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Times(2)
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Times(2)
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Times(2)
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Times(2)
		ssaoMock.EXPECT().SetSlot(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil)).Once()
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
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		lhMock.EXPECT().Bgp("ssao_lit").Return(bgpMock).Once()
		suite.scene.ssaoHandler = ssaoMock

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
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		lhMock.EXPECT().Bgp("ssao_lit").Return(bgpMock).Once()
		suite.scene.ssaoHandler = ssaoMock
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
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		lhMock.EXPECT().Bgp("ssao_lit").Return(bgpMock).Once()
		suite.scene.ssaoHandler = ssaoMock
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
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		lhMock.EXPECT().Bgp("ssao_lit").Return(bgpMock).Once()
		suite.scene.ssaoHandler = ssaoMock
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
		suite.scene.lightHandler = lhMock
		suite.scene.initGBuffer()
	})

	suite.Run("zero screen width returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.gBufferHandler = gbufMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 480
		suite.scene.initGBuffer()
	})

	suite.Run("zero screen height returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.gBufferHandler = gbufMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 0
		suite.scene.initGBuffer()
	})

	suite.Run("RegisterGBufferPipeline static error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.gBufferHandler = gbufMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_bones": "64u"}
		suite.rendererMock.EXPECT().CreateGBufferTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Times(2)
		gbufMock.EXPECT().SetNormalTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetNormalTextureView(mock.Anything).Maybe()
		gbufMock.EXPECT().SetAlbedoTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetAlbedoTextureView(mock.Anything).Maybe()
		gbufMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		gbufMock.EXPECT().SetSlot(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).
			Return(errors.New("pipe err")).Once()
		suite.Panics(func() { suite.scene.initGBuffer() })
	})

	suite.Run("RegisterGBufferPipeline skinned error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.gBufferHandler = gbufMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_bones": "64u"}
		suite.rendererMock.EXPECT().CreateGBufferTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Times(2)
		gbufMock.EXPECT().SetNormalTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetNormalTextureView(mock.Anything).Maybe()
		gbufMock.EXPECT().SetAlbedoTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetAlbedoTextureView(mock.Anything).Maybe()
		gbufMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		gbufMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		gbufMock.EXPECT().SetSlot(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(nil).Once()
		gbufMock.EXPECT().SetPipelineKey("static", "gbuffer_static").Once()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(errors.New("skinned pipe err")).Once()
		suite.Panics(func() { suite.scene.initGBuffer() })
	})

	suite.Run("happy path completes all steps", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.gBufferHandler = gbufMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.injections = map[string]string{"max_bones": "64u"}
		suite.rendererMock.EXPECT().CreateGBufferTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil),
				(*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Times(2)
		gbufMock.EXPECT().SetNormalTexture(mock.Anything).Times(2)
		gbufMock.EXPECT().SetNormalTextureView(mock.Anything).Times(2)
		gbufMock.EXPECT().SetAlbedoTexture(mock.Anything).Times(2)
		gbufMock.EXPECT().SetAlbedoTextureView(mock.Anything).Times(2)
		gbufMock.EXPECT().SetDepthTexture(mock.Anything).Times(2)
		gbufMock.EXPECT().SetDepthTextureView(mock.Anything).Times(2)
		gbufMock.EXPECT().SetSlot(mock.Anything).Maybe()
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
		suite.scene.lightHandler = lhMock
		suite.scene.initContactShadows()
	})

	suite.Run("GBufferHandler not enabled returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(false).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.initContactShadows()
	})

	suite.Run("zero screen width returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 600
		suite.scene.initContactShadows()
	})

	suite.Run("zero screen height returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 0
		suite.scene.initContactShadows()
	})

	suite.Run("RegisterPipelines error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateContactShadowTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Times(2)
		csMock.EXPECT().SetTexture(mock.Anything).Times(2)
		csMock.EXPECT().SetTextureView(mock.Anything).Times(2)
		csMock.EXPECT().SetSlot(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil)).Once()
		csMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).
			Return(errors.New("pipe err")).Once()
		suite.Panics(func() { suite.scene.initContactShadows() })
	})

	suite.Run("InitBindGroup error panics", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateContactShadowTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Times(2)
		csMock.EXPECT().SetTexture(mock.Anything).Times(2)
		csMock.EXPECT().SetTextureView(mock.Anything).Times(2)
		csMock.EXPECT().SetSlot(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil)).Once()
		csMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		csMock.EXPECT().SetPipelineKey("contact_shadow_compute", "contact_shadow_compute").Once()
		csBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		csMock.EXPECT().Bgp("contact_shadow_compute").Return(csBGPMock).Once()
		gbufMock.EXPECT().SetSlot(mock.Anything).Maybe()
		csBGPMock.EXPECT().SetSlot(mock.Anything).Maybe()
		gbufMock.EXPECT().DepthTextureView().Return((*wgpu.TextureView)(nil)).Once()
		gbufMock.EXPECT().NormalTextureView().Return((*wgpu.TextureView)(nil)).Once()
		gbufMock.EXPECT().AlbedoTextureView().Return((*wgpu.TextureView)(nil)).Once()
		csBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(csBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("bgp err")).Once()
		suite.Panics(func() { suite.scene.initContactShadows() })
	})

	suite.Run("happy path completes", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbufMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.scene.gBufferHandler = gbufMock
		gbufMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.rendererMock.EXPECT().CreateContactShadowTextures(800, 600).
			Return((*wgpu.TextureView)(nil), (*wgpu.Texture)(nil)).Times(2)
		csMock.EXPECT().SetTexture(mock.Anything).Times(2)
		csMock.EXPECT().SetTextureView(mock.Anything).Times(2)
		csMock.EXPECT().SetSlot(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return((*wgpu.Sampler)(nil)).Once()
		csMock.EXPECT().SetLinearSampler(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		csMock.EXPECT().SetPipelineKey("contact_shadow_compute", "contact_shadow_compute").Once()
		csBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		csMock.EXPECT().Bgp("contact_shadow_compute").Return(csBGPMock).Once()
		gbufMock.EXPECT().SetSlot(mock.Anything).Maybe()
		csBGPMock.EXPECT().SetSlot(mock.Anything).Maybe()
		gbufMock.EXPECT().DepthTextureView().Return((*wgpu.TextureView)(nil)).Twice()
		gbufMock.EXPECT().NormalTextureView().Return((*wgpu.TextureView)(nil)).Twice()
		gbufMock.EXPECT().AlbedoTextureView().Return((*wgpu.TextureView)(nil)).Twice()
		csBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(csBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Twice()
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
		bgpMock.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgpMock.EXPECT().SetSlot(0).Return().Once()
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
		bgpMock.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgpMock.EXPECT().SetSlot(0).Return().Once()
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

	suite.Run("InitBindGroup slot 1 error panics", func() {
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
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgpMock.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("slot1 err")).Once()
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
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil).Once()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Once()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgp0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgp1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_data_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_0").Return(bgp0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp0.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp0.EXPECT().SetSlot(0).Return().Once()
		shMock.EXPECT().SetBgp("csm_data_1", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_1").Return(bgp1).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp1.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp1.EXPECT().SetSlot(0).Return().Once()
		lhMock.EXPECT().MaxGPULights().Return(1).Once()
		shMock.EXPECT().LightShadowTileSize().Return(256).Once()
	}

	setupThroughStaticPipelines := func(shMock *light_mocks.MockShadowHandler) {
		shMock.EXPECT().SetLightShadowAtlasSlots(6).Once()
		shMock.EXPECT().SetLightShadowAtlasCols(3).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(768, 512).Return(nil, nil).Once()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Once()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Once()
		spotKeys := []string{"spot_shadow_0", "spot_shadow_1", "spot_shadow_2", "spot_shadow_3", "spot_shadow_4", "spot_shadow_5"}
		for _, key := range spotKeys {
			spotBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
			shMock.EXPECT().SetBgp(key, mock.Anything).Once()
			shMock.EXPECT().Bgp(key).Return(spotBGP).Once()
			suite.rendererMock.EXPECT().InitBindGroup(spotBGP, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
			spotBGP.EXPECT().SetSlot(1).Return().Once()
			suite.rendererMock.EXPECT().InitBindGroup(spotBGP, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
			spotBGP.EXPECT().SetSlot(0).Return().Once()
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

	suite.Run("InitBindGroup CSM cascade error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil).Once()
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

	suite.Run("InitBindGroup CSM cascade slot 1 error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil).Once()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Once()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgp0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_data_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_0").Return(bgp0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp0.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("slot1 err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, nil) })
	})

	suite.Run("uniform buffer entry populates sizeOverrides", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil).Once()
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
		bgp0.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp0.EXPECT().SetSlot(0).Return().Once()
		shMock.EXPECT().SetBgp("csm_data_1", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_1").Return(bgp1).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp1.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp1.EXPECT().SetSlot(0).Return().Once()
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
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil).Once()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Once()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgp0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgp1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_data_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_0").Return(bgp0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp0.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp0.EXPECT().SetSlot(0).Return().Once()
		shMock.EXPECT().SetBgp("csm_data_1", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_1").Return(bgp1).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp1.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp1.EXPECT().SetSlot(0).Return().Once()
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
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(2048, 1024).Return(nil, nil).Once()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Once()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Once()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil).Once()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Once()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgp0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgp1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_data_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_0").Return(bgp0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp0.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp0.EXPECT().SetSlot(0).Return().Once()
		shMock.EXPECT().SetBgp("csm_data_1", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_data_1").Return(bgp1).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp1.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgp1, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgp1.EXPECT().SetSlot(0).Return().Once()
		lhMock.EXPECT().MaxGPULights().Return(1).Once()
		shMock.EXPECT().LightShadowTileSize().Return(16384).Once()
		shMock.EXPECT().SetLightShadowAtlasSlots(1).Once()
		shMock.EXPECT().SetLightShadowAtlasCols(1).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(16384, 16384).Return(nil, nil).Once()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Once()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Once()
		spotBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("spot_shadow_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("spot_shadow_0").Return(spotBGP).Once()
		suite.rendererMock.EXPECT().InitBindGroup(spotBGP, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		spotBGP.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(spotBGP, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		spotBGP.EXPECT().SetSlot(0).Return().Once()
		suite.rendererMock.EXPECT().RegisterShadowDepthPipeline(mock.Anything).Return(nil).Times(3)
		shMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Times(3)
		suite.scene.lightHandler = lhMock
		suite.scene.initShadowMap(shaderMock, nil)
	})

	suite.Run("InitBindGroup spot slot error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		setupThroughCascades(lhMock, shMock, shaderMock)
		shMock.EXPECT().SetLightShadowAtlasSlots(6).Once()
		shMock.EXPECT().SetLightShadowAtlasCols(3).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(768, 512).Return(nil, nil).Once()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Once()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Once()
		spotBGP0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("spot_shadow_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("spot_shadow_0").Return(spotBGP0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(spotBGP0, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("spot bgp err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, nil) })
	})

	suite.Run("InitBindGroup spot shadow slot 1 error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		setupThroughCascades(lhMock, shMock, shaderMock)
		shMock.EXPECT().SetLightShadowAtlasSlots(6).Once()
		shMock.EXPECT().SetLightShadowAtlasCols(3).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(768, 512).Return(nil, nil).Once()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Once()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Once()
		spotBGP0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("spot_shadow_0", mock.Anything).Once()
		shMock.EXPECT().Bgp("spot_shadow_0").Return(spotBGP0).Once()
		suite.rendererMock.EXPECT().InitBindGroup(spotBGP0, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		spotBGP0.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(spotBGP0, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("slot1 err")).Once()
		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.initShadowMap(shaderMock, nil) })
	})

	suite.Run("RegisterShadowDepthPipeline static error panics", func() {
		lhMock, shMock := newBaseHandlers()
		shaderMock := shader_mocks.NewMockShader(suite.T())
		setupThroughCascades(lhMock, shMock, shaderMock)
		shMock.EXPECT().SetLightShadowAtlasSlots(6).Once()
		shMock.EXPECT().SetLightShadowAtlasCols(3).Once()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(768, 512).Return(nil, nil).Once()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Once()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Once()
		spotKeys := []string{"spot_shadow_0", "spot_shadow_1", "spot_shadow_2", "spot_shadow_3", "spot_shadow_4", "spot_shadow_5"}
		for _, key := range spotKeys {
			spotBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
			shMock.EXPECT().SetBgp(key, mock.Anything).Once()
			shMock.EXPECT().Bgp(key).Return(spotBGP).Once()
			suite.rendererMock.EXPECT().InitBindGroup(spotBGP, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
			spotBGP.EXPECT().SetSlot(1).Return().Once()
			suite.rendererMock.EXPECT().InitBindGroup(spotBGP, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
			spotBGP.EXPECT().SetSlot(0).Return().Once()
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
		bgpMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
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
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
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
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
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
		csMock.EXPECT().Enabled().Return(false).Twice()
		csMock.EXPECT().SetSlot(0).Return().Once()
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("contact shadow handler TextureView nil fallback path succeeds", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().Enabled().Return(true).Twice()
		csMock.EXPECT().LinearSampler().Return(&wgpu.Sampler{}).Twice()
		csMock.EXPECT().SetSlot(0).Return().Twice()
		csMock.EXPECT().SetSlot(1).Return().Once()
		csMock.EXPECT().TextureView().Return(nil).Twice()
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("contact shadow handler LinearSampler nil fallback path succeeds", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().Enabled().Return(true).Twice()
		csMock.EXPECT().LinearSampler().Return(nil).Twice()
		csMock.EXPECT().SetSlot(0).Return().Once()
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("contact shadow handler all OK bindings 5 and 6 set", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := setupBGP(shMock, fragMock)
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().Enabled().Return(true).Twice()
		csMock.EXPECT().LinearSampler().Return(&wgpu.Sampler{}).Times(4)
		csMock.EXPECT().SetSlot(0).Return().Twice()
		csMock.EXPECT().SetSlot(1).Return().Once()
		csMock.EXPECT().TextureView().Return(&wgpu.TextureView{}).Times(4)
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		bgpMock.EXPECT().SetTextureView(5, mock.Anything).Twice()
		bgpMock.EXPECT().SetSampler(6, mock.Anything).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("initializes both bgp slots and resets slot 0 at end", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shMock.EXPECT().SetBgp("csm_shadow_lit", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_shadow_lit").Return(bgpMock).Once()
		fragMock.EXPECT().BindGroupLayoutDescriptor(4).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgpMock.EXPECT().SetTextureView(0, mock.Anything).Once()
		bgpMock.EXPECT().SetSampler(1, mock.Anything).Once()
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(nil).Once()
		bgpMock.EXPECT().SetSlot(0).Return().Once()
		bgpMock.EXPECT().SetSlot(1).Return().Once()
		bgpMock.EXPECT().SetSlot(0).Return().Once()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 5, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitSampler(bgpMock, 6, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initCSMShadowLitBindGroup(fragMock) })
	})

	suite.Run("contact shadow slot selection binds matching slot views and resets slot 0", func() {
		lhMock, shMock, fragMock := makeReadyShader([]shader.Annotation{makeProviderDecl()})
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		view0 := &wgpu.TextureView{}
		view1 := &wgpu.TextureView{}
		sampler := &wgpu.Sampler{}

		shMock.EXPECT().SetBgp("csm_shadow_lit", mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_shadow_lit").Return(bgpMock).Once()
		fragMock.EXPECT().BindGroupLayoutDescriptor(4).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgpMock.EXPECT().SetTextureView(0, mock.Anything).Once()
		bgpMock.EXPECT().SetSampler(1, mock.Anything).Once()
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()

		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().Enabled().Return(true).Twice()
		csMock.EXPECT().LinearSampler().Return(sampler).Times(4)
		csMock.EXPECT().SetSlot(0).Return().Twice()
		csMock.EXPECT().SetSlot(1).Return().Once()
		csMock.EXPECT().TextureView().Return(view0).Twice()
		csMock.EXPECT().TextureView().Return(view1).Twice()
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()

		bgpMock.EXPECT().SetSlot(0).Return().Once()
		bgpMock.EXPECT().SetSlot(1).Return().Once()
		bgpMock.EXPECT().SetSlot(0).Return().Once()
		bgpMock.EXPECT().SetTextureView(5, view0).Once()
		bgpMock.EXPECT().SetSampler(6, sampler).Once()
		bgpMock.EXPECT().SetTextureView(5, view1).Once()
		bgpMock.EXPECT().SetSampler(6, sampler).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()

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
		})).Return(nil).Twice()
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
		})).Return(nil).Twice()
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
		})).Return(nil).Twice()
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
		})).Return(nil).Twice()
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
		})).Return(nil).Twice()
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
		csMock.EXPECT().Enabled().Return(true).Twice()
		csMock.EXPECT().LinearSampler().Return(&wgpu.Sampler{}).Times(4)
		csMock.EXPECT().SetSlot(0).Return().Twice()
		csMock.EXPECT().SetSlot(1).Return().Once()
		csMock.EXPECT().TextureView().Return(&wgpu.TextureView{}).Times(4)
		lhMock.EXPECT().ContactShadowHandler().Return(csMock).Once()
		bgpMock.EXPECT().SetTextureView(5, mock.Anything).Twice()
		bgpMock.EXPECT().SetSampler(6, mock.Anything).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.MatchedBy(func(m map[int]uint64) bool {
			return len(m) == 2
		})).Return(nil).Twice()
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
		cullBGPMock.EXPECT().SetBuffer(2, mock.Anything).Maybe()
		cullBGPMock.EXPECT().SetBuffer(3, mock.Anything).Maybe()

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
		lightsBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		cullBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		tileBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()

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

	suite.Run("InitBindGroup cullBGP slot 1 error panics", func() {
		lhMock, _, cullBGPMock, _, cullShaderMock, litShaderMock := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("slot1 err")).Once()
		suite.Panics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("RegisterPipelines error panics", func() {
		lhMock, _, cullBGPMock, _, cullShaderMock, litShaderMock := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.Panics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("InitBindGroup for tileBGP error panics", func() {
		lhMock, _, cullBGPMock, tileBGPMock, cullShaderMock, litShaderMock := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("tile err")).Once()
		suite.Panics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("InitBindGroup tileBGP slot 1 error panics", func() {
		lhMock, _, cullBGPMock, tileBGPMock, cullShaderMock, litShaderMock := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("slot1 err")).Once()
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
		cullBGPMock.EXPECT().SetBuffer(2, mock.Anything).Maybe()
		cullBGPMock.EXPECT().SetBuffer(3, mock.Anything).Maybe()

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
		tileBGPMock.EXPECT().SetBuffer(1, mock.Anything).Maybe()
		lightsBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		cullBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		tileBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()

		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
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
		cullBGPMock.EXPECT().SetBuffer(2, mock.Anything).Maybe()
		cullBGPMock.EXPECT().SetBuffer(3, mock.Anything).Maybe()

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
		tileBGPMock.EXPECT().SetBuffer(2, mock.Anything).Maybe()
		lightsBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		cullBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		tileBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()

		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})

	suite.Run("full happy path both Buffer(2) and Buffer(3) nil", func() {
		lhMock, _, cullBGPMock, tileBGPMock, cullShaderMock, litShaderMock := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.initLightCullResources(cullShaderMock, litShaderMock, 800, 600) })
	})
}

func (suite *sceneImplTest) TestInitSSR() {
	makeReadyHandlers := func() (
		*light_mocks.MockLightingHandler,
		*ssr_mocks.MockHandler,
		*gbuffer_mocks.MockGBufferHandler,
		*composition_mocks.MockHandler,
	) {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		compMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.ssrHandler = ssrMock
		suite.scene.gBufferHandler = gbMock
		suite.scene.compositionHandler = compMock
		gbMock.EXPECT().Enabled().Return(true).Maybe()
		compMock.EXPECT().Enabled().Return(true).Maybe()
		return lhMock, ssrMock, gbMock, compMock
	}

	makeFullBase := func(mipCount int) (
		*light_mocks.MockLightingHandler,
		*ssr_mocks.MockHandler,
		*gbuffer_mocks.MockGBufferHandler,
		*composition_mocks.MockHandler,
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
			Return(&wgpu.TextureView{}, &wgpu.Texture{}).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().
			Return(&wgpu.Sampler{}).Maybe()
		suite.rendererMock.EXPECT().CreateHiZTextures(mock.Anything, mock.Anything).
			Return(&wgpu.TextureView{}, &wgpu.Texture{}, mipReadViews, mipStorageViews, mipCount).Maybe()
		ssrMock.EXPECT().SetSSRTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetSSRTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipCount(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipReadViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZStorageViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMaxTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMaxTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMaxMipReadViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMaxStorageViews(mock.Anything).Maybe()
		gbMock.EXPECT().DepthTextureView().Return(&wgpu.TextureView{}).Maybe()
		gbMock.EXPECT().NormalTextureView().Return(&wgpu.TextureView{}).Maybe()
		compMock.EXPECT().HDRTextureView().Return(&wgpu.TextureView{}).Maybe()
		gbMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(nil).Maybe()
		return lhMock, ssrMock, gbMock, compMock
	}

	suite.Run("ssrHandler nil returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.gBufferHandler = gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.compositionHandler = composition_mocks.NewMockHandler(suite.T())
		suite.scene.ssrHandler = nil
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("gbHandler nil returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		suite.scene.ssrHandler = ssrMock
		suite.scene.compositionHandler = composition_mocks.NewMockHandler(suite.T())
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("compHandler nil returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.ssrHandler = ssrMock
		suite.scene.gBufferHandler = gbMock
		suite.scene.compositionHandler = nil
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("gbHandler not enabled returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		compMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.ssrHandler = ssrMock
		suite.scene.gBufferHandler = gbMock
		suite.scene.compositionHandler = compMock
		gbMock.EXPECT().Enabled().Return(false).Maybe()
		compMock.EXPECT().Enabled().Return(true).Maybe()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("compHandler not enabled returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		compMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.ssrHandler = ssrMock
		suite.scene.gBufferHandler = gbMock
		suite.scene.compositionHandler = compMock
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

	suite.Run("InitBindGroup hiz_init_max error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(1)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("init_max err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("RegisterPipelines hizDownMax error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(1)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("max down err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("InitBindGroup hiz_down_max mip loop error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(2)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample_max", "hiz_downsample_max").Once()
		// MIN mip loop i=1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_1", mock.Anything).Once()
		// MAX mip loop i=1 FAILS (line 1490)
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("max mip err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("InitBindGroup hiz_init_1 error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(1)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample_max", "hiz_downsample_max").Once()
		// slot-0 mip loops skipped (mipCount=1)
		// hizInitBGP1 FAILS (line 1502)
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("init_1 err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("InitBindGroup hiz_init_max_1 error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(1)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample_max", "hiz_downsample_max").Once()
		// slot-0 mip loops skipped (mipCount=1)
		// hizInitBGP1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_1", mock.Anything).Once()
		// hizInitMaxBGP1 FAILS (line 1511)
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("init_max_1 err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("InitBindGroup hiz_down_1_1 mip loop error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(2)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample_max", "hiz_downsample_max").Once()
		// slot-0 MIN mip loop i=1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_1", mock.Anything).Once()
		// slot-0 MAX mip loop i=1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_max_1", mock.Anything).Once()
		// hizInitBGP1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_1", mock.Anything).Once()
		// hizInitMaxBGP1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max_1", mock.Anything).Once()
		// slot-1 MIN mip loop i=1 FAILS (line 1523)
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("down_1_1 err")).Once()
		suite.Panics(func() { suite.scene.initSSR() })
	})

	suite.Run("InitBindGroup hiz_down_max_1_1 mip loop error panics", func() {
		lhMock, ssrMock, _, _ := makeFullBase(2)
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_init", "hiz_init").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample_max", "hiz_downsample_max").Once()
		// slot-0 MIN mip loop i=1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_1", mock.Anything).Once()
		// slot-0 MAX mip loop i=1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_max_1", mock.Anything).Once()
		// hizInitBGP1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_1", mock.Anything).Once()
		// hizInitMaxBGP1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max_1", mock.Anything).Once()
		// slot-1 MIN mip loop i=1 succeeds
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_1_1", mock.Anything).Once()
		// slot-1 MAX mip loop i=1 FAILS (line 1535)
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("down_max_1_1 err")).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample_max", "hiz_downsample_max").Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample_max", "hiz_downsample_max").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_1", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max_1", mock.Anything).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample_max", "hiz_downsample_max").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_1", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max_1", mock.Anything).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample_max", "hiz_downsample_max").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_1", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max_1", mock.Anything).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max", mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample", "hiz_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		ssrMock.EXPECT().SetPipelineKey("hiz_downsample_max", "hiz_downsample_max").Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_1", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_max_1", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_1", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_init_max_1", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_1_1", mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		ssrMock.EXPECT().SetBgp("hiz_down_max_1_1", mock.Anything).Once()
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
		*composition_mocks.MockHandler,
		*ssr_mocks.MockHandler,
		*bgp_mocks.MockBindGroupProvider,
	) {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())

		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.buildInjectionMap()

		suite.scene.compositionHandler = chMock
		suite.scene.ssrHandler = ssrMock

		suite.rendererMock.EXPECT().SampleCount().Return(uint32(1)).Maybe()
		suite.rendererMock.EXPECT().CreateCompositionTextures(800, 600, uint32(1)).
			Return(nil, nil, nil, nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().SetRenderTargetFormat(wgpu.TextureFormatRGBA16Float).Maybe()

		chMock.EXPECT().SetHDRTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetHDRTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetMSAATexture(mock.Anything).Maybe()
		chMock.EXPECT().SetMSAATextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetSlot(mock.Anything).Maybe()

		return lhMock, chMock, ssrMock, bgpMock
	}

	makeFullBase := func() (
		*light_mocks.MockLightingHandler,
		*composition_mocks.MockHandler,
		*ssr_mocks.MockHandler,
		*bgp_mocks.MockBindGroupProvider,
	) {
		lhMock, chMock, ssrMock, bgpMock := makeBase()

		suite.rendererMock.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}).Maybe()
		chMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterCompositionPipeline(mock.Anything).Return(nil).Maybe()
		chMock.EXPECT().SetPipelineKey("composition", "composition").Maybe()
		chMock.EXPECT().Bgp("composition").Return(bgpMock).Maybe()
		bgpMock.EXPECT().SetTextureView(0, mock.Anything).Maybe()
		bgpMock.EXPECT().SetSampler(1, mock.Anything).Maybe()
		bgpMock.EXPECT().SetSampler(3, mock.Anything).Maybe()
		bgpMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()

		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return((*wgpu.Buffer)(nil)).Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		chMock.EXPECT().Exposure().Return(float32(1.0)).Maybe()
		chMock.EXPECT().SetExposureBuffer(mock.Anything).Maybe()
		chMock.EXPECT().HDRTextureView().Return(nil).Maybe()
		lumBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lumBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		lumBGPMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()
		chMock.EXPECT().Bgp("luminance_compute").Return(lumBGPMock).Maybe()
		lumBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(lumBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		chMock.EXPECT().BloomEnabled().Return(false).Maybe()
		suite.rendererMock.EXPECT().InitTextureView(bgpMock, 6, mock.Anything).Return(nil).Maybe()

		return lhMock, chMock, ssrMock, bgpMock
	}

	suite.Run("nil CompositionHandler returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initComposition() })
	})

	suite.Run("zero screenWidth returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.compositionHandler = chMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 600
		suite.NotPanics(func() { suite.scene.initComposition() })
	})

	suite.Run("zero screenHeight returns early", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.compositionHandler = chMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 0
		suite.NotPanics(func() { suite.scene.initComposition() })
	})

	suite.Run("RegisterCompositionPipeline error panics", func() {
		lhMock, chMock, _, _ := makeBase()
		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}).Once()
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
		compMock *composition_mocks.MockHandler,
		ssrMock *ssr_mocks.MockHandler,
		camMock *camera_mocks.MockCamera,
		camBGPMock *bgp_mocks.MockBindGroupProvider,
	) {
		lhMock = light_mocks.NewMockLightingHandler(suite.T())
		shMock = light_mocks.NewMockShadowHandler(suite.T())
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		compMock = composition_mocks.NewMockHandler(suite.T())
		ssrMock = ssr_mocks.NewMockHandler(suite.T())
		camMock = camera_mocks.NewMockCamera(suite.T())
		camBGPMock = bgp_mocks.NewMockBindGroupProvider(suite.T())

		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		shMock.EXPECT().ShadowMapResolution().Return(1024).Maybe()
		shMock.EXPECT().CascadeCount().Return(2).Maybe()
		shMock.EXPECT().SetCSMAtlasTexture(mock.Anything).Maybe()
		shMock.EXPECT().SetCSMAtlasTextureView(mock.Anything).Maybe()
		shMock.EXPECT().SetComparisonSampler(mock.Anything).Maybe()
		shMock.EXPECT().SetBgp(mock.Anything, mock.Anything).Maybe()
		shBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		shMock.EXPECT().Bgp(mock.Anything).Return(shBGPMock).Maybe()
		shMock.EXPECT().LightShadowTileSize().Return(256).Maybe()
		shMock.EXPECT().SetLightShadowAtlasSlots(mock.Anything).Maybe()
		shMock.EXPECT().SetLightShadowAtlasCols(mock.Anything).Maybe()
		shMock.EXPECT().SetLightShadowAtlas(mock.Anything).Maybe()
		shMock.EXPECT().SetLightShadowAtlasView(mock.Anything).Maybe()
		shMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		shMock.EXPECT().CSMAtlasTextureView().Return(nil).Maybe()
		shMock.EXPECT().ComparisonSampler().Return(nil).Maybe()
		shMock.EXPECT().PCFSamples().Return(uint32(16)).Maybe()
		shMock.EXPECT().PCFSamplesSpot().Return(uint32(8)).Maybe()
		shMock.EXPECT().LightShadowAtlasView().Return(nil).Maybe()
		shMock.EXPECT().LightShadowAtlasSlots().Return(6).Maybe()

		csBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		csBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		csBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
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
		csMock.EXPECT().SetSlot(mock.Anything).Maybe()

		suite.scene.gBufferHandler = gbMock
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
		gbMock.EXPECT().AlbedoTextureView().Return(nil).Maybe()
		gbMock.EXPECT().SetSlot(mock.Anything).Maybe()

		ssaoBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssaoBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		ssaoBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		suite.scene.ssaoHandler = ssaoMock
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
		ssaoMock.EXPECT().SetSlot(mock.Anything).Maybe()

		lightsBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lightsBGPMock.EXPECT().Buffer(mock.Anything).Return(nil).Maybe()
		lightsBGPMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()
		lightsBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		lhMock.EXPECT().Bgp("lights").Return(lightsBGPMock).Maybe()
		lightCullBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lightCullBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		lhMock.EXPECT().Bgp("light_cull").Return(lightCullBGPMock).Maybe()
		tileLitBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		tileLitBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		lhMock.EXPECT().Bgp("tile_lit").Return(tileLitBGPMock).Maybe()
		ssaoLitBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssaoLitBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		lhMock.EXPECT().Bgp("ssao_lit").Return(ssaoLitBGPMock).Maybe()
		lhMock.EXPECT().MaxGPULights().Return(1).Maybe()
		lhMock.EXPECT().Resize(mock.Anything, mock.Anything).Maybe()
		lhMock.EXPECT().TileCountX().Return(4).Maybe()
		lhMock.EXPECT().TileCountY().Return(4).Maybe()
		lhMock.EXPECT().TileSize().Return(16).Maybe()
		lhMock.EXPECT().MaxLightsPerTile().Return(32).Maybe()
		lhMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()

		suite.scene.compositionHandler = compMock
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
		compMock.EXPECT().SetSlot(mock.Anything).Maybe()

		ssrInternalBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssrInternalBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		ssrInternalBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		suite.scene.ssrHandler = ssrMock
		ssrMock.EXPECT().SetSSRTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetSSRTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipCount(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipReadViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZStorageViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMaxTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMaxTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMaxMipReadViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMaxStorageViews(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		ssrMock.EXPECT().Bgp(mock.Anything).Return(ssrInternalBGPMock).Maybe()
		ssrMock.EXPECT().Resize(mock.Anything, mock.Anything).Maybe()
		ssrMock.EXPECT().SetEnabled(mock.Anything).Maybe()
		ssrMock.EXPECT().SetBgp(mock.Anything, mock.Anything).Maybe()
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()

		camMock.EXPECT().BindGroupProvider().Return(camBGPMock).Maybe()
		camBGPMock.EXPECT().SetBindGroupLayout(mock.Anything).Maybe()
		camBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()

		suite.rendererMock.EXPECT().MaxTextureDimension2D().Return(uint32(0)).Maybe()
		suite.rendererMock.EXPECT().CreateShadowDepthTexture(mock.Anything, mock.Anything).Return(nil, nil).Maybe()
		suite.rendererMock.EXPECT().CreateComparisonSampler().Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterShadowDepthPipeline(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateGBufferTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateSSAOTextures(mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateContactShadowTextures(mock.Anything, mock.Anything).Return(nil, nil).Maybe()
		suite.rendererMock.EXPECT().SampleCount().Return(uint32(1)).Maybe()
		suite.rendererMock.EXPECT().CreateCompositionTextures(mock.Anything, mock.Anything, mock.Anything).Return(nil, nil, nil, nil, nil, nil).Maybe()
		suite.rendererMock.EXPECT().SetRenderTargetFormat(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateSSRTextures(mock.Anything, mock.Anything).Return(nil, nil).Maybe()
		suite.rendererMock.EXPECT().CreateHiZTextures(mock.Anything, mock.Anything).Return(nil, nil, make([]*wgpu.TextureView, 1), make([]*wgpu.TextureView, 1), 1).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterCompositionPipeline(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return((*wgpu.Buffer)(nil)).Maybe()
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

	mkInitCompBGP := func(compMock *composition_mocks.MockHandler, ssrMock *ssr_mocks.MockHandler) *bgp_mocks.MockBindGroupProvider {
		bgpMockComp := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpMockComp.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		bgpMockComp.EXPECT().SetSampler(mock.Anything, mock.Anything).Maybe()
		bgpMockComp.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()
		bgpMockComp.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		compMock.EXPECT().Bgp("composition").Return(bgpMockComp).Once()
		ssrMock.EXPECT().SSRTextureView().Return(nil).Once()
		lumBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lumBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		lumBGPMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()
		lumBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		compMock.EXPECT().Bgp("luminance_compute").Return(lumBGPMock).Maybe()
		return bgpMockComp
	}

	suite.Run("SSR disabled skips re-bind SetEnabled called", func() {
		lhMock, _, compMock, ssrMock, camMock, _ := makeMocks()
		setupSceneFields(lhMock, camMock)
		suite.scene.taaHandler = nil
		mkInitCompBGP(compMock, ssrMock)
		ssrMock.EXPECT().Enabled().Return(false).Once()
		lhMock.EXPECT().SetEnabled(true).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.NotPanics(func() { suite.scene.initLighting(800, 600) })
	})

	suite.Run("SSR comp enabled Bgp nil inner block skipped SetEnabled called", func() {
		lhMock, _, compMock, ssrMock, camMock, _ := makeMocks()
		setupSceneFields(lhMock, camMock)
		suite.scene.taaHandler = nil
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
		suite.scene.taaHandler = nil
		mkInitCompBGP(compMock, ssrMock)
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		compBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
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
		compBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
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
		suite.scene.taaHandler = nil
		mkInitCompBGP(compMock, ssrMock)
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		compBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
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

	suite.Run("full happy path completes", func() {
		makeBase()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).Return((*wgpu.Buffer)(nil)).Once()
		suite.NotPanics(func() { suite.scene.initPhysics() })
	})
}

func (suite *sceneImplTest) TestInitPhysicsSyncGroup() {
	makeBase := func() (phMock *physics_mocks.MockPhysics, animMock *animator_mocks.MockAnimator, animBGPMock *bgp_mocks.MockBindGroupProvider) {
		phMock = physics_mocks.NewMockPhysics(suite.T())
		buffersBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		animMock = animator_mocks.NewMockAnimator(suite.T())
		animBGPMock = bgp_mocks.NewMockBindGroupProvider(suite.T())
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
		return phMock, animMock, animBGPMock
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
		animBGPMock.EXPECT().Buffer(mock.Anything).Return((*wgpu.Buffer)(nil)).Twice()
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(1).Once()
		animBGPMock.EXPECT().SetSlot(0).Return().Once()
		animBGPMock.EXPECT().SetSlot(1).Return().Twice()
		suite.rendererMock.EXPECT().Pipeline("sync_pipeline_key").Return(pipeMock).Maybe()
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(syncShaderMock).Maybe()
		syncShaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Once()
		suite.NotPanics(func() { suite.scene.initPhysicsSyncGroup(animMock) })
		suite.GreaterOrEqual(suite.scene.physicsAnimBinding, 0)
	})

	suite.Run("InitBindGroup error panics", func() {
		_, animMock, animBGPMock := makeBase()
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(1).Once()
		animBGPMock.EXPECT().SetSlot(0).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("init err")).Once()
		suite.Panics(func() { suite.scene.initPhysicsSyncGroup(animMock) })
	})

	suite.Run("second InitBindGroup error panics", func() {
		_, animMock, animBGPMock := makeBase()
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(1).Once()
		animBGPMock.EXPECT().SetSlot(0).Return().Once()
		animBGPMock.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("slot1 init err")).Once()
		suite.Panics(func() { suite.scene.initPhysicsSyncGroup(animMock) })
	})

	suite.Run("physicsSyncGroup nil initialized on first call", func() {
		_, animMock, animBGPMock := makeBase()
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(1).Once()
		animBGPMock.EXPECT().SetSlot(0).Return().Once()
		animBGPMock.EXPECT().SetSlot(1).Return().Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Once()
		id := suite.scene.initPhysicsSyncGroup(animMock)
		suite.Equal(0, id)
		suite.NotNil(suite.scene.physicsSyncGroup)
	})

	suite.Run("second call increments groupID", func() {
		_, animMock, animBGPMock := makeBase()
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: bgp_mocks.NewMockBindGroupProvider(suite.T())}
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(1).Once()
		animBGPMock.EXPECT().SetSlot(0).Return().Once()
		animBGPMock.EXPECT().SetSlot(1).Return().Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Once()
		id := suite.scene.initPhysicsSyncGroup(animMock)
		suite.Equal(1, id)
	})

	suite.Run("full happy path returns 0 and populates maps", func() {
		_, animMock, animBGPMock := makeBase()
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(1).Once()
		animBGPMock.EXPECT().SetSlot(0).Return().Once()
		animBGPMock.EXPECT().SetSlot(1).Return().Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
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

	suite.Run("InitBindGroup slot 1 error panics", func() {
		shaderMock, camMock, bgpMock := makeBase()
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{cameraDecl}).Once()
		camMock.EXPECT().BindGroupProvider().Return(bgpMock).Once()
		shaderMock.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		bgpMock.EXPECT().SetBindGroupLayout((*wgpu.BindGroupLayout)(nil)).Once()
		bgpMock.EXPECT().SetSlot(1).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("slot1 err")).Once()
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
		bgpMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.NotPanics(func() { suite.scene.reinitCameraBGPForLitPipeline(shaderMock) })
	})
}

func (suite *sceneImplTest) TestInitSimpleShadowAnimationProvider() {
	suite.Run("nil animator returns early", func() {
		suite.NotPanics(func() { suite.scene.initSimpleShadowAnimationProvider(nil, 0) })
	})

	suite.Run("negative animationDataBinding returns early", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		suite.NotPanics(func() { suite.scene.initSimpleShadowAnimationProvider(animMock, -1) })
	})

	suite.Run("nil compute provider returns early", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().ComputeBindGroupProvider().Return(nil).Once()
		suite.NotPanics(func() { suite.scene.initSimpleShadowAnimationProvider(animMock, 0) })
	})

	suite.Run("missing slot 0 animation buffer panics", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())

		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		animMock.EXPECT().Model().Return(nil).Once()
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(1).Once()
		computeBGP.EXPECT().SetSlot(0).Return().Once()
		computeBGP.EXPECT().Buffer(0).Return(nil).Once()
		computeBGP.EXPECT().SetSlot(1).Return().Once()

		suite.Panics(func() { suite.scene.initSimpleShadowAnimationProvider(animMock, 0) })
	})

	suite.Run("missing slot 1 animation buffer panics", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		slot0Buffer := &wgpu.Buffer{}

		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		animMock.EXPECT().Model().Return(nil).Once()
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(0).Once()
		computeBGP.EXPECT().SetSlot(0).Return().Twice()
		computeBGP.EXPECT().Buffer(0).Return(slot0Buffer).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		computeBGP.EXPECT().SetSlot(1).Return().Once()
		computeBGP.EXPECT().Buffer(0).Return(nil).Once()

		suite.Panics(func() { suite.scene.initSimpleShadowAnimationProvider(animMock, 0) })
	})

	suite.Run("success path replaces existing provider and releases old provider", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		slot0Buffer := &wgpu.Buffer{}
		slot1Buffer := &wgpu.Buffer{}

		oldProvider := bgp_mocks.NewMockBindGroupProvider(suite.T())
		oldProvider.EXPECT().SetSlot(0).Return().Once()
		oldProvider.EXPECT().SetBuffers(mock.Anything).Return().Once()
		oldProvider.EXPECT().SetSlot(1).Return().Once()
		oldProvider.EXPECT().SetBuffers(mock.Anything).Return().Once()
		oldProvider.EXPECT().Release().Once()

		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{animMock: oldProvider}

		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		animMock.EXPECT().Model().Return(nil).Once()
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(1).Once()
		computeBGP.EXPECT().SetSlot(0).Return().Once()
		computeBGP.EXPECT().Buffer(0).Return(slot0Buffer).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		computeBGP.EXPECT().SetSlot(1).Return().Twice()
		computeBGP.EXPECT().Buffer(0).Return(slot1Buffer).Once()

		suite.NotPanics(func() { suite.scene.initSimpleShadowAnimationProvider(animMock, 0) })

		newProvider := suite.scene.shadowAnimationProviders[animMock]
		suite.NotNil(newProvider)
		suite.NotEqual(oldProvider, newProvider)
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
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		anim := startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
		suite.NotNil(anim)
	})

	suite.Run("mesh provider with nil vertex buffer calls InitMeshBuffers", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpMock.EXPECT().VertexBuffer().Return(nil).Once()
		mdl.EXPECT().MeshProvider().Return(bgpMock).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("mesh provider with non-nil vertex buffer skips InitMeshBuffers", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpMock.EXPECT().VertexBuffer().Return(new(wgpu.Buffer)).Once()
		mdl.EXPECT().MeshProvider().Return(bgpMock).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("InitMeshBuffers error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = vs
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpMock.EXPECT().VertexBuffer().Return(nil).Once()
		mdl.EXPECT().MeshProvider().Return(bgpMock).Once()
		mdl.EXPECT().VertexData().Return(nil).Once()
		mdl.EXPECT().IndexData().Return(nil).Once()
		mdl.EXPECT().IndexCount().Return(0).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		suite.rendererMock.EXPECT().InitMeshBuffers(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("mesh err")).Once()
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("AnnotationArgAnimationData sets compute group", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("AnnotationArgSkeletalAnimationData with array< prefix sets compute group", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("compute group decl with nil Group is skipped", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().LODCount().Return(1).Maybe()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().RenderMaterials().Return(nil).Maybe()
		b := 1
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   nil,
			Binding: &b,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgAnimationData},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		cs.EXPECT().Key().Return("ck").Maybe()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(0).Maybe()
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("compute group decl non-BindingGroup type is skipped", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("output group BindingGroup InstanceData array< prefix sets group and binding", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("output group BindingGroup InstanceData with nil binding skips binding update", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("output group set from Provider Animator", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("output decl with nil Group is skipped", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("output group Provider with non-Animator arg does not set outputGroup", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("matching output descriptor entry overrides perInstanceOutputSize", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("output descriptor entry zero MinBindingSize does not override perInstanceOutputSize", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("typed compute entries IndirectArgs BoneInfo ModelData AnimGlobals GlobalData default storage", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("typed BoneInfo ModelData with zero MinBindingSize no size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("default ReadOnlyStorage binding with MinBindingSize adds size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("default non-storage binding with MinBindingSize no size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("compute binding types loop skips Provider decls", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("compute binding types loop skips BindingGroup decls with nil Binding", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("raw output binding sets computeOutputBinding and size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("raw packed binding adds size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("raw scratch binding adds size override", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("raw provider loop skips non-Provider decls", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Maybe()
		mdl.EXPECT().LODCount().Return(1).Maybe()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Maybe()
		mdl.EXPECT().RenderMaterials().Return(nil).Maybe()
		b := 2
		g := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &g,
			Binding: &b,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgAnimationData},
		}
		cs.EXPECT().Declarations().Return([]shader.Annotation{decl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		cs.EXPECT().Key().Return("ck").Maybe()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CurrentFrameSlot().Return(0).Maybe()
		suite.scene.createAnimator(mdl, cs, vs, fs)
	})

	suite.Run("raw provider loop skips Provider decls with nil Binding", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("output size override for matching outputInstanceBinding storage entry", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("InitBindGroup compute error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("compute bg err")).Once()
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("InitBindGroup output error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("output bg err")).Once()
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("RegisterPipelines compute error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("cp err")).Once()
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("RegisterPipelines render error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("rp err")).Once()
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("material with non-nil BGP is skipped", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("material with empty pipeline key uses fragmentShader for initMaterialGPU", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("material pipeline key nil at Pipeline registers new pipeline second nil uses fragmentShader", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(3)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("material PipelineOptions non-empty non-PipelineBuilderOption items filtered", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(3)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("material PipelineOptions with valid PipelineBuilderOption is included", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(3)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("material already registered Pipeline skips registration and uses Shader for frag", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("material pipeline Shader returns nil falls back to fragmentShader", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("material RegisterPipelines error panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("mat pipe err")).Once()
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("initMaterialGPU error panics", func() {
		mdl, cs, vs, fs := makeBase()
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("mat bg err")).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("skeletal backend nil skeleton selects skeletal type", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		anim := startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
		suite.NotNil(anim)
	})

	suite.Run("skeletal binding discovery nil Binding skips decl", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("skeletal binding discovery BoneInfo and AnimatorPacked both found", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("skeletal binding discovery array< prefix stripped for BoneInfo", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("skeletal binding discovery Provider non-AnimatorPacked arg does not set packedBinding", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(true).Maybe()
		mdl.EXPECT().Skeleton().Return(nil).Times(2)
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
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
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		anim := startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
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
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
	})

	suite.Run("compute output buffer shared to output BGP when annotation present", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
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
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)

		anim := startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
		suite.NotNil(anim)
	})

	suite.Run("shadowIndirectBuffers populated", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		mockBuf := &wgpu.Buffer{}
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(mockBuf).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)

		anim := startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs))
		suite.NotNil(anim)
		suite.Contains(suite.scene.shadowIndirectBuffers, anim)
		suite.NotNil(suite.scene.shadowIndirectBuffers[anim])
	})
}

func (suite *sceneImplTest) TestAcquireCompositionFrame() {
	suite.Run("nil lifecycle returns nil and does not begin composition frame", func() {
		suite.scene.lc = nil
		suite.NoError(suite.scene.AcquireCompositionFrame())
		suite.rendererMock.AssertNotCalled(suite.T(), "BeginCompositionFrame")
	})

	suite.Run("nil renderer returns nil", func() {
		suite.scene.lc = lifecycle.NewLifecycle()
		suite.scene.r = nil
		suite.NoError(suite.scene.AcquireCompositionFrame())
	})

	suite.Run("non-running scene returns nil", func() {
		suite.scene.lc = lifecycle.NewLifecycle()
		suite.NoError(suite.scene.AcquireCompositionFrame())
		suite.rendererMock.AssertNotCalled(suite.T(), "BeginCompositionFrame")
	})

	suite.Run("BeginCompositionFrame error propagated", func() {
		suite.scene.lc = lifecycle.NewLifecycle()
		suite.NoError(suite.scene.lc.SetState(lifecycle.LifecycleStateStarting))
		suite.NoError(suite.scene.lc.SetState(lifecycle.LifecycleStateRunning))
		suite.rendererMock.EXPECT().BeginCompositionFrame().Return(errors.New("fail")).Once()
		suite.Error(suite.scene.AcquireCompositionFrame())
	})

	suite.Run("BeginCompositionFrame success", func() {
		suite.scene.lc = lifecycle.NewLifecycle()
		suite.NoError(suite.scene.lc.SetState(lifecycle.LifecycleStateStarting))
		suite.NoError(suite.scene.lc.SetState(lifecycle.LifecycleStateRunning))
		suite.rendererMock.EXPECT().BeginCompositionFrame().Return(nil).Once()
		suite.NoError(suite.scene.AcquireCompositionFrame())
	})
}

func (suite *sceneImplTest) TestTransitionChildLifecycle() {
	suite.Run("nil lifecycle returns nil", func() {
		suite.NoError(transitionChildLifecycle(nil, lifecycle.LifecycleStateRunning))
	})

	suite.Run("running target transitions paused to running", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStatePaused))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateRunning))
		suite.Equal(lifecycle.LifecycleStateRunning, lc.State())
	})

	suite.Run("running target leaves already running lifecycle unchanged", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateRunning))
		suite.Equal(lifecycle.LifecycleStateRunning, lc.State())
	})

	suite.Run("running target transitions draining to running", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateDraining))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateRunning))
		suite.Equal(lifecycle.LifecycleStateRunning, lc.State())
	})

	suite.Run("running target leaves default branch state unchanged", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRegistered))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateRunning))
		suite.Equal(lifecycle.LifecycleStateRegistered, lc.State())
	})

	suite.Run("paused target transitions running to paused", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStatePaused))
		suite.Equal(lifecycle.LifecycleStatePaused, lc.State())
	})

	suite.Run("paused target leaves default branch state unchanged", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRegistered))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStatePaused))
		suite.Equal(lifecycle.LifecycleStateRegistered, lc.State())
	})

	suite.Run("draining target transitions running to draining", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateDraining))
		suite.Equal(lifecycle.LifecycleStateDraining, lc.State())
	})

	suite.Run("draining target transitions errored to draining", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateErrored))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateDraining))
		suite.Equal(lifecycle.LifecycleStateDraining, lc.State())
	})

	suite.Run("draining target leaves already draining lifecycle unchanged", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateDraining))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateDraining))
		suite.Equal(lifecycle.LifecycleStateDraining, lc.State())
	})

	suite.Run("draining target leaves default branch state unchanged", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStatePaused))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateDraining))
		suite.Equal(lifecycle.LifecycleStatePaused, lc.State())
	})

	suite.Run("stopped target transitions registered to stopped", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRegistered))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateStopped))
		suite.Equal(lifecycle.LifecycleStateStopped, lc.State())
	})

	suite.Run("stopped target transitions paused to stopped", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStatePaused))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateStopped))
		suite.Equal(lifecycle.LifecycleStateStopped, lc.State())
	})

	suite.Run("stopped target transitions draining to stopped", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateDraining))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateStopped))
		suite.Equal(lifecycle.LifecycleStateStopped, lc.State())
	})

	suite.Run("stopped target leaves already stopped lifecycle unchanged", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateStopped))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateStopped))
		suite.Equal(lifecycle.LifecycleStateStopped, lc.State())
	})

	suite.Run("stopped target transitions running through draining to stopped", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateStopped))
		suite.Equal(lifecycle.LifecycleStateStopped, lc.State())
	})

	suite.Run("stopped target transitions errored through draining to stopped", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateErrored))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateStopped))
		suite.Equal(lifecycle.LifecycleStateStopped, lc.State())
	})

	suite.Run("stopped target returns error when intermediate draining transition fails", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		lc.OnTransitionTo(lifecycle.LifecycleStateDraining, lifecycle.Hook(func() error {
			return errors.New("drain failed")
		}))
		err := transitionChildLifecycle(lc, lifecycle.LifecycleStateStopped)
		suite.Error(err)
		suite.Equal(lifecycle.LifecycleStateDraining, lc.State())
	})

	suite.Run("stopped target leaves default branch state unchanged", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateStarting))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateStopped))
		suite.Equal(lifecycle.LifecycleStateStarting, lc.State())
	})

	suite.Run("removed target transitions stopped to removed", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateStopped))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateRemoved))
		suite.Equal(lifecycle.LifecycleStateRemoved, lc.State())
	})

	suite.Run("removed target transitions running through stop to removed", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateRemoved))
		suite.Equal(lifecycle.LifecycleStateRemoved, lc.State())
	})

	suite.Run("removed target no-ops when already removed", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRemoved))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateRemoved))
		suite.Equal(lifecycle.LifecycleStateRemoved, lc.State())
	})

	suite.Run("removed target leaves default branch state unchanged", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateStarting))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleStateRemoved))
		suite.Equal(lifecycle.LifecycleStateStarting, lc.State())
	})

	suite.Run("removed target returns error when stop transition fails", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		lc.OnTransitionTo(lifecycle.LifecycleStateDraining, lifecycle.Hook(func() error {
			return errors.New("drain failed")
		}))
		err := transitionChildLifecycle(lc, lifecycle.LifecycleStateRemoved)
		suite.Error(err)
		suite.Equal(lifecycle.LifecycleStateDraining, lc.State())
	})

	suite.Run("unknown target returns nil and leaves state unchanged", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		suite.NoError(transitionChildLifecycle(lc, lifecycle.LifecycleState(999)))
		suite.Equal(lifecycle.LifecycleStateRunning, lc.State())
	})
}

func (suite *sceneImplTest) TestSceneLifecycleChildren() {
	suite.Run("returns animators without nil entries and current physics handler", func() {
		animOne := animator_mocks.NewMockAnimator(suite.T())
		animTwo := animator_mocks.NewMockAnimator(suite.T())
		mdlOne := model_mocks.NewMockModel(suite.T())
		mdlTwo := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{
			mdlOne: {animOne, nil},
			mdlTwo: {nil, animTwo},
		}
		ph := physics.NewPhysics()
		suite.scene.physicsHandler = ph

		anims, physicsHandler := suite.scene.sceneLifecycleChildren()

		suite.Len(anims, 2)
		suite.ElementsMatch([]animator.Animator{animOne, animTwo}, anims)
		suite.Equal(ph, physicsHandler)
	})
}

func (suite *sceneImplTest) TestTransitionSceneChildren() {
	suite.Run("transitions animator and physics children to target state", func() {
		animLCOne := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStatePaused))
		animLCTwo := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateDraining))
		animOne := animator_mocks.NewMockAnimator(suite.T())
		animTwo := animator_mocks.NewMockAnimator(suite.T())
		animOne.EXPECT().Lifecycle().Return(animLCOne).Maybe()
		animTwo.EXPECT().Lifecycle().Return(animLCTwo).Maybe()

		phLC := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStatePaused))
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(phLC).Maybe()

		mdlOne := model_mocks.NewMockModel(suite.T())
		mdlTwo := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{
			mdlOne: {animOne},
			mdlTwo: {animTwo},
		}
		suite.scene.physicsHandler = phMock

		err := suite.scene.transitionSceneChildren(lifecycle.LifecycleStateRunning)

		suite.NoError(err)
		suite.Equal(lifecycle.LifecycleStateRunning, animLCOne.State())
		suite.Equal(lifecycle.LifecycleStateRunning, animLCTwo.State())
		suite.Equal(lifecycle.LifecycleStateRunning, phLC.State())
	})

	suite.Run("joins transition errors with animator model names and physics context", func() {
		animNamedLC := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		animNamedLC.OnTransitionTo(lifecycle.LifecycleStateDraining, lifecycle.Hook(func() error {
			return errors.New("anim named failed")
		}))
		namedAnim := animator_mocks.NewMockAnimator(suite.T())
		namedAnim.EXPECT().Lifecycle().Return(animNamedLC).Maybe()
		namedModel := model_mocks.NewMockModel(suite.T())
		namedModel.EXPECT().Name().Return("hero").Maybe()
		namedAnim.EXPECT().Model().Return(namedModel).Maybe()

		animNilLC := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		animNilLC.OnTransitionTo(lifecycle.LifecycleStateDraining, lifecycle.Hook(func() error {
			return errors.New("anim nil failed")
		}))
		nilModelAnim := animator_mocks.NewMockAnimator(suite.T())
		nilModelAnim.EXPECT().Lifecycle().Return(animNilLC).Maybe()
		nilModelAnim.EXPECT().Model().Return(nil).Maybe()

		phLC := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		phLC.OnTransitionTo(lifecycle.LifecycleStateDraining, lifecycle.Hook(func() error {
			return errors.New("physics failed")
		}))
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(phLC).Maybe()

		mdlOne := model_mocks.NewMockModel(suite.T())
		mdlTwo := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{
			mdlOne: {namedAnim},
			mdlTwo: {nilModelAnim},
		}
		suite.scene.physicsHandler = phMock

		err := suite.scene.transitionSceneChildren(lifecycle.LifecycleStateDraining)

		suite.Error(err)
		suite.Contains(err.Error(), `animator "hero" lifecycle transition to 4 failed`)
		suite.Contains(err.Error(), `animator "<nil>" lifecycle transition to 4 failed`)
		suite.Contains(err.Error(), "physics lifecycle transition to 4 failed")
	})

	suite.Run("transitions animators when physics handler is nil", func() {
		animLC := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStatePaused))
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(animLC).Maybe()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{
			model_mocks.NewMockModel(suite.T()): {animMock},
		}
		suite.scene.physicsHandler = nil

		err := suite.scene.transitionSceneChildren(lifecycle.LifecycleStateRunning)

		suite.NoError(err)
		suite.Equal(lifecycle.LifecycleStateRunning, animLC.State())
	})
}

func (suite *sceneImplTest) TestRegisterLifecycleHooks() {
	suite.Run("nil scene lifecycle is a no-op", func() {
		s := newSceneLifecycleHelper(suite.rendererMock)
		s.lc = nil
		suite.NotPanics(func() { s.registerLifecycleHooks() })
	})

	suite.Run("pause resume drain and stop transitions fan out to children", func() {
		animLC := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(animLC).Maybe()

		phLC := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(phLC).Maybe()

		suite.scene.animatorPool = map[model.Model][]animator.Animator{
			model_mocks.NewMockModel(suite.T()): {animMock},
		}
		suite.scene.physicsHandler = phMock
		suite.scene.registerLifecycleHooks()

		suite.NoError(suite.scene.lc.SetState(lifecycle.LifecycleStateStarting))
		suite.NoError(suite.scene.lc.SetState(lifecycle.LifecycleStateRunning))
		suite.NoError(suite.scene.lc.SetState(lifecycle.LifecycleStatePaused))
		suite.NoError(suite.scene.lc.SetState(lifecycle.LifecycleStateRunning))
		suite.NoError(suite.scene.lc.SetState(lifecycle.LifecycleStateDraining))
		suite.NoError(suite.scene.lc.SetState(lifecycle.LifecycleStateStopped))

		suite.Equal(lifecycle.LifecycleStateStopped, animLC.State())
		suite.Equal(lifecycle.LifecycleStateStopped, phLC.State())
	})

	suite.Run("removed transition fans out children and runs cleanup", func() {
		rendererMock := renderer_mocks.NewMockRenderer(suite.T())
		rendererMock.EXPECT().WaitIdle().Return().Once()
		s := newSceneLifecycleHelper(rendererMock)

		animLC := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateStopped))
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(animLC).Maybe()

		phLC := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateStopped))
		phMock := physics_mocks.NewMockPhysics(suite.T())
		phMock.EXPECT().Lifecycle().Return(phLC).Maybe()
		phMock.EXPECT().Bgps().Return(map[string]bind_group_provider.BindGroupProvider{}).Maybe()
		phMock.EXPECT().Buffers().Return(nil).Once()
		phMock.EXPECT().StagingBuffer().Return(nil).Once()

		s.animatorPool = map[model.Model][]animator.Animator{
			model_mocks.NewMockModel(suite.T()): {animMock},
		}
		s.physicsHandler = phMock
		s.registerLifecycleHooks()

		suite.NoError(s.lc.SetState(lifecycle.LifecycleStateStopped))
		suite.NoError(s.lc.SetState(lifecycle.LifecycleStateRemoved))
		suite.Equal(lifecycle.LifecycleStateRemoved, animLC.State())
		suite.Equal(lifecycle.LifecycleStateRemoved, phLC.State())
		suite.Nil(s.physicsHandler)
		suite.Len(s.animatorPool, 0)
	})
}

func (suite *sceneImplTest) TestReleaseSceneResources() {
	suite.Run("resets scene-owned state maps and flags", func() {
		suite.rendererMock.EXPECT().WaitIdle().Return().Once()

		suite.scene.lightObjects = []game_object.GameObject{nil}
		suite.scene.lightShadowEntries = []light.GPULightShadowEntry{{}}
		suite.scene.lightShadowMap = map[light.Light]uint32{light.NewLight(light.LightTypeDirectional): 1}
		suite.scene.lightPrevSlotMap = map[light.Light]uint32{light.NewLight(light.LightTypePoint): 2}
		suite.scene.writePool = []bind_group_provider.BufferWrite{{Binding: 1}}
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{bind_group_provider.NewBindGroupProvider("draw_bg")}
		suite.scene.drawDeclsPool = []shader.Annotation{{Line: 1}}
		suite.scene.drawGroupProvidersPool = map[int]bind_group_provider.BindGroupProvider{1: bind_group_provider.NewBindGroupProvider("draw_provider")}
		suite.scene.drawBindGroupCache = map[drawCacheKey][]bind_group_provider.BindGroupProvider{
			{pipelineKey: "key"}: {bind_group_provider.NewBindGroupProvider("cache_provider")},
		}
		suite.scene.shadowIndirectBuffers = map[animator.Animator]*wgpu.Buffer{nil: nil}
		suite.scene.animIndirectBinding = map[animator.Animator]int{nil: 5}
		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{nil: bind_group_provider.NewBindGroupProvider("shadow_anim")}
		suite.scene.instanceLookup = map[animator.Animator]map[uint32]uint64{nil: map[uint32]uint64{0: 1}}
		suite.scene.lodLevelCache = map[animator.Animator]int{nil: 1}
		suite.scene.injections = map[string]string{"x": "y"}
		suite.scene.drawCacheDirty = false
		suite.scene.postProcessingInitialized = true
		suite.scene.tileBufferCapacity = 17
		suite.scene.animatorPool = map[model.Model][]animator.Animator{model_mocks.NewMockModel(suite.T()): {nil}}
		suite.scene.registry = map[uint64]game_object.GameObject{1: nil}
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{7: nil}
		suite.scene.physicsSyncAnimMap = map[animator.Animator]int{nil: 3}
		suite.scene.physicsHandler = physics.NewPhysics()

		err := suite.scene.releaseSceneResources()

		suite.NoError(err)
		suite.Nil(suite.scene.lightObjects)
		suite.Nil(suite.scene.lightShadowEntries)
		suite.Nil(suite.scene.lightShadowMap)
		suite.Nil(suite.scene.lightPrevSlotMap)
		suite.Nil(suite.scene.writePool)
		suite.Nil(suite.scene.drawBindGroupsPool)
		suite.Nil(suite.scene.drawDeclsPool)
		suite.Nil(suite.scene.drawGroupProvidersPool)
		suite.Nil(suite.scene.drawBindGroupCache)
		suite.Nil(suite.scene.shadowIndirectBuffers)
		suite.Nil(suite.scene.animIndirectBinding)
		suite.Nil(suite.scene.shadowAnimationProviders)
		suite.Nil(suite.scene.instanceLookup)
		suite.Nil(suite.scene.lodLevelCache)
		suite.Nil(suite.scene.injections)
		suite.True(suite.scene.drawCacheDirty)
		suite.False(suite.scene.postProcessingInitialized)
		suite.Equal(0, suite.scene.tileBufferCapacity)
		suite.Nil(suite.scene.physicsHandler)
		suite.NotNil(suite.scene.animatorPool)
		suite.NotNil(suite.scene.registry)
		suite.NotNil(suite.scene.physicsSyncGroup)
		suite.NotNil(suite.scene.physicsSyncAnimMap)
		suite.Len(suite.scene.animatorPool, 0)
		suite.Len(suite.scene.registry, 0)
		suite.Len(suite.scene.physicsSyncGroup, 0)
		suite.Len(suite.scene.physicsSyncAnimMap, 0)
	})
}

func (suite *sceneImplTest) TestReleasePhysicsResources() {
	suite.Run("nil physics handler and nil groups are safely skipped", func() {
		suite.scene.physicsHandler = nil
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: nil}
		suite.scene.boneParticleUpdateGroups = []*boneParticleUpdateGroup{nil, &boneParticleUpdateGroup{bgp: nil}}
		suite.scene.physicsSyncWrites = []bind_group_provider.BufferWrite{{Binding: 0}}

		suite.scene.releasePhysicsResources()

		suite.NotNil(suite.scene.physicsSyncGroup)
		suite.NotNil(suite.scene.physicsSyncAnimMap)
		suite.Len(suite.scene.physicsSyncGroup, 0)
		suite.Len(suite.scene.physicsSyncAnimMap, 0)
		suite.Nil(suite.scene.physicsSyncWrites)
		suite.Nil(suite.scene.boneParticleUpdateGroups)
	})

	suite.Run("releases physics sync and bone update providers", func() {
		syncGroupBGP := bind_group_provider.NewBindGroupProvider("physics_sync_group")
		boneBGP := bind_group_provider.NewBindGroupProvider("bone_update_group")
		ph := physics.NewPhysics()

		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{42: syncGroupBGP}
		suite.scene.boneParticleUpdateGroups = []*boneParticleUpdateGroup{&boneParticleUpdateGroup{bgp: boneBGP}}
		suite.scene.physicsHandler = ph

		suite.scene.releasePhysicsResources()

		suite.Nil(suite.scene.physicsHandler)
		suite.NotNil(suite.scene.physicsSyncGroup)
		suite.NotNil(suite.scene.physicsSyncAnimMap)
		suite.Len(suite.scene.physicsSyncGroup, 0)
		suite.Len(suite.scene.physicsSyncAnimMap, 0)
		for _, bgp := range ph.Bgps() {
			suite.Nil(bgp)
		}
	})
}

func (suite *sceneImplTest) TestReleaseLightingResources() {
	suite.Run("nil lighting handler returns without panic", func() {
		s := newSceneLifecycleHelper(nil)
		s.lightHandler = nil
		suite.NotPanics(func() { s.releaseLightingResources() })
	})

	suite.Run("releases lighting providers and disables handlers", func() {
		suite.scene.lightHandler = light.NewLightingHandler()
		suite.scene.ssaoHandler.SetEnabled(true)
		shadowHandler := suite.scene.lightHandler.ShadowHandler()
		shadowHandler.SetBgp("csm_shadow_lit", bind_group_provider.NewBindGroupProvider("csm_shadow_lit"))
		shadowHandler.SetBgp("spot_shadow", bind_group_provider.NewBindGroupProvider("spot_shadow"))

		suite.scene.releaseLightingResources()

		suite.False(suite.scene.lightHandler.Enabled())
		suite.False(suite.scene.lightHandler.ContactShadowHandler().Enabled())
		suite.Nil(suite.scene.lightHandler.Bgp("lights"))
		suite.Nil(suite.scene.lightHandler.Bgp("light_cull"))
		suite.Nil(suite.scene.lightHandler.Bgp("tile_lit"))
		suite.Nil(suite.scene.lightHandler.Bgp("ssao_lit"))
		suite.Nil(shadowHandler.Bgp("csm_shadow_lit"))
		suite.Nil(shadowHandler.Bgp("spot_shadow"))
	})
}

func (suite *sceneImplTest) TestReleasePostProcessingResources() {
	suite.Run("nil handlers are safely skipped", func() {
		s := &scene{}
		suite.NotPanics(func() { s.releasePostProcessingResources() })
	})

	suite.Run("enabled handlers are disabled and provider references cleared", func() {
		s := newSceneLifecycleHelper(nil)
		s.compositionHandler.SetEnabled(true)
		s.compositionHandler.SetBloomEnabled(true)
		s.ssaoHandler.SetEnabled(true)
		s.ssrHandler.SetEnabled(true)
		s.taaHandler.SetEnabled(true)

		s.releasePostProcessingResources()

		suite.False(s.compositionHandler.Enabled())
		suite.False(s.ssaoHandler.Enabled())
		suite.False(s.ssrHandler.Enabled())
		suite.False(s.taaHandler.Enabled())
		suite.Nil(s.compositionHandler.Bgp("composition"))
		suite.Nil(s.compositionHandler.Bgp("luminance_compute"))
		suite.Nil(s.ssaoHandler.Bgp("ssao_compute"))
		suite.Nil(s.ssrHandler.Bgp("ssr_compute"))
		suite.Nil(s.taaHandler.Bgp("taa_resolve_0"))
		suite.Nil(s.taaHandler.Bgp("taa_resolve_1"))
		suite.Nil(s.taaHandler.Bgp("taa_sharpen_0"))
		suite.Nil(s.taaHandler.Bgp("taa_sharpen_1"))
	})
}

func (suite *sceneImplTest) TestPrepareLuminance() {
	suite.Run("nil light handler returns early", func() {
		suite.scene.lightHandler = nil
		suite.NotPanics(func() { suite.scene.PrepareLuminance(0.016) })
	})

	suite.Run("composition handler disabled returns early", func() {
		suite.NotPanics(func() { suite.scene.PrepareLuminance(0.016) })
	})

	suite.Run("auto exposure disabled returns early", func() {
		suite.scene.compositionHandler.SetEnabled(true)
		suite.NotPanics(func() { suite.scene.PrepareLuminance(0.016) })
	})

	suite.Run("dispatches luminance compute when enabled", func() {
		suite.scene.compositionHandler.SetEnabled(true)
		suite.scene.compositionHandler.SetAutoExposureEnabled(true)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareLuminance(0.016) })
	})
}

func (suite *sceneImplTest) TestPrepareBloom() {
	suite.Run("nil light handler returns early", func() {
		suite.scene.lightHandler = nil
		suite.NotPanics(func() { suite.scene.PrepareBloom() })
	})

	suite.Run("nil renderer returns early", func() {
		suite.scene.r = nil
		suite.NotPanics(func() { suite.scene.PrepareBloom() })
	})

	suite.Run("bloom disabled returns early", func() {
		suite.scene.compositionHandler.SetEnabled(true)
		suite.NotPanics(func() { suite.scene.PrepareBloom() })
	})

	suite.Run("dispatches bloom compute when enabled", func() {
		suite.scene.compositionHandler.SetEnabled(true)
		suite.scene.compositionHandler.SetBloomEnabled(true)
		suite.scene.compositionHandler.SetBloomMipCount(1)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return().Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndComputeFrame().Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareBloom() })
	})

	suite.Run("dispatches bloom compute with mipCount 2 exercises upsample loop", func() {
		suite.scene.compositionHandler.SetEnabled(true)
		suite.scene.compositionHandler.SetBloomEnabled(true)
		suite.scene.compositionHandler.SetBloomMipCount(2)
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return().Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Times(3)
		suite.rendererMock.EXPECT().EndComputeFrame().Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareBloom() })
	})
}

func (suite *sceneImplTest) TestSyncFrameSlot() {
	suite.Run("no-op when lightHandler is nil", func() {
		suite.scene.lightHandler = nil
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(0) })
	})

	suite.Run("calls SetSlot on GBufferHandler", func() {
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		gbhMock.EXPECT().SetSlot(1).Return().Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(nil).Once()
		mockLH.EXPECT().ShadowHandler().Return(nil).Once()
		mockLH.EXPECT().Bgp("lights").Return(nil).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(nil).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(nil).Once()
		suite.scene.lightHandler = mockLH
		suite.scene.gBufferHandler = gbhMock
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("calls SetSlot on SSAOHandler", func() {
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		ssaoMock.EXPECT().SetSlot(1).Return().Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(nil).Once()
		mockLH.EXPECT().ShadowHandler().Return(nil).Once()
		mockLH.EXPECT().Bgp("lights").Return(nil).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(nil).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(nil).Once()
		suite.scene.lightHandler = mockLH
		suite.scene.ssaoHandler = ssaoMock
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("calls SetSlot on ContactShadowHandler", func() {
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().SetSlot(1).Return().Once()
		csMock.EXPECT().Bgp("contact_shadow_compute").Return(nil).Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(csMock).Once()
		mockLH.EXPECT().ShadowHandler().Return(nil).Once()
		mockLH.EXPECT().Bgp("lights").Return(nil).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(nil).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(nil).Once()
		suite.scene.lightHandler = mockLH
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("calls SetSlot on contact shadow compute bind group when present", func() {
		csComputeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		csComputeBGP.EXPECT().SetSlot(1).Return().Once()
		csMock := light_mocks.NewMockContactShadowHandler(suite.T())
		csMock.EXPECT().SetSlot(1).Return().Once()
		csMock.EXPECT().Bgp("contact_shadow_compute").Return(csComputeBGP).Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(csMock).Once()
		mockLH.EXPECT().ShadowHandler().Return(nil).Once()
		mockLH.EXPECT().Bgp("lights").Return(nil).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(nil).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(nil).Once()
		suite.scene.lightHandler = mockLH
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("calls SetSlot on CompositionHandler and its luminance_compute BGP", func() {
		lumBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lumBGP.EXPECT().SetSlot(1).Return().Once()
		chMock := composition_mocks.NewMockHandler(suite.T())
		chMock.EXPECT().SetSlot(1).Return().Once()
		chMock.EXPECT().Bgp("luminance_compute").Return(lumBGP).Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(nil).Once()
		mockLH.EXPECT().ShadowHandler().Return(nil).Once()
		mockLH.EXPECT().Bgp("lights").Return(nil).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(nil).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(nil).Once()
		suite.scene.lightHandler = mockLH
		suite.scene.compositionHandler = chMock
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("calls SetSlot on SSRHandler", func() {
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		ssrMock.EXPECT().SetSlot(1).Return().Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(nil).Once()
		mockLH.EXPECT().ShadowHandler().Return(nil).Once()
		mockLH.EXPECT().Bgp("lights").Return(nil).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(nil).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(nil).Once()
		suite.scene.lightHandler = mockLH
		suite.scene.ssrHandler = ssrMock
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("calls SetSlot on TAAHandler", func() {
		taaMock := taa_mocks.NewMockHandler(suite.T())
		taaMock.EXPECT().SetSlot(1).Return().Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(nil).Once()
		mockLH.EXPECT().ShadowHandler().Return(nil).Once()
		mockLH.EXPECT().Bgp("lights").Return(nil).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(nil).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(nil).Once()
		suite.scene.lightHandler = mockLH
		suite.scene.taaHandler = taaMock
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("calls SetSlot on lightHandler BGPs lights light_cull tile_lit", func() {
		bgpLights := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpLights.EXPECT().SetSlot(0).Return().Once()
		bgpCull := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpCull.EXPECT().SetSlot(0).Return().Once()
		bgpTile := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpTile.EXPECT().SetSlot(0).Return().Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(nil).Once()
		mockLH.EXPECT().ShadowHandler().Return(nil).Once()
		mockLH.EXPECT().Bgp("lights").Return(bgpLights).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(bgpCull).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(bgpTile).Once()
		suite.scene.lightHandler = mockLH
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(0) })
	})

	suite.Run("calls SetSlot on shadow handler cascade and atlas BGPs", func() {
		bgpCSMShadowLit := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpCSMShadowLit.EXPECT().SetSlot(0).Return().Once()
		bgpCSM0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpCSM0.EXPECT().SetSlot(0).Return().Once()
		bgpCSM1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpCSM1.EXPECT().SetSlot(0).Return().Once()
		bgpAtlas0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpAtlas0.EXPECT().SetSlot(0).Return().Once()
		bgpAtlas1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bgpAtlas1.EXPECT().SetSlot(0).Return().Once()
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		shMock.EXPECT().CascadeCount().Return(2).Times(3)
		shMock.EXPECT().LightShadowAtlasSlots().Return(2).Times(3)
		shMock.EXPECT().Bgp("csm_shadow_lit").Return(bgpCSMShadowLit).Once()
		shMock.EXPECT().Bgp("csm_data_0").Return(bgpCSM0).Once()
		shMock.EXPECT().Bgp("csm_data_1").Return(bgpCSM1).Once()
		shMock.EXPECT().Bgp("spot_shadow_0").Return(bgpAtlas0).Once()
		shMock.EXPECT().Bgp("spot_shadow_1").Return(bgpAtlas1).Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(nil).Once()
		mockLH.EXPECT().ShadowHandler().Return(shMock).Once()
		mockLH.EXPECT().Bgp("lights").Return(nil).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(nil).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(nil).Once()
		suite.scene.lightHandler = mockLH
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(0) })
	})

	suite.Run("calls SetSlot on camera BindGroupProvider", func() {
		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camBGP.EXPECT().SetSlot(1).Return().Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Once()
		suite.scene.cam = camMock
		suite.scene.lightHandler = nil
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("calls SetSlot on animator BGPs", func() {
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		computeBGP.EXPECT().SetSlot(0).Return().Once()
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		outputBGP.EXPECT().SetSlot(0).Return().Once()
		hizBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		hizBGP.EXPECT().SetSlot(0).Return().Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		animMock.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		animMock.EXPECT().HiZBindGroupProvider().Return(hizBGP).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.scene.lightHandler = nil
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(0) })
	})

	suite.Run("calls SetSlot on shadow animation provider", func() {
		shadowBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shadowBGP.EXPECT().SetSlot(1).Return().Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().ComputeBindGroupProvider().Return(nil).Once()
		animMock.EXPECT().OutputBindGroupProvider().Return(nil).Once()
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{animMock: shadowBGP}
		suite.scene.lightHandler = nil
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("calls SetSlot for non-nil physics sync groups only", func() {
		physicsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		physicsBGP.EXPECT().SetSlot(1).Return().Once()
		suite.scene.lightHandler = nil
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{
			0: physicsBGP,
			1: nil,
		}
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("nil camera provider and nil animator shadow providers are skipped", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().ComputeBindGroupProvider().Return(nil).Once()
		animMock.EXPECT().OutputBindGroupProvider().Return(nil).Once()
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{animMock: nil}
		suite.scene.physicsSyncGroup = map[int]bind_group_provider.BindGroupProvider{0: nil}
		suite.scene.lightHandler = nil

		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})
}

func (suite *sceneImplTest) TestInitSSAOMissingBranches() {
	suite.Run("should return early when GBuffer is not enabled", func() {
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.NotPanics(func() { suite.scene.initSSAO() })
	})

	suite.Run("should return early when screen dimensions are zero for SSAO", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		gbhMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.gBufferHandler = gbhMock
		suite.scene.ssaoHandler = ssaoMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 0
		suite.NotPanics(func() { suite.scene.initSSAO() })
	})
}

func (suite *sceneImplTest) TestInitGBufferMissingBranches() {
	suite.Run("should return early when screen dimensions are zero", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		suite.scene.gBufferHandler = gbhMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 0
		suite.NotPanics(func() { suite.scene.initGBuffer() })
	})
}

func (suite *sceneImplTest) TestInitContactShadowsMissingBranches() {
	suite.Run("should return early when GBuffer is not enabled", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		cshMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		gbhMock.EXPECT().Enabled().Return(false).Once()
		lhMock.EXPECT().ContactShadowHandler().Return(cshMock).Once()
		suite.scene.gBufferHandler = gbhMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.NotPanics(func() { suite.scene.initContactShadows() })
	})

	suite.Run("should return early when screen dimensions are zero", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		cshMock := light_mocks.NewMockContactShadowHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		gbhMock.EXPECT().Enabled().Return(true).Once()
		lhMock.EXPECT().ContactShadowHandler().Return(cshMock).Once()
		suite.scene.gBufferHandler = gbhMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 0
		suite.NotPanics(func() { suite.scene.initContactShadows() })
	})
}

func (suite *sceneImplTest) TestInitShadowMapMissingBranches() {
	suite.Run("should return early when shadowVertShader is nil", func() {
		suite.NotPanics(func() { suite.scene.initShadowMap(nil, nil) })
	})
}

func (suite *sceneImplTest) TestInitLightCullResourcesMissingBranches() {
	suite.Run("should return early when cullComputeShader is nil", func() {
		suite.NotPanics(func() { suite.scene.initLightCullResourcesLocked(nil, nil, 800, 600) })
	})

	suite.Run("continues when slot1 lights buffer is nil", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lightsBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		cullBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		tileBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		cullShaderMock := shader_mocks.NewMockShader(suite.T())
		litShaderMock := shader_mocks.NewMockShader(suite.T())

		lhMock.EXPECT().Bgp("lights").Return(lightsBGPMock).Once()
		lhMock.EXPECT().Bgp("light_cull").Return(cullBGPMock).Once()
		lhMock.EXPECT().Bgp("tile_lit").Return(tileBGPMock).Once()
		lhMock.EXPECT().Resize(800, 600).Return().Once()
		lhMock.EXPECT().TileCountX().Return(4).Once()
		lhMock.EXPECT().TileCountY().Return(4).Once()
		lhMock.EXPECT().MaxLightsPerTile().Return(32).Maybe()
		lhMock.EXPECT().SetPipelineKey("light_cull", "light_cull_compute").Once()

		lightsBGPMock.EXPECT().Buffer(1).Return(&wgpu.Buffer{}).Twice()
		lightsBGPMock.EXPECT().Buffer(1).Return(nil).Once()
		lightsBGPMock.EXPECT().SetSlot(1).Return().Once()
		lightsBGPMock.EXPECT().SetSlot(0).Return().Once()

		cullBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		cullBGPMock.EXPECT().SetBuffer(2, mock.Anything).Return().Maybe()
		cullBGPMock.EXPECT().SetBuffer(3, mock.Anything).Return().Maybe()
		cullBGPMock.EXPECT().SetBuffer(1, mock.Anything).Return().Once()
		cullBGPMock.EXPECT().Buffer(2).Return(nil).Twice()
		cullBGPMock.EXPECT().Buffer(3).Return(nil).Twice()

		tileBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()

		cullShaderMock.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()

		grp := 5
		tileDecl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &grp,
			Args:  []shader.AnnotationArg{"", "", shader.AnnotationArgTileUniforms},
		}
		litShaderMock.EXPECT().Declarations().Return([]shader.Annotation{tileDecl}).Once()
		litShaderMock.EXPECT().BindGroupLayoutDescriptor(5).Return(wgpu.BindGroupLayoutDescriptor{}).Once()

		suite.scene.lightHandler = lhMock
		suite.rendererMock.EXPECT().InitBindGroup(cullBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(tileBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()

		suite.NotPanics(func() { suite.scene.initLightCullResourcesLocked(cullShaderMock, litShaderMock, 800, 600) })
	})
}

func (suite *sceneImplTest) TestInitCompositionMissingBranches() {
	suite.Run("should return early when screen dimensions are zero", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.compositionHandler = chMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 0
		suite.NotPanics(func() { suite.scene.initComposition() })
	})
}

func (suite *sceneImplTest) TestInitBloomMissingBranches() {
	suite.Run("should return early when half dimensions are zero", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(true).Once()
		suite.NotPanics(func() { suite.scene.initBloom(chMock, bgpMock, 0, 0) })
	})
}

func (suite *sceneImplTest) TestInitSSRMissingBranches() {
	suite.Run("should return early when GBuffer or Composition handler is not enabled", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		gbhMock.EXPECT().Enabled().Return(false).Once()
		suite.scene.ssrHandler = ssrMock
		suite.scene.gBufferHandler = gbhMock
		suite.scene.compositionHandler = chMock
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.initSSR() })
	})

	suite.Run("should return early when screen dimensions are zero", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		gbhMock.EXPECT().Enabled().Return(true).Once()
		chMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.ssrHandler = ssrMock
		suite.scene.gBufferHandler = gbhMock
		suite.scene.compositionHandler = chMock
		suite.scene.lightHandler = lhMock
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 0
		suite.NotPanics(func() { suite.scene.initSSR() })
	})
}

func (suite *sceneImplTest) TestReinitCameraBGPMissingBranches() {
	suite.Run("should return early when litFragShader is nil", func() {
		suite.NotPanics(func() { suite.scene.reinitCameraBGPForLitPipeline(nil) })
	})

	suite.Run("should return early when camera bind group provider is nil", func() {
		grp := 1
		cameraDecl := shader.Annotation{
			Type:  shader.AnnotationTypeBindingGroup,
			Group: &grp,
			Args:  []shader.AnnotationArg{"", "", shader.AnnotationArgCamera},
		}
		shaderMock := shader_mocks.NewMockShader(suite.T())
		shaderMock.EXPECT().Declarations().Return([]shader.Annotation{cameraDecl}).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		suite.scene.cam = camMock
		suite.NotPanics(func() { suite.scene.reinitCameraBGPForLitPipeline(shaderMock) })
	})
}

func (suite *sceneImplTest) TestPruneAnimatorMissingBranches() {
	suite.Run("should keep non-empty pool when other animators remain for the same model", func() {
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		suite.scene.instanceLookup = make(map[animator.Animator]map[uint32]uint64)
		mdlMock := model_mocks.NewMockModel(suite.T())
		animMock1 := animator_mocks.NewMockAnimator(suite.T())
		animMock2 := animator_mocks.NewMockAnimator(suite.T())
		animMock1.EXPECT().Model().Return(mdlMock).Once()
		animMock1.EXPECT().Release().Once()
		suite.scene.animatorPool[mdlMock] = []animator.Animator{animMock1, animMock2}
		suite.scene.pruneAnimator(animMock1)
		suite.Equal(1, len(suite.scene.animatorPool[mdlMock]))
		suite.Equal(animMock2, suite.scene.animatorPool[mdlMock][0])
	})
}

func (suite *sceneImplTest) TestRefreshAnimatorHiZBindGroupsMissingBranches() {
	suite.Run("should return early when lightHandler SSRHandler is nil", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})

	suite.Run("should skip animator when hiZBGP is nil", func() {
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(nil).Maybe()
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()
		mdlMock := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool[mdlMock] = []animator.Animator{animMock}
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})

	suite.Run("should skip animator when model is nil", func() {
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(nil).Maybe()
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock
		animMock := animator_mocks.NewMockAnimator(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		animMock.EXPECT().HiZBindGroupProvider().Return(bgpMock).Once()
		animMock.EXPECT().Model().Return(nil).Once()
		mdlMock := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool[mdlMock] = []animator.Animator{animMock}
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})

	suite.Run("should skip animator when pipeline is nil", func() {
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(nil).Maybe()
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock
		animMock := animator_mocks.NewMockAnimator(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mdlMock := model_mocks.NewMockModel(suite.T())
		animMock.EXPECT().HiZBindGroupProvider().Return(bgpMock).Once()
		animMock.EXPECT().Model().Return(mdlMock).Once()
		mdlMock.EXPECT().ComputePipelineKey().Return("some_compute_key").Once()
		suite.rendererMock.EXPECT().Pipeline("some_compute_key").Return(nil).Once()
		suite.scene.animatorPool[mdlMock] = []animator.Animator{animMock}
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})

	suite.Run("should skip animator when compute shader is nil", func() {
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(nil).Maybe()
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock
		animMock := animator_mocks.NewMockAnimator(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mdlMock := model_mocks.NewMockModel(suite.T())
		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		animMock.EXPECT().HiZBindGroupProvider().Return(bgpMock).Once()
		animMock.EXPECT().Model().Return(mdlMock).Once()
		mdlMock.EXPECT().ComputePipelineKey().Return("some_compute_key").Once()
		suite.rendererMock.EXPECT().Pipeline("some_compute_key").Return(pipeMock).Once()
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(nil).Once()
		suite.scene.animatorPool[mdlMock] = []animator.Animator{animMock}
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})
}

func (suite *sceneImplTest) TestReleaseResolutionDependentResourcesMissingBranches() {
	suite.Run("should not panic when all handlers are enabled but no GPU resources are allocated", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		cshMock := light_mocks.NewMockContactShadowHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())

		suite.scene.gBufferHandler = gbhMock
		suite.scene.ssaoHandler = ssaoMock
		lhMock.EXPECT().Bgp("ssao_lit").Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(cshMock).Maybe()
		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		suite.scene.compositionHandler = chMock
		suite.scene.ssrHandler = ssrMock

		gbhMock.EXPECT().Enabled().Return(true)
		gbhMock.EXPECT().SetSlot(mock.Anything).Maybe()
		gbhMock.EXPECT().NormalTextureView().Return(nil).Maybe()
		gbhMock.EXPECT().NormalTexture().Return(nil).Maybe()
		gbhMock.EXPECT().SetNormalTextureView(mock.Anything).Maybe()
		gbhMock.EXPECT().SetNormalTexture(mock.Anything).Maybe()
		gbhMock.EXPECT().AlbedoTextureView().Return(nil).Maybe()
		gbhMock.EXPECT().AlbedoTexture().Return(nil).Maybe()
		gbhMock.EXPECT().SetAlbedoTextureView(mock.Anything).Maybe()
		gbhMock.EXPECT().SetAlbedoTexture(mock.Anything).Maybe()
		gbhMock.EXPECT().DepthTextureView().Return(nil).Maybe()
		gbhMock.EXPECT().DepthTexture().Return(nil).Maybe()
		gbhMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		gbhMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()

		ssaoMock.EXPECT().Enabled().Return(true)
		ssaoMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssaoMock.EXPECT().RawTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().RawTexture().Return(nil).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().BlurredTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().BlurredTexture().Return(nil).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().ScratchTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().ScratchTexture().Return(nil).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().Bgp(mock.Anything).Return(nil).Maybe()

		cshMock.EXPECT().Enabled().Return(true)
		cshMock.EXPECT().SetSlot(mock.Anything).Maybe()
		cshMock.EXPECT().TextureView().Return(nil).Maybe()
		cshMock.EXPECT().Texture().Return(nil).Maybe()
		cshMock.EXPECT().SetTextureView(mock.Anything).Maybe()
		cshMock.EXPECT().SetTexture(mock.Anything).Maybe()
		cshMock.EXPECT().Bgp(mock.Anything).Return(nil).Maybe()

		shMock.EXPECT().CSMAtlasTexture().Return(nil)

		chMock.EXPECT().Enabled().Return(true)
		chMock.EXPECT().BloomMipCount().Return(0)
		chMock.EXPECT().SetSlot(mock.Anything).Maybe()
		chMock.EXPECT().HDRTextureView().Return(nil).Maybe()
		chMock.EXPECT().HDRTexture().Return(nil).Maybe()
		chMock.EXPECT().SetHDRTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetHDRTexture(mock.Anything).Maybe()
		chMock.EXPECT().MSAATextureView().Return(nil).Maybe()
		chMock.EXPECT().MSAATexture().Return(nil).Maybe()
		chMock.EXPECT().SetMSAATextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetMSAATexture(mock.Anything).Maybe()
		chMock.EXPECT().DepthTextureView().Return(nil).Maybe()
		chMock.EXPECT().DepthTexture().Return(nil).Maybe()
		chMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		chMock.EXPECT().BloomDownTexture().Return(nil).Maybe()
		chMock.EXPECT().BloomDownReadViews().Return(nil).Maybe()
		chMock.EXPECT().BloomDownStorageViews().Return(nil).Maybe()
		chMock.EXPECT().SetBloomDownTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomDownReadViews(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomDownStorageViews(mock.Anything).Maybe()
		chMock.EXPECT().BloomUpTexture().Return(nil).Maybe()
		chMock.EXPECT().BloomUpReadViews().Return(nil).Maybe()
		chMock.EXPECT().BloomUpStorageViews().Return(nil).Maybe()
		chMock.EXPECT().BloomUpMip0View().Return(nil).Maybe()
		chMock.EXPECT().SetBloomUpTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomUpReadViews(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomUpStorageViews(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomUpMip0View(mock.Anything).Maybe()
		chMock.EXPECT().LinearSampler().Return(nil).Maybe()
		chMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		chMock.EXPECT().Bgp(mock.Anything).Return(nil).Maybe()

		ssrMock.EXPECT().Enabled().Return(true)
		ssrMock.EXPECT().HiZMipCount().Return(0)
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().SSRTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().SSRTexture().Return(nil).Maybe()
		ssrMock.EXPECT().SetSSRTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetSSRTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZTexture().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMipReadViews().Return(nil).Maybe()
		ssrMock.EXPECT().HiZStorageViews().Return(nil).Maybe()
		ssrMock.EXPECT().SetHiZTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipReadViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZStorageViews(mock.Anything).Maybe()
		ssrMock.EXPECT().Bgp(mock.Anything).Return(nil).Maybe()

		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.releaseResolutionDependentResources() })
	})
}

func (suite *sceneImplTest) TestResizeMissingBranches() {
	suite.Run("should return early when dimensions are unchanged", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		suite.scene.cam = camMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.NotPanics(func() { suite.scene.Resize(800, 600) })
	})

	suite.Run("should enter needTileCullReinit path when tile count exceeds buffer capacity", func() {
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().SetAspect(mock.Anything).Return().Maybe()
		suite.scene.cam = camMock
		suite.scene.tileBufferCapacity = -1
		suite.scene.buildInjectionMap()
		suite.rendererMock.EXPECT().Resize(1280, 720).Return().Once()
		suite.NotPanics(func() { suite.scene.Resize(1280, 720) })
	})
}

func (suite *sceneImplTest) TestPrepareCompositionMissingBranches() {
	suite.Run("should set ToneMappingEnabled in params when tone mapping is enabled", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.compositionHandler = chMock
		chMock.EXPECT().Enabled().Return(true).Once()
		chMock.EXPECT().Exposure().Return(float32(1.0)).Once()
		chMock.EXPECT().ToneMappingEnabled().Return(true).Once()
		chMock.EXPECT().AutoExposureEnabled().Return(false).Once()
		chMock.EXPECT().BloomEnabled().Return(false).Once()
		chMock.EXPECT().Bgp("composition").Return(bgpMock).Once()
		chMock.EXPECT().PipelineKey("composition").Return("composition_pipe").Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginCompositionPass().Return().Once()
		suite.rendererMock.EXPECT().CompositionDrawCall(mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().EndCompositionPass().Return().Once()
		suite.rendererMock.EXPECT().EndCompositionFrame().Return().Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.PrepareComposition() })
	})

	suite.Run("should set AutoExposureEnabled in params when auto exposure is enabled", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.compositionHandler = chMock
		chMock.EXPECT().Enabled().Return(true).Once()
		chMock.EXPECT().Exposure().Return(float32(1.0)).Once()
		chMock.EXPECT().ToneMappingEnabled().Return(false).Once()
		chMock.EXPECT().AutoExposureEnabled().Return(true).Once()
		chMock.EXPECT().BloomEnabled().Return(false).Once()
		chMock.EXPECT().Bgp("composition").Return(bgpMock).Once()
		chMock.EXPECT().PipelineKey("composition").Return("composition_pipe").Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginCompositionPass().Return().Once()
		suite.rendererMock.EXPECT().CompositionDrawCall(mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().EndCompositionPass().Return().Once()
		suite.rendererMock.EXPECT().EndCompositionFrame().Return().Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.PrepareComposition() })
	})

	suite.Run("should set BloomEnabled and BloomIntensity in params when bloom is enabled", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.scene.compositionHandler = chMock
		chMock.EXPECT().Enabled().Return(true).Once()
		chMock.EXPECT().Exposure().Return(float32(1.0)).Once()
		chMock.EXPECT().ToneMappingEnabled().Return(false).Once()
		chMock.EXPECT().AutoExposureEnabled().Return(false).Once()
		chMock.EXPECT().BloomEnabled().Return(true).Once()
		chMock.EXPECT().BloomIntensity().Return(float32(0.5)).Once()
		chMock.EXPECT().Bgp("composition").Return(bgpMock).Once()
		chMock.EXPECT().PipelineKey("composition").Return("composition_pipe").Once()
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().BeginCompositionPass().Return().Once()
		suite.rendererMock.EXPECT().CompositionDrawCall(mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().EndCompositionPass().Return().Once()
		suite.rendererMock.EXPECT().EndCompositionFrame().Return().Once()
		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.PrepareComposition() })
	})
}

func (suite *sceneImplTest) TestWorldAABBMissingBranches() {
	suite.Run("should swap lo and hi when scale is negative", func() {
		modelMin := [3]float32{-1, -1, -1}
		modelMax := [3]float32{1, 1, 1}
		pos := [3]float32{0, 0, 0}
		scale := [3]float32{-2, -2, -2}
		wMin, wMax := worldAABB(modelMin, modelMax, pos, scale)
		suite.InDelta(float64(-2), float64(wMin[0]), 1e-6)
		suite.InDelta(float64(2), float64(wMax[0]), 1e-6)
		suite.InDelta(float64(-2), float64(wMin[1]), 1e-6)
		suite.InDelta(float64(2), float64(wMax[1]), 1e-6)
		suite.InDelta(float64(-2), float64(wMin[2]), 1e-6)
		suite.InDelta(float64(2), float64(wMax[2]), 1e-6)
	})

	suite.Run("should not swap when scale is positive", func() {
		modelMin := [3]float32{-1, -1, -1}
		modelMax := [3]float32{1, 1, 1}
		pos := [3]float32{2, 3, 4}
		scale := [3]float32{2, 2, 2}
		wMin, wMax := worldAABB(modelMin, modelMax, pos, scale)
		suite.InDelta(float64(0), float64(wMin[0]), 1e-6)
		suite.InDelta(float64(4), float64(wMax[0]), 1e-6)
		suite.InDelta(float64(1), float64(wMin[1]), 1e-6)
		suite.InDelta(float64(5), float64(wMax[1]), 1e-6)
		suite.InDelta(float64(2), float64(wMin[2]), 1e-6)
		suite.InDelta(float64(6), float64(wMax[2]), 1e-6)
	})
}

func (suite *sceneImplTest) TestPrepareLightsMissingBranches() {
	suite.Run("sort path non-nil ctrl at origin covers dist2 clamp alongside directional MaxFloat32", func() {
		ctrl := camera_mocks.NewMockCameraController(suite.T())
		ctrl.EXPECT().Position().Return(float32(0), float32(0), float32(0)).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Controller().Return(ctrl).Once()
		suite.scene.cam = camMock

		pointLight := light.NewLight(light.LightTypePoint)
		dirLight := light.NewLight(light.LightTypeDirectional)
		farPoint := light.NewLight(light.LightTypePoint)
		farPoint.SetPosition(10, 10, 10)

		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(bgpMock).Once()
		mockLH.EXPECT().Lights().Return([]light.Light{pointLight, dirLight, farPoint}).Once()
		mockLH.EXPECT().MaxGPULights().Return(2).Once()
		mockLH.EXPECT().MarshalLightBuffer(mock.Anything, mock.Anything).Return(make([]byte, 64)).Once()
		suite.scene.lightHandler = mockLH
		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Once()
		suite.NotPanics(func() { suite.scene.PrepareLights() })
	})
}

func (suite *sceneImplTest) TestNewSceneMissingBranches() {
	suite.Run("should panic when second InitBindGroup fails for slot 1", func() {
		bgp := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(bgp).Once()
		bgp.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().SetInjections(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("slot 1 err")).Once()
		suite.Panics(func() {
			_ = NewScene("test_slot1_err", camMock, suite.rendererMock)
		})
	})
}

func (suite *sceneImplTest) TestInitBloomDisabledFallback() {
	suite.Run("should init 1x1 fallback texture when bloom is disabled", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().InitTextureView(compBGPMock, 6, mock.Anything).Return(nil).Once()
		suite.NotPanics(func() { suite.scene.initBloom(chMock, compBGPMock, 800, 600) })
	})
}

func (suite *sceneImplTest) TestInitBloomHappyPath() {
	suite.Run("should init bloom textures and BGPs when bloom is enabled with mipCount 1", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(true).Once()
		downReadViews := make([]*wgpu.TextureView, 1)
		downStorageViews := make([]*wgpu.TextureView, 1)
		upReadViews := make([]*wgpu.TextureView, 1)
		upStorageViews := make([]*wgpu.TextureView, 1)
		suite.rendererMock.EXPECT().CreateBloomTextures(400, 300).Return(
			nil, downReadViews, downStorageViews, nil, upReadViews, upStorageViews, nil, 1, nil,
		).Once()
		chMock.EXPECT().SetBloomDownTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpMip0View(mock.Anything).Once()
		chMock.EXPECT().SetBloomMipCount(1).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		chMock.EXPECT().SetPipelineKey("bloom_downsample", "bloom_downsample").Once()
		chMock.EXPECT().SetPipelineKey("bloom_upsample", "bloom_upsample").Once()
		// down loop: i=0 accesses HDRTextureView (first mip) and LinearSampler
		chMock.EXPECT().LinearSampler().Return(nil).Once()
		chMock.EXPECT().HDRTextureView().Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		chMock.EXPECT().SetBgp("bloom_down_0", mock.Anything).Once()
		// up loop: mipCount-2 = -1 < 0, so nothing runs
		compBGPMock.EXPECT().SetTextureView(6, mock.Anything).Once()
		suite.NotPanics(func() { suite.scene.initBloom(chMock, compBGPMock, 800, 600) })
	})
}

func (suite *sceneImplTest) TestReleaseResolutionDependentResourcesWithBloomAndSSRBGPLoops() {
	suite.Run("should call Release on bloom and SSR HiZ BGPs when mip counts are non-zero", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		cshMock := light_mocks.NewMockContactShadowHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())

		suite.scene.gBufferHandler = gbhMock
		suite.scene.ssaoHandler = ssaoMock
		lhMock.EXPECT().Bgp("ssao_lit").Return(nil).Maybe()
		lhMock.EXPECT().ContactShadowHandler().Return(cshMock).Maybe()
		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		suite.scene.compositionHandler = chMock
		suite.scene.ssrHandler = ssrMock

		gbhMock.EXPECT().Enabled().Return(false)
		ssaoMock.EXPECT().Enabled().Return(false)
		cshMock.EXPECT().Enabled().Return(false)
		shMock.EXPECT().CSMAtlasTexture().Return(nil)

		bloomDown0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bloomDown0.EXPECT().SetSamplers(mock.Anything).Once()
		bloomDown0.EXPECT().Release().Once()
		bloomUp0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		bloomUp0.EXPECT().SetSamplers(mock.Anything).Once()
		bloomUp0.EXPECT().Release().Once()

		hizInitBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		hizInitBGP.EXPECT().Release().Once()
		hizDown1BGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		hizDown1BGP.EXPECT().Release().Once()
		hizInitMaxBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		hizInitMaxBGP.EXPECT().Release().Once()
		hizDownMax1BGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		hizDownMax1BGP.EXPECT().Release().Once()

		chMock.EXPECT().Enabled().Return(true)
		chMock.EXPECT().BloomMipCount().Return(1)
		chMock.EXPECT().SetSlot(mock.Anything).Maybe()
		chMock.EXPECT().HDRTextureView().Return(nil).Maybe()
		chMock.EXPECT().HDRTexture().Return(nil).Maybe()
		chMock.EXPECT().SetHDRTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetHDRTexture(mock.Anything).Maybe()
		chMock.EXPECT().MSAATextureView().Return(nil).Maybe()
		chMock.EXPECT().MSAATexture().Return(nil).Maybe()
		chMock.EXPECT().SetMSAATextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetMSAATexture(mock.Anything).Maybe()
		chMock.EXPECT().DepthTextureView().Return(nil).Maybe()
		chMock.EXPECT().DepthTexture().Return(nil).Maybe()
		chMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		chMock.EXPECT().BloomDownTexture().Return(nil).Maybe()
		chMock.EXPECT().BloomDownReadViews().Return(nil).Maybe()
		chMock.EXPECT().BloomDownStorageViews().Return(nil).Maybe()
		chMock.EXPECT().SetBloomDownTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomDownReadViews(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomDownStorageViews(mock.Anything).Maybe()
		chMock.EXPECT().BloomUpTexture().Return(nil).Maybe()
		chMock.EXPECT().BloomUpReadViews().Return(nil).Maybe()
		chMock.EXPECT().BloomUpStorageViews().Return(nil).Maybe()
		chMock.EXPECT().BloomUpMip0View().Return(nil).Maybe()
		chMock.EXPECT().SetBloomUpTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomUpReadViews(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomUpStorageViews(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomUpMip0View(mock.Anything).Maybe()
		chMock.EXPECT().Bgp("composition").Return(nil).Maybe()
		chMock.EXPECT().Bgp("luminance_compute").Return(nil).Maybe()
		chMock.EXPECT().Bgp("bloom_down_0").Return(bloomDown0).Once()
		chMock.EXPECT().Bgp("bloom_up_0").Return(bloomUp0).Once()
		chMock.EXPECT().LinearSampler().Return(nil).Maybe()
		chMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()

		ssrMock.EXPECT().Enabled().Return(true)
		ssrMock.EXPECT().HiZMipCount().Return(2)
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().SSRTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().SSRTexture().Return(nil).Maybe()
		ssrMock.EXPECT().SetSSRTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetSSRTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZTexture().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMipReadViews().Return(nil).Maybe()
		ssrMock.EXPECT().HiZStorageViews().Return(nil).Maybe()
		ssrMock.EXPECT().SetHiZTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipReadViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZStorageViews(mock.Anything).Maybe()
		ssrMock.EXPECT().Bgp("ssr_compute").Return(nil).Maybe()
		ssrMock.EXPECT().Bgp("hiz_init").Return(hizInitBGP).Once()
		ssrMock.EXPECT().Bgp("hiz_down_1").Return(hizDown1BGP).Once()
		ssrMock.EXPECT().Bgp("hiz_init_max").Return(hizInitMaxBGP).Once()
		ssrMock.EXPECT().Bgp("hiz_down_max_1").Return(hizDownMax1BGP).Once()

		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.releaseResolutionDependentResources() })
	})
}

func (suite *sceneImplTest) TestRefreshAnimatorHiZBindGroupsHappyPath() {
	suite.Run("calls InitBindGroup once per slot when animator has valid hiZBGP pipeline and compute shader", func() {
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		ssrMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(&wgpu.TextureView{}).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(&wgpu.TextureView{}).Maybe()
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock

		hiZBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		hiZBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		hiZBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Return().Maybe()

		computeShdrMock := shader_mocks.NewMockShader(suite.T())
		computeShdrMock.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{}).Once()

		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShdrMock).Once()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		mdlMock := model_mocks.NewMockModel(suite.T())
		animMock.EXPECT().HiZBindGroupProvider().Return(hiZBGPMock).Once()
		animMock.EXPECT().Model().Return(mdlMock).Once()
		mdlMock.EXPECT().ComputePipelineKey().Return("hiz_compute").Once()
		suite.rendererMock.EXPECT().Pipeline("hiz_compute").Return(pipeMock).Once()
		suite.rendererMock.EXPECT().InitBindGroup(hiZBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})
}

func (suite *sceneImplTest) TestSyncFrameSlotCompositionHandlerLuminanceBGPNil() {
	suite.Run("calls SetSlot on CompositionHandler but skips nil luminance BGP", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		chMock.EXPECT().SetSlot(1).Return().Once()
		chMock.EXPECT().Bgp("luminance_compute").Return(nil).Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(nil).Once()
		mockLH.EXPECT().ShadowHandler().Return(nil).Once()
		mockLH.EXPECT().Bgp("lights").Return(nil).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(nil).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(nil).Once()
		suite.scene.lightHandler = mockLH
		suite.scene.compositionHandler = chMock
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})

	suite.Run("calls SetSlot on csm_shadow_lit bind group", func() {
		shadowBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		shadowBGP.EXPECT().SetSlot(1).Return().Once()
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		shMock.EXPECT().Bgp("csm_shadow_lit").Return(shadowBGP).Once()
		shMock.EXPECT().CascadeCount().Return(0).Once()
		shMock.EXPECT().LightShadowAtlasSlots().Return(0).Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().ContactShadowHandler().Return(nil).Once()
		mockLH.EXPECT().ShadowHandler().Return(shMock).Once()
		mockLH.EXPECT().Bgp("lights").Return(nil).Once()
		mockLH.EXPECT().Bgp("light_cull").Return(nil).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(nil).Once()
		suite.scene.lightHandler = mockLH
		suite.NotPanics(func() { suite.scene.SyncFrameSlot(1) })
	})
}

func (suite *sceneImplTest) TestResizePostProcessingHappyPath() {
	suite.Run("runs WaitIdle and initSSAOLitBindGroup when postProcessingInitialized is true", func() {
		// Use the real lightHandler (all sub-handlers disabled by default).
		// releaseResolutionDependentResources skips all blocks (disabled/nil resources).
		// initSSAOLitBindGroup parses the real lit shader and installs fallback texture/sampler.
		suite.scene.buildInjectionMap()
		suite.scene.postProcessingInitialized = true
		suite.rendererMock.EXPECT().WaitIdle().Return().Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.NotPanics(func() { suite.scene.resizePostProcessing(800, 600) })
	})

	suite.Run("calls initCSMShadowLitBindGroup when CSMAtlasTexture is non-nil", func() {
		suite.scene.buildInjectionMap()
		suite.scene.postProcessingInitialized = true
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		suite.scene.lightHandler.Bgps()["ssao_lit"] = bind_group_provider.NewBindGroupProvider("ssao_lit")

		shadowHandler := suite.scene.lightHandler.ShadowHandler()
		shadowHandler.SetCSMAtlasTexture(&wgpu.Texture{})
		shadowHandler.SetCSMAtlasTextureView(&wgpu.TextureView{})
		shadowHandler.SetComparisonSampler(&wgpu.Sampler{})
		shadowHandler.SetLightShadowAtlasSlots(64)

		suite.rendererMock.EXPECT().WaitIdle().Return().Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)

		suite.NotPanics(func() { suite.scene.resizePostProcessing(800, 600) })
		suite.NotNil(shadowHandler.Bgp("csm_shadow_lit"))
	})
}

func (suite *sceneImplTest) TestPrepareShadowsPointLightNoCastShadows() {
	suite.Run("point cube face: model CastsShadows false skips all faces in outerScan", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		suite.scene.cullingDisabled = true

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(false).Maybe()
		mapKey := model_mocks.NewMockModel(suite.T())
		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Maybe()
		mockAnim.EXPECT().Model().Return(mockModel).Maybe()
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})
}

func (suite *sceneImplTest) TestPrepareLightCullingZeroLights() {
	suite.Run("zero enabled lights still dispatches cull shader", func() {
		suite.scene.lightHandler.SetEnabled(true)
		suite.scene.lightHandler.Resize(800, 600)

		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().InverseProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ViewMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().Near().Return(float32(0.1)).Once()
		camMock.EXPECT().Far().Return(float32(1000.0)).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Twice()
		suite.rendererMock.EXPECT().BeginComputeFrame().Return().Once()
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndComputeFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareLightCulling() })
	})
}

func (suite *sceneImplTest) TestDrawCallsAdditionalAnnotations() {
	suite.Run("OverlayParams nil mat BGP and nil effect resolves nil provider skips material", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgOverlayParams},
		}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("koa1").Once()
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(nil).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("koa1").Return(mockPipe).Once()
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("EffectParams nil effect and nil mat Provider resolves nil provider skips material", func() {
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		zero := 0
		decl := shader.Annotation{
			Type:    shader.AnnotationTypeBindingGroup,
			Group:   &zero,
			Binding: &zero,
			Args:    []shader.AnnotationArg{"", "", shader.AnnotationArgEffectParams},
		}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("kep1").Once()
		matMock.EXPECT().Provider(0).Return(nil).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Once()
		mockModel.EXPECT().EffectProvider().Return(nil).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Once()
		animMock.EXPECT().Model().Return(mockModel).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.rendererMock.EXPECT().Pipeline("kep1").Return(mockPipe).Once()
		suite.NoError(suite.scene.DrawCalls())
	})
}

func (suite *sceneImplTest) TestInitBloomMultiMip() {
	suite.Run("should init bloom BGPs with mipCount=2 covering down i>0 read view and up loop body", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(true).Once()
		downReadViews := []*wgpu.TextureView{&wgpu.TextureView{}, &wgpu.TextureView{}}
		downStorageViews := []*wgpu.TextureView{&wgpu.TextureView{}, &wgpu.TextureView{}}
		upReadViews := []*wgpu.TextureView{&wgpu.TextureView{}, &wgpu.TextureView{}}
		upStorageViews := []*wgpu.TextureView{&wgpu.TextureView{}, &wgpu.TextureView{}}
		upMip0View := &wgpu.TextureView{}
		suite.rendererMock.EXPECT().CreateBloomTextures(400, 300).Return(
			nil, downReadViews, downStorageViews, nil, upReadViews, upStorageViews, upMip0View, 2, nil,
		).Once()
		chMock.EXPECT().SetBloomDownTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpMip0View(mock.Anything).Once()
		chMock.EXPECT().SetBloomMipCount(2).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		chMock.EXPECT().SetPipelineKey("bloom_downsample", "bloom_downsample").Once()
		chMock.EXPECT().SetPipelineKey("bloom_upsample", "bloom_upsample").Once()
		chMock.EXPECT().LinearSampler().Return(nil).Once()
		chMock.EXPECT().HDRTextureView().Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		chMock.EXPECT().SetBgp("bloom_down_0", mock.Anything).Once()
		chMock.EXPECT().SetBgp("bloom_down_1", mock.Anything).Once()
		chMock.EXPECT().SetBgp("bloom_up_0", mock.Anything).Once()
		compBGPMock.EXPECT().SetTextureView(6, upMip0View).Once()
		suite.NotPanics(func() { suite.scene.initBloom(chMock, compBGPMock, 800, 600) })
	})
}

func (suite *sceneImplTest) TestInitLuminancePanicPaths() {
	suite.Run("should panic when RegisterPipelines fails", func() {
		suite.scene.buildInjectionMap()
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).
			Return(&wgpu.Buffer{}).Once()
		chMock.EXPECT().Exposure().Return(float32(1.0)).Once()
		suite.rendererMock.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Once()
		chMock.EXPECT().SetExposureBuffer(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("reg err")).Once()
		suite.Panics(func() { suite.scene.initLuminance(chMock, compBGPMock) })
	})

	suite.Run("should panic when InitBindGroup slot 0 fails", func() {
		suite.scene.buildInjectionMap()
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lumBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).
			Return(&wgpu.Buffer{}).Once()
		chMock.EXPECT().Exposure().Return(float32(1.0)).Once()
		suite.rendererMock.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Once()
		chMock.EXPECT().SetExposureBuffer(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		chMock.EXPECT().Bgp("luminance_compute").Return(lumBGPMock).Once()
		chMock.EXPECT().HDRTextureView().Return(nil).Once()
		lumBGPMock.EXPECT().SetTextureView(0, mock.Anything).Once()
		lumBGPMock.EXPECT().SetBuffer(2, mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(lumBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("bg err")).Once()
		suite.Panics(func() { suite.scene.initLuminance(chMock, compBGPMock) })
	})

	suite.Run("should panic when InitBindGroup slot 1 fails", func() {
		suite.scene.buildInjectionMap()
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lumBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).
			Return(&wgpu.Buffer{}).Once()
		chMock.EXPECT().Exposure().Return(float32(1.0)).Once()
		suite.rendererMock.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Once()
		chMock.EXPECT().SetExposureBuffer(mock.Anything).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		chMock.EXPECT().Bgp("luminance_compute").Return(lumBGPMock).Once()
		chMock.EXPECT().HDRTextureView().Return(nil).Once()
		lumBGPMock.EXPECT().SetTextureView(0, mock.Anything).Once()
		lumBGPMock.EXPECT().SetBuffer(2, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(lumBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Once()
		lumBGPMock.EXPECT().SetSlot(1).Once()
		suite.rendererMock.EXPECT().InitBindGroup(lumBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("slot1 err")).Once()
		suite.Panics(func() { suite.scene.initLuminance(chMock, compBGPMock) })
	})
}

func (suite *sceneImplTest) TestRefreshAnimatorHiZBindGroupsFallbackView() {
	suite.Run("should use hizFallbackView when SSR hizView is nil", func() {
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		fallbackView := &wgpu.TextureView{}
		suite.scene.hizFallbackView = fallbackView

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(&wgpu.TextureView{}).Maybe()
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock

		hiZBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		hiZBGPMock.EXPECT().SetSlot(mock.Anything).Maybe()
		hiZBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()

		computeShdrMock := shader_mocks.NewMockShader(suite.T())
		computeShdrMock.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{}).Once()

		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShdrMock).Once()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		mdlMock := model_mocks.NewMockModel(suite.T())
		animMock.EXPECT().HiZBindGroupProvider().Return(hiZBGPMock).Once()
		animMock.EXPECT().Model().Return(mdlMock).Once()
		mdlMock.EXPECT().ComputePipelineKey().Return("hiz_compute").Once()
		suite.rendererMock.EXPECT().Pipeline("hiz_compute").Return(pipeMock).Once()
		suite.rendererMock.EXPECT().InitBindGroup(hiZBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(nil).Twice()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})

	suite.Run("should skip all slots when hizView and hizFallbackView are both nil", func() {
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)
		suite.scene.hizFallbackView = nil

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(nil).Maybe()
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock

		hiZBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		hiZBGPMock.EXPECT().SetSlot(mock.Anything).Maybe()

		computeShdrMock := shader_mocks.NewMockShader(suite.T())
		computeShdrMock.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{}).Once()

		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShdrMock).Once()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		mdlMock := model_mocks.NewMockModel(suite.T())
		animMock.EXPECT().HiZBindGroupProvider().Return(hiZBGPMock).Once()
		animMock.EXPECT().Model().Return(mdlMock).Once()
		mdlMock.EXPECT().ComputePipelineKey().Return("hiz_compute").Once()
		suite.rendererMock.EXPECT().Pipeline("hiz_compute").Return(pipeMock).Once()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})

	suite.Run("should skip slot when InitBindGroup returns error", func() {
		suite.scene.animatorPool = make(map[model.Model][]animator.Animator)

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(&wgpu.TextureView{}).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(&wgpu.TextureView{}).Maybe()
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock

		hiZBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		hiZBGPMock.EXPECT().SetSlot(mock.Anything).Maybe()
		hiZBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()

		computeShdrMock := shader_mocks.NewMockShader(suite.T())
		computeShdrMock.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{}).Once()

		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShdrMock).Once()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		mdlMock := model_mocks.NewMockModel(suite.T())
		animMock.EXPECT().HiZBindGroupProvider().Return(hiZBGPMock).Once()
		animMock.EXPECT().Model().Return(mdlMock).Once()
		mdlMock.EXPECT().ComputePipelineKey().Return("hiz_compute").Once()
		suite.rendererMock.EXPECT().Pipeline("hiz_compute").Return(pipeMock).Once()
		suite.rendererMock.EXPECT().InitBindGroup(hiZBGPMock, mock.Anything, mock.Anything, mock.Anything).
			Return(errors.New("bg err")).Maybe()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})
}

func (suite *sceneImplTest) TestDrawCallsDirtyPrePassGuardContinues() {
	suite.Run("nil compute pool fires continue in pre-pass", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_nilpool").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Twice()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_nilpool").Return(mockPipe).Twice()
		suite.rendererMock.EXPECT().DrawCall("k_nilpool", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("six bad animators plus good animator trigger all pre-pass guard continues", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		// badAnim1: InstanceCount=0 fires pre-pass guard 1 and serial guard 1.
		badAnim1 := animator_mocks.NewMockAnimator(suite.T())
		badAnim1.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		badAnim1.EXPECT().InstanceCount().Return(uint32(0)).Twice()

		// badAnim2: InstanceCount=1, Model=nil fires pre-pass guard 2 and serial guard 2.
		badAnim2 := animator_mocks.NewMockAnimator(suite.T())
		badAnim2.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		badAnim2.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		badAnim2.EXPECT().Model().Return(nil).Twice()

		// badAnim3: valid model but empty materials fires pre-pass guard 3 and serial guard 3.
		meshBGP3 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		model3 := model_mocks.NewMockModel(suite.T())
		model3.EXPECT().RenderMaterials().Return([]material.Material{}).Twice()
		model3.EXPECT().LODMeshProvider(0).Return(meshBGP3).Once()
		badAnim3 := animator_mocks.NewMockAnimator(suite.T())
		badAnim3.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		badAnim3.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		badAnim3.EXPECT().Model().Return(model3).Twice()

		// badAnim4: valid model, mat with empty pipeline key fires pre-pass guard 4 and serial guard 4.
		mat4 := material_mocks.NewMockMaterial(suite.T())
		mat4.EXPECT().PipelineKey().Return("").Twice()
		meshBGP4 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		model4 := model_mocks.NewMockModel(suite.T())
		model4.EXPECT().RenderMaterials().Return([]material.Material{mat4}).Twice()
		model4.EXPECT().LODMeshProvider(0).Return(meshBGP4).Once()
		badAnim4 := animator_mocks.NewMockAnimator(suite.T())
		badAnim4.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		badAnim4.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		badAnim4.EXPECT().Model().Return(model4).Twice()

		// badAnim5: mat with valid key but nil pipeline fires pre-pass guard 5 and serial guard 5.
		mat5 := material_mocks.NewMockMaterial(suite.T())
		mat5.EXPECT().PipelineKey().Return("k_bad5").Twice()
		meshBGP5 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		model5 := model_mocks.NewMockModel(suite.T())
		model5.EXPECT().RenderMaterials().Return([]material.Material{mat5}).Twice()
		model5.EXPECT().LODMeshProvider(0).Return(meshBGP5).Once()
		suite.rendererMock.EXPECT().Pipeline("k_bad5").Return(nil).Twice()
		badAnim5 := animator_mocks.NewMockAnimator(suite.T())
		badAnim5.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		badAnim5.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		badAnim5.EXPECT().Model().Return(model5).Twice()

		// badAnim6: valid pipeline but nil vertex shader fires pre-pass guard 6 and serial guard 6.
		mat6 := material_mocks.NewMockMaterial(suite.T())
		mat6.EXPECT().PipelineKey().Return("k_bad6").Twice()
		meshBGP6 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		model6 := model_mocks.NewMockModel(suite.T())
		model6.EXPECT().RenderMaterials().Return([]material.Material{mat6}).Twice()
		model6.EXPECT().LODMeshProvider(0).Return(meshBGP6).Once()
		pipe6 := pipeline_mocks.NewMockPipeline(suite.T())
		pipe6.EXPECT().Shader(shader.ShaderTypeVertex).Return(nil).Twice()
		suite.rendererMock.EXPECT().Pipeline("k_bad6").Return(pipe6).Twice()
		badAnim6 := animator_mocks.NewMockAnimator(suite.T())
		badAnim6.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		badAnim6.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		badAnim6.EXPECT().Model().Return(model6).Twice()

		// goodAnim: fully valid, empty decls. Goroutine runs but returns nil,nil (maxGroup=-1, valid=false).
		// Serial fallback fires and produces a DrawCall.
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matGood := material_mocks.NewMockMaterial(suite.T())
		matGood.EXPECT().PipelineKey().Return("k_good").Twice()
		modelGood := model_mocks.NewMockModel(suite.T())
		modelGood.EXPECT().RenderMaterials().Return([]material.Material{matGood}).Twice()
		modelGood.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		vertShdrGood := shader_mocks.NewMockShader(suite.T())
		vertShdrGood.EXPECT().Declarations().Return([]shader.Annotation{}).Twice()
		pipeGood := pipeline_mocks.NewMockPipeline(suite.T())
		pipeGood.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrGood).Twice()
		pipeGood.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Twice()
		suite.rendererMock.EXPECT().Pipeline("k_good").Return(pipeGood).Twice()
		suite.rendererMock.EXPECT().DrawCall("k_good", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		goodAnim := animator_mocks.NewMockAnimator(suite.T())
		goodAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		goodAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		goodAnim.EXPECT().Model().Return(modelGood).Twice()
		goodAnim.EXPECT().CullingEnabled().Return(false).Once()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{
			mapKey: {badAnim1, badAnim2, badAnim3, badAnim4, badAnim5, badAnim6, goodAnim},
		}
		suite.NoError(suite.scene.DrawCalls())
	})
}

func (suite *sceneImplTest) TestDrawCallsGoroutineInfrastructure() {
	suite.Run("goroutine processes multi-group annotations and serial hits cache", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		camBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		camMock := camera_mocks.NewMockCamera(suite.T())
		// Called twice in goroutine: g0 Provider+Camera and g1 BindingGroup+Camera.
		camMock.EXPECT().BindGroupProvider().Return(camBGP).Twice()
		suite.scene.cam = camMock

		animBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		g0, g1, g2, g3 := 0, 1, 2, 3

		vertAnnotations := []shader.Annotation{
			// nil group → if decl.Group == nil { continue } in goroutine loop.
			{Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
			// g0, Provider+Camera → camBGP; also exercises if g > maxGroup { maxGroup = g }.
			{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
			// g0 duplicate → if _, exists := groupProviders[g]; exists { continue }.
			{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgAnimator}},
			// g1, BindingGroup+Camera (non-array) → cam.BindGroupProvider().
			{Group: &g1, Type: shader.AnnotationTypeBindingGroup, Args: []shader.AnnotationArg{"storage", "v", shader.AnnotationArgCamera}},
			// g2, BindingGroup+InstanceData via array stripping → OutputBindGroupProvider().
			{Group: &g2, Type: shader.AnnotationTypeBindingGroup, Args: []shader.AnnotationArg{"storage", "v", shader.AnnotationArg("array<" + string(shader.AnnotationArgInstanceData) + ">")}},
		}
		fragAnnotations := []shader.Annotation{
			// g3, Provider+Animator → OutputBindGroupProvider(); also exercises frag shader non-nil branch.
			{Group: &g3, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgAnimator}},
		}

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_infra").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()

		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return(vertAnnotations).Once()
		fragShdrMock := shader_mocks.NewMockShader(suite.T())
		fragShdrMock.EXPECT().Declarations().Return(fragAnnotations).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(fragShdrMock).Once()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		// Called twice in goroutine: g2 BindingGroup+InstanceData (array stripped) and g3 Provider+Animator.
		animMock.EXPECT().OutputBindGroupProvider().Return(animBGP).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()

		suite.rendererMock.EXPECT().Pipeline("k_infra").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_infra", meshBGP, uint32(1), mock.MatchedBy(func(bgs []bind_group_provider.BindGroupProvider) bool {
			return len(bgs) == 4
		})).Return(nil).Once()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})
}

func (suite *sceneImplTest) TestDrawCallsGoroutineInvalidPath() {
	suite.Run("goroutine returns nil nil when no valid group provider", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		// cam returns nil so the goroutine's g1 provider is nil → !valid → return nil, nil.
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().BindGroupProvider().Return(nil).Twice()
		suite.scene.cam = camMock

		animBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		g0, g1 := 0, 1
		annotations := []shader.Annotation{
			// nil group → if decl.Group == nil { continue }.
			{Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
			// g0, Provider+Animator → animBGP (non-nil → groupProviders[0] set).
			{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgAnimator}},
			// g0 duplicate → if _, exists := groupProviders[g]; exists { continue }.
			{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
			// g1, Provider+Camera → nil → if provider != nil FALSE → groupProviders[1] not set → !valid.
			{Group: &g1, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgCamera}},
		}

		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_ginvalid").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()

		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return(annotations).Twice()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Twice()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Twice()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		// No DrawCall: goroutine invalid + serial skipMaterial. Only pre-pass guard + serial guard.
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Twice()
		animMock.EXPECT().Model().Return(mockModel).Twice()
		// Goroutine g0 + serial fallback g0.
		animMock.EXPECT().OutputBindGroupProvider().Return(animBGP).Twice()

		suite.rendererMock.EXPECT().Pipeline("k_ginvalid").Return(mockPipe).Twice()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})
}

func (suite *sceneImplTest) TestDrawCallsGoroutineAnnotationCoverage() {
	suite.Run("goroutine provider lights enabled resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		lightsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(lightsBGP).Once()
		suite.scene.lightHandler = mockLH

		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgLights}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_glights").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_glights").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_glights", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine provider shadow enabled resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		shadowBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockSH := light_mocks.NewMockShadowHandler(suite.T())
		mockSH.EXPECT().Bgp("csm_shadow_lit").Return(shadowBGP).Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().ShadowHandler().Return(mockSH).Once()
		suite.scene.lightHandler = mockLH

		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgShadow}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gshadow").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gshadow").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gshadow", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine provider tiles enabled resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		tilesBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(tilesBGP).Once()
		suite.scene.lightHandler = mockLH

		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgTiles}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gtiles").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gtiles").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gtiles", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine provider ssao enabled resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		ssaoBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("ssao_lit").Return(ssaoBGP).Once()
		suite.scene.lightHandler = mockLH

		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgSSAO}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gssao").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gssao").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gssao", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine provider effect non-nil resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		effectBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgEffect}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_geffect").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		mockModel.EXPECT().EffectProvider().Return(effectBGP).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_geffect").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_geffect", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine provider material provider non-nil resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gmatprov").Twice()
		matMock.EXPECT().Provider(0).Return(matBGP).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gmatprov").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gmatprov", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine provider material provider nil falls to bindgroup", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeProvider, Args: []shader.AnnotationArg{shader.AnnotationArgMaterial}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gmatbgp").Twice()
		matMock.EXPECT().Provider(0).Return(nil).Once()
		matMock.EXPECT().BindGroupProvider().Return(matBGP).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gmatbgp").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gmatbgp", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine binding group light enabled resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		lightsBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("lights").Return(lightsBGP).Once()
		suite.scene.lightHandler = mockLH

		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeBindingGroup, Args: []shader.AnnotationArg{"storage", "v", shader.AnnotationArgLight}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gbglight").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gbglight").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gbglight", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine binding group shadow uniform enabled resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		shadowBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockSH := light_mocks.NewMockShadowHandler(suite.T())
		mockSH.EXPECT().Bgp("csm_shadow_lit").Return(shadowBGP).Once()
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().ShadowHandler().Return(mockSH).Once()
		suite.scene.lightHandler = mockLH

		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeBindingGroup, Args: []shader.AnnotationArg{"storage", "v", shader.AnnotationArgShadowUniform}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gbgshadowuni").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gbgshadowuni").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gbgshadowuni", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine binding group tile uniforms enabled resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		tilesBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		mockLH := light_mocks.NewMockLightingHandler(suite.T())
		mockLH.EXPECT().Enabled().Return(true).Once()
		mockLH.EXPECT().Bgp("tile_lit").Return(tilesBGP).Once()
		suite.scene.lightHandler = mockLH

		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeBindingGroup, Args: []shader.AnnotationArg{"storage", "v", shader.AnnotationArgTileUniforms}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gbgtileuni").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gbgtileuni").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gbgtileuni", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine binding group overlay params mat bgp non-nil resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeBindingGroup, Args: []shader.AnnotationArg{"storage", "v", shader.AnnotationArgOverlayParams}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gbgoverlay1").Twice()
		matMock.EXPECT().BindGroupProvider().Return(matBGP).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gbgoverlay1").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gbgoverlay1", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine binding group overlay params mat nil effect non-nil resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		effectBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeBindingGroup, Args: []shader.AnnotationArg{"storage", "v", shader.AnnotationArgOverlayParams}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gbgoverlay2").Twice()
		matMock.EXPECT().BindGroupProvider().Return(nil).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		mockModel.EXPECT().EffectProvider().Return(effectBGP).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gbgoverlay2").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gbgoverlay2", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine binding group effect params effect non-nil resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		effectBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeBindingGroup, Args: []shader.AnnotationArg{"storage", "v", shader.AnnotationArgEffectParams}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gbgeffect1").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		mockModel.EXPECT().EffectProvider().Return(effectBGP).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gbgeffect1").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gbgeffect1", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})

	suite.Run("goroutine binding group effect params effect nil mat provider non-nil resolves", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)

		matBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		g0 := 0
		decl := shader.Annotation{Group: &g0, Type: shader.AnnotationTypeBindingGroup, Args: []shader.AnnotationArg{"storage", "v", shader.AnnotationArgEffectParams}}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_gbgeffect2").Twice()
		matMock.EXPECT().Provider(0).Return(matBGP).Once()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		mockModel.EXPECT().EffectProvider().Return(nil).Once()
		vertShdrMock := shader_mocks.NewMockShader(suite.T())
		vertShdrMock.EXPECT().Declarations().Return([]shader.Annotation{decl}).Once()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(vertShdrMock).Once()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Once()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_gbgeffect2").Return(mockPipe).Once()
		suite.rendererMock.EXPECT().DrawCall("k_gbgeffect2", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
	})
}

func (suite *sceneImplTest) TestInitCompositionSSRNonNilPath() {
	suite.Run("should call SetTextureView for SSR binding when SSR texture view is non-nil", func() {
		suite.scene.buildInjectionMap()
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssrView := &wgpu.TextureView{}

		suite.scene.compositionHandler = chMock
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock

		suite.rendererMock.EXPECT().SampleCount().Return(uint32(1)).Maybe()
		suite.rendererMock.EXPECT().CreateCompositionTextures(800, 600, mock.Anything).Return(
			&wgpu.TextureView{}, &wgpu.Texture{}, nil, nil, &wgpu.TextureView{}, &wgpu.Texture{},
		).Maybe()
		chMock.EXPECT().SetHDRTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetHDRTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetMSAATexture(mock.Anything).Maybe()
		chMock.EXPECT().SetMSAATextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetSlot(mock.Anything).Maybe()

		suite.rendererMock.EXPECT().SetRenderTargetFormat(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}).Maybe()
		chMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterCompositionPipeline(mock.Anything).Return(nil).Maybe()
		chMock.EXPECT().SetPipelineKey("composition", "composition").Maybe()
		chMock.EXPECT().Bgp("composition").Return(compBGPMock).Maybe()
		compBGPMock.EXPECT().SetTextureView(0, mock.Anything).Maybe()
		compBGPMock.EXPECT().SetSampler(1, mock.Anything).Maybe()
		compBGPMock.EXPECT().SetSampler(3, mock.Anything).Maybe()
		compBGPMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()

		// initLuminance mocks
		suite.rendererMock.EXPECT().CreateBuffer(mock.Anything, mock.Anything, mock.Anything).
			Return((*wgpu.Buffer)(nil)).Maybe()
		suite.rendererMock.EXPECT().WriteRawBuffer(mock.Anything, mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		chMock.EXPECT().Exposure().Return(float32(1.0)).Maybe()
		chMock.EXPECT().SetExposureBuffer(mock.Anything).Maybe()
		chMock.EXPECT().HDRTextureView().Return(nil).Maybe()
		lumBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lumBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		lumBGPMock.EXPECT().SetBuffer(mock.Anything, mock.Anything).Maybe()
		chMock.EXPECT().Bgp("luminance_compute").Return(lumBGPMock).Maybe()
		lumBGPMock.EXPECT().SetSlot(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(lumBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		// initBloom mocks (bloom disabled path)
		chMock.EXPECT().BloomEnabled().Return(false).Maybe()
		suite.rendererMock.EXPECT().InitTextureView(compBGPMock, 6, mock.Anything).Return(nil).Maybe()

		// Post-initLuminance/initBloom: SSR texture binding + final InitBindGroup + Resize + SetEnabled
		ssrMock.EXPECT().SSRTextureView().Return(ssrView).Times(2)
		compBGPMock.EXPECT().SetTextureView(2, mock.Anything).Once()
		suite.rendererMock.EXPECT().InitBindGroup(compBGPMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		chMock.EXPECT().Resize(800, 600).Once()
		chMock.EXPECT().SetEnabled(true).Once()

		suite.NotPanics(func() { suite.scene.initComposition() })
	})
}

func (suite *sceneImplTest) TestInitLightBindGroupStorageType() {
	suite.Run("should set size override for BufferBindingTypeStorage", func() {
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
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage}},
			},
		}
		shaderMock.EXPECT().BindGroupLayoutDescriptor(lightGroupIdx).Return(desc).Once()

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		bgpMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lhMock.EXPECT().Bgp("lights").Return(bgpMock).Once()
		lhMock.EXPECT().MaxGPULights().Return(64).Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgpMock.EXPECT().SetSlot(1).Return().Once()
		suite.rendererMock.EXPECT().InitBindGroup(bgpMock, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		bgpMock.EXPECT().SetSlot(0).Return().Once()
		suite.scene.lightHandler = lhMock
		suite.scene.initLightBindGroup(shaderMock)
	})
}

func (suite *sceneImplTest) TestPrepareShadowsOuterScanBreak() {
	suite.Run("point cube face: break outerScan fires when model AABB intersects face frustum", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		suite.scene.cullingDisabled = true

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.0002)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Maybe()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Maybe()
		mockModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Maybe()
		mockModel.EXPECT().BoundingRadius().Return(float32(1000)).Maybe()
		mockModel.EXPECT().LODMeshProvider(0).Return(nil).Maybe()
		mockModel.EXPECT().LODCount().Return(1).Maybe()

		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Maybe()
		mockAnim.EXPECT().Model().Return(mockModel).Maybe()
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{1, 1, 1}).Maybe()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})
}

func (suite *sceneImplTest) TestResizePostProcessingGBufferEnabled() {
	suite.Run("calls initGBuffer when GBufferHandler is enabled", func() {
		suite.scene.buildInjectionMap()
		suite.scene.postProcessingInitialized = true
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		cshMock := light_mocks.NewMockContactShadowHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())

		suite.scene.gBufferHandler = gbhMock
		suite.scene.ssaoHandler = ssaoMock
		lhMock.EXPECT().ContactShadowHandler().Return(cshMock).Maybe()
		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		suite.scene.compositionHandler = chMock
		suite.scene.ssrHandler = ssrMock
		lhMock.EXPECT().Bgp("ssao_lit").Return(nil).Maybe()

		gbhMock.EXPECT().Enabled().Return(true).Maybe()
		gbhMock.EXPECT().SetSlot(mock.Anything).Maybe()
		gbhMock.EXPECT().NormalTextureView().Return(nil).Maybe()
		gbhMock.EXPECT().NormalTexture().Return(nil).Maybe()
		gbhMock.EXPECT().SetNormalTextureView(mock.Anything).Maybe()
		gbhMock.EXPECT().SetNormalTexture(mock.Anything).Maybe()
		gbhMock.EXPECT().AlbedoTextureView().Return(nil).Maybe()
		gbhMock.EXPECT().AlbedoTexture().Return(nil).Maybe()
		gbhMock.EXPECT().SetAlbedoTextureView(mock.Anything).Maybe()
		gbhMock.EXPECT().SetAlbedoTexture(mock.Anything).Maybe()
		gbhMock.EXPECT().DepthTextureView().Return(nil).Maybe()
		gbhMock.EXPECT().DepthTexture().Return(nil).Maybe()
		gbhMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		gbhMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()

		ssaoMock.EXPECT().Enabled().Return(false).Maybe()
		ssaoMock.EXPECT().BlurredTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().LinearSampler().Return(nil).Maybe()
		cshMock.EXPECT().Enabled().Return(false).Maybe()
		shMock.EXPECT().CSMAtlasTexture().Return(nil).Maybe()
		chMock.EXPECT().Enabled().Return(false).Maybe()
		chMock.EXPECT().BloomMipCount().Return(0).Maybe()
		ssrMock.EXPECT().Enabled().Return(false).Maybe()
		ssrMock.EXPECT().HiZMipCount().Return(0).Maybe()

		suite.rendererMock.EXPECT().CreateGBufferTextures(800, 600).Return(
			&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{},
		).Twice()
		gbhMock.EXPECT().SetPipelineKey(mock.Anything, mock.Anything).Maybe()
		gbhMock.EXPECT().Resize(800, 600).Once()
		gbhMock.EXPECT().SetEnabled(true).Once()
		suite.rendererMock.EXPECT().RegisterGBufferPipeline(mock.Anything).Return(nil).Twice()

		suite.rendererMock.EXPECT().WaitIdle().Return().Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.resizePostProcessing(800, 600) })
	})
}

// TestPrepareShadowsFrustumInvisible covers scene.go line 1535: the continue fired when
// a point light's sphere is outside the camera frustum inside an already-open atlas pass.
// The atlas pass is opened by the spot light whose slot makes hasAtlasWork true.
func (suite *sceneImplTest) TestPrepareShadowsFrustumInvisible() {
	suite.Run("point cube face: frustum-invisible light skips during open atlas pass", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(7)
		sh.SetLightShadowAtlasCols(7)

		sl := light_mocks.NewMockLight(suite.T())
		sl.EXPECT().Enabled().Return(true).Maybe()
		sl.EXPECT().CastsShadows().Return(true).Maybe()
		sl.EXPECT().Type().Return(light.LightTypeSpot).Maybe()
		sl.EXPECT().Position().Return([3]float32{0, 0, 0}).Maybe()
		sl.EXPECT().Direction().Return([3]float32{0, -1, 0}).Maybe()
		sl.EXPECT().Range().Return(float32(10)).Maybe()
		sl.EXPECT().OuterCone().Return(float32(0.5)).Maybe()
		sl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		sl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		suite.scene.lightHandler.AddLight(sl)

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{999, 999, 999}).Maybe()
		pl.EXPECT().Range().Return(float32(1)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		var view, proj, vp [16]float32
		common.LookAt(view[:], 0, 0, 0, 0, 0, -1, 0, 1, 0)
		common.Perspective(proj[:], float32(math.Pi/2), 1.0, 0.1, 100.0)
		common.Mul4(vp[:], proj[:], view[:])
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().ViewProjectionMatrix().Return(vp).Once()
		suite.scene.cam = camMock

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})
}

// TestPrepareShadowsOuterScanRangeExceed covers scene.go lines 1569-1570: the continue
// fired when an instance's distance from the point light exceeds rng+boundR*maxS in the
// outerScan geometry-discovery loop.
func (suite *sceneImplTest) TestPrepareShadowsOuterScanRangeExceed() {
	suite.Run("point cube face: instance beyond light range triggers range-check continue", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		suite.scene.cullingDisabled = true

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 0, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(10)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().CastsShadows().Return(true).Maybe()
		mockModel.EXPECT().BoundingMin().Return([3]float32{-1, -1, -1}).Maybe()
		mockModel.EXPECT().BoundingMax().Return([3]float32{1, 1, 1}).Maybe()
		mockModel.EXPECT().BoundingRadius().Return(float32(1)).Maybe()

		mockAnim := animator_mocks.NewMockAnimator(suite.T())
		mockAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		mockAnim.EXPECT().InstanceCount().Return(uint32(1)).Maybe()
		mockAnim.EXPECT().Model().Return(mockModel).Maybe()
		// Instance is ~173 units from the light; rng+boundR*maxS = 10+1*1 = 11, so 173 > 11.
		mockAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{100, 100, 100}, [3]float32{1, 1, 1}).Maybe()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {mockAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})
}

func (suite *sceneImplTest) TestInitBloomPanicArms() {
	suite.Run("should panic when InitTextureView fails and bloom is disabled", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().InitTextureView(compBGPMock, 6, mock.Anything).Return(errors.New("fail")).Once()
		suite.Panics(func() { suite.scene.initBloom(chMock, compBGPMock, 800, 600) })
	})

	suite.Run("should panic when CreateBloomTextures fails", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(true).Once()
		suite.rendererMock.EXPECT().CreateBloomTextures(400, 300).Return(
			nil, nil, nil, nil, nil, nil, nil, 0, errors.New("fail"),
		).Once()
		suite.Panics(func() { suite.scene.initBloom(chMock, compBGPMock, 800, 600) })
	})

	suite.Run("should panic when RegisterPipelines fails for downsample pipeline", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(true).Once()
		downReadViews := make([]*wgpu.TextureView, 1)
		downStorageViews := make([]*wgpu.TextureView, 1)
		upReadViews := make([]*wgpu.TextureView, 1)
		upStorageViews := make([]*wgpu.TextureView, 1)
		suite.rendererMock.EXPECT().CreateBloomTextures(400, 300).Return(
			nil, downReadViews, downStorageViews, nil, upReadViews, upStorageViews, nil, 1, nil,
		).Once()
		chMock.EXPECT().SetBloomDownTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpMip0View(mock.Anything).Once()
		chMock.EXPECT().SetBloomMipCount(1).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("fail")).Once()
		suite.Panics(func() { suite.scene.initBloom(chMock, compBGPMock, 800, 600) })
	})

	suite.Run("should panic when RegisterPipelines fails for upsample pipeline", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(true).Once()
		downReadViews := make([]*wgpu.TextureView, 1)
		downStorageViews := make([]*wgpu.TextureView, 1)
		upReadViews := make([]*wgpu.TextureView, 1)
		upStorageViews := make([]*wgpu.TextureView, 1)
		suite.rendererMock.EXPECT().CreateBloomTextures(400, 300).Return(
			nil, downReadViews, downStorageViews, nil, upReadViews, upStorageViews, nil, 1, nil,
		).Once()
		chMock.EXPECT().SetBloomDownTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpMip0View(mock.Anything).Once()
		chMock.EXPECT().SetBloomMipCount(1).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		chMock.EXPECT().SetPipelineKey("bloom_downsample", "bloom_downsample").Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("fail")).Once()
		suite.Panics(func() { suite.scene.initBloom(chMock, compBGPMock, 800, 600) })
	})

	suite.Run("should panic when InitBindGroup fails in downsample loop", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(true).Once()
		downReadViews := make([]*wgpu.TextureView, 1)
		downStorageViews := make([]*wgpu.TextureView, 1)
		upReadViews := make([]*wgpu.TextureView, 1)
		upStorageViews := make([]*wgpu.TextureView, 1)
		suite.rendererMock.EXPECT().CreateBloomTextures(400, 300).Return(
			nil, downReadViews, downStorageViews, nil, upReadViews, upStorageViews, nil, 1, nil,
		).Once()
		chMock.EXPECT().SetBloomDownTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpMip0View(mock.Anything).Once()
		chMock.EXPECT().SetBloomMipCount(1).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		chMock.EXPECT().SetPipelineKey("bloom_downsample", "bloom_downsample").Once()
		chMock.EXPECT().SetPipelineKey("bloom_upsample", "bloom_upsample").Once()
		chMock.EXPECT().LinearSampler().Return(nil).Once()
		chMock.EXPECT().HDRTextureView().Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("fail")).Once()
		suite.Panics(func() { suite.scene.initBloom(chMock, compBGPMock, 800, 600) })
	})

	suite.Run("should panic when InitBindGroup fails in upsample loop", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(true).Once()
		downReadViews := make([]*wgpu.TextureView, 2)
		downStorageViews := make([]*wgpu.TextureView, 2)
		upReadViews := make([]*wgpu.TextureView, 2)
		upStorageViews := make([]*wgpu.TextureView, 2)
		suite.rendererMock.EXPECT().CreateBloomTextures(400, 300).Return(
			nil, downReadViews, downStorageViews, nil, upReadViews, upStorageViews, nil, 2, nil,
		).Once()
		chMock.EXPECT().SetBloomDownTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpMip0View(mock.Anything).Once()
		chMock.EXPECT().SetBloomMipCount(2).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		chMock.EXPECT().SetPipelineKey("bloom_downsample", "bloom_downsample").Once()
		chMock.EXPECT().SetPipelineKey("bloom_upsample", "bloom_upsample").Once()
		chMock.EXPECT().LinearSampler().Return(nil).Once()
		chMock.EXPECT().HDRTextureView().Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("fail")).Once()
		chMock.EXPECT().SetBgp("bloom_down_0", mock.Anything).Once()
		chMock.EXPECT().SetBgp("bloom_down_1", mock.Anything).Once()
		suite.Panics(func() { suite.scene.initBloom(chMock, compBGPMock, 800, 600) })
	})

	suite.Run("should cover upReadViews else branch when i is not mipCount-2 in upsample loop", func() {
		chMock := composition_mocks.NewMockHandler(suite.T())
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().BloomEnabled().Return(true).Once()
		downReadViews := make([]*wgpu.TextureView, 3)
		downStorageViews := make([]*wgpu.TextureView, 3)
		upReadViews := make([]*wgpu.TextureView, 3)
		upStorageViews := make([]*wgpu.TextureView, 3)
		upMip0View := &wgpu.TextureView{}
		suite.rendererMock.EXPECT().CreateBloomTextures(400, 300).Return(
			nil, downReadViews, downStorageViews, nil, upReadViews, upStorageViews, upMip0View, 3, nil,
		).Once()
		chMock.EXPECT().SetBloomDownTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomDownStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpTexture(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpReadViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpStorageViews(mock.Anything).Once()
		chMock.EXPECT().SetBloomUpMip0View(mock.Anything).Once()
		chMock.EXPECT().SetBloomMipCount(3).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		chMock.EXPECT().SetPipelineKey("bloom_downsample", "bloom_downsample").Once()
		chMock.EXPECT().SetPipelineKey("bloom_upsample", "bloom_upsample").Once()
		chMock.EXPECT().LinearSampler().Return(nil).Once()
		chMock.EXPECT().HDRTextureView().Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(5)
		chMock.EXPECT().SetBgp("bloom_down_0", mock.Anything).Once()
		chMock.EXPECT().SetBgp("bloom_down_1", mock.Anything).Once()
		chMock.EXPECT().SetBgp("bloom_down_2", mock.Anything).Once()
		chMock.EXPECT().SetBgp("bloom_up_1", mock.Anything).Once()
		chMock.EXPECT().SetBgp("bloom_up_0", mock.Anything).Once()
		compBGPMock.EXPECT().SetTextureView(6, upMip0View).Once()
		suite.NotPanics(func() { suite.scene.initBloom(chMock, compBGPMock, 800, 600) })
	})
}
func (suite *sceneImplTest) TestCreateAnimatorUncoveredPaths() {
	makeBase := func() (*model_mocks.MockModel, *shader_mocks.MockShader, *shader_mocks.MockShader, *shader_mocks.MockShader) {
		return model_mocks.NewMockModel(suite.T()),
			shader_mocks.NewMockShader(suite.T()),
			shader_mocks.NewMockShader(suite.T()),
			shader_mocks.NewMockShader(suite.T())
	}

	suite.Run("should panic when InitBindGroup slot 1 compute fails", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("slot1 compute fail")).Once()
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("should init Hi-Z BGP when HiZBindGroupProvider returns non-nil", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.scene.hizFallbackView = &wgpu.TextureView{}
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(5)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.NotPanics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("should share packed and scratch buffers for slot 1 compute", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		packed := 2
		scratch := 3
		packedDecl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Binding: &packed,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorPacked},
		}
		scratchDecl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Binding: &scratch,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorScratch},
		}
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		mdl.EXPECT().SetComputePipelineKey(mock.Anything).Return().Once()
		mdl.EXPECT().RenderMaterials().Return(nil).Once()
		cs.EXPECT().Declarations().Return([]shader.Annotation{packedDecl, scratchDecl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().Key().Return("ck").Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			RunAndReturn(func(provider bind_group_provider.BindGroupProvider, desc wgpu.BindGroupLayoutDescriptor, usages map[int]wgpu.BufferUsage, sizes map[int]uint64) error {
				provider.SetBuffer(packed, &wgpu.Buffer{})
				provider.SetBuffer(scratch, &wgpu.Buffer{})
				return nil
			}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().CreateBuffer("shadow_indirect", uint64(20), wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst).Return(&wgpu.Buffer{}).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Times(2)
		suite.NotPanics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("Hi-Z BGP slot 0 InitBindGroup panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.scene.hizFallbackView = &wgpu.TextureView{}
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("hiz slot0 fail")).Once()
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("Hi-Z BGP slot 1 InitBindGroup panics", func() {
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		cs.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.scene.hizFallbackView = &wgpu.TextureView{}
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(4)
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("hiz slot1 fail")).Once()
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})

	suite.Run("output BGP slot 1 InitBindGroup panics", func() {
		rawOutputBnd := 1
		outputDecl := shader.Annotation{
			Type:    shader.AnnotationTypeProvider,
			Binding: &rawOutputBnd,
			Args:    []shader.AnnotationArg{shader.AnnotationArgAnimatorOutput},
		}
		mdl, cs, vs, fs := makeBase()
		_ = fs
		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(1).Once()
		mdl.EXPECT().Name().Return("m").Maybe()
		cs.EXPECT().Declarations().Return([]shader.Annotation{outputDecl}).Maybe()
		cs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{
			Entries: []wgpu.BindGroupLayoutEntry{
				{Binding: 1, Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 64}},
			},
		}).Once()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Once()
		vs.EXPECT().BindGroupLayoutDescriptor(0).Return(wgpu.BindGroupLayoutDescriptor{}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
			RunAndReturn(func(provider bind_group_provider.BindGroupProvider, _ wgpu.BindGroupLayoutDescriptor, _ map[int]wgpu.BufferUsage, _ map[int]uint64) error {
				provider.SetBuffer(rawOutputBnd, &wgpu.Buffer{})
				return nil
			}).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(2)
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("output slot1 fail")).Once()
		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})
}

func (suite *sceneImplTest) TestShadowAnimatorBindGroup() {
	suite.Run("nil animator returns nil", func() {
		result := suite.scene.shadowAnimatorBindGroup(nil)
		suite.Nil(result)
	})

	suite.Run("simple backend returns scene shadow animation provider", func() {
		shadowBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().BackendType().Return(animator.BackendTypeSimple).Once()
		suite.scene.shadowAnimationProviders = map[animator.Animator]bind_group_provider.BindGroupProvider{animMock: shadowBGP}
		result := suite.scene.shadowAnimatorBindGroup(animMock)
		suite.Equal(shadowBGP, result)
	})

	suite.Run("non-simple backend returns animator output provider", func() {
		outputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().BackendType().Return(animator.BackendTypeSkeletal).Once()
		animMock.EXPECT().OutputBindGroupProvider().Return(outputBGP).Once()
		result := suite.scene.shadowAnimatorBindGroup(animMock)
		suite.Equal(outputBGP, result)
	})
}

func (suite *sceneImplTest) TestReleaseSharedBufferBindGroupProvider() {
	suite.Run("nil provider no-op", func() {
		suite.NotPanics(func() { releaseSharedBufferBindGroupProvider(nil) })
	})

	suite.Run("non-nil provider clears both slots before release", func() {
		provider := bgp_mocks.NewMockBindGroupProvider(suite.T())
		slotOrder := make([]int, 0, 2)
		currentSlot := -1
		clearedSlots := map[int]bool{}

		provider.EXPECT().SetSlot(mock.Anything).Run(func(slot int) {
			currentSlot = slot
			slotOrder = append(slotOrder, slot)
		}).Return().Twice()
		provider.EXPECT().SetBuffers(mock.Anything).Run(func(buffers map[int]*wgpu.Buffer) {
			suite.Len(buffers, 0)
			clearedSlots[currentSlot] = true
		}).Return().Twice()
		provider.EXPECT().Release().Run(func() {
			suite.True(clearedSlots[0])
			suite.True(clearedSlots[1])
		}).Return().Once()

		releaseSharedBufferBindGroupProvider(provider)

		suite.Equal([]int{0, 1}, slotOrder)
	})
}

func (suite *sceneImplTest) TestAnimLODLevel() {
	suite.Run("lodEnabled true cache hit returns cached level", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		suite.scene.lodEnabled = true
		suite.scene.lodLevelCache = map[animator.Animator]int{animMock: 2}
		result := suite.scene.animLODLevel(animMock)
		suite.Equal(2, result)
	})

	suite.Run("lodEnabled true cache miss returns zero", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		suite.scene.lodEnabled = true
		suite.scene.lodLevelCache = map[animator.Animator]int{}
		result := suite.scene.animLODLevel(animMock)
		suite.Equal(0, result)
	})
}

func (suite *sceneImplTest) TestAnimShadowLODLevel() {
	suite.Run("mdl nil returns base", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Model().Return(nil).Once()
		suite.scene.lodEnabled = false
		result := suite.scene.animShadowLODLevel(animMock)
		suite.Equal(0, result)
	})

	suite.Run("level exceeds maxLevel clamps to maxLevel", func() {
		animMock := animator_mocks.NewMockAnimator(suite.T())
		mockModel := model_mocks.NewMockModel(suite.T())
		animMock.EXPECT().Model().Return(mockModel).Once()
		mockModel.EXPECT().LODCount().Return(2).Once()
		suite.scene.lodEnabled = false
		suite.scene.lodShadowBias = 5
		result := suite.scene.animShadowLODLevel(animMock)
		suite.Equal(1, result)
	})
}

func (suite *sceneImplTest) TestPrepareComputeLODPath() {
	suite.Run("phase1 lodEnabled computes LOD level and writes cache", func() {
		suite.scene.cullingDisabled = true
		suite.scene.lodEnabled = true
		suite.scene.lod1Distance = 10.0
		suite.scene.lod2Distance = 30.0
		suite.scene.lodLevelCache = make(map[animator.Animator]int)
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)
		suite.scene.buildInjectionMap()
		realShdr := shader.NewShader("_pc_lod1", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipe := pipeline.NewPipeline("pc-lod1-key", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShdr),
		)
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ctrlMock := camera_mocks.NewMockCameraController(suite.T())
		ctrlMock.EXPECT().Position().Return(float32(0), float32(0), float32(0)).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		camMock.EXPECT().Controller().Return(ctrlMock).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("pc-lod1-key").Times(2)
		mockModel.EXPECT().LODCount().Return(2).Times(2)
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(8)
		animMock.EXPECT().Model().Return(mockModel).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().SetScreenSize(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().SetProjectionX(mock.Anything).Return().Once()
		animMock.EXPECT().SetHiZMipCount(mock.Anything).Return().Once()
		animMock.EXPECT().SetViewProj(mock.Anything).Return().Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(1)).Once()
		animMock.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{5, 0, 0}, [3]float32{1, 1, 1}).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("pc-lod1-key").Return(realPipe).Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "pc-lod1-key" && d[0].Providers[0].Provider == computeBGP
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
		suite.Contains(suite.scene.lodLevelCache, animMock)
		suite.Equal(0, suite.scene.lodLevelCache[animMock])
	})
}

func (suite *sceneImplTest) TestDrawCallsDirtyCacheElseMerge() {
	suite.Run("drawCacheDirty true parallel pre-pass then serial fallback and builtCache merge", func() {
		suite.scene.drawCacheDirty = true
		suite.scene.drawBindGroupCache = make(map[drawCacheKey][]bind_group_provider.BindGroupProvider)
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)
		suite.scene.drawBindGroupsPool = []bind_group_provider.BindGroupProvider{}
		meshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		matMock := material_mocks.NewMockMaterial(suite.T())
		matMock.EXPECT().PipelineKey().Return("k_dirty").Twice()
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().LODMeshProvider(0).Return(meshBGP).Once()
		mockModel.EXPECT().RenderMaterials().Return([]material.Material{matMock}).Twice()
		renderShdrMock := shader_mocks.NewMockShader(suite.T())
		renderShdrMock.EXPECT().Declarations().Return([]shader.Annotation{}).Twice()
		mockPipe := pipeline_mocks.NewMockPipeline(suite.T())
		mockPipe.EXPECT().Shader(shader.ShaderTypeVertex).Return(renderShdrMock).Twice()
		mockPipe.EXPECT().Shader(shader.ShaderTypeFragment).Return(nil).Twice()
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(3)
		animMock.EXPECT().Model().Return(mockModel).Twice()
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		suite.rendererMock.EXPECT().Pipeline("k_dirty").Return(mockPipe).Twice()
		suite.rendererMock.EXPECT().DrawCall("k_dirty", meshBGP, uint32(1), mock.Anything).Return(nil).Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NoError(suite.scene.DrawCalls())
		suite.NotNil(suite.scene.drawBindGroupCache)
	})
}

func (suite *sceneImplTest) TestPrepareShadowsSecondLoopGuards() {
	suite.Run("point cube face second loop skips zero instance count animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		goodMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		goodOutputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		goodModel := model_mocks.NewMockModel(suite.T())
		goodModel.EXPECT().CastsShadows().Return(true).Times(14)
		goodModel.EXPECT().LODCount().Return(1).Times(6)
		goodModel.EXPECT().LODMeshProvider(0).Return(goodMeshBGP).Times(6)
		goodModel.EXPECT().Skinned().Return(false).Times(6)
		goodModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		goodModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Times(12)
		goodModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Times(12)
		goodModel.EXPECT().BoundingRadius().Return(float32(1000)).Times(12)
		goodAnim := animator_mocks.NewMockAnimator(suite.T())
		goodAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		goodAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		goodAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(14)
		goodAnim.EXPECT().Model().Return(goodModel).Times(20)
		goodAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{1, 1, 1}).Once()
		goodAnim.EXPECT().OutputBindGroupProvider().Return(goodOutputBGP).Maybe()

		badAnim := animator_mocks.NewMockAnimator(suite.T())
		badAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		badAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		badAnim.EXPECT().InstanceCount().Return(uint32(0)).Times(8)

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {goodAnim, badAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face second loop skips nil model animator", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		goodMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		goodOutputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		goodModel := model_mocks.NewMockModel(suite.T())
		goodModel.EXPECT().CastsShadows().Return(true).Times(14)
		goodModel.EXPECT().LODCount().Return(1).Times(6)
		goodModel.EXPECT().LODMeshProvider(0).Return(goodMeshBGP).Times(6)
		goodModel.EXPECT().Skinned().Return(false).Times(6)
		goodModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		goodModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Times(12)
		goodModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Times(12)
		goodModel.EXPECT().BoundingRadius().Return(float32(1000)).Times(12)
		goodAnim := animator_mocks.NewMockAnimator(suite.T())
		goodAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		goodAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		goodAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(14)
		goodAnim.EXPECT().Model().Return(goodModel).Times(20)
		goodAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{1, 1, 1}).Once()
		goodAnim.EXPECT().OutputBindGroupProvider().Return(goodOutputBGP).Maybe()

		badAnim := animator_mocks.NewMockAnimator(suite.T())
		badAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		badAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		badAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(8)
		badAnim.EXPECT().Model().Return(nil).Times(8)

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {goodAnim, badAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face second loop skips out of range instance", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(20)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		goodMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		goodOutputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		goodModel := model_mocks.NewMockModel(suite.T())
		goodModel.EXPECT().CastsShadows().Return(true).Times(14)
		goodModel.EXPECT().LODCount().Return(1).Times(6)
		goodModel.EXPECT().LODMeshProvider(0).Return(goodMeshBGP).Times(6)
		goodModel.EXPECT().Skinned().Return(false).Times(6)
		goodModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		goodModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Times(12)
		goodModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Times(12)
		goodModel.EXPECT().BoundingRadius().Return(float32(1000)).Times(12)
		goodAnim := animator_mocks.NewMockAnimator(suite.T())
		goodAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		goodAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		goodAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(14)
		goodAnim.EXPECT().Model().Return(goodModel).Times(20)
		goodAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{1, 1, 1}).Once()
		goodAnim.EXPECT().OutputBindGroupProvider().Return(goodOutputBGP).Maybe()

		badMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		badModel := model_mocks.NewMockModel(suite.T())
		badModel.EXPECT().CastsShadows().Return(true).Times(8)
		badModel.EXPECT().LODCount().Return(1).Times(6)
		badModel.EXPECT().LODMeshProvider(0).Return(badMeshBGP).Times(6)
		badModel.EXPECT().Skinned().Return(false).Times(6)
		badModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		badModel.EXPECT().BoundingMin().Return([3]float32{-0.5, -0.5, -0.5}).Times(6)
		badModel.EXPECT().BoundingMax().Return([3]float32{0.5, 0.5, 0.5}).Times(6)
		badModel.EXPECT().BoundingRadius().Return(float32(0.5)).Times(6)
		badAnim := animator_mocks.NewMockAnimator(suite.T())
		badAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		badAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		badAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(8)
		badAnim.EXPECT().Model().Return(badModel).Times(14)
		badAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{1000, 0, 0}, [3]float32{1, 1, 1}).Once()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {goodAnim, badAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})

	suite.Run("point cube face second loop skips instance not visible in frustum", func() {
		suite.scene.lightHandler.SetEnabled(true)
		sh := suite.scene.lightHandler.ShadowHandler()
		sh.SetLightShadowAtlasSlots(6)
		sh.SetLightShadowAtlasCols(6)
		sh.SetPipelineKey("shadow_static_back", "shadow_static_back")

		pl := light_mocks.NewMockLight(suite.T())
		pl.EXPECT().Enabled().Return(true).Maybe()
		pl.EXPECT().CastsShadows().Return(true).Maybe()
		pl.EXPECT().Type().Return(light.LightTypePoint).Maybe()
		pl.EXPECT().Position().Return([3]float32{0, 5, 0}).Maybe()
		pl.EXPECT().Range().Return(float32(1000)).Maybe()
		pl.EXPECT().ShadowBias().Return(float32(0.005)).Maybe()
		pl.EXPECT().InnerCone().Return(float32(0)).Maybe()
		pl.EXPECT().OuterCone().Return(float32(0)).Maybe()
		pl.EXPECT().Direction().Return([3]float32{}).Maybe()
		suite.scene.lightHandler.AddLight(pl)

		goodMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		goodOutputBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		goodModel := model_mocks.NewMockModel(suite.T())
		goodModel.EXPECT().CastsShadows().Return(true).Times(14)
		goodModel.EXPECT().LODCount().Return(1).Times(6)
		goodModel.EXPECT().LODMeshProvider(0).Return(goodMeshBGP).Times(6)
		goodModel.EXPECT().Skinned().Return(false).Times(6)
		goodModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		goodModel.EXPECT().BoundingMin().Return([3]float32{-1000, -1000, -1000}).Times(12)
		goodModel.EXPECT().BoundingMax().Return([3]float32{1000, 1000, 1000}).Times(12)
		goodModel.EXPECT().BoundingRadius().Return(float32(1000)).Times(12)
		goodAnim := animator_mocks.NewMockAnimator(suite.T())
		goodAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		goodAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		goodAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(14)
		goodAnim.EXPECT().Model().Return(goodModel).Times(20)
		goodAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{0, 0, 0}, [3]float32{1, 1, 1}).Once()
		goodAnim.EXPECT().OutputBindGroupProvider().Return(goodOutputBGP).Maybe()

		badMeshBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		badModel := model_mocks.NewMockModel(suite.T())
		badModel.EXPECT().CastsShadows().Return(true).Times(8)
		badModel.EXPECT().LODCount().Return(1).Times(6)
		badModel.EXPECT().LODMeshProvider(0).Return(badMeshBGP).Times(6)
		badModel.EXPECT().Skinned().Return(false).Times(6)
		badModel.EXPECT().ShadowCullMode().Return(model.ShadowCullModeBack).Times(6)
		badModel.EXPECT().BoundingMin().Return([3]float32{-0.01, -0.01, -0.01}).Times(6)
		badModel.EXPECT().BoundingMax().Return([3]float32{0.01, 0.01, 0.01}).Times(6)
		badModel.EXPECT().BoundingRadius().Return(float32(0.01)).Times(6)
		badAnim := animator_mocks.NewMockAnimator(suite.T())
		badAnim.EXPECT().BackendType().Return(animator.BackendTypeSimple).Maybe()
		badAnim.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Maybe()
		badAnim.EXPECT().InstanceCount().Return(uint32(1)).Times(8)
		badAnim.EXPECT().Model().Return(badModel).Times(14)
		badAnim.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{0, 5, 0}, [3]float32{1, 1, 1}).Once()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {goodAnim, badAnim}}

		suite.rendererMock.EXPECT().WriteBuffers(mock.Anything).Return().Maybe()
		suite.rendererMock.EXPECT().BeginShadowFrame().Return().Once()
		suite.rendererMock.EXPECT().BeginShadowAtlasPass(mock.Anything).Return().Once()
		suite.rendererMock.EXPECT().SetShadowViewport(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return().Times(6)
		suite.rendererMock.EXPECT().EndShadowAtlasPass().Return().Once()
		suite.rendererMock.EXPECT().EndShadowFrame().Return().Once()

		suite.NotPanics(func() { suite.scene.PrepareShadows() })
	})
}

func (suite *sceneImplTest) TestPrepareComputeLODRemainingBranches() {
	suite.Run("phase1 lodEnabled lod1 distance branch", func() {
		suite.scene.cullingDisabled = true
		suite.scene.lodEnabled = true
		suite.scene.lod1Distance = 10.0
		suite.scene.lod2Distance = 30.0
		suite.scene.lodLevelCache = make(map[animator.Animator]int)
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)
		suite.scene.buildInjectionMap()
		realShdr := shader.NewShader("_pclod_a", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipe := pipeline.NewPipeline("pclod-a-key", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShdr),
		)
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ctrlMock := camera_mocks.NewMockCameraController(suite.T())
		ctrlMock.EXPECT().Position().Return(float32(0), float32(0), float32(0)).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		camMock.EXPECT().Controller().Return(ctrlMock).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("pclod-a-key").Times(2)
		mockModel.EXPECT().LODCount().Return(3).Times(2)
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(8)
		animMock.EXPECT().Model().Return(mockModel).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().SetScreenSize(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().SetProjectionX(mock.Anything).Return().Once()
		animMock.EXPECT().SetHiZMipCount(mock.Anything).Return().Once()
		animMock.EXPECT().SetViewProj(mock.Anything).Return().Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(1)).Once()
		animMock.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{15, 0, 0}, [3]float32{1, 1, 1}).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("pclod-a-key").Return(realPipe).Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "pclod-a-key" && d[0].Providers[0].Provider == computeBGP
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
		suite.Contains(suite.scene.lodLevelCache, animMock)
		suite.Equal(1, suite.scene.lodLevelCache[animMock])
	})

	suite.Run("phase1 lodEnabled lod2 distance branch and maxLevel clamp", func() {
		suite.scene.cullingDisabled = true
		suite.scene.lodEnabled = true
		suite.scene.lod1Distance = 10.0
		suite.scene.lod2Distance = 30.0
		suite.scene.lodLevelCache = make(map[animator.Animator]int)
		suite.scene.computePool = worker.NewDynamicWorkerPool(1, 256, 1*time.Second)
		suite.scene.buildInjectionMap()
		realShdr := shader.NewShader("_pclod_b", shader.ShaderTypeCompute,
			"engine/renderer/animator/assets/simple-compute.wgsl",
			shader.WithInjections(suite.scene.injections),
		)
		realPipe := pipeline.NewPipeline("pclod-b-key", pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(realShdr),
		)
		computeBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ctrlMock := camera_mocks.NewMockCameraController(suite.T())
		ctrlMock.EXPECT().Position().Return(float32(0), float32(0), float32(0)).Once()
		camMock := camera_mocks.NewMockCamera(suite.T())
		camMock.EXPECT().Update().Return().Once()
		camMock.EXPECT().ViewProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().ProjectionMatrix().Return([16]float32{}).Once()
		camMock.EXPECT().BindGroupProvider().Return(nil).Once()
		camMock.EXPECT().Controller().Return(ctrlMock).Once()
		suite.scene.cam = camMock
		suite.scene.writePool = []bind_group_provider.BufferWrite{}
		mockModel := model_mocks.NewMockModel(suite.T())
		mockModel.EXPECT().ComputePipelineKey().Return("pclod-b-key").Times(2)
		mockModel.EXPECT().LODCount().Return(2).Times(2)
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Lifecycle().Return(lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))).Times(4)
		animMock.EXPECT().InstanceCount().Return(uint32(1)).Times(8)
		animMock.EXPECT().Model().Return(mockModel).Times(2)
		animMock.EXPECT().CullingEnabled().Return(false).Once()
		animMock.EXPECT().StagedWriteData().Return(nil).Once()
		animMock.EXPECT().SetScreenSize(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().SetProjectionX(mock.Anything).Return().Once()
		animMock.EXPECT().SetHiZMipCount(mock.Anything).Return().Once()
		animMock.EXPECT().SetViewProj(mock.Anything).Return().Once()
		animMock.EXPECT().PrepareFrame(mock.Anything, mock.Anything).Return().Once()
		animMock.EXPECT().Flush(mock.Anything, mock.Anything, mock.Anything).Return(uint32(1)).Once()
		animMock.EXPECT().InstanceTransform(uint32(0)).Return([3]float32{35, 0, 0}, [3]float32{1, 1, 1}).Once()
		animMock.EXPECT().ComputeBindGroupProvider().Return(computeBGP).Once()
		animMock.EXPECT().HiZBindGroupProvider().Return(nil).Once()
		suite.rendererMock.EXPECT().Pipeline("pclod-b-key").Return(realPipe).Times(2)
		suite.rendererMock.EXPECT().DispatchComputeBatch(mock.MatchedBy(func(d []renderer.ComputeDispatch) bool {
			return len(d) == 1 && d[0].PipelineKey == "pclod-b-key" && d[0].Providers[0].Provider == computeBGP
		})).Return().Once()
		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.PrepareCompute(0.016) })
		suite.Contains(suite.scene.lodLevelCache, animMock)
		suite.Equal(1, suite.scene.lodLevelCache[animMock])
	})
}

func (suite *sceneImplTest) TestLifecycle() {
	suite.Run("returns scene lifecycle", func() {
		lc := lifecycle.NewLifecycle()
		s := &scene{mu: &sync.RWMutex{}, lc: lc}
		suite.Equal(lc, s.Lifecycle())
	})
}

func (suite *sceneImplTest) TestWithCullingDisabled() {
	suite.Run("sets cullingDisabled true", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithCullingDisabled(true)
		opt(s)
		suite.Equal(true, s.cullingDisabled)
	})

	suite.Run("sets cullingDisabled false", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithCullingDisabled(false)
		opt(s)
		suite.Equal(false, s.cullingDisabled)
	})
}

func (suite *sceneImplTest) TestWithLODEnabled() {
	suite.Run("sets lodEnabled true", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithLODEnabled(true)
		opt(s)
		suite.Equal(true, s.lodEnabled)
	})

	suite.Run("sets lodEnabled false", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithLODEnabled(false)
		opt(s)
		suite.Equal(false, s.lodEnabled)
	})
}

func (suite *sceneImplTest) TestWithLODDistances() {
	suite.Run("sets lod1Distance and lod2Distance", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithLODDistances(10.0, 30.0)
		opt(s)
		suite.Equal(float32(10.0), s.lod1Distance)
		suite.Equal(float32(30.0), s.lod2Distance)
	})
}

func (suite *sceneImplTest) TestWithLODShadowBias() {
	suite.Run("sets lodShadowBias", func() {
		s := &scene{mu: &sync.RWMutex{}}
		opt := WithLODShadowBias(2)
		opt(s)
		suite.Equal(2, s.lodShadowBias)
	})
}

func (suite *sceneImplTest) TestPruneAnimatorMarkAllDirty() {
	suite.Run("calls MarkAllDirty on non-nil ShadowHandler", func() {
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		shMock.EXPECT().MarkAllDirty().Once()
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		lhMock.EXPECT().ShadowHandler().Return(shMock).Once()
		suite.scene.lightHandler = lhMock
		animMock := animator_mocks.NewMockAnimator(suite.T())
		animMock.EXPECT().Model().Return(nil).Once()
		animMock.EXPECT().Release().Once()
		suite.NotPanics(func() { suite.scene.pruneAnimator(animMock) })
	})
}

func (suite *sceneImplTest) TestRefreshAnimatorHiZBindGroupsNilLightHandler() {
	suite.Run("returns immediately when lightHandler is nil", func() {
		suite.scene.lightHandler = nil
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})
}

func (suite *sceneImplTest) TestRefreshAnimatorHiZBindGroupsMaxViewNilPaths() {
	suite.Run("skips slot when maxHizView is nil and hizFallbackView is nil", func() {
		suite.scene.hizFallbackView = nil

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(&wgpu.TextureView{}).Maybe()
		ssrMock.EXPECT().HiZMaxTextureView().Return(nil).Maybe()
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock

		hiZBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		hiZBGPMock.EXPECT().SetSlot(mock.Anything).Maybe()

		computeShdrMock := shader_mocks.NewMockShader(suite.T())
		computeShdrMock.EXPECT().BindGroupLayoutDescriptor(1).Return(wgpu.BindGroupLayoutDescriptor{}).Once()

		pipeMock := pipeline_mocks.NewMockPipeline(suite.T())
		pipeMock.EXPECT().Shader(shader.ShaderTypeCompute).Return(computeShdrMock).Once()

		animMock := animator_mocks.NewMockAnimator(suite.T())
		mdlMock := model_mocks.NewMockModel(suite.T())
		animMock.EXPECT().HiZBindGroupProvider().Return(hiZBGPMock).Once()
		animMock.EXPECT().Model().Return(mdlMock).Once()
		mdlMock.EXPECT().ComputePipelineKey().Return("hiz_maxnil_compute").Once()
		suite.rendererMock.EXPECT().Pipeline("hiz_maxnil_compute").Return(pipeMock).Once()

		mapKey := model_mocks.NewMockModel(suite.T())
		suite.scene.animatorPool = map[model.Model][]animator.Animator{mapKey: {animMock}}
		suite.NotPanics(func() { suite.scene.refreshAnimatorHiZBindGroups() })
	})
}

func (suite *sceneImplTest) TestCreateAnimatorLODMeshInitPanics() {
	suite.Run("panics when InitMeshBuffers fails for LOD level 1", func() {
		mdl := model_mocks.NewMockModel(suite.T())
		cs := shader_mocks.NewMockShader(suite.T())
		vs := shader_mocks.NewMockShader(suite.T())
		fs := shader_mocks.NewMockShader(suite.T())

		mdl.EXPECT().Skinned().Return(false).Maybe()
		mdl.EXPECT().BoundingRadius().Return(float32(1.0)).Once()
		mdl.EXPECT().BoundingMin().Return([3]float32{}).Once()
		mdl.EXPECT().BoundingMax().Return([3]float32{}).Once()
		mdl.EXPECT().MeshProvider().Return(nil).Once()
		mdl.EXPECT().LODCount().Return(2).Once()
		lodBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lodBGP.EXPECT().VertexBuffer().Return(nil).Once()
		mdl.EXPECT().LODMeshProvider(1).Return(lodBGP).Once()
		mdl.EXPECT().LODVertexData(1).Return([]byte{1}).Once()
		mdl.EXPECT().LODIndexData(1).Return([]byte{1}).Once()
		mdl.EXPECT().LODIndexCount(1).Return(3).Once()
		mdl.EXPECT().Name().Return("test-lod-model").Maybe()

		cs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()
		vs.EXPECT().Declarations().Return([]shader.Annotation{}).Maybe()

		suite.rendererMock.EXPECT().InitMeshBuffers(lodBGP, []byte{1}, []byte{1}, 3).Return(errors.New("lod mesh fail")).Once()

		suite.Panics(func() { startAnimator(suite.scene.createAnimator(mdl, cs, vs, fs)) })
	})
}
func (suite *sceneImplTest) TestReleaseResolutionDependentResourcesBGPPaths() {
	suite.Run("sets all bgp bind groups to nil for all handlers enabled", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		cshMock := light_mocks.NewMockContactShadowHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())

		suite.scene.gBufferHandler = gbhMock
		suite.scene.ssaoHandler = ssaoMock
		lhMock.EXPECT().ContactShadowHandler().Return(cshMock).Maybe()
		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		suite.scene.compositionHandler = chMock
		suite.scene.ssrHandler = ssrMock

		gbhMock.EXPECT().Enabled().Return(false).Maybe()

		ssaoMock.EXPECT().Enabled().Return(true).Maybe()
		ssaoMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssaoMock.EXPECT().RawTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().RawTexture().Return(nil).Maybe()
		ssaoMock.EXPECT().SetRawTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetRawTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().BlurredTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().BlurredTexture().Return(nil).Maybe()
		ssaoMock.EXPECT().SetBlurredTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetBlurredTexture(mock.Anything).Maybe()
		ssaoMock.EXPECT().ScratchTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().ScratchTexture().Return(nil).Maybe()
		ssaoMock.EXPECT().SetScratchTextureView(mock.Anything).Maybe()
		ssaoMock.EXPECT().SetScratchTexture(mock.Anything).Maybe()
		ssaoBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssaoBGP.EXPECT().BindGroup().Return(nil).Once()
		ssaoBGP.EXPECT().SetBindGroup(mock.Anything).Once()
		ssaoMock.EXPECT().Bgp("ssao_compute").Return(ssaoBGP).Once()
		ssaoMock.EXPECT().Bgp("ssao_blur_h").Return(nil).Maybe()
		ssaoMock.EXPECT().Bgp("ssao_blur_v").Return(nil).Maybe()

		ssaoLitBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssaoLitBGP.EXPECT().BindGroup().Return(nil).Once()
		ssaoLitBGP.EXPECT().SetBindGroup(mock.Anything).Once()
		lhMock.EXPECT().Bgp("ssao_lit").Return(ssaoLitBGP).Maybe()

		cshMock.EXPECT().Enabled().Return(true).Maybe()
		cshMock.EXPECT().SetSlot(mock.Anything).Maybe()
		cshMock.EXPECT().TextureView().Return(nil).Maybe()
		cshMock.EXPECT().Texture().Return(nil).Maybe()
		cshMock.EXPECT().SetTextureView(mock.Anything).Maybe()
		cshMock.EXPECT().SetTexture(mock.Anything).Maybe()
		csBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		csBGP.EXPECT().SetSlot(0).Return().Twice()
		csBGP.EXPECT().SetSlot(1).Return().Once()
		csBGP.EXPECT().BindGroup().Return(nil).Twice()
		csBGP.EXPECT().SetBindGroup(mock.Anything).Twice()
		cshMock.EXPECT().Bgp("contact_shadow_compute").Return(csBGP).Once()

		shMock.EXPECT().CSMAtlasTexture().Return(&wgpu.Texture{}).Maybe()
		csmBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		csmBGP.EXPECT().BindGroup().Return(nil).Once()
		csmBGP.EXPECT().SetBindGroup(mock.Anything).Once()
		shMock.EXPECT().Bgp("csm_shadow_lit").Return(csmBGP).Once()

		chMock.EXPECT().Enabled().Return(true).Maybe()
		chMock.EXPECT().BloomMipCount().Return(0).Maybe()
		chMock.EXPECT().SetSlot(mock.Anything).Maybe()
		chMock.EXPECT().HDRTextureView().Return(nil).Maybe()
		chMock.EXPECT().HDRTexture().Return(nil).Maybe()
		chMock.EXPECT().SetHDRTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetHDRTexture(mock.Anything).Maybe()
		chMock.EXPECT().MSAATextureView().Return(nil).Maybe()
		chMock.EXPECT().MSAATexture().Return(nil).Maybe()
		chMock.EXPECT().SetMSAATextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetMSAATexture(mock.Anything).Maybe()
		chMock.EXPECT().DepthTextureView().Return(nil).Maybe()
		chMock.EXPECT().DepthTexture().Return(nil).Maybe()
		chMock.EXPECT().SetDepthTextureView(mock.Anything).Maybe()
		chMock.EXPECT().SetDepthTexture(mock.Anything).Maybe()
		chMock.EXPECT().BloomDownTexture().Return(nil).Maybe()
		chMock.EXPECT().BloomDownReadViews().Return(nil).Maybe()
		chMock.EXPECT().BloomDownStorageViews().Return(nil).Maybe()
		chMock.EXPECT().SetBloomDownTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomDownReadViews(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomDownStorageViews(mock.Anything).Maybe()
		chMock.EXPECT().BloomUpTexture().Return(nil).Maybe()
		chMock.EXPECT().BloomUpReadViews().Return(nil).Maybe()
		chMock.EXPECT().BloomUpStorageViews().Return(nil).Maybe()
		chMock.EXPECT().BloomUpMip0View().Return(nil).Maybe()
		chMock.EXPECT().SetBloomUpTexture(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomUpReadViews(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomUpStorageViews(mock.Anything).Maybe()
		chMock.EXPECT().SetBloomUpMip0View(mock.Anything).Maybe()
		compBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		compBGP.EXPECT().SetSampler(1, mock.Anything).Once()
		compBGP.EXPECT().SetSampler(3, mock.Anything).Once()
		compBGP.EXPECT().BindGroup().Return(nil).Once()
		compBGP.EXPECT().SetBindGroup(mock.Anything).Once()
		chMock.EXPECT().Bgp("composition").Return(compBGP).Once()
		lumBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		lumBGP.EXPECT().BindGroup().Return(nil).Once()
		lumBGP.EXPECT().SetBindGroup(mock.Anything).Once()
		chMock.EXPECT().Bgp("luminance_compute").Return(lumBGP).Once()
		chMock.EXPECT().LinearSampler().Return(nil).Maybe()
		chMock.EXPECT().SetLinearSampler(mock.Anything).Maybe()

		ssrMock.EXPECT().Enabled().Return(true).Maybe()
		ssrMock.EXPECT().HiZMipCount().Return(0).Maybe()
		ssrMock.EXPECT().SetSlot(mock.Anything).Maybe()
		ssrMock.EXPECT().SSRTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().SSRTexture().Return(nil).Maybe()
		ssrMock.EXPECT().SetSSRTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetSSRTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().HiZTextureView().Return(nil).Maybe()
		ssrMock.EXPECT().HiZTexture().Return(nil).Maybe()
		ssrMock.EXPECT().HiZMipReadViews().Return(nil).Maybe()
		ssrMock.EXPECT().HiZStorageViews().Return(nil).Maybe()
		ssrMock.EXPECT().SetHiZTextureView(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZTexture(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZMipReadViews(mock.Anything).Maybe()
		ssrMock.EXPECT().SetHiZStorageViews(mock.Anything).Maybe()
		ssrBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		ssrBGP.EXPECT().BindGroup().Return(nil).Once()
		ssrBGP.EXPECT().SetBindGroup(mock.Anything).Once()
		ssrMock.EXPECT().Bgp("ssr_compute").Return(ssrBGP).Once()
		ssrMock.EXPECT().Bgp("hiz_init").Return(nil).Maybe()
		ssrMock.EXPECT().Bgp("hiz_init_max").Return(nil).Maybe()

		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.releaseResolutionDependentResources() })
	})
}
func (suite *sceneImplTest) TestResizePostProcessingCSMShadowLitBindGroup() {
	suite.Run("calls initCSMShadowLitBindGroup when CSMAtlasTexture is non-nil", func() {
		suite.scene.buildInjectionMap()
		suite.scene.postProcessingInitialized = true
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		cshMock := light_mocks.NewMockContactShadowHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())

		suite.scene.gBufferHandler = gbhMock
		suite.scene.ssaoHandler = ssaoMock
		lhMock.EXPECT().ContactShadowHandler().Return(cshMock).Maybe()
		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		suite.scene.compositionHandler = chMock
		suite.scene.ssrHandler = ssrMock
		lhMock.EXPECT().Bgp("ssao_lit").Return(nil).Maybe()

		gbhMock.EXPECT().Enabled().Return(false).Maybe()
		ssaoMock.EXPECT().Enabled().Return(false).Maybe()
		cshMock.EXPECT().Enabled().Return(false).Maybe()
		chMock.EXPECT().Enabled().Return(false).Maybe()
		ssrMock.EXPECT().Enabled().Return(false).Maybe()

		// CSMAtlasTexture returns non-nil to enter the initCSMShadowLitBindGroup branch.
		shMock.EXPECT().CSMAtlasTexture().Return(&wgpu.Texture{}).Maybe()
		// Release phase: Bgp("csm_shadow_lit") returns nil so the bgp block is skipped (no Release).
		shMock.EXPECT().Bgp("csm_shadow_lit").Return(nil).Maybe()
		// initCSMShadowLitBindGroup: CSMAtlasTextureView returns nil → early return, no panic.
		shMock.EXPECT().CSMAtlasTextureView().Return(nil).Maybe()

		suite.rendererMock.EXPECT().WaitIdle().Return().Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.resizePostProcessing(800, 600) })
	})
}

func (suite *sceneImplTest) TestResizePostProcessingSSRCompositionCombinedPanic() {
	suite.Run("panics when InitBindGroup fails in SSR+Composition combined rebind", func() {
		suite.scene.buildInjectionMap()
		suite.scene.postProcessingInitialized = true
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		cshMock := light_mocks.NewMockContactShadowHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())

		suite.scene.gBufferHandler = gbhMock
		suite.scene.ssaoHandler = ssaoMock
		lhMock.EXPECT().ContactShadowHandler().Return(cshMock).Maybe()
		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		suite.scene.compositionHandler = chMock
		suite.scene.ssrHandler = ssrMock
		lhMock.EXPECT().Bgp("ssao_lit").Return(nil).Maybe()

		ssaoMock.EXPECT().BlurredTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().LinearSampler().Return(nil).Maybe()

		gbhMock.EXPECT().Enabled().Return(false).Maybe()
		ssaoMock.EXPECT().Enabled().Return(false).Maybe()
		cshMock.EXPECT().Enabled().Return(false).Maybe()
		shMock.EXPECT().CSMAtlasTexture().Return(nil).Maybe()

		// chMock.Enabled: false (release), false (initComposition check), true (combined block check)
		chEnabledCount := 0
		chMock.EXPECT().Enabled().RunAndReturn(func() bool {
			chEnabledCount++
			return chEnabledCount >= 3
		}).Maybe()

		// ssrMock.Enabled: false (release), false (initSSR check), true (combined block check)
		ssrEnabledCount := 0
		ssrMock.EXPECT().Enabled().RunAndReturn(func() bool {
			ssrEnabledCount++
			return ssrEnabledCount >= 3
		}).Maybe()

		// Combined block internals — compBGP is non-nil and SSRTextureView is non-nil
		compBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())
		chMock.EXPECT().Bgp("composition").Return(compBGP).Once()
		ssrView := &wgpu.TextureView{}
		ssrMock.EXPECT().SSRTextureView().Return(ssrView).Maybe()
		compBGP.EXPECT().SetTextureView(2, ssrView).Once()

		// Specific failure for compBGP registered BEFORE the catch-all so it wins
		suite.rendererMock.EXPECT().InitBindGroup(compBGP, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("rebind fail")).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		suite.rendererMock.EXPECT().WaitIdle().Return().Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		suite.scene.lightHandler = lhMock
		suite.Panics(func() { suite.scene.resizePostProcessing(800, 600) })
	})

	suite.Run("does not rebind composition when SSR texture view is nil", func() {
		suite.scene.buildInjectionMap()
		suite.scene.postProcessingInitialized = true
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600

		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbhMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		cshMock := light_mocks.NewMockContactShadowHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		chMock := composition_mocks.NewMockHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())
		compBGP := bgp_mocks.NewMockBindGroupProvider(suite.T())

		suite.scene.gBufferHandler = gbhMock
		suite.scene.ssaoHandler = ssaoMock
		lhMock.EXPECT().ContactShadowHandler().Return(cshMock).Maybe()
		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		suite.scene.compositionHandler = chMock
		suite.scene.ssrHandler = ssrMock
		lhMock.EXPECT().Bgp("ssao_lit").Return(nil).Maybe()

		ssaoMock.EXPECT().BlurredTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().LinearSampler().Return(nil).Maybe()

		gbhMock.EXPECT().Enabled().Return(false).Maybe()
		ssaoMock.EXPECT().Enabled().Return(false).Maybe()
		cshMock.EXPECT().Enabled().Return(false).Maybe()
		shMock.EXPECT().CSMAtlasTexture().Return(nil).Maybe()

		chEnabledCount := 0
		chMock.EXPECT().Enabled().RunAndReturn(func() bool {
			chEnabledCount++
			return chEnabledCount >= 3
		}).Maybe()

		ssrEnabledCount := 0
		ssrMock.EXPECT().Enabled().RunAndReturn(func() bool {
			ssrEnabledCount++
			return ssrEnabledCount >= 3
		}).Maybe()

		chMock.EXPECT().Bgp("composition").Return(compBGP).Once()
		ssrMock.EXPECT().SSRTextureView().Return(nil).Once()

		suite.rendererMock.EXPECT().WaitIdle().Return().Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		suite.scene.lightHandler = lhMock
		suite.NotPanics(func() { suite.scene.resizePostProcessing(800, 600) })
		compBGP.AssertNotCalled(suite.T(), "SetTextureView", 2, mock.Anything)
		suite.rendererMock.AssertNotCalled(suite.T(), "InitBindGroup", compBGP, mock.Anything, mock.Anything, mock.Anything)
	})
}

func (suite *sceneImplTest) TestInitTAA() {
	makeBase := func() (*gbuffer_mocks.MockGBufferHandler, *composition_mocks.MockHandler) {
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		compMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.gBufferHandler = gbMock
		suite.scene.compositionHandler = compMock
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		compMock.EXPECT().LuminanceWorkgroupSize().Return(256).Maybe()
		suite.scene.buildInjectionMap()
		gbMock.EXPECT().Enabled().Return(true).Maybe()
		compMock.EXPECT().Enabled().Return(true).Maybe()
		gbMock.EXPECT().SetSlot(mock.Anything).Maybe()
		gbMock.EXPECT().DepthTextureView().Return(&wgpu.TextureView{}).Maybe()
		compMock.EXPECT().SetSlot(mock.Anything).Maybe()
		compMock.EXPECT().HDRTextureView().Return(&wgpu.TextureView{}).Maybe()
		return gbMock, compMock
	}

	makeCreates := func() {
		suite.rendererMock.EXPECT().CreateTAATextures(800, 600).Return(
			&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{},
		).Once()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}).Once()
		suite.rendererMock.EXPECT().CreateSharpenTexture(800, 600).Return(
			&wgpu.TextureView{}, &wgpu.Texture{},
		).Once()
	}

	suite.Run("taaHandler nil returns early", func() {
		suite.scene.taaHandler = nil
		suite.NotPanics(func() { suite.scene.initTAA() })
	})

	suite.Run("gbHandler nil returns early", func() {
		suite.scene.gBufferHandler = nil
		suite.NotPanics(func() { suite.scene.initTAA() })
	})

	suite.Run("compHandler nil returns early", func() {
		suite.scene.compositionHandler = nil
		suite.NotPanics(func() { suite.scene.initTAA() })
	})

	suite.Run("gbHandler not enabled returns early", func() {
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		compMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.gBufferHandler = gbMock
		suite.scene.compositionHandler = compMock
		gbMock.EXPECT().Enabled().Return(false).Once()
		suite.NotPanics(func() { suite.scene.initTAA() })
		_ = compMock
	})

	suite.Run("compHandler not enabled returns early", func() {
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		compMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.gBufferHandler = gbMock
		suite.scene.compositionHandler = compMock
		gbMock.EXPECT().Enabled().Return(true).Once()
		compMock.EXPECT().Enabled().Return(false).Once()
		suite.NotPanics(func() { suite.scene.initTAA() })
	})

	suite.Run("zero screenWidth returns early", func() {
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		compMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.gBufferHandler = gbMock
		suite.scene.compositionHandler = compMock
		gbMock.EXPECT().Enabled().Return(true).Once()
		compMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.screenWidth = 0
		suite.scene.screenHeight = 600
		suite.NotPanics(func() { suite.scene.initTAA() })
	})

	suite.Run("zero screenHeight returns early", func() {
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		compMock := composition_mocks.NewMockHandler(suite.T())
		suite.scene.gBufferHandler = gbMock
		suite.scene.compositionHandler = compMock
		gbMock.EXPECT().Enabled().Return(true).Once()
		compMock.EXPECT().Enabled().Return(true).Once()
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 0
		suite.NotPanics(func() { suite.scene.initTAA() })
	})

	suite.Run("RegisterPipelines resolve error panics", func() {
		makeBase()
		makeCreates()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("resolve pipe fail")).Once()
		suite.Panics(func() { suite.scene.initTAA() })
	})

	suite.Run("RegisterPipelines sharpen error panics", func() {
		makeBase()
		makeCreates()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(errors.New("sharpen pipe fail")).Once()
		suite.Panics(func() { suite.scene.initTAA() })
	})

	suite.Run("InitBindGroup resolve slot 0 error panics", func() {
		makeBase()
		makeCreates()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("bgp0 fail")).Once()
		suite.Panics(func() { suite.scene.initTAA() })
	})

	suite.Run("InitBindGroup resolve slot 1 error panics", func() {
		makeBase()
		makeCreates()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("bgp1 fail")).Once()
		suite.Panics(func() { suite.scene.initTAA() })
	})

	suite.Run("InitBindGroup CAS slot 0 error panics", func() {
		makeBase()
		makeCreates()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("cas0 fail")).Once()
		suite.Panics(func() { suite.scene.initTAA() })
	})

	suite.Run("InitBindGroup CAS slot 1 error panics", func() {
		makeBase()
		makeCreates()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(3)
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("cas1 fail")).Once()
		suite.Panics(func() { suite.scene.initTAA() })
	})

	suite.Run("composition BGP nil returns early after setting enabled", func() {
		_, compMock := makeBase()
		makeCreates()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(4)
		compMock.EXPECT().Bgp("composition").Return(nil).Once()
		suite.NotPanics(func() { suite.scene.initTAA() })
		suite.True(suite.scene.taaHandler.Enabled())
	})

	suite.Run("InitBindGroup composition slot 0 error panics", func() {
		_, compMock := makeBase()
		makeCreates()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(4)
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		compMock.EXPECT().Bgp("composition").Return(compBGPMock).Once()
		compBGPMock.EXPECT().SetSlot(mock.Anything).Maybe()
		compBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("comp0 fail")).Once()
		suite.Panics(func() { suite.scene.initTAA() })
	})

	suite.Run("InitBindGroup composition slot 1 error panics", func() {
		_, compMock := makeBase()
		makeCreates()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(4)
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		compMock.EXPECT().Bgp("composition").Return(compBGPMock).Once()
		compBGPMock.EXPECT().SetSlot(mock.Anything).Maybe()
		compBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Once()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(errors.New("comp1 fail")).Once()
		suite.Panics(func() { suite.scene.initTAA() })
	})

	suite.Run("full happy path sets enabled and rebinds composition", func() {
		_, compMock := makeBase()
		makeCreates()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Twice()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Times(4)
		compBGPMock := bgp_mocks.NewMockBindGroupProvider(suite.T())
		compMock.EXPECT().Bgp("composition").Return(compBGPMock).Once()
		compBGPMock.EXPECT().SetSlot(mock.Anything).Maybe()
		compBGPMock.EXPECT().SetTextureView(mock.Anything, mock.Anything).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Twice()
		suite.NotPanics(func() { suite.scene.initTAA() })
		suite.True(suite.scene.taaHandler.Enabled())
	})
}

func (suite *sceneImplTest) TestReleaseResolutionDependentResourcesTAA() {
	suite.Run("releases TAA textures and bind groups when enabled", func() {
		taaMock := taa_mocks.NewMockHandler(suite.T())
		suite.scene.taaHandler = taaMock
		taaMock.EXPECT().Enabled().Return(true).Once()
		taaMock.EXPECT().SetSlot(mock.Anything).Maybe()
		taaMock.EXPECT().TAATextureView().Return(nil).Maybe()
		taaMock.EXPECT().TAATexture().Return(nil).Maybe()
		taaMock.EXPECT().SetTAATextureView(mock.Anything).Maybe()
		taaMock.EXPECT().SetTAATexture(mock.Anything).Maybe()
		taaMock.EXPECT().Bgp(mock.Anything).Return(nil).Maybe()
		taaMock.EXPECT().SharpenTextureView().Return(nil).Maybe()
		taaMock.EXPECT().SharpenTexture().Return(nil).Maybe()
		taaMock.EXPECT().SetSharpenTextureView(mock.Anything).Maybe()
		taaMock.EXPECT().SetSharpenTexture(mock.Anything).Maybe()
		suite.NotPanics(func() { suite.scene.releaseResolutionDependentResources() })
	})

	suite.Run("enters non-nil BGP branch and calls SetBindGroup nil", func() {
		taaMock := taa_mocks.NewMockHandler(suite.T())
		suite.scene.taaHandler = taaMock
		taaMock.EXPECT().Enabled().Return(true).Once()
		taaMock.EXPECT().SetSlot(mock.Anything).Maybe()
		taaMock.EXPECT().TAATextureView().Return(nil).Maybe()
		taaMock.EXPECT().TAATexture().Return(nil).Maybe()
		taaMock.EXPECT().SetTAATextureView(mock.Anything).Maybe()
		taaMock.EXPECT().SetTAATexture(mock.Anything).Maybe()
		taaMock.EXPECT().SharpenTextureView().Return(nil).Maybe()
		taaMock.EXPECT().SharpenTexture().Return(nil).Maybe()
		taaMock.EXPECT().SetSharpenTextureView(mock.Anything).Maybe()
		taaMock.EXPECT().SetSharpenTexture(mock.Anything).Maybe()

		resolveBGP0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		resolveBGP1 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sharpenBGP0 := bgp_mocks.NewMockBindGroupProvider(suite.T())
		sharpenBGP1 := bgp_mocks.NewMockBindGroupProvider(suite.T())

		resolveBGP0.EXPECT().BindGroup().Return(nil).Once()
		resolveBGP0.EXPECT().SetBindGroup(mock.Anything).Once()
		resolveBGP1.EXPECT().BindGroup().Return(nil).Once()
		resolveBGP1.EXPECT().SetBindGroup(mock.Anything).Once()
		sharpenBGP0.EXPECT().BindGroup().Return(nil).Once()
		sharpenBGP0.EXPECT().SetBindGroup(mock.Anything).Once()
		sharpenBGP1.EXPECT().BindGroup().Return(nil).Once()
		sharpenBGP1.EXPECT().SetBindGroup(mock.Anything).Once()

		taaMock.EXPECT().Bgp("taa_resolve_0").Return(resolveBGP0).Once()
		taaMock.EXPECT().Bgp("taa_resolve_1").Return(resolveBGP1).Once()
		taaMock.EXPECT().Bgp("taa_sharpen_0").Return(sharpenBGP0).Once()
		taaMock.EXPECT().Bgp("taa_sharpen_1").Return(sharpenBGP1).Once()

		suite.NotPanics(func() { suite.scene.releaseResolutionDependentResources() })
	})
}

func (suite *sceneImplTest) TestResizePostProcessingTAAEnabled() {
	suite.Run("calls initTAA when taaHandler is enabled", func() {
		lhMock := light_mocks.NewMockLightingHandler(suite.T())
		gbMock := gbuffer_mocks.NewMockGBufferHandler(suite.T())
		ssaoMock := ssao_mocks.NewMockHandler(suite.T())
		cshMock := light_mocks.NewMockContactShadowHandler(suite.T())
		shMock := light_mocks.NewMockShadowHandler(suite.T())
		compMock := composition_mocks.NewMockHandler(suite.T())
		ssrMock := ssr_mocks.NewMockHandler(suite.T())

		suite.scene.gBufferHandler = gbMock
		suite.scene.ssaoHandler = ssaoMock
		suite.scene.compositionHandler = compMock
		suite.scene.ssrHandler = ssrMock
		suite.scene.lightHandler = lhMock
		suite.scene.taaHandler = taa.NewHandler()
		suite.scene.taaHandler.SetEnabled(true)
		suite.scene.screenWidth = 800
		suite.scene.screenHeight = 600
		compMock.EXPECT().LuminanceWorkgroupSize().Return(256).Maybe()
		lhMock.EXPECT().TileSize().Return(16).Maybe()
		lhMock.EXPECT().MaxLightsPerTile().Return(64).Maybe()
		ssaoMock.EXPECT().MaxSamples().Return(16).Maybe()
		lhMock.EXPECT().ShadowHandler().Return(shMock).Maybe()
		shMock.EXPECT().PCFSamples().Return(16).Maybe()
		shMock.EXPECT().PCFSamplesSpot().Return(16).Maybe()
		suite.scene.buildInjectionMap()
		suite.scene.postProcessingInitialized = true

		lhMock.EXPECT().ContactShadowHandler().Return(cshMock).Maybe()
		lhMock.EXPECT().Bgp("ssao_lit").Return(nil).Maybe()

		gbEnabledCount := 0
		gbMock.EXPECT().Enabled().RunAndReturn(func() bool {
			gbEnabledCount++
			return gbEnabledCount >= 3
		}).Maybe()
		ssaoMock.EXPECT().Enabled().Return(false).Maybe()
		cshMock.EXPECT().Enabled().Return(false).Maybe()
		shMock.EXPECT().CSMAtlasTexture().Return(nil).Maybe()
		compEnabledCount := 0
		compMock.EXPECT().Enabled().RunAndReturn(func() bool {
			compEnabledCount++
			return compEnabledCount >= 3
		}).Maybe()
		ssrMock.EXPECT().Enabled().Return(false).Maybe()

		ssaoMock.EXPECT().BlurredTextureView().Return(nil).Maybe()
		ssaoMock.EXPECT().LinearSampler().Return(nil).Maybe()

		gbMock.EXPECT().SetSlot(mock.Anything).Maybe()
		gbMock.EXPECT().DepthTextureView().Return(&wgpu.TextureView{}).Maybe()
		compMock.EXPECT().SetSlot(mock.Anything).Maybe()
		compMock.EXPECT().HDRTextureView().Return(&wgpu.TextureView{}).Maybe()
		compMock.EXPECT().Bgp("composition").Return(nil).Maybe()

		suite.rendererMock.EXPECT().WaitIdle().Return().Once()
		suite.rendererMock.EXPECT().InitTextureView(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitSampler(mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().CreateTAATextures(800, 600).Return(
			&wgpu.TextureView{}, &wgpu.Texture{}, &wgpu.TextureView{}, &wgpu.Texture{},
		).Once()
		suite.rendererMock.EXPECT().CreateLinearSampler().Return(&wgpu.Sampler{}).Once()
		suite.rendererMock.EXPECT().CreateSharpenTexture(800, 600).Return(
			&wgpu.TextureView{}, &wgpu.Texture{},
		).Once()
		suite.rendererMock.EXPECT().RegisterPipelines(mock.Anything).Return(nil).Maybe()
		suite.rendererMock.EXPECT().InitBindGroup(mock.Anything, mock.Anything, mock.Anything, mock.Anything).Return(nil).Maybe()

		suite.NotPanics(func() { suite.scene.resizePostProcessing(800, 600) })
		suite.True(suite.scene.taaHandler.Enabled())
	})
}

func (suite *sceneImplTest) TestBuilderOptions() {
	suite.Run("WithCullingDisabled sets cullingDisabled", func() {
		s := &scene{}
		opt := WithCullingDisabled(true)
		opt(s)
		suite.True(s.cullingDisabled)
	})

	suite.Run("WithGBufferHandler sets gBufferHandler", func() {
		s := &scene{}
		h := gbuffer.NewGBufferHandler()
		opt := WithGBufferHandler(h)
		opt(s)
		suite.Equal(h, s.gBufferHandler)
	})

	suite.Run("WithSSAOHandler sets ssaoHandler", func() {
		s := &scene{}
		h := ssao.NewHandler()
		opt := WithSSAOHandler(h)
		opt(s)
		suite.Equal(h, s.ssaoHandler)
	})

	suite.Run("WithCompositionHandler sets compositionHandler", func() {
		s := &scene{}
		h := composition.NewHandler()
		opt := WithCompositionHandler(h)
		opt(s)
		suite.Equal(h, s.compositionHandler)
	})

	suite.Run("WithSSRHandler sets ssrHandler", func() {
		s := &scene{}
		h := ssr.NewHandler()
		opt := WithSSRHandler(h)
		opt(s)
		suite.Equal(h, s.ssrHandler)
	})

	suite.Run("WithScreenSize sets width and height", func() {
		s := &scene{}
		opt := WithScreenSize(1920, 1080)
		opt(s)
		suite.Equal(1920, s.screenWidth)
		suite.Equal(1080, s.screenHeight)
	})

	suite.Run("WithMaxBonesGPU sets the value", func() {
		s := &scene{}
		opt := WithMaxBonesGPU(128)
		opt(s)
		suite.Equal(uint64(128), s.maxBonesGPU)
	})

	suite.Run("WithMaxBonesGPU clamps zero to one", func() {
		s := &scene{}
		opt := WithMaxBonesGPU(0)
		opt(s)
		suite.Equal(uint64(1), s.maxBonesGPU)
	})

	suite.Run("WithLODEnabled sets the field", func() {
		s := &scene{}
		opt := WithLODEnabled(true)
		opt(s)
		suite.True(s.lodEnabled)
	})

	suite.Run("WithLODDistances sets both thresholds", func() {
		s := &scene{}
		opt := WithLODDistances(50.0, 100.0)
		opt(s)
		suite.Equal(float32(50.0), s.lod1Distance)
		suite.Equal(float32(100.0), s.lod2Distance)
	})

	suite.Run("WithLODShadowBias sets the value", func() {
		s := &scene{}
		opt := WithLODShadowBias(2)
		opt(s)
		suite.Equal(2, s.lodShadowBias)
	})
}
