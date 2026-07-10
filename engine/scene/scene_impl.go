package scene

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"

	"github.com/Carmen-Shannon/automation/tools/worker"
	"github.com/oliverbestmann/webgpu/wgpu"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/gbuffer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/composition"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/ssao"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/ssr"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/taa"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
)

// boneParticleUpdateGroup tracks the GPU resources needed to transform a kinematic
// body's particles through the skeletal animator's bone matrices each frame. Each
// kinematic-animated body gets its own group with a BGP that binds the shared
// particle/body buffers from physics and the scratch_matrices buffer from the
// owning Animator.
type boneParticleUpdateGroup struct {
	bgp           bind_group_provider.BindGroupProvider
	particleStart uint32
	particleCount uint32
	boneCount     uint32
	instanceIndex uint32
}

// drawCacheKey uniquely identifies a cached bind group slice for one
// (animator, material pipeline) pair.
type drawCacheKey struct {
	anim        animator.Animator
	pipelineKey string
}

type scene struct {
	mu *sync.RWMutex

	common.DelegateImpl[Scene]

	name string
	lc   lifecycle.Lifecycle

	animatorPool map[model.Model][]animator.Animator
	registry     map[uint64]game_object.GameObject // non-ephemeral objects by ID
	nextID       uint64

	physicsHandler physics.Physics

	// Per-animator sync dispatch state. Each unique Animator that has physics
	// bodies gets its own group with a dedicated sync_map buffer and AnimationData
	// reference. This allows the sync shader to write each body's transform to the
	// correct Animator's AnimationData slot without cross-group interference
	// (bodies not belonging to a group are masked by a sentinel).
	physicsSyncGroup   map[int]bind_group_provider.BindGroupProvider
	physicsSyncAnimMap map[animator.Animator]int         // animator -> group ID in physicsSyncGroup
	physicsSyncWrites  []bind_group_provider.BufferWrite // staged per-group sync_map writes
	physicsAnimBinding int                               // cached AnimationData binding in compute shader (-1 = not resolved)

	// Per-kinematic-body bone particle update state. Each kinematic body with
	// skeletal animation gets a boneParticleUpdateGroup that dispatches the
	// bone_particle_update shader after animator compute to transform particles
	// through the current bone matrices.
	boneParticleUpdateGroups []*boneParticleUpdateGroup
	boneUpdateGPUReady       bool // true once the bone_update pipeline is registered

	cam camera.Camera
	r   renderer.Renderer

	screenWidth  int
	screenHeight int

	cullingDisabled bool // when true, skips frustum plane distribution to animators

	// Lighting subsystem — manages lights, shadow mapping, and Forward+ culling state.
	lightHandler   light.LightingHandler
	gBufferHandler gbuffer.Handler

	// Post-processing handlers — owned by the scene, initialized lazily.
	ssaoHandler        ssao.Handler
	ssrHandler         ssr.Handler
	compositionHandler composition.Handler
	taaHandler         taa.Handler

	lightObjects []game_object.GameObject // objects with attached lights (ephemeral and non-ephemeral)

	// Per-frame spot/point shadow state, populated by PrepareShadows.
	lightShadowEntries []light.GPULightShadowEntry // rebuilt each frame
	lightShadowMap     map[light.Light]uint32      // light → entry index (current frame)
	lightPrevSlotMap   map[light.Light]uint32      // light → entry index (previous frame, for migration detection)

	tileBufferCapacity int // number of tiles the tile GPU buffers were sized for

	// Pre-allocated slices reused each frame to avoid per-frame allocations.
	writePool              []bind_group_provider.BufferWrite             // reusable coalesced buffer write slice
	drawBindGroupsPool     []bind_group_provider.BindGroupProvider       // reusable bind group slice for DrawCalls
	drawDeclsPool          []shader.Annotation                           // reusable annotations slice for DrawCalls
	drawGroupProvidersPool map[int]bind_group_provider.BindGroupProvider // reusable group-providers map for DrawCalls

	// computePool manages a bounded set of reusable goroutines for the parallel
	// CPU prep phase of PrepareCompute. Workers persist across frames, avoiding
	// per-frame goroutine spawn/teardown overhead.
	computePool    worker.DynamicWorkerPool
	computeWorkers int // stored so we can log/inspect the configured count
	maxBonesGPU    uint64

	// instanceLookup provides O(1) reverse lookup from (Animator, instanceSlot) → objID.
	// Maintained by Add/Remove so the swap-remove fixup in Remove avoids an O(N) registry scan.
	instanceLookup map[animator.Animator]map[uint32]uint64

	// shadowIndirectBuffers holds a dedicated DrawIndexedIndirect argument buffer for each
	// animator. Written by the CPU pre-pass in PrepareShadows and consumed by every shadow
	// depth pass for that animator without going through the compute culling path.
	shadowIndirectBuffers map[animator.Animator]*wgpu.Buffer
	animIndirectBinding   map[animator.Animator]int
	// shadowAnimationProviders binds raw simple AnimationData for non-skinned shadow depth shaders.
	shadowAnimationProviders map[animator.Animator]bind_group_provider.BindGroupProvider

	injections map[string]string

	// hizFallbackView is a 1×1 R32Float placeholder bound to the animator_hiz slot
	// before the real SSR Hi-Z pyramid is available (e.g. during createAnimator).
	hizFallbackTexture *wgpu.Texture
	hizFallbackView    *wgpu.TextureView

	postProcessingInitialized bool

	lodEnabled    bool
	lod1Distance  float32
	lod2Distance  float32
	lodShadowBias int
	lodLevelCache map[animator.Animator]int

	// drawBindGroupCache caches the resolved []BindGroupProvider for each
	// (animator, pipeline) pair. Rebuilt in parallel when drawCacheDirty is true;
	// consumed directly by the serial draw loop each frame.
	drawBindGroupCache map[drawCacheKey][]bind_group_provider.BindGroupProvider
	drawCacheDirty     bool
}

// sceneLifecycleChildren snapshots the current scene lifecycle children.
//
// Returns:
//   - []animator.Animator: snapshot of all registered animators
//   - physics.Physics: current physics handler, if present
func (s *scene) sceneLifecycleChildren() ([]animator.Animator, physics.Physics) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	anims := make([]animator.Animator, 0)
	for _, group := range s.animatorPool {
		for _, a := range group {
			if a != nil {
				anims = append(anims, a)
			}
		}
	}

	return anims, s.physicsHandler
}

// transitionChildLifecycle advances a child lifecycle toward the requested target state
// using only legal lifecycle transitions.
//
// Parameters:
//   - lc: child lifecycle to transition
//   - target: target lifecycle state
//
// Returns:
//   - error: error if any attempted transition fails
func transitionChildLifecycle(lc lifecycle.Lifecycle, target lifecycle.LifecycleState) error {
	if lc == nil {
		return nil
	}

	switch target {
	case lifecycle.LifecycleStateRunning:
		switch lc.State() {
		case lifecycle.LifecycleStateRunning:
			return nil
		case lifecycle.LifecycleStatePaused, lifecycle.LifecycleStateDraining:
			return lc.SetState(lifecycle.LifecycleStateRunning)
		default:
			return nil
		}

	case lifecycle.LifecycleStatePaused:
		if lc.State() == lifecycle.LifecycleStateRunning {
			return lc.SetState(lifecycle.LifecycleStatePaused)
		}
		return nil

	case lifecycle.LifecycleStateDraining:
		switch lc.State() {
		case lifecycle.LifecycleStateDraining:
			return nil
		case lifecycle.LifecycleStateRunning, lifecycle.LifecycleStateErrored:
			return lc.SetState(lifecycle.LifecycleStateDraining)
		default:
			return nil
		}

	case lifecycle.LifecycleStateStopped:
		switch lc.State() {
		case lifecycle.LifecycleStateStopped:
			return nil
		case lifecycle.LifecycleStateRegistered, lifecycle.LifecycleStatePaused, lifecycle.LifecycleStateDraining:
			return lc.SetState(lifecycle.LifecycleStateStopped)
		case lifecycle.LifecycleStateRunning:
			if err := lc.SetState(lifecycle.LifecycleStateDraining); err != nil {
				return err
			}
			if lc.State() == lifecycle.LifecycleStateDraining {
				return lc.SetState(lifecycle.LifecycleStateStopped)
			}
			return nil
		case lifecycle.LifecycleStateErrored:
			if err := lc.SetState(lifecycle.LifecycleStateDraining); err != nil {
				return err
			}
			if lc.State() == lifecycle.LifecycleStateDraining {
				return lc.SetState(lifecycle.LifecycleStateStopped)
			}
			return nil
		default:
			return nil
		}

	case lifecycle.LifecycleStateRemoved:
		if lc.State() == lifecycle.LifecycleStateRemoved {
			return nil
		}
		if err := transitionChildLifecycle(lc, lifecycle.LifecycleStateStopped); err != nil {
			return err
		}
		if lc.State() == lifecycle.LifecycleStateStopped {
			return lc.SetState(lifecycle.LifecycleStateRemoved)
		}
		return nil

	default:
		return nil
	}
}

// transitionSceneChildren transitions all scene child lifecycles to the target state.
//
// Parameters:
//   - target: target lifecycle state
//
// Returns:
//   - error: joined errors from failed child transitions
func (s *scene) transitionSceneChildren(target lifecycle.LifecycleState) error {
	anims, ph := s.sceneLifecycleChildren()

	var errs []error
	for _, a := range anims {
		if a == nil {
			continue
		}
		if err := transitionChildLifecycle(a.Lifecycle(), target); err != nil {
			modelName := "<nil>"
			if mdl := a.Model(); mdl != nil {
				modelName = mdl.Name()
			}
			errs = append(errs, fmt.Errorf("animator %q lifecycle transition to %v failed: %w", modelName, target, err))
		}
	}

	if ph != nil {
		if err := transitionChildLifecycle(ph.Lifecycle(), target); err != nil {
			errs = append(errs, fmt.Errorf("physics lifecycle transition to %v failed: %w", target, err))
		}
	}

	return errors.Join(errs...)
}

// registerLifecycleHooks wires scene lifecycle transitions to child lifecycle fan-out
// and scene resource cleanup.
func (s *scene) registerLifecycleHooks() {
	if s.lc == nil {
		return
	}

	s.lc.OnTransitionTo(lifecycle.LifecycleStateRunning, lifecycle.Hook(func() error {
		return s.transitionSceneChildren(lifecycle.LifecycleStateRunning)
	}))

	s.lc.OnTransitionTo(lifecycle.LifecycleStatePaused, lifecycle.Hook(func() error {
		return s.transitionSceneChildren(lifecycle.LifecycleStatePaused)
	}))

	s.lc.OnTransitionTo(lifecycle.LifecycleStateDraining, lifecycle.Hook(func() error {
		return s.transitionSceneChildren(lifecycle.LifecycleStateDraining)
	}))

	s.lc.OnTransitionTo(lifecycle.LifecycleStateStopped, lifecycle.Hook(func() error {
		return s.transitionSceneChildren(lifecycle.LifecycleStateStopped)
	}))

	s.lc.OnTransitionTo(lifecycle.LifecycleStateRemoved, lifecycle.Hook(func() error {
		childErr := s.transitionSceneChildren(lifecycle.LifecycleStateRemoved)
		cleanupErr := s.releaseSceneResources()
		return errors.Join(childErr, cleanupErr)
	}))
}

// releaseSceneResources releases scene-owned GPU resources for lifecycle Removed.
//
// Returns:
//   - error: joined errors if cleanup fails
func (s *scene) releaseSceneResources() error {
	if s.r != nil {
		s.r.WaitIdle()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.releaseResolutionDependentResources()
	s.releaseLightingResources()
	s.releasePostProcessingResources()
	s.releasePhysicsResources()

	if s.hizFallbackView != nil {
		s.hizFallbackView.Release()
		s.hizFallbackView = nil
	}
	if s.hizFallbackTexture != nil {
		s.hizFallbackTexture.Release()
		s.hizFallbackTexture = nil
	}

	s.lightObjects = nil
	s.lightShadowEntries = nil
	s.lightShadowMap = nil
	s.lightPrevSlotMap = nil
	s.writePool = nil
	s.drawBindGroupsPool = nil
	s.drawDeclsPool = nil
	s.drawGroupProvidersPool = nil
	s.drawBindGroupCache = nil
	s.shadowIndirectBuffers = nil
	s.animIndirectBinding = nil
	s.shadowAnimationProviders = nil
	s.instanceLookup = nil
	s.lodLevelCache = nil
	s.injections = nil
	s.drawCacheDirty = true
	s.postProcessingInitialized = false
	s.tileBufferCapacity = 0

	s.animatorPool = make(map[model.Model][]animator.Animator)
	s.registry = make(map[uint64]game_object.GameObject)
	s.physicsSyncGroup = make(map[int]bind_group_provider.BindGroupProvider)
	s.physicsSyncAnimMap = make(map[animator.Animator]int)

	return nil
}

// releasePhysicsResources releases physics GPU resources owned by the scene and
// clears any shared buffer references before releasing dependent providers.
func (s *scene) releasePhysicsResources() {
	for _, bgp := range s.physicsSyncGroup {
		if bgp == nil {
			continue
		}
		for slot := 0; slot < 2; slot++ {
			bgp.SetSlot(slot)
			bgp.SetBuffer(0, nil)
			bgp.SetBuffer(2, nil)
			bgp.SetBuffer(3, nil)
		}
		bgp.Release()
	}
	s.physicsSyncGroup = make(map[int]bind_group_provider.BindGroupProvider)
	s.physicsSyncAnimMap = make(map[animator.Animator]int)
	s.physicsSyncWrites = nil

	for _, group := range s.boneParticleUpdateGroups {
		if group == nil || group.bgp == nil {
			continue
		}
		for slot := 0; slot < 2; slot++ {
			group.bgp.SetSlot(slot)
			group.bgp.SetBuffer(0, nil)
			group.bgp.SetBuffer(1, nil)
			group.bgp.SetBuffer(2, nil)
			group.bgp.SetBuffer(4, nil)
		}
		group.bgp.Release()
	}
	s.boneParticleUpdateGroups = nil

	ph := s.physicsHandler
	if ph == nil {
		return
	}

	for key, bgp := range ph.Bgps() {
		if bgp == nil {
			continue
		}
		for slot := 0; slot < 2; slot++ {
			bgp.SetSlot(slot)
			bgp.SetBuffers(nil)
		}
		bgp.Release()
		ph.Bgps()[key] = nil
	}

	if buffers := ph.Buffers(); buffers != nil {
		buffers.Release()
	}

	if staging := ph.StagingBuffer(); staging != nil {
		staging.Release()
		ph.SetStagingBuffer(nil)
	}

	s.physicsHandler = nil
}

// releaseLightingResources releases lighting and shadow GPU resources that are not
// handled by resolution-dependent cleanup.
func (s *scene) releaseLightingResources() {
	lh := s.lightHandler
	if lh == nil {
		return
	}

	csHandler := lh.ContactShadowHandler()
	contactEnabled := csHandler != nil && csHandler.Enabled()

	if csHandler != nil {
		for key, bgp := range csHandler.Bgps() {
			if bgp == nil {
				continue
			}
			bgp.SetTextureViews(nil)
			bgp.Release()
			csHandler.SetBgp(key, nil)
		}
	}

	sh := lh.ShadowHandler()
	var csmShadowLit bind_group_provider.BindGroupProvider
	if sh != nil {
		csmShadowLit = sh.Bgp("csm_shadow_lit")
		if csmShadowLit != nil {
			if contactEnabled {
				csmShadowLit.SetTextureView(5, nil)
				csmShadowLit.SetSampler(6, nil)
				if csSampler := csHandler.LinearSampler(); csSampler != nil {
					csSampler.Release()
					csHandler.SetLinearSampler(nil)
				}
			}
			csmShadowLit.Release()
			sh.SetBgp("csm_shadow_lit", nil)
		} else if contactEnabled {
			if csSampler := csHandler.LinearSampler(); csSampler != nil {
				csSampler.Release()
				csHandler.SetLinearSampler(nil)
			}
		}

		for key, bgp := range sh.Bgps() {
			if bgp == nil || key == "csm_shadow_lit" {
				continue
			}
			bgp.Release()
			sh.SetBgp(key, nil)
		}

		if csmShadowLit == nil {
			if view := sh.CSMAtlasTextureView(); view != nil {
				view.Release()
			}
			if view := sh.LightShadowAtlasView(); view != nil {
				view.Release()
			}
			if cmpSampler := sh.ComparisonSampler(); cmpSampler != nil {
				cmpSampler.Release()
			}
		}

		if tex := sh.CSMAtlasTexture(); tex != nil {
			tex.Release()
		}
		if tex := sh.LightShadowAtlas(); tex != nil {
			tex.Release()
		}

		sh.SetCSMAtlasTexture(nil)
		sh.SetCSMAtlasTextureView(nil)
		sh.SetLightShadowAtlas(nil)
		sh.SetLightShadowAtlasView(nil)
		sh.SetComparisonSampler(nil)
	}

	if tileBGP := lh.Bgp("tile_lit"); tileBGP != nil {
		for slot := 0; slot < 2; slot++ {
			tileBGP.SetSlot(slot)
			tileBGP.SetBuffer(1, nil)
			tileBGP.SetBuffer(2, nil)
		}
		tileBGP.Release()
		lh.Bgps()["tile_lit"] = nil
	}

	if cullBGP := lh.Bgp("light_cull"); cullBGP != nil {
		for slot := 0; slot < 2; slot++ {
			cullBGP.SetSlot(slot)
			cullBGP.SetBuffer(1, nil)
		}
		cullBGP.Release()
		lh.Bgps()["light_cull"] = nil
	}

	if lightsBGP := lh.Bgp("lights"); lightsBGP != nil {
		lightsBGP.Release()
		lh.Bgps()["lights"] = nil
	}

	if ssaoLitBGP := lh.Bgp("ssao_lit"); ssaoLitBGP != nil {
		if s.ssaoHandler != nil && s.ssaoHandler.Enabled() {
			ssaoLitBGP.SetTextureViews(nil)
		}
		ssaoLitBGP.Release()
		lh.Bgps()["ssao_lit"] = nil
		if s.ssaoHandler != nil {
			s.ssaoHandler.SetLinearSampler(nil)
		}
	}

	if csHandler != nil {
		csHandler.SetEnabled(false)
		csHandler.SetLinearSampler(nil)
	}

	lh.SetEnabled(false)
}

// releasePostProcessingResources releases post-processing providers and any persistent
// non-resolution resources that remain after resolution-dependent cleanup.
func (s *scene) releasePostProcessingResources() {
	if ch := s.compositionHandler; ch != nil {
		lumBGP := ch.Bgp("luminance_compute")
		if lumBGP != nil {
			lumBGP.SetTextureViews(nil)
			for slot := 0; slot < 2; slot++ {
				lumBGP.SetSlot(slot)
				lumBGP.SetBuffer(2, nil)
			}
			lumBGP.Release()
			ch.SetBgp("luminance_compute", nil)
		}

		compBGP := ch.Bgp("composition")
		if compBGP != nil {
			compBGP.SetTextureView(0, nil)
			if s.ssrHandler != nil && s.ssrHandler.Enabled() {
				compBGP.SetTextureView(2, nil)
			}
			if ch.BloomEnabled() {
				compBGP.SetTextureView(6, nil)
			}
			compBGP.SetSampler(1, nil)
			compBGP.SetSampler(3, nil)
			compBGP.Release()
			ch.SetBgp("composition", nil)
		} else {
			if expBuf := ch.ExposureBuffer(); expBuf != nil {
				expBuf.Release()
			}
		}

		if samp := ch.LinearSampler(); samp != nil {
			samp.Release()
		}

		ch.SetExposureBuffer(nil)
		ch.SetLinearSampler(nil)
		ch.SetEnabled(false)
	}

	if ssaoH := s.ssaoHandler; ssaoH != nil {
		for _, key := range []string{"ssao_compute", "ssao_blur_h", "ssao_blur_v"} {
			bgp := ssaoH.Bgp(key)
			if bgp == nil {
				continue
			}
			bgp.SetTextureViews(nil)
			bgp.Release()
			ssaoH.SetBgp(key, nil)
		}
		ssaoH.SetEnabled(false)
	}

	if ssrH := s.ssrHandler; ssrH != nil {
		for key, bgp := range ssrH.Bgps() {
			if bgp == nil {
				continue
			}
			bgp.SetTextureViews(nil)
			bgp.SetSamplers(nil)
			bgp.Release()
			ssrH.SetBgp(key, nil)
		}
		if samp := ssrH.LinearSampler(); samp != nil {
			samp.Release()
		}
		ssrH.SetLinearSampler(nil)
		ssrH.SetEnabled(false)
	}

	if taaH := s.taaHandler; taaH != nil {
		if samp := taaH.LinearSampler(); samp != nil {
			samp.Release()
		}
		taaH.SetLinearSampler(nil)

		for key, bgp := range taaH.Bgps() {
			if bgp == nil {
				continue
			}
			bgp.SetTextureViews(nil)
			bgp.SetSamplers(nil)
			bgp.Release()
			taaH.SetBgp(key, nil)
		}

		taaH.SetEnabled(false)
	}
}

// shadowPipelineKey resolves the shadow depth pipeline key for the given model
// type and cull mode from the lighting handler's pipeline key map.
func (s *scene) shadowPipelineKey(skinned bool, mode model.ShadowCullMode) string {
	prefix := "shadow_static_"
	if skinned && s.lightHandler.ShadowHandler().PipelineKey("shadow_skinned_back") != "" {
		prefix = "shadow_skinned_"
	}
	tag := "back"
	switch mode {
	case model.ShadowCullModeFront:
		tag = "front"
	case model.ShadowCullModeNone:
		tag = "none"
	}
	return s.lightHandler.ShadowHandler().PipelineKey(prefix + tag)
}

// shadowAnimatorBindGroup returns the animator bind group provider expected by
// the shadow depth pipelines.
//
// Parameters:
//   - a: the animator issuing the shadow draw
//
// Returns:
//   - bind_group_provider.BindGroupProvider: the provider bound at shadow group 1
func (s *scene) shadowAnimatorBindGroup(a animator.Animator) bind_group_provider.BindGroupProvider {
	if a == nil {
		return nil
	}
	if a.BackendType() == animator.BackendTypeSimple {
		return s.shadowAnimationProviders[a]
	}
	return a.OutputBindGroupProvider()
}

// animLODLevel returns the LOD level for the given animator from the per-frame cache.
// Returns 0 (base mesh) when LOD is disabled or the animator has no cached level.
func (s *scene) animLODLevel(a animator.Animator) int {
	if !s.lodEnabled {
		return 0
	}
	if level, ok := s.lodLevelCache[a]; ok {
		return level
	}
	return 0
}

// animShadowLODLevel returns the LOD level for shadow rendering of the given
// animator. Adds lodShadowBias to the base LOD level, clamped to the model's
// maximum available LOD.
func (s *scene) animShadowLODLevel(a animator.Animator) int {
	base := s.animLODLevel(a)
	mdl := a.Model()
	if mdl == nil {
		return base
	}
	level := base + s.lodShadowBias
	maxLevel := mdl.LODCount() - 1
	if level > maxLevel {
		level = maxLevel
	}
	return level
}

// buildInjectionMap builds the injection map for WGSL shader pre-processing,
// including dynamic values from the light handler.
func (s *scene) buildInjectionMap() {
	m := map[string]string{
		"max_bones":              fmt.Sprintf("%du", s.maxBonesGPU),
		"empty_sentinel":         fmt.Sprintf("0x%Xu", uint32(0xFFFFFFFF)),
		"flag_active":            fmt.Sprintf("%du", physics.PhysicsStateActive),
		"flag_static":            fmt.Sprintf("%du", physics.PhysicsStateStatic),
		"flag_kinematic":         fmt.Sprintf("%du", physics.PhysicsStateKinematic),
		"light_type_directional": fmt.Sprintf("%du", light.LightTypeDirectional),
		"light_type_point":       fmt.Sprintf("%du", light.LightTypePoint),
		"light_type_spot":        fmt.Sprintf("%du", light.LightTypeSpot),
	}
	if s.lightHandler != nil {
		ts := s.lightHandler.TileSize()
		m["tile_size"] = fmt.Sprintf("%du", ts)
		m["luminance_workgroup_size"] = fmt.Sprintf("%du", s.compositionHandler.LuminanceWorkgroupSize())
		m["max_lights_per_tile"] = fmt.Sprintf("%du", s.lightHandler.MaxLightsPerTile())
		m["num_threads"] = fmt.Sprintf("%du", ts*ts)
		m["max_ssao_samples"] = fmt.Sprintf("%du", s.ssaoHandler.MaxSamples())
		m["pcf_samples"] = fmt.Sprintf("%du", s.lightHandler.ShadowHandler().PCFSamples())
		m["pcf_samples_spot"] = fmt.Sprintf("%du", s.lightHandler.ShadowHandler().PCFSamplesSpot())
	}
	if s.physicsHandler != nil {
		m["slots_per_cell"] = fmt.Sprintf("%du", s.physicsHandler.SlotsPerCell())
		m["body_idx_mask"] = fmt.Sprintf("0x%Xu", s.physicsHandler.BodyIdxMask())
	}
	s.injections = m
}

// generateSSAOKernel generates a hemisphere sample kernel of the given size
// (clamped to [1, 32]) as a flat byte buffer of array<vec4<f32>, 32> (512 bytes).
// Samples are distributed in a unit hemisphere with an accelerating distribution
// that biases samples closer to the origin.
func (s *scene) generateSSAOKernel(sampleCount int) []byte {
	if sampleCount < 1 {
		sampleCount = 1
	}
	if sampleCount > s.ssaoHandler.MaxSamples() {
		sampleCount = s.ssaoHandler.MaxSamples()
	}

	buf := make([]byte, s.ssaoHandler.MaxSamples()*16) // MaxSamples × vec4<f32>
	off := 0
	for i := 0; i < s.ssaoHandler.MaxSamples(); i++ {
		var x, y, z float32
		if i < sampleCount {
			// Random point in a hemisphere (z >= 0).
			x = rand.Float32()*2.0 - 1.0
			y = rand.Float32()*2.0 - 1.0
			z = rand.Float32() // [0, 1] — hemisphere
			length := float32(math.Sqrt(float64(x*x + y*y + z*z)))
			if length > 0.0001 {
				x /= length
				y /= length
				z /= length
			}
			// Accelerating distribution: samples near the center of the hemisphere
			// are denser, tapering off toward the edges.
			scale := float32(i) / float32(sampleCount)
			scale = 0.1 + scale*scale*0.9
			x *= scale
			y *= scale
			z *= scale
		}
		binary.LittleEndian.PutUint32(buf[off:off+4], math.Float32bits(x))
		binary.LittleEndian.PutUint32(buf[off+4:off+8], math.Float32bits(y))
		binary.LittleEndian.PutUint32(buf[off+8:off+12], math.Float32bits(z))
		binary.LittleEndian.PutUint32(buf[off+12:off+16], 0) // w = 0
		off += 16
	}
	return buf
}

// initSSAO initializes the SSAO subsystem: creates screen-sized occlusion textures,
// registers the SSAO compute and bilateral blur pipelines, and pre-creates all bind
// group providers with correctly-sized GPU buffers. The G-Buffer must be initialized
// before calling this method.
func (s *scene) initSSAO() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.ssaoHandler == nil || s.gBufferHandler == nil {
		return
	}
	if !s.gBufferHandler.Enabled() {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	// When half-resolution is enabled, allocate SSAO textures at half the
	// screen dimensions in each axis (quarter pixel count).
	ssaoW := w
	ssaoHeight := h
	if s.ssaoHandler.HalfResolution() {
		ssaoW = max(w/2, 1)
		ssaoHeight = max(h/2, 1)
	}

	// 1. Create SSAO textures (raw, blurred, scratch at SSAO res).
	rawView, rawTex, blurView, blurTex, scratchView, scratchTex := s.r.CreateSSAOTextures(ssaoW, ssaoHeight)
	s.ssaoHandler.SetRawTexture(rawTex)
	s.ssaoHandler.SetRawTextureView(rawView)
	s.ssaoHandler.SetBlurredTexture(blurTex)
	s.ssaoHandler.SetBlurredTextureView(blurView)
	s.ssaoHandler.SetScratchTexture(scratchTex)
	s.ssaoHandler.SetScratchTextureView(scratchView)

	// 2. Create or reuse linear sampler for the blurred SSAO texture in the lit shader.
	linearSamp := s.r.CreateLinearSampler()
	s.ssaoHandler.SetLinearSampler(linearSamp)

	// 3. Register SSAO compute pipeline.
	ssaoCompShader := shader.NewShader("_ssao_compute", shader.ShaderTypeCompute, "engine/renderer/postprocessing/ssao/assets/ssao-compute.wgsl", shader.WithInjections(s.injections))
	ssaoCompKey := "ssao_compute"
	ssaoCompPipe := pipeline.NewPipeline(ssaoCompKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(ssaoCompShader),
	)
	if err := s.r.RegisterPipelines(ssaoCompPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register SSAO compute pipeline: %v", err))
	}
	s.ssaoHandler.SetPipelineKey("ssao_compute", ssaoCompKey)

	// 4. Register bilateral blur compute pipeline.
	blurCompShader := shader.NewShader("_ssao_blur_compute", shader.ShaderTypeCompute, "engine/renderer/postprocessing/ssao/assets/ssao-blur-compute.wgsl", shader.WithInjections(s.injections))
	blurCompKey := "ssao_blur_compute"
	blurCompPipe := pipeline.NewPipeline(blurCompKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(blurCompShader),
	)
	if err := s.r.RegisterPipelines(blurCompPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register SSAO blur compute pipeline: %v", err))
	}
	s.ssaoHandler.SetPipelineKey("ssao_blur", blurCompKey)

	// 5. Create SSAO compute bind group provider.
	ssaoDesc := ssaoCompShader.BindGroupLayoutDescriptor(0)
	ssaoSizeOverrides := map[int]uint64{
		4: uint64((&ssao.GPUSSAOParams{}).Size()), // ssao_params uniform
		5: 32 * 16,                                // ssao_kernel: array<vec4<f32>, 32> = 512 bytes
	}
	ssaoBGP := s.ssaoHandler.Bgp("ssao_compute")
	ssaoBGP.SetTextureView(0, s.gBufferHandler.DepthTextureView())
	ssaoBGP.SetTextureView(1, s.gBufferHandler.NormalTextureView())
	ssaoBGP.SetTextureView(3, rawView)
	if err := s.r.InitBindGroup(ssaoBGP, ssaoDesc, nil, ssaoSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSAO compute bind group: %v", err))
	}

	// 6. Create blur bind group providers (bilateral blur, depth-aware).
	blurDesc := blurCompShader.BindGroupLayoutDescriptor(0)
	blurSizeOverrides := map[int]uint64{
		2: uint64((&ssao.GPUBlurParams{}).Size()),
	}

	// Horizontal: raw → scratch, depth from G-Buffer hardware depth texture.
	blurHBGP := s.ssaoHandler.Bgp("ssao_blur_h")
	blurHBGP.SetTextureView(0, rawView)
	blurHBGP.SetTextureView(1, scratchView)
	blurHBGP.SetTextureView(3, s.gBufferHandler.DepthTextureView())
	if err := s.r.InitBindGroup(blurHBGP, blurDesc, nil, blurSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSAO blur horizontal bind group: %v", err))
	}

	// Vertical: scratch → blurred, depth from G-Buffer hardware depth texture.
	blurVBGP := s.ssaoHandler.Bgp("ssao_blur_v")
	blurVBGP.SetTextureView(0, scratchView)
	blurVBGP.SetTextureView(1, blurView)
	blurVBGP.SetTextureView(3, s.gBufferHandler.DepthTextureView())
	if err := s.r.InitBindGroup(blurVBGP, blurDesc, nil, blurSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSAO blur vertical bind group: %v", err))
	}

	// 7. Generate hemisphere sample kernel and write to the SSAO compute BGP buffer.
	kernelData := s.generateSSAOKernel(s.ssaoHandler.SampleCount())
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: ssaoBGP, Binding: 5, Offset: 0, Data: kernelData},
	})

	s.ssaoHandler.Resize(w, h)
	s.ssaoHandler.SetEnabled(true)

	// Initialize slot 1 SSAO textures
	ssaoH := s.ssaoHandler
	ssaoH.SetSlot(1)
	rawView1, rawTex1, blurView1, blurTex1, scratchView1, scratchTex1 := s.r.CreateSSAOTextures(ssaoW, ssaoHeight)
	ssaoH.SetRawTexture(rawTex1)
	ssaoH.SetRawTextureView(rawView1)
	ssaoH.SetBlurredTexture(blurTex1)
	ssaoH.SetBlurredTextureView(blurView1)
	ssaoH.SetScratchTexture(scratchTex1)
	ssaoH.SetScratchTextureView(scratchView1)
	ssaoH.SetSlot(0)
}

// initSSAOLitBindGroup creates the SSAO bind group provider used by the lit
// fragment shader at @group(6). When the SSAO subsystem is enabled, the real
// blurred occlusion texture and linear sampler are bound. When SSAO is disabled
// or absent, a 1×1 white fallback texture is created so the shader reads ao=1.0
// (no darkening), keeping the bind group layout valid without any conditional
// branching in the shader.
func (s *scene) initSSAOLitBindGroup(litFragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if litFragmentShader == nil {
		return
	}

	// Resolve the SSAO bind group index from the lit fragment shader's annotations.
	ssaoGroup := -1
	for _, decl := range litFragmentShader.Declarations() {
		if decl.Type == shader.AnnotationTypeProvider && decl.Group != nil && decl.Args[0] == shader.AnnotationArgSSAO {
			ssaoGroup = *decl.Group
			break
		}
	}
	if ssaoGroup < 0 {
		return
	}

	bgp := s.lightHandler.Bgp("ssao_lit")
	desc := litFragmentShader.BindGroupLayoutDescriptor(ssaoGroup)

	// Determine whether SSAO is enabled and the blurred texture is available.
	ssaoReady := s.ssaoHandler.Enabled() &&
		s.ssaoHandler.BlurredTextureView() != nil && s.ssaoHandler.LinearSampler() != nil

	if ssaoReady {
		// Bind the real blurred SSAO texture and linear sampler.
		for _, entry := range desc.Entries {
			binding := int(entry.Binding)
			if entry.Texture.SampleType != wgpu.TextureSampleTypeBindingNotUsed {
				bgp.SetTextureView(binding, s.ssaoHandler.BlurredTextureView())
			}
			if entry.Sampler.Type != wgpu.SamplerBindingTypeBindingNotUsed {
				bgp.SetSampler(binding, s.ssaoHandler.LinearSampler())
			}
		}
	} else {
		// Create a 1×1 white fallback texture (ao=1.0, no darkening).
		for _, entry := range desc.Entries {
			binding := int(entry.Binding)
			if entry.Texture.SampleType != wgpu.TextureSampleTypeBindingNotUsed {
				fallback := common.TextureStagingData{
					Pixels: []byte{255, 255, 255, 255},
					Width:  1,
					Height: 1,
					Linear: true,
				}
				if err := s.r.InitTextureView(bgp, binding, fallback); err != nil {
					panic(fmt.Sprintf("scene: failed to init SSAO fallback texture: %v", err))
				}
			}
			if entry.Sampler.Type != wgpu.SamplerBindingTypeBindingNotUsed {
				fallbackSampler := common.SamplerStagingData{
					AddressModeU:  wgpu.AddressModeClampToEdge,
					AddressModeV:  wgpu.AddressModeClampToEdge,
					AddressModeW:  wgpu.AddressModeClampToEdge,
					MagFilter:     wgpu.FilterModeLinear,
					MinFilter:     wgpu.FilterModeLinear,
					MipmapFilter:  wgpu.MipmapFilterModeLinear,
					LodMinClamp:   0,
					LodMaxClamp:   1,
					MaxAnisotropy: 1,
				}
				if err := s.r.InitSampler(bgp, binding, fallbackSampler); err != nil {
					panic(fmt.Sprintf("scene: failed to init SSAO fallback sampler: %v", err))
				}
			}
		}
	}

	if err := s.r.InitBindGroup(bgp, desc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSAO lit bind group: %v", err))
	}
}

// initGBuffer initializes the G-Buffer MRT textures and registers the static
// and skinned G-Buffer render pipelines. The G-Buffer pre-pass writes per-pixel
// position, normal, and albedo data into screen-sized textures consumed by
// downstream screen-space effects (SSAO, SSR).
func (s *scene) initGBuffer() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.gBufferHandler == nil {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	// Create G-Buffer MRT textures (normal + albedo + depth; position is
	// reconstructed from depth at read time by compute shaders).
	normView, normTex, albView, albTex, depthView, depthTex := s.r.CreateGBufferTextures(w, h)
	s.gBufferHandler.SetNormalTexture(normTex)
	s.gBufferHandler.SetNormalTextureView(normView)
	s.gBufferHandler.SetAlbedoTexture(albTex)
	s.gBufferHandler.SetAlbedoTextureView(albView)
	s.gBufferHandler.SetDepthTexture(depthTex)
	s.gBufferHandler.SetDepthTextureView(depthView)

	// Initialize slot 1
	gbh := s.gBufferHandler
	gbh.SetSlot(1)
	normView1, normTex1, albView1, albTex1, depthView1, depthTex1 := s.r.CreateGBufferTextures(w, h)
	gbh.SetNormalTexture(normTex1)
	gbh.SetNormalTextureView(normView1)
	gbh.SetAlbedoTexture(albTex1)
	gbh.SetAlbedoTextureView(albView1)
	gbh.SetDepthTexture(depthTex1)
	gbh.SetDepthTextureView(depthView1)
	gbh.SetSlot(0)

	// Load shaders for the G-Buffer pass.
	gbufferFrag := shader.NewShader("_gbuffer_frag", shader.ShaderTypeFragment, "engine/renderer/gbuffer/assets/gbuffer-frag.wgsl", shader.WithInjections(s.injections))
	staticVert := shader.NewShader("_gbuffer_static_vert", shader.ShaderTypeVertex, "engine/light/assets/lit-vert.wgsl", shader.WithInjections(s.injections))
	skinnedVert := shader.NewShader("_gbuffer_skinned_vert", shader.ShaderTypeVertex, "engine/light/assets/lit-skinned-vert.wgsl", shader.WithInjections(s.injections))

	// Register static G-Buffer pipeline.
	staticKey := "gbuffer_static"
	staticPipe := pipeline.NewPipeline(staticKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(staticVert),
		pipeline.WithFragmentShader(gbufferFrag),
	)
	if err := s.r.RegisterGBufferPipeline(staticPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register static G-Buffer pipeline: %v", err))
	}
	s.gBufferHandler.SetPipelineKey("static", staticKey)

	// Register skinned G-Buffer pipeline.
	skinnedKey := "gbuffer_skinned"
	skinnedPipe := pipeline.NewPipeline(skinnedKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(skinnedVert),
		pipeline.WithFragmentShader(gbufferFrag),
	)
	if err := s.r.RegisterGBufferPipeline(skinnedPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register skinned G-Buffer pipeline: %v", err))
	}
	s.gBufferHandler.SetPipelineKey("skinned", skinnedKey)

	s.gBufferHandler.Resize(w, h)
	s.gBufferHandler.SetEnabled(true)
}

// initContactShadows initializes the contact shadow subsystem: creates a
// screen-sized R32Float output texture, registers the contact shadow compute
// pipeline, and pre-creates the bind group provider with correctly-sized GPU
// buffers. The G-Buffer must be initialized before calling this method.
func (s *scene) initContactShadows() {
	s.mu.Lock()
	defer s.mu.Unlock()

	csHandler := s.lightHandler.ContactShadowHandler()
	gbh := s.gBufferHandler
	if csHandler == nil || gbh == nil {
		return
	}
	if !gbh.Enabled() {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	// 1. Create contact shadow output texture (full screen resolution).
	csView, csTex := s.r.CreateContactShadowTextures(w, h)
	csHandler.SetTexture(csTex)
	csHandler.SetTextureView(csView)

	// Initialize slot 1
	csHandler.SetSlot(1)
	csView1, csTex1 := s.r.CreateContactShadowTextures(w, h)
	csHandler.SetTexture(csTex1)
	csHandler.SetTextureView(csView1)
	csHandler.SetSlot(0)

	// 2. Create linear sampler for the lit shader to sample the contact shadow texture.
	csLinearSamp := s.r.CreateLinearSampler()
	csHandler.SetLinearSampler(csLinearSamp)

	// 3. Register contact shadow compute pipeline.
	csCompShader := shader.NewShader("_contact_shadow_compute", shader.ShaderTypeCompute, "engine/light/assets/contact-shadow-compute.wgsl", shader.WithInjections(s.injections))
	csCompKey := "contact_shadow_compute"
	csCompPipe := pipeline.NewPipeline(csCompKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(csCompShader),
	)
	if err := s.r.RegisterPipelines(csCompPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register contact shadow compute pipeline: %v", err))
	}
	csHandler.SetPipelineKey("contact_shadow_compute", csCompKey)

	// 4. Create contact shadow compute bind group provider.
	csDesc := csCompShader.BindGroupLayoutDescriptor(0)
	csSizeOverrides := map[int]uint64{
		4: uint64((&light.GPUContactShadowParams{}).Size()),
	}
	csBGP := csHandler.Bgp("contact_shadow_compute")

	// Slot 0 bind group — output texture is csView (slot 0).
	gbh.SetSlot(0)
	csBGP.SetSlot(0)
	csBGP.SetTextureView(0, gbh.DepthTextureView())
	csBGP.SetTextureView(1, gbh.NormalTextureView())
	csBGP.SetTextureView(2, gbh.AlbedoTextureView())
	csBGP.SetTextureView(3, csView)
	if err := s.r.InitBindGroup(csBGP, csDesc, nil, csSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init contact shadow compute bind group slot 0: %v", err))
	}

	// Slot 1 bind group — output texture is csView1 (slot 1).
	gbh.SetSlot(1)
	csBGP.SetSlot(1)
	csBGP.SetTextureView(0, gbh.DepthTextureView())
	csBGP.SetTextureView(1, gbh.NormalTextureView())
	csBGP.SetTextureView(2, gbh.AlbedoTextureView())
	csBGP.SetTextureView(3, csView1)
	if err := s.r.InitBindGroup(csBGP, csDesc, nil, csSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init contact shadow compute bind group slot 1: %v", err))
	}

	// Reset to slot 0.
	gbh.SetSlot(0)
	csBGP.SetSlot(0)

	csHandler.SetEnabled(true)
}

// initLightBindGroup initializes the GPU resources for the light storage buffer
// using the layout descriptor from the given fragment shader's light group.
func (s *scene) initLightBindGroup(fragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if fragmentShader == nil {
		return
	}

	// Resolve the lights bind group index from the shader's pre-processor
	// declarations by matching the LightHeader struct type annotation.
	lightGroup := -1
	for _, decl := range fragmentShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgLightHeader {
			lightGroup = *decl.Group
			break
		}
	}
	if lightGroup < 0 {
		return
	}

	bgp := s.lightHandler.Bgp("lights")

	// Build buffer size overrides: the light storage buffer (binding 1) must hold
	// MaxGPULights entries so it can accommodate dynamic light counts each frame.
	descriptor := fragmentShader.BindGroupLayoutDescriptor(lightGroup)
	sizeOverrides := make(map[int]uint64)
	for _, entry := range descriptor.Entries {
		binding := int(entry.Binding)
		if entry.Buffer.Type == wgpu.BufferBindingTypeReadOnlyStorage || entry.Buffer.Type == wgpu.BufferBindingTypeStorage {
			// Storage buffer: size it for max lights (header is in a separate uniform binding).
			sizeOverrides[binding] = uint64(s.lightHandler.MaxGPULights()) * uint64((&light.GPULight{}).Size())
		}
	}

	if err := s.r.InitBindGroup(bgp, descriptor, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init light bind group: %v", err))
	}
	bgp.SetSlot(1)
	if err := s.r.InitBindGroup(bgp, descriptor, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init light bind group slot 1: %v", err))
	}
	bgp.SetSlot(0)
}

// initShadowMap initializes the CSM atlas depth texture, per-cascade bind group
// providers, and shadow depth render pipelines for PCF shadow mapping.
func (s *scene) initShadowMap(shadowVertShader, shadowSkinnedVertShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if shadowVertShader == nil {
		return
	}

	sh := s.lightHandler.ShadowHandler()
	res := sh.ShadowMapResolution()
	cascadeCount := sh.CascadeCount()
	atlasW := cascadeCount * res

	maxDim := int(s.r.MaxTextureDimension2D())
	if maxDim > 0 && atlasW > maxDim {
		panic(fmt.Sprintf(
			"scene: CSM atlas width %d (%d cascades × %d resolution) exceeds device MaxTextureDimension2D (%d). "+
				"Reduce WithShadowMapResolution.",
			atlasW, cascadeCount, res, maxDim,
		))
	}

	// Create CSM atlas depth texture: (atlasW × res).
	depthView, depthTex := s.r.CreateShadowDepthTexture(atlasW, res)
	sh.SetCSMAtlasTexture(depthTex)
	sh.SetCSMAtlasTextureView(depthView)

	// Create comparison sampler for PCF shadow sampling in the lit fragment shader.
	compSampler := s.r.CreateComparisonSampler()
	sh.SetComparisonSampler(compSampler)

	// Create one "csm_data_N" BGP per cascade — each holds a GPUShadowUniform buffer
	// for the per-cascade shadow depth pass vertex shader (group 0, binding 0).
	shadowGroup := 0
	for _, decl := range shadowVertShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgShadowUniform {
			shadowGroup = *decl.Group
			break
		}
	}
	desc := shadowVertShader.BindGroupLayoutDescriptor(shadowGroup)
	sizeOverrides := make(map[int]uint64)
	for _, entry := range desc.Entries {
		if entry.Buffer.Type == wgpu.BufferBindingTypeUniform {
			sizeOverrides[int(entry.Binding)] = uint64((&light.GPUShadowUniform{}).Size())
		}
	}
	for i := 0; i < cascadeCount; i++ {
		bgpKey := fmt.Sprintf("csm_data_%d", i)
		sh.SetBgp(bgpKey, bind_group_provider.NewBindGroupProvider(bgpKey))
		bgp := sh.Bgp(bgpKey)
		if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init CSM data bind group %d: %v", i, err))
		}
		bgp.SetSlot(1)
		if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init CSM data bind group %d slot 1: %v", i, err))
		}
		bgp.SetSlot(0)
	}

	// Compute 2D grid atlas dimensions.
	// Worst case: every GPU light is a point light needing 6 atlas slots.
	totalSlots := int(s.lightHandler.MaxGPULights()) * 6
	tileSize := sh.LightShadowTileSize()

	// Cap grid dimensions at GPU texture limits.
	// The WebGPU spec guarantees at least 8192 for maxTextureDimension2D.
	// Some wgpu backends report uint32-max instead of the real limit,
	// so we clamp to the spec-guaranteed minimum as a safe upper bound.
	const safeMaxTextureDim = 8192
	effectiveMaxDim := maxDim
	if effectiveMaxDim <= 0 || effectiveMaxDim > safeMaxTextureDim {
		effectiveMaxDim = safeMaxTextureDim
	}
	maxTilesPerAxis := effectiveMaxDim / tileSize
	if maxTilesPerAxis < 1 {
		maxTilesPerAxis = 1
	}

	cols := min(int(math.Ceil(math.Sqrt(float64(totalSlots)))), maxTilesPerAxis)
	rows := int(math.Ceil(float64(totalSlots) / float64(cols)))
	if rows > maxTilesPerAxis {
		rows = maxTilesPerAxis
	}
	// Physical atlas capacity.
	atlasCapacity := cols * rows
	sh.SetLightShadowAtlasSlots(atlasCapacity)
	sh.SetLightShadowAtlasCols(cols)

	spotAtlasW := cols * tileSize
	spotAtlasH := rows * tileSize
	spotView, spotTex := s.r.CreateShadowDepthTexture(spotAtlasW, spotAtlasH)
	sh.SetLightShadowAtlas(spotTex)
	sh.SetLightShadowAtlasView(spotView)

	// Create per-slot BGPs for spot/point shadow depth passes.
	for i := range atlasCapacity {
		bgpKey := fmt.Sprintf("spot_shadow_%d", i)
		sh.SetBgp(bgpKey, bind_group_provider.NewBindGroupProvider(bgpKey))
		bgp := sh.Bgp(bgpKey)
		if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init spot shadow bind group %d: %v", i, err))
		}
		bgp.SetSlot(1)
		if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init spot shadow bind group %d slot 1: %v", i, err))
		}
		bgp.SetSlot(0)
	}

	// Register shadow depth render pipelines for each ShadowCullMode variant.
	cullModes := []struct {
		mode model.ShadowCullMode
		wgpu wgpu.CullMode
		tag  string
	}{
		{model.ShadowCullModeBack, wgpu.CullModeBack, "back"},
		{model.ShadowCullModeFront, wgpu.CullModeFront, "front"},
		{model.ShadowCullModeNone, wgpu.CullModeNone, "none"},
	}

	for _, cm := range cullModes {
		key := "shadow_depth_static_" + cm.tag
		sp := pipeline.NewPipeline(key, pipeline.PipelineTypeRender,
			pipeline.WithVertexShader(shadowVertShader),
			pipeline.WithCullMode(cm.wgpu),
		)
		if err := s.r.RegisterShadowDepthPipeline(sp); err != nil {
			panic(fmt.Sprintf("scene: failed to register shadow depth static pipeline (%s): %v", cm.tag, err))
		}
		sh.SetPipelineKey("shadow_static_"+cm.tag, key)
	}

	if shadowSkinnedVertShader != nil {
		for _, cm := range cullModes {
			key := "shadow_depth_skinned_" + cm.tag
			ssp := pipeline.NewPipeline(key, pipeline.PipelineTypeRender,
				pipeline.WithVertexShader(shadowSkinnedVertShader),
				pipeline.WithCullMode(cm.wgpu),
			)
			if err := s.r.RegisterShadowDepthPipeline(ssp); err != nil {
				panic(fmt.Sprintf("scene: failed to register shadow depth skinned pipeline (%s): %v", cm.tag, err))
			}
			sh.SetPipelineKey("shadow_skinned_"+cm.tag, key)
		}
	}
}

// initCSMShadowLitBindGroup initializes the "csm_shadow_lit" bind group provider
// that lit fragment shaders use to sample the CSM atlas texture. Binds:
//   - @binding(0): CSM atlas texture view (Depth32Float shadow depth atlas)
//   - @binding(1): comparison sampler
//   - @binding(2): GPUCSMData uniform buffer (192 bytes: 32 + 2*80)
func (s *scene) initCSMShadowLitBindGroup(litFragmentShader shader.Shader) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if litFragmentShader == nil {
		return
	}

	sh := s.lightHandler.ShadowHandler()
	if sh.CSMAtlasTextureView() == nil || sh.ComparisonSampler() == nil {
		return // initShadowMap must be called first
	}

	// Resolve the shadow bind group index from the lit fragment shader's
	// provider annotation matching AnnotationArgShadow.
	shadowGroup := -1
	for _, decl := range litFragmentShader.Declarations() {
		if decl.Type == shader.AnnotationTypeProvider && decl.Group != nil && decl.Args[0] == shader.AnnotationArgShadow {
			shadowGroup = *decl.Group
			break
		}
	}
	if shadowGroup < 0 {
		return
	}

	sh.SetBgp("csm_shadow_lit", bind_group_provider.NewBindGroupProvider("csm_shadow_lit"))
	bgp := sh.Bgp("csm_shadow_lit")
	desc := litFragmentShader.BindGroupLayoutDescriptor(shadowGroup)

	// Pre-set texture views and samplers for explicit bindings.
	bgp.SetTextureView(0, sh.CSMAtlasTextureView())
	bgp.SetSampler(1, sh.ComparisonSampler())
	if sh.LightShadowAtlasView() != nil {
		bgp.SetTextureView(3, sh.LightShadowAtlasView())
	}

	sizeOverrides := make(map[int]uint64)
	for _, decl := range litFragmentShader.Declarations() {
		if decl.Type != shader.AnnotationTypeBindingGroup || decl.Group == nil || *decl.Group != shadowGroup || decl.Binding == nil {
			continue
		}
		typeArg := string(decl.Args[2])
		if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
			typeArg = strings.TrimSuffix(stripped, ">")
		}
		switch shader.AnnotationArg(typeArg) {
		case shader.AnnotationArgCSMData:
			sizeOverrides[*decl.Binding] = uint64((&light.GPUCSMData{}).Size())
		case shader.AnnotationArgLightShadowEntry:
			sizeOverrides[*decl.Binding] = uint64(sh.LightShadowAtlasSlots()) * uint64((&light.GPULightShadowEntry{}).Size())
		}
	}

	csHandler := s.lightHandler.ContactShadowHandler()
	fallbackInitialized := false
	initFallback := func() {
		if fallbackInitialized {
			return
		}
		fallback := common.TextureStagingData{
			Pixels: []byte{255, 255, 255, 255},
			Width:  1,
			Height: 1,
			Linear: true,
		}
		if err := s.r.InitTextureView(bgp, 5, fallback); err != nil {
			panic(fmt.Sprintf("scene: failed to init contact shadow fallback texture: %v", err))
		}
		fallbackSampler := common.SamplerStagingData{
			AddressModeU:  wgpu.AddressModeClampToEdge,
			AddressModeV:  wgpu.AddressModeClampToEdge,
			AddressModeW:  wgpu.AddressModeClampToEdge,
			MagFilter:     wgpu.FilterModeLinear,
			MinFilter:     wgpu.FilterModeLinear,
			MipmapFilter:  wgpu.MipmapFilterModeLinear,
			LodMinClamp:   0,
			LodMaxClamp:   1,
			MaxAnisotropy: 1,
		}
		if err := s.r.InitSampler(bgp, 6, fallbackSampler); err != nil {
			panic(fmt.Sprintf("scene: failed to init contact shadow fallback sampler: %v", err))
		}
		fallbackInitialized = true
	}

	for slot := 0; slot < 2; slot++ {
		bgp.SetSlot(slot)
		useContactShadow := false
		if csHandler != nil && csHandler.Enabled() && csHandler.LinearSampler() != nil {
			csHandler.SetSlot(slot)
			if csHandler.TextureView() != nil {
				bgp.SetTextureView(5, csHandler.TextureView())
				bgp.SetSampler(6, csHandler.LinearSampler())
				useContactShadow = true
			}
		}
		if !useContactShadow {
			initFallback()
		}
		if err := s.r.InitBindGroup(bgp, desc, nil, sizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init CSM shadow lit bind group slot %d: %v", slot, err))
		}
	}

	if csHandler != nil {
		csHandler.SetSlot(0)
	}
	bgp.SetSlot(0)
}

// initLightCullResources initializes the Forward+ light culling pipeline and buffer resources.
func (s *scene) initLightCullResources(cullComputeShader, litFragmentShader shader.Shader, screenWidth, screenHeight int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLightCullResourcesLocked(cullComputeShader, litFragmentShader, screenWidth, screenHeight)
}

// initLightCullResourcesLocked is the lock-free inner body of initLightCullResources.
// Caller must hold s.mu.
func (s *scene) initLightCullResourcesLocked(cullComputeShader, litFragmentShader shader.Shader, screenWidth, screenHeight int) {
	if cullComputeShader == nil || litFragmentShader == nil {
		return
	}
	lightsBGP := s.lightHandler.Bgp("lights")
	if lightsBGP.Buffer(1) == nil {
		return // initLightBindGroup must be called first
	}

	s.lightHandler.Resize(screenWidth, screenHeight)
	tileCountX := s.lightHandler.TileCountX()
	tileCountY := s.lightHandler.TileCountY()

	numTiles := uint64(tileCountX) * uint64(tileCountY)

	// ── 1. Create compute BGP (cull shader's @group(0)) ────────────────
	// binding 0: cull_uniforms (uniform, 160 bytes)
	// binding 1: cull_lights (storage, read) — shared from lightsBGP binding 1
	// binding 2: tile_light_counts (storage, rw) — new buffer
	// binding 3: tile_light_indices (storage, rw) — new buffer
	cullBGP := s.lightHandler.Bgp("light_cull")

	// Nil out the tile storage buffer slots so InitBindGroup always recreates
	// them at the current capacity. The old GPU buffer objects are still referenced
	// by tileBGP bindings 1 and 2 and must not be released here; they will be
	// released implicitly when tileBGP is rebuilt below and the wgpu device drops
	// the last reference.
	for _, slot := range []int{0, 1} {
		cullBGP.SetSlot(slot)
		cullBGP.SetBuffer(2, nil)
		cullBGP.SetBuffer(3, nil)
	}
	cullBGP.SetSlot(0)

	// Pre-set the lights buffer from lightsBGP so InitBindGroup reuses it.
	if lightsBuffer := lightsBGP.Buffer(1); lightsBuffer != nil {
		cullBGP.SetBuffer(1, lightsBuffer)
	}

	cullDesc := cullComputeShader.BindGroupLayoutDescriptor(0)
	sizeOverrides := map[int]uint64{
		0: uint64((&light.GPULightCullUniforms{}).Size()),           // LightCullUniforms
		2: numTiles * 4,                                             // tile_light_counts: one u32 per tile
		3: numTiles * uint64(s.lightHandler.MaxLightsPerTile()) * 4, // tile_light_indices
	}

	if err := s.r.InitBindGroup(cullBGP, cullDesc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init light cull bind group: %v", err))
	}

	// Init cull BGP for slot 1: share lights storage buffer from slot 1, create fresh cull uniforms and tile buffers.
	lightsBGP.SetSlot(1)
	slot1LightsBuffer := lightsBGP.Buffer(1)
	lightsBGP.SetSlot(0)
	cullBGP.SetSlot(1)
	if slot1LightsBuffer != nil {
		cullBGP.SetBuffer(1, slot1LightsBuffer)
	}
	if err := s.r.InitBindGroup(cullBGP, cullDesc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init cull bind group slot 1: %v", err))
	}
	slot1CountsBuf := cullBGP.Buffer(2)
	slot1IndicesBuf := cullBGP.Buffer(3)
	cullBGP.SetSlot(0)

	// ── 2. Register the cull compute pipeline ──────────────────────────
	pipeKey := "light_cull_compute"
	cp := pipeline.NewPipeline(pipeKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(cullComputeShader),
	)
	if err := s.r.RegisterPipelines(cp); err != nil {
		panic(fmt.Sprintf("scene: failed to register light cull compute pipeline: %v", err))
	}
	s.lightHandler.SetPipelineKey("light_cull", pipeKey)

	// ── 3. Create fragment tile BGP (lit frag shader's @group(5)) ──────
	// binding 0: tile_uniforms (uniform, 8 bytes)
	// binding 1: tile_light_counts (storage, read) — shared from cullBGP binding 2
	// binding 2: tile_light_indices (storage, read) — shared from cullBGP binding 3
	tileBGP := s.lightHandler.Bgp("tile_lit")

	if countsBuf := cullBGP.Buffer(2); countsBuf != nil {
		tileBGP.SetBuffer(1, countsBuf)
	}
	if indicesBuf := cullBGP.Buffer(3); indicesBuf != nil {
		tileBGP.SetBuffer(2, indicesBuf)
	}

	// Resolve the tile bind group index from the shader's pre-processor
	// declarations by matching the TileUniforms struct type annotation.
	tileGroup := -1
	for _, decl := range litFragmentShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgTileUniforms {
			tileGroup = *decl.Group
			break
		}
	}
	if tileGroup < 0 {
		panic("scene: lit fragment shader has no tile bind group")
	}

	tileDesc := litFragmentShader.BindGroupLayoutDescriptor(tileGroup)
	tileSizeOverrides := map[int]uint64{
		0: uint64((&light.GPUTileUniforms{}).Size()), // TileUniforms
	}
	if err := s.r.InitBindGroup(tileBGP, tileDesc, nil, tileSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init tile lit bind group: %v", err))
	}

	// Init tile BGP for slot 1: share slot-1 tile storage buffers from cull BGP, create fresh tile uniforms.
	tileBGP.SetSlot(1)
	if slot1CountsBuf != nil {
		tileBGP.SetBuffer(1, slot1CountsBuf)
	}
	if slot1IndicesBuf != nil {
		tileBGP.SetBuffer(2, slot1IndicesBuf)
	}
	if err := s.r.InitBindGroup(tileBGP, tileDesc, nil, tileSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init tile lit bind group slot 1: %v", err))
	}
	tileBGP.SetSlot(0)

	// ── 4. Write initial tile uniforms ─────────────────────────────────
	tileUniforms := light.GPUTileUniforms{
		TileCountX:       uint32(tileCountX),
		MaxLightsPerTile: uint32(s.lightHandler.MaxLightsPerTile()),
		ScreenWidth:      uint32(s.screenWidth),
		ScreenHeight:     uint32(s.screenHeight),
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: tileBGP, Binding: 0, Offset: 0, Data: tileUniforms.Marshal()},
	})

	s.tileBufferCapacity = tileCountX * tileCountY
}

// initComposition initializes the composition and tone mapping subsystem: creates
// offscreen HDR render targets (with optional MSAA textures), a linear sampler,
// registers the fullscreen composition pipeline, and creates the composition bind
// group provider with pre-set texture views. The G-Buffer must be initialized
// before calling this method so that SSR textures are available for binding.
func (s *scene) initComposition() {
	s.mu.Lock()
	defer s.mu.Unlock()

	ch := s.compositionHandler
	if ch == nil {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	sampleCount := s.r.SampleCount()

	// 1. Create HDR + optional MSAA + depth textures.
	hdrView, hdrTex, msaaView, msaaTex, depthView, depthTex := s.r.CreateCompositionTextures(w, h, sampleCount)
	ch.SetHDRTexture(hdrTex)
	ch.SetHDRTextureView(hdrView)
	ch.SetMSAATexture(msaaTex)
	ch.SetMSAATextureView(msaaView)
	ch.SetDepthTexture(depthTex)
	ch.SetDepthTextureView(depthView)

	// Initialize slot 1 composition textures
	ch.SetSlot(1)
	hdrView1, hdrTex1, msaaView1, msaaTex1, depthView1, depthTex1 := s.r.CreateCompositionTextures(w, h, sampleCount)
	ch.SetHDRTexture(hdrTex1)
	ch.SetHDRTextureView(hdrView1)
	ch.SetMSAATexture(msaaTex1)
	ch.SetMSAATextureView(msaaView1)
	ch.SetDepthTexture(depthTex1)
	ch.SetDepthTextureView(depthView1)
	ch.SetSlot(0)

	// Override the render pipeline color target format to RGBA16Float so that
	// all subsequently registered pipelines target the offscreen HDR texture
	// instead of the swapchain surface format.
	s.r.SetRenderTargetFormat(wgpu.TextureFormatRGBA16Float)

	// 2. Create linear sampler for HDR and SSR texture sampling.
	linearSamp := s.r.CreateLinearSampler()
	ch.SetLinearSampler(linearSamp)

	// 3. Load composition shaders and register the fullscreen pipeline.
	compVert := shader.NewShader("_composition_vert", shader.ShaderTypeVertex, "engine/renderer/postprocessing/composition/assets/composition-vert.wgsl", shader.WithInjections(s.injections))
	compFrag := shader.NewShader("_composition_frag", shader.ShaderTypeFragment, "engine/renderer/postprocessing/composition/assets/composition-frag.wgsl", shader.WithInjections(s.injections))

	compKey := "composition"
	compPipe := pipeline.NewPipeline(compKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(compVert),
		pipeline.WithFragmentShader(compFrag),
	)
	if err := s.r.RegisterCompositionPipeline(compPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register composition pipeline: %v", err))
	}
	ch.SetPipelineKey("composition", compKey)

	// 4. Initialize the composition bind group (group 0): hdr_texture, hdr_sampler,
	// ssr_texture, ssr_sampler, composition_params uniform.
	compBGP := ch.Bgp("composition")
	compDesc := compFrag.BindGroupLayoutDescriptor(0)

	// Pre-set the HDR texture view and sampler.
	compBGP.SetTextureView(0, hdrView)
	compBGP.SetSampler(1, linearSamp)

	// Bind the SSR texture if available, otherwise create a 1×1 black fallback.
	ssrHandler := s.ssrHandler
	if ssrHandler.SSRTextureView() != nil {
		compBGP.SetTextureView(2, ssrHandler.SSRTextureView())
	} else {
		fallback := common.TextureStagingData{
			Pixels: []byte{0, 0, 0, 0},
			Width:  1,
			Height: 1,
			Linear: true,
		}
		if err := s.r.InitTextureView(compBGP, 2, fallback); err != nil {
			panic(fmt.Sprintf("scene: failed to init SSR fallback texture for composition: %v", err))
		}
	}
	compBGP.SetSampler(3, linearSamp)

	s.initLuminance(ch, compBGP)

	s.initBloom(ch, compBGP, w, h)

	sizeOverrides := map[int]uint64{
		4: uint64((&composition.GPUCompositionParams{}).Size()),
		5: 4, // exposure_buffer: single f32
	}
	if err := s.r.InitBindGroup(compBGP, compDesc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init composition bind group: %v", err))
	}

	ch.Resize(w, h)
	ch.SetEnabled(true)
}

// initLuminance creates the luminance compute pipeline and exposure storage buffer
// used by the auto-exposure system. It registers the luminance pipeline, creates a
// persistent 4-byte exposure buffer (initialized to the composition handler's default
// exposure), wires the luminance BGP, and sets the exposure buffer at binding 5 of
// compBGP so it is included when the composition bind group is finalized.
//
// Must be called after the HDR texture view is set on ch and after bindings 0–3 are
// set on compBGP, but before s.r.InitBindGroup(compBGP, ...) is called.
func (s *scene) initLuminance(ch composition.Handler, compBGP bind_group_provider.BindGroupProvider) {
	expBuf := s.r.CreateBuffer("luminance_exposure", 4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc|wgpu.BufferUsageCopyDst)

	initData := make([]byte, 4)
	binary.LittleEndian.PutUint32(initData, math.Float32bits(ch.Exposure()))
	s.r.WriteRawBuffer(expBuf, 0, initData)
	ch.SetExposureBuffer(expBuf)

	lumShader := shader.NewShader("_luminance_compute", shader.ShaderTypeCompute, "engine/renderer/postprocessing/composition/assets/luminance-compute.wgsl", shader.WithInjections(s.injections))
	lumPipe := pipeline.NewPipeline("luminance_compute", pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(lumShader))
	if err := s.r.RegisterPipelines(lumPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register luminance compute pipeline: %v", err))
	}

	lumBGP := ch.Bgp("luminance_compute")
	lumDesc := lumShader.BindGroupLayoutDescriptor(0)
	lumBGP.SetTextureView(0, ch.HDRTextureView())
	lumBGP.SetBuffer(2, expBuf)
	lumSizeOverrides := map[int]uint64{
		1: (&composition.GPULuminanceParams{}).Size(),
	}
	if err := s.r.InitBindGroup(lumBGP, lumDesc, nil, lumSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init luminance compute bind group: %v", err))
	}

	// Init luminance BGP for slot 1.
	// expBuf (binding 2) is written by the GPU luminance compute shader each frame.
	// Both slots share the same physical expBuf — within-frame GPU-GPU, queue ordering guarantees no cross-frame hazard.
	// Only the luminance params buffer (binding 1) is CPU-written per frame and needs a fresh slot-1 copy.
	lumBGP.SetSlot(1)
	lumBGP.SetBuffer(2, expBuf)
	if err := s.r.InitBindGroup(lumBGP, lumDesc, nil, lumSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init luminance bind group slot 1: %v", err))
	}
	lumBGP.SetSlot(0)

	compBGP.SetBuffer(5, expBuf)
}

// initBloom creates bloom mip chain textures, registers bloom compute pipelines,
// and creates per-mip downsample and upsample bind group providers. When bloom is
// disabled, a 1×1 black fallback texture is bound at composition binding 6.
func (s *scene) initBloom(ch composition.Handler, compBGP bind_group_provider.BindGroupProvider, width, height int) {
	if !ch.BloomEnabled() {
		fallback := common.TextureStagingData{
			Pixels: []byte{0, 0, 0, 0},
			Width:  1,
			Height: 1,
			Linear: true,
		}
		if err := s.r.InitTextureView(compBGP, 6, fallback); err != nil {
			panic(fmt.Sprintf("scene: failed to init bloom fallback texture: %v", err))
		}
		return
	}

	halfW := width / 2
	halfH := height / 2
	if halfW <= 0 || halfH <= 0 {
		return
	}

	downTex, downReadViews, downStorageViews, upTex, upReadViews, upStorageViews, upMip0View, mipCount, err :=
		s.r.CreateBloomTextures(halfW, halfH)
	if err != nil {
		panic(fmt.Sprintf("scene: failed to create bloom textures: %v", err))
	}
	ch.SetBloomDownTexture(downTex)
	ch.SetBloomDownReadViews(downReadViews)
	ch.SetBloomDownStorageViews(downStorageViews)
	ch.SetBloomUpTexture(upTex)
	ch.SetBloomUpReadViews(upReadViews)
	ch.SetBloomUpStorageViews(upStorageViews)
	ch.SetBloomUpMip0View(upMip0View)
	ch.SetBloomMipCount(mipCount)

	downShader := shader.NewShader("_bloom_downsample", shader.ShaderTypeCompute, "engine/renderer/postprocessing/composition/assets/bloom-downsample.wgsl", shader.WithInjections(s.injections))
	downPipe := pipeline.NewPipeline("bloom_downsample", pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(downShader))
	if err := s.r.RegisterPipelines(downPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register bloom downsample pipeline: %v", err))
	}
	ch.SetPipelineKey("bloom_downsample", "bloom_downsample")

	upShader := shader.NewShader("_bloom_upsample", shader.ShaderTypeCompute, "engine/renderer/postprocessing/composition/assets/bloom-upsample.wgsl", shader.WithInjections(s.injections))
	upPipe := pipeline.NewPipeline("bloom_upsample", pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(upShader))
	if err := s.r.RegisterPipelines(upPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register bloom upsample pipeline: %v", err))
	}
	ch.SetPipelineKey("bloom_upsample", "bloom_upsample")

	linearSamp := ch.LinearSampler()

	downDesc := downShader.BindGroupLayoutDescriptor(0)
	bloomParamSize := uint64((&composition.GPUBloomParams{}).Size())
	for i := 0; i < mipCount; i++ {
		bgpName := fmt.Sprintf("bloom_down_%d", i)
		bgp := bind_group_provider.NewBindGroupProvider(bgpName)

		if i == 0 {
			bgp.SetTextureView(0, ch.HDRTextureView())
		} else {
			bgp.SetTextureView(0, downReadViews[i-1])
		}
		bgp.SetSampler(1, linearSamp)
		bgp.SetTextureView(2, downStorageViews[i])

		sizeOverrides := map[int]uint64{3: bloomParamSize}
		if err := s.r.InitBindGroup(bgp, downDesc, nil, sizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init bloom downsample bind group mip %d: %v", i, err))
		}
		ch.SetBgp(bgpName, bgp)
	}

	upDesc := upShader.BindGroupLayoutDescriptor(0)
	for i := mipCount - 2; i >= 0; i-- {
		bgpName := fmt.Sprintf("bloom_up_%d", i)
		bgp := bind_group_provider.NewBindGroupProvider(bgpName)

		if i == mipCount-2 {
			bgp.SetTextureView(0, downReadViews[mipCount-1])
		} else {
			bgp.SetTextureView(0, upReadViews[i+1])
		}
		bgp.SetSampler(1, linearSamp)
		bgp.SetTextureView(2, downReadViews[i])
		bgp.SetTextureView(3, upStorageViews[i])

		if err := s.r.InitBindGroup(bgp, upDesc, nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init bloom upsample bind group mip %d: %v", i, err))
		}
		ch.SetBgp(bgpName, bgp)
	}

	compBGP.SetTextureView(6, upMip0View)
}

// prepareLuminance dispatches the luminance compute shader to update the adapted
// exposure storage buffer based on the current HDR frame. No-ops if auto-exposure
// is disabled or the composition handler is not initialized.
func (s *scene) prepareLuminance(dt float32) {
	if s.lightHandler == nil {
		return
	}
	ch := s.compositionHandler
	if ch == nil || !ch.Enabled() || !ch.AutoExposureEnabled() {
		return
	}

	params := composition.GPULuminanceParams{
		ScreenWidth:         uint32(ch.ScreenWidth()),
		ScreenHeight:        uint32(ch.ScreenHeight()),
		AdaptSpeed:          ch.AdaptSpeed(),
		DeltaTime:           dt,
		MinExposure:         ch.MinExposure(),
		MaxExposure:         ch.MaxExposure(),
		KeyValue:            0.18,
		AutoExposureEnabled: 1,
	}

	lumBGP := ch.Bgp("luminance_compute")
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: lumBGP, Binding: 1, Offset: 0, Data: params.Marshal()},
	})
	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{PipelineKey: "luminance_compute", Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: lumBGP}}, WorkGroupCount: [3]uint32{1, 1, 1}},
	})
}

// initSSR initializes the SSR (screen-space reflections) subsystem: creates the
// half-resolution SSR output texture, the Hi-Z depth pyramid with per-mip views
// and bind groups, registers all compute pipelines (Hi-Z init, Hi-Z downsample,
// SSR compute), and wires up the SSR compute bind group with G-Buffer, HDR, and
// Hi-Z texture views. The G-Buffer and composition subsystems must be initialized
// before calling this method.
func (s *scene) initSSR() {
	s.mu.Lock()
	defer s.mu.Unlock()

	ssrHandler := s.ssrHandler
	gbHandler := s.gBufferHandler
	compHandler := s.compositionHandler
	if ssrHandler == nil || gbHandler == nil || compHandler == nil {
		return
	}
	if !gbHandler.Enabled() || !compHandler.Enabled() {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	// SSR runs at half-resolution.
	halfW := w / 2
	halfH := h / 2
	if halfW <= 0 {
		halfW = 1
	}
	if halfH <= 0 {
		halfH = 1
	}

	// 1. Create SSR output texture (RGBA16Float, storage + texture binding).
	ssrView, ssrTex := s.r.CreateSSRTextures(halfW, halfH)
	ssrHandler.SetSSRTexture(ssrTex)
	ssrHandler.SetSSRTextureView(ssrView)

	// 2. Create linear sampler for composition shader to sample SSR result.
	linearSamp := s.r.CreateLinearSampler()
	ssrHandler.SetLinearSampler(linearSamp)

	// 3. Create Hi-Z depth pyramid texture with full mip chain and per-mip views.
	hizView, hizTex, mipReadViews, mipStorageViews, mipCount := s.r.CreateHiZTextures(w, h)
	ssrHandler.SetHiZTexture(hizTex)
	ssrHandler.SetHiZTextureView(hizView)
	ssrHandler.SetHiZMipCount(mipCount)
	ssrHandler.SetHiZMipReadViews(mipReadViews)
	ssrHandler.SetHiZStorageViews(mipStorageViews)

	// MAX Hi-Z pyramid for slot 0 (same dimensions/mip count as MIN).
	maxHizView, maxHizTex, maxMipReadViews, maxMipStorageViews, _ := s.r.CreateHiZTextures(w, h)
	ssrHandler.SetHiZMaxTexture(maxHizTex)
	ssrHandler.SetHiZMaxTextureView(maxHizView)
	ssrHandler.SetHiZMaxMipReadViews(maxMipReadViews)
	ssrHandler.SetHiZMaxStorageViews(maxMipStorageViews)

	// Initialize slot 1 SSR textures
	ssrHandler.SetSlot(1)
	ssrView1, ssrTex1 := s.r.CreateSSRTextures(halfW, halfH)
	ssrHandler.SetSSRTexture(ssrTex1)
	ssrHandler.SetSSRTextureView(ssrView1)
	hizView1, hizTex1, mipReadViews1, mipStorageViews1, mipCount1 := s.r.CreateHiZTextures(w, h)
	ssrHandler.SetHiZTexture(hizTex1)
	ssrHandler.SetHiZTextureView(hizView1)
	ssrHandler.SetHiZMipCount(mipCount1)
	ssrHandler.SetHiZMipReadViews(mipReadViews1)
	ssrHandler.SetHiZStorageViews(mipStorageViews1)

	// MAX Hi-Z pyramid for slot 1.
	maxHizView1, maxHizTex1, maxMipReadViews1, maxMipStorageViews1, _ := s.r.CreateHiZTextures(w, h)
	ssrHandler.SetHiZMaxTexture(maxHizTex1)
	ssrHandler.SetHiZMaxTextureView(maxHizView1)
	ssrHandler.SetHiZMaxMipReadViews(maxMipReadViews1)
	ssrHandler.SetHiZMaxStorageViews(maxMipStorageViews1)
	ssrHandler.SetSlot(0)

	// 4. Register Hi-Z init compute pipeline (copies depth → Hi-Z mip 0).
	hizInitShader := shader.NewShader("_hiz_init", shader.ShaderTypeCompute, "engine/renderer/postprocessing/ssr/assets/hiz-init-compute.wgsl", shader.WithInjections(s.injections))
	hizInitKey := "hiz_init"
	hizInitPipe := pipeline.NewPipeline(hizInitKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(hizInitShader),
	)
	if err := s.r.RegisterPipelines(hizInitPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register Hi-Z init pipeline: %v", err))
	}
	ssrHandler.SetPipelineKey("hiz_init", hizInitKey)

	// 5. Create Hi-Z init bind group: binding 0 = gbuffer_depth, binding 1 = hiz mip 0 storage.
	hizInitBGP := bind_group_provider.NewBindGroupProvider("hiz_init")
	hizInitBGP.SetTextureView(0, gbHandler.DepthTextureView())
	hizInitBGP.SetTextureView(1, mipStorageViews[0])
	hizInitDesc := hizInitShader.BindGroupLayoutDescriptor(0)
	if err := s.r.InitBindGroup(hizInitBGP, hizInitDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init Hi-Z init bind group: %v", err))
	}
	ssrHandler.SetBgp("hiz_init", hizInitBGP)

	// BGP: hiz_init_max (SLOT 0) — writes GBuffer depth to MAX mip 0.
	hizInitMaxBGP := bind_group_provider.NewBindGroupProvider("hiz_init_max")
	hizInitMaxBGP.SetTextureView(0, gbHandler.DepthTextureView())
	hizInitMaxBGP.SetTextureView(1, maxMipStorageViews[0])
	if err := s.r.InitBindGroup(hizInitMaxBGP, hizInitDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init Hi-Z init_max bind group: %v", err))
	}
	ssrHandler.SetBgp("hiz_init_max", hizInitMaxBGP)

	// 6. Register Hi-Z downsample compute pipeline (min of 2×2 from prev mip).
	hizDownShader := shader.NewShader("_hiz_downsample", shader.ShaderTypeCompute, "engine/renderer/postprocessing/ssr/assets/hiz-downsample-compute.wgsl", shader.WithInjections(s.injections))
	hizDownKey := "hiz_downsample"
	hizDownPipe := pipeline.NewPipeline(hizDownKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(hizDownShader),
	)
	if err := s.r.RegisterPipelines(hizDownPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register Hi-Z downsample pipeline: %v", err))
	}
	ssrHandler.SetPipelineKey("hiz_downsample", hizDownKey)

	// Register MAX downsample pipeline (max of 2×2 from prev mip).
	hizDownMaxShader := shader.NewShader("_hiz_downsample_max", shader.ShaderTypeCompute, "engine/renderer/postprocessing/ssr/assets/hiz-downsample-max-compute.wgsl", shader.WithInjections(s.injections))
	hizDownMaxKey := "hiz_downsample_max"
	hizDownMaxPipe := pipeline.NewPipeline(hizDownMaxKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(hizDownMaxShader),
	)
	if err := s.r.RegisterPipelines(hizDownMaxPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register Hi-Z max downsample pipeline: %v", err))
	}
	ssrHandler.SetPipelineKey("hiz_downsample_max", hizDownMaxKey)

	// 7. Create per-mip downsample bind groups: for each mip level 1..N-1,
	//    binding 0 = read view of mip N-1, binding 1 = storage view of mip N.
	hizDownDesc := hizDownShader.BindGroupLayoutDescriptor(0)
	for i := 1; i < mipCount; i++ {
		bgpName := fmt.Sprintf("hiz_down_%d", i)
		bgp := bind_group_provider.NewBindGroupProvider(bgpName)
		bgp.SetTextureView(0, mipReadViews[i-1])
		bgp.SetTextureView(1, mipStorageViews[i])
		if err := s.r.InitBindGroup(bgp, hizDownDesc, nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init Hi-Z downsample bind group mip %d: %v", i, err))
		}
		ssrHandler.SetBgp(bgpName, bgp)
	}

	// BGPs: hiz_down_max_N (SLOT 0) — MAX downsample for mips 1..mipCount-1.
	for i := 1; i < mipCount; i++ {
		maxBGPName := fmt.Sprintf("hiz_down_max_%d", i)
		maxBGP := bind_group_provider.NewBindGroupProvider(maxBGPName)
		maxBGP.SetTextureView(0, maxMipReadViews[i-1])
		maxBGP.SetTextureView(1, maxMipStorageViews[i])
		if err := s.r.InitBindGroup(maxBGP, hizDownDesc, nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init Hi-Z max downsample bind group mip %d: %v", i, err))
		}
		ssrHandler.SetBgp(maxBGPName, maxBGP)
	}

	// 7b. Slot 1 Hi-Z init BGP: same layout as slot 0 but bound to slot 1 GBuffer depth and slot 1 mip 0 storage.
	gbHandler.SetSlot(1)
	hizInitBGP1 := bind_group_provider.NewBindGroupProvider("hiz_init_1")
	hizInitBGP1.SetTextureView(0, gbHandler.DepthTextureView())
	hizInitBGP1.SetTextureView(1, mipStorageViews1[0])
	hizInitDesc1 := hizInitShader.BindGroupLayoutDescriptor(0)
	if err := s.r.InitBindGroup(hizInitBGP1, hizInitDesc1, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init Hi-Z init_1 bind group: %v", err))
	}
	ssrHandler.SetBgp("hiz_init_1", hizInitBGP1)

	// BGP: hiz_init_max_1 (SLOT 1).
	hizInitMaxBGP1 := bind_group_provider.NewBindGroupProvider("hiz_init_max_1")
	hizInitMaxBGP1.SetTextureView(0, gbHandler.DepthTextureView())
	hizInitMaxBGP1.SetTextureView(1, maxMipStorageViews1[0])
	if err := s.r.InitBindGroup(hizInitMaxBGP1, hizInitDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init Hi-Z init_max_1 bind group: %v", err))
	}
	ssrHandler.SetBgp("hiz_init_max_1", hizInitMaxBGP1)
	gbHandler.SetSlot(0)

	// 7c. Slot 1 Hi-Z downsample BGPs: same layout as slot 0 but bound to slot 1 mip views.
	for i := 1; i < mipCount1; i++ {
		bgpName1 := fmt.Sprintf("hiz_down_%d_1", i)
		bgp1 := bind_group_provider.NewBindGroupProvider(bgpName1)
		bgp1.SetTextureView(0, mipReadViews1[i-1])
		bgp1.SetTextureView(1, mipStorageViews1[i])
		if err := s.r.InitBindGroup(bgp1, hizDownDesc, nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init Hi-Z downsample_1 bind group mip %d: %v", i, err))
		}
		ssrHandler.SetBgp(bgpName1, bgp1)
	}

	// BGPs: hiz_down_max_N_1 (SLOT 1).
	for i := 1; i < mipCount1; i++ {
		maxBGPName1 := fmt.Sprintf("hiz_down_max_%d_1", i)
		maxBGP1 := bind_group_provider.NewBindGroupProvider(maxBGPName1)
		maxBGP1.SetTextureView(0, maxMipReadViews1[i-1])
		maxBGP1.SetTextureView(1, maxMipStorageViews1[i])
		if err := s.r.InitBindGroup(maxBGP1, hizDownDesc, nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init Hi-Z max downsample_1 bind group mip %d: %v", i, err))
		}
		ssrHandler.SetBgp(maxBGPName1, maxBGP1)
	}

	// 8. Load SSR compute shader and register compute pipeline.
	ssrCompShader := shader.NewShader("_ssr_compute", shader.ShaderTypeCompute, "engine/renderer/postprocessing/ssr/assets/ssr-compute.wgsl", shader.WithInjections(s.injections))
	ssrCompKey := "ssr_compute"
	ssrCompPipe := pipeline.NewPipeline(ssrCompKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(ssrCompShader),
	)
	if err := s.r.RegisterPipelines(ssrCompPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register SSR compute pipeline: %v", err))
	}
	ssrHandler.SetPipelineKey("ssr_compute", ssrCompKey)

	// 9. Initialize SSR compute bind group (group 0):
	//   binding 0: ssr_params uniform
	//   binding 1: gbuffer_normal texture
	//   binding 2: gbuffer_depth texture
	//   binding 3: hdr_texture
	//   binding 4: ssr_output storage texture
	//   binding 5: hiz_texture (full mip chain)
	ssrBGP := ssrHandler.Bgp("ssr_compute")
	ssrDesc := ssrCompShader.BindGroupLayoutDescriptor(0)

	ssrBGP.SetTextureView(1, gbHandler.NormalTextureView())
	ssrBGP.SetTextureView(2, gbHandler.DepthTextureView())
	ssrBGP.SetTextureView(3, compHandler.HDRTextureView())
	ssrBGP.SetTextureView(4, ssrView)
	ssrBGP.SetTextureView(5, hizView)

	ssrSizeOverrides := map[int]uint64{
		0: uint64((&ssr.GPUSSRParams{}).Size()),
	}
	if err := s.r.InitBindGroup(ssrBGP, ssrDesc, nil, ssrSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init SSR compute bind group: %v", err))
	}

	ssrHandler.Resize(w, h)
	ssrHandler.SetEnabled(true)
	s.refreshAnimatorHiZBindGroups()
}

// initTAA initializes the TAA resolve subsystem: creates two full-resolution
// RGBA16Float ping-pong textures, registers the TAA resolve compute pipeline,
// and creates both slot-indexed compute BGPs. When TAA is enabled, the composition
// BGP's binding 0 is rebound from the raw HDR texture to the TAA resolved texture,
// so the composition pass samples the temporally accumulated output instead.
//
// Must be called after initSSR() (G-Buffer depth views exist) and after
// initComposition() (composition BGP and exposure buffer exist).
func (s *scene) initTAA() {
	s.mu.Lock()
	defer s.mu.Unlock()

	taaH := s.taaHandler
	gbH := s.gBufferHandler
	compH := s.compositionHandler
	if taaH == nil || gbH == nil || compH == nil {
		return
	}
	if !gbH.Enabled() || !compH.Enabled() {
		return
	}

	w := s.screenWidth
	h := s.screenHeight
	if w <= 0 || h <= 0 {
		return
	}

	// 1. Create two ping-pong RGBA16Float textures.
	view0, tex0, view1, tex1 := s.r.CreateTAATextures(w, h)

	taaH.SetSlot(0)
	taaH.SetTAATexture(tex0)
	taaH.SetTAATextureView(view0)

	taaH.SetSlot(1)
	taaH.SetTAATexture(tex1)
	taaH.SetTAATextureView(view1)

	taaH.SetSlot(0)

	// 2. Create a shared linear sampler for history sampling.
	linearSamp := s.r.CreateLinearSampler()
	taaH.SetLinearSampler(linearSamp)

	// 2b. Create the CAS sharpening output texture (single, not ping-ponged).
	sharpenView, sharpenTex := s.r.CreateSharpenTexture(w, h)
	taaH.SetSharpenTexture(sharpenTex)
	taaH.SetSharpenTextureView(sharpenView)

	// 3. Register the TAA resolve compute pipeline.
	taaShader := shader.NewShader("_taa_resolve", shader.ShaderTypeCompute,
		"engine/renderer/postprocessing/taa/assets/taa-resolve.wgsl", shader.WithInjections(s.injections))
	taaKey := "taa_resolve"
	taaPipe := pipeline.NewPipeline(taaKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(taaShader),
	)
	if err := s.r.RegisterPipelines(taaPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register TAA resolve pipeline: %v", err))
	}
	taaH.SetPipelineKey("taa_resolve", taaKey)

	// 3b. Register the CAS sharpening compute pipeline.
	casShader := shader.NewShader("_taa_sharpen", shader.ShaderTypeCompute,
		"engine/renderer/postprocessing/taa/assets/taa-sharpen.wgsl", shader.WithInjections(s.injections))
	casKey := "taa_sharpen"
	casPipe := pipeline.NewPipeline(casKey, pipeline.PipelineTypeCompute,
		pipeline.WithComputeShader(casShader),
	)
	if err := s.r.RegisterPipelines(casPipe); err != nil {
		panic(fmt.Sprintf("scene: failed to register TAA sharpen pipeline: %v", err))
	}
	taaH.SetPipelineKey("taa_sharpen", casKey)

	casDesc := casShader.BindGroupLayoutDescriptor(0)
	taaDesc := taaShader.BindGroupLayoutDescriptor(0)
	taaSizeOverrides := map[int]uint64{
		0: uint64((&taa.GPUTAAParams{}).Size()),
	}

	// 4. Create BGP for slot 0: hdr=view0, history=view1, depth=gbH slot 0, resolved=view0 storage.
	bgp0 := taaH.Bgp("taa_resolve_0")
	compH.SetSlot(0)
	bgp0.SetTextureView(1, compH.HDRTextureView())
	gbH.SetSlot(0)
	bgp0.SetTextureView(3, gbH.DepthTextureView())
	bgp0.SetTextureView(2, view1) // history: previous slot's output
	bgp0.SetTextureView(4, view0) // taa_resolved: storage write target for this slot
	bgp0.SetSampler(5, linearSamp)
	if err := s.r.InitBindGroup(bgp0, taaDesc, nil, taaSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init TAA BGP slot 0: %v", err))
	}

	// 5. Create BGP for slot 1: hdr=view1, history=view0, depth=gbH slot 1, resolved=view1 storage.
	bgp1 := taaH.Bgp("taa_resolve_1")
	compH.SetSlot(1)
	bgp1.SetTextureView(1, compH.HDRTextureView())
	gbH.SetSlot(1)
	bgp1.SetTextureView(3, gbH.DepthTextureView())
	bgp1.SetTextureView(2, view0) // history: previous slot's output
	bgp1.SetTextureView(4, view1) // taa_resolved: storage write target for this slot
	bgp1.SetSampler(5, linearSamp)
	if err := s.r.InitBindGroup(bgp1, taaDesc, nil, taaSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init TAA BGP slot 1: %v", err))
	}

	// 5b. Create slot-indexed CAS BGPs.
	//     Slot 0: CAS reads view0 (the slot-0 resolve output); writes to sharpenView.
	//     Slot 1: CAS reads view1 (the slot-1 resolve output); writes to sharpenView.
	//     Both slots share the same sharpenView output — CAS overwrites it each frame.
	bgpCAS0 := taaH.Bgp("taa_sharpen_0")
	bgpCAS0.SetTextureView(0, view0)
	bgpCAS0.SetTextureView(1, sharpenView)
	bgpCAS0.SetSampler(2, linearSamp)
	if err := s.r.InitBindGroup(bgpCAS0, casDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init CAS BGP slot 0: %v", err))
	}

	bgpCAS1 := taaH.Bgp("taa_sharpen_1")
	bgpCAS1.SetTextureView(0, view1)
	bgpCAS1.SetTextureView(1, sharpenView)
	bgpCAS1.SetSampler(2, linearSamp)
	if err := s.r.InitBindGroup(bgpCAS1, casDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init CAS BGP slot 1: %v", err))
	}

	// Restore slots to 0.
	compH.SetSlot(0)
	gbH.SetSlot(0)
	taaH.SetSlot(0)

	taaH.Resize(w, h)
	taaH.SetEnabled(true)

	// 6. Rebind composition input from raw HDR to the shared TAA sharpen output
	//    so composition presents the CAS-filtered TAA result.
	compBGP := compH.Bgp("composition")
	if compBGP == nil {
		return
	}

	compFrag := shader.NewShader("_composition_frag_taa_rebind", shader.ShaderTypeFragment,
		"engine/renderer/postprocessing/composition/assets/composition-frag.wgsl", shader.WithInjections(s.injections))
	compDesc := compFrag.BindGroupLayoutDescriptor(0)
	compSizeOverrides := map[int]uint64{
		4: uint64((&composition.GPUCompositionParams{}).Size()),
	}

	compBGP.SetSlot(0)
	compBGP.SetTextureView(0, sharpenView)
	if err := s.r.InitBindGroup(compBGP, compDesc, nil, compSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to re-init composition BGP slot 0 for TAA sharpen output: %v", err))
	}

	compBGP.SetSlot(1)
	compBGP.SetTextureView(0, sharpenView)
	if err := s.r.InitBindGroup(compBGP, compDesc, nil, compSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to re-init composition BGP slot 1 for TAA sharpen output: %v", err))
	}

	compBGP.SetSlot(0)
}

// initLighting initializes the entire lighting pipeline in the correct order:
// light storage buffer, shadow map resources, shadow lit bind group, and Forward+
// light culling. All lighting shaders are loaded internally from the engine's
// standard light shader assets.
func (s *scene) initLighting(screenWidth, screenHeight int) {
	s.buildInjectionMap()

	litFragShader := shader.NewShader("_lit_frag_csm", shader.ShaderTypeFragment, "engine/light/assets/lit-frag-csm.wgsl", shader.WithInjections(s.injections))
	cullComputeShader := shader.NewShader("_light_cull_compute", shader.ShaderTypeCompute, "engine/light/assets/light-cull-compute.wgsl", shader.WithInjections(s.injections))

	// 1. Light storage buffer (must be first — other steps share this buffer).
	s.initLightBindGroup(litFragShader)

	// 2. Shadow resources — creates depth-only atlas texture, comparison sampler,
	// and PCF shadow render pipelines.
	shadowVertShader := shader.NewShader("_shadow_depth_vert", shader.ShaderTypeVertex, "engine/light/assets/shadow-depth-vert.wgsl", shader.WithInjections(s.injections))
	shadowSkinnedVertShader := shader.NewShader("_shadow_depth_skinned_vert", shader.ShaderTypeVertex, "engine/light/assets/shadow-depth-skinned-vert.wgsl", shader.WithInjections(s.injections))
	s.initShadowMap(shadowVertShader, shadowSkinnedVertShader)

	// 3. Forward+ tile culling pipeline and shared tile buffers (references lights buffer from step 1).
	s.initLightCullResources(cullComputeShader, litFragShader, screenWidth, screenHeight)

	// 4. Re-create the camera bind group with merged VERTEX|FRAGMENT visibility.
	//
	// The camera's bind group was originally created in NewScene from the vertex
	// shader alone (visibility = VERTEX). The lit fragment shader also declares the
	// camera group (visibility = FRAGMENT). The render pipeline merges these into
	// VERTEX|FRAGMENT. WebGPU requires exact bind group layout equivalence, so the
	// camera BGL must be recreated with the combined visibility to pass validation.
	s.reinitCameraBGPForLitPipeline(litFragShader)

	// 5. G-Buffer MRT pre-pass (required by SSAO and SSR).
	s.initGBuffer()

	// 6. SSAO — hemisphere sampling + bilateral blur (requires G-Buffer).
	s.initSSAO()

	// 7. Contact shadows — screen-space ray march (requires G-Buffer).
	s.initContactShadows()

	// 8. Shadow lit BGP (fragment-side shadow sampling — references shadow resources from step 2
	// and the contact-shadow texture from step 7).
	s.initCSMShadowLitBindGroup(litFragShader)

	// 9. SSAO lit bind group — binds blurred SSAO texture at @group(6) for the lit
	// fragment shader. When SSAO is disabled, a 1×1 white fallback is used (ao=1.0).
	s.initSSAOLitBindGroup(litFragShader)

	// 10. Composition — offscreen HDR render target + fullscreen tone-mapping pass.
	// Must come before SSR so the HDR texture exists for SSR to read.
	s.initComposition()

	// 11. SSR — screen-space reflections compute pass (requires G-Buffer + composition HDR texture).
	// Create a 1×1 fallback Hi-Z texture for animators that are registered before the real
	// SSR pyramid is built. The shader guards access with hiz_mip_count == 0.
	if s.hizFallbackView == nil {
		hizFallbackView, hizFallbackTex, _, _, _ := s.r.CreateHiZTextures(1, 1)
		s.hizFallbackTexture = hizFallbackTex
		s.hizFallbackView = hizFallbackView
	}
	s.initSSR()

	// Re-bind the SSR texture on the composition BGP now that it exists.
	if s.ssrHandler.Enabled() && s.compositionHandler.Enabled() {
		compBGP := s.compositionHandler.Bgp("composition")
		if compBGP != nil && s.ssrHandler.SSRTextureView() != nil {
			compBGP.SetTextureView(2, s.ssrHandler.SSRTextureView())
			// Rebuild the composition bind group to pick up the real SSR texture.
			compFrag := shader.NewShader("_composition_frag_rebind", shader.ShaderTypeFragment, "engine/renderer/postprocessing/composition/assets/composition-frag.wgsl", shader.WithInjections(s.injections))
			compDesc := compFrag.BindGroupLayoutDescriptor(0)
			sizeOverrides := map[int]uint64{
				4: uint64((&composition.GPUCompositionParams{}).Size()),
			}
			if err := s.r.InitBindGroup(compBGP, compDesc, nil, sizeOverrides); err != nil {
				panic(fmt.Sprintf("scene: failed to re-init composition bind group with SSR texture: %v", err))
			}
		}
	}

	// 12. TAA — temporal accumulation resolve pass (requires G-Buffer depth + composition HDR).
	s.initTAA()

	// 13. Mark the lighting subsystem as GPU-initialized.
	s.lightHandler.SetEnabled(true)
	s.postProcessingInitialized = true
}

// initPhysics creates GPU buffers, bind groups, and compute pipelines for the
// physics simulation. Called once when the first rigid body is added to the scene.
// Follows the same InitBindGroup + shared buffer + pipeline registration pattern
// used by createAnimator. Caller must hold s.mu write lock.
func (s *scene) initPhysics() {
	s.buildInjectionMap()
	ph := s.physicsHandler

	type stageEntry struct {
		name string
		path string
	}

	stages := []stageEntry{
		{"particle_values", "engine/physics/assets/particle-values.wgsl"},
		{"aabb_reduce", "engine/physics/assets/aabb-reduce.wgsl"},
		{"grid_build_params", "engine/physics/assets/grid-build-params.wgsl"},
		{"grid_clear", "engine/physics/assets/grid-clear.wgsl"},
		{"grid_insert", "engine/physics/assets/grid-insert.wgsl"},
		{"collision", "engine/physics/assets/collision-reaction.wgsl"},
		{"momenta", "engine/physics/assets/compute-momenta.wgsl"},
		{"integrate", "engine/physics/assets/integrate.wgsl"},
		{"sync", "engine/physics/assets/physics-sync.wgsl"},
	}

	shaders := make(map[string]shader.Shader, len(stages))
	for _, st := range stages {
		shaders[st.name] = shader.NewShader("physics_"+st.name, shader.ShaderTypeCompute, st.path, shader.WithInjections(s.injections))
	}

	// Canonical buffer indices on the buffers BGP. These are the contract between
	// physics.go staged writes and the GPU buffer layout.
	annotatedBufferIndex := map[shader.AnnotationArg]int{
		shader.AnnotationArgPhysicsBody:       0,
		shader.AnnotationArgPhysicsParticle:   1,
		shader.AnnotationArgPhysicsGrid:       2,
		shader.AnnotationArgPhysicsGlobals:    3,
		shader.AnnotationArgPhysicsGridParams: 4,
	}

	// Manually declared WGSL bindings (atomic types not expressible via annotations)
	// are matched by their variable name from the parsed shader source.
	manualVarBufferIndex := map[string]int{
		"aabb":     5, // AABB atomics (6 × atomic<u32>)
		"grid":     2, // atomic view of the grid cell buffer
		"sync_map": 7, // sync mapping (body index → animator instance ID)
	}

	// Derive per-shader binding→canonical buffer index maps from declarations and
	// var names. Simultaneously collect layout entries for the unified buffers BGP.
	bufferMaps := make(map[string]map[int]int, len(stages))
	collected := make(map[int]wgpu.BindGroupLayoutEntry)

	for _, st := range stages {
		sh := shaders[st.name]
		desc := sh.BindGroupLayoutDescriptor(0)
		bmap := make(map[int]int, len(desc.Entries))

		// Resolve annotated bindings via the declaration list
		for _, decl := range sh.Declarations() {
			typeArg := string(decl.Args[2])
			if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
				typeArg = strings.TrimSuffix(stripped, ">")
			}
			if bufIdx, ok := annotatedBufferIndex[shader.AnnotationArg(typeArg)]; ok {
				bmap[*decl.Binding] = bufIdx
			}
		}

		// Resolve remaining (manually-declared) bindings by WGSL variable name
		for _, entry := range desc.Entries {
			b := int(entry.Binding)
			if _, done := bmap[b]; done {
				continue
			}
			if bufIdx, ok := manualVarBufferIndex[sh.BindGroupVarName(0, b)]; ok {
				bmap[b] = bufIdx
			}
		}

		// Collect layout entries at canonical indices for the buffers BGP descriptor.
		// Prefer entries with larger MinBindingSize so the struct-based entry (e.g.
		// GridCell=16) wins over the atomic element size (u32=4).
		for _, entry := range desc.Entries {
			if canonIdx, ok := bmap[int(entry.Binding)]; ok {
				if existing, exists := collected[canonIdx]; !exists || entry.Buffer.MinBindingSize > existing.Buffer.MinBindingSize {
					e := entry
					e.Binding = uint32(canonIdx)
					collected[canonIdx] = e
				}
			}
		}

		bufferMaps[st.name] = bmap
	}

	// The sync mapping buffer is referenced by the sync shader at binding 1 (manual
	// var) and populated via staged writes during RegisterBody. Add it manually to the
	// buffers BGP descriptor since its canonical index (7) is not discovered from other
	// annotated stages.
	collected[7] = wgpu.BindGroupLayoutEntry{
		Binding: 7, Visibility: wgpu.ShaderStageCompute,
		Buffer: wgpu.BufferBindingLayout{Type: wgpu.BufferBindingTypeStorage, MinBindingSize: 4},
	}

	// Assemble the buffers BGP descriptor from collected entries, sorted by binding.
	buffersEntries := make([]wgpu.BindGroupLayoutEntry, 0, len(collected))
	for i := range 8 {
		if e, ok := collected[i]; ok {
			buffersEntries = append(buffersEntries, e)
		}
	}

	maxBodies := uint64(ph.MaxBodies())
	maxParticles := uint64(ph.MaxParticles())
	maxGridCells := uint64(ph.MaxGridCells())

	buffersDesc := wgpu.BindGroupLayoutDescriptor{
		Label:   "physics_buffers",
		Entries: buffersEntries,
	}

	buffersSizeOverrides := map[int]uint64{
		0: maxBodies * uint64((&physics.GPUBody{}).Size()),
		1: maxParticles * uint64((&physics.GPUParticle{}).Size()),
		2: maxGridCells * uint64((&physics.GPUGridCell{}).Size()),
		3: uint64((&physics.GPUPhysicsGlobals{}).Size()),
		4: uint64((&physics.GPUGridParams{}).Size()),
		5: 24,            // aabbAtomics: 6 × u32 (no struct type)
		7: maxBodies * 4, // syncMapping: u32 per body
	}

	buffersUsageOverrides := map[int]wgpu.BufferUsage{
		0: wgpu.BufferUsageCopySrc, // allow copy-to-staging for readback
	}

	if err := s.r.InitBindGroup(ph.Buffers(), buffersDesc, buffersUsageOverrides, buffersSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init physics buffers BGP: %v", err))
	}

	// Wire shared physical buffers from the buffers BGP into each per-shader BGP,
	// then create bind groups. InitBindGroup skips buffer creation for bindings
	// that already have a buffer set via SetBuffer.
	for _, st := range stages {
		bgp := ph.Bgp(st.name)
		for shaderBinding, canonIdx := range bufferMaps[st.name] {
			bgp.SetBuffer(shaderBinding, ph.Buffers().Buffer(canonIdx))
		}
		// Sync InitBindGroup is deferred to Add() because binding 2
		// (AnimationData) comes from the Animator, not the physics buffers.
		if st.name == "sync" {
			continue
		}
		desc := shaders[st.name].BindGroupLayoutDescriptor(0)
		if err := s.r.InitBindGroup(bgp, desc, nil, nil); err != nil {
			panic(fmt.Sprintf("scene: failed to init physics BGP %q: %v", st.name, err))
		}
	}

	// Register compute pipelines and store their keys on the physics handler.
	for _, st := range stages {
		sh := shaders[st.name]
		p := pipeline.NewPipeline(sh.Key(), pipeline.PipelineTypeCompute, pipeline.WithComputeShader(sh))
		if err := s.r.RegisterPipelines(p); err != nil {
			panic(fmt.Sprintf("scene: failed to register physics pipeline %q: %v", st.name, err))
		}
		ph.SetPipelineKey(st.name, p.PipelineKey())
	}

	// Register the bone_particle_update pipeline separately. Its BGP is created
	// per-group in createBoneParticleUpdateGroup because it requires buffers from
	// both the physics handler and a specific Animator.
	{
		boneUpdateShader := shader.NewShader("physics_bone_update", shader.ShaderTypeCompute,
			"engine/physics/assets/bone-particle-update.wgsl", shader.WithInjections(s.injections))
		boneUpdatePipe := pipeline.NewPipeline(boneUpdateShader.Key(), pipeline.PipelineTypeCompute,
			pipeline.WithComputeShader(boneUpdateShader))
		if err := s.r.RegisterPipelines(boneUpdatePipe); err != nil {
			panic(fmt.Sprintf("scene: failed to register bone_update pipeline: %v", err))
		}
		ph.SetPipelineKey("bone_update", boneUpdatePipe.PipelineKey())
	}

	// Create a staging buffer for GPU→CPU readback of body positions and quaternions.
	// Sized for the full body buffer so any number of bodies up to maxBodies can be read back.
	stagingSize := maxBodies * uint64((&physics.GPUBody{}).Size())
	stagingBuf := s.r.CreateBuffer("physics_staging", stagingSize, wgpu.BufferUsageMapRead|wgpu.BufferUsageCopyDst)
	ph.SetStagingBuffer(stagingBuf)
}

// initPhysicsSyncGroup creates a per-animator sync bind group provider for the
// physics sync shader dispatch. Each group has its own sync_map buffer (initialized
// to the 0xFFFFFFFF sentinel so the shader skips non-member bodies) and references
// the animator's AnimationData buffer at binding 2. The bodies and globals buffers
// are shared from the physics handler. Caller must hold s.mu write lock.
//
// Parameters:
//   - anim: the Animator that owns the bodies in this sync group
//   - computeShader: the compute shader used by the Animator (for AnimationData binding discovery)
//
// Returns:
//   - int: the ID of the new sync group in s.physicsSyncGroup
func (s *scene) initPhysicsSyncGroup(anim animator.Animator) int {
	ph := s.physicsHandler
	currentSlot := s.r.CurrentFrameSlot()

	// Discover the AnimationData binding index from the standard simple compute shader (cached).
	if s.physicsAnimBinding < 0 {
		syncComputeShader := shader.NewShader("_sync_compute_init", shader.ShaderTypeCompute, "engine/renderer/animator/assets/simple-compute.wgsl", shader.WithInjections(s.injections))
		for _, decl := range syncComputeShader.Declarations() {
			typeArg := string(decl.Args[2])
			if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
				typeArg = strings.TrimSuffix(stripped, ">")
			}
			if shader.AnnotationArg(typeArg) == shader.AnnotationArgAnimationData {
				s.physicsAnimBinding = *decl.Binding
				break
			}
		}
	}

	if s.physicsSyncGroup == nil {
		s.physicsSyncGroup = make(map[int]bind_group_provider.BindGroupProvider)
	}
	groupID := uint32(len(s.physicsSyncGroup))
	bgpLabel := fmt.Sprintf("physics_sync_group_%d", groupID)
	bgp := bind_group_provider.NewBindGroupProvider(bgpLabel)
	animCompute := anim.ComputeBindGroupProvider()
	syncShader := s.r.Pipeline(ph.PipelineKey("sync")).Shader(shader.ShaderTypeCompute)
	syncDesc := syncShader.BindGroupLayoutDescriptor(0)
	sizeOverrides := map[int]uint64{
		1: uint64(ph.MaxBodies()) * 4,
	}

	// Initialize slot 0, letting InitBindGroup create the per-group sync_map buffer.
	animCompute.SetSlot(0)
	bgp.SetSlot(0)
	bgp.SetBuffer(0, ph.Buffers().Buffer(0))
	bgp.SetBuffer(3, ph.Buffers().Buffer(3))
	bgp.SetBuffer(2, animCompute.Buffer(s.physicsAnimBinding))
	if err := s.r.InitBindGroup(bgp, syncDesc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init physics sync BGP for group %d: %v", groupID, err))
	}
	syncMapBuf := bgp.Buffer(1)

	// Initialize slot 1 with the matching AnimationData buffer while sharing the
	// same sync_map buffer so staged writes remain coherent across both slots.
	animCompute.SetSlot(1)
	bgp.SetSlot(1)
	bgp.SetBuffer(0, ph.Buffers().Buffer(0))
	bgp.SetBuffer(1, syncMapBuf)
	bgp.SetBuffer(2, animCompute.Buffer(s.physicsAnimBinding))
	bgp.SetBuffer(3, ph.Buffers().Buffer(3))
	if err := s.r.InitBindGroup(bgp, syncDesc, nil, sizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init physics sync BGP slot 1 for group %d: %v", groupID, err))
	}

	animCompute.SetSlot(currentSlot)
	bgp.SetSlot(currentSlot)

	// Initialize the sync_map buffer to all 0xFFFFFFFF (sentinel). The shader
	// checks this value and skips bodies not belonging to the group.
	sentinelData := make([]byte, ph.MaxBodies()*4)
	for i := 0; i < len(sentinelData); i += 4 {
		binary.LittleEndian.PutUint32(sentinelData[i:i+4], 0xFFFFFFFF)
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: bgp, Binding: 1, Offset: 0, Data: sentinelData},
	})

	s.physicsSyncGroup[int(groupID)] = bgp
	s.physicsSyncAnimMap[anim] = int(groupID)

	return int(groupID)
}

// reinitCameraBGPForLitPipeline recreates the camera's bind group with merged
// VERTEX|FRAGMENT visibility so it matches the lit render pipeline's layout.
//
// The camera BGL was originally created from the vertex shader alone (VERTEX).
// When the lit fragment shader also declares the same camera group, the render
// pipeline merges the layout entries with VERTEX|FRAGMENT visibility. WebGPU
// requires exact bind group layout equivalence, so the camera BGL must be
// recreated with the combined visibility to avoid SetBindGroup validation errors.
//
// The existing camera uniform buffer is preserved — only the layout and bind
// group objects are recreated.
//
// Parameters:
//   - litFragShader: the lit fragment shader that may declare a camera group
func (s *scene) reinitCameraBGPForLitPipeline(litFragShader shader.Shader) {
	if litFragShader == nil {
		return
	}

	// Resolve the camera group index from the shader's pre-processor
	// declarations by matching the Camera struct type annotation.
	cameraGroup := -1
	for _, decl := range litFragShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Group != nil && decl.Args[2] == shader.AnnotationArgCamera {
			cameraGroup = *decl.Group
			break
		}
	}
	if cameraGroup < 0 {
		return // Fragment shader doesn't declare a camera group; no re-init needed.
	}

	bgp := s.cam.BindGroupProvider()
	if bgp == nil {
		return
	}

	// Grab the fragment shader's descriptor and add VERTEX visibility to every
	// entry so the resulting layout matches both shader stages.
	fragDesc := litFragShader.BindGroupLayoutDescriptor(cameraGroup)
	entries := make([]wgpu.BindGroupLayoutEntry, len(fragDesc.Entries))
	copy(entries, fragDesc.Entries)
	for i := range entries {
		entries[i].Visibility |= wgpu.ShaderStageVertex
	}
	mergedDesc := wgpu.BindGroupLayoutDescriptor{
		Label:   fragDesc.Label,
		Entries: entries,
	}

	// Clear the old layout so InitBindGroup creates a new one from mergedDesc.
	bgp.SetBindGroupLayout(nil)
	if err := s.r.InitBindGroup(bgp, mergedDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to reinit camera bind group for lit pipeline: %v", err))
	}
	bgp.SetSlot(1)
	if err := s.r.InitBindGroup(bgp, mergedDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to reinit camera bind group slot 1 for lit pipeline: %v", err))
	}
	bgp.SetSlot(0)
}

// patchSyncMapEntry stages a write to the per-group sync_map buffer that maps the
// given object's physics body slot to a new Animator instance ID. Pass 0xFFFFFFFF
// as instanceID to sentinel the entry (disabling sync for that body). Caller must
// hold s.mu write lock. No-op if the object has no physics body or its Animator has
// no sync group.
func (s *scene) patchSyncMapEntry(anim animator.Animator, objID uint64, instanceID uint32) {
	if s.physicsHandler == nil || anim == nil {
		return
	}
	bodyIdx, ok := s.physicsHandler.BodyIndex(objID)
	if !ok {
		return
	}
	sgIdx, exists := s.physicsSyncAnimMap[anim]
	if !exists {
		return
	}
	data := make([]byte, 4)
	binary.LittleEndian.PutUint32(data, instanceID)
	s.physicsSyncWrites = append(s.physicsSyncWrites, bind_group_provider.BufferWrite{
		Provider: s.physicsSyncGroup[sgIdx],
		Binding:  1,
		Offset:   uint64(bodyIdx) * 4,
		Data:     data,
	})
}

// createBoneParticleUpdateGroup sets up a per-kinematic-body bind group provider
// for the bone_particle_update compute shader. The BGP wires the shared particle and
// body buffers from the physics handler with the scratch_matrices buffer from the
// Animator's compute bind group (binding 5 in skeletal-compute.wgsl). A small uniform
// at binding 3 carries the particle range, bone count, and animator instance index.
// Caller must hold s.mu write lock.
//
// Parameters:
//   - anim: the skeletal Animator owning the bone matrices
//   - bodyIndex: the GPU body slot returned by RegisterBody
//   - mdl: the Model with skeleton data (for bone count)
//   - instanceIndex: the Animator instance slot for scratch_matrices indexing
func (s *scene) createBoneParticleUpdateGroup(anim animator.Animator, bodyIndex int, mdl model.Model, instanceIndex uint32) {
	ph := s.physicsHandler

	particleStart, particleCount := ph.BodyParticleInfo(bodyIndex)
	if particleCount == 0 {
		return
	}

	boneCount := uint32(len(mdl.Skeleton().Bones))

	// scratch_matrices lives at binding 5 in the skeletal-compute shader's BGP.
	// This is a manually-declared WGSL binding (no @oxy:group annotation).
	const scratchBinding = 5

	bgpLabel := fmt.Sprintf("bone_particle_update_%d", len(s.boneParticleUpdateGroups))
	bgp := bind_group_provider.NewBindGroupProvider(bgpLabel)

	// Wire shared physics buffers and the animator's scratch_matrices buffer.
	// model_data lives at binding 6 in the skeletal-compute shader's BGP.
	const modelDataBinding = 6

	bgp.SetBuffer(0, ph.Buffers().Buffer(1))                                 // particles
	bgp.SetBuffer(1, ph.Buffers().Buffer(0))                                 // bodies
	bgp.SetBuffer(2, anim.ComputeBindGroupProvider().Buffer(scratchBinding)) // scratch_matrices
	// Binding 3 (params uniform) is left unset so InitBindGroup creates a new buffer.
	bgp.SetBuffer(4, anim.ComputeBindGroupProvider().Buffer(modelDataBinding)) // model_data

	// Use the bone_update shader's layout descriptor to initialize the bind group.
	boneUpdateKey := ph.PipelineKey("bone_update")
	boneUpdatePipe := s.r.Pipeline(boneUpdateKey)
	if boneUpdatePipe == nil {
		panic("scene: bone_update pipeline not registered")
	}
	boneUpdateShader := boneUpdatePipe.Shader(shader.ShaderTypeCompute)
	boneUpdateDesc := boneUpdateShader.BindGroupLayoutDescriptor(0)

	if err := s.r.InitBindGroup(bgp, boneUpdateDesc, nil, nil); err != nil {
		panic(fmt.Sprintf("scene: failed to init bone particle update BGP: %v", err))
	}

	// Upload the constant params uniform (does not change frame-to-frame).
	paramsData := make([]byte, 16)
	binary.LittleEndian.PutUint32(paramsData[0:4], particleStart)
	binary.LittleEndian.PutUint32(paramsData[4:8], particleCount)
	binary.LittleEndian.PutUint32(paramsData[8:12], boneCount)
	binary.LittleEndian.PutUint32(paramsData[12:16], instanceIndex)
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: bgp, Binding: 3, Offset: 0, Data: paramsData},
	})

	s.boneParticleUpdateGroups = append(s.boneParticleUpdateGroups, &boneParticleUpdateGroup{
		bgp:           bgp,
		particleStart: particleStart,
		particleCount: particleCount,
		boneCount:     boneCount,
		instanceIndex: instanceIndex,
	})
}

// initMaterialGPU creates GPU resources (textures, samplers, bind group) for a single Material
// by inspecting the fragment shader's pre-processed Declarations for @oxy:provider annotations
// with the "material" identity. Multiple material groups are supported: each group with an
// @oxy:provider material annotation gets its own BindGroupProvider, enabling a single material
// to own resources across several bind groups (e.g. textures at group 2, effect uniforms at group 3).
// Per-binding roles (diffuse_texture, normal_texture, etc.) are resolved from the declaration Args,
// eliminating the need for variable-name string matching.
//
// Parameters:
//   - mat: the Material to initialize GPU resources for
//   - fragmentShader: the fragment shader whose @oxy:provider material annotations define the layout
//   - providerName: a unique name prefix for the created BindGroupProviders
//
// Returns:
//   - error: an error if GPU resource creation fails
func (s *scene) initMaterialGPU(mat material.Material, fragmentShader shader.Shader, providerName string) error {
	// Phase 1: Collect all groups annotated with @oxy:provider material and their binding roles.
	type groupInfo struct {
		bindingRoles map[int]shader.AnnotationArg
	}
	materialGroups := make(map[int]*groupInfo)

	for _, decl := range fragmentShader.Declarations() {
		if decl.Type != shader.AnnotationTypeProvider || decl.Group == nil {
			continue
		}
		if decl.Args[0] != shader.AnnotationArgMaterial {
			continue
		}
		g := *decl.Group
		if _, exists := materialGroups[g]; !exists {
			materialGroups[g] = &groupInfo{bindingRoles: make(map[int]shader.AnnotationArg)}
		}
		if len(decl.Args) > 1 && decl.Binding != nil {
			materialGroups[g].bindingRoles[*decl.Binding] = decl.Args[1]
		}
	}

	if len(materialGroups) == 0 {
		return nil
	}

	// Sort group indices so the lowest group (typically the texture group) becomes the
	// primary BindGroupProvider for backward-compatible access via mat.BindGroupProvider().
	groupIndices := make([]int, 0, len(materialGroups))
	for g := range materialGroups {
		groupIndices = append(groupIndices, g)
	}
	sort.Ints(groupIndices)

	// Texture role → material texture lookup
	type textureBinding struct {
		tex  *common.ImportedTexture
		role shader.AnnotationArg
	}
	roleToTexture := map[shader.AnnotationArg]textureBinding{
		shader.AnnotationArgDiffuseTexture:           {tex: mat.DiffuseTexture(), role: shader.AnnotationArgDiffuseTexture},
		shader.AnnotationArgNormalTexture:            {tex: mat.NormalTexture(), role: shader.AnnotationArgNormalTexture},
		shader.AnnotationArgMetallicRoughnessTexture: {tex: mat.MetallicRoughnessTexture(), role: shader.AnnotationArgMetallicRoughnessTexture},
	}
	textureSamplerPairs := map[shader.AnnotationArg]shader.AnnotationArg{
		shader.AnnotationArgDiffuseTexture:           shader.AnnotationArgDiffuseSampler,
		shader.AnnotationArgNormalTexture:            shader.AnnotationArgNormalSampler,
		shader.AnnotationArgMetallicRoughnessTexture: shader.AnnotationArgMetallicRoughnessSampler,
	}

	// Phase 2: For each material group, create a BGP, process textures/samplers, init bind group.
	firstGroup := true
	for _, groupIdx := range groupIndices {
		gi := materialGroups[groupIdx]
		provName := fmt.Sprintf("%s_g%d", providerName, groupIdx)
		provider := bind_group_provider.NewBindGroupProvider(provName)

		// Build binding→role reverse map for this group
		roleToBinding := make(map[shader.AnnotationArg]int)
		for binding, role := range gi.bindingRoles {
			roleToBinding[role] = binding
		}

		// Process user-supplied textures + their paired samplers
		for texRole, tb := range roleToTexture {
			if tb.tex == nil {
				continue
			}
			texBindingIdx, hasTexBinding := roleToBinding[texRole]
			if !hasTexBinding {
				continue
			}

			samplerRole := textureSamplerPairs[texRole]
			samplerBindingIdx, hasSamplerBinding := roleToBinding[samplerRole]

			pixels, width, height, err := tb.tex.Decode()
			if err != nil {
				return fmt.Errorf("failed to decode %s texture: %w", texRole, err)
			}
			isLinear := texRole == shader.AnnotationArgNormalTexture || texRole == shader.AnnotationArgMetallicRoughnessTexture
			stagingData := common.TextureStagingData{
				Pixels: pixels,
				Width:  width,
				Height: height,
				Linear: isLinear,
			}
			if err := s.r.InitTextureView(provider, texBindingIdx, stagingData); err != nil {
				return fmt.Errorf("failed to init %s texture view: %w", texRole, err)
			}
			if hasSamplerBinding {
				samplerData := common.SamplerStagingData{
					AddressModeU:  wgpu.AddressModeRepeat,
					AddressModeV:  wgpu.AddressModeRepeat,
					AddressModeW:  wgpu.AddressModeRepeat,
					MagFilter:     wgpu.FilterModeLinear,
					MinFilter:     wgpu.FilterModeLinear,
					MipmapFilter:  wgpu.MipmapFilterModeLinear,
					LodMinClamp:   0,
					LodMaxClamp:   32,
					MaxAnisotropy: 1,
				}
				if tb.tex.SamplerData != nil {
					samplerData = *tb.tex.SamplerData
				}
				if err := s.r.InitSampler(provider, samplerBindingIdx, samplerData); err != nil {
					return fmt.Errorf("failed to init %s sampler: %w", samplerRole, err)
				}
			}
		}

		// Fill in fallback textures/samplers for any texture or sampler bindings without data.
		descriptor := fragmentShader.BindGroupLayoutDescriptor(groupIdx)
		for _, entry := range descriptor.Entries {
			binding := int(entry.Binding)
			isTexture := entry.Texture.SampleType != wgpu.TextureSampleTypeBindingNotUsed
			isSampler := entry.Sampler.Type != wgpu.SamplerBindingTypeBindingNotUsed

			if isTexture && provider.TextureView(binding) == nil {
				role := gi.bindingRoles[binding]
				var pixel [4]byte
				switch role {
				case shader.AnnotationArgNormalTexture:
					pixel = [4]byte{128, 128, 255, 255}
				case shader.AnnotationArgMetallicRoughnessTexture:
					// Encode the material's scalar roughness (G) and metallic (B) into the
					// fallback 1×1 texture so shaders that read these values from the texture
					// (e.g. the G-Buffer pass for SSR) get the correct material properties.
					roughByte := byte(mat.Roughness() * 255)
					metalByte := byte(mat.Metallic() * 255)
					pixel = [4]byte{0, roughByte, metalByte, 255}
				default:
					pixel = [4]byte{255, 255, 255, 255}
				}
				isLinear := role == shader.AnnotationArgNormalTexture || role == shader.AnnotationArgMetallicRoughnessTexture
				fallback := common.TextureStagingData{
					Pixels: pixel[:],
					Width:  1,
					Height: 1,
					Linear: isLinear,
				}
				if err := s.r.InitTextureView(provider, binding, fallback); err != nil {
					return fmt.Errorf("failed to init fallback texture at binding %d: %w", binding, err)
				}
			}

			if isSampler && provider.Sampler(binding) == nil {
				fallbackSampler := common.SamplerStagingData{
					AddressModeU:  wgpu.AddressModeRepeat,
					AddressModeV:  wgpu.AddressModeRepeat,
					AddressModeW:  wgpu.AddressModeRepeat,
					MagFilter:     wgpu.FilterModeLinear,
					MinFilter:     wgpu.FilterModeLinear,
					MipmapFilter:  wgpu.MipmapFilterModeLinear,
					LodMinClamp:   0,
					LodMaxClamp:   32,
					MaxAnisotropy: 1,
				}
				if err := s.r.InitSampler(provider, binding, fallbackSampler); err != nil {
					return fmt.Errorf("failed to init fallback sampler at binding %d: %w", binding, err)
				}
			}
		}

		// Check for a material_params uniform binding in this group.
		paramsBinding := -1
		for binding, role := range gi.bindingRoles {
			if role == shader.AnnotationArgMaterialParams {
				paramsBinding = binding
				break
			}
		}
		var sizeOverrides map[int]uint64
		if paramsBinding >= 0 {
			sizeOverrides = map[int]uint64{
				paramsBinding: uint64((&material.GPUMaterialParams{}).Size()),
			}
		}

		if err := s.r.InitBindGroup(provider, descriptor, nil, sizeOverrides); err != nil {
			return fmt.Errorf("failed to init material bind group for group %d: %w", groupIdx, err)
		}

		if paramsBinding >= 0 {
			params := material.GPUMaterialParams{AlphaCutoff: mat.AlphaCutoff()}
			s.r.WriteBuffers([]bind_group_provider.BufferWrite{
				{Provider: provider, Binding: paramsBinding, Offset: 0, Data: params.Marshal()},
			})
		}

		// First material group becomes the primary BindGroupProvider for backward compat.
		if firstGroup {
			mat.SetBindGroupProvider(provider)
			firstGroup = false
		}
		mat.SetProvider(groupIdx, provider)
	}

	return nil
}

// createAnimator creates a new Animator for the given Model and registers lifecycle
// hooks that initialize and prune animator GPU resources. Caller must hold s.mu write lock.
//
// Parameters:
//   - mdl: the Model to create an Animator for
//   - computeShader: the compute shader for the animator's compute pipeline
//   - vertexShader: the vertex shader for the render pipeline
//   - fragmentShader: the fragment shader for the render pipeline
//   - pipelineOpts: optional pipeline builder options for the render pipeline
//
// Returns:
//   - animator.Animator: the configured Animator
func (s *scene) createAnimator(mdl model.Model, computeShader, vertexShader, fragmentShader shader.Shader, pipelineOpts ...pipeline.PipelineBuilderOption) animator.Animator {
	// Pick backend type based on whether the model uses skeletal animation
	backendType := animator.BackendTypeSimple
	if mdl.Skinned() {
		backendType = animator.BackendTypeSkeletal
	}

	// Discover binding indices for the skeletal animator's bone and packed animation buffers.
	// boneBinding targets the BoneInfo declaration (receives bone data via SetBone/Flush).
	// packedBinding targets the raw "anim_packed" buffer (receives clip/channel/keyframe data via AddClip).
	// For simple animators these default to 0 and are unused.
	boneBinding := 0
	packedBinding := 0
	if backendType == animator.BackendTypeSkeletal {
		for _, decl := range computeShader.Declarations() {
			if decl.Binding == nil {
				continue
			}
			switch decl.Type {
			case shader.AnnotationTypeBindingGroup:
				typeArg := string(decl.Args[2])
				if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
					typeArg = strings.TrimSuffix(stripped, ">")
				}
				if shader.AnnotationArg(typeArg) == shader.AnnotationArgBoneInfo {
					boneBinding = *decl.Binding
				}
			case shader.AnnotationTypeProvider:
				if decl.Args[0] == shader.AnnotationArgAnimatorPacked {
					packedBinding = *decl.Binding
				}
			}
		}
	}

	anim := animator.NewAnimator(backendType, animator.WithModel(mdl, boneBinding, packedBinding))
	anim.SetBoundingRadius(mdl.BoundingRadius())
	anim.SetBoundingBox(mdl.BoundingMin(), mdl.BoundingMax())
	anim.Lifecycle().OnTransitionTo(lifecycle.LifecycleStateStarting, lifecycle.Hook(func() error {
		s.initAnimatorGPU(anim, computeShader, vertexShader, fragmentShader, pipelineOpts...)
		return nil
	}))
	anim.Lifecycle().OnTransitionTo(lifecycle.LifecycleStateRemoved, lifecycle.Hook(func() error {
		s.pruneAnimator(anim)
		return nil
	}))

	return anim
}

// initAnimatorGPU initializes all renderer resources for an Animator and its model.
//
// Parameters:
//   - anim: the Animator to initialize
//   - computeShader: the compute shader for the animator's compute pipeline
//   - vertexShader: the vertex shader for the render pipeline
//   - fragmentShader: the fragment shader for the render pipeline
//   - pipelineOpts: optional pipeline builder options for the render pipeline
func (s *scene) initAnimatorGPU(anim animator.Animator, computeShader, vertexShader, fragmentShader shader.Shader, pipelineOpts ...pipeline.PipelineBuilderOption) {
	mdl := anim.Model()
	if mdl == nil {
		panic("scene: cannot init animator GPU resources without a Model")
	}

	backendType := anim.BackendType()

	// Init mesh provider GPU resources if not already done (e.g. hand-built models
	// skip this, while loader-produced models will already have VertexBuffer set).
	if meshBGP := mdl.MeshProvider(); meshBGP != nil && meshBGP.VertexBuffer() == nil {
		if err := s.r.InitMeshBuffers(meshBGP, mdl.VertexData(), mdl.IndexData(), mdl.IndexCount()); err != nil {
			panic(fmt.Sprintf("scene: failed to init mesh BGP for model %q: %v", mdl.Name(), err))
		}
	}

	for level := 1; level < mdl.LODCount(); level++ {
		lodBGP := mdl.LODMeshProvider(level)
		if lodBGP != nil && lodBGP.VertexBuffer() == nil {
			lodVData := mdl.LODVertexData(level)
			lodIData := mdl.LODIndexData(level)
			lodICount := mdl.LODIndexCount(level)
			if lodVData != nil && lodIData != nil {
				if err := s.r.InitMeshBuffers(lodBGP, lodVData, lodIData, lodICount); err != nil {
					panic(fmt.Sprintf("scene: failed to init LOD%d mesh BGP for model %q: %v", level, mdl.Name(), err))
				}
			}
		}
	}

	// Identify the compute group from the compute shader's declarations.
	// The animation data binding (simple or skeletal) identifies the correct group.
	computeGroup := 0
	for _, decl := range computeShader.Declarations() {
		if decl.Type != shader.AnnotationTypeBindingGroup || decl.Group == nil {
			continue
		}
		typeArg := string(decl.Args[2])
		if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
			typeArg = strings.TrimSuffix(stripped, ">")
		}
		switch shader.AnnotationArg(typeArg) {
		case shader.AnnotationArgAnimationData, shader.AnnotationArgSkeletalAnimationData:
			computeGroup = *decl.Group
		}
	}

	// Identify the output group in the vertex shader (contains the instance buffer
	// that the compute shader's output feeds into).
	// For static models this is an @oxy:group with AnnotationArgInstanceData.
	// For skinned models this is an @oxy:provider with AnnotationArgAnimator (raw vec4 buffer).
	outputGroup := 0
	outputInstanceBinding := 0
	for _, decl := range vertexShader.Declarations() {
		if decl.Group == nil {
			continue
		}
		switch decl.Type {
		case shader.AnnotationTypeBindingGroup:
			typeArg := string(decl.Args[2])
			if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
				typeArg = strings.TrimSuffix(stripped, ">")
			}
			if shader.AnnotationArg(typeArg) == shader.AnnotationArgInstanceData {
				outputGroup = *decl.Group
				if decl.Binding != nil {
					outputInstanceBinding = *decl.Binding
				}
			}
		case shader.AnnotationTypeProvider:
			if decl.Args[0] == shader.AnnotationArgAnimator {
				outputGroup = *decl.Group
				// Provider identifies the group; the raw instance binding is always 0.
				outputInstanceBinding = 0
			}
		}
	}

	// Derive the per-instance output size from the vertex shader's instance buffer.
	// The compute shader's output buffer and the vertex shader's instance buffer are
	// backed by the same physical GPU buffer, so the per-instance stride must match.
	outputDesc := vertexShader.BindGroupLayoutDescriptor(outputGroup)
	perInstanceOutputSize := uint64((&animator.GPUInstanceData{}).Size())
	for _, entry := range outputDesc.Entries {
		if int(entry.Binding) == outputInstanceBinding && entry.Buffer.MinBindingSize > 0 {
			perInstanceOutputSize = entry.Buffer.MinBindingSize
			break
		}
	}

	// For skeletal animators the output stride is NOT the array element size (vec4 = 16 bytes)
	// but the full per-instance payload: 1 model matrix + 1 flag vec4 + MAX_BONES bone matrices.
	// The WGSL parser returns the element stride for runtime-sized arrays (array<vec4<f32>> → 16),
	// which must be scaled up to the actual per-instance stride that both the compute and vertex
	// shaders use (5 vec4 for model+flags, then 4 vec4 per bone matrix).
	if backendType == animator.BackendTypeSkeletal {
		perInstanceOutputSize = 80 + s.maxBonesGPU*64
	}

	// Compute skeletal-specific sizing context (bone count, packed buffer size).
	var boneCount uint64
	var packedBufferSize uint64
	if backendType == animator.BackendTypeSkeletal && mdl.Skinned() && mdl.Skeleton() != nil {
		boneCount = uint64(len(mdl.Skeleton().Bones))

		// Compute packed animation buffer size from model data.
		// Packed layout: [clips × 4 u32] [channels × 8 u32] [keyframes × 16 u32]
		totalClips := 0
		totalChannels := 0
		totalKeyframes := 0
		for _, clip := range mdl.Animations() {
			totalClips++
			for _, ch := range clip.Channels {
				totalChannels++
				totalKeyframes += len(ch.PositionKeys) + len(ch.RotationKeys) + len(ch.ScaleKeys)
			}
		}
		totalU32s := totalClips*4 + totalChannels*8 + totalKeyframes*16
		packedBufferSize = uint64(totalU32s) * 4
		if packedBufferSize < 4 {
			packedBufferSize = 4
		}
	}

	// Build buffer size and usage overrides for the compute group.
	// Simple animators: all storage buffers are per-instance (maxInst × element stride).
	// Skeletal animators: bone and packed data are shared (not per-instance), scratch needs
	// extra capacity for blending (2 slots per instance × boneCount matrices).
	maxInst := uint64(anim.MaxInstances())
	computeDesc := computeShader.BindGroupLayoutDescriptor(computeGroup)
	computeSizeOverrides := make(map[int]uint64)
	computeUsageOverrides := make(map[int]wgpu.BufferUsage)

	// Build a binding→type map from the compute shader's declarations for typed bindings.
	computeBindingTypes := make(map[int]shader.AnnotationArg)
	animationDataBinding := -1
	for _, decl := range computeShader.Declarations() {
		if decl.Type != shader.AnnotationTypeBindingGroup || decl.Binding == nil {
			continue
		}
		typeArg := string(decl.Args[2])
		if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
			typeArg = strings.TrimSuffix(stripped, ">")
		}
		typedArg := shader.AnnotationArg(typeArg)
		computeBindingTypes[*decl.Binding] = typedArg
		if typedArg == shader.AnnotationArgAnimationData {
			animationDataBinding = *decl.Binding
		}
	}

	// Resolve raw bindings from provider declarations for output, packed, and scratch buffers.
	rawOutputBinding, rawPackedBinding, rawScratchBinding := -1, -1, -1
	for _, decl := range computeShader.Declarations() {
		if decl.Type != shader.AnnotationTypeProvider || decl.Binding == nil {
			continue
		}
		switch decl.Args[0] {
		case shader.AnnotationArgAnimatorOutput:
			rawOutputBinding = *decl.Binding
		case shader.AnnotationArgAnimatorPacked:
			rawPackedBinding = *decl.Binding
		case shader.AnnotationArgAnimatorScratch:
			rawScratchBinding = *decl.Binding
		}
	}

	computeOutputBinding := -1
	for _, entry := range computeDesc.Entries {
		binding := int(entry.Binding)

		// Check annotated bindings first.
		if typeArg, ok := computeBindingTypes[binding]; ok {
			switch typeArg {
			case shader.AnnotationArgIndirectArgs:
				// Indirect args buffer needs the Indirect usage flag for DrawIndexedIndirect.
				computeUsageOverrides[binding] = wgpu.BufferUsageIndirect
			case shader.AnnotationArgBoneInfo:
				// Shared bone info buffer: one entry per bone, not per-instance.
				if entry.Buffer.MinBindingSize > 0 {
					computeSizeOverrides[binding] = boneCount * entry.Buffer.MinBindingSize
				}
			case shader.AnnotationArgModelData:
				// Per-instance model matrices from CPU.
				if entry.Buffer.MinBindingSize > 0 {
					computeSizeOverrides[binding] = maxInst * entry.Buffer.MinBindingSize
				}
			case shader.AnnotationArgAnimationGlobals, shader.AnnotationArgGlobalData:
				// Uniform buffer — fixed size from the parser, no override needed.
			default:
				// Per-instance storage buffers (animation data, skeletal animation data, etc.).
				if (entry.Buffer.Type == wgpu.BufferBindingTypeStorage || entry.Buffer.Type == wgpu.BufferBindingTypeReadOnlyStorage) &&
					entry.Buffer.MinBindingSize > 0 {
					computeSizeOverrides[binding] = maxInst * entry.Buffer.MinBindingSize
				}
			}
			continue
		}

		// Handle raw (un-annotated) bindings by resolved var name.
		switch binding {
		case rawOutputBinding:
			// Output buffer stores per-instance data that the vertex shader reads.
			computeSizeOverrides[binding] = maxInst * perInstanceOutputSize
			computeOutputBinding = binding
		case rawPackedBinding:
			// Packed animation data buffer: clips, channels, keyframes packed as u32 array.
			computeSizeOverrides[binding] = packedBufferSize
		case rawScratchBinding:
			// Scratch bone matrix workspace: 2 slots per instance (for blending) × boneCount × mat4x4.
			computeSizeOverrides[binding] = maxInst * boneCount * 2 * 64
		}
	}

	if err := s.r.InitBindGroup(anim.ComputeBindGroupProvider(), computeDesc, computeUsageOverrides, computeSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init compute BGP for model %q: %v", mdl.Name(), err))
	}

	// Share the compute output buffer with the vertex shader's output BGP.
	// The compute shader writes per-instance data into the output buffer through the compute BGP,
	// and the vertex shader reads it from instance_buffer through the output BGP.
	// These must reference the same physical GPU buffer.
	if computeOutputBinding >= 0 {
		if sharedBuf := anim.ComputeBindGroupProvider().Buffer(computeOutputBinding); sharedBuf != nil {
			anim.OutputBindGroupProvider().SetBuffer(outputInstanceBinding, sharedBuf)
		}
	}

	// Build buffer size overrides for the output group (vertex shader instance buffer).
	// The shared buffer is already set on the output BGP for the instance binding,
	// so InitBindGroup will reuse it rather than creating a new buffer.
	outputSizeOverrides := make(map[int]uint64)
	for _, entry := range outputDesc.Entries {
		if int(entry.Binding) == outputInstanceBinding &&
			(entry.Buffer.Type == wgpu.BufferBindingTypeStorage || entry.Buffer.Type == wgpu.BufferBindingTypeReadOnlyStorage) &&
			entry.Buffer.MinBindingSize > 0 {
			outputSizeOverrides[int(entry.Binding)] = maxInst * perInstanceOutputSize
		}
	}

	if err := s.r.InitBindGroup(anim.OutputBindGroupProvider(), outputDesc, nil, outputSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init output BGP for model %q: %v", mdl.Name(), err))
	}

	// Slot-1 init for compute BGP and output BGP.
	// CPU-written bindings get fresh slot-1 buffers; GPU-only bindings share slot-0 physical buffers.
	computeProvider := anim.ComputeBindGroupProvider()
	outputProvider := anim.OutputBindGroupProvider()

	var slot0OutputBuf, slot0PackedBuf, slot0ScratchBuf *wgpu.Buffer
	if computeOutputBinding >= 0 {
		slot0OutputBuf = computeProvider.Buffer(computeOutputBinding)
	}
	if rawPackedBinding >= 0 {
		slot0PackedBuf = computeProvider.Buffer(rawPackedBinding)
	}
	if rawScratchBinding >= 0 {
		slot0ScratchBuf = computeProvider.Buffer(rawScratchBinding)
	}

	computeProvider.SetSlot(1)
	if slot0OutputBuf != nil {
		computeProvider.SetBuffer(computeOutputBinding, slot0OutputBuf)
	}
	if slot0PackedBuf != nil {
		computeProvider.SetBuffer(rawPackedBinding, slot0PackedBuf)
	}
	if slot0ScratchBuf != nil {
		computeProvider.SetBuffer(rawScratchBinding, slot0ScratchBuf)
	}
	if err := s.r.InitBindGroup(computeProvider, computeDesc, computeUsageOverrides, computeSizeOverrides); err != nil {
		panic(fmt.Sprintf("scene: failed to init compute BGP slot 1 for model %q: %v", mdl.Name(), err))
	}
	computeProvider.SetSlot(0)

	if slot0OutputBuf != nil {
		outputProvider.SetSlot(1)
		outputProvider.SetBuffer(outputInstanceBinding, slot0OutputBuf)
		if err := s.r.InitBindGroup(outputProvider, outputDesc, nil, outputSizeOverrides); err != nil {
			panic(fmt.Sprintf("scene: failed to init output BGP slot 1 for model %q: %v", mdl.Name(), err))
		}
		outputProvider.SetSlot(0)
	}

	if backendType == animator.BackendTypeSimple && animationDataBinding >= 0 {
		s.initSimpleShadowAnimationProvider(anim, animationDataBinding)
	}

	// Init Hi-Z bind group for the animator (both slots).
	// If the real SSR Hi-Z pyramid isn't ready yet, use the 1×1 fallback texture.
	// The shader guards access via hiz_mip_count == 0.
	hiZBGP := anim.HiZBindGroupProvider()
	if hiZBGP != nil {
		var hizView, maxHizView *wgpu.TextureView
		if s.lightHandler != nil {
			if ssrH := s.ssrHandler; ssrH != nil {
				hizView = ssrH.HiZTextureView()
				maxHizView = ssrH.HiZMaxTextureView()
			}
		}
		if hizView == nil {
			hizView = s.hizFallbackView
		}
		if maxHizView == nil {
			maxHizView = s.hizFallbackView
		}
		if hizView != nil && maxHizView != nil {
			hiZDesc := computeShader.BindGroupLayoutDescriptor(1)
			hiZBGP.SetTextureView(0, hizView)
			hiZBGP.SetTextureView(1, maxHizView)
			if err := s.r.InitBindGroup(hiZBGP, hiZDesc, nil, nil); err != nil {
				panic(fmt.Sprintf("scene: failed to init Hi-Z BGP for model %q: %v", mdl.Name(), err))
			}
			hiZBGP.SetSlot(1)
			hiZBGP.SetTextureView(0, hizView)
			hiZBGP.SetTextureView(1, maxHizView)
			if err := s.r.InitBindGroup(hiZBGP, hiZDesc, nil, nil); err != nil {
				panic(fmt.Sprintf("scene: failed to init Hi-Z BGP slot 1 for model %q: %v", mdl.Name(), err))
			}
			hiZBGP.SetSlot(0)
		}
	}

	shadowBuf := s.r.CreateBuffer("shadow_indirect", 20, wgpu.BufferUsageIndirect|wgpu.BufferUsageCopyDst)
	s.shadowIndirectBuffers[anim] = shadowBuf
	for _, decl := range computeShader.Declarations() {
		if decl.Type == shader.AnnotationTypeBindingGroup && decl.Binding != nil {
			typeArg := string(decl.Args[2])
			if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
				typeArg = strings.TrimSuffix(stripped, ">")
			}
			if shader.AnnotationArg(typeArg) == shader.AnnotationArgIndirectArgs {
				s.animIndirectBinding[anim] = *decl.Binding
				break
			}
		}
	}

	// Register compute pipeline
	cp := pipeline.NewPipeline(computeShader.Key(), pipeline.PipelineTypeCompute, pipeline.WithComputeShader(computeShader))
	if err := s.r.RegisterPipelines(cp); err != nil {
		panic(fmt.Sprintf("scene: failed to register compute pipeline for model %q: %v", mdl.Name(), err))
	}
	anim.Model().SetComputePipelineKey(cp.PipelineKey())

	// Register render pipeline with the model name as key, matching Material.PipelineKey()
	renderOpts := append([]pipeline.PipelineBuilderOption{
		pipeline.WithVertexShader(vertexShader),
		pipeline.WithFragmentShader(fragmentShader),
	}, pipelineOpts...)
	rp := pipeline.NewPipeline(mdl.Name(), pipeline.PipelineTypeRender, renderOpts...)
	if err := s.r.RegisterPipelines(rp); err != nil {
		panic(fmt.Sprintf("scene: failed to register render pipeline for model %q: %v", mdl.Name(), err))
	}

	// Init material GPU resources (textures, samplers, bind groups) for each material
	// that doesn't already have a bind group provider set. For materials with a non-empty
	// PipelineKey, a per-material pipeline is registered (or reused if already present).
	for i, mat := range mdl.RenderMaterials() {
		if mat.BindGroupProvider() != nil {
			continue
		}
		// Register or reuse the per-material pipeline.
		matPipelineKey := mat.PipelineKey()
		if matPipelineKey != "" && s.r.Pipeline(matPipelineKey) == nil {
			var matPipeOpts []pipeline.PipelineBuilderOption
			// Check if the material provides explicit pipeline options (shaders, blend, cull, etc.)
			if rawOpts := mat.PipelineOptions(); len(rawOpts) > 0 {
				typedOpts := make([]pipeline.PipelineBuilderOption, 0, len(rawOpts))
				for _, raw := range rawOpts {
					if opt, ok := raw.(pipeline.PipelineBuilderOption); ok {
						typedOpts = append(typedOpts, opt)
					}
				}
				// Prepend the model's standard vertex+fragment shaders so material opts
				// can override them via explicit WithVertexShader/WithFragmentShader entries.
				matPipeOpts = append([]pipeline.PipelineBuilderOption{
					pipeline.WithVertexShader(vertexShader),
					pipeline.WithFragmentShader(fragmentShader),
				}, typedOpts...)
			} else {
				// No material-level options: use the model's standard vertex+fragment shaders.
				matPipeOpts = []pipeline.PipelineBuilderOption{
					pipeline.WithVertexShader(vertexShader),
					pipeline.WithFragmentShader(fragmentShader),
				}
			}
			mp := pipeline.NewPipeline(matPipelineKey, pipeline.PipelineTypeRender, matPipeOpts...)
			if err := s.r.RegisterPipelines(mp); err != nil {
				panic(fmt.Sprintf("scene: failed to register pipeline %q for model %q material %d: %v", matPipelineKey, mdl.Name(), i, err))
			}
		}
		// Determine which fragment shader to use for GPU resource init.
		var fragShaderForMat shader.Shader
		if matPipelineKey != "" {
			if p := s.r.Pipeline(matPipelineKey); p != nil {
				fragShaderForMat = p.Shader(shader.ShaderTypeFragment)
			}
		}
		if fragShaderForMat == nil {
			fragShaderForMat = fragmentShader
		}
		providerName := fmt.Sprintf("%s_material_%d", mdl.Name(), i)
		if err := s.initMaterialGPU(mat, fragShaderForMat, providerName); err != nil {
			panic(fmt.Sprintf("scene: failed to init material GPU for model %q material %d: %v", mdl.Name(), i, err))
		}
	}

}

// initSimpleShadowAnimationProvider initializes the raw AnimationData bind group
// used by the non-skinned shadow depth shader.
//
// Parameters:
//   - anim: the simple animator whose AnimationData buffers back shadow draws
//   - animationDataBinding: compute bind-group binding index for array<animation_data>
func (s *scene) initSimpleShadowAnimationProvider(anim animator.Animator, animationDataBinding int) {
	if anim == nil || animationDataBinding < 0 {
		return
	}

	computeProvider := anim.ComputeBindGroupProvider()
	if computeProvider == nil {
		return
	}

	shadowVertShader := shader.NewShader("_shadow_simple_anim_init", shader.ShaderTypeVertex,
		"engine/light/assets/shadow-depth-vert.wgsl", shader.WithInjections(s.injections))
	shadowDesc := shadowVertShader.BindGroupLayoutDescriptor(1)
	bgp := bind_group_provider.NewBindGroupProvider(fmt.Sprintf("shadow_simple_anim_%p", anim))

	modelName := "<nil>"
	if mdl := anim.Model(); mdl != nil {
		modelName = mdl.Name()
	}

	currentSlot := s.r.CurrentFrameSlot()

	computeProvider.SetSlot(0)
	slot0AnimBuf := computeProvider.Buffer(animationDataBinding)
	if slot0AnimBuf == nil {
		computeProvider.SetSlot(currentSlot)
		panic(fmt.Sprintf("scene: missing AnimationData buffer at binding %d (slot 0) for model %q", animationDataBinding, modelName))
	}
	bgp.SetSlot(0)
	bgp.SetBuffer(0, slot0AnimBuf)
	if err := s.r.InitBindGroup(bgp, shadowDesc, nil, nil); err != nil {
		computeProvider.SetSlot(currentSlot)
		panic(fmt.Sprintf("scene: failed to init simple shadow animation BGP for model %q: %v", modelName, err))
	}

	computeProvider.SetSlot(1)
	slot1AnimBuf := computeProvider.Buffer(animationDataBinding)
	if slot1AnimBuf == nil {
		computeProvider.SetSlot(currentSlot)
		panic(fmt.Sprintf("scene: missing AnimationData buffer at binding %d (slot 1) for model %q", animationDataBinding, modelName))
	}
	bgp.SetSlot(1)
	bgp.SetBuffer(0, slot1AnimBuf)
	if err := s.r.InitBindGroup(bgp, shadowDesc, nil, nil); err != nil {
		computeProvider.SetSlot(currentSlot)
		panic(fmt.Sprintf("scene: failed to init simple shadow animation BGP slot 1 for model %q: %v", modelName, err))
	}

	computeProvider.SetSlot(currentSlot)
	bgp.SetSlot(currentSlot)

	if s.shadowAnimationProviders == nil {
		s.shadowAnimationProviders = make(map[animator.Animator]bind_group_provider.BindGroupProvider)
	}
	if old := s.shadowAnimationProviders[anim]; old != nil {
		releaseSharedBufferBindGroupProvider(old)
	}
	s.shadowAnimationProviders[anim] = bgp
}

// releaseSharedBufferBindGroupProvider clears shared buffer references before
// releasing provider-owned bind groups and layout.
//
// Parameters:
//   - provider: bind group provider that may reference externally-owned buffers
func releaseSharedBufferBindGroupProvider(provider bind_group_provider.BindGroupProvider) {
	if provider == nil {
		return
	}

	for slot := 0; slot < 2; slot++ {
		provider.SetSlot(slot)
		provider.SetBuffers(nil)
	}

	provider.Release()
}

// computeWorkgroupSize2D returns the workgroup size for a 2D compute dispatch.
//
// Falls back to the provided defaults if the pipeline or shader is not available.
func (s *scene) computeWorkgroupSize2D(pipeKey string, defaultX, defaultY uint32) (uint32, uint32) {
	pipe := s.r.Pipeline(pipeKey)
	if pipe == nil {
		return defaultX, defaultY
	}
	shdr := pipe.Shader(shader.ShaderTypeCompute)
	if shdr == nil {
		return defaultX, defaultY
	}
	wgSize := shdr.WorkgroupSize()
	x := wgSize[0]
	y := wgSize[1]
	if x == 0 {
		x = defaultX
	}
	if y == 0 {
		y = defaultY
	}
	return x, y
}

// releaseResolutionDependentResources releases old GPU textures and bind groups
// before resize re-initialization.
func (s *scene) releaseResolutionDependentResources() {
	gbh := s.gBufferHandler
	if gbh.Enabled() {
		for slot := 0; slot < 2; slot++ {
			gbh.SetSlot(slot)
			if v := gbh.NormalTextureView(); v != nil {
				v.Release()
			}
			if t := gbh.NormalTexture(); t != nil {
				t.Release()
			}
			gbh.SetNormalTextureView(nil)
			gbh.SetNormalTexture(nil)
			if v := gbh.AlbedoTextureView(); v != nil {
				v.Release()
			}
			if t := gbh.AlbedoTexture(); t != nil {
				t.Release()
			}
			gbh.SetAlbedoTextureView(nil)
			gbh.SetAlbedoTexture(nil)
			if v := gbh.DepthTextureView(); v != nil {
				v.Release()
			}
			if t := gbh.DepthTexture(); t != nil {
				t.Release()
			}
			gbh.SetDepthTextureView(nil)
			gbh.SetDepthTexture(nil)
		}
		gbh.SetSlot(0)
	}

	ssaoH := s.ssaoHandler
	if ssaoH.Enabled() {
		for slot := 0; slot < 2; slot++ {
			ssaoH.SetSlot(slot)
			if v := ssaoH.RawTextureView(); v != nil {
				v.Release()
			}
			if t := ssaoH.RawTexture(); t != nil {
				t.Release()
			}
			ssaoH.SetRawTextureView(nil)
			ssaoH.SetRawTexture(nil)
			if v := ssaoH.BlurredTextureView(); v != nil {
				v.Release()
			}
			if t := ssaoH.BlurredTexture(); t != nil {
				t.Release()
			}
			ssaoH.SetBlurredTextureView(nil)
			ssaoH.SetBlurredTexture(nil)
			if v := ssaoH.ScratchTextureView(); v != nil {
				v.Release()
			}
			if t := ssaoH.ScratchTexture(); t != nil {
				t.Release()
			}
			ssaoH.SetScratchTextureView(nil)
			ssaoH.SetScratchTexture(nil)
		}
		ssaoH.SetSlot(0)
		for _, key := range []string{"ssao_compute", "ssao_blur_h", "ssao_blur_v"} {
			if bgp := ssaoH.Bgp(key); bgp != nil {
				if bg := bgp.BindGroup(); bg != nil {
					bg.Release()
				}
				bgp.SetBindGroup(nil)
			}
		}
	}

	if bgp := s.lightHandler.Bgp("ssao_lit"); bgp != nil {
		if bg := bgp.BindGroup(); bg != nil {
			bg.Release()
		}
		bgp.SetBindGroup(nil)
	}

	csh := s.lightHandler.ContactShadowHandler()
	if csh.Enabled() {
		for slot := 0; slot < 2; slot++ {
			csh.SetSlot(slot)
			if v := csh.TextureView(); v != nil {
				v.Release()
			}
			if t := csh.Texture(); t != nil {
				t.Release()
			}
			csh.SetTextureView(nil)
			csh.SetTexture(nil)
		}
		csh.SetSlot(0)
		if bgp := csh.Bgp("contact_shadow_compute"); bgp != nil {
			for slot := 0; slot < 2; slot++ {
				bgp.SetSlot(slot)
				if bg := bgp.BindGroup(); bg != nil {
					bg.Release()
				}
				bgp.SetBindGroup(nil)
			}
			bgp.SetSlot(0)
		}
	}

	sh := s.lightHandler.ShadowHandler()
	if sh.CSMAtlasTexture() != nil {
		if bgp := sh.Bgp("csm_shadow_lit"); bgp != nil {
			if bg := bgp.BindGroup(); bg != nil {
				bg.Release()
			}
			bgp.SetBindGroup(nil)
		}
	}

	ch := s.compositionHandler
	if ch.Enabled() {
		mipCount := ch.BloomMipCount()
		for slot := 0; slot < 2; slot++ {
			ch.SetSlot(slot)
			if v := ch.HDRTextureView(); v != nil {
				v.Release()
			}
			if t := ch.HDRTexture(); t != nil {
				t.Release()
			}
			ch.SetHDRTextureView(nil)
			ch.SetHDRTexture(nil)
			if v := ch.MSAATextureView(); v != nil {
				v.Release()
			}
			if t := ch.MSAATexture(); t != nil {
				t.Release()
			}
			ch.SetMSAATextureView(nil)
			ch.SetMSAATexture(nil)
			if v := ch.DepthTextureView(); v != nil {
				v.Release()
			}
			if t := ch.DepthTexture(); t != nil {
				t.Release()
			}
			ch.SetDepthTextureView(nil)
			ch.SetDepthTexture(nil)
			if ch.BloomDownTexture() != nil {
				ch.BloomDownTexture().Release()
			}
			for _, v := range ch.BloomDownReadViews() {
				if v != nil {
					v.Release()
				}
			}
			for _, v := range ch.BloomDownStorageViews() {
				if v != nil {
					v.Release()
				}
			}
			ch.SetBloomDownTexture(nil)
			ch.SetBloomDownReadViews(nil)
			ch.SetBloomDownStorageViews(nil)
			if ch.BloomUpTexture() != nil {
				ch.BloomUpTexture().Release()
			}
			for _, v := range ch.BloomUpReadViews() {
				if v != nil {
					v.Release()
				}
			}
			for _, v := range ch.BloomUpStorageViews() {
				if v != nil {
					v.Release()
				}
			}
			if ch.BloomUpMip0View() != nil {
				ch.BloomUpMip0View().Release()
			}
			ch.SetBloomUpTexture(nil)
			ch.SetBloomUpReadViews(nil)
			ch.SetBloomUpStorageViews(nil)
			ch.SetBloomUpMip0View(nil)
		}
		ch.SetSlot(0)
		if bgp := ch.Bgp("composition"); bgp != nil {
			bgp.SetSampler(1, nil)
			bgp.SetSampler(3, nil)
			if bg := bgp.BindGroup(); bg != nil {
				bg.Release()
			}
			bgp.SetBindGroup(nil)
		}
		if bgp := ch.Bgp("luminance_compute"); bgp != nil {
			if bg := bgp.BindGroup(); bg != nil {
				bg.Release()
			}
			bgp.SetBindGroup(nil)
		}
		for i := 0; i < mipCount; i++ {
			if bgp := ch.Bgp(fmt.Sprintf("bloom_down_%d", i)); bgp != nil {
				bgp.SetSamplers(nil)
				bgp.Release()
			}
			if bgp := ch.Bgp(fmt.Sprintf("bloom_up_%d", i)); bgp != nil {
				bgp.SetSamplers(nil)
				bgp.Release()
			}
		}
		if samp := ch.LinearSampler(); samp != nil {
			samp.Release()
		}
		ch.SetLinearSampler(nil)
	}

	ssrH := s.ssrHandler
	if ssrH.Enabled() {
		hizMipCount := ssrH.HiZMipCount()
		for slot := 0; slot < 2; slot++ {
			ssrH.SetSlot(slot)
			if v := ssrH.SSRTextureView(); v != nil {
				v.Release()
			}
			if t := ssrH.SSRTexture(); t != nil {
				t.Release()
			}
			ssrH.SetSSRTextureView(nil)
			ssrH.SetSSRTexture(nil)
			if v := ssrH.HiZTextureView(); v != nil {
				v.Release()
			}
			if t := ssrH.HiZTexture(); t != nil {
				t.Release()
			}
			for _, v := range ssrH.HiZMipReadViews() {
				if v != nil {
					v.Release()
				}
			}
			for _, v := range ssrH.HiZStorageViews() {
				if v != nil {
					v.Release()
				}
			}
			ssrH.SetHiZTextureView(nil)
			ssrH.SetHiZTexture(nil)
			ssrH.SetHiZMipReadViews(nil)
			ssrH.SetHiZStorageViews(nil)
		}
		ssrH.SetSlot(0)
		if bgp := ssrH.Bgp("ssr_compute"); bgp != nil {
			if bg := bgp.BindGroup(); bg != nil {
				bg.Release()
			}
			bgp.SetBindGroup(nil)
		}
		if bgp := ssrH.Bgp("hiz_init"); bgp != nil {
			bgp.Release()
		}
		for i := 1; i < hizMipCount; i++ {
			if bgp := ssrH.Bgp(fmt.Sprintf("hiz_down_%d", i)); bgp != nil {
				bgp.Release()
			}
		}
		if bgp := ssrH.Bgp("hiz_init_max"); bgp != nil {
			bgp.Release()
		}
		for i := 1; i < hizMipCount; i++ {
			if bgp := ssrH.Bgp(fmt.Sprintf("hiz_down_max_%d", i)); bgp != nil {
				bgp.Release()
			}
		}
	}

	taaH := s.taaHandler
	if taaH != nil && taaH.Enabled() {
		for slot := 0; slot < 2; slot++ {
			taaH.SetSlot(slot)
			if v := taaH.TAATextureView(); v != nil {
				v.Release()
			}
			if t := taaH.TAATexture(); t != nil {
				t.Release()
			}
			taaH.SetTAATextureView(nil)
			taaH.SetTAATexture(nil)
		}
		taaH.SetSlot(0)
		for _, key := range []string{"taa_resolve_0", "taa_resolve_1"} {
			if bgp := taaH.Bgp(key); bgp != nil {
				if bg := bgp.BindGroup(); bg != nil {
					bg.Release()
				}
				bgp.SetBindGroup(nil)
			}
		}

		if sv := taaH.SharpenTextureView(); sv != nil {
			sv.Release()
		}
		if st := taaH.SharpenTexture(); st != nil {
			st.Release()
		}
		taaH.SetSharpenTextureView(nil)
		taaH.SetSharpenTexture(nil)

		for _, key := range []string{"taa_sharpen_0", "taa_sharpen_1"} {
			if bgp := taaH.Bgp(key); bgp != nil {
				if bg := bgp.BindGroup(); bg != nil {
					bg.Release()
				}
				bgp.SetBindGroup(nil)
			}
		}
	}
}

// (RegisterPipelines checks pipelineCache for existing keys). InitBindGroup
// reuses existing layouts and buffers.
//
// Parameters:
//   - w: new width in pixels
//   - h: new height in pixels
func (s *scene) resizePostProcessing(w, h int) {
	if !s.postProcessingInitialized {
		return
	}

	s.r.WaitIdle()
	s.releaseResolutionDependentResources()
	s.drawBindGroupCache = nil

	if s.gBufferHandler.Enabled() {
		s.initGBuffer()
	}
	if s.ssaoHandler.Enabled() {
		s.initSSAO()
	}
	if s.lightHandler.ContactShadowHandler().Enabled() {
		s.initContactShadows()
	}

	litFragShader := shader.NewShader("_lit_frag_resize", shader.ShaderTypeFragment,
		"engine/light/assets/lit-frag-csm.wgsl", shader.WithInjections(s.injections))

	s.initSSAOLitBindGroup(litFragShader)

	if s.lightHandler.ShadowHandler().CSMAtlasTexture() != nil {
		s.initCSMShadowLitBindGroup(litFragShader)
	}

	if s.compositionHandler.Enabled() {
		s.initComposition()
	}
	if s.ssrHandler.Enabled() {
		s.initSSR()
	}

	if s.ssrHandler.Enabled() && s.compositionHandler.Enabled() {
		compBGP := s.compositionHandler.Bgp("composition")
		if compBGP != nil && s.ssrHandler.SSRTextureView() != nil {
			compBGP.SetTextureView(2, s.ssrHandler.SSRTextureView())
			compFrag := shader.NewShader("_composition_frag_rebind", shader.ShaderTypeFragment,
				"engine/renderer/postprocessing/composition/assets/composition-frag.wgsl", shader.WithInjections(s.injections))
			compDesc := compFrag.BindGroupLayoutDescriptor(0)
			sizeOverrides := map[int]uint64{
				4: uint64((&composition.GPUCompositionParams{}).Size()),
			}
			if err := s.r.InitBindGroup(compBGP, compDesc, nil, sizeOverrides); err != nil {
				panic(fmt.Sprintf("scene: failed to re-init composition bind group on resize: %v", err))
			}
		}
	}

	if s.taaHandler != nil && s.taaHandler.Enabled() {
		s.initTAA()
	}
}

// pruneAnimator removes an empty animator from the pool and releases all GPU
// resources it owns.
// Mesh provider GPU resources are not released — they are model-owned and may
// be shared across animators.
func (s *scene) pruneAnimator(a animator.Animator) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Remove from animatorPool.
	mdl := a.Model()
	if mdl != nil {
		pool := s.animatorPool[mdl]
		for i, p := range pool {
			if p == a {
				pool = append(pool[:i], pool[i+1:]...)
				break
			}
		}
		if len(pool) == 0 {
			delete(s.animatorPool, mdl)
		} else {
			s.animatorPool[mdl] = pool
		}
	}

	// Release shadow indirect buffer.
	if buf, ok := s.shadowIndirectBuffers[a]; ok && buf != nil {
		buf.Release()
	}
	delete(s.shadowIndirectBuffers, a)
	delete(s.animIndirectBinding, a)
	if bgp, ok := s.shadowAnimationProviders[a]; ok && bgp != nil {
		releaseSharedBufferBindGroupProvider(bgp)
	}
	delete(s.shadowAnimationProviders, a)

	// Release animator GPU resources (compute + output BGPs, both slots).
	a.Release()

	// Remove reverse-index entry.
	delete(s.instanceLookup, a)
	delete(s.lodLevelCache, a)

	// Notify the shadow system to re-render next frame, clearing any stale atlas entries.
	if sh := s.lightHandler.ShadowHandler(); sh != nil {
		sh.MarkAllDirty()
	}
}

// refreshAnimatorHiZBindGroups rebuilds the Hi-Z bind groups for all registered animators
// using the current SSR Hi-Z texture views. Must be called when the Hi-Z texture changes
// (e.g. after initSSR or a resize). Lock-free — caller must hold s.mu or guarantee
// exclusive access (initSSR already holds s.mu.Lock()).
func (s *scene) refreshAnimatorHiZBindGroups() {
	if s.lightHandler == nil {
		return
	}
	ssrH := s.ssrHandler
	if ssrH == nil {
		return
	}

	hizViews := [2]*wgpu.TextureView{}
	maxHizViews := [2]*wgpu.TextureView{}
	for slot := 0; slot < 2; slot++ {
		ssrH.SetSlot(slot)
		hizViews[slot] = ssrH.HiZTextureView()
		maxHizViews[slot] = ssrH.HiZMaxTextureView()
	}
	ssrH.SetSlot(0)

	for _, animGroup := range s.animatorPool {
		for _, a := range animGroup {
			hiZBGP := a.HiZBindGroupProvider()
			if hiZBGP == nil {
				continue
			}
			mdl := a.Model()
			if mdl == nil {
				continue
			}
			computeKey := mdl.ComputePipelineKey()
			pipe := s.r.Pipeline(computeKey)
			if pipe == nil {
				continue
			}
			computeShdr := pipe.Shader(shader.ShaderTypeCompute)
			if computeShdr == nil {
				continue
			}
			hiZDesc := computeShdr.BindGroupLayoutDescriptor(1)
			for slot := 0; slot < 2; slot++ {
				view := hizViews[slot]
				if view == nil {
					view = s.hizFallbackView
				}
				if view == nil {
					continue
				}
				maxView := maxHizViews[slot]
				if maxView == nil {
					maxView = s.hizFallbackView
				}
				if maxView == nil {
					continue
				}
				hiZBGP.SetSlot(slot)
				hiZBGP.SetTextureView(0, view)
				hiZBGP.SetTextureView(1, maxView)
				if err := s.r.InitBindGroup(hiZBGP, hiZDesc, nil, nil); err != nil {
					continue
				}
			}
			hiZBGP.SetSlot(0)
		}
	}
}

func (s *scene) prepareTAA() {
	if s.lightHandler == nil || s.cam == nil {
		return
	}
	taaH := s.taaHandler
	if taaH == nil || !taaH.Enabled() {
		return
	}

	// Halton sequence for sub-pixel jitter (base 2 for X, base 3 for Y).
	// Returns value in [-0.5, 0.5] pixel space, then converted to NDC.
	halton := func(index uint64, base uint64) float32 {
		var result float64
		f := 1.0 / float64(base)
		i := index
		for i > 0 {
			result += f * float64(i%base)
			i /= base
			f /= float64(base)
		}
		return float32(result - 0.5) // center around 0
	}

	// Wrap the Halton sequence at N=8 per Yang et al. [YNS*09] §3.1 recommendation.
	// UE4 uses an 8-sample Halton(2,3) sequence by default. Bounded wrapping ensures
	// all 8 sub-pixel positions are evenly covered each cycle regardless of frame count,
	// and avoids floating-point precision issues in the Halton computation at high indices.
	nextIdx := (taaH.FrameIndex() + 1) % 8
	jitterScale := taaH.JitterScale()
	jx := halton(nextIdx, 2) * jitterScale
	jy := halton(nextIdx, 3) * jitterScale
	taaH.AdvanceFrame(jx, jy)

	// Convert pixel jitter to projection-matrix NDC element units.
	sw := float32(taaH.ScreenWidth())
	sh := float32(taaH.ScreenHeight())
	var ndcX, ndcY float32
	if sw > 0 {
		ndcX = jx * 2.0 / sw
	}
	if sh > 0 {
		ndcY = jy * 2.0 / sh
	}

	// Schedule the jitter for the NEXT frame's cam.Update().
	s.cam.SetJitter(ndcX, ndcY)

	// Build GPUTAAParams from the camera's CURRENT (jittered) matrices.
	// currVP reflects the jittered VP for the current frame (applied last frame's prepareTAA jitter).
	// prevVP reflects the jittered VP that was active in the previous frame.
	currVP := s.cam.ViewProjectionMatrix()
	var invCurrVP [16]float32
	common.Invert4(invCurrVP[:], currVP[:])
	rawHistoryOnly := float32(0.0)
	if taaH.RawHistoryOnly() {
		rawHistoryOnly = 1.0
	}

	params := taa.GPUTAAParams{
		InvCurrViewProj:           invCurrVP,
		PrevViewProj:              s.cam.PrevViewProjectionMatrix(),
		JitterCurr:                [2]float32{taaH.JitterX(), taaH.JitterY()},
		JitterPrev:                [2]float32{taaH.PrevJitterX(), taaH.PrevJitterY()},
		ScreenWidth:               float32(taaH.ScreenWidth()),
		ScreenHeight:              float32(taaH.ScreenHeight()),
		BlendFactor:               taaH.BlendFactor(),
		HistoryRectificationScale: taaH.HistoryRectificationScale(),
		RawHistoryOnly:            rawHistoryOnly,
	}

	// Select the active slot's BGP using the renderer's current frame slot.
	slot := s.r.CurrentFrameSlot()

	bgpKey := "taa_resolve_0"
	if slot == 1 {
		bgpKey = "taa_resolve_1"
	}
	bgp := taaH.Bgp(bgpKey)

	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: bgp, Binding: 0, Offset: 0, Data: params.Marshal()},
	})

	w := uint32(taaH.ScreenWidth())
	h := uint32(taaH.ScreenHeight())
	wgX := (w + 7) / 8
	wgY := (h + 7) / 8

	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{
			PipelineKey:    taaH.PipelineKey("taa_resolve"),
			Providers:      []renderer.ComputeGroupProvider{{Group: 0, Provider: bgp}},
			WorkGroupCount: [3]uint32{wgX, wgY, 1},
		},
	})

	casBGPKey := "taa_sharpen_0"
	if slot == 1 {
		casBGPKey = "taa_sharpen_1"
	}
	casBGP := taaH.Bgp(casBGPKey)

	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{
			PipelineKey:    taaH.PipelineKey("taa_sharpen"),
			Providers:      []renderer.ComputeGroupProvider{{Group: 0, Provider: casBGP}},
			WorkGroupCount: [3]uint32{wgX, wgY, 1},
		},
	})
}
