// Package scene provides the orchestration layer for frame-level scene execution and rendering.
//
// It coordinates the [Scene] lifecycle around camera and renderer ownership, animator-driven
// draw preparation, light and shadow integration, physics hooks, and per-frame render/compute
// preparation flow. Construct scenes with [NewScene], and use the exported [Scene] interface as
// the primary entrypoint for scene management.
package scene

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/Carmen-Shannon/automation/tools/worker"
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
)

// Scene manages a collection of Animators (registered implicitly via Add) and an
// optional registry of non-ephemeral GameObjects, with a Camera and Renderer for
// rendering. Rendering is driven entirely by the registered Animator list — each
// Animator owns its instance data and material.
// Scenes can be hot-swapped via the Active flag to switch between different views or levels.
// Thread-safe for concurrent access.
type Scene interface {
	// Name returns the scene's identifier.
	Name() string

	// SetName sets the scene's identifier.
	SetName(name string)

	// Active returns whether this scene is currently active for rendering.
	Active() bool

	// SetActive sets whether this scene is active for rendering.
	SetActive(active bool)

	// Camera returns the scene's camera.
	Camera() camera.Camera

	// SetCamera replaces the scene's camera.
	//
	// Parameters:
	//   - cam: the new camera
	SetCamera(cam camera.Camera)

	// Renderer returns the scene's renderer.
	Renderer() renderer.Renderer

	// SetRenderer replaces the scene's renderer.
	//
	// Parameters:
	//   - r: the new renderer
	SetRenderer(r renderer.Renderer)

	// SetPhysicsHandler replaces the scene's physics handler. This should be called
	// before adding any rigid body objects. If not set, a default handler is created
	// lazily when the first rigid body object is added.
	//
	// Parameters:
	//   - ph: the pre-configured Physics instance
	SetPhysicsHandler(ph physics.Physics)

	// Count returns the number of persisted GameObjects in the scene's registry. Does not include ephemeral objects.
	//
	// Returns:
	//   - int: count of non-ephemeral GameObjects in the registry
	Count() int

	// CountEphemeral returns the number of ephemeral GameObjects currently being rendered through the scene's animators.
	//
	// Returns:
	//   - int: count of ephemeral GameObjects currently rendered
	CountEphemeral() int

	// Add adds a GameObject to the scene. The scene's Renderer must be attached
	// and the object must carry a Model. The scene automatically creates and manages
	// an Animator for each unique Model, registers its compute and render pipelines,
	// initializes GPU resources, and adds a new instance wired with the object's
	// initial transform data. If the object is not ephemeral it is also persisted
	// in the registry for later lookup or removal by ID.
	//
	// Compute, vertex, and fragment shaders are resolved automatically from the
	// engine's standard shader assets based on whether the model is skinned.
	//
	// Panics if the scene has no Renderer or the object has no Model.
	//
	// Parameters:
	//   - obj: the GameObject to add
	//   - pipelineOpts: optional pipeline builder options for the render pipeline (e.g., blending)
	//
	// Returns:
	//   - uint64: the assigned object ID
	AddGameObject(obj game_object.GameObject, pipelineOpts ...pipeline.PipelineBuilderOption) uint64

	// Get retrieves a non-ephemeral GameObject by its ID.
	// Returns nil if not found.
	//
	// Parameters:
	//   - id: the object's unique ID
	//
	// Returns:
	//   - game_object.GameObject: the object or nil
	Get(id uint64) game_object.GameObject

	// RemoveGameObject removes a non-ephemeral GameObject from the registry by ID
	// and swap-removes the instance data from its animator.
	//
	// Parameters:
	//   - id: the object's unique ID
	RemoveGameObject(id uint64)

	// SyncFrameSlot switches all dual-slot GPU resources (BGPs and post-processing textures)
	// to the given frame slot. Must be called once per frame after SyncGPUTimestamps and
	// before any Prepare* calls.
	//
	// Parameters:
	//   - slot: the frame slot index to activate (0 or 1)
	SyncFrameSlot(slot int)

	// PrepareCompute updates camera matrices, advances animation state,
	// uploads staged buffer writes, and dispatches all compute shaders for this scene.
	// Must be called within a BeginComputeFrame/EndComputeFrame block on the renderer.
	//
	// Parameters:
	//   - deltaTime: elapsed time since the last frame in seconds
	PrepareCompute(deltaTime float32)

	// CullingDisabled returns whether GPU frustum culling is explicitly disabled for this scene.
	// When true, the scene will not distribute frustum planes to animators, keeping them in
	// non-culled mode even when a camera is present.
	//
	// Returns:
	//   - bool: true if culling is disabled
	CullingDisabled() bool

	// SetCullingDisabled enables or disables GPU frustum culling for this scene.
	// When set to true, the scene skips frustum plane distribution and animators
	// fall back to non-culled rendering with regular draw calls.
	//
	// Parameters:
	//   - disabled: true to disable culling, false to enable it
	SetCullingDisabled(disabled bool)

	// DrawCalls issues instanced draw calls for each registered animator.
	// Must be called within a BeginFrame/EndFrame block on the renderer.
	//
	// Returns:
	//   - error: error if a draw call fails
	DrawCalls() error

	// AddLight adds a light source to the scene and lazily initializes the full
	// lighting pipeline (light storage buffer, shadow map, Forward+ culling) on
	// the first call. Subsequent calls simply append the light without
	// re-initializing GPU resources. The lighting shaders are loaded internally
	// from the engine's standard light shader assets. Screen dimensions for
	// Forward+ tile culling are taken from the scene's stored screen size
	// (set via Resize or WithScreenSize).
	//
	// Parameters:
	//   - l: the Light to add
	AddLight(l light.Light)

	// Resize updates the scene's stored screen dimensions and propagates the
	// change to the renderer surface, camera aspect ratio, and (when lighting
	// is enabled) the Forward+ tile grid. Call this from the window's resize
	// callback or whenever the surface dimensions change.
	//
	// Parameters:
	//   - width: the new width in pixels
	//   - height: the new height in pixels
	Resize(width, height int)

	// RemoveLight removes a light source from the scene by reference.
	//
	// Parameters:
	//   - l: the Light to remove
	RemoveLight(l light.Light)

	// DetachLight removes a game object's attached light from the scene's tracking
	// and light lists. This is the cleanup counterpart for objects whose lights
	// were auto-registered during Add(). Non-ephemeral objects are cleaned up
	// automatically via Remove(), but ephemeral object owners must call this
	// explicitly when the object's lifetime ends.
	//
	// Parameters:
	//   - obj: the GameObject whose attached light should be detached
	DetachLight(obj game_object.GameObject)

	// Lights returns all lights currently registered in the scene.
	//
	// Returns:
	//   - []light.Light: the scene's light list
	Lights() []light.Light

	// AmbientColor returns the scene's ambient light color.
	//
	// Returns:
	//   - [3]float32: the ambient RGB color
	AmbientColor() [3]float32

	// SetAmbientColor sets the scene's ambient light color.
	//
	// Parameters:
	//   - color: the ambient RGB color
	SetAmbientColor(color [3]float32)

	// PrepareShadows computes the directional light's view-projection, updates the
	// shadow uniform buffer, and renders the depth-only shadow pass for all drawables.
	// Must be called after PrepareCompute and before BeginFrame each frame.
	// No-ops if no shadow map has been initialized or no shadow-casting directional
	// light exists.
	PrepareShadows()

	// PrepareLights marshals the current light list into the GPU light buffer.
	// Must be called after PrepareShadows (so that the shadow index map is up to date)
	// and before PrepareGBuffer each frame. No-ops if lighting is not initialized.
	PrepareLights()

	// PrepareLightCulling updates the light cull uniform buffer and dispatches the
	// light culling compute shader. Must be called after PrepareCompute (so lights
	// are uploaded) and before DrawCalls.
	PrepareLightCulling()

	// PrepareGBuffer renders the G-Buffer MRT pre-pass for all drawables.
	// Writes world position, normals, and albedo to off-screen textures
	// consumed by screen-space effects (SSAO, SSR). Must be called after
	// PrepareCompute and before PrepareSSAO each frame.
	// No-ops if the G-Buffer subsystem has not been initialized.
	PrepareGBuffer()

	// PrepareSSAO dispatches the SSAO hemisphere sampling compute shader
	// and the bilateral blur passes. Must be called after PrepareGBuffer
	// (so G-Buffer textures are populated) and before DrawCalls.
	// No-ops if the SSAO subsystem has not been initialized.
	PrepareSSAO()

	// PrepareContactShadows dispatches the contact shadow screen-space ray
	// march compute shader. Must be called after PrepareGBuffer (so the
	// G-Buffer depth texture is populated) and before DrawCalls.
	// No-ops if the contact shadow subsystem has not been initialized.
	PrepareContactShadows()

	// PrepareSSR dispatches the SSR compute shader to perform screen-space ray
	// marching against the G-Buffer. Must be called after DrawCalls (so the HDR
	// texture is populated) and before PrepareComposition. No-ops if the SSR
	// subsystem has not been initialized.
	PrepareSSR()

	// PrepareLuminance dispatches the luminance compute shader to update the
	// adapted exposure storage buffer based on the current HDR frame luminance.
	// Must be called after DrawCalls (so the HDR texture is populated) and
	// after PrepareSSR. No-ops if auto-exposure is disabled or the composition
	// handler has not been initialized.
	//
	// Parameters:
	//   - dt: elapsed time since the last frame in seconds
	PrepareLuminance(dt float32)

	// PrepareBloom dispatches the bloom downsample and upsample compute passes
	// to produce the final bloom texture from the HDR scene output. Must be
	// called after PrepareLuminance and before PrepareComposition. No-ops if
	// bloom is disabled or the composition handler has not been initialized.
	PrepareBloom()

	// PrepareComposition runs the fullscreen composition pass: samples the HDR lit
	// texture and optional SSR texture, applies ACES tone mapping and gamma
	// correction, and writes the final LDR result to the swapchain.
	// AcquireCompositionFrame must be called before this method each frame.
	// Must be called after DrawCalls (and PrepareSSR if active) and before Present.
	// No-ops if the composition subsystem has not been initialized.
	PrepareComposition()

	// AcquireCompositionFrame acquires the swapchain image for the composition
	// pass. Must be called immediately before PrepareComposition each frame.
	// No-ops if the renderer is nil or the scene is inactive, returning nil.
	//
	// Returns:
	//   - error: an error if the swapchain image could not be acquired
	AcquireCompositionFrame() error

	// BeginHDRFrame starts the HDR render pass using this scene's composition
	// handler textures. Returns an error if the composition handler is
	// not initialized or the render pass cannot be started.
	//
	// Returns:
	//   - error: an error if the HDR frame could not be started
	BeginHDRFrame() error
}

var _ Scene = &scene{}

func (s *scene) Name() string                         { return s.name }
func (s *scene) SetName(name string)                  { s.name = name }
func (s *scene) Active() bool                         { return s.active }
func (s *scene) SetActive(active bool)                { s.active = active }
func (s *scene) Camera() camera.Camera                { return s.cam }
func (s *scene) SetCamera(cam camera.Camera)          { s.cam = cam }
func (s *scene) Renderer() renderer.Renderer          { return s.r }
func (s *scene) SetRenderer(r renderer.Renderer)      { s.r = r }
func (s *scene) SetPhysicsHandler(ph physics.Physics) { s.physicsHandler = ph }
func (s *scene) CullingDisabled() bool                { return s.cullingDisabled }
func (s *scene) SetCullingDisabled(disabled bool)     { s.cullingDisabled = disabled }
func (s *scene) AmbientColor() [3]float32             { return s.lightHandler.AmbientColor() }
func (s *scene) SetAmbientColor(color [3]float32)     { s.lightHandler.SetAmbientColor(color) }
func (s *scene) Get(id uint64) game_object.GameObject { return s.registry[id] }

func (s *scene) AddLight(l light.Light) {
	if !s.lightHandler.Enabled() {
		s.initLighting(s.screenWidth, s.screenHeight)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.lightHandler.AddLight(l)
}

func (s *scene) Resize(width, height int) {
	s.mu.Lock()
	if s.screenWidth == width && s.screenHeight == height {
		s.mu.Unlock()
		return
	}
	s.screenWidth = width
	s.screenHeight = height
	s.r.Resize(width, height)

	if height > 0 {
		s.cam.SetAspect(float32(width) / float32(height))
	}
	if s.lightHandler.Enabled() {
		s.lightHandler.Resize(width, height)
	}
	if s.lightHandler.GBufferHandler().Enabled() {
		s.lightHandler.GBufferHandler().Resize(width, height)
	}
	if s.lightHandler.SSAOHandler().Enabled() {
		s.lightHandler.SSAOHandler().Resize(width, height)
	}
	if s.lightHandler.CompositionHandler().Enabled() {
		s.lightHandler.CompositionHandler().Resize(width, height)
	}
	if s.lightHandler.SSRHandler().Enabled() {
		s.lightHandler.SSRHandler().Resize(width, height)
	}
	needTileCullReinit := s.lightHandler.TileCountX()*s.lightHandler.TileCountY() > s.tileBufferCapacity
	s.mu.Unlock()

	// Recreate resolution-dependent GPU resources (textures + bind groups).
	s.resizePostProcessing(width, height)

	// If the new tile count exceeds the allocated tile buffer capacity,
	// re-init the Forward+ cull resources with larger buffers.
	if needTileCullReinit {
		cullComputeShader := shader.NewShader("_light_cull_compute", shader.ShaderTypeCompute,
			"engine/light/assets/light-cull-compute.wgsl", shader.WithInjections(s.injections))
		litFragShader := shader.NewShader("_lit_frag_csm", shader.ShaderTypeFragment,
			"engine/light/assets/lit-frag-csm.wgsl", shader.WithInjections(s.injections))
		s.initLightCullResources(cullComputeShader, litFragShader, width, height)
	}
}

func (s *scene) RemoveLight(l light.Light) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lightHandler.RemoveLight(l)
	delete(s.lightPrevSlotMap, l)
}

func (s *scene) DetachLight(obj game_object.GameObject) {
	s.mu.Lock()
	defer s.mu.Unlock()
	l := obj.Light()
	if l == nil {
		return
	}
	s.lightHandler.RemoveLight(l)
	for i, o := range s.lightObjects {
		if o == obj {
			s.lightObjects = append(s.lightObjects[:i], s.lightObjects[i+1:]...)
			break
		}
	}
}

func (s *scene) Lights() []light.Light {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lightHandler.Lights()
}

func (s *scene) PrepareGBuffer() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.lightHandler.GBufferHandler() == nil || !s.lightHandler.GBufferHandler().Enabled() || s.r == nil {
		return
	}

	if err := s.r.BeginGBufferFrame(); err != nil {
		return
	}
	s.r.BeginGBufferPass(
		s.lightHandler.GBufferHandler().NormalTextureView(),
		s.lightHandler.GBufferHandler().AlbedoTextureView(),
		s.lightHandler.GBufferHandler().DepthTextureView(),
	)

	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}

			mdl := a.Model()
			if mdl == nil {
				continue
			}
			meshProvider := mdl.MeshProvider()
			if meshProvider == nil {
				continue
			}

			// Select the appropriate G-Buffer pipeline based on whether the model is skinned.
			pipeKey := s.lightHandler.GBufferHandler().PipelineKey("static")
			if mdl.Skinned() {
				pipeKey = s.lightHandler.GBufferHandler().PipelineKey("skinned")
			}
			if pipeKey == "" {
				continue
			}

			// Build bind groups for the G-Buffer pass:
			//   group(0) = camera BGP
			//   group(1) = output BGP (instance/bone matrices from compute shader)
			//   group(2) = material BGP
			cameraBGP := s.cam.BindGroupProvider()
			if cameraBGP == nil {
				continue
			}

			mats := mdl.RenderMaterials()
			if len(mats) == 0 {
				continue
			}
			matBGP := mats[0].BindGroupProvider()
			if matBGP == nil {
				continue
			}

			gbufferBindGroups := []bind_group_provider.BindGroupProvider{
				cameraBGP,
				a.OutputBindGroupProvider(),
				matBGP,
			}

			// Use indirect draw when GPU frustum culling is active.
			if a.CullingEnabled() {
				if key := mdl.ComputePipelineKey(); key != "" {
					if rp := s.r.Pipeline(key); rp != nil {
						if cs := rp.Shader(shader.ShaderTypeCompute); cs != nil {
							indirectBinding := 0
							for _, decl := range cs.Declarations() {
								if decl.Type == shader.AnnotationTypeBindingGroup && decl.Binding != nil {
									typeArg := string(decl.Args[2])
									if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
										typeArg = strings.TrimSuffix(stripped, ">")
									}
									if shader.AnnotationArg(typeArg) == shader.AnnotationArgIndirectArgs {
										indirectBinding = *decl.Binding
										break
									}
								}
							}
							if indBuf := a.IndirectBuffer(indirectBinding); indBuf != nil {
								_ = s.r.GBufferDrawCallIndirect(pipeKey, meshProvider, indBuf, gbufferBindGroups)
								continue
							}
						}
					}
				}
			}

			_ = s.r.GBufferDrawCall(pipeKey, meshProvider, uint32(a.InstanceCount()), gbufferBindGroups)
		}
	}

	s.r.EndGBufferPass()
	s.r.EndGBufferFrame()
}

func (s *scene) PrepareSSAO() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.lightHandler.SSAOHandler() == nil || !s.lightHandler.SSAOHandler().Enabled() || s.r == nil || s.cam == nil {
		return
	}

	w := s.lightHandler.SSAOHandler().ScreenWidth()
	h := s.lightHandler.SSAOHandler().ScreenHeight()

	// Compute the G-Buffer coordinate scale: 2.0 when running at half-resolution,
	// 1.0 at full resolution. The SSAO compute and blur shaders multiply their
	// texture coordinates by this value when reading from full-res G-Buffer textures.
	var gbufferScale float32 = 1.0
	var gbufferScaleI int32 = 1
	if s.lightHandler.SSAOHandler().HalfResolution() {
		gbufferScale = 2.0
		gbufferScaleI = 2
	}

	ssaoW := w
	ssaoH := h
	if s.lightHandler.SSAOHandler().HalfResolution() {
		ssaoW = max(w/2, 1)
		ssaoH = max(h/2, 1)
	}

	// Compute the inverse view-projection matrix for depth-to-world reconstruction.
	vp := s.cam.ViewProjectionMatrix()
	var invVP [16]float32
	common.Invert4(invVP[:], vp[:])

	// Get camera position from the controller.
	var camPos [3]float32
	if ctrl := s.cam.Controller(); ctrl != nil {
		camPos[0], camPos[1], camPos[2] = ctrl.Position()
	}

	// Compute view-adaptive world-space radius from screen-space pixel radius.
	screenRadius := s.lightHandler.SSAOHandler().ScreenRadius()
	camDist := float32(1.0)
	if ctrl := s.cam.Controller(); ctrl != nil {
		if d := ctrl.Radius(); d > 0 {
			camDist = d
		}
	}
	fov := s.cam.Fov()
	worldRadius := screenRadius * (2.0 * camDist * float32(math.Tan(float64(fov/2.0)))) / float32(ssaoH)

	// Build and write SSAO uniform parameters.
	ssaoParams := light.GPUSSAOParams{
		Projection:     s.cam.ViewProjectionMatrix(),
		InvViewProj:    invVP,
		Radius:         worldRadius,
		Bias:           s.lightHandler.SSAOHandler().Bias(),
		Power:          s.lightHandler.SSAOHandler().Power(),
		SampleCount:    uint32(s.lightHandler.SSAOHandler().SampleCount()),
		ScreenWidth:    float32(ssaoW),
		ScreenHeight:   float32(ssaoH),
		GBufferScale:   gbufferScale,
		CameraPosition: camPos,
	}

	ssaoBGP := s.lightHandler.SSAOHandler().Bgp("ssao_compute")
	blurHBGP := s.lightHandler.SSAOHandler().Bgp("ssao_blur_h")
	blurVBGP := s.lightHandler.SSAOHandler().Bgp("ssao_blur_v")

	// Write SSAO params to the compute BGP uniform buffer.
	hParams := light.GPUBlurParams{
		Direction:    [2]int32{1, 0},
		Radius:       int32(s.lightHandler.SSAOHandler().BlurRadius()),
		GBufferScale: gbufferScaleI,
	}
	vParams := light.GPUBlurParams{
		Direction:    [2]int32{0, 1},
		Radius:       int32(s.lightHandler.SSAOHandler().BlurRadius()),
		GBufferScale: gbufferScaleI,
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: ssaoBGP, Binding: 4, Offset: 0, Data: ssaoParams.Marshal()},
		{Provider: blurHBGP, Binding: 2, Offset: 0, Data: hParams.Marshal()},
		{Provider: blurVBGP, Binding: 2, Offset: 0, Data: vParams.Marshal()},
	})

	// Dispatch SSAO compute + bilateral blur.
	ssaoWGX, ssaoWGY := s.computeWorkgroupSize2D(s.lightHandler.SSAOHandler().PipelineKey("ssao_compute"), 16, 16)
	workGroupsX := (uint32(ssaoW) + ssaoWGX - 1) / ssaoWGX
	workGroupsY := (uint32(ssaoH) + ssaoWGY - 1) / ssaoWGY
	wg := [3]uint32{workGroupsX, workGroupsY, 1}

	// Each dispatch is a separate compute pass to ensure automatic GPU barriers
	// between the SSAO compute output and the blur reads (READ-AFTER-WRITE).
	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{PipelineKey: s.lightHandler.SSAOHandler().PipelineKey("ssao_compute"), Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ssaoBGP}}, WorkGroupCount: wg},
	})
	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{PipelineKey: s.lightHandler.SSAOHandler().PipelineKey("ssao_blur"), Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: blurHBGP}}, WorkGroupCount: wg},
	})
	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{PipelineKey: s.lightHandler.SSAOHandler().PipelineKey("ssao_blur"), Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: blurVBGP}}, WorkGroupCount: wg},
	})
}

func (s *scene) PrepareContactShadows() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	csHandler := s.lightHandler.ContactShadowHandler()
	if csHandler == nil || !csHandler.Enabled() || s.r == nil || s.cam == nil {
		return
	}
	if csHandler.TextureView() == nil {
		return
	}

	// Find directional light direction.
	var lightDir [3]float32
	found := false
	for _, l := range s.lightHandler.Lights() {
		if l.Enabled() && l.CastsShadows() && l.Type() == light.LightTypeDirectional {
			lightDir = l.Direction()
			found = true
			break
		}
	}
	if !found {
		return
	}

	// Compute inverse view-projection.
	vp := s.cam.ViewProjectionMatrix()
	var invVP [16]float32
	common.Invert4(invVP[:], vp[:])

	// Camera position.
	var camPos [3]float32
	if ctrl := s.cam.Controller(); ctrl != nil {
		camPos[0], camPos[1], camPos[2] = ctrl.Position()
	}

	csW := float32(s.screenWidth)
	csH := float32(s.screenHeight)

	params := light.GPUContactShadowParams{
		ViewProj:       vp,
		InvViewProj:    invVP,
		LightDirection: lightDir,
		StepCount:      uint32(csHandler.StepCount()),
		MaxDistance:    csHandler.MaxDistance(),
		Thickness:      csHandler.Thickness(),
		ScreenWidth:    csW,
		ScreenHeight:   csH,
		CameraPosition: camPos,
	}

	csBGP := csHandler.Bgp("contact_shadow_compute")
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: csBGP, Binding: 2, Offset: 0, Data: params.Marshal()},
	})

	csWGX, csWGY := s.computeWorkgroupSize2D(csHandler.PipelineKey("contact_shadow_compute"), 16, 16)
	workGroupsX := (uint32(s.screenWidth) + csWGX - 1) / csWGX
	workGroupsY := (uint32(s.screenHeight) + csWGY - 1) / csWGY

	if err := s.r.BeginComputeFrame(); err != nil {
		return
	}
	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{PipelineKey: csHandler.PipelineKey("contact_shadow_compute"), Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: csBGP}}, WorkGroupCount: [3]uint32{workGroupsX, workGroupsY, 1}},
	})
	s.r.EndComputeFrame()
}

func (s *scene) PrepareSSR() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ssrHandler := s.lightHandler.SSRHandler()
	compHandler := s.lightHandler.CompositionHandler()
	if ssrHandler == nil || !ssrHandler.Enabled() || compHandler == nil || !compHandler.Enabled() || s.r == nil || s.cam == nil {
		return
	}

	w := ssrHandler.ScreenWidth()
	h := ssrHandler.ScreenHeight()

	// Determine which slot's Hi-Z BGPs to use for this frame.
	slot := s.r.CurrentFrameSlot()
	hizInitKey := "hiz_init"
	hizInitMaxKey := "hiz_init_max"
	downKeySuffix := ""
	if slot == 1 {
		hizInitKey = "hiz_init_1"
		hizInitMaxKey = "hiz_init_max_1"
		downKeySuffix = "_1"
	}

	// Build and write SSR uniform parameters.
	ssrParams := light.GPUSSRParams{
		Projection:      s.cam.ProjectionMatrix(),
		InvProjection:   s.cam.InverseProjectionMatrix(),
		View:            s.cam.ViewMatrix(),
		MaxDistance:     ssrHandler.MaxDistance(),
		Thickness:       ssrHandler.Thickness(),
		Stride:          ssrHandler.Stride(),
		MaxSteps:        uint32(ssrHandler.MaxSteps()),
		ScreenWidth:     float32(w),
		ScreenHeight:    float32(h),
		RoughnessCutoff: ssrHandler.RoughnessCutoff(),
		HiZMipCount:     uint32(ssrHandler.HiZMipCount()),
	}

	ssrBGP := ssrHandler.Bgp("ssr_compute")
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: ssrBGP, Binding: 0, Offset: 0, Data: ssrParams.Marshal()},
	})

	if err := s.r.BeginComputeFrame(); err != nil {
		return
	}

	// --- Hi-Z depth pyramid generation ---
	mipCount := ssrHandler.HiZMipCount()

	// Pass 0: copy GBuffer depth → Hi-Z mip 0 (full resolution).
	hizInitBGP := ssrHandler.Bgp(hizInitKey)
	hizIWGX, hizIWGY := s.computeWorkgroupSize2D(ssrHandler.PipelineKey("hiz_init"), 8, 8)
	initWGX := (uint32(w) + hizIWGX - 1) / hizIWGX
	initWGY := (uint32(h) + hizIWGY - 1) / hizIWGY
	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{
			PipelineKey:    ssrHandler.PipelineKey("hiz_init"),
			Providers:      []renderer.ComputeGroupProvider{{Group: 0, Provider: hizInitBGP}},
			WorkGroupCount: [3]uint32{initWGX, initWGY, 1},
		},
	})

	// Passes 1..N-1: min-downsample each mip level from the previous.
	// Each dispatch is a separate pass to provide implicit barriers for the READ-AFTER-WRITE chain.
	mipW := w
	mipH := h
	hizDWGX, hizDWGY := s.computeWorkgroupSize2D(ssrHandler.PipelineKey("hiz_downsample"), 8, 8)
	for i := 1; i < mipCount; i++ {
		mipW = max(mipW/2, 1)
		mipH = max(mipH/2, 1)
		bgp := ssrHandler.Bgp(fmt.Sprintf("hiz_down_%d%s", i, downKeySuffix))
		wgX := (uint32(mipW) + hizDWGX - 1) / hizDWGX
		wgY := (uint32(mipH) + hizDWGY - 1) / hizDWGY
		s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
			{
				PipelineKey:    ssrHandler.PipelineKey("hiz_downsample"),
				Providers:      []renderer.ComputeGroupProvider{{Group: 0, Provider: bgp}},
				WorkGroupCount: [3]uint32{wgX, wgY, 1},
			},
		})
	}

	// ── MAX Hi-Z pyramid: mip 0 init + downsample chain ──────────────────────────
	// Mip 0: copy GBuffer depth → MAX mip 0 (same pipeline as min init, different BGP).
	hizInitMaxBGP := ssrHandler.Bgp(hizInitMaxKey)
	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{{
		PipelineKey:    ssrHandler.PipelineKey("hiz_init"),
		Providers:      []renderer.ComputeGroupProvider{{Group: 0, Provider: hizInitMaxBGP}},
		WorkGroupCount: [3]uint32{initWGX, initWGY, 1},
	}})

	// Mips 1..N-1: max-downsample.
	mipWMax := w
	mipHMax := h
	hizDMaxWGX, hizDMaxWGY := s.computeWorkgroupSize2D(ssrHandler.PipelineKey("hiz_downsample_max"), 8, 8)
	for i := 1; i < mipCount; i++ {
		mipWMax = max(mipWMax/2, 1)
		mipHMax = max(mipHMax/2, 1)
		bgpMax := ssrHandler.Bgp(fmt.Sprintf("hiz_down_max_%d%s", i, downKeySuffix))
		wgMaxX := (uint32(mipWMax) + hizDMaxWGX - 1) / hizDMaxWGX
		wgMaxY := (uint32(mipHMax) + hizDMaxWGY - 1) / hizDMaxWGY
		s.r.DispatchComputeBatch([]renderer.ComputeDispatch{{
			PipelineKey:    ssrHandler.PipelineKey("hiz_downsample_max"),
			Providers:      []renderer.ComputeGroupProvider{{Group: 0, Provider: bgpMax}},
			WorkGroupCount: [3]uint32{wgMaxX, wgMaxY, 1},
		}})
	}

	// --- SSR compute dispatch at half-resolution ---
	halfW := w / 2
	halfH := h / 2
	if halfW <= 0 {
		halfW = 1
	}
	if halfH <= 0 {
		halfH = 1
	}
	ssrWGX, ssrWGY := s.computeWorkgroupSize2D(ssrHandler.PipelineKey("ssr_compute"), 8, 8)
	workGroupsX := (uint32(halfW) + ssrWGX - 1) / ssrWGX
	workGroupsY := (uint32(halfH) + ssrWGY - 1) / ssrWGY
	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{
			PipelineKey:    ssrHandler.PipelineKey("ssr_compute"),
			Providers:      []renderer.ComputeGroupProvider{{Group: 0, Provider: ssrBGP}},
			WorkGroupCount: [3]uint32{workGroupsX, workGroupsY, 1},
		},
	})

	s.r.EndComputeFrame()
}

func (s *scene) PrepareLuminance(dt float32) {
	s.prepareLuminance(dt)
}

func (s *scene) PrepareBloom() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.lightHandler == nil || s.r == nil {
		return
	}
	ch := s.lightHandler.CompositionHandler()
	if ch == nil || !ch.Enabled() || !ch.BloomEnabled() || ch.BloomMipCount() <= 0 {
		return
	}

	mipCount := ch.BloomMipCount()

	// Write bloom params for each downsample BGP.
	writes := make([]bind_group_provider.BufferWrite, mipCount)
	for i := 0; i < mipCount; i++ {
		params := light.GPUBloomParams{}
		if i == 0 {
			params.Threshold = ch.BloomThreshold()
		}
		writes[i] = bind_group_provider.BufferWrite{
			Provider: ch.Bgp(fmt.Sprintf("bloom_down_%d", i)),
			Binding:  3,
			Offset:   0,
			Data:     params.Marshal(),
		}
	}
	s.r.WriteBuffers(writes)

	if err := s.r.BeginComputeFrame(); err != nil {
		return
	}

	// Pre-compute mip dimensions.
	halfW := ch.ScreenWidth() / 2
	halfH := ch.ScreenHeight() / 2
	mipDims := make([][2]int, mipCount)
	w, h := halfW, halfH
	for i := 0; i < mipCount; i++ {
		mipDims[i] = [2]int{w, h}
		w = max(w/2, 1)
		h = max(h/2, 1)
	}

	// Downsample chain.
	downKey := ch.PipelineKey("bloom_downsample")
	for i := 0; i < mipCount; i++ {
		wgX := (uint32(mipDims[i][0]) + 7) / 8
		wgY := (uint32(mipDims[i][1]) + 7) / 8
		s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
			{PipelineKey: downKey, Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ch.Bgp(fmt.Sprintf("bloom_down_%d", i))}}, WorkGroupCount: [3]uint32{wgX, wgY, 1}},
		})
	}

	// Upsample chain: from lowest mip up to mip 0.
	upKey := ch.PipelineKey("bloom_upsample")
	for i := mipCount - 2; i >= 0; i-- {
		wgX := (uint32(mipDims[i][0]) + 7) / 8
		wgY := (uint32(mipDims[i][1]) + 7) / 8
		s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
			{PipelineKey: upKey, Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ch.Bgp(fmt.Sprintf("bloom_up_%d", i))}}, WorkGroupCount: [3]uint32{wgX, wgY, 1}},
		})
	}

	s.r.EndComputeFrame()
}

func (s *scene) PrepareComposition() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	compHandler := s.lightHandler.CompositionHandler()
	if compHandler == nil || !compHandler.Enabled() || s.r == nil {
		return
	}

	// Write composition params uniform.
	compParams := light.GPUCompositionParams{
		Exposure: compHandler.Exposure(),
	}
	if compHandler.ToneMappingEnabled() {
		compParams.ToneMappingEnabled = 1
	}
	if compHandler.AutoExposureEnabled() {
		compParams.AutoExposureEnabled = 1
	}
	if compHandler.BloomEnabled() {
		compParams.BloomEnabled = 1
		compParams.BloomIntensity = compHandler.BloomIntensity()
	}
	compBGP := compHandler.Bgp("composition")
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: compBGP, Binding: 4, Offset: 0, Data: compParams.Marshal()},
	})

	s.r.BeginCompositionPass()
	_ = s.r.CompositionDrawCall(compHandler.PipelineKey("composition"), []bind_group_provider.BindGroupProvider{compBGP})
	s.r.EndCompositionPass()
	s.r.EndCompositionFrame()
}

func (s *scene) AcquireCompositionFrame() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.r == nil || !s.active {
		return nil
	}
	return s.r.BeginCompositionFrame()
}

func (s *scene) BeginHDRFrame() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ch := s.lightHandler.CompositionHandler()
	if ch == nil || !ch.Enabled() || s.r == nil {
		return fmt.Errorf("scene %q: composition not initialized", s.name)
	}

	sampleCount := s.r.SampleCount()
	if sampleCount > 1 && ch.MSAATextureView() != nil {
		// MSAA active: render to multi-sampled texture, resolve into HDR.
		return s.r.BeginHDRFrame(ch.MSAATextureView(), ch.HDRTextureView(), ch.DepthTextureView(), sampleCount)
	}
	// No MSAA: render directly to HDR texture.
	return s.r.BeginHDRFrame(ch.HDRTextureView(), nil, ch.DepthTextureView(), 1)
}

func (s *scene) PrepareLights() {
	if !s.lightHandler.Enabled() {
		return
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	lightsBGP := s.lightHandler.Bgp("lights")
	rawLights := s.lightHandler.Lights()
	lightsToMarshal := rawLights
	if len(rawLights) > int(s.lightHandler.MaxGPULights()) {
		const lightSortEpsilon = float32(1e-4)
		var eyeX, eyeY, eyeZ float32
		if ctrl := s.cam.Controller(); ctrl != nil {
			eyeX, eyeY, eyeZ = ctrl.Position()
		}
		importanceOf := func(l light.Light) float32 {
			if l.Type() == light.LightTypeDirectional {
				return math.MaxFloat32
			}
			pos := l.Position()
			dx := pos[0] - eyeX
			dy := pos[1] - eyeY
			dz := pos[2] - eyeZ
			dist2 := dx*dx + dy*dy + dz*dz
			if dist2 < lightSortEpsilon {
				dist2 = lightSortEpsilon
			}
			return l.Intensity() * l.Range() / dist2
		}
		sorted := make([]light.Light, len(rawLights))
		copy(sorted, rawLights)
		slices.SortFunc(sorted, func(a, b light.Light) int {
			impA := importanceOf(a)
			impB := importanceOf(b)
			if impA > impB {
				return -1
			}
			if impA < impB {
				return 1
			}
			return 0
		})
		lightsToMarshal = sorted
	}
	lightData := s.lightHandler.MarshalLightBuffer(lightsToMarshal, s.lightShadowMap)
	writes := []bind_group_provider.BufferWrite{
		{
			Provider: lightsBGP,
			Binding:  0, // light_header uniform
			Offset:   0,
			Data:     lightData[:16], // GPULightHeader is 16 bytes
		},
	}
	if len(lightData) > 16 {
		writes = append(writes, bind_group_provider.BufferWrite{
			Provider: lightsBGP,
			Binding:  1, // lights storage array
			Offset:   0,
			Data:     lightData[16:],
		})
	}
	s.r.WriteBuffers(writes)
}

// shadowTransformEntry caches a single instance's position and scale for
// use across all shadow pass iterations within a frame.
type shadowTransformEntry struct {
	pos   [3]float32
	scale [3]float32
}

// worldAABB transforms a model-space AABB to world space using instance
// position and scale. Per-axis min/max swap handles negative scale correctly.
func worldAABB(modelMin, modelMax [3]float32, pos, scale [3]float32) (wMin, wMax [3]float32) {
	for i := range 3 {
		lo := scale[i] * modelMin[i]
		hi := scale[i] * modelMax[i]
		if lo > hi {
			lo, hi = hi, lo
		}
		wMin[i] = pos[i] + lo
		wMax[i] = pos[i] + hi
	}
	return
}

func (s *scene) PrepareShadows() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.lightHandler.Enabled() || s.r == nil {
		return
	}

	sh := s.lightHandler.ShadowHandler()

	// Find the first enabled, shadow-casting directional light.
	var shadowLight light.Light
	for _, l := range s.lightHandler.Lights() {
		if l.Enabled() && l.CastsShadows() && l.Type() == light.LightTypeDirectional {
			shadowLight = l
			break
		}
	}

	writes := []bind_group_provider.BufferWrite{}

	// ── CSM cascade data (only when a directional shadow light exists) ──
	var cascadeCount int
	var res int
	var csmData *light.GPUCSMData
	if shadowLight != nil {
		res = sh.ShadowMapResolution()
		cascadeCount = sh.CascadeCount()
		innerRadius := sh.ShadowInnerRadius()

		csmData = &light.GPUCSMData{
			TexelSize:         [2]float32{1.0 / float32(res), 1.0 / float32(res)},
			Bias:              shadowLight.ShadowBias(),
			InnerRadius:       innerRadius,
			PCFRadius:         sh.PCFRadius(),
			ShadowMaxDistance: sh.ShadowFar(),
		}

		camNear := s.cam.Near()
		camFar := sh.ShadowFar()
		var cameraPosition [3]float32
		if ctrl := s.cam.Controller(); ctrl != nil {
			cameraPosition[0], cameraPosition[1], cameraPosition[2] = ctrl.Position()
		}
		csmData.ComputeCascades(
			shadowLight.Direction(),
			camNear, camFar,
			s.cam.Fov(),
			s.cam.Aspect(),
			s.cam.ViewMatrix(),
			cameraPosition,
			innerRadius,
			sh.ShadowNormalBiasScale(),
			res,
		)

		// Write CSM data to the csm_shadow_lit BGP uniform buffer (binding 2).
		csmShadowLitBGP := sh.Bgp("csm_shadow_lit")
		if csmShadowLitBGP.Buffer(2) != nil {
			writes = append(writes, bind_group_provider.BufferWrite{
				Provider: csmShadowLitBGP,
				Binding:  2,
				Offset:   0,
				Data:     csmData.Marshal(),
			})
		}

		// Write per-cascade ShadowUniform to "csm_data_i" BGPs.
		for i := 0; i < cascadeCount; i++ {
			c := csmData.Cascades[i]
			cascadeUniform := light.GPUShadowUniform{LightVP: c.LightVP}
			bgpKey := fmt.Sprintf("csm_data_%d", i)
			cascadeBGP := sh.Bgp(bgpKey)
			writes = append(writes, bind_group_provider.BufferWrite{
				Provider: cascadeBGP,
				Binding:  0,
				Offset:   0,
				Data:     cascadeUniform.Marshal(),
			})
		}
	}

	// ── Spot/point shadow VP computation (always) ──
	s.lightShadowEntries = s.lightShadowEntries[:0]
	s.lightShadowMap = make(map[light.Light]uint32, sh.LightShadowAtlasSlots())
	slotIdx := 0
	maxSlots := sh.LightShadowAtlasSlots()
	tileSize := sh.LightShadowTileSize()

	if s.lightPrevSlotMap == nil {
		s.lightPrevSlotMap = make(map[light.Light]uint32)
	}

	// Extract camera frustum for per-light visibility rejection.
	// Lights whose entire influence sphere lies outside the camera frustum are skipped entirely —
	// no atlas slot, no shadow depth pass, no lightShadowMap entry.
	var cameraFrustum common.Frustum
	hasCameraFrustum := !s.cullingDisabled && s.cam != nil
	if hasCameraFrustum {
		vp := s.cam.ViewProjectionMatrix()
		cameraFrustum = common.ExtractFrustumFromMatrix(vp[:])
	}

	spotFrustums := make(map[light.Light]common.Frustum)

	for _, l := range s.lightHandler.Lights() {
		if slotIdx >= maxSlots {
			break
		}
		if !l.Enabled() || !l.CastsShadows() || l.Type() != light.LightTypeSpot {
			continue
		}

		pos := l.Position()
		dir := l.Direction()
		rng := l.Range()
		outerCos := l.OuterCone()

		halfAngle := float32(math.Acos(float64(outerCos)))
		fovY := halfAngle * 2.0
		near := float32(math.Max(1.0, float64(rng)*0.005))
		far := rng

		var view, proj, vp [16]float32
		target := [3]float32{pos[0] + dir[0], pos[1] + dir[1], pos[2] + dir[2]}

		up := [3]float32{0, 1, 0}
		absDy := dir[1]
		if absDy < 0 {
			absDy = -absDy
		}
		if absDy > 0.99 {
			up = [3]float32{1, 0, 0}
		}

		common.LookAt(view[:], pos[0], pos[1], pos[2], target[0], target[1], target[2], up[0], up[1], up[2])
		common.Perspective(proj[:], fovY, 1.0, near, far)
		common.Mul4(vp[:], proj[:], view[:])
		spotFrustums[l] = common.ExtractFrustumFromMatrix(vp[:])

		cols := sh.LightShadowAtlasCols()
		rows := maxSlots / cols
		col := slotIdx % cols
		row := slotIdx / cols
		atlasRect := [4]float32{
			float32(col) / float32(cols),
			float32(row) / float32(rows),
			1.0 / float32(cols),
			1.0 / float32(rows),
		}

		entry := light.GPULightShadowEntry{
			LightVP:    vp,
			AtlasRect:  atlasRect,
			Bias:       l.ShadowBias(),
			Near:       near,
			Far:        far,
			ShadowType: light.ShadowTypeSpot,
		}

		s.lightShadowMap[l] = uint32(slotIdx)
		s.lightShadowEntries = append(s.lightShadowEntries, entry)
		newSlot := uint32(slotIdx)
		if prev, ok := s.lightPrevSlotMap[l]; ok && prev != newSlot {
			sh.ForceMarkDirty(l)
		}
		s.lightPrevSlotMap[l] = newSlot

		spotUniform := light.GPUShadowUniform{LightVP: vp}
		bgpKey := fmt.Sprintf("spot_shadow_%d", slotIdx)
		spotBGP := sh.Bgp(bgpKey)
		writes = append(writes, bind_group_provider.BufferWrite{
			Provider: spotBGP,
			Binding:  0,
			Offset:   0,
			Data:     spotUniform.Marshal(),
		})

		slotIdx++
	}

	// Determine whether any enabled shadow-casting point light exists so the
	// bail check below does not prematurely skip the unified point-light loop.
	hasPointShadows := false
	for _, l := range s.lightHandler.Lights() {
		if l.Enabled() && l.CastsShadows() && l.Type() == light.LightTypePoint {
			hasPointShadows = true
			break
		}
	}

	// Bail if no shadow work at all.
	if shadowLight == nil && len(s.lightShadowEntries) == 0 && !hasPointShadows {
		return
	}

	s.r.WriteBuffers(writes)

	for _, anims := range s.animatorPool {
		for _, a := range anims {
			if a.InstanceCount() == 0 {
				continue
			}
			mdl := a.Model()
			if mdl == nil || !mdl.CastsShadows() {
				continue
			}
			if buf, ok := s.shadowIndirectBuffers[a]; ok && buf != nil {
				args := animator.GPUIndirectArgs{
					IndexCount:    uint32(mdl.IndexCount()),
					InstanceCount: a.InstanceCount(),
					FirstIndex:    0,
					BaseVertex:    0,
					FirstInstance: 0,
				}
				s.r.WriteRawBuffer(buf, 0, args.Marshal())
			}
		}
	}

	// Build a per-frame transform cache for all shadow-casting animators.
	// One lock acquisition per instance at cache-build time; zero locks during face iteration.
	shadowTransforms := make(map[animator.Animator][]shadowTransformEntry)
	for _, anims := range s.animatorPool {
		for _, a := range anims {
			count := a.InstanceCount()
			if count == 0 {
				continue
			}
			mdl := a.Model()
			if mdl == nil || !mdl.CastsShadows() {
				continue
			}
			entries := make([]shadowTransformEntry, count)
			for idx := uint32(0); idx < count; idx++ {
				p, sc := a.InstanceTransform(idx)
				entries[idx] = shadowTransformEntry{pos: p, scale: sc}
			}
			shadowTransforms[a] = entries
		}
	}

	atlasCleared := false
	if err := s.r.BeginShadowFrame(); err != nil {
		return
	}

	// CSM cascade depth passes.
	if shadowLight != nil {
		for i := 0; i < cascadeCount; i++ {
			bgpKey := fmt.Sprintf("csm_data_%d", i)
			cascadeBGP := sh.Bgp(bgpKey)
			x := uint32(i * res)
			s.r.BeginShadowDepthPass(
				sh.CSMAtlasTextureView(),
				x, 0, uint32(res), uint32(res),
				i == 0,
			)

			cascadeFrustum := common.ExtractFrustumFromMatrix(csmData.Cascades[i].LightVP[:])

			for _, anim := range s.animatorPool {
				for _, a := range anim {
					if a.InstanceCount() == 0 {
						continue
					}
					mdl := a.Model()
					if mdl == nil || !mdl.CastsShadows() {
						continue
					}
					meshProvider := mdl.MeshProvider()
					if meshProvider == nil {
						continue
					}

					cullMode := mdl.ShadowCullMode()
					pipeKey := s.shadowPipelineKey(mdl.Skinned(), cullMode)
					if pipeKey == "" {
						continue
					}

					mdlMin, mdlMax := mdl.BoundingMin(), mdl.BoundingMax()
					visible := false
					for _, entry := range shadowTransforms[a] {
						wMin, wMax := worldAABB(mdlMin, mdlMax, entry.pos, entry.scale)
						if cascadeFrustum.IntersectAABB(wMin, wMax) {
							visible = true
							break
						}
					}
					if !visible {
						continue
					}

					shadowBindGroups := []bind_group_provider.BindGroupProvider{
						cascadeBGP,
						a.OutputBindGroupProvider(),
					}

					if buf, ok := s.shadowIndirectBuffers[a]; ok && buf != nil {
						_ = s.r.ShadowDrawCallIndirect(pipeKey, meshProvider, buf, shadowBindGroups)
					}
				}
			}

			s.r.EndShadowPass()
		}
	}

	// Spot and point shadow depth passes — only re-render dirty lights.
	// Clean lights retain their cached atlas tile content from prior frames.
	for _, l := range s.lightHandler.Lights() {
		if !l.Enabled() || !l.CastsShadows() || l.Type() != light.LightTypeSpot {
			continue
		}
		slotI, ok := s.lightShadowMap[l]
		if !ok {
			continue
		}

		i := int(slotI)
		bgpKey := fmt.Sprintf("spot_shadow_%d", i)
		spotBGP := sh.Bgp(bgpKey)
		cols := sh.LightShadowAtlasCols()
		col := uint32(i % cols)
		row := uint32(i / cols)
		x := col * uint32(tileSize)
		y := row * uint32(tileSize)
		s.r.BeginShadowDepthPass(
			sh.LightShadowAtlasView(),
			x, y, uint32(tileSize), uint32(tileSize),
			!atlasCleared,
		)
		atlasCleared = true

		spotFrustum := spotFrustums[l]
		lPos := l.Position()
		lRange := l.Range()
		for _, anim := range s.animatorPool {
			for _, a := range anim {
				if a.InstanceCount() == 0 {
					continue
				}
				mdl := a.Model()
				if mdl == nil || !mdl.CastsShadows() {
					continue
				}
				meshProvider := mdl.MeshProvider()
				if meshProvider == nil {
					continue
				}

				cullMode := mdl.ShadowCullMode()
				pipeKey := s.shadowPipelineKey(mdl.Skinned(), cullMode)
				if pipeKey == "" {
					continue
				}

				// Sphere rejection + frustum cull using pre-built transform cache.
				mdlMin, mdlMax := mdl.BoundingMin(), mdl.BoundingMax()
				boundR := mdl.BoundingRadius()
				visible := false
				for _, entry := range shadowTransforms[a] {
					iPos := entry.pos
					iScale := entry.scale
					dx := iPos[0] - lPos[0]
					dy := iPos[1] - lPos[1]
					dz := iPos[2] - lPos[2]
					dist := float32(math.Sqrt(float64(dx*dx + dy*dy + dz*dz)))
					maxS := iScale[0]
					if iScale[1] > maxS {
						maxS = iScale[1]
					}
					if iScale[2] > maxS {
						maxS = iScale[2]
					}
					if dist > lRange+boundR*maxS {
						continue
					}
					wMin, wMax := worldAABB(mdlMin, mdlMax, iPos, iScale)
					if spotFrustum.IntersectAABB(wMin, wMax) {
						visible = true
						break
					}
				}
				if !visible {
					continue
				}

				shadowBindGroups := []bind_group_provider.BindGroupProvider{
					spotBGP,
					a.OutputBindGroupProvider(),
				}

				if buf, ok := s.shadowIndirectBuffers[a]; ok && buf != nil {
					_ = s.r.ShadowDrawCallIndirect(pipeKey, meshProvider, buf, shadowBindGroups)
				}
			}
		}

		s.r.EndShadowPass()
	}

	// Unified point light loop: VP computation, buffer writes, dirty check, and render.
	// All six cube-face passes for each light are handled in a single iteration,
	// eliminating the cross-loop dirty-flag dependency of the prior two-loop design.
	for _, l := range s.lightHandler.Lights() {
		if slotIdx+5 >= maxSlots {
			break
		}
		if !l.Enabled() || !l.CastsShadows() || l.Type() != light.LightTypePoint {
			continue
		}

		pos := l.Position()
		rng := l.Range()
		near := float32(math.Max(1.0, float64(rng)*0.005))
		far := rng

		var proj [16]float32
		common.Perspective(proj[:], math.Pi/2.0, 1.0, near, far)

		type cubeFace struct {
			dir [3]float32
			up  [3]float32
		}
		faces := [6]cubeFace{
			{dir: [3]float32{1, 0, 0}, up: [3]float32{0, -1, 0}},
			{dir: [3]float32{-1, 0, 0}, up: [3]float32{0, -1, 0}},
			{dir: [3]float32{0, 1, 0}, up: [3]float32{0, 0, 1}},
			{dir: [3]float32{0, -1, 0}, up: [3]float32{0, 0, -1}},
			{dir: [3]float32{0, 0, 1}, up: [3]float32{0, -1, 0}},
			{dir: [3]float32{0, 0, -1}, up: [3]float32{0, -1, 0}},
		}

		s.lightShadowMap[l] = uint32(len(s.lightShadowEntries))
		newSlot := s.lightShadowMap[l]
		if prev, ok := s.lightPrevSlotMap[l]; ok && prev != newSlot {
			sh.ForceMarkDirty(l)
		}
		s.lightPrevSlotMap[l] = newSlot

		cols := sh.LightShadowAtlasCols()
		rows := maxSlots / cols

		var ptWrites []bind_group_provider.BufferWrite
		var faceFrustums [6]common.Frustum
		for fi := 0; fi < 6; fi++ {
			target := [3]float32{
				pos[0] + faces[fi].dir[0],
				pos[1] + faces[fi].dir[1],
				pos[2] + faces[fi].dir[2],
			}
			var view, vp [16]float32
			common.LookAt(view[:], pos[0], pos[1], pos[2], target[0], target[1], target[2], faces[fi].up[0], faces[fi].up[1], faces[fi].up[2])
			common.Mul4(vp[:], proj[:], view[:])
			faceFrustums[fi] = common.ExtractFrustumFromMatrix(vp[:])

			si := slotIdx + fi
			col := si % cols
			row := si / cols
			atlasRect := [4]float32{
				float32(col) / float32(cols),
				float32(row) / float32(rows),
				1.0 / float32(cols),
				1.0 / float32(rows),
			}

			entry := light.GPULightShadowEntry{
				LightVP:    vp,
				AtlasRect:  atlasRect,
				Bias:       l.ShadowBias(),
				Near:       near,
				Far:        far,
				ShadowType: light.ShadowTypeCubeFace,
			}
			s.lightShadowEntries = append(s.lightShadowEntries, entry)

			bgpKey := fmt.Sprintf("spot_shadow_%d", si)
			bgp := sh.Bgp(bgpKey)
			uniform := light.GPUShadowUniform{LightVP: vp}
			ptWrites = append(ptWrites, bind_group_provider.BufferWrite{
				Provider: bgp,
				Binding:  0,
				Offset:   0,
				Data:     uniform.Marshal(),
			})
		}
		s.r.WriteBuffers(ptWrites)

		lightVisible := !hasCameraFrustum || cameraFrustum.IntersectSphere(pos, l.Range())
		if lightVisible {
			for fi := 0; fi < 6; fi++ {
				faceFrustum := faceFrustums[fi]
				si := slotIdx + fi
				bgpKey := fmt.Sprintf("spot_shadow_%d", si)
				spotBGP := sh.Bgp(bgpKey)
				col := uint32(si % cols)
				row := uint32(si / cols)
				x := col * uint32(tileSize)
				y := row * uint32(tileSize)
				s.r.BeginShadowDepthPass(
					sh.LightShadowAtlasView(),
					x, y, uint32(tileSize), uint32(tileSize),
					!atlasCleared,
				)
				atlasCleared = true

				for _, anim := range s.animatorPool {
					for _, a := range anim {
						if a.InstanceCount() == 0 {
							continue
						}
						mdl := a.Model()
						if mdl == nil || !mdl.CastsShadows() {
							continue
						}
						meshProvider := mdl.MeshProvider()
						if meshProvider == nil {
							continue
						}

						cullMode := mdl.ShadowCullMode()
						pipeKey := s.shadowPipelineKey(mdl.Skinned(), cullMode)
						if pipeKey == "" {
							continue
						}

						// Sphere rejection + frustum cull using pre-built transform cache.
						mdlMin, mdlMax := mdl.BoundingMin(), mdl.BoundingMax()
						boundR := mdl.BoundingRadius()
						visible := false
						for _, entry := range shadowTransforms[a] {
							iPos := entry.pos
							iScale := entry.scale
							dx := iPos[0] - pos[0]
							dy := iPos[1] - pos[1]
							dz := iPos[2] - pos[2]
							dist := float32(math.Sqrt(float64(dx*dx + dy*dy + dz*dz)))
							maxS := iScale[0]
							if iScale[1] > maxS {
								maxS = iScale[1]
							}
							if iScale[2] > maxS {
								maxS = iScale[2]
							}
							if dist > rng+boundR*maxS {
								continue
							}
							wMin, wMax := worldAABB(mdlMin, mdlMax, iPos, iScale)
							if faceFrustum.IntersectAABB(wMin, wMax) {
								visible = true
								break
							}
						}
						if !visible {
							continue
						}

						shadowBindGroups := []bind_group_provider.BindGroupProvider{
							spotBGP,
							a.OutputBindGroupProvider(),
						}

						if buf, ok := s.shadowIndirectBuffers[a]; ok && buf != nil {
							_ = s.r.ShadowDrawCallIndirect(pipeKey, meshProvider, buf, shadowBindGroups)
						}
					}
				}

				s.r.EndShadowPass()
			}
		}
		slotIdx += 6
	}

	// Write the complete shadow entry data (spot + point) to csm_shadow_lit
	// binding 4 now that all entries have been populated by both loops.
	if csmBGP := sh.Bgp("csm_shadow_lit"); csmBGP != nil && csmBGP.Buffer(4) != nil && len(s.lightShadowEntries) > 0 {
		entryData := make([]byte, 0, len(s.lightShadowEntries)*96)
		for _, e := range s.lightShadowEntries {
			entryData = append(entryData, e.Marshal()...)
		}
		s.r.WriteBuffers([]bind_group_provider.BufferWrite{
			{Provider: csmBGP, Binding: 4, Offset: 0, Data: entryData},
		})
	}

	s.r.EndShadowFrame()
}

func (s *scene) PrepareLightCulling() {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.lightHandler.Enabled() || s.r == nil || s.cam == nil {
		return
	}

	// Count enabled lights. Even when zero we must still dispatch the cull
	// shader so that tile counts are zeroed out — otherwise stale tile data
	// from the previous frame causes disabled lights to keep rendering.
	var lightCount uint32
	for _, l := range s.lightHandler.Lights() {
		if l.Enabled() {
			lightCount++
		}
	}

	// Build and write cull uniforms.
	cullBGP := s.lightHandler.Bgp("light_cull")
	uniforms := light.GPULightCullUniforms{
		InvProj:      s.cam.InverseProjectionMatrix(),
		ViewMatrix:   s.cam.ViewMatrix(),
		TileCountX:   uint32(s.lightHandler.TileCountX()),
		TileCountY:   uint32(s.lightHandler.TileCountY()),
		ScreenWidth:  uint32(s.lightHandler.ScreenWidth()),
		ScreenHeight: uint32(s.lightHandler.ScreenHeight()),
		LightCount:   lightCount,
		Near:         s.cam.Near(),
		Far:          s.cam.Far(),
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: cullBGP, Binding: 0, Offset: 0, Data: uniforms.Marshal()},
	})

	// Also update tile uniforms for the lit shader so screen dimensions
	// and tile counts stay in sync after resize.
	tileBGP := s.lightHandler.Bgp("tile_lit")
	tileUniforms := light.GPUTileUniforms{
		TileCountX:       uint32(s.lightHandler.TileCountX()),
		MaxLightsPerTile: uint32(s.lightHandler.MaxLightsPerTile()),
		ScreenWidth:      uint32(s.lightHandler.ScreenWidth()),
		ScreenHeight:     uint32(s.lightHandler.ScreenHeight()),
	}
	s.r.WriteBuffers([]bind_group_provider.BufferWrite{
		{Provider: tileBGP, Binding: 0, Offset: 0, Data: tileUniforms.Marshal()},
	})

	// Dispatch the light culling compute shader.
	if err := s.r.BeginComputeFrame(); err != nil {
		return
	}
	s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
		{PipelineKey: s.lightHandler.PipelineKey("light_cull"), Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: cullBGP}}, WorkGroupCount: [3]uint32{uint32(s.lightHandler.TileCountX()), uint32(s.lightHandler.TileCountY()), 1}},
	})
	s.r.EndComputeFrame()
}

func (s *scene) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.registry)
}

func (s *scene) CountEphemeral() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	count := 0
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			count += int(a.InstanceCount())
		}
	}
	return count
}

func (s *scene) AddGameObject(obj game_object.GameObject, pipelineOpts ...pipeline.PipelineBuilderOption) uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	mdl := obj.Model()
	if mdl == nil {
		panic("scene: cannot Add a GameObject without a Model")
	}

	// Auto-resolve standard shaders based on skinning and whether lighting is active.
	var computeShader, vertexShader, fragmentShader shader.Shader
	if mdl.Skinned() {
		computeShader = shader.NewShader(mdl.Name()+"_compute", shader.ShaderTypeCompute, "engine/renderer/animator/assets/skeletal-compute.wgsl", shader.WithInjections(s.injections))
		if s.lightHandler.Enabled() {
			vertexShader = shader.NewShader(mdl.Name()+"_vertex", shader.ShaderTypeVertex, "engine/light/assets/lit-skinned-vert.wgsl", shader.WithInjections(s.injections))
		} else {
			vertexShader = shader.NewShader(mdl.Name()+"_vertex", shader.ShaderTypeVertex, "engine/model/assets/skinned-vert.wgsl", shader.WithInjections(s.injections))
		}
	} else {
		computeShader = shader.NewShader(mdl.Name()+"_compute", shader.ShaderTypeCompute, "engine/renderer/animator/assets/simple-compute.wgsl", shader.WithInjections(s.injections))
		if s.lightHandler.Enabled() {
			vertexShader = shader.NewShader(mdl.Name()+"_vertex", shader.ShaderTypeVertex, "engine/light/assets/lit-vert.wgsl", shader.WithInjections(s.injections))
		} else {
			vertexShader = shader.NewShader(mdl.Name()+"_vertex", shader.ShaderTypeVertex, "engine/model/assets/simple-vert.wgsl", shader.WithInjections(s.injections))
		}
	}
	if s.lightHandler.Enabled() {
		fragmentShader = shader.NewShader(mdl.Name()+"_fragment", shader.ShaderTypeFragment, "engine/light/assets/lit-frag-csm.wgsl", shader.WithInjections(s.injections))
	} else {
		fragmentShader = shader.NewShader(mdl.Name()+"_fragment", shader.ShaderTypeFragment, "engine/model/assets/textured-frag.wgsl", shader.WithInjections(s.injections))
	}

	if obj.ID() == 0 {
		obj.SetID(atomic.AddUint64(&s.nextID, 1) - 1)
	}

	// Lookup or create an Animator for this Model
	animPool, exists := s.animatorPool[mdl]
	var anim animator.Animator
	if !exists {
		anim = s.createAnimator(mdl, computeShader, vertexShader, fragmentShader, pipelineOpts...)
		animPool = []animator.Animator{anim}
		s.animatorPool[mdl] = animPool
	} else {
		for _, a := range animPool {
			if a.InstanceCount() < a.MaxInstances() {
				anim = a
				break
			}
		}
		if anim == nil {
			anim = s.createAnimator(mdl, computeShader, vertexShader, fragmentShader, pipelineOpts...)
			animPool = append(animPool, anim)
			s.animatorPool[mdl] = animPool
		}
	}

	// Capture initial transform from the GameObject BEFORE wiring the animator.
	// TransformData returns the builder-supplied initial values (position, scale,
	// rotation, rotation speed) when the animator is nil. Once SetAnimator is called,
	// it would read from the animator's zero-initialized instance slot instead.
	pos, scale, rot, rotSpeed := obj.TransformData()

	// Wire the object to the animator and add an instance
	obj.SetAnimator(anim)
	idx, err := anim.AddInstance()
	if err != nil {
		panic(fmt.Sprintf("scene: failed to add instance for model %q: %v", mdl.Name(), err))
	}
	obj.SetAnimatorInstanceID(int(idx))

	// Push initial transform data from the GameObject into the animator slot
	anim.SetInstanceData(idx, pos, scale, rotSpeed, rot)

	// Update reverse-index so Remove() can find the swapped object in O(1).
	if s.instanceLookup[anim] == nil {
		s.instanceLookup[anim] = make(map[uint32]uint64)
	}
	s.instanceLookup[anim][idx] = obj.ID()

	// Persist non-ephemeral objects in the registry
	if !obj.Ephemeral() {
		s.registry[obj.ID()] = obj
	}

	// If the object has an attached light, track it for automatic position sync
	// and register the light with the handler's light list.
	if l := obj.Light(); l != nil {
		s.lightObjects = append(s.lightObjects, obj)
		s.lightHandler.AddLight(l)
	}

	if obj.RigidBody() != nil && s.physicsHandler != nil {
		bodyIndex := s.physicsHandler.RegisterBody(obj.ID(), [3]float32{pos[0], pos[1], pos[2]}, [3]float32{rot[0], rot[1], rot[2]}, obj.RigidBody(), uint32(obj.AnimatorInstanceID()))
		if !s.physicsGPUReady {
			s.initPhysics()
			s.physicsGPUReady = true
		}

		// Ensure this animator has a sync group. Each unique Animator that
		// owns physics bodies gets its own sync dispatch with a per-group
		// sync_map buffer (sentinel-initialized) and the animator's own
		// AnimationData buffer. Bodies not belonging to a group are skipped
		// by the shader via the 0xFFFFFFFF sentinel.
		if s.physicsSyncAnimMap == nil {
			s.physicsSyncAnimMap = make(map[animator.Animator]int)
		}
		if s.physicsSyncGroup == nil {
			s.physicsSyncGroup = make(map[int]bind_group_provider.BindGroupProvider)
		}
		sgIdx, exists := s.physicsSyncAnimMap[anim]
		if !exists {
			sgIdx = s.initPhysicsSyncGroup(anim)
		}

		// Stage a write of this body's instance_id into the group's sync_map buffer.
		instanceData := make([]byte, 4)
		binary.LittleEndian.PutUint32(instanceData, uint32(obj.AnimatorInstanceID()))
		s.physicsSyncWrites = append(s.physicsSyncWrites, bind_group_provider.BufferWrite{
			Provider: s.physicsSyncGroup[sgIdx],
			Binding:  1,
			Offset:   uint64(bodyIndex) * 4,
			Data:     instanceData,
		})

		// For kinematic bodies on skeletal animators, create a bone particle
		// update group so the bone_particle_update shader transforms their
		// particles through the current bone matrices each frame.
		rb := obj.RigidBody()
		mdl := obj.Model()
		if rb.Kinematic() && mdl != nil && mdl.Skinned() && mdl.Skeleton() != nil {
			s.createBoneParticleUpdateGroup(anim, bodyIndex, mdl, uint32(obj.AnimatorInstanceID()))
		}
	}

	return obj.ID()
}

func (s *scene) RemoveGameObject(id uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	obj, exists := s.registry[id]
	if !exists {
		return
	}

	delete(s.registry, id)

	// Remove attached light from scene tracking lists
	if l := obj.Light(); l != nil {
		s.lightHandler.RemoveLight(l)
		sh := s.lightHandler.ShadowHandler()
		sh.OnLightRemoved(l)
		for i, o := range s.lightObjects {
			if o == obj {
				s.lightObjects = append(s.lightObjects[:i], s.lightObjects[i+1:]...)
				break
			}
		}
	}

	// Swap-remove the instance data from the animator and patch the physics
	// sync_map so the sync shader writes each body's transform to the correct
	// Animator instance slot after the swap.
	anim := obj.Animator()
	if anim != nil {
		removedIdx := obj.AnimatorInstanceID()
		if removedIdx >= 0 {
			swappedFrom, swapped := anim.RemoveInstance(uint32(removedIdx))

			// Remove the deleted object's entry from the reverse-index.
			if lut := s.instanceLookup[anim]; lut != nil {
				delete(lut, uint32(removedIdx))
			}

			if swapped {
				// The instance at swappedFrom was moved into removedIdx.
				// Use the reverse-index for O(1) lookup instead of scanning
				// the entire registry.
				if lut := s.instanceLookup[anim]; lut != nil {
					if swappedObjID, ok := lut[swappedFrom]; ok {
						if o, exists := s.registry[swappedObjID]; exists {
							o.SetAnimatorInstanceID(removedIdx)
							s.patchSyncMapEntry(anim, o.ID(), uint32(removedIdx))
						}
						// Update the reverse-index: swappedFrom is gone, now lives at removedIdx.
						delete(lut, swappedFrom)
						lut[uint32(removedIdx)] = swappedObjID
					}
				}
			}
			obj.SetAnimatorInstanceID(-1)
		}
	}

	// Prune empty animator from pool.
	if anim != nil && anim.InstanceCount() == 0 {
		s.pruneAnimator(anim)
	}

	// Sentinel the removed body's sync_map entry and deactivate its GPU slot.
	if s.physicsHandler != nil {
		s.patchSyncMapEntry(anim, obj.ID(), 0xFFFFFFFF)
		s.physicsHandler.RemoveBody(obj.ID())
	}
}

func (s *scene) PrepareCompute(deltaTime float32) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Update camera matrices and write VP matrix to GPU once per frame
	var gpuPlanes [6]animator.GPUFrustumPlane
	hasFrustum := false
	s.cam.Update()
	vpMat := s.cam.ViewProjectionMatrix()

	// Gather Hi-Z data for animator occlusion culling.
	hiZMipCount := 0
	projX := float32(0)
	if s.lightHandler != nil {
		if ssrHandler := s.lightHandler.SSRHandler(); ssrHandler != nil {
			hiZMipCount = ssrHandler.HiZMipCount()
		}
	}
	projMat := s.cam.ProjectionMatrix()
	projX = projMat[0]
	if camBGP := s.cam.BindGroupProvider(); camBGP != nil {
		camUniform := camera.GPUCameraUniform{
			ViewProj: vpMat,
			View:     s.cam.ViewMatrix(),
		}
		if ctrl := s.cam.Controller(); ctrl != nil {
			camUniform.CameraPosition[0], camUniform.CameraPosition[1], camUniform.CameraPosition[2] = ctrl.Position()
		}
		s.r.WriteBuffers([]bind_group_provider.BufferWrite{
			{
				Provider: camBGP,
				Binding:  0,
				Offset:   0,
				Data:     camUniform.Marshal(),
			},
		})
	}

	// Extract frustum planes from the VP matrix for GPU-side culling
	frustum := common.ExtractFrustumFromMatrix(vpMat[:])
	for i := range 6 {
		gpuPlanes[i] = animator.GPUFrustumPlane{
			Normal:   frustum.Planes[i].Normal,
			Distance: frustum.Planes[i].Distance,
		}
	}
	hasFrustum = !s.cullingDisabled

	// Sync attached lights: copy each game object's world position to its light.
	for _, obj := range s.lightObjects {
		if l := obj.Light(); l != nil && obj.Enabled() {
			x, y, z := obj.Position()
			l.SetPosition(x, y, z)
		}
	}

	// Process all animator groups in three phases:
	// Pre-pass (serial): rebuild GPU buffers for any groups that grew since last frame.
	// Phase 1 (parallel): fan out CPU-only prep work across goroutines.
	// Phase 2 (serial): coalesce buffer writes and dispatch compute shaders.

	// Pre-pass: serial RebuildGPU for animators that grew — requires GPU access.
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}
			// we don't currently have a RebuildGPU step and we shouldn't have one because we just want to ignore dead animators
			// ideally we should remove the dead animator from the pool, but should that be done here or elsewhere?
			// if a.NeedsRebuild() {
			// 	if err := a.RebuildGPU(s.r.InitBindGroups); err != nil {
			// 		continue
			// 	}
			// }
		}
	}

	// Phase 1: parallel CPU prep — submit each animator's prep work to the
	// compute pool. Workers are reused across frames (no goroutine spawn overhead).
	// A WaitGroup provides per-frame barrier sync since pool.Wait() blocks until
	// workers idle-exit which is unsuitable for frame-rate workloads.
	var wg sync.WaitGroup
	taskID := 0
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}

			mdl := a.Model()
			if mdl == nil {
				continue
			}
			pipeKey := mdl.ComputePipelineKey()
			if pipeKey == "" {
				continue
			}
			pipe := s.r.Pipeline(pipeKey)
			if pipe == nil {
				continue
			}
			shdr := pipe.Shader(shader.ShaderTypeCompute)
			if shdr == nil {
				continue
			}

			wg.Add(1)
			aCap := a // capture for closure
			id := taskID
			taskID++
			s.computePool.SubmitTask(worker.Task{
				ID: id,
				Do: func() (any, error) {
					defer wg.Done()

					uniformBinding, instanceBinding, boneBinding, modelBinding := 0, 0, 0, 0
					for _, decl := range shdr.Declarations() {
						if decl.Type != shader.AnnotationTypeBindingGroup || decl.Binding == nil {
							continue
						}
						typeArg := string(decl.Args[2])
						if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
							typeArg = strings.TrimSuffix(stripped, ">")
						}
						switch shader.AnnotationArg(typeArg) {
						case shader.AnnotationArgGlobalData, shader.AnnotationArgAnimationGlobals:
							uniformBinding = *decl.Binding
						case shader.AnnotationArgAnimationData, shader.AnnotationArgSkeletalAnimationData:
							instanceBinding = *decl.Binding
						case shader.AnnotationArgBoneInfo:
							boneBinding = *decl.Binding
						case shader.AnnotationArgModelData:
							modelBinding = *decl.Binding
						}
					}

					// Feed frustum planes to the animator for GPU-side culling.
					// This must happen before PrepareFrame so the uniform data includes the planes.
					if hasFrustum {
						aCap.SetFrustumPlanes(gpuPlanes)
					}

					aCap.SetScreenSize(s.screenWidth, s.screenHeight)
					aCap.SetProjectionX(projX)
					aCap.SetHiZMipCount(hiZMipCount)
					aCap.SetViewProj(vpMat)

					aCap.PrepareFrame(deltaTime, uniformBinding)
					aCap.Flush(instanceBinding, boneBinding, modelBinding)
					return nil, nil
				},
			})
		}
	}
	wg.Wait()

	// Phase 2: coalesced GPU submission — collect all buffer writes from all animators into a single
	// slice, then submit once to the renderer. This reduces mutex acquisitions from N to 1 for writes.
	// For each animator with culling enabled, reset the indirect args buffer to zero instance count
	// before collecting its writes, so the compute shader can atomically count visible instances.
	allWrites := s.writePool[:0]
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}
			if a.CullingEnabled() {
				if m := a.Model(); m != nil {
					if mp := m.MeshProvider(); mp != nil {
						pipeKey := m.ComputePipelineKey()
						if pipeKey == "" {
							continue
						}
						pipe := s.r.Pipeline(pipeKey)
						if pipe == nil {
							continue
						}
						shdr := pipe.Shader(shader.ShaderTypeCompute)
						if shdr == nil {
							continue
						}

						indirectBinding := 0
						for _, decl := range shdr.Declarations() {
							if decl.Type != shader.AnnotationTypeBindingGroup || decl.Binding == nil {
								continue
							}
							typeArg := string(decl.Args[2])
							if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
								typeArg = strings.TrimSuffix(stripped, ">")
							}
							if shader.AnnotationArg(typeArg) == shader.AnnotationArgIndirectArgs {
								indirectBinding = *decl.Binding
								break
							}
						}
						a.ResetIndirectArgs(uint32(mp.IndexCount()), indirectBinding)
					}
				}
			}

			allWrites = append(allWrites, a.StagedWriteData()...)
		}
	}
	s.writePool = allWrites

	if len(allWrites) > 0 {
		s.r.WriteBuffers(allWrites)
	}

	// ── Physics compute dispatch ───────────────────────────────────────
	// Runs BEFORE the animator compute dispatches so the sync shader's writes to
	// AnimationData are visible when the animator reads it for model-matrix
	// generation and frustum culling. The 8-stage GPU rigid body pipeline uses a
	// fixed-timestep accumulator. PrepareStep returns the number of substeps for
	// this frame and the marshaled globals uniform. Each substep dispatches the full
	// pipeline: particle values → AABB reduce → grid build params → grid clear →
	// grid insert → collision → compute momenta → integrate. Between substeps,
	// only the AABB atomics buffer is reset; body and particle data persist in the
	// storage buffers across stages.
	if ph := s.physicsHandler; ph != nil && ph.Enabled() {
		// Process any pending GPU→CPU readback from the previous frame's copy command.
		// By this point the compute command buffer containing the CopyBufferToBuffer has
		// been submitted (EndComputeFrame from the prior frame), so the staging buffer is
		// safe to map synchronously. This only runs when game logic called RequestReadback.
		if ph.ReadbackPending() {
			bodySize := uint64((&physics.GPUBody{}).Size())
			readSize := uint64(ph.BodiesCount()) * bodySize
			if readSize > 0 {
				data, err := s.r.ReadMappedBuffer(ph.StagingBuffer(), 0, readSize)
				if err == nil {
					ph.ProcessReadback(data)
				}
			}
			ph.ClearReadbackPending()
		}

		// Collect staged writes (body registrations, removals, force drains).
		// PrepareStep MUST be called first so that force drains from ApplyForce()
		// are staged into the same write batch as body registrations. Without this,
		// newly spawned bodies spend their first physics frame with zero external
		// force (no gravity), causing them to sit at the spawn point while collision
		// forces from overlapping neighbors fling them apart.
		substeps, globalsData := ph.PrepareStep(deltaTime)
		physWrites := ph.StagedWriteData()

		// Append per-group sync_map writes staged during Add() calls.
		if len(s.physicsSyncWrites) > 0 {
			physWrites = append(physWrites, s.physicsSyncWrites...)
			s.physicsSyncWrites = s.physicsSyncWrites[:0]
		}

		if substeps > 0 {
			// Write globals uniform once — it is constant across all substeps
			// since fixedDt does not change within a frame.
			physWrites = append(physWrites, bind_group_provider.BufferWrite{
				Provider: ph.Buffers(),
				Binding:  3,
				Offset:   0,
				Data:     globalsData,
			})

			if len(physWrites) > 0 {
				s.r.WriteBuffers(physWrites)
			}

			// physDispatchGroups computes the number of work groups needed to cover
			// itemCount invocations for the shader behind the given pipeline key.
			// The workgroup size is read from the parsed WGSL source, not hardcoded.
			physDispatchGroups := func(pipeKey string, itemCount uint32) [3]uint32 {
				pipe := s.r.Pipeline(pipeKey)
				if pipe == nil {
					return [3]uint32{1, 1, 1}
				}
				shdr := pipe.Shader(shader.ShaderTypeCompute)
				if shdr == nil {
					return [3]uint32{1, 1, 1}
				}
				wgSize := shdr.WorkgroupSize()
				xSize := wgSize[0]
				if xSize == 0 {
					xSize = 1
				}
				groups := (itemCount + xSize - 1) / xSize
				if groups == 0 {
					groups = 1
				}
				return [3]uint32{groups, 1, 1}
			}

			particleCount := uint32(ph.ParticleCount())
			bodyCount := uint32(ph.BodiesCount())

			// Pre-build AABB atomics reset payload (6 × u32):
			//   indices 0–2 (min): 0xFFFFFFFF (largest sortable uint → will be atomicMin'd down)
			//   indices 3–5 (max): 0x00000000 (smallest sortable uint → will be atomicMax'd up)
			aabbReset := make([]byte, 24)
			binary.LittleEndian.PutUint32(aabbReset[0:4], 0xFFFFFFFF)
			binary.LittleEndian.PutUint32(aabbReset[4:8], 0xFFFFFFFF)
			binary.LittleEndian.PutUint32(aabbReset[8:12], 0xFFFFFFFF)
			// indices 3–5 are already zero from make()

			pvKey := ph.PipelineKey("particle_values")
			arKey := ph.PipelineKey("aabb_reduce")
			gbKey := ph.PipelineKey("grid_build_params")
			gcKey := ph.PipelineKey("grid_clear")
			giKey := ph.PipelineKey("grid_insert")
			crKey := ph.PipelineKey("collision")
			cmKey := ph.PipelineKey("momenta")
			iKey := ph.PipelineKey("integrate")

			for sub := 0; sub < substeps; sub++ {
				// WriteBuffers must precede the dispatch batch (Queue.writeBuffer takes effect
				// before the next GPU submit, providing the AABB reset before any dispatch runs).
				s.r.WriteBuffers([]bind_group_provider.BufferWrite{
					{Provider: ph.Buffers(), Binding: 5, Offset: 0, Data: aabbReset},
				})
				// Each physics pipeline stage depends on the previous stage's output.
				// Separate compute passes provide automatic GPU barriers (READ-AFTER-WRITE).
				s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
					{PipelineKey: pvKey, Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ph.Bgp("particle_values")}}, WorkGroupCount: physDispatchGroups(pvKey, particleCount)},
				})
				s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
					{PipelineKey: arKey, Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ph.Bgp("aabb_reduce")}}, WorkGroupCount: physDispatchGroups(arKey, particleCount)},
				})
				s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
					{PipelineKey: gbKey, Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ph.Bgp("grid_build_params")}}, WorkGroupCount: physDispatchGroups(gbKey, 1)},
				})
				s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
					{PipelineKey: gcKey, Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ph.Bgp("grid_clear")}}, WorkGroupCount: physDispatchGroups(gcKey, uint32(ph.MaxGridCells()))},
				})
				s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
					{PipelineKey: giKey, Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ph.Bgp("grid_insert")}}, WorkGroupCount: physDispatchGroups(giKey, particleCount)},
				})
				s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
					{PipelineKey: crKey, Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ph.Bgp("collision")}}, WorkGroupCount: physDispatchGroups(crKey, particleCount)},
				})
				s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
					{PipelineKey: cmKey, Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ph.Bgp("momenta")}}, WorkGroupCount: physDispatchGroups(cmKey, bodyCount)},
				})
				s.r.DispatchComputeBatch([]renderer.ComputeDispatch{
					{PipelineKey: iKey, Providers: []renderer.ComputeGroupProvider{{Group: 0, Provider: ph.Bgp("integrate")}}, WorkGroupCount: physDispatchGroups(iKey, bodyCount)},
				})
			}

			// After all substeps, sync physics results back to each Animator's
			// AnimationData buffer. Each sync group dispatches the sync shader with
			// its own BGP that binds the correct AnimationData buffer and per-group
			// sync_map (sentinel-filtered so non-member bodies are skipped).
			if len(s.physicsSyncGroup) > 0 {
				syncKey := ph.PipelineKey("sync")
				syncWG := physDispatchGroups(syncKey, bodyCount)
				syncDispatches := make([]renderer.ComputeDispatch, 0, len(s.physicsSyncGroup))
				for i := 0; i < len(s.physicsSyncGroup); i++ {
					if bgp, ok := s.physicsSyncGroup[i]; ok {
						syncDispatches = append(syncDispatches, renderer.ComputeDispatch{
							PipelineKey:    syncKey,
							Providers:      []renderer.ComputeGroupProvider{{Group: 0, Provider: bgp}},
							WorkGroupCount: syncWG,
						})
					}
				}
				if len(syncDispatches) > 0 {
					s.r.DispatchComputeBatch(syncDispatches)
				}
			}

			// If game logic requested a readback, encode a GPU→GPU copy of the bodies
			// buffer into the staging buffer. The next frame will map and process it.
			if ph.ConsumeReadbackRequest() {
				if staging := ph.StagingBuffer(); staging != nil {
					copySize := uint64(bodyCount) * uint64((&physics.GPUBody{}).Size())
					s.r.CopyBufferToBuffer(ph.Buffers().Buffer(0), staging, 0, 0, copySize)
				}
			}
		} else if len(physWrites) > 0 {
			// No substeps this frame (accumulator hasn't reached fixedDt yet),
			// but we still need to flush registration/removal writes.
			s.r.WriteBuffers(physWrites)
		}
	}

	// Dispatch compute shaders for each registered animator with instances.
	// This runs AFTER the physics block so the sync shader's writes to
	// AnimationData (positions/rotations of physics-controlled bodies) are
	// visible when the animator's compute shader builds model matrices and
	// performs frustum culling.
	var animDispatches []renderer.ComputeDispatch
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}
			mdl := a.Model()
			if mdl == nil {
				continue
			}
			key := mdl.ComputePipelineKey()
			if key == "" {
				continue
			}
			pipe := s.r.Pipeline(key)
			if pipe == nil {
				continue
			}
			shdr := pipe.Shader(shader.ShaderTypeCompute)
			if shdr == nil {
				continue
			}
			// Dispatch the correct number of workgroups to cover all instances.
			// shdr.WorkgroupSize() returns the per-workgroup thread count (e.g. 256),
			// NOT the number of groups. We need ceil(instanceCount / workgroupSize).
			wgSize := shdr.WorkgroupSize()
			xSize := wgSize[0]
			if xSize == 0 {
				xSize = 1
			}
			instCount := a.InstanceCount()
			groups := (instCount + xSize - 1) / xSize
			animDispatches = append(animDispatches, renderer.ComputeDispatch{
				PipelineKey: key,
				Providers: []renderer.ComputeGroupProvider{
					{Group: 0, Provider: a.ComputeBindGroupProvider()},
					{Group: 1, Provider: a.HiZBindGroupProvider()},
				},
				WorkGroupCount: [3]uint32{groups, 1, 1},
			})
		}
	}
	if len(animDispatches) > 0 {
		s.r.DispatchComputeBatch(animDispatches)
	}

	// Dispatch bone particle update for kinematic bodies after animator compute.
	// The animator has populated scratch_matrices with current bone world matrices;
	// this shader transforms each kinematic particle through its bone matrix so the
	// next frame's physics collision pipeline sees the animated pose.
	if ph := s.physicsHandler; ph != nil && len(s.boneParticleUpdateGroups) > 0 {
		boneUpdateKey := ph.PipelineKey("bone_update")
		boneUpdatePipe := s.r.Pipeline(boneUpdateKey)
		if boneUpdatePipe != nil {
			boneUpdateShader := boneUpdatePipe.Shader(shader.ShaderTypeCompute)
			if boneUpdateShader != nil {
				wgSize := boneUpdateShader.WorkgroupSize()
				xSize := wgSize[0]
				if xSize == 0 {
					xSize = 1
				}
				boneDispatches := make([]renderer.ComputeDispatch, 0, len(s.boneParticleUpdateGroups))
				for _, bg := range s.boneParticleUpdateGroups {
					groups := (bg.particleCount + xSize - 1) / xSize
					if groups == 0 {
						groups = 1
					}
					boneDispatches = append(boneDispatches, renderer.ComputeDispatch{
						PipelineKey:    boneUpdateKey,
						Providers:      []renderer.ComputeGroupProvider{{Group: 0, Provider: bg.bgp}},
						WorkGroupCount: [3]uint32{groups, 1, 1},
					})
				}
				if len(boneDispatches) > 0 {
					s.r.DispatchComputeBatch(boneDispatches)
				}
			}
		}
	}
}

func (s *scene) DrawCalls() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, anim := range s.animatorPool {
		for _, a := range anim {
			if a.InstanceCount() == 0 {
				continue
			}

			mdl := a.Model()
			if mdl == nil {
				continue
			}
			meshProvider := mdl.MeshProvider()
			if meshProvider == nil {
				continue
			}

			mats := mdl.RenderMaterials()
			if len(mats) == 0 {
				continue
			}

			for _, mat := range mats {
				pipelineKey := mat.PipelineKey()
				if pipelineKey == "" {
					continue
				}

				// Look up the render pipeline to discover bind group layouts from both shaders.
				rp := s.r.Pipeline(pipelineKey)
				if rp == nil {
					continue
				}
				renderShader := rp.Shader(shader.ShaderTypeVertex)
				if renderShader == nil {
					continue
				}

				// Collect declarations from vertex and fragment shaders.
				allDecls := s.drawDeclsPool[:0]
				allDecls = append(allDecls, renderShader.Declarations()...)
				if fragShader := rp.Shader(shader.ShaderTypeFragment); fragShader != nil {
					allDecls = append(allDecls, fragShader.Declarations()...)
				}
				s.drawDeclsPool = allDecls

				// Build bind groups dynamically by matching each group's var names to a provider.
				// Groups are iterated in index order so bindGroups[i] maps to @group(i).
				maxGroup := -1
				clear(s.drawGroupProvidersPool)
				groupProviders := s.drawGroupProvidersPool
				for _, decl := range allDecls {
					if decl.Group == nil {
						continue
					}
					g := *decl.Group
					if g > maxGroup {
						maxGroup = g
					}
					if _, exists := groupProviders[g]; exists {
						continue
					}

					var provider bind_group_provider.BindGroupProvider
					switch decl.Type {
					case shader.AnnotationTypeProvider:
						switch decl.Args[0] {
						case shader.AnnotationArgCamera:
							provider = s.cam.BindGroupProvider()
						case shader.AnnotationArgMaterial:
							if mp := mat.Provider(g); mp != nil {
								provider = mp
							} else {
								provider = mat.BindGroupProvider()
							}
						case shader.AnnotationArgLights:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("lights")
							}
						case shader.AnnotationArgShadow:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.ShadowHandler().Bgp("csm_shadow_lit")
							}
						case shader.AnnotationArgTiles:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("tile_lit")
							}
						case shader.AnnotationArgEffect:
							if ep := mdl.EffectProvider(); ep != nil {
								provider = ep
							}
						case shader.AnnotationArgAnimator:
							provider = a.OutputBindGroupProvider()
						case shader.AnnotationArgSSAO:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("ssao_lit")
							}
						}
					case shader.AnnotationTypeBindingGroup:
						typeArg := string(decl.Args[2])
						if stripped, ok := strings.CutPrefix(typeArg, "array<"); ok {
							typeArg = strings.TrimSuffix(stripped, ">")
						}
						switch shader.AnnotationArg(typeArg) {
						case shader.AnnotationArgCamera:
							provider = s.cam.BindGroupProvider()
						case shader.AnnotationArgInstanceData:
							provider = a.OutputBindGroupProvider()
						case shader.AnnotationArgLight, shader.AnnotationArgLightHeader:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("lights")
							}
						case shader.AnnotationArgShadowUniform, shader.AnnotationArgCSMData:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.ShadowHandler().Bgp("csm_shadow_lit")
							}
						case shader.AnnotationArgTileUniforms:
							if s.lightHandler.Enabled() {
								provider = s.lightHandler.Bgp("tile_lit")
							}
						case shader.AnnotationArgOverlayParams:
							if bp := mat.BindGroupProvider(); bp != nil {
								provider = bp
							} else if ep := mdl.EffectProvider(); ep != nil {
								provider = ep
							}
						case shader.AnnotationArgEffectParams:
							if ep := mdl.EffectProvider(); ep != nil {
								provider = ep
							} else if mp := mat.Provider(g); mp != nil {
								provider = mp
							}
						}
					}

					if provider != nil {
						groupProviders[g] = provider
					}
				}

				bindGroups := s.drawBindGroupsPool[:0]
				skipMaterial := false
				for g := 0; g <= maxGroup; g++ {
					provider, ok := groupProviders[g]
					if !ok || provider == nil {
						skipMaterial = true
						break
					}
					bindGroups = append(bindGroups, provider)
				}
				if skipMaterial {
					continue
				}

				// Use indirect draw when GPU frustum culling is active — the compute shader writes
				// the visible instance count into the indirect args buffer, avoiding CPU readback.
				if a.CullingEnabled() {
					var indirectBinding int
					if key := mdl.ComputePipelineKey(); key != "" {
						rp := s.r.Pipeline(key)
						if rp == nil {
							continue
						}
						if cs := rp.Shader(shader.ShaderTypeCompute); cs != nil {
							for _, d := range cs.Declarations() {
								if d.Type == shader.AnnotationTypeBindingGroup && d.Binding != nil {
									arg := string(d.Args[2])
									if stripped, ok := strings.CutPrefix(arg, "array<"); ok {
										arg = strings.TrimSuffix(stripped, ">")
									}
									if shader.AnnotationArg(arg) == shader.AnnotationArgIndirectArgs {
										indirectBinding = *d.Binding
										break
									}
								}
							}
						}
					}
					if indBuf := a.IndirectBuffer(indirectBinding); indBuf != nil {
						if err := s.r.DrawCallIndirect(pipelineKey, meshProvider, indBuf, bindGroups); err != nil {
							return fmt.Errorf("indirect draw call failed for animator in scene %q: %w", s.name, err)
						}
						continue
					}
				}

				if err := s.r.DrawCall(pipelineKey, meshProvider, uint32(a.InstanceCount()), bindGroups); err != nil {
					return fmt.Errorf("draw call failed for animator in scene %q: %w", s.name, err)
				}
			}
		}
	}

	return nil
}
