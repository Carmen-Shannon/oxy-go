//go:build ignore

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine"
	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/loader"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	bgp "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

// ── Benchmark Configuration ────────────────────────────────────────
const (
	// litFoxInitialCount is the number of foxes in the first ramp step.
	litFoxInitialCount = 10
	// litFoxRampInterval is how long each ramp step runs before ramping.
	litFoxRampInterval = 5 * time.Second
	// litFoxFPSThreshold is the FPS floor — the benchmark stops ramping when
	// the average FPS over the ramp interval falls below this value.
	litFoxFPSThreshold = 30.0
	// litFoxRampFactor is the multiplier applied each step (1.5 = +50%).
	litFoxRampFactor = 1.5
	// litFoxMaxStep caps the absolute number of foxes added per ramp step
	// to prevent massive spawn stalls at high counts.
	litFoxMaxStep = 2500
	// litFoxSpacing determines how far apart foxes are placed in the XZ grid.
	litFoxSpacing = 120.0
	// litFoxMaxSide is the maximum number of foxes per side on the XZ plane
	// before new objects start stacking upward in layers.
	litFoxMaxSide = 60
	// litFoxLayerSpacing is the vertical distance between stacked layers.
	litFoxLayerSpacing = 150.0
)

func main() {
	// ── Engine + Window ─────────────────────────────────────────────
	eng := engine.NewEngine(
		engine.WithProfiling(true),
		engine.WithTickRate(60),
		engine.WithWindow(window.NewWindow(
			window.WithTitle("Oxy Benchmark — Many Foxes (Lit)"),
			window.WithWidth(1920),
			window.WithHeight(1080),
		)),
	)

	// ── Renderer (uncapped FPS) ─────────────────────────────────────
	r := renderer.NewRenderer(
		renderer.BackendTypeWGPU,
		eng.Window(),
		renderer.WithPresentMode(renderer.PresentModeUncapped),
	)

	// ── Camera ──────────────────────────────────────────────────────
	cam := camera.NewCamera(
		camera.WithFov(float32(45.0*math.Pi/180.0)),
		camera.WithAspect(float32(eng.Window().Width())/float32(eng.Window().Height())),
		camera.WithNear(0.1),
		camera.WithFar(100000),
		camera.WithController(camera.NewCameraController(
			camera.WithRadius(5000),
			camera.WithTarget(0, 50, 0),
			camera.WithElevation(1.2),
			camera.WithAzimuth(0.3),
			camera.WithRadiusBounds(1, 200000),
			camera.WithZoomSpeed(50.0),
			camera.WithMouseSensitivity(0.002),
			camera.WithPanSpeed(50.0),
		)),
	)

	// ── Shaders ─────────────────────────────────────────────────────
	// Skeletal compute + lit skinned vertex + lit fragment for fox meshes
	computeShader := shader.NewShader("skeletal_compute", shader.ShaderTypeCompute, "examples/assets/shaders/skeletal-compute.wgsl")
	litVert := shader.NewShader("lit_skinned_vert", shader.ShaderTypeVertex, "examples/assets/shaders/lit-skinned-vert.wgsl")
	litFrag := shader.NewShader("lit_frag", shader.ShaderTypeFragment, "examples/assets/shaders/lit-frag.wgsl")

	// Static compute + lit vertex for non-skinned geometry (ground plane)
	staticCompute := shader.NewShader("simple_compute", shader.ShaderTypeCompute, "examples/assets/shaders/simple-compute.wgsl")
	staticLitVert := shader.NewShader("lit_vert", shader.ShaderTypeVertex, "examples/assets/shaders/lit-vert.wgsl")

	// Shadow + light-cull shaders for forward+ pipeline
	shadowVert := shader.NewShader("shadow_depth_vert", shader.ShaderTypeVertex, "examples/assets/shaders/shadow-depth-vert.wgsl")
	shadowSkinnedVert := shader.NewShader("shadow_depth_skinned_vert", shader.ShaderTypeVertex, "examples/assets/shaders/shadow-depth-skinned-vert.wgsl")
	cullCompute := shader.NewShader("light_cull_compute", shader.ShaderTypeCompute, "examples/assets/shaders/light-cull-compute.wgsl")

	// ── Scene ───────────────────────────────────────────────────────
	// Shadow half-extent must cover the full fox grid at any sun angle.
	// Grid half = litFoxMaxSide * litFoxSpacing / 2 = 3600.
	// Worst-case diagonal (45° sun): 3600 * sqrt(2) ≈ 5091. Using 5500 for margin.
	// At 5500 half-extent with 4096 resolution, each shadow texel covers ~2.7 world units.
	sc := scene.NewScene("Many Foxes Lit Bench", cam, r, litVert,
		scene.WithActive(true),
		scene.WithShadowMapResolution(4096),
		scene.WithShadowHalfExtent(4500),
		scene.WithShadowNearFar(0.1, 15000),
		scene.WithShadowBias(0.001),
		scene.WithShadowNormalBiasScale(1.0),
	)

	// ── Lights ──────────────────────────────────────────────────────
	// Directional sun light (shadow-casting, auto-orbiting)
	sun := light.NewLight(light.LightTypeDirectional,
		light.WithDirection(0, -1, 0),
		light.WithColor(1.0, 0.95, 0.85),
		light.WithIntensity(1.5),
		light.WithCastsShadows(true),
		light.WithEnabled(true),
	)
	sc.AddLight(sun)

	// Blue point light — outside the grid corner, elevated to illuminate from above
	// Fox grid half-extent is ~3540, so position well beyond the grid edge.
	bluePoint := light.NewLight(light.LightTypePoint,
		light.WithPosition(-4500, 2000, 4500),
		light.WithColor(0.2, 0.4, 1.0),
		light.WithIntensity(2.5),
		light.WithRange(15000),
		light.WithEnabled(false),
		light.WithCastsShadows(true),
	)
	sc.AddLight(bluePoint)

	// Orange point light — opposite corner, mirrored from blue
	orangePoint := light.NewLight(light.LightTypePoint,
		light.WithPosition(4500, 2000, -4500),
		light.WithColor(1.0, 0.5, 0.1),
		light.WithIntensity(2.5),
		light.WithRange(15000),
		light.WithEnabled(false),
		light.WithCastsShadows(true),
	)
	sc.AddLight(orangePoint)

	// Green spot light — high above and behind the grid, angled down at center
	spot := light.NewLight(light.LightTypeSpot,
		light.WithPosition(0, 3000, 5000),
		light.WithDirection(0, -0.5, -1),
		light.WithColor(0.0, 1.0, 0.5),
		light.WithIntensity(3.0),
		light.WithRange(15000),
		light.WithSpotCone(30, 45),
		light.WithEnabled(false),
		light.WithCastsShadows(true),
	)
	sc.AddLight(spot)

	// Very dim ambient — dark scene makes lighting effects much more visible
	sc.SetAmbientColor([3]float32{0.01, 0.01, 0.02})

	// ── Initialize Full Lighting Pipeline ───────────────────────────
	sc.InitLighting(litFrag, shadowVert, shadowSkinnedVert, cullCompute,
		eng.Window().Width(), eng.Window().Height(),
	)

	// ── Load Fox Model ──────────────────────────────────────────────
	ldr := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))
	foxModel, err := ldr.Load("examples/assets/models/Fox.glb", litFrag)
	if err != nil {
		log.Fatalf("Failed to load Fox model: %v", err)
	}

	// ── Ground Plane ───────────────────────────────────────────────
	// A large flat slab under the foxes so directional shadow is visible.
	groundHalf := float32(litFoxMaxSide) * litFoxSpacing * 0.6 // cover full grid + margin
	groundVerts, groundIdx := buildLitFoxGroundPlane(groundHalf)
	groundObj := game_object.NewGameObject(
		game_object.WithModel(model.NewModel(
			model.WithName("ground_plane"),
			model.WithBoundingRadius(groundHalf*2),
			model.WithVertexData(common.SliceToBytes(groundVerts)),
			model.WithIndexData(common.SliceToBytes(groundIdx)),
			model.WithIndexCount(len(groundIdx)),
			model.WithMeshProvider(bgp.NewBindGroupProvider("ground_mesh")),
			model.WithRenderMaterials(material.NewMaterial(
				material.WithName("ground_material"),
				material.WithBaseColor([4]float32{0.35, 0.35, 0.35, 1.0}),
				material.WithPipelineKey("ground_plane"),
			)),
		)),
		game_object.WithPosition(0, -15, 0),
		game_object.WithScale(1, 1, 1),
	)
	groundMats := groundObj.Model().RenderMaterials()
	if len(groundMats) > 0 {
		if err := ldr.InitMaterialGPU(groundMats[0], litFrag, "ground_material"); err != nil {
			log.Fatalf("Failed to init ground material GPU: %v", err)
		}
	}
	sc.Add(groundObj, staticCompute, staticLitVert, litFrag)

	// ── Sun Indicator Sphere ────────────────────────────────────────
	// A small orange sphere that tracks the shadow-map eye position for
	// the directional sun light, making it easy to visualize where the light is.
	sunSphereVerts, sunSphereIdx := buildLitFoxSphere(50, 12, 16, [4]float32{1.0, 0.6, 0.1, 1.0})
	sunIndicator := game_object.NewGameObject(
		game_object.WithModel(model.NewModel(
			model.WithName("sun_indicator"),
			model.WithBoundingRadius(20000.0),
			model.WithVertexData(common.SliceToBytes(sunSphereVerts)),
			model.WithIndexData(common.SliceToBytes(sunSphereIdx)),
			model.WithIndexCount(len(sunSphereIdx)),
			model.WithMeshProvider(bgp.NewBindGroupProvider("sun_sphere_mesh")),
			model.WithRenderMaterials(material.NewMaterial(
				material.WithName("sun_sphere_material"),
				material.WithBaseColor([4]float32{1.0, 0.6, 0.1, 1.0}),
				material.WithPipelineKey("sun_indicator"),
			)),
		)),
		game_object.WithPosition(0, 5000, 0),
		game_object.WithScale(1, 1, 1),
		game_object.WithEphemeral(true),
	)
	sunMats := sunIndicator.Model().RenderMaterials()
	if len(sunMats) > 0 {
		if err := ldr.InitMaterialGPU(sunMats[0], litFrag, "sun_sphere_material"); err != nil {
			log.Fatalf("Failed to init sun indicator material GPU: %v", err)
		}
	}
	sc.Add(sunIndicator, staticCompute, staticLitVert, litFrag)

	eng.AddScene(0, sc)

	// ── Spawn initial batch ─────────────────────────────────────────
	foxObjects := spawnLitFoxes(sc, foxModel, computeShader, litVert, litFrag, litFoxInitialCount, 0)
	currentCount := litFoxInitialCount

	// Start each fox playing a random animation at a random time offset
	animCount := foxModel.AnimationCount()
	randomizeLitFoxAnimations(foxObjects, animCount)

	// Override the bounding radius on the fox animator to prevent premature
	// frustum culling — animated poses (running strides, tail swings) extend
	// well beyond the rest-pose bounding sphere.
	if len(foxObjects) > 0 && foxObjects[0].Animator() != nil {
		foxObjects[0].Animator().SetBoundingRadius(200.0)
	}

	// ── Sun orbit state ─────────────────────────────────────────────
	var sunAngle float64

	// ── Benchmark tracking ──────────────────────────────────────────
	var (
		rampStart  = time.Now()
		frameCount int
		bestCount  int
		bestFPS    float64
		stopped    bool
	)

	animNames := foxModel.AnimationNames()
	fmt.Println("╔══════════════════════════════════════════════════════╗")
	fmt.Println("║  Oxy Engine Benchmark — Many Foxes (Lit)            ║")
	fmt.Println("╠══════════════════════════════════════════════════════╣")
	fmt.Printf("║  Initial count: %d foxes                            ║\n", litFoxInitialCount)
	fmt.Printf("║  Ramp interval: %v                              ║\n", litFoxRampInterval)
	fmt.Printf("║  FPS threshold: %.0f                                ║\n", litFoxFPSThreshold)
	fmt.Println("║  Lighting: 1 sun (shadow) + 2 point + 1 spot       ║")
	fmt.Println("║  Sun orbits automatically (day/night cycle)         ║")
	fmt.Println("║  Animations:                                        ║")
	for i, name := range animNames {
		fmt.Printf("║    %d = %-46s║\n", i, name)
	}
	fmt.Println("║  Camera: WASD=Pan  Q/E=Up/Down  Scroll=Zoom        ║")
	fmt.Println("║          Middle-mouse drag=Orbit                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════╝")
	log.Printf("[Bench] Starting with %d lit foxes (%d animations, %d bones)",
		currentCount, animCount, len(foxModel.Skeleton().Bones))

	// ── Per-frame render callback for FPS sampling ──────────────────
	eng.SetRenderCallback(func(_ float32) {
		frameCount++

		if stopped {
			return
		}

		elapsed := time.Since(rampStart)
		if elapsed >= litFoxRampInterval {
			fps := float64(frameCount) / elapsed.Seconds()
			log.Printf("[Bench] %d foxes → %.1f FPS (%.2f ms/frame)", currentCount, fps, 1000.0/fps)

			if fps > bestFPS {
				bestFPS = fps
				bestCount = currentCount
			}

			if fps < litFoxFPSThreshold {
				log.Printf("[Bench] ════════════════════════════════════════════")
				log.Printf("[Bench] RESULT: Below %.0f FPS threshold at %d foxes (%.1f FPS)", litFoxFPSThreshold, currentCount, fps)
				log.Printf("[Bench] Best sustained: %d foxes @ %.1f FPS", bestCount, bestFPS)
				log.Printf("[Bench] ════════════════════════════════════════════")
				stopped = true
				rampStart = time.Now()
				frameCount = 0
				return
			}

			// Compute next count: multiply by ramp factor, cap the delta
			nextCount := int(math.Ceil(float64(currentCount) * litFoxRampFactor))
			delta := nextCount - currentCount
			if delta > litFoxMaxStep {
				delta = litFoxMaxStep
				nextCount = currentCount + delta
			}

			spawnStart := time.Now()
			newFoxes := spawnLitFoxes(sc, foxModel, computeShader, litVert, litFrag, delta, currentCount)
			randomizeLitFoxAnimations(newFoxes, animCount)
			spawnDuration := time.Since(spawnStart)
			foxObjects = append(foxObjects, newFoxes...)
			currentCount = nextCount

			// Only pull camera back if the grid has grown beyond current view.
			// Compute the diagonal of the actual grid extent and ensure the camera
			// is far enough to see it, but never snap inward.
			effectiveSide := litFoxMaxSide
			if currentCount < litFoxMaxSide*litFoxMaxSide {
				effectiveSide = int(math.Ceil(math.Sqrt(float64(currentCount))))
			}
			layers := currentCount/(litFoxMaxSide*litFoxMaxSide) + 1
			gridW := float32(effectiveSide) * litFoxSpacing
			gridH := float32(layers) * litFoxLayerSpacing
			diag := float32(math.Sqrt(float64(gridW*gridW + gridH*gridH)))
			minRadius := diag * 0.6
			if minRadius > cam.Controller().Radius() {
				cam.Controller().SetRadius(minRadius)
			}

			log.Printf("[Bench] Ramping → %d foxes (+%d, spawn took %v)", currentCount, delta, spawnDuration.Round(time.Millisecond))
			rampStart = time.Now()
			frameCount = 0
		}
	})

	// ── Input + Sun Orbit ───────────────────────────────────────────
	setupLitFoxBenchInput(eng, cam, sun, &sunAngle, sunIndicator)

	log.Println("[Bench] Starting Oxy Engine — Many Foxes (Lit) Benchmark")
	eng.Run()
}

// spawnLitFoxes adds `count` fox GameObjects into the scene using a stable
// grid layout. Objects fill an XZ grid (up to litFoxMaxSide per axis) and once
// a layer is full, new objects stack upward on subsequent layers. This keeps
// the scene visually flat for maximum lighting coverage before growing vertically.
// Returns the created GameObjects so their animations can be randomized.
func spawnLitFoxes(
	sc scene.Scene,
	foxModel model.Model,
	computeShader, vertexShader, fragmentShader shader.Shader,
	count, startIdx int,
) []game_object.GameObject {
	objects := make([]game_object.GameObject, 0, count)

	for i := 0; i < count; i++ {
		idx := startIdx + i
		col := idx % litFoxMaxSide
		row := (idx / litFoxMaxSide) % litFoxMaxSide
		layer := idx / (litFoxMaxSide * litFoxMaxSide)

		cx := (float32(col) - float32(litFoxMaxSide-1)/2.0) * litFoxSpacing
		cz := (float32(row) - float32(litFoxMaxSide-1)/2.0) * litFoxSpacing
		cy := float32(layer) * litFoxLayerSpacing

		obj := game_object.NewGameObject(
			game_object.WithModel(foxModel),
			game_object.WithEphemeral(true),
			game_object.WithPosition(cx, cy, cz),
			game_object.WithScale(1, 1, 1),
		)
		sc.Add(obj, computeShader, vertexShader, fragmentShader)
		objects = append(objects, obj)
	}

	return objects
}

// randomizeLitFoxAnimations starts each fox playing a random animation clip with
// a random time offset so they're not all perfectly in sync.
func randomizeLitFoxAnimations(foxes []game_object.GameObject, animCount int) {
	if animCount == 0 {
		return
	}
	for _, fox := range foxes {
		anim := fox.Animator()
		if anim == nil {
			continue
		}
		clipIdx := uint32(rand.Intn(animCount))
		anim.PlayAnimation(uint32(fox.AnimatorInstanceID()), clipIdx, true)
		anim.SetAnimationTime(uint32(fox.AnimatorInstanceID()), rand.Float32()*5.0)
	}
}

// setupLitFoxBenchInput wires camera controls and the automatic sun orbit for
// the lit fox benchmark. The sun orbits continuously to simulate a day/night
// cycle: overhead → horizon → underneath → back overhead.
//
// Parameters:
//   - eng: the engine instance for registering callbacks
//   - cam: camera to control with input
//   - sun: the directional light whose direction is orbited
//   - sunAngle: pointer to the current sun orbit angle in radians
//   - sunIndicator: game object whose position tracks the shadow-map eye
func setupLitFoxBenchInput(eng engine.Engine, cam camera.Camera, sun light.Light, sunAngle *float64, sunIndicator game_object.GameObject) {
	keyState := make(map[uint32]bool)

	eng.Window().SetKeyDownCallback(func(keyCode uint32) {
		keyState[keyCode] = true
	})
	eng.Window().SetKeyUpCallback(func(keyCode uint32) {
		keyState[keyCode] = false
	})

	var dragging bool
	var lastX, lastY int32

	eng.Window().SetMiddleMouseDownCallback(func(x, y int32) {
		dragging = true
		lastX, lastY = x, y
	})
	eng.Window().SetMiddleMouseUpCallback(func(_, _ int32) {
		dragging = false
	})
	eng.Window().SetMouseMoveCallback(func(x, y int32) {
		if !dragging {
			return
		}
		dx := float32(x - lastX)
		dy := float32(y - lastY)
		cam.Controller().SetAzimuth(cam.Controller().Azimuth() + dx*cam.Controller().MouseSensitivity())
		cam.Controller().SetElevation(cam.Controller().Elevation() - dy*cam.Controller().MouseSensitivity())
		lastX, lastY = x, y
	})
	eng.Window().SetScrollCallback(func(delta float32) {
		cam.Controller().Zoom(delta)
	})

	eng.SetTickCallback(func(dt float32) {
		// Auto-orbit the sun: rotate direction in the YZ plane to simulate
		// a continuous day/night cycle (~30 seconds per full orbit at 0.2 rad/s).
		*sunAngle += float64(dt) * 0.2
		if *sunAngle > 2*math.Pi {
			*sunAngle -= 2 * math.Pi
		}
		dirY := float32(-math.Cos(*sunAngle))
		dirZ := float32(-math.Sin(*sunAngle))
		sun.SetDirection(0, dirY, dirZ)

		// Update sun indicator sphere position to show the shadow-map eye.
		// Eye = center - lightDir * far * 0.5 (matches ComputeDirectionalLightVP).
		{
			dir := sun.Direction()
			const shadowFarHalf = 7500.0 // far=15000 * 0.5
			// Shadow frustum center = camera target = (0, 50, 0)
			sunIndicator.SetPosition(
				0-dir[0]*shadowFarHalf,
				50-dir[1]*shadowFarHalf,
				0-dir[2]*shadowFarHalf,
			)
		}

		if keyState[common.KeyW] {
			cam.Controller().PanForward(1)
		}
		if keyState[common.KeyS] {
			cam.Controller().PanForward(-1)
		}
		if keyState[common.KeyA] {
			cam.Controller().PanRight(-1)
		}
		if keyState[common.KeyD] {
			cam.Controller().PanRight(1)
		}
		if keyState[common.KeyQ] {
			cam.Controller().PanUp(1)
		}
		if keyState[common.KeyE] {
			cam.Controller().PanUp(-1)
		}
	})
}

// buildLitFoxGroundPlane creates a large flat slab (6-face box with negligible
// thickness) for shadow-receiving. Normals point outward so the lit fragment
// shader lights the top face correctly.
//
// Parameters:
//   - half: half-extent of the slab in X and Z
//
// Returns:
//   - []model.GPUVertex: 24 vertices forming the slab (4 per face × 6 faces)
//   - []uint32: 36 indices forming 12 triangles
func buildLitFoxGroundPlane(half float32) ([]model.GPUVertex, []uint32) {
	color := [4]float32{0.35, 0.35, 0.35, 1.0}
	thickness := float32(20.0)

	v := func(px, py, pz, nx, ny, nz, u, vt, tx, ty, tz, tw float32) model.GPUVertex {
		return model.GPUVertex{
			Position: [3]float32{px, py, pz},
			Normal:   [3]float32{nx, ny, nz},
			TexCoord: [2]float32{u, vt},
			Color:    color,
			Tangent:  [4]float32{tx, ty, tz, tw},
		}
	}

	vertices := []model.GPUVertex{
		// Top face (+Y)
		v(-half, thickness, -half, 0, 1, 0, 0, 0, 1, 0, 0, 1),
		v(half, thickness, -half, 0, 1, 0, 1, 0, 1, 0, 0, 1),
		v(half, thickness, half, 0, 1, 0, 1, 1, 1, 0, 0, 1),
		v(-half, thickness, half, 0, 1, 0, 0, 1, 1, 0, 0, 1),

		// Bottom face (-Y)
		v(-half, -thickness, half, 0, -1, 0, 0, 0, 1, 0, 0, 1),
		v(half, -thickness, half, 0, -1, 0, 1, 0, 1, 0, 0, 1),
		v(half, -thickness, -half, 0, -1, 0, 1, 1, 1, 0, 0, 1),
		v(-half, -thickness, -half, 0, -1, 0, 0, 1, 1, 0, 0, 1),

		// Front face (+Z)
		v(-half, -thickness, half, 0, 0, 1, 0, 0, 1, 0, 0, 1),
		v(half, -thickness, half, 0, 0, 1, 1, 0, 1, 0, 0, 1),
		v(half, thickness, half, 0, 0, 1, 1, 1, 1, 0, 0, 1),
		v(-half, thickness, half, 0, 0, 1, 0, 1, 1, 0, 0, 1),

		// Back face (-Z)
		v(half, -thickness, -half, 0, 0, -1, 0, 0, -1, 0, 0, 1),
		v(-half, -thickness, -half, 0, 0, -1, 1, 0, -1, 0, 0, 1),
		v(-half, thickness, -half, 0, 0, -1, 1, 1, -1, 0, 0, 1),
		v(half, thickness, -half, 0, 0, -1, 0, 1, -1, 0, 0, 1),

		// Right face (+X)
		v(half, -thickness, half, 1, 0, 0, 0, 0, 0, 0, 1, 1),
		v(half, -thickness, -half, 1, 0, 0, 1, 0, 0, 0, 1, 1),
		v(half, thickness, -half, 1, 0, 0, 1, 1, 0, 0, 1, 1),
		v(half, thickness, half, 1, 0, 0, 0, 1, 0, 0, 1, 1),

		// Left face (-X)
		v(-half, -thickness, -half, -1, 0, 0, 0, 0, 0, 0, -1, 1),
		v(-half, -thickness, half, -1, 0, 0, 1, 0, 0, 0, -1, 1),
		v(-half, thickness, half, -1, 0, 0, 1, 1, 0, 0, -1, 1),
		v(-half, thickness, -half, -1, 0, 0, 0, 1, 0, 0, -1, 1),
	}

	indices := []uint32{
		0, 2, 1, 0, 3, 2, // top
		4, 6, 5, 4, 7, 6, // bottom
		8, 10, 9, 8, 11, 10, // front
		12, 14, 13, 12, 15, 14, // back
		16, 18, 17, 16, 19, 18, // right
		20, 22, 21, 20, 23, 22, // left
	}

	return vertices, indices
}

// buildLitFoxSphere creates a UV sphere mesh for use as a visual indicator
// (e.g. light position marker). The sphere is centered at the origin with
// the given radius.
//
// Parameters:
//   - radius: the radius of the sphere in world units
//   - rings: the number of horizontal rings (latitude subdivisions, minimum 2)
//   - segments: the number of vertical segments (longitude subdivisions, minimum 3)
//   - color: RGBA vertex color for all vertices
//
// Returns:
//   - []model.GPUVertex: vertices forming the sphere surface
//   - []uint32: triangle indices with CCW winding
func buildLitFoxSphere(radius float32, rings, segments int, color [4]float32) ([]model.GPUVertex, []uint32) {
	var vertices []model.GPUVertex
	var indices []uint32

	for r := 0; r <= rings; r++ {
		phi := math.Pi * float64(r) / float64(rings)
		y := float32(math.Cos(phi)) * radius
		ringRadius := float32(math.Sin(phi)) * radius

		for s := 0; s <= segments; s++ {
			theta := 2.0 * math.Pi * float64(s) / float64(segments)
			x := ringRadius * float32(math.Cos(theta))
			z := ringRadius * float32(math.Sin(theta))

			nx := float32(math.Sin(phi) * math.Cos(theta))
			ny := float32(math.Cos(phi))
			nz := float32(math.Sin(phi) * math.Sin(theta))

			u := float32(s) / float32(segments)
			v := float32(r) / float32(rings)

			tx := float32(-math.Sin(theta))
			tz := float32(math.Cos(theta))

			vertices = append(vertices, model.GPUVertex{
				Position: [3]float32{x, y, z},
				Normal:   [3]float32{nx, ny, nz},
				TexCoord: [2]float32{u, v},
				Color:    color,
				Tangent:  [4]float32{tx, 0, tz, 1},
			})
		}
	}

	stride := segments + 1
	for r := 0; r < rings; r++ {
		for s := 0; s < segments; s++ {
			a := uint32(r*stride + s)
			b := uint32(r*stride + s + 1)
			c := uint32((r+1)*stride + s)
			d := uint32((r+1)*stride + s + 1)

			indices = append(indices, a, c, b)
			indices = append(indices, b, c, d)
		}
	}

	return vertices, indices
}
