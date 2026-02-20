//go:build ignore

package main

import (
	"encoding/binary"
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
	// litCubeInitialCount is the number of cubes in the first ramp step.
	litCubeInitialCount = 1000
	// litCubeRampInterval is how long each ramp step runs before increasing.
	litCubeRampInterval = 5 * time.Second
	// litCubeFPSThreshold is the FPS floor — the benchmark stops ramping when
	// the average FPS over the ramp interval falls below this value.
	litCubeFPSThreshold = 30.0
	// litCubeSpacing determines how far apart cubes are placed in the grid.
	litCubeSpacing = 3.0
	// litCubeRampFactor is the multiplier applied each step (1.5 = +50%).
	litCubeRampFactor = 1.5
	// litCubeMaxStep caps the absolute number of cubes added per ramp step
	// to prevent long spawn freezes at high counts.
	litCubeMaxStep = 1_000_000
	// litCubeMaxSide is the maximum number of cubes per side on the XZ plane
	// before new objects start stacking upward in layers.
	litCubeMaxSide = 200
)

func main() {
	// ── Engine + Window ─────────────────────────────────────────────
	eng := engine.NewEngine(
		engine.WithProfiling(true),
		engine.WithTickRate(60),
		engine.WithWindow(window.NewWindow(
			window.WithTitle("Oxy Benchmark — Many Cubes (Lit)"),
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
		camera.WithFov(float32(60.0*math.Pi/180.0)),
		camera.WithAspect(float32(eng.Window().Width())/float32(eng.Window().Height())),
		camera.WithNear(0.1),
		camera.WithFar(100000),
		camera.WithController(camera.NewCameraController(
			camera.WithRadius(50),
			camera.WithTarget(0, 0, 0),
			camera.WithElevation(0.6),
			camera.WithAzimuth(0.3),
			camera.WithRadiusBounds(1, 200000),
			camera.WithZoomSpeed(50.0),
			camera.WithMouseSensitivity(0.002),
		)),
	)

	// ── Shaders ─────────────────────────────────────────────────────
	// Compute + vertex for static (non-skinned) lit geometry
	computeShader := shader.NewShader("simple_compute", shader.ShaderTypeCompute, "examples/assets/shaders/simple-compute.wgsl")
	litVert := shader.NewShader("lit_vert", shader.ShaderTypeVertex, "examples/assets/shaders/lit-vert.wgsl")
	litFrag := shader.NewShader("lit_frag", shader.ShaderTypeFragment, "examples/assets/shaders/lit-frag.wgsl")

	// Shadow + light-cull shaders for forward+ pipeline
	shadowVert := shader.NewShader("shadow_depth_vert", shader.ShaderTypeVertex, "examples/assets/shaders/shadow-depth-vert.wgsl")
	// Shadow skinned vert needed by InitLighting even if we don't have skinned meshes
	shadowSkinnedVert := shader.NewShader("shadow_depth_skinned_vert", shader.ShaderTypeVertex, "examples/assets/shaders/shadow-depth-skinned-vert.wgsl")
	cullCompute := shader.NewShader("light_cull_compute", shader.ShaderTypeCompute, "examples/assets/shaders/light-cull-compute.wgsl")

	// ── Loader (for material GPU init of procedural geometry) ────────
	ldr := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))

	// ── Build Cube Model ────────────────────────────────────────────
	cubeModel := buildLitCubeModel()

	// Initialize the cube material's GPU resources (fallback textures, samplers,
	// and the @group(2) bind group required by lit-frag.wgsl).
	cubeMats := cubeModel.RenderMaterials()
	if len(cubeMats) > 0 {
		if err := ldr.InitMaterialGPU(cubeMats[0], litFrag, "cube_lit_material"); err != nil {
			log.Fatalf("Failed to init cube material GPU: %v", err)
		}
	}

	// ── Scene ───────────────────────────────────────────────────────
	sc := scene.NewScene("Many Cubes Lit Bench", cam, r, litVert,
		scene.WithActive(true),
		scene.WithShadowHalfExtent(500),
		scene.WithShadowNearFar(0.1, 5000),
		scene.WithShadowBias(0.001),
	)

	// ── Lights ──────────────────────────────────────────────────────
	// Directional sun light (shadow-casting, auto-orbiting)
	sun := light.NewLight(light.LightTypeDirectional,
		light.WithDirection(0, -1, 0),
		light.WithColor(1.0, 0.95, 0.85),
		light.WithIntensity(3.0),
		light.WithCastsShadows(true),
		light.WithEnabled(true),
	)
	sc.AddLight(sun)

	// Blue point light — outside the grid corner, elevated to illuminate from above
	// Cube grid half-extent is ~300, so position well beyond the grid edge.
	bluePoint := light.NewLight(light.LightTypePoint,
		light.WithPosition(-500, 250, 400),
		light.WithColor(0.2, 0.4, 1.0),
		light.WithIntensity(4.0),
		light.WithRange(5000),
		light.WithCastsShadows(true),
		light.WithEnabled(true),
	)
	sc.AddLight(bluePoint)

	// Orange point light — opposite corner, mirrored from blue
	orangePoint := light.NewLight(light.LightTypePoint,
		light.WithPosition(500, 250, -400),
		light.WithColor(1.0, 0.5, 0.1),
		light.WithIntensity(4.0),
		light.WithRange(5000),
		light.WithCastsShadows(true),
		light.WithEnabled(true),
	)
	sc.AddLight(orangePoint)

	// Green spot light — high above and behind the grid, angled down at center
	spot := light.NewLight(light.LightTypeSpot,
		light.WithPosition(0, 400, 500),
		light.WithDirection(0, -0.7, -1),
		light.WithColor(0.0, 1.0, 0.5),
		light.WithIntensity(5.0),
		light.WithRange(5000),
		light.WithSpotCone(30, 40),
		light.WithCastsShadows(true),
		light.WithEnabled(true),
	)
	sc.AddLight(spot)

	// Very dim ambient — dark scene makes lighting effects much more visible
	sc.SetAmbientColor([3]float32{0.02, 0.02, 0.03})

	// ── Initialize Full Lighting Pipeline ───────────────────────────
	sc.InitLighting(litFrag, shadowVert, shadowSkinnedVert, cullCompute,
		eng.Window().Width(), eng.Window().Height(),
	)

	eng.AddScene(0, sc)

	// ── Ground Plane ───────────────────────────────────────────────
	// A large flat slab under the cubes so directional shadow is visible.
	groundHalf := float32(litCubeMaxSide) * litCubeSpacing * 0.6
	groundVerts, groundIdx := buildLitCubeGroundPlane(groundHalf)
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
		game_object.WithPosition(0, -1.5, 0),
		game_object.WithScale(1, 1, 1),
	)
	groundMats := groundObj.Model().RenderMaterials()
	if len(groundMats) > 0 {
		if err := ldr.InitMaterialGPU(groundMats[0], litFrag, "ground_material"); err != nil {
			log.Fatalf("Failed to init ground material GPU: %v", err)
		}
	}
	sc.Add(groundObj, computeShader, litVert, litFrag)

	// ── Spawn initial batch ─────────────────────────────────────────
	spawnLitCubes(sc, cubeModel, computeShader, litVert, litFrag, litCubeInitialCount)
	currentCount := litCubeInitialCount

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

	fmt.Println("╔══════════════════════════════════════════════════════╗")
	fmt.Println("║  Oxy Engine Benchmark — Many Cubes (Lit)            ║")
	fmt.Println("╠══════════════════════════════════════════════════════╣")
	fmt.Printf("║  Initial count: %d cubes                          ║\n", litCubeInitialCount)
	fmt.Printf("║  Ramp interval: %v                              ║\n", litCubeRampInterval)
	fmt.Printf("║  FPS threshold: %.0f                                ║\n", litCubeFPSThreshold)
	fmt.Println("║  Lighting: 1 sun (shadow) + 2 point + 1 spot       ║")
	fmt.Println("║  Sun orbits automatically (day/night cycle)         ║")
	fmt.Println("║  Camera: WASD=Pan  Q/E=Up/Down  Scroll=Zoom        ║")
	fmt.Println("║          Middle-mouse drag=Orbit                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════╝")
	log.Printf("[Bench] Starting with %d lit cubes", currentCount)

	// ── Per-frame render callback for FPS sampling ──────────────────
	eng.SetRenderCallback(func(_ float32) {
		frameCount++

		if stopped {
			return
		}

		elapsed := time.Since(rampStart)
		if elapsed >= litCubeRampInterval {
			fps := float64(frameCount) / elapsed.Seconds()
			log.Printf("[Bench] %d cubes → %.1f FPS (%.2f ms/frame)", currentCount, fps, 1000.0/fps)

			if fps > bestFPS {
				bestFPS = fps
				bestCount = currentCount
			}

			if fps < litCubeFPSThreshold {
				log.Printf("[Bench] ════════════════════════════════════════════")
				log.Printf("[Bench] RESULT: Below %.0f FPS threshold at %d cubes (%.1f FPS)", litCubeFPSThreshold, currentCount, fps)
				log.Printf("[Bench] Best sustained: %d cubes @ %.1f FPS", bestCount, bestFPS)
				log.Printf("[Bench] ════════════════════════════════════════════")
				stopped = true
				return
			}

			// Compute next count: multiply by ramp factor, cap the delta
			nextCount := int(math.Ceil(float64(currentCount) * litCubeRampFactor))
			delta := nextCount - currentCount
			if delta > litCubeMaxStep {
				delta = litCubeMaxStep
				nextCount = currentCount + delta
			}

			spawnStart := time.Now()
			spawnLitCubes(sc, cubeModel, computeShader, litVert, litFrag, delta)
			spawnDuration := time.Since(spawnStart)
			currentCount = nextCount

			// Only pull camera back if the grid has grown beyond current view.
			effectiveSide := litCubeMaxSide
			if currentCount < litCubeMaxSide*litCubeMaxSide {
				effectiveSide = int(math.Ceil(math.Sqrt(float64(currentCount))))
			}
			layers := currentCount/(litCubeMaxSide*litCubeMaxSide) + 1
			gridW := float32(effectiveSide) * litCubeSpacing
			gridH := float32(layers) * litCubeSpacing
			diag := float32(math.Sqrt(float64(gridW*gridW + gridH*gridH)))
			minRadius := diag * 0.7
			if minRadius > cam.Controller().Radius() {
				cam.Controller().SetRadius(minRadius)
			}

			log.Printf("[Bench] Ramping → %d cubes (+%d, spawn took %v)", currentCount, delta, spawnDuration.Round(time.Millisecond))
			rampStart = time.Now()
			frameCount = 0
		}
	})

	// ── Input + Sun Orbit ───────────────────────────────────────────
	setupLitCubeBenchInput(eng, cam, sun, &sunAngle)

	log.Println("[Bench] Starting Oxy Engine — Many Cubes (Lit) Benchmark")
	eng.Run()
}

// spawnLitCubes adds `count` ephemeral lit cube GameObjects into the scene using
// a stable grid layout. Objects fill an XZ grid (up to litCubeMaxSide per axis) and
// once a layer is full, new objects stack upward on subsequent layers.
// Each cube gets a random rotation speed for visual variety.
func spawnLitCubes(
	sc scene.Scene,
	cubeModel model.Model,
	computeShader, vertexShader, fragmentShader shader.Shader,
	count int,
) {
	existing := sc.CountEphemeral()

	for i := 0; i < count; i++ {
		idx := existing + i
		col := idx % litCubeMaxSide
		row := (idx / litCubeMaxSide) % litCubeMaxSide
		layer := idx / (litCubeMaxSide * litCubeMaxSide)

		cx := (float32(col) - float32(litCubeMaxSide-1)/2.0) * litCubeSpacing
		cz := (float32(row) - float32(litCubeMaxSide-1)/2.0) * litCubeSpacing
		cy := float32(layer) * litCubeSpacing

		obj := game_object.NewGameObject(
			game_object.WithModel(cubeModel),
			game_object.WithEphemeral(true),
			game_object.WithPosition(cx, cy, cz),
			game_object.WithScale(1, 1, 1),
			game_object.WithRotationSpeed(
				rand.Float32()*2.0-1.0,
				rand.Float32()*2.0-1.0,
				rand.Float32()*2.0-1.0,
			),
		)
		sc.Add(obj, computeShader, vertexShader, fragmentShader)
	}
}

// buildLitCubeModel creates a unit cube model with per-vertex rainbow colors
// and proper normals for lit rendering. Each face has a distinct color and
// outward-facing normal.
func buildLitCubeModel() model.Model {
	faceColors := [][4]float32{
		{1, 0, 0, 1}, // +X red
		{0, 1, 0, 1}, // -X green
		{0, 0, 1, 1}, // +Y blue
		{1, 1, 0, 1}, // -Y yellow
		{1, 0, 1, 1}, // +Z magenta
		{0, 1, 1, 1}, // -Z cyan
	}

	type faceData struct {
		positions [4][3]float32
		normal    [3]float32
		tangent   [4]float32
	}

	faces := []faceData{
		// +X
		{positions: [4][3]float32{{0.5, -0.5, -0.5}, {0.5, 0.5, -0.5}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}}, normal: [3]float32{1, 0, 0}, tangent: [4]float32{0, 0, 1, 1}},
		// -X
		{positions: [4][3]float32{{-0.5, -0.5, 0.5}, {-0.5, 0.5, 0.5}, {-0.5, 0.5, -0.5}, {-0.5, -0.5, -0.5}}, normal: [3]float32{-1, 0, 0}, tangent: [4]float32{0, 0, -1, 1}},
		// +Y
		{positions: [4][3]float32{{-0.5, 0.5, -0.5}, {-0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, -0.5}}, normal: [3]float32{0, 1, 0}, tangent: [4]float32{1, 0, 0, 1}},
		// -Y
		{positions: [4][3]float32{{-0.5, -0.5, 0.5}, {-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5}, {0.5, -0.5, 0.5}}, normal: [3]float32{0, -1, 0}, tangent: [4]float32{1, 0, 0, 1}},
		// +Z
		{positions: [4][3]float32{{-0.5, -0.5, 0.5}, {0.5, -0.5, 0.5}, {0.5, 0.5, 0.5}, {-0.5, 0.5, 0.5}}, normal: [3]float32{0, 0, 1}, tangent: [4]float32{1, 0, 0, 1}},
		// -Z
		{positions: [4][3]float32{{0.5, -0.5, -0.5}, {-0.5, -0.5, -0.5}, {-0.5, 0.5, -0.5}, {0.5, 0.5, -0.5}}, normal: [3]float32{0, 0, -1}, tangent: [4]float32{-1, 0, 0, 1}},
	}

	vertices := make([]model.GPUVertex, 0, 24)
	for fi, face := range faces {
		for _, pos := range face.positions {
			vertices = append(vertices, model.GPUVertex{
				Position: pos,
				Normal:   face.normal,
				TexCoord: [2]float32{0, 0},
				Color:    faceColors[fi],
				Tangent:  face.tangent,
			})
		}
	}

	indices := make([]uint32, 0, 36)
	for fi := range 6 {
		base := uint32(fi * 4)
		indices = append(indices,
			base+0, base+1, base+2,
			base+0, base+2, base+3,
		)
	}

	vertexBytes := common.SliceToBytes(vertices)
	indexBytes := make([]byte, len(indices)*4)
	for i, idx := range indices {
		binary.LittleEndian.PutUint32(indexBytes[i*4:], idx)
	}

	meshProvider := bgp.NewBindGroupProvider("cube_lit_mesh")

	mat := material.NewMaterial(
		material.WithName("cube_lit_rainbow"),
		material.WithBaseColor([4]float32{1, 1, 1, 1}),
		material.WithPipelineKey("CubeLitModel"),
	)

	return model.NewModel(
		model.WithName("CubeLitModel"),
		model.WithMeshProvider(meshProvider),
		model.WithVertexData(vertexBytes),
		model.WithIndexData(indexBytes),
		model.WithIndexCount(36),
		model.WithBoundingRadius(0.87),
		model.WithRenderMaterials(mat),
	)
}

// setupLitCubeBenchInput wires camera controls and the automatic sun orbit for
// the lit cube benchmark. The sun orbits continuously to simulate a day/night
// cycle: overhead → horizon → underneath → back overhead.
func setupLitCubeBenchInput(eng engine.Engine, cam camera.Camera, sun light.Light, sunAngle *float64) {
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

// buildLitCubeGroundPlane creates a large flat slab (6-face box with negligible
// thickness) for shadow-receiving. Normals point outward so the lit fragment
// shader lights the top face correctly.
//
// Parameters:
//   - half: half-extent of the slab in X and Z
//
// Returns:
//   - []model.GPUVertex: 24 vertices forming the slab (4 per face × 6 faces)
//   - []uint32: 36 indices forming 12 triangles
func buildLitCubeGroundPlane(half float32) ([]model.GPUVertex, []uint32) {
	color := [4]float32{0.35, 0.35, 0.35, 1.0}
	thickness := float32(0.5)

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
