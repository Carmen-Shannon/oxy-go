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
	// benchInitialCount is the number of cubes in the first ramp step.
	benchInitialCount = 1000
	// benchRampInterval is how long each ramp step runs before increasing.
	benchRampInterval = 5 * time.Second
	// benchFPSThreshold is the FPS floor — the benchmark stops ramping when
	// the average FPS over the ramp interval falls below this value.
	benchFPSThreshold = 30.0
	// benchCubeSpacing determines how far apart cubes are placed in the grid.
	benchCubeSpacing = 3.0
	// benchRampFactor is the multiplier applied each step (1.5 = +50%).
	benchRampFactor = 1.5
	// benchMaxStep caps the absolute number of cubes added per ramp step
	// to prevent long spawn freezes at high counts.
	benchMaxStep = 1_000_000
	// benchMaxSide is the maximum number of cubes per side on the XZ plane
	// before new objects start stacking upward in layers.
	benchMaxSide = 200
)

func main() {
	// ── Engine + Window ─────────────────────────────────────────────
	eng := engine.NewEngine(
		engine.WithProfiling(true),
		engine.WithTickRate(60),
		engine.WithWindow(window.NewWindow(
			window.WithTitle("Oxy Benchmark — Many Cubes"),
			window.WithWidth(1920),
			window.WithHeight(1080),
		)),
	)

	// ── Renderer (uncapped FPS) ─────────────────────────────────────
	// Toggle WithForceSoftwareRenderer(true) to benchmark CPU-only rendering
	// via a software Vulkan ICD (requires SwiftShader or lavapipe installed).
	r := renderer.NewRenderer(
		renderer.BackendTypeWGPU,
		eng.Window(),
		renderer.WithPresentMode(renderer.PresentModeUncapped),
		// renderer.WithForceSoftwareRenderer(true),
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
	computeShader := shader.NewShader("simple_compute", shader.ShaderTypeCompute, "examples/assets/shaders/simple-compute.wgsl")
	vertexShader := shader.NewShader("simple_vert", shader.ShaderTypeVertex, "examples/assets/shaders/simple-vert.wgsl")
	fragmentShader := shader.NewShader("rainbow_frag", shader.ShaderTypeFragment, "examples/assets/shaders/rainbow-frag.wgsl")

	// ── Build Cube Model ────────────────────────────────────────────
	cubeModel := buildRainbowCubeModel()

	// ── Scene ───────────────────────────────────────────────────────
	sc := scene.NewScene("Many Cubes Bench", cam, r, vertexShader,
		scene.WithActive(true),
	)

	eng.AddScene(0, sc)

	// ── Spawn initial batch ─────────────────────────────────────────
	spawnCubes(sc, cubeModel, computeShader, vertexShader, fragmentShader, benchInitialCount)
	currentCount := benchInitialCount

	// ── Benchmark tracking ──────────────────────────────────────────
	var (
		rampStart  = time.Now()
		frameCount int
		bestCount  int
		bestFPS    float64
		stopped    bool // true once we've dipped below the FPS threshold
	)

	fmt.Println("╔══════════════════════════════════════════════════════╗")
	fmt.Println("║  Oxy Engine Benchmark — Many Cubes                  ║")
	fmt.Println("╠══════════════════════════════════════════════════════╣")
	fmt.Printf("║  Initial count: %d cubes                          ║\n", benchInitialCount)
	fmt.Printf("║  Ramp interval: %v                              ║\n", benchRampInterval)
	fmt.Printf("║  FPS threshold: %.0f                                ║\n", benchFPSThreshold)
	fmt.Println("║  Camera: WASD=Pan  Q/E=Up/Down  Scroll=Zoom        ║")
	fmt.Println("║          Middle-mouse drag=Orbit                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════╝")
	log.Printf("[Bench] Starting with %d cubes", currentCount)

	// ── Per-frame render callback for FPS sampling ──────────────────
	eng.SetRenderCallback(func(_ float32) {
		frameCount++

		if stopped {
			return
		}

		elapsed := time.Since(rampStart)
		if elapsed >= benchRampInterval {
			fps := float64(frameCount) / elapsed.Seconds()
			log.Printf("[Bench] %d cubes → %.1f FPS (%.2f ms/frame)", currentCount, fps, 1000.0/fps)

			if fps > bestFPS {
				bestFPS = fps
				bestCount = currentCount
			}

			if fps < benchFPSThreshold {
				log.Printf("[Bench] ════════════════════════════════════════════")
				log.Printf("[Bench] RESULT: Below %.0f FPS threshold at %d cubes (%.1f FPS)", benchFPSThreshold, currentCount, fps)
				log.Printf("[Bench] Best sustained: %d cubes @ %.1f FPS", bestCount, bestFPS)
				log.Printf("[Bench] ════════════════════════════════════════════")
				stopped = true
				return
			}

			// Compute next count: multiply by ramp factor, cap the delta
			nextCount := int(math.Ceil(float64(currentCount) * benchRampFactor))
			delta := nextCount - currentCount
			if delta > benchMaxStep {
				delta = benchMaxStep
				nextCount = currentCount + delta
			}

			spawnStart := time.Now()
			spawnCubes(sc, cubeModel, computeShader, vertexShader, fragmentShader, delta)
			spawnDuration := time.Since(spawnStart)
			currentCount = nextCount

			// Only pull camera back if the grid has grown beyond current view.
			effectiveSide := benchMaxSide
			if currentCount < benchMaxSide*benchMaxSide {
				effectiveSide = int(math.Ceil(math.Sqrt(float64(currentCount))))
			}
			layers := currentCount/(benchMaxSide*benchMaxSide) + 1
			gridW := float32(effectiveSide) * benchCubeSpacing
			gridH := float32(layers) * benchCubeSpacing
			diag := float32(math.Sqrt(float64(gridW*gridW + gridH*gridH)))
			minRadius := diag * 0.7
			if minRadius > cam.Controller().Radius() {
				cam.Controller().SetRadius(minRadius)
			}

			log.Printf("[Bench] Ramping → %d cubes (+%d, spawn took %v)", currentCount, delta, spawnDuration.Round(time.Millisecond))
			// Reset timer AFTER spawning so spawn cost doesn't eat into measurement
			rampStart = time.Now()
			frameCount = 0
		}
	})

	// ── Input ───────────────────────────────────────────────────────
	setupBenchInput(eng, cam)

	log.Println("[Bench] Starting Oxy Engine — Many Cubes Benchmark")
	eng.Run()
}

// spawnCubes adds `count` ephemeral cube GameObjects into the scene using a stable
// grid layout. Objects fill an XZ grid (up to benchMaxSide per axis) and once a layer
// is full, new objects stack upward on subsequent layers.
// Each cube gets a random rotation speed for visual variety.
func spawnCubes(
	sc scene.Scene,
	cubeModel model.Model,
	computeShader, vertexShader, fragmentShader shader.Shader,
	count int,
) {
	existing := sc.CountEphemeral()

	for i := 0; i < count; i++ {
		idx := existing + i
		col := idx % benchMaxSide
		row := (idx / benchMaxSide) % benchMaxSide
		layer := idx / (benchMaxSide * benchMaxSide)

		cx := (float32(col) - float32(benchMaxSide-1)/2.0) * benchCubeSpacing
		cz := (float32(row) - float32(benchMaxSide-1)/2.0) * benchCubeSpacing
		cy := float32(layer) * benchCubeSpacing

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

// buildRainbowCubeModel creates a unit cube model with per-vertex rainbow colors.
// Each face has a distinct color for visual clarity.
func buildRainbowCubeModel() model.Model {
	// 6 faces × 4 vertices = 24 vertices
	// Each face has a unique color
	faceColors := [][4]float32{
		{1, 0, 0, 1}, // +X red
		{0, 1, 0, 1}, // -X green
		{0, 0, 1, 1}, // +Y blue
		{1, 1, 0, 1}, // -Y yellow
		{1, 0, 1, 1}, // +Z magenta
		{0, 1, 1, 1}, // -Z cyan
	}

	// Face definitions: 4 positions + normal per face
	type faceData struct {
		positions [4][3]float32
		normal    [3]float32
	}

	faces := []faceData{
		// +X
		{positions: [4][3]float32{{0.5, -0.5, -0.5}, {0.5, 0.5, -0.5}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}}, normal: [3]float32{1, 0, 0}},
		// -X
		{positions: [4][3]float32{{-0.5, -0.5, 0.5}, {-0.5, 0.5, 0.5}, {-0.5, 0.5, -0.5}, {-0.5, -0.5, -0.5}}, normal: [3]float32{-1, 0, 0}},
		// +Y
		{positions: [4][3]float32{{-0.5, 0.5, -0.5}, {-0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, -0.5}}, normal: [3]float32{0, 1, 0}},
		// -Y
		{positions: [4][3]float32{{-0.5, -0.5, 0.5}, {-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5}, {0.5, -0.5, 0.5}}, normal: [3]float32{0, -1, 0}},
		// +Z
		{positions: [4][3]float32{{-0.5, -0.5, 0.5}, {0.5, -0.5, 0.5}, {0.5, 0.5, 0.5}, {-0.5, 0.5, 0.5}}, normal: [3]float32{0, 0, 1}},
		// -Z
		{positions: [4][3]float32{{0.5, -0.5, -0.5}, {-0.5, -0.5, -0.5}, {-0.5, 0.5, -0.5}, {0.5, 0.5, -0.5}}, normal: [3]float32{0, 0, -1}},
	}

	// Build vertices (model.GPUVertex: 64 bytes)
	vertices := make([]model.GPUVertex, 0, 24)
	for fi, face := range faces {
		for _, pos := range face.positions {
			vertices = append(vertices, model.GPUVertex{
				Position: pos,
				Normal:   face.normal,
				TexCoord: [2]float32{0, 0},
				Color:    faceColors[fi],
			})
		}
	}

	// Build indices: 6 faces × 2 tris × 3 = 36 indices
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

	meshProvider := bgp.NewBindGroupProvider("cube_mesh")

	mat := material.NewMaterial(
		material.WithName("cube_rainbow"),
		material.WithPipelineKey("CubeModel"),
	)

	return model.NewModel(
		model.WithName("CubeModel"),
		model.WithMeshProvider(meshProvider),
		model.WithVertexData(vertexBytes),
		model.WithIndexData(indexBytes),
		model.WithIndexCount(36),
		model.WithBoundingRadius(0.87), // sqrt(0.5^2 + 0.5^2 + 0.5^2) ≈ 0.866
		model.WithRenderMaterials(mat),
	)
}

// setupBenchInput wires camera controls for the benchmark viewer.
func setupBenchInput(eng engine.Engine, cam camera.Camera) {
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

	eng.SetTickCallback(func(_ float32) {
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
