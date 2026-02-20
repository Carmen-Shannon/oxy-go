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
	"github.com/Carmen-Shannon/oxy-go/engine/loader"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	bgp "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

// ── Benchmark Configuration ────────────────────────────────────────
const (
	// foxInitialCount is the number of foxes in the first ramp step.
	foxInitialCount = 10
	// foxRampInterval is how long each ramp step runs before ramping.
	foxRampInterval = 5 * time.Second
	// foxFPSThreshold is the FPS floor — the benchmark stops ramping when
	// the average FPS over the ramp interval falls below this value.
	foxFPSThreshold = 30.0
	// foxRampFactor is the multiplier applied each step (1.5 = +50%).
	foxRampFactor = 1.5
	// foxMaxStep caps the absolute number of foxes added per ramp step
	// to prevent massive spawn stalls at high counts.
	foxMaxStep = 1500
	// foxSpacing determines how far apart foxes are placed in the XZ grid.
	foxSpacing = 120.0
	// foxMaxSide is the maximum number of foxes per side on the XZ plane
	// before new objects start stacking upward in layers.
	foxMaxSide = 30
	// foxLayerSpacing is the vertical distance between stacked layers.
	foxLayerSpacing = 150.0
)

func main() {
	// ── Engine + Window ─────────────────────────────────────────────
	eng := engine.NewEngine(
		engine.WithProfiling(true),
		engine.WithTickRate(60),
		engine.WithWindow(window.NewWindow(
			window.WithTitle("Oxy Benchmark — Many Foxes"),
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
		camera.WithFov(float32(45.0*math.Pi/180.0)),
		camera.WithAspect(float32(eng.Window().Width())/float32(eng.Window().Height())),
		camera.WithNear(0.1),
		camera.WithFar(100000),
		camera.WithController(camera.NewCameraController(
			camera.WithRadius(500),
			camera.WithTarget(0, 40, 0),
			camera.WithElevation(0.5),
			camera.WithAzimuth(0.3),
			camera.WithRadiusBounds(1, 200000),
			camera.WithZoomSpeed(50.0),
			camera.WithMouseSensitivity(0.002), camera.WithPanSpeed(50.0))),
	)

	// ── Shaders ─────────────────────────────────────────────────────
	computeShader := shader.NewShader("skeletal_compute", shader.ShaderTypeCompute, "examples/assets/shaders/skeletal-compute.wgsl")
	vertexShader := shader.NewShader("skinned_vert", shader.ShaderTypeVertex, "examples/assets/shaders/skinned-vert.wgsl")
	fragmentShader := shader.NewShader("textured_frag", shader.ShaderTypeFragment, "examples/assets/shaders/textured-frag.wgsl")

	// ── Scene ───────────────────────────────────────────────────────
	sc := scene.NewScene("Many Foxes Bench", cam, r, vertexShader,
		scene.WithActive(true),
	)

	// ── Load Fox Model ──────────────────────────────────────────────
	ldr := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))
	foxModel, err := ldr.Load("examples/assets/models/Fox.glb", fragmentShader)
	if err != nil {
		log.Fatalf("Failed to load Fox model: %v", err)
	}

	// The textured-frag shader has @group(3) with an effect_tint uniform.
	// DrawCalls requires a valid EffectProvider for that group — without one the
	// bind-group resolution fails and the material is silently skipped.
	effectBGP := bgp.NewBindGroupProvider("fox_bench_effect_tint")
	if err := r.InitBindGroup(effectBGP, fragmentShader.BindGroupLayoutDescriptor(3), nil, nil); err != nil {
		log.Fatalf("Failed to init effect tint bind group: %v", err)
	}
	foxModel.SetEffectProvider(effectBGP)
	// Write initial tint (no effect: alpha = 0)
	noTint := [4]float32{0, 0, 0, 0}
	r.WriteBuffers([]bgp.BufferWrite{{
		Provider: effectBGP,
		Binding:  0,
		Offset:   0,
		Data:     common.SliceToBytes(noTint[:]),
	}})

	eng.AddScene(0, sc)

	// ── Spawn initial batch ─────────────────────────────────────────
	foxObjects := spawnFoxes(sc, foxModel, computeShader, vertexShader, fragmentShader, foxInitialCount, 0)
	currentCount := foxInitialCount

	// Start each fox playing a random animation at a random time offset
	animCount := foxModel.AnimationCount()
	randomizeFoxAnimations(foxObjects, animCount)

	// Override the bounding radius on the fox animator to prevent premature
	// frustum culling — animated poses extend beyond the rest-pose bounding sphere.
	if len(foxObjects) > 0 && foxObjects[0].Animator() != nil {
		foxObjects[0].Animator().SetBoundingRadius(200.0)
	}

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
	fmt.Println("║  Oxy Engine Benchmark — Many Foxes                  ║")
	fmt.Println("╠══════════════════════════════════════════════════════╣")
	fmt.Printf("║  Initial count: %d foxes                            ║\n", foxInitialCount)
	fmt.Printf("║  Ramp interval: %v                              ║\n", foxRampInterval)
	fmt.Printf("║  FPS threshold: %.0f                                ║\n", foxFPSThreshold)
	fmt.Println("║  Animations:                                        ║")
	for i, name := range animNames {
		fmt.Printf("║    %d = %-46s║\n", i, name)
	}
	fmt.Println("║  Camera: WASD=Pan  Q/E=Up/Down  Scroll=Zoom        ║")
	fmt.Println("║          Middle-mouse drag=Orbit                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════╝")
	log.Printf("[Bench] Starting with %d foxes (%d animations, %d bones)",
		currentCount, animCount, len(foxModel.Skeleton().Bones))

	// ── Per-frame render callback for FPS sampling ──────────────────
	eng.SetRenderCallback(func(_ float32) {
		frameCount++

		if stopped {
			return
		}

		elapsed := time.Since(rampStart)
		if elapsed >= foxRampInterval {
			fps := float64(frameCount) / elapsed.Seconds()
			log.Printf("[Bench] %d foxes → %.1f FPS (%.2f ms/frame)", currentCount, fps, 1000.0/fps)

			if fps > bestFPS {
				bestFPS = fps
				bestCount = currentCount
			}

			if fps < foxFPSThreshold {
				log.Printf("[Bench] ════════════════════════════════════════════")
				log.Printf("[Bench] RESULT: Below %.0f FPS threshold at %d foxes (%.1f FPS)", foxFPSThreshold, currentCount, fps)
				log.Printf("[Bench] Best sustained: %d foxes @ %.1f FPS", bestCount, bestFPS)
				log.Printf("[Bench] ════════════════════════════════════════════")
				stopped = true
				rampStart = time.Now()
				frameCount = 0
				return
			}

			// Compute next count: multiply by ramp factor, cap the delta
			nextCount := int(math.Ceil(float64(currentCount) * foxRampFactor))
			delta := nextCount - currentCount
			if delta > foxMaxStep {
				delta = foxMaxStep
				nextCount = currentCount + delta
			}

			spawnStart := time.Now()
			newFoxes := spawnFoxes(sc, foxModel, computeShader, vertexShader, fragmentShader, delta, currentCount)
			randomizeFoxAnimations(newFoxes, animCount)
			spawnDuration := time.Since(spawnStart)
			foxObjects = append(foxObjects, newFoxes...)
			currentCount = nextCount

			// Only pull camera back if the grid has grown beyond current view.
			effectiveSide := foxMaxSide
			if currentCount < foxMaxSide*foxMaxSide {
				effectiveSide = int(math.Ceil(math.Sqrt(float64(currentCount))))
			}
			layers := currentCount/(foxMaxSide*foxMaxSide) + 1
			gridW := float32(effectiveSide) * foxSpacing
			gridH := float32(layers) * foxLayerSpacing
			diag := float32(math.Sqrt(float64(gridW*gridW + gridH*gridH)))
			minRadius := diag * 0.6
			if minRadius > cam.Controller().Radius() {
				cam.Controller().SetRadius(minRadius)
			}

			log.Printf("[Bench] Ramping → %d foxes (+%d, spawn took %v)", currentCount, delta, spawnDuration.Round(time.Millisecond))
			// Reset timer AFTER spawning so spawn cost doesn't eat into measurement
			rampStart = time.Now()
			frameCount = 0
		}
	})

	// ── Input ───────────────────────────────────────────────────────
	setupFoxBenchInput(eng, cam)

	log.Println("[Bench] Starting Oxy Engine — Many Foxes Benchmark")
	eng.Run()
}

// spawnFoxes adds `count` fox GameObjects into the scene using a stable grid layout.
// Objects fill an XZ grid (up to foxMaxSide per axis) and once a layer is full,
// new objects stack upward on subsequent layers. Returns the created GameObjects
// so their animations can be randomized.
func spawnFoxes(
	sc scene.Scene,
	foxModel model.Model,
	computeShader, vertexShader, fragmentShader shader.Shader,
	count, startIdx int,
) []game_object.GameObject {
	objects := make([]game_object.GameObject, 0, count)

	for i := 0; i < count; i++ {
		idx := startIdx + i
		col := idx % foxMaxSide
		row := (idx / foxMaxSide) % foxMaxSide
		layer := idx / (foxMaxSide * foxMaxSide)

		cx := (float32(col) - float32(foxMaxSide-1)/2.0) * foxSpacing
		cz := (float32(row) - float32(foxMaxSide-1)/2.0) * foxSpacing
		cy := float32(layer) * foxLayerSpacing

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

// randomizeFoxAnimations starts each fox playing a random animation clip with
// a random time offset so they're not all perfectly in sync.
func randomizeFoxAnimations(foxes []game_object.GameObject, animCount int) {
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
		// Randomize starting time so foxes don't all animate in lockstep
		anim.SetAnimationTime(uint32(fox.AnimatorInstanceID()), rand.Float32()*5.0)
	}
}

// setupFoxBenchInput wires camera controls for the benchmark viewer.
func setupFoxBenchInput(eng engine.Engine, cam camera.Camera) {
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
