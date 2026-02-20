//go:build ignore

package main

import (
	"fmt"
	"log"
	"math"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine"
	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

func main() {
	// ── Engine + Window ─────────────────────────────────────────────────
	eng := engine.NewEngine(
		engine.WithProfiling(true),
		engine.WithTickRate(60),
		engine.WithWindow(window.NewWindow(
			window.WithTitle("Oxy Engine - Scene Test"),
			window.WithWidth(1920),
			window.WithHeight(1080),
		)),
	)

	// ── Renderer ────────────────────────────────────────────────────────
	r := renderer.NewRenderer(
		renderer.BackendTypeWGPU,
		eng.Window(),
		renderer.WithPresentMode(renderer.PresentModeUncapped),
	)

	// ── Camera ──────────────────────────────────────────────────────────
	cam := camera.NewCamera(
		camera.WithFov(float32(45.0*math.Pi/180.0)),
		camera.WithAspect(float32(eng.Window().Width())/float32(eng.Window().Height())),
		camera.WithNear(0.01),
		camera.WithFar(10000),
		camera.WithController(camera.NewCameraController(
			camera.WithRadius(100),
			camera.WithTarget(0, 0, 0),
			camera.WithElevation(0.3),
			camera.WithAzimuth(0.5),
			camera.WithPanSpeed(1.0),
			camera.WithRadiusBounds(1, 20000),
			camera.WithZoomSpeed(16.0),
			camera.WithMouseSensitivity(0.002),
		)),
	)

	// ── Shaders ─────────────────────────────────────────────────────────
	computeShader := shader.NewShader("simple_compute", shader.ShaderTypeCompute, "examples/assets/shaders/simple-compute.wgsl")
	vertexShader := shader.NewShader("simple_vert", shader.ShaderTypeVertex, "examples/assets/shaders/simple-vert.wgsl")
	fragmentShader := shader.NewShader("rainbow_frag", shader.ShaderTypeFragment, "examples/assets/shaders/rainbow-frag.wgsl")

	// ── Scene ───────────────────────────────────────────────────────────
	// NewScene inits the camera BGP from the vertex shader's camera bind group layout.
	sc := scene.NewScene("Scene Test", cam, r, vertexShader,
		scene.WithActive(true),
	)

	// ── Rainbow Cube ────────────────────────────────────────────────────
	cubeVerts, cubeIdx := buildCube()
	cube := game_object.NewGameObject(
		game_object.WithModel(model.NewModel(
			model.WithName("rainbow_cube"),
			model.WithBoundingRadius(1.0),
			model.WithVertexData(common.SliceToBytes(cubeVerts)),
			model.WithIndexData(common.SliceToBytes(cubeIdx)),
			model.WithIndexCount(len(cubeIdx)),
			model.WithMeshProvider(bind_group_provider.NewBindGroupProvider(
				"cube_mesh",
			)),
			model.WithRenderMaterials(material.NewMaterial(
				material.WithName("rainbow_cube"),
				material.WithPipelineKey("rainbow_cube"),
			)),
		)),
		game_object.WithPosition(0, 0, 0),
		game_object.WithScale(10, 10, 10),
		game_object.WithRotationSpeed(2.0, 1.0, 0),
		game_object.WithEphemeral(true),
	)

	_ = sc.Add(cube, computeShader, vertexShader, fragmentShader)

	eng.AddScene(0, sc)

	// ── Input Handling ──────────────────────────────────────────────────
	setupInput(eng, cam)

	fmt.Println("╔══════════════════════════════════════════════════════╗")
	fmt.Println("║  Oxy Engine - Scene Test                            ║")
	fmt.Println("╠══════════════════════════════════════════════════════╣")
	fmt.Println("║  Camera: WASD=Pan  Q/E=Up/Down  Scroll=Zoom        ║")
	fmt.Println("║          Middle-mouse drag=Orbit                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════╝")

	log.Println("Starting Oxy Engine - Scene Test")
	eng.Run()
}

// setupInput wires camera controls: WASD/QE planar movement, middle-mouse orbit,
// and scroll zoom.
//
// Parameters:
//   - eng: the engine instance providing window callbacks and tick
//   - cam: the camera to control
func setupInput(eng engine.Engine, cam camera.Camera) {
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

// buildCube returns 8 vertices with distinct rainbow colors and 36 indices
// forming 12 triangles (2 per face). All outward faces wind counter-clockwise.
//
// Returns:
//   - []model.GPUVertex: the cube vertices with rainbow colors
//   - []uint32: the triangle indices
func buildCube() ([]model.GPUVertex, []uint32) {
	pos := [8][3]float32{
		{-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5},
		{0.5, 0.5, -0.5}, {-0.5, 0.5, -0.5},
		{-0.5, -0.5, 0.5}, {0.5, -0.5, 0.5},
		{0.5, 0.5, 0.5}, {-0.5, 0.5, 0.5},
	}
	col := [8][4]float32{
		{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}, {1, 1, 0, 1},
		{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 1, 1}, {1, 0.5, 0, 1},
	}

	vertices := make([]model.GPUVertex, 8)
	for i := 0; i < 8; i++ {
		vertices[i] = model.GPUVertex{Position: pos[i], Color: col[i]}
	}

	indices := []uint32{
		4, 5, 6, 4, 6, 7, // Front  (+Z)
		1, 0, 3, 1, 3, 2, // Back   (-Z)
		5, 1, 2, 5, 2, 6, // Right  (+X)
		0, 4, 7, 0, 7, 3, // Left   (-X)
		3, 7, 6, 3, 6, 2, // Top    (+Y)
		0, 1, 5, 0, 5, 4, // Bottom (-Y)
	}

	return vertices, indices
}
