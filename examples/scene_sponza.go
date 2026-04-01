//go:build ignore

package main

import (
	"fmt"
	"log"
	"math"
	"os"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine"
	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/loader"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

func main() {
	// ── Engine + Window ─────────────────────────────────────────────────
	eng := engine.NewEngine(
		engine.WithProfiling(true),
		engine.WithTickRate(60),
		engine.WithWindow(window.NewWindow(
			window.WithTitle("Oxy Engine - Sponza (Auto-Exposure / CSM+PCF / SSAO)"),
			window.WithWidth(1920),
			window.WithHeight(1080),
		)),
	)

	// ── Renderer ────────────────────────────────────────────────────────
	r := renderer.NewRenderer(
		renderer.BackendTypeWGPU,
		eng.Window(),
		renderer.WithPresentMode(renderer.PresentModeUncapped),
		renderer.WithMSAA(renderer.MSAA4x),
		renderer.WithGPUSerializedProfiling(false),
	)

	// ── Camera ──────────────────────────────────────────────────────────
	cam := camera.NewCamera(
		camera.WithFov(float32(80.0*math.Pi/180.0)),
		camera.WithAspect(float32(eng.Window().Width())/float32(eng.Window().Height())),
		camera.WithNear(0.05),
		camera.WithFar(1000),
		camera.WithController(camera.NewCameraController(
			camera.WithRadius(20),
			camera.WithTarget(0, 2, 0),
			camera.WithElevation(0.3),
			camera.WithAzimuth(0.0),
			camera.WithRadiusBounds(1, 40),
			camera.WithZoomSpeed(1.0),
			camera.WithMouseSensitivity(0.002),
		)),
	)

	// ── Scene ───────────────────────────────────────────────────────────
	sc := scene.NewScene("Sponza Scene", cam, r,
		scene.WithActive(true),
		scene.WithScreenSize(eng.Window().Width(), eng.Window().Height()),
		scene.WithLighting(light.NewLightingHandler(
			light.WithShadowHandler(light.NewShadowHandler(
				light.WithPCFRadius(1.0),
				light.WithShadowNearFar(0.05, 1000),
				light.WithShadowNormalBiasScale(1.0),
				light.WithShadowMapResolution(1024),
				light.WithShadowInnerRadius(5),
			)),
			light.WithGBufferHandler(light.NewGBufferHandler()),
			light.WithSSAOHandler(light.NewSSAOHandler(
				light.WithSSAOSampleCount(16),
				light.WithSSAOScreenRadius(24.0),
				light.WithSSAOBias(0.025),
				light.WithSSAOPower(2.0),
				light.WithSSAOBlurRadius(2),
				light.WithSSAOHalfResolution(true),
			)),
			light.WithCompositionHandler(light.NewCompositionHandler(
				light.WithToneMappingEnabled(true),
				light.WithExposure(1.0),
				light.WithAutoExposure(true),
				light.WithAdaptSpeed(8.0),
				light.WithMinExposure(0.001),
				light.WithMaxExposure(2.0),
			)),
			light.WithSSRHandler(light.NewSSRHandler(
				light.WithSSRMaxSteps(32),
				light.WithSSRMaxDistance(10.0),
				light.WithSSRThickness(2.0),
				light.WithSSRStride(1.5),
				light.WithSSRRoughnessCutoff(0.5),
			)),
		)),
	)

	// ── Lights ──────────────────────────────────────────────────────────
	// Single directional sun light angled from upper-left
	sun := light.NewLight(light.LightTypeDirectional,
		light.WithDirection(-0.3, -1.0, -0.5),
		light.WithColor(1.0, 0.95, 0.85),
		light.WithIntensity(1.5),
		light.WithCastsShadows(true),
		light.WithShadowBias(0.04),
		light.WithEnabled(true),
	)
	sc.AddLight(sun)

	// Very dark ambient so indoor shadowed corridors read dramatically
	sc.SetAmbientColor([3]float32{0.04, 0.04, 0.06})

	var pointLights []light.Light

	// Upper corridor fire lights (ceiling-mounted, warm amber/orange)
	for _, cfg := range [][6]float32{
		{-5, 7, -0.5, 1.0, 0.6, 0.2},
		{-5, 7, 0.5, 0.95, 0.55, 0.18},
		{-2, 7, -0.5, 1.0, 0.55, 0.15},
		{-2, 7, 0.5, 0.95, 0.5, 0.13},
		{1, 7, -0.5, 0.95, 0.5, 0.2},
		{1, 7, 0.5, 1.0, 0.55, 0.18},
		{4, 7, -0.5, 1.0, 0.65, 0.25},
		{4, 7, 0.5, 0.95, 0.6, 0.22},
	} {
		l := light.NewLight(light.LightTypePoint,
			light.WithPosition(cfg[0], cfg[1], cfg[2]),
			light.WithColor(cfg[3], cfg[4], cfg[5]),
			light.WithIntensity(0.5),
			light.WithRange(10.0),
			light.WithCastsShadows(true),
			light.WithEnabled(true),
		)
		sc.AddLight(l)
		pointLights = append(pointLights, l)
	}

	// Corner floor lights (atrium corners, deeper orange/red)
	for _, cfg := range [][6]float32{
		{9, 1, -3.5, 1.0, 0.45, 0.1},
		{9, 1, 3.5, 0.95, 0.5, 0.15},
		{-9, 1, 3.5, 1.0, 0.45, 0.1},
		{-9, 1, -3.5, 0.95, 0.5, 0.15},
	} {
		l := light.NewLight(light.LightTypePoint,
			light.WithPosition(cfg[0], cfg[1], cfg[2]),
			light.WithColor(cfg[3], cfg[4], cfg[5]),
			light.WithIntensity(0.3),
			light.WithRange(10.0),
			light.WithCastsShadows(true),
			light.WithEnabled(true),
		)
		sc.AddLight(l)
		pointLights = append(pointLights, l)
	}

	// Inner floor accent lights (warmest fire red-orange)
	for _, cfg := range [][6]float32{
		{4, 1, -1.5, 1.0, 0.4, 0.1},
		{4, 1, 1.5, 0.95, 0.45, 0.12},
		{-5, 1, 1.5, 1.0, 0.4, 0.1},
		{-5, 1, -1.5, 0.95, 0.45, 0.12},
	} {
		l := light.NewLight(light.LightTypePoint,
			light.WithPosition(cfg[0], cfg[1], cfg[2]),
			light.WithColor(cfg[3], cfg[4], cfg[5]),
			light.WithIntensity(0.1),
			light.WithRange(5.0),
			light.WithCastsShadows(true),
			light.WithEnabled(true),
		)
		sc.AddLight(l)
		pointLights = append(pointLights, l)
	}

	// ── Load Sponza Model ───────────────────────────────────────────────
	ldr := loader.NewLoader(loader.BackendTypeGLTF)
	sponzaModels, err := ldr.LoadAll("examples/assets/models/sponza/Sponza.gltf")
	if err != nil {
		log.Fatalf("Failed to load Sponza model: %v", err)
	}

	for _, m := range sponzaModels {
		obj := game_object.NewGameObject(
			game_object.WithModel(m),
			game_object.WithPosition(0, 0, 0),
			game_object.WithScale(1, 1, 1),
		)
		_ = sc.AddGameObject(obj)
	}

	eng.AddScene(0, sc)

	// ── Input Handling ──────────────────────────────────────────────────
	setupSponzaInput(eng, cam, sun, pointLights)

	fmt.Println("╔══════════════════════════════════════════════════════╗")
	fmt.Println("║  Oxy Engine - Sponza (Auto-Exposure / CSM+PCF / SSAO)║")
	fmt.Println("╠══════════════════════════════════════════════════════╣")
	fmt.Println("║  Camera: WASD=Pan  Q/E=Up/Down  Scroll=Zoom          ║")
	fmt.Println("║          Middle-mouse drag=Orbit                      ║")
	fmt.Println("║  L:      Toggle sun (directional light)               ║")
	fmt.Println("║  F:      Toggle point lights                          ║")
	fmt.Println("║  Shadow: CSM + PCF (dual-cascade sphere shadows)      ║")
	fmt.Println("║  SSAO:   Half-resolution AO, 16 samples               ║")
	fmt.Println("║  Expo:   Auto-exposure (bright atrium ↔ dark interior)║")
	fmt.Println("╚══════════════════════════════════════════════════════╝")

	log.Println("Starting Oxy Engine - Sponza Scene")
	eng.Run()
}

// setupSponzaInput wires camera controls (WASD/QE planar movement, middle-mouse orbit,
// scroll zoom) and sun toggling (L key) and point light toggling (F key) for the Sponza scene.
//
// Parameters:
//   - eng: the engine instance providing window callbacks and tick
//   - cam: the camera to control
//   - sun: the directional light to toggle with the L key
//   - pointLights: the slice of all point lights to toggle with the F key
func setupSponzaInput(eng engine.Engine, cam camera.Camera, sun light.Light, pointLights []light.Light) {
	keyState := make(map[uint32]bool)
	pointsOn := true

	eng.Window().SetKeyDownCallback(func(keyCode uint32) {
		keyState[keyCode] = true

		// L toggles the directional sun light
		if keyCode == common.KeyL {
			sun.SetEnabled(!sun.Enabled())
			if sun.Enabled() {
				fmt.Println("[Light] Sun ON")
			} else {
				fmt.Println("[Light] Sun OFF")
			}
		}

		// F toggles all point lights
		if keyCode == common.KeyF {
			pointsOn = !pointsOn
			for _, pl := range pointLights {
				pl.SetEnabled(pointsOn)
			}
			if pointsOn {
				fmt.Println("[Light] Point lights ON")
			} else {
				fmt.Println("[Light] Point lights OFF")
			}
		}

		// Space logs the camera position to a file
		if keyCode == common.KeySpace {
			x, y, z := cam.Controller().Position()
			line := fmt.Sprintf("%.3f, %.3f, %.3f\n", x, y, z)

			f, err := os.OpenFile("examples/assets/sponza_positions.txt", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			if err != nil {
				fmt.Printf("Failed to open position log: %v\n", err)
			} else {
				_, _ = f.WriteString(line)
				f.Close()
				fmt.Printf("Captured position: %.3f, %.3f, %.3f\n", x, y, z)
			}
		}
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

		ctrl := cam.Controller()
		px, py, pz := ctrl.Position()
		r := ctrl.Radius()

		ctrl.SetAzimuth(ctrl.Azimuth() + dx*ctrl.MouseSensitivity())
		ctrl.SetElevation(ctrl.Elevation() - dy*ctrl.MouseSensitivity())

		// Use post-clamp values so the target stays consistent with the actual stored angles
		elev := ctrl.Elevation()
		azim := ctrl.Azimuth()
		cosElev := float32(math.Cos(float64(elev)))
		sinElev := float32(math.Sin(float64(elev)))
		cosAzim := float32(math.Cos(float64(azim)))
		sinAzim := float32(math.Sin(float64(azim)))
		ctrl.SetTarget(
			px-r*cosElev*sinAzim,
			py-r*sinElev,
			pz-r*cosElev*cosAzim,
		)

		lastX, lastY = x, y
	})

	eng.Window().SetScrollCallback(func(delta float32) {
		cam.Controller().Zoom(delta)
	})

	eng.SetTickCallback(func(_ float32) {
		if keyState[common.KeyW] {
			cam.Controller().PanForward(0.05)
		}
		if keyState[common.KeyS] {
			cam.Controller().PanForward(-0.05)
		}
		if keyState[common.KeyA] {
			cam.Controller().PanRight(-0.05)
		}
		if keyState[common.KeyD] {
			cam.Controller().PanRight(0.05)
		}
		if keyState[common.KeyQ] {
			cam.Controller().PanUp(0.05)
		}
		if keyState[common.KeyE] {
			cam.Controller().PanUp(-0.05)
		}
	})
}
