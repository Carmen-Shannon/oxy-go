//go:build ignore

package main

import (
	"fmt"
	"log"
	"math"
	"sync"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine"
	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/loader"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
	"github.com/cogentcore/webgpu/wgpu"
)

func main() {
	// ── Engine + Window ─────────────────────────────────────────────────
	eng := engine.NewEngine(
		engine.WithProfiling(true),
		engine.WithTickRate(60),
		engine.WithWindow(window.NewWindow(
			window.WithTitle("Oxy Engine - Lit Scene (Forward+ / CSM+PCF / SSAO / SSR / HDR)"),
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
		camera.WithNear(0.1),
		camera.WithFar(20000),
		camera.WithController(camera.NewCameraController(
			camera.WithRadius(200),
			camera.WithTarget(0, 40, 0),
			camera.WithElevation(0.3),
			camera.WithAzimuth(0.5),
			camera.WithPanSpeed(1.0),
			camera.WithRadiusBounds(1, 20000),
			camera.WithZoomSpeed(16.0),
			camera.WithMouseSensitivity(0.002),
		)),
	)

	// ── Scene ───────────────────────────────────────────────────────────
	sc := scene.NewScene("Lit Scene", cam, r,
		scene.WithActive(true),
		scene.WithScreenSize(eng.Window().Width(), eng.Window().Height()),
		scene.WithLighting(light.NewLightingHandler(
			light.WithShadowHandler(light.NewShadowHandler(
				light.WithPCFRadius(2.0),
				light.WithShadowNearFar(0.1, 2000),
				light.WithShadowNormalBiasScale(2.0),
				light.WithShadowMapResolution(2048),
				light.WithShadowInnerRadius(1000),
			)),
			light.WithGBufferHandler(light.NewGBufferHandler()),
			light.WithSSAOHandler(light.NewSSAOHandler(
				light.WithSSAOSampleCount(16),
				light.WithSSAORadius(0.1),
				light.WithSSAOBias(0.025),
				light.WithSSAOPower(2.0),
				light.WithSSAOBlurRadius(2),
				light.WithSSAOHalfResolution(true),
			)),
			light.WithCompositionHandler(light.NewCompositionHandler(
				light.WithToneMappingEnabled(true),
				light.WithExposure(1.5),
			)),
			light.WithSSRHandler(light.NewSSRHandler(
				light.WithSSRMaxSteps(64),
				light.WithSSRMaxDistance(300.0),
				light.WithSSRThickness(2.0),
				light.WithSSRStride(1.5),
				light.WithSSRRoughnessCutoff(0.5),
			)),
		)),
	)

	// ── Lights ──────────────────────────────────────────────────────────
	// Directional sun light (shadow-casting)
	sun := light.NewLight(light.LightTypeDirectional,
		light.WithDirection(0, -1, 0),
		light.WithColor(1.0, 0.95, 0.85),
		light.WithIntensity(0.5),
		light.WithCastsShadows(true),
		light.WithShadowBias(0.0005),
		light.WithEnabled(false),
	)
	sc.AddLight(sun)

	// Blue point light (left of the model)
	bluePoint := light.NewLight(light.LightTypePoint,
		light.WithPosition(-50, 50, 30),
		light.WithColor(0.2, 0.4, 1.0),
		light.WithIntensity(1.0),
		light.WithRange(200),
		light.WithEnabled(true),
		light.WithCastsShadows(true),
		light.WithShadowBias(0.0005),
	)
	sc.AddLight(bluePoint)

	// Orange point light (right of the model)
	orangePoint := light.NewLight(light.LightTypePoint,
		light.WithPosition(50, 50, -30),
		light.WithColor(1.0, 0.5, 0.1),
		light.WithIntensity(1.0),
		light.WithRange(200),
		light.WithEnabled(true),
		light.WithCastsShadows(true),
		light.WithShadowBias(0.0005),
	)
	sc.AddLight(orangePoint)

	// Spot light (overhead, angled down)
	spot := light.NewLight(light.LightTypeSpot,
		light.WithPosition(0, 100, 80),
		light.WithDirection(0, -1, -0.5),
		light.WithColor(0.0, 1.0, 0.5),
		light.WithIntensity(1.0),
		light.WithRange(200),
		light.WithSpotCone(15, 15),
		light.WithEnabled(false),
		light.WithCastsShadows(true),
		light.WithShadowBias(0.0005),
	)
	sc.AddLight(spot)

	// Set ambient color (dim fill light so shadows aren't fully black)
	sc.SetAmbientColor([3]float32{0.08, 0.08, 0.12})

	// ── Load Fox Model ──────────────────────────────────────────────────
	ldr := loader.NewLoader(loader.BackendTypeGLTF)
	foxModel, err := ldr.Load("examples/assets/models/Fox.glb")
	if err != nil {
		log.Fatalf("Failed to load Fox model: %v", err)
	}

	fox := game_object.NewGameObject(
		game_object.WithModel(foxModel),
		game_object.WithPosition(0, 0, 0),
		game_object.WithScale(1, 1, 1),
	)

	_ = sc.AddGameObject(fox)

	// Start initial animation (first clip, looped)
	if foxModel.AnimationCount() > 0 {
		fox.Animator().PlayAnimation(0, 0, true)
	}

	// ── Reflective Floor ──────────────────────────────────────────────────
	// A reflective metallic floor beneath the fox. Roughness is low and
	// metallic is high so the surface picks up screen-space reflections
	// (SSR). The fox shadow from the directional sun also appears here.
	// Press V to cycle transparency.
	quadAlpha := float32(1.0)
	quadVerts, quadIdx := buildLitQuad(quadAlpha)
	quadObj := game_object.NewGameObject(
		game_object.WithModel(model.NewModel(
			model.WithName("reflective_floor"),
			model.WithBoundingRadius(100.0),
			model.WithVertexData(common.SliceToBytes(quadVerts)),
			model.WithIndexData(common.SliceToBytes(quadIdx)),
			model.WithIndexCount(len(quadIdx)),
			model.WithShadowCullMode(model.ShadowCullModeNone),
			model.WithMeshProvider(bind_group_provider.NewBindGroupProvider("quad_mesh")),
			model.WithRenderMaterials(material.NewMaterial(
				material.WithName("quad_material"),
				material.WithBaseColor([4]float32{0.7, 0.8, 0.95, quadAlpha}),
				material.WithRoughness(0.3),
				material.WithMetallic(0.0),
				material.WithPipelineKey("reflective_floor"),
			)),
			model.WithCastsShadows(true),
		)),
		game_object.WithPosition(0, -1, 0),
		game_object.WithScale(1, 1, 1),
		game_object.WithEphemeral(true),
	)

	_ = sc.AddGameObject(quadObj,
		pipeline.WithBlendEnabled(true),
		pipeline.WithBlendState(&wgpu.BlendState{
			Color: wgpu.BlendComponent{
				Operation: wgpu.BlendOperationAdd,
				SrcFactor: wgpu.BlendFactorSrcAlpha,
				DstFactor: wgpu.BlendFactorOneMinusSrcAlpha,
			},
			Alpha: wgpu.BlendComponent{
				Operation: wgpu.BlendOperationAdd,
				SrcFactor: wgpu.BlendFactorOne,
				DstFactor: wgpu.BlendFactorZero,
			},
		}),
	)

	// ── Sun Indicator ───────────────────────────────────────────────────
	// A small orange sphere that shows the shadow-map eye position for the
	// directional sun light. This makes it easy to see where the light is.
	sunSphereVerts, sunSphereIdx := buildSphere(5, 12, 16, [4]float32{1.0, 0.6, 0.1, 1.0})
	sunIndicator := game_object.NewGameObject(
		game_object.WithModel(model.NewModel(
			model.WithName("sun_indicator"),
			model.WithBoundingRadius(20000.0),
			model.WithVertexData(common.SliceToBytes(sunSphereVerts)),
			model.WithIndexData(common.SliceToBytes(sunSphereIdx)),
			model.WithIndexCount(len(sunSphereIdx)),
			model.WithMeshProvider(bind_group_provider.NewBindGroupProvider(
				"sun_sphere_mesh",
			)),
			model.WithRenderMaterials(material.NewMaterial(
				material.WithName("sun_sphere_material"),
				material.WithBaseColor([4]float32{1.0, 0.6, 0.1, 1.0}),
				material.WithPipelineKey("sun_indicator"),
			)),
			model.WithCastsShadows(false),
			model.WithShadowCullMode(model.ShadowCullModeBack),
		)),
		game_object.WithPosition(0, 240, 0),
		game_object.WithScale(1, 1, 1),
		game_object.WithEphemeral(true),
	)
	_ = sc.AddGameObject(sunIndicator)

	eng.AddScene(0, sc)

	// ── Input Handling ──────────────────────────────────────────────────
	setupLitInput(eng, cam, fox, sun, bluePoint, orangePoint, spot, quadObj, sunIndicator)

	fmt.Println("╔══════════════════════════════════════════════════════╗")
	fmt.Println("║  Oxy Engine - Lit Scene (Forward+ / CSM+PCF)         ║")
	fmt.Println("╠══════════════════════════════════════════════════════╣")
	fmt.Println("║  Camera: WASD=Pan  Q/E=Up/Down  Scroll=Zoom        ║")
	fmt.Println("║          Middle-mouse drag=Orbit                    ║")
	fmt.Println("║  L:      Toggle sun (directional light)             ║")
	fmt.Println("║  F:      Toggle point lights                        ║")
	fmt.Println("║  G:      Toggle spot light                          ║")
	fmt.Println("║  T:      Toggle sun orbit (day/night cycle)          ║")
	fmt.Println("║  V:      Cycle quad transparency                     ║")
	fmt.Println("║  1-3:    Switch fox animation                       ║")
	fmt.Println("║  Shadow: CSM + PCF (dual-cascade sphere shadows)      ║")
	fmt.Println("╚══════════════════════════════════════════════════════╝")

	log.Println("Starting Oxy Engine - Lit Scene (Forward+ / CSM+PCF)")
	eng.Run()
}

// setupLitInput wires camera controls (WASD/QE planar movement, middle-mouse orbit,
// scroll zoom), number-key animation switching, light toggling (L=sun, F=point lights,
// G=spot), sun orbit (T), and quad transparency cycling (V).
//
// Parameters:
//   - eng: the engine instance providing window callbacks and tick
//   - cam: the camera to control
//   - fox: the fox game object for animation control
//   - sun: the directional light to toggle
//   - bluePoint: the blue point light to toggle
//   - orangePoint: the orange point light to toggle
//   - spot: the spot light to toggle
//   - quad: the transparent quad game object for alpha cycling
//   - sunIndicator: the sun indicator sphere whose position tracks the shadow eye
func setupLitInput(
	eng engine.Engine,
	cam camera.Camera,
	fox game_object.GameObject,
	sun, bluePoint, orangePoint, spot light.Light,
	quad game_object.GameObject,
	sunIndicator game_object.GameObject,
) {
	var keyState sync.Map
	animCount := fox.Model().AnimationCount()
	pointsOn := true
	sunOrbit := false
	var sunAngle float64 // current orbit angle in radians

	// Transparency levels to cycle through with V key
	alphaLevels := []float32{1.0, 0.75, 0.5, 0.25, 0.0}
	alphaIdx := 0

	eng.Window().SetKeyDownCallback(func(keyCode uint32) {
		keyState.Store(keyCode, true)

		// Number keys 1-9 switch animations with a smooth blend transition
		if keyCode >= common.Key1 && keyCode <= common.Key9 {
			clipIdx := int(keyCode - common.Key1)
			if clipIdx < animCount {
				fox.Animator().BlendToAnimation(0, uint32(clipIdx), 0.3)
			}
		}

		// L toggles the directional sun light
		if keyCode == common.KeyL {
			sun.SetEnabled(!sun.Enabled())
			if sun.Enabled() {
				fmt.Println("[Light] Sun ON")
			} else {
				fmt.Println("[Light] Sun OFF")
			}
		}

		// F toggles both point lights
		if keyCode == common.KeyF {
			pointsOn = !pointsOn
			bluePoint.SetEnabled(pointsOn)
			orangePoint.SetEnabled(pointsOn)
			if pointsOn {
				fmt.Println("[Light] Point lights ON")
			} else {
				fmt.Println("[Light] Point lights OFF")
			}
		}

		// G toggles the spot light
		if keyCode == common.KeyG {
			spot.SetEnabled(!spot.Enabled())
			if spot.Enabled() {
				fmt.Println("[Light] Spot light ON")
			} else {
				fmt.Println("[Light] Spot light OFF")
			}
		}

		// T toggles the sun orbit (day/night cycle)
		if keyCode == common.KeyT {
			sunOrbit = !sunOrbit
			if sunOrbit {
				fmt.Println("[Light] Sun orbit ON")
			} else {
				fmt.Println("[Light] Sun orbit OFF")
			}
		}

		// V cycles the quad's vertex color alpha
		if keyCode == common.KeyV {
			alphaIdx = (alphaIdx + 1) % len(alphaLevels)
			newAlpha := alphaLevels[alphaIdx]
			newVerts, newIdx := buildLitQuad(newAlpha)
			quad.Model().SetVertexData(common.SliceToBytes(newVerts))
			quad.Model().SetIndexData(common.SliceToBytes(newIdx))
			fmt.Printf("[Quad] Alpha = %.2f\n", newAlpha)
		}
	})

	eng.Window().SetKeyUpCallback(func(keyCode uint32) {
		keyState.Store(keyCode, false)
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
		// Animate sun orbit: rotate the direction vector around the X axis
		// to simulate the sun arcing from east horizon → overhead → west horizon.
		// One full orbit takes about 30 seconds at ~0.2 rad/s.
		if sunOrbit && sun.Enabled() {
			sunAngle += float64(dt) * 0.2
			if sunAngle > 2*math.Pi {
				sunAngle -= 2 * math.Pi
			}
			// Sun direction orbits in the YZ plane (no X offset) with the sun
			// starting directly overhead (angle 0 → direction (0,-1,0)) and
			// rotating in a clean vertical circle above/below the scene.
			dirY := float32(-math.Cos(sunAngle))
			dirZ := float32(-math.Sin(sunAngle))
			sun.SetDirection(0, dirY, dirZ)
		}

		{
			dir := sun.Direction()
			const sunDistance = 200.0
			sunIndicator.SetPosition(
				0-dir[0]*sunDistance,
				40-dir[1]*sunDistance,
				0-dir[2]*sunDistance,
			)
		}

		if v, ok := keyState.Load(common.KeyW); ok && v.(bool) {
			cam.Controller().PanForward(1)
		}
		if v, ok := keyState.Load(common.KeyS); ok && v.(bool) {
			cam.Controller().PanForward(-1)
		}
		if v, ok := keyState.Load(common.KeyA); ok && v.(bool) {
			cam.Controller().PanRight(-1)
		}
		if v, ok := keyState.Load(common.KeyD); ok && v.(bool) {
			cam.Controller().PanRight(1)
		}
		if v, ok := keyState.Load(common.KeyQ); ok && v.(bool) {
			cam.Controller().PanUp(1)
		}
		if v, ok := keyState.Load(common.KeyE); ok && v.(bool) {
			cam.Controller().PanUp(-1)
		}
	})
}

// buildLitQuad creates a thin 3D slab (box) suitable for the lit rendering
// pipeline. The slab is a closed mesh in the XZ plane with proper front/back
// faces on all 6 sides, enabling correct shadow mapping with CullModeFront.
// It acts as a canopy that blocks sunlight from above.
//
// The slab is 80 units wide in both X and Z, and 2 units thick in Y, centered
// at the origin. Position it in the scene via the game object's WithPosition option.
//
// Parameters:
//   - alpha: the vertex color alpha value (1.0 = fully opaque, 0.0 = fully transparent)
//
// Returns:
//   - []model.GPUVertex: 24 vertices forming the slab (4 per face × 6 faces)
//   - []uint32: 36 indices forming 12 triangles
func buildLitQuad(alpha float32) ([]model.GPUVertex, []uint32) {
	color := [4]float32{0.3, 0.5, 0.9, alpha}
	half := float32(80)       // half-extent in X and Z
	thickness := float32(1.0) // half-thickness in Y

	// Helper to make a vertex
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
		// Top face (+Y), tangent +X
		v(-half, thickness, -half, 0, 1, 0, 0, 0, 1, 0, 0, 1), // 0
		v(half, thickness, -half, 0, 1, 0, 1, 0, 1, 0, 0, 1),  // 1
		v(half, thickness, half, 0, 1, 0, 1, 1, 1, 0, 0, 1),   // 2
		v(-half, thickness, half, 0, 1, 0, 0, 1, 1, 0, 0, 1),  // 3

		// Bottom face (-Y), tangent +X
		v(-half, -thickness, half, 0, -1, 0, 0, 0, 1, 0, 0, 1),  // 4
		v(half, -thickness, half, 0, -1, 0, 1, 0, 1, 0, 0, 1),   // 5
		v(half, -thickness, -half, 0, -1, 0, 1, 1, 1, 0, 0, 1),  // 6
		v(-half, -thickness, -half, 0, -1, 0, 0, 1, 1, 0, 0, 1), // 7

		// Front face (+Z), tangent +X
		v(-half, -thickness, half, 0, 0, 1, 0, 0, 1, 0, 0, 1), // 8
		v(half, -thickness, half, 0, 0, 1, 1, 0, 1, 0, 0, 1),  // 9
		v(half, thickness, half, 0, 0, 1, 1, 1, 1, 0, 0, 1),   // 10
		v(-half, thickness, half, 0, 0, 1, 0, 1, 1, 0, 0, 1),  // 11

		// Back face (-Z), tangent -X
		v(half, -thickness, -half, 0, 0, -1, 0, 0, -1, 0, 0, 1),  // 12
		v(-half, -thickness, -half, 0, 0, -1, 1, 0, -1, 0, 0, 1), // 13
		v(-half, thickness, -half, 0, 0, -1, 1, 1, -1, 0, 0, 1),  // 14
		v(half, thickness, -half, 0, 0, -1, 0, 1, -1, 0, 0, 1),   // 15

		// Right face (+X), tangent +Z
		v(half, -thickness, half, 1, 0, 0, 0, 0, 0, 0, 1, 1),  // 16
		v(half, -thickness, -half, 1, 0, 0, 1, 0, 0, 0, 1, 1), // 17
		v(half, thickness, -half, 1, 0, 0, 1, 1, 0, 0, 1, 1),  // 18
		v(half, thickness, half, 1, 0, 0, 0, 1, 0, 0, 1, 1),   // 19

		// Left face (-X), tangent -Z
		v(-half, -thickness, -half, -1, 0, 0, 0, 0, 0, 0, -1, 1), // 20
		v(-half, -thickness, half, -1, 0, 0, 1, 0, 0, 0, -1, 1),  // 21
		v(-half, thickness, half, -1, 0, 0, 1, 1, 0, 0, -1, 1),   // 22
		v(-half, thickness, -half, -1, 0, 0, 0, 1, 0, 0, -1, 1),  // 23
	}

	// CCW winding for each face when viewed from outside the box
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

// buildSphere creates a UV sphere mesh for use as a visual indicator (e.g. light
// position marker). The sphere is centered at the origin with the given radius.
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
func buildSphere(radius float32, rings, segments int, color [4]float32) ([]model.GPUVertex, []uint32) {
	var vertices []model.GPUVertex
	var indices []uint32

	for r := 0; r <= rings; r++ {
		phi := math.Pi * float64(r) / float64(rings) // 0 (top) → π (bottom)
		y := float32(math.Cos(phi)) * radius
		ringRadius := float32(math.Sin(phi)) * radius

		for s := 0; s <= segments; s++ {
			theta := 2.0 * math.Pi * float64(s) / float64(segments)
			x := ringRadius * float32(math.Cos(theta))
			z := ringRadius * float32(math.Sin(theta))

			// Normal is the normalized position for a unit sphere
			nx := float32(math.Sin(phi) * math.Cos(theta))
			ny := float32(math.Cos(phi))
			nz := float32(math.Sin(phi) * math.Sin(theta))

			u := float32(s) / float32(segments)
			v := float32(r) / float32(rings)

			// Tangent along the longitude direction (+theta)
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

	// Build triangle indices with CCW winding
	stride := segments + 1
	for r := 0; r < rings; r++ {
		for s := 0; s < segments; s++ {
			a := uint32(r*stride + s)
			b := uint32(r*stride + s + 1)
			c := uint32((r+1)*stride + s)
			d := uint32((r+1)*stride + s + 1)

			// Upper triangle
			indices = append(indices, a, c, b)
			// Lower triangle
			indices = append(indices, b, c, d)
		}
	}

	return vertices, indices
}
