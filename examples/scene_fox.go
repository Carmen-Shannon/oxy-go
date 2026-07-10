//go:build ignore

package main

import (
	"fmt"
	"log"
	"math"

	"github.com/oliverbestmann/webgpu/wgpu"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine"
	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/loader"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	bgp "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/postprocessing/taa"
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
			window.WithTitle("Oxy Engine - Fox Skeletal Animation"),
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
		camera.WithNear(0.5),
		camera.WithFar(10000),
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
	sc := scene.NewScene("Fox Animation Test", cam, r,
		scene.WithTAAHandler(taa.NewHandler(
			taa.WithTAAHistoryRectificationScale(0.1),
			taa.WithTAAJitterScale(0.6),
		)),
	)

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
	injections := buildDefaultInjectionMap()

	// ── Tint Overlay Material ──────────────────────────────────────────────
	tintMat := material.NewMaterial(
		material.WithName("tint"),
		material.WithPipelineKey(foxModel.Name()+"_tint"),
		material.WithPipelineOptions(
			pipeline.WithFragmentShader(shader.NewShader("tint_frag", shader.ShaderTypeFragment, "examples/assets/shaders/tint-overlay-frag.wgsl", shader.WithInjections(injections))),
			pipeline.WithBlendEnabled(true),
			pipeline.WithBlendState(&wgpu.BlendState{
				Color: wgpu.BlendComponent{
					SrcFactor: wgpu.BlendFactorSrcAlpha,
					DstFactor: wgpu.BlendFactorOneMinusSrcAlpha,
					Operation: wgpu.BlendOperationAdd,
				},
				Alpha: wgpu.BlendComponent{
					SrcFactor: wgpu.BlendFactorOne,
					DstFactor: wgpu.BlendFactorOneMinusSrcAlpha,
					Operation: wgpu.BlendOperationAdd,
				},
			}),
			pipeline.WithDepthCompare(wgpu.CompareFunctionLessEqual),
			pipeline.WithDepthWriteEnabled(false),
		),
	)

	// ── Rainbow Material ───────────────────────────────────────────────────
	rainbowMat := material.NewMaterial(
		material.WithName("rainbow"),
		material.WithPipelineKey(foxModel.Name()+"_rainbow"),
		material.WithPipelineOptions(
			pipeline.WithFragmentShader(shader.NewShader("rainbow_frag", shader.ShaderTypeFragment, "examples/assets/shaders/skinned-rainbow-frag.wgsl", shader.WithInjections(injections))),
		),
	)

	// ── Overlay Material (inverted hull outline) ───────────────────────────
	overlayMat := material.NewMaterial(
		material.WithName("overlay"),
		material.WithPipelineKey(foxModel.Name()+"_overlay"),
		material.WithPipelineOptions(
			pipeline.WithVertexShader(shader.NewShader("outline_vert", shader.ShaderTypeVertex, "examples/assets/shaders/outline-vert.wgsl", shader.WithInjections(injections))),
			pipeline.WithFragmentShader(shader.NewShader("overlay_frag", shader.ShaderTypeFragment, "examples/assets/shaders/overlay-frag.wgsl", shader.WithInjections(injections))),
			pipeline.WithCullMode(wgpu.CullModeFront),
			pipeline.WithDepthCompare(wgpu.CompareFunctionLessEqual),
			pipeline.WithDepthWriteEnabled(false),
		),
	)

	// ── Wood Texture Material ──────────────────────────────────────────────
	woodMat := material.NewMaterial(
		material.WithName("wood"),
		material.WithBaseColor([4]float32{1, 1, 1, 1}),
		material.WithDiffuseTexture(&common.ImportedTexture{
			Name:     "wood_diffuse",
			Path:     "examples/assets/textures/wood.png",
			MimeType: "image/png",
		}),
		material.WithPipelineKey(foxModel.Name()+"_wood"),
	)

	// Save the original materials so we can restore them when toggling back.
	originalMats := make([]material.Material, len(foxModel.RenderMaterials()))
	copy(originalMats, foxModel.RenderMaterials())

	// Extend the model's material list to include all variants for GPU init at Add time.
	allMats := make([]material.Material, len(originalMats), len(originalMats)+4)
	copy(allMats, originalMats)
	allMats = append(allMats, tintMat, rainbowMat, overlayMat, woodMat)
	foxModel.SetRenderMaterials(allMats)
	_ = sc.AddGameObject(fox)
	// Restore original materials as the active set (variants inactive by default).
	foxModel.SetRenderMaterials(originalMats)

	// Start initial animation (first clip, looped)
	if foxModel.AnimationCount() > 0 {
		fox.Animator().PlayAnimation(0, 0, true)
	}

	eng.AddScene(0, sc)

	// ── Input Handling ──────────────────────────────────────────────────
	setupFoxInput(eng, cam, fox, r, originalMats, tintMat, rainbowMat, overlayMat, woodMat)

	// Print animation names for user reference
	animNames := foxModel.AnimationNames()
	fmt.Println("╔══════════════════════════════════════════════════════╗")
	fmt.Println("║  Oxy Engine - Fox Skeletal Animation                ║")
	fmt.Println("╠══════════════════════════════════════════════════════╣")
	fmt.Println("║  Camera: WASD=Pan  Q/E=Up/Down  Scroll=Zoom        ║")
	fmt.Println("║          Middle-mouse drag=Orbit                    ║")
	fmt.Println("║  Space:  Toggle Textured ↔ Rainbow shader          ║")
	fmt.Println("║  X:      Toggle wood texture (runtime swap)          ║")
	fmt.Println("║  V:      Toggle damage tint (uniform approach)      ║")
	fmt.Println("║  B:      Toggle outline (multi-pass approach)        ║")
	fmt.Println("║  Animations:                                        ║")
	for i, name := range animNames {
		fmt.Printf("║    %d = %-46s║\n", i+1, name)
	}
	fmt.Println("╚══════════════════════════════════════════════════════╝")

	log.Println("Starting Oxy Engine - Fox Skeletal Animation")
	eng.Run()
}

func buildDefaultInjectionMap() map[string]string {
	return map[string]string{
		"max_bones":              fmt.Sprintf("%du", uint64(64)),
		"empty_sentinel":         fmt.Sprintf("0x%Xu", uint32(0xFFFFFFFF)),
		"max_ssao_samples":       fmt.Sprintf("%du", uint32(32)),
		"pcf_samples":            fmt.Sprintf("%du", uint32(16)),
		"flag_active":            fmt.Sprintf("%du", physics.PhysicsStateActive),
		"flag_static":            fmt.Sprintf("%du", physics.PhysicsStateStatic),
		"flag_kinematic":         fmt.Sprintf("%du", physics.PhysicsStateKinematic),
		"light_type_directional": fmt.Sprintf("%du", light.LightTypeDirectional),
		"light_type_point":       fmt.Sprintf("%du", light.LightTypePoint),
		"light_type_spot":        fmt.Sprintf("%du", light.LightTypeSpot),
	}
}

// setupFoxInput wires camera controls (WASD/QE movement, middle-mouse orbit, scroll zoom),
// number-key animation switching with blend transitions, spacebar shader toggling,
// X-key wood texture toggle, V-key tint toggle, and B-key overlay toggle.
func setupFoxInput(
	eng engine.Engine,
	cam camera.Camera,
	fox game_object.GameObject,
	r renderer.Renderer,
	originalMats []material.Material,
	tintMat, rainbowMat, overlayMat, woodMat material.Material,
) {
	keyState := make(map[uint32]bool)
	animCount := fox.Model().AnimationCount()
	usingRainbow := false
	usingWood := false
	tintActive := false
	overlayActive := false

	// rebuildMaterials reconstructs the full material list from current state.
	// Each material already carries its own pipeline key (auto-derived from its
	// fragment shader path), so we only need to swap which materials are active.
	rebuildMaterials := func() {
		var baseMats []material.Material
		switch {
		case usingRainbow:
			baseMats = []material.Material{rainbowMat}
		case usingWood:
			baseMats = []material.Material{woodMat}
		default:
			restored := make([]material.Material, len(originalMats))
			copy(restored, originalMats)
			baseMats = restored
		}

		if tintActive {
			baseMats = append(baseMats, tintMat)
		}
		if overlayActive {
			baseMats = append(baseMats, overlayMat)
		}
		fox.Model().SetRenderMaterials(baseMats)
	}

	eng.Window().SetKeyDownCallback(func(keyCode uint32) {
		keyState[keyCode] = true

		// Number keys 1-9 switch animations with a smooth blend transition
		if keyCode >= common.Key1 && keyCode <= common.Key9 {
			clipIdx := int(keyCode - common.Key1)
			if clipIdx < animCount {
				fox.Animator().BlendToAnimation(0, uint32(clipIdx), 0.3)
			}
		}

		// Spacebar toggles between textured and rainbow fragment shaders.
		// Enabling rainbow disables wood since the rainbow shader generates
		// colors procedurally and ignores textures.
		if keyCode == common.KeySpace {
			usingRainbow = !usingRainbow
			if usingRainbow && usingWood {
				usingWood = false
				fmt.Println("[Texture] Wood OFF (overridden by rainbow)")
			}
			rebuildMaterials()
			if usingRainbow {
				fmt.Println("[Shader] Switched to: Rainbow")
			} else {
				fmt.Println("[Shader] Switched to: Textured")
			}
		}

		// X key toggles the wood texture (runtime texture swap).
		// Enabling wood disables rainbow since the rainbow shader ignores
		// textures entirely.
		if keyCode == common.KeyX {
			usingWood = !usingWood
			if usingWood && usingRainbow {
				usingRainbow = false
				fmt.Println("[Shader] Rainbow OFF (overridden by wood)")
			}
			rebuildMaterials()
			if usingWood {
				fmt.Println("[Texture] Wood ON")
			} else {
				fmt.Println("[Texture] Wood OFF (original restored)")
			}
		}

		// V key toggles the tint overlay (Approach 1)
		if keyCode == common.KeyV {
			tintActive = !tintActive
			var tint [4]float32
			if tintActive {
				tint = [4]float32{1.0, 0.0, 0.0, 0.5} // red, 50% intensity
			}
			r.WriteBuffers([]bgp.BufferWrite{{
				Provider: tintMat.Provider(2),
				Binding:  0,
				Offset:   0,
				Data:     common.SliceToBytes(tint[:]),
			}})
			rebuildMaterials()
			if tintActive {
				fmt.Println("[Effect] Tint ON (red 50%)")
			} else {
				fmt.Println("[Effect] Tint OFF")
			}
		}

		// B key toggles the overlay pass (Approach 2)
		if keyCode == common.KeyB {
			overlayActive = !overlayActive
			var overlayColor [4]float32
			if overlayActive {
				overlayColor = [4]float32{0.0, 0.0, 0.0, 1.0} // solid black outline
			}
			r.WriteBuffers([]bgp.BufferWrite{{
				Provider: overlayMat.BindGroupProvider(),
				Binding:  0,
				Offset:   0,
				Data:     common.SliceToBytes(overlayColor[:]),
			}})
			rebuildMaterials()
			if overlayActive {
				fmt.Println("[Outline] ON")
			} else {
				fmt.Println("[Outline] OFF")
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
