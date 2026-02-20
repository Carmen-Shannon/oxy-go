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
	"github.com/Carmen-Shannon/oxy-go/engine/loader"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	bgp "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
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
		camera.WithNear(0.01),
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

	// ── Shaders ─────────────────────────────────────────────────────────
	computeShader := shader.NewShader("skeletal_compute", shader.ShaderTypeCompute, "examples/assets/shaders/skeletal-compute.wgsl")
	vertexShader := shader.NewShader("skinned_vert", shader.ShaderTypeVertex, "examples/assets/shaders/skinned-vert.wgsl")
	outlineVert := shader.NewShader("outline_vert", shader.ShaderTypeVertex, "examples/assets/shaders/outline-vert.wgsl")
	texturedFrag := shader.NewShader("textured_frag", shader.ShaderTypeFragment, "examples/assets/shaders/textured-frag.wgsl")
	rainbowFrag := shader.NewShader("skinned_rainbow_frag", shader.ShaderTypeFragment, "examples/assets/shaders/skinned-rainbow-frag.wgsl")
	overlayFrag := shader.NewShader("overlay_frag", shader.ShaderTypeFragment, "examples/assets/shaders/overlay-frag.wgsl")

	// Use textured as the primary fragment shader for loading (sets up material GPU resources)
	fragmentShader := texturedFrag

	// ── Scene ───────────────────────────────────────────────────────────
	sc := scene.NewScene("Fox Animation Test", cam, r, vertexShader,
		scene.WithActive(true),
	)

	// ── Load Fox Model ──────────────────────────────────────────────────
	ldr := loader.NewLoader(loader.BackendTypeGLTF, loader.WithRenderer(r))
	foxModel, err := ldr.Load("examples/assets/models/Fox.glb", fragmentShader)
	if err != nil {
		log.Fatalf("Failed to load Fox model: %v", err)
	}

	fox := game_object.NewGameObject(
		game_object.WithModel(foxModel),
		game_object.WithPosition(0, 0, 0),
		game_object.WithScale(1, 1, 1),
	)

	_ = sc.Add(fox, computeShader, vertexShader, fragmentShader)

	// ── Approach 1: Effect Tint Uniform (@group(3) in textured, @group(2) in rainbow) ──
	// Create a bind group provider for the tint uniform and attach it to the model.
	// Both fragment shaders read this uniform and mix the tint with their output color.
	effectBGP := bgp.NewBindGroupProvider("fox_effect_tint")
	// Init from textured shader's @group(3) layout — it's a single uniform vec4<f32>.
	if err := r.InitBindGroup(effectBGP, texturedFrag.BindGroupLayoutDescriptor(3), nil, nil); err != nil {
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

	// ── Register alternate render pipelines ─────────────────────────────
	// Rainbow pipeline (spacebar toggle)
	rainbowPipelineKey := foxModel.Name() + "_rainbow"
	rainbowPipeline := pipeline.NewPipeline(rainbowPipelineKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(vertexShader),
		pipeline.WithFragmentShader(rainbowFrag),
	)
	if err := r.RegisterPipelines(rainbowPipeline); err != nil {
		log.Fatalf("Failed to register rainbow pipeline: %v", err)
	}

	// ── Approach 2: Multi-Pass Outline (inverted hull) ─────────────────
	// Register an outline pipeline: inflated vertices + front-face culling = silhouette.
	// Depth test stays ON so the outline is occluded by closer objects. Depth write
	// is OFF so the outline doesn't interfere with the normal geometry's depth.
	overlayPipelineKey := foxModel.Name() + "_overlay"
	overlayPipeline := pipeline.NewPipeline(overlayPipelineKey, pipeline.PipelineTypeRender,
		pipeline.WithVertexShader(outlineVert),
		pipeline.WithFragmentShader(overlayFrag),
		pipeline.WithCullMode(wgpu.CullModeFront), // render only back faces
		pipeline.WithDepthWriteEnabled(false),     // don't occlude the normal pass
	)
	if err := r.RegisterPipelines(overlayPipeline); err != nil {
		log.Fatalf("Failed to register overlay pipeline: %v", err)
	}

	// Pre-create the overlay material with its own bind group provider.
	// The overlay shader has @group(2) with a single uniform (overlay_color vec4).
	overlayBGP := bgp.NewBindGroupProvider("fox_overlay_material")
	if err := r.InitBindGroup(overlayBGP, overlayFrag.BindGroupLayoutDescriptor(2), nil, nil); err != nil {
		log.Fatalf("Failed to init overlay bind group: %v", err)
	}
	// Write initial outline color (solid black)
	overlayColor := [4]float32{0.0, 0.0, 0.0, 1.0}
	r.WriteBuffers([]bgp.BufferWrite{{
		Provider: overlayBGP,
		Binding:  0,
		Offset:   0,
		Data:     common.SliceToBytes(overlayColor[:]),
	}})

	overlayMat := material.NewMaterial(
		material.WithName("overlay"),
		material.WithPipelineKey(overlayPipelineKey),
	)
	overlayMat.SetBindGroupProvider(overlayBGP)

	// ── Wood Texture Material (X-key hot-swap) ─────────────────────────
	// Create a material that samples a .png file from disk as the diffuse
	// texture. InitMaterialGPU decodes the image, uploads it to the GPU,
	// and creates fallback normal/metallic-roughness textures automatically.
	woodMat := material.NewMaterial(
		material.WithName("wood"),
		material.WithBaseColor([4]float32{1, 1, 1, 1}),
		material.WithDiffuseTexture(&common.ImportedTexture{
			Name:     "wood_diffuse",
			Path:     "examples/assets/textures/wood.png",
			MimeType: "image/png",
		}),
		material.WithPipelineKey(foxModel.Name()),
	)
	if err := ldr.InitMaterialGPU(woodMat, fragmentShader, "wood"); err != nil {
		log.Fatalf("Failed to init wood material GPU: %v", err)
	}

	// Save the original materials so we can restore them when toggling back.
	originalMats := make([]material.Material, len(foxModel.RenderMaterials()))
	copy(originalMats, foxModel.RenderMaterials())

	// Start initial animation (first clip, looped)
	if foxModel.AnimationCount() > 0 {
		fox.Animator().PlayAnimation(0, 0, true)
	}

	eng.AddScene(0, sc)

	// ── Input Handling ──────────────────────────────────────────────────
	setupFoxInput(eng, cam, fox, r, effectBGP, overlayMat, woodMat, originalMats, foxModel.Name(), rainbowPipelineKey)

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

// setupFoxInput wires camera controls (WASD/QE movement, middle-mouse orbit, scroll zoom),
// number-key animation switching with blend transitions, spacebar shader toggling,
// X-key wood texture toggle, V-key tint toggle, and B-key overlay toggle.
func setupFoxInput(
	eng engine.Engine,
	cam camera.Camera,
	fox game_object.GameObject,
	r renderer.Renderer,
	effectBGP bgp.BindGroupProvider,
	overlayMat material.Material,
	woodMat material.Material,
	originalMats []material.Material,
	texturedKey, rainbowKey string,
) {
	keyState := make(map[uint32]bool)
	animCount := fox.Model().AnimationCount()
	usingRainbow := false
	usingWood := false
	tintActive := false
	overlayActive := false

	eng.Window().SetKeyDownCallback(func(keyCode uint32) {
		keyState[keyCode] = true

		// Number keys 1-9 switch animations with a smooth blend transition
		if keyCode >= common.Key1 && keyCode <= common.Key9 {
			clipIdx := int(keyCode - common.Key1)
			if clipIdx < animCount {
				fox.Animator().BlendToAnimation(0, uint32(clipIdx), 0.3)
			}
		}

		// Spacebar toggles between textured and rainbow fragment shaders
		if keyCode == common.KeySpace {
			usingRainbow = !usingRainbow
			newKey := texturedKey
			if usingRainbow {
				newKey = rainbowKey
			}
			// Only swap the base materials, not the overlay material
			for _, mat := range fox.Model().RenderMaterials() {
				if mat.Name() == "overlay" {
					continue
				}
				mat.SetPipelineKey(newKey)
			}
			if usingRainbow {
				fmt.Println("[Shader] Switched to: Rainbow")
			} else {
				fmt.Println("[Shader] Switched to: Textured")
			}
		}

		// X key toggles the wood texture (runtime texture swap)
		if keyCode == common.KeyX {
			usingWood = !usingWood
			if usingWood {
				// Replace all base materials with the wood-textured material
				fox.Model().SetRenderMaterials([]material.Material{woodMat})
				fmt.Println("[Texture] Wood ON")
			} else {
				// Restore original materials
				restored := make([]material.Material, len(originalMats))
				copy(restored, originalMats)
				fox.Model().SetRenderMaterials(restored)
				fmt.Println("[Texture] Wood OFF (original restored)")
			}
		}

		// V key toggles the tint uniform (Approach 1)
		if keyCode == common.KeyV {
			tintActive = !tintActive
			var tint [4]float32
			if tintActive {
				tint = [4]float32{1.0, 0.0, 0.0, 0.5} // red, 50% intensity
			}
			r.WriteBuffers([]bgp.BufferWrite{{
				Provider: effectBGP,
				Binding:  0,
				Offset:   0,
				Data:     common.SliceToBytes(tint[:]),
			}})
			if tintActive {
				fmt.Println("[Effect] Tint ON (red 50%)")
			} else {
				fmt.Println("[Effect] Tint OFF")
			}
		}

		// B key toggles the overlay pass (Approach 2)
		if keyCode == common.KeyB {
			overlayActive = !overlayActive
			mats := fox.Model().RenderMaterials()
			if overlayActive {
				// Append the overlay material for a second draw pass
				fox.Model().SetRenderMaterials(append(mats, overlayMat))
				fmt.Println("[Outline] ON")
			} else {
				// Remove the overlay material
				filtered := make([]material.Material, 0, len(mats))
				for _, m := range mats {
					if m.Name() != "overlay" {
						filtered = append(filtered, m)
					}
				}
				fox.Model().SetRenderMaterials(filtered)
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
