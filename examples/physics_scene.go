//go:build ignore

package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine"
	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/game_object"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/loader"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/physics"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

// ── Physical constants ──────────────────────────────────────────────────────

// gravity is the gravitational acceleration constant (m/s²).
const gravity = -9.81

// dropletRadius is the visual radius of each fluid droplet sphere.
const dropletRadius = 0.15

// particleRadius is the DEM collision particle radius. All bodies share the
// same particle diameter in the current pipeline (globals.particle_diameter).
const particleRadius = 0.15

// wallHalfThick is the half-extent of each wall's thin dimension.
// Full wall thickness = 2 × wallHalfThick = 0.90 m = 3 particle diameters,
// giving three layers of collision particles for robust containment
// under heavy stacking loads.
const wallHalfThick = 0.45

// ── Container geometry ──────────────────────────────────────────────────────
// Open-top pool. Inner floor surface at y=0.
const (
	boxW = 20.0 // inner width  (X axis)
	boxH = 6.0  // inner height (Y axis)
	boxD = 16.0 // inner depth  (Z axis)
)

// ── Droplet stream parameters ────────────────────────────────────────────────
// Droplets are spawned in a constant stream above the pool, like water
// from a faucet. When the maximum live count is reached, the oldest
// droplets are removed via Scene.Remove and new ones are spawned via
// Scene.Add — a true circular spawn/despawn system.
const (
	maxDroplets   = 250000 // maximum simultaneous live droplets
	spawnPerTick  = 10     // new droplets spawned per spawn event
	spawnInterval = 1      // ticks between spawn events (60 Hz / 2 = 30 droplets/sec)
	spawnY        = 18.0   // Y position of spawn point (well above the 4m box top)
	spawnRadius   = 3.0    // XZ scatter radius around stream center (tight faucet)
	jitterAmp     = 0.01   // ±jitter per axis on spawn position
)

// ── DEM coefficients ────────────────────────────────────────────────────────
// Tuned for FLUID behavior with penetration-weighted DEM damping.
// The collision shader uses damp_weight = penetration/diameter so that
// shallow first-touch contacts produce negligible damping (preventing
// velocity reversal) while deep overlaps get full dissipation.
//
// Wall contacts use per-axis averaging (Fix B) with surface-normal gating
// (Fix A) to prevent reversed spring forces in multi-layer walls and avoid
// cross-axis force dilution at corners. Effective stiffness per contact
// group is exactly k=500, well within the stability limit.
//
// At full-diameter penetration (worst case), effective single-contact
// damping ratio ξ = c / (2√(mk)) = 10 / (2√(0.05×500)) = 10/10 = 1.0
// (critically damped — optimal energy absorption).
//
// Stability limits for m=0.05 kg, dt=1/120 s:
//   - Spring:   k < 4m/dt² = 2880        → 500 (safe per contact)
//   - Damping:  c < 2m/dt  = 12          → 10  (sub-critical at deep overlap)
//   - Shear:    tangential friction       → 4   (viscous sliding)
const (
	fluidSpring  = 500.0
	fluidDamping = 10.0
	fluidShear   = 4.0
)

// dropletMass is the per-particle mass in kilograms.
//
// 0.05 kg gives:
//   - Gravity force: 0.05 × 9.81 ≈ 0.49 N
//   - Rest penetration vs spring(500): 0.49/500 ≈ 0.001 m (0.3% of diameter)
//   - Critical damping at full overlap: 2√(mk) = 2√(25) = 10, c=10 → ξ=1.0
//   - Critical dt for spring: 2√(m/k) = 0.020 s > 0.0083 s — stable
const dropletMass = 0.05

func main() {
	fmt.Printf("[Physics] Droplet mass: %.4f kg (radius=%.3f m)\n", dropletMass, dropletRadius)

	// ── Engine + Window ─────────────────────────────────────────────────
	eng := engine.NewEngine(
		engine.WithProfiling(true),
		engine.WithTickRate(60),
		engine.WithWindow(window.NewWindow(
			window.WithTitle("Oxy Engine - Fluid Physics Demo"),
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
			camera.WithRadius(40),
			camera.WithTarget(0, 1.5, 0),
			camera.WithElevation(0.6),
			camera.WithAzimuth(0.8),
			camera.WithPanSpeed(0.15),
			camera.WithRadiusBounds(3, 150),
			camera.WithZoomSpeed(3.0),
			camera.WithMouseSensitivity(0.002),
		)),
	)

	// ── Physics ─────────────────────────────────────────────────────────
	// No analytical boundary planes — the container is a single voxelized
	// static mesh whose surface particles provide DEM collision on all faces.
	// The thicker walls (3 particle layers) and dynamic V_MAX clamping in
	// integrate.wgsl prevent tunneling under heavy stacking loads.
	ph := physics.NewPhysics(
		physics.WithFixedDt(1.0/120.0),
		physics.WithMaxSubsteps(8),
		physics.WithMaxBodies(250003),
		physics.WithMaxParticles(310000),
		physics.WithSpringCoeff(fluidSpring),
		physics.WithDampingCoeff(fluidDamping),
		physics.WithShearCoeff(fluidShear),
		physics.WithGravity([3]float32{0, gravity, 0}),
	)

	// ── Scene ───────────────────────────────────────────────────────────
	sc := scene.NewScene("Fluid Physics", cam, r,
		scene.WithActive(true),
		scene.WithScreenSize(eng.Window().Width(), eng.Window().Height()),
		scene.WithLighting(light.NewLightingHandler(
			light.WithShadowHalfExtent(60),
			light.WithShadowNearFar(0.1, 120),
			light.WithShadowBias(0.001),
		)),
	)
	sc.SetPhysicsHandler(ph)

	// ── Lighting ───────────────────────────────────────────────────────
	// Directional sun light offset from center, angled toward the box.
	// Direction vector points from the light toward the scene.
	sun := light.NewLight(light.LightTypeDirectional,
		light.WithDirection(-0.4, -1.0, -0.3),
		light.WithColor(1.0, 0.95, 0.85),
		light.WithIntensity(1.5),
		light.WithCastsShadows(true),
		light.WithEnabled(true),
	)
	sc.AddLight(sun)

	// Dim ambient fill so shadows are visible but not fully black
	sc.SetAmbientColor([3]float32{0.12, 0.12, 0.15})

	// Loader for loading fox model
	ldr := loader.NewLoader(loader.BackendTypeGLTF)

	// ── Container (voxelized static box) ────────────────────────────────
	// A single open-top box mesh forms the containment pool. VoxelizeMesh
	// with surfaceOnly=true generates DEM collision particles on every
	// surface face. The box is registered as a static RigidBody (mass=0)
	// so its particles participate in the spatial grid and produce contact
	// forces against droplets, but the box itself never moves.
	wallColor := [4]float32{0.3, 0.35, 0.4, 1.0}
	boxModel := buildOpenBoxModel("container",
		float32(boxW)/2, float32(boxH)/2, float32(boxD)/2,
		float32(wallHalfThick), wallColor,
	)

	boxParticles := physics.VoxelizeMesh(boxModel, particleRadius, true)
	fmt.Printf("[Physics] Container voxelized: %d surface particles\n", len(boxParticles))

	boxBody := physics.NewRigidBody(
		physics.WithMass(0),
		physics.WithStatic(true),
		physics.WithActive(true),
		physics.WithParticles(boxParticles),
		physics.WithParticleRadius(particleRadius),
	)

	// Position at origin — model geometry is built with the inner floor
	// surface at y=0, walls rising to y=boxH, and centered on XZ.
	boxObj := game_object.NewGameObject(
		game_object.WithModel(boxModel),
		game_object.WithPosition(0, 0, 0),
		game_object.WithScale(1, 1, 1),
		game_object.WithRigidBody(boxBody),
	)
	sc.Add(boxObj)

	// ── Fox model (animated obstacle in the center of the box) ─────────
	// Load the fox glTF model and place it at the center of the box floor.
	// Scaled down to fit inside the container (model is ~100 units tall at
	// scale 1, box inner height is 6 m). A random animation plays in a loop
	// so particles fall over and interact with the animated geometry.
	foxModel, err := ldr.Load("examples/assets/models/Fox.glb")
	if err != nil {
		log.Fatalf("Failed to load Fox model: %v", err)
	}

	foxScale := float32(0.04) // ~4 m tall inside the 6 m box

	// Voxelize the fox mesh into DEM collision particles. Voxelization works
	// in model space, so we use particleRadius/foxScale as the model-space
	// radius. Positions are kept in model space (NOT scaled) because the
	// bone_particle_update shader transforms them through bone world matrices
	// that already include the fox's world transform (with scale).
	modelSpaceRadius := particleRadius / foxScale
	foxParticles := physics.VoxelizeMesh(foxModel, modelSpaceRadius, true)

	// Assign each voxel particle to its nearest bone and transform the
	// particle's model-space position into bone-local space via the bone's
	// inverse bind matrix. The GPU bone_particle_update shader transforms
	// these back to world space each frame using the current bone matrices.
	physics.AssignBoneIndices(foxParticles, foxModel)
	fmt.Printf("[Physics] Fox voxelized: %d surface particles (bone-assigned)\n", len(foxParticles))

	foxBody := physics.NewRigidBody(
		physics.WithMass(0),
		physics.WithKinematic(true),
		physics.WithActive(true),
		physics.WithParticles(foxParticles),
		physics.WithParticleRadius(particleRadius),
	)

	fox := game_object.NewGameObject(
		game_object.WithModel(foxModel),
		game_object.WithPosition(0, 0, 0),
		game_object.WithScale(foxScale, foxScale, foxScale),
		game_object.WithRigidBody(foxBody),
	)
	sc.Add(fox)

	// Start a random animation on the fox, looping
	if foxModel.AnimationCount() > 0 {
		clip := uint32(rand.Intn(foxModel.AnimationCount()))
		fox.Animator().PlayAnimation(uint32(fox.AnimatorInstanceID()), clip, true)
	}

	// ── Shared droplet assets ───────────────────────────────────────────
	// A single sphere Model is shared across all droplets (instanced rendering).
	// Each droplet is a single DEM collision sphere at body origin.
	dropletColor := [4]float32{0.2, 0.5, 0.9, 1.0}
	sphereModel := buildSphereModel("droplet", dropletRadius, 2, dropletColor)
	sphereParticles := []physics.Particle{{LocalPosition: [3]float32{0, 0, 0}}}

	eng.AddScene(0, sc)

	// ── Input handling + streaming spawn ─────────────────────────
	setupFluidInput(eng, cam, sc, ph, sphereModel, sphereParticles)

	fmt.Println("╔══════════════════════════════════════════════════════╗")
	fmt.Println("║  Oxy Engine - Fluid Physics Demo                    ║")
	fmt.Println("╠══════════════════════════════════════════════════════╣")
	fmt.Println("║  Camera: WASD=Pan  Q/E=Up/Down  Scroll=Zoom        ║")
	fmt.Println("║          Middle-mouse drag=Orbit                    ║")
	fmt.Println("║  Space: Toggle spawning on/off                      ║")
	fmt.Println("║                                                     ║")
	fmt.Println("║  Streaming fluid: 30 droplets/sec from a source.    ║")
	fmt.Println("║  Oldest droplets removed when pool reaches 250000.  ║")
	fmt.Println("║  Directional sun with shadow-casting particles.     ║")
	fmt.Println("╚══════════════════════════════════════════════════════╝")

	log.Println("Starting Oxy Engine - Fluid Physics Demo")
	eng.Run()
}

// tangentFromNormal computes a tangent vector (with handedness w=1) perpendicular
// to the given normal. Used by procedural geometry builders so the lit fragment
// shader has a valid TBN matrix for normal mapping.
//
// Parameters:
//   - nx, ny, nz: the surface normal components (assumed normalized)
//
// Returns:
//   - [4]float32: tangent xyz with w=1 for bitangent handedness
func tangentFromNormal(nx, ny, nz float32) [4]float32 {
	// Choose an up vector that isn't parallel to the normal.
	// If the normal is nearly vertical (|ny| > 0.9), use the Z-axis instead.
	var ux, uy, uz float32
	if ny > 0.9 || ny < -0.9 {
		ux, uy, uz = 0, 0, 1
	} else {
		ux, uy, uz = 0, 1, 0
	}
	// cross(normal, up) gives the tangent direction
	tx := ny*uz - nz*uy
	ty := nz*ux - nx*uz
	tz := nx*uy - ny*ux
	// normalize
	l := float32(math.Sqrt(float64(tx*tx + ty*ty + tz*tz)))
	if l < 1e-6 {
		return [4]float32{1, 0, 0, 1}
	}
	return [4]float32{tx / l, ty / l, tz / l, 1}
}

// buildOpenBoxModel creates an open-top box (floor + 4 walls) as a single mesh.
// Each slab is a watertight 6-face sub-box so VoxelizeMesh ray-casting produces
// correct inside/outside classifications. The inner floor surface sits at y=0,
// walls rise from y=0 to y=2*hy, and the box is centered on XZ.
//
// Parameters:
//   - name: the unique model name
//   - ihx: inner half-width (X axis)
//   - ihy: inner half-height (Y axis, wall height = 2*ihy)
//   - ihz: inner half-depth (Z axis)
//   - wt: wall/floor thickness
//   - color: RGBA vertex color
//
// Returns:
//   - model.Model: the fully configured Model
func buildOpenBoxModel(name string, ihx, ihy, ihz, wt float32, color [4]float32) model.Model {
	var verts []model.GPUVertex
	var indices []uint32

	v := func(px, py, pz, nx, ny, nz float32) model.GPUVertex {
		t := tangentFromNormal(nx, ny, nz)
		return model.GPUVertex{
			Position: [3]float32{px, py, pz},
			Normal:   [3]float32{nx, ny, nz},
			Color:    color,
			Tangent:  t,
		}
	}

	// addBox appends a closed 6-face axis-aligned box at position (cx,cy,cz)
	// with half-extents (hx,hy,hz) to the shared vertex/index slices.
	addBox := func(cx, cy, cz, hx, hy, hz float32) {
		base := uint32(len(verts))
		x0, x1 := cx-hx, cx+hx
		y0, y1 := cy-hy, cy+hy
		z0, z1 := cz-hz, cz+hz

		verts = append(verts,
			// Top face (+Y)
			v(x0, y1, z0, 0, 1, 0), v(x1, y1, z0, 0, 1, 0),
			v(x1, y1, z1, 0, 1, 0), v(x0, y1, z1, 0, 1, 0),
			// Bottom face (-Y)
			v(x0, y0, z0, 0, -1, 0), v(x1, y0, z0, 0, -1, 0),
			v(x1, y0, z1, 0, -1, 0), v(x0, y0, z1, 0, -1, 0),
			// Front face (+Z)
			v(x0, y0, z1, 0, 0, 1), v(x1, y0, z1, 0, 0, 1),
			v(x1, y1, z1, 0, 0, 1), v(x0, y1, z1, 0, 0, 1),
			// Back face (-Z)
			v(x1, y0, z0, 0, 0, -1), v(x0, y0, z0, 0, 0, -1),
			v(x0, y1, z0, 0, 0, -1), v(x1, y1, z0, 0, 0, -1),
			// Right face (+X)
			v(x1, y0, z1, 1, 0, 0), v(x1, y0, z0, 1, 0, 0),
			v(x1, y1, z0, 1, 0, 0), v(x1, y1, z1, 1, 0, 0),
			// Left face (-X)
			v(x0, y0, z0, -1, 0, 0), v(x0, y0, z1, -1, 0, 0),
			v(x0, y1, z1, -1, 0, 0), v(x0, y1, z0, -1, 0, 0),
		)

		// Top face (+Y): winding CCW from above
		b := base + 0
		indices = append(indices, b, b+2, b+1, b, b+3, b+2)
		// Bottom, Front, Back, Right, Left: reversed winding for CCW from
		// their respective outward normals
		for _, off := range [5]uint32{4, 8, 12, 16, 20} {
			b = base + off
			indices = append(indices, b, b+1, b+2, b, b+2, b+3)
		}
	}

	wallH := ihy      // half-height of the walls
	wallCY := wallH   // center Y of the walls (bottom at y=0, top at 2*ihy)
	ohx := ihx + 2*wt // outer half-width including full wall thickness (2×wt per side)
	ohz := ihz + 2*wt // outer half-depth including full wall thickness (2×wt per side)

	// Floor slab — full outer footprint, top surface at y=0
	// wt is used directly as the half-thickness so the floor is 2*wt thick
	// (≥ 2 particle diameters), giving at least 2 voxel layers for robust
	// collision containment.
	addBox(0, -wt, 0, ohx, wt, ohz)
	// Left wall (-X)
	addBox(-(ihx + wt), wallCY, 0, wt, wallH, ohz)
	// Right wall (+X)
	addBox(ihx+wt, wallCY, 0, wt, wallH, ohz)
	// Back wall (-Z) — fits between side walls (inner width span)
	addBox(0, wallCY, -(ihz + wt), ihx, wallH, wt)
	// Front wall (+Z) — fits between side walls (inner width span)
	addBox(0, wallCY, ihz+wt, ihx, wallH, wt)

	// Bounding radius: distance from origin to farthest corner
	br := float32(math.Sqrt(float64(ohx*ohx + (2*ihy)*(2*ihy) + ohz*ohz)))

	return model.NewModel(
		model.WithName(name),
		model.WithBoundingRadius(br),
		model.WithVertexData(common.SliceToBytes(verts)),
		model.WithIndexData(common.SliceToBytes(indices)),
		model.WithIndexCount(len(indices)),
		model.WithMeshProvider(bind_group_provider.NewBindGroupProvider(name+"_mesh")),
		model.WithRenderMaterials(material.NewMaterial(
			material.WithName(name),
			material.WithPipelineKey(name),
		)),
	)
}

// buildBoxModel creates an axis-aligned box centered at the origin with the given
// half-extents on each axis and a uniform vertex color. The box has 24 vertices
// (4 per face for correct normals) and 36 indices with CCW winding.
//
// Parameters:
//   - name: the unique model name
//   - hx: half-extent along the X axis
//   - hy: half-extent along the Y axis
//   - hz: half-extent along the Z axis
//   - color: RGBA vertex color
//
// Returns:
//   - model.Model: the fully configured Model
func buildBoxModel(name string, hx, hy, hz float32, color [4]float32) model.Model {
	v := func(px, py, pz, nx, ny, nz float32) model.GPUVertex {
		t := tangentFromNormal(nx, ny, nz)
		return model.GPUVertex{
			Position: [3]float32{px, py, pz},
			Normal:   [3]float32{nx, ny, nz},
			Color:    color,
			Tangent:  t,
		}
	}

	verts := []model.GPUVertex{
		// Top face (+Y)
		v(-hx, hy, -hz, 0, 1, 0),
		v(hx, hy, -hz, 0, 1, 0),
		v(hx, hy, hz, 0, 1, 0),
		v(-hx, hy, hz, 0, 1, 0),
		// Bottom face (-Y)
		v(-hx, -hy, -hz, 0, -1, 0),
		v(hx, -hy, -hz, 0, -1, 0),
		v(hx, -hy, hz, 0, -1, 0),
		v(-hx, -hy, hz, 0, -1, 0),
		// Front face (+Z)
		v(-hx, -hy, hz, 0, 0, 1),
		v(hx, -hy, hz, 0, 0, 1),
		v(hx, hy, hz, 0, 0, 1),
		v(-hx, hy, hz, 0, 0, 1),
		// Back face (-Z)
		v(hx, -hy, -hz, 0, 0, -1),
		v(-hx, -hy, -hz, 0, 0, -1),
		v(-hx, hy, -hz, 0, 0, -1),
		v(hx, hy, -hz, 0, 0, -1),
		// Right face (+X)
		v(hx, -hy, hz, 1, 0, 0),
		v(hx, -hy, -hz, 1, 0, 0),
		v(hx, hy, -hz, 1, 0, 0),
		v(hx, hy, hz, 1, 0, 0),
		// Left face (-X)
		v(-hx, -hy, -hz, -1, 0, 0),
		v(-hx, -hy, hz, -1, 0, 0),
		v(-hx, hy, hz, -1, 0, 0),
		v(-hx, hy, -hz, -1, 0, 0),
	}

	indices := []uint32{
		0, 2, 1, 0, 3, 2, // top    CCW from +Y
		4, 5, 6, 4, 6, 7, // bottom CCW from -Y
		8, 9, 10, 8, 10, 11, // front  CCW from +Z
		12, 13, 14, 12, 14, 15, // back   CCW from -Z
		16, 17, 18, 16, 18, 19, // right  CCW from +X
		20, 21, 22, 20, 22, 23, // left   CCW from -X
	}

	br := float32(math.Sqrt(float64(hx*hx + hy*hy + hz*hz)))

	return model.NewModel(
		model.WithName(name),
		model.WithBoundingRadius(br),
		model.WithVertexData(common.SliceToBytes(verts)),
		model.WithIndexData(common.SliceToBytes(indices)),
		model.WithIndexCount(len(indices)),
		model.WithMeshProvider(bind_group_provider.NewBindGroupProvider(name+"_mesh")),
		model.WithRenderMaterials(material.NewMaterial(
			material.WithName(name),
			material.WithPipelineKey(name),
		)),
	)
}

// buildSphereModel creates a UV sphere centered at the origin with outward-facing
// normals and a uniform vertex color.
//
// Parameters:
//   - name: the unique model name
//   - radius: sphere radius
//   - stacks: number of horizontal rings (latitude divisions, minimum 3)
//   - slices: number of vertical segments (longitude divisions, minimum 3)
//   - color: RGBA vertex color
//
// Returns:
//   - model.Model: the fully configured Model
//
// buildSphereModel generates an icosphere Model with the given number of
// subdivision levels, avoiding the pole-singularity artifacts of a UV sphere.
// subdivisions=2 gives 162 verts / 960 tris (good for small instanced spheres).
//
// Parameters:
//   - name: unique identifier for the model and its material/pipeline
//   - radius: sphere radius in world units
//   - subdivisions: number of recursive subdivisions of the base icosahedron
//   - color: RGBA vertex color applied to every vertex
//
// Returns:
//   - model.Model: the constructed icosphere model
func buildSphereModel(name string, radius float32, subdivisions int, color [4]float32) model.Model {
	type edge struct{ a, b uint32 }

	// Start with icosahedron (12 vertices, 20 faces).
	t := float32((1.0 + math.Sqrt(5.0)) / 2.0) // golden ratio
	baseVerts := [][3]float32{
		{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
		{0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
		{t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1},
	}
	// Normalize base vertices to unit sphere.
	for i := range baseVerts {
		l := float32(math.Sqrt(float64(baseVerts[i][0]*baseVerts[i][0] +
			baseVerts[i][1]*baseVerts[i][1] +
			baseVerts[i][2]*baseVerts[i][2])))
		baseVerts[i][0] /= l
		baseVerts[i][1] /= l
		baseVerts[i][2] /= l
	}

	baseTris := [][3]uint32{
		{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
		{1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
		{3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
		{4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1},
	}

	positions := make([][3]float32, len(baseVerts))
	copy(positions, baseVerts)
	triangles := make([][3]uint32, len(baseTris))
	copy(triangles, baseTris)

	// Subdivide: split each triangle into 4 by inserting midpoints on edges.
	midpointCache := make(map[edge]uint32)
	getMidpoint := func(a, b uint32) uint32 {
		e := edge{a, b}
		if a > b {
			e = edge{b, a}
		}
		if idx, ok := midpointCache[e]; ok {
			return idx
		}
		mid := [3]float32{
			(positions[a][0] + positions[b][0]) * 0.5,
			(positions[a][1] + positions[b][1]) * 0.5,
			(positions[a][2] + positions[b][2]) * 0.5,
		}
		// Project onto unit sphere.
		l := float32(math.Sqrt(float64(mid[0]*mid[0] + mid[1]*mid[1] + mid[2]*mid[2])))
		mid[0] /= l
		mid[1] /= l
		mid[2] /= l
		idx := uint32(len(positions))
		positions = append(positions, mid)
		midpointCache[e] = idx
		return idx
	}

	for s := 0; s < subdivisions; s++ {
		var next [][3]uint32
		midpointCache = make(map[edge]uint32) // reset cache per level
		for _, tri := range triangles {
			a := getMidpoint(tri[0], tri[1])
			b := getMidpoint(tri[1], tri[2])
			c := getMidpoint(tri[2], tri[0])
			next = append(next,
				[3]uint32{tri[0], a, c},
				[3]uint32{tri[1], b, a},
				[3]uint32{tri[2], c, b},
				[3]uint32{a, b, c},
			)
		}
		triangles = next
	}

	// Build GPU vertices with smooth (per-vertex) normals. For a sphere on a
	// unit icosphere, the smooth normal at each vertex is simply the normalized
	// position — pointing radially outward. Vertices are shared across adjacent
	// triangles via indexing, giving a smooth-shaded appearance.
	verts := make([]model.GPUVertex, 0, len(positions))
	for _, p := range positions {
		// Scale to radius; normal = normalized position (radially outward)
		pos := [3]float32{p[0] * radius, p[1] * radius, p[2] * radius}
		n := p // already unit-length from subdivision
		tng := tangentFromNormal(n[0], n[1], n[2])
		verts = append(verts, model.GPUVertex{
			Position: pos,
			Normal:   n,
			Color:    color,
			Tangent:  tng,
		})
	}

	indices := make([]uint32, 0, len(triangles)*3)
	for _, tri := range triangles {
		indices = append(indices, tri[0], tri[1], tri[2])
	}

	return model.NewModel(
		model.WithName(name),
		model.WithBoundingRadius(radius),
		model.WithCastsShadows(true),
		model.WithShadowCullMode(model.ShadowCullModeFront),
		model.WithVertexData(common.SliceToBytes(verts)),
		model.WithIndexData(common.SliceToBytes(indices)),
		model.WithIndexCount(len(indices)),
		model.WithMeshProvider(bind_group_provider.NewBindGroupProvider(name+"_mesh")),
		model.WithRenderMaterials(material.NewMaterial(
			material.WithName(name),
			material.WithPipelineKey(name),
		)),
	)
}

// dropletEntry tracks one live droplet in the circular spawn buffer.
type dropletEntry struct {
	objID uint64
	body  physics.RigidBody
}

// setupFluidInput wires camera controls and implements the streaming droplet
// spawner. Every spawnInterval ticks (60 Hz / 2 = 30 Hz), spawnPerTick new
// droplets are created via Scene.Add at the spawn point above the box. When live count reaches
// maxDroplets, the oldest droplets are removed via Scene.Remove before
// new ones are spawned — a circular FIFO. Gravity is applied to all live
// bodies every tick.
//
// Parameters:
//   - eng: the engine instance for tick callbacks and input
//   - cam: the camera for orbit/pan/zoom controls
//   - sc: the scene to add/remove droplets from
//   - ph: the physics handler for diagnostic logging
//   - sphereModel: the shared sphere Model for all droplets
//   - sphereParticles: the DEM particle definition for each droplet body
func setupFluidInput(
	eng engine.Engine,
	cam camera.Camera,
	sc scene.Scene,
	ph physics.Physics,
	sphereModel model.Model,
	sphereParticles []physics.Particle,
) {
	keyState := make(map[uint32]bool)
	spawning := true

	eng.Window().SetKeyDownCallback(func(keyCode uint32) {
		keyState[keyCode] = true
		if keyCode == common.KeySpace {
			spawning = !spawning
			if spawning {
				fmt.Println("[Spawn] ON")
			} else {
				fmt.Println("[Spawn] OFF")
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

	// Circular FIFO of live droplets
	liveDroplets := make([]dropletEntry, 0, maxDroplets+spawnPerTick)
	var tickCounter int

	eng.SetTickCallback(func(_ float32) {
		if spawning && tickCounter%spawnInterval == 0 {
			// Remove oldest droplets if we're at capacity
			removeCount := len(liveDroplets) + spawnPerTick - maxDroplets
			if removeCount > 0 {
				for i := 0; i < removeCount && len(liveDroplets) > 0; i++ {
					oldest := liveDroplets[0]
					sc.Remove(oldest.objID)
					liveDroplets = liveDroplets[1:]
				}
			}

			// Spawn new droplets at the source point above the box
			for i := 0; i < spawnPerTick; i++ {
				// Random position within a small circular area at spawnY
				angle := rand.Float64() * 2 * math.Pi
				dist := rand.Float64() * float64(spawnRadius)
				px := float32(math.Cos(angle)*dist) + float32(rand.Float64()*2-1)*jitterAmp
				// Stagger spawn Y across the batch so particles within a single
				// tick don't overlap. Each particle is offset by one full diameter
				// (2 × particleRadius) from the previous.
				diameter := float32(particleRadius * 2.0)
				py := float32(spawnY) + float32(i)*diameter + float32(rand.Float64()*2-1)*jitterAmp
				pz := float32(math.Sin(angle)*dist) + float32(rand.Float64()*2-1)*jitterAmp

				// Initial downward velocity sized so successive spawns are
				// spaced at least one diameter apart:
				// gap = v × (spawnInterval / tickRate) = 12 × (2/60) = 0.40 m > 0.30 m diameter.
				// Gravity handles additional acceleration after spawn.
				// Random angular velocity on spawn so each droplet visually
				// spins. Without this, vertically-falling droplets produce
				// zero tangential velocity at floor contact, meaning shear
				// friction generates no torque and the spheres never rotate.
				angVel := [3]float32{
					float32(rand.Float64()*10 - 5),
					float32(rand.Float64()*10 - 5),
					float32(rand.Float64()*10 - 5),
				}

				rb := physics.NewRigidBody(
					physics.WithMass(dropletMass),
					physics.WithBounce(0.05),
					physics.WithFriction(0.4),
					physics.WithActive(true),
					physics.WithVelocity([3]float32{0, -12.0, 0}),
					physics.WithAngularVelocity(angVel),
					physics.WithParticles(sphereParticles),
					physics.WithParticleRadius(particleRadius),
				)

				obj := game_object.NewGameObject(
					game_object.WithModel(sphereModel),
					game_object.WithPosition(px, py, pz),
					game_object.WithScale(1, 1, 1),
					game_object.WithRigidBody(rb),
				)
				objID := sc.Add(obj)
				liveDroplets = append(liveDroplets, dropletEntry{objID: objID, body: rb})
			}
		}

		// Every second (60 ticks), request a GPU→CPU readback of body positions
		// so we can detect escaped droplets on the next cycle.
		tickCounter++
		if tickCounter%60 == 0 {
			ph.RequestReadback()
		}

		// After a readback has been processed (positions updated on CPU-side
		// RigidBodies), sweep the live list and remove any droplet whose
		// Y position has fallen below -5. This prevents escaped particles
		// from poisoning the spatial grid's AABB.
		if tickCounter%60 == 30 {
			n := 0
			escaped := 0
			for _, d := range liveDroplets {
				pos := d.body.Position()
				if pos[1] < -5.0 {
					sc.Remove(d.objID)
					escaped++
				} else {
					liveDroplets[n] = d
					n++
				}
			}
			liveDroplets = liveDroplets[:n]
			if escaped > 0 {
				fmt.Printf("[Cleanup] Removed %d escaped droplets (Y < -5)\n", escaped)
			}
		}

		// Per-second diagnostic log (every 60 ticks at 60 Hz)
		if tickCounter%60 == 0 {
			fmt.Printf("[Stats] live=%d  bodies=%d  particles=%d  sceneObjs=%d\n",
				len(liveDroplets), ph.BodiesCount(), ph.ParticleCount(), sc.Count())
		}

		// Camera controls
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
