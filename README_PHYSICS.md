# Oxy Physics System

The `physics` package implements a GPU-accelerated rigid body physics simulation built on the Discrete Element Method (DEM) described in [GPU Gems 3, Chapter 29](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus). Rigid bodies are decomposed into spherical particles, and all collision detection, force computation, and integration run entirely on the GPU via WebGPU compute shaders. A fixed-timestep accumulator on the CPU drives per-frame dispatches while asynchronous readback keeps CPU-side state in sync.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Simulation Pipeline](#simulation-pipeline)
- [RigidBody](#rigidbody)
  - [Creating a RigidBody](#creating-a-rigidbody)
  - [RigidBody Builder Options](#rigidbody-builder-options)
  - [RigidBody Interface](#rigidbody-interface)
- [Particle](#particle)
  - [VoxelizeMesh](#voxelizemesh)
  - [AssignBoneIndices](#assignboneindices)
- [Constants](#constants)
- [Physics Controller](#physics-controller)
  - [Creating a Physics Controller](#creating-a-physics-controller)
  - [Physics Builder Options](#physics-builder-options)
  - [Physics Interface](#physics-interface)
- [GPU Types](#gpu-types)
  - [GPUBody](#gpubody)
  - [GPUParticle](#gpuparticle)
  - [GPUGridCell](#gpugridcell)
  - [GPUPhysicsGlobals](#gpuphysicsglobals)
  - [GPUGridParams](#gpugridparams)
- [WGSL Assets](#wgsl-assets)
- [Usage Example](#usage-example)

---

## Overview

The physics system is built around three core concepts:

1. **RigidBody** — A physical object defined by mass, restitution, friction, and a set of spherical collision particles. Rigid bodies can be dynamic, static (immovable), or kinematic (script/animation-driven).
2. **Particle** — A spherical collision element composing a rigid body. Particles are generated from mesh geometry via ray-cast voxelization and carry body-local position offsets and surface normals.
3. **Physics** — The controller that owns GPU buffers, manages body registration/removal, drives the fixed-timestep accumulator, stages buffer writes, and processes asynchronous readback of simulation results.

The entire collision and integration pipeline executes on the GPU. The CPU is responsible only for body registration, force/torque application, timestep control, and readback processing.

---

## Architecture

```
physics/
├── physics.go              Physics controller interface
├── physics_impl.go         Unexported physicsImpl struct and method bodies
├── physics_builder.go      PhysicsBuilderOption functional options
├── rigid_body.go           RigidBody interface
├── rigid_body_impl.go      Unexported rigidBody struct and method bodies
├── rigid_body_builder.go   RigidBodyOption functional options
├── particle.go             Particle struct, VoxelizeMesh, AssignBoneIndices
├── gpu_types.go            GPU-aligned structs (GPUBody, GPUParticle, etc.)
└── assets/                 WGSL compute shader sources
    ├── body.wgsl               Body struct definition
    ├── particle.wgsl           Particle struct definition
    ├── physics-globals.wgsl    PhysicsGlobals uniform definition
    ├── grid-params.wgsl        GridParams struct definition
    ├── grid-cell.wgsl          GridCell struct definition
    ├── particle-values.wgsl    Particle world-position and velocity update
    ├── aabb-reduce.wgsl        AABB reduction for spatial grid bounds
    ├── grid-build-params.wgsl  Grid dimension computation from AABB
    ├── grid-clear.wgsl         Grid cell reset
    ├── grid-insert.wgsl        Particle-to-cell insertion
    ├── collision-reaction.wgsl DEM collision force computation
    ├── compute-momenta.wgsl    Force → momentum accumulation per body
    ├── integrate.wgsl          Symplectic Euler integration
    ├── bone-particle-update.wgsl Kinematic bone-driven particle update
    └── physics-sync.wgsl       Body position/quaternion → Animator sync
```

---

## Simulation Pipeline

Each frame the physics system runs a fixed-timestep loop. `PrepareStep(dt)` advances the accumulator and returns the number of substeps to dispatch (capped at `maxSubsteps` to avoid the spiral-of-death). For each substep the Scene dispatches the following compute stages in order:

| Stage | Shader                 | Description                                                                                  |
| ----- | ---------------------- | -------------------------------------------------------------------------------------------- |
| 1     | `particle-values`      | Update each particle's world position and velocity from its parent body's state              |
| 2     | `aabb-reduce`          | Compute the global AABB of all active particles                                              |
| 3     | `grid-build-params`    | Derive grid origin and cell dimensions from the AABB and particle diameter                   |
| 4     | `grid-clear`           | Zero all grid cells                                                                          |
| 5     | `grid-insert`          | Insert each particle into its spatial grid cell                                              |
| 6     | `collision-reaction`   | DEM spring/damping/shear force computation per particle pair, plus boundary plane collisions |
| 7     | `compute-momenta`      | Accumulate per-particle forces into per-body linear and angular momentum                     |
| 8     | `integrate`            | Symplectic Euler integration of position, quaternion, and momenta                            |
| 9     | `bone-particle-update` | (Kinematic bodies only) Transform bone-local particles to world space using bone matrices    |
| 10    | `physics-sync`         | Write body positions and quaternions back to the Animator instance buffer                    |

After the substep loop completes, the Scene optionally initiates an asynchronous GPU readback of the body buffer so CPU-side `RigidBody` state (position, quaternion) stays in sync.

---

## RigidBody

### Creating a RigidBody

```go
rb := physics.NewRigidBody(
    physics.WithMass(5.0),
    physics.WithBounce(0.4),
    physics.WithFriction(0.6),
    physics.WithParticleRadius(0.05),
    physics.WithParticles(particles), // from VoxelizeMesh
)
```

Defaults applied before options:

| Parameter   | Default |
| ----------- | ------- |
| Mass        | `1.0`   |
| Bounce      | `0.5`   |
| Friction    | `0.5`   |
| SurfaceOnly | `true`  |
| Active      | `false` |
| Static      | `false` |
| Kinematic   | `false` |

If `mass` is zero and the body is not kinematic it is automatically marked static. The inverse inertia tensor is computed from the particle layout when particles are provided. Single-particle bodies fall back to the analytical inertia of a solid sphere ($I = \frac{2}{5} m r^2$).

### RigidBody Builder Options

| Option                | Type         | Description                          |
| --------------------- | ------------ | ------------------------------------ |
| `WithMass`            | `float32`    | Body mass                            |
| `WithInverseMass`     | `float32`    | Explicit inverse mass override       |
| `WithBounce`          | `float32`    | Coefficient of restitution           |
| `WithFriction`        | `float32`    | Friction coefficient                 |
| `WithVelocity`        | `[3]float32` | Initial linear velocity              |
| `WithAngularVelocity` | `[3]float32` | Initial angular velocity             |
| `WithActive`          | `bool`       | Whether the body is active           |
| `WithStatic`          | `bool`       | Whether the body is immovable        |
| `WithKinematic`       | `bool`       | Whether the body is animation-driven |
| `WithParticles`       | `[]Particle` | Collision particle set               |
| `WithParticleRadius`  | `float32`    | Radius of each spherical particle    |

### RigidBody Interface

The `RigidBody` interface embeds `common.Delegate[RigidBody]` and exposes the following methods:

**Properties**

| Method                       | Returns      | Description                                             |
| ---------------------------- | ------------ | ------------------------------------------------------- |
| `Position()`                 | `[3]float32` | World-space position (updated from GPU readback)        |
| `Quaternion()`               | `[4]float32` | Orientation quaternion XYZW (updated from GPU readback) |
| `Velocity()`                 | `[3]float32` | Linear velocity                                         |
| `AngularVelocity()`          | `[3]float32` | Angular velocity                                        |
| `Mass()`                     | `float32`    | Body mass                                               |
| `InverseMass()`              | `float32`    | Inverse of mass                                         |
| `Bounce()`                   | `float32`    | Coefficient of restitution                              |
| `Friction()`                 | `float32`    | Friction coefficient                                    |
| `InverseInertiaTensorBody()` | `[9]float32` | Body-space inverse inertia tensor (3×3 flattened)       |
| `Active()`                   | `bool`       | Whether the body is active                              |
| `Static()`                   | `bool`       | Whether the body is immovable                           |
| `Kinematic()`                | `bool`       | Whether the body is animation-driven                    |
| `Particles()`                | `[]Particle` | Collision particle set                                  |
| `ParticleRadius()`           | `float32`    | Radius of each particle                                 |
| `SurfaceOnly()`              | `bool`       | Whether only surface particles are generated            |

**Setters**

All property getters have corresponding `Set*` methods. Additionally:

| Method                | Parameters   | Description                                 |
| --------------------- | ------------ | ------------------------------------------- |
| `ApplyForce(force)`   | `[3]float32` | Accumulate a force vector (goroutine-safe)  |
| `ApplyTorque(torque)` | `[3]float32` | Accumulate a torque vector (goroutine-safe) |

`ApplyForce` and `ApplyTorque` are safe to call from any goroutine (e.g. the engine tick callback). Accumulated forces and torques are drained by the physics controller each frame and uploaded as `GPUBody.ExternalForce` and `GPUBody.ExternalTorque`.

---

## Particle

```go
type Particle struct {
    LocalPosition [3]float32  // Body-local position offset
    SurfaceNormal [3]float32  // Outward-facing surface normal (surface-only particles)
    BodyIndex     uint32      // Owning body index (assigned during registration)
    BoneIndex     uint32      // Bone index for kinematic skeletal bodies
}
```

### VoxelizeMesh

```go
func VoxelizeMesh(src model.Model, particleRadius float32, surfaceOnly bool) []Particle
```

Converts a Model's mesh geometry into spherical particles via ray-cast voxelization following the method described in [GPU Gems 3, §29.1.3](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus).

The algorithm:

1. Extracts vertex positions and triangle indices from the Model's retained byte data
2. Computes the axis-aligned bounding box (AABB) of the mesh
3. Subdivides the AABB into a uniform 3D grid with cell size equal to the particle diameter
4. For each (Y, Z) column, casts a ray along +X and collects triangle intersections using the Möller–Trumbore algorithm
5. Parity of crossings determines inside/outside status for each voxel
6. When `surfaceOnly` is true, an additional filter removes interior voxels, keeping only those adjacent to at least one empty neighbor

The column ray-casting phase is parallelized across CPU cores. Surface normals are computed as the average direction toward empty neighbors, enabling the collision shader to prevent reversed spring forces.

### AssignBoneIndices

```go
func AssignBoneIndices(particles []Particle, src model.Model)
```

Associates each voxel particle with its nearest bone from a skinned model. For each particle, the nearest vertex is found by squared distance, and that vertex's primary bone (highest blend weight) is assigned. The particle's position is then transformed from model space to bone-local space using the bone's inverse bind matrix, enabling per-frame bone-driven updates in the `bone-particle-update` compute shader.

---

## Constants

### PhysicsState

The `PhysicsState` type is a bitmask used to classify a rigid body's simulation mode.

```go
type PhysicsState int

const (
    PhysicsStateActive    PhysicsState = 1 // Body participates in full simulation
    PhysicsStateStatic    PhysicsState = 2 // Body is immovable (collides but never moves)
    PhysicsStateKinematic PhysicsState = 4 // Body is animation-driven (not gravity-affected)
)
```

---

## Physics Controller

### Creating a Physics Controller

```go
phys := physics.NewPhysics(
    physics.WithFixedDt(1.0 / 60.0),
    physics.WithMaxSubsteps(4),
    physics.WithMaxBodies(256),
    physics.WithMaxParticles(2048),
    physics.WithMaxGridCells(128 * 128 * 128),
    physics.WithGravity([3]float32{0, -9.81, 0}),
    physics.WithSpringCoeff(1.0),
    physics.WithDampingCoeff(0.1),
    physics.WithShearCoeff(0.5),
    physics.WithBoundaryPlanes([][6]float32{
        {0, 1, 0, 0, -100, 100},  // ground plane
    }),
)
```

Defaults applied before options:

| Parameter     | Default           |
| ------------- | ----------------- |
| FixedDt       | `1/60` (60 Hz)    |
| MaxSubsteps   | `4`               |
| MaxBodies     | `256`             |
| MaxParticles  | `2048`            |
| MaxGridCells  | `128 × 128 × 128` |
| SpringCoeff   | `1.0`             |
| DampingCoeff  | `0.1`             |
| ShearCoeff    | `0.5`             |
| Gravity       | `{0, 0, 0}`       |
| BoundaryCount | `0`               |

The system starts disabled and becomes active upon the first `RegisterBody` call.

### Physics Builder Options

| Option               | Type           | Description                                                     |
| -------------------- | -------------- | --------------------------------------------------------------- |
| `WithFixedDt`        | `float32`      | Fixed timestep in seconds                                       |
| `WithMaxSubsteps`    | `int`          | Max substeps per frame (spiral-of-death cap)                    |
| `WithMaxBodies`      | `int`          | Maximum rigid body count (GPU buffer size)                      |
| `WithMaxParticles`   | `int`          | Maximum particle count (GPU buffer size)                        |
| `WithMaxGridCells`   | `int`          | Maximum spatial grid cells (x×y×z cap)                          |
| `WithSpringCoeff`    | `float32`      | DEM spring coefficient $k$                                      |
| `WithDampingCoeff`   | `float32`      | DEM damping coefficient $\eta$                                  |
| `WithShearCoeff`     | `float32`      | DEM tangential friction coefficient $\mu_t$                     |
| `WithGravity`        | `[3]float32`   | Gravitational acceleration vector (m/s²)                        |
| `WithBoundaryPlanes` | `[][6]float32` | Up to 6 containment half-planes `[nx, ny, nz, d, y_min, y_max]` |

### Physics Interface

**State Queries**

| Method            | Returns  | Description                                                                 |
| ----------------- | -------- | --------------------------------------------------------------------------- |
| `Enabled()`       | `bool`   | Whether the physics system is active                                        |
| `BodiesCount()`   | `int`    | Number of allocated body slots                                              |
| `ParticleCount()` | `int`    | Total particles across all bodies                                           |
| `MaxBodies()`     | `int`    | Maximum body capacity                                                       |
| `MaxParticles()`  | `int`    | Maximum particle capacity                                                   |
| `MaxGridCells()`  | `int`    | Maximum grid cell count                                                     |
| `SlotsPerCell()`  | `uint32` | Returns the number of particle slots per grid cell                          |
| `BodyIdxMask()`   | `uint32` | Returns the bitmask used to extract body indices from packed grid cell data |

**Buffer Management**

| Method                      | Returns                        | Description                             |
| --------------------------- | ------------------------------ | --------------------------------------- |
| `Buffers()`                 | `BindGroupProvider`            | Primary physics storage buffer provider |
| `Bgp(key)`                  | `BindGroupProvider`            | Stage-specific bind group provider      |
| `Bgps()`                    | `map[string]BindGroupProvider` | All compute-stage providers             |
| `PipelineKey(key)`          | `string`                       | Compute pipeline cache key for a stage  |
| `SetPipelineKey(name, key)` | —                              | Associate a pipeline key with a stage   |

**Body Registration**

| Method             | Parameters                                  | Returns        | Description                                            |
| ------------------ | ------------------------------------------- | -------------- | ------------------------------------------------------ |
| `RegisterBody`     | `objID, position, rotation, rb, instanceID` | `int`          | Register a body, upload GPU data, return slot index    |
| `RemoveBody`       | `objID`                                     | —              | Deactivate a body and return its slot to the free list |
| `BodyIndex`        | `objID`                                     | `int, bool`    | Look up the slot index for an object ID                |
| `BodyParticleInfo` | `bodyIndex`                                 | `start, count` | Particle range for a body slot                         |

**Simulation Loop**

| Method            | Parameters | Returns                 | Description                                        |
| ----------------- | ---------- | ----------------------- | -------------------------------------------------- |
| `PrepareStep`     | `dt`       | `substeps, globalsData` | Advance accumulator, drain forces, build globals   |
| `StagedWriteData` | —          | `[]BufferWrite`         | Drain pending GPU buffer writes                    |
| `ProcessReadback` | `data`     | —                       | Unmarshal body positions/quaternions from readback |

**Readback Control**

| Method                     | Description                                      |
| -------------------------- | ------------------------------------------------ |
| `RequestReadback()`        | Flag that a GPU→CPU readback should be initiated |
| `ConsumeReadbackRequest()` | Check and clear the readback request flag        |
| `ReadbackPending()`        | Whether a readback copy is in-flight             |
| `ClearReadbackPending()`   | Reset the pending flag after mapping completes   |
| `StagingBuffer()`          | Get the GPU staging buffer for readback          |
| `SetStagingBuffer(buf)`    | Set the GPU staging buffer                       |

---

## GPU Types

All GPU types match their corresponding WGSL struct definitions exactly and provide `Size()` and `Marshal()` methods for GPU upload. WGSL source strings are embedded via `//go:embed` directives.

### GPUBody

**Size:** 160 bytes (std430 aligned)

| Field           | Type          | Offset | Description                                       |
| --------------- | ------------- | ------ | ------------------------------------------------- |
| Position        | `vec4<f32>`   | 0      | World-space center of mass                        |
| Quaternion      | `vec4<f32>`   | 16     | Orientation quaternion (xyzw)                     |
| LinearMomentum  | `vec4<f32>`   | 32     | Linear momentum                                   |
| AngularMomentum | `vec4<f32>`   | 48     | Angular momentum                                  |
| InvInertiaTBody | `mat3x3<f32>` | 64     | Body-space inverse inertia tensor                 |
| InverseMass     | `f32`         | 112    | 1/mass (0 for static)                             |
| ParticleStart   | `u32`         | 116    | Start index into particle buffer                  |
| ParticleCount   | `u32`         | 120    | Number of particles                               |
| Flags           | `u32`         | 124    | Bit-packed: active (0), static (1), kinematic (2) |
| ExternalForce   | `vec4<f32>`   | 128    | CPU-uploaded per-frame force                      |
| ExternalTorque  | `vec4<f32>`   | 144    | CPU-uploaded per-frame torque                     |

### GPUParticle

**Size:** 96 bytes (std430 aligned)

| Field         | Type        | Offset | Description                                       |
| ------------- | ----------- | ------ | ------------------------------------------------- |
| WorldPosition | `vec4<f32>` | 0      | Current world-space position                      |
| Velocity      | `vec4<f32>` | 16     | Current velocity                                  |
| RelPosition   | `vec4<f32>` | 32     | Relative position to body center                  |
| Force         | `vec4<f32>` | 48     | Accumulated collision force                       |
| LocalPosition | `vec4<f32>` | 64     | Body-local rest position (w = bitcast body index) |
| SurfaceNormal | `vec4<f32>` | 80     | Outward surface normal (w = 1.0 if valid static)  |

### GPUGridCell

**Size:** 64 bytes (std430 aligned)

| Field    | Type        | Offset | Description                                     |
| -------- | ----------- | ------ | ----------------------------------------------- |
| Indices0 | `vec4<u32>` | 0      | Particle indices slots 0–3 (0xFFFFFFFF = empty) |
| Indices1 | `vec4<u32>` | 16     | Particle indices slots 4–7                      |
| Indices2 | `vec4<u32>` | 32     | Particle indices slots 8–11                     |
| Indices3 | `vec4<u32>` | 48     | Particle indices slots 12–15                    |

### GPUPhysicsGlobals

**Size:** 240 bytes (std140 / WGSL uniform aligned)

| Field            | Type            | Offset | Description                            |
| ---------------- | --------------- | ------ | -------------------------------------- |
| DeltaTime        | `f32`           | 0      | Simulation timestep (seconds)          |
| ParticleDiameter | `f32`           | 4      | Diameter of all particles              |
| SpringCoeff      | `f32`           | 8      | DEM spring coefficient $k$             |
| DampingCoeff     | `f32`           | 12     | DEM damping coefficient $\eta$         |
| ShearCoeff       | `f32`           | 16     | DEM tangential friction $\mu_t$        |
| BodyCount        | `u32`           | 20     | Number of active rigid bodies          |
| ParticleCount    | `u32`           | 24     | Total particle count                   |
| MaxGridCells     | `u32`           | 28     | Maximum grid cells (x×y×z cap)         |
| BoundaryCount    | `u32`           | 32     | Number of active boundary planes (0–6) |
| GravityX/Y/Z     | `f32`           | 36–44  | Gravitational acceleration components  |
| BoundaryPlanes   | `6 × vec4<f32>` | 48     | Half-plane definitions (normal.xyz, d) |
| BoundaryYRanges  | `6 × vec4<f32>` | 144    | Per-plane Y activation range           |

### GPUGridParams

**Size:** 32 bytes (std430 aligned)

Written by the GPU AABB reduction shader rather than uploaded from the CPU.

| Field      | Type        | Offset | Description                      |
| ---------- | ----------- | ------ | -------------------------------- |
| GridOrigin | `vec4<f32>` | 0      | World-space grid origin          |
| GridDims   | `vec4<u32>` | 16     | Grid dimensions (x, y, z, total) |

---

## WGSL Assets

The `assets/` directory contains all WGSL compute shader sources and struct definitions used by the physics system. These are embedded into Go variables via `//go:embed` directives.

**Struct Definitions**

| File                   | Embedded Variable         | Description                   |
| ---------------------- | ------------------------- | ----------------------------- |
| `body.wgsl`            | `GPUBodySource`           | Body struct layout            |
| `particle.wgsl`        | `GPUParticleSource`       | Particle struct layout        |
| `physics-globals.wgsl` | `GPUPhysicsGlobalsSource` | PhysicsGlobals uniform layout |
| `grid-params.wgsl`     | `GPUGridParamsSource`     | GridParams struct layout      |
| `grid-cell.wgsl`       | `GPUGridCellSource`       | GridCell struct layout        |

**Compute Shaders**

| File                        | Pipeline Stage    | Description                                           |
| --------------------------- | ----------------- | ----------------------------------------------------- |
| `particle-values.wgsl`      | particle_values   | Derive world positions and velocities from body state |
| `aabb-reduce.wgsl`          | aabb_reduce       | Parallel AABB reduction across all particles          |
| `grid-build-params.wgsl`    | grid_build_params | Compute grid origin and dimensions                    |
| `grid-clear.wgsl`           | grid_clear        | Reset all grid cell slots to empty sentinel           |
| `grid-insert.wgsl`          | grid_insert       | Hash particles into grid cells                        |
| `collision-reaction.wgsl`   | collision         | DEM spring/damping/shear forces + boundary planes     |
| `compute-momenta.wgsl`      | momenta           | Sum particle forces into body momentum changes        |
| `integrate.wgsl`            | integrate         | Symplectic Euler position/quaternion/momentum update  |
| `bone-particle-update.wgsl` | bone_particle     | Update kinematic bone-driven particles                |
| `physics-sync.wgsl`         | sync              | Copy body transforms to the Animator instance buffer  |

---

## Usage Example

```go
package main

import (
    "github.com/Carmen-Shannon/oxy-go/engine"
    "github.com/Carmen-Shannon/oxy-go/engine/game_object"
    "github.com/Carmen-Shannon/oxy-go/engine/physics"
    "github.com/Carmen-Shannon/oxy-go/engine/window"
)

func main() {
    eng := engine.NewEngine(
        engine.WithWindow(window.NewWindow(
            window.WithTitle("Physics Demo"),
            window.WithWidth(1280),
            window.WithHeight(720),
        )),
    )

    // Create a physics system with gravity and a ground plane
    phys := physics.NewPhysics(
        physics.WithGravity([3]float32{0, -9.81, 0}),
        physics.WithBoundaryPlanes([][6]float32{
            {0, 1, 0, 0, -100, 100}, // ground at y=0
        }),
        physics.WithSpringCoeff(2000.0),
        physics.WithDampingCoeff(5.0),
    )

    // After loading a model, voxelize it into particles
    // particles := physics.VoxelizeMesh(model, 0.05, true)

    // Create the rigid body with the voxelized particles
    // rb := physics.NewRigidBody(
    //     physics.WithMass(5.0),
    //     physics.WithBounce(0.4),
    //     physics.WithFriction(0.6),
    //     physics.WithParticleRadius(0.05),
    //     physics.WithParticles(particles),
    // )

    // Attach to a game object and register with the scene
    // obj := game_object.NewGameObject(
    //     game_object.WithRigidBody(rb),
    // )

    // Apply forces during the tick callback
    eng.SetTickCallback(func(dt float32) {
        // rb.ApplyForce([3]float32{0, 50, 0}) // upward impulse
    })

    eng.Run()
}
```

See [examples/physics_scene.go](examples/physics_scene.go) for a complete runnable physics demo with rigid bodies, lighting, and boundary planes.
