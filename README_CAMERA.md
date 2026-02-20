# Oxy Camera System

The `camera` package provides the camera and camera controller systems for the Oxy engine. A `Camera` holds perspective settings (FOV, aspect ratio, near/far planes) and computes view/projection matrices each frame by reading positional state from an attached `CameraController`.

---

## Table of Contents

- [Overview](#overview)
- [Camera](#camera)
  - [Creating a Camera](#creating-a-camera)
  - [Camera Builder Options](#camera-builder-options)
  - [Camera Interface](#camera-interface)
- [CameraController](#cameracontroller)
  - [Creating a Controller](#creating-a-controller)
  - [Controller Builder Options](#controller-builder-options)
  - [CameraController Interface](#cameracontroller-interface)
  - [Orbit Controls](#orbit-controls)
  - [Planar Controls](#planar-controls)
- [GPU Types](#gpu-types)
- [Usage Example](#usage-example)

---

## Overview

The camera system is split into two concerns:

1. **Camera** — Owns perspective parameters and matrices. Reads position/target from a controller each frame via `Update()`.
2. **CameraController** — Owns positional state (position, target) and provides orbit and planar control methods. Supports both third-person orbit controls (spherical coordinates around a pivot) and first-person-style planar panning simultaneously.

This separation allows the same controller to be reused or swapped without recreating the camera, and keeps input handling decoupled from matrix computation.

---

## Camera

### Creating a Camera

```go
cam := camera.NewCamera(
    camera.WithFov(45.0 * (math.Pi / 180.0)),
    camera.WithAspect(16.0 / 9.0),
    camera.WithNear(0.1),
    camera.WithFar(1000.0),
    camera.WithController(ctrl),
)
```

The constructor applies sensible defaults before processing options:

| Parameter    | Default            |
| ------------ | ------------------ |
| Up vector    | `(0, 1, 0)`        |
| FOV          | `45°` (in radians) |
| Aspect ratio | `1.0`              |
| Near plane   | `0.1`              |
| Far plane    | `100.0`            |
| Controller   | `nil`              |

A `BindGroupProvider` is created automatically with a unique name (`camera_0`, `camera_1`, …).

### Camera Builder Options

All options follow the `CameraBuilderOption` functional option pattern.

| Option                  | Parameters                   | Description                                    |
| ----------------------- | ---------------------------- | ---------------------------------------------- |
| `WithUp`                | `x, y, z float32`            | Sets the camera's up vector                    |
| `WithFov`               | `fov float32`                | Sets the field of view in radians              |
| `WithAspect`            | `aspect float32`             | Sets the aspect ratio (width / height)         |
| `WithNear`              | `near float32`               | Sets the near clipping plane distance          |
| `WithFar`               | `far float32`                | Sets the far clipping plane distance           |
| `WithController`        | `ctrl CameraController`      | Attaches a camera controller                   |
| `WithBindGroupProvider` | `provider BindGroupProvider` | Overrides the auto-created bind group provider |

### Camera Interface

```go
type Camera interface {
    // Getters
    Up() (x, y, z float32)
    Fov() float32
    Aspect() float32
    Near() float32
    Far() float32
    ViewMatrix() [16]float32
    ProjectionMatrix() [16]float32
    ViewProjectionMatrix() [16]float32
    InverseProjectionMatrix() [16]float32
    Controller() CameraController
    BindGroupProvider() bind_group_provider.BindGroupProvider

    // Setters (recompute matrices where applicable)
    SetUp(x, y, z float32)
    SetFov(fov float32)
    SetAspect(aspect float32)
    SetNear(near float32)
    SetFar(far float32)
    SetController(ctrl CameraController)
    SetBindGroupProvider(provider bind_group_provider.BindGroupProvider)

    // Per-frame update
    Update()
}
```

**Key methods:**

| Method                      | Description                                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `Update()`                  | Reads position/target from the attached controller and recomputes all matrices. Call once per frame. No-op if no controller is attached. |
| `ViewMatrix()`              | Returns the 4×4 view matrix (column-major)                                                                                               |
| `ProjectionMatrix()`        | Returns the 4×4 perspective projection matrix (column-major)                                                                             |
| `ViewProjectionMatrix()`    | Returns the combined view-projection matrix (column-major)                                                                               |
| `InverseProjectionMatrix()` | Returns the inverse projection matrix, used by the Forward+ light culling compute shader                                                 |

---

## CameraController

### Creating a Controller

```go
ctrl := camera.NewCameraController(
    camera.WithRadius(250.0),
    camera.WithAzimuth(0.0),
    camera.WithElevation(math.Pi / 6),
    camera.WithTarget(0, 0, 0),
    camera.WithRadiusBounds(20.0, 2000.0),
    camera.WithOrbitSpeed(0.03),
    camera.WithMouseSensitivity(0.005),
    camera.WithZoomSpeed(15.0),
    camera.WithPanSpeed(1.0),
)
```

Defaults applied before options:

| Parameter         | Default            |
| ----------------- | ------------------ |
| Target            | `(0, 0, 0)`        |
| Radius            | `250.0`            |
| Azimuth           | `0.0`              |
| Elevation         | `π/6` (~30°)       |
| Min radius        | `20.0`             |
| Max radius        | `2000.0`           |
| Min elevation     | `0.05`             |
| Max elevation     | `π/2 - 0.1` (~80°) |
| Orbit speed       | `0.03` rad/step    |
| Mouse sensitivity | `0.005`            |
| Zoom speed        | `15.0`             |
| Pan speed         | `1.0`              |

### Controller Builder Options

All options follow the `CameraControllerOption` functional option pattern.

| Option                 | Parameters            | Description                                        |
| ---------------------- | --------------------- | -------------------------------------------------- |
| `WithRadius`           | `radius float32`      | Initial orbit distance from target                 |
| `WithAzimuth`          | `azimuth float32`     | Initial horizontal angle in radians (0 = +Z axis)  |
| `WithElevation`        | `elevation float32`   | Initial vertical angle in radians (0 = horizontal) |
| `WithTarget`           | `x, y, z float32`     | Look-at / pivot point                              |
| `WithRadiusBounds`     | `min, max float32`    | Min and max zoom distance                          |
| `WithElevationBounds`  | `min, max float32`    | Min and max elevation angles in radians            |
| `WithOrbitSpeed`       | `speed float32`       | Keyboard orbit speed (radians per step)            |
| `WithMouseSensitivity` | `sensitivity float32` | Mouse drag sensitivity multiplier                  |
| `WithZoomSpeed`        | `speed float32`       | Zoom input multiplier                              |
| `WithPanSpeed`         | `speed float32`       | Pan input multiplier                               |

### CameraController Interface

The `CameraController` interface is a union of shared controls, orbit controls, and planar controls:

```go
type CameraController interface {
    orbitCameraController
    planarCameraController

    Position() (x, y, z float32)
    Target() (x, y, z float32)
    SetTarget(x, y, z float32)
    SetPosition(x, y, z float32)
    Zoom(delta float32)
}
```

| Method                 | Description                                                                     |
| ---------------------- | ------------------------------------------------------------------------------- |
| `Position()`           | Returns the camera's computed world-space position                              |
| `Target()`             | Returns the current look-at point                                               |
| `SetTarget(x, y, z)`   | Sets the look-at/pivot point and recomputes position from spherical coordinates |
| `SetPosition(x, y, z)` | Sets the camera position directly (does not update spherical coordinates)       |
| `Zoom(delta)`          | Adjusts orbit radius; positive delta zooms in, clamped to min/max bounds        |

### Orbit Controls

Orbit methods modify spherical coordinates (radius, azimuth, elevation) relative to the target and recompute the camera position.

| Method                              | Description                                                       |
| ----------------------------------- | ----------------------------------------------------------------- |
| `OrbitLeft()`                       | Rotates left by one orbit speed step                              |
| `OrbitRight()`                      | Rotates right by one orbit speed step                             |
| `OrbitUp()`                         | Tilts upward by one orbit speed step (clamped to max elevation)   |
| `OrbitDown()`                       | Tilts downward by one orbit speed step (clamped to min elevation) |
| `Radius()` / `SetRadius(r)`         | Get/set orbit distance (clamped to bounds)                        |
| `Azimuth()` / `SetAzimuth(a)`       | Get/set horizontal angle in radians                               |
| `Elevation()` / `SetElevation(e)`   | Get/set vertical angle in radians (clamped to bounds)             |
| `MinRadius()` / `MaxRadius()`       | Query radius bounds                                               |
| `MinElevation()` / `MaxElevation()` | Query elevation bounds                                            |
| `OrbitSpeed()`                      | Keyboard orbit speed (radians/step)                               |
| `MouseSensitivity()`                | Mouse drag multiplier                                             |
| `ZoomSpeed()`                       | Zoom input multiplier                                             |

### Planar Controls

Planar methods translate both position and target along the camera's local axes, preserving the orbit relationship. This enables first-person-style panning without disrupting orbit angles.

| Method              | Description                                                                |
| ------------------- | -------------------------------------------------------------------------- |
| `PanRight(delta)`   | Translates along the local right axis (positive = right)                   |
| `PanUp(delta)`      | Translates along the local up axis (positive = up)                         |
| `PanForward(delta)` | Translates along the local forward axis / dolly (positive = toward target) |
| `PanSpeed()`        | Pan input multiplier                                                       |

---

## GPU Types

The package provides a GPU-aligned uniform struct for uploading camera data to shaders.

### GPUCameraUniform

| Field            | Type          | Offset | Description                                     |
| ---------------- | ------------- | ------ | ----------------------------------------------- |
| `ViewProj`       | `[16]float32` | 0      | Combined view-projection matrix (`mat4x4<f32>`) |
| `CameraPosition` | `[3]float32`  | 64     | World-space camera position (`vec3<f32>`)       |
| `_pad`           | `float32`     | 76     | Padding to 80 bytes                             |

**Total size:** 80 bytes (std430 / WGSL aligned)

The corresponding WGSL struct definition is embedded via `GPUCameraUniformSource` and can be injected into shaders using the `@oxy:include camera` annotation (see [Shader Annotation System](README_ANNOTATIONS.md)).

```go
uniform := &camera.GPUCameraUniform{
    ViewProj:       cam.ViewProjectionMatrix(),
    CameraPosition: [3]float32{px, py, pz},
}
buf := uniform.Marshal() // []byte ready for GPU upload
```

---

## Usage Example

A typical setup combining camera and controller in an engine tick loop:

```go
package main

import (
    "math"

    "github.com/Carmen-Shannon/oxy-go/engine/camera"
)

func main() {
    // Create a controller with orbit defaults
    ctrl := camera.NewCameraController(
        camera.WithRadius(10.0),
        camera.WithTarget(0, 1, 0),
        camera.WithElevation(float32(math.Pi / 6)),
    )

    // Create a camera attached to the controller
    cam := camera.NewCamera(
        camera.WithFov(60.0 * (math.Pi / 180.0)),
        camera.WithAspect(16.0 / 9.0),
        camera.WithNear(0.1),
        camera.WithFar(500.0),
        camera.WithController(ctrl),
    )

    // Each frame:
    cam.Update() // recomputes matrices from controller state

    // Read matrices for GPU upload
    _ = cam.ViewProjectionMatrix()

    // Orbit controls (e.g. from key input)
    ctrl.OrbitLeft()
    ctrl.Zoom(1.0) // zoom in

    // Pan controls (e.g. from mouse drag)
    ctrl.PanRight(2.0)
    ctrl.PanUp(1.0)
}
```
