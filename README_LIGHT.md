# Oxy Light System

The `light` package provides the lighting, shadow mapping, and Forward+ tile culling systems for the Oxy engine. It supports directional, point, and spot light types, all sharing a single `Light` interface. Lights are marshaled into GPU storage buffers each frame and evaluated in a tiled Forward+ rendering pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Light Types](#light-types)
- [Creating a Light](#creating-a-light)
- [Builder Options](#builder-options)
- [Light Interface](#light-interface)
  - [Properties](#properties)
  - [Setters](#setters)
- [Forward+ Light Culling](#forward-light-culling)
  - [Constants](#constants)
  - [TileCounts](#tilecounts)
- [Shadow Mapping](#shadow-mapping)
  - [Shadow Constants](#shadow-constants)
- [GPU Types](#gpu-types)
  - [GPULight](#gpulight)
  - [GPULightHeader](#gpulightheader)
  - [GPUShadowData](#gpushadowdata)
  - [GPUShadowUniform](#gpushadowuniform)
  - [GPULightCullUniforms](#gpulightculluniforms)
  - [GPUTileUniforms](#gputileuniforms)
- [Helper Functions](#helper-functions)
- [Usage Example](#usage-example)

---

## Overview

The light system is designed around three pillars:

1. **Light** — A scene-level entity with type, position, direction, color, intensity, range, and cone angles. All three light types share the same interface; type-specific properties return zero values when not applicable.
2. **Forward+ Tile Culling** — The screen is divided into tiles (`TileSize × TileSize` pixels). A compute shader assigns lights to tiles so the fragment shader only evaluates lights relevant to each tile.
3. **Shadow Mapping** — Shadow-casting lights render a depth-only pass each frame. The shadow data (light view-projection, texel size, bias) is uploaded as a GPU uniform for PCF-sampled shadow comparison in the lit fragment shader.

---

## Light Types

```go
type LightType int

const (
    LightTypeDirectional LightType = iota  // Sun/moon — no position, only direction, no attenuation
    LightTypePoint                         // Bulbs/lanterns — emits in all directions from a position
    LightTypeSpot                          // Flashlights/lamps — cone from a position along a direction
)
```

| Type        | Position    | Direction       | Range        | Cone Angles            | Attenuation        |
| ----------- | ----------- | --------------- | ------------ | ---------------------- | ------------------ |
| Directional | N/A         | Light direction | N/A          | N/A                    | None (uniform)     |
| Point       | World-space | N/A             | Max distance | N/A                    | Distance-based     |
| Spot        | World-space | Cone axis       | Max distance | Inner/outer half-angle | Distance + angular |

---

## Creating a Light

```go
// Directional (sun)
sun := light.NewLight(light.LightTypeDirectional,
    light.WithDirection(0, -1, 0.5),
    light.WithColor(1, 0.95, 0.9),
    light.WithIntensity(1.5),
    light.WithCastsShadows(true),
)

// Point
bulb := light.NewLight(light.LightTypePoint,
    light.WithPosition(0, 3, 0),
    light.WithColor(1, 0.8, 0.6),
    light.WithIntensity(2.0),
    light.WithRange(20.0),
)

// Spot
flashlight := light.NewLight(light.LightTypeSpot,
    light.WithPosition(0, 5, 0),
    light.WithDirection(0, -1, 0),
    light.WithColor(1, 1, 1),
    light.WithIntensity(3.0),
    light.WithRange(30.0),
    light.WithSpotCone(25, 35), // inner/outer half-angles in degrees
)
```

Defaults applied before options:

| Parameter    | Default               |
| ------------ | --------------------- |
| Position     | `(0, 0, 0)`           |
| Direction    | `(0, -1, 0)`          |
| Color        | `(1, 1, 1)` (white)   |
| Intensity    | `1.0`                 |
| Range        | `10.0`                |
| Inner cone   | `cos(25°)` ≈ `0.9063` |
| Outer cone   | `cos(35°)` ≈ `0.8192` |
| Enabled      | `true`                |
| Ephemeral    | `false`               |
| CastsShadows | `false`               |

---

## Builder Options

All options follow the `LightBuilderOption` functional option pattern.

| Option             | Parameters                   | Description                                                 |
| ------------------ | ---------------------------- | ----------------------------------------------------------- |
| `WithPosition`     | `x, y, z float32`            | World-space position (point/spot)                           |
| `WithDirection`    | `x, y, z float32`            | Direction vector (normalized internally)                    |
| `WithColor`        | `r, g, b float32`            | RGB color                                                   |
| `WithIntensity`    | `intensity float32`          | Scalar intensity multiplier                                 |
| `WithRange`        | `lightRange float32`         | Max attenuation distance (point/spot)                       |
| `WithSpotCone`     | `innerDeg, outerDeg float32` | Inner/outer cone half-angles in degrees (stored as cosines) |
| `WithEnabled`      | `enabled bool`               | Whether the light is active for rendering                   |
| `WithEphemeral`    | `ephemeral bool`             | Marks as ephemeral (not persisted in scene registry)        |
| `WithCastsShadows` | `castsShadows bool`          | Enables shadow map generation for this light                |

---

## Light Interface

### Properties

| Method                   | Description                                                                    |
| ------------------------ | ------------------------------------------------------------------------------ |
| `Type() LightType`       | Light type (directional, point, or spot)                                       |
| `Position() [3]float32`  | World-space position (meaningless for directional)                             |
| `Direction() [3]float32` | Normalized direction (meaningless for point)                                   |
| `Color() [3]float32`     | RGB color                                                                      |
| `Intensity() float32`    | Scalar intensity multiplier                                                    |
| `Range() float32`        | Max attenuation distance (meaningless for directional)                         |
| `InnerCone() float32`    | `cos(inner half-angle)` for spot lights                                        |
| `OuterCone() float32`    | `cos(outer half-angle)` for spot lights                                        |
| `Enabled() bool`         | Whether the light is active; disabled lights are skipped during GPU marshaling |
| `Ephemeral() bool`       | Whether the light is short-lived (particle-emitted)                            |
| `CastsShadows() bool`    | Whether the light is eligible for shadow map generation                        |

### Setters

| Method                                    | Description                                          |
| ----------------------------------------- | ---------------------------------------------------- |
| `SetPosition(x, y, z float32)`            | Sets world-space position                            |
| `SetDirection(x, y, z float32)`           | Sets direction (normalized internally)               |
| `SetColor(r, g, b float32)`               | Sets RGB color                                       |
| `SetIntensity(intensity float32)`         | Sets scalar intensity                                |
| `SetRange(lightRange float32)`            | Sets max attenuation distance                        |
| `SetSpotCone(innerDeg, outerDeg float32)` | Sets cone half-angles in degrees (stored as cosines) |
| `SetEnabled(enabled bool)`                | Enables or disables the light                        |
| `SetEphemeral(ephemeral bool)`            | Marks as ephemeral                                   |
| `SetCastsShadows(castsShadows bool)`      | Enables or disables shadow casting                   |

---

## Forward+ Light Culling

The engine uses a Forward+ (tiled forward) rendering pipeline. The screen is divided into a grid of tiles, and a compute shader assigns lights to tiles so the lit fragment shader only loops over lights that actually overlap each tile.

### Constants

| Constant           | Value  | Description                                                               |
| ------------------ | ------ | ------------------------------------------------------------------------- |
| `TileSize`         | `16`   | Width and height of each screen-space tile in pixels                      |
| `MaxLightsPerTile` | `256`  | Maximum light indices stored per tile; excess lights are silently dropped |
| `MaxGPULights`     | `1024` | Maximum lights marshaled into the GPU storage buffer per frame            |

### TileCounts

```go
func TileCounts(screenWidth, screenHeight int) (tileCountX, tileCountY uint32)
```

Computes the number of tiles in each dimension for a given screen resolution. Used to size the tile light index buffer and configure the compute dispatch.

---

## Shadow Mapping

Shadow-casting lights render a depth-only pass into a shadow map texture. The lit fragment shader samples this texture with PCF (percentage-closer filtering) to produce soft shadow edges.

### Shadow Constants

| Constant                       | Value   | Description                                                              |
| ------------------------------ | ------- | ------------------------------------------------------------------------ |
| `ShadowMapResolution`          | `2048`  | Default shadow depth texture size (width and height in texels)           |
| `DefaultShadowHalfExtent`      | `40.0`  | Orthographic frustum half-extent in world units                          |
| `DefaultShadowNear`            | `0.1`   | Near plane for shadow projection                                         |
| `DefaultShadowFar`             | `200.0` | Far plane for shadow projection                                          |
| `DefaultShadowBias`            | `0.001` | Constant depth bias to reduce shadow acne                                |
| `DefaultShadowNormalBiasScale` | `3.0`   | Multiplier on texel world-size for normal-offset bias (typical: 2.0–4.0) |

---

## GPU Types

All GPU types have a corresponding embedded WGSL source (`*Source` variable) that can be injected into shaders via `@oxy:include` annotations (see [Shader Annotation System](README_ANNOTATIONS.md)).

Each GPU struct provides:

- `Size() int` — struct size in bytes
- `Marshal() []byte` — serializes to a byte buffer for GPU upload

### GPULight

Per-light data uploaded to the light storage buffer.

| Field          | Type         | Offset | Description                    |
| -------------- | ------------ | ------ | ------------------------------ |
| `Position`     | `[3]float32` | 0      | World-space position           |
| `LightType`    | `uint32`     | 12     | 0=directional, 1=point, 2=spot |
| `Color`        | `[3]float32` | 16     | RGB color                      |
| `Intensity`    | `float32`    | 28     | Scalar multiplier              |
| `Direction`    | `[3]float32` | 32     | Normalized direction           |
| `LightRange`   | `float32`    | 44     | Attenuation cutoff             |
| `InnerCone`    | `float32`    | 48     | cos(inner half-angle)          |
| `OuterCone`    | `float32`    | 52     | cos(outer half-angle)          |
| `CastsShadows` | `uint32`     | 56     | 1=casts, 0=does not            |
| `_pad`         | `uint32`     | 60     | Padding to 64 bytes            |

**Size:** 64 bytes

### GPULightHeader

Header prepended to the light storage buffer.

| Field          | Type         | Offset | Description                                  |
| -------------- | ------------ | ------ | -------------------------------------------- |
| `AmbientColor` | `[3]float32` | 0      | Scene ambient RGB                            |
| `LightCount`   | `uint32`     | 12     | Number of active lights following the header |

**Size:** 16 bytes

### GPUShadowData

Directional shadow data for the lit fragment shader.

| Field        | Type          | Offset | Description                                           |
| ------------ | ------------- | ------ | ----------------------------------------------------- |
| `LightVP`    | `[16]float32` | 0      | Orthographic view-projection from light's perspective |
| `TexelSize`  | `[2]float32`  | 64     | `1.0 / resolution` for PCF offset calculations        |
| `Bias`       | `float32`     | 72     | Depth comparison bias                                 |
| `NormalBias` | `float32`     | 76     | World-space normal-offset distance                    |

**Size:** 80 bytes

Additional methods:

- `ComputeDirectionalLightVP(lightDir, centerX, centerY, centerZ, halfExtent, near, far)` — Builds the orthographic view-projection matrix centered on the camera position.
- `ComputeNormalBias(halfExtent, scale, resolution)` — Derives the world-space normal-offset bias from shadow map parameters.

### GPUShadowUniform

Shadow vertex shader uniform (light view-projection only).

| Field     | Type          | Offset | Description                                           |
| --------- | ------------- | ------ | ----------------------------------------------------- |
| `LightVP` | `[16]float32` | 0      | Orthographic view-projection from light's perspective |

**Size:** 64 bytes

### GPULightCullUniforms

Uniform data for the light culling compute shader.

| Field          | Type          | Offset | Description               |
| -------------- | ------------- | ------ | ------------------------- |
| `InvProj`      | `[16]float32` | 0      | Inverse projection matrix |
| `ViewMatrix`   | `[16]float32` | 64     | Camera view matrix        |
| `TileCountX`   | `uint32`      | 128    | Tile columns              |
| `TileCountY`   | `uint32`      | 132    | Tile rows                 |
| `ScreenWidth`  | `uint32`      | 136    | Screen width in pixels    |
| `ScreenHeight` | `uint32`      | 140    | Screen height in pixels   |
| `LightCount`   | `uint32`      | 144    | Active light count        |
| `Near`         | `float32`     | 148    | Camera near plane         |
| `Far`          | `float32`     | 152    | Camera far plane          |
| `_pad`         | `uint32`      | 156    | Padding to 160 bytes      |

**Size:** 160 bytes

### GPUTileUniforms

Fragment shader uniform for tile-based light indexing.

| Field              | Type     | Offset | Description                |
| ------------------ | -------- | ------ | -------------------------- |
| `TileCountX`       | `uint32` | 0      | Number of tile columns     |
| `MaxLightsPerTile` | `uint32` | 4      | Max light indices per tile |

**Size:** 8 bytes

---

## Helper Functions

| Function                                                        | Description                                                                                                                          |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `ToGPULight(l Light) GPULight`                                  | Converts a `Light` to its GPU-aligned struct representation                                                                          |
| `MarshalLightBuffer(lights []Light, ambient [3]float32) []byte` | Marshals a header + enabled lights into a single byte buffer for GPU upload. Only enabled lights are included, up to `MaxGPULights`. |

---

## Usage Example

```go
package main

import (
    "github.com/Carmen-Shannon/oxy-go/engine/light"
)

func main() {
    // Create a directional sun light with shadows
    sun := light.NewLight(light.LightTypeDirectional,
        light.WithDirection(0.3, -1, 0.5),
        light.WithColor(1, 0.95, 0.9),
        light.WithIntensity(1.5),
        light.WithCastsShadows(true),
    )

    // Create a point light
    torch := light.NewLight(light.LightTypePoint,
        light.WithPosition(5, 2, 0),
        light.WithColor(1, 0.6, 0.3),
        light.WithIntensity(3.0),
        light.WithRange(15.0),
    )

    // Marshal for GPU upload
    lights := []light.Light{sun, torch}
    ambient := [3]float32{0.05, 0.05, 0.08}
    buf := light.MarshalLightBuffer(lights, ambient)
    _ = buf // upload to GPU storage buffer

    // Compute tile counts for Forward+ culling
    tileX, tileY := light.TileCounts(1280, 720)
    _, _ = tileX, tileY

    // Set up shadow data for the directional light
    shadow := &light.GPUShadowData{
        TexelSize: [2]float32{
            1.0 / float32(light.ShadowMapResolution),
            1.0 / float32(light.ShadowMapResolution),
        },
        Bias: light.DefaultShadowBias,
    }
    shadow.ComputeDirectionalLightVP(
        sun.Direction(),
        0, 0, 0, // camera center
        light.DefaultShadowHalfExtent,
        light.DefaultShadowNear,
        light.DefaultShadowFar,
    )
    shadow.ComputeNormalBias(
        light.DefaultShadowHalfExtent,
        light.DefaultShadowNormalBiasScale,
        light.ShadowMapResolution,
    )
    shadowBuf := shadow.Marshal()
    _ = shadowBuf // upload to GPU uniform buffer
}
```
