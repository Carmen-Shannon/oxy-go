# Oxy Light System

The `light` package provides lighting, shadow mapping, Forward+ tile culling, and contact shadows for the Oxy engine. It supports directional, point, and spot light types, all sharing a single `Light` interface. Lights are marshaled into GPU storage buffers each frame and evaluated in a tiled Forward+ rendering pipeline.

Shadow mapping uses a dual-cascade sphere-based CSM approach with a `Depth32Float` atlas, hardware `sampler_comparison`, and 16-tap Poisson PCF. Contact shadows are computed via a screen-space ray march that detects fine-detail occlusion at surface contacts.

> Post-processing handlers (SSAO, SSR, Composition, Bloom, TAA) are documented in [README_POSTPROCESSOR.md](README_POSTPROCESSOR.md). The G-Buffer handler is documented in the gbuffer section of [README_RENDERER.md](README_RENDERER.md).

---

## Table of Contents

- [Overview](#overview)
- [Light Types](#light-types)
- [Creating a Light](#creating-a-light)
- [Builder Options](#builder-options)
- [Light Interface](#light-interface)
  - [Properties](#properties)
  - [Setters](#setters)
- [LightingHandler](#lightinghandler)
  - [Creating a LightingHandler](#creating-a-lightinghandler)
  - [LightingHandler Builder Options](#lightinghandler-builder-options)
  - [LightingHandler Interface](#lightinghandler-interface)
- [Forward+ Light Culling](#forward-light-culling)
  - [Constants](#constants)
  - [TileCounts](#tilecounts)
- [Shadow Mapping (PCF)](#shadow-mapping-pcf)
  - [Shadow Constants](#shadow-constants)
- [Sub-Handlers](#sub-handlers)
  - [ShadowHandler](#shadowhandler)
  - [ContactShadowHandler](#contactshadowhandler)
- [GPU Types](#gpu-types)
  - [GPULight](#gpulight)
  - [GPULightHeader](#gpulightheader)
  - [GPUCSMData](#gpucsmdata)
  - [GPUCSMCascade](#gpucsmcascade)
  - [GPUShadowUniform](#gpushadowuniform)
  - [GPULightShadowEntry](#gpulightshadowentry)
  - [GPULightCullUniforms](#gpulightculluniforms)
  - [GPUTileUniforms](#gputileuniforms)
  - [GPUContactShadowParams](#gpucontactshadowparams)
- [Helper Functions](#helper-functions)
- [Usage Example](#usage-example)
- [Files](#files)

---

## Overview

The light system is designed around four pillars:

1. **Light** — A scene-level entity with type, position, direction, color, intensity, range, cone angles, and per-light shadow bias. All three light types share the same interface; type-specific properties return zero values when not applicable.
2. **Forward+ Tile Culling** — The screen is divided into tiles (`TileSize × TileSize` pixels). A compute shader assigns lights to tiles so the fragment shader only evaluates lights relevant to each tile.
3. **Dual-Cascade PCF Shadow Mapping** — Shadow-casting directional lights render into a `Depth32Float` atlas (two cascades). Cascade 0 is a camera-centered sphere with configurable inner radius; cascade 1 is frustum-fit for the full depth range. 16-tap Poisson PCF provides soft shadow edges. Spot and point lights use a separate `Depth32Float` atlas described by `GPULightShadowEntry` records.
4. **Sub-Handler Architecture** — The `LightingHandler` owns lazily-initialized sub-handlers (`ShadowHandler`, `ContactShadowHandler`) that manage their own textures, pipelines, and bind groups. Post-processing sub-handlers (GBuffer, SSAO, SSR, Composition, TAA) have been extracted to separate packages under `engine/renderer/`.

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
| ShadowBias   | `0.005`               |

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
| `WithShadowBias`   | `bias float32`               | Depth comparison bias for this light's shadow map           |

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
| `ShadowBias() float32`   | Per-light depth comparison bias for shadow map generation                      |

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
| `SetShadowBias(bias float32)`             | Sets the per-light shadow bias                       |

---

## LightingHandler

The `LightingHandler` manages the light list, ambient color, Forward+ tile culling, shadow sub-handlers, and all associated GPU resources (bind group providers, pipeline keys). It is created via `NewLightingHandler` with builder options and attached to a scene. GPU resources are initialized lazily by the scene when the first light is added. Shadow resources are owned by the `ShadowHandler` sub-handler.

Thread safety is provided by the owning scene's mutex — the handler itself does not perform internal locking.

### Creating a LightingHandler

```go
handler := light.NewLightingHandler(
    light.WithAmbientColor([3]float32{0.05, 0.05, 0.08}),
    light.WithShadowHandler(light.NewShadowHandler(
        light.WithShadowNearFar(0.1, 300.0),
        light.WithShadowNormalBiasScale(3.0),
        light.WithShadowMapResolution(4096),
        light.WithPCFRadius(1.5),
        light.WithShadowInnerRadius(50.0),
    )),
)
```

Defaults applied before options:

| Parameter        | Default             |
| ---------------- | ------------------- |
| Enabled          | `false`             |
| Lights           | empty               |
| Ambient color    | `(0, 0, 0)` (black) |
| TileSize         | `16`                |
| MaxLightsPerTile | `256`               |
| MaxGPULights     | `1024`              |

The constructor pre-creates the following named `BindGroupProvider` entries: `"lights"`, `"light_cull"`, `"tile_lit"`, `"ssao_lit"`, `"probe_lit"`, `"composition_lit"`, `"ssr_lit"`. Shadow-related BGPs live on `ShadowHandler`.

### LightingHandler Builder Options

All options follow the `LightingHandlerOption` functional option pattern.

| Option                     | Parameters                     | Description                                          |
| -------------------------- | ------------------------------ | ---------------------------------------------------- |
| `WithAmbientColor`         | `color [3]float32`             | Initial ambient light color                          |
| `WithShadowHandler`        | `h ShadowHandler`              | Overrides the default shadow handler                 |
| `WithContactShadowHandler` | `handler ContactShadowHandler` | Overrides the default contact shadow handler         |
| `WithTileSize`             | `size int`                     | Forward+ tile size in pixels (default 16)            |
| `WithMaxLightsPerTile`     | `max int`                      | Max light indices per tile (default 256)             |
| `WithMaxGPULights`         | `max int`                      | Max lights marshaled to GPU per frame (default 1024) |

### LightingHandler Interface

#### Light Management

| Method                              | Description                                                               |
| ----------------------------------- | ------------------------------------------------------------------------- |
| `Enabled() bool`                    | Whether the lighting subsystem is GPU-initialized and ready for rendering |
| `SetEnabled(enabled bool)`          | Marks the subsystem as GPU-initialized                                    |
| `Lights() []Light`                  | Returns a copy of the current light list                                  |
| `AddLight(l Light)`                 | Appends a light to the handler's list                                     |
| `RemoveLight(l Light)`              | Removes a light by reference equality                                     |
| `AmbientColor() [3]float32`         | Returns the scene's ambient light color                                   |
| `SetAmbientColor(color [3]float32)` | Sets the scene's ambient light color                                      |

#### Bind Group Providers & Pipelines

| Method                             | Description                                                       |
| ---------------------------------- | ----------------------------------------------------------------- |
| `Bgp(key string)`                  | Retrieves a BindGroupProvider by key (see default BGP list above) |
| `Bgps()`                           | Returns the full map of bind group providers                      |
| `PipelineKey(name string) string`  | Retrieves the pipeline key for a given name                       |
| `PipelineKeys() map[string]string` | Returns the full map of pipeline name-to-key mappings             |
| `SetPipelineKey(name, key string)` | Stores a pipeline key under the given name                        |

#### Sub-Handler Accessors

| Method                                        | Description                        |
| --------------------------------------------- | ---------------------------------- |
| `ShadowHandler() ShadowHandler`               | Returns the shadow handler         |
| `ContactShadowHandler() ContactShadowHandler` | Returns the contact shadow handler |

#### Screen & Tile State

| Method                                                                      | Description                                                  |
| --------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `ScreenWidth() int`                                                         | Current screen width in pixels for tile calculations         |
| `ScreenHeight() int`                                                        | Current screen height in pixels for tile calculations        |
| `TileCountX() int`                                                          | Number of Forward+ tile columns                              |
| `TileCountY() int`                                                          | Number of Forward+ tile rows                                 |
| `Resize(width, height int)`                                                 | Updates screen dimensions and recalculates tile counts       |
| `TileSize() int`                                                            | Configured tile size in pixels                               |
| `MaxLightsPerTile() int`                                                    | Maximum light indices stored per tile                        |
| `MaxGPULights() int`                                                        | Maximum lights marshaled to GPU per frame                    |
| `SetMaxGPULights(max int)`                                                  | Sets the maximum GPU light count                             |
| `MarshalLightBuffer(lights []Light, shadowIndices map[Light]uint32) []byte` | Marshals enabled lights into a GPU storage buffer byte slice |

---

## Forward+ Light Culling

The engine uses a Forward+ (tiled forward) rendering pipeline. The screen is divided into a grid of tiles, and a compute shader assigns lights to tiles so the lit fragment shader only loops over lights that actually overlap each tile.

### Constants

| Constant           | Value | Description                                                               |
| ------------------ | ----- | ------------------------------------------------------------------------- |
| `TileSize`         | `16`  | Width and height of each screen-space tile in pixels                      |
| `MaxLightsPerTile` | `256` | Maximum light indices stored per tile; excess lights are silently dropped |

`MaxGPULights` is configurable per handler via `WithMaxGPULights` (default `1024`) and accessed via `LightingHandler.MaxGPULights()`.

### TileCounts

```go
func TileCounts(screenWidth, screenHeight int) (tileCountX, tileCountY uint32)
```

Computes the number of tiles in each dimension for a given screen resolution. Used to size the tile light index buffer and configure the compute dispatch.

---

## Shadow Mapping (PCF)

Shadow-casting directional lights render a depth-only pass into a `Depth32Float` atlas each frame via a dual-cascade sphere-based CSM. The lit fragment shader applies 16-tap Poisson PCF using a `sampler_comparison`. Spot and point lights share a separate `Depth32Float` atlas; their shadow data is described per-light by `GPULightShadowEntry` records in a GPU storage buffer.

- **Cascade 0** — Camera-centered sphere of radius `ShadowInnerRadius` (default `100.0`). Provides constant texel density around the camera for near-field detail.
- **Cascade 1** — Frustum-fit over the full camera depth range for wide-area coverage.
- **PCF kernel** — 16-tap Poisson disk, radius configurable via `WithPCFRadius` (default `1.0` texels).
- **Spot lights** — Perspective VP from light position/direction/outer cone, stored as `GPULightShadowEntry`.
- **Point lights** — 6 consecutive atlas slots (one per cube face), each with a 90° FOV perspective VP.

### Shadow Constants

| Constant                       | Source          | Description                                          |
| ------------------------------ | --------------- | ---------------------------------------------------- |
| `DefaultShadowNear`            | `ShadowHandler` | Near plane for shadow projection (`0.1`)             |
| `DefaultShadowFar`             | `ShadowHandler` | Far plane for shadow projection (`200.0`)            |
| `DefaultShadowNormalBiasScale` | `ShadowHandler` | Normal-offset bias multiplier (`3.0`)                |
| `DefaultShadowMapResolution`   | `ShadowHandler` | CSM atlas resolution in texels (`2048`)              |
| `DefaultPCFRadius`             | `ShadowHandler` | Poisson disk radius in texels (`1.0`)                |
| `DefaultPCFSamples`            | `ShadowHandler` | Poisson disk tap count (`16`)                        |
| `DefaultShadowInnerRadius`     | `ShadowHandler` | Inner cascade sphere radius in world units (`100.0`) |
| `DefaultLightShadowTileSize`   | `ShadowHandler` | Spot/point atlas tile size in texels (`1024`)        |

---

## Sub-Handlers

The `LightingHandler` delegates shadow and contact-shadow work to dedicated sub-handlers. Default instances are auto-created in the `NewLightingHandler` constructor unless an explicit override is provided via builder options. GPU resources for each sub-handler are initialized lazily by the scene when lighting is first enabled. All sub-handlers share the same thread-safety model — locking is owned by the enclosing scene.

> Post-processing sub-handlers (GBufferHandler, SSAOHandler, SSRHandler, CompositionHandler, TAAHandler) have been extracted to separate packages. See [README_POSTPROCESSOR.md](README_POSTPROCESSOR.md) and [README_RENDERER.md](README_RENDERER.md).

### ShadowHandler

The `ShadowHandler` manages all shadow map resources — the dual-cascade directional CSM atlas, the spot/point light shadow atlas, per-slot depth-pass bind group providers, and all associated pipeline keys.

```go
shadow := light.NewShadowHandler(
    light.WithShadowNearFar(0.1, 300.0),
    light.WithShadowNormalBiasScale(3.0),
    light.WithShadowMapResolution(4096),
    light.WithPCFRadius(1.5),
    light.WithShadowInnerRadius(50.0),
)
```

**Builder Options (`ShadowHandlerOption`):**

| Option                      | Parameters          | Default        | Description                                     |
| --------------------------- | ------------------- | -------------- | ----------------------------------------------- |
| `WithShadowNearFar`         | `near, far float32` | `0.1`, `200.0` | Near/far planes for shadow projection           |
| `WithShadowNormalBiasScale` | `scale float32`     | `3.0`          | Normal-offset bias multiplier                   |
| `WithShadowMapResolution`   | `resolution int`    | `2048`         | CSM atlas resolution in texels                  |
| `WithPCFRadius`             | `radius float32`    | `1.0`          | Poisson disk PCF kernel radius in texels        |
| `WithPCFSamples`            | `samples uint32`    | `16`           | Poisson disk tap count for directional CSM PCF  |
| `WithPCFSamplesSpot`        | `samples uint32`    | `8`            | Poisson disk tap count for spot/point light PCF |
| `WithShadowInnerRadius`     | `radius float32`    | `100.0`        | Inner cascade sphere radius in world units      |
| `WithLightShadowTileSize`   | `size int`          | `1024`         | Tile size for the spot/point atlas in texels    |

**Key Interface Methods:**

| Method                                                                                       | Description                          |
| -------------------------------------------------------------------------------------------- | ------------------------------------ |
| `ShadowNear() float32`                                                                       | Near plane for shadow projection     |
| `ShadowFar() float32`                                                                        | Far plane for shadow projection      |
| `ShadowNormalBiasScale() float32`                                                            | Normal-offset bias multiplier        |
| `ShadowMapResolution() int`                                                                  | CSM atlas resolution in texels       |
| `PCFRadius() float32`                                                                        | PCF kernel radius in texels          |
| `PCFSamples() uint32`                                                                        | PCF tap count for directional CSM    |
| `PCFSamplesSpot() uint32`                                                                    | PCF tap count for spot/point lights  |
| `ShadowInnerRadius() float32`                                                                | Inner cascade sphere radius          |
| `LightShadowTileSize() int`                                                                  | Spot/point atlas tile size           |
| `CascadeCount() int`                                                                         | Always 2 (dual-cascade)              |
| `ComparisonSampler() *wgpu.Sampler` / `SetComparisonSampler(s *wgpu.Sampler)`                | Depth comparison sampler             |
| `CSMAtlasTexture() *wgpu.Texture` / `SetCSMAtlasTexture(t *wgpu.Texture)`                    | Directional CSM atlas texture        |
| `CSMAtlasTextureView() *wgpu.TextureView` / `SetCSMAtlasTextureView(tv *wgpu.TextureView)`   | Directional CSM atlas texture view   |
| `LightShadowAtlas() *wgpu.Texture` / `SetLightShadowAtlas(t *wgpu.Texture)`                  | Spot/point shadow atlas texture      |
| `LightShadowAtlasView() *wgpu.TextureView` / `SetLightShadowAtlasView(tv *wgpu.TextureView)` | Spot/point shadow atlas texture view |
| `LightShadowAtlasSlots() int` / `SetLightShadowAtlasSlots(n int)`                            | Number of allocated atlas slots      |
| `LightShadowAtlasCols() int` / `SetLightShadowAtlasCols(n int)`                              | Number of atlas columns              |
| `Bgp(key string) bind_group_provider.BindGroupProvider`                                      | Retrieves a shadow BGP by key        |
| `Bgps() map[string]bind_group_provider.BindGroupProvider`                                    | Returns all shadow BGPs              |
| `PipelineKey(name string) string`                                                            | Retrieves a pipeline key             |
| `PipelineKeys() map[string]string`                                                           | Returns all pipeline keys            |
| `SetPipelineKey(name, key string)`                                                           | Stores a pipeline key                |

**ShadowType constants:**

```go
const (
    ShadowTypeSpot     ShadowType = 0  // Perspective spot/point-face shadow
    ShadowTypeCubeFace ShadowType = 1  // Point light cube face
)
```

### ContactShadowHandler

The `ContactShadowHandler` manages a screen-space ray march compute pass that detects fine-detail occlusion at surface contacts (feet on ground, model creases). It runs after SSAO and its result is multiplied into the directional light contribution.

```go
contact := light.NewContactShadowHandler(
    light.WithContactShadowsEnabled(true),
    light.WithContactShadowStepCount(16),
    light.WithContactShadowMaxDistance(1.0),
    light.WithContactShadowThickness(0.05),
)
```

**Builder Options (`ContactShadowHandlerOption`):**

| Option                         | Parameters          | Default | Description                          |
| ------------------------------ | ------------------- | ------- | ------------------------------------ |
| `WithContactShadowsEnabled`    | `enabled bool`      | `true`  | Whether contact shadows are computed |
| `WithContactShadowStepCount`   | `count int`         | `16`    | Ray march steps per pixel            |
| `WithContactShadowMaxDistance` | `dist float32`      | `1.0`   | Max march distance in world units    |
| `WithContactShadowThickness`   | `thickness float32` | `0.05`  | NDC depth thickness tolerance        |

**Key Interface Methods:**

| Method                                                                     | Description                        |
| -------------------------------------------------------------------------- | ---------------------------------- |
| `Enabled() bool` / `SetEnabled(enabled bool)`                              | Whether contact shadows are active |
| `StepCount() int`                                                          | Ray march step count               |
| `MaxDistance() float32`                                                    | Max march distance in world units  |
| `Thickness() float32`                                                      | Depth thickness tolerance in NDC   |
| `Texture() *wgpu.Texture` / `SetTexture(t *wgpu.Texture)`                  | Output contact shadow texture      |
| `TextureView() *wgpu.TextureView` / `SetTextureView(tv *wgpu.TextureView)` | Output texture view                |
| `LinearSampler() *wgpu.Sampler` / `SetLinearSampler(s *wgpu.Sampler)`      | Linear sampler                     |
| `Bgp(key string) bind_group_provider.BindGroupProvider`                    | Retrieves a BGP by key             |
| `Bgps() map[string]bind_group_provider.BindGroupProvider`                  | Returns all BGPs                   |
| `PipelineKey(name string) string`                                          | Retrieves a pipeline key           |
| `PipelineKeys() map[string]string`                                         | Returns all pipeline keys          |

**Default BGPs:** `"contact_shadow_compute"`

---

## GPU Types

All GPU types have a corresponding embedded WGSL source (`*Source` variable) that can be injected into shaders via `@oxy:include` annotations (see [Shader Annotation System](README_ANNOTATIONS.md)).

Each GPU struct provides:

- `Size() int` — struct size in bytes
- `Marshal() []byte` — serializes to a byte buffer for GPU upload

### GPULight

Per-light data uploaded to the light storage buffer.

| Field          | Type         | Offset | Description                                                        |
| -------------- | ------------ | ------ | ------------------------------------------------------------------ |
| `Position`     | `[3]float32` | 0      | World-space position                                               |
| `LightType`    | `uint32`     | 12     | 0=directional, 1=point, 2=spot                                     |
| `Color`        | `[3]float32` | 16     | RGB color                                                          |
| `Intensity`    | `float32`    | 28     | Scalar multiplier                                                  |
| `Direction`    | `[3]float32` | 32     | Normalized direction                                               |
| `LightRange`   | `float32`    | 44     | Attenuation cutoff                                                 |
| `InnerCone`    | `float32`    | 48     | cos(inner half-angle)                                              |
| `OuterCone`    | `float32`    | 52     | cos(outer half-angle)                                              |
| `CastsShadows` | `uint32`     | 56     | 1=casts, 0=does not                                                |
| `ShadowIndex`  | `uint32`     | 60     | Index into the light shadow entry buffer; `0xFFFFFFFF` = no shadow |

**Size:** 64 bytes

### GPULightHeader

Header prepended to the light storage buffer.

| Field          | Type         | Offset | Description                                  |
| -------------- | ------------ | ------ | -------------------------------------------- |
| `AmbientColor` | `[3]float32` | 0      | Scene ambient RGB                            |
| `LightCount`   | `uint32`     | 12     | Number of active lights following the header |

**Size:** 16 bytes

### GPUCSMCascade

One cascade block in the directional shadow atlas. 80 bytes.

| Field        | Type          | Offset | Description                            |
| ------------ | ------------- | ------ | -------------------------------------- |
| `LightVP`    | `[16]float32` | 0      | Light view-projection matrix           |
| `ShadowNear` | `float32`     | 64     | Shadow near plane                      |
| `ShadowFar`  | `float32`     | 68     | Shadow far plane                       |
| `CamFar`     | `float32`     | 72     | Camera far plane for cascade selection |
| `NormalBias` | `float32`     | 76     | World-space normal-offset bias         |

---

### GPUCSMData

Top-level CSM uniform written to the shadow bind group each frame. 192 bytes total (32-byte header + 2 × 80-byte `GPUCSMCascade` blocks).

| Field               | Type               | Offset | Description                                     |
| ------------------- | ------------------ | ------ | ----------------------------------------------- |
| `TexelSize`         | `[2]float32`       | 0      | `1 / resolution` for each atlas axis            |
| `Bias`              | `float32`          | 8      | Depth comparison bias                           |
| `InnerRadius`       | `float32`          | 12     | Inner cascade sphere radius in world units      |
| `PCFRadius`         | `float32`          | 16     | Poisson disk radius in texels                   |
| `ShadowMaxDistance` | `float32`          | 20     | Maximum shadow distance                         |
| `_pad0`             | `float32`          | 24     | Padding                                         |
| `_pad1`             | `float32`          | 28     | Padding                                         |
| `Cascades`          | `[2]GPUCSMCascade` | 32     | Two cascade blocks (inner sphere + frustum-fit) |

**Method:**

```go
func (d *GPUCSMData) ComputeCascades(
    lightDir [3]float32,
    camNear, camFar, camFov, camAspect float32,
    camViewMatrix [16]float32,
    cameraPosition [3]float32,
    innerRadius, normalBiasScale float32,
    resolution int,
)
```

Computes both cascade view-projection matrices, normal bias, and cascade selection parameters from the camera and light state, writing results directly into `Cascades[0]` and `Cascades[1]`.

### GPUShadowUniform

Shadow vertex shader uniform for a single depth pass.

| Field     | Type          | Offset | Description                                           |
| --------- | ------------- | ------ | ----------------------------------------------------- |
| `LightVP` | `[16]float32` | 0      | Light view-projection matrix for a single shadow pass |

Total size: **64 bytes**.

### GPULightShadowEntry

Per-light shadow atlas entry written into a GPU storage buffer for spot and point lights. 96 bytes.

| Field        | Type          | Offset | Description                                                |
| ------------ | ------------- | ------ | ---------------------------------------------------------- |
| `LightVP`    | `[16]float32` | 0      | Light view-projection matrix                               |
| `AtlasRect`  | `[4]float32`  | 64     | `(u_offset, v_offset, u_scale, v_scale)` in atlas UV space |
| `Bias`       | `float32`     | 80     | Depth comparison bias                                      |
| `Near`       | `float32`     | 84     | Shadow near plane                                          |
| `Far`        | `float32`     | 88     | Shadow far plane                                           |
| `ShadowType` | `ShadowType`  | 92     | `0` = spot / `1` = cube face                               |

Spot lights produce 1 entry per light. Point lights produce 6 consecutive entries (one per cube face, selected in the lit shader by the dominant axis of `light → fragment`).

### GPUContactShadowParams

Uniform data for the contact shadow compute shader.

| Field            | Type          | Offset | Description                                   |
| ---------------- | ------------- | ------ | --------------------------------------------- |
| `ViewProj`       | `[16]float32` | 0      | View-projection matrix (column-major)         |
| `InvViewProj`    | `[16]float32` | 64     | Inverse view-projection matrix (column-major) |
| `LightDirection` | `[3]float32`  | 128    | Directional light direction (world space)     |
| `StepCount`      | `uint32`      | 140    | Number of ray march steps                     |
| `MaxDistance`    | `float32`     | 144    | Max ray march distance in world units         |
| `Thickness`      | `float32`     | 148    | Depth thickness tolerance in NDC depth space  |
| `ScreenWidth`    | `float32`     | 152    | Output texture width in pixels                |
| `ScreenHeight`   | `float32`     | 156    | Output texture height in pixels               |
| `CameraPosition` | `[3]float32`  | 160    | World-space camera position                   |
| `_pad`           | `float32`     | 172    | Padding to 176-byte alignment                 |

**Size:** 176 bytes

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

| Field              | Type     | Offset | Description                           |
| ------------------ | -------- | ------ | ------------------------------------- |
| `TileCountX`       | `uint32` | 0      | Number of tile columns                |
| `MaxLightsPerTile` | `uint32` | 4      | Maximum light indices stored per tile |
| `ScreenWidth`      | `uint32` | 8      | Screen width in pixels                |
| `ScreenHeight`     | `uint32` | 12     | Screen height in pixels               |

Total size: **16 bytes**.

---

## Helper Functions

| Function                                                                                        | Description                                                                                                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ToGPULight(l Light) GPULight`                                                                  | Converts a `Light` to its GPU-aligned struct representation                                                                                                                                                                                           |
| `(h LightingHandler) MarshalLightBuffer(lights []Light, shadowIndices map[Light]uint32) []byte` | Interface method on `LightingHandler`. Marshals enabled lights into a GPU storage buffer byte slice using the handler's internal ambient color. `shadowIndices` maps each shadow-casting light to its slot index in the `GPULightShadowEntry` buffer. |

---

## Usage Example

```go
package main

import (
    "github.com/Carmen-Shannon/oxy-go/engine/light"
)

func main() {
    // Create directional light (sun)
    sun := light.NewLight(light.LightTypeDirectional,
        light.WithDirection(0, -1, 0.5),
        light.WithColor(1.0, 0.95, 0.8),
        light.WithIntensity(2.0),
        light.WithCastsShadows(true),
        light.WithShadowBias(0.001),
    )

    // Create a point light
    bulb := light.NewLight(light.LightTypePoint,
        light.WithPosition(0, 3, 0),
        light.WithColor(1.0, 0.8, 0.6),
        light.WithIntensity(5.0),
        light.WithRange(20.0),
        light.WithCastsShadows(true),
    )

    // Create a spot light
    flashlight := light.NewLight(light.LightTypeSpot,
        light.WithPosition(0, 5, 0),
        light.WithDirection(0, -1, 0),
        light.WithColor(1, 1, 1),
        light.WithIntensity(8.0),
        light.WithRange(30.0),
        light.WithSpotCone(15, 25),
        light.WithCastsShadows(true),
    )

    // Create the lighting handler with shadow configuration
    handler := light.NewLightingHandler(
        light.WithAmbientColor([3]float32{0.05, 0.05, 0.08}),
        light.WithShadowHandler(light.NewShadowHandler(
            light.WithShadowNearFar(0.1, 300.0),
            light.WithShadowNormalBiasScale(3.0),
            light.WithShadowMapResolution(4096),
            light.WithPCFRadius(1.5),
            light.WithShadowInnerRadius(50.0),
        )),
        light.WithContactShadowHandler(light.NewContactShadowHandler(
            light.WithContactShadowsEnabled(true),
            light.WithContactShadowStepCount(16),
            light.WithContactShadowMaxDistance(1.0),
            light.WithContactShadowThickness(0.05),
        )),
    )

    // Add lights
    handler.AddLight(sun)
    handler.AddLight(bulb)
    handler.AddLight(flashlight)

    // Query tile layout
    tileX, tileY := light.TileCounts(1280, 720)
    _, _ = tileX, tileY

    _ = handler
}
```

---

## Files

| File                                | Contents                                                                                                                                                                                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `light.go`                          | `Light` interface and `LightType` constants                                                                                                                                                                                  |
| `light_impl.go`                     | Unexported `light` struct and method implementations                                                                                                                                                                         |
| `light_builder.go`                  | `LightBuilderOption` type, `With*` functions, `NewLight` constructor                                                                                                                                                         |
| `light_handler.go`                  | `LightingHandler` interface                                                                                                                                                                                                  |
| `light_handler_impl.go`             | Unexported `lightingHandler` struct and method implementations                                                                                                                                                               |
| `light_handler_builder.go`          | `LightingHandlerOption` type, `With*` functions, `NewLightingHandler` constructor                                                                                                                                            |
| `shadow_handler.go`                 | `ShadowHandler` interface and `ShadowType` constants                                                                                                                                                                         |
| `shadow_handler_impl.go`            | Unexported `shadowHandler` struct                                                                                                                                                                                            |
| `shadow_handler_builder.go`         | `ShadowHandlerOption` type and `NewShadowHandler` constructor                                                                                                                                                                |
| `contact_shadow_handler.go`         | `ContactShadowHandler` interface                                                                                                                                                                                             |
| `contact_shadow_handler_impl.go`    | Unexported `contactShadowHandler` struct                                                                                                                                                                                     |
| `contact_shadow_handler_builder.go` | `ContactShadowHandlerOption` type and `NewContactShadowHandler` constructor                                                                                                                                                  |
| `gpu_types.go`                      | All GPU-marshaled structs (`GPULight`, `GPULightHeader`, `GPUCSMData`, `GPUCSMCascade`, `GPUShadowUniform`, `GPULightShadowEntry`, `GPULightCullUniforms`, `GPUTileUniforms`, `GPUContactShadowParams`) and helper functions |
