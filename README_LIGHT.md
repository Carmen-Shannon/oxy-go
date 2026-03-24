# Oxy Light System

The `light` package provides the lighting, shadow mapping, Forward+ tile culling, and full global illumination (GI) pipeline for the Oxy engine. It supports directional, point, and spot light types, all sharing a single `Light` interface. Lights are marshaled into GPU storage buffers each frame and evaluated in a tiled Forward+ rendering pipeline.

Shadow mapping uses a dual-cascade sphere-based CSM approach with a `Depth32Float` atlas, hardware `sampler_comparison`, and 16-tap Poisson PCF. Contact shadows are computed via a screen-space ray march that detects fine-detail occlusion at surface contacts.

The GI pipeline includes a G-Buffer MRT pre-pass, Screen-Space Ambient Occlusion (SSAO) with bilateral blur, Screen-Space Reflections (SSR) via Hi-Z ray marching, and a final composition pass with ACES tone mapping and HDR rendering.

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
- [GI Sub-Handlers](#gi-sub-handlers)
  - [GBufferHandler](#gbufferhandler)
  - [SSAOHandler](#ssaohandler)
  - [SSRHandler](#ssrhandler)
  - [CompositionHandler](#compositionhandler)
  - [ShadowHandler](#shadowhandler)
  - [ContactShadowHandler](#contactshadowhandler)
- [GPU Types](#gpu-types)
  - [GPULight](#gpulight)
  - [GPUCSMData](#gpucsmdata)
  - [GPUCSMCascade](#gpucsmcascade)
  - [GPUShadowUniform](#gpushadowuniform)
  - [GPULightShadowEntry](#gpulightshadowentry)
  - [GPUSSAOParams](#gpussaoparams)
  - [GPUBlurParams](#gpublurparams)
  - [GPUTileUniforms](#gputileuniforms)
  - [GPULuminanceParams](#gpuluminanceparams)
  - [GPUContactShadowParams](#gpucontactshadowparams)
- [Helper Functions](#helper-functions)
- [Usage Example](#usage-example)
- [Files](#files)

---

## Overview

The light system is designed around five pillars:

1. **Light** — A scene-level entity with type, position, direction, color, intensity, range, cone angles, and per-light shadow bias. All three light types share the same interface; type-specific properties return zero values when not applicable.
2. **Forward+ Tile Culling** — The screen is divided into tiles (`TileSize × TileSize` pixels). A compute shader assigns lights to tiles so the fragment shader only evaluates lights relevant to each tile.
3. **Dual-Cascade PCF Shadow Mapping** — Shadow-casting directional lights render into a `Depth32Float` atlas (two cascades). Cascade 0 is a camera-centered sphere with configurable inner radius; cascade 1 is frustum-fit for the full depth range. 16-tap Poisson PCF provides soft shadow edges. Spot and point lights use a separate `Depth32Float` atlas described by `GPULightShadowEntry` records.
4. **Global Illumination Pipeline** — A G-Buffer MRT pre-pass captures per-pixel normals, albedo, and linear depth. SSAO uses hemisphere sampling on the G-Buffer to compute ambient occlusion. A screen-space contact-shadow pass detects fine-detail occlusion at surface contacts. SSR via Hi-Z ray marching adds specular reflections. A final composition pass applies ACES tone mapping to the HDR result.
5. **Sub-Handler Architecture** — The `LightingHandler` owns lazily-initialized sub-handlers (`GBufferHandler`, `SSAOHandler`, `CompositionHandler`, `SSRHandler`, `ShadowHandler`, `ContactShadowHandler`) that manage their own textures, pipelines, and bind groups.

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

The `LightingHandler` manages the light list, ambient color, Forward+ tile culling, GI sub-handlers, and all associated GPU resources (bind group providers, pipeline keys). It is created via `NewLightingHandler` with builder options and attached to a scene. GPU resources are initialized lazily by the scene when the first light is added. Shadow resources are owned by the `ShadowHandler` sub-handler.

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

The constructor pre-creates the following named `BindGroupProvider` entries: `"lights"`, `"light_cull"`, `"tile_lit"`, `"ssao_lit"`, `"probe_lit"`, `"composition_lit"`, `"ssr_lit"`. Shadow-related BGPs live on `ShadowHandler`; SSAO blur BGPs live on `SSAOHandler`.

The constructor auto-creates default sub-handlers (`GBufferHandler`, `SSAOHandler`, `CompositionHandler`, `SSRHandler`, `ShadowHandler`, `ContactShadowHandler`) if not explicitly provided via builder options.

### LightingHandler Builder Options

All options follow the `LightingHandlerOption` functional option pattern.

| Option                     | Parameters                     | Description                                            |
| -------------------------- | ------------------------------ | ------------------------------------------------------ |
| `WithAmbientColor`         | `color [3]float32`             | Initial ambient light color                            |
| `WithGBufferHandler`       | `handler GBufferHandler`       | Overrides the default G-Buffer handler                 |
| `WithSSAOHandler`          | `handler SSAOHandler`          | Overrides the default SSAO handler                     |
| `WithCompositionHandler`   | `handler CompositionHandler`   | Overrides the default composition/tone mapping handler |
| `WithSSRHandler`           | `handler SSRHandler`           | Overrides the default SSR handler                      |
| `WithShadowHandler`        | `h ShadowHandler`              | Overrides the default shadow handler                   |
| `WithContactShadowHandler` | `handler ContactShadowHandler` | Overrides the default contact shadow handler           |
| `WithTileSize`             | `size int`                     | Forward+ tile size in pixels (default 16)              |
| `WithMaxLightsPerTile`     | `max int`                      | Max light indices per tile (default 256)               |
| `WithMaxGPULights`         | `max int`                      | Max lights marshaled to GPU per frame (default 1024)   |

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

#### GI Sub-Handler Accessors

| Method                                        | Description                                            |
| --------------------------------------------- | ------------------------------------------------------ |
| `GBufferHandler() GBufferHandler`             | Returns the G-Buffer handler, or nil if not configured |
| `SSAOHandler() SSAOHandler`                   | Returns the SSAO handler, or nil if not configured     |
| `CompositionHandler() CompositionHandler`     | Returns the composition/tone mapping handler, or nil   |
| `SSRHandler() SSRHandler`                     | Returns the SSR handler, or nil if not configured      |
| `ShadowHandler() ShadowHandler`               | Returns the shadow handler                             |
| `ContactShadowHandler() ContactShadowHandler` | Returns the contact shadow handler                     |

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

## GI Sub-Handlers

The `LightingHandler` delegates each stage of the GI pipeline to a dedicated sub-handler. Default instances are auto-created in the `NewLightingHandler` constructor unless an explicit override is provided via builder options. GPU resources for each sub-handler are initialized lazily by the scene when GI is first enabled. All sub-handlers share the same thread-safety model — locking is owned by the enclosing scene.

### GBufferHandler

The `GBufferHandler` manages the multiple render target (MRT) textures written during the geometry pre-pass. Downstream screen-space effects (SSAO, SSR) read from these textures.

```go
gbuf := light.NewGBufferHandler(
    light.WithGBufferScreenSize(1280, 720),
)
```

| Builder Option          | Parameters          | Default | Description               |
| ----------------------- | ------------------- | ------- | ------------------------- |
| `WithGBufferScreenSize` | `width, height int` | 0, 0    | Initial screen dimensions |

**MRT Textures:**

| Texture | Format      | Contents                                         |
| ------- | ----------- | ------------------------------------------------ |
| Normal  | RGBA16Float | World normal XYZ (packed [0,1]) + roughness in W |
| Albedo  | RGBA8Unorm  | Albedo RGB + metallic in A                       |
| Depth   | Depth24Plus | Shared depth for the pre-pass                    |

**Key Interface Methods:**

| Method                                          | Description                                            |
| ----------------------------------------------- | ------------------------------------------------------ |
| `Enabled() bool`                                | Whether GPU resources are initialized                  |
| `PositionTexture() / SetPositionTexture`        | World-space position MRT texture                       |
| `NormalTexture() / SetNormalTexture`            | Normals + roughness MRT texture                        |
| `AlbedoTexture() / SetAlbedoTexture`            | Albedo + metallic MRT texture                          |
| `DepthTexture() / SetDepthTexture`              | Shared depth texture for the G-Buffer pass             |
| `ScreenWidth() int` / `SetScreenWidth(w int)`   | Screen width in pixels                                 |
| `ScreenHeight() int` / `SetScreenHeight(h int)` | Screen height in pixels                                |
| `PipelineKey(name) / SetPipelineKey`            | Pipeline key storage for the G-Buffer render pipeline  |
| `Resize(width, height)`                         | Updates screen dimensions (does not recreate textures) |

### SSAOHandler

The `SSAOHandler` manages the hemisphere sampling kernel, noise texture, raw and blurred occlusion textures, and bilateral blur pipeline. The raw AO is computed in a compute shader and then blurred with a separable bilateral filter to preserve edges.

```go
ssao := light.NewSSAOHandler(
    light.WithSSAOSampleCount(16),
    light.WithSSAOScreenRadius(24.0),
    light.WithSSAOBias(0.025),
    light.WithSSAOPower(2.0),
    light.WithSSAOBlurRadius(4),
    light.WithSSAOHalfResolution(false),
)
```

| Builder Option           | Parameters       | Default | Description                                                                                                                                                 |
| ------------------------ | ---------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WithSSAOScreenSize`     | `width, height`  | 0, 0    | Initial screen dimensions                                                                                                                                   |
| `WithSSAOSampleCount`    | `count int`      | 16      | Hemisphere samples per pixel (1–32)                                                                                                                         |
| `WithSSAOScreenRadius`   | `pixels float32` | 24.0    | Screen-space sampling radius in pixels — the engine auto-computes the world-space hemisphere radius each frame from camera distance, FOV, and screen height |
| `WithSSAOBias`           | `bias float32`   | 0.025   | Depth bias to prevent self-occlusion                                                                                                                        |
| `WithSSAOPower`          | `power float32`  | 2.0     | Exponent for AO contrast                                                                                                                                    |
| `WithSSAOBlurRadius`     | `radius int`     | 4       | Bilateral blur half-width in texels                                                                                                                         |
| `WithSSAOHalfResolution` | `enabled bool`   | false   | Allocate textures at half resolution (¼ pixel count)                                                                                                        |
| `WithSSAOMaxSamples`     | `max int`        | 32      | GPU compile-time upper bound for sample array size                                                                                                          |

**Key Textures:**

| Texture | Format      | Description                                           |
| ------- | ----------- | ----------------------------------------------------- |
| Raw     | R8Unorm     | Pre-blur SSAO output                                  |
| Blurred | R8Unorm     | Final AO bound to the lit shader                      |
| Scratch | R8Unorm     | Intermediate texture between horizontal/vertical blur |
| Noise   | RGBA16Float | 4×4 random rotation vectors for kernel rotation       |

**Default BGPs:** `"ssao_compute"`, `"ssao_blur_h"`, `"ssao_blur_v"`

**Key Interface Methods:**

| Method                                                                | Description                                |
| --------------------------------------------------------------------- | ------------------------------------------ |
| `MaxSamples() int`                                                    | Returns the compile-time max samples value |
| `HalfResolution() bool` / `SetHalfResolution(enabled bool)`           | Whether SSAO runs at half resolution       |
| `LinearSampler() *wgpu.Sampler` / `SetLinearSampler(s *wgpu.Sampler)` | Linear sampler for SSAO texture lookups    |

The `CompositionHandler` manages the offscreen HDR render target, MSAA resolve, depth texture, and the full-screen composition pipeline that applies ACES tone mapping and gamma correction. When composition is active, the lit pass renders to an RGBA16Float texture instead of the swapchain; the composition pass then samples the HDR result (and any SSR contribution) and writes the final LDR output.

```go
comp := light.NewCompositionHandler(
    light.WithToneMappingEnabled(true),
    light.WithExposure(1.0),
)
```

| Builder Option               | Parameters          | Default | Description                                                 |
| ---------------------------- | ------------------- | ------- | ----------------------------------------------------------- |
| `WithCompositionScreenSize`  | `width, height`     | 0, 0    | Initial screen dimensions                                   |
| `WithToneMappingEnabled`     | `enabled bool`      | true    | Enables ACES tone mapping                                   |
| `WithExposure`               | `exposure float32`  | 1.0     | HDR exposure multiplier (1.0=neutral)                       |
| `WithAutoExposure`           | `enabled bool`      | false   | Enable GPU-driven luminance-based exposure adaptation.      |
| `WithAdaptSpeed`             | `speed float32`     | 1.0     | Rate at which exposure converges to the target (seconds⁻¹). |
| `WithMinExposure`            | `min float32`       | 0.1     | Minimum clamp for the adapted exposure value.               |
| `WithMaxExposure`            | `max float32`       | 10.0    | Maximum clamp for the adapted exposure value.               |
| `WithLuminanceWorkgroupSize` | `size int`          | 16      | Workgroup tile dimension for the luminance compute shader.  |
| `WithBloomEnabled`           | `enabled bool`      | false   | Enables bloom post-processing.                              |
| `WithBloomThreshold`         | `threshold float32` | 1.0     | Brightness threshold for bloom extraction (soft-knee).      |
| `WithBloomIntensity`         | `intensity float32` | 0.5     | Multiplier for the bloom contribution in the final image.   |

**Key Textures:**

| Texture | Format      | Description                                                |
| ------- | ----------- | ---------------------------------------------------------- |
| HDR     | RGBA16Float | Offscreen render target for the lit pass                   |
| MSAA    | RGBA16Float | Multi-sampled color attachment (resolves into HDR texture) |
| Depth   | Depth24Plus | Depth buffer for the offscreen HDR render pass             |

**Default BGPs:** `"composition"`

**Key Interface Methods:**

| Method                            | Returns        | Description                                                        |
| --------------------------------- | -------------- | ------------------------------------------------------------------ |
| `ToneMappingEnabled() bool`       | `bool`         | Whether ACES tone mapping is active.                               |
| `SetToneMappingEnabled(bool)`     | —              | Toggle tone mapping at runtime.                                    |
| `Exposure() float32`              | `float32`      | Current HDR exposure multiplier.                                   |
| `SetExposure(float32)`            | —              | Set the HDR exposure multiplier.                                   |
| `AutoExposureEnabled() bool`      | `bool`         | Whether GPU-driven auto-exposure is active.                        |
| `SetAutoExposureEnabled(bool)`    | —              | Toggle auto-exposure at runtime.                                   |
| `AdaptSpeed() float32`            | `float32`      | Exposure adaptation rate (exposure units/second).                  |
| `SetAdaptSpeed(float32)`          | —              | Set the adaptation rate.                                           |
| `MinExposure() float32`           | `float32`      | Minimum adapted exposure clamp value.                              |
| `SetMinExposure(float32)`         | —              | Set the minimum exposure clamp.                                    |
| `MaxExposure() float32`           | `float32`      | Maximum adapted exposure clamp value.                              |
| `SetMaxExposure(float32)`         | —              | Set the maximum exposure clamp.                                    |
| `ExposureBuffer() *wgpu.Buffer`   | `*wgpu.Buffer` | The GPU storage buffer holding the current adapted exposure value. |
| `SetExposureBuffer(*wgpu.Buffer)` | —              | Set the exposure storage buffer (called during init).              |
| `BloomEnabled() bool`             | `bool`         | Whether bloom post-processing is active.                           |
| `SetBloomEnabled(bool)`           | —              | Toggle bloom at runtime.                                           |
| `BloomThreshold() float32`        | `float32`      | Brightness threshold for bloom extraction.                         |
| `SetBloomThreshold(float32)`      | —              | Set the bloom brightness threshold.                                |
| `BloomIntensity() float32`        | `float32`      | Multiplier for the bloom contribution.                             |
| `SetBloomIntensity(float32)`      | —              | Set the bloom intensity multiplier.                                |

#### Bloom

The bloom system extracts bright pixels from the HDR buffer using a soft-knee brightness threshold, then progressively downsamples (13-tap box-tent filter, CoD:AW style) and upsamples (9-tap tent filter with additive blending) through a mip chain to produce a smooth glow. The result is added to the final image in the composition shader, after exposure but before tone mapping.

| Builder Option       | Parameters          | Default | Description                                                                                                |
| -------------------- | ------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `WithBloomEnabled`   | `enabled bool`      | `false` | Enables/disables bloom.                                                                                    |
| `WithBloomThreshold` | `threshold float32` | `1.0`   | Brightness threshold for bloom extraction; pixels below this contribute less. Uses a soft-knee transition. |
| `WithBloomIntensity` | `intensity float32` | `0.5`   | Multiplier for the bloom contribution added to the final image.                                            |

**Implementation details:**

- Two RGBA16Float mip chain textures (down chain + up chain) at half screen resolution, max 6 mip levels
- Per-mip BGPs with separate `DispatchComputeBatch` per mip for GPU barriers
- Threshold only applied on the first downsample pass (mip 0); subsequent mips downsample without threshold
- When disabled, a 1×1 black fallback texture is bound to composition binding 6

### SSRHandler

The `SSRHandler` manages ray march configuration, the SSR result texture, the Hi-Z depth pyramid (for accelerated ray marching), and associated compute pipelines. The SSR compute shader reads from the G-Buffer normals and the Hi-Z pyramid to march reflected rays in screen space.

```go
ssr := light.NewSSRHandler(
    light.WithSSRMaxSteps(64),
    light.WithSSRMaxDistance(50.0),
    light.WithSSRThickness(0.1),
    light.WithSSRStride(1.0),
    light.WithSSRRoughnessCutoff(0.5),
)
```

| Builder Option           | Parameters          | Default | Description                                     |
| ------------------------ | ------------------- | ------- | ----------------------------------------------- |
| `WithSSRScreenSize`      | `width, height`     | 0, 0    | Initial screen dimensions                       |
| `WithSSRMaxSteps`        | `steps int`         | 64      | Maximum ray march steps per pixel               |
| `WithSSRMaxDistance`     | `distance float32`  | 50.0    | Maximum ray march distance in view space        |
| `WithSSRThickness`       | `thickness float32` | 0.1     | Depth tolerance for hit detection               |
| `WithSSRStride`          | `stride float32`    | 1.0     | Step stride multiplier (1.0 = uniform stepping) |
| `WithSSRRoughnessCutoff` | `cutoff float32`    | 0.5     | Roughness above which SSR is skipped            |

**Key Textures:**

| Texture | Format      | Description                                       |
| ------- | ----------- | ------------------------------------------------- |
| SSR     | RGBA16Float | Half-resolution result (RGB=color, A=confidence)  |
| Hi-Z    | R32Float    | Full mip chain min-depth pyramid for ray marching |

**Hi-Z Pyramid:** The SSR handler also stores per-mip read views (`HiZMipReadViews`) and storage views (`HiZStorageViews`) for the Hi-Z downsample pipeline. `HiZMipCount()` returns the number of mip levels.

**Default BGPs:** `"ssr_compute"`

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

| Option                      | Parameters          | Default        | Description                                  |
| --------------------------- | ------------------- | -------------- | -------------------------------------------- |
| `WithShadowNearFar`         | `near, far float32` | `0.1`, `200.0` | Near/far planes for shadow projection        |
| `WithShadowNormalBiasScale` | `scale float32`     | `3.0`          | Normal-offset bias multiplier                |
| `WithShadowMapResolution`   | `resolution int`    | `2048`         | CSM atlas resolution in texels               |
| `WithPCFRadius`             | `radius float32`    | `1.0`          | Poisson disk PCF kernel radius in texels     |
| `WithPCFSamples`            | `samples uint32`    | `16`           | Poisson disk tap count                       |
| `WithShadowInnerRadius`     | `radius float32`    | `100.0`        | Inner cascade sphere radius in world units   |
| `WithLightShadowTileSize`   | `size int`          | `1024`         | Tile size for the spot/point atlas in texels |

**Key Interface Methods:**

| Method                                                                                       | Description                          |
| -------------------------------------------------------------------------------------------- | ------------------------------------ |
| `ShadowNear() float32`                                                                       | Near plane for shadow projection     |
| `ShadowFar() float32`                                                                        | Far plane for shadow projection      |
| `ShadowNormalBiasScale() float32`                                                            | Normal-offset bias multiplier        |
| `ShadowMapResolution() int`                                                                  | CSM atlas resolution in texels       |
| `PCFRadius() float32`                                                                        | PCF kernel radius in texels          |
| `PCFSamples() uint32`                                                                        | PCF tap count                        |
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

### GPUBlurParams

Uniform data for the separable SSAO blur compute shader.

| Field          | Type       | Offset | Description                                                      |
| -------------- | ---------- | ------ | ---------------------------------------------------------------- |
| `Direction`    | `[2]int32` | 0      | `(1,0)` for horizontal pass, `(0,1)` for vertical pass           |
| `Radius`       | `int32`    | 8      | Half-width of the filter kernel in texels                        |
| `GBufferScale` | `int32`    | 12     | Lookup scale factor for half-resolution SSAO                     |
| `CascadeWidth` | `int32`    | 16     | Per-cascade atlas column width; `0` disables horizontal clamping |
| `_pad`         | `int32`    | 20     | Padding to 24 bytes                                              |

Total size: **24 bytes**.

### GPUGBufferOutput

G-Buffer MRT fragment output (written by the G-Buffer pre-pass, not typically CPU-uploaded).

| Field      | Type         | Offset | Description                                      |
| ---------- | ------------ | ------ | ------------------------------------------------ |
| `Position` | `[4]float32` | 0      | World XYZ + linear depth in W                    |
| `Normal`   | `[4]float32` | 16     | World normal XYZ (packed [0,1]) + roughness in W |
| `Albedo`   | `[4]float32` | 32     | Albedo RGB + metallic in A                       |

**Size:** 48 bytes

### GPUSSAOParams

Uniform data for the SSAO compute shader.

| Field            | Type          | Offset | Description                                     |
| ---------------- | ------------- | ------ | ----------------------------------------------- |
| `Projection`     | `[16]float32` | 0      | View-projection matrix                          |
| `InvViewProj`    | `[16]float32` | 64     | Inverse view-projection matrix                  |
| `Radius`         | `float32`     | 128    | Hemisphere sample radius in world units         |
| `Bias`           | `float32`     | 132    | Depth comparison bias to prevent self-occlusion |
| `Power`          | `float32`     | 136    | Exponent applied to the final AO value          |
| `SampleCount`    | `uint32`      | 140    | Number of hemisphere samples (max 32)           |
| `ScreenWidth`    | `float32`     | 144    | Screen width in pixels                          |
| `ScreenHeight`   | `float32`     | 148    | Screen height in pixels                         |
| `GBufferScale`   | `float32`     | 152    | Coordinate multiplier for G-Buffer lookups      |
| `_pad`           | `float32`     | 156    | Padding                                         |
| `CameraPosition` | `[3]float32`  | 160    | World-space camera position                     |
| `_pad2`          | `float32`     | 172    | Padding to 176 bytes                            |

**Size:** 176 bytes

### GPUCompositionParams

Uniform data for the composition fragment shader.

| Field                 | Type      | Offset | Description                                      |
| --------------------- | --------- | ------ | ------------------------------------------------ |
| `ToneMappingEnabled`  | `uint32`  | 0      | 1=ACES tone mapping applied, 0=bypassed          |
| `Exposure`            | `float32` | 4      | Exposure multiplier before tone mapping          |
| `AutoExposureEnabled` | `uint32`  | 8      | Non-zero when GPU-driven auto-exposure is active |
| `_pad2`               | `uint32`  | 12     | Padding to 16 bytes                              |

**Size:** 16 bytes

### GPULuminanceParams

Uniform data for the luminance compute pass used by auto-exposure adaptation.

| Offset | Field                   | Type  | Description                                        |
| ------ | ----------------------- | ----- | -------------------------------------------------- |
| 0      | `screen_width`          | `u32` | HDR texture width in pixels.                       |
| 4      | `screen_height`         | `u32` | HDR texture height in pixels.                      |
| 8      | `adapt_speed`           | `f32` | Exposure adaptation speed (units/second).          |
| 12     | `delta_time`            | `f32` | Frame delta time in seconds.                       |
| 16     | `min_exposure`          | `f32` | Minimum clamped exposure value.                    |
| 20     | `max_exposure`          | `f32` | Maximum clamped exposure value.                    |
| 24     | `key_value`             | `f32` | Middle-gray key value for exposure mapping (0.18). |
| 28     | `auto_exposure_enabled` | `u32` | Non-zero when auto-exposure is active.             |

**Size:** 32 bytes

**Workgroup size:** Configurable via `WithLuminanceWorkgroupSize(size int)` on `CompositionHandler`; defaults to `16`.

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

### GPUSSRParams

Uniform data for the screen-space reflections compute shader.

| Field             | Type          | Offset | Description                                    |
| ----------------- | ------------- | ------ | ---------------------------------------------- |
| `Projection`      | `[16]float32` | 0      | Projection matrix                              |
| `InvProjection`   | `[16]float32` | 64     | Inverse projection matrix                      |
| `View`            | `[16]float32` | 128    | View matrix                                    |
| `MaxDistance`     | `float32`     | 192    | Maximum ray march distance in view space       |
| `Thickness`       | `float32`     | 196    | Depth thickness tolerance for hit detection    |
| `Stride`          | `float32`     | 200    | Step stride multiplier (unused in Hi-Z mode)   |
| `MaxSteps`        | `uint32`      | 204    | Maximum number of ray march steps              |
| `ScreenWidth`     | `float32`     | 208    | Screen width in pixels                         |
| `ScreenHeight`    | `float32`     | 212    | Screen height in pixels                        |
| `RoughnessCutoff` | `float32`     | 216    | Roughness above which SSR is skipped           |
| `HiZMipCount`     | `uint32`      | 220    | Number of Hi-Z mip levels in the depth pyramid |

**Size:** 224 bytes

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

| File                                | Contents                                                                                                                                                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `light.go`                          | `Light` interface and `LightType` constants                                                                                                                                                                                                |
| `light_impl.go`                     | Unexported `light` struct and method implementations                                                                                                                                                                                       |
| `light_builder.go`                  | `LightBuilderOption` type, `With*` functions, `NewLight` constructor                                                                                                                                                                       |
| `light_handler.go`                  | `LightingHandler` interface                                                                                                                                                                                                                |
| `light_handler_impl.go`             | Unexported `lightingHandler` struct and method implementations                                                                                                                                                                             |
| `light_handler_builder.go`          | `LightingHandlerOption` type, `With*` functions, `NewLightingHandler` constructor                                                                                                                                                          |
| `gbuffer_handler.go`                | `GBufferHandler` interface                                                                                                                                                                                                                 |
| `gbuffer_handler_impl.go`           | Unexported `gBufferHandler` struct                                                                                                                                                                                                         |
| `gbuffer_handler_builder.go`        | `GBufferHandlerOption` type and `NewGBufferHandler` constructor                                                                                                                                                                            |
| `ssao_handler.go`                   | `SSAOHandler` interface                                                                                                                                                                                                                    |
| `ssao_handler_impl.go`              | Unexported `ssaoHandler` struct                                                                                                                                                                                                            |
| `ssao_handler_builder.go`           | `SSAOHandlerOption` type and `NewSSAOHandler` constructor                                                                                                                                                                                  |
| `ssr_handler.go`                    | `SSRHandler` interface                                                                                                                                                                                                                     |
| `ssr_handler_impl.go`               | Unexported `ssrHandler` struct                                                                                                                                                                                                             |
| `ssr_handler_builder.go`            | `SSRHandlerOption` type and `NewSSRHandler` constructor                                                                                                                                                                                    |
| `composition_handler.go`            | `CompositionHandler` interface                                                                                                                                                                                                             |
| `composition_handler_impl.go`       | Unexported `compositionHandler` struct                                                                                                                                                                                                     |
| `composition_handler_builder.go`    | `CompositionHandlerOption` type and `NewCompositionHandler` constructor                                                                                                                                                                    |
| `shadow_handler.go`                 | `ShadowHandler` interface and `ShadowType` constants                                                                                                                                                                                       |
| `shadow_handler_impl.go`            | Unexported `shadowHandler` struct                                                                                                                                                                                                          |
| `shadow_handler_builder.go`         | `ShadowHandlerOption` type and `NewShadowHandler` constructor                                                                                                                                                                              |
| `contact_shadow_handler.go`         | `ContactShadowHandler` interface                                                                                                                                                                                                           |
| `contact_shadow_handler_impl.go`    | Unexported `contactShadowHandler` struct                                                                                                                                                                                                   |
| `contact_shadow_handler_builder.go` | `ContactShadowHandlerOption` type and `NewContactShadowHandler` constructor                                                                                                                                                                |
| `gpu_types.go`                      | All GPU-marshaled structs (`GPULight`, `GPUCSMData`, `GPUCSMCascade`, `GPUShadowUniform`, `GPULightShadowEntry`, `GPUSSAOParams`, `GPUBlurParams`, `GPUTileUniforms`, `GPULuminanceParams`, `GPUContactShadowParams`) and helper functions |
