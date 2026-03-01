# Oxy Light System

The `light` package provides the lighting, shadow mapping, Forward+ tile culling, and full global illumination (GI) pipeline for the Oxy engine. It supports directional, point, and spot light types, all sharing a single `Light` interface. Lights are marshaled into GPU storage buffers each frame and evaluated in a tiled Forward+ rendering pipeline.

Shadow mapping uses Variance Shadow Maps (VSM) with an optional Percentage-Closer Soft Shadows (PCSS) mode backed by a Summed-Area Table. A separable Gaussian blur produces constant-width soft shadows in default VSM mode, while PCSS uses per-pixel variable-width filtering for contact-hardening soft shadows.

The GI pipeline includes a G-Buffer MRT pre-pass, Screen-Space Ambient Occlusion (SSAO) with bilateral blur, irradiance probe grids storing L2 spherical harmonics for diffuse indirect lighting, Screen-Space Reflections (SSR) via Hi-Z ray marching, and a final composition pass with ACES tone mapping and HDR rendering.

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
- [Shadow Mapping (VSM)](#shadow-mapping-vsm)
  - [Shadow Constants](#shadow-constants)
  - [VSM Constants](#vsm-constants)
- [PCSS (Percentage-Closer Soft Shadows)](#pcss-percentage-closer-soft-shadows)
- [GI Sub-Handlers](#gi-sub-handlers)
  - [GBufferHandler](#gbufferhandler)
  - [SSAOHandler](#ssaohandler)
  - [CompositionHandler](#compositionhandler)
  - [SSRHandler](#ssrhandler)
  - [IrradianceProbeGrid](#irradianceprobegrid)
- [GPU Types](#gpu-types)
  - [GPULight](#gpulight)
  - [GPULightHeader](#gpulightheader)
  - [GPUShadowData](#gpushadowdata)
  - [GPUShadowUniform](#gpushadowuniform)
  - [GPUBlurParams](#gpublurparams)
  - [GPUSATParams](#gpusatparams)
  - [GPUGBufferOutput](#gpugbufferoutput)
  - [GPUSSAOParams](#gpussaoparams)
  - [GPUIrradianceProbe](#gpuirradianceprobe)
  - [GPUProbeGridParams](#gpuprobegridparams)
  - [GPUProbeBakeCamera](#gpuprobebakecamera)
  - [GPUSHProjectParams](#gpushprojectparams)
  - [GPUCompositionParams](#gpucompositionparams)
  - [GPUSSRParams](#gpussrparams)
  - [GPULightCullUniforms](#gpulightculluniforms)
  - [GPUTileUniforms](#gputileuniforms)
- [Helper Functions](#helper-functions)
- [Usage Example](#usage-example)

---

## Overview

The light system is designed around five pillars:

1. **Light** — A scene-level entity with type, position, direction, color, intensity, range, and cone angles. All three light types share the same interface; type-specific properties return zero values when not applicable.
2. **Forward+ Tile Culling** — The screen is divided into tiles (`TileSize × TileSize` pixels). A compute shader assigns lights to tiles so the fragment shader only evaluates lights relevant to each tile.
3. **Variance Shadow Mapping (VSM)** — Shadow-casting lights render a depth-moments pass (storing `depth` and `depth²`) into an RG32Float texture each frame. A separable blur smooths the moments for constant-width soft shadows. Chebyshev's inequality is used in the lit fragment shader instead of PCF for smoother, filter-friendly shadow boundaries. An optional PCSS mode replaces the blur with a Summed-Area Table for per-pixel variable-width penumbrae.
4. **Global Illumination Pipeline** — A G-Buffer MRT pre-pass captures per-pixel normals, albedo, and linear depth. SSAO uses hemisphere sampling on the G-Buffer to compute ambient occlusion. An irradiance probe grid stores L2 spherical harmonics for diffuse indirect lighting. SSR via Hi-Z ray marching adds specular reflections. A final composition pass applies ACES tone mapping to the HDR result.
5. **Sub-Handler Architecture** — The `LightingHandler` owns five lazily-initialized sub-handlers (`GBufferHandler`, `SSAOHandler`, `CompositionHandler`, `SSRHandler`, `IrradianceProbeGrid`) that manage their own textures, pipelines, and bind groups.

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

The `Light` interface embeds `common.Delegate[Light]`, exposing `SetDelegate(delegate Light)`. In production code the delegate is set to the instance itself during construction. In test code the delegate can be replaced with a mock.

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

## LightingHandler

The `LightingHandler` manages the light list, ambient color, VSM shadow mapping configuration, PCSS/SAT state, Forward+ tile culling, GI sub-handlers, and all associated GPU resources (bind group providers, pipeline keys, shadow textures). It is created via `NewLightingHandler` with builder options and attached to a scene. GPU resources are initialized lazily by the scene when the first light is added.

Thread safety is provided by the owning scene's mutex — the handler itself does not perform internal locking.

### Creating a LightingHandler

```go
handler := light.NewLightingHandler(
    light.WithShadowHalfExtent(50.0),
    light.WithShadowNearFar(0.1, 300.0),
    light.WithShadowBias(0.002),
    light.WithShadowNormalBiasScale(3.0),
    light.WithShadowMapResolution(4096),
    light.WithAmbientColor([3]float32{0.05, 0.05, 0.08}),
    light.WithVSMBlurRadius(6),
    light.WithVSMMinVariance(0.0001),
    light.WithVSMLightBleedReduction(0.4),
    light.WithPCSSEnabled(true),
    light.WithVSMLightSize(2.0),
)
```

Defaults applied before options:

| Parameter                 | Default                                 |
| ------------------------- | --------------------------------------- |
| Enabled                   | `false`                                 |
| Lights                    | empty                                   |
| Ambient color             | `(0, 0, 0)` (black)                     |
| Shadow half-extent        | `DefaultShadowHalfExtent` (`40.0`)      |
| Shadow near               | `DefaultShadowNear` (`0.1`)             |
| Shadow far                | `DefaultShadowFar` (`200.0`)            |
| Shadow bias               | `DefaultShadowBias` (`0.001`)           |
| Shadow normal bias        | `DefaultShadowNormalBiasScale` (`3.0`)  |
| Shadow map resolution     | `ShadowMapResolution` (`2048`)          |
| VSM blur radius           | `DefaultVSMBlurRadius` (`4`)            |
| VSM min variance          | `DefaultVSMMinVariance` (`0.00001`)     |
| VSM light bleed reduction | `DefaultVSMLightBleedReduction` (`0.3`) |
| VSM light size            | `DefaultVSMLightSize` (`1.0`)           |
| PCSS enabled              | `false`                                 |

The constructor pre-creates the following named `BindGroupProvider` entries: `"lights"`, `"shadow_data"`, `"shadow_lit"`, `"light_cull"`, `"tile_lit"`, `"vsm_blur_h"`, `"vsm_blur_v"`, `"sat_prepare"`, `"ssao_lit"`, `"probe_lit"`, `"composition_lit"`, `"ssr_lit"`. When PCSS is enabled, additional per-pass `"sat_pass_N"` BGPs are created for each recursive-doubling prefix-sum dispatch.

The constructor also auto-creates default GI sub-handlers (`GBufferHandler`, `SSAOHandler`, `CompositionHandler`, `SSRHandler`) if not explicitly provided via builder options. The `IrradianceProbeGrid` must be provided explicitly if probe-based GI is desired.

### LightingHandler Builder Options

All options follow the `LightingHandlerOption` functional option pattern.

| Option                       | Parameters                    | Description                                                    |
| ---------------------------- | ----------------------------- | -------------------------------------------------------------- |
| `WithShadowHalfExtent`       | `halfExtent float32`          | Orthographic frustum half-extent in world units                |
| `WithShadowNearFar`          | `near, far float32`           | Near/far planes for shadow projection                          |
| `WithShadowBias`             | `bias float32`                | Depth comparison bias to reduce shadow acne                    |
| `WithShadowNormalBiasScale`  | `scale float32`               | Normal-offset bias multiplier on per-texel world-size          |
| `WithShadowMapResolution`    | `resolution int`              | Shadow depth texture resolution in texels                      |
| `WithAmbientColor`           | `color [3]float32`            | Initial ambient light color                                    |
| `WithVSMBlurRadius`          | `radius int`                  | Half-width (in texels) of the separable VSM blur kernel        |
| `WithVSMMinVariance`         | `minVariance float32`         | Minimum variance clamp for Chebyshev's inequality              |
| `WithVSMLightBleedReduction` | `reduction float32`           | Exponent to reduce light-bleeding artifacts (typical: 0.1–0.6) |
| `WithVSMLightSize`           | `size float32`                | World-space light size for PCSS penumbra estimation            |
| `WithPCSSEnabled`            | `enabled bool`                | Enables PCSS variable-width soft shadows via SAT               |
| `WithGBufferHandler`         | `handler GBufferHandler`      | Overrides the default G-Buffer handler                         |
| `WithSSAOHandler`            | `handler SSAOHandler`         | Overrides the default SSAO handler                             |
| `WithProbeGrid`              | `handler IrradianceProbeGrid` | Attaches a pre-configured irradiance probe grid                |
| `WithCompositionHandler`     | `handler CompositionHandler`  | Overrides the default composition/tone mapping handler         |
| `WithSSRHandler`             | `handler SSRHandler`          | Overrides the default SSR handler                              |

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

#### VSM Shadow Resources

| Method                                            | Description                                                  |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `VSMTexture() *wgpu.Texture`                      | Returns the RG32Float variance shadow map texture            |
| `SetVSMTexture(t *wgpu.Texture)`                  | Sets the VSM texture                                         |
| `VSMTextureView() *wgpu.TextureView`              | Returns the VSM texture view                                 |
| `SetVSMTextureView(tv *wgpu.TextureView)`         | Sets the VSM texture view                                    |
| `VSMScratchTexture() *wgpu.Texture`               | Returns the scratch texture for the separable blur pass      |
| `SetVSMScratchTexture(t *wgpu.Texture)`           | Sets the scratch texture                                     |
| `VSMScratchTextureView() *wgpu.TextureView`       | Returns the scratch texture view                             |
| `SetVSMScratchTextureView(tv *wgpu.TextureView)`  | Sets the scratch texture view                                |
| `VSMAuxDepthTexture() *wgpu.Texture`              | Returns the auxiliary Depth32Float texture for VSM z-testing |
| `SetVSMAuxDepthTexture(t *wgpu.Texture)`          | Sets the auxiliary depth texture                             |
| `VSMAuxDepthTextureView() *wgpu.TextureView`      | Returns the auxiliary depth texture view                     |
| `SetVSMAuxDepthTextureView(tv *wgpu.TextureView)` | Sets the auxiliary depth texture view                        |
| `VSMLinearSampler() *wgpu.Sampler`                | Returns the linear sampler for VSM texture lookups           |
| `SetVSMLinearSampler(s *wgpu.Sampler)`            | Sets the linear sampler                                      |

#### VSM Configuration (Read-Only)

| Method                             | Description                                              |
| ---------------------------------- | -------------------------------------------------------- |
| `VSMBlurRadius() int`              | Half-width (in texels) of the separable blur kernel      |
| `VSMMinVariance() float32`         | Minimum variance clamp for Chebyshev's inequality        |
| `VSMLightBleedReduction() float32` | Exponent applied to reduce light-bleeding artifacts      |
| `VSMLightSize() float32`           | World-space area light size for PCSS penumbra estimation |

#### PCSS / SAT Resources

| Method                                     | Description                                             |
| ------------------------------------------ | ------------------------------------------------------- |
| `PCSSEnabled() bool`                       | Whether PCSS contact-hardening soft shadows are enabled |
| `SetPCSSEnabled(enabled bool)`             | Enables or disables PCSS                                |
| `SATTextureA() *wgpu.Texture`              | Returns the first RGBA32Float SAT ping-pong texture     |
| `SetSATTextureA(t *wgpu.Texture)`          | Sets SAT texture A                                      |
| `SATTextureAView() *wgpu.TextureView`      | Returns the view for SAT texture A                      |
| `SetSATTextureAView(tv *wgpu.TextureView)` | Sets the view for SAT texture A                         |
| `SATTextureB() *wgpu.Texture`              | Returns the second RGBA32Float SAT ping-pong texture    |
| `SetSATTextureB(t *wgpu.Texture)`          | Sets SAT texture B                                      |
| `SATTextureBView() *wgpu.TextureView`      | Returns the view for SAT texture B                      |
| `SetSATTextureBView(tv *wgpu.TextureView)` | Sets the view for SAT texture B                         |

#### GI Sub-Handler Accessors

| Method                                    | Description                                                 |
| ----------------------------------------- | ----------------------------------------------------------- |
| `GBufferHandler() GBufferHandler`         | Returns the G-Buffer handler, or nil if not configured      |
| `SSAOHandler() SSAOHandler`               | Returns the SSAO handler, or nil if not configured          |
| `ProbeGrid() IrradianceProbeGrid`         | Returns the irradiance probe grid, or nil if not configured |
| `CompositionHandler() CompositionHandler` | Returns the composition/tone mapping handler, or nil        |
| `SSRHandler() SSRHandler`                 | Returns the SSR handler, or nil if not configured           |

#### Shadow Configuration (Read-Only)

| Method                            | Description                                     |
| --------------------------------- | ----------------------------------------------- |
| `ShadowHalfExtent() float32`      | Orthographic frustum half-extent in world units |
| `ShadowNear() float32`            | Near plane for shadow projection                |
| `ShadowFar() float32`             | Far plane for shadow projection                 |
| `ShadowBias() float32`            | Depth comparison bias                           |
| `ShadowNormalBiasScale() float32` | Normal-offset bias multiplier                   |
| `ShadowMapResolution() int`       | Shadow depth texture resolution in texels       |

#### Screen & Tile State

| Method                      | Description                                            |
| --------------------------- | ------------------------------------------------------ |
| `ScreenWidth() int`         | Current screen width in pixels for tile calculations   |
| `ScreenHeight() int`        | Current screen height in pixels for tile calculations  |
| `TileCountX() uint32`       | Number of Forward+ tile columns                        |
| `TileCountY() uint32`       | Number of Forward+ tile rows                           |
| `Resize(width, height int)` | Updates screen dimensions and recalculates tile counts |

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

## Shadow Mapping (VSM)

Shadow-casting lights render a depth-moments pass into an RG32Float variance shadow map texture each frame. The VSM fragment shader outputs `(depth, depth²)` where depth is linearly normalized between the shadow near and far planes. A separable Gaussian blur smooths the moments to produce soft shadow edges. In the lit fragment shader, Chebyshev's inequality replaces PCF for smoother, filter-friendly shadow boundaries.

When PCSS is enabled, the blur pass is replaced by a Summed-Area Table (SAT) generation pipeline. The lit shader uses the SAT to perform per-pixel variable-width box filtering of the moments, producing contact-hardening soft shadows where the penumbra width is proportional to the blocker-to-receiver distance.

### Shadow Constants

| Constant                       | Value   | Description                                                              |
| ------------------------------ | ------- | ------------------------------------------------------------------------ |
| `ShadowMapResolution`          | `2048`  | Default shadow depth texture size (width and height in texels)           |
| `DefaultShadowHalfExtent`      | `40.0`  | Orthographic frustum half-extent in world units                          |
| `DefaultShadowNear`            | `0.1`   | Near plane for shadow projection                                         |
| `DefaultShadowFar`             | `200.0` | Far plane for shadow projection                                          |
| `DefaultShadowBias`            | `0.001` | Constant depth bias to reduce shadow acne                                |
| `DefaultShadowNormalBiasScale` | `3.0`   | Multiplier on texel world-size for normal-offset bias (typical: 2.0–4.0) |

### VSM Constants

| Constant                        | Value     | Description                                                                          |
| ------------------------------- | --------- | ------------------------------------------------------------------------------------ |
| `DefaultVSMBlurRadius`          | `4`       | Half-width (texels) of the separable blur kernel; full width is `2*radius+1`         |
| `DefaultVSMMinVariance`         | `0.00001` | Minimum variance clamp for Chebyshev's inequality (prevents hard edges on flat geom) |
| `DefaultVSMLightBleedReduction` | `0.3`     | Exponent to reduce light-bleeding artifacts (typical: 0.1–0.6)                       |
| `DefaultVSMLightSize`           | `1.0`     | World-space area light size for PCSS penumbra estimation                             |

---

## PCSS (Percentage-Closer Soft Shadows)

PCSS provides contact-hardening soft shadows by varying the filter width per pixel based on the blocker distance estimated from the VSM moments. When enabled via `WithPCSSEnabled(true)`, the shadow pipeline replaces the constant-width blur with a Summed-Area Table:

1. **Prepare** — The RG32Float moments are distributed into RGBA32Float with precision splitting.
2. **Recursive doubling** — `2×log₂(resolution)` prefix-sum passes (horizontal then vertical) build the complete SAT, ping-ponging between two RGBA32Float textures.
3. **Lit shader** — The fragment shader uses the SAT to compute a variable-width box filter over the moments, then applies Chebyshev's inequality with the filtered mean and variance.

All SAT passes are pre-created with dedicated bind group providers and uniform buffers so that every dispatch is batched into a single GPU command encoder submission.

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

| Texture  | Format      | Contents                                         |
| -------- | ----------- | ------------------------------------------------ |
| Position | RGBA16Float | World XYZ + linear depth in W                    |
| Normal   | RGBA16Float | World normal XYZ (packed [0,1]) + roughness in W |
| Albedo   | RGBA8Unorm  | Albedo RGB + metallic in A                       |
| Depth    | Depth24Plus | Shared depth for the pre-pass                    |

**Key Interface Methods:**

| Method                                   | Description                                            |
| ---------------------------------------- | ------------------------------------------------------ |
| `Enabled() bool`                         | Whether GPU resources are initialized                  |
| `PositionTexture() / SetPositionTexture` | World-space position MRT texture                       |
| `NormalTexture() / SetNormalTexture`     | Normals + roughness MRT texture                        |
| `AlbedoTexture() / SetAlbedoTexture`     | Albedo + metallic MRT texture                          |
| `DepthTexture() / SetDepthTexture`       | Shared depth texture for the G-Buffer pass             |
| `PipelineKey(name) / SetPipelineKey`     | Pipeline key storage for the G-Buffer render pipeline  |
| `Resize(width, height)`                  | Updates screen dimensions (does not recreate textures) |

### SSAOHandler

The `SSAOHandler` manages the hemisphere sampling kernel, noise texture, raw and blurred occlusion textures, and bilateral blur pipeline. The raw AO is computed in a compute shader and then blurred with a separable bilateral filter to preserve edges.

```go
ssao := light.NewSSAOHandler(
    light.WithSSAOSampleCount(16),
    light.WithSSAORadius(0.5),
    light.WithSSAOBias(0.025),
    light.WithSSAOPower(2.0),
    light.WithSSAOBlurRadius(4),
    light.WithSSAOHalfResolution(false),
)
```

| Builder Option           | Parameters       | Default | Description                                          |
| ------------------------ | ---------------- | ------- | ---------------------------------------------------- |
| `WithSSAOScreenSize`     | `width, height`  | 0, 0    | Initial screen dimensions                            |
| `WithSSAOSampleCount`    | `count int`      | 16      | Hemisphere samples per pixel (1–32)                  |
| `WithSSAORadius`         | `radius float32` | 0.5     | Sampling radius in world units                       |
| `WithSSAOBias`           | `bias float32`   | 0.025   | Depth bias to prevent self-occlusion                 |
| `WithSSAOPower`          | `power float32`  | 2.0     | Exponent for AO contrast                             |
| `WithSSAOBlurRadius`     | `radius int`     | 4       | Bilateral blur half-width in texels                  |
| `WithSSAOHalfResolution` | `enabled bool`   | false   | Allocate textures at half resolution (¼ pixel count) |

**Key Textures:**

| Texture | Format      | Description                                           |
| ------- | ----------- | ----------------------------------------------------- |
| Raw     | R8Unorm     | Pre-blur SSAO output                                  |
| Blurred | R8Unorm     | Final AO bound to the lit shader                      |
| Scratch | R8Unorm     | Intermediate texture between horizontal/vertical blur |
| Noise   | RGBA16Float | 4×4 random rotation vectors for kernel rotation       |

**Default BGPs:** `"ssao_compute"`, `"ssao_blur_h"`, `"ssao_blur_v"`

### CompositionHandler

The `CompositionHandler` manages the offscreen HDR render target, MSAA resolve, depth texture, and the full-screen composition pipeline that applies ACES tone mapping and gamma correction. When composition is active, the lit pass renders to an RGBA16Float texture instead of the swapchain; the composition pass then samples the HDR result (and any SSR contribution) and writes the final LDR output.

```go
comp := light.NewCompositionHandler(
    light.WithToneMappingEnabled(true),
    light.WithExposure(1.0),
)
```

| Builder Option              | Parameters         | Default | Description                           |
| --------------------------- | ------------------ | ------- | ------------------------------------- |
| `WithCompositionScreenSize` | `width, height`    | 0, 0    | Initial screen dimensions             |
| `WithToneMappingEnabled`    | `enabled bool`     | true    | Enables ACES tone mapping             |
| `WithExposure`              | `exposure float32` | 1.0     | HDR exposure multiplier (1.0=neutral) |

**Key Textures:**

| Texture | Format      | Description                                                |
| ------- | ----------- | ---------------------------------------------------------- |
| HDR     | RGBA16Float | Offscreen render target for the lit pass                   |
| MSAA    | RGBA16Float | Multi-sampled color attachment (resolves into HDR texture) |
| Depth   | Depth24Plus | Depth buffer for the offscreen HDR render pass             |

**Default BGPs:** `"composition"`

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

### IrradianceProbeGrid

The `IrradianceProbeGrid` stores a regular 3-D grid of irradiance probes, each containing L2 spherical harmonic (SH) coefficients that encode low-frequency indirect illumination. During baking, the scene is rendered from each probe position into a tiny cubemap (6 faces) and projected into SH coefficients via a compute shader. The SH data is uploaded to a GPU storage buffer and sampled in the lit fragment shader for diffuse indirect lighting via trilinear probe interpolation.

Unlike the other sub-handlers, the probe grid must be explicitly provided via `WithProbeGrid(...)` — it is not auto-created.

```go
probes := light.NewIrradianceProbeGrid(
    light.WithProbeGridCounts(8, 4, 8),
    light.WithProbeGridBounds(
        [3]float32{-10, -2, -10},
        [3]float32{10, 6, 10},
    ),
    light.WithProbeBakeResolution(32),
)
```

| Builder Option            | Parameters            | Default                | Description                           |
| ------------------------- | --------------------- | ---------------------- | ------------------------------------- |
| `WithProbeGridCounts`     | `x, y, z int`         | 8, 4, 8                | Probes per axis                       |
| `WithProbeGridBounds`     | `min, max [3]float32` | (-10,-2,-10)→(10,6,10) | World-space AABB of the grid          |
| `WithProbeBakeResolution` | `resolution int`      | 32                     | Cubemap face resolution (px per edge) |

**Key Interface Methods:**

| Method                                                | Description                                              |
| ----------------------------------------------------- | -------------------------------------------------------- |
| `CountX() / CountY() / CountZ()`                      | Probe counts per axis                                    |
| `TotalProbes()`                                       | Total probes (X × Y × Z)                                 |
| `GridMin() / GridMax() / Spacing()`                   | World-space bounds and per-axis spacing                  |
| `ProbeIndex(x, y, z)`                                 | Flat index from grid coordinates                         |
| `Probe(index) / SetProbe(index, p)`                   | CPU-side probe data access                               |
| `DirtyProbes() / MarkAllDirty() / ClearDirtyProbes()` | Incremental bake management                              |
| `ProbeBuffer() / SetProbeBuffer()`                    | GPU storage buffer for the full probe array              |
| `GridParamsBuffer() / SetGridParamsBuffer()`          | GPU uniform buffer for `GPUProbeGridParams`              |
| `BakeColorTexture() / BakeDepthTexture()`             | Cubemap face bake render targets (reused per-probe/face) |

**Default BGPs:** `"probe_grid"`, `"probe_sh_project"`, `"probe_bake_camera"`

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

Directional shadow data for the lit fragment shader (VSM mode).

| Field                 | Type          | Offset | Description                                                  |
| --------------------- | ------------- | ------ | ------------------------------------------------------------ |
| `LightVP`             | `[16]float32` | 0      | Orthographic view-projection from light's perspective        |
| `LightView`           | `[16]float32` | 64     | View-only matrix (no projection) for VSM linear depth        |
| `TexelSize`           | `[2]float32`  | 128    | `1.0 / resolution` for VSM texel calculations                |
| `Bias`                | `float32`     | 136    | Depth comparison bias                                        |
| `NormalBias`          | `float32`     | 140    | World-space normal-offset distance                           |
| `ShadowNear`          | `float32`     | 144    | Near plane for linear depth normalization                    |
| `ShadowFar`           | `float32`     | 148    | Far plane for linear depth normalization                     |
| `MinVariance`         | `float32`     | 152    | Minimum variance clamp for Chebyshev's inequality            |
| `LightBleedReduction` | `float32`     | 156    | Exponent to reduce light-bleeding artifacts                  |
| `LightSize`           | `float32`     | 160    | World-space light size for PCSS penumbra estimation          |
| `ShadowHalfExtent`    | `float32`     | 164    | Orthographic frustum half-size for world-to-texel conversion |
| `_pad`                | `[2]float32`  | 168    | Padding to 176 bytes (16-byte alignment)                     |

**Size:** 176 bytes

Additional methods:

- `ComputeDirectionalLightVP(lightDir, centerX, centerY, centerZ, halfExtent, near, far)` — Builds the orthographic view-projection matrix centered on the camera position. Also stores the view-only matrix and near/far planes for VSM linear depth.
- `ComputeNormalBias(halfExtent, scale, resolution)` — Derives the world-space normal-offset bias from shadow map parameters.

### GPUShadowUniform

Shadow vertex shader uniform for VSM depth-moments pass.

| Field        | Type          | Offset | Description                                           |
| ------------ | ------------- | ------ | ----------------------------------------------------- |
| `LightVP`    | `[16]float32` | 0      | Orthographic view-projection from light's perspective |
| `LightView`  | `[16]float32` | 64     | View-only matrix for linear depth in VSM              |
| `ShadowNear` | `float32`     | 128    | Near plane for linear depth normalization             |
| `ShadowFar`  | `float32`     | 132    | Far plane for linear depth normalization              |

**Size:** 136 bytes

### GPUBlurParams

Uniform data for the separable VSM blur compute shader.

| Field          | Type       | Offset | Description                                                          |
| -------------- | ---------- | ------ | -------------------------------------------------------------------- |
| `Direction`    | `[2]int32` | 0      | `(1,0)` for horizontal, `(0,1)` for vertical                         |
| `Radius`       | `int32`    | 8      | Half-width of the box filter kernel in texels                        |
| `GBufferScale` | `int32`    | 12     | Coordinate multiplier for depth lookups (1 = full-res, 2 = half-res) |

**Size:** 16 bytes

### GPUSATParams

Uniform data for the SAT recursive-doubling compute shader.

| Field       | Type       | Offset | Description                                                          |
| ----------- | ---------- | ------ | -------------------------------------------------------------------- |
| `Direction` | `[2]int32` | 0      | `(1,0)` for horizontal, `(0,1)` for vertical                         |
| `Offset`    | `int32`    | 8      | `2^k` step offset for recursive doubling; 0 = precision distribution |
| `_pad`      | `int32`    | 12     | Padding to 16 bytes                                                  |

**Size:** 16 bytes

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

### GPUIrradianceProbe

Per-probe data stored in the GPU storage buffer. Contains L2 spherical harmonics (9 coefficients per RGB channel).

| Field      | Type          | Offset | Description                                           |
| ---------- | ------------- | ------ | ----------------------------------------------------- |
| `Position` | `[4]float32`  | 0      | World XYZ + status in W (0=inactive, 1=active)        |
| `SH_R`     | `[12]float32` | 16     | L2 SH red coefficients (indices 0–8 used, 9–11 pad)   |
| `SH_G`     | `[12]float32` | 64     | L2 SH green coefficients (indices 0–8 used, 9–11 pad) |
| `SH_B`     | `[12]float32` | 112    | L2 SH blue coefficients (indices 0–8 used, 9–11 pad)  |

**Size:** 160 bytes

Additional helper: `MarshalProbeBuffer(probes []GPUIrradianceProbe) []byte` marshals a slice of probes into a tightly-packed buffer for GPU upload.

### GPUProbeGridParams

Uniform data describing the irradiance probe grid layout.

| Field         | Type         | Offset | Description                               |
| ------------- | ------------ | ------ | ----------------------------------------- |
| `GridMin`     | `[3]float32` | 0      | World-space minimum corner of the grid    |
| `ProbeCountX` | `uint32`     | 12     | Probes along X                            |
| `GridMax`     | `[3]float32` | 16     | World-space maximum corner of the grid    |
| `ProbeCountY` | `uint32`     | 28     | Probes along Y                            |
| `Spacing`     | `[3]float32` | 32     | Distance between adjacent probes per axis |
| `ProbeCountZ` | `uint32`     | 44     | Probes along Z                            |
| `TotalProbes` | `uint32`     | 48     | Total probes (X × Y × Z)                  |
| `_pad`        | `[7]uint32`  | 52     | Padding to 80 bytes (WGSL alignment)      |

**Size:** 80 bytes

### GPUProbeBakeCamera

Camera uniform for probe cubemap face baking.

| Field            | Type          | Offset | Description                         |
| ---------------- | ------------- | ------ | ----------------------------------- |
| `ViewProj`       | `[16]float32` | 0      | Cubemap face view-projection matrix |
| `CameraPosition` | `[3]float32`  | 64     | Probe world-space position          |
| `_pad`           | `float32`     | 76     | Padding to 80 bytes                 |

**Size:** 80 bytes

### GPUSHProjectParams

Uniform data for the SH projection compute shader.

| Field        | Type     | Offset | Description                        |
| ------------ | -------- | ------ | ---------------------------------- |
| `ProbeIndex` | `uint32` | 0      | Target probe index                 |
| `FaceIndex`  | `uint32` | 4      | Cubemap face being projected (0–5) |
| `Resolution` | `uint32` | 8      | Bake resolution (pixels per face)  |
| `_pad`       | `uint32` | 12     | Padding to 16 bytes                |

**Size:** 16 bytes

### GPUCompositionParams

Uniform data for the composition fragment shader.

| Field                | Type      | Offset | Description                             |
| -------------------- | --------- | ------ | --------------------------------------- |
| `ToneMappingEnabled` | `uint32`  | 0      | 1=ACES tone mapping applied, 0=bypassed |
| `Exposure`           | `float32` | 4      | Exposure multiplier before tone mapping |
| `_pad1`              | `uint32`  | 8      | Padding                                 |
| `_pad2`              | `uint32`  | 12     | Padding to 16 bytes                     |

**Size:** 16 bytes

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
| `MarshalProbeBuffer(probes []GPUIrradianceProbe) []byte`        | Marshals a probe array into a tightly-packed byte buffer for GPU storage buffer upload.                                              |

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

    // Create a lighting handler with VSM and PCSS enabled
    handler := light.NewLightingHandler(
        light.WithShadowHalfExtent(50.0),
        light.WithShadowNearFar(0.1, 300.0),
        light.WithShadowBias(0.002),
        light.WithShadowNormalBiasScale(3.0),
        light.WithShadowMapResolution(4096),
        light.WithAmbientColor([3]float32{0.05, 0.05, 0.08}),
        light.WithVSMBlurRadius(6),
        light.WithVSMMinVariance(0.0001),
        light.WithVSMLightBleedReduction(0.4),
        light.WithPCSSEnabled(true),
        light.WithVSMLightSize(2.0),
    )
    _ = handler // attach to scene via scene.WithLighting(handler)

    // Marshal for GPU upload
    lights := []light.Light{sun, torch}
    ambient := [3]float32{0.05, 0.05, 0.08}
    buf := light.MarshalLightBuffer(lights, ambient)
    _ = buf // upload to GPU storage buffer

    // Compute tile counts for Forward+ culling
    tileX, tileY := light.TileCounts(1280, 720)
    _, _ = tileX, tileY

    // Set up shadow data for the directional light (VSM mode)
    shadow := &light.GPUShadowData{
        TexelSize:           [2]float32{
            1.0 / float32(light.ShadowMapResolution),
            1.0 / float32(light.ShadowMapResolution),
        },
        Bias:                light.DefaultShadowBias,
        MinVariance:         light.DefaultVSMMinVariance,
        LightBleedReduction: light.DefaultVSMLightBleedReduction,
        LightSize:           light.DefaultVSMLightSize,
        ShadowHalfExtent:    light.DefaultShadowHalfExtent,
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
