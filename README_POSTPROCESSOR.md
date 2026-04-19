# Oxy Post-Processing System

The `postprocessing` package provides the screen-space post-processing pipeline handlers for the Oxy engine. It contains four sub-handlers extracted from the former `engine/light` package:

- **SSAOHandler** — Screen-Space Ambient Occlusion with hemisphere sampling and bilateral blur
- **CompositionHandler** — HDR composition with ACES tone mapping, auto-exposure, and bloom
- **SSRHandler** — Screen-Space Reflections via Hi-Z ray marching
- **TAAHandler** — Temporal Anti-Aliasing with Halton jitter, YCoCg neighborhood clamping, and CAS sharpening

Each handler manages its own textures, pipeline keys, and bind group providers. GPU resources are initialized lazily by the scene.

> Lighting, shadows, and contact shadows are documented in [README_LIGHT.md](README_LIGHT.md). The G-Buffer handler is documented in [README_RENDERER.md](README_RENDERER.md).

---

## Table of Contents

- [Overview](#overview)
- [SSAOHandler](#ssaohandler)
  - [Creating an SSAOHandler](#creating-an-ssaohandler)
  - [SSAOHandler Builder Options](#ssaohandler-builder-options)
  - [SSAOHandler Interface](#ssaohandler-interface)
  - [SSAO Textures](#ssao-textures)
  - [SSAO Default BGPs](#ssao-default-bgps)
- [CompositionHandler](#compositionhandler)
  - [Creating a CompositionHandler](#creating-a-compositionhandler)
  - [CompositionHandler Builder Options](#compositionhandler-builder-options)
  - [CompositionHandler Interface](#compositionhandler-interface)
  - [Composition Textures](#composition-textures)
  - [Composition Default BGPs](#composition-default-bgps)
  - [Bloom](#bloom)
- [SSRHandler](#ssrhandler)
  - [Creating an SSRHandler](#creating-an-ssrhandler)
  - [SSRHandler Builder Options](#ssrhandler-builder-options)
  - [SSRHandler Interface](#ssrhandler-interface)
  - [SSR Textures](#ssr-textures)
  - [Hi-Z Pyramid](#hi-z-pyramid)
  - [SSR Default BGPs](#ssr-default-bgps)
- [TAAHandler](#taahandler)
  - [Creating a TAAHandler](#creating-a-taahandler)
  - [TAAHandler Builder Options](#taahandler-builder-options)
  - [TAAHandler Interface](#taahandler-interface)
  - [TAA Textures](#taa-textures)
  - [TAA Default BGPs](#taa-default-bgps)
- [GPU Types](#gpu-types)
  - [GPUSSAOParams](#gpussaoparams)
  - [GPUBlurParams](#gpublurparams)
  - [GPUCompositionParams](#gpucompositionparams)
  - [GPULuminanceParams](#gpuluminanceparams)
  - [GPUSSRParams](#gpussrparams)
  - [GPUBloomParams](#gpubloomparams)
  - [GPUTAAParams](#gputaaparams)
- [Embedded WGSL Sources](#embedded-wgsl-sources)
- [Files](#files)

---

## Overview

The post-processing system is organized as four independent handler interfaces, each following the same pattern:

1. **Handler** — An exported interface with getters, setters, texture accessors, pipeline key storage, and bind group provider maps.
2. **Builder** — A functional-option constructor (`New*Handler(opts ...)`) that applies defaults then user overrides.
3. **GPU Types** — `Marshal()`-able structs matching WGSL uniform layouts, uploaded to GPU each frame.

All four handlers share a common lifecycle:

- Created via builder options and attached to the scene's lighting/rendering setup.
- GPU resources (textures, samplers, bind groups) are initialized lazily by the scene on first use.
- On window resize, the scene calls `Resize(width, height)` to update stored dimensions, then releases and recreates GPU textures at the new resolution.
- Double-buffered bind group providers use `SetSlot(slot)` to select the active frames-in-flight slot.

---

## SSAOHandler

The `SSAOHandler` manages Screen-Space Ambient Occlusion — a hemisphere-sampling compute pass that detects local occlusion from the G-Buffer depth and normal textures, followed by a separable bilateral blur pass.

### Creating an SSAOHandler

```go
ssao := postprocessing.NewSSAOHandler(
    postprocessing.WithSSAOScreenSize(1280, 720),
    postprocessing.WithSSAOSampleCount(16),
    postprocessing.WithSSAOScreenRadius(24.0),
    postprocessing.WithSSAOBias(0.025),
    postprocessing.WithSSAOPower(2.0),
    postprocessing.WithSSAOBlurRadius(4),
    postprocessing.WithSSAOHalfResolution(false),
)
```

Defaults applied before options:

| Parameter      | Default |
| -------------- | ------- |
| Screen size    | `0, 0`  |
| SampleCount    | `16`    |
| MaxSamples     | `32`    |
| ScreenRadius   | `24.0`  |
| Bias           | `0.025` |
| Power          | `2.0`   |
| BlurRadius     | `4`     |
| HalfResolution | `false` |

### SSAOHandler Builder Options

All options follow the `SSAOHandlerOption` functional option pattern.

| Option                   | Parameters          | Default | Description                                                                                 |
| ------------------------ | ------------------- | ------- | ------------------------------------------------------------------------------------------- |
| `WithSSAOScreenSize`     | `width, height int` | `0, 0`  | Initial screen dimensions                                                                   |
| `WithSSAOSampleCount`    | `count int`         | `16`    | Hemisphere samples per pixel (1–MaxSamples)                                                 |
| `WithSSAOMaxSamples`     | `max int`           | `32`    | GPU compile-time upper bound for sample array size                                          |
| `WithSSAOScreenRadius`   | `pixels float32`    | `24.0`  | Screen-space sampling radius in pixels — engine auto-computes world-space radius each frame |
| `WithSSAOBias`           | `bias float32`      | `0.025` | Depth bias to prevent self-occlusion                                                        |
| `WithSSAOPower`          | `power float32`     | `2.0`   | Exponent for AO contrast                                                                    |
| `WithSSAOBlurRadius`     | `radius int`        | `4`     | Bilateral blur half-width in texels                                                         |
| `WithSSAOHalfResolution` | `enabled bool`      | `false` | Allocate textures at half resolution (¼ pixel count)                                        |

### SSAOHandler Interface

| Method                                              | Description                            |
| --------------------------------------------------- | -------------------------------------- |
| `Enabled() bool` / `SetEnabled(bool)`               | Whether GPU resources are initialized  |
| `SetSlot(slot int)`                                 | Sets the active double-buffer slot     |
| `ScreenWidth() int` / `ScreenHeight() int`          | Current screen dimensions              |
| `SampleCount() int`                                 | Hemisphere samples per pixel           |
| `MaxSamples() int`                                  | Compile-time max samples value         |
| `ScreenRadius() float32`                            | Screen-space sampling radius in pixels |
| `Bias() float32`                                    | Depth comparison bias                  |
| `Power() float32`                                   | AO contrast exponent                   |
| `BlurRadius() int`                                  | Bilateral blur half-width              |
| `HalfResolution() bool` / `SetHalfResolution(bool)` | Whether SSAO runs at half resolution   |
| `RawTexture()` / `SetRawTexture`                    | Pre-blur SSAO output texture           |
| `BlurredTexture()` / `SetBlurredTexture`            | Final AO texture bound to lit shader   |
| `ScratchTexture()` / `SetScratchTexture`            | Intermediate blur texture              |
| `LinearSampler()` / `SetLinearSampler`              | Linear sampler for texture lookups     |
| `PipelineKey(name)` / `SetPipelineKey(name, key)`   | Pipeline key storage                   |
| `Bgp(key)` / `Bgps()`                               | Bind group provider access             |
| `Resize(width, height)`                             | Updates stored screen dimensions       |

### SSAO Textures

| Texture | Format  | Description                      |
| ------- | ------- | -------------------------------- |
| Raw     | R8Unorm | Pre-blur SSAO output             |
| Blurred | R8Unorm | Final AO bound to the lit shader |
| Scratch | R8Unorm | Intermediate between H/V blur    |

### SSAO Default BGPs

`"ssao_compute"`, `"ssao_blur_h"`, `"ssao_blur_v"`

---

## CompositionHandler

The `CompositionHandler` manages the HDR composition pass — ACES tone mapping, manual or auto-exposure, and bloom. It owns the offscreen HDR render target (`RGBA16Float`) that the lit shader writes into, plus the MSAA and depth attachments for that render pass.

### Creating a CompositionHandler

```go
comp := postprocessing.NewCompositionHandler(
    postprocessing.WithCompositionScreenSize(1280, 720),
    postprocessing.WithToneMappingEnabled(true),
    postprocessing.WithExposure(1.0),
    postprocessing.WithAutoExposure(true),
    postprocessing.WithAdaptSpeed(1.0),
    postprocessing.WithMinExposure(0.1),
    postprocessing.WithMaxExposure(10.0),
    postprocessing.WithBloomEnabled(true),
    postprocessing.WithBloomThreshold(1.0),
    postprocessing.WithBloomIntensity(0.5),
)
```

Defaults applied before options:

| Parameter              | Default |
| ---------------------- | ------- |
| Screen size            | `0, 0`  |
| ToneMappingEnabled     | `true`  |
| Exposure               | `1.0`   |
| AutoExposure           | `false` |
| AdaptSpeed             | `1.0`   |
| MinExposure            | `0.1`   |
| MaxExposure            | `10.0`  |
| LuminanceWorkgroupSize | `16`    |
| BloomEnabled           | `false` |
| BloomThreshold         | `1.0`   |
| BloomIntensity         | `0.5`   |

### CompositionHandler Builder Options

All options follow the `CompositionHandlerOption` functional option pattern.

| Option                       | Parameters          | Default | Description                                           |
| ---------------------------- | ------------------- | ------- | ----------------------------------------------------- |
| `WithCompositionScreenSize`  | `width, height int` | `0, 0`  | Initial screen dimensions                             |
| `WithToneMappingEnabled`     | `enabled bool`      | `true`  | Enables ACES tone mapping                             |
| `WithExposure`               | `exposure float32`  | `1.0`   | HDR exposure multiplier (1.0=neutral)                 |
| `WithAutoExposure`           | `enabled bool`      | `false` | Enable GPU-driven luminance-based exposure adaptation |
| `WithAdaptSpeed`             | `speed float32`     | `1.0`   | Rate at which exposure converges (seconds⁻¹)          |
| `WithMinExposure`            | `min float32`       | `0.1`   | Minimum clamp for adapted exposure                    |
| `WithMaxExposure`            | `max float32`       | `10.0`  | Maximum clamp for adapted exposure                    |
| `WithLuminanceWorkgroupSize` | `size int`          | `16`    | Workgroup tile dimension for luminance compute        |
| `WithBloomEnabled`           | `enabled bool`      | `false` | Enables bloom post-processing                         |
| `WithBloomThreshold`         | `threshold float32` | `1.0`   | Brightness threshold for bloom extraction (soft-knee) |
| `WithBloomIntensity`         | `intensity float32` | `0.5`   | Multiplier for bloom contribution                     |

### CompositionHandler Interface

| Method                                                        | Description                                  |
| ------------------------------------------------------------- | -------------------------------------------- |
| `Enabled() bool` / `SetEnabled(bool)`                         | Whether GPU resources are initialized        |
| `SetSlot(slot int)`                                           | Sets the active double-buffer slot           |
| `ScreenWidth() int` / `ScreenHeight() int`                    | Current screen dimensions                    |
| `ToneMappingEnabled() bool`                                   | Whether ACES tone mapping is active          |
| `Exposure() float32` / `SetExposure(float32)`                 | HDR exposure multiplier                      |
| `AutoExposureEnabled() bool` / `SetAutoExposureEnabled(bool)` | GPU-driven auto-exposure toggle              |
| `AdaptSpeed() float32` / `SetAdaptSpeed(float32)`             | Exposure adaptation rate                     |
| `MinExposure() float32` / `SetMinExposure(float32)`           | Min exposure clamp                           |
| `MaxExposure() float32` / `SetMaxExposure(float32)`           | Max exposure clamp                           |
| `LuminanceWorkgroupSize() int`                                | Workgroup dimension for luminance compute    |
| `ExposureBuffer()` / `SetExposureBuffer`                      | GPU storage buffer for adapted exposure      |
| `BloomEnabled() bool` / `SetBloomEnabled(bool)`               | Bloom toggle                                 |
| `BloomThreshold() float32` / `SetBloomThreshold(float32)`     | Bloom extraction threshold                   |
| `BloomIntensity() float32` / `SetBloomIntensity(float32)`     | Bloom contribution multiplier                |
| `BloomMipCount() int` / `SetBloomMipCount(int)`               | Number of bloom mip levels                   |
| `BloomDownTexture()` / `BloomDownViews()`                     | Bloom downsample mip chain texture and views |
| `BloomUpTexture()` / `BloomUpViews()`                         | Bloom upsample mip chain texture and views   |
| `HDRTexture()` / `SetHDRTexture`                              | Offscreen HDR render target                  |
| `MSAATexture()` / `SetMSAATexture`                            | MSAA color attachment                        |
| `DepthTexture()` / `SetDepthTexture`                          | Depth buffer for HDR pass                    |
| `LinearSampler()` / `SetLinearSampler`                        | Linear sampler                               |
| `PipelineKey(name)` / `SetPipelineKey(name, key)`             | Pipeline key storage                         |
| `Bgp(key)` / `Bgps()`                                         | Bind group provider access                   |
| `Resize(width, height)`                                       | Updates stored screen dimensions             |

### Composition Textures

| Texture | Format      | Description                          |
| ------- | ----------- | ------------------------------------ |
| HDR     | RGBA16Float | Offscreen render target for lit pass |
| MSAA    | RGBA16Float | Multi-sampled color attachment       |
| Depth   | Depth24Plus | Depth buffer for HDR render pass     |

### Composition Default BGPs

`"composition"`

### Bloom

The bloom system extracts bright pixels from the HDR buffer using a soft-knee brightness threshold, then progressively downsamples (13-tap box-tent filter, CoD:AW style) and upsamples (9-tap tent filter with additive blending) through a mip chain to produce a smooth glow. The result is added to the final image in the composition shader, after exposure but before tone mapping.

- Two `RGBA16Float` mip chain textures (down chain + up chain) at half screen resolution, max 6 mip levels
- Per-mip BGPs with separate `DispatchComputeBatch` per mip for GPU barriers
- Threshold only applied on first downsample pass; subsequent passes downsample without threshold
- When disabled, a 1×1 black fallback texture is bound to composition binding 6

---

## SSRHandler

The `SSRHandler` manages Screen-Space Reflections via hierarchical ray marching against a Hi-Z depth pyramid. It produces a half-resolution result texture bound to the composition shader.

### Creating an SSRHandler

```go
ssr := postprocessing.NewSSRHandler(
    postprocessing.WithSSRScreenSize(1280, 720),
    postprocessing.WithSSRMaxSteps(64),
    postprocessing.WithSSRMaxDistance(50.0),
    postprocessing.WithSSRThickness(0.1),
    postprocessing.WithSSRStride(1.0),
    postprocessing.WithSSRRoughnessCutoff(0.5),
)
```

Defaults applied before options:

| Parameter       | Default |
| --------------- | ------- |
| Screen size     | `0, 0`  |
| MaxSteps        | `64`    |
| MaxDistance     | `50.0`  |
| Thickness       | `0.1`   |
| Stride          | `1.0`   |
| RoughnessCutoff | `0.5`   |

### SSRHandler Builder Options

All options follow the `SSRHandlerOption` functional option pattern.

| Option                   | Parameters          | Default | Description                              |
| ------------------------ | ------------------- | ------- | ---------------------------------------- |
| `WithSSRScreenSize`      | `width, height int` | `0, 0`  | Initial screen dimensions                |
| `WithSSRMaxSteps`        | `steps int`         | `64`    | Maximum ray march steps per pixel        |
| `WithSSRMaxDistance`     | `distance float32`  | `50.0`  | Maximum ray march distance in view space |
| `WithSSRThickness`       | `thickness float32` | `0.1`   | Depth tolerance for hit detection        |
| `WithSSRStride`          | `stride float32`    | `1.0`   | Step stride multiplier                   |
| `WithSSRRoughnessCutoff` | `cutoff float32`    | `0.5`   | Roughness above which SSR is skipped     |

### SSRHandler Interface

| Method                                                              | Description                                    |
| ------------------------------------------------------------------- | ---------------------------------------------- |
| `Enabled() bool` / `SetEnabled(bool)`                               | Whether GPU resources are initialized          |
| `SetSlot(slot int)`                                                 | Sets the active double-buffer slot             |
| `ScreenWidth() int` / `ScreenHeight() int`                          | Current screen dimensions                      |
| `MaxSteps() int`                                                    | Maximum ray march steps                        |
| `MaxDistance() float32`                                             | Maximum march distance                         |
| `Thickness() float32`                                               | Depth tolerance for hit detection              |
| `Stride() float32`                                                  | Step stride multiplier                         |
| `RoughnessCutoff() float32`                                         | Roughness above which SSR is skipped           |
| `SSRTexture()` / `SetSSRTexture`                                    | SSR result texture                             |
| `HiZTexture()` / `SetHiZTexture`                                    | Hi-Z depth pyramid texture                     |
| `HiZMipCount()` / `SetHiZMipCount`                                  | Number of Hi-Z mip levels                      |
| `HiZMipReadViews()` / `HiZStorageViews()`                           | Per-mip read/storage views for Hi-Z downsample |
| `HiZMaxTexture()` / `HiZMaxMipReadViews()` / `HiZMaxStorageViews()` | MAX Hi-Z pyramid for occlusion culling         |
| `LinearSampler()` / `SetLinearSampler`                              | Linear sampler                                 |
| `PipelineKey(name)` / `SetPipelineKey(name, key)`                   | Pipeline key storage                           |
| `Bgp(key)` / `Bgps()`                                               | Bind group provider access                     |
| `Resize(width, height)`                                             | Updates stored screen dimensions               |

### SSR Textures

| Texture    | Format      | Description                                      |
| ---------- | ----------- | ------------------------------------------------ |
| SSR        | RGBA16Float | Half-resolution result (RGB=color, A=confidence) |
| Hi-Z (MIN) | R32Float    | Min-depth pyramid for SSR ray marching           |
| Hi-Z (MAX) | R32Float    | Max-depth pyramid for occlusion culling          |

### Hi-Z Pyramid

The SSR system builds two hierarchical depth pyramids each frame from the G-Buffer depth texture:

- **MIN pyramid** — Each mip stores the minimum depth of the 4 parent texels. Used by the SSR compute shader for conservative ray-depth intersection during hierarchical ray marching.
- **MAX pyramid** — Each mip stores the maximum depth of the 4 parent texels. Used by the Hi-Z occlusion culling compute pass to conservatively reject objects whose nearest clip-space Z is farther than the maximum stored depth in the pyramid footprint.

Both pyramids use `R32Float` format with per-mip storage views for the downsample compute shader and per-mip read views for the consumer shaders.

### SSR Default BGPs

`"ssr_compute"`

---

## TAAHandler

The `TAAHandler` manages Temporal Anti-Aliasing. Each frame, the camera projection is sub-pixel jittered using a Halton(2,3) sequence. A compute shader reprojects the previous frame's resolved output into the current frame's coordinate space, clamps the historical sample using YCoCg neighborhood min/max, and blends with the current frame. A final CAS (Contrast Adaptive Sharpening) pass counteracts the slight softening introduced by temporal accumulation.

### Creating a TAAHandler

```go
taa := postprocessing.NewTAAHandler(
    postprocessing.WithTAAScreenSize(1280, 720),
    postprocessing.WithTAABlendFactor(0.1),
    postprocessing.WithTAAHistoryRectificationScale(1.0),
    postprocessing.WithTAAJitterScale(1.0),
)
```

Defaults applied before options:

| Parameter                 | Default |
| ------------------------- | ------- |
| Screen size               | `0, 0`  |
| BlendFactor               | `0.1`   |
| HistoryRectificationScale | `1.0`   |
| JitterScale               | `1.0`   |

### TAAHandler Builder Options

All options follow the `TAAHandlerOption` functional option pattern.

| Option                             | Parameters          | Default | Description                                                                                              |
| ---------------------------------- | ------------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `WithTAAScreenSize`                | `width, height int` | `0, 0`  | Initial screen dimensions                                                                                |
| `WithTAABlendFactor`               | `f float32`         | `0.1`   | Weight given to current frame during temporal blending. Lower = more smoothing/ghosting. Range: 0.05–0.2 |
| `WithTAAHistoryRectificationScale` | `scale float32`     | `1.0`   | Expansion scale for YCoCg history clamp box                                                              |
| `WithTAAJitterScale`               | `scale float32`     | `1.0`   | Multiplier applied to Halton jitter offsets. Higher = more sub-pixel jitter amplitude                    |

### TAAHandler Interface

| Method                                                                          | Description                                                                  |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `Enabled() bool` / `SetEnabled(bool)`                                           | Whether TAA is active                                                        |
| `SetSlot(slot int)`                                                             | Sets the active double-buffer slot                                           |
| `ScreenWidth() int` / `ScreenHeight() int`                                      | Current screen dimensions                                                    |
| `BlendFactor() float32` / `SetBlendFactor(float32)`                             | Temporal blend weight                                                        |
| `HistoryRectificationScale() float32` / `SetHistoryRectificationScale(float32)` | YCoCg clamp box scale                                                        |
| `RawHistoryOnly() bool` / `SetRawHistoryOnly(bool)`                             | Diagnostic: bypass neighborhood clamping                                     |
| `JitterScale() float32` / `SetJitterScale(float32)`                             | Halton jitter multiplier                                                     |
| `JitterX() float32` / `JitterY() float32`                                       | Current frame jitter offsets                                                 |
| `PrevJitterX() float32` / `PrevJitterY() float32`                               | Previous frame jitter offsets                                                |
| `FrameIndex() uint64`                                                           | Current TAA frame counter                                                    |
| `AdvanceFrame(jitterX, jitterY float32)`                                        | Promotes current jitter to previous, sets new jitter, increments frame index |
| `TAATexture()` / `SetTAATexture`                                                | Ping-pong resolved output texture                                            |
| `SharpenTexture()` / `SetSharpenTexture`                                        | CAS sharpened output texture                                                 |
| `LinearSampler()` / `SetLinearSampler`                                          | Linear sampler                                                               |
| `PipelineKey(name)` / `SetPipelineKey(name, key)`                               | Pipeline key storage                                                         |
| `Bgp(key)` / `Bgps()`                                                           | Bind group provider access                                                   |
| `Resize(width, height)`                                                         | Updates stored screen dimensions                                             |

### TAA Textures

| Texture         | Format      | Description                                          |
| --------------- | ----------- | ---------------------------------------------------- |
| TAA (ping-pong) | RGBA16Float | Resolved TAA output (2 textures, alternated by slot) |
| Sharpen         | RGBA16Float | CAS sharpened output                                 |

### TAA Default BGPs

`"taa_resolve_0"`, `"taa_resolve_1"`, `"taa_sharpen_0"`, `"taa_sharpen_1"`

---

## GPU Types

All GPU types have a corresponding embedded WGSL source (`*Source` variable) that can be injected into shaders via `@oxy:include` annotations (see [Shader Annotation System](README_ANNOTATIONS.md)).

Each GPU struct provides:

- `Size() int` — struct size in bytes
- `Marshal() []byte` — serializes to a byte buffer for GPU upload

### GPUSSAOParams

Uniform data for the SSAO compute shader. 176 bytes.

| Field            | Type          | Offset | Description                                   |
| ---------------- | ------------- | ------ | --------------------------------------------- |
| `Projection`     | `[16]float32` | 0      | View-projection matrix (column-major)         |
| `InvViewProj`    | `[16]float32` | 64     | Inverse view-projection matrix (column-major) |
| `Radius`         | `float32`     | 128    | Hemisphere radius in world units              |
| `Bias`           | `float32`     | 132    | Depth comparison bias                         |
| `Power`          | `float32`     | 136    | AO contrast exponent                          |
| `SampleCount`    | `uint32`      | 140    | Number of hemisphere samples                  |
| `ScreenWidth`    | `float32`     | 144    | Screen width in pixels                        |
| `ScreenHeight`   | `float32`     | 148    | Screen height in pixels                       |
| `GBufferScale`   | `float32`     | 152    | Coordinate multiplier for half-res            |
| `_pad`           | `float32`     | 156    | Padding                                       |
| `CameraPosition` | `[3]float32`  | 160    | World-space camera position                   |
| `_pad2`          | `float32`     | 172    | Padding                                       |

**Size:** 176 bytes

### GPUBlurParams

Uniform data for the separable bilateral blur compute shader. 24 bytes.

| Field          | Type       | Offset | Description                                   |
| -------------- | ---------- | ------ | --------------------------------------------- |
| `Direction`    | `[2]int32` | 0      | `(1,0)` horizontal, `(0,1)` vertical          |
| `Radius`       | `int32`    | 8      | Blur kernel half-width in texels              |
| `GBufferScale` | `int32`    | 12     | Lookup scale for half-res SSAO                |
| `CascadeWidth` | `int32`    | 16     | Per-cascade column width; 0 disables clamping |
| `_pad`         | `int32`    | 20     | Padding                                       |

**Size:** 24 bytes

### GPUCompositionParams

Uniform data for the composition fragment shader. 32 bytes.

| Field                 | Type      | Offset | Description                        |
| --------------------- | --------- | ------ | ---------------------------------- |
| `ToneMappingEnabled`  | `uint32`  | 0      | 1=ACES applied, 0=bypassed         |
| `Exposure`            | `float32` | 4      | Exposure multiplier                |
| `AutoExposureEnabled` | `uint32`  | 8      | Non-zero when auto-exposure active |
| `BloomEnabled`        | `uint32`  | 12     | Non-zero when bloom is active      |
| `BloomIntensity`      | `float32` | 16     | Bloom contribution multiplier      |
| `_pad5`               | `uint32`  | 20     | Padding                            |
| `_pad6`               | `uint32`  | 24     | Padding                            |
| `_pad7`               | `uint32`  | 28     | Padding                            |

**Size:** 32 bytes

### GPULuminanceParams

Uniform data for the luminance compute shader. 32 bytes.

| Field                 | Type      | Offset | Description                        |
| --------------------- | --------- | ------ | ---------------------------------- |
| `ScreenWidth`         | `uint32`  | 0      | HDR texture width                  |
| `ScreenHeight`        | `uint32`  | 4      | HDR texture height                 |
| `AdaptSpeed`          | `float32` | 8      | Exposure adaptation speed          |
| `DeltaTime`           | `float32` | 12     | Frame delta time                   |
| `MinExposure`         | `float32` | 16     | Minimum clamped exposure           |
| `MaxExposure`         | `float32` | 20     | Maximum clamped exposure           |
| `KeyValue`            | `float32` | 24     | Middle-gray key value (0.18)       |
| `AutoExposureEnabled` | `uint32`  | 28     | Non-zero when auto-exposure active |

**Size:** 32 bytes

### GPUSSRParams

Uniform data for the SSR compute shader. 224 bytes.

| Field             | Type          | Offset | Description                                |
| ----------------- | ------------- | ------ | ------------------------------------------ |
| `Projection`      | `[16]float32` | 0      | Projection matrix (column-major)           |
| `InvProjection`   | `[16]float32` | 64     | Inverse projection matrix (column-major)   |
| `View`            | `[16]float32` | 128    | View matrix (column-major)                 |
| `MaxDistance`     | `float32`     | 192    | Max ray march distance in view space       |
| `Thickness`       | `float32`     | 196    | Depth tolerance for hit detection          |
| `Stride`          | `float32`     | 200    | Step stride multiplier                     |
| `MaxSteps`        | `uint32`      | 204    | Max ray march steps                        |
| `ScreenWidth`     | `float32`     | 208    | Screen width in pixels                     |
| `ScreenHeight`    | `float32`     | 212    | Screen height in pixels                    |
| `RoughnessCutoff` | `float32`     | 216    | Roughness above which SSR is skipped       |
| `HiZMipCount`     | `uint32`      | 220    | Number of Hi-Z mip levels in depth pyramid |

**Size:** 224 bytes

### GPUBloomParams

Uniform data for the bloom downsample compute shader. 16 bytes.

| Field       | Type      | Offset | Description                                                     |
| ----------- | --------- | ------ | --------------------------------------------------------------- |
| `Threshold` | `float32` | 0      | Brightness threshold for bloom extraction; 0 disables filtering |
| `_pad0`     | `uint32`  | 4      | Padding                                                         |
| `_pad1`     | `uint32`  | 8      | Padding                                                         |
| `_pad2`     | `uint32`  | 12     | Padding                                                         |

**Size:** 16 bytes

### GPUTAAParams

Uniform data for the TAA resolve compute shader. 176 bytes.

| Field                       | Type          | Offset | Description                                        |
| --------------------------- | ------------- | ------ | -------------------------------------------------- |
| `InvCurrViewProj`           | `[16]float32` | 0      | Inverse of jittered current view-projection matrix |
| `PrevViewProj`              | `[16]float32` | 64     | Previous frame's jittered view-projection matrix   |
| `JitterCurr`                | `[2]float32`  | 128    | Current NDC jitter (x, y)                          |
| `JitterPrev`                | `[2]float32`  | 136    | Previous NDC jitter (x, y)                         |
| `ScreenWidth`               | `float32`     | 144    | Screen width in pixels                             |
| `ScreenHeight`              | `float32`     | 148    | Screen height in pixels                            |
| `BlendFactor`               | `float32`     | 152    | Weight for the current frame (e.g. 0.1)            |
| `HistoryRectificationScale` | `float32`     | 156    | YCoCg clamp expansion scale (1.0 = baseline)       |
| `RawHistoryOnly`            | `float32`     | 160    | 1.0 = output raw reprojected history (diagnostic)  |
| `_pad`                      | `[3]float32`  | 164    | Padding to 16-byte struct alignment                |

**Size:** 176 bytes

---

## Embedded WGSL Sources

All GPU types have a corresponding `//go:embed` variable for shader injection via `@oxy:include`:

| Variable                     | Asset File                       |
| ---------------------------- | -------------------------------- |
| `GPUBlurParamsSource`        | `assets/blur-params.wgsl`        |
| `GPUSSAOParamsSource`        | `assets/ssao-params.wgsl`        |
| `GPUCompositionParamsSource` | `assets/composition-params.wgsl` |
| `GPULuminanceParamsSource`   | `assets/luminance-params.wgsl`   |
| `GPUSSRParamsSource`         | `assets/ssr-params.wgsl`         |
| `GPUBloomParamsSource`       | `assets/bloom-params.wgsl`       |
| `GPUTAAParamsSource`         | `assets/taa-params.wgsl`         |

---

## Files

| File                             | Contents                                                                                |
| -------------------------------- | --------------------------------------------------------------------------------------- |
| `ssao_handler.go`                | `SSAOHandler` interface                                                                 |
| `ssao_handler_impl.go`           | Unexported `ssaoHandlerImpl` struct                                                     |
| `ssao_handler_builder.go`        | `SSAOHandlerOption` type, `With*` functions, `NewSSAOHandler` constructor               |
| `composition_handler.go`         | `CompositionHandler` interface                                                          |
| `composition_handler_impl.go`    | Unexported `compositionHandlerImpl` struct                                              |
| `composition_handler_builder.go` | `CompositionHandlerOption` type, `With*` functions, `NewCompositionHandler` constructor |
| `ssr_handler.go`                 | `SSRHandler` interface                                                                  |
| `ssr_handler_impl.go`            | Unexported `ssrHandlerImpl` struct                                                      |
| `ssr_handler_builder.go`         | `SSRHandlerOption` type, `With*` functions, `NewSSRHandler` constructor                 |
| `taa_handler.go`                 | `TAAHandler` interface                                                                  |
| `taa_handler_impl.go`            | Unexported `taaHandlerImpl` struct                                                      |
| `taa_handler_builder.go`         | `TAAHandlerOption` type, `With*` functions, `NewTAAHandler` constructor                 |
| `gpu_types.go`                   | All GPU-marshaled structs and embedded WGSL sources                                     |
