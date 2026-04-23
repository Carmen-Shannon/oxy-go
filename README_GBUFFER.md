# Oxy G-Buffer System

The gbuffer package provides the scene G-Buffer handler used by deferred-style screen-space rendering passes. It owns the multiple render target (MRT) textures written during the G-Buffer pre-pass and exposes accessors used by downstream effects such as SSAO and SSR.

This subsystem is intentionally state-focused: the owning scene initializes and binds GPU resources, while the handler stores screen dimensions, pipeline key mappings, active frame slot selection, and texture/view references.

---

## Table of Contents

- [Overview](#overview)
- [Package Path](#package-path)
- [GBufferHandler](#gbufferhandler)
  - [Creating a GBufferHandler](#creating-a-gbufferhandler)
  - [GBufferHandler Builder Options](#gbufferhandler-builder-options)
  - [GBufferHandler Interface](#gbufferhandler-interface)
  - [G-Buffer Textures](#g-buffer-textures)
- [GPU Types](#gpu-types)
  - [GPUGBufferOutput](#gpugbufferoutput)
  - [GPUGBufferOutputSource](#gpugbufferoutputsource)
- [Files](#files)

---

## Overview

The G-Buffer subsystem stores per-pixel geometric and material data from the pre-pass into three attachments:

- Normal target (world-space normal + roughness)
- Albedo target (albedo + metallic)
- Depth target

The handler uses a two-slot texture/view layout for frames-in-flight. Call SetSlot(slot) before reading or writing slot-bound resources.

---

## Package Path

- Directory: [engine/renderer/gbuffer](engine/renderer/gbuffer)
- Import path: github.com/Carmen-Shannon/oxy-go/engine/renderer/gbuffer

---

## GBufferHandler

GBufferHandler is the exported interface for managing G-Buffer state and texture references.

### Creating a GBufferHandler

```go
import "github.com/Carmen-Shannon/oxy-go/engine/renderer/gbuffer"

handler := gbuffer.NewGBufferHandler(
	gbuffer.WithScreenSize(1920, 1080),
)
```

Defaults applied before options:

| Parameter        | Default   |
| ---------------- | --------- |
| Enabled          | false     |
| Screen size      | 0, 0      |
| Pipeline key map | empty map |
| Active slot      | 0         |

### GBufferHandler Builder Options

All options follow the GBufferHandlerOption functional option pattern.

| Option         | Parameters        | Description                                              |
| -------------- | ----------------- | -------------------------------------------------------- |
| WithScreenSize | width, height int | Sets initial dimensions used for G-Buffer texture sizing |

### GBufferHandler Interface

| Method                                                                             | Description                                                    |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Enabled() bool / SetEnabled(enabled bool)                                          | Gets or sets whether the G-Buffer subsystem is GPU-initialized |
| SetSlot(slot int)                                                                  | Selects the active frames-in-flight texture slot               |
| ScreenWidth() int / ScreenHeight() int                                             | Returns the current stored dimensions used for texture sizing  |
| PipelineKey(name string) string                                                    | Gets a pipeline key by logical name                            |
| PipelineKeys() map[string]string                                                   | Returns the full pipeline key map                              |
| SetPipelineKey(name, key string)                                                   | Stores a pipeline key by logical name                          |
| NormalTexture() *wgpu.Texture / SetNormalTexture(t *wgpu.Texture)                  | Gets or sets the current-slot normal MRT texture               |
| NormalTextureView() *wgpu.TextureView / SetNormalTextureView(tv *wgpu.TextureView) | Gets or sets the current-slot normal texture view              |
| AlbedoTexture() *wgpu.Texture / SetAlbedoTexture(t *wgpu.Texture)                  | Gets or sets the current-slot albedo MRT texture               |
| AlbedoTextureView() *wgpu.TextureView / SetAlbedoTextureView(tv *wgpu.TextureView) | Gets or sets the current-slot albedo texture view              |
| DepthTexture() *wgpu.Texture / SetDepthTexture(t *wgpu.Texture)                    | Gets or sets the current-slot depth texture                    |
| DepthTextureView() *wgpu.TextureView / SetDepthTextureView(tv *wgpu.TextureView)   | Gets or sets the current-slot depth texture view               |
| Resize(width, height int)                                                          | Updates the stored dimensions for texture sizing               |

### G-Buffer Textures

| Target | Format      | Contents                               |
| ------ | ----------- | -------------------------------------- |
| Normal | RGBA16Float | World-space normal (xyz) and roughness |
| Albedo | RGBA8Unorm  | Albedo (rgb) and metallic              |
| Depth  | Depth24Plus | G-Buffer pre-pass depth                |

Each target is stored as a two-slot array in the implementation (index selected via SetSlot).

---

## GPU Types

### GPUGBufferOutput

GPUGBufferOutput is the GPU-aligned representation of one G-Buffer fragment output.

- Layout: 3 x vec4<f32>
- Size: 48 bytes
- Fields:
  - Position [4]float32
  - Normal [4]float32
  - Albedo [4]float32

Methods:

- Size() int: Returns the struct size in bytes.
- Marshal() []byte: Serializes Position, Normal, and Albedo into little-endian byte layout matching WGSL.

### GPUGBufferOutputSource

GPUGBufferOutputSource is the embedded canonical WGSL source for the GBufferOutput struct definition:

- Embedded from [engine/renderer/gbuffer/assets/gbuffer-output.wgsl](engine/renderer/gbuffer/assets/gbuffer-output.wgsl)
- Intended to match GPUGBufferOutput exactly (48 bytes, std430 aligned)

---

## Files

- [engine/renderer/gbuffer/gbuffer_handler.go](engine/renderer/gbuffer/gbuffer_handler.go): Exported interface and method implementations
- [engine/renderer/gbuffer/gbuffer_handler_builder.go](engine/renderer/gbuffer/gbuffer_handler_builder.go): Functional options and constructor
- [engine/renderer/gbuffer/gbuffer_handler_impl.go](engine/renderer/gbuffer/gbuffer_handler_impl.go): Internal handler state layout
- [engine/renderer/gbuffer/gpu_types.go](engine/renderer/gbuffer/gpu_types.go): GPU type and embedded WGSL source definitions
- [engine/renderer/gbuffer/assets/gbuffer-output.wgsl](engine/renderer/gbuffer/assets/gbuffer-output.wgsl): WGSL struct source embedded by GPUGBufferOutputSource
