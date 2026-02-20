# Oxy Engine

A 3D game engine written in pure Go, powered by WebGPU via [cogentcore/webgpu](https://github.com/cogentcore/webgpu).

[![Go Reference](https://pkg.go.dev/badge/github.com/Carmen-Shannon/oxy-go.svg)](https://pkg.go.dev/github.com/Carmen-Shannon/oxy-go)

```
go get github.com/Carmen-Shannon/oxy-go
```

> **Go 1.25.6+** required.

---

## Overview

Oxy is a forward-rendering 3D engine built from scratch in Go. It uses WebGPU for GPU access and GLFW for windowing, with no CGo wrappers beyond those two dependencies. The engine is designed around a modular, interface-first architecture using the option-builder pattern throughout.

### Key Features

- **Forward+ Rendering** — Tiled light culling compute pass followed by a lit forward render pass.
- **Skeletal Animation** — GPU-driven skeletal animation via compute shaders with bone blending, channel interpolation, and indirect draw.
- **Shadow Mapping** — Depth-only shadow passes with PCF sampling and configurable shadow uniforms.
- **glTF Loader** — Full glTF 2.0 import pipeline: meshes, materials, skeletons, and animations.
- **WGSL Shader Annotations** — A custom pre-processor that embeds resource metadata directly in WGSL source files, enabling declarative GPU resource wiring with zero string-based lookups at runtime. See the [Annotation System Documentation](README_ANNOTATIONS.md).
- **Scene Graph** — Scenes manage cameras, lights, game objects, pipelines, shaders, and bind group providers in a single composable unit.
- **Profiler** — Built-in frame timing profiler for engine tick, render, and per-phase metrics.

---

## Architecture

```
engine/
├── camera/          Camera, CameraController, GPU uniform types
├── game_object/     GameObject with transform, model, and animation state
├── light/           Point/directional lights, shadow maps, forward+ tile culling
├── loader/          glTF 2.0 importer (meshes, materials, skeletons, animations)
├── model/           Model, Mesh, GPU vertex types, instance data
├── profiler/        Frame timing profiler
├── renderer/
│   ├── animator/    GPU compute animation backends (simple + skeletal)
│   ├── bind_group_provider/  Bind group creation and buffer writes
│   ├── material/    Material GPU types (overlay, effect params)
│   ├── pipeline/    Render and compute pipeline management
│   └── shader/      Shader loading, WGSL parsing, annotation pre-processor
├── scene/           Scene graph, draw calls, compute dispatch, resource wiring
└── window/          GLFW window abstraction

common/              Shared types, math utilities, key codes, frustum culling
assets/
├── models/          glTF model assets
└── shaders/         WGSL shader source files
```

---

## Quick Start

The repository includes several runnable example scenes in the `examples` directory:

| File                | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `scene.go`          | Basic unlit scene with a rotating cube                          |
| `scene_fox.go`      | Animated fox model with skeletal animation                      |
| `scene_lit.go`      | Full forward+ lit scene with shadows and multiple lights        |
| `many_cubes.go`     | Instanced rendering stress test (unlit)                         |
| `many_foxes.go`     | Instanced skeletal animation stress test (unlit)                |
| `many_cubes_lit.go` | Lit instanced cube stress test with shadows and orbiting sun    |
| `many_foxes_lit.go` | Lit instanced skeletal animation stress test with full lighting |

Run any example with:

```bash
go run examples/scene_lit.go
```

### Minimal Example

```go
package main

import (
    "github.com/Carmen-Shannon/oxy-go/engine"
    "github.com/Carmen-Shannon/oxy-go/engine/window"
)

func main() {
    eng := engine.NewEngine(
        engine.WithProfiler(true), // profiler option
        engine.WithTickRate(60), // engine tick rate in hz
        engine.WithWindow(window.NewWindow(
            window.WithTitle("Oxy Engine"),
            window.WithWidth(1280),
            window.WithHeight(720),
        )),
    )

    eng.SetTickCallback(func(dt float32) {
        // game logic here
    })

    eng.SetRenderCallback(func(dt float32) {
        // render calls here
    })

    eng.Run()
}
```

---

## Design Principles

### Option-Builder Pattern

All constructors use variadic functional options instead of config structs:

```go
ctrl := camera.NewCameraController(
    camera.WithTarget(0, 0, 0),
    camera.WithRadius(5),
    camera.WithElevation(0.4),
)

cam := camera.NewCamera(
    camera.WithFov(45.0 * (math.Pi / 180.0)),
    camera.WithAspect(16.0 / 9.0),
    camera.WithController(ctrl),
)
```

### Interface-First

Every major system exposes an interface backed by an unexported struct. Compile-time checks enforce implementation compliance:

```go
var _ Renderer = &renderer{}
```

### Shader Annotations

WGSL shaders use `@oxy:` annotations to declare their resource requirements directly in the shader source. The engine's Scene reads these declarations at load time to wire GPU bind groups automatically. See [README_ANNOTATIONS.md](README_ANNOTATIONS.md) for the full specification.

---

## Documentation

The [`engine`](README_ENGINE.md) package is the **main entrypoint** of oxy-go. It represents the highest-level instance of the engine itself — the single object that owns the window, manages scenes by z-index, and drives all render and game logic through its concurrent tick and render loops.

- [Common](README_COMMON.md) — Shared types, math utilities (matrix ops, projection, byte conversions), frustum culling, virtual key codes, and generic helpers.
- [Engine](README_ENGINE.md) — Engine interface, tick/render loops, scene management, profiling, builder options, and shutdown lifecycle.
- [Camera System](README_CAMERA.md) — Camera and CameraController interfaces, builder options, orbit/planar controls, and GPU uniform types.
- [GameObject System](README_GAME_OBJECT.md) — GameObject interface, builder options, transform lifecycle, and light attachment.
- [Light System](README_LIGHT.md) — Light types, Forward+ tile culling, shadow mapping, GPU types, and builder options.
- [Loader System](README_LOADER.md) — Model loading and caching, glTF/GLB support, mesh/material/skeleton/animation extraction, and shader-driven GPU resource initialization.
- [Model System](README_MODEL.md) — Model interface, GPU vertex types, skeleton and animation data structures, import types, and WGSL assets.
- [Renderer System](README_RENDERER.md) — Renderer interface, pipeline cache, frame lifecycle (compute → shadow → render → present), backend types, builder options, and sub-package index.
  - [Animator](README_ANIMATOR.md) — GPU compute animation backends (simple + skeletal), per-instance transform staging, frustum culling, skeletal clip blending, and GPU type definitions.
  - [Bind Group Provider](README_BGP.md) — GPU bind group abstraction, per-entity resource storage (buffers, textures, samplers), batched buffer writes, and release lifecycle.
  - [Material](README_MATERIAL.md) — Material interface, surface properties, texture references, GPU uniform types (overlay/effect params), and builder options.
  - [Pipeline](README_PIPELINE.md) — Render and compute pipeline configuration, depth/blend/cull state, shader attachment, and builder options.
  - [Shader](README_SHADER.md) — WGSL shader loading, annotation pre-processor, bind group layout extraction, vertex layout parsing, and workgroup size resolution.
- [Scene System](README_SCENE.md) — Scene interface, object management, animator pool, lighting/shadow/Forward+ initialization, frame lifecycle, parallel compute prep, and annotation-driven draw calls.
- [Window System](README_WINDOW.md) — GLFW-based windowing, input callbacks, high-DPI handling, WebGPU surface creation, and builder options.
- [Shader Annotation System](README_ANNOTATIONS.md) — Full syntax reference, placement rules, and examples for the `@oxy:include`, `@oxy:group`, and `@oxy:provider` annotations.

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE)
