# Material

The `engine/renderer/material` package defines the render material abstraction for the oxy-go engine. A material encapsulates surface properties (color, metallic, roughness), texture references (diffuse, normal, metallic-roughness), and GPU resource bindings (pipeline key, bind group provider) needed for draw calls. Materials are created at model load time by the Loader and wired to GPU resources during the scene initialization phase.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/renderer/material`

---

## Architecture

```
Material (public interface)
 ├─ common.Delegate[Material]   ← mock / test delegation
 └─ material (unexported struct)
      └─ common.DelegateImpl[Material]
```

The package follows the standard oxy-go interface-first pattern: a single public `Material` interface backed by an unexported `material` struct with a compile-time implementation check. The interface embeds `common.Delegate[Material]` for mock/test delegation support.

---

## Material Interface

### Read-Only Surface Properties

Set at creation time via builder options and read-only through the interface.

| Method                                               | Description                                                    |
| ---------------------------------------------------- | -------------------------------------------------------------- |
| `Name() string`                                      | Material identifier (from glTF or manually assigned)           |
| `BaseColor() [4]float32`                             | Albedo/diffuse RGBA color (default `{1,1,1,1}`)                |
| `Metallic() float32`                                 | Metallic factor: `0.0` = dielectric, `1.0` = metal (default 0) |
| `Roughness() float32`                                | Roughness factor: `0.0` = smooth, `1.0` = rough (default 1)    |
| `DiffuseTexture() *common.ImportedTexture`           | Diffuse/albedo texture reference, or nil                       |
| `NormalTexture() *common.ImportedTexture`            | Normal map texture reference, or nil                           |
| `MetallicRoughnessTexture() *common.ImportedTexture` | Metallic-roughness map reference, or nil                       |
| `FragmentShaderPath() string`                        | Path to the fragment shader source this material uses          |

### Mutable GPU Bindings

Set during the scene's GPU-init phase after construction.

| Method                                                                   | Description                                             |
| ------------------------------------------------------------------------ | ------------------------------------------------------- |
| `PipelineKey() string`                                                   | Key identifying the render pipeline this material uses  |
| `BindGroupProvider() bind_group_provider.BindGroupProvider`              | Bind group provider holding GPU resources               |
| `SetPipelineKey(key string)`                                             | Updates the pipeline key                                |
| `SetBindGroupProvider(provider bind_group_provider.BindGroupProvider)`   | Updates the bind group provider                         |
| `SetFragmentShaderPath(path string)`                                     | Updates the fragment shader source path                 |
| `Provider(group int) bind_group_provider.BindGroupProvider`              | Per-group bind group provider lookup                    |
| `SetProvider(group int, provider bind_group_provider.BindGroupProvider)` | Sets the bind group provider for a specific group index |

---

## Builder Options

The `NewMaterial` constructor accepts variadic `MaterialBuilderOption` functions:

| Option                              | Description                                   |
| ----------------------------------- | --------------------------------------------- |
| `WithName(name)`                    | Sets the material identifier                  |
| `WithBaseColor(color)`              | Sets the albedo/diffuse RGBA color            |
| `WithMetallic(metallic)`            | Sets the metallic factor                      |
| `WithRoughness(roughness)`          | Sets the roughness factor                     |
| `WithDiffuseTexture(tex)`           | Sets the diffuse/albedo texture reference     |
| `WithNormalTexture(tex)`            | Sets the normal map texture reference         |
| `WithMetallicRoughnessTexture(tex)` | Sets the metallic-roughness texture reference |
| `WithPipelineKey(key)`              | Sets the render pipeline key                  |
| `WithBindGroupProvider(provider)`   | Sets the bind group provider                  |
| `WithFragmentShaderPath(path)`      | Sets the fragment shader source path          |

---

## Constructor

```go
func NewMaterial(options ...MaterialBuilderOption) Material
```

Creates a new `Material` with sensible defaults: white base color `{1,1,1,1}`, metallic `0.0`, roughness `1.0`, and an empty `providers` map. Sets `m.Delegate = m` so delegation routes to itself by default. Builder options are applied after defaults.

---

## GPU Types

The package defines two GPU-aligned uniform structs for fragment shader parameters, each with an embedded WGSL source file:

| Type               | Size | WGSL Asset            | Description                                         |
| ------------------ | ---- | --------------------- | --------------------------------------------------- |
| `GPUOverlayParams` | 16 B | `overlay-params.wgsl` | RGBA overlay color written to all fragments         |
| `GPUEffectParams`  | 16 B | `effect-params.wgsl`  | RGB tint color + alpha blend intensity for textures |

Both types implement `Size() int` and `Marshal() []byte` for GPU buffer upload.

The package also exports the embedded WGSL source strings for each type:

| Variable                 | Description                              |
| ------------------------ | ---------------------------------------- |
| `GPUOverlayParamsSource` | Embedded WGSL source for `OverlayParams` |
| `GPUEffectParamsSource`  | Embedded WGSL source for `EffectParams`  |

---

## Files

| File                  | Purpose                                                        |
| --------------------- | -------------------------------------------------------------- |
| `material.go`         | `Material` interface, `material` struct, constructor, impls    |
| `material_builder.go` | `MaterialBuilderOption` type and 10 builder functions          |
| `gpu_types.go`        | `GPUOverlayParams`, `GPUEffectParams` with Size/Marshal + WGSL |

### Assets

| File                  | Description                                |
| --------------------- | ------------------------------------------ |
| `overlay-params.wgsl` | WGSL `OverlayParams` struct (16 B, 1 vec4) |
| `effect-params.wgsl`  | WGSL `EffectParams` struct (16 B, 1 vec4)  |
