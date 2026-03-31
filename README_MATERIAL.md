# Material

The `engine/renderer/material` package defines the render material abstraction for the oxy-go engine. A material encapsulates surface properties (color, metallic, roughness), texture references (diffuse, normal, metallic-roughness), and GPU resource bindings (pipeline key, bind group provider) needed for draw calls. Materials are created at model load time by the Loader and wired to GPU resources during the scene initialization phase.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/renderer/material`

---

## Architecture

```
Material (public interface)
 └─ material (unexported struct in material_impl.go)
```

The package follows the standard oxy-go 3-file layout: a single public `Material` interface in `material.go`, an unexported `material` struct in `material_impl.go`, and all builder functions in `material_builder.go`.

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
| `AlphaCutoff() float32`                              | Alpha discard threshold (`0.0`–`1.0`), used in MASK alpha mode |

### Mutable GPU Bindings

Set during the scene's GPU-init phase after construction.

| Method                                                                   | Description                                             |
| ------------------------------------------------------------------------ | ------------------------------------------------------- |
| `PipelineKey() string`                                                   | Key identifying the render pipeline this material uses  |
| `BindGroupProvider() bind_group_provider.BindGroupProvider`              | Bind group provider holding GPU resources               |
| `SetPipelineKey(key string)`                                             | Updates the pipeline key                                |
| `SetBindGroupProvider(provider bind_group_provider.BindGroupProvider)`   | Updates the bind group provider                         |
| `Provider(group int) bind_group_provider.BindGroupProvider`              | Per-group bind group provider lookup                    |
| `SetProvider(group int, provider bind_group_provider.BindGroupProvider)` | Sets the bind group provider for a specific group index |
| `PipelineOptions() []any`                                                | Additional pipeline options passed during pipeline init |

---

## Builder Options

The `NewMaterial` constructor accepts variadic `MaterialBuilderOption` functions:

| Option                              | Description                                     |
| ----------------------------------- | ----------------------------------------------- |
| `WithName(name)`                    | Sets the material identifier                    |
| `WithBaseColor(color)`              | Sets the albedo/diffuse RGBA color              |
| `WithMetallic(metallic)`            | Sets the metallic factor                        |
| `WithRoughness(roughness)`          | Sets the roughness factor                       |
| `WithDiffuseTexture(tex)`           | Sets the diffuse/albedo texture reference       |
| `WithNormalTexture(tex)`            | Sets the normal map texture reference           |
| `WithMetallicRoughnessTexture(tex)` | Sets the metallic-roughness texture reference   |
| `WithPipelineKey(key)`              | Sets the render pipeline key                    |
| `WithBindGroupProvider(provider)`   | Sets the bind group provider                    |
| `WithAlphaCutoff(cutoff)`           | Sets the alpha discard threshold                |
| `WithPipelineOptions(opts ...any)`  | Sets additional pipeline initialization options |

---

## Constructor

```go
func NewMaterial(options ...MaterialBuilderOption) Material
```

Creates a new `Material` with sensible defaults: white base color `{1,1,1,1}`, metallic `0.0`, roughness `1.0`, alpha cutoff `0.01`, and an empty `providers` map. Builder options are applied after defaults.

---

## GPU Types

The package defines three GPU-aligned uniform structs for fragment shader parameters:

| Type                | Size | WGSL Asset            | Description                                                              |
| ------------------- | ---- | --------------------- | ------------------------------------------------------------------------ |
| `GPUMaterialParams` | 4 B  | —                     | Per-material scalar parameters (alpha cutoff) for the GPU uniform buffer |
| `GPUOverlayParams`  | 16 B | `overlay-params.wgsl` | RGBA overlay color written to all fragments                              |
| `GPUEffectParams`   | 16 B | `effect-params.wgsl`  | RGB tint color + alpha blend intensity for textures                      |

Both types implement `Size() int` and `Marshal() []byte` for GPU buffer upload.

The package also exports the embedded WGSL source strings for each type:

| Variable                 | Description                              |
| ------------------------ | ---------------------------------------- |
| `GPUOverlayParamsSource` | Embedded WGSL source for `OverlayParams` |
| `GPUEffectParamsSource`  | Embedded WGSL source for `EffectParams`  |

---

## Files

| File                  | Purpose                                                             |
| --------------------- | ------------------------------------------------------------------- |
| `material.go`         | `Material` interface and exported forwarding method implementations |
| `material_impl.go`    | Unexported `material` struct and internal field definitions         |
| `material_builder.go` | `MaterialBuilderOption` type and 11 builder functions               |
| `gpu_types.go`        | `GPUOverlayParams`, `GPUEffectParams` with Size/Marshal + WGSL      |

### Assets

| File                  | Description                                |
| --------------------- | ------------------------------------------ |
| `overlay-params.wgsl` | WGSL `OverlayParams` struct (16 B, 1 vec4) |
| `effect-params.wgsl`  | WGSL `EffectParams` struct (16 B, 1 vec4)  |
