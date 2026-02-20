# Shader

The `engine/renderer/shader` package handles WGSL shader loading, parsing, annotation pre-processing, and bind group layout generation for the oxy-go engine. It is the primary bridge between raw WGSL source files and the GPU pipeline/resource-wiring systems. When a shader is created, its source is read from disk, pre-processed for `@oxy:` annotations, and fully parsed to extract entry points, vertex layouts, workgroup sizes, and bind group layout descriptors — all before any GPU objects are created.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/renderer/shader`

---

## Architecture

```
Shader (public interface)
 └─ shader (unexported struct)
        └─ PreProcessor (public interface)
              └─ preProcessor (unexported struct)
```

The `Shader` holds a `PreProcessor` internally. During construction, the pre-processor scans the WGSL source for `@oxy:` annotations, replaces them with generated WGSL code (struct injections, bind group declarations), and records a declarations list. The parsed shader then exposes everything the Renderer and Scene need to create pipelines and wire bind groups.

---

## Shader Types

| Constant             | Value | Description                            |
| -------------------- | ----- | -------------------------------------- |
| `ShaderTypeCompute`  | `0`   | Compute shader with `@compute` entry   |
| `ShaderTypeVertex`   | `1`   | Vertex shader with `@vertex` entry     |
| `ShaderTypeFragment` | `2`   | Fragment shader with `@fragment` entry |

---

## Shader Interface

### Source & Identity

| Method                    | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `Key() string`            | Unique identifier for caching and lookups              |
| `Source() string`         | Processed WGSL source (annotations replaced)           |
| `ShaderType() ShaderType` | The shader stage type                                  |
| `EntryPoint() string`     | Entry point function name (e.g. `"vs_main"`)           |
| `Module()`                | `*wgpu.ShaderModuleDescriptor` for GPU module creation |

### Bind Group Metadata

| Method                                         | Description                                      |
| ---------------------------------------------- | ------------------------------------------------ |
| `BindGroupLayoutDescriptor(key int)`           | Layout descriptor for a specific group index     |
| `BindGroupLayoutDescriptors()`                 | All layout descriptors keyed by group index      |
| `BindGroupVarName(group, binding int) string`  | Variable name at a given group/binding           |
| `BindGroupFromVarName(group int, name string)` | Reverse lookup: binding index from variable name |
| `BindGroupVarNames()`                          | All variable names keyed by group and binding    |

### Vertex & Compute Metadata

| Method                      | Description                              |
| --------------------------- | ---------------------------------------- |
| `VertexLayout(key int)`     | Vertex buffer layout for a specific key  |
| `VertexLayouts()`           | All vertex buffer layouts                |
| `WorkgroupSize() [3]uint32` | Compute workgroup dimensions `[x, y, z]` |

### Declarations

| Method                        | Description                                                         |
| ----------------------------- | ------------------------------------------------------------------- |
| `Declarations() []Annotation` | `@oxy:group` and `@oxy:provider` annotations from the pre-processor |

---

## Constructor

```go
func NewShader(key string, shaderType ShaderType, sourcePath string) Shader
```

Creates a new `Shader` by reading the WGSL file at `sourcePath`, running it through the pre-processor, and extracting all metadata:

1. Pre-processes `@oxy:` annotations (struct injection, bind group declarations)
2. Builds the `wgpu.ShaderModuleDescriptor`
3. Parses the entry point name for the given shader type
4. Parses vertex buffer layouts (vertex shaders only)
5. Parses workgroup size (compute shaders only)
6. Parses bind group layout descriptors with `MinBindingSize` resolution

Panics if `sourcePath` is empty or the file cannot be read.

---

## Pre-Processor

The `PreProcessor` interface scans WGSL source for `@oxy:` annotations and produces processed output.

| Method                        | Description                                                     |
| ----------------------------- | --------------------------------------------------------------- |
| `Process(source string)`      | Returns processed WGSL and error; resets declarations each call |
| `Declarations() []Annotation` | Annotations collected during the last `Process` call            |

### Annotation Types

| Type       | Syntax                                                        | Effect                                                 |
| ---------- | ------------------------------------------------------------- | ------------------------------------------------------ |
| `include`  | `//@oxy:include <struct_type>`                                | Injects embedded WGSL struct source at annotation site |
| `group`    | `//@oxy:group <group> <binding> <addr_space> <var> <type>`    | Generates `@group/@binding var<...>` declaration       |
| `provider` | `//@oxy:provider <group> <binding> <identity> [binding_role]` | Registers provider identity; no WGSL output            |

### Struct Registry

The pre-processor maps annotation argument keys to embedded WGSL sources from GPU type packages across the engine. Currently registered types:

| Key                       | WGSL Type               | Source Package |
| ------------------------- | ----------------------- | -------------- |
| `camera`                  | `CameraUniform`         | `camera`       |
| `vertex`                  | `VertexInput`           | `model`        |
| `skinned_vertex`          | `VertexInput`           | `model`        |
| `overlay_params`          | `OverlayParams`         | `material`     |
| `effect_params`           | `EffectParams`          | `material`     |
| `light`                   | `Light`                 | `light`        |
| `light_header`            | `LightHeader`           | `light`        |
| `light_cull_uniforms`     | `LightCullUniforms`     | `light`        |
| `shadow_data`             | `ShadowData`            | `light`        |
| `shadow_uniform`          | `ShadowUniform`         | `light`        |
| `tile_uniforms`           | `TileUniforms`          | `light`        |
| `animation_data`          | `AnimationData`         | `animator`     |
| `skeletal_animation_data` | `SkeletalAnimationData` | `animator`     |
| `animation_globals`       | `AnimationGlobals`      | `animator`     |
| `frustum_plane`           | `FrustumPlane`          | `animator`     |
| `global_data`             | `GlobalData`            | `animator`     |
| `indirect_args`           | `IndirectArgs`          | `animator`     |
| `bone_info`               | `BoneInfo`              | `animator`     |
| `instance_data`           | `InstanceData`          | `animator`     |
| `model_data`              | `ModelData`             | `model`        |

---

## WGSL Parser

The parsing subsystem extracts GPU metadata from processed WGSL source without a full AST. It operates via regex-based extraction functions and type layout computation.

### Parser Functions

| Function                | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| `parseVertexLayouts`    | Extracts `wgpu.VertexBufferLayout` from structs with `@location` fields |
| `parseBindGroupLayouts` | Extracts `wgpu.BindGroupLayoutDescriptor` from `@group/@binding` decls  |
| `parseWorkgroupSize`    | Extracts `@workgroup_size(x, y, z)` dimensions                          |
| `parseEntryPoint`       | Extracts the entry point function name for a given shader type          |
| `parseStructBlocks`     | Finds all `struct { ... }` blocks and parses their fields               |
| `parseStructFields`     | Parses individual struct fields with `@location`/`@builtin` attributes  |

### Resource Classification

The `classifyResource` function determines the bind group layout entry type from a WGSL declaration's address space and type name:

- **Buffers**: `uniform` → `BufferBindingTypeUniform`, `storage` → `Storage` or `ReadOnlyStorage`
- **Samplers**: `sampler` → `Filtering`, `sampler_comparison` → `Comparison`
- **Sampled textures**: `texture_2d<f32>` etc. → dimension + sample type
- **Depth textures**: `texture_depth_2d` etc. → `TextureSampleTypeDepth`
- **Storage textures**: `texture_storage_2d<format, access>` → format + access + dimension

### Type Layout Resolution

`MinBindingSize` is computed for all buffer bindings by resolving WGSL types (primitives, structs, fixed/runtime arrays) to their byte size and alignment per the [WGSL specification](https://www.w3.org/TR/WGSL/#alignment-and-size). Struct layouts are resolved iteratively to handle inter-struct dependencies.

---

## Files

| File                     | Purpose                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| `shader.go`              | `Shader` interface, `shader` struct, `NewShader`, `parseSourceFromPath`                          |
| `annotations.go`         | Annotation types, argument constants, validation slices, `parseAnnotation`                       |
| `pre_processor.go`       | `PreProcessor` interface, struct/address-space registries, `Process`                             |
| `wgsl_parser.go`         | Vertex layout, bind group layout, workgroup, entry point parsers                                 |
| `wgsl_parser_backend.go` | Primitive layout map, type resolution, struct sizing, resource classification, comment stripping |
| `wgsl_parser_types.go`   | Internal parser data types (`vertexFormatInfo`, `parsedField`, etc.)                             |
