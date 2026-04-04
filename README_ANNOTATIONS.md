# Oxy Shader Annotation System

The Oxy annotation system is a WGSL pre-processor that embeds semantic metadata directly into shader source files. Annotations are parsed at shader load time to:

1. **Inject** WGSL struct definitions from Go-embedded `.wgsl` asset files
2. **Generate** `@group`/`@binding` variable declarations with correct types and address spaces
3. **Register** resource provider identities so the Scene can wire GPU bind groups without hard-coded string lookups

All annotations live in WGSL comment lines (prefixed with `//`) so they are invisible to the WGSL compiler. The pre-processor replaces annotation lines with generated WGSL (or removes them entirely for provider annotations) before the source reaches the GPU.

---

## Table of Contents

- [Annotation Syntax](#annotation-syntax)
  - [@oxy:include](#oxyinclude)
  - [@oxy:group](#oxygroup)
  - [@oxy:provider](#oxyprovider)
  - [@oxy:inject](#oxyinject)
- [Struct Type Arguments](#struct-type-arguments)
- [Address Space Arguments](#address-space-arguments)
- [Provider Identity Arguments](#provider-identity-arguments)
- [Material Binding Role Arguments](#material-binding-role-arguments)
- [Placement Rules](#placement-rules)
- [How It Works End-to-End](#how-it-works-end-to-end)
- [Example: Annotating a Lit Vertex Shader](#example-annotating-a-lit-vertex-shader)
- [Example: Annotating a Compute Shader](#example-annotating-a-compute-shader)
- [Adding a New Struct Type](#adding-a-new-struct-type)

---

## Annotation Syntax

All annotations are single-line WGSL comments beginning with `//@oxy:`. Whitespace before the `//` is permitted.

### @oxy:include

Injects the WGSL struct definition source for a registered type. The struct source is embedded from the corresponding Go GPU type's `.wgsl` asset file at compile time.

**Syntax:**

```wgsl
//@oxy:include <struct_type>
```

**Parameters:**

| Position | Name          | Description                                                                        |
| -------- | ------------- | ---------------------------------------------------------------------------------- |
| 1        | `struct_type` | A registered struct type key (see [Struct Type Arguments](#struct-type-arguments)) |

**Behavior:** The annotation line is replaced with the full WGSL struct definition. No declaration is produced.

**Example:**

```wgsl
//@oxy:include camera
// Becomes:
// struct CameraUniform {
//     view_projection: mat4x4<f32>,
//     camera_position: vec3<f32>,
//     _pad: f32,
// };
```

---

### @oxy:group

Generates a `@group(N) @binding(M) var<...>` declaration and registers it as a declaration for Scene-level resource wiring.

**Syntax:**

```wgsl
//@oxy:group <group> <binding> <address_space> <var_name> <type>
```

**Parameters:**

| Position | Name            | Description                                                                                                         |
| -------- | --------------- | ------------------------------------------------------------------------------------------------------------------- |
| 1        | `group`         | Integer `@group` index                                                                                              |
| 2        | `binding`       | Integer `@binding` index                                                                                            |
| 3        | `address_space` | One of `storage_uniform`, `storage_read`, `storage_read_write`                                                      |
| 4        | `var_name`      | The WGSL variable name to emit                                                                                      |
| 5        | `type`          | A registered struct type key, optionally wrapped in `array<>` (see [Struct Type Arguments](#struct-type-arguments)) |

**Behavior:** The annotation line is replaced with a generated WGSL declaration. An `Annotation` with `Type = AnnotationTypeBindingGroup` is appended to the declarations list.

**Generated output pattern:**

```wgsl
@group(G) @binding(B) var<address_space> var_name: Type;
```

**Annotation.Args layout:** `[0]` = address space, `[1]` = var name, `[2]` = type key

**Examples:**

```wgsl
//@oxy:group 0 0 storage_uniform camera camera
// Generates: @group(0) @binding(0) var<uniform> camera: CameraUniform;

//@oxy:group 0 1 storage_read_write instance_data array<animation_data>
// Generates: @group(0) @binding(1) var<storage, read_write> instance_data: array<AnimationData>;

//@oxy:group 3 1 storage_read lights array<light>
// Generates: @group(3) @binding(1) var<storage, read> lights: array<Light>;
```

---

### @oxy:provider

Registers a resource provider identity for a specific group and binding without generating any WGSL output. The actual WGSL `@group`/`@binding` declaration must be written manually on the line immediately below the annotation.

This is used for bindings containing raw WGSL types that have no registered struct in the pre-processor (textures, samplers, flat arrays of primitives like `array<u32>`, `array<f32>`, `array<vec4<f32>>`).

An optional `binding_role` qualifier can be appended after the provider identity to declare the semantic purpose of an individual binding within a multi-binding provider group. This lets the loader resolve binding indices from declarations instead of relying on variable-name string matching. See [Material Binding Role Arguments](#material-binding-role-arguments) for the valid role keys.

**Syntax:**

```wgsl
//@oxy:provider <group> <binding> <provider_identity> [<binding_role>]
```

**Parameters:**

| Position | Name                | Required | Description                                                                                                  |
| -------- | ------------------- | -------- | ------------------------------------------------------------------------------------------------------------ |
| 1        | `group`             | Yes      | Integer `@group` index                                                                                       |
| 2        | `binding`           | Yes      | Integer `@binding` index for the hand-written declaration on the next line                                   |
| 3        | `provider_identity` | Yes      | One of the valid provider identity keys (see [Provider Identity Arguments](#provider-identity-arguments))    |
| 4        | `binding_role`      | No       | One of the valid binding role keys (see [Material Binding Role Arguments](#material-binding-role-arguments)) |

**Behavior:** The annotation line is consumed (removed from WGSL output). An `Annotation` with `Type = AnnotationTypeProvider` is appended to the declarations list. The hand-written WGSL binding immediately below is preserved.

**Annotation.Args layout:** `[0]` = provider identity, `[1]` = binding role (present only when the annotation includes a fourth argument)

**Examples:**

```wgsl
// Per-binding material provider with roles:
//@oxy:provider 2 0 material diffuse_texture
@group(2) @binding(0) var diffuse_texture: texture_2d<f32>;
//@oxy:provider 2 1 material diffuse_sampler
@group(2) @binding(1) var diffuse_sampler: sampler;

// Provider without a binding role (single-binding or role-agnostic):
//@oxy:provider 0 4 animator_output
@group(0) @binding(4) var<storage, read_write> output_transforms: array<f32>;
```

---

### @oxy:inject

Replaces the annotation line with a WGSL `const` declaration whose value is supplied at pre-process time from the injection map passed to `Process()`. This allows Go constants to be injected into WGSL shaders without hard-coding values in shader source.

**Syntax:**

```wgsl
//@oxy:inject <const_name> <wgsl_type> <injection_key>
```

**Parameters:**

| Position | Name            | Description                                                                                   |
| -------- | --------------- | --------------------------------------------------------------------------------------------- |
| 1        | `const_name`    | The WGSL identifier for the emitted `const` declaration                                       |
| 2        | `wgsl_type`     | One of `u32`, `f32`, or `i32`                                                                 |
| 3        | `injection_key` | A registered injection key whose value is provided in the injection map passed to `Process()` |

**Behavior:** The annotation line is replaced with `const <const_name>: <wgsl_type> = <value>;`. No `Annotation` declaration entry is produced — `@oxy:inject` is purely a source substitution.

**Valid injection keys:** `max_bones`, `max_ssao_samples`, `tile_size`, `max_lights_per_tile`, `num_threads`, `slots_per_cell`, `flag_active`, `flag_static`, `flag_kinematic`, `empty_sentinel`, `body_idx_mask`, `pcf_samples`, `pcf_samples_spot`, `light_type_directional`, `light_type_point`, `light_type_spot`, `luminance_workgroup_size`

**Example:**

```wgsl
//@oxy:inject MAX_BONES u32 max_bones
// Becomes:
// const MAX_BONES: u32 = 256;
```

---

## Struct Type Arguments

These are the valid `struct_type` values for `@oxy:include` and `@oxy:group` annotations. Each maps to a Go GPU type with an embedded `.wgsl` asset file.

| Argument Key              | WGSL Type Name          | Go Source                           | Asset File                                                     |
| ------------------------- | ----------------------- | ----------------------------------- | -------------------------------------------------------------- |
| `camera`                  | `CameraUniform`         | `camera.GPUCameraUniform`           | `engine/camera/assets/camera-uniform.wgsl`                     |
| `vertex`\*                | `VertexInput`           | `model.GPUVertex`                   | `engine/model/assets/vertex.wgsl`                              |
| `skinned_vertex`\*        | `VertexInput`           | `model.GPUSkinnedVertex`            | `engine/model/assets/skinned-vertex.wgsl`                      |
| `overlay_params`          | `OverlayParams`         | `material.GPUOverlayParams`         | `engine/renderer/material/assets/overlay-params.wgsl`          |
| `effect_params`           | `EffectParams`          | `material.GPUEffectParams`          | `engine/renderer/material/assets/effect-params.wgsl`           |
| `light`                   | `Light`                 | `light.GPULight`                    | `engine/light/assets/light.wgsl`                               |
| `light_header`            | `LightHeader`           | `light.GPULightHeader`              | `engine/light/assets/light-header.wgsl`                        |
| `light_cull_uniforms`\*   | `LightCullUniforms`     | `light.GPULightCullUniforms`        | `engine/light/assets/light-cull-uniforms.wgsl`                 |
| `shadow_uniform`          | `ShadowUniform`         | `light.GPUShadowUniform`            | `engine/light/assets/shadow-uniform.wgsl`                      |
| `tile_uniforms`           | `TileUniforms`          | `light.GPUTileUniforms`             | `engine/light/assets/tile-uniforms.wgsl`                       |
| `csm_data`                | `CSMData`               | `light.GPUCSMData`                  | `engine/light/assets/csm-data.wgsl`                            |
| `contact_shadow_params`\* | `ContactShadowParams`   | `light.GPUContactShadowParams`      | `engine/light/assets/contact-shadow-params.wgsl`               |
| `light_shadow_entry`      | `LightShadowEntry`      | `light.GPULightShadowEntry`         | `engine/light/assets/light-shadow-entry.wgsl`                  |
| `model_data`              | `ModelData`             | `model.GPUModelData`                | `engine/model/assets/model-data.wgsl`                          |
| `instance_data`           | `InstanceData`          | `animator.GPUInstanceData`          | `engine/renderer/animator/assets/instance-data.wgsl`           |
| `animation_data`          | `AnimationData`         | `animator.GPUAnimationData`         | `engine/renderer/animator/assets/animation-data.wgsl`          |
| `skeletal_animation_data` | `SkeletalAnimationData` | `animator.GPUSkeletalAnimationData` | `engine/renderer/animator/assets/skeletal-animation-data.wgsl` |
| `animation_globals`       | `AnimationGlobals`      | `animator.GPUAnimationGlobals`      | `engine/renderer/animator/assets/animation-globals.wgsl`       |
| `frustum_plane`\*         | `FrustumPlane`          | `animator.GPUFrustumPlane`          | `engine/renderer/animator/assets/frustum-plane.wgsl`           |
| `global_data`             | `GlobalData`            | `animator.GPUGlobalData`            | `engine/renderer/animator/assets/simple-globals.wgsl`          |
| `indirect_args`           | `IndirectArgs`          | `animator.GPUIndirectArgs`          | `engine/renderer/animator/assets/indirect-args.wgsl`           |
| `bone_info`               | `BoneInfo`              | `animator.GPUBoneInfo`              | `engine/renderer/animator/assets/bone-info.wgsl`               |
| `physics_body`            | `Body`                  | `physics.GPUBody`                   | `engine/physics/assets/body.wgsl`                              |
| `physics_particle`        | `Particle`              | `physics.GPUParticle`               | `engine/physics/assets/particle.wgsl`                          |
| `physics_grid`            | `GridCell`              | `physics.GPUGridCell`               | `engine/physics/assets/grid-cell.wgsl`                         |
| `physics_globals`         | `PhysicsGlobals`        | `physics.GPUPhysicsGlobals`         | `engine/physics/assets/physics-globals.wgsl`                   |
| `physics_grid_params`     | `GridParams`            | `physics.GPUGridParams`             | `engine/physics/assets/grid-params.wgsl`                       |
| `gbuffer_output`          | `GBufferOutput`         | `light.GPUGBufferOutput`            | `engine/light/assets/gbuffer-output.wgsl`                      |
| `ssao_params`\*           | `SSAOParams`            | `light.GPUSSAOParams`               | `engine/light/assets/ssao-params.wgsl`                         |
| `blur_params`\*           | `BlurParams`            | `light.GPUBlurParams`               | `engine/light/assets/blur-params.wgsl`                         |
| `composition_params`\*    | `CompositionParams`     | `light.GPUCompositionParams`        | `engine/light/assets/composition-params.wgsl`                  |
| `ssr_params`\*            | `SSRParams`             | `light.GPUSSRParams`                | `engine/light/assets/ssr-params.wgsl`                          |
| `luminance_params`\*      | `LuminanceParams`       | `light.GPULuminanceParams`          | `engine/light/assets/luminance-params.wgsl`                    |
| `bloom_params`\*          | `BloomParams`           | `light.GPUBloomParams`              | `engine/light/assets/bloom-params.wgsl`                        |

\* Unexported keys — used internally by the pre-processor but cannot be matched from outside the shader package.

---

## Address Space Arguments

These are the valid `address_space` values for `@oxy:group` annotations.

| Argument Key         | Generated WGSL             |
| -------------------- | -------------------------- |
| `storage_uniform`    | `var<uniform>`             |
| `storage_read`       | `var<storage, read>`       |
| `storage_read_write` | `var<storage, read_write>` |

---

## Provider Identity Arguments

These are the valid `provider_identity` values for `@oxy:provider` annotations. Each maps to a specific Scene-level resource provider that the Scene's draw call and compute setup logic uses to wire BindGroupProviders.

| Argument Key       | Description                                                  | Typical Bindings                                         |
| ------------------ | ------------------------------------------------------------ | -------------------------------------------------------- |
| `camera`           | Camera uniform provider                                      | `CameraUniform`                                          |
| `material`         | Material textures, samplers, and uniforms                    | `texture_2d`, `sampler`, material params                 |
| `lights`           | Light storage buffer provider                                | `LightHeader`, `array<Light>`                            |
| `shadow`           | Shadow map texture and comparison sampler                    | `texture_depth_2d_array`, `sampler_comparison`           |
| `tiles`            | Forward+ tile culling data                                   | `array<u32>` counts/indices                              |
| `effect`           | Visual effect/overlay parameters                             | `OverlayParams`, `EffectParams`                          |
| `animator`         | Skinned vertex shader instance buffer                        | `array<vec4<f32>>` bone/transform data                   |
| `animator_output`  | Compute shader output transforms buffer                      | `array<f32>` (shared with vertex shader instance buffer) |
| `animator_packed`  | Packed animation data (clips, channels, keyframes)           | `array<u32>` flat packed buffer                          |
| `animator_scratch` | Scratch bone matrix workspace for blending                   | `array<mat4x4<f32>>`                                     |
| `ssao`             | SSAO blurred occlusion texture + sampler (lit pass)          | `texture_2d<f32>`, `sampler`                             |
| `contact_shadows`  | Contact shadow occlusion texture + sampler (lit pass)        | `texture_2d<f32>`, `sampler`                             |
| `composition`      | Composition HDR + SSR textures + sampler + params            | `texture_2d<f32>`, `sampler`, `CompositionParams`        |
| `ssr`              | SSR compute I/O (G-Buffer + HDR + output + params)           | `texture_2d<f32>`, `texture_storage_2d`, `SSRParams`     |
| `hiz_init`         | Hi-Z init (G-Buffer depth → Hi-Z mip 0 copy)                 | `texture_2d<f32>`, `texture_storage_2d`                  |
| `hiz_down`         | Hi-Z downsample (mip N-1 → mip N min-downsample)             | `texture_2d<f32>`, `texture_storage_2d`                  |
| `hiz_down_max`     | Hi-Z MAX pyramid downsample (mip N-1 → mip N)                | `texture_2d<f32>`, `texture_storage_2d`                  |
| `animator_hiz`     | Animator Hi-Z min pyramid for SSR ray march visibility       | `texture_2d<f32>`                                        |
| `animator_max_hiz` | Animator MAX Hi-Z pyramid for per-instance occlusion culling | `texture_2d<f32>`                                        |

---

## Material Binding Role Arguments

These are the valid `binding_role` values for the optional fourth argument of `@oxy:provider` annotations. They qualify individual bindings within a material provider group, telling the loader which texture or sampler role each binding fulfils.

| Argument Key                 | Description                                                          |
| ---------------------------- | -------------------------------------------------------------------- |
| `diffuse_texture`            | Diffuse / base-color `texture_2d<f32>` binding                       |
| `diffuse_sampler`            | Sampler paired with the diffuse texture                              |
| `normal_texture`             | Tangent-space normal map `texture_2d<f32>` binding                   |
| `normal_sampler`             | Sampler paired with the normal map                                   |
| `metallic_roughness_texture` | Combined metallic-roughness `texture_2d<f32>` binding                |
| `metallic_roughness_sampler` | Sampler paired with the metallic-roughness texture                   |
| `ssao_texture`               | SSAO blurred occlusion `texture_2d<f32>` binding                     |
| `ssao_sampler`               | Sampler paired with the SSAO occlusion texture                       |
| `gbuffer_normal`             | G-Buffer world-normal `texture_2d<f32>` binding                      |
| `gbuffer_depth`              | G-Buffer depth `texture_2d<f32>` binding                             |
| `hdr_texture`                | HDR lit result `texture_2d<f32>` binding                             |
| `ssr_output`                 | SSR compute output `texture_storage_2d` binding                      |
| `ssr_texture`                | SSR result `texture_2d<f32>` binding (composition)                   |
| `composition_sampler`        | Linear sampler for composition pass                                  |
| `hiz_out`                    | Hi-Z output `texture_storage_2d` binding                             |
| `hiz_in`                     | Hi-Z input `texture_2d<f32>` binding (previous mip)                  |
| `hiz_texture`                | Full Hi-Z depth pyramid `texture_2d<f32>` binding                    |
| `hiz_max_texture`            | MAX Hi-Z depth pyramid `texture_2d<f32>` binding (occlusion culling) |
| `spot_shadow_texture`        | Spot/point light shadow atlas depth texture binding                  |
| `contact_shadow_texture`     | Contact shadow occlusion `texture_2d<f32>` binding                   |
| `contact_shadow_sampler`     | Sampler paired with the contact shadow occlusion texture             |
| `material_params`            | Per-material scalar parameters uniform binding                       |

**Usage:** Binding roles qualify the semantic purpose of an individual binding within any multi-binding `@oxy:provider` group — not just `material` providers. The remaining roles listed above (`ssao_*`, `gbuffer_*`, `hdr_*`, `ssr_*`, `composition_*`, `hiz_*`) are used by the GI subsystem providers (`ssao`, `composition`, `ssr`, `hiz_init`, `hiz_down`). Each binding in the material group should have its own `@oxy:provider` annotation with a role:

```wgsl
//@oxy:provider 2 0 material diffuse_texture
@group(2) @binding(0) var diffuse_texture: texture_2d<f32>;
//@oxy:provider 2 1 material diffuse_sampler
@group(2) @binding(1) var diffuse_sampler: sampler;
//@oxy:provider 2 2 material normal_texture
@group(2) @binding(2) var normal_texture: texture_2d<f32>;
// ...
```

The loader reads these roles from `Shader.Declarations()` to resolve per-binding texture and sampler assignments without any variable-name string matching.

---

## Placement Rules

### @oxy:include

- Place **before** any `@oxy:group` or hand-written `@group`/`@binding` declarations that reference the included struct type.
- Multiple includes can appear in sequence at the top of the bind group section.
- Each include injects the struct source in-place, so order matters if one struct references another (e.g. `frustum_plane` must precede `global_data` or `animation_globals` since those structs contain `FrustumPlane` fields).

### @oxy:group

- Place on the line **immediately before** any commented-out reference copy of the declaration (optional but recommended for readability).
- The annotation replaces itself with the generated `@group(G) @binding(B) var<...> name: Type;` line.
- The var name should match the name used in the shader body.

### @oxy:provider

- Place on the line **immediately before** each hand-written `@group`/`@binding` declaration that belongs to the provider.
- For multi-binding provider groups (e.g. material textures and samplers), use one `@oxy:provider` annotation per binding with a `binding_role` qualifier rather than a single annotation covering the whole group.
- The annotation is consumed (produces no WGSL output); the raw WGSL binding immediately below it is preserved.
- The `binding` index must match the `@binding(N)` on the declaration that follows.

---

## How It Works End-to-End

```
┌─────────────────────────────────────────────────┐
│  WGSL Source File (with @oxy: annotations)      │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  PreProcessor.Process()                         │
│  ├─ @oxy:include → inject struct source         │
│  ├─ @oxy:group  → emit declaration + record     │
│  ├─ @oxy:provider → consume line + record       │
│  └─ @oxy:inject → emit const from value map     │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌───────────────────┐    ┌─────────────────────┐
│  Processed WGSL   │    │  Declarations List   │
│  (valid WGSL for  │    │  ([]Annotation with  │
│   GPU compilation) │    │   Group, Binding,    │
│                   │    │   Type/Identity/Role)│
└───────────────────┘    └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                      ▼
              ┌─────────────────────┐ ┌──────────────────┐
              │  Scene Methods      │ │  Loader          │
              │  ├─ DrawCalls()     │ │  initMaterialGPU │
              │  ├─ PrepareCompute()│ │                  │
              │  ├─ createAnimator()│ │  Match binding   │
              │  └─ PrepareShadows()│ │  roles to texture│
              │                     │ │  & sampler slots │
              │  Match declarations │ └──────────────────┘
              │  to providers:      │
              │  camera, material,  │
              │  lights, shadow,    │
              │  tiles, animator... │
              └─────────────────────┘
```

1. **Shader Load** — `NewShader()` reads the `.wgsl` file, runs the PreProcessor, then passes the processed source to the WGSL parser for layout extraction.
2. **Pre-Processing** — Annotations are parsed line-by-line. `@oxy:include` injects struct source. `@oxy:group` emits a `@group/@binding` declaration and records it. `@oxy:provider` only records. `@oxy:inject` emits a `const` declaration from the injection value map.
3. **WGSL Parsing** — The processed source (now valid WGSL) is parsed to extract `BindGroupLayoutDescriptors`, `BindGroupVarNames`, vertex layouts, workgroup sizes, and entry points.
4. **Scene Wiring** — Scene methods iterate `Shader.Declarations()` to semantically match each group/binding to the correct `BindGroupProvider` based on the annotation's type argument or provider identity — no string matching on variable names required.
5. **Loader Wiring** — The Loader's `initMaterialGPU` also iterates `Declarations()` to find material provider annotations with binding roles, resolving per-binding texture and sampler assignments declaratively instead of searching variable names.

---

## Example: Annotating a Lit Vertex Shader

```wgsl
// ── Struct includes ────────────────────────────────────────────────
//@oxy:include vertex
//@oxy:include camera
//@oxy:include instance_data

// ── Bind groups ────────────────────────────────────────────────────
//@oxy:group 0 0 storage_uniform camera camera
// Generates: @group(0) @binding(0) var<uniform> camera: CameraUniform;

//@oxy:group 1 0 storage_read instance_buffer array<instance_data>
// Generates: @group(1) @binding(0) var<storage, read> instance_buffer: array<InstanceData>;

@vertex
fn main(input: VertexInput, @builtin(instance_index) idx: u32) -> ... {
    let model_matrix = instance_buffer[idx].model;
    // ...
}
```

**Declarations produced:**

1. `AnnotationTypeBindingGroup` — group=0, binding=0, type=`camera`
2. `AnnotationTypeBindingGroup` — group=1, binding=0, type=`array<instance_data>`

The Scene matches group 0 to the camera provider and group 1 to the animator output provider based on the `instance_data` type.

---

## Example: Annotating a Compute Shader

```wgsl
//@oxy:include animation_data
//@oxy:include frustum_plane
//@oxy:include global_data
//@oxy:include indirect_args

// ── Bind group 0 ───────────────────────────────────────────────────
//@oxy:group 0 0 storage_uniform globals global_data
//@oxy:group 0 1 storage_read_write instance_data array<animation_data>

// Raw binding — no registered struct, use provider annotation:
//@oxy:provider 0 2 animator_output
@group(0) @binding(2) var<storage, read_write> output_transforms: array<f32>;

//@oxy:group 0 3 storage_read_write indirect_args indirect_args
```

**Declarations produced:**

1. `AnnotationTypeBindingGroup` — group=0, binding=0, type=`global_data`
2. `AnnotationTypeBindingGroup` — group=0, binding=1, type=`array<animation_data>`
3. `AnnotationTypeProvider` — group=0, binding=2, identity=`animator_output`
4. `AnnotationTypeBindingGroup` — group=0, binding=3, type=`indirect_args`

The Scene's `createAnimator` uses these declarations to resolve buffer sizing and provider wiring without any `BindGroupFromVarName` string lookups.

---

## Adding a New Struct Type

To register a new GPU struct for use with `@oxy:include` and `@oxy:group` annotations:

1. **Create the GPU struct** in the appropriate package (e.g. `engine/physics/gpu_types.go`) with `Size() int` and `Marshal() []byte` methods.
2. **Create the `.wgsl` asset file** in the package's `assets/` directory with the WGSL struct definition. Embed it via `//go:embed` and export the source string.
3. **Add the argument constant** in `engine/renderer/shader/annotations.go`:
   - Add a new `AnnotationArg` constant (exported if the key must be referenced outside the `shader` package, unexported for engine-internal types — e.g. `AnnotationArgPhysicsBody AnnotationArg = "physics_body"`).
   - Append the key to the `validStructTypes` slice.
4. **Register the struct source** in `engine/renderer/shader/pre_processor.go`'s `structRegistry` map, mapping the key to the embedded WGSL source string.
5. **Use the annotation** in your `.wgsl` shader files with `@oxy:include <key>` or `@oxy:group ... <key>`.

---
