# Loader System

The `engine/loader` package provides 3D model loading and caching for the oxy-go engine. It currently supports **glTF 2.0** (.gltf) and **GLB** (.glb) files, extracting meshes, materials, skeletons, and animations into engine-ready data structures.

---

## Architecture

The loader is organized in layers:

```
Loader (public API + model cache)
  └── loaderBackend (format-specific dispatch)
        └── gltfImporter (orchestration)
              ├── gltfParser         (JSON/GLB parse + accessor reads)
              ├── gltfMeshExtractor  (vertex, index, tangent data)
              ├── gltfMaterialExtractor (textures, samplers, PBR params)
              ├── gltfSkeletonExtractor (bone hierarchy, topological sort)
              └── gltfAnimationExtractor (keyframe channels)
```

Only the top-level `Loader` interface and its builder types are exported. Everything below is internal to the package.

---

## Creating a Loader

```go
import (
    "github.com/Carmen-Shannon/oxy-go/engine/loader"
    "github.com/Carmen-Shannon/oxy-go/engine/renderer"
)

ldr := loader.NewLoader(loader.BackendTypeGLTF,
    loader.WithRenderer(myRenderer),
)
```

### Backend Types

| Constant          | Description                    |
| ----------------- | ------------------------------ |
| `BackendTypeGLTF` | Selects the glTF / GLB backend |

### Builder Options

| Option                                 | Description                                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `WithRenderer(r renderer.Renderer)`    | Sets the Renderer used for GPU resource creation (mesh buffers, textures, samplers, bind groups). |
| `WithModel(key string, m model.Model)` | Pre-populates the model cache with an existing model.                                             |

---

## Loader Interface

| Method                                                                                                | Description                                                                             |
| ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `Load(path string, fragmentShader shader.Shader) (model.Model, error)`                                | Full import — meshes, skeleton, animations, materials. Caches by file path.             |
| `LoadMeshOnly(path string, fragmentShader shader.Shader) (model.Model, error)`                        | Fast import — meshes and materials only, skips skeleton/animation. Caches by file path. |
| `LoadReader(name string, r io.Reader, isGLB bool, fragmentShader shader.Shader) (model.Model, error)` | Import from a reader stream (embedded resources, network). Caches by the given name.    |
| `Get(name string) model.Model`                                                                        | Retrieve a cached model by name. Returns nil if not found.                              |
| `Models() map[string]model.Model`                                                                     | Returns a copy of the full model cache.                                                 |
| `InitMaterialGPU(mat material.Material, fragmentShader shader.Shader, providerName string) error`     | Initializes GPU resources for a hand-built material that bypasses the Load pipeline.    |

All `Load*` methods are cache-aware: if a model has already been loaded under the same key, the cached version is returned immediately.

---

## Fragment Shader Integration

The `fragmentShader` parameter is required by `Load`, `LoadMeshOnly`, `LoadReader`, and `InitMaterialGPU`. The loader reads the shader's pre-processed `Declarations()` to locate material bindings without any variable-name string matching.

### How it works

1. **Locate the material group** — scan declarations for `@oxy:provider` annotations whose provider identity is `material`. The group index of the first match identifies the material bind group.
2. **Resolve per-binding roles** — each provider annotation can carry an optional _binding role_ argument that declares what the binding is for. The loader maps roles to binding indices:

   | Role                         | Material data                     |
   | ---------------------------- | --------------------------------- |
   | `diffuse_texture`            | Base-color / diffuse image        |
   | `diffuse_sampler`            | Sampler for the above             |
   | `normal_texture`             | Tangent-space normal map          |
   | `normal_sampler`             | Sampler for the above             |
   | `metallic_roughness_texture` | Combined metallic-roughness image |
   | `metallic_roughness_sampler` | Sampler for the above             |

3. **Upload textures & samplers** — for each role that the model provides, decode the image to RGBA pixels and create the GPU texture view / sampler at the declared binding index.
4. **Fill fallback placeholders** — any shader-declared texture or sampler binding that the model doesn't populate gets a 1×1 placeholder (e.g. a flat normal `(128, 128, 255, 255)`, or a white diffuse `(255, 255, 255, 255)`).

### Example shader annotations

```wgsl
//@oxy:provider 2 0 material diffuse_texture
@group(2) @binding(0) var diffuse_texture: texture_2d<f32>;
//@oxy:provider 2 1 material diffuse_sampler
@group(2) @binding(1) var diffuse_sampler: sampler;
//@oxy:provider 2 2 material normal_texture
@group(2) @binding(2) var normal_texture: texture_2d<f32>;
//@oxy:provider 2 3 material normal_sampler
@group(2) @binding(3) var normal_sampler: sampler;
```

This declaration-driven approach means the loader works with any fragment shader layout — no hard-coded binding indices or variable-name conventions required.

---

## glTF Feature Support

### Meshes

- Triangle primitives (mode 4, the default)
- Attributes: `POSITION`, `NORMAL`, `TANGENT`, `TEXCOORD_0`, `COLOR_0`, `JOINTS_0`, `WEIGHTS_0`
- Auto-generated smooth normals when `NORMAL` is absent
- Auto-generated MikkTSpace-compatible tangents when `TANGENT` is absent
- Vertex colors in VEC3/VEC4 × FLOAT / UNSIGNED_BYTE / UNSIGNED_SHORT formats
- Per-primitive material index and bounding box calculation

### Materials (PBR Metallic-Roughness)

- Base color factor and texture
- Metallic / roughness factors and combined texture
- Normal map
- Texture image sources: external file, buffer view (GLB), data URI (base64)
- Sampler parameters (filter modes, wrap modes) converted to WebGPU equivalents

### Skeletons

- Skin → bone hierarchy with inverse bind matrices
- Topological sort guarantees parents are processed before children
- Bone index remapping applied to mesh vertices after sort

### Animations

- Translation, rotation, and scale keyframe channels
- Per-bone channel merging (all TRS channels for one bone in a single `AnimationChannel`)
- Skin-scoped extraction (only animations relevant to a skeleton)
- Timestamps in seconds (glTF spec)

### File Formats

- `.gltf` — JSON with optional external buffer/image files
- `.glb` — Binary container with embedded JSON + BIN chunks
- Data URIs — base64-encoded inline buffers and images

---

## Usage Example

```go
// Create a renderer (assumed already initialized).
rend := renderer.NewRenderer(renderer.BackendTypeWGPU, hwnd, hinstance)

// Create loader with the renderer.
ldr := loader.NewLoader(loader.BackendTypeGLTF,
    loader.WithRenderer(rend),
)

// Load a model — meshes, skeleton, animations, materials, and GPU resources.
fragShader := rend.Shader("lit-frag")
mdl, err := ldr.Load("assets/models/fox.glb", fragShader)
if err != nil {
    log.Fatal(err)
}

// Access the cached model later.
cached := ldr.Get("assets/models/fox.glb")

// Load a static model (mesh + materials only, no skeleton).
cube, err := ldr.LoadMeshOnly("assets/models/cube.gltf", fragShader)
if err != nil {
    log.Fatal(err)
}
```
