# Loader System

The `engine/loader` package provides 3D model loading and caching for the oxy-go engine. It currently supports **glTF 2.0** (.gltf) and **GLB** (.glb) files, extracting meshes, materials, skeletons, and animations into CPU-side engine-ready data structures. GPU resource creation is deferred to the Scene when a model is added via `Add()`.

---

## Architecture

The loader is organized in layers:

```
Loader (public API + model cache)
  └── loaderBackend (format-specific dispatch)
        └── gltfLoaderBackend → gltfLoaderBackendImpl (glTF backend)
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
)

ldr := loader.NewLoader(loader.BackendTypeGLTF)
```

### Backend Types

| Constant          | Description                    |
| ----------------- | ------------------------------ |
| `BackendTypeGLTF` | Selects the glTF / GLB backend |

### Builder Options

| Option                                 | Description                                           |
| -------------------------------------- | ----------------------------------------------------- |
| `WithModel(key string, m model.Model)` | Pre-populates the model cache with an existing model. |

---

## Loader Interface

| Method                                   | Description                                                                                                                               |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `Load(path string) (model.Model, error)` | Full import — meshes, skeleton, animations, materials. Caches by file path. Returns CPU-side data only; GPU init is handled by the Scene. |
| `Get(name string) model.Model`           | Retrieve a cached model by name. Returns nil if not found.                                                                                |
| `Models() map[string]model.Model`        | Returns the internal model cache map directly. Callers must not mutate the returned map.                                                  |

All `Load` calls are cache-aware: if a model has already been loaded under the same key, the cached version is returned immediately.

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
- Alpha mode (`OPAQUE`, `MASK`, `BLEND`) and alpha cutoff threshold

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
// Create loader with the glTF backend.
ldr := loader.NewLoader(loader.BackendTypeGLTF)

// Load a model — meshes, skeleton, animations, and materials (CPU-side only).
// GPU resources are initialized by the Scene when the model is added via Add().
mdl, err := ldr.Load("assets/models/fox.glb")
if err != nil {
    log.Fatal(err)
}

// Access the cached model later.
cached := ldr.Get("assets/models/fox.glb")
_ = cached
```
