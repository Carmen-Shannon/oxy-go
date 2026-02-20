# Oxy Model System

The `model` package defines the engine's GPU-ready 3D model representation. A `Model` bundles mesh vertex/index data, skeleton hierarchy, animation clips, imported and render-ready materials, and a `BindGroupProvider` for GPU mesh resources. Models are produced by the Loader after importing a file and are consumed by the Scene, Animator, and Renderer during draw calls.

---

## Table of Contents

- [Overview](#overview)
- [Creating a Model](#creating-a-model)
- [Builder Options](#builder-options)
- [Model Interface](#model-interface)
  - [Identity & Mesh Data](#identity--mesh-data)
  - [Skeleton & Animation](#skeleton--animation)
  - [Materials](#materials)
  - [GPU Providers](#gpu-providers)
- [Data Types](#data-types)
  - [Transform](#transform)
  - [Bone & Skeleton](#bone--skeleton)
  - [Animation Types](#animation-types)
  - [Import Types](#import-types)
- [GPU Types](#gpu-types)
  - [GPUVertex](#gpuvertex)
  - [GPUSkinnedVertex](#gpuskinnedvertex)
  - [GPUModelData](#gpumodeldata)
- [WGSL Assets](#wgsl-assets)
- [Helper Functions](#helper-functions)
- [Usage Example](#usage-example)

---

## Overview

The model system is the bridge between file import and GPU rendering. It stores:

1. **Mesh data** — Raw vertex and index byte buffers plus a `BindGroupProvider` for GPU-side mesh resources.
2. **Skeleton** — A `Bone` hierarchy with inverse bind matrices, used by the skeletal animator.
3. **Animations** — `AnimationClip` objects containing per-bone keyframe channels (translation, rotation, scale).
4. **Materials** — Both raw `ImportedMaterial` data (from the loader) and GPU-configured `material.Material` instances (for draw calls).

The `Model` interface exposes all of these through a read-heavy API with selective setters for fields that change after construction (render materials, effect provider, compute pipeline key).

---

## Creating a Model

Models are typically created by the Loader, but can be constructed manually:

```go
mdl := model.NewModel(
    model.WithName("my_cube"),
    model.WithSkinned(false),
    model.WithVertexData(vertexBytes),
    model.WithIndexData(indexBytes),
    model.WithIndexCount(36),
    model.WithBoundingRadius(1.732),
)
```

The constructor applies no defaults — all fields start at their zero values. Builder options set only the fields you provide.

---

## Builder Options

All options follow the `ModelBuilderOption` functional option pattern.

| Option                   | Parameters                            | Description                                                           |
| ------------------------ | ------------------------------------- | --------------------------------------------------------------------- |
| `WithName`               | `name string`                         | Sets the model identifier                                             |
| `WithSkinned`            | `skinned bool`                        | Marks the model as using skeletal animation                           |
| `WithSkeleton`           | `skeleton *Skeleton`                  | Sets the bone hierarchy                                               |
| `WithAnimations`         | `animations []*AnimationClip`         | Sets the animation clips                                              |
| `WithImportedMaterials`  | `materials []common.ImportedMaterial` | Sets the raw imported materials from the model file                   |
| `WithMeshProvider`       | `provider BindGroupProvider`          | Sets the GPU mesh bind group provider (vertex/index buffers)          |
| `WithBoundingRadius`     | `radius float32`                      | Manually sets the bounding sphere radius (overrides auto-computation) |
| `WithRenderMaterials`    | `mats ...material.Material`           | Sets the GPU-configured render materials                              |
| `WithComputePipelineKey` | `key string`                          | Sets the compute pipeline key for the model's animator                |
| `WithVertexData`         | `data []byte`                         | Sets the raw vertex byte buffer                                       |
| `WithIndexData`          | `data []byte`                         | Sets the raw index byte buffer                                        |
| `WithIndexCount`         | `count int`                           | Sets the number of indices in the mesh                                |

---

## Model Interface

### Identity & Mesh Data

| Method                       | Description                                              |
| ---------------------------- | -------------------------------------------------------- |
| `Name() string`              | Returns the model identifier                             |
| `Skinned() bool`             | Reports whether the model uses skeletal animation        |
| `VertexData() []byte`        | Returns the raw vertex byte buffer                       |
| `SetVertexData(data []byte)` | Replaces the raw vertex byte buffer                      |
| `IndexData() []byte`         | Returns the raw index byte buffer                        |
| `SetIndexData(data []byte)`  | Replaces the raw index byte buffer                       |
| `IndexCount() int`           | Returns the number of indices in the mesh                |
| `SetIndexCount(count int)`   | Sets the index count                                     |
| `BoundingRadius() float32`   | Returns the bounding sphere radius (for frustum culling) |

### Skeleton & Animation

| Method                               | Description                                             |
| ------------------------------------ | ------------------------------------------------------- |
| `Skeleton() *Skeleton`               | Returns the bone hierarchy, or `nil` for static models  |
| `Animations() []*AnimationClip`      | Returns all animation clips                             |
| `AnimationCount() int`               | Returns the number of animation clips                   |
| `AnimationNames() []string`          | Returns the names of all animation clips                |
| `GetAnimationIndex(name string) int` | Returns the index of a named clip, or `-1` if not found |

### Materials

| Method                                          | Description                                   |
| ----------------------------------------------- | --------------------------------------------- |
| `ImportedMaterials() []common.ImportedMaterial` | Returns the raw materials from the model file |
| `RenderMaterials() []material.Material`         | Returns the GPU-configured render materials   |
| `SetRenderMaterials(mats []material.Material)`  | Replaces the render material list             |

### GPU Providers

| Method                                          | Description                                                |
| ----------------------------------------------- | ---------------------------------------------------------- |
| `MeshProvider() BindGroupProvider`              | Returns the bind group provider holding GPU mesh resources |
| `EffectProvider() BindGroupProvider`            | Returns the per-model effect parameter provider, or `nil`  |
| `SetEffectProvider(provider BindGroupProvider)` | Assigns an effect parameter bind group provider            |
| `ComputePipelineKey() string`                   | Returns the compute pipeline key for the model's animator  |
| `SetComputePipelineKey(key string)`             | Sets the compute pipeline key                              |

---

## Data Types

### Transform

Decomposed transform for animation interpolation.

| Field         | Type         | Description                         |
| ------------- | ------------ | ----------------------------------- |
| `Translation` | `[3]float32` | Position offset                     |
| `Rotation`    | `[4]float32` | Quaternion orientation (x, y, z, w) |
| `Scale`       | `[3]float32` | Per-axis scale factor               |

### Bone & Skeleton

**Bone:**

| Field               | Type          | Description                                              |
| ------------------- | ------------- | -------------------------------------------------------- |
| `Name`              | `string`      | Bone identifier (for debugging and animation targeting)  |
| `ParentIndex`       | `int32`       | Index of the parent bone (`-1` for root bones)           |
| `InverseBindMatrix` | `[16]float32` | Transforms from model space to bone space at bind pose   |
| `LocalTransform`    | `Transform`   | Current transform relative to parent (updated per frame) |

**Skeleton:**

| Field             | Type               | Description                     |
| ----------------- | ------------------ | ------------------------------- |
| `Bones`           | `[]Bone`           | All bones in the hierarchy      |
| `RootBoneIndices` | `[]int32`          | Indices of bones with no parent |
| `BoneNameToIndex` | `map[string]int32` | Bone name to index lookup       |

### Animation Types

**AnimationClip:**

| Field            | Type                 | Description                  |
| ---------------- | -------------------- | ---------------------------- |
| `Name`           | `string`             | Animation identifier         |
| `Duration`       | `float32`            | Total length in seconds      |
| `TicksPerSecond` | `float32`            | Sample rate of the animation |
| `Channels`       | `[]AnimationChannel` | Per-bone keyframe data       |

**AnimationChannel:**

| Field          | Type                   | Description                     |
| -------------- | ---------------------- | ------------------------------- |
| `BoneIndex`    | `int32`                | Index of the animated bone      |
| `PositionKeys` | `[]VectorKeyframe`     | Translation keyframes           |
| `RotationKeys` | `[]QuaternionKeyframe` | Rotation keyframes (quaternion) |
| `ScaleKeys`    | `[]VectorKeyframe`     | Scale keyframes                 |

**VectorKeyframe:**

| Field   | Type         | Description                   |
| ------- | ------------ | ----------------------------- |
| `Time`  | `float32`    | Keyframe timestamp in seconds |
| `Value` | `[3]float32` | 3D vector at this keyframe    |

**QuaternionKeyframe:**

| Field   | Type         | Description                        |
| ------- | ------------ | ---------------------------------- |
| `Time`  | `float32`    | Keyframe timestamp in seconds      |
| `Value` | `[4]float32` | Quaternion at this keyframe (xyzw) |

### Import Types

**ImportedModel** — Universal format produced by importers (glTF, etc.):

| Field        | Type                        | Description                                 |
| ------------ | --------------------------- | ------------------------------------------- |
| `Name`       | `string`                    | Model identifier                            |
| `Meshes`     | `[]ImportedMesh`            | All mesh data                               |
| `Skeleton`   | `*Skeleton`                 | Bone hierarchy (`nil` for static models)    |
| `Animations` | `[]*AnimationClip`          | Animation clips bundled with the model      |
| `Materials`  | `[]common.ImportedMaterial` | Material library referenced by mesh indices |

**ImportedMesh:**

| Field           | Type                 | Description                                          |
| --------------- | -------------------- | ---------------------------------------------------- |
| `Name`          | `string`             | Mesh identifier                                      |
| `Vertices`      | `[]GPUSkinnedVertex` | Vertices (includes bone data even for static meshes) |
| `Indices`       | `[]uint32`           | Triangle indices                                     |
| `MaterialIndex` | `int`                | Index into `ImportedModel.Materials`                 |
| `BoundingMin`   | `[3]float32`         | AABB minimum corner                                  |
| `BoundingMax`   | `[3]float32`         | AABB maximum corner                                  |

---

## GPU Types

### GPUVertex

GPU-aligned vertex for static (non-skinned) meshes. Size: **64 bytes**.

| Field      | Type         | Offset | Size     | Description                                    |
| ---------- | ------------ | ------ | -------- | ---------------------------------------------- |
| `Position` | `[3]float32` | 0      | 12 bytes | Vertex position in model space                 |
| `Normal`   | `[3]float32` | 12     | 12 bytes | Normal vector for lighting                     |
| `TexCoord` | `[2]float32` | 24     | 8 bytes  | UV texture coordinate                          |
| `Color`    | `[4]float32` | 32     | 16 bytes | Per-vertex RGBA color                          |
| `Tangent`  | `[4]float32` | 48     | 16 bytes | Tangent (xyz) + handedness (w) for normal maps |

Methods: `Size() int`, `Marshal() []byte`

### GPUSkinnedVertex

Extends `GPUVertex` with bone skinning data. Size: **96 bytes** (64 base + 32 skinning).

| Field         | Type         | Offset | Size     | Description                          |
| ------------- | ------------ | ------ | -------- | ------------------------------------ |
| _(GPUVertex)_ | —            | 0      | 64 bytes | All base vertex fields               |
| `BoneIndices` | `[4]uint32`  | 64     | 16 bytes | Indices of up to 4 influencing bones |
| `BoneWeights` | `[4]float32` | 80     | 16 bytes | Blend weights (must sum to 1.0)      |

Methods: `Size() int`, `Marshal() []byte`

### GPUModelData

Per-instance model-to-world transform matrix. Size: **64 bytes**.

| Field   | Type          | Offset | Size     | Description                   |
| ------- | ------------- | ------ | -------- | ----------------------------- |
| `Model` | `[16]float32` | 0      | 64 bytes | 4×4 column-major model matrix |

Methods: `Size() int`, `Marshal() []byte`

---

## WGSL Assets

Each GPU type has a corresponding `.wgsl` asset file embedded at compile time via `//go:embed`:

| Go Type            | WGSL Struct Name | Asset File                                | Annotation Key     |
| ------------------ | ---------------- | ----------------------------------------- | ------------------ |
| `GPUVertex`        | `VertexInput`    | `engine/model/assets/vertex.wgsl`         | `vertex`\*         |
| `GPUSkinnedVertex` | `VertexInput`    | `engine/model/assets/skinned_vertex.wgsl` | `skinned_vertex`\* |
| `GPUModelData`     | `ModelData`      | `engine/model/assets/model_data.wgsl`     | `model_data`       |

\* Unexported annotation keys — used internally by the shader pre-processor.

---

## Helper Functions

| Function                                                     | Description                                                                |
| ------------------------------------------------------------ | -------------------------------------------------------------------------- |
| `ComputeBoundingRadius(vertices []GPUSkinnedVertex) float32` | Computes the bounding sphere radius as the max vertex distance from origin |

---

## Usage Example

```go
// Typical flow: the Loader produces a Model from a glTF file.
mdl, err := ldr.Load("assets/models/fox.glb", fragmentShader)
if err != nil {
    log.Fatal(err)
}

// Query model properties
fmt.Println("Name:", mdl.Name())
fmt.Println("Skinned:", mdl.Skinned())
fmt.Println("Animations:", mdl.AnimationNames())
fmt.Println("Bounding radius:", mdl.BoundingRadius())

// Access mesh data
fmt.Println("Vertex bytes:", len(mdl.VertexData()))
fmt.Println("Index count:", mdl.IndexCount())

// Materials are set by the loader during GPU init
for i, mat := range mdl.RenderMaterials() {
    fmt.Printf("Material %d: %s\n", i, mat.Name())
}

// Create a game object from the model
obj := game_object.NewGameObject(
    game_object.WithModel(mdl),
    game_object.WithPosition(0, 0, 0),
    game_object.WithEnabled(true),
)
```

---
