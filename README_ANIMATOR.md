# Animator System

The `engine/renderer/animator` package implements the GPU compute animation system. It provides two pluggable backends — **simple** (per-instance position/rotation/scale) and **skeletal** (bone-based animation with clip blending) — behind a single `Animator` interface. Each frame, the animator stages dirty instance data into GPU buffer writes that are dispatched via a compute shader to produce final transforms for the vertex shader.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/renderer/animator`

**Parent package:** [`engine/renderer`](README_RENDERER.md)

---

## Architecture

```
Animator (public interface)
 └─ animator (unexported struct)
      ├── backendType   — BackendTypeSimple or BackendTypeSkeletal
      ├── backend       — AnimatorBackend (union interface)
      │    ├── simpleAnimatorBackendImpl
      │    └── skeletalAnimatorBackendImpl
      └── model         — associated Model (skeleton + animations)
```

`AnimatorBackend` is a union interface embedding both `simpleAnimatorBackend` and `skeletalAnimatorBackend`. Each concrete implementation provides the full method set — methods that don't apply to a given backend type are implemented as no-ops.

---

## Backend Types

| Constant              | Value | Description                                                                                                                                    |
| --------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `BackendTypeSimple`   | `0`   | Per-instance position, rotation, and scale driven by a compute shader. Uses sparse dirty tracking with a bitset for efficient partial uploads. |
| `BackendTypeSkeletal` | `1`   | Bone-based animation with per-instance playback state, clip blending, and packed animation data. Uses contiguous dirty range tracking.         |

---

## Constructor

```go
func NewAnimator(
    backendType AnimatorBackendType,
    options ...AnimatorBuilderOption,
) Animator
```

Creates a new Animator with the specified backend. The backend is instantiated first, then builder options are applied.

---

## Builder Options

| Option                                     | Description                                                                |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| `WithMaxInstances(maxInstances)`           | Sets the maximum number of instances the animator can manage.              |
| `WithModel(m, boneBinding, packedBinding)` | Assigns a Model and flattens its skeleton/animation data into the backend. |

---

## Animator Interface

### Instance Management

| Method                                 | Description                                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `AddInstance() (uint32, error)`        | Registers a new instance. Auto-grows capacity if full. Returns the instance index.                |
| `RemoveInstance(index) (uint32, bool)` | Swap-removes an instance. Returns the swapped index and whether a swap occurred. _(Experimental)_ |
| `InstanceCount() uint32`               | Number of active instances.                                                                       |
| `MaxInstances() uint32`                | Current capacity.                                                                                 |
| `Grow(newMax)`                         | Increases capacity, preserving existing data. Sets `needsRebuild`. _(Experimental)_               |
| `NeedsRebuild() bool`                  | Whether GPU buffers need recreation after a `Grow`. _(Experimental)_                              |
| `ClearNeedsRebuild()`                  | Resets the rebuild flag after GPU resources are recreated. _(Experimental)_                       |

### Transform (Simple Backend)

These methods no-op on skeletal backends:

| Method                                                          | Description                                                        |
| --------------------------------------------------------------- | ------------------------------------------------------------------ |
| `SetInstanceTransform(index, posXYZ, scaleXYZ)`                 | Sets position and scale for an instance.                           |
| `SetInstanceRotation(index, rotSpeedXYZ, rotXYZ)`               | Sets rotation speed and current rotation.                          |
| `SetInstanceData(index, posXYZ, scaleXYZ, rotSpeedXYZ, rotXYZ)` | Sets all transform data in a single call (reduced mutex overhead). |
| `InstanceTransform(index) (pos, scale)`                         | Returns position and scale.                                        |
| `InstanceRotation(index) (rotSpeed, rot)`                       | Returns rotation speed and current rotation.                       |

### Skeletal Animation

These methods no-op on simple backends:

| Method                                                                                                                                | Description                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `SetBoneCount(count)`                                                                                                                 | Allocates the bone slice. Must be called before `SetBone`.  |
| `SetBone(index, inverseBindMatrix, localTranslation, localRotation, localScale, parentIndex, binding)`                                | Sets bone data at the given skeleton index.                 |
| `AddClip(duration, ticksPerSecond, channels, keyframeTimes, keyframeTranslations, keyframeRotations, keyframeScales, binding) uint32` | Adds a flattened animation clip. Returns the clip index.    |
| `PlayAnimation(instanceIndex, clipIndex, loop)`                                                                                       | Starts playback of a clip on an instance.                   |
| `BlendToAnimation(instanceIndex, targetClipIndex, blendDuration)`                                                                     | Smoothly transitions to a new clip over the given duration. |
| `SetAnimationTime(instanceIndex, time)`                                                                                               | Sets the playback position.                                 |
| `SetAnimationSpeed(instanceIndex, speed)`                                                                                             | Sets the speed multiplier (1.0 = normal).                   |
| `IsBlending(instanceIndex) bool`                                                                                                      | Whether an instance is currently blending.                  |
| `BlendProgress(instanceIndex) float32`                                                                                                | Blend progress from 0.0 to 1.0.                             |
| `CancelBlend(instanceIndex)`                                                                                                          | Stops an in-progress blend.                                 |

### Model

| Method                                    | Description                                                                                        |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `Model() Model`                           | Returns the associated Model, or `nil`.                                                            |
| `SetModel(m, boneBinding, packedBinding)` | Assigns a Model. For skinned models, flattens skeleton bones and animation clips into the backend. |

### Frustum Culling

| Method                                   | Description                                                                    |
| ---------------------------------------- | ------------------------------------------------------------------------------ |
| `SetFrustumPlanes(planes)`               | Updates the six frustum planes for GPU culling. Enables culling on first call. |
| `SetBoundingRadius(radius)`              | Sets the object-space bounding sphere radius.                                  |
| `BoundingRadius() float32`               | Returns the current bounding radius.                                           |
| `CullingEnabled() bool`                  | Whether culling is active.                                                     |
| `IndirectBuffer(binding) *wgpu.Buffer`   | Returns the GPU indirect draw arguments buffer, or `nil`.                      |
| `ResetIndirectArgs(indexCount, binding)` | Zeros the indirect args instance count before each compute dispatch.           |

### Frame Lifecycle

| Method                                                     | Description                                                                                                                                   |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `PrepareFrame(deltaTime, binding)`                         | Advances animation state, stages per-frame uniform data. For skeletal backends, advances playback time, handles looping and blend resolution. |
| `Flush(instanceBinding, boneBinding, modelBinding) uint32` | Stages dirty instance data as GPU buffer writes. Returns the number of instances flushed.                                                     |
| `StagedWriteData() []BufferWrite`                          | Returns and clears pending GPU buffer writes for the Renderer to submit.                                                                      |
| `Release()`                                                | Frees all GPU resources held by this animator.                                                                                                |

### Metadata

| Method                              | Description               |
| ----------------------------------- | ------------------------- |
| `BackendType() AnimatorBackendType` | Returns the backend type. |

---

## GPU Types

All GPU structs are std430-aligned and have embedded WGSL source files loaded via `//go:embed`. Each type implements `Size() int` and `Marshal() []byte`.

| Go Type                    | WGSL Type               | Size  | Backend  | Description                                                    |
| -------------------------- | ----------------------- | ----- | -------- | -------------------------------------------------------------- |
| `GPUInstanceData`          | `InstanceData`          | 64 B  | Output   | Per-instance 4×4 model matrix (compute output).                |
| `GPUAnimationData`         | `AnimationData`         | 64 B  | Simple   | Per-instance rotation, position, scale (compute input).        |
| `GPUSkeletalAnimationData` | `SkeletalAnimationData` | 48 B  | Skeletal | Per-instance clip index, time, blend weight.                   |
| `GPUAnimationGlobals`      | `AnimationGlobals`      | 128 B | Skeletal | Per-frame uniform: counts, offsets, frustum planes.            |
| `GPUGlobalData`            | `GlobalData`            | 112 B | Simple   | Per-frame uniform: instance count, delta time, frustum planes. |
| `GPUFrustumPlane`          | `FrustumPlane`          | 16 B  | Both     | Single frustum plane (normal + distance).                      |
| `GPUIndirectArgs`          | `IndirectArgs`          | 20 B  | Both     | DrawIndexedIndirect arguments written by compute shader.       |
| `GPUBoneInfo`              | `BoneInfo`              | 112 B | Skeletal | Inverse bind matrix, local transform, parent index.            |
| `GPUKeyFrame`              | —                       | 64 B  | Skeletal | Time, translation, rotation, scale per keyframe.               |
| `GPUChannelHeader`         | —                       | 32 B  | Skeletal | Bone index + keyframe offsets/counts per channel.              |
| `GPUClipHeader`            | —                       | 16 B  | Skeletal | Duration, ticks/sec, channel offset/count per clip.            |

---

## Dirty Tracking

### Simple Backend

Uses **sparse bitset tracking**. Each mutated instance index is enqueued into `dirtyIndices` with O(1) dedup via a `uint64` bitset. On `Flush`, indices are sorted (insertion sort) and coalesced into contiguous GPU buffer writes to minimize write commands.

### Skeletal Backend

Uses **contiguous dirty range tracking** (`dirtyStart`/`dirtyEnd`). All instances in the dirty range are uploaded in a single write. Separate dirty flags exist for instance data, bone data, and model matrices.

---

## Packed Animation Buffer (Skeletal)

The skeletal backend packs all animation data into a single `array<u32>` GPU buffer:

```
[ clip headers (4 u32 each) ][ channel headers (8 u32 each) ][ keyframes (16 u32 each) ]
```

Offsets (`channelDataOffset`, `keyframeDataOffset`) are stored in the per-frame `AnimationGlobals` uniform so the compute shader can index into each section.

---

## WGSL Assets

| Asset File                     | Embedded Variable                | WGSL Struct             |
| ------------------------------ | -------------------------------- | ----------------------- |
| `animation_data.wgsl`          | `GPUAnimationDataSource`         | `AnimationData`         |
| `animation_globals.wgsl`       | `GPUAnimationGlobalsSource`      | `AnimationGlobals`      |
| `bone_info.wgsl`               | `GPUBoneInfoSource`              | `BoneInfo`              |
| `frustum_plane.wgsl`           | `GPUFrustumPlaneSource`          | `FrustumPlane`          |
| `indirect_args.wgsl`           | `GPUIndirectArgsSource`          | `IndirectArgs`          |
| `instance_data.wgsl`           | `GPUInstanceDataSource`          | `InstanceData`          |
| `simple_globals.wgsl`          | `GPUGlobalDataSource`            | `GlobalData`            |
| `skeletal_animation_data.wgsl` | `GPUSkeletalAnimationDataSource` | `SkeletalAnimationData` |

---

## Files

| File                           | Purpose                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `animator.go`                  | `Animator` interface, `animator` struct, `NewAnimator` constructor, all delegation methods                                      |
| `animator_backend.go`          | `AnimatorBackendType` enum, `AnimatorBackend` union interface                                                                   |
| `animator_builder.go`          | `AnimatorBuilderOption` type and builder functions                                                                              |
| `gpu_types.go`                 | All GPU-aligned structs with `Size()`, `Marshal()`, and embedded WGSL sources                                                   |
| `simple_animator_backend.go`   | `simpleAnimatorBackend` interface + `simpleAnimatorBackendImpl` (sparse dirty tracking, transform staging)                      |
| `skeletal_animator_backend.go` | `skeletalAnimatorBackend` interface + `skeletalAnimatorBackendImpl` (bone data, clip storage, blend transitions, packed buffer) |
| `assets/`                      | 8 embedded `.wgsl` struct definition files                                                                                      |
