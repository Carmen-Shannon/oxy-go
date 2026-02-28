# Bind Group Provider

The `engine/renderer/bind_group_provider` package provides the abstraction layer between engine components and their GPU-side bind group resources. Every entity that needs GPU bindings — cameras, game objects, lights, materials, animators — holds a `BindGroupProvider` to describe its resource requirements. The Renderer then uses the provider to create, update, and release the underlying WebGPU objects.

**Package path:** `github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider`

---

## Architecture

```
BindGroupProvider (public interface)
 ├─ common.Delegate[BindGroupProvider]   ← mock / test delegation
 └─ bindGroupProvider (unexported struct)
      └─ common.DelegateImpl[BindGroupProvider]
```

The package follows the standard oxy-go interface-first pattern: a single public `BindGroupProvider` interface backed by an unexported `bindGroupProvider` struct with a compile-time implementation check. The interface embeds `common.Delegate[BindGroupProvider]` for mock/test delegation support.

---

## Usage Pattern

```
1. Component creates a BindGroupProvider with a label and optional builder options
2. Component stores the provider via its SetBindGroupProvider() method
3. Scene/Renderer calls Renderer.InitBindGroup(provider, ...) to create GPU resources
4. Scene/Renderer calls Renderer.WriteBuffers([]BufferWrite{...}) to update uniforms
5. Component reads BindGroup() during draw calls for shader binding
6. Component calls Release() on shutdown to free GPU memory
```

The provider itself does **not** allocate GPU resources — it is a data holder. Allocation is performed by the `Renderer` when `InitBindGroup`, `InitMeshBuffers`, `InitTextureView`, or `InitSampler` is called.

---

## Resource Categories

A `BindGroupProvider` manages three categories of GPU resources:

### Bind Group Resources

| Field / Method                                            | Description                                         |
| --------------------------------------------------------- | --------------------------------------------------- |
| `BindGroup() *wgpu.BindGroup`                             | The WebGPU bind group created for shader binding    |
| `BindGroupLayout() *wgpu.BindGroupLayout`                 | The bind group layout used to create the bind group |
| `Buffer(binding int) *wgpu.Buffer`                        | A GPU buffer at the given binding index             |
| `Buffers() map[int]*wgpu.Buffer`                          | All buffers keyed by binding index                  |
| `TextureView(binding int) *wgpu.TextureView`              | A GPU texture view at the given binding index       |
| `TextureViews() map[int]*wgpu.TextureView`                | All texture views keyed by binding index            |
| `Sampler(binding int) *wgpu.Sampler`                      | A GPU sampler at the given binding index            |
| `Samplers() map[int]*wgpu.Sampler`                        | All samplers keyed by binding index                 |
| `SetBindGroup(bg *wgpu.BindGroup)`                        | Updates the bind group                              |
| `SetBindGroupLayout(bgl *wgpu.BindGroupLayout)`           | Updates the bind group layout                       |
| `SetBuffer(binding int, buf *wgpu.Buffer)`                | Sets a single buffer at a binding index             |
| `SetBuffers(buffers map[int]*wgpu.Buffer)`                | Replaces the entire buffer map                      |
| `SetTextureView(binding int, tv *wgpu.TextureView)`       | Sets a single texture view at a binding index       |
| `SetTextureViews(textureViews map[int]*wgpu.TextureView)` | Replaces the entire texture view map                |
| `SetSampler(binding int, s *wgpu.Sampler)`                | Sets a single sampler at a binding index            |
| `SetSamplers(samplers map[int]*wgpu.Sampler)`             | Replaces the entire sampler map                     |

### Vertex Pulling Resources

| Field / Method                      | Description                                            |
| ----------------------------------- | ------------------------------------------------------ |
| `VertexBuffer() *wgpu.Buffer`       | GPU vertex buffer for mesh vertex data                 |
| `IndexBuffer() *wgpu.Buffer`        | GPU index buffer for mesh index data                   |
| `IndexCount() int`                  | Number of indices, used by the Renderer for draw calls |
| `SetVertexBuffer(buf *wgpu.Buffer)` | Updates the vertex buffer                              |
| `SetIndexBuffer(buf *wgpu.Buffer)`  | Updates the index buffer                               |
| `SetIndexCount(count int)`          | Updates the index count                                |

### Metadata & Lifecycle

| Field / Method   | Description                                   |
| ---------------- | --------------------------------------------- |
| `Label() string` | Debug label for profiling and diagnostics     |
| `Release()`      | Frees all GPU resources held by this provider |

---

## Builder Options

The `NewBindGroupProvider` constructor accepts a required label and variadic `BindGroupProviderOption` functions:

| Option                         | Description                                             |
| ------------------------------ | ------------------------------------------------------- |
| `WithBindGroup(bg)`            | Pre-sets the bind group (typically set by the Renderer) |
| `WithBindGroupLayout(bgl)`     | Pre-sets the bind group layout                          |
| `WithBuffer(binding, buf)`     | Sets a single buffer at a specific binding index        |
| `WithBuffers(map)`             | Replaces the entire buffer map                          |
| `WithTextureView(binding, tv)` | Sets a single texture view at a binding index           |
| `WithTextureViews(map)`        | Replaces the entire texture view map                    |
| `WithSampler(binding, s)`      | Sets a single sampler at a binding index                |
| `WithSamplers(map)`            | Replaces the entire sampler map                         |
| `WithVertexBuffer(buf)`        | Pre-sets the vertex buffer                              |
| `WithIndexBuffer(buf)`         | Pre-sets the index buffer                               |
| `WithIndexCount(count)`        | Pre-sets the index count                                |

---

## Constructor

```go
func NewBindGroupProvider(label string, options ...BindGroupProviderOption) BindGroupProvider
```

Creates a new `BindGroupProvider` with initialized empty maps for buffers, texture views, and samplers. Sets `p.Delegate = p` so delegation routes to itself by default. Builder options are applied after map initialization.

---

## BufferWrite

The `BufferWrite` struct describes a single GPU buffer write operation. It is consumed by `Renderer.WriteBuffers()` to batch-upload uniform and storage buffer data each frame.

```go
type BufferWrite struct {
    Provider BindGroupProvider  // target provider
    Binding  int                // binding index within the provider
    Offset   uint64             // byte offset into the buffer
    Data     []byte             // raw bytes to write
}
```

---

## Release

Calling `Release()` on a provider frees all GPU resources in the following order:

1. Texture views (per-binding)
2. Samplers (per-binding)
3. Buffers (per-binding)
4. Bind group
5. Bind group layout
6. Vertex buffer
7. Index buffer

Each resource is released and removed from its map/nil'd to prevent double-free.

---

## Files

| File                             | Purpose                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------- |
| `bind_group_provider.go`         | `BindGroupProvider` interface, `bindGroupProvider` struct, constructor, impls |
| `bind_group_provider_builder.go` | `BindGroupProviderOption` type and builder functions                          |
| `buffer_write.go`                | `BufferWrite` struct for batched GPU buffer writes                            |
