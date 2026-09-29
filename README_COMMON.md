# Common Package

The `common` package provides shared types, math utilities, and constants used throughout the oxy-go engine. It contains plain structs and standalone functions — no interface-wrapped systems — serving as the foundation that every other package imports.

**Package path:** `github.com/Carmen-Shannon/oxy-go/common`

---

## Files

| File                                    | Purpose                                                                                                                         |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `delegate.go`                           | Generic delegation interface and embeddable implementation for mock/test routing                                                |
| `frustum.go`                            | View frustum representation and plane extraction for culling                                                                    |
| `key_codes.go`                          | Cross-platform virtual key codes matching GLFW                                                                                  |
| `math.go`                               | 4×4 matrix math, projection, view, model transforms, and unsafe byte conversions                                                |
| `readfile_native.go` / `readfile_js.go` | Platform file-read abstraction: `ReadFile` uses the native filesystem on desktop builds and the browser Fetch API under GOOS=js |
| `types.go`                              | Staging data structs for textures, samplers, and imported materials from model files                                            |
| `utils.go`                              | Generic utility functions                                                                                                       |

---

## Frustum Culling (`frustum.go`)

Provides a `Frustum` struct containing six `Plane` values representing the view frustum. Planes are extracted from a combined view-projection matrix using the Gribb/Hartmann method.

### Types

| Type      | Description                                                   |
| --------- | ------------------------------------------------------------- |
| `Plane`   | A plane in 3D space: normal `[3]float32` + distance `float32` |
| `Frustum` | Six indexed planes: Left, Right, Bottom, Top, Near, Far       |

### Constants

| Constant        | Value | Description                |
| --------------- | ----- | -------------------------- |
| `FrustumLeft`   | 0     | Left frustum plane index   |
| `FrustumRight`  | 1     | Right frustum plane index  |
| `FrustumBottom` | 2     | Bottom frustum plane index |
| `FrustumTop`    | 3     | Top frustum plane index    |
| `FrustumNear`   | 4     | Near frustum plane index   |
| `FrustumFar`    | 5     | Far frustum plane index    |

### Functions

| Function                     | Description                                                              |
| ---------------------------- | ------------------------------------------------------------------------ |
| `ExtractFrustumFromMatrix()` | Extracts and normalizes six frustum planes from a column-major VP matrix |

### Methods

| Method              | Description                                                                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `IntersectSphere()` | Returns `true` if a sphere intersects or is inside the frustum; uses signed plane distances; conservative (may return `true` for edge cases)                                   |
| `IntersectAABB()`   | Returns `true` if an axis-aligned bounding box intersects or is inside the frustum; uses the positive-vertex method per plane; conservative (may return `true` for edge cases) |

**Reference:** [Gribb/Hartmann plane extraction (PDF)](https://www8.cs.umu.se/kurser/5DV051/HT12/lab/plane_extraction.pdf)

---

## Delegation (`delegate.go`)

Provides a generic delegation pattern for routing method calls through a replaceable instance. In production code the delegate is set to the instance itself during construction; in tests it can be swapped for a mock.

### Types

| Type              | Description                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------- |
| `Delegate[T any]` | Interface defining `SetDelegate(delegate T)` for embedding delegation support             |
| `DelegateImpl[T]` | Embeddable struct implementing `Delegate[T]` with a public `Delegate T` field for routing |

---

## Key Codes (`key_codes.go`)

Platform-independent virtual key constants matching [GLFW key codes](https://pkg.go.dev/github.com/go-gl/glfw/v3.4/glfw#Key). Printable keys use their ASCII values; special keys use GLFW-assigned values.

### Printable Keys

`KeyA`–`KeyZ` (all 26 letters, 65–90), `KeySpace` (32), `Key0`–`Key9` (48–57).

### Special Keys

| Constant         | Value   | Description                 |
| ---------------- | ------- | --------------------------- |
| `KeyEsc`         | 256     | Escape (GLFW)               |
| `KeyTab`         | 258     | Tab (GLFW)                  |
| `KeyBackspace`   | 259     | Backspace (GLFW)            |
| `KeyRight`       | 262     | Right Arrow (GLFW)          |
| `KeyLeft`        | 263     | Left Arrow (GLFW)           |
| `KeyDown`        | 264     | Down Arrow (GLFW)           |
| `KeyUp`          | 265     | Up Arrow (GLFW)             |
| `KeyF1`–`KeyF12` | 290–301 | Function keys F1–F12 (GLFW) |
| `KeyLeftShift`   | 340     | Left Shift                  |
| `KeyLeftCtrl`    | 341     | Left Ctrl (GLFW)            |
| `KeyLeftAlt`     | 342     | Left Alt (GLFW)             |
| `KeyRightShift`  | 344     | Right Shift                 |
| `KeyRightCtrl`   | 345     | Right Ctrl (GLFW)           |
| `KeyRightAlt`    | 346     | Right Alt (GLFW)            |

---

## Math Utilities (`math.go`)

All matrix operations use **column-major** layout (OpenGL/WebGPU convention). Matrices are represented as `[]float32` of length 16.

### Matrix Functions

| Function             | Description                                                                                                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `Identity()`         | Resets a 4×4 slice to the identity matrix                                                                         |
| `Mul4()`             | Multiplies two 4×4 matrices: `out = a * b`                                                                        |
| `Perspective()`      | Builds a perspective projection matrix (WebGPU clip space `[0, 1]`)                                               |
| `BuildModelMatrix()` | Constructs a model matrix from position, Euler rotation (Y×X×Z), and scale                                        |
| `Invert4()`          | Computes the cofactor-based inverse of a 4×4 matrix; returns `false` if singular                                  |
| `Invert3x3()`        | Computes the inverse of a 3×3 row-major matrix (`[9]float32`); returns the inverted array and `false` if singular |
| `LookAt()`           | Builds a view matrix from eye position, target point, and up vector                                               |

### Byte Conversion Functions

| Function          | Description                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| `SliceToBytes()`  | Reinterprets any typed slice as `[]byte` via `unsafe` for GPU uploads  |
| `StructToBytes()` | Reinterprets a struct pointer as `[]byte` via `unsafe` for GPU uploads |

> **Warning:** Both functions return views into the original memory — the caller must not modify the returned bytes.

### Scalar & Grid Functions

| Function         | Description                                                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `ClampInt()`     | Constrains `v` to the range `[lo, hi]`                                                                                      |
| `NormalizeF64()` | Normalizes a 3-component float64 vector, returned as float32; returns a zero vector if the input length is effectively zero |
| `CellKey()`      | Computes the flattened grid-cell index for a 3D position within a uniform grid; clamps each axis to `[0, gridRes-1]`        |

---

## Platform File Reads (`readfile_native.go` / `readfile_js.go`)

Platform file-read seam: one exported `ReadFile` function with two build-tagged implementations, so engine code reads files identically on desktop and in the browser.

### Functions

| Function                                | Description                                                             |
| --------------------------------------- | ----------------------------------------------------------------------- |
| `ReadFile(path string) ([]byte, error)` | Reads the file (or fetched resource) at `path` and returns its contents |

The engine's loader calls `ReadFile` at its three filesystem touch points — model files, external glTF buffers, and texture images — on both native and browser/WASM builds. On native builds the implementation delegates directly to `os.ReadFile`. Under GOOS=js there is no local filesystem in a browser tab, so the implementation reads through the browser Fetch API; fetch URLs resolve relative to the page origin, so assets must be served by the hosting HTTP server.

> **Use `common.ReadFile` instead of `os.ReadFile`** anywhere files are read in engine code. It centralizes platform-divergent file reads in a single seam, keeps native behavior identical (`ReadFile` IS `os.ReadFile` on desktop builds), and is required for the engine's browser/WASM target where `os.ReadFile` cannot work.

---

## Staging & Import Types (`types.go`)

Plain structs used to shuttle data between the loader, material, and bind group provider systems.

### Types

| Type                 | Description                                                                                                                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TextureStagingData` | RGBA pixel data (`[]byte`) + width/height + `Linear` flag, staged for GPU texture upload. When `Linear` is true, uses `RGBA8Unorm` instead of `RGBA8UnormSrgb` (for non-color data like normal maps). |
| `SamplerStagingData` | Sampler configuration (address modes, filter modes, LOD clamps, anisotropy, compare function)                                                                                                         |
| `ImportedMaterial`   | Material properties from a model file: name, base color, metallic, roughness, alpha mode, alpha cutoff, and texture paths/data                                                                        |
| `ImportedTexture`    | Texture data from a model file: name, embedded bytes or file path, MIME type, optional sampler override. `Width` and `Height` fields are populated after calling `Decode()`.                          |

### Methods

| Method                     | Description                                                                 |
| -------------------------- | --------------------------------------------------------------------------- |
| `ImportedTexture.Decode()` | Decodes embedded or file-based PNG/JPEG to raw RGBA pixels + width + height |

---

## Generic Utilities (`utils.go`)

| Function     | Description                                                                |
| ------------ | -------------------------------------------------------------------------- |
| `Coalesce()` | Returns the first non-zero value from a variadic list of comparable values |
| `ToPtr()`    | Returns a pointer to the given value of any type `T`                       |

---

## Usage Examples

### Frustum extraction

```go
vp := make([]float32, 16)
common.Mul4(vp, viewMatrix, projMatrix)
frustum := common.ExtractFrustumFromMatrix(vp)
```

### GPU buffer upload

```go
vertices := []MyVertex{ /* ... */ }
data := common.SliceToBytes(vertices)
queue.WriteBuffer(buffer, 0, data)
```

### Model matrix

```go
model := make([]float32, 16)
common.BuildModelMatrix(model,
    0, 1, 0,       // position
    0, 3.14, 0,    // rotation (radians)
    1, 1, 1,        // scale
)
```

### Platform file read

```go
data, err := common.ReadFile("assets/model.gltf")
if err != nil {
    return err
}
```
