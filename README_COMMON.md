# Common Package

The `common` package provides shared types, math utilities, and constants used throughout the oxy-go engine. It contains plain structs and standalone functions — no interface-wrapped systems — serving as the foundation that every other package imports.

---

## Files

| File           | Purpose                                                                              |
| -------------- | ------------------------------------------------------------------------------------ |
| `frustum.go`   | View frustum representation and plane extraction for culling                         |
| `key_codes.go` | Cross-platform virtual key codes matching GLFW                                       |
| `math.go`      | 4×4 matrix math, projection, view, model transforms, and unsafe byte conversions     |
| `types.go`     | Staging data structs for textures, samplers, and imported materials from model files |
| `utils.go`     | Generic utility functions                                                            |

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

**Reference:** [Gribb/Hartmann plane extraction (PDF)](https://www8.cs.umu.se/kurser/5DV051/HT12/lab/plane_extraction.pdf)

---

## Key Codes (`key_codes.go`)

Platform-independent virtual key constants matching [GLFW key codes](https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw#Key). Printable keys use their ASCII values; special keys use GLFW-assigned values.

### Printable Keys

`KeyA` (65) through `KeyX` (88), `KeySpace` (32), `Key0`–`Key9` (48–57).

### Special Keys

| Constant        | Value | Description      |
| --------------- | ----- | ---------------- |
| `KeyBackspace`  | 259   | Backspace (GLFW) |
| `KeyEsc`        | 256   | Escape (GLFW)    |
| `KeyLeftShift`  | 340   | Left Shift       |
| `KeyRightShift` | 344   | Right Shift      |

---

## Math Utilities (`math.go`)

All matrix operations use **column-major** layout (OpenGL/WebGPU convention). Matrices are represented as `[]float32` of length 16.

### Matrix Functions

| Function             | Description                                                                      |
| -------------------- | -------------------------------------------------------------------------------- |
| `Identity()`         | Resets a 4×4 slice to the identity matrix                                        |
| `Mul4()`             | Multiplies two 4×4 matrices: `out = a * b`                                       |
| `Perspective()`      | Builds a perspective projection matrix (WebGPU clip space `[0, 1]`)              |
| `BuildModelMatrix()` | Constructs a model matrix from position, Euler rotation (Y×X×Z), and scale       |
| `Invert4()`          | Computes the cofactor-based inverse of a 4×4 matrix; returns `false` if singular |
| `LookAt()`           | Builds a view matrix from eye position, target point, and up vector              |

### Byte Conversion Functions

| Function          | Description                                                            |
| ----------------- | ---------------------------------------------------------------------- |
| `SliceToBytes()`  | Reinterprets any typed slice as `[]byte` via `unsafe` for GPU uploads  |
| `StructToBytes()` | Reinterprets a struct pointer as `[]byte` via `unsafe` for GPU uploads |

> **Warning:** Both functions return views into the original memory — the caller must not modify the returned bytes.

---

## Staging & Import Types (`types.go`)

Plain structs used to shuttle data between the loader, material, and bind group provider systems.

### Types

| Type                 | Description                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| `TextureStagingData` | RGBA pixel data (`[]byte`) + width/height, staged for GPU texture upload                          |
| `SamplerStagingData` | Sampler configuration (address modes, filter modes, LOD clamps, anisotropy, compare function)     |
| `ImportedMaterial`   | Material properties from a model file: base color, metallic, roughness, texture paths/data        |
| `ImportedTexture`    | Texture data from a model file: embedded bytes or file path, MIME type, optional sampler override |

### Methods

| Method                     | Description                                                                 |
| -------------------------- | --------------------------------------------------------------------------- |
| `ImportedTexture.Decode()` | Decodes embedded or file-based PNG/JPEG to raw RGBA pixels + width + height |

---

## Generic Utilities (`utils.go`)

| Function     | Description                                                                |
| ------------ | -------------------------------------------------------------------------- |
| `Coalesce()` | Returns the first non-zero value from a variadic list of comparable values |

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
