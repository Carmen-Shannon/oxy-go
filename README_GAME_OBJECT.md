# Oxy GameObject System

The `game_object` package defines the scene entity abstraction for the Oxy engine. A `GameObject` binds a `Model` to an `Animator` instance and provides transform access (position, rotation, scale) through the animator's internal arrays, eliminating per-object data duplication.

---

## Table of Contents

- [Overview](#overview)
- [Creating a GameObject](#creating-a-gameobject)
- [Builder Options](#builder-options)
- [GameObject Interface](#gameobject-interface)
  - [Identity & State](#identity--state)
  - [Model & Animator](#model--animator)
  - [Transform Access](#transform-access)
  - [Light Attachment](#light-attachment)
- [Transform Lifecycle](#transform-lifecycle)
- [Usage Example](#usage-example)

---

## Overview

A `GameObject` is the primary scene entity in Oxy. It connects three systems:

1. **Model** — The mesh and material data to render.
2. **Animator** — The GPU-side system that manages per-instance transform arrays and (optionally) skeletal animation.
3. **Light** — An optional attached light whose position is synced from the object's transform each frame by the scene.

Transform data (position, scale, rotation, rotation speed) is not stored directly on the object. Instead, the object holds an `animatorInstanceID` that indexes into the animator's internal arrays. Before an animator is assigned (e.g. during construction), initial transform values are stored locally via builder options and applied when the object is added to a scene.

---

## Creating a GameObject

```go
obj := game_object.NewGameObject(
    game_object.WithID(1),
    game_object.WithModel(myModel),
    game_object.WithPosition(0, 1, 0),
    game_object.WithScale(1, 1, 1),
    game_object.WithRotation(0, 0, 0),
    game_object.WithRotationSpeed(0, 0.5, 0),
    game_object.WithEnabled(true),
)
```

Defaults applied before options:

| Parameter      | Default     |
| -------------- | ----------- |
| ID             | `0`         |
| Enabled        | `false`     |
| Ephemeral      | `false`     |
| Model          | `nil`       |
| Position       | `(0, 0, 0)` |
| Scale          | `(1, 1, 1)` |
| Rotation       | `(0, 0, 0)` |
| Rotation speed | `(0, 0, 0)` |
| Light          | `nil`       |

---

## Builder Options

All options follow the `GameObjectBuilderOption` functional option pattern.

| Option              | Parameters           | Description                                                       |
| ------------------- | -------------------- | ----------------------------------------------------------------- |
| `WithID`            | `id uint64`          | Sets the object's unique identifier                               |
| `WithEnabled`       | `enabled bool`       | Enables or disables the object for rendering                      |
| `WithEphemeral`     | `ephemeral bool`     | Marks the object as ephemeral (not persisted in scene registry)   |
| `WithModel`         | `m model.Model`      | Associates a Model with the object                                |
| `WithPosition`      | `x, y, z float32`    | Sets the initial position (applied when added to a scene)         |
| `WithScale`         | `sx, sy, sz float32` | Sets the initial scale (applied when added to a scene)            |
| `WithRotation`      | `rx, ry, rz float32` | Sets the initial rotation (applied when added to a scene)         |
| `WithRotationSpeed` | `rx, ry, rz float32` | Sets the initial rotation speed (applied when added to a scene)   |
| `WithLight`         | `l light.Light`      | Attaches a light whose position syncs from the object's transform |

---

## GameObject Interface

### Identity & State

| Method                     | Description                                                               |
| -------------------------- | ------------------------------------------------------------------------- |
| `ID() uint64`              | Returns the object's unique identifier                                    |
| `SetID(id uint64)`         | Sets the object's unique identifier                                       |
| `Enabled() bool`           | Returns whether the object is enabled for rendering                       |
| `SetEnabled(enabled bool)` | Enables or disables the object                                            |
| `Ephemeral() bool`         | Returns whether the object is ephemeral (not persisted in scene registry) |

### Model & Animator

| Method                                | Description                                                    |
| ------------------------------------- | -------------------------------------------------------------- |
| `Model() model.Model`                 | Returns the associated Model, or `nil`                         |
| `SetModel(m model.Model)`             | Associates a Model with this object                            |
| `Animator() animator.Animator`        | Returns the associated Animator, or `nil`                      |
| `SetAnimator(anim animator.Animator)` | Sets the Animator for this object                              |
| `AnimatorInstanceID() int`            | Returns the instance index within the Animator (`-1` if unset) |
| `SetAnimatorInstanceID(id int)`       | Sets the instance index within the Animator                    |

### Transform Access

All transform getters read from the Animator's internal arrays via `animatorInstanceID`. If no Animator is assigned, they return the initial values set during construction.

All transform setters write through to the Animator, preserving sibling values (e.g. `SetPosition` preserves the current scale). If no Animator is assigned, they update the local initial values instead.

| Method                                                   | Description                                   |
| -------------------------------------------------------- | --------------------------------------------- |
| `Position() (x, y, z float32)`                           | Returns the instance's current position       |
| `SetPosition(x, y, z float32)`                           | Updates the instance's position               |
| `Scale() (sx, sy, sz float32)`                           | Returns the instance's current scale          |
| `SetScale(sx, sy, sz float32)`                           | Updates the instance's scale                  |
| `Rotation() (rx, ry, rz float32)`                        | Returns the instance's current rotation       |
| `SetRotation(rx, ry, rz float32)`                        | Updates the instance's rotation               |
| `RotationSpeed() (rx, ry, rz float32)`                   | Returns the instance's current rotation speed |
| `SetRotationSpeed(rx, ry, rz float32)`                   | Updates the instance's rotation speed         |
| `TransformData() (pos, scale, rot, rotSpeed [3]float32)` | Reads all transform data in a single call     |

### Light Attachment

| Method                    | Description                                                                                                            |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `Light() light.Light`     | Returns the attached Light, or `nil`                                                                                   |
| `SetLight(l light.Light)` | Attaches a Light (pass `nil` to detach). The scene syncs the light's position from this object's transform each frame. |

---

## Transform Lifecycle

```
Construction (no Animator)          After Scene.Add (Animator assigned)
┌─────────────────────────┐        ┌──────────────────────────────────┐
│ WithPosition(x, y, z)   │        │ SetPosition(x, y, z)            │
│ WithScale(sx, sy, sz)   │  ───►  │   └─► animator.SetInstanceTransform()
│ WithRotation(rx, ry, rz)│        │ Position()                      │
│ Values stored locally    │        │   └─► animator.InstanceTransform()
└─────────────────────────┘        └──────────────────────────────────┘
```

1. **During construction** — Builder options store initial transform values in local fields on the object.
2. **Scene.Add** — The scene assigns an Animator and instance ID, then writes the initial values into the animator's arrays.
3. **At runtime** — All transform getters/setters read from and write through to the Animator. The local initial values are no longer used.

---

## Usage Example

```go
package main

import (
    "github.com/Carmen-Shannon/oxy-go/engine/game_object"
    "github.com/Carmen-Shannon/oxy-go/engine/light"
    "github.com/Carmen-Shannon/oxy-go/engine/model"
)

func main() {
    // Load a model (simplified)
    mdl := model.NewModel( /* ... */ )

    // Create a point light to attach
    pointLight := light.NewLight( /* ... */ )

    // Create the game object
    obj := game_object.NewGameObject(
        game_object.WithID(1),
        game_object.WithEnabled(true),
        game_object.WithModel(mdl),
        game_object.WithPosition(0, 5, 0),
        game_object.WithScale(2, 2, 2),
        game_object.WithRotationSpeed(0, 0.5, 0),
        game_object.WithLight(pointLight),
    )

    // After adding to a scene, transforms read from the Animator:
    // scene.Add(obj)

    // Query transform data
    x, y, z := obj.Position()
    sx, sy, sz := obj.Scale()
    _, _, _ = x, y, z
    _, _, _ = sx, sy, sz

    // Modify at runtime (writes through to Animator if assigned)
    obj.SetPosition(10, 5, 0)
    obj.SetRotationSpeed(0, 1.0, 0)
}
```
