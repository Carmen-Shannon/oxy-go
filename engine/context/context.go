package context

import (
	"maps"

	"github.com/Carmen-Shannon/oxy-go/engine/scene"
)

type Context interface {
	// Scene returns the scene associated with the given key, nil if no scene exists for the key specified
	//
	// Parameters:
	//   - key: The unique identifier for the scene to be retrieved.
	//
	// Returns:
	//   - The scene associated with the provided key, or nil if no such scene exists.
	//   - A boolean indicating whether the scene was found in the context (true) or not (false).
	Scene(key int) (scene.Scene, bool)

	// Scenes returns a map of all scenes currently registered in the context, where the keys are the unique identifiers for the scenes and the values are the corresponding scene instances.
	//
	// Returns:
	//   - A map containing all registered scenes in the context, with scene keys as integers and scene instances as values.
	Scenes() map[int]scene.Scene

	// SetScenes replaces the current scenes map in the context with the provided scenes map.
	// This should never really be called, it is automatically managed via AddScene and RemoveScene on the engine interface.
	//
	// Parameters:
	//   - scenes: A map of scene keys to scene instances that will replace the current scenes in the context.
	SetScenes(scenes map[int]scene.Scene)

	// Get retrieves a value from the context associated with the given key.
	// If the value does not exist within the context, it returns nil and false.
	// It is the responsibility of the caller to set and assert the correct type for the retrieved value.
	//
	// Parameters:
	//   - key: The unique identifier for the value to be retrieved from the context.
	//
	// Returns:
	//   - The value associated with the provided key, or nil if no such value exists.
	//   - A boolean indicating whether the value was found in the context (true) or not (false).
	Get(key string) (any, bool)

	// Set stores a value in the context associated with the given key.
	//
	// Parameters:
	//   - key: The unique identifier for the value to be stored in the context.
	//   - value: The value to be stored in the context, which can be of any type.
	Set(key string, value any)
}

func (c *context) Scene(key int) (scene.Scene, bool) {
	c.sceneMu.Lock()
	defer c.sceneMu.Unlock()
	scene, exists := c.scenes[key]
	return scene, exists
}

func (c *context) Scenes() map[int]scene.Scene {
	c.sceneMu.Lock()
	defer c.sceneMu.Unlock()
	return c.scenes
}

func (c *context) SetScenes(scenes map[int]scene.Scene) {
	c.sceneMu.Lock()
	defer c.sceneMu.Unlock()
	copied := make(map[int]scene.Scene, len(scenes))
	maps.Copy(copied, scenes)
	c.scenes = copied
}

func (c *context) Get(key string) (any, bool) {
	c.valuesMu.Lock()
	defer c.valuesMu.Unlock()
	value, exists := c.values[key]
	return value, exists
}

func (c *context) Set(key string, value any) {
	c.valuesMu.Lock()
	defer c.valuesMu.Unlock()
	c.values[key] = value
}
