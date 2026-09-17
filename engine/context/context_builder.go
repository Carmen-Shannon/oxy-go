package context

import (
	"sync"

	"github.com/Carmen-Shannon/oxy-go/engine/scene"
)

type ContextBuilderOption func(*context)

// WithScenes sets the scenes map for the underlying Context instance.
//
// Parameters:
//   - scenes: A map of scene keys to scene instances that will be used to initialize the Context's scenes.
//
// Returns:
//   - A ContextBuilderOption that can be passed to NewContext to set the initial scenes for the Context.
func WithScenes(scenes map[int]scene.Scene) ContextBuilderOption {
	return func(c *context) {
		c.scenes = scenes
	}
}

// NewContext creates a new Context instance with the provided options.
// Scenes should be provided after they are added to the engine.
//
// Parameters:
//   - options: A variadic list of ContextBuilderOption functions that can be used to configure the Context instance.
//
// Returns:
//   - A new Context instance configured according to the provided options.
func NewContext(options ...ContextBuilderOption) Context {
	c := &context{
		sceneMu:  &sync.Mutex{},
		valuesMu: &sync.Mutex{},
		scenes:   make(map[int]scene.Scene),
		values:   make(map[string]any),
	}
	for _, opt := range options {
		opt(c)
	}
	return c
}
