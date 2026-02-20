package loader

import (
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
)

// LoaderBuilderOption is a functional option for configuring a Loader via NewLoader.
type LoaderBuilderOption func(*loader)

// WithRenderer is an option builder that sets the Renderer used by the Loader.
//
// Parameters:
//   - r: the renderer instance
//
// Returns:
//   - LoaderBuilderOption: a function that applies the renderer option to a loader
func WithRenderer(r renderer.Renderer) LoaderBuilderOption {
	return func(l *loader) {
		l.renderer = r
	}
}

// WithModel is an option builder that pre-populates the model cache with a model.
//
// Parameters:
//   - key: the cache key for the model
//   - model: the model to cache
//
// Returns:
//   - LoaderBuilderOption: a function that applies the model option to a loader
func WithModel(key string, model model.Model) LoaderBuilderOption {
	return func(l *loader) {
		l.modelCache[key] = model
	}
}
