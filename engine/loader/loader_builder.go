package loader

import (
	"sync"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// LoaderBuilderOption is a functional option for configuring a Loader via NewLoader.
type LoaderBuilderOption func(*loader)

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

// NewLoader creates a new Loader instance with the specified backend type and options applied.
//
// Parameters:
//   - backendType: the type of loader backend to use (e.g., BackendTypeGLTF)
//   - options: a variadic list of LoaderBuilderOption functions to configure the Loader
//
// Returns:
//   - Loader: a new instance of Loader configured with the provided backend and options
func NewLoader(backendType LoaderBackendType, options ...LoaderBuilderOption) Loader {
	l := &loader{
		mu:         sync.RWMutex{},
		modelCache: make(map[string]model.Model),
	}

	switch backendType {
	case BackendTypeGLTF:
		l.backend = newGLTFLoaderBackend()
	}

	for _, option := range options {
		option(l)
	}
	return l
}
