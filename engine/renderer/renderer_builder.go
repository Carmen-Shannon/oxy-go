package renderer

import (
	"sync"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
	"github.com/Carmen-Shannon/oxy-go/engine/window"
)

// RendererBuilderOption is a functional option applied to a renderer during construction via NewRenderer.
type RendererBuilderOption func(*renderer)

// WithPipeline pre-registers a single Pipeline in the renderer's pipeline cache under the given key.
//
// Parameters:
//   - key: the unique identifier for the pipeline
//   - p: the Pipeline to cache
//
// Returns:
//   - RendererBuilderOption: a function that applies the pipeline option to a renderer
func WithPipeline(key string, p pipeline.Pipeline) RendererBuilderOption {
	return func(r *renderer) {
		r.pipelineCache[key] = p
	}
}

// WithPipelines replaces the renderer's entire pipeline cache with the provided map.
//
// Parameters:
//   - pipelines: a map of pipeline keys to their corresponding Pipeline objects
//
// Returns:
//   - RendererBuilderOption: a function that applies the pipelines option to a renderer
func WithPipelines(pipelines map[string]pipeline.Pipeline) RendererBuilderOption {
	return func(r *renderer) {
		r.pipelineCache = pipelines
	}
}

// WithPresentMode sets the surface present mode which controls how frames are delivered to the display.
//
// Parameters:
//   - mode: the PresentMode to use (VSync or Uncapped)
//
// Returns:
//   - RendererBuilderOption: a function that applies the present mode option to a renderer
func WithPresentMode(mode PresentMode) RendererBuilderOption {
	return func(r *renderer) {
		r.pendingPresentMode = &mode
	}
}

// WithMSAA sets the multisample anti-aliasing sample count for the renderer.
// When not specified, the default is MSAA4x. Use MSAAOff to disable MSAA entirely.
// Higher values (MSAA8x, MSAA16x) are adapter-dependent and may not be supported
// by all hardware.
//
// Parameters:
//   - count: the MSAASampleCount to use (MSAAOff, MSAA4x, MSAA8x, or MSAA16x)
//
// Returns:
//   - RendererBuilderOption: a function that applies the MSAA option to a renderer
func WithMSAA(count MSAASampleCount) RendererBuilderOption {
	return func(r *renderer) {
		r.pendingMSAA = &count
	}
}

// WithForceSoftwareRenderer forces WGPU to use a CPU/software fallback adapter instead of
// hardware GPU acceleration. This requires a software Vulkan ICD to be installed on the system
// (e.g. SwiftShader or lavapipe). Useful for benchmarking CPU vs GPU rendering performance.
//
// Parameters:
//   - force: true to force the software fallback adapter, false to use hardware (default)
//
// Returns:
//   - RendererBuilderOption: a function that applies the force software renderer option to a renderer
func WithForceSoftwareRenderer(force bool) RendererBuilderOption {
	return func(r *renderer) {
		r.forceFallbackAdapter = force
	}
}

// NewRenderer creates a new Renderer instance with the specified backend type and surface descriptor.
// The surface descriptor is platform-specific and is typically obtained from Window.GetSurfaceDescriptor().
//
// Parameters:
//   - backendType: the type of rendering backend to use (e.g., WGPU)
//   - surfaceDescriptor: the platform-specific surface descriptor for WebGPU surface creation
//   - options: variadic list of RendererBuilderOption functions to configure the Renderer
//
// Returns:
//   - Renderer: a new instance of Renderer configured with the specified backend and options
func NewRenderer(backendType RendererBackendType, window window.Window, options ...RendererBuilderOption) Renderer {
	r := &renderer{
		mu:            &sync.Mutex{},
		pipelineCache: make(map[string]pipeline.Pipeline),
		backendType:   backendType,
	}

	// Apply options first so config flags (e.g. forceFallbackAdapter) are
	// available before the backend requests a GPU adapter.
	for _, opt := range options {
		opt(r)
	}

	msaa := MSAA4x // default
	if r.pendingMSAA != nil {
		msaa = *r.pendingMSAA
	}

	switch backendType {
	case BackendTypeWGPU:
		fallthrough
	default:
		r.backend = newWGPURendererBackend(window.SurfaceDescriptor(), r.forceFallbackAdapter, msaa)
	}

	if r.pendingPresentMode != nil {
		r.backend.SetPresentMode(*r.pendingPresentMode)
	}

	r.backend.ConfigureSurface(window.Width(), window.Height())
	r.Delegate = r
	return r
}
