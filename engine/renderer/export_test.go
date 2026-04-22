package renderer

import (
	"sync"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
)

// NewRendererWithBackend constructs a renderer with the given backend, bypassing
// GPU device creation. Intended exclusively for use in tests.
func NewRendererWithBackend(backend RendererBackend) Renderer {
	r := &renderer{
		mu:            &sync.Mutex{},
		pipelineCache: make(map[string]pipeline.Pipeline),
		backend:       backend,
	}
	r.Delegate = r
	return r
}

// ApplyOption applies a RendererBuilderOption to an existing Renderer. For use in tests only.
func ApplyOption(r Renderer, opt RendererBuilderOption) {
	opt(r.(*renderer))
}

// RendererPendingPresentMode returns the pending present mode field. For use in tests only.
func RendererPendingPresentMode(r Renderer) *PresentMode {
	return r.(*renderer).pendingPresentMode
}

// RendererPendingMSAA returns the pending MSAA field. For use in tests only.
func RendererPendingMSAA(r Renderer) *MSAASampleCount {
	return r.(*renderer).pendingMSAA
}

// RendererForceFallbackAdapter returns the forceFallbackAdapter field. For use in tests only.
func RendererForceFallbackAdapter(r Renderer) bool {
	return r.(*renderer).forceFallbackAdapter
}

func RendererGPUSerializedProfiling(r Renderer) bool {
	return r.(*renderer).gpuSerializedProfiling
}
