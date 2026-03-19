package renderer

import (
	"sync"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/pipeline"
)

// renderer is the implementation of the Renderer interface.
type renderer struct {
	common.DelegateImpl[Renderer]

	mu *sync.Mutex

	pipelineCache map[string]pipeline.Pipeline

	backendType RendererBackendType
	backend     RendererBackend

	// Pre-creation config collected from builder options
	forceFallbackAdapter bool
	pendingPresentMode   *PresentMode
	pendingMSAA          *MSAASampleCount
	injections           map[string]string
}
