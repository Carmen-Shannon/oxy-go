package context

import (
	"sync"

	"github.com/Carmen-Shannon/oxy-go/engine/scene"
)

type context struct {
	sceneMu, valuesMu *sync.Mutex

	scenes map[int]scene.Scene

	values map[string]any
}
