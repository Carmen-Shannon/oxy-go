package profiler

import (
	"runtime"
	"time"
)

// profiler tracks frame rate and memory statistics for performance monitoring.
// Outputs stats to the log at a configurable interval.
type profiler struct {
	frameCount     int
	lastTime       time.Time
	updateInterval time.Duration
	memStats       runtime.MemStats
	lastGCCount    uint32
	lastTotalAlloc uint64
}
