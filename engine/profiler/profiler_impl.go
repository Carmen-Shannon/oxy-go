package profiler

import (
	"runtime"
	"time"
)

// sectionStats accumulates timing data for a named render phase.
type sectionStats struct {
	totalNs      int64
	minNs        int64
	maxNs        int64
	count        int64
	pendingStart time.Time
	hasPending   bool
}

// metricStats accumulates scalar metric data under a named label.
type metricStats struct {
	total float64
	count int64
}

// profiler tracks frame rate and memory statistics for performance monitoring.
// Outputs stats to the log at a configurable interval.
type profiler struct {
	frameCount     int
	lastTime       time.Time
	updateInterval time.Duration
	memStats       runtime.MemStats
	lastGCCount    uint32
	lastTotalAlloc uint64

	sections     map[string]*sectionStats
	sectionOrder []string

	metrics     map[string]*metricStats
	metricOrder []string

	started bool
}
