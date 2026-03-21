package profiler

import (
	"runtime"
	"time"
)

type ProfilerBuilderOption func(*profiler)

// WithUpdateInterval sets the interval at which the profiler logs performance statistics.
//
// Parameters:
//   - interval: the duration between profiler updates
//
// Returns:
//   - ProfilerBuilderOption: a function that applies the update interval option to a profiler
func WithUpdateInterval(interval time.Duration) ProfilerBuilderOption {
	return func(p *profiler) {
		p.updateInterval = interval
	}
}

// NewProfiler creates a new Profiler with default settings.
// Update interval defaults to 1 second.
//
// Returns:
//   - Profiler: the newly created profiler instance
func NewProfiler(options ...ProfilerBuilderOption) Profiler {
	p := &profiler{
		frameCount:     0,
		lastTime:       time.Now(),
		updateInterval: time.Second,
		memStats:       runtime.MemStats{},
	}
	for _, opt := range options {
		opt(p)
	}
	return p
}
