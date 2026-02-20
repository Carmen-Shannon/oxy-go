package profiler

import (
	"log"
	"runtime"
	"time"
)

// Profiler tracks frame rate and memory statistics for performance monitoring.
// Outputs stats to the log at a configurable interval.
type Profiler struct {
	frameCount     int
	lastTime       time.Time
	updateInterval time.Duration
	memStats       runtime.MemStats
	lastGCCount    uint32
	lastTotalAlloc uint64
}

// NewProfiler creates a new Profiler with default settings.
// Update interval defaults to 1 second.
//
// Returns:
//   - *Profiler: the newly created profiler instance
func NewProfiler() *Profiler {
	return &Profiler{
		frameCount:     0,
		lastTime:       time.Now(),
		updateInterval: time.Second,
		memStats:       runtime.MemStats{},
	}
}

// Tick should be called once per frame to track frame timing.
// Logs performance statistics when the update interval has elapsed.
// Statistics include: FPS, heap usage, allocation rate, GC count/pause times, total memory.
//
// Returns:
//   - bool: true if stats were logged this tick, false otherwise
func (p *Profiler) Tick() bool {
	p.frameCount++
	currentTime := time.Now()
	elapsed := currentTime.Sub(p.lastTime)

	if elapsed >= p.updateInterval {
		fps := float64(p.frameCount) / elapsed.Seconds()

		runtime.ReadMemStats(&p.memStats)
		// Alloc: Bytes of allocated heap objects (live memory)
		// TotalAlloc: Cumulative bytes allocated for heap objects (increases forever, tracks churn)
		// Sys: Total bytes of memory obtained from the OS (actual process footprint)
		allocMB := float64(p.memStats.Alloc) / 1024 / 1024
		sysMB := float64(p.memStats.Sys) / 1024 / 1024

		// Calculate allocation rate (MB/sec)
		allocDelta := p.memStats.TotalAlloc - p.lastTotalAlloc
		allocRateMB := float64(allocDelta) / 1024 / 1024 / elapsed.Seconds()

		// Calculate GC pause stats (last pause and max recent pause)
		gcCount := p.memStats.NumGC
		var lastPauseUs, maxPauseUs uint64
		if gcCount > 0 {
			// PauseNs is a circular buffer of last 256 GC pauses
			lastPauseUs = p.memStats.PauseNs[(gcCount-1)%256] / 1000

			// Find max pause since last tick
			startIdx := p.lastGCCount
			if gcCount-startIdx > 256 {
				startIdx = gcCount - 256
			}
			for i := startIdx; i < gcCount; i++ {
				pause := p.memStats.PauseNs[i%256] / 1000
				if pause > maxPauseUs {
					maxPauseUs = pause
				}
			}
		}

		log.Printf("[Profiler] FPS: %.2f | Heap: %.2f MB | Alloc Rate: %.2f MB/s | GC: %d (last: %d µs, max: %d µs) | Sys: %.2f MB",
			fps, allocMB, allocRateMB, gcCount, lastPauseUs, maxPauseUs, sysMB)

		p.frameCount = 0
		p.lastTime = currentTime
		p.lastGCCount = gcCount
		p.lastTotalAlloc = p.memStats.TotalAlloc
		return true
	}

	return false
}
