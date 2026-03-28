package profiler

import (
	"log"
	"math"
	"runtime"
	"time"
)

type Profiler interface {
	// Tick should be called once per frame to track frame timing.
	// Logs performance statistics when the update interval has elapsed.
	// Statistics include: FPS, heap usage, allocation rate, GC count/pause times, total memory.
	// When section or metric data has been recorded, it is also emitted and then reset.
	//
	// Returns:
	//   - bool: true if stats were logged this tick, false otherwise
	Tick() bool

	// Section brackets a named timing section. Call the returned func to stop timing.
	// Typical usage: defer profiler.Section("PrepareShadows")()
	// If profiling is not active (Tick has never been called), this is a no-op.
	//
	// Parameters:
	//   - label: name of the timing section
	//
	// Returns:
	//   - func(): call this function to stop the timing section
	Section(label string) func()

	// Record stores a scalar metric value under the given label.
	// Reported in the Tick() output alongside timing sections.
	//
	// Parameters:
	//   - label: name of the metric
	//   - value: scalar value to record
	Record(label string, value float64)
}

func (p *profiler) Tick() bool {
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

		if len(p.sections) > 0 {
			log.Printf("[Profiler] Frame phase timings (avg over %s):", elapsed.Round(time.Millisecond))
			var totalAvgMs float64
			for _, label := range p.sectionOrder {
				s := p.sections[label]
				if s.count == 0 {
					continue
				}
				totalAvgMs += float64(s.totalNs) / float64(s.count) / 1e6
			}
			for _, label := range p.sectionOrder {
				s := p.sections[label]
				if s.count == 0 {
					continue
				}
				avgMs := float64(s.totalNs) / float64(s.count) / 1e6
				minMs := float64(s.minNs) / 1e6
				maxMs := float64(s.maxNs) / 1e6
				var pct float64
				if totalAvgMs > 0 {
					pct = (avgMs / totalAvgMs) * 100.0
				}
				log.Printf("  %-24s avg: %8.3fms   min: %8.3fms   max: %8.3fms   (%5.1f%%)", label, avgMs, minMs, maxMs, pct)
			}
		}

		if len(p.metrics) > 0 {
			log.Printf("[Profiler] Metrics:")
			for _, label := range p.metricOrder {
				m := p.metrics[label]
				if m.count == 0 {
					continue
				}
				avgVal := m.total / float64(m.count)
				log.Printf("  %-24s avg: %.2fms", label, avgVal)
			}
		}

		p.frameCount = 0
		p.lastTime = currentTime
		p.lastGCCount = gcCount
		p.lastTotalAlloc = p.memStats.TotalAlloc
		p.started = true

		for _, s := range p.sections {
			s.totalNs = 0
			s.minNs = math.MaxInt64
			s.maxNs = 0
			s.count = 0
		}
		for _, m := range p.metrics {
			m.total = 0
			m.count = 0
		}

		return true
	}

	return false
}

func (p *profiler) Section(label string) func() {
	if _, ok := p.sections[label]; !ok {
		p.sections[label] = &sectionStats{minNs: math.MaxInt64}
		p.sectionOrder = append(p.sectionOrder, label)
	}
	s := p.sections[label]
	s.hasPending = true
	s.pendingStart = time.Now()
	return func() {
		if !s.hasPending {
			return
		}
		elapsed := time.Since(s.pendingStart).Nanoseconds()
		s.hasPending = false
		s.totalNs += elapsed
		s.count++
		if elapsed < s.minNs {
			s.minNs = elapsed
		}
		if elapsed > s.maxNs {
			s.maxNs = elapsed
		}
	}
}

func (p *profiler) Record(label string, value float64) {
	if _, ok := p.metrics[label]; !ok {
		p.metrics[label] = &metricStats{}
		p.metricOrder = append(p.metricOrder, label)
	}
	m := p.metrics[label]
	m.total += value
	m.count++
}
