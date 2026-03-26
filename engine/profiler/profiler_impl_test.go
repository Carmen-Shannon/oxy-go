package profiler

import (
	"math"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type profilerImplTest struct {
	suite.Suite
}

func TestRunProfilerImplTests(t *testing.T) {
	suite.Run(t, new(profilerImplTest))
}

func (suite *profilerImplTest) TestTick() {
	suite.Run("returns false on second call after lastTime reset", func() {
		p := &profiler{
			updateInterval: time.Second,
			lastTime:       time.Now().Add(-time.Hour),
		}
		first := p.Tick()
		suite.True(first)
		second := p.Tick()
		suite.False(second)
	})
}

func (suite *profilerImplTest) TestTickGCCountOverflow() {
	suite.Run("startIdx clamped when gcCount far exceeds lastGCCount", func() {
		for i := 0; i < 260; i++ {
			runtime.GC()
		}
		p := &profiler{
			updateInterval: time.Nanosecond,
			lastTime:       time.Time{},
			lastGCCount:    0,
		}
		result := p.Tick()
		suite.True(result)
	})
}

func (suite *profilerImplTest) TestSection() {
	suite.Run("accumulates timing for a named section", func() {
		p := &profiler{
			sections:     make(map[string]*sectionStats),
			sectionOrder: []string{},
		}
		stop1 := p.Section("foo")
		time.Sleep(time.Microsecond)
		stop1()
		stop2 := p.Section("foo")
		time.Sleep(time.Microsecond)
		stop2()
		s := p.sections["foo"]
		suite.Equal(int64(2), s.count)
		suite.Greater(s.totalNs, int64(0))
		suite.GreaterOrEqual(s.maxNs, s.minNs)
	})

	suite.Run("initializes minNs to MaxInt64 before first stop", func() {
		p := &profiler{
			sections:     make(map[string]*sectionStats),
			sectionOrder: []string{},
		}
		p.Section("bar")
		s := p.sections["bar"]
		suite.Equal(int64(math.MaxInt64), s.minNs)
	})

	suite.Run("noop if hasPending is false", func() {
		p := &profiler{
			sections:     make(map[string]*sectionStats),
			sectionOrder: []string{},
		}
		stop := p.Section("baz")
		p.sections["baz"].hasPending = false
		stop()
		s := p.sections["baz"]
		suite.Equal(int64(0), s.totalNs)
		suite.Equal(int64(0), s.count)
	})

	suite.Run("preserves sectionOrder", func() {
		p := &profiler{
			sections:     make(map[string]*sectionStats),
			sectionOrder: []string{},
		}
		p.Section("alpha")
		p.Section("beta")
		suite.Equal([]string{"alpha", "beta"}, p.sectionOrder)
	})

	suite.Run("reuses existing entry", func() {
		p := &profiler{
			sections:     make(map[string]*sectionStats),
			sectionOrder: []string{},
		}
		stop1 := p.Section("foo")
		time.Sleep(time.Microsecond)
		stop1()
		stop2 := p.Section("foo")
		time.Sleep(time.Microsecond)
		stop2()
		suite.Len(p.sections, 1)
		suite.Equal(int64(2), p.sections["foo"].count)
	})
}

func (suite *profilerImplTest) TestRecord() {
	suite.Run("accumulates values", func() {
		p := &profiler{
			metrics:     make(map[string]*metricStats),
			metricOrder: []string{},
		}
		p.Record("metric", 5.0)
		p.Record("metric", 3.0)
		suite.InDelta(8.0, p.metrics["metric"].total, 1e-6)
		suite.Equal(int64(2), p.metrics["metric"].count)
	})

	suite.Run("initializes new entry", func() {
		p := &profiler{
			metrics:     make(map[string]*metricStats),
			metricOrder: []string{},
		}
		p.Record("new_metric", 1.0)
		_, ok := p.metrics["new_metric"]
		suite.True(ok)
		suite.Contains(p.metricOrder, "new_metric")
	})

	suite.Run("preserves metricOrder", func() {
		p := &profiler{
			metrics:     make(map[string]*metricStats),
			metricOrder: []string{},
		}
		p.Record("a", 1.0)
		p.Record("b", 2.0)
		suite.Equal([]string{"a", "b"}, p.metricOrder)
	})
}

func (suite *profilerImplTest) TestTickResets() {
	suite.Run("resets section accumulators after logging", func() {
		p := &profiler{
			sections:       make(map[string]*sectionStats),
			sectionOrder:   []string{},
			metrics:        make(map[string]*metricStats),
			metricOrder:    []string{},
			updateInterval: time.Nanosecond,
			lastTime:       time.Time{},
		}
		stop := p.Section("x")
		stop()
		result := p.Tick()
		suite.True(result)
		s := p.sections["x"]
		suite.Equal(int64(0), s.totalNs)
		suite.Equal(int64(0), s.count)
		suite.Equal(int64(math.MaxInt64), s.minNs)
	})

	suite.Run("resets metric accumulators after logging", func() {
		p := &profiler{
			sections:       make(map[string]*sectionStats),
			sectionOrder:   []string{},
			metrics:        make(map[string]*metricStats),
			metricOrder:    []string{},
			updateInterval: time.Nanosecond,
			lastTime:       time.Time{},
		}
		p.Record("y", 1.0)
		result := p.Tick()
		suite.True(result)
		m := p.metrics["y"]
		suite.InDelta(0.0, m.total, 1e-6)
		suite.Equal(int64(0), m.count)
	})
}
