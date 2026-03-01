package profiler_test

import (
	"runtime"
	"testing"
	"time"

	"github.com/Carmen-Shannon/oxy-go/engine/profiler"
	"github.com/stretchr/testify/suite"
)

type profilerTest struct {
	suite.Suite
}

func TestProfiler(t *testing.T) {
	suite.Run(t, new(profilerTest))
}

func (suite *profilerTest) TestNewProfiler() {
	suite.Run("returns a non-nil profiler", func() {
		p := profiler.NewProfiler()
		suite.NotNil(p)
	})

	suite.Run("tick returns false immediately after creation", func() {
		p := profiler.NewProfiler()
		suite.False(p.Tick())
	})
}

func (suite *profilerTest) TestTick() {
	suite.Run("returns false when update interval has not elapsed", func() {
		p := profiler.NewProfiler()
		for i := 0; i < 10; i++ {
			suite.False(p.Tick())
		}
	})

	suite.Run("returns true after update interval elapses", func() {
		p := profiler.NewProfiler()
		// Force a GC cycle so the GC stats branch is exercised.
		runtime.GC()

		// Tick repeatedly until the 1-second default interval has passed.
		deadline := time.After(3 * time.Second)
		triggered := false
		for !triggered {
			select {
			case <-deadline:
				suite.Fail("tick never returned true within timeout")
				return
			default:
				if p.Tick() {
					triggered = true
				} else {
					time.Sleep(10 * time.Millisecond)
				}
			}
		}
		suite.True(triggered)
	})

	suite.Run("returns false again immediately after logging", func() {
		p := profiler.NewProfiler()
		runtime.GC()

		// Wait for the first true tick.
		deadline := time.After(3 * time.Second)
		for {
			select {
			case <-deadline:
				suite.Fail("tick never returned true within timeout")
				return
			default:
				if p.Tick() {
					// Immediately after a true return, next tick should be false.
					suite.False(p.Tick())
					return
				}
				time.Sleep(10 * time.Millisecond)
			}
		}
	})
}
