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

func TestRunProfilerTests(t *testing.T) {
	suite.Run(t, new(profilerTest))
}

func (suite *profilerTest) TestNewProfiler() {
	suite.Run("default interval fires after 1 second", func() {
		p := profiler.NewProfiler()
		result := p.Tick()
		suite.False(result)
	})

	suite.Run("WithUpdateInterval is applied", func() {
		p := profiler.NewProfiler(profiler.WithUpdateInterval(0))
		result := p.Tick()
		suite.True(result)
	})
}

func (suite *profilerTest) TestTick() {
	suite.Run("returns false when interval has not elapsed", func() {
		p := profiler.NewProfiler(profiler.WithUpdateInterval(time.Hour))
		result := p.Tick()
		suite.False(result)
	})

	suite.Run("returns true when interval has elapsed", func() {
		runtime.GC()
		p := profiler.NewProfiler(profiler.WithUpdateInterval(0))
		result := p.Tick()
		suite.True(result)
	})
}
