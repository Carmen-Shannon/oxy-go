package profiler

import (
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
