package queue_test

import (
	"testing"
	"time"

	"github.com/Carmen-Shannon/oxy-go/engine/command"
	"github.com/Carmen-Shannon/oxy-go/engine/context"
	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
	"github.com/Carmen-Shannon/oxy-go/engine/queue"
	"github.com/stretchr/testify/suite"
)

func TestRunQueueTests(t *testing.T) {
	suite.Run(t, new(queueTest))
}

type queueTest struct {
	suite.Suite
}

func (suite *queueTest) TestNewQueue() {
	suite.Run("should create a queue with defaults", func() {
		q := queue.NewQueue[command.Command]()
		suite.NotNil(q)
		suite.NotNil(q.AsyncLifecycle())
		suite.NotNil(q.LinearLifecycle())
	})

	suite.Run("should apply WithAsyncBufferSize", func() {
		q := queue.NewQueue(queue.WithAsyncBufferSize[command.Command](5))
		suite.NotNil(q)
	})

	suite.Run("should apply WithLinearBufferSize", func() {
		q := queue.NewQueue(queue.WithLinearBufferSize[command.Command](3))
		suite.NotNil(q)
	})
}

func (suite *queueTest) TestSubmit() {
	suite.Run("should route async command to the async queue", func() {
		q := queue.NewQueue[command.Command]()
		cmd := command.NewCommand(command.CommandTypeAsync, command.WithCommandFunc(func(ctx context.Context) error {
			return nil
		}))

		done := make(chan struct{})
		executed := make(chan struct{}, 1)
		cmd2 := command.NewCommand(command.CommandTypeAsync, command.WithCommandFunc(func(ctx context.Context) error {
			executed <- struct{}{}
			return nil
		}))

		ctx := context.NewContext()
		q.Submit(cmd)
		q.Submit(cmd2)
		q.Start(ctx, done)

		select {
		case <-executed:
		case <-time.After(time.Second):
			suite.Fail("async command was not executed within timeout")
		}

		close(done)
	})

	suite.Run("should route linear command to the linear queue", func() {
		q := queue.NewQueue[command.Command]()
		executed := make(chan struct{}, 1)
		cmd := command.NewCommand(command.CommandTypeLinear, command.WithCommandFunc(func(ctx context.Context) error {
			executed <- struct{}{}
			return nil
		}))

		done := make(chan struct{})
		ctx := context.NewContext()
		q.Submit(cmd)
		q.Start(ctx, done)

		select {
		case <-executed:
		case <-time.After(time.Second):
			suite.Fail("linear command was not executed within timeout")
		}

		close(done)
	})

	suite.Run("should panic on invalid command type", func() {
		q := queue.NewQueue[command.Command]()
		cmd := command.NewCommand(command.CommandType("invalid"))
		suite.Panics(func() {
			q.Submit(cmd)
		})
	})
}

func (suite *queueTest) TestAsyncLifecycle() {
	suite.Run("should return a non-nil lifecycle in registered state", func() {
		q := queue.NewQueue[command.Command]()
		lc := q.AsyncLifecycle()
		suite.NotNil(lc)
		suite.Equal(lifecycle.LifecycleStateRegistered, lc.State())
	})
}

func (suite *queueTest) TestLinearLifecycle() {
	suite.Run("should return a non-nil lifecycle in registered state", func() {
		q := queue.NewQueue[command.Command]()
		lc := q.LinearLifecycle()
		suite.NotNil(lc)
		suite.Equal(lifecycle.LifecycleStateRegistered, lc.State())
	})
}

func (suite *queueTest) TestStart() {
	suite.Run("should transition lifecycles to running", func() {
		q := queue.NewQueue[command.Command]()
		done := make(chan struct{})
		ctx := context.NewContext()
		q.Start(ctx, done)

		suite.Equal(lifecycle.LifecycleStateRunning, q.AsyncLifecycle().State())
		suite.Equal(lifecycle.LifecycleStateRunning, q.LinearLifecycle().State())

		close(done)
	})

	suite.Run("should execute submitted async commands", func() {
		q := queue.NewQueue[command.Command]()
		executed := make(chan struct{}, 1)
		cmd := command.NewCommand(command.CommandTypeAsync, command.WithCommandFunc(func(ctx context.Context) error {
			executed <- struct{}{}
			return nil
		}))

		done := make(chan struct{})
		ctx := context.NewContext()
		q.Submit(cmd)
		q.Start(ctx, done)

		select {
		case <-executed:
		case <-time.After(time.Second):
			suite.Fail("command was not executed within timeout")
		}

		close(done)
	})

	suite.Run("should be idempotent for async lifecycle", func() {
		q := queue.NewQueue[command.Command]()
		done := make(chan struct{})
		ctx := context.NewContext()
		
		q.Start(ctx, done)
		suite.Equal(lifecycle.LifecycleStateRunning, q.AsyncLifecycle().State())

		q.Start(ctx, done)
		suite.Equal(lifecycle.LifecycleStateRunning, q.AsyncLifecycle().State())

		close(done)
	})
}
