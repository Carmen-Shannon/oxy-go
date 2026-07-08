// Package queue provides a queue styled interface for submitting and handling custom engine commands
package queue

import (
	"github.com/Carmen-Shannon/oxy-go/engine/command"
	"github.com/Carmen-Shannon/oxy-go/engine/context"
	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
)

type Queue[T command.Command] interface {
	// Submit sends a command to the relevant queue based on the command type.
	// Submit will panic if the provided command has an underlying type that is not supported.
	//
	// Parameters:
	//   - cmd: The command to be submitted. The command's type will determine which queue it is sent to (async or linear).
	Submit(cmd T)

	// AsyncLifecycle returns the lifecycle associated with the async command queue.
	//
	// Returns:
	//   - The lifecycle instance that manages the async command queue, allowing for lifecycle hooks and management of async commands.
	AsyncLifecycle() lifecycle.Lifecycle

	// LinearLifecycle returns the lifecycle associated with the linear command queue.
	//
	// Returns:
	//   - The lifecycle instance that manages the linear command queue, allowing for lifecycle hooks and management of linear commands.
	LinearLifecycle() lifecycle.Lifecycle

	// Start begins draining both the async and linear command queues in background goroutines.
	// The provided engine context is passed to each command's Run method.
	// When done is closed, the drain goroutines exit immediately (hard-stop).
	//
	// Parameters:
	//   - ctx: the engine context passed to each command's Run method
	//   - done: channel that signals the drain goroutines to stop
	Start(ctx context.Context, done <-chan struct{})
}

func (q *queue[T]) Submit(cmd T) {
	switch cmd.Type() {
	case command.CommandTypeAsync:
		q.async <- cmd
	case command.CommandTypeLinear:
		q.linear <- cmd
	default:
		panic("invalid command type submitted to queue")
	}
}

func (q *queue[T]) AsyncLifecycle() lifecycle.Lifecycle {
	return q.asyncLC
}

func (q *queue[T]) LinearLifecycle() lifecycle.Lifecycle {
	return q.linearLC
}

func (q *queue[T]) Start(ctx context.Context, done <-chan struct{}) {
	q.asyncOnce.Do(func() {
		q.ctx = ctx
		_ = q.asyncLC.SetState(lifecycle.LifecycleStateStarting)
		_ = q.asyncLC.SetState(lifecycle.LifecycleStateRunning)
		go func() {
			for {
				select {
				case cmd := <-q.async:
					_ = cmd.Run(q.ctx)
				case <-done:
					return
				}
			}
		}()
	})
	q.linearOnce.Do(func() {
		if q.ctx == nil {
			q.ctx = ctx
		}
		_ = q.linearLC.SetState(lifecycle.LifecycleStateStarting)
		_ = q.linearLC.SetState(lifecycle.LifecycleStateRunning)
		go func() {
			for {
				select {
				case cmd := <-q.linear:
					_ = cmd.Run(q.ctx)
				case <-done:
					return
				}
			}
		}()
	})
}
