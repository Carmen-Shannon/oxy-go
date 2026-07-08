package queue

import (
	"github.com/Carmen-Shannon/oxy-go/engine/command"
	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
)

type QueueBuilderOption[T command.Command] func(*queue[T])

// WithAsyncBufferSize sets the buffer size for the async command channel.
//
// Parameters:
//   - size: The desired buffer size for the async command channel. If not set, defaults to 100.
//
// Returns:
//   - A QueueBuilderOption that can be passed to NewQueue to configure the async channel buffer size.
func WithAsyncBufferSize[T command.Command](size int) QueueBuilderOption[T] {
	return func(q *queue[T]) {
		q.async = make(chan T, size)
	}
}

// WithLinearBufferSize sets the buffer size for the linear command channel.
//
// Parameters:
//   - size: The desired buffer size for the linear command channel. If not set, defaults to 10.
//
// Returns:
//   - A QueueBuilderOption that can be passed to NewQueue to configure the linear channel buffer size.
func WithLinearBufferSize[T command.Command](size int) QueueBuilderOption[T] {
	return func(q *queue[T]) {
		q.linear = make(chan T, size)
	}
}

// NewQueue creates a new Queue instance with optional configuration.
//
// Parameters:
//   - options: A variadic list of QueueBuilderOption functions that can be used to customize the queue's behavior and settings. If no options are provided, the queue will be initialized with default settings.
//
// Returns:
//   - A new Queue instance configured according to the provided options.
func NewQueue[T command.Command](options ...QueueBuilderOption[T]) Queue[T] {
	q := &queue[T]{
		async:    make(chan T, 100),
		linear:   make(chan T, 10),
		asyncLC:  lifecycle.NewLifecycle(),
		linearLC: lifecycle.NewLifecycle(),
	}
	for _, opt := range options {
		opt(q)
	}
	return q
}
