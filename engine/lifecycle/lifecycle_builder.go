package lifecycle

import "sync"

type LifecycleBuilderOption func(*lifecycleImpl)

// WithState sets the initial lifecycle state of the lifecycle instance, by default LifecycleStateRegistered is set.
//
// Parameters:
//   - state: the initial lifecycle state to set for the lifecycle instance
//
// Returns:
//   - LifecycleBuilderOption: a builder option that can be passed to the NewLifecycle function to configure the initial state of the lifecycle instance
func WithState(state LifecycleState) LifecycleBuilderOption {
	return func(l *lifecycleImpl) {
		l.state = state
	}
}

// NewLifecycle creates a new instance of the Lifecycle interface.
//
// Parameters:
//   - opts: variadic options for configuring the lifecycle instance, such as setting the initial state.
//
// Returns:
//   - Lifecycle: a new instance of the Lifecycle interface configured with the provided options.
func NewLifecycle(opts ...LifecycleBuilderOption) Lifecycle {
	l := &lifecycleImpl{
		transitionMu: &sync.Mutex{},
		stateMu:      &sync.Mutex{},
		hooksMu:      &sync.Mutex{},
		state:        LifecycleStateRegistered,
		toHooks:      make(map[LifecycleState][]*Hook),
		fromHooks:    make(map[LifecycleState][]*Hook),
	}
	for _, opt := range opts {
		opt(l)
	}
	return l
}
