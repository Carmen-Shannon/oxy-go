// Package lifecycle provides an interface for defining lifecycles for various engine interfaces
package lifecycle

// Lifecycle defines the interface for managing lifecycles of various engine interfaces
type Lifecycle interface {
	// State returns the current lifecycle state of the consumer
	//
	// Returns:
	//   - LifecycleState: the current lifecycle state of the consumer
	State() LifecycleState

	// SetState attempts to transition the lifecycle to a new state.
	//
	// State transitions are executed synchronously, processing all OnTransitionFrom hooks
	// for the old state, updating the state, and then processing all OnTransitionTo hooks
	// for the new state. A transition cannot be interrupted or happen concurrently.
	//
	// WARNING: Because transitions hold the state lock until complete, calling SetState
	// synchronously from inside a lifecycle hook will result in a permanent deadlock.
	// If a hook must trigger a subsequent state change, it must do so asynchronously.
	//
	// Example of what NOT to do (will deadlock):
	//
	//	lc.OnTransitionTo(LifecycleStateRunning, func() error {
	//		// DEADLOCK: The current transition lock is still held
	//		return lc.SetState(LifecycleStateStopped)
	//	})
	//
	// Example of the correct asynchronous approach:
	//
	//	lc.OnTransitionTo(LifecycleStateRunning, func() error {
	//		if shouldStop {
	//			// Correct: Queue the next state transition on a new goroutine
	//			go lc.SetState(LifecycleStateStopped)
	//		}
	//		return nil
	//	})
	//
	// Parameters:
	//   - state: the new lifecycle state to transition to
	//
	// Returns:
	//   - error: an error if the transition is not allowed, or aggregated errors from hooks
	SetState(state LifecycleState) error

	// OnTransitionTo registers a hook to be executed when a transition to a specific lifecycle state occurs
	//
	// Parameters:
	//   - to: the lifecycle state that triggers the execution of the hook
	//   - hook: the function to execute when the transition occurs
	//
	// Returns:
	//   - cleanup: a function that can be called to remove or unregister the hook
	OnTransitionTo(to LifecycleState, hook Hook) (cleanup func())

	// OnTransitionFrom registers a hook to be executed when a transition from a specific lifecycle state occurs
	//
	// Parameters:
	//   - from: the lifecycle state that triggers the execution of the hook
	//   - hook: the function to execute when the transition occurs
	//
	// Returns:
	//   - cleanup: a function that can be called to remove or unregister the hook
	OnTransitionFrom(from LifecycleState, hook Hook) (cleanup func())
}

var _ Lifecycle = &lifecycleImpl{}

func (l *lifecycleImpl) SetState(state LifecycleState) error { return l.transitionTo(state) }

func (l *lifecycleImpl) State() LifecycleState {
	l.stateMu.Lock()
	defer l.stateMu.Unlock()
	return l.state
}

func (l *lifecycleImpl) OnTransitionTo(to LifecycleState, hook Hook) (cleanup func()) {
	l.hooksMu.Lock()
	defer l.hooksMu.Unlock()
	h := &hook
	l.toHooks[to] = append(l.toHooks[to], h)

	return func() {
		l.removeHook(to, h, true, false)
	}
}

func (l *lifecycleImpl) OnTransitionFrom(from LifecycleState, hook Hook) (cleanup func()) {
	l.hooksMu.Lock()
	defer l.hooksMu.Unlock()
	h := &hook
	l.fromHooks[from] = append(l.fromHooks[from], h)

	return func() {
		l.removeHook(from, h, false, true)
	}
}

type LifecycleState int

const (
	LifecycleStateRegistered LifecycleState = iota
	LifecycleStateStarting
	LifecycleStateRunning
	LifecycleStatePaused
	LifecycleStateDraining
	LifecycleStateStopped
	LifecycleStateErrored
	LifecycleStateRemoved
)
