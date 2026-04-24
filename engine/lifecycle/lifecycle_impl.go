package lifecycle

import (
	"errors"
	"fmt"
	"slices"
	"sync"
)

type lifecycleImpl struct {
	transitionMu, stateMu, hooksMu *sync.Mutex

	state LifecycleState

	toHooks, fromHooks map[LifecycleState][]*Hook
}

type lifecycleTransition struct {
	from LifecycleState
	to   LifecycleState
}

var (
	lifecycleTransitions = []lifecycleTransition{
		{from: LifecycleStateRegistered, to: LifecycleStateStarting},
		{from: LifecycleStateRegistered, to: LifecycleStateStopped},
		{from: LifecycleStateStarting, to: LifecycleStateRunning},
		{from: LifecycleStateStarting, to: LifecycleStateErrored},
		{from: LifecycleStateRunning, to: LifecycleStatePaused},
		{from: LifecycleStateRunning, to: LifecycleStateDraining},
		{from: LifecycleStateRunning, to: LifecycleStateErrored},
		{from: LifecycleStatePaused, to: LifecycleStateRunning},
		{from: LifecycleStatePaused, to: LifecycleStateStopped},
		{from: LifecycleStateDraining, to: LifecycleStateRunning},
		{from: LifecycleStateDraining, to: LifecycleStateStopped},
		{from: LifecycleStateDraining, to: LifecycleStateErrored},
		{from: LifecycleStateErrored, to: LifecycleStateDraining},
		{from: LifecycleStateStopped, to: LifecycleStateRemoved},
	}
)

// transitionTo manages the transition from a previous lifecycle state to a new lifecycle state.
// the validation governing the allowance of a transition depends on the unexported lifecycleTransitions slice.
//
// Parameters:
//   - state: the new lifecycle state to transition to
//
// Returns:
//   - error: an error if the transition is not allowed, otherwise nil
func (l *lifecycleImpl) transitionTo(state LifecycleState) error {
	l.transitionMu.Lock()
	defer l.transitionMu.Unlock()

	l.stateMu.Lock()
	oldState := l.state
	l.stateMu.Unlock()

	if !slices.ContainsFunc(lifecycleTransitions, func(t lifecycleTransition) bool {
		return t.from == oldState && t.to == state
	}) {
		return fmt.Errorf("cannot transition from %v to %v", oldState, state)
	}

	l.hooksMu.Lock()
	froms := append([]*Hook(nil), l.fromHooks[oldState]...)
	l.hooksMu.Unlock()

	var errs []error
	for _, hookPtr := range froms {
		if hookPtr == nil || *hookPtr == nil {
			continue
		}
		if err := (*hookPtr)(); err != nil {
			errs = append(errs, err)
		}
	}

	l.stateMu.Lock()
	l.state = state
	l.stateMu.Unlock()

	l.hooksMu.Lock()
	tos := append([]*Hook(nil), l.toHooks[state]...)
	l.hooksMu.Unlock()

	for _, hookPtr := range tos {
		if hookPtr == nil || *hookPtr == nil {
			continue
		}
		if err := (*hookPtr)(); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}

// removeHook removes a hook from the lifecycle's registered hooks for a given state and transition direction.
//
// Parameters:
//   - state: the lifecycle state associated with the hook to be removed
//   - hook: a pointer to the Hook function to be removed
//   - to: a boolean indicating whether the hook is registered for transitions to the specified state
//   - from: a boolean indicating whether the hook is registered for transitions from the specified state
func (l *lifecycleImpl) removeHook(state LifecycleState, hook *Hook, to, from bool) {
	l.hooksMu.Lock()
	defer l.hooksMu.Unlock()
	if to {
		hooks := l.toHooks[state]
		for i, h := range hooks {
			if h == hook {
				hooks[i] = hooks[len(hooks)-1]
				l.toHooks[state] = hooks[:len(hooks)-1]
				break
			}
		}
	}
	if from {
		hooks := l.fromHooks[state]
		for i, h := range hooks {
			if h == hook {
				hooks[i] = hooks[len(hooks)-1]
				l.fromHooks[state] = hooks[:len(hooks)-1]
				break
			}
		}
	}
}
