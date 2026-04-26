package animator

import (
	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// AnimatorBuilderOption is a functional option for configuring an Animator during construction.
type AnimatorBuilderOption func(*animator)

// WithMaxInstances is an option builder that sets the maximum number of instances the Animator can manage.
//
// Parameters:
//   - maxInstances: the maximum number of instances to support
//
// Returns:
//   - AnimatorBuilderOption: a function that applies the max instances option to an animator
func WithMaxInstances(maxInstances int) AnimatorBuilderOption {
	return func(a *animator) {
		a.backend.SetMaxInstances(uint32(maxInstances))
	}
}

// WithModel is an option builder that assigns a Model to the Animator during construction.
// This calls SetModel internally, which flattens skeleton and animation data into the backend.
//
// Parameters:
//   - m: the Model to associate with this animator
//   - boneBinding: the binding index for the bone data buffer in the compute shader
//   - packedBinding: the binding index for the packed animation data buffer in the compute shader
//
// Returns:
//   - AnimatorBuilderOption: a function that applies the model option to an animator
func WithModel(m model.Model, boneBinding, packedBinding int) AnimatorBuilderOption {
	return func(a *animator) {
		a.SetModel(m, boneBinding, packedBinding)
	}
}

// NewAnimator creates a new Animator instance with the specified backend type.
// The backend is created based on the type and then configured using the provided options.
// Binding indices are configured via WithBinding options rather than fixed struct parameters,
// allowing any shader binding layout.
//
// Parameters:
//   - backendType: the type of animation backend to use (BackendTypeSimple or BackendTypeSkeletal)
//   - options: variadic list of AnimatorBuilderOption functions to configure the Animator
//
// Returns:
//   - Animator: a new instance of Animator configured with the specified backend and options
func NewAnimator(backendType AnimatorBackendType, options ...AnimatorBuilderOption) Animator {
	a := &animator{
		backendType: backendType,
		lc:          lifecycle.NewLifecycle(),
	}
	switch backendType {
	case BackendTypeSkeletal:
		a.backend = newSkeletalAnimatorBackend()
	case BackendTypeSimple:
		fallthrough
	default:
		a.backend = newSimpleAnimatorBackend()
	}
	for _, opt := range options {
		opt(a)
	}
	a.Delegate = a
	return a
}
