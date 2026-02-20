package animator

import (
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
