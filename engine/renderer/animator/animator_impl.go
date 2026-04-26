package animator

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// animator is the implementation of the Animator interface.
type animator struct {
	common.DelegateImpl[Animator]

	backendType AnimatorBackendType
	backend     AnimatorBackend
	model       model.Model

	lc lifecycle.Lifecycle
}
