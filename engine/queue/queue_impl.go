package queue

import (
	"sync"

	"github.com/Carmen-Shannon/oxy-go/engine/command"
	"github.com/Carmen-Shannon/oxy-go/engine/context"
	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
)

type queue[T command.Command] struct {
	async  chan T
	linear chan T

	asyncLC  lifecycle.Lifecycle
	linearLC lifecycle.Lifecycle

	ctx        context.Context
	asyncOnce  sync.Once
	linearOnce sync.Once
}
