// Package command provides an interface for defining commands to send and receive from the Engine
package command

import "github.com/Carmen-Shannon/oxy-go/engine/context"

type Command interface {
	// Type returns the CommandType of this command, which determines how the command will be processed by the engine's command queues.
	//
	// Returns:
	//   - The CommandType of this command, which can be used to determine which queue the command will be submitted to (async or linear).
	Type() CommandType

	// Run executes the command's logic using the provided context. The context provides access to engine resources and state that may be needed to execute the command.
	//
	// Parameters:
	//   - ctx: The context that will be passed to the command's Run method, providing access to engine resources and state.
	//
	// Returns:
	//   - An error if the command execution fails, or nil if the command executes successfully.
	Run(ctx context.Context) error
}

type CommandFunc func(ctx context.Context) error
type CommandType string

const (
	CommandTypeAsync  CommandType = "async"
	CommandTypeLinear CommandType = "linear"
)

func (c *command) Type() CommandType {
	return c.cmdType
}

func (c *command) Run(ctx context.Context) error {
	if c.cb == nil {
		panic("command callback is nil, you must issue a CommandFunc using WithCommandFunc when creating this command before it is submitted.")
	}
	return c.cb(ctx)
}
