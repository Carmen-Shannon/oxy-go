package command

type CommandBuilderOption func(*command)

// WithCommandFunc is a CommandBuilderOption that sets the CommandFunc for a command.
//
// Parameters:
//   - cb: The CommandFunc to be executed when the command is run.
//
// Returns:
//   - A CommandBuilderOption that can be passed to NewCommand to configure the command's execution function.
func WithCommandFunc(cb CommandFunc) CommandBuilderOption {
	return func(c *command) {
		c.cb = cb
	}
}

// NewCommand creates a new Command based on the provided CommandType and applies any given builder options to it.
// This function will return an instance of either AsyncCommand or LinearCommand depending on the CommandType specified.
// If an unsupported CommandType is provided, it will return nil.
//
// Parameters:
//   - cmdType: The type of command to create (e.g., CommandTypeAsync or CommandTypeLinear).
//   - options: A variadic list of CommandBuilderOption functions that can be used to configure the command.
//
// Returns:
//   - Command: An instance of a Command (either AsyncCommand or LinearCommand) configured according to the provided options, or nil if the CommandType is unsupported.
func NewCommand(cmdType CommandType, options ...CommandBuilderOption) Command {
	c := &command{
		cmdType: cmdType,
	}
	for _, opt := range options {
		opt(c)
	}
	return c
}
