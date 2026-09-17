package command_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/command"
	"github.com/Carmen-Shannon/oxy-go/engine/context"
)

func TestRunCommandTests(t *testing.T) {
	suite.Run(t, new(commandTest))
}

type commandTest struct {
	suite.Suite
}

func (suite *commandTest) TestNewCommand() {
	suite.Run("should create an async command", func() {
		cmd := command.NewCommand(command.CommandTypeAsync)
		suite.NotNil(cmd)
		suite.Equal(command.CommandTypeAsync, cmd.Type())
	})

	suite.Run("should create a linear command", func() {
		cmd := command.NewCommand(command.CommandTypeLinear)
		suite.NotNil(cmd)
		suite.Equal(command.CommandTypeLinear, cmd.Type())
	})
}

func (suite *commandTest) TestWithCommandFunc() {
	suite.Run("should set the callback function on the command", func() {
		called := false
		cmd := command.NewCommand(command.CommandTypeAsync, command.WithCommandFunc(func(ctx context.Context) error {
			called = true
			return nil
		}))
		suite.NotNil(cmd)
		_ = cmd.Run(context.NewContext())
		suite.True(called)
	})
}

func (suite *commandTest) TestType() {
	suite.Run("should return CommandTypeAsync for an async command", func() {
		cmd := command.NewCommand(command.CommandTypeAsync)
		suite.Equal(command.CommandTypeAsync, cmd.Type())
	})

	suite.Run("should return CommandTypeLinear for a linear command", func() {
		cmd := command.NewCommand(command.CommandTypeLinear)
		suite.Equal(command.CommandTypeLinear, cmd.Type())
	})
}

func (suite *commandTest) TestRun() {
	suite.Run("should execute the callback with the provided context", func() {
		ctx := context.NewContext()
		receivedCtx := false
		cmd := command.NewCommand(command.CommandTypeAsync, command.WithCommandFunc(func(c context.Context) error {
			receivedCtx = (c == ctx)
			return nil
		}))
		err := cmd.Run(ctx)
		suite.NoError(err)
		suite.True(receivedCtx)
	})

	suite.Run("should panic when the callback is nil", func() {
		cmd := command.NewCommand(command.CommandTypeAsync)
		suite.Panics(func() {
			_ = cmd.Run(context.NewContext())
		})
	})
}
