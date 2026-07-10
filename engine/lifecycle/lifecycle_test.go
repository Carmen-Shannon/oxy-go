package lifecycle_test

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/lifecycle"
)

func TestRunLifecycleTests(t *testing.T) {
	suite.Run(t, new(lifecycleTest))
}

type lifecycleTest struct {
	suite.Suite
	lifecycle lifecycle.Lifecycle
}

func (suite *lifecycleTest) SetupSubTest() {
	suite.lifecycle = lifecycle.NewLifecycle()
}

func (suite *lifecycleTest) TestWithState() {
	suite.Run("should set the initial lifecycle state when using WithState", func() {
		lc := lifecycle.NewLifecycle(lifecycle.WithState(lifecycle.LifecycleStateRunning))
		suite.Equal(lifecycle.LifecycleStateRunning, lc.State())
	})
}

func (suite *lifecycleTest) TestSetState() {
	suite.Run("should transition to a valid lifecycle state", func() {
		err := suite.lifecycle.SetState(lifecycle.LifecycleStateStarting)
		suite.NoError(err)
		suite.Equal(lifecycle.LifecycleStateStarting, suite.lifecycle.State())
	})

	suite.Run("should skip nil hooks", func() {
		cleanupTo := suite.lifecycle.OnTransitionTo(lifecycle.LifecycleStateRunning, nil)
		cleanupFrom := suite.lifecycle.OnTransitionFrom(lifecycle.LifecycleStateStarting, nil)
		defer cleanupTo()
		defer cleanupFrom()

		err := suite.lifecycle.SetState(lifecycle.LifecycleStateStarting)
		suite.NoError(err)
		suite.Equal(lifecycle.LifecycleStateStarting, suite.lifecycle.State())

		err = suite.lifecycle.SetState(lifecycle.LifecycleStateRunning)
		suite.NoError(err)
		suite.Equal(lifecycle.LifecycleStateRunning, suite.lifecycle.State())
	})

	suite.Run("should return an error when transitioning to an invalid lifecycle state", func() {
		err := suite.lifecycle.SetState(lifecycle.LifecycleStateRunning)
		suite.Error(err)
		suite.Equal(lifecycle.LifecycleStateRegistered, suite.lifecycle.State())
	})

	suite.Run("should return an error when a lifecycle hook returns an error during a transition", func() {
		errorFunc := func() error {
			return errors.New("error")
		}
		cleanupTo := suite.lifecycle.OnTransitionTo(lifecycle.LifecycleStateStarting, errorFunc)
		cleanupFrom := suite.lifecycle.OnTransitionFrom(lifecycle.LifecycleStateRegistered, errorFunc)
		defer cleanupTo()
		defer cleanupFrom()

		err := suite.lifecycle.SetState(lifecycle.LifecycleStateStarting)
		suite.Error(err)
		suite.Equal(lifecycle.LifecycleStateStarting, suite.lifecycle.State())
	})
}
