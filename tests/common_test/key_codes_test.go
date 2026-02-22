package common_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/stretchr/testify/suite"
)

func TestKeyCodes(t *testing.T) {
	suite.Run(t, new(keyCodesTest))
}

type keyCodesTest struct {
	suite.Suite
}

func (suite *keyCodesTest) TestAlphaKeys() {
	suite.Run("KeyW equals ASCII 'W'", func() {
		suite.Equal(int('W'), common.KeyW)
	})

	suite.Run("KeyA equals ASCII 'A'", func() {
		suite.Equal(int('A'), common.KeyA)
	})

	suite.Run("KeyS equals ASCII 'S'", func() {
		suite.Equal(int('S'), common.KeyS)
	})

	suite.Run("KeyD equals ASCII 'D'", func() {
		suite.Equal(int('D'), common.KeyD)
	})

	suite.Run("KeyQ equals ASCII 'Q'", func() {
		suite.Equal(int('Q'), common.KeyQ)
	})

	suite.Run("KeyE equals ASCII 'E'", func() {
		suite.Equal(int('E'), common.KeyE)
	})

	suite.Run("KeyB equals ASCII 'B'", func() {
		suite.Equal(int('B'), common.KeyB)
	})

	suite.Run("KeyC equals ASCII 'C'", func() {
		suite.Equal(int('C'), common.KeyC)
	})

	suite.Run("KeyF equals ASCII 'F'", func() {
		suite.Equal(int('F'), common.KeyF)
	})

	suite.Run("KeyG equals ASCII 'G'", func() {
		suite.Equal(int('G'), common.KeyG)
	})

	suite.Run("KeyL equals ASCII 'L'", func() {
		suite.Equal(int('L'), common.KeyL)
	})

	suite.Run("KeyM equals ASCII 'M'", func() {
		suite.Equal(int('M'), common.KeyM)
	})

	suite.Run("KeyT equals ASCII 'T'", func() {
		suite.Equal(int('T'), common.KeyT)
	})

	suite.Run("KeyV equals ASCII 'V'", func() {
		suite.Equal(int('V'), common.KeyV)
	})

	suite.Run("KeyX equals ASCII 'X'", func() {
		suite.Equal(int('X'), common.KeyX)
	})
}

func (suite *keyCodesTest) TestNumericKeys() {
	suite.Run("Key0 equals ASCII '0'", func() {
		suite.Equal(int('0'), common.Key0)
	})

	suite.Run("Key1 equals ASCII '1'", func() {
		suite.Equal(int('1'), common.Key1)
	})

	suite.Run("Key2 equals ASCII '2'", func() {
		suite.Equal(int('2'), common.Key2)
	})

	suite.Run("Key3 equals ASCII '3'", func() {
		suite.Equal(int('3'), common.Key3)
	})

	suite.Run("Key4 equals ASCII '4'", func() {
		suite.Equal(int('4'), common.Key4)
	})

	suite.Run("Key5 equals ASCII '5'", func() {
		suite.Equal(int('5'), common.Key5)
	})

	suite.Run("Key6 equals ASCII '6'", func() {
		suite.Equal(int('6'), common.Key6)
	})

	suite.Run("Key7 equals ASCII '7'", func() {
		suite.Equal(int('7'), common.Key7)
	})

	suite.Run("Key8 equals ASCII '8'", func() {
		suite.Equal(int('8'), common.Key8)
	})

	suite.Run("Key9 equals ASCII '9'", func() {
		suite.Equal(int('9'), common.Key9)
	})
}

func (suite *keyCodesTest) TestSpecialKeys() {
	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw#Key
	suite.Run("KeySpace equals ASCII 32", func() {
		suite.Equal(32, common.KeySpace)
	})

	suite.Run("KeyBackspace equals GLFW 259", func() {
		suite.Equal(259, common.KeyBackspace)
	})

	suite.Run("KeyEsc equals GLFW 256", func() {
		suite.Equal(256, common.KeyEsc)
	})
}

func (suite *keyCodesTest) TestModifierKeys() {
	// Reference: https://pkg.go.dev/github.com/go-gl/glfw/v3.3/glfw#Key
	suite.Run("KeyLeftShift equals GLFW 340", func() {
		suite.Equal(340, common.KeyLeftShift)
	})

	suite.Run("KeyRightShift equals GLFW 344", func() {
		suite.Equal(344, common.KeyRightShift)
	})
}
