package common_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/stretchr/testify/suite"
)

func TestUtils(t *testing.T) {
	suite.Run(t, new(utilsTest))
}

type utilsTest struct {
	suite.Suite
}

func (suite *utilsTest) TestCoalesce() {
	suite.Run("returns first non-zero int", func() {
		result := common.Coalesce(0, 0, 5, 3)
		suite.Equal(5, result)
	})

	suite.Run("returns first value when it is non-zero", func() {
		result := common.Coalesce(7, 0, 3)
		suite.Equal(7, result)
	})

	suite.Run("returns zero when all values are zero", func() {
		result := common.Coalesce(0, 0, 0)
		suite.Equal(0, result)
	})

	suite.Run("returns zero for no arguments", func() {
		result := common.Coalesce[int]()
		suite.Equal(0, result)
	})

	suite.Run("single non-zero value is returned", func() {
		result := common.Coalesce(42)
		suite.Equal(42, result)
	})

	suite.Run("single zero value returns zero", func() {
		result := common.Coalesce(0)
		suite.Equal(0, result)
	})

	suite.Run("returns first non-empty string", func() {
		result := common.Coalesce("", "", "hello", "world")
		suite.Equal("hello", result)
	})

	suite.Run("returns empty string when all are empty", func() {
		result := common.Coalesce("", "", "")
		suite.Equal("", result)
	})

	suite.Run("returns first non-zero float64", func() {
		result := common.Coalesce(0.0, 0.0, 3.14)
		suite.Equal(3.14, result)
	})

	suite.Run("returns first true bool", func() {
		result := common.Coalesce(false, false, true, false)
		suite.Equal(true, result)
	})

	suite.Run("returns false when all bools are false", func() {
		result := common.Coalesce(false, false, false)
		suite.Equal(false, result)
	})
}
