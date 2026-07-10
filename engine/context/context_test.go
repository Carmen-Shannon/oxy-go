package context_test

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/context"
	"github.com/Carmen-Shannon/oxy-go/engine/scene"
	scene_mocks "github.com/Carmen-Shannon/oxy-go/engine/scene/mocks"
)

func TestRunContextTests(t *testing.T) {
	suite.Run(t, new(contextTest))
}

type contextTest struct {
	suite.Suite
}

func (suite *contextTest) TestNewContext() {
	suite.Run("should create a context with default values", func() {
		ctx := context.NewContext()
		suite.NotNil(ctx)

		scenes := ctx.Scenes()
		suite.NotNil(scenes)
		suite.Empty(scenes)

		val, exists := ctx.Get("any-key")
		suite.False(exists)
		suite.Nil(val)
	})

	suite.Run("should apply WithScenes option", func() {
		mockScene := scene_mocks.NewMockScene(suite.T())
		scenes := map[int]scene.Scene{
			0: mockScene,
		}
		ctx := context.NewContext(context.WithScenes(scenes))
		suite.NotNil(ctx)
		suite.Equal(scenes, ctx.Scenes())
	})
}

func (suite *contextTest) TestScene() {
	suite.Run("should return scene for existing key", func() {
		mockScene := scene_mocks.NewMockScene(suite.T())
		scenes := map[int]scene.Scene{
			0: mockScene,
		}
		ctx := context.NewContext(context.WithScenes(scenes))
		s, exists := ctx.Scene(0)
		suite.True(exists)
		suite.Equal(mockScene, s)
	})

	suite.Run("should return false for missing key", func() {
		ctx := context.NewContext()
		s, exists := ctx.Scene(42)
		suite.False(exists)
		suite.Nil(s)
	})
}

func (suite *contextTest) TestScenes() {
	suite.Run("should return all registered scenes", func() {
		mockScene0 := scene_mocks.NewMockScene(suite.T())
		mockScene1 := scene_mocks.NewMockScene(suite.T())
		scenes := map[int]scene.Scene{
			0: mockScene0,
			1: mockScene1,
		}
		ctx := context.NewContext(context.WithScenes(scenes))
		result := ctx.Scenes()
		suite.Equal(2, len(result))
		suite.Equal(mockScene0, result[0])
		suite.Equal(mockScene1, result[1])
	})
}

func (suite *contextTest) TestSetScenes() {
	suite.Run("should replace all scenes and copy the input map", func() {
		mockScene0 := scene_mocks.NewMockScene(suite.T())
		mockScene1 := scene_mocks.NewMockScene(suite.T())
		original := map[int]scene.Scene{
			0: mockScene0,
		}

		ctx := context.NewContext(context.WithScenes(original))
		suite.Equal(1, len(ctx.Scenes()))

		replacement := map[int]scene.Scene{
			1: mockScene1,
		}
		ctx.SetScenes(replacement)
		suite.Equal(1, len(ctx.Scenes()))
		suite.Equal(mockScene1, ctx.Scenes()[1])
		_, exists := ctx.Scenes()[0]
		suite.False(exists)
	})

	suite.Run("should not reflect mutations to the original map after SetScenes", func() {
		mockScene := scene_mocks.NewMockScene(suite.T())
		original := map[int]scene.Scene{
			0: mockScene,
		}
		ctx := context.NewContext()
		ctx.SetScenes(original)

		original[99] = scene_mocks.NewMockScene(suite.T())

		suite.Equal(1, len(ctx.Scenes()))
		_, exists := ctx.Scenes()[99]
		suite.False(exists)
	})
}

func (suite *contextTest) TestGet() {
	suite.Run("should return value for existing key", func() {
		ctx := context.NewContext()
		ctx.Set("hello", "world")
		val, exists := ctx.Get("hello")
		suite.True(exists)
		suite.Equal("world", val)
	})

	suite.Run("should return false for missing key", func() {
		ctx := context.NewContext()
		val, exists := ctx.Get("nonexistent")
		suite.False(exists)
		suite.Nil(val)
	})
}

func (suite *contextTest) TestSet() {
	suite.Run("should store and retrieve a value", func() {
		ctx := context.NewContext()
		ctx.Set("key", 42)
		val, exists := ctx.Get("key")
		suite.True(exists)
		suite.Equal(42, val)
	})

	suite.Run("should overwrite existing value", func() {
		ctx := context.NewContext()
		ctx.Set("key", "first")
		ctx.Set("key", "second")
		val, exists := ctx.Get("key")
		suite.True(exists)
		suite.Equal("second", val)
	})
}
