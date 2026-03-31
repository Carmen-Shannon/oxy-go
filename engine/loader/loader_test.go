package loader_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/loader"
	mocks "github.com/Carmen-Shannon/oxy-go/engine/model/mocks"
	"github.com/stretchr/testify/suite"
)

func TestRunLoaderTests(t *testing.T) {
	suite.Run(t, new(loaderTest))
}

type loaderTest struct {
	suite.Suite
	loader loader.Loader
}

func (suite *loaderTest) SetupSubTest() {
	suite.loader = loader.NewLoader(loader.BackendTypeGLTF)
}

func (suite *loaderTest) TestNewLoader() {
	suite.Run("returns non-nil loader for GLTF backend", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		suite.NotNil(l)
	})
}

func (suite *loaderTest) TestWithModel() {
	suite.Run("pre-populates model cache so Get returns the model", func() {
		mockMdl := mocks.NewMockModel(suite.T())
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("key", mockMdl))
		suite.NotNil(l.Get("key"))
	})
}

func (suite *loaderTest) TestModels() {
	suite.Run("empty map initially", func() {
		m := suite.loader.Models()
		suite.NotNil(m)
		suite.Len(m, 0)
	})

	suite.Run("returns populated map after WithModel", func() {
		mockMdl := mocks.NewMockModel(suite.T())
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("mykey", mockMdl))
		m := l.Models()
		suite.Len(m, 1)
		suite.NotNil(m["mykey"])
	})
}

func (suite *loaderTest) TestGet() {
	suite.Run("returns nil for missing key", func() {
		suite.Nil(suite.loader.Get("nonexistent"))
	})

	suite.Run("returns model for known key", func() {
		mockMdl := mocks.NewMockModel(suite.T())
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("k", mockMdl))
		suite.NotNil(l.Get("k"))
	})
}

func (suite *loaderTest) TestLoad() {
	suite.Run("returns cached model on second call", func() {
		mockMdl := mocks.NewMockModel(suite.T())
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("myfile.gltf", mockMdl))
		result, err := l.Load("myfile.gltf")
		suite.NoError(err)
		suite.NotNil(result)
	})

	suite.Run("returns error for unsupported extension", func() {
		result, err := suite.loader.Load("model.xyz")
		suite.Nil(result)
		suite.Error(err)
	})

	suite.Run("returns error when backend fails for valid extension", func() {
		result, err := suite.loader.Load("nonexistent.gltf")
		suite.Nil(result)
		suite.Error(err)
	})
}
