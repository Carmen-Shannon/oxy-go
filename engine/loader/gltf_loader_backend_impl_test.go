package loader

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

type mockGltfImporter struct {
	result *model.ImportedModel
	err    error
}

func (m *mockGltfImporter) Import(_ string) (*model.ImportedModel, error) {
	return m.result, m.err
}

type gltfLoaderBackendImplTest struct {
	suite.Suite
	mock    *mockGltfImporter
	backend *gltfLoaderBackendImpl
}

func (suite *gltfLoaderBackendImplTest) SetupSubTest() {
	suite.mock = &mockGltfImporter{}
	suite.backend = &gltfLoaderBackendImpl{importer: suite.mock}
}

func (suite *gltfLoaderBackendImplTest) TestNewGLTFLoaderBackend() {
	suite.Run("returns non-nil backend", func() {
		b := newGLTFLoaderBackend()
		suite.NotNil(b)
	})
}

func (suite *gltfLoaderBackendImplTest) TestLoad() {
	suite.Run("delegates to importer and returns model on success", func() {
		suite.mock.result = &model.ImportedModel{Name: "fox"}
		result, err := suite.backend.Load("any.gltf")
		suite.NoError(err)
		suite.Equal("fox", result.Name)
	})

	suite.Run("propagates importer error", func() {
		suite.mock.err = errors.New("fail")
		result, err := suite.backend.Load("bad.gltf")
		suite.Error(err)
		suite.Nil(result)
	})
}

func TestGltfLoaderBackendImplTest(t *testing.T) {
	suite.Run(t, new(gltfLoaderBackendImplTest))
}
