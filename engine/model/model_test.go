package model_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/stretchr/testify/suite"
)

func TestRunModelTests(t *testing.T) {
	suite.Run(t, new(modelTest))
}

type modelTest struct {
	suite.Suite
	model model.Model
}

func (suite *modelTest) SetupSubTest() {
	suite.model = model.NewModel()
}

func (suite *modelTest) TestNewModel() {
	suite.Run("should return non-nil model with default CastsShadows true", func() {
		suite.NotNil(suite.model)
		suite.True(suite.model.CastsShadows())
	})
}

func (suite *modelTest) TestWithName() {
	suite.Run("should set and return the model name", func() {
		m := model.NewModel(model.WithName("hero"))
		suite.Equal("hero", m.Name())
	})
}

func (suite *modelTest) TestName() {
	suite.Run("should return empty string by default", func() {
		suite.Equal("", suite.model.Name())
	})
}

func (suite *modelTest) TestWithSkinned() {
	suite.Run("should set and return true when WithSkinned(true)", func() {
		m := model.NewModel(model.WithSkinned(true))
		suite.True(m.Skinned())
	})
}

func (suite *modelTest) TestSkinned() {
	suite.Run("should return false by default", func() {
		suite.False(suite.model.Skinned())
	})
}

func (suite *modelTest) TestWithSkeleton() {
	suite.Run("should set and return the skeleton", func() {
		sk := &model.Skeleton{Bones: []model.Bone{{Name: "root"}}}
		m := model.NewModel(model.WithSkeleton(sk))
		suite.Equal(sk, m.Skeleton())
	})
}

func (suite *modelTest) TestSkeleton() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.model.Skeleton())
	})
}

func (suite *modelTest) TestWithAnimations() {
	suite.Run("should set and return animation clips", func() {
		clips := []*model.AnimationClip{{Name: "walk"}, {Name: "run"}}
		m := model.NewModel(model.WithAnimations(clips))
		suite.Equal(clips, m.Animations())
	})
}

func (suite *modelTest) TestAnimations() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.model.Animations())
	})
}

func (suite *modelTest) TestWithImportedMaterials() {
	suite.Run("should set and return imported materials", func() {
		mats := []common.ImportedMaterial{{Name: "mat0"}}
		m := model.NewModel(model.WithImportedMaterials(mats))
		suite.Equal(mats, m.ImportedMaterials())
	})
}

func (suite *modelTest) TestImportedMaterials() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.model.ImportedMaterials())
	})
}

func (suite *modelTest) TestWithMeshProvider() {
	suite.Run("should set and return the mesh provider", func() {
		p := bind_group_provider.NewBindGroupProvider("test")
		m := model.NewModel(model.WithMeshProvider(p))
		suite.Equal(p, m.MeshProvider())
	})
}

func (suite *modelTest) TestMeshProvider() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.model.MeshProvider())
	})
}

func (suite *modelTest) TestWithComputePipelineKey() {
	suite.Run("should set and return via WithComputePipelineKey", func() {
		m := model.NewModel(model.WithComputePipelineKey("skinned-compute"))
		suite.Equal("skinned-compute", m.ComputePipelineKey())
	})
}

func (suite *modelTest) TestComputePipelineKey() {
	suite.Run("should return empty string by default", func() {
		suite.Equal("", suite.model.ComputePipelineKey())
	})
}

func (suite *modelTest) TestSetComputePipelineKey() {
	suite.Run("should update the compute pipeline key", func() {
		suite.model.SetComputePipelineKey("new-key")
		suite.Equal("new-key", suite.model.ComputePipelineKey())
	})
}

func (suite *modelTest) TestAnimationCount() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.model.AnimationCount())
	})

	suite.Run("should return the number of animation clips", func() {
		clips := []*model.AnimationClip{{Name: "walk"}, {Name: "run"}}
		m := model.NewModel(model.WithAnimations(clips))
		suite.Equal(2, m.AnimationCount())
	})
}

func (suite *modelTest) TestAnimationNames() {
	suite.Run("should return empty slice by default", func() {
		suite.Empty(suite.model.AnimationNames())
	})

	suite.Run("should return names of all animation clips", func() {
		clips := []*model.AnimationClip{{Name: "walk"}, {Name: "run"}}
		m := model.NewModel(model.WithAnimations(clips))
		suite.Equal([]string{"walk", "run"}, m.AnimationNames())
	})
}

func (suite *modelTest) TestGetAnimationIndex() {
	suite.Run("should return -1 when animation not found", func() {
		suite.Equal(-1, suite.model.GetAnimationIndex("idle"))
	})

	suite.Run("should return the correct index when found", func() {
		clips := []*model.AnimationClip{{Name: "walk"}, {Name: "run"}}
		m := model.NewModel(model.WithAnimations(clips))
		suite.Equal(1, m.GetAnimationIndex("run"))
	})
}

func (suite *modelTest) TestWithVertexData() {
	suite.Run("should set and return vertex data", func() {
		data := []byte{1, 2, 3, 4}
		m := model.NewModel(model.WithVertexData(data))
		suite.Equal(data, m.VertexData())
	})
}

func (suite *modelTest) TestVertexData() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.model.VertexData())
	})
}

func (suite *modelTest) TestSetVertexData() {
	suite.Run("should update vertex data", func() {
		data := []byte{5, 6, 7, 8}
		suite.model.SetVertexData(data)
		suite.Equal(data, suite.model.VertexData())
	})
}

func (suite *modelTest) TestWithIndexData() {
	suite.Run("should set and return index data", func() {
		data := []byte{9, 10, 11, 12}
		m := model.NewModel(model.WithIndexData(data))
		suite.Equal(data, m.IndexData())
	})
}

func (suite *modelTest) TestIndexData() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.model.IndexData())
	})
}

func (suite *modelTest) TestSetIndexData() {
	suite.Run("should update index data", func() {
		data := []byte{1, 2, 3}
		suite.model.SetIndexData(data)
		suite.Equal(data, suite.model.IndexData())
	})
}

func (suite *modelTest) TestWithIndexCount() {
	suite.Run("should set and return the index count", func() {
		m := model.NewModel(model.WithIndexCount(42))
		suite.Equal(42, m.IndexCount())
	})
}

func (suite *modelTest) TestIndexCount() {
	suite.Run("should return 0 by default", func() {
		suite.Equal(0, suite.model.IndexCount())
	})
}

func (suite *modelTest) TestSetIndexCount() {
	suite.Run("should update index count", func() {
		suite.model.SetIndexCount(100)
		suite.Equal(100, suite.model.IndexCount())
	})
}

func (suite *modelTest) TestWithRenderMaterials() {
	suite.Run("should set and return render materials", func() {
		mat := material.NewMaterial()
		m := model.NewModel(model.WithRenderMaterials(mat))
		suite.Len(m.RenderMaterials(), 1)
		suite.Equal(mat, m.RenderMaterials()[0])
	})
}

func (suite *modelTest) TestRenderMaterials() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.model.RenderMaterials())
	})
}

func (suite *modelTest) TestSetRenderMaterials() {
	suite.Run("should update render materials", func() {
		mat := material.NewMaterial()
		suite.model.SetRenderMaterials([]material.Material{mat})
		suite.Len(suite.model.RenderMaterials(), 1)
		suite.Equal(mat, suite.model.RenderMaterials()[0])
	})
}

func (suite *modelTest) TestWithBoundingRadius() {
	suite.Run("should set and return the bounding radius", func() {
		m := model.NewModel(model.WithBoundingRadius(5.5))
		suite.InDelta(float32(5.5), m.BoundingRadius(), 1e-6)
	})
}

func (suite *modelTest) TestBoundingRadius() {
	suite.Run("should return 0 by default", func() {
		suite.InDelta(float32(0), suite.model.BoundingRadius(), 1e-6)
	})
}

func (suite *modelTest) TestEffectProvider() {
	suite.Run("should return nil by default", func() {
		suite.Nil(suite.model.EffectProvider())
	})
}

func (suite *modelTest) TestSetEffectProvider() {
	suite.Run("should set and return the effect provider", func() {
		p := bind_group_provider.NewBindGroupProvider("effect")
		suite.model.SetEffectProvider(p)
		suite.Equal(p, suite.model.EffectProvider())
	})
}

func (suite *modelTest) TestWithCastsShadows() {
	suite.Run("should default to true", func() {
		suite.True(suite.model.CastsShadows())
	})

	suite.Run("should allow setting to false via WithCastsShadows(false)", func() {
		m := model.NewModel(model.WithCastsShadows(false))
		suite.False(m.CastsShadows())
	})
}

func (suite *modelTest) TestCastsShadows() {
	suite.Run("should default to true", func() {
		suite.True(suite.model.CastsShadows())
	})
}

func (suite *modelTest) TestSetCastsShadows() {
	suite.Run("should update castsShadows", func() {
		suite.model.SetCastsShadows(false)
		suite.False(suite.model.CastsShadows())
	})
}

func (suite *modelTest) TestWithShadowCullMode() {
	suite.Run("should return zero value by default", func() {
		suite.Equal(model.ShadowCullMode(0), suite.model.ShadowCullMode())
	})

	suite.Run("should set and return the shadow cull mode", func() {
		m := model.NewModel(model.WithShadowCullMode(model.ShadowCullModeFront))
		suite.Equal(model.ShadowCullModeFront, m.ShadowCullMode())
	})
}

func (suite *modelTest) TestShadowCullMode() {
	suite.Run("should return zero value by default", func() {
		suite.Equal(model.ShadowCullMode(0), suite.model.ShadowCullMode())
	})
}

func (suite *modelTest) TestSetShadowCullMode() {
	suite.Run("should update shadow cull mode", func() {
		suite.model.SetShadowCullMode(model.ShadowCullModeNone)
		suite.Equal(model.ShadowCullModeNone, suite.model.ShadowCullMode())
	})
}
