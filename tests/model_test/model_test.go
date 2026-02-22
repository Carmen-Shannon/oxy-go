package model_test

import (
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/stretchr/testify/suite"
)

type modelTest struct {
	suite.Suite
}

func TestModel(t *testing.T) {
	suite.Run(t, new(modelTest))
}

func (suite *modelTest) TestNewModel() {
	suite.Run("default model is non-nil", func() {
		m := model.NewModel()
		suite.NotNil(m)
	})

	suite.Run("default name is empty string", func() {
		m := model.NewModel()
		suite.Equal("", m.Name())
	})

	suite.Run("default skinned is false", func() {
		m := model.NewModel()
		suite.False(m.Skinned())
	})

	suite.Run("default skeleton is nil", func() {
		m := model.NewModel()
		suite.Nil(m.Skeleton())
	})

	suite.Run("default animations is nil", func() {
		m := model.NewModel()
		suite.Nil(m.Animations())
	})

	suite.Run("default imported materials is nil", func() {
		m := model.NewModel()
		suite.Nil(m.ImportedMaterials())
	})

	suite.Run("default mesh provider is nil", func() {
		m := model.NewModel()
		suite.Nil(m.MeshProvider())
	})

	suite.Run("default compute pipeline key is empty", func() {
		m := model.NewModel()
		suite.Equal("", m.ComputePipelineKey())
	})

	suite.Run("default bounding radius is zero", func() {
		m := model.NewModel()
		suite.InDelta(0.0, m.BoundingRadius(), 1e-6)
	})

	suite.Run("default vertex data is nil", func() {
		m := model.NewModel()
		suite.Nil(m.VertexData())
	})

	suite.Run("default index data is nil", func() {
		m := model.NewModel()
		suite.Nil(m.IndexData())
	})

	suite.Run("default index count is zero", func() {
		m := model.NewModel()
		suite.Equal(0, m.IndexCount())
	})

	suite.Run("default animation count is zero", func() {
		m := model.NewModel()
		suite.Equal(0, m.AnimationCount())
	})

	suite.Run("default animation names is empty", func() {
		m := model.NewModel()
		suite.Empty(m.AnimationNames())
	})

	suite.Run("default render materials is nil", func() {
		m := model.NewModel()
		suite.Nil(m.RenderMaterials())
	})

	suite.Run("default effect provider is nil", func() {
		m := model.NewModel()
		suite.Nil(m.EffectProvider())
	})
}

func (suite *modelTest) TestWithName() {
	suite.Run("sets the model name", func() {
		m := model.NewModel(model.WithName("TestModel"))
		suite.Equal("TestModel", m.Name())
	})

	suite.Run("empty string is valid", func() {
		m := model.NewModel(model.WithName(""))
		suite.Equal("", m.Name())
	})
}

func (suite *modelTest) TestWithSkinned() {
	suite.Run("sets skinned to true", func() {
		m := model.NewModel(model.WithSkinned(true))
		suite.True(m.Skinned())
	})

	suite.Run("sets skinned to false", func() {
		m := model.NewModel(model.WithSkinned(false))
		suite.False(m.Skinned())
	})
}

func (suite *modelTest) TestWithSkeleton() {
	suite.Run("sets the skeleton", func() {
		skel := &model.Skeleton{
			Bones: []model.Bone{
				{Name: "root", ParentIndex: -1},
			},
			RootBoneIndices: []int32{0},
			BoneNameToIndex: map[string]int32{"root": 0},
		}
		m := model.NewModel(model.WithSkeleton(skel))
		suite.NotNil(m.Skeleton())
		suite.Equal("root", m.Skeleton().Bones[0].Name)
	})

	suite.Run("nil skeleton is valid", func() {
		m := model.NewModel(model.WithSkeleton(nil))
		suite.Nil(m.Skeleton())
	})
}

func (suite *modelTest) TestWithAnimations() {
	suite.Run("sets animation clips", func() {
		anims := []*model.AnimationClip{
			{Name: "walk", Duration: 1.0, TicksPerSecond: 30},
			{Name: "run", Duration: 0.5, TicksPerSecond: 30},
		}
		m := model.NewModel(model.WithAnimations(anims))
		suite.Len(m.Animations(), 2)
		suite.Equal("walk", m.Animations()[0].Name)
		suite.Equal("run", m.Animations()[1].Name)
	})

	suite.Run("nil animations is valid", func() {
		m := model.NewModel(model.WithAnimations(nil))
		suite.Nil(m.Animations())
	})
}

func (suite *modelTest) TestWithImportedMaterials() {
	suite.Run("sets imported materials", func() {
		mats := []common.ImportedMaterial{
			{Name: "material_0"},
			{Name: "material_1"},
		}
		m := model.NewModel(model.WithImportedMaterials(mats))
		suite.Len(m.ImportedMaterials(), 2)
		suite.Equal("material_0", m.ImportedMaterials()[0].Name)
	})

	suite.Run("empty slice is valid", func() {
		m := model.NewModel(model.WithImportedMaterials([]common.ImportedMaterial{}))
		suite.Empty(m.ImportedMaterials())
	})
}

func (suite *modelTest) TestWithMeshProvider() {
	suite.Run("sets the mesh provider", func() {
		provider := bind_group_provider.NewBindGroupProvider("test_mesh")
		m := model.NewModel(model.WithMeshProvider(provider))
		suite.NotNil(m.MeshProvider())
	})

	suite.Run("nil provider is valid", func() {
		m := model.NewModel(model.WithMeshProvider(nil))
		suite.Nil(m.MeshProvider())
	})
}

func (suite *modelTest) TestWithBoundingRadius() {
	suite.Run("sets the bounding radius", func() {
		m := model.NewModel(model.WithBoundingRadius(5.5))
		suite.InDelta(5.5, m.BoundingRadius(), 1e-6)
	})

	suite.Run("zero radius is valid", func() {
		m := model.NewModel(model.WithBoundingRadius(0.0))
		suite.InDelta(0.0, m.BoundingRadius(), 1e-6)
	})
}

func (suite *modelTest) TestWithComputePipelineKey() {
	suite.Run("sets the compute pipeline key", func() {
		m := model.NewModel(model.WithComputePipelineKey("skinning_pipeline"))
		suite.Equal("skinning_pipeline", m.ComputePipelineKey())
	})
}

func (suite *modelTest) TestWithVertexData() {
	suite.Run("sets vertex data bytes", func() {
		data := []byte{1, 2, 3, 4, 5, 6, 7, 8}
		m := model.NewModel(model.WithVertexData(data))
		suite.Equal(data, m.VertexData())
	})

	suite.Run("nil vertex data is valid", func() {
		m := model.NewModel(model.WithVertexData(nil))
		suite.Nil(m.VertexData())
	})
}

func (suite *modelTest) TestWithIndexData() {
	suite.Run("sets index data bytes", func() {
		data := []byte{0, 1, 0, 2, 0, 3}
		m := model.NewModel(model.WithIndexData(data))
		suite.Equal(data, m.IndexData())
	})

	suite.Run("nil index data is valid", func() {
		m := model.NewModel(model.WithIndexData(nil))
		suite.Nil(m.IndexData())
	})
}

func (suite *modelTest) TestWithIndexCount() {
	suite.Run("sets the index count", func() {
		m := model.NewModel(model.WithIndexCount(36))
		suite.Equal(36, m.IndexCount())
	})

	suite.Run("zero index count is valid", func() {
		m := model.NewModel(model.WithIndexCount(0))
		suite.Equal(0, m.IndexCount())
	})
}

func (suite *modelTest) TestWithRenderMaterials() {
	suite.Run("sets render materials", func() {
		mat := material.NewMaterial(material.WithName("test_mat"))
		m := model.NewModel(model.WithRenderMaterials(mat))
		suite.Len(m.RenderMaterials(), 1)
	})

	suite.Run("no render materials results in empty slice", func() {
		m := model.NewModel(model.WithRenderMaterials())
		suite.Empty(m.RenderMaterials())
	})
}

func (suite *modelTest) TestMultipleBuilderOptions() {
	suite.Run("all options applied in order", func() {
		skel := &model.Skeleton{
			Bones:           []model.Bone{{Name: "hip", ParentIndex: -1}},
			RootBoneIndices: []int32{0},
			BoneNameToIndex: map[string]int32{"hip": 0},
		}
		anims := []*model.AnimationClip{{Name: "idle", Duration: 2.0}}
		mats := []common.ImportedMaterial{{Name: "skin"}}

		m := model.NewModel(
			model.WithName("character"),
			model.WithSkinned(true),
			model.WithSkeleton(skel),
			model.WithAnimations(anims),
			model.WithImportedMaterials(mats),
			model.WithBoundingRadius(2.5),
			model.WithComputePipelineKey("compute_skin"),
			model.WithVertexData([]byte{0xFF}),
			model.WithIndexData([]byte{0x00}),
			model.WithIndexCount(3),
		)

		suite.Equal("character", m.Name())
		suite.True(m.Skinned())
		suite.NotNil(m.Skeleton())
		suite.Len(m.Animations(), 1)
		suite.Len(m.ImportedMaterials(), 1)
		suite.InDelta(2.5, m.BoundingRadius(), 1e-6)
		suite.Equal("compute_skin", m.ComputePipelineKey())
		suite.Equal([]byte{0xFF}, m.VertexData())
		suite.Equal([]byte{0x00}, m.IndexData())
		suite.Equal(3, m.IndexCount())
	})

	suite.Run("later options override earlier ones", func() {
		m := model.NewModel(
			model.WithName("first"),
			model.WithName("second"),
		)
		suite.Equal("second", m.Name())
	})
}

func (suite *modelTest) TestAnimationCount() {
	suite.Run("returns zero for no animations", func() {
		m := model.NewModel()
		suite.Equal(0, m.AnimationCount())
	})

	suite.Run("returns correct count for multiple animations", func() {
		anims := []*model.AnimationClip{
			{Name: "a"},
			{Name: "b"},
			{Name: "c"},
		}
		m := model.NewModel(model.WithAnimations(anims))
		suite.Equal(3, m.AnimationCount())
	})
}

func (suite *modelTest) TestAnimationNames() {
	suite.Run("returns empty slice for no animations", func() {
		m := model.NewModel()
		suite.Empty(m.AnimationNames())
	})

	suite.Run("returns names in order", func() {
		anims := []*model.AnimationClip{
			{Name: "walk"},
			{Name: "run"},
			{Name: "jump"},
		}
		m := model.NewModel(model.WithAnimations(anims))
		names := m.AnimationNames()
		suite.Equal([]string{"walk", "run", "jump"}, names)
	})
}

func (suite *modelTest) TestGetAnimationIndex() {
	suite.Run("returns -1 for no animations", func() {
		m := model.NewModel()
		suite.Equal(-1, m.GetAnimationIndex("walk"))
	})

	suite.Run("returns correct index for existing animation", func() {
		anims := []*model.AnimationClip{
			{Name: "idle"},
			{Name: "walk"},
			{Name: "run"},
		}
		m := model.NewModel(model.WithAnimations(anims))
		suite.Equal(0, m.GetAnimationIndex("idle"))
		suite.Equal(1, m.GetAnimationIndex("walk"))
		suite.Equal(2, m.GetAnimationIndex("run"))
	})

	suite.Run("returns -1 for non-existent animation", func() {
		anims := []*model.AnimationClip{{Name: "walk"}}
		m := model.NewModel(model.WithAnimations(anims))
		suite.Equal(-1, m.GetAnimationIndex("fly"))
	})

	suite.Run("returns -1 for empty name", func() {
		anims := []*model.AnimationClip{{Name: "walk"}}
		m := model.NewModel(model.WithAnimations(anims))
		suite.Equal(-1, m.GetAnimationIndex(""))
	})
}

func (suite *modelTest) TestSetComputePipelineKey() {
	suite.Run("updates the compute pipeline key", func() {
		m := model.NewModel(model.WithComputePipelineKey("old"))
		m.SetComputePipelineKey("new")
		suite.Equal("new", m.ComputePipelineKey())
	})

	suite.Run("can set to empty string", func() {
		m := model.NewModel(model.WithComputePipelineKey("key"))
		m.SetComputePipelineKey("")
		suite.Equal("", m.ComputePipelineKey())
	})
}

func (suite *modelTest) TestSetRenderMaterials() {
	suite.Run("replaces render materials", func() {
		mat1 := material.NewMaterial(material.WithName("mat1"))
		mat2 := material.NewMaterial(material.WithName("mat2"))
		m := model.NewModel(model.WithRenderMaterials(mat1))
		m.SetRenderMaterials([]material.Material{mat2})
		suite.Len(m.RenderMaterials(), 1)
	})

	suite.Run("setting nil clears render materials", func() {
		mat := material.NewMaterial(material.WithName("mat"))
		m := model.NewModel(model.WithRenderMaterials(mat))
		m.SetRenderMaterials(nil)
		suite.Nil(m.RenderMaterials())
	})
}

func (suite *modelTest) TestSetEffectProvider() {
	suite.Run("sets and retrieves effect provider", func() {
		provider := bind_group_provider.NewBindGroupProvider("effect")
		m := model.NewModel()
		m.SetEffectProvider(provider)
		suite.NotNil(m.EffectProvider())
	})

	suite.Run("setting nil clears effect provider", func() {
		provider := bind_group_provider.NewBindGroupProvider("effect")
		m := model.NewModel()
		m.SetEffectProvider(provider)
		m.SetEffectProvider(nil)
		suite.Nil(m.EffectProvider())
	})
}

func (suite *modelTest) TestSetVertexData() {
	suite.Run("replaces vertex data", func() {
		m := model.NewModel(model.WithVertexData([]byte{1, 2, 3}))
		m.SetVertexData([]byte{4, 5, 6, 7})
		suite.Equal([]byte{4, 5, 6, 7}, m.VertexData())
	})

	suite.Run("setting nil clears vertex data", func() {
		m := model.NewModel(model.WithVertexData([]byte{1}))
		m.SetVertexData(nil)
		suite.Nil(m.VertexData())
	})
}

func (suite *modelTest) TestSetIndexData() {
	suite.Run("replaces index data", func() {
		m := model.NewModel(model.WithIndexData([]byte{1, 2, 3}))
		m.SetIndexData([]byte{10, 20})
		suite.Equal([]byte{10, 20}, m.IndexData())
	})

	suite.Run("setting nil clears index data", func() {
		m := model.NewModel(model.WithIndexData([]byte{1}))
		m.SetIndexData(nil)
		suite.Nil(m.IndexData())
	})
}

func (suite *modelTest) TestSetIndexCount() {
	suite.Run("replaces index count", func() {
		m := model.NewModel(model.WithIndexCount(6))
		m.SetIndexCount(12)
		suite.Equal(12, m.IndexCount())
	})

	suite.Run("setting zero is valid", func() {
		m := model.NewModel(model.WithIndexCount(100))
		m.SetIndexCount(0)
		suite.Equal(0, m.IndexCount())
	})
}

func (suite *modelTest) TestSetDelegate() {
	suite.Run("model has delegate set to itself after construction", func() {
		m := model.NewModel(model.WithName("self"))
		// Calling any method confirms the delegate is functional
		suite.Equal("self", m.Name())
	})
}
