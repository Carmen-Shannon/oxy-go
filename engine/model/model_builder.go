package model

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
)

// ModelBuilderOption is a functional option for configuring a Model via NewModel.
type ModelBuilderOption func(*model)

// WithName is an option builder that sets the name of the Model.
//
// Parameters:
//   - name: the model identifier
//
// Returns:
//   - ModelBuilderOption: a function that applies the name option to a model
func WithName(name string) ModelBuilderOption {
	return func(m *model) {
		m.name = name
	}
}

// WithSkinned is an option builder that sets whether the Model uses skeletal animation.
//
// Parameters:
//   - skinned: true if the model has bone data
//
// Returns:
//   - ModelBuilderOption: a function that applies the skinned option to a model
func WithSkinned(skinned bool) ModelBuilderOption {
	return func(m *model) {
		m.skinned = skinned
	}
}

// WithSkeleton is an option builder that sets the bone hierarchy of the Model.
//
// Parameters:
//   - skeleton: the skeleton to set
//
// Returns:
//   - ModelBuilderOption: a function that applies the skeleton option to a model
func WithSkeleton(skeleton *Skeleton) ModelBuilderOption {
	return func(m *model) {
		m.skeleton = skeleton
	}
}

// WithAnimations is an option builder that sets the animation clips of the Model.
//
// Parameters:
//   - animations: the animation clips to set
//
// Returns:
//   - ModelBuilderOption: a function that applies the animations option to a model
func WithAnimations(animations []*AnimationClip) ModelBuilderOption {
	return func(m *model) {
		m.animations = animations
	}
}

// WithImportedMaterials is an option builder that sets the raw imported materials of the Model.
//
// Parameters:
//   - materials: the imported materials to set
//
// Returns:
//   - ModelBuilderOption: a function that applies the imported materials option to a model
func WithImportedMaterials(materials []common.ImportedMaterial) ModelBuilderOption {
	return func(m *model) {
		m.importedMaterials = materials
	}
}

// WithMeshProvider is an option builder that sets the BindGroupProvider for mesh GPU resources.
//
// Parameters:
//   - provider: the BindGroupProvider holding vertex/index buffers and bind group data
//
// Returns:
//   - ModelBuilderOption: a function that applies the mesh provider option to a model
func WithMeshProvider(provider bind_group_provider.BindGroupProvider) ModelBuilderOption {
	return func(m *model) {
		m.meshProvider = provider
	}
}

// WithBoundingRadius is an option builder that manually sets the bounding sphere radius.
// Use this to override the auto-computed value from ComputeBoundingRadius when a manually
// tuned conservative bound is preferred.
//
// Parameters:
//   - radius: the bounding radius to set
//
// Returns:
//   - ModelBuilderOption: a function that applies the bounding radius option to a model
func WithBoundingRadius(radius float32) ModelBuilderOption {
	return func(m *model) {
		m.boundingRadius = radius
	}
}

// WithRenderMaterials is an option builder that sets the render-ready materials for the Model.
// These are GPU-configured Material instances used during DrawCalls.
//
// Parameters:
//   - mats: the render-ready materials to set
//
// Returns:
//   - ModelBuilderOption: a function that applies the render materials option to a model
func WithRenderMaterials(mats ...material.Material) ModelBuilderOption {
	return func(m *model) {
		m.renderMaterials = mats
	}
}

// WithComputePipelineKey is an option builder that sets the compute pipeline key for this model's animator.
// This is used to look up the correct pipeline when creating or updating the animator.
//
// Parameters:
//   - key: the compute pipeline key to set
//
// Returns:
//   - ModelBuilderOption: a function that applies the compute pipeline key option to a model
func WithComputePipelineKey(key string) ModelBuilderOption {
	return func(m *model) {
		m.computePipelineKey = key
	}
}

// WithVertexData is an option builder that sets the raw vertex data for this model's mesh.
//
// Parameters:
//   - data: the vertex data to set
//
// Returns:
//   - ModelBuilderOption: a function that applies the vertex data option to a model
func WithVertexData(data []byte) ModelBuilderOption {
	return func(m *model) {
		m.vertexData = data
	}
}

// WithIndexData is an option builder that sets the raw index data for this model's mesh.
//
// Parameters:
//   - data: the index data to set
//
// Returns:
//   - ModelBuilderOption: a function that applies the index data option to a model
func WithIndexData(data []byte) ModelBuilderOption {
	return func(m *model) {
		m.indexData = data
	}
}

// WithIndexCount is an option builder that sets the number of indices in the model's mesh.
//
// Parameters:
//   - count: the index count to set
//
// Returns:
//   - ModelBuilderOption: a function that applies the index count option to a model
func WithIndexCount(count int) ModelBuilderOption {
	return func(m *model) {
		m.indexCount = count
	}
}
