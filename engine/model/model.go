package model

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
)

// model is the implementation of the Model interface.
type model struct {
	name                  string
	skinned               bool
	skeleton              *Skeleton
	animations            []*AnimationClip
	importedMaterials     []common.ImportedMaterial
	renderMaterials       []material.Material
	meshProvider          bind_group_provider.BindGroupProvider
	effectProvider        bind_group_provider.BindGroupProvider
	computePipelineKey    string
	boundingRadius        float32
	vertexData, indexData []byte
	indexCount            int
}

// Model defines the interface for a loaded 3D model.
// A Model is a GPU-ready container holding mesh data via a BindGroupProvider,
// skeleton hierarchy, animation clips, and material properties.
// It is produced by the Loader after importing and processing a model file.
type Model interface {
	// Name retrieves the model identifier.
	//
	// Returns:
	//   - string: the model name
	Name() string

	// Skinned reports whether this model uses skeletal animation.
	//
	// Returns:
	//   - bool: true if the model has bone data
	Skinned() bool

	// Skeleton retrieves the bone hierarchy for this model.
	// Returns nil for static (non-skinned) models.
	//
	// Returns:
	//   - *Skeleton: the skeleton or nil
	Skeleton() *Skeleton

	// Animations retrieves all animation clips bundled with this model.
	//
	// Returns:
	//   - []*AnimationClip: the animation clips
	Animations() []*AnimationClip

	// ImportedMaterials retrieves the raw material properties imported from the model file.
	//
	// Returns:
	//   - []common.ImportedMaterial: the imported materials
	ImportedMaterials() []common.ImportedMaterial

	// MeshProvider retrieves the BindGroupProvider holding GPU mesh resources.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the mesh provider
	MeshProvider() bind_group_provider.BindGroupProvider

	// ComputePipelineKey returns the unique key for the compute pipeline used by this model's animator.
	// This is used to look up the correct pipeline when creating or updating the animator.
	//
	// Returns:
	//   - string: the compute pipeline key
	ComputePipelineKey() string

	// AnimationCount returns the number of available animation clips.
	//
	// Returns:
	//   - int: the animation count
	AnimationCount() int

	// AnimationNames returns the names of all animation clips.
	//
	// Returns:
	//   - []string: the animation clip names
	AnimationNames() []string

	// VertexData returns the raw vertex data for this model's mesh.
	//
	// Returns:
	//   - []byte: the vertex data
	VertexData() []byte

	// IndexData returns the raw index data for this model's mesh.
	//
	// Returns:
	//   - []byte: the index data
	IndexData() []byte

	// IndexCount returns the number of indices in the model's mesh.
	//
	// Returns:
	//   - int: the index count
	IndexCount() int

	// GetAnimationIndex returns the index of an animation by name, or -1 if not found.
	//
	// Parameters:
	//   - name: the animation clip name to search for
	//
	// Returns:
	//   - int: the animation index, or -1 if not found
	GetAnimationIndex(name string) int

	// RenderMaterials retrieves the render-ready materials for this model.
	// These are GPU-configured Material instances used during DrawCalls,
	// as opposed to the raw common.ImportedMaterial data from the loader.
	//
	// Returns:
	//   - []material.Material: the render-ready materials
	RenderMaterials() []material.Material

	// SetRenderMaterials replaces the render-ready material list for this model.
	//
	// Parameters:
	//   - mats: the render-ready materials to set
	SetRenderMaterials(mats []material.Material)

	// BoundingRadius returns the bounding sphere radius for this model, measured as
	// the maximum vertex distance from the origin. Used by frustum culling.
	//
	// Returns:
	//   - float32: the bounding radius
	BoundingRadius() float32

	// SetComputePipelineKey sets the compute pipeline key for this model's animator.
	//
	// Parameters:
	//   - key: the compute pipeline key to set
	SetComputePipelineKey(key string)

	// EffectProvider returns the bind group provider for per-model GPU effect
	// parameters (e.g. a tint uniform), or nil if none has been configured.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the effect provider, or nil
	EffectProvider() bind_group_provider.BindGroupProvider

	// SetEffectProvider assigns a bind group provider for per-model GPU effect parameters.
	//
	// Parameters:
	//   - provider: the effect bind group provider to associate
	SetEffectProvider(provider bind_group_provider.BindGroupProvider)

	// SetVertexData sets the raw vertex data for this model's mesh.
	//
	// Parameters:
	//   - data: the vertex data to set
	SetVertexData(data []byte)

	// SetIndexData sets the raw index data for this model's mesh.
	//
	// Parameters:
	//   - data: the index data to set
	SetIndexData(data []byte)

	// SetIndexCount sets the number of indices in the model's mesh.
	//
	// Parameters:
	//   - count: the index count to set
	SetIndexCount(count int)
}

var _ Model = &model{}

// NewModel creates a new Model instance with the specified options applied.
//
// Parameters:
//   - options: a variadic list of ModelBuilderOption functions to configure the Model
//
// Returns:
//   - Model: a new instance of Model configured with the provided options
func NewModel(options ...ModelBuilderOption) Model {
	m := &model{}
	for _, opt := range options {
		opt(m)
	}
	return m
}

func (m *model) Name() string {
	return m.name
}

func (m *model) Skinned() bool {
	return m.skinned
}

func (m *model) Skeleton() *Skeleton {
	return m.skeleton
}

func (m *model) Animations() []*AnimationClip {
	return m.animations
}

func (m *model) ImportedMaterials() []common.ImportedMaterial {
	return m.importedMaterials
}

func (m *model) MeshProvider() bind_group_provider.BindGroupProvider {
	return m.meshProvider
}

func (m *model) ComputePipelineKey() string {
	return m.computePipelineKey
}

func (m *model) SetComputePipelineKey(key string) {
	m.computePipelineKey = key
}

func (m *model) AnimationCount() int {
	return len(m.animations)
}

func (m *model) AnimationNames() []string {
	names := make([]string, len(m.animations))
	for i, anim := range m.animations {
		names[i] = anim.Name
	}
	return names
}

func (m *model) VertexData() []byte {
	return m.vertexData
}

func (m *model) SetVertexData(data []byte) {
	m.vertexData = data
}

func (m *model) IndexData() []byte {
	return m.indexData
}

func (m *model) SetIndexData(data []byte) {
	m.indexData = data
}

func (m *model) IndexCount() int {
	return m.indexCount
}

func (m *model) SetIndexCount(count int) {
	m.indexCount = count
}

func (m *model) GetAnimationIndex(name string) int {
	for i, anim := range m.animations {
		if anim.Name == name {
			return i
		}
	}
	return -1
}

func (m *model) RenderMaterials() []material.Material {
	return m.renderMaterials
}

func (m *model) SetRenderMaterials(mats []material.Material) {
	m.renderMaterials = mats
}

func (m *model) BoundingRadius() float32 {
	return m.boundingRadius
}

func (m *model) EffectProvider() bind_group_provider.BindGroupProvider {
	return m.effectProvider
}

func (m *model) SetEffectProvider(provider bind_group_provider.BindGroupProvider) {
	m.effectProvider = provider
}
