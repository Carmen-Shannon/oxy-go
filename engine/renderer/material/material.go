// Package material defines renderer material abstractions for surface shading and resource setup.
//
// Its primary interface is [Material], which models render surface properties and
// GPU binding associations used by renderer initialization and draw paths.
package material

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// Material defines the interface for a render material, encapsulating surface
// properties, texture references, and GPU resource bindings needed for draw calls.
//
// Surface properties (name, base color, metallic, roughness, textures) are set at
// load time and are read-only through this interface. GPU resource references
// (pipeline key, bind group provider) are mutable so they can be configured after
// construction during the Loader GPU-init phase.
type Material interface {
	common.Delegate[Material]

	// Name retrieves the material identifier.
	//
	// Returns:
	//   - string: the name of the material
	Name() string

	// BaseColor retrieves the albedo/diffuse RGBA color of the material.
	//
	// Returns:
	//   - [4]float32: the base color as RGBA values
	BaseColor() [4]float32

	// Metallic retrieves the metallic factor of the material.
	// A value of 0.0 represents a dielectric surface, 1.0 represents a fully metallic surface.
	//
	// Returns:
	//   - float32: the metallic factor
	Metallic() float32

	// Roughness retrieves the roughness factor of the material.
	// A value of 0.0 represents a perfectly smooth surface, 1.0 represents a fully rough surface.
	//
	// Returns:
	//   - float32: the roughness factor
	Roughness() float32

	// AlphaCutoff retrieves the alpha discard threshold for this material.
	// Fragments with alpha below this value are discarded in alpha-tested rendering.
	//
	// Returns:
	//   - float32: the alpha cutoff threshold
	AlphaCutoff() float32

	// DiffuseTexture retrieves the diffuse/albedo texture data reference, or nil if none is set.
	//
	// Returns:
	//   - *common.ImportedTexture: the diffuse texture, or nil
	DiffuseTexture() *common.ImportedTexture

	// NormalTexture retrieves the normal map texture data reference, or nil if none is set.
	//
	// Returns:
	//   - *common.ImportedTexture: the normal texture, or nil
	NormalTexture() *common.ImportedTexture

	// MetallicRoughnessTexture retrieves the metallic-roughness texture data reference, or nil if none is set.
	//
	// Returns:
	//   - *common.ImportedTexture: the metallic-roughness texture, or nil
	MetallicRoughnessTexture() *common.ImportedTexture

	// PipelineKey retrieves the key identifying the render pipeline this material uses.
	//
	// Returns:
	//   - string: the pipeline key
	PipelineKey() string

	// BindGroupProvider retrieves the bind group provider holding GPU-side resources for this material.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the bind group provider, or nil if not yet initialized
	BindGroupProvider() bind_group_provider.BindGroupProvider

	// SetPipelineKey sets the render pipeline key for this material.
	//
	// Parameters:
	//   - key: the pipeline key to associate with this material
	SetPipelineKey(key string)

	// SetBindGroupProvider sets the bind group provider for this material.
	//
	// Parameters:
	//   - provider: the bind group provider containing GPU resources for this material
	SetBindGroupProvider(provider bind_group_provider.BindGroupProvider)

	// PipelineOptions retrieves the pipeline builder options associated with this material.
	// These are applied when the scene registers the material's render pipeline at Add time.
	// Each value is a pipeline.PipelineBuilderOption passed as any to avoid import cycles.
	//
	// Returns:
	//   - []any: the pipeline builder options, or nil if none are set
	PipelineOptions() []any

	// Provider retrieves the bind group provider associated with the specified group index.
	// Returns nil if no provider has been set for the given group.
	//
	// Parameters:
	//   - group: the bind group index to look up
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the provider for the group, or nil
	Provider(group int) bind_group_provider.BindGroupProvider

	// SetProvider stores a bind group provider for the specified group index.
	// This allows a material to own GPU resources across multiple bind groups
	// (e.g. textures at one group and effect uniforms at another).
	//
	// Parameters:
	//   - group: the bind group index
	//   - provider: the bind group provider to associate with the group
	SetProvider(group int, provider bind_group_provider.BindGroupProvider)
}

var _ Material = &material{}

func (m *material) Name() string                            { return m.name }
func (m *material) BaseColor() [4]float32                   { return m.baseColor }
func (m *material) Metallic() float32                       { return m.metallic }
func (m *material) Roughness() float32                      { return m.roughness }
func (m *material) AlphaCutoff() float32                    { return m.alphaCutoff }
func (m *material) DiffuseTexture() *common.ImportedTexture { return m.diffuseTexture }
func (m *material) NormalTexture() *common.ImportedTexture  { return m.normalTexture }
func (m *material) PipelineOptions() []any                  { return m.pipelineOpts }
func (m *material) PipelineKey() string                     { return m.pipelineKey }
func (m *material) SetPipelineKey(key string)               { m.pipelineKey = key }

func (m *material) MetallicRoughnessTexture() *common.ImportedTexture {
	return m.metallicRoughnessTexture
}

func (m *material) BindGroupProvider() bind_group_provider.BindGroupProvider {
	return m.bindGroupProvider
}

func (m *material) SetBindGroupProvider(provider bind_group_provider.BindGroupProvider) {
	m.bindGroupProvider = provider
}

func (m *material) Provider(group int) bind_group_provider.BindGroupProvider {
	return m.providers[group]
}

func (m *material) SetProvider(group int, provider bind_group_provider.BindGroupProvider) {
	m.providers[group] = provider
}
