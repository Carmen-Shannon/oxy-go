package material

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// material is the implementation of the Material interface.
type material struct {
	name                     string
	baseColor                [4]float32
	metallic                 float32
	roughness                float32
	diffuseTexture           *common.ImportedTexture
	normalTexture            *common.ImportedTexture
	metallicRoughnessTexture *common.ImportedTexture
	pipelineKey              string
	bindGroupProvider        bind_group_provider.BindGroupProvider
}

// Material defines the interface for a render material, encapsulating surface
// properties, texture references, and GPU resource bindings needed for draw calls.
//
// Surface properties (name, base color, metallic, roughness, textures) are set at
// load time and are read-only through this interface. GPU resource references
// (pipeline key, bind group provider) are mutable so they can be configured after
// construction during the Loader GPU-init phase.
type Material interface {
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
}

var _ Material = &material{}

// NewMaterial creates a new Material instance configured with the provided options.
//
// Parameters:
//   - options: variadic list of MaterialBuilderOption functions to configure the material
//
// Returns:
//   - Material: a new Material instance
func NewMaterial(options ...MaterialBuilderOption) Material {
	m := &material{
		baseColor: [4]float32{1, 1, 1, 1},
		metallic:  0.0,
		roughness: 1.0,
	}
	for _, opt := range options {
		opt(m)
	}
	return m
}

func (m *material) Name() string {
	return m.name
}

func (m *material) BaseColor() [4]float32 {
	return m.baseColor
}

func (m *material) Metallic() float32 {
	return m.metallic
}

func (m *material) Roughness() float32 {
	return m.roughness
}

func (m *material) DiffuseTexture() *common.ImportedTexture {
	return m.diffuseTexture
}

func (m *material) NormalTexture() *common.ImportedTexture {
	return m.normalTexture
}

func (m *material) MetallicRoughnessTexture() *common.ImportedTexture {
	return m.metallicRoughnessTexture
}

func (m *material) PipelineKey() string {
	return m.pipelineKey
}

func (m *material) BindGroupProvider() bind_group_provider.BindGroupProvider {
	return m.bindGroupProvider
}

func (m *material) SetPipelineKey(key string) {
	m.pipelineKey = key
}

func (m *material) SetBindGroupProvider(provider bind_group_provider.BindGroupProvider) {
	m.bindGroupProvider = provider
}
