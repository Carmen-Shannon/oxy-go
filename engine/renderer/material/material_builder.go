package material

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// MaterialBuilderOption is a function that configures a material instance during construction.
type MaterialBuilderOption func(*material)

// WithName is an option builder that sets the name of the material.
//
// Parameters:
//   - name: the identifier for the material
//
// Returns:
//   - MaterialBuilderOption: a function that applies the name option to a material
func WithName(name string) MaterialBuilderOption {
	return func(m *material) {
		m.name = name
	}
}

// WithBaseColor is an option builder that sets the albedo/diffuse RGBA color of the material.
//
// Parameters:
//   - color: the base color as RGBA float32 values
//
// Returns:
//   - MaterialBuilderOption: a function that applies the base color option to a material
func WithBaseColor(color [4]float32) MaterialBuilderOption {
	return func(m *material) {
		m.baseColor = color
	}
}

// WithMetallic is an option builder that sets the metallic factor of the material.
//
// Parameters:
//   - metallic: the metallic factor (0.0 = dielectric, 1.0 = metal)
//
// Returns:
//   - MaterialBuilderOption: a function that applies the metallic option to a material
func WithMetallic(metallic float32) MaterialBuilderOption {
	return func(m *material) {
		m.metallic = metallic
	}
}

// WithRoughness is an option builder that sets the roughness factor of the material.
//
// Parameters:
//   - roughness: the roughness factor (0.0 = smooth, 1.0 = rough)
//
// Returns:
//   - MaterialBuilderOption: a function that applies the roughness option to a material
func WithRoughness(roughness float32) MaterialBuilderOption {
	return func(m *material) {
		m.roughness = roughness
	}
}

// WithDiffuseTexture is an option builder that sets the diffuse/albedo texture reference.
//
// Parameters:
//   - tex: the imported texture data for the diffuse map
//
// Returns:
//   - MaterialBuilderOption: a function that applies the diffuse texture option to a material
func WithDiffuseTexture(tex *common.ImportedTexture) MaterialBuilderOption {
	return func(m *material) {
		m.diffuseTexture = tex
	}
}

// WithNormalTexture is an option builder that sets the normal map texture reference.
//
// Parameters:
//   - tex: the imported texture data for the normal map
//
// Returns:
//   - MaterialBuilderOption: a function that applies the normal texture option to a material
func WithNormalTexture(tex *common.ImportedTexture) MaterialBuilderOption {
	return func(m *material) {
		m.normalTexture = tex
	}
}

// WithMetallicRoughnessTexture is an option builder that sets the metallic-roughness texture reference.
//
// Parameters:
//   - tex: the imported texture data for the metallic-roughness map
//
// Returns:
//   - MaterialBuilderOption: a function that applies the metallic-roughness texture option to a material
func WithMetallicRoughnessTexture(tex *common.ImportedTexture) MaterialBuilderOption {
	return func(m *material) {
		m.metallicRoughnessTexture = tex
	}
}

// WithPipelineKey is an option builder that sets the render pipeline key for the material.
//
// Parameters:
//   - key: the pipeline key to associate with the material
//
// Returns:
//   - MaterialBuilderOption: a function that applies the pipeline key option to a material
func WithPipelineKey(key string) MaterialBuilderOption {
	return func(m *material) {
		m.pipelineKey = key
	}
}

// WithBindGroupProvider is an option builder that sets the bind group provider for the material.
//
// Parameters:
//   - provider: the bind group provider containing GPU resources for the material
//
// Returns:
//   - MaterialBuilderOption: a function that applies the bind group provider option to a material
func WithBindGroupProvider(provider bind_group_provider.BindGroupProvider) MaterialBuilderOption {
	return func(m *material) {
		m.bindGroupProvider = provider
	}
}
