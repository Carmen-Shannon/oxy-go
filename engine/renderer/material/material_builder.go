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

// WithAlphaCutoff is an option builder that sets the alpha discard threshold for the material.
//
// Parameters:
//   - cutoff: the alpha cutoff threshold
//
// Returns:
//   - MaterialBuilderOption: a function that applies the alpha cutoff option to a material
func WithAlphaCutoff(cutoff float32) MaterialBuilderOption {
	return func(m *material) {
		m.alphaCutoff = cutoff
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

// WithPipelineOptions is an option builder that sets the pipeline builder options for the material.
// These options are applied by the scene when registering the material's render pipeline at Add time,
// and may include render state overrides (blend, cull mode, depth), custom vertex or fragment shaders,
// and any other [pipeline.PipelineBuilderOption] values. Each option is stored as any to avoid
// import cycles between the material and pipeline packages.
//
// Parameters:
//   - opts: variadic pipeline builder options to associate with the material
//
// Returns:
//   - MaterialBuilderOption: a function that applies the pipeline options to a material
func WithPipelineOptions(opts ...any) MaterialBuilderOption {
	return func(m *material) {
		m.pipelineOpts = opts
	}
}

// NewMaterial creates a new Material instance configured with the provided options.
//
// Parameters:
//   - options: variadic list of MaterialBuilderOption functions to configure the material
//
// Returns:
//   - Material: a new Material instance
func NewMaterial(options ...MaterialBuilderOption) Material {
	m := &material{
		baseColor:   [4]float32{1, 1, 1, 1},
		metallic:    0.0,
		roughness:   1.0,
		alphaCutoff: 0.01,
		providers:   make(map[int]bind_group_provider.BindGroupProvider),
	}
	for _, opt := range options {
		opt(m)
	}
	m.Delegate = m
	return m
}
