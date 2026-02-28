package material

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// material is the implementation of the Material interface.
type material struct {
	common.DelegateImpl[Material]

	name                     string
	baseColor                [4]float32
	metallic                 float32
	roughness                float32
	diffuseTexture           *common.ImportedTexture
	normalTexture            *common.ImportedTexture
	metallicRoughnessTexture *common.ImportedTexture
	pipelineKey              string
	bindGroupProvider        bind_group_provider.BindGroupProvider
	fragmentShaderPath       string
	providers                map[int]bind_group_provider.BindGroupProvider
}

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

	// FragmentShaderPath retrieves the path to the fragment shader associated with this material.
	// An empty string indicates that the engine default fragment shader should be used.
	//
	// Returns:
	//   - string: the fragment shader asset path, or empty for the engine default
	FragmentShaderPath() string

	// SetFragmentShaderPath sets the path to the fragment shader for this material.
	//
	// Parameters:
	//   - path: the fragment shader asset path
	SetFragmentShaderPath(path string)

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
		providers: make(map[int]bind_group_provider.BindGroupProvider),
	}
	for _, opt := range options {
		opt(m)
	}
	m.Delegate = m
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

func (m *material) FragmentShaderPath() string {
	return m.fragmentShaderPath
}

func (m *material) SetFragmentShaderPath(path string) {
	m.fragmentShaderPath = path
}

func (m *material) Provider(group int) bind_group_provider.BindGroupProvider {
	return m.providers[group]
}

func (m *material) SetProvider(group int, provider bind_group_provider.BindGroupProvider) {
	m.providers[group] = provider
}
