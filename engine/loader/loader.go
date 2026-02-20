package loader

import (
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"sync"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"

	"github.com/cogentcore/webgpu/wgpu"
)

// LoaderBackendType identifies the model file format backend to use.
type LoaderBackendType int

const (
	// BackendTypeGLTF selects the glTF/GLB loader backend.
	BackendTypeGLTF LoaderBackendType = iota
)

// loader is the implementation of the Loader interface.
type loader struct {
	mu sync.RWMutex

	renderer renderer.Renderer

	modelCache map[string]model.Model

	backend loaderBackend
}

// Loader defines the public-facing interface for loading and caching 3D models.
// It abstracts the file format (glTF, GLB, etc.) behind a generic backend and
// manages a cache of previously loaded models.
type Loader interface {
	// Load imports a model file and caches the result.
	// If the model is already cached (by file path), the cached version is returned.
	// The backend is selected based on the file extension (.gltf/.glb → glTF backend).
	// The fragment shader is used to discover bind group layouts for initializing
	// material GPU resources (textures, samplers, bind groups).
	//
	// Parameters:
	//   - path: the file path to the model file
	//   - fragmentShader: the fragment shader whose bind group layouts drive material GPU init
	//
	// Returns:
	//   - model.Model: the loaded and cached model
	//   - error: error if loading fails
	Load(path string, fragmentShader shader.Shader) (model.Model, error)

	// LoadMeshOnly imports only mesh and material data, skipping skeleton and animations.
	// Useful for static models that don't need animation support.
	// The fragment shader is used to discover bind group layouts for initializing
	// material GPU resources (textures, samplers, bind groups).
	//
	// Parameters:
	//   - path: the file path to the model file
	//   - fragmentShader: the fragment shader whose bind group layouts drive material GPU init
	//
	// Returns:
	//   - model.Model: the loaded model (mesh and materials only)
	//   - error: error if loading fails
	LoadMeshOnly(path string, fragmentShader shader.Shader) (model.Model, error)

	// LoadReader imports a model from a reader stream and caches it by the given name.
	// The fragment shader is used to discover bind group layouts for initializing
	// material GPU resources (textures, samplers, bind groups).
	//
	// Parameters:
	//   - name: the cache key for the loaded model
	//   - r: the reader providing model data
	//   - isGLB: true if the reader provides GLB binary data
	//   - fragmentShader: the fragment shader whose bind group layouts drive material GPU init
	//
	// Returns:
	//   - model.Model: the loaded model
	//   - error: error if loading fails
	LoadReader(name string, r io.Reader, isGLB bool, fragmentShader shader.Shader) (model.Model, error)

	// Get retrieves a cached model by name. Returns nil if not found.
	//
	// Parameters:
	//   - name: the cache key to look up
	//
	// Returns:
	//   - model.Model: the cached model or nil
	Get(name string) model.Model

	// Models returns the full model cache.
	//
	// Returns:
	//   - map[string]model.Model: all cached models keyed by name
	Models() map[string]model.Model

	// InitMaterialGPU initializes GPU resources (fallback textures, samplers, bind group)
	// for a render material using the provided fragment shader's bind group layout. This
	// is required for procedural/hand-built models that bypass the Load pipeline but need
	// to render with lit fragment shaders that declare material texture bindings.
	//
	// Parameters:
	//   - mat: the Material to initialize GPU resources on
	//   - fragmentShader: the fragment shader providing bind group layout information
	//   - providerName: a unique name for the material's bind group provider
	//
	// Returns:
	//   - error: error if GPU resource creation fails
	InitMaterialGPU(mat material.Material, fragmentShader shader.Shader, providerName string) error
}

var _ Loader = &loader{}

// NewLoader creates a new Loader instance with the specified backend type and options applied.
//
// Parameters:
//   - backendType: the type of loader backend to use (e.g., BackendTypeGLTF)
//   - options: a variadic list of LoaderBuilderOption functions to configure the Loader
//
// Returns:
//   - Loader: a new instance of Loader configured with the provided backend and options
func NewLoader(backendType LoaderBackendType, options ...LoaderBuilderOption) Loader {
	l := &loader{
		mu:         sync.RWMutex{},
		modelCache: make(map[string]model.Model),
	}

	switch backendType {
	case BackendTypeGLTF:
		l.backend = newGLTFLoaderBackend()
	}

	for _, option := range options {
		option(l)
	}
	return l
}

func (l *loader) Load(path string, fragmentShader shader.Shader) (model.Model, error) {
	l.mu.RLock()
	if cached, ok := l.modelCache[path]; ok {
		l.mu.RUnlock()
		return cached, nil
	}
	l.mu.RUnlock()

	backend, err := l.resolveBackend(path)
	if err != nil {
		return nil, err
	}

	imported, err := backend.Load(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load %s: %w", path, err)
	}

	m, err := l.importedToModel(imported, fragmentShader)
	if err != nil {
		return nil, err
	}

	l.mu.Lock()
	l.modelCache[path] = m
	l.mu.Unlock()

	return m, nil
}

func (l *loader) LoadMeshOnly(path string, fragmentShader shader.Shader) (model.Model, error) {
	l.mu.RLock()
	if cached, ok := l.modelCache[path]; ok {
		l.mu.RUnlock()
		return cached, nil
	}
	l.mu.RUnlock()

	backend, err := l.resolveBackend(path)
	if err != nil {
		return nil, err
	}

	imported, err := backend.LoadMeshOnly(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load %s: %w", path, err)
	}

	m, err := l.importedToModel(imported, fragmentShader)
	if err != nil {
		return nil, err
	}

	l.mu.Lock()
	l.modelCache[path] = m
	l.mu.Unlock()

	return m, nil
}

func (l *loader) LoadReader(name string, r io.Reader, isGLB bool, fragmentShader shader.Shader) (model.Model, error) {
	l.mu.RLock()
	if cached, ok := l.modelCache[name]; ok {
		l.mu.RUnlock()
		return cached, nil
	}
	l.mu.RUnlock()

	imported, err := l.backend.LoadReader(r, isGLB)
	if err != nil {
		return nil, fmt.Errorf("failed to load from reader %q: %w", name, err)
	}

	m, err := l.importedToModel(imported, fragmentShader)
	if err != nil {
		return nil, err
	}

	l.mu.Lock()
	l.modelCache[name] = m
	l.mu.Unlock()

	return m, nil
}

func (l *loader) Get(name string) model.Model {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.modelCache[name]
}

func (l *loader) Models() map[string]model.Model {
	l.mu.RLock()
	defer l.mu.RUnlock()

	result := make(map[string]model.Model, len(l.modelCache))
	for k, v := range l.modelCache {
		result[k] = v
	}
	return result
}

func (l *loader) InitMaterialGPU(mat material.Material, fragmentShader shader.Shader, providerName string) error {
	if l.renderer == nil {
		return fmt.Errorf("loader: cannot InitMaterialGPU without a Renderer")
	}
	return l.initMaterialGPU(mat, fragmentShader, providerName, 0)
}

// resolveBackend selects an appropriate loader backend based on the file extension.
// Currently only glTF/GLB is supported.
func (l *loader) resolveBackend(path string) (loaderBackend, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".gltf", ".glb":
		return l.backend, nil
	default:
		return nil, fmt.Errorf("unsupported model format: %s", ext)
	}
}

// importedToModel converts an ImportedModel (CPU data) into a Model (engine-ready).
// It combines all mesh vertex and index data into a single BindGroupProvider, uploads
// the data to the GPU via InitBindGroup when a Renderer is available, and initializes
// material GPU resources (textures, samplers, bind groups) using the fragment shader's
// bind group layout descriptors.
//
// Parameters:
//   - imported: the CPU-side ImportedModel containing mesh, skeleton, animation, and material data
//   - fragmentShader: the fragment shader used to discover bind group layouts for material GPU init
//
// Returns:
//   - model.Model: the engine-ready Model with GPU mesh resources
//   - error: error if GPU resource creation fails
func (l *loader) importedToModel(imported *model.ImportedModel, fragmentShader shader.Shader) (model.Model, error) {
	skinned := imported.Skeleton != nil && len(imported.Skeleton.Bones) > 0

	// Combine all meshes into one vertex + index buffer
	var allVertexBytes []byte
	var allIndexBytes []byte
	totalIndices := 0
	indexOffset := uint32(0)

	for _, mesh := range imported.Meshes {
		allVertexBytes = append(allVertexBytes, common.SliceToBytes(mesh.Vertices)...)

		// Reindex: offset each index by the running vertex count across meshes
		adjusted := make([]uint32, len(mesh.Indices))
		for i, idx := range mesh.Indices {
			adjusted[i] = idx + indexOffset
		}
		allIndexBytes = append(allIndexBytes, common.SliceToBytes(adjusted)...)

		totalIndices += len(mesh.Indices)
		indexOffset += uint32(len(mesh.Vertices))
	}

	// Create BindGroupProvider with staged vertex/index data
	provider := bind_group_provider.NewBindGroupProvider(
		imported.Name + "_mesh",
	)

	// Upload to GPU if renderer is available
	if l.renderer != nil {
		if err := l.renderer.InitMeshBuffers(provider, allVertexBytes, allIndexBytes, totalIndices); err != nil {
			return nil, fmt.Errorf("failed to init mesh bind group for %q: %w", imported.Name, err)
		}
	}

	mdl := model.NewModel(
		model.WithName(imported.Name),
		model.WithSkinned(skinned),
		model.WithSkeleton(imported.Skeleton),
		model.WithAnimations(imported.Animations),
		model.WithImportedMaterials(imported.Materials),
		model.WithMeshProvider(provider),
	)

	// Convert imported materials into render-ready Materials with GPU resources.
	renderMats := make([]material.Material, len(imported.Materials))
	for i, imp := range imported.Materials {
		mat := material.NewMaterial(
			material.WithName(imp.Name),
			material.WithBaseColor(imp.BaseColor),
			material.WithMetallic(imp.Metallic),
			material.WithRoughness(imp.Roughness),
			material.WithDiffuseTexture(imp.DiffuseTexture),
			material.WithNormalTexture(imp.NormalTexture),
			material.WithMetallicRoughnessTexture(imp.MetallicRoughnessTexture),
			material.WithPipelineKey(imported.Name),
		)

		// Initialize material GPU resources when a renderer and fragment shader are available.
		if l.renderer != nil && fragmentShader != nil {
			if err := l.initMaterialGPU(mat, fragmentShader, imported.Name, i); err != nil {
				return nil, fmt.Errorf("failed to init material GPU resources for %q material %d: %w", imported.Name, i, err)
			}
		}

		renderMats[i] = mat
	}
	mdl.SetRenderMaterials(renderMats)

	return mdl, nil
}

// initMaterialGPU creates GPU resources (textures, samplers, bind group) for a single Material
// by inspecting the fragment shader's pre-processed Declarations for @oxy:provider annotations
// with the "material" identity. Per-binding roles (diffuse_texture, normal_texture, etc.) are
// resolved from the declaration Args, eliminating the need for variable-name string matching.
//
// Parameters:
//   - mat: the Material to initialize GPU resources on
//   - fragmentShader: the fragment shader providing declarations and bind group layout information
//   - modelName: the model name, used for provider naming
//   - materialIndex: the index of this material within the model, used for provider naming
//
// Returns:
//   - error: error if GPU resource creation fails
func (l *loader) initMaterialGPU(mat material.Material, fragmentShader shader.Shader, modelName string, materialIndex int) error {
	// Scan declarations for material provider entries and collect the group index
	// plus per-binding role mappings.
	materialGroupIdx := -1
	bindingRoles := make(map[int]shader.AnnotationArg) // binding index → role

	for _, decl := range fragmentShader.Declarations() {
		if decl.Type != shader.AnnotationTypeProvider || decl.Group == nil {
			continue
		}
		if decl.Args[0] != shader.AnnotationArgMaterial {
			continue
		}
		if materialGroupIdx < 0 {
			materialGroupIdx = *decl.Group
		}
		if len(decl.Args) > 1 && decl.Binding != nil {
			bindingRoles[*decl.Binding] = decl.Args[1]
		}
	}

	if materialGroupIdx < 0 {
		// No material provider declared in this shader; nothing to init.
		return nil
	}

	providerName := fmt.Sprintf("%s_material_%d", modelName, materialIndex)
	provider := bind_group_provider.NewBindGroupProvider(providerName)

	// Map each material binding role to its texture data from the Material.
	type textureBinding struct {
		tex  *common.ImportedTexture
		role shader.AnnotationArg
	}
	roleToTexture := map[shader.AnnotationArg]textureBinding{
		shader.AnnotationArgDiffuseTexture:           {tex: mat.DiffuseTexture(), role: shader.AnnotationArgDiffuseTexture},
		shader.AnnotationArgNormalTexture:            {tex: mat.NormalTexture(), role: shader.AnnotationArgNormalTexture},
		shader.AnnotationArgMetallicRoughnessTexture: {tex: mat.MetallicRoughnessTexture(), role: shader.AnnotationArgMetallicRoughnessTexture},
	}

	// Pair each texture role with its sampler role so we can locate both bindings.
	textureSamplerPairs := map[shader.AnnotationArg]shader.AnnotationArg{
		shader.AnnotationArgDiffuseTexture:           shader.AnnotationArgDiffuseSampler,
		shader.AnnotationArgNormalTexture:            shader.AnnotationArgNormalSampler,
		shader.AnnotationArgMetallicRoughnessTexture: shader.AnnotationArgMetallicRoughnessSampler,
	}

	// Build reverse lookup: role → binding index.
	roleToBinding := make(map[shader.AnnotationArg]int)
	for binding, role := range bindingRoles {
		roleToBinding[role] = binding
	}

	for texRole, tb := range roleToTexture {
		if tb.tex == nil {
			continue
		}

		texBindingIdx, hasTexBinding := roleToBinding[texRole]
		if !hasTexBinding {
			continue
		}

		samplerRole := textureSamplerPairs[texRole]
		samplerBindingIdx, hasSamplerBinding := roleToBinding[samplerRole]

		// Decode texture to RGBA pixels.
		pixels, width, height, err := tb.tex.Decode()
		if err != nil {
			return fmt.Errorf("failed to decode %s texture: %w", texRole, err)
		}

		stagingData := common.TextureStagingData{
			Pixels: pixels,
			Width:  width,
			Height: height,
		}

		if err := l.renderer.InitTextureView(provider, texBindingIdx, stagingData); err != nil {
			return fmt.Errorf("failed to init %s texture view: %w", texRole, err)
		}

		// Init sampler using glTF sampler params if available, otherwise default to linear/repeat.
		if hasSamplerBinding {
			samplerData := common.SamplerStagingData{
				AddressModeU:  wgpu.AddressModeRepeat,
				AddressModeV:  wgpu.AddressModeRepeat,
				AddressModeW:  wgpu.AddressModeRepeat,
				MagFilter:     wgpu.FilterModeLinear,
				MinFilter:     wgpu.FilterModeLinear,
				MipmapFilter:  wgpu.MipmapFilterModeLinear,
				LodMinClamp:   0,
				LodMaxClamp:   32,
				MaxAnisotropy: 1,
			}
			if tb.tex.SamplerData != nil {
				samplerData = *tb.tex.SamplerData
			}
			if err := l.renderer.InitSampler(provider, samplerBindingIdx, samplerData); err != nil {
				return fmt.Errorf("failed to init %s sampler: %w", samplerRole, err)
			}
		}
	}

	// Fill fallback 1×1 placeholder textures for any shader-declared texture/sampler
	// bindings that were not populated above (e.g. when a model lacks a normal map).
	// Without these, InitBindGroup would fail with "no texture view" errors.
	descriptor := fragmentShader.BindGroupLayoutDescriptor(materialGroupIdx)
	for _, entry := range descriptor.Entries {
		binding := int(entry.Binding)
		isTexture := entry.Texture.SampleType != wgpu.TextureSampleTypeUndefined
		isSampler := entry.Sampler.Type != wgpu.SamplerBindingTypeUndefined

		if isTexture && provider.TextureView(binding) == nil {
			// Determine fallback pixel based on the binding role from declarations.
			role := bindingRoles[binding]
			var pixel [4]byte
			switch role {
			case shader.AnnotationArgNormalTexture:
				// Flat tangent-space normal pointing straight up: (0.5, 0.5, 1.0) → (128, 128, 255, 255)
				pixel = [4]byte{128, 128, 255, 255}
			case shader.AnnotationArgMetallicRoughnessTexture:
				// glTF packing: R=unused(0), G=roughness(1.0=fully rough), B=metallic(0=dielectric)
				pixel = [4]byte{0, 255, 0, 255}
			default:
				// White 1×1 texture (no-op multiply)
				pixel = [4]byte{255, 255, 255, 255}
			}
			fallback := common.TextureStagingData{
				Pixels: pixel[:],
				Width:  1,
				Height: 1,
			}
			if err := l.renderer.InitTextureView(provider, binding, fallback); err != nil {
				return fmt.Errorf("failed to init fallback texture at binding %d: %w", binding, err)
			}
		}

		if isSampler && provider.Sampler(binding) == nil {
			fallbackSampler := common.SamplerStagingData{
				AddressModeU:  wgpu.AddressModeRepeat,
				AddressModeV:  wgpu.AddressModeRepeat,
				AddressModeW:  wgpu.AddressModeRepeat,
				MagFilter:     wgpu.FilterModeLinear,
				MinFilter:     wgpu.FilterModeLinear,
				MipmapFilter:  wgpu.MipmapFilterModeLinear,
				LodMinClamp:   0,
				LodMaxClamp:   32,
				MaxAnisotropy: 1,
			}
			if err := l.renderer.InitSampler(provider, binding, fallbackSampler); err != nil {
				return fmt.Errorf("failed to init fallback sampler at binding %d: %w", binding, err)
			}
		}
	}

	// Create the bind group from the shader's layout descriptor for this group.
	if err := l.renderer.InitBindGroup(provider, descriptor, nil, nil); err != nil {
		return fmt.Errorf("failed to init material bind group: %w", err)
	}

	mat.SetBindGroupProvider(provider)
	return nil
}
