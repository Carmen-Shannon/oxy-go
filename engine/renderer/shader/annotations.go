package shader

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// annotationPrefix is the marker that identifies an Oxy annotation within a WGSL comment line.
// Every annotation must appear on a line beginning with "//" followed by this prefix.
const annotationPrefix = "@oxy:"

// AnnotationType identifies the kind of annotation parsed from a WGSL comment line.
// Each type corresponds to a distinct pre-processor action and produces different
// fields on the resulting Annotation struct.
type AnnotationType string

const (
	// annotationTypeInclude injects the WGSL source of a registered struct definition
	// into the shader at the annotation site. The struct source is embedded from the
	// corresponding Go GPU type's .wgsl asset file. This annotation does not produce
	// a declaration and is consumed entirely during pre-processing.
	//
	// Syntax: //@oxy:include <struct_type>
	//
	// Example: //@oxy:include camera
	annotationTypeInclude AnnotationType = "include"

	// AnnotationTypeBindingGroup generates a WGSL @group/@binding variable declaration
	// and appends an Annotation to the PreProcessor's declarations list. The declaration
	// carries the group index, binding index, and the resolved struct type, enabling the
	// Scene to semantically match bindings to resource providers without string lookups.
	//
	// Syntax: //@oxy:group <group> <binding> <address_space> <var_name> <type>
	//
	// Example: //@oxy:group 0 0 storage_uniform camera camera
	AnnotationTypeBindingGroup AnnotationType = "group"

	// AnnotationTypeProvider registers a resource provider identity for a group and binding
	// without generating any WGSL output. The WGSL binding declaration remains hand-written
	// in the shader source directly below the annotation. This is used for bindings that
	// contain raw WGSL types (textures, samplers, flat arrays of primitives) which have no
	// corresponding registered struct in the pre-processor's struct registry.
	//
	// An optional binding role can be appended after the provider identity to declare the
	// semantic purpose of an individual binding within a multi-binding provider group.
	// This allows the loader to resolve binding indices from declarations instead of
	// relying on variable-name string matching.
	//
	// Syntax:
	//   //@oxy:provider <group> <binding> <provider_identity>
	//   //@oxy:provider <group> <binding> <provider_identity> <binding_role>
	//
	// Examples:
	//   //@oxy:provider 2 0 material diffuse_texture
	//   //@oxy:provider 4 0 shadow
	AnnotationTypeProvider AnnotationType = "provider"

	// annotationTypeInject replaces the annotation line with a typed WGSL const
	// declaration whose value is looked up at pre-process time from an injection map
	// passed to Process(). Like annotationTypeInclude, this annotation is consumed
	// entirely during pre-processing and does not produce a declaration entry.
	//
	// Syntax: //@oxy:inject <const_name> <wgsl_type> <injection_key>
	//
	// Example: //@oxy:inject MAX_BONES u32 max_bones
	annotationTypeInject AnnotationType = "inject"
)

// Annotation represents a single parsed @oxy: annotation from a WGSL shader source line.
// It carries the annotation type, its arguments, the source line number, and optional
// group/binding indices. Annotations of type AnnotationTypeBindingGroup and
// AnnotationTypeProvider are appended to the PreProcessor's declarations list for
// consumption by the Scene during resource wiring.
type Annotation struct {
	// Type identifies which annotation was parsed (include, group, or provider).
	Type AnnotationType

	// Args holds the annotation's arguments. The contents depend on Type:
	//   - include:  [0] = struct type key (e.g. "camera")
	//   - group:    [0] = address space, [1] = var name, [2] = WGSL type key
	//   - provider: [0] = provider identity (e.g. "material", "animator_output"), [1] = binding role (optional, e.g. "diffuse_texture")
	Args []AnnotationArg

	// Line is the 1-based line number in the original WGSL source where this annotation
	// was found. Used for error reporting.
	Line int

	// Group is the @group index for group and provider annotations. Nil for include annotations.
	Group *int

	// Binding is the @binding index for group and provider annotations. Nil for include annotations.
	Binding *int
}

// AnnotationArg is a typed string constant used as an argument in annotations.
// Arguments fall into three categories: struct type keys (used with include and group),
// address space identifiers (used with group), and provider identity keys (used with provider).
type AnnotationArg string

// ── Struct type arguments ──────────────────────────────────────────────────────
// These identify registered WGSL struct types. They can appear in @oxy:include annotations
// (to inject the struct source) and in @oxy:group annotations (as the type field, optionally
// wrapped in array<>). Each maps to a Go GPU type with an embedded .wgsl asset file.

const (
	// AnnotationArgCamera identifies the CameraUniform struct.
	// Source: engine/camera/assets/camera-uniform.wgsl
	AnnotationArgCamera AnnotationArg = "camera"

	// annotationArgVertex identifies the VertexInput struct for static (non-skinned) meshes.
	// Source: engine/model/assets/vertex.wgsl
	annotationArgVertex AnnotationArg = "vertex"

	// annotationArgSkinnedVertex identifies the VertexInput struct for skinned meshes with bone weights.
	// Source: engine/model/assets/skinned-vertex.wgsl
	annotationArgSkinnedVertex AnnotationArg = "skinned_vertex"

	// AnnotationArgOverlayParams identifies the OverlayParams material struct.
	// Source: engine/renderer/material/assets/overlay-params.wgsl
	AnnotationArgOverlayParams AnnotationArg = "overlay_params"

	// AnnotationArgEffectParams identifies the EffectParams material struct.
	// Source: engine/renderer/material/assets/effect-params.wgsl
	AnnotationArgEffectParams AnnotationArg = "effect_params"

	// AnnotationArgLight identifies the Light struct for per-light GPU data.
	// Source: engine/light/assets/light.wgsl
	AnnotationArgLight AnnotationArg = "light"

	// AnnotationArgLightHeader identifies the LightHeader struct containing light count and ambient color.
	// Source: engine/light/assets/light-header.wgsl
	AnnotationArgLightHeader AnnotationArg = "light_header"

	// annotationArgLightCullUniforms identifies the LightCullUniforms struct for tile-based light culling.
	// Source: engine/light/assets/light-cull-uniforms.wgsl
	annotationArgLightCullUniforms AnnotationArg = "light_cull_uniforms"

	// AnnotationArgShadowUniform identifies the ShadowUniform struct for the shadow depth pass.
	// Source: engine/light/assets/shadow-uniform.wgsl
	AnnotationArgShadowUniform AnnotationArg = "shadow_uniform"

	// AnnotationArgCSMData identifies the CSMData and CSMCascade structs for cascaded shadow map data.
	// Source: engine/light/assets/csm-data.wgsl
	AnnotationArgCSMData AnnotationArg = "csm_data"

	// AnnotationArgLightShadowEntry identifies the LightShadowEntry struct for per-light shadow data.
	// Source: engine/light/assets/light-shadow-entry.wgsl
	AnnotationArgLightShadowEntry AnnotationArg = "light_shadow_entry"

	// AnnotationArgTileUniforms identifies the TileUniforms struct for Forward+ tile configuration.
	// Source: engine/light/assets/tile-uniforms.wgsl
	AnnotationArgTileUniforms AnnotationArg = "tile_uniforms"

	// AnnotationArgModelData identifies the ModelData struct holding per-instance model matrices.
	// Source: engine/model/assets/model-data.wgsl
	AnnotationArgModelData AnnotationArg = "model_data"

	// AnnotationArgInstanceData identifies the InstanceData struct for per-instance transform output.
	// Source: engine/renderer/animator/assets/instance-data.wgsl
	AnnotationArgInstanceData AnnotationArg = "instance_data"

	// AnnotationArgAnimationData identifies the AnimationData struct for simple (non-skeletal) animation state.
	// Source: engine/renderer/animator/assets/animation-data.wgsl
	AnnotationArgAnimationData AnnotationArg = "animation_data"

	// AnnotationArgSkeletalAnimationData identifies the SkeletalAnimationData struct for skeletal animation state.
	// Source: engine/renderer/animator/assets/skeletal-animation-data.wgsl
	AnnotationArgSkeletalAnimationData AnnotationArg = "skeletal_animation_data"

	// AnnotationArgAnimationGlobals identifies the AnimationGlobals struct for skeletal compute uniforms.
	// Source: engine/renderer/animator/assets/animation-globals.wgsl
	AnnotationArgAnimationGlobals AnnotationArg = "animation_globals"

	// annotationArgFrustumPlane identifies the FrustumPlane struct used inside uniform structs for GPU culling.
	// Source: engine/renderer/animator/assets/frustum-plane.wgsl
	annotationArgFrustumPlane AnnotationArg = "frustum_plane"

	// AnnotationArgGlobalData identifies the GlobalData struct for simple compute shader uniforms.
	// Source: engine/renderer/animator/assets/simple-globals.wgsl
	AnnotationArgGlobalData AnnotationArg = "global_data"

	// AnnotationArgIndirectArgs identifies the IndirectArgs struct matching WebGPU's DrawIndexedIndirect layout.
	// Source: engine/renderer/animator/assets/indirect-args.wgsl
	AnnotationArgIndirectArgs AnnotationArg = "indirect_args"

	// AnnotationArgBoneInfo identifies the BoneInfo struct holding per-bone inverse bind matrices and hierarchy data.
	// Source: engine/renderer/animator/assets/bone-info.wgsl
	AnnotationArgBoneInfo AnnotationArg = "bone_info"

	// AnnotationArgPhysicsBody identifies the PhysicsBody struct for rigid body simulation.
	// Source: engine/physics/assets/body.wgsl
	AnnotationArgPhysicsBody AnnotationArg = "physics_body"

	// AnnotationArgPhysicsParticle identifies the Particle struct for particle-based fluid simulation.
	// Source: engine/physics/assets/particle.wgsl
	AnnotationArgPhysicsParticle AnnotationArg = "physics_particle"

	// AnnotationArgPhysicsGrid identifies the GridCell struct for spatial partitioning in physics simulation.
	// Source: engine/physics/assets/grid-cell.wgsl
	AnnotationArgPhysicsGrid AnnotationArg = "physics_grid"

	// AnnotationArgPhysicsGlobals identifies the PhysicsGlobals struct for physics simulation parameters.
	// Source: engine/physics/assets/physics-globals.wgsl
	AnnotationArgPhysicsGlobals AnnotationArg = "physics_globals"

	// AnnotationArgPhysicsGridParams identifies the GridParams struct for GPU-writable grid origin and dimensions.
	// Source: engine/physics/assets/grid-params.wgsl
	AnnotationArgPhysicsGridParams AnnotationArg = "physics_grid_params"

	// AnnotationArgGBufferOutput identifies the GBufferOutput struct for the G-Buffer MRT fragment shader.
	// Source: engine/light/assets/gbuffer-output.wgsl
	AnnotationArgGBufferOutput AnnotationArg = "gbuffer_output"

	// annotationArgSSAOParams identifies the SSAOParams struct for the SSAO compute shader.
	// Source: engine/light/assets/ssao-params.wgsl
	annotationArgSSAOParams AnnotationArg = "ssao_params"

	// annotationArgBlurParams identifies the BlurParams struct for the separable blur compute shader.
	// Source: engine/light/assets/blur-params.wgsl
	annotationArgBlurParams AnnotationArg = "blur_params"

	// annotationArgCompositionParams identifies the CompositionParams struct for the composition fragment shader.
	// Source: engine/light/assets/composition-params.wgsl
	annotationArgCompositionParams AnnotationArg = "composition_params"

	// annotationArgSSRParams identifies the SSRParams struct for the SSR compute shader.
	// Source: engine/light/assets/ssr-params.wgsl
	annotationArgSSRParams AnnotationArg = "ssr_params"

	// annotationArgContactShadowParams identifies the ContactShadowParams struct for the contact shadow compute shader.
	// Source: engine/light/assets/contact-shadow-params.wgsl
	annotationArgContactShadowParams AnnotationArg = "contact_shadow_params"

	// annotationArgLuminanceParams identifies the LuminanceParams struct for the luminance compute shader.
	// Source: engine/light/assets/luminance-params.wgsl
	annotationArgLuminanceParams AnnotationArg = "luminance_params"

	// annotationArgBloomParams identifies the GPUBloomParams struct type used by the bloom
	// downsample compute shader for threshold configuration.
	// Source: engine/light/assets/bloom-params.wgsl
	annotationArgBloomParams AnnotationArg = "bloom_params"

	// annotationArgTAAParams identifies the TAAParams struct for the TAA resolve compute shader.
	// Source: engine/light/assets/taa-params.wgsl
	annotationArgTAAParams AnnotationArg = "taa_params"
)

// ── Address space arguments ────────────────────────────────────────────────────
// These specify the WGSL variable address space in @oxy:group annotations.
// They map to WGSL var<> declarations.

const (
	// annotationArgStorageTypeUniform maps to var<uniform> in WGSL.
	annotationArgStorageTypeUniform AnnotationArg = "storage_uniform"

	// annotationArgStorageTypeRead maps to var<storage, read> in WGSL.
	annotationArgStorageTypeRead AnnotationArg = "storage_read"

	// annotationArgStorageTypeReadWrite maps to var<storage, read_write> in WGSL.
	annotationArgStorageTypeReadWrite AnnotationArg = "storage_read_write"
)

// ── Provider identity arguments ────────────────────────────────────────────────
// These identify which Scene-level resource provider owns a bind group. Used in
// @oxy:provider annotations and matched by the Scene's draw call and compute setup
// logic to wire the correct BindGroupProvider for each group.

const (
	// AnnotationArgMaterial identifies the material provider (textures, samplers, material uniforms).
	AnnotationArgMaterial AnnotationArg = "material"

	// AnnotationArgLights identifies the lights provider (light header + light storage array).
	AnnotationArgLights AnnotationArg = "lights"

	// AnnotationArgShadow identifies the shadow provider (shadow depth texture, comparison sampler, shadow uniform).
	AnnotationArgShadow AnnotationArg = "shadow"

	// AnnotationArgTiles identifies the Forward+ tile provider (tile counts and tile light indices).
	AnnotationArgTiles AnnotationArg = "tiles"

	// AnnotationArgEffect identifies the effect/overlay provider (visual effect parameters).
	AnnotationArgEffect AnnotationArg = "effect"

	// AnnotationArgAnimator identifies the animator provider for vertex shaders with raw instance buffers (e.g. skinned vertex array<vec4<f32>>).
	AnnotationArgAnimator AnnotationArg = "animator"

	// AnnotationArgAnimatorOutput identifies the compute shader's output_transforms buffer that is shared with the vertex shader's instance buffer.
	AnnotationArgAnimatorOutput AnnotationArg = "animator_output"

	// AnnotationArgAnimatorPacked identifies the packed animation data buffer (flat array<u32> of clips, channels, keyframes).
	AnnotationArgAnimatorPacked AnnotationArg = "animator_packed"

	// AnnotationArgAnimatorScratch identifies the scratch bone matrix workspace buffer used during skeletal animation blending.
	AnnotationArgAnimatorScratch AnnotationArg = "animator_scratch"

	// AnnotationArgSSAO identifies the SSAO provider (blurred occlusion texture + sampler for the lit shader).
	AnnotationArgSSAO AnnotationArg = "ssao"

	// AnnotationArgComposition identifies the composition provider (HDR texture + SSR texture + sampler + params uniform).
	AnnotationArgComposition AnnotationArg = "composition"

	// AnnotationArgSSR identifies the SSR provider (G-Buffer textures + HDR texture + SSR output + params uniform).
	AnnotationArgSSR AnnotationArg = "ssr"

	// AnnotationArgHiZInit identifies the Hi-Z init provider (GBuffer depth → Hi-Z mip 0 copy).
	AnnotationArgHiZInit AnnotationArg = "hiz_init"

	// AnnotationArgHiZDown identifies the Hi-Z downsample provider (mip N-1 → mip N min-downsample).
	AnnotationArgHiZDown AnnotationArg = "hiz_down"

	// AnnotationArgHiZDownMax identifies the hiz_down_max compute provider (MAX pyramid downsample).
	AnnotationArgHiZDownMax AnnotationArg = "hiz_down_max"

	// AnnotationArgContactShadows identifies the contact shadows provider (contact shadow texture + sampler for the lit shader).
	AnnotationArgContactShadows AnnotationArg = "contact_shadows"

	// AnnotationArgAnimatorHiZ identifies the animator Hi-Z occlusion culling provider
	// (Hi-Z full mip chain texture for the animator compute shaders).
	AnnotationArgAnimatorHiZ AnnotationArg = "animator_hiz"

	// AnnotationArgAnimatorMaxHiZ identifies the animator MAX Hi-Z occlusion culling provider.
	AnnotationArgAnimatorMaxHiZ AnnotationArg = "animator_max_hiz"

	// AnnotationArgTAA identifies the TAA resolve provider.
	AnnotationArgTAA AnnotationArg = "taa"

	// AnnotationArgTAASharpen identifies the CAS (Contrast Adaptive Sharpening) post-TAA provider.
	AnnotationArgTAASharpen AnnotationArg = "taa_sharpen"
)

// ── Injection key arguments ────────────────────────────────────────────────────
// These identify registered injection keys for @oxy:inject annotations. Each maps
// to a Go-side value that is provided in the injection map parameter of Process().

const (
	// annotationArgInjectMaxBones injects the maximum bone count for skinned mesh shaders.
	// Go source: engine/scene/scene.go — maxBonesGPU constant.
	annotationArgInjectMaxBones AnnotationArg = "max_bones"

	// annotationArgInjectMaxSSAOSamples injects the maximum SSAO kernel sample count
	// (caps the uniform array size and sample loop).
	// Go source: engine/scene/scene.go — SSAO sample count clamp.
	annotationArgInjectMaxSSAOSamples AnnotationArg = "max_ssao_samples"

	// annotationArgInjectTileSize injects the Forward+ light-culling tile width/height in pixels.
	// Go source: engine/light/light_handler_builder.go — tileSize field.
	annotationArgInjectTileSize AnnotationArg = "tile_size"

	// annotationArgInjectMaxLightsPerTile injects the maximum number of lights stored per tile
	// in the Forward+ light index buffer.
	// Go source: engine/light/light_handler_builder.go — maxLightsPerTile field.
	annotationArgInjectMaxLightsPerTile AnnotationArg = "max_lights_per_tile"

	// annotationArgInjectNumThreads injects the total workgroup thread count for light culling
	// (pre-computed as tileSize × tileSize in Go).
	// Go source: derived from tileSize in engine/light/light_handler_builder.go.
	annotationArgInjectNumThreads AnnotationArg = "num_threads"

	// annotationArgInjectSlotsPerCell injects the number of body slots per physics grid cell.
	// Go source: implicit from GPUGridCell struct layout in engine/physics/gpu_types.go.
	annotationArgInjectSlotsPerCell AnnotationArg = "slots_per_cell"

	// annotationArgInjectFlagActive injects the physics body "active" bit flag value.
	// Go source: engine/physics/physics.go — rigid body flag constants.
	annotationArgInjectFlagActive AnnotationArg = "flag_active"

	// annotationArgInjectFlagStatic injects the physics body "static" bit flag value.
	// Go source: engine/physics/physics.go — rigid body flag constants.
	annotationArgInjectFlagStatic AnnotationArg = "flag_static"

	// annotationArgInjectFlagKinematic injects the physics body "kinematic" bit flag value.
	// Go source: engine/physics/physics.go — rigid body flag constants.
	annotationArgInjectFlagKinematic AnnotationArg = "flag_kinematic"

	// annotationArgInjectEmptySentinel injects the empty-cell sentinel value used in physics
	// grid cells to mark unused slots (typically 0xFFFFFFFF).
	// Go source: engine/scene/scene.go — grid cell initialization constant.
	annotationArgInjectEmptySentinel AnnotationArg = "empty_sentinel"

	// annotationArgInjectBodyIdxMask injects the bitmask used to extract the body index
	// from a packed body-index/bone-index u32 in physics shaders.
	// Go source: engine/physics/physics.go — packing logic (bodyIndex | boneIndex<<24).
	annotationArgInjectBodyIdxMask AnnotationArg = "body_idx_mask"

	// annotationArgInjectPCFSamples injects the number of Poisson-disk PCF shadow samples
	// used in CSM and spot/point shadow sampling.
	// Go source: engine/light/shadow_handler.go (or configurable via builder).
	annotationArgInjectPCFSamples AnnotationArg = "pcf_samples"

	// annotationArgInjectPCFSamplesSpot injects the number of Poisson-disk PCF shadow samples
	// used specifically for spot light shadow sampling.
	// Go source: engine/light/shadow_handler.go (or configurable via builder).
	annotationArgInjectPCFSamplesSpot AnnotationArg = "pcf_samples_spot"

	// annotationArgInjectLightTypeDirectional injects the integer constant identifying
	// directional lights in the GPU Light struct's type field.
	// Go source: engine/light/light.go — LightTypeDirectional iota value.
	annotationArgInjectLightTypeDirectional AnnotationArg = "light_type_directional"

	// annotationArgInjectLightTypePoint injects the integer constant identifying
	// point lights in the GPU Light struct's type field.
	// Go source: engine/light/light.go — LightTypePoint iota value.
	annotationArgInjectLightTypePoint AnnotationArg = "light_type_point"

	// annotationArgInjectLightTypeSpot injects the integer constant identifying
	// spot lights in the GPU Light struct's type field.
	// Go source: engine/light/light.go — LightTypeSpot iota value.
	annotationArgInjectLightTypeSpot AnnotationArg = "light_type_spot"

	// annotationArgInjectLuminanceWorkgroupSize injects the luminance compute workgroup tile dimension.
	// Go source: engine/light/composition_handler.go — CompositionHandler.LuminanceWorkgroupSize().
	annotationArgInjectLuminanceWorkgroupSize AnnotationArg = "luminance_workgroup_size"
)

// ── Material binding role arguments ────────────────────────────────────────────
// These qualify individual bindings within a material provider group. They appear
// as the optional fourth argument of an @oxy:provider annotation when the provider
// identity is "material", telling the loader which texture or sampler role each
// binding fulfils without relying on variable-name string matching.

const (
	// AnnotationArgDiffuseTexture identifies a diffuse / base-color texture binding.
	AnnotationArgDiffuseTexture AnnotationArg = "diffuse_texture"

	// AnnotationArgDiffuseSampler identifies the sampler paired with the diffuse texture.
	AnnotationArgDiffuseSampler AnnotationArg = "diffuse_sampler"

	// AnnotationArgNormalTexture identifies a tangent-space normal map texture binding.
	AnnotationArgNormalTexture AnnotationArg = "normal_texture"

	// AnnotationArgNormalSampler identifies the sampler paired with the normal map.
	AnnotationArgNormalSampler AnnotationArg = "normal_sampler"

	// AnnotationArgMetallicRoughnessTexture identifies a combined metallic-roughness texture binding.
	AnnotationArgMetallicRoughnessTexture AnnotationArg = "metallic_roughness_texture"

	// AnnotationArgMetallicRoughnessSampler identifies the sampler paired with the metallic-roughness texture.
	AnnotationArgMetallicRoughnessSampler AnnotationArg = "metallic_roughness_sampler"

	// AnnotationArgSSAOTexture identifies the SSAO blurred occlusion texture binding role.
	AnnotationArgSSAOTexture AnnotationArg = "ssao_texture"

	// AnnotationArgSSAOSampler identifies the sampler paired with the SSAO occlusion texture.
	AnnotationArgSSAOSampler AnnotationArg = "ssao_sampler"

	// AnnotationArgGBufferNormal identifies the G-Buffer normal texture binding role.
	AnnotationArgGBufferNormal AnnotationArg = "gbuffer_normal"

	// AnnotationArgGBufferDepth identifies the G-Buffer depth texture binding role.
	AnnotationArgGBufferDepth AnnotationArg = "gbuffer_depth"

	// AnnotationArgHDRTexture identifies the HDR lit result texture binding role.
	AnnotationArgHDRTexture AnnotationArg = "hdr_texture"

	// AnnotationArgSSROutput identifies the SSR compute output storage texture binding role.
	AnnotationArgSSROutput AnnotationArg = "ssr_output"

	// AnnotationArgSSRTexture identifies the SSR result texture binding role (sampled in composition).
	AnnotationArgSSRTexture AnnotationArg = "ssr_texture"

	// AnnotationArgCompositionSampler identifies the linear sampler binding role for composition.
	AnnotationArgCompositionSampler AnnotationArg = "composition_sampler"

	// AnnotationArgHiZOut identifies the Hi-Z output storage texture binding role.
	AnnotationArgHiZOut AnnotationArg = "hiz_out"

	// AnnotationArgHiZIn identifies the Hi-Z input texture binding role (previous mip read view).
	AnnotationArgHiZIn AnnotationArg = "hiz_in"

	// AnnotationArgHiZTexture identifies the full Hi-Z depth pyramid texture binding role.
	AnnotationArgHiZTexture AnnotationArg = "hiz_texture"

	// AnnotationArgMaxHiZTexture identifies the MAX Hi-Z depth pyramid texture binding role (occlusion culling).
	AnnotationArgMaxHiZTexture AnnotationArg = "hiz_max_texture"

	// AnnotationArgSpotShadowTexture identifies the spot/point shadow atlas depth texture binding role.
	AnnotationArgSpotShadowTexture AnnotationArg = "spot_shadow_texture"

	// AnnotationArgContactShadowTexture identifies the contact shadow occlusion texture binding role.
	AnnotationArgContactShadowTexture AnnotationArg = "contact_shadow_texture"

	// AnnotationArgContactShadowSampler identifies the contact shadow sampler binding role.
	AnnotationArgContactShadowSampler AnnotationArg = "contact_shadow_sampler"

	// AnnotationArgMaterialParams identifies the per-material scalar parameters uniform binding role.
	AnnotationArgMaterialParams AnnotationArg = "material_params"

	// AnnotationArgTAAHDRTexture identifies the current-frame HDR texture binding role for TAA.
	AnnotationArgTAAHDRTexture AnnotationArg = "taa_hdr_texture"

	// AnnotationArgTAAHistoryTexture identifies the history accumulation texture binding role for TAA.
	AnnotationArgTAAHistoryTexture AnnotationArg = "taa_history_texture"

	// AnnotationArgTAADepth identifies the G-Buffer depth texture binding role for TAA reprojection.
	AnnotationArgTAADepth AnnotationArg = "taa_depth"

	// AnnotationArgTAAResolved identifies the TAA resolved output storage texture binding role.
	AnnotationArgTAAResolved AnnotationArg = "taa_resolved"

	// AnnotationArgTAASampler identifies the linear sampler binding role for TAA history sampling.
	AnnotationArgTAASampler AnnotationArg = "taa_sampler"

	// AnnotationArgTAASharpenInput identifies the TAA resolved texture input binding role for CAS.
	AnnotationArgTAASharpenInput AnnotationArg = "taa_sharpen_input"

	// AnnotationArgTAASharpenOutput identifies the sharpened storage texture output binding role for CAS.
	AnnotationArgTAASharpenOutput AnnotationArg = "taa_sharpen_output"
)

var validInjectionKeys = []AnnotationArg{
	annotationArgInjectMaxBones,
	annotationArgInjectMaxSSAOSamples,
	annotationArgInjectTileSize,
	annotationArgInjectMaxLightsPerTile,
	annotationArgInjectNumThreads,
	annotationArgInjectSlotsPerCell,
	annotationArgInjectFlagActive,
	annotationArgInjectFlagStatic,
	annotationArgInjectFlagKinematic,
	annotationArgInjectEmptySentinel,
	annotationArgInjectBodyIdxMask,
	annotationArgInjectPCFSamples,
	annotationArgInjectPCFSamplesSpot,
	annotationArgInjectLightTypeDirectional,
	annotationArgInjectLightTypePoint,
	annotationArgInjectLightTypeSpot,
	annotationArgInjectLuminanceWorkgroupSize,
}

var validWGSLTypes = []string{"u32", "f32", "i32"}

// validStructTypes lists all AnnotationArg values that are accepted as struct type
// arguments in @oxy:include and @oxy:group annotations. Each entry must have a
// corresponding registryEntry in the PreProcessor's structRegistry.
var validStructTypes = []AnnotationArg{
	AnnotationArgCamera,
	annotationArgVertex,
	annotationArgSkinnedVertex,
	AnnotationArgOverlayParams,
	AnnotationArgEffectParams,
	AnnotationArgLight,
	AnnotationArgLightHeader,
	annotationArgLightCullUniforms,
	AnnotationArgShadowUniform,
	AnnotationArgCSMData,
	AnnotationArgLightShadowEntry,
	AnnotationArgTileUniforms,
	AnnotationArgAnimationData,
	AnnotationArgSkeletalAnimationData,
	AnnotationArgAnimationGlobals,
	annotationArgFrustumPlane,
	AnnotationArgGlobalData,
	AnnotationArgIndirectArgs,
	AnnotationArgBoneInfo,
	AnnotationArgInstanceData,
	AnnotationArgModelData,
	AnnotationArgPhysicsBody,
	AnnotationArgPhysicsParticle,
	AnnotationArgPhysicsGrid,
	AnnotationArgPhysicsGlobals,
	AnnotationArgPhysicsGridParams,
	AnnotationArgGBufferOutput,
	annotationArgSSAOParams,
	annotationArgBlurParams,
	annotationArgCompositionParams,
	annotationArgSSRParams,
	annotationArgContactShadowParams,
	annotationArgLuminanceParams,
	annotationArgBloomParams,
	annotationArgTAAParams,
}

// validAddressSpaces lists all AnnotationArg values that are accepted as address
// space arguments in @oxy:group annotations. Each maps to a WGSL var<> declaration.
var validAddressSpaces = []AnnotationArg{
	annotationArgStorageTypeUniform,
	annotationArgStorageTypeRead,
	annotationArgStorageTypeReadWrite,
}

// validProviderIdentities lists all AnnotationArg values that are accepted as
// provider identity arguments in @oxy:provider annotations. Each maps to a
// Scene-level resource provider used during draw call and compute setup wiring.
var validProviderIdentities = []AnnotationArg{
	AnnotationArgCamera,
	AnnotationArgMaterial,
	AnnotationArgLights,
	AnnotationArgShadow,
	AnnotationArgTiles,
	AnnotationArgEffect,
	AnnotationArgAnimator,
	AnnotationArgAnimatorOutput,
	AnnotationArgAnimatorPacked,
	AnnotationArgAnimatorScratch,
	AnnotationArgSSAO,
	AnnotationArgComposition,
	AnnotationArgSSR,
	AnnotationArgHiZInit,
	AnnotationArgHiZDown,
	AnnotationArgHiZDownMax,
	AnnotationArgContactShadows,
	AnnotationArgAnimatorHiZ,
	AnnotationArgAnimatorMaxHiZ,
	AnnotationArgTAA,
	AnnotationArgTAASharpen,
}

// validBindingRoles lists all AnnotationArg values that are accepted as binding
// role qualifiers in @oxy:provider annotations. These identify the semantic purpose
// of individual bindings within a material provider group.
var validBindingRoles = []AnnotationArg{
	AnnotationArgDiffuseTexture,
	AnnotationArgDiffuseSampler,
	AnnotationArgNormalTexture,
	AnnotationArgNormalSampler,
	AnnotationArgMetallicRoughnessTexture,
	AnnotationArgMetallicRoughnessSampler,
	AnnotationArgSSAOTexture,
	AnnotationArgSSAOSampler,
	AnnotationArgGBufferNormal,
	AnnotationArgGBufferDepth,
	AnnotationArgHDRTexture,
	AnnotationArgSSROutput,
	AnnotationArgSSRTexture,
	AnnotationArgCompositionSampler,
	AnnotationArgHiZOut,
	AnnotationArgHiZIn,
	AnnotationArgHiZTexture,
	AnnotationArgMaxHiZTexture,
	AnnotationArgSpotShadowTexture,
	AnnotationArgContactShadowTexture,
	AnnotationArgContactShadowSampler,
	AnnotationArgMaterialParams,
	AnnotationArgTAAHDRTexture,
	AnnotationArgTAAHistoryTexture,
	AnnotationArgTAADepth,
	AnnotationArgTAAResolved,
	AnnotationArgTAASampler,
	AnnotationArgTAASharpenInput,
	AnnotationArgTAASharpenOutput,
}

// parseAnnotation attempts to parse a single line of WGSL source as an @oxy: annotation.
// Returns nil with no error for lines that do not contain the annotation prefix. Returns
// a populated Annotation for valid annotations, or an error describing the problem for
// malformed annotations with correct prefix but invalid syntax or unknown arguments.
//
// Parameters:
//   - line: the raw WGSL source line to parse
//   - lineNum: the 1-based line number for error reporting
//
// Returns:
//   - *Annotation: the parsed annotation, or nil if the line is not an annotation
//   - error: a descriptive error if the annotation is malformed
func parseAnnotation(line string, lineNum int) (*Annotation, error) {
	trimmed := strings.TrimSpace(line)
	_, after, ok := strings.Cut(trimmed, annotationPrefix)
	if !ok {
		return nil, nil
	}

	args := strings.Fields(after)
	if len(args) == 0 {
		return nil, fmt.Errorf("line %d: empty @oxy annotation", lineNum)
	}

	switch args[0] {
	case string(annotationTypeInclude):
		if len(args) != 2 {
			return nil, fmt.Errorf("line %d: @oxy include annotation requires exactly one argument", lineNum)
		}
		if !slices.Contains(validStructTypes, AnnotationArg(args[1])) {
			return nil, fmt.Errorf("line %d: unknown struct type %q in @oxy include annotation", lineNum, args[1])
		}
		return &Annotation{
			Type: annotationTypeInclude,
			Args: []AnnotationArg{AnnotationArg(args[1])},
			Line: lineNum,
		}, nil
	case string(AnnotationTypeBindingGroup):
		if len(args) != 6 {
			return nil, fmt.Errorf("line %d: @oxy group annotation requires exactly four arguments (group number, binding number, address space, struct type)", lineNum)
		}
		groupInt, err := strconv.Atoi(args[1])
		if err != nil {
			return nil, fmt.Errorf("line %d: invalid group number %q in @oxy group annotation: %v", lineNum, args[1], err)
		}
		bindingInt, err := strconv.Atoi(args[2])
		if err != nil {
			return nil, fmt.Errorf("line %d: invalid binding number %q in @oxy group annotation: %v", lineNum, args[2], err)
		}
		if !slices.Contains(validAddressSpaces, AnnotationArg(args[3])) {
			return nil, fmt.Errorf("line %d: unknown address space %q in @oxy group annotation", lineNum, args[3])
		}
		typeArg := args[5]
		if inner, ok := strings.CutPrefix(typeArg, "array<"); ok {
			inner = strings.TrimSuffix(inner, ">")
			if !slices.Contains(validStructTypes, AnnotationArg(inner)) {
				return nil, fmt.Errorf("line %d: unknown array element type %q in @oxy group annotation", lineNum, inner)
			}
		} else {
			if !slices.Contains(validStructTypes, AnnotationArg(typeArg)) {
				return nil, fmt.Errorf("line %d: unknown struct type %q in @oxy group annotation", lineNum, typeArg)
			}
		}
		return &Annotation{
			Type:    AnnotationTypeBindingGroup,
			Args:    []AnnotationArg{AnnotationArg(args[3]), AnnotationArg(args[4]), AnnotationArg(args[5])},
			Line:    lineNum,
			Group:   &groupInt,
			Binding: &bindingInt,
		}, nil
	case string(annotationTypeInject):
		if len(args) != 4 {
			return nil, fmt.Errorf("line %d: @oxy:inject annotation requires exactly three arguments (const_name, wgsl_type, injection_key)", lineNum)
		}
		if !slices.Contains(validWGSLTypes, args[2]) {
			return nil, fmt.Errorf("line %d: unsupported WGSL type %q in @oxy:inject annotation", lineNum, args[2])
		}
		if !slices.Contains(validInjectionKeys, AnnotationArg(args[3])) {
			return nil, fmt.Errorf("line %d: unknown injection key %q in @oxy:inject annotation", lineNum, args[3])
		}
		return &Annotation{
			Type: annotationTypeInject,
			Args: []AnnotationArg{AnnotationArg(args[1]), AnnotationArg(args[2]), AnnotationArg(args[3])},
			Line: lineNum,
		}, nil
	case string(AnnotationTypeProvider):
		if len(args) < 4 || len(args) > 5 {
			return nil, fmt.Errorf("line %d: @oxy provider annotation requires three or four arguments (group, binding, provider identity[, binding role])", lineNum)
		}
		groupInt, err := strconv.Atoi(args[1])
		if err != nil {
			return nil, fmt.Errorf("line %d: invalid group number %q: %v", lineNum, args[1], err)
		}
		bindingInt, err := strconv.Atoi(args[2])
		if err != nil {
			return nil, fmt.Errorf("line %d: invalid binding number %q in @oxy provider annotation: %v", lineNum, args[2], err)
		}
		if !slices.Contains(validProviderIdentities, AnnotationArg(args[3])) {
			return nil, fmt.Errorf("line %d: unknown provider identity %q in @oxy provider annotation", lineNum, args[3])
		}
		providerArgs := []AnnotationArg{AnnotationArg(args[3])}
		if len(args) == 5 {
			if !slices.Contains(validBindingRoles, AnnotationArg(args[4])) {
				return nil, fmt.Errorf("line %d: unknown binding role %q in @oxy provider annotation", lineNum, args[4])
			}
			providerArgs = append(providerArgs, AnnotationArg(args[4]))
		}
		return &Annotation{
			Type:    AnnotationTypeProvider,
			Args:    providerArgs,
			Line:    lineNum,
			Group:   &groupInt,
			Binding: &bindingInt,
		}, nil
	default:
		return nil, fmt.Errorf("line %d: unknown @oxy annotation type %q", lineNum, args[0])
	}
}
