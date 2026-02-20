// annotations.go defines the annotation types, argument constants, and parser for the
// Oxy WGSL shader pre-processor. Annotations are single-line WGSL comments prefixed
// with @oxy: that drive automatic struct injection, bind group declaration, and resource
// provider registration. The parsed results are stored as Annotation values and consumed
// by the PreProcessor and Scene to wire GPU resources without manual low-level plumbing.
//
// See ANNOTATIONS_README.md at the repository root for full syntax documentation and examples.
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
	// Source: engine/camera/assets/camera_uniform.wgsl
	AnnotationArgCamera AnnotationArg = "camera"

	// annotationArgVertex identifies the VertexInput struct for static (non-skinned) meshes.
	// Source: engine/model/assets/vertex.wgsl
	annotationArgVertex AnnotationArg = "vertex"

	// annotationArgSkinnedVertex identifies the VertexInput struct for skinned meshes with bone weights.
	// Source: engine/model/assets/skinned_vertex.wgsl
	annotationArgSkinnedVertex AnnotationArg = "skinned_vertex"

	// AnnotationArgOverlayParams identifies the OverlayParams material struct.
	// Source: engine/renderer/material/assets/overlay_params.wgsl
	AnnotationArgOverlayParams AnnotationArg = "overlay_params"

	// AnnotationArgEffectParams identifies the EffectParams material struct.
	// Source: engine/renderer/material/assets/effect_params.wgsl
	AnnotationArgEffectParams AnnotationArg = "effect_params"

	// AnnotationArgLight identifies the Light struct for per-light GPU data.
	// Source: engine/light/assets/light.wgsl
	AnnotationArgLight AnnotationArg = "light"

	// AnnotationArgLightHeader identifies the LightHeader struct containing light count and ambient color.
	// Source: engine/light/assets/light_header.wgsl
	AnnotationArgLightHeader AnnotationArg = "light_header"

	// annotationArgLightCullUniforms identifies the LightCullUniforms struct for tile-based light culling.
	// Source: engine/light/assets/light_cull_uniforms.wgsl
	annotationArgLightCullUniforms AnnotationArg = "light_cull_uniforms"

	// AnnotationArgShadowData identifies the ShadowData struct for the lit fragment shader's shadow sampling.
	// Source: engine/light/assets/shadow_data.wgsl
	AnnotationArgShadowData AnnotationArg = "shadow_data"

	// AnnotationArgShadowUniform identifies the ShadowUniform struct for the shadow depth pass.
	// Source: engine/light/assets/shadow_uniform.wgsl
	AnnotationArgShadowUniform AnnotationArg = "shadow_uniform"

	// AnnotationArgTileUniforms identifies the TileUniforms struct for Forward+ tile configuration.
	// Source: engine/light/assets/tile_uniforms.wgsl
	AnnotationArgTileUniforms AnnotationArg = "tile_uniforms"

	// AnnotationArgModelData identifies the ModelData struct holding per-instance model matrices.
	// Source: engine/model/assets/model_data.wgsl
	AnnotationArgModelData AnnotationArg = "model_data"

	// AnnotationArgInstanceData identifies the InstanceData struct for per-instance transform output.
	// Source: engine/renderer/animator/assets/instance_data.wgsl
	AnnotationArgInstanceData AnnotationArg = "instance_data"

	// AnnotationArgAnimationData identifies the AnimationData struct for simple (non-skeletal) animation state.
	// Source: engine/renderer/animator/assets/animation_data.wgsl
	AnnotationArgAnimationData AnnotationArg = "animation_data"

	// AnnotationArgSkeletalAnimationData identifies the SkeletalAnimationData struct for skeletal animation state.
	// Source: engine/renderer/animator/assets/skeletal_animation_data.wgsl
	AnnotationArgSkeletalAnimationData AnnotationArg = "skeletal_animation_data"

	// AnnotationArgAnimationGlobals identifies the AnimationGlobals struct for skeletal compute uniforms.
	// Source: engine/renderer/animator/assets/animation_globals.wgsl
	AnnotationArgAnimationGlobals AnnotationArg = "animation_globals"

	// annotationArgFrustumPlane identifies the FrustumPlane struct used inside uniform structs for GPU culling.
	// Source: engine/renderer/animator/assets/frustum_plane.wgsl
	annotationArgFrustumPlane AnnotationArg = "frustum_plane"

	// AnnotationArgGlobalData identifies the GlobalData struct for simple compute shader uniforms.
	// Source: engine/renderer/animator/assets/simple_globals.wgsl
	AnnotationArgGlobalData AnnotationArg = "global_data"

	// AnnotationArgIndirectArgs identifies the IndirectArgs struct matching WebGPU's DrawIndexedIndirect layout.
	// Source: engine/renderer/animator/assets/indirect_args.wgsl
	AnnotationArgIndirectArgs AnnotationArg = "indirect_args"

	// AnnotationArgBoneInfo identifies the BoneInfo struct holding per-bone inverse bind matrices and hierarchy data.
	// Source: engine/renderer/animator/assets/bone_info.wgsl
	AnnotationArgBoneInfo AnnotationArg = "bone_info"
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
)

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
	AnnotationArgShadowData,
	AnnotationArgShadowUniform,
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
