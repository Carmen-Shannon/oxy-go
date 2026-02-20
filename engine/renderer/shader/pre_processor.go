// pre_processor.go implements the Oxy WGSL shader pre-processor. It scans shader
// source code for @oxy: annotations, replaces them with generated WGSL declarations
// or injected struct source, and collects a declarations list that the Scene uses to
// semantically wire GPU resources to bind groups without manual string lookups.
//
// The pre-processor maintains two registries:
//   - structRegistry: maps AnnotationArg keys to embedded WGSL struct sources and their
//     resolved type names. Used by @oxy:include (to inject the struct source) and
//     @oxy:group (to resolve the WGSL type name in the generated declaration).
//   - addressSpaceRegistry: maps address space argument keys to WGSL var<> syntax strings.
//
// See ANNOTATIONS_README.md at the repository root for full annotation documentation.
package shader

import (
	"fmt"
	"strings"

	"github.com/Carmen-Shannon/oxy-go/engine/camera"
	"github.com/Carmen-Shannon/oxy-go/engine/light"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
)

// registryEntry pairs a WGSL struct source string (embedded from a .wgsl asset file)
// with the resolved WGSL type name used in generated @group/@binding declarations.
type registryEntry struct {
	// Source is the raw WGSL struct definition text injected by @oxy:include.
	Source string

	// Type is the WGSL type name emitted in @oxy:group declarations (e.g. "CameraUniform", "Light").
	Type string
}

// preProcessor is the implementation of the PreProcessor interface.
type preProcessor struct {
	// structRegistry maps struct type argument keys to their embedded WGSL source and type name.
	structRegistry map[AnnotationArg]registryEntry

	// addressSpaceRegistry maps address space argument keys to WGSL var<> syntax strings.
	addressSpaceRegistry map[AnnotationArg]string

	// declarations accumulates annotations of type AnnotationTypeBindingGroup and
	// AnnotationTypeProvider during a Process call. Reset at the start of each Process invocation.
	declarations []Annotation
}

// PreProcessor processes raw WGSL shader source code containing @oxy: annotations,
// replacing them with generated declarations or injected struct sources while collecting
// a declarations list for downstream resource wiring by the Scene.
type PreProcessor interface {
	// Process takes raw WGSL shader source code and pre-processes it by replacing
	// @oxy: annotations with their corresponding WGSL output. @oxy:include annotations
	// are replaced with embedded struct source text. @oxy:group annotations are replaced
	// with generated @group/@binding variable declarations. @oxy:provider annotations
	// produce no WGSL output but are recorded in the declarations list.
	//
	// The declarations list is reset at the start of each call and can be retrieved
	// via Declarations() after Process returns.
	//
	// Parameters:
	//   - source: the raw WGSL shader source code containing annotations to be processed
	//
	// Returns:
	//   - string: the processed WGSL shader source code with annotations replaced
	//   - error: an error if any annotation is malformed or references an unknown type
	Process(source string) (string, error)

	// Declarations returns the list of AnnotationTypeBindingGroup and AnnotationTypeProvider
	// annotations collected during the most recent call to Process, in source-order.
	// Returns nil if Process has not been called.
	//
	// Returns:
	//   - []Annotation: the declarations collected during the last Process call
	Declarations() []Annotation
}

var _ PreProcessor = &preProcessor{}

// NewPreProcessor creates a new PreProcessor with all registered struct types and
// address space mappings pre-populated. The struct registry maps annotation argument
// keys to their embedded WGSL source and resolved WGSL type names from the engine's
// GPU type packages.
//
// Returns:
//   - PreProcessor: a ready-to-use pre-processor instance
func NewPreProcessor() PreProcessor {
	return &preProcessor{
		structRegistry: map[AnnotationArg]registryEntry{
			AnnotationArgCamera:                {Source: camera.GPUCameraUniformSource, Type: "CameraUniform"},
			annotationArgVertex:                {Source: model.GPUVertexSource, Type: "VertexInput"},
			annotationArgSkinnedVertex:         {Source: model.GPUSkinnedVertexSource, Type: "VertexInput"},
			AnnotationArgOverlayParams:         {Source: material.GPUOverlayParamsSource, Type: "OverlayParams"},
			AnnotationArgEffectParams:          {Source: material.GPUEffectParamsSource, Type: "EffectParams"},
			AnnotationArgLight:                 {Source: light.GPULightSource, Type: "Light"},
			AnnotationArgLightHeader:           {Source: light.GPULightHeaderSource, Type: "LightHeader"},
			AnnotationArgShadowData:            {Source: light.GPUShadowDataSource, Type: "ShadowData"},
			AnnotationArgShadowUniform:         {Source: light.GPUShadowUniformSource, Type: "ShadowUniform"},
			annotationArgLightCullUniforms:     {Source: light.GPULightCullUniformsSource, Type: "LightCullUniforms"},
			AnnotationArgTileUniforms:          {Source: light.GPUTileUniformsSource, Type: "TileUniforms"},
			AnnotationArgAnimationData:         {Source: animator.GPUAnimationDataSource, Type: "AnimationData"},
			AnnotationArgSkeletalAnimationData: {Source: animator.GPUSkeletalAnimationDataSource, Type: "SkeletalAnimationData"},
			AnnotationArgAnimationGlobals:      {Source: animator.GPUAnimationGlobalsSource, Type: "AnimationGlobals"},
			annotationArgFrustumPlane:          {Source: animator.GPUFrustumPlaneSource, Type: "FrustumPlane"},
			AnnotationArgGlobalData:            {Source: animator.GPUGlobalDataSource, Type: "GlobalData"},
			AnnotationArgIndirectArgs:          {Source: animator.GPUIndirectArgsSource, Type: "IndirectArgs"},
			AnnotationArgBoneInfo:              {Source: animator.GPUBoneInfoSource, Type: "BoneInfo"},
			AnnotationArgInstanceData:          {Source: animator.GPUInstanceDataSource, Type: "InstanceData"},
			AnnotationArgModelData:             {Source: model.GPUModelDataSource, Type: "ModelData"},
		},
		addressSpaceRegistry: map[AnnotationArg]string{
			annotationArgStorageTypeUniform:   "var<uniform>",
			annotationArgStorageTypeRead:      "var<storage, read>",
			annotationArgStorageTypeReadWrite: "var<storage, read_write>",
		},
	}
}

func (p *preProcessor) Process(source string) (string, error) {
	p.declarations = p.declarations[:0]

	lines := strings.Split(source, "\n")
	out := make([]string, 0, len(lines))

	// iterate through each line of the source and attempt to parse it as an annotation, if it's an annotation replace it with the corresponding source from the registry, otherwise keep the line as is.
	for i, line := range lines {
		a, err := parseAnnotation(line, i+1)
		if err != nil {
			return "", err
		}
		if a == nil {
			out = append(out, line)
			continue
		}

		// handle annotation based on its type and arguments
		switch a.Type {
		case annotationTypeInclude:
			entry, ok := p.structRegistry[a.Args[0]]
			if !ok {
				return "", fmt.Errorf("line %d: unknown @oxy:include argument %q", i+1, a.Args[0])
			}

			out = append(out, entry.Source)
		case AnnotationTypeBindingGroup:
			addrSpace := p.addressSpaceRegistry[a.Args[0]]
			varName := string(a.Args[1])
			var wgslType string
			if inner, ok := strings.CutPrefix(string(a.Args[2]), "array<"); ok {
				inner = strings.TrimSuffix(inner, ">")
				entry := p.structRegistry[AnnotationArg(inner)]
				wgslType = fmt.Sprintf("array<%s>", entry.Type)
			} else {
				entry := p.structRegistry[a.Args[2]]
				wgslType = entry.Type
			}

			out = append(out, fmt.Sprintf("@group(%d) @binding(%d) %s %s: %s;", *a.Group, *a.Binding, addrSpace, varName, wgslType))
			p.declarations = append(p.declarations, *a)
		case AnnotationTypeProvider:
			p.declarations = append(p.declarations, *a)
		default:
			return "", fmt.Errorf("line %d: unknown annotation type %q", i+1, a.Type)
		}

	}
	return strings.Join(out, "\n"), nil
}

func (p *preProcessor) Declarations() []Annotation {
	return p.declarations
}
