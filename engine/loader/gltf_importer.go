package loader

import (
	"fmt"
	"io"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// gltfImporterImpl is the implementation of the gltfImporter interface.
type gltfImporterImpl struct{}

// gltfImporter defines the interface for orchestrating a full glTF/GLB import.
// It combines the parser and all extractors to produce a complete ImportedModel.
type gltfImporter interface {
	// Import loads a glTF/GLB file and extracts all data into an ImportedModel.
	// This includes meshes, skeleton, animations, and materials.
	//
	// Parameters:
	//   - path: the file path to the glTF or GLB file
	//
	// Returns:
	//   - *model.ImportedModel: the fully populated imported model
	//   - error: error if import fails
	Import(path string) (*model.ImportedModel, error)

	// ImportReader loads a glTF document from a reader and extracts all data.
	// The reader should provide a complete glTF JSON or GLB binary stream.
	//
	// Parameters:
	//   - r: the reader providing glTF/GLB data
	//   - isGLB: true if the reader provides GLB binary data, false for glTF JSON
	//
	// Returns:
	//   - *model.ImportedModel: the fully populated imported model
	//   - error: error if import fails
	ImportReader(r io.Reader, isGLB bool) (*model.ImportedModel, error)

	// ImportMeshOnly loads a glTF/GLB file and extracts only mesh and material data.
	// Skeleton and animation extraction is skipped for faster loading of static models.
	//
	// Parameters:
	//   - path: the file path to the glTF or GLB file
	//
	// Returns:
	//   - *model.ImportedModel: the imported model with meshes and materials only
	//   - error: error if import fails
	ImportMeshOnly(path string) (*model.ImportedModel, error)
}

var _ gltfImporter = &gltfImporterImpl{}

// newGLTFImporter creates a new glTF importer.
//
// Returns:
//   - gltfImporter: the importer
func newGLTFImporter() gltfImporter {
	return &gltfImporterImpl{}
}

func (imp *gltfImporterImpl) Import(path string) (*model.ImportedModel, error) {
	parser := newGLTFParser()
	if err := parser.Parse(path); err != nil {
		return nil, fmt.Errorf("failed to parse %s: %w", path, err)
	}

	return imp.importFromParser(parser, path)
}

func (imp *gltfImporterImpl) ImportReader(r io.Reader, isGLB bool) (*model.ImportedModel, error) {
	parser := newGLTFParser()
	if err := parser.ParseReader(r, isGLB); err != nil {
		return nil, fmt.Errorf("failed to parse from reader: %w", err)
	}

	return imp.importFromParser(parser, "")
}

func (imp *gltfImporterImpl) ImportMeshOnly(path string) (*model.ImportedModel, error) {
	parser := newGLTFParser()
	if err := parser.Parse(path); err != nil {
		return nil, fmt.Errorf("failed to parse %s: %w", path, err)
	}

	doc := parser.Document()
	if doc == nil {
		return nil, fmt.Errorf("no document after parsing")
	}

	meshExtractor := newGLTFMeshExtractor(parser)
	materialExtractor := newGLTFMaterialExtractor(parser)

	// Extract meshes
	meshes, err := meshExtractor.ExtractAllMeshes()
	if err != nil {
		return nil, fmt.Errorf("mesh extraction failed: %w", err)
	}

	// Extract materials
	var materials []*common.ImportedMaterial
	if len(doc.Materials) > 0 {
		materials, err = materialExtractor.ExtractAllMaterials()
		if err != nil {
			return nil, fmt.Errorf("material extraction failed: %w", err)
		}
	}

	name := gltfExtractModelName(doc, path)

	return &model.ImportedModel{
		Name:      name,
		Meshes:    meshes,
		Materials: gltfFlattenMaterials(materials),
	}, nil
}

// importFromParser performs a full import from a parser that has already loaded a document.
//
// Parameters:
//   - parser: the glTF parser that has already loaded a document
//   - fallbackPath: optional file path used as a fallback for model naming
func (imp *gltfImporterImpl) importFromParser(parser gltfParser, fallbackPath string) (*model.ImportedModel, error) {
	doc := parser.Document()
	if doc == nil {
		return nil, fmt.Errorf("no document after parsing")
	}

	meshExtractor := newGLTFMeshExtractor(parser)
	skeletonExtractor := newGLTFSkeletonExtractor(parser)
	animationExtractor := newGLTFAnimationExtractor(parser)
	materialExtractor := newGLTFMaterialExtractor(parser)

	// Extract all meshes
	meshes, err := meshExtractor.ExtractAllMeshes()
	if err != nil {
		return nil, fmt.Errorf("mesh extraction failed: %w", err)
	}

	// Extract skeleton if any skins exist.
	// For simplicity, we use the first skin as the model's skeleton.
	// Most glTF models have a single skin per mesh group.
	var skeleton *model.Skeleton
	var boneMapping map[int]int32 // glTF node index → sorted bone index
	var oldToNew map[int32]int32  // pre-sort bone index → post-sort bone index

	if len(doc.Skins) > 0 {
		skinIndex := 0

		// Try to find the skin associated with the first mesh
		if len(doc.Meshes) > 0 {
			if si := skeletonExtractor.FindSkeletonForMesh(0); si >= 0 {
				skinIndex = si
			}
		}

		var err error
		skeleton, oldToNew, err = skeletonExtractor.ExtractSkeletonWithMapping(skinIndex)
		if err != nil {
			return nil, fmt.Errorf("skeleton extraction failed: %w", err)
		}

		// Build the node → new bone index mapping for animations.
		// skin.Joints[i] = glTF node index for original bone index i.
		// oldToNew[i] = new bone index after topological sort.
		skin := &doc.Skins[skinIndex]
		boneMapping = make(map[int]int32, len(skin.Joints))
		for originalBoneIdx, nodeIdx := range skin.Joints {
			newBoneIdx, ok := oldToNew[int32(originalBoneIdx)]
			if ok {
				boneMapping[nodeIdx] = newBoneIdx
			}
		}

		// Remap bone indices in mesh vertices to match sorted skeleton
		gltfRemapMeshBoneIndices(meshes, oldToNew)
	}

	// Extract animations
	var animations []*model.AnimationClip
	if len(doc.Animations) > 0 && boneMapping != nil {
		// Use the skin-scoped extraction so we only get relevant animations
		skinIndex := 0
		if len(doc.Meshes) > 0 {
			if si := skeletonExtractor.FindSkeletonForMesh(0); si >= 0 {
				skinIndex = si
			}
		}

		animations, err = animationExtractor.ExtractAnimationsForSkeleton(skinIndex, boneMapping)
		if err != nil {
			return nil, fmt.Errorf("animation extraction failed: %w", err)
		}
	} else if len(doc.Animations) > 0 {
		// No skeleton but animations exist — extract with empty mapping
		animations, err = animationExtractor.ExtractAllAnimations(make(map[int]int32))
		if err != nil {
			return nil, fmt.Errorf("animation extraction failed: %w", err)
		}
	}

	// Extract materials
	var materials []*common.ImportedMaterial
	if len(doc.Materials) > 0 {
		materials, err = materialExtractor.ExtractAllMaterials()
		if err != nil {
			return nil, fmt.Errorf("material extraction failed: %w", err)
		}
	}

	name := gltfExtractModelName(doc, fallbackPath)

	return &model.ImportedModel{
		Name:       name,
		Meshes:     meshes,
		Skeleton:   skeleton,
		Animations: animations,
		Materials:  gltfFlattenMaterials(materials),
	}, nil
}

// --- Helper Functions ---

// gltfRemapMeshBoneIndices updates the BoneIndices of all skinned vertices
// to reflect the topologically sorted bone order from the skeleton extractor.
func gltfRemapMeshBoneIndices(meshes []model.ImportedMesh, oldToNew map[int32]int32) {
	if len(oldToNew) == 0 {
		return
	}

	for i := range meshes {
		for j := range meshes[i].Vertices {
			v := &meshes[i].Vertices[j]
			for k := 0; k < 4; k++ {
				if newIdx, ok := oldToNew[int32(v.BoneIndices[k])]; ok {
					v.BoneIndices[k] = uint32(newIdx)
				}
			}
		}
	}
}

// gltfExtractModelName derives a model name from the document asset or a file path fallback.
func gltfExtractModelName(doc *gltfDocument, fallbackPath string) string {
	// Try scene name first
	if doc.Scene != nil && *doc.Scene < len(doc.Scenes) {
		if name := doc.Scenes[*doc.Scene].Name; name != "" {
			return name
		}
	}

	// Try asset title from extras (not standard, but common)
	if doc.Asset.Generator != "" {
		// Asset.Generator is not a good name, skip
	}

	if fallbackPath != "" {
		return fallbackPath
	}

	return "unnamed_model"
}

// gltfFlattenMaterials converts a slice of material pointers to a value slice.
func gltfFlattenMaterials(materials []*common.ImportedMaterial) []common.ImportedMaterial {
	if materials == nil {
		return nil
	}

	result := make([]common.ImportedMaterial, len(materials))
	for i, m := range materials {
		if m != nil {
			result[i] = *m
		}
	}

	return result
}
