package loader

import (
	"fmt"
	"math"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// gltfSkeletonExtractorImpl is the implementation of the gltfSkeletonExtractor interface.
type gltfSkeletonExtractorImpl struct {
	parser gltfParser
}

// gltfSkeletonExtractor defines the interface for extracting skeleton/bone data from a parsed glTF document.
// It converts glTF skin definitions into engine-ready Skeleton structs with topologically sorted bones.
type gltfSkeletonExtractor interface {
	// ExtractSkeleton extracts a skeleton from a skin by index.
	//
	// Parameters:
	//   - skinIndex: the index of the skin to extract
	//
	// Returns:
	//   - *model.Skeleton: the extracted skeleton with topologically sorted bones
	//   - error: error if extraction fails
	ExtractSkeleton(skinIndex int) (*model.Skeleton, error)

	// ExtractSkeletonWithMapping extracts a skeleton and returns the old-to-new bone index mapping.
	// This mapping is needed to remap mesh bone indices after topological sorting.
	//
	// Parameters:
	//   - skinIndex: the index of the skin to extract
	//
	// Returns:
	//   - *model.Skeleton: the extracted skeleton
	//   - map[int32]int32: mapping from old bone index to new bone index
	//   - error: error if extraction fails
	ExtractSkeletonWithMapping(skinIndex int) (*model.Skeleton, map[int32]int32, error)

	// ExtractAllSkeletons extracts all skeletons from the document.
	//
	// Returns:
	//   - []*model.Skeleton: all skeletons
	//   - error: error if extraction fails
	ExtractAllSkeletons() ([]*model.Skeleton, error)

	// FindSkeletonForMesh finds which skeleton (skin) is associated with a mesh.
	// Returns -1 if no skeleton is found.
	//
	// Parameters:
	//   - meshIndex: the mesh index to find a skeleton for
	//
	// Returns:
	//   - int: the skin index, or -1 if none
	FindSkeletonForMesh(meshIndex int) int
}

var _ gltfSkeletonExtractor = &gltfSkeletonExtractorImpl{}

// newGLTFSkeletonExtractor creates a new skeleton extractor for a parsed document.
//
// Parameters:
//   - parser: the parser containing a loaded document
//
// Returns:
//   - gltfSkeletonExtractor: the skeleton extractor
func newGLTFSkeletonExtractor(parser gltfParser) gltfSkeletonExtractor {
	return &gltfSkeletonExtractorImpl{parser: parser}
}

func (e *gltfSkeletonExtractorImpl) ExtractSkeleton(skinIndex int) (*model.Skeleton, error) {
	skeleton, _, err := e.extractSkeletonInternal(skinIndex)
	return skeleton, err
}

func (e *gltfSkeletonExtractorImpl) ExtractSkeletonWithMapping(skinIndex int) (*model.Skeleton, map[int32]int32, error) {
	return e.extractSkeletonInternal(skinIndex)
}

func (e *gltfSkeletonExtractorImpl) ExtractAllSkeletons() ([]*model.Skeleton, error) {
	doc := e.parser.Document()
	if doc == nil {
		return nil, fmt.Errorf("no document loaded")
	}

	skeletons := make([]*model.Skeleton, len(doc.Skins))
	for i := range doc.Skins {
		skeleton, err := e.ExtractSkeleton(i)
		if err != nil {
			return nil, fmt.Errorf("skin %d: %w", i, err)
		}
		skeletons[i] = skeleton
	}

	return skeletons, nil
}

func (e *gltfSkeletonExtractorImpl) FindSkeletonForMesh(meshIndex int) int {
	doc := e.parser.Document()
	if doc == nil {
		return -1
	}

	for _, node := range doc.Nodes {
		if node.Mesh != nil && *node.Mesh == meshIndex && node.Skin != nil {
			return *node.Skin
		}
	}

	return -1
}

// extractSkeletonInternal is the shared implementation for ExtractSkeleton and ExtractSkeletonWithMapping.
func (e *gltfSkeletonExtractorImpl) extractSkeletonInternal(skinIndex int) (*model.Skeleton, map[int32]int32, error) {
	doc := e.parser.Document()
	if doc == nil {
		return nil, nil, fmt.Errorf("no document loaded")
	}
	if skinIndex < 0 || skinIndex >= len(doc.Skins) {
		return nil, nil, fmt.Errorf("skin index %d out of range", skinIndex)
	}

	skin := &doc.Skins[skinIndex]

	// Read inverse bind matrices (optional but usually present)
	var inverseBindMatrices [][16]float32
	if skin.InverseBindMatrices != nil {
		var err error
		inverseBindMatrices, err = e.parser.ReadMat4Accessor(*skin.InverseBindMatrices)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to read inverse bind matrices: %w", err)
		}
	}

	// Build bone hierarchy from joint nodes
	bones := make([]model.Bone, len(skin.Joints))
	boneNameToIndex := make(map[string]int32)

	// First pass: create bones and map names
	for i, jointIndex := range skin.Joints {
		if jointIndex < 0 || jointIndex >= len(doc.Nodes) {
			return nil, nil, fmt.Errorf("joint %d: invalid node index %d", i, jointIndex)
		}

		node := &doc.Nodes[jointIndex]
		bone := &bones[i]

		bone.Name = node.Name
		if bone.Name == "" {
			bone.Name = fmt.Sprintf("bone_%d", i)
		}
		boneNameToIndex[bone.Name] = int32(i)

		if i < len(inverseBindMatrices) {
			bone.InverseBindMatrix = inverseBindMatrices[i]
		} else {
			bone.InverseBindMatrix = gltfIdentityMatrix()
		}

		bone.LocalTransform = gltfExtractNodeTransform(node)
	}

	// Second pass: establish parent relationships
	nodeIndexToBoneIndex := make(map[int]int32)
	for boneIdx, jointNodeIdx := range skin.Joints {
		nodeIndexToBoneIndex[jointNodeIdx] = int32(boneIdx)
	}

	var rootBoneIndices []int32
	for boneIdx, jointNodeIdx := range skin.Joints {
		parentFound := false

		for nodeIdx, node := range doc.Nodes {
			for _, childIdx := range node.Children {
				if childIdx == jointNodeIdx {
					if parentBoneIdx, ok := nodeIndexToBoneIndex[nodeIdx]; ok {
						bones[boneIdx].ParentIndex = parentBoneIdx
						parentFound = true
					}
					break
				}
			}
			if parentFound {
				break
			}
		}

		if !parentFound {
			bones[boneIdx].ParentIndex = -1
			rootBoneIndices = append(rootBoneIndices, int32(boneIdx))
		}
	}

	// Sort bones in topological order (parents before children)
	sortedBones, sortedRootIndices, sortedNameToIndex, oldToNewMapping := gltfTopologicalSortBones(bones, rootBoneIndices, boneNameToIndex)

	return &model.Skeleton{
		Bones:           sortedBones,
		RootBoneIndices: sortedRootIndices,
		BoneNameToIndex: sortedNameToIndex,
	}, oldToNewMapping, nil
}

// --- Helper Functions ---

// gltfExtractNodeTransform extracts TRS transform from a glTF node.
func gltfExtractNodeTransform(node *gltfNode) model.Transform {
	transform := model.Transform{
		Translation: [3]float32{0, 0, 0},
		Rotation:    [4]float32{0, 0, 0, 1},
		Scale:       [3]float32{1, 1, 1},
	}

	if node.Matrix != nil {
		return gltfDecomposeMatrix(*node.Matrix)
	}

	if node.Translation != nil {
		transform.Translation = *node.Translation
	}
	if node.Rotation != nil {
		transform.Rotation = *node.Rotation
	}
	if node.Scale != nil {
		transform.Scale = *node.Scale
	}

	return transform
}

// gltfIdentityMatrix returns a 4x4 identity matrix.
func gltfIdentityMatrix() [16]float32 {
	return [16]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
}

// gltfDecomposeMatrix decomposes a 4x4 column-major matrix into translation, rotation (quaternion), and scale.
// This is an approximation that assumes no shear.
func gltfDecomposeMatrix(m [16]float32) model.Transform {
	var t model.Transform

	// Extract translation (column 3)
	t.Translation = [3]float32{m[12], m[13], m[14]}

	// Extract scale (length of each column)
	sx := gltfVectorLength(m[0], m[1], m[2])
	sy := gltfVectorLength(m[4], m[5], m[6])
	sz := gltfVectorLength(m[8], m[9], m[10])
	t.Scale = [3]float32{sx, sy, sz}

	// Avoid division by zero
	if sx < 0.0001 {
		sx = 1
	}
	if sy < 0.0001 {
		sy = 1
	}
	if sz < 0.0001 {
		sz = 1
	}

	// Build rotation matrix (normalized columns)
	r := [9]float32{
		m[0] / sx, m[1] / sx, m[2] / sx,
		m[4] / sy, m[5] / sy, m[6] / sy,
		m[8] / sz, m[9] / sz, m[10] / sz,
	}

	// Extract quaternion from rotation matrix
	t.Rotation = gltfMatrixToQuaternion(r)

	return t
}

// gltfVectorLength computes the length of a 3D vector.
func gltfVectorLength(x, y, z float32) float32 {
	return float32(math.Sqrt(float64(x*x + y*y + z*z)))
}

// gltfMatrixToQuaternion converts a 3x3 rotation matrix to a quaternion.
// Matrix is in row-major order: [r00, r01, r02, r10, r11, r12, r20, r21, r22].
// Returns quaternion as [x, y, z, w].
func gltfMatrixToQuaternion(m [9]float32) [4]float32 {
	r00, r01, r02 := m[0], m[1], m[2]
	r10, r11, r12 := m[3], m[4], m[5]
	r20, r21, r22 := m[6], m[7], m[8]

	trace := r00 + r11 + r22

	var x, y, z, w float32

	if trace > 0 {
		s := float32(math.Sqrt(float64(trace+1.0))) * 2
		w = 0.25 * s
		x = (r21 - r12) / s
		y = (r02 - r20) / s
		z = (r10 - r01) / s
	} else if r00 > r11 && r00 > r22 {
		s := float32(math.Sqrt(float64(1.0+r00-r11-r22))) * 2
		w = (r21 - r12) / s
		x = 0.25 * s
		y = (r01 + r10) / s
		z = (r02 + r20) / s
	} else if r11 > r22 {
		s := float32(math.Sqrt(float64(1.0+r11-r00-r22))) * 2
		w = (r02 - r20) / s
		x = (r01 + r10) / s
		y = 0.25 * s
		z = (r12 + r21) / s
	} else {
		s := float32(math.Sqrt(float64(1.0+r22-r00-r11))) * 2
		w = (r10 - r01) / s
		x = (r02 + r20) / s
		y = (r12 + r21) / s
		z = 0.25 * s
	}

	// Normalize quaternion
	length := float32(math.Sqrt(float64(x*x + y*y + z*z + w*w)))
	if length > 0.0001 {
		x /= length
		y /= length
		z /= length
		w /= length
	}

	return [4]float32{x, y, z, w}
}

// gltfTopologicalSortBones sorts bones so that parents always come before children.
// This is required for GPU bone matrix computation where we iterate bones in order
// and multiply by the parent's already-computed world matrix.
//
// Parameters:
//   - bones: original bone array
//   - rootIndices: indices of root bones (no parent)
//   - nameToIndex: original name-to-index mapping
//
// Returns:
//   - []model.Bone: sorted bone array with updated parent indices
//   - []int32: new root indices
//   - map[string]int32: updated name-to-index mapping
//   - map[int32]int32: old bone index to new bone index mapping
func gltfTopologicalSortBones(bones []model.Bone, rootIndices []int32, nameToIndex map[string]int32) ([]model.Bone, []int32, map[string]int32, map[int32]int32) {
	if len(bones) == 0 {
		return bones, rootIndices, nameToIndex, make(map[int32]int32)
	}

	// Build children map (old indices)
	children := make(map[int32][]int32)
	for i, bone := range bones {
		if bone.ParentIndex >= 0 {
			children[bone.ParentIndex] = append(children[bone.ParentIndex], int32(i))
		}
	}

	// BFS from roots to get topological order
	sorted := make([]int32, 0, len(bones))
	queue := make([]int32, 0, len(rootIndices))
	for _, r := range rootIndices {
		queue = append(queue, r)
	}

	for len(queue) > 0 {
		oldIdx := queue[0]
		queue = queue[1:]
		sorted = append(sorted, oldIdx)

		for _, childIdx := range children[oldIdx] {
			queue = append(queue, childIdx)
		}
	}

	// If we didn't get all bones (disconnected), append remaining
	if len(sorted) < len(bones) {
		visited := make(map[int32]bool)
		for _, idx := range sorted {
			visited[idx] = true
		}
		for i := range bones {
			if !visited[int32(i)] {
				sorted = append(sorted, int32(i))
			}
		}
	}

	// Build old-to-new index mapping
	oldToNew := make(map[int32]int32)
	for newIdx, oldIdx := range sorted {
		oldToNew[oldIdx] = int32(newIdx)
	}

	// Create new bone array with updated parent indices
	newBones := make([]model.Bone, len(bones))
	newNameToIndex := make(map[string]int32)
	var newRootIndices []int32

	for newIdx, oldIdx := range sorted {
		bone := bones[oldIdx]

		if bone.ParentIndex >= 0 {
			bone.ParentIndex = oldToNew[bone.ParentIndex]
		} else {
			newRootIndices = append(newRootIndices, int32(newIdx))
		}

		newBones[newIdx] = bone
		newNameToIndex[bone.Name] = int32(newIdx)
	}

	return newBones, newRootIndices, newNameToIndex, oldToNew
}
