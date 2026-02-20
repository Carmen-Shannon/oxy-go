package model

import (
	"github.com/Carmen-Shannon/oxy-go/common"
)

// --- Transform & Skeleton Types ---

// Transform represents a decomposed transform for animation interpolation.
type Transform struct {
	// Translation is the position offset.
	Translation [3]float32

	// Rotation is the orientation as a quaternion (x, y, z, w).
	Rotation [4]float32

	// Scale is the scale factor along each axis.
	Scale [3]float32
}

// Bone represents a single bone in a skeleton hierarchy.
type Bone struct {
	// Name is the bone's identifier (for debugging and animation targeting).
	Name string

	// ParentIndex is the index of the parent bone (-1 for root bones).
	ParentIndex int32

	// InverseBindMatrix transforms from model space to bone space at bind pose.
	// This is the inverse of the bone's world transform when the mesh was bound.
	InverseBindMatrix [16]float32

	// LocalTransform is the bone's transform relative to its parent.
	// Updated each frame during animation playback.
	LocalTransform Transform
}

// Skeleton represents a bone hierarchy for skeletal animation.
type Skeleton struct {
	// Bones is the array of all bones in the skeleton.
	Bones []Bone

	// RootBoneIndices are indices of bones with no parent.
	RootBoneIndices []int32

	// BoneNameToIndex maps bone names to their indices for quick lookup.
	BoneNameToIndex map[string]int32
}

// --- Animation Types ---

// AnimationClip represents a single animation (walk, run, attack, etc.).
type AnimationClip struct {
	// Name is the animation identifier.
	Name string

	// Duration is the total length of the animation in seconds.
	Duration float32

	// TicksPerSecond is the sample rate of the animation.
	TicksPerSecond float32

	// Channels contains animation data for each animated bone.
	Channels []AnimationChannel
}

// AnimationChannel contains keyframe data for a single bone.
type AnimationChannel struct {
	// BoneIndex is the index of the bone this channel animates.
	BoneIndex int32

	// PositionKeys are keyframes for translation.
	PositionKeys []VectorKeyframe

	// RotationKeys are keyframes for rotation (quaternion).
	RotationKeys []QuaternionKeyframe

	// ScaleKeys are keyframes for scale.
	ScaleKeys []VectorKeyframe
}

// VectorKeyframe stores a 3D vector value at a specific time.
type VectorKeyframe struct {
	// Time is the keyframe timestamp in seconds.
	Time float32

	// Value is the 3D vector value at this keyframe.
	Value [3]float32
}

// QuaternionKeyframe stores a quaternion rotation at a specific time.
type QuaternionKeyframe struct {
	// Time is the keyframe timestamp in seconds.
	Time float32

	// Value is the quaternion value at this keyframe (x, y, z, w).
	Value [4]float32
}

// --- Import Types ---

// ImportedModel represents a 3D model loaded from an external format.
// This is the universal format that importers (glTF, FBX, etc.) produce.
type ImportedModel struct {
	// Name is the model identifier.
	Name string

	// Meshes contains all mesh data (may have multiple meshes/submeshes).
	Meshes []ImportedMesh

	// Skeleton is the bone hierarchy (nil for static models).
	Skeleton *Skeleton

	// Animations are all animation clips bundled with the model.
	Animations []*AnimationClip

	// Materials are referenced materials (indices into a material library).
	Materials []common.ImportedMaterial
}

// ImportedMesh represents a single mesh within an imported model.
type ImportedMesh struct {
	// Name is the mesh identifier.
	Name string

	// Vertices are the mesh vertices for all models, including bone skinning data for animated meshes.
	Vertices []GPUSkinnedVertex

	// Indices are the triangle indices.
	Indices []uint32

	// MaterialIndex references ImportedModel.Materials.
	MaterialIndex int

	// BoundingMin is the minimum corner of the axis-aligned bounding box.
	BoundingMin [3]float32

	// BoundingMax is the maximum corner of the axis-aligned bounding box.
	BoundingMax [3]float32
}
