package loader

import (
	"fmt"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// gltfAnimationExtractorImpl is the implementation of the gltfAnimationExtractor interface.
type gltfAnimationExtractorImpl struct {
	parser gltfParser
}

// gltfAnimationExtractor defines the interface for extracting animation data from a parsed glTF document.
// It converts glTF animation definitions into engine-ready AnimationClip structs with keyframe data.
//
// The boneMapping parameter maps glTF node indices to bone indices in the topologically sorted skeleton.
// This is produced by the skeleton extractor and ensures that animation channels target the correct bones
// after skeleton reordering.
type gltfAnimationExtractor interface {
	// ExtractAnimation extracts a single animation by index.
	// The boneMapping maps glTF node indices to sorted skeleton bone indices.
	//
	// Parameters:
	//   - animIndex: the index of the animation in the document
	//   - boneMapping: maps glTF node index to skeleton bone index
	//
	// Returns:
	//   - *model.AnimationClip: the extracted animation clip
	//   - error: error if extraction fails
	ExtractAnimation(animIndex int, boneMapping map[int]int32) (*model.AnimationClip, error)

	// ExtractAnimationsForSkeleton extracts all animations that target joints belonging to a skin.
	//
	// Parameters:
	//   - skinIndex: the skin index whose joint set determines which animations to extract
	//   - boneMapping: maps glTF node index to skeleton bone index
	//
	// Returns:
	//   - []*model.AnimationClip: animations that animate at least one joint of the skin
	//   - error: error if extraction fails
	ExtractAnimationsForSkeleton(skinIndex int, boneMapping map[int]int32) ([]*model.AnimationClip, error)

	// ExtractAllAnimations extracts every animation from the document.
	//
	// Parameters:
	//   - boneMapping: maps glTF node index to skeleton bone index
	//
	// Returns:
	//   - []*model.AnimationClip: all extracted animation clips
	//   - error: error if extraction fails
	ExtractAllAnimations(boneMapping map[int]int32) ([]*model.AnimationClip, error)
}

var _ gltfAnimationExtractor = &gltfAnimationExtractorImpl{}

// newGLTFAnimationExtractor creates a new animation extractor for a parsed document.
//
// Parameters:
//   - parser: the parser containing a loaded document
//
// Returns:
//   - gltfAnimationExtractor: the animation extractor
func newGLTFAnimationExtractor(parser gltfParser) gltfAnimationExtractor {
	return &gltfAnimationExtractorImpl{parser: parser}
}

func (e *gltfAnimationExtractorImpl) ExtractAnimation(animIndex int, boneMapping map[int]int32) (*model.AnimationClip, error) {
	doc := e.parser.Document()
	if doc == nil {
		return nil, fmt.Errorf("no document loaded")
	}
	if animIndex < 0 || animIndex >= len(doc.Animations) {
		return nil, fmt.Errorf("animation index %d out of range", animIndex)
	}

	anim := &doc.Animations[animIndex]

	// channelMap groups channels by bone index so we can merge translation/rotation/scale
	// into a single AnimationChannel per bone.
	channelMap := make(map[int32]*model.AnimationChannel)

	var maxTime float32

	for i := range anim.Channels {
		ch := &anim.Channels[i]

		// Skip channels with no target node (e.g. morph targets)
		if ch.Target.Node == nil {
			continue
		}
		nodeIndex := *ch.Target.Node

		// Map glTF node index → skeleton bone index
		boneIndex, ok := boneMapping[nodeIndex]
		if !ok {
			// This channel targets a node that isn't in the skeleton; skip it
			continue
		}

		if ch.Sampler < 0 || ch.Sampler >= len(anim.Samplers) {
			return nil, fmt.Errorf("animation %q channel %d: invalid sampler index %d", anim.Name, i, ch.Sampler)
		}
		sampler := &anim.Samplers[ch.Sampler]

		// Read keyframe timestamps
		timestamps, err := e.parser.ReadScalarAccessor(sampler.Input)
		if err != nil {
			return nil, fmt.Errorf("animation %q channel %d: failed to read timestamps: %w", anim.Name, i, err)
		}

		// Track max timestamp for duration
		if len(timestamps) > 0 {
			if t := timestamps[len(timestamps)-1]; t > maxTime {
				maxTime = t
			}
		}

		// Get or create channel entry for this bone
		animCh, exists := channelMap[boneIndex]
		if !exists {
			animCh = &model.AnimationChannel{BoneIndex: boneIndex}
			channelMap[boneIndex] = animCh
		}

		// Read and store keyframe values based on target path
		switch ch.Target.Path {
		case gltfAnimPathTranslation:
			values, err := e.parser.ReadVec3Accessor(sampler.Output)
			if err != nil {
				return nil, fmt.Errorf("animation %q channel %d: failed to read translation values: %w", anim.Name, i, err)
			}
			keys := make([]model.VectorKeyframe, min(len(timestamps), len(values)))
			for j := range keys {
				keys[j] = model.VectorKeyframe{Time: timestamps[j], Value: values[j]}
			}
			animCh.PositionKeys = keys

		case gltfAnimPathRotation:
			values, err := e.parser.ReadVec4Accessor(sampler.Output)
			if err != nil {
				return nil, fmt.Errorf("animation %q channel %d: failed to read rotation values: %w", anim.Name, i, err)
			}
			keys := make([]model.QuaternionKeyframe, min(len(timestamps), len(values)))
			for j := range keys {
				keys[j] = model.QuaternionKeyframe{Time: timestamps[j], Value: values[j]}
			}
			animCh.RotationKeys = keys

		case gltfAnimPathScale:
			values, err := e.parser.ReadVec3Accessor(sampler.Output)
			if err != nil {
				return nil, fmt.Errorf("animation %q channel %d: failed to read scale values: %w", anim.Name, i, err)
			}
			keys := make([]model.VectorKeyframe, min(len(timestamps), len(values)))
			for j := range keys {
				keys[j] = model.VectorKeyframe{Time: timestamps[j], Value: values[j]}
			}
			animCh.ScaleKeys = keys

		case gltfAnimPathWeights:
			// Morph target weights are not supported; skip
			continue
		}
	}

	// Flatten channel map into slice
	channels := make([]model.AnimationChannel, 0, len(channelMap))
	for _, ch := range channelMap {
		channels = append(channels, *ch)
	}

	name := anim.Name
	if name == "" {
		name = fmt.Sprintf("animation_%d", animIndex)
	}

	return &model.AnimationClip{
		Name:           name,
		Duration:       maxTime,
		TicksPerSecond: 1.0, // glTF timestamps are always in seconds
		Channels:       channels,
	}, nil
}

func (e *gltfAnimationExtractorImpl) ExtractAnimationsForSkeleton(skinIndex int, boneMapping map[int]int32) ([]*model.AnimationClip, error) {
	doc := e.parser.Document()
	if doc == nil {
		return nil, fmt.Errorf("no document loaded")
	}
	if skinIndex < 0 || skinIndex >= len(doc.Skins) {
		return nil, fmt.Errorf("skin index %d out of range", skinIndex)
	}

	// Build set of joint node indices for this skin
	jointSet := make(map[int]bool)
	for _, j := range doc.Skins[skinIndex].Joints {
		jointSet[j] = true
	}

	var clips []*model.AnimationClip

	for animIdx := range doc.Animations {
		anim := &doc.Animations[animIdx]

		// Check if any channel targets a joint of this skin
		relevant := false
		for _, ch := range anim.Channels {
			if ch.Target.Node != nil && jointSet[*ch.Target.Node] {
				relevant = true
				break
			}
		}
		if !relevant {
			continue
		}

		clip, err := e.ExtractAnimation(animIdx, boneMapping)
		if err != nil {
			return nil, fmt.Errorf("animation %d: %w", animIdx, err)
		}
		clips = append(clips, clip)
	}

	return clips, nil
}

func (e *gltfAnimationExtractorImpl) ExtractAllAnimations(boneMapping map[int]int32) ([]*model.AnimationClip, error) {
	doc := e.parser.Document()
	if doc == nil {
		return nil, fmt.Errorf("no document loaded")
	}

	clips := make([]*model.AnimationClip, len(doc.Animations))
	for i := range doc.Animations {
		clip, err := e.ExtractAnimation(i, boneMapping)
		if err != nil {
			return nil, fmt.Errorf("animation %d: %w", i, err)
		}
		clips[i] = clip
	}

	return clips, nil
}
