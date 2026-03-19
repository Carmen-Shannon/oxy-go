package loader

import (
	"errors"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/stretchr/testify/suite"
)

// mockGltfParser is a hand-rolled mock for the unexported gltfParser interface.
type mockGltfParser struct {
	doc           *gltfDocument
	readScalarErr error
	readVec3Err   error
	readVec4Err   error
	scalarResult  []float32
	vec3Result    [][3]float32
	vec4Result    [][4]float32
}

func (m *mockGltfParser) Parse(_ string) error                         { return nil }
func (m *mockGltfParser) Document() *gltfDocument                      { return m.doc }
func (m *mockGltfParser) BaseDir() string                              { return "" }
func (m *mockGltfParser) ReadAccessorData(_ int) ([]byte, error)       { return nil, nil }
func (m *mockGltfParser) ReadVec2Accessor(_ int) ([][2]float32, error) { return nil, nil }
func (m *mockGltfParser) ReadVec3Accessor(_ int) ([][3]float32, error) {
	return m.vec3Result, m.readVec3Err
}
func (m *mockGltfParser) ReadVec4Accessor(_ int) ([][4]float32, error) {
	return m.vec4Result, m.readVec4Err
}
func (m *mockGltfParser) ReadScalarAccessor(_ int) ([]float32, error) {
	return m.scalarResult, m.readScalarErr
}
func (m *mockGltfParser) ReadMat4Accessor(_ int) ([][16]float32, error) { return nil, nil }
func (m *mockGltfParser) ReadIndicesAccessor(_ int) ([]uint32, error)   { return nil, nil }
func (m *mockGltfParser) ReadJointsAccessor(_ int) ([][4]uint32, error) { return nil, nil }

type gltfAnimationExtractorImplTest struct {
	suite.Suite
	mockParser *mockGltfParser
	extractor  gltfAnimationExtractor
}

func TestGltfAnimationExtractorImpl(t *testing.T) {
	suite.Run(t, new(gltfAnimationExtractorImplTest))
}

func (suite *gltfAnimationExtractorImplTest) SetupSubTest() {
	suite.mockParser = &mockGltfParser{}
	suite.extractor = newGLTFAnimationExtractor(suite.mockParser)
}

func (suite *gltfAnimationExtractorImplTest) TestExtractAnimation() {
	suite.Run("nil document returns error", func() {
		suite.mockParser.doc = nil
		_, err := suite.extractor.ExtractAnimation(0, nil)
		suite.Error(err)
	})

	suite.Run("animIndex out of range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{{Name: "anim"}},
		}
		_, err := suite.extractor.ExtractAnimation(1, nil)
		suite.Error(err)
	})

	suite.Run("animIndex negative returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{{Name: "anim"}},
		}
		_, err := suite.extractor.ExtractAnimation(-1, nil)
		suite.Error(err)
	})

	suite.Run("channel with nil target node is skipped", func() {
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Test",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: nil, Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		clip, err := suite.extractor.ExtractAnimation(0, map[int]int32{})
		suite.NoError(err)
		suite.NotNil(clip)
		suite.Len(clip.Channels, 0)
	})

	suite.Run("channel targeting node not in boneMapping is skipped", func() {
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Test",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(99), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		clip, err := suite.extractor.ExtractAnimation(0, map[int]int32{0: 0})
		suite.NoError(err)
		suite.NotNil(clip)
		suite.Len(clip.Channels, 0)
	})

	suite.Run("invalid sampler index returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Test",
					Channels: []gltfAnimChannel{
						{Sampler: 5, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{},
				},
			},
		}
		_, err := suite.extractor.ExtractAnimation(0, map[int]int32{0: 0})
		suite.Error(err)
	})

	suite.Run("ReadScalarAccessor error returns error", func() {
		suite.mockParser.readScalarErr = errors.New("fail")
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Test",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		_, err := suite.extractor.ExtractAnimation(0, map[int]int32{0: 0})
		suite.Error(err)
	})

	suite.Run("translation path sets PositionKeys", func() {
		suite.mockParser.scalarResult = []float32{0.0, 0.5}
		suite.mockParser.vec3Result = [][3]float32{{1, 2, 3}, {4, 5, 6}}
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Walk",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		clip, err := suite.extractor.ExtractAnimation(0, map[int]int32{0: 0})
		suite.NoError(err)
		suite.NotNil(clip)
		suite.Len(clip.Channels, 1)
		suite.Len(clip.Channels[0].PositionKeys, 2)
		suite.InDelta(float32(0.0), clip.Channels[0].PositionKeys[0].Time, 1e-6)
		suite.Equal([3]float32{1, 2, 3}, clip.Channels[0].PositionKeys[0].Value)
		suite.InDelta(float32(0.5), clip.Channels[0].PositionKeys[1].Time, 1e-6)
		suite.Equal([3]float32{4, 5, 6}, clip.Channels[0].PositionKeys[1].Value)
	})

	suite.Run("ReadVec3Accessor error on translation returns error", func() {
		suite.mockParser.scalarResult = []float32{0.0}
		suite.mockParser.readVec3Err = errors.New("fail")
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Test",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		_, err := suite.extractor.ExtractAnimation(0, map[int]int32{0: 0})
		suite.Error(err)
	})

	suite.Run("rotation path sets RotationKeys", func() {
		suite.mockParser.scalarResult = []float32{0.0, 1.0}
		suite.mockParser.vec4Result = [][4]float32{{0, 0, 0, 1}, {0, 1, 0, 0}}
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Run",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(1), Path: gltfAnimPathRotation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		clip, err := suite.extractor.ExtractAnimation(0, map[int]int32{1: 2})
		suite.NoError(err)
		suite.NotNil(clip)
		suite.Len(clip.Channels, 1)
		suite.Equal(int32(2), clip.Channels[0].BoneIndex)
		suite.Len(clip.Channels[0].RotationKeys, 2)
		suite.Equal([4]float32{0, 0, 0, 1}, clip.Channels[0].RotationKeys[0].Value)
		suite.Equal([4]float32{0, 1, 0, 0}, clip.Channels[0].RotationKeys[1].Value)
	})

	suite.Run("ReadVec4Accessor error on rotation returns error", func() {
		suite.mockParser.scalarResult = []float32{0.0}
		suite.mockParser.readVec4Err = errors.New("fail")
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Test",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathRotation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		_, err := suite.extractor.ExtractAnimation(0, map[int]int32{0: 0})
		suite.Error(err)
	})

	suite.Run("scale path sets ScaleKeys", func() {
		suite.mockParser.scalarResult = []float32{0.0, 0.25}
		suite.mockParser.vec3Result = [][3]float32{{1, 1, 1}, {2, 2, 2}}
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Scale",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathScale}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		clip, err := suite.extractor.ExtractAnimation(0, map[int]int32{0: 0})
		suite.NoError(err)
		suite.NotNil(clip)
		suite.Len(clip.Channels, 1)
		suite.Len(clip.Channels[0].ScaleKeys, 2)
		suite.Equal([3]float32{1, 1, 1}, clip.Channels[0].ScaleKeys[0].Value)
		suite.Equal([3]float32{2, 2, 2}, clip.Channels[0].ScaleKeys[1].Value)
	})

	suite.Run("ReadVec3Accessor error on scale returns error", func() {
		suite.mockParser.scalarResult = []float32{0.0}
		suite.mockParser.readVec3Err = errors.New("fail")
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Test",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathScale}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		_, err := suite.extractor.ExtractAnimation(0, map[int]int32{0: 0})
		suite.Error(err)
	})

	suite.Run("weights path is skipped", func() {
		suite.mockParser.scalarResult = []float32{0.0}
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Morph",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathWeights}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		clip, err := suite.extractor.ExtractAnimation(0, map[int]int32{0: 0})
		suite.NoError(err)
		suite.NotNil(clip)
		suite.Len(clip.Channels, 1)
		suite.Len(clip.Channels[0].PositionKeys, 0)
		suite.Len(clip.Channels[0].RotationKeys, 0)
		suite.Len(clip.Channels[0].ScaleKeys, 0)
	})

	suite.Run("empty animation name produces auto-generated name", func() {
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{Name: "", Channels: []gltfAnimChannel{}, Samplers: []gltfAnimSampler{}},
			},
		}
		clip, err := suite.extractor.ExtractAnimation(0, map[int]int32{})
		suite.NoError(err)
		suite.NotNil(clip)
		suite.Equal("animation_0", clip.Name)
	})

	suite.Run("named animation preserves name", func() {
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{Name: "Walk", Channels: []gltfAnimChannel{}, Samplers: []gltfAnimSampler{}},
			},
		}
		clip, err := suite.extractor.ExtractAnimation(0, map[int]int32{})
		suite.NoError(err)
		suite.NotNil(clip)
		suite.Equal("Walk", clip.Name)
	})

	suite.Run("duration equals max timestamp", func() {
		suite.mockParser.scalarResult = []float32{0.0, 1.0}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}, {1, 1, 1}}
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "DurTest",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathTranslation}},
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(1), Path: gltfAnimPathScale}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		clip, err := suite.extractor.ExtractAnimation(0, map[int]int32{0: 0, 1: 1})
		suite.NoError(err)
		suite.NotNil(clip)
		suite.InDelta(float32(1.0), clip.Duration, 1e-6)
	})

	suite.Run("TicksPerSecond is always 1.0", func() {
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{Name: "T", Channels: []gltfAnimChannel{}, Samplers: []gltfAnimSampler{}},
			},
		}
		clip, err := suite.extractor.ExtractAnimation(0, map[int]int32{})
		suite.NoError(err)
		suite.NotNil(clip)
		suite.InDelta(float32(1.0), clip.TicksPerSecond, 1e-6)
	})
}

func (suite *gltfAnimationExtractorImplTest) TestExtractAnimationsForSkeleton() {
	suite.Run("nil document returns error", func() {
		suite.mockParser.doc = nil
		_, err := suite.extractor.ExtractAnimationsForSkeleton(0, nil)
		suite.Error(err)
	})

	suite.Run("skinIndex out of range returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Skins: []gltfSkin{{Joints: []int{0}}},
		}
		_, err := suite.extractor.ExtractAnimationsForSkeleton(1, nil)
		suite.Error(err)
	})

	suite.Run("skinIndex negative returns error", func() {
		suite.mockParser.doc = &gltfDocument{
			Skins: []gltfSkin{{Joints: []int{0}}},
		}
		_, err := suite.extractor.ExtractAnimationsForSkeleton(-1, nil)
		suite.Error(err)
	})

	suite.Run("no animations return nil clips", func() {
		suite.mockParser.doc = &gltfDocument{
			Skins:      []gltfSkin{{Joints: []int{0, 1}}},
			Animations: nil,
		}
		clips, err := suite.extractor.ExtractAnimationsForSkeleton(0, map[int]int32{0: 0, 1: 1})
		suite.NoError(err)
		suite.Nil(clips)
	})

	suite.Run("animation not targeting skin joint is skipped", func() {
		suite.mockParser.doc = &gltfDocument{
			Skins: []gltfSkin{{Joints: []int{1, 2, 3}}},
			Animations: []gltfAnimation{
				{
					Name: "NotRelevant",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(5), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		clips, err := suite.extractor.ExtractAnimationsForSkeleton(0, map[int]int32{5: 0})
		suite.NoError(err)
		suite.Nil(clips)
	})

	suite.Run("animation targeting skin joint is included", func() {
		suite.mockParser.scalarResult = []float32{0.0, 1.0}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}, {1, 0, 0}}
		suite.mockParser.doc = &gltfDocument{
			Skins: []gltfSkin{{Joints: []int{1, 2}}},
			Animations: []gltfAnimation{
				{
					Name: "Walk",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(1), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		clips, err := suite.extractor.ExtractAnimationsForSkeleton(0, map[int]int32{1: 0, 2: 1})
		suite.NoError(err)
		suite.Len(clips, 1)
		suite.Equal("Walk", clips[0].Name)
	})

	suite.Run("error from ExtractAnimation propagates", func() {
		suite.mockParser.readScalarErr = errors.New("read fail")
		suite.mockParser.doc = &gltfDocument{
			Skins: []gltfSkin{{Joints: []int{0}}},
			Animations: []gltfAnimation{
				{
					Name: "Fail",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		_, err := suite.extractor.ExtractAnimationsForSkeleton(0, map[int]int32{0: 0})
		suite.Error(err)
	})
}

func (suite *gltfAnimationExtractorImplTest) TestExtractAllAnimations() {
	suite.Run("nil document returns error", func() {
		suite.mockParser.doc = nil
		_, err := suite.extractor.ExtractAllAnimations(nil)
		suite.Error(err)
	})

	suite.Run("empty animations list returns empty slice", func() {
		suite.mockParser.doc = &gltfDocument{
			Animations: nil,
		}
		clips, err := suite.extractor.ExtractAllAnimations(map[int]int32{})
		suite.NoError(err)
		suite.NotNil(clips)
		suite.Len(clips, 0)
	})

	suite.Run("all animations extracted successfully", func() {
		suite.mockParser.scalarResult = []float32{0.0, 1.0}
		suite.mockParser.vec3Result = [][3]float32{{0, 0, 0}, {1, 0, 0}}
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Anim1",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
				{
					Name: "Anim2",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(1), Path: gltfAnimPathScale}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		clips, err := suite.extractor.ExtractAllAnimations(map[int]int32{0: 0, 1: 1})
		suite.NoError(err)
		suite.Len(clips, 2)
		suite.Equal("Anim1", clips[0].Name)
		suite.Equal("Anim2", clips[1].Name)
	})

	suite.Run("error on one animation propagates", func() {
		suite.mockParser.readScalarErr = errors.New("fail on second")
		suite.mockParser.doc = &gltfDocument{
			Animations: []gltfAnimation{
				{
					Name: "Good",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(0), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
				{
					Name: "Bad",
					Channels: []gltfAnimChannel{
						{Sampler: 0, Target: gltfAnimTarget{Node: common.ToPtr(1), Path: gltfAnimPathTranslation}},
					},
					Samplers: []gltfAnimSampler{{Input: 0, Output: 1}},
				},
			},
		}
		_, err := suite.extractor.ExtractAllAnimations(map[int]int32{0: 0, 1: 1})
		suite.Error(err)
	})
}
