package animator_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	"github.com/stretchr/testify/suite"
)

func TestRunAnimatorTests(t *testing.T) {
	suite.Run(t, new(animatorTest))
}

type animatorTest struct {
	suite.Suite
}

// --- BackendType constants ---

func (suite *animatorTest) TestAnimatorBackendTypeConstants() {
	suite.Run("BackendTypeSimple should equal AnimatorBackendType(0)", func() {
		suite.Equal(animator.AnimatorBackendType(0), animator.BackendTypeSimple)
	})
	suite.Run("BackendTypeSkeletal should equal AnimatorBackendType(1)", func() {
		suite.Equal(animator.AnimatorBackendType(1), animator.BackendTypeSkeletal)
	})
}

// --- GPUInstanceData ---

func (suite *animatorTest) TestGPUInstanceData() {
	suite.Run("Size should return 64", func() {
		g := &animator.GPUInstanceData{}
		suite.Equal(64, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUInstanceData{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode Model[0] at offset 0", func() {
		g := &animator.GPUInstanceData{Model: [16]float32{1}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(1), binary.LittleEndian.Uint32(buf[0:4]))
	})
	suite.Run("Marshal should encode Model[15] at offset 60", func() {
		g := &animator.GPUInstanceData{}
		g.Model[15] = 7
		buf := g.Marshal()
		suite.Equal(math.Float32bits(7), binary.LittleEndian.Uint32(buf[60:64]))
	})
}

// --- GPUAnimationData ---

func (suite *animatorTest) TestGPUAnimationData() {
	suite.Run("Size should return 64", func() {
		g := &animator.GPUAnimationData{}
		suite.Equal(64, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUAnimationData{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode RotSpeed[0] at offset 0", func() {
		g := &animator.GPUAnimationData{RotSpeed: [3]float32{3.14, 0, 0}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(3.14), binary.LittleEndian.Uint32(buf[0:4]))
	})
	suite.Run("Marshal should encode Rot[0] at offset 16", func() {
		g := &animator.GPUAnimationData{Rot: [3]float32{1.5, 0, 0}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(1.5), binary.LittleEndian.Uint32(buf[16:20]))
	})
	suite.Run("Marshal should encode Pos[0] at offset 32", func() {
		g := &animator.GPUAnimationData{Pos: [3]float32{9.0, 0, 0}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(9.0), binary.LittleEndian.Uint32(buf[32:36]))
	})
	suite.Run("Marshal should encode Scale[0] at offset 48", func() {
		g := &animator.GPUAnimationData{Scale: [3]float32{2.0, 0, 0}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(2.0), binary.LittleEndian.Uint32(buf[48:52]))
	})
	suite.Run("Marshal padding bytes should be zero", func() {
		g := &animator.GPUAnimationData{}
		buf := g.Marshal()
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[12:16]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[28:32]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[44:48]))
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[60:64]))
	})
}

// --- GPUFrustumPlane ---

func (suite *animatorTest) TestGPUFrustumPlane() {
	suite.Run("Size should return 16", func() {
		g := &animator.GPUFrustumPlane{}
		suite.Equal(16, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUFrustumPlane{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode Normal[0] at offset 0", func() {
		g := &animator.GPUFrustumPlane{Normal: [3]float32{1, 2, 3}, Distance: 4}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(1), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(math.Float32bits(2), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(math.Float32bits(3), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(math.Float32bits(4), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

// --- GPUGlobalData ---

func (suite *animatorTest) TestGPUGlobalData() {
	suite.Run("Size should return 224", func() {
		g := &animator.GPUGlobalData{}
		suite.Equal(224, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUGlobalData{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode InstanceCount at offset 0", func() {
		g := &animator.GPUGlobalData{InstanceCount: 42}
		buf := g.Marshal()
		suite.Equal(uint32(42), binary.LittleEndian.Uint32(buf[0:4]))
	})
	suite.Run("Marshal should encode DeltaTime at offset 4", func() {
		g := &animator.GPUGlobalData{DeltaTime: 0.016}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(0.016), binary.LittleEndian.Uint32(buf[4:8]))
	})
	suite.Run("Marshal should encode BoundingRadius at offset 8", func() {
		g := &animator.GPUGlobalData{BoundingRadius: 5.5}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(5.5), binary.LittleEndian.Uint32(buf[8:12]))
	})
	suite.Run("Marshal should encode first plane Normal[0] at offset 32", func() {
		g := &animator.GPUGlobalData{}
		g.Planes[0] = animator.GPUFrustumPlane{Normal: [3]float32{1, 0, 0}, Distance: 10}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(1), binary.LittleEndian.Uint32(buf[32:36]))
		suite.Equal(math.Float32bits(10), binary.LittleEndian.Uint32(buf[44:48]))
	})
	suite.Run("Marshal should encode ScreenWidth at offset 12", func() {
		g := &animator.GPUGlobalData{ScreenWidth: 1920}
		buf := g.Marshal()
		suite.Equal(uint32(1920), binary.LittleEndian.Uint32(buf[12:16]))
	})
	suite.Run("Marshal should encode ScreenHeight at offset 16", func() {
		g := &animator.GPUGlobalData{ScreenHeight: 1080}
		buf := g.Marshal()
		suite.Equal(uint32(1080), binary.LittleEndian.Uint32(buf[16:20]))
	})
	suite.Run("Marshal should encode HiZMipCount at offset 20", func() {
		g := &animator.GPUGlobalData{HiZMipCount: 7}
		buf := g.Marshal()
		suite.Equal(uint32(7), binary.LittleEndian.Uint32(buf[20:24]))
	})
	suite.Run("Marshal should encode ProjX at offset 24", func() {
		g := &animator.GPUGlobalData{ProjX: 1.5}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(1.5), binary.LittleEndian.Uint32(buf[24:28]))
	})
	suite.Run("Marshal should encode ViewProj[0] at offset 128", func() {
		g := &animator.GPUGlobalData{}
		g.ViewProj[0] = 2.5
		buf := g.Marshal()
		suite.Equal(math.Float32bits(2.5), binary.LittleEndian.Uint32(buf[128:132]))
	})
	suite.Run("Marshal should encode BoundingMin at offset 192", func() {
		g := &animator.GPUGlobalData{BoundingMin: [3]float32{-1, -2, -3}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(-1), binary.LittleEndian.Uint32(buf[192:196]))
		suite.Equal(math.Float32bits(-2), binary.LittleEndian.Uint32(buf[196:200]))
		suite.Equal(math.Float32bits(-3), binary.LittleEndian.Uint32(buf[200:204]))
	})
	suite.Run("Marshal should encode BoundingMax at offset 208", func() {
		g := &animator.GPUGlobalData{BoundingMax: [3]float32{4, 5, 6}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(4), binary.LittleEndian.Uint32(buf[208:212]))
		suite.Equal(math.Float32bits(5), binary.LittleEndian.Uint32(buf[212:216]))
		suite.Equal(math.Float32bits(6), binary.LittleEndian.Uint32(buf[216:220]))
	})
}

// --- GPUAnimationGlobals ---

func (suite *animatorTest) TestGPUAnimationGlobals() {
	suite.Run("Size should return 240", func() {
		g := &animator.GPUAnimationGlobals{}
		suite.Equal(240, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUAnimationGlobals{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode InstanceCount at offset 0", func() {
		g := &animator.GPUAnimationGlobals{InstanceCount: 7}
		buf := g.Marshal()
		suite.Equal(uint32(7), binary.LittleEndian.Uint32(buf[0:4]))
	})
	suite.Run("Marshal should encode BoneCount at offset 4", func() {
		g := &animator.GPUAnimationGlobals{BoneCount: 32}
		buf := g.Marshal()
		suite.Equal(uint32(32), binary.LittleEndian.Uint32(buf[4:8]))
	})
	suite.Run("Marshal should encode ChannelDataOffset at offset 12", func() {
		g := &animator.GPUAnimationGlobals{ChannelDataOffset: 99}
		buf := g.Marshal()
		suite.Equal(uint32(99), binary.LittleEndian.Uint32(buf[12:16]))
	})
	suite.Run("Marshal should encode first plane Normal[0] at offset 32", func() {
		g := &animator.GPUAnimationGlobals{}
		g.Planes[0] = animator.GPUFrustumPlane{Normal: [3]float32{0, 1, 0}, Distance: 3}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(0), binary.LittleEndian.Uint32(buf[32:36]))
		suite.Equal(math.Float32bits(1), binary.LittleEndian.Uint32(buf[36:40]))
	})
	suite.Run("Marshal should encode KeyframeDataOffset at offset 16", func() {
		g := &animator.GPUAnimationGlobals{KeyframeDataOffset: 77}
		buf := g.Marshal()
		suite.Equal(uint32(77), binary.LittleEndian.Uint32(buf[16:20]))
	})
	suite.Run("Marshal should encode ScreenWidth at offset 20", func() {
		g := &animator.GPUAnimationGlobals{ScreenWidth: 2560}
		buf := g.Marshal()
		suite.Equal(uint32(2560), binary.LittleEndian.Uint32(buf[20:24]))
	})
	suite.Run("Marshal should encode ScreenHeight at offset 24", func() {
		g := &animator.GPUAnimationGlobals{ScreenHeight: 1440}
		buf := g.Marshal()
		suite.Equal(uint32(1440), binary.LittleEndian.Uint32(buf[24:28]))
	})
	suite.Run("Marshal should encode HiZMipCount at offset 28", func() {
		g := &animator.GPUAnimationGlobals{HiZMipCount: 8}
		buf := g.Marshal()
		suite.Equal(uint32(8), binary.LittleEndian.Uint32(buf[28:32]))
	})
	suite.Run("Marshal should encode ProjX at offset 128", func() {
		g := &animator.GPUAnimationGlobals{ProjX: 3.14}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(3.14), binary.LittleEndian.Uint32(buf[128:132]))
	})
	suite.Run("Marshal should encode ViewProj[0] at offset 144", func() {
		g := &animator.GPUAnimationGlobals{}
		g.ViewProj[0] = 7.0
		buf := g.Marshal()
		suite.Equal(math.Float32bits(7.0), binary.LittleEndian.Uint32(buf[144:148]))
	})
	suite.Run("Marshal should encode BoundingMin at offset 208", func() {
		g := &animator.GPUAnimationGlobals{BoundingMin: [3]float32{-5, -6, -7}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(-5), binary.LittleEndian.Uint32(buf[208:212]))
		suite.Equal(math.Float32bits(-6), binary.LittleEndian.Uint32(buf[212:216]))
		suite.Equal(math.Float32bits(-7), binary.LittleEndian.Uint32(buf[216:220]))
	})
	suite.Run("Marshal should encode BoundingMax at offset 224", func() {
		g := &animator.GPUAnimationGlobals{BoundingMax: [3]float32{8, 9, 10}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(8), binary.LittleEndian.Uint32(buf[224:228]))
		suite.Equal(math.Float32bits(9), binary.LittleEndian.Uint32(buf[228:232]))
		suite.Equal(math.Float32bits(10), binary.LittleEndian.Uint32(buf[232:236]))
	})
}

// --- GPUIndirectArgs ---

func (suite *animatorTest) TestGPUIndirectArgs() {
	suite.Run("Size should return 20", func() {
		g := &animator.GPUIndirectArgs{}
		suite.Equal(20, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUIndirectArgs{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode IndexCount at offset 0", func() {
		g := &animator.GPUIndirectArgs{IndexCount: 36}
		buf := g.Marshal()
		suite.Equal(uint32(36), binary.LittleEndian.Uint32(buf[0:4]))
	})
	suite.Run("Marshal should encode InstanceCount at offset 4", func() {
		g := &animator.GPUIndirectArgs{InstanceCount: 5}
		buf := g.Marshal()
		suite.Equal(uint32(5), binary.LittleEndian.Uint32(buf[4:8]))
	})
	suite.Run("Marshal should encode BaseVertex (signed) at offset 12", func() {
		g := &animator.GPUIndirectArgs{BaseVertex: -1}
		buf := g.Marshal()
		suite.Equal(uint32(0xFFFFFFFF), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

// --- GPUBoneInfo ---

func (suite *animatorTest) TestGPUBoneInfo() {
	suite.Run("Size should return 112", func() {
		g := &animator.GPUBoneInfo{}
		suite.Equal(112, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUBoneInfo{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode InverseBindMatrix[0] at offset 0", func() {
		g := &animator.GPUBoneInfo{}
		g.InverseBindMatrix[0] = 1.5
		buf := g.Marshal()
		suite.Equal(math.Float32bits(1.5), binary.LittleEndian.Uint32(buf[0:4]))
	})
	suite.Run("Marshal should encode LocalTranslation at offset 64", func() {
		g := &animator.GPUBoneInfo{LocalTranslation: [3]float32{2, 3, 4}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(2), binary.LittleEndian.Uint32(buf[64:68]))
		suite.Equal(math.Float32bits(3), binary.LittleEndian.Uint32(buf[68:72]))
		suite.Equal(math.Float32bits(4), binary.LittleEndian.Uint32(buf[72:76]))
	})
	suite.Run("Marshal should encode ParentIndex at offset 76", func() {
		g := &animator.GPUBoneInfo{ParentIndex: -1}
		buf := g.Marshal()
		suite.Equal(uint32(0xFFFFFFFF), binary.LittleEndian.Uint32(buf[76:80]))
	})
	suite.Run("Marshal should encode LocalRotation at offset 96", func() {
		g := &animator.GPUBoneInfo{LocalRotation: [4]float32{0, 0, 0, 1}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(1), binary.LittleEndian.Uint32(buf[108:112]))
	})
}

// --- GPUKeyFrame ---

func (suite *animatorTest) TestGPUKeyFrame() {
	suite.Run("Size should return 64", func() {
		g := &animator.GPUKeyFrame{}
		suite.Equal(64, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUKeyFrame{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode Time at offset 0", func() {
		g := &animator.GPUKeyFrame{Time: 1.25}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(1.25), binary.LittleEndian.Uint32(buf[0:4]))
	})
	suite.Run("Marshal should encode Translation at offset 16", func() {
		g := &animator.GPUKeyFrame{Translation: [3]float32{5, 6, 7}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(5), binary.LittleEndian.Uint32(buf[16:20]))
		suite.Equal(math.Float32bits(6), binary.LittleEndian.Uint32(buf[20:24]))
		suite.Equal(math.Float32bits(7), binary.LittleEndian.Uint32(buf[24:28]))
	})
	suite.Run("Marshal should encode Rotation at offset 32", func() {
		g := &animator.GPUKeyFrame{Rotation: [4]float32{0, 0, 0, 1}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(1), binary.LittleEndian.Uint32(buf[44:48]))
	})
	suite.Run("Marshal should encode Scale at offset 48", func() {
		g := &animator.GPUKeyFrame{Scale: [3]float32{2, 2, 2}}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(2), binary.LittleEndian.Uint32(buf[48:52]))
	})
}

// --- GPUChannelHeader ---

func (suite *animatorTest) TestGPUChannelHeader() {
	suite.Run("Size should return 32", func() {
		g := &animator.GPUChannelHeader{}
		suite.Equal(32, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUChannelHeader{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode BoneIndex at offset 0", func() {
		g := &animator.GPUChannelHeader{BoneIndex: 7}
		buf := g.Marshal()
		suite.Equal(uint32(7), binary.LittleEndian.Uint32(buf[0:4]))
	})
	suite.Run("Marshal should encode ScaleKeyCount at offset 24", func() {
		g := &animator.GPUChannelHeader{ScaleKeyCount: 3}
		buf := g.Marshal()
		suite.Equal(uint32(3), binary.LittleEndian.Uint32(buf[24:28]))
	})
	suite.Run("Marshal padding at offset 28 should be zero", func() {
		g := &animator.GPUChannelHeader{}
		buf := g.Marshal()
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[28:32]))
	})
}

// --- GPUClipHeader ---

func (suite *animatorTest) TestGPUClipHeader() {
	suite.Run("Size should return 16", func() {
		g := &animator.GPUClipHeader{}
		suite.Equal(16, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUClipHeader{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode Duration at offset 0", func() {
		g := &animator.GPUClipHeader{Duration: 2.5}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(2.5), binary.LittleEndian.Uint32(buf[0:4]))
	})
	suite.Run("Marshal should encode TicksPerSecond at offset 4", func() {
		g := &animator.GPUClipHeader{TicksPerSecond: 24}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(24), binary.LittleEndian.Uint32(buf[4:8]))
	})
	suite.Run("Marshal should encode ChannelOffset at offset 8", func() {
		g := &animator.GPUClipHeader{ChannelOffset: 10}
		buf := g.Marshal()
		suite.Equal(uint32(10), binary.LittleEndian.Uint32(buf[8:12]))
	})
	suite.Run("Marshal should encode ChannelCount at offset 12", func() {
		g := &animator.GPUClipHeader{ChannelCount: 3}
		buf := g.Marshal()
		suite.Equal(uint32(3), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

// --- GPUSkeletalAnimationData ---

func (suite *animatorTest) TestGPUSkeletalAnimationData() {
	suite.Run("Size should return 48", func() {
		g := &animator.GPUSkeletalAnimationData{}
		suite.Equal(48, g.Size())
	})
	suite.Run("Marshal length should equal Size", func() {
		g := &animator.GPUSkeletalAnimationData{}
		suite.Len(g.Marshal(), g.Size())
	})
	suite.Run("Marshal should encode AnimationIndex at offset 0", func() {
		g := &animator.GPUSkeletalAnimationData{AnimationIndex: 2}
		buf := g.Marshal()
		suite.Equal(uint32(2), binary.LittleEndian.Uint32(buf[0:4]))
	})
	suite.Run("Marshal should encode AnimationTime at offset 4", func() {
		g := &animator.GPUSkeletalAnimationData{AnimationTime: 1.5}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(1.5), binary.LittleEndian.Uint32(buf[4:8]))
	})
	suite.Run("Marshal should encode BlendWeight at offset 8", func() {
		g := &animator.GPUSkeletalAnimationData{BlendWeight: 0.75}
		buf := g.Marshal()
		suite.Equal(math.Float32bits(0.75), binary.LittleEndian.Uint32(buf[8:12]))
	})
	suite.Run("Marshal should encode SecondaryAnimIndex at offset 12", func() {
		g := &animator.GPUSkeletalAnimationData{SecondaryAnimIndex: 1}
		buf := g.Marshal()
		suite.Equal(uint32(1), binary.LittleEndian.Uint32(buf[12:16]))
	})
}
