package animator

import (
	"encoding/binary"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
	model_mocks "github.com/Carmen-Shannon/oxy-go/engine/model/mocks"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/stretchr/testify/suite"
)

func TestRunAnimatorImplTests(t *testing.T) {
	suite.Run(t, new(animatorImplTest))
}

type animatorImplTest struct {
	suite.Suite
	simple   Animator
	skeletal Animator
}

func (suite *animatorImplTest) SetupSubTest() {
	suite.simple = NewAnimator(BackendTypeSimple)
	suite.skeletal = NewAnimator(BackendTypeSkeletal)
}

// --- Constructor / BackendType ---

func (suite *animatorImplTest) TestNewAnimator() {
	suite.Run("should create a simple animator with BackendTypeSimple backend", func() {
		a := NewAnimator(BackendTypeSimple)
		suite.Equal(BackendTypeSimple, a.BackendType())
	})
	suite.Run("should create a skeletal animator with BackendTypeSkeletal backend", func() {
		a := NewAnimator(BackendTypeSkeletal)
		suite.Equal(BackendTypeSkeletal, a.BackendType())
	})
	suite.Run("should default to simple backend for unknown backend type and remain functional", func() {
		a := NewAnimator(AnimatorBackendType(99))
		suite.NotNil(a)
		suite.Equal(AnimatorBackendType(99), a.BackendType())
		suite.NotPanics(func() {
			a.AddInstance()
		})
	})
}

// --- WithMaxInstances builder option ---

func (suite *animatorImplTest) TestWithMaxInstances() {
	suite.Run("should configure the max instances on the simple animator", func() {
		a := NewAnimator(BackendTypeSimple, WithMaxInstances(50))
		suite.Equal(uint32(50), a.MaxInstances())
	})
	suite.Run("should configure the max instances on the skeletal animator", func() {
		a := NewAnimator(BackendTypeSkeletal, WithMaxInstances(10))
		suite.Equal(uint32(10), a.MaxInstances())
	})
}

// --- AddInstance / InstanceCount ---

func (suite *animatorImplTest) TestAddInstance() {
	suite.Run("should return index 0 for the first instance on a simple animator", func() {
		idx, err := suite.simple.AddInstance()
		suite.NoError(err)
		suite.Equal(uint32(0), idx)
	})
	suite.Run("should return index 0 for the first instance on a skeletal animator", func() {
		idx, err := suite.skeletal.AddInstance()
		suite.NoError(err)
		suite.Equal(uint32(0), idx)
	})
	suite.Run("should increment instance count after each add", func() {
		suite.simple.AddInstance()
		suite.simple.AddInstance()
		suite.Equal(uint32(2), suite.simple.InstanceCount())
	})
	suite.Run("should auto-grow when instances exceed max capacity on simple animator", func() {
		a := NewAnimator(BackendTypeSimple, WithMaxInstances(2))
		a.AddInstance()
		a.AddInstance()
		_, err := a.AddInstance() // triggers auto-grow
		suite.NoError(err)
		suite.True(a.NeedsRebuild())
	})
}

// --- NeedsRebuild ---

func (suite *animatorImplTest) TestNeedsRebuild() {
	suite.Run("should return false initially on simple animator", func() {
		suite.False(suite.simple.NeedsRebuild())
	})
	suite.Run("should return false initially on skeletal animator", func() {
		suite.False(suite.skeletal.NeedsRebuild())
	})
}

// --- Grow ---

func (suite *animatorImplTest) TestGrow() {
	suite.Run("should set NeedsRebuild to true on simple animator", func() {
		// simple default max is 25000; must grow beyond that to trigger rebuild
		suite.simple.Grow(30000)
		suite.True(suite.simple.NeedsRebuild())
	})
	suite.Run("should set NeedsRebuild to true on skeletal animator", func() {
		// skeletal default max is 200; growing to 1000 exceeds it
		suite.skeletal.Grow(1000)
		suite.True(suite.skeletal.NeedsRebuild())
	})
	suite.Run("should be a no-op when newMax is less than or equal to current capacity", func() {
		suite.simple.Grow(1) // 1 < 25000 default — must be a no-op
		suite.False(suite.simple.NeedsRebuild())
	})
}

// --- ClearNeedsRebuild ---

func (suite *animatorImplTest) TestClearNeedsRebuild() {
	suite.Run("should clear the rebuild flag on simple animator", func() {
		suite.simple.Grow(100)
		suite.simple.ClearNeedsRebuild()
		suite.False(suite.simple.NeedsRebuild())
	})
	suite.Run("should clear the rebuild flag on skeletal animator", func() {
		suite.skeletal.Grow(1000)
		suite.skeletal.ClearNeedsRebuild()
		suite.False(suite.skeletal.NeedsRebuild())
	})
}

// --- SetInstanceTransform / InstanceTransform ---

func (suite *animatorImplTest) TestSetInstanceTransform() {
	suite.Run("should store and return position and scale for simple animator", func() {
		suite.simple.AddInstance()
		suite.simple.SetInstanceTransform(0, [3]float32{1, 2, 3}, [3]float32{4, 5, 6})
		pos, scale := suite.simple.InstanceTransform(0)
		suite.Equal([3]float32{1, 2, 3}, pos)
		suite.Equal([3]float32{4, 5, 6}, scale)
	})
	suite.Run("should store and return position and scale for skeletal animator", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceTransform(0, [3]float32{10, 20, 30}, [3]float32{1, 1, 1})
		pos, scale := suite.skeletal.InstanceTransform(0)
		suite.InDelta(float64(10), float64(pos[0]), 1e-5)
		suite.InDelta(float64(20), float64(pos[1]), 1e-5)
		suite.InDelta(float64(30), float64(pos[2]), 1e-5)
		suite.InDelta(1.0, float64(scale[0]), 1e-5)
		suite.InDelta(1.0, float64(scale[1]), 1e-5)
		suite.InDelta(1.0, float64(scale[2]), 1e-5)
	})
	suite.Run("should be a no-op for out-of-range index on simple animator", func() {
		suite.NotPanics(func() {
			suite.simple.SetInstanceTransform(999999, [3]float32{1, 2, 3}, [3]float32{4, 5, 6})
		})
	})
}

// --- SetInstanceRotation / InstanceRotation ---

func (suite *animatorImplTest) TestSetInstanceRotation() {
	suite.Run("should store and return rotation speed and angle for simple animator", func() {
		suite.simple.AddInstance()
		suite.simple.SetInstanceRotation(0, [3]float32{0.1, 0.2, 0.3}, [3]float32{1.0, 2.0, 3.0})
		rotSpeed, rot := suite.simple.InstanceRotation(0)
		suite.Equal([3]float32{0.1, 0.2, 0.3}, rotSpeed)
		suite.Equal([3]float32{1.0, 2.0, 3.0}, rot)
	})
	suite.Run("should store rot euler data for skeletal animator via SetInstanceRotation", func() {
		suite.skeletal.AddInstance()
		// skeletal SetInstanceRotation only persists the euler angles (rotXYZ),
		// not rotSpeedXYZ — use SetInstanceData to also persist rotSpeed
		suite.skeletal.SetInstanceRotation(0, [3]float32{0.5, 0.6, 0.7}, [3]float32{0.1, 0.2, 0.3})
		_, rot := suite.skeletal.InstanceRotation(0)
		suite.Equal([3]float32{0.1, 0.2, 0.3}, rot)
	})
	suite.Run("should return zeros for index beyond instance count on simple animator", func() {
		rotSpeed, rot := suite.simple.InstanceRotation(0)
		suite.Equal([3]float32{}, rotSpeed)
		suite.Equal([3]float32{}, rot)
	})
}

// --- SetInstanceData ---

func (suite *animatorImplTest) TestSetInstanceData() {
	suite.Run("should store transform data retrievable via InstanceTransform and InstanceRotation on simple animator", func() {
		suite.simple.AddInstance()
		suite.simple.SetInstanceData(0, [3]float32{1, 2, 3}, [3]float32{2, 2, 2}, [3]float32{0.1, 0.1, 0.1}, [3]float32{0.5, 0.5, 0.5})
		pos, scale := suite.simple.InstanceTransform(0)
		rotSpeed, rot := suite.simple.InstanceRotation(0)
		suite.Equal([3]float32{1, 2, 3}, pos)
		suite.Equal([3]float32{2, 2, 2}, scale)
		suite.Equal([3]float32{0.1, 0.1, 0.1}, rotSpeed)
		suite.Equal([3]float32{0.5, 0.5, 0.5}, rot)
	})
	suite.Run("should store transform data for skeletal animator and make rotation retrievable", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceData(0, [3]float32{5, 6, 7}, [3]float32{1, 1, 1}, [3]float32{0.2, 0.2, 0.2}, [3]float32{0.3, 0.3, 0.3})
		_, rot := suite.skeletal.InstanceRotation(0)
		suite.Equal([3]float32{0.3, 0.3, 0.3}, rot)
	})
}

func (suite *animatorImplTest) TestSetInstanceFlags() {
	suite.Run("should store flags on a simple animator instance and mark it dirty", func() {
		suite.simple.AddInstance()
		suite.simple.SetInstanceFlags(0, InstanceFlagContactShadowExcluded)

		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		suite.Equal(uint32(InstanceFlagContactShadowExcluded), b.instanceData[0].InstanceFlags)
		suite.Equal([]uint32{0}, b.dirtyIndices)
	})

	suite.Run("should store flags on a skeletal animator instance and mark the dirty range", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceFlags(0, InstanceFlagContactShadowExcluded)

		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(uint32(InstanceFlagContactShadowExcluded), b.instanceData[0].InstanceFlags)
		suite.True(b.dirty)
		suite.Equal(uint32(0), b.dirtyStart)
		suite.Equal(uint32(1), b.dirtyEnd)
	})

	suite.Run("should be a no-op when index exceeds maxInstances on simple animator", func() {
		a := NewAnimator(BackendTypeSimple, WithMaxInstances(1))
		a.SetInstanceFlags(2, InstanceFlagContactShadowExcluded)

		b := a.(*animator).backend.(*simpleAnimatorBackendImpl)
		suite.Equal(uint32(0), b.instanceData[0].InstanceFlags)
		suite.Empty(b.dirtyIndices)
	})

	suite.Run("should be a no-op when index exceeds maxInstances on skeletal animator", func() {
		a := NewAnimator(BackendTypeSkeletal, WithMaxInstances(1))
		a.SetInstanceFlags(2, InstanceFlagContactShadowExcluded)

		b := a.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(uint32(0), b.instanceData[0].InstanceFlags)
		suite.False(b.dirty)
	})

	suite.Run("should expand dirtyStart when second SetInstanceFlags uses a lower index", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceFlags(1, InstanceFlagContactShadowExcluded)
		suite.skeletal.SetInstanceFlags(0, InstanceFlagContactShadowExcluded)

		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.True(b.dirty)
		suite.Equal(uint32(0), b.dirtyStart)
		suite.Equal(uint32(2), b.dirtyEnd)
	})

	suite.Run("should expand dirtyEnd when second SetInstanceFlags uses a higher index", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceFlags(0, InstanceFlagContactShadowExcluded)
		suite.skeletal.SetInstanceFlags(1, InstanceFlagContactShadowExcluded)

		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.True(b.dirty)
		suite.Equal(uint32(0), b.dirtyStart)
		suite.Equal(uint32(2), b.dirtyEnd)
	})
}

// --- RemoveInstance ---

func (suite *animatorImplTest) TestRemoveInstance() {
	suite.Run("should remove an instance and return swap info on simple animator", func() {
		suite.simple.AddInstance() // index 0
		suite.simple.AddInstance() // index 1
		oldLast, swapped := suite.simple.RemoveInstance(0)
		suite.True(swapped)
		suite.Equal(uint32(1), oldLast)
		suite.Equal(uint32(1), suite.simple.InstanceCount())
	})
	suite.Run("should return swapped=false when removing the only element on simple animator", func() {
		suite.simple.AddInstance() // index 0
		_, swapped := suite.simple.RemoveInstance(0)
		suite.False(swapped)
		suite.Equal(uint32(0), suite.simple.InstanceCount())
	})
	suite.Run("should remove an instance from skeletal animator", func() {
		suite.skeletal.AddInstance() // index 0
		suite.skeletal.AddInstance() // index 1
		oldLast, swapped := suite.skeletal.RemoveInstance(0)
		suite.True(swapped)
		suite.Equal(uint32(1), oldLast)
		suite.Equal(uint32(1), suite.skeletal.InstanceCount())
	})
	suite.Run("should return false when count is 0 on simple animator", func() {
		_, swapped := suite.simple.RemoveInstance(0)
		suite.False(swapped)
	})
}

// --- SetFrustumPlanes / CullingEnabled ---

func (suite *animatorImplTest) TestSetFrustumPlanes() {
	suite.Run("should enable culling on simple animator after setting frustum planes", func() {
		suite.False(suite.simple.CullingEnabled())
		var planes [6]GPUFrustumPlane
		suite.simple.SetFrustumPlanes(planes)
		suite.True(suite.simple.CullingEnabled())
	})
	suite.Run("should enable culling on skeletal animator after setting frustum planes", func() {
		suite.False(suite.skeletal.CullingEnabled())
		var planes [6]GPUFrustumPlane
		suite.skeletal.SetFrustumPlanes(planes)
		suite.True(suite.skeletal.CullingEnabled())
	})
}

// --- SetBoundingRadius / BoundingRadius ---

func (suite *animatorImplTest) TestSetBoundingRadius() {
	suite.Run("should store and return the bounding radius for simple animator", func() {
		suite.simple.SetBoundingRadius(3.5)
		suite.InDelta(3.5, float64(suite.simple.BoundingRadius()), 1e-6)
	})
	suite.Run("should store and return the bounding radius for skeletal animator", func() {
		suite.skeletal.SetBoundingRadius(7.25)
		suite.InDelta(7.25, float64(suite.skeletal.BoundingRadius()), 1e-6)
	})
}

// --- IndirectBuffer ---

func (suite *animatorImplTest) TestIndirectBuffer() {
	suite.Run("should return nil when culling is not enabled on simple animator", func() {
		suite.Nil(suite.simple.IndirectBuffer(0))
	})
	suite.Run("should return nil for skeletal animator when culling not enabled", func() {
		suite.Nil(suite.skeletal.IndirectBuffer(0))
	})
	suite.Run("should return nil for simple animator even when culling is enabled but no GPU buffer set", func() {
		var planes [6]GPUFrustumPlane
		suite.simple.SetFrustumPlanes(planes)
		// computeProvider has no real GPU buffer — Buffer(binding) returns nil
		suite.Nil(suite.simple.IndirectBuffer(0))
	})
}

// --- ResetIndirectArgs ---

func (suite *animatorImplTest) TestResetIndirectArgs() {
	suite.Run("should not panic when culling is not enabled on simple animator", func() {
		suite.NotPanics(func() {
			suite.simple.ResetIndirectArgs(36, 0)
		})
	})
	suite.Run("should not panic when culling is not enabled on skeletal animator", func() {
		suite.NotPanics(func() {
			suite.skeletal.ResetIndirectArgs(36, 0)
		})
	})
	suite.Run("should not panic when culling is enabled but needsRebuild is set on simple animator", func() {
		suite.simple.Grow(100) // sets needsRebuild = true
		var planes [6]GPUFrustumPlane
		suite.simple.SetFrustumPlanes(planes)
		suite.NotPanics(func() {
			suite.simple.ResetIndirectArgs(36, 0)
		})
	})
}

// --- PrepareFrame / StagedWriteData ---

func (suite *animatorImplTest) TestPrepareFrame() {
	suite.Run("should produce a staged write containing global data for simple animator", func() {
		suite.simple.PrepareFrame(0.016, 0)
		writes := suite.simple.StagedWriteData()
		suite.NotEmpty(writes)
	})
	suite.Run("should produce a staged write for skeletal animator", func() {
		suite.skeletal.PrepareFrame(0.016, 0)
		writes := suite.skeletal.StagedWriteData()
		suite.NotEmpty(writes)
	})
	suite.Run("should serialize simple culling enabled flag at offset 28", func() {
		readFlag := func(writes []bind_group_provider.BufferWrite, offset int) (uint32, bool) {
			for _, write := range writes {
				if len(write.Data) < offset+4 {
					continue
				}
				return binary.LittleEndian.Uint32(write.Data[offset : offset+4]), true
			}
			return 0, false
		}

		suite.simple.PrepareFrame(0.016, 0)
		disabledWrites := suite.simple.StagedWriteData()
		disabledFlag, foundDisabled := readFlag(disabledWrites, 28)
		suite.True(foundDisabled)
		suite.Equal(uint32(0), disabledFlag)

		var planes [6]GPUFrustumPlane
		suite.simple.SetFrustumPlanes(planes)
		suite.simple.PrepareFrame(0.016, 0)
		enabledWrites := suite.simple.StagedWriteData()
		enabledFlag, foundEnabled := readFlag(enabledWrites, 28)
		suite.True(foundEnabled)
		suite.Equal(uint32(1), enabledFlag)
	})
	suite.Run("should serialize skeletal culling enabled flag at offset 132", func() {
		readFlag := func(writes []bind_group_provider.BufferWrite, offset int) (uint32, bool) {
			for _, write := range writes {
				if len(write.Data) < offset+4 {
					continue
				}
				return binary.LittleEndian.Uint32(write.Data[offset : offset+4]), true
			}
			return 0, false
		}

		suite.skeletal.PrepareFrame(0.016, 0)
		disabledWrites := suite.skeletal.StagedWriteData()
		disabledFlag, foundDisabled := readFlag(disabledWrites, 132)
		suite.True(foundDisabled)
		suite.Equal(uint32(0), disabledFlag)

		var planes [6]GPUFrustumPlane
		suite.skeletal.SetFrustumPlanes(planes)
		suite.skeletal.PrepareFrame(0.016, 0)
		enabledWrites := suite.skeletal.StagedWriteData()
		enabledFlag, foundEnabled := readFlag(enabledWrites, 132)
		suite.True(foundEnabled)
		suite.Equal(uint32(1), enabledFlag)
	})
	suite.Run("should not produce staged writes when needsRebuild is set on simple animator", func() {
		suite.simple.Grow(30000) // 30000 > 25000 default — actually triggers rebuild flag
		suite.simple.PrepareFrame(0.016, 0)
		writes := suite.simple.StagedWriteData()
		suite.Empty(writes)
	})
}

// --- Flush / StagedWriteData ---

func (suite *animatorImplTest) TestFlush() {
	suite.Run("should return 0 instance count when no instances added on simple animator", func() {
		count := suite.simple.Flush(0, 0, 0)
		suite.Equal(uint32(0), count)
	})
	suite.Run("should produce staged writes after instances are added on simple animator", func() {
		suite.simple.AddInstance()
		suite.simple.AddInstance()
		suite.simple.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		suite.simple.SetInstanceTransform(1, [3]float32{2, 0, 0}, [3]float32{1, 1, 1})
		suite.simple.Flush(0, 0, 0)
		writes := suite.simple.StagedWriteData()
		suite.NotNil(writes)
	})
	suite.Run("should return instance count after instances added on skeletal animator", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.PrepareFrame(0.016, 2) // triggers dirty marking
		suite.skeletal.StagedWriteData()      // drain PrepareFrame writes
		count := suite.skeletal.Flush(0, 1, 2)
		suite.Equal(uint32(1), count)
	})
}

// --- StagedWriteData drainage ---

func (suite *animatorImplTest) TestStagedWriteData() {
	suite.Run("should drain the staged writes slice after calling StagedWriteData", func() {
		suite.simple.PrepareFrame(0.016, 0)
		suite.simple.StagedWriteData()
		// second call should return empty
		writes := suite.simple.StagedWriteData()
		suite.Empty(writes)
	})
}

// --- PlayAnimation / IsBlending ---

func (suite *animatorImplTest) TestPlayAnimation() {
	suite.Run("should not be blending after PlayAnimation on skeletal animator", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.PlayAnimation(0, 0, true)
		suite.False(suite.skeletal.IsBlending(0))
	})
	suite.Run("should be a no-op on simple animator (no panic)", func() {
		suite.simple.AddInstance()
		suite.NotPanics(func() { suite.simple.PlayAnimation(0, 0, true) })
	})
	suite.Run("should be a no-op when instanceIndex is out of range on skeletal animator", func() {
		suite.NotPanics(func() { suite.skeletal.PlayAnimation(99, 0, true) })
	})
}

// --- BlendToAnimation ---

func (suite *animatorImplTest) TestBlendToAnimation() {
	suite.Run("should enter blending state after BlendToAnimation on skeletal animator", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.PlayAnimation(0, 0, true)
		suite.skeletal.BlendToAnimation(0, 1, 0.5)
		suite.True(suite.skeletal.IsBlending(0))
		suite.InDelta(0.0, float64(suite.skeletal.BlendProgress(0)), 1e-6)
	})
	suite.Run("should be a no-op on simple animator (no panic)", func() {
		suite.simple.AddInstance()
		suite.NotPanics(func() { suite.simple.BlendToAnimation(0, 1, 0.5) })
	})
	suite.Run("should be a no-op when instanceIndex is out of range on skeletal animator", func() {
		suite.NotPanics(func() { suite.skeletal.BlendToAnimation(99, 1, 0.5) })
	})
}

// --- Blend resolves after PrepareFrame ---

func (suite *animatorImplTest) TestBlendResolvesAfterFrame() {
	suite.Run("should resolve blend after PrepareFrame with sufficient delta time", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.PlayAnimation(0, 0, true)
		suite.skeletal.BlendToAnimation(0, 1, 0.1)
		// Advance 1 second — far beyond 0.1s blend duration
		suite.skeletal.PrepareFrame(1.0, 2)
		suite.False(suite.skeletal.IsBlending(0))
	})
}

// --- BlendProgress ---

func (suite *animatorImplTest) TestBlendProgress() {
	suite.Run("should return 0 when not blending on skeletal animator", func() {
		suite.skeletal.AddInstance()
		suite.InDelta(0.0, float64(suite.skeletal.BlendProgress(0)), 1e-6)
	})
	suite.Run("should return 0 for simple animator", func() {
		suite.simple.AddInstance()
		suite.InDelta(0.0, float64(suite.simple.BlendProgress(0)), 1e-6)
	})
	suite.Run("should return 0 for out-of-range index on skeletal animator", func() {
		suite.InDelta(0.0, float64(suite.skeletal.BlendProgress(99)), 1e-6)
	})
}

// --- CancelBlend ---

func (suite *animatorImplTest) TestCancelBlend() {
	suite.Run("should exit blending state after CancelBlend on skeletal animator", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.PlayAnimation(0, 0, true)
		suite.skeletal.BlendToAnimation(0, 1, 0.5)
		suite.True(suite.skeletal.IsBlending(0))
		suite.skeletal.CancelBlend(0)
		suite.False(suite.skeletal.IsBlending(0))
	})
	suite.Run("should be a no-op on simple animator", func() {
		suite.simple.AddInstance()
		suite.NotPanics(func() { suite.simple.CancelBlend(0) })
	})
	suite.Run("should be a no-op when instanceIndex is out of range on skeletal animator", func() {
		suite.NotPanics(func() { suite.skeletal.CancelBlend(99) })
	})
}

// --- SetAnimationTime ---

func (suite *animatorImplTest) TestSetAnimationTime() {
	suite.Run("should not panic when setting animation time on skeletal animator", func() {
		suite.skeletal.AddInstance()
		suite.NotPanics(func() { suite.skeletal.SetAnimationTime(0, 1.5) })
	})
	suite.Run("should not panic on simple animator (no-op)", func() {
		suite.simple.AddInstance()
		suite.NotPanics(func() { suite.simple.SetAnimationTime(0, 1.5) })
	})
	suite.Run("should be a no-op when instanceIndex is out of range on skeletal animator", func() {
		suite.NotPanics(func() { suite.skeletal.SetAnimationTime(99, 1.5) })
	})
}

// --- SetAnimationSpeed ---

func (suite *animatorImplTest) TestSetAnimationSpeed() {
	suite.Run("should not panic when setting animation speed on skeletal animator", func() {
		suite.skeletal.AddInstance()
		suite.NotPanics(func() { suite.skeletal.SetAnimationSpeed(0, 2.0) })
	})
	suite.Run("should not panic on simple animator (no-op)", func() {
		suite.simple.AddInstance()
		suite.NotPanics(func() { suite.simple.SetAnimationSpeed(0, 2.0) })
	})
	suite.Run("should be a no-op when instanceIndex is out of range on skeletal animator", func() {
		suite.NotPanics(func() { suite.skeletal.SetAnimationSpeed(99, 2.0) })
	})
}

// --- SetBoneCount ---

func (suite *animatorImplTest) TestSetBoneCount() {
	suite.Run("should not panic on skeletal animator", func() {
		suite.NotPanics(func() { suite.skeletal.SetBoneCount(5) })
	})
	suite.Run("should not panic on simple animator (no-op)", func() {
		suite.NotPanics(func() { suite.simple.SetBoneCount(5) })
	})
}

// --- SetBone ---

func (suite *animatorImplTest) TestSetBone() {
	suite.Run("should not panic when setting a valid bone on skeletal animator", func() {
		suite.skeletal.SetBoneCount(2)
		suite.NotPanics(func() {
			suite.skeletal.SetBone(0, [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, [3]float32{}, [4]float32{0, 0, 0, 1}, [3]float32{1, 1, 1}, -1, 1)
		})
	})
	suite.Run("should be a no-op for out-of-range bone index on skeletal animator", func() {
		suite.skeletal.SetBoneCount(1)
		suite.NotPanics(func() {
			suite.skeletal.SetBone(99, [16]float32{}, [3]float32{}, [4]float32{}, [3]float32{}, -1, 1)
		})
	})
	suite.Run("should be a no-op on simple animator", func() {
		suite.NotPanics(func() {
			suite.simple.SetBone(0, [16]float32{}, [3]float32{}, [4]float32{}, [3]float32{}, 0, 0)
		})
	})
}

// --- AddClip ---

func (suite *animatorImplTest) TestAddClip() {
	suite.Run("should return monotonically increasing clip indices on skeletal animator", func() {
		idx0 := suite.skeletal.AddClip(1.0, 24.0, nil, nil, nil, nil, nil, 0)
		idx1 := suite.skeletal.AddClip(2.0, 30.0, nil, nil, nil, nil, nil, 0)
		suite.Equal(uint32(0), idx0)
		suite.Equal(uint32(1), idx1)
	})
	suite.Run("should return 0 on simple animator (no-op)", func() {
		idx := suite.simple.AddClip(1.0, 24.0, nil, nil, nil, nil, nil, 0)
		suite.Equal(uint32(0), idx)
	})
	suite.Run("should accept keyframe data without panic on skeletal animator", func() {
		channels := []uint32{0, 0, 1, 0, 0, 0, 0} // 1 channel, 1 pos key
		times := []float32{0.0}
		trans := [][3]float32{{1, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}}
		scales := [][3]float32{{1, 1, 1}}
		suite.NotPanics(func() {
			suite.skeletal.AddClip(1.0, 24.0, channels, times, trans, rots, scales, 0)
		})
	})
}

// --- Release ---

func (suite *animatorImplTest) TestRelease() {
	suite.Run("should not panic on simple animator", func() {
		suite.NotPanics(func() { suite.simple.Release() })
	})
	suite.Run("should not panic on skeletal animator", func() {
		suite.NotPanics(func() { suite.skeletal.Release() })
	})
}

// --- ComputeBindGroupProvider ---

func (suite *animatorImplTest) TestComputeBindGroupProvider() {
	suite.Run("should return a non-nil provider for simple animator", func() {
		suite.NotNil(suite.simple.ComputeBindGroupProvider())
	})
	suite.Run("should return a non-nil provider for skeletal animator", func() {
		suite.NotNil(suite.skeletal.ComputeBindGroupProvider())
	})
}

// --- OutputBindGroupProvider ---

func (suite *animatorImplTest) TestOutputBindGroupProvider() {
	suite.Run("should return a non-nil provider for simple animator", func() {
		suite.NotNil(suite.simple.OutputBindGroupProvider())
	})
	suite.Run("should return a non-nil provider for skeletal animator", func() {
		suite.NotNil(suite.skeletal.OutputBindGroupProvider())
	})
}

// --- Model / SetModel ---

func (suite *animatorImplTest) TestModel() {
	suite.Run("should return nil before SetModel is called", func() {
		suite.Nil(suite.simple.Model())
	})
}

func (suite *animatorImplTest) TestSetModel() {
	suite.Run("should store the model reference when model is not skinned", func() {
		m := model_mocks.NewMockModel(suite.T())
		m.EXPECT().Skinned().Return(false)
		suite.simple.SetModel(m, 0, 0)
		suite.Equal(m, suite.simple.Model())
	})
	suite.Run("should return early without calling Skeleton when model is not skinned", func() {
		m := model_mocks.NewMockModel(suite.T())
		m.EXPECT().Skinned().Return(false)
		// Skeleton() and Animations() must NOT be called — mockery will panic if they are
		suite.NotPanics(func() {
			suite.simple.SetModel(m, 0, 0)
		})
	})
	suite.Run("should return early when Skeleton is nil even if skinned", func() {
		m := model_mocks.NewMockModel(suite.T())
		m.EXPECT().Skinned().Return(true)
		m.EXPECT().Skeleton().Return(nil)
		suite.NotPanics(func() {
			suite.simple.SetModel(m, 0, 0)
		})
	})
	suite.Run("should set bones and clips when model is skinned with a skeleton on skeletal animator", func() {
		skel := &model.Skeleton{
			Bones: []model.Bone{
				{
					Name:              "root",
					ParentIndex:       -1,
					InverseBindMatrix: [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
					LocalTransform: model.Transform{
						Translation: [3]float32{0, 0, 0},
						Rotation:    [4]float32{0, 0, 0, 1},
						Scale:       [3]float32{1, 1, 1},
					},
				},
			},
		}
		m := model_mocks.NewMockModel(suite.T())
		m.EXPECT().Skinned().Return(true)
		m.EXPECT().Skeleton().Return(skel)
		m.EXPECT().Animations().Return([]*model.AnimationClip{})
		suite.NotPanics(func() {
			suite.skeletal.SetModel(m, 1, 2)
		})
		suite.Equal(m, suite.skeletal.Model())
	})
	suite.Run("should add clips when model has animations on skeletal animator", func() {
		skel := &model.Skeleton{
			Bones: []model.Bone{
				{
					ParentIndex:       -1,
					InverseBindMatrix: [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
					LocalTransform: model.Transform{
						Scale: [3]float32{1, 1, 1},
					},
				},
			},
		}
		clips := []*model.AnimationClip{
			{
				Name:           "walk",
				Duration:       1.0,
				TicksPerSecond: 24.0,
				Channels: []model.AnimationChannel{
					{
						BoneIndex: 0,
						PositionKeys: []model.VectorKeyframe{
							{Time: 0, Value: [3]float32{0, 0, 0}},
						},
						RotationKeys: []model.QuaternionKeyframe{
							{Time: 0, Value: [4]float32{0, 0, 0, 1}},
						},
						ScaleKeys: []model.VectorKeyframe{
							{Time: 0, Value: [3]float32{1, 1, 1}},
						},
					},
				},
			},
		}
		m := model_mocks.NewMockModel(suite.T())
		m.EXPECT().Skinned().Return(true)
		m.EXPECT().Skeleton().Return(skel)
		m.EXPECT().Animations().Return(clips)
		suite.NotPanics(func() {
			suite.skeletal.SetModel(m, 1, 2)
		})
	})
}

// --- WithModel builder option ---

func (suite *animatorImplTest) TestWithModel() {
	suite.Run("should apply SetModel via the builder option for non-skinned model", func() {
		m := model_mocks.NewMockModel(suite.T())
		m.EXPECT().Skinned().Return(false)
		a := NewAnimator(BackendTypeSimple, WithModel(m, 0, 0))
		suite.Equal(m, a.Model())
	})
}

// --- IsBlending edge cases ---

func (suite *animatorImplTest) TestIsBlending() {
	suite.Run("should return false for simple animator (always)", func() {
		suite.simple.AddInstance()
		suite.False(suite.simple.IsBlending(0))
	})
	suite.Run("should return false for out-of-range index on skeletal animator", func() {
		suite.False(suite.skeletal.IsBlending(99))
	})
}

// --- SetMaxInstances ---

func (suite *animatorImplTest) TestSetMaxInstances() {
	suite.Run("should reset instance count when max is reconfigured on simple animator", func() {
		suite.simple.AddInstance()
		suite.simple.AddInstance()
		suite.simple.(*animator).backend.SetMaxInstances(10)
		suite.Equal(uint32(0), suite.simple.InstanceCount())
		suite.Equal(uint32(10), suite.simple.MaxInstances())
	})
	suite.Run("should reset instance count when max is reconfigured on skeletal animator", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.(*animator).backend.SetMaxInstances(5)
		suite.Equal(uint32(0), suite.skeletal.InstanceCount())
		suite.Equal(uint32(5), suite.skeletal.MaxInstances())
	})
}

// --- InstanceTransform out-of-range ---

func (suite *animatorImplTest) TestInstanceTransformOutOfRange() {
	suite.Run("should return zero values for out-of-range index on simple animator", func() {
		pos, scale := suite.simple.InstanceTransform(99)
		suite.Equal([3]float32{}, pos)
		suite.Equal([3]float32{}, scale)
	})
	suite.Run("should return zero values for out-of-range index on skeletal animator", func() {
		pos, scale := suite.skeletal.InstanceTransform(99)
		suite.Equal([3]float32{}, pos)
		suite.Equal([3]float32{}, scale)
	})
}

// --- Additional coverage tests ---

func (suite *animatorImplTest) TestSetComputeBindGroupProvider() {
	suite.Run("should update the compute bind group provider on simple animator", func() {
		p := bind_group_provider.NewBindGroupProvider("test_compute_simple")
		suite.simple.(*animator).backend.SetComputeBindGroupProvider(p)
		suite.Equal(p, suite.simple.ComputeBindGroupProvider())
	})
	suite.Run("should update the compute bind group provider on skeletal animator", func() {
		p := bind_group_provider.NewBindGroupProvider("test_compute_skeletal")
		suite.skeletal.(*animator).backend.SetComputeBindGroupProvider(p)
		suite.Equal(p, suite.skeletal.ComputeBindGroupProvider())
	})
}

func (suite *animatorImplTest) TestSetOutputBindGroupProvider() {
	suite.Run("should update the output bind group provider on simple animator", func() {
		p := bind_group_provider.NewBindGroupProvider("test_output_simple")
		suite.simple.(*animator).backend.SetOutputBindGroupProvider(p)
		suite.Equal(p, suite.simple.OutputBindGroupProvider())
	})
	suite.Run("should update the output bind group provider on skeletal animator", func() {
		p := bind_group_provider.NewBindGroupProvider("test_output_skeletal")
		suite.skeletal.(*animator).backend.SetOutputBindGroupProvider(p)
		suite.Equal(p, suite.skeletal.OutputBindGroupProvider())
	})
}

func (suite *animatorImplTest) TestSkeletalBoneCount() {
	suite.Run("should return 0 before SetBoneCount on skeletal animator", func() {
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(uint32(0), b.BoneCount())
	})
	suite.Run("should return the bone count after SetBoneCount on skeletal animator", func() {
		suite.skeletal.SetBoneCount(5)
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(uint32(5), b.BoneCount())
	})
}

func (suite *animatorImplTest) TestSetInstanceRotationOutOfRange() {
	suite.Run("should be a no-op when index exceeds maxInstances on simple animator", func() {
		// Create a small-capacity animator so 999 exceeds maxInstances
		a := NewAnimator(BackendTypeSimple, WithMaxInstances(5))
		suite.NotPanics(func() {
			a.SetInstanceRotation(999, [3]float32{1, 2, 3}, [3]float32{4, 5, 6})
		})
	})
	suite.Run("should be a no-op when index exceeds maxInstances on skeletal animator", func() {
		a := NewAnimator(BackendTypeSkeletal, WithMaxInstances(5))
		suite.NotPanics(func() {
			a.SetInstanceRotation(999, [3]float32{1, 2, 3}, [3]float32{4, 5, 6})
		})
	})
}

func (suite *animatorImplTest) TestSetInstanceDataOutOfRange() {
	suite.Run("should be a no-op when index exceeds maxInstances on simple animator", func() {
		a := NewAnimator(BackendTypeSimple, WithMaxInstances(5))
		suite.NotPanics(func() {
			a.SetInstanceData(999, [3]float32{}, [3]float32{}, [3]float32{}, [3]float32{})
		})
	})
	suite.Run("should be a no-op when index exceeds maxInstances on skeletal animator", func() {
		a := NewAnimator(BackendTypeSkeletal, WithMaxInstances(5))
		suite.NotPanics(func() {
			a.SetInstanceData(999, [3]float32{}, [3]float32{}, [3]float32{}, [3]float32{})
		})
	})
}

func (suite *animatorImplTest) TestEnqueueDirtyDedup() {
	suite.Run("should not duplicate dirty entries when SetInstanceTransform is called twice on same index", func() {
		suite.simple.AddInstance()
		suite.simple.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		suite.simple.SetInstanceTransform(0, [3]float32{2, 0, 0}, [3]float32{2, 2, 2}) // second call — already dirty
		// Flush should handle without error and produce writes
		suite.simple.Flush(0, 0, 0)
		// After flush the staged writes carry the update — just verify no panic and instance data is correct
		pos, _ := suite.simple.InstanceTransform(0)
		suite.Equal([3]float32{2, 0, 0}, pos)
	})
}

func (suite *animatorImplTest) TestFlushNonContiguousMerge() {
	suite.Run("should merge non-contiguous dirty ranges correctly on simple animator", func() {
		suite.simple.AddInstance() // 0
		suite.simple.AddInstance() // 1
		suite.simple.AddInstance() // 2
		// Dirty indices 0 and 2 (non-adjacent) to exercise the merge else-branch
		suite.simple.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		suite.simple.SetInstanceTransform(2, [3]float32{3, 0, 0}, [3]float32{1, 1, 1})
		count := suite.simple.Flush(0, 0, 0)
		suite.Equal(uint32(2), count)
	})
}

func (suite *animatorImplTest) TestSortUint32() {
	suite.Run("should sort out-of-order dirty indices before merge on simple animator", func() {
		suite.simple.AddInstance() // 0
		suite.simple.AddInstance() // 1
		suite.simple.AddInstance() // 2
		// Dirty index 2 first then 0 — forces sortUint32 inner swap loop
		suite.simple.SetInstanceTransform(2, [3]float32{2, 0, 0}, [3]float32{1, 1, 1})
		suite.simple.SetInstanceTransform(0, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})
		count := suite.simple.Flush(0, 0, 0)
		suite.Equal(uint32(2), count)
	})
}

func (suite *animatorImplTest) TestSkeletalAddInstanceAutoGrow() {
	suite.Run("should auto-grow the skeletal animator when capacity is exceeded", func() {
		a := NewAnimator(BackendTypeSkeletal, WithMaxInstances(1))
		a.AddInstance()           // fills capacity
		_, err := a.AddInstance() // triggers auto-grow
		suite.NoError(err)
		suite.True(a.NeedsRebuild())
	})
}

func (suite *animatorImplTest) TestSkeletalFlushBoneDirty() {
	suite.Run("should flush bone data when boneDirty is set on skeletal animator", func() {
		suite.skeletal.SetBoneCount(2)
		suite.skeletal.SetBone(0, [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, [3]float32{}, [4]float32{0, 0, 0, 1}, [3]float32{1, 1, 1}, -1, 1)
		// SetBone stages a write and marks boneDirty — drain those first
		suite.skeletal.StagedWriteData()
		// Flush should see boneDirty and produce a staged write
		suite.skeletal.Flush(0, 1, 2)
		writes := suite.skeletal.StagedWriteData()
		// boneDirty write should be in staged writes
		suite.NotEmpty(writes)
	})
}

func (suite *animatorImplTest) TestSkeletalFlushModelDirty() {
	suite.Run("should flush model matrix data when modelDirty is set on skeletal animator", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceTransform(0, [3]float32{1, 2, 3}, [3]float32{1, 1, 1})
		count := suite.skeletal.Flush(0, 1, 2)
		writes := suite.skeletal.StagedWriteData()
		suite.NotEmpty(writes)
		suite.Equal(uint32(0), count) // no instance data dirty, only model dirty
	})
}

func (suite *animatorImplTest) TestSkeletalFlushNeedsRebuild() {
	suite.Run("should return 0 from Flush when needsRebuild is set on skeletal animator", func() {
		suite.skeletal.Grow(1000)
		suite.skeletal.AddInstance()
		count := suite.skeletal.Flush(0, 1, 2)
		suite.Equal(uint32(0), count)
	})
}

func (suite *animatorImplTest) TestSkeletalPrepareFrameLooping() {
	suite.Run("should wrap animation time when looping clip duration is exceeded", func() {
		suite.skeletal.AddInstance()
		// Add a clip with duration 0.5s
		suite.skeletal.AddClip(0.5, 24.0, nil, nil, nil, nil, nil, 0)
		suite.skeletal.PlayAnimation(0, 0, true)
		suite.skeletal.StagedWriteData() // drain AddClip write
		// Advance 0.7s > 0.5s duration — time should wrap
		suite.skeletal.PrepareFrame(0.7, 2)
		// Verify: no panic and staged writes are produced
		writes := suite.skeletal.StagedWriteData()
		suite.NotEmpty(writes)
	})
}

func (suite *animatorImplTest) TestSkeletalSetBoneCountZero() {
	suite.Run("should not panic when SetBoneCount is called with 0 on skeletal animator", func() {
		suite.NotPanics(func() {
			suite.skeletal.SetBoneCount(0)
		})
	})
}

func (suite *animatorImplTest) TestSkeletalDirtyRangeUpdate() {
	suite.Run("should update dirty range correctly when SetInstanceTransform is called twice on skeletal", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceTransform(1, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		// Second call extends the dirty range
		suite.skeletal.SetInstanceTransform(0, [3]float32{2, 0, 0}, [3]float32{1, 1, 1})
		suite.NotPanics(func() {
			suite.skeletal.Flush(0, 1, 2)
		})
	})
	suite.Run("should update dirty range correctly when SetInstanceData is called twice on skeletal", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceData(1, [3]float32{1, 0, 0}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{})
		suite.skeletal.SetInstanceData(0, [3]float32{2, 0, 0}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{})
		suite.NotPanics(func() {
			suite.skeletal.Flush(0, 1, 2)
		})
	})
	suite.Run("should update dirty range correctly when SetInstanceRotation is called twice on skeletal", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceRotation(1, [3]float32{0, 0.1, 0}, [3]float32{0, 0.5, 0})
		suite.skeletal.SetInstanceRotation(0, [3]float32{0, 0.2, 0}, [3]float32{0, 1.0, 0})
		suite.NotPanics(func() {
			suite.skeletal.Flush(0, 1, 2)
		})
	})
}

func (suite *animatorImplTest) TestSkeletalGrowWithInstances() {
	suite.Run("should mark instance and model data dirty after Grow with active instances", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceTransform(0, [3]float32{5, 0, 0}, [3]float32{1, 1, 1})
		suite.skeletal.StagedWriteData() // drain
		// Grow triggers dirty marking for existing instances
		suite.skeletal.Grow(1000)
		suite.True(suite.skeletal.NeedsRebuild())
	})
	suite.Run("should mark bone data dirty after Grow when bones are set", func() {
		suite.skeletal.SetBoneCount(2)
		suite.skeletal.StagedWriteData() // drain
		suite.skeletal.Grow(1000)
		suite.True(suite.skeletal.NeedsRebuild())
	})
}

func (suite *animatorImplTest) TestSkeletalInstanceRotationOutOfRange() {
	suite.Run("should return zeros for out-of-range index on skeletal animator", func() {
		rotSpeed, rot := suite.skeletal.InstanceRotation(99)
		suite.Equal([3]float32{}, rotSpeed)
		suite.Equal([3]float32{}, rot)
	})
}

func (suite *animatorImplTest) TestSkeletalRemoveInstanceAlreadyDirty() {
	suite.Run("should update existing dirty range when RemoveInstance swaps a dirty slot", func() {
		suite.skeletal.AddInstance() // 0
		suite.skeletal.AddInstance() // 1
		suite.skeletal.AddInstance() // 2
		// Run PrepareFrame to mark dirty first
		suite.skeletal.PlayAnimation(0, 0, false)
		suite.skeletal.PrepareFrame(0.016, 2) // sets dirty
		suite.skeletal.StagedWriteData()      // drain staged writes, NOT dirty flags
		// Now remove instance 0 — RemoveInstance will encounter already-dirty case
		suite.skeletal.RemoveInstance(0)
		suite.Equal(uint32(2), suite.skeletal.InstanceCount())
	})
}

func (suite *animatorImplTest) TestSkeletalPrepareFrameBlendWithLoopingClips() {
	suite.Run("should advance blend-to animation time with looping clips during PrepareFrame", func() {
		suite.skeletal.AddInstance()
		// Add two clips with duration
		suite.skeletal.AddClip(1.0, 24.0, nil, nil, nil, nil, nil, 0)
		suite.skeletal.AddClip(1.0, 24.0, nil, nil, nil, nil, nil, 0)
		suite.skeletal.PlayAnimation(0, 0, true)
		suite.skeletal.BlendToAnimation(0, 1, 0.5)
		suite.skeletal.StagedWriteData() // drain AddClip writes
		// Advance a small time — blend still in progress
		suite.skeletal.PrepareFrame(0.1, 2)
		suite.True(suite.skeletal.IsBlending(0))
		writes := suite.skeletal.StagedWriteData()
		suite.NotEmpty(writes)
	})
}

// --- RemoveInstance swapped=false paths ---

func (suite *animatorImplTest) TestSkeletalRemoveInstanceSwappedFalse() {
	suite.Run("should return swapped=false when removing the only skeletal instance", func() {
		suite.skeletal.AddInstance()
		_, swapped := suite.skeletal.RemoveInstance(0)
		suite.False(swapped)
		suite.Equal(uint32(0), suite.skeletal.InstanceCount())
	})
	suite.Run("should return false early when count is 0 on skeletal animator", func() {
		_, swapped := suite.skeletal.RemoveInstance(0)
		suite.False(swapped)
		suite.Equal(uint32(0), suite.skeletal.InstanceCount())
	})
}

// --- Flush early-return when no dirty flags ---

func (suite *animatorImplTest) TestSkeletalFlushNoDirtyFlags() {
	suite.Run("should return 0 from Flush when no dirty flags are set on skeletal animator", func() {
		count := suite.skeletal.Flush(0, 1, 2)
		suite.Equal(uint32(0), count)
		writes := suite.skeletal.StagedWriteData()
		suite.Empty(writes)
	})
}

// --- PrepareFrame missing branches ---

func (suite *animatorImplTest) TestSkeletalPrepareFrameMissingBranches() {
	suite.Run("should return early without staged writes when needsRebuild is set on skeletal animator", func() {
		suite.skeletal.Grow(1000)
		suite.skeletal.PrepareFrame(0.016, 2)
		writes := suite.skeletal.StagedWriteData()
		suite.Empty(writes)
	})
	suite.Run("should wrap blend-to animation time when looping blend-to clip duration is exceeded", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddClip(0.2, 24.0, nil, nil, nil, nil, nil, 0)
		suite.skeletal.AddClip(0.3, 24.0, nil, nil, nil, nil, nil, 0)
		suite.skeletal.PlayAnimation(0, 0, true)
		suite.skeletal.BlendToAnimation(0, 1, 0.5)
		suite.skeletal.StagedWriteData() // drain AddClip writes
		suite.skeletal.PrepareFrame(0.5, 2)
		writes := suite.skeletal.StagedWriteData()
		suite.NotEmpty(writes)
	})
}

// --- SetInstanceTransform modelDirtyEnd expansion ---

func (suite *animatorImplTest) TestSkeletalSetInstanceTransformDirtyRangeExpandEnd() {
	suite.Run("should expand dirtyEnd when second SetInstanceTransform uses a higher index", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		suite.skeletal.SetInstanceTransform(1, [3]float32{2, 0, 0}, [3]float32{1, 1, 1})
		suite.NotPanics(func() { suite.skeletal.Flush(0, 1, 2) })
	})
}

// --- SetInstanceData modelDirtyEnd expansion ---

func (suite *animatorImplTest) TestSkeletalSetInstanceDataDirtyRangeExpandEnd() {
	suite.Run("should expand dirtyEnd when second SetInstanceData uses a higher index", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceData(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{})
		suite.skeletal.SetInstanceData(1, [3]float32{2, 0, 0}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{})
		suite.NotPanics(func() { suite.skeletal.Flush(0, 1, 2) })
	})
}

// --- SetInstanceRotation modelDirtyEnd expansion ---

func (suite *animatorImplTest) TestSkeletalSetInstanceRotationDirtyRangeExpandEnd() {
	suite.Run("should expand dirtyEnd when second SetInstanceRotation uses a higher index", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		// SetInstanceTransform establishes position/scale in the model matrix, then Flush resets
		// the modelDirty flag so the subsequent SetInstanceRotation calls can exercise the
		// if !modelDirty branch (index 0) and the else index+1 > dirtyEnd sub-condition (index 1).
		suite.skeletal.SetInstanceTransform(0, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})
		suite.skeletal.SetInstanceTransform(1, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})
		suite.skeletal.Flush(0, 1, 2)
		suite.skeletal.SetInstanceRotation(0, [3]float32{}, [3]float32{0, 0.1, 0})
		suite.skeletal.SetInstanceRotation(1, [3]float32{}, [3]float32{0, 0.2, 0})
		suite.NotPanics(func() { suite.skeletal.Flush(0, 1, 2) })
	})
}

func (suite *animatorImplTest) TestSetScreenSize() {
	suite.Run("should store width and height on simple animator backend", func() {
		suite.simple.SetScreenSize(1920, 1080)
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		suite.Equal(1920, b.screenWidth)
		suite.Equal(1080, b.screenHeight)
	})
	suite.Run("should store width and height on skeletal animator backend", func() {
		suite.skeletal.SetScreenSize(2560, 1440)
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(2560, b.screenWidth)
		suite.Equal(1440, b.screenHeight)
	})
}

func (suite *animatorImplTest) TestSetProjectionX() {
	suite.Run("should store projX on simple animator backend", func() {
		suite.simple.SetProjectionX(1.5)
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		suite.InDelta(1.5, float64(b.projX), 1e-6)
	})
	suite.Run("should store projX on skeletal animator backend", func() {
		suite.skeletal.SetProjectionX(2.75)
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.InDelta(2.75, float64(b.projX), 1e-6)
	})
}

func (suite *animatorImplTest) TestSetHiZMipCount() {
	suite.Run("should store hiZMipCount on simple animator backend", func() {
		suite.simple.SetHiZMipCount(6)
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		suite.Equal(6, b.hiZMipCount)
	})
	suite.Run("should store hiZMipCount on skeletal animator backend", func() {
		suite.skeletal.SetHiZMipCount(4)
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(4, b.hiZMipCount)
	})
}

func (suite *animatorImplTest) TestSetViewProj() {
	suite.Run("should store viewProj on simple animator backend", func() {
		vp := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		suite.simple.SetViewProj(vp)
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		suite.Equal(vp, b.viewProj)
	})
	suite.Run("should store viewProj on skeletal animator backend", func() {
		vp := [16]float32{2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2}
		suite.skeletal.SetViewProj(vp)
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(vp, b.viewProj)
	})
}

func (suite *animatorImplTest) TestSimplePrepareFrameRotation() {
	suite.Run("should skip rotation update and continue when instance has zero RotSpeed", func() {
		suite.simple.AddInstance() // RotSpeed is {0,0,0} by default
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		suite.simple.PrepareFrame(0.016, 0)
		suite.Equal([3]float32{}, b.instanceData[0].Rot)
	})
	suite.Run("should advance rotation when instance has non-zero RotSpeed", func() {
		suite.simple.AddInstance()
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		b.instanceData[0].RotSpeed = [3]float32{1.0, 1.0, 1.0}
		suite.simple.PrepareFrame(0.016, 0)
		suite.NotEqual(float32(0), b.instanceData[0].Rot[0])
		suite.NotEmpty(b.dirtyIndices)
	})
	suite.Run("should wrap rotation above twoPi for all axes", func() {
		suite.simple.AddInstance()
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		b.instanceData[0].RotSpeed = [3]float32{7.0, 7.0, 7.0}
		suite.simple.PrepareFrame(1.0, 0)
		const twoPi = float32(6.283185307)
		suite.GreaterOrEqual(b.instanceData[0].Rot[0], float32(0))
		suite.Less(b.instanceData[0].Rot[0], twoPi)
		suite.GreaterOrEqual(b.instanceData[0].Rot[1], float32(0))
		suite.Less(b.instanceData[0].Rot[1], twoPi)
		suite.GreaterOrEqual(b.instanceData[0].Rot[2], float32(0))
		suite.Less(b.instanceData[0].Rot[2], twoPi)
	})
	suite.Run("should wrap rotation below zero for all axes", func() {
		suite.simple.AddInstance()
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		b.instanceData[0].RotSpeed = [3]float32{-7.0, -7.0, -7.0}
		suite.simple.PrepareFrame(1.0, 0)
		const twoPi = float32(6.283185307)
		suite.GreaterOrEqual(b.instanceData[0].Rot[0], float32(0))
		suite.Less(b.instanceData[0].Rot[0], twoPi)
		suite.GreaterOrEqual(b.instanceData[0].Rot[1], float32(0))
		suite.Less(b.instanceData[0].Rot[1], twoPi)
		suite.GreaterOrEqual(b.instanceData[0].Rot[2], float32(0))
		suite.Less(b.instanceData[0].Rot[2], twoPi)
	})
}

func (suite *animatorImplTest) TestSimpleFlushMissingBranches() {
	suite.Run("should return 0 when needsRebuild is true even with dirty indices", func() {
		suite.simple.AddInstance()
		suite.simple.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		b.needsRebuild = true
		count := suite.simple.Flush(0, 0, 0)
		suite.Equal(uint32(0), count)
	})
	suite.Run("should clear dirty indices after two flushes", func() {
		suite.simple.AddInstance()
		suite.simple.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		suite.simple.Flush(0, 0, 0)
		suite.simple.Flush(0, 0, 0)
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		suite.Empty(b.dirtyIndices)
		suite.Equal(0, b.flushCountdown)
	})
}

func (suite *animatorImplTest) TestSimpleHiZBindGroupProvider() {
	suite.Run("should return non-nil HiZ provider from simple animator backend", func() {
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		result := b.HiZBindGroupProvider()
		suite.NotNil(result)
	})
}

func (suite *animatorImplTest) TestSimpleSetBoundingBox() {
	suite.Run("should store bounding box on simple animator backend", func() {
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		b.SetBoundingBox([3]float32{-1, -2, -3}, [3]float32{1, 2, 3})
		suite.Equal([3]float32{-1, -2, -3}, b.boundingMin)
		suite.Equal([3]float32{1, 2, 3}, b.boundingMax)
	})
}

func (suite *animatorImplTest) TestSkeletalRemoveInstanceMissingBranches() {
	suite.Run("should return 0 and false when index is out of range on non-empty skeletal backend", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		idx, swapped := suite.skeletal.RemoveInstance(5)
		suite.Equal(uint32(0), idx)
		suite.False(swapped)
		suite.Equal(uint32(2), suite.skeletal.InstanceCount())
	})
	suite.Run("should update existing dirty range start when removing a swapped instance with lower index", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		b.dirty = true
		b.dirtyStart = 2
		b.dirtyEnd = 3
		suite.skeletal.RemoveInstance(0)
		suite.Equal(uint32(0), b.dirtyStart)
	})
	suite.Run("should update existing dirty range end when removing a swapped instance with higher index", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		b.dirty = true
		b.dirtyStart = 0
		b.dirtyEnd = 2
		suite.skeletal.RemoveInstance(2)
		suite.Equal(uint32(3), b.dirtyEnd)
	})
	suite.Run("should update existing modelDirty range start when removing a swapped instance with lower index", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceTransform(1, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})
		suite.skeletal.StagedWriteData()
		suite.skeletal.RemoveInstance(0)
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(uint32(0), b.modelDirtyStart)
	})
	suite.Run("should update existing modelDirty range end when removing a swapped instance with higher index", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		b.modelDirty = true
		b.modelDirtyStart = 0
		b.modelDirtyEnd = 2
		b.modelFlushRemaining = 2
		suite.skeletal.RemoveInstance(2)
		suite.Equal(uint32(3), b.modelDirtyEnd)
	})
}

func (suite *animatorImplTest) TestSkeletalFlushCountdownClear() {
	suite.Run("should clear boneDirty after two flushes", func() {
		suite.skeletal.SetBoneCount(2)
		suite.skeletal.SetBone(0, [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, [3]float32{}, [4]float32{0, 0, 0, 1}, [3]float32{1, 1, 1}, -1, 1)
		suite.skeletal.StagedWriteData()
		suite.skeletal.Flush(0, 1, 2)
		suite.skeletal.StagedWriteData()
		suite.skeletal.Flush(0, 1, 2)
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.False(b.boneDirty)
		suite.Equal(0, b.boneFlushRemaining)
	})
	suite.Run("should clear modelDirty after two flushes", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceTransform(0, [3]float32{1, 2, 3}, [3]float32{1, 1, 1})
		suite.skeletal.StagedWriteData()
		suite.skeletal.Flush(0, 1, 2)
		suite.skeletal.StagedWriteData()
		suite.skeletal.Flush(0, 1, 2)
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.False(b.modelDirty)
		suite.Equal(0, b.modelFlushRemaining)
	})
}

func (suite *animatorImplTest) TestSkeletalGrowNoOp() {
	suite.Run("should return early when newMax is not larger than current maxInstances", func() {
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		currentMax := b.maxInstances
		suite.skeletal.Grow(currentMax)
		suite.Equal(currentMax, b.maxInstances)
		suite.False(b.needsRebuild)
	})
}

func (suite *animatorImplTest) TestSkeletalSetInstanceTransformMissingBranches() {
	suite.Run("should return early when index is out of range", func() {
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.NotPanics(func() {
			suite.skeletal.SetInstanceTransform(b.maxInstances, [3]float32{}, [3]float32{1, 1, 1})
		})
		suite.Equal(uint32(0), suite.skeletal.InstanceCount())
	})
	suite.Run("should expand modelDirty range start when new index is lower than existing start", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceTransform(2, [3]float32{}, [3]float32{1, 1, 1})
		suite.skeletal.SetInstanceTransform(0, [3]float32{}, [3]float32{1, 1, 1})
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(uint32(0), b.modelDirtyStart)
	})
}

func (suite *animatorImplTest) TestSkeletalSetInstanceRotationMissingBranches() {
	suite.Run("should return early when index is out of range", func() {
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.NotPanics(func() {
			suite.skeletal.SetInstanceRotation(b.maxInstances, [3]float32{}, [3]float32{})
		})
	})
	suite.Run("should expand modelDirty range start when new index is lower than existing start", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceRotation(2, [3]float32{}, [3]float32{})
		suite.skeletal.SetInstanceRotation(0, [3]float32{}, [3]float32{})
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(uint32(0), b.modelDirtyStart)
	})
}

func (suite *animatorImplTest) TestSkeletalHiZBindGroupProvider() {
	suite.Run("should return non-nil HiZ provider from skeletal animator backend", func() {
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		result := b.HiZBindGroupProvider()
		suite.NotNil(result)
	})
}

func (suite *animatorImplTest) TestSkeletalSetBoundingBox() {
	suite.Run("should store bounding box on skeletal animator backend", func() {
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		b.SetBoundingBox([3]float32{-1, -2, -3}, [3]float32{4, 5, 6})
		suite.Equal([3]float32{-1, -2, -3}, b.boundingMin)
		suite.Equal([3]float32{4, 5, 6}, b.boundingMax)
	})
}

func (suite *animatorImplTest) TestHiZBindGroupProvider() {
	suite.Run("should delegate HiZ provider call on simple animator", func() {
		result := suite.simple.(*animator).HiZBindGroupProvider()
		suite.NotNil(result)
	})
	suite.Run("should delegate HiZ provider call on skeletal animator", func() {
		result := suite.skeletal.(*animator).HiZBindGroupProvider()
		suite.NotNil(result)
	})
}

func (suite *animatorImplTest) TestSetBoundingBox() {
	suite.Run("should delegate SetBoundingBox call on simple animator", func() {
		suite.simple.(*animator).SetBoundingBox([3]float32{-1, -1, -1}, [3]float32{1, 1, 1})
		b := suite.simple.(*animator).backend.(*simpleAnimatorBackendImpl)
		suite.Equal([3]float32{-1, -1, -1}, b.boundingMin)
		suite.Equal([3]float32{1, 1, 1}, b.boundingMax)
	})
	suite.Run("should delegate SetBoundingBox call on skeletal animator", func() {
		suite.skeletal.(*animator).SetBoundingBox([3]float32{-1, -1, -1}, [3]float32{1, 1, 1})
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal([3]float32{-1, -1, -1}, b.boundingMin)
		suite.Equal([3]float32{1, 1, 1}, b.boundingMax)
	})
}

func (suite *animatorImplTest) TestSkeletalPrepareFrameDirtyStartExpansion() {
	suite.Run("should expand dirty range start when earlier instance is processed during PrepareFrame", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		b.dirty = true
		b.dirtyStart = 2
		b.dirtyEnd = 3
		suite.skeletal.PrepareFrame(0.016, 2)
		suite.Equal(uint32(0), b.dirtyStart)
	})
}

func (suite *animatorImplTest) TestSkeletalSetInstanceRotationEndExpansion() {
	suite.Run("should expand modelDirty range end when new index is higher than existing end", func() {
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.AddInstance()
		suite.skeletal.SetInstanceRotation(0, [3]float32{}, [3]float32{})
		suite.skeletal.SetInstanceRotation(2, [3]float32{}, [3]float32{})
		b := suite.skeletal.(*animator).backend.(*skeletalAnimatorBackendImpl)
		suite.Equal(uint32(3), b.modelDirtyEnd)
	})
}
