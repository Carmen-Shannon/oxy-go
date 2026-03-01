package animator_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/animator"
	modelmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/model"
	"github.com/stretchr/testify/suite"
)

type animatorTest struct {
	suite.Suite
}

func TestAnimator(t *testing.T) {
	suite.Run(t, new(animatorTest))
}

func (suite *animatorTest) TestNewAnimatorSimple() {
	suite.Run("creates a simple backend by default", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		suite.NotNil(a)
		suite.Equal(animator.BackendTypeSimple, a.BackendType())
	})

	suite.Run("simple backend has default max instances of 25000", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		suite.Equal(uint32(25000), a.MaxInstances())
	})

	suite.Run("simple backend starts with zero instance count", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		suite.Equal(uint32(0), a.InstanceCount())
	})

	suite.Run("WithMaxInstances overrides default for simple backend", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(100))
		suite.Equal(uint32(100), a.MaxInstances())
	})

	suite.Run("compute and output providers are initialized", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		suite.NotNil(a.ComputeBindGroupProvider())
		suite.NotNil(a.OutputBindGroupProvider())
	})
}

func (suite *animatorTest) TestNewAnimatorSkeletal() {
	suite.Run("creates a skeletal backend", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal)
		suite.NotNil(a)
		suite.Equal(animator.BackendTypeSkeletal, a.BackendType())
	})

	suite.Run("skeletal backend has default max instances of 200", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal)
		suite.Equal(uint32(200), a.MaxInstances())
	})

	suite.Run("skeletal backend starts with zero instance count", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal)
		suite.Equal(uint32(0), a.InstanceCount())
	})

	suite.Run("WithMaxInstances overrides default for skeletal backend", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(50))
		suite.Equal(uint32(50), a.MaxInstances())
	})

	suite.Run("compute and output providers are initialized", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal)
		suite.NotNil(a.ComputeBindGroupProvider())
		suite.NotNil(a.OutputBindGroupProvider())
	})
}

func (suite *animatorTest) TestAddInstanceSimple() {
	suite.Run("first instance returns index 0", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		idx, err := a.AddInstance()
		suite.NoError(err)
		suite.Equal(uint32(0), idx)
	})

	suite.Run("sequential instances return incrementing indices", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		idx0, err := a.AddInstance()
		suite.NoError(err)
		suite.Equal(uint32(0), idx0)
		idx1, err := a.AddInstance()
		suite.NoError(err)
		suite.Equal(uint32(1), idx1)
		idx2, err := a.AddInstance()
		suite.NoError(err)
		suite.Equal(uint32(2), idx2)
		suite.Equal(uint32(3), a.InstanceCount())
	})

	suite.Run("auto-grows when capacity exceeded", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(2))
		_, _ = a.AddInstance()
		_, _ = a.AddInstance()
		suite.Equal(uint32(2), a.MaxInstances())

		idx, err := a.AddInstance()
		suite.NoError(err)
		suite.Equal(uint32(2), idx)
		suite.True(a.MaxInstances() > 2)
		suite.Equal(uint32(3), a.InstanceCount())
	})
}

func (suite *animatorTest) TestAddInstanceSkeletal() {
	suite.Run("first instance returns index 0", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		idx, err := a.AddInstance()
		suite.NoError(err)
		suite.Equal(uint32(0), idx)
	})

	suite.Run("auto-grows when capacity exceeded", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(2))
		_, _ = a.AddInstance()
		_, _ = a.AddInstance()
		idx, err := a.AddInstance()
		suite.NoError(err)
		suite.Equal(uint32(2), idx)
		suite.True(a.MaxInstances() > 2)
	})
}

func (suite *animatorTest) TestSetInstanceTransformSimple() {
	suite.Run("stores and retrieves position and scale", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		pos := [3]float32{1.0, 2.0, 3.0}
		scale := [3]float32{4.0, 5.0, 6.0}
		a.SetInstanceTransform(0, pos, scale)

		gotPos, gotScale := a.InstanceTransform(0)
		suite.Equal(pos, gotPos)
		suite.Equal(scale, gotScale)
	})

	suite.Run("out of bounds index is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		a.SetInstanceTransform(999, [3]float32{1, 2, 3}, [3]float32{1, 1, 1})
		// should not panic
	})

	suite.Run("returns zeros for out of bounds get", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		pos, scale := a.InstanceTransform(999)
		suite.Equal([3]float32{}, pos)
		suite.Equal([3]float32{}, scale)
	})
}

func (suite *animatorTest) TestSetInstanceRotationSimple() {
	suite.Run("stores and retrieves rotation speed and rotation", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		rotSpeed := [3]float32{0.1, 0.2, 0.3}
		rot := [3]float32{1.0, 2.0, 3.0}
		a.SetInstanceRotation(0, rotSpeed, rot)

		gotRotSpeed, gotRot := a.InstanceRotation(0)
		suite.Equal(rotSpeed, gotRotSpeed)
		suite.Equal(rot, gotRot)
	})

	suite.Run("out of bounds index is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		a.SetInstanceRotation(999, [3]float32{0.1, 0.2, 0.3}, [3]float32{1, 2, 3})
	})

	suite.Run("returns zeros for out of bounds get", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		rotSpeed, rot := a.InstanceRotation(999)
		suite.Equal([3]float32{}, rotSpeed)
		suite.Equal([3]float32{}, rot)
	})
}

func (suite *animatorTest) TestSetInstanceDataSimple() {
	suite.Run("sets all transform data in one call", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		pos := [3]float32{10, 20, 30}
		scale := [3]float32{2, 3, 4}
		rotSpeed := [3]float32{0.5, 0.6, 0.7}
		rot := [3]float32{1.1, 2.2, 3.3}
		a.SetInstanceData(0, pos, scale, rotSpeed, rot)

		gotPos, gotScale := a.InstanceTransform(0)
		suite.Equal(pos, gotPos)
		suite.Equal(scale, gotScale)

		gotRotSpeed, gotRot := a.InstanceRotation(0)
		suite.Equal(rotSpeed, gotRotSpeed)
		suite.Equal(rot, gotRot)
	})

	suite.Run("out of bounds index is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		a.SetInstanceData(999, [3]float32{}, [3]float32{}, [3]float32{}, [3]float32{})
	})
}

func (suite *animatorTest) TestFlushSimple() {
	suite.Run("returns 0 when nothing is dirty", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		count := a.Flush(0, 1, 2)
		suite.Equal(uint32(0), count)
	})

	suite.Run("returns dirty count after modifying instances", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		_, _ = a.AddInstance()
		a.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(1, [3]float32{2, 0, 0}, [3]float32{1, 1, 1})

		count := a.Flush(0, 1, 2)
		suite.Equal(uint32(2), count)
	})

	suite.Run("second flush returns 0 when nothing changed", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		a.Flush(0, 1, 2)

		count := a.Flush(0, 1, 2)
		suite.Equal(uint32(0), count)
	})

	suite.Run("deduplicates dirty indices for same instance", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(0, [3]float32{2, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(0, [3]float32{3, 0, 0}, [3]float32{1, 1, 1})

		count := a.Flush(0, 1, 2)
		suite.Equal(uint32(1), count)
	})

	suite.Run("flush returns 0 during needsRebuild", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(2))
		_, _ = a.AddInstance()
		a.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		a.Grow(10)
		suite.True(a.NeedsRebuild())

		count := a.Flush(0, 1, 2)
		suite.Equal(uint32(0), count)
	})
}

func (suite *animatorTest) TestStagedWriteDataSimple() {
	suite.Run("returns empty slice initially", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		writes := a.StagedWriteData()
		suite.Len(writes, 0)
	})

	suite.Run("drains writes after flush", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		a.Flush(0, 1, 2)

		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)

		// second drain is empty
		writes2 := a.StagedWriteData()
		suite.Len(writes2, 0)
	})
}

func (suite *animatorTest) TestPrepareFrameSimple() {
	suite.Run("stages per-frame uniform data", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.PrepareFrame(0.016, 0)

		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)
	})

	suite.Run("no-op during needsRebuild", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(2))
		_, _ = a.AddInstance()
		a.Grow(10)
		suite.True(a.NeedsRebuild())

		a.PrepareFrame(0.016, 0)
		writes := a.StagedWriteData()
		suite.Len(writes, 0)
	})
}

func (suite *animatorTest) TestGrowSimple() {
	suite.Run("increases capacity", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		a.Grow(100)
		suite.Equal(uint32(100), a.MaxInstances())
	})

	suite.Run("preserves existing instance data", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(4))
		_, _ = a.AddInstance()
		pos := [3]float32{7, 8, 9}
		scale := [3]float32{2, 3, 4}
		a.SetInstanceTransform(0, pos, scale)
		a.Flush(0, 1, 2)
		_ = a.StagedWriteData()

		a.Grow(20)
		a.ClearNeedsRebuild()

		gotPos, gotScale := a.InstanceTransform(0)
		suite.Equal(pos, gotPos)
		suite.Equal(scale, gotScale)
	})

	suite.Run("sets needsRebuild flag", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(4))
		suite.False(a.NeedsRebuild())
		a.Grow(20)
		suite.True(a.NeedsRebuild())
	})

	suite.Run("no-op when newMax is less than or equal to current", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		a.Grow(5)
		suite.Equal(uint32(10), a.MaxInstances())
		suite.False(a.NeedsRebuild())
	})
}

func (suite *animatorTest) TestRemoveInstanceSimple() {
	suite.Run("swap-removes middle instance", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance() // 0
		_, _ = a.AddInstance() // 1
		_, _ = a.AddInstance() // 2
		a.SetInstanceTransform(0, [3]float32{10, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(1, [3]float32{20, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(2, [3]float32{30, 0, 0}, [3]float32{1, 1, 1})

		old, swapped := a.RemoveInstance(0)
		suite.True(swapped)
		suite.Equal(uint32(2), old)
		suite.Equal(uint32(2), a.InstanceCount())

		// slot 0 should now contain what was in slot 2
		pos, _ := a.InstanceTransform(0)
		suite.InDelta(float32(30), pos[0], 1e-6)
	})

	suite.Run("removing last instance does not swap", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		_, _ = a.AddInstance()

		_, swapped := a.RemoveInstance(1)
		suite.False(swapped)
		suite.Equal(uint32(1), a.InstanceCount())
	})

	suite.Run("removing from empty animator returns false", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, swapped := a.RemoveInstance(0)
		suite.False(swapped)
	})

	suite.Run("out of bounds index returns false", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		_, swapped := a.RemoveInstance(999)
		suite.False(swapped)
	})
}

func (suite *animatorTest) TestNeedsRebuildAndClearSimple() {
	suite.Run("starts as false", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		suite.False(a.NeedsRebuild())
	})

	suite.Run("clear resets the flag", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(2))
		a.Grow(10)
		suite.True(a.NeedsRebuild())
		a.ClearNeedsRebuild()
		suite.False(a.NeedsRebuild())
	})
}

func (suite *animatorTest) TestFrustumCullingSimple() {
	suite.Run("culling is initially disabled", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		suite.False(a.CullingEnabled())
	})

	suite.Run("setting frustum planes enables culling", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		planes := [6]animator.GPUFrustumPlane{}
		for i := range 6 {
			planes[i] = animator.GPUFrustumPlane{Normal: [3]float32{0, 1, 0}, Distance: 1.0}
		}
		a.SetFrustumPlanes(planes)
		suite.True(a.CullingEnabled())
	})

	suite.Run("bounding radius is stored and retrieved", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		a.SetBoundingRadius(5.5)
		suite.InDelta(float32(5.5), a.BoundingRadius(), 1e-6)
	})

	suite.Run("indirect buffer returns nil when culling is disabled", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		buf := a.IndirectBuffer(0)
		suite.Nil(buf)
	})
}

func (suite *animatorTest) TestResetIndirectArgsSimple() {
	suite.Run("stages write when culling is enabled", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		a.SetFrustumPlanes([6]animator.GPUFrustumPlane{})
		a.ResetIndirectArgs(36, 3)

		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)
	})

	suite.Run("no-op when culling is disabled", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		a.ResetIndirectArgs(36, 3)
		writes := a.StagedWriteData()
		suite.Len(writes, 0)
	})

	suite.Run("no-op during needsRebuild", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(2))
		a.SetFrustumPlanes([6]animator.GPUFrustumPlane{})
		a.Grow(10)
		a.ResetIndirectArgs(36, 3)
		writes := a.StagedWriteData()
		suite.Len(writes, 0)
	})
}

func (suite *animatorTest) TestSimpleBackendSkeletalNoOps() {
	suite.Run("SetBoneCount is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		a.SetBoneCount(10) // should not panic
	})

	suite.Run("SetBone is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		a.SetBone(0, [16]float32{}, [3]float32{}, [4]float32{}, [3]float32{}, -1, 0)
	})

	suite.Run("AddClip returns 0", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		idx := a.AddClip(1.0, 30.0, nil, nil, nil, nil, nil, 0)
		suite.Equal(uint32(0), idx)
	})

	suite.Run("PlayAnimation is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		a.PlayAnimation(0, 0, true)
	})

	suite.Run("BlendToAnimation is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		a.BlendToAnimation(0, 1, 0.5)
	})

	suite.Run("SetAnimationTime is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		a.SetAnimationTime(0, 1.0)
	})

	suite.Run("SetAnimationSpeed is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		a.SetAnimationSpeed(0, 2.0)
	})

	suite.Run("IsBlending returns false", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		suite.False(a.IsBlending(0))
	})

	suite.Run("BlendProgress returns 0", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		suite.InDelta(float32(0), a.BlendProgress(0), 1e-6)
	})

	suite.Run("CancelBlend is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		a.CancelBlend(0)
	})

}

func (suite *animatorTest) TestReleaseSimple() {
	suite.Run("release does not panic", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetInstanceTransform(0, [3]float32{1, 2, 3}, [3]float32{1, 1, 1})
		a.Release()
	})
}

func (suite *animatorTest) TestModelSimple() {
	suite.Run("model is nil by default", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple)
		suite.Nil(a.Model())
	})
}

func (suite *animatorTest) TestSetInstanceTransformSkeletal() {
	suite.Run("stores and retrieves position and scale from model matrix", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		pos := [3]float32{5, 10, 15}
		scale := [3]float32{2, 3, 4}
		a.SetInstanceTransform(0, pos, scale)

		gotPos, gotScale := a.InstanceTransform(0)
		suite.InDelta(pos[0], gotPos[0], 1e-5)
		suite.InDelta(pos[1], gotPos[1], 1e-5)
		suite.InDelta(pos[2], gotPos[2], 1e-5)
		suite.InDelta(scale[0], gotScale[0], 1e-5)
		suite.InDelta(scale[1], gotScale[1], 1e-5)
		suite.InDelta(scale[2], gotScale[2], 1e-5)
	})

	suite.Run("out of bounds index is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetInstanceTransform(999, [3]float32{1, 2, 3}, [3]float32{1, 1, 1})
	})
}

func (suite *animatorTest) TestSetInstanceRotationSkeletal() {
	suite.Run("stores rotation and rebuilds model matrix", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		// Set transform first (needed so matrix has valid pos/scale)
		a.SetInstanceTransform(0, [3]float32{1, 2, 3}, [3]float32{1, 1, 1})
		a.SetInstanceRotation(0, [3]float32{0.1, 0.2, 0.3}, [3]float32{0.5, 0.6, 0.7})

		gotRotSpeed, gotRot := a.InstanceRotation(0)
		// Skeletal stores rotation speed in separate tracking, not in GPU data
		suite.Equal([3]float32{}, gotRotSpeed) // rotSpeed not stored for skeletal
		suite.InDelta(float32(0.7), gotRot[2], 1e-5)
	})

	suite.Run("out of bounds index is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetInstanceRotation(999, [3]float32{}, [3]float32{})
	})
}

func (suite *animatorTest) TestSetInstanceDataSkeletal() {
	suite.Run("sets all transform data and builds model matrix", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		pos := [3]float32{10, 20, 30}
		scale := [3]float32{2, 2, 2}
		a.SetInstanceData(0, pos, scale, [3]float32{}, [3]float32{0, 0, 0})

		gotPos, gotScale := a.InstanceTransform(0)
		suite.InDelta(pos[0], gotPos[0], 1e-5)
		suite.InDelta(pos[1], gotPos[1], 1e-5)
		suite.InDelta(pos[2], gotPos[2], 1e-5)
		suite.InDelta(scale[0], gotScale[0], 1e-5)
		suite.InDelta(scale[1], gotScale[1], 1e-5)
		suite.InDelta(scale[2], gotScale[2], 1e-5)
	})
}

func (suite *animatorTest) TestFlushSkeletal() {
	suite.Run("returns 0 when nothing is dirty", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		count := a.Flush(0, 1, 2)
		suite.Equal(uint32(0), count)
	})

	suite.Run("flush returns 0 during needsRebuild", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(2))
		_, _ = a.AddInstance()
		a.Grow(10)
		count := a.Flush(0, 1, 2)
		suite.Equal(uint32(0), count)
	})
}

func (suite *animatorTest) TestPrepareFrameSkeletal() {
	suite.Run("stages per-frame uniform data", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.PrepareFrame(0.016, 0)

		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)
	})

	suite.Run("no-op during needsRebuild", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(2))
		_, _ = a.AddInstance()
		a.Grow(10)

		a.PrepareFrame(0.016, 0)
		writes := a.StagedWriteData()
		suite.Len(writes, 0)
	})
}

func (suite *animatorTest) TestGrowSkeletal() {
	suite.Run("increases capacity", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.Grow(100)
		suite.Equal(uint32(100), a.MaxInstances())
	})

	suite.Run("preserves existing instance data", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(4))
		_, _ = a.AddInstance()
		a.SetInstanceTransform(0, [3]float32{7, 8, 9}, [3]float32{2, 3, 4})
		a.Flush(0, 1, 2)
		_ = a.StagedWriteData()

		a.Grow(20)
		a.ClearNeedsRebuild()

		pos, scale := a.InstanceTransform(0)
		suite.InDelta(float32(7), pos[0], 1e-5)
		suite.InDelta(float32(8), pos[1], 1e-5)
		suite.InDelta(float32(9), pos[2], 1e-5)
		suite.InDelta(float32(2), scale[0], 1e-5)
		suite.InDelta(float32(3), scale[1], 1e-5)
		suite.InDelta(float32(4), scale[2], 1e-5)
	})

	suite.Run("sets needsRebuild flag", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(4))
		a.Grow(20)
		suite.True(a.NeedsRebuild())
	})

	suite.Run("no-op when newMax is less than or equal to current", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.Grow(5)
		suite.Equal(uint32(10), a.MaxInstances())
		suite.False(a.NeedsRebuild())
	})

	suite.Run("preserves bone dirty flag when bones exist", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(4))
		a.SetBoneCount(2)
		a.SetBone(0, [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, [3]float32{}, [4]float32{0, 0, 0, 1}, [3]float32{1, 1, 1}, -1, 0)
		_ = a.StagedWriteData() // drain

		a.Grow(20)
		suite.True(a.NeedsRebuild())
	})
}

func (suite *animatorTest) TestRemoveInstanceSkeletal() {
	suite.Run("swap-removes middle instance", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance() // 0
		_, _ = a.AddInstance() // 1
		_, _ = a.AddInstance() // 2
		a.SetInstanceTransform(0, [3]float32{10, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(1, [3]float32{20, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(2, [3]float32{30, 0, 0}, [3]float32{1, 1, 1})

		old, swapped := a.RemoveInstance(0)
		suite.True(swapped)
		suite.Equal(uint32(2), old)
		suite.Equal(uint32(2), a.InstanceCount())

		pos, _ := a.InstanceTransform(0)
		suite.InDelta(float32(30), pos[0], 1e-5)
	})

	suite.Run("removing last instance does not swap", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		_, _ = a.AddInstance()
		_, swapped := a.RemoveInstance(1)
		suite.False(swapped)
		suite.Equal(uint32(1), a.InstanceCount())
	})

	suite.Run("removing from empty returns false", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, swapped := a.RemoveInstance(0)
		suite.False(swapped)
	})

	suite.Run("out of bounds index returns false", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		_, swapped := a.RemoveInstance(999)
		suite.False(swapped)
	})
}

func (suite *animatorTest) TestSetBoneCountSkeletal() {
	suite.Run("allocates bone data", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetBoneCount(5)
		// Should not panic and should allow SetBone calls up to index 4
		a.SetBone(0, [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, [3]float32{}, [4]float32{0, 0, 0, 1}, [3]float32{1, 1, 1}, -1, 0)
		a.SetBone(4, [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, [3]float32{}, [4]float32{0, 0, 0, 1}, [3]float32{1, 1, 1}, 0, 0)
	})
}

func (suite *animatorTest) TestSetBoneSkeletal() {
	suite.Run("out of bounds index is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetBoneCount(2)
		// index 5 > boneCount 2, should not panic
		a.SetBone(5, [16]float32{}, [3]float32{}, [4]float32{}, [3]float32{}, -1, 0)
	})

	suite.Run("stages bone buffer write", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetBoneCount(2)
		_ = a.StagedWriteData() // drain from SetBoneCount

		a.SetBone(0, [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, [3]float32{0, 1, 0}, [4]float32{0, 0, 0, 1}, [3]float32{1, 1, 1}, -1, 1)
		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)
	})
}

func (suite *animatorTest) TestAddClipSkeletal() {
	suite.Run("returns clip index 0 for first clip", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetBoneCount(2)
		_ = a.StagedWriteData()

		// 1 channel targeting bone 0 with 2 position keyframes
		channels := []uint32{
			0, // boneIndex
			0, // posKeyOffset
			2, // posKeyCount
			0, // rotKeyOffset (none)
			0, // rotKeyCount
			0, // scaleKeyOffset (none)
			0, // scaleKeyCount
		}
		times := []float32{0.0, 1.0}
		translations := [][3]float32{{0, 0, 0}, {1, 0, 0}}
		rotations := [][4]float32{{0, 0, 0, 1}, {0, 0, 0, 1}}
		scales := [][3]float32{{1, 1, 1}, {1, 1, 1}}

		idx := a.AddClip(2.0, 30.0, channels, times, translations, rotations, scales, 2)
		suite.Equal(uint32(0), idx)
	})

	suite.Run("second clip returns index 1", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetBoneCount(1)
		_ = a.StagedWriteData()

		channels := []uint32{0, 0, 1, 0, 0, 0, 0}
		times := []float32{0.0}
		trans := [][3]float32{{0, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}}

		idx0 := a.AddClip(1.0, 30.0, channels, times, trans, rots, scls, 2)
		suite.Equal(uint32(0), idx0)
		idx1 := a.AddClip(2.0, 30.0, channels, times, trans, rots, scls, 2)
		suite.Equal(uint32(1), idx1)
	})

	suite.Run("stages packed buffer write", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetBoneCount(1)
		_ = a.StagedWriteData()

		channels := []uint32{0, 0, 1, 0, 0, 0, 0}
		times := []float32{0.0}
		trans := [][3]float32{{0, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}}

		a.AddClip(1.0, 30.0, channels, times, trans, rots, scls, 2)
		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)
	})
}

func (suite *animatorTest) TestPlayAnimationSkeletal() {
	suite.Run("does not panic with valid instance", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(1)
		channels := []uint32{0, 0, 1, 0, 0, 0, 0}
		a.AddClip(1.0, 30.0, channels, []float32{0.0}, [][3]float32{{0, 0, 0}}, [][4]float32{{0, 0, 0, 1}}, [][3]float32{{1, 1, 1}}, 2)
		_ = a.StagedWriteData()

		a.PlayAnimation(0, 0, true)
	})

	suite.Run("out of bounds instance is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.PlayAnimation(999, 0, true)
	})
}

func (suite *animatorTest) TestBlendToAnimationSkeletal() {
	suite.Run("sets blending state", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(1)
		channels := []uint32{0, 0, 1, 0, 0, 0, 0}
		kfArgs := []float32{0.0}
		trans := [][3]float32{{0, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}}
		a.AddClip(1.0, 30.0, channels, kfArgs, trans, rots, scls, 2)
		a.AddClip(2.0, 30.0, channels, kfArgs, trans, rots, scls, 2)
		_ = a.StagedWriteData()

		a.PlayAnimation(0, 0, true)
		a.BlendToAnimation(0, 1, 0.5)
		suite.True(a.IsBlending(0))
	})

	suite.Run("out of bounds instance is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.BlendToAnimation(999, 0, 0.5)
	})
}

func (suite *animatorTest) TestBlendProgressSkeletal() {
	suite.Run("returns 0 when not blending", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		suite.InDelta(float32(0), a.BlendProgress(0), 1e-6)
	})

	suite.Run("progress increases with PrepareFrame", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(1)
		channels := []uint32{0, 0, 1, 0, 0, 0, 0}
		kf := []float32{0.0}
		trans := [][3]float32{{0, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}}
		a.AddClip(2.0, 30.0, channels, kf, trans, rots, scls, 2)
		a.AddClip(2.0, 30.0, channels, kf, trans, rots, scls, 2)
		_ = a.StagedWriteData()

		a.PlayAnimation(0, 0, true)
		a.BlendToAnimation(0, 1, 1.0) // 1 second blend

		// Advance half the blend duration
		a.PrepareFrame(0.5, 0)
		_ = a.StagedWriteData()

		progress := a.BlendProgress(0)
		suite.InDelta(float32(0.5), progress, 1e-5)
	})

	suite.Run("returns 0 for out of bounds instance", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		suite.InDelta(float32(0), a.BlendProgress(999), 1e-6)
	})
}

func (suite *animatorTest) TestCancelBlendSkeletal() {
	suite.Run("stops blending", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(1)
		channels := []uint32{0, 0, 1, 0, 0, 0, 0}
		kf := []float32{0.0}
		trans := [][3]float32{{0, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}}
		a.AddClip(2.0, 30.0, channels, kf, trans, rots, scls, 2)
		a.AddClip(2.0, 30.0, channels, kf, trans, rots, scls, 2)
		_ = a.StagedWriteData()

		a.PlayAnimation(0, 0, true)
		a.BlendToAnimation(0, 1, 1.0)
		suite.True(a.IsBlending(0))

		a.CancelBlend(0)
		suite.False(a.IsBlending(0))
	})

	suite.Run("out of bounds is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.CancelBlend(999)
	})
}

func (suite *animatorTest) TestSetAnimationTimeSkeletal() {
	suite.Run("does not panic with valid instance", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetAnimationTime(0, 2.5)
	})

	suite.Run("out of bounds is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetAnimationTime(999, 2.5)
	})
}

func (suite *animatorTest) TestSetAnimationSpeedSkeletal() {
	suite.Run("does not panic with valid instance", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetAnimationSpeed(0, 2.0)
	})

	suite.Run("out of bounds is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetAnimationSpeed(999, 2.0)
	})
}

func (suite *animatorTest) TestIsBlendingSkeletal() {
	suite.Run("returns false for out of bounds", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		suite.False(a.IsBlending(999))
	})

	suite.Run("returns false when not blending", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		suite.False(a.IsBlending(0))
	})
}

func (suite *animatorTest) TestFrustumCullingSkeletal() {
	suite.Run("culling is initially disabled", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		suite.False(a.CullingEnabled())
	})

	suite.Run("setting frustum planes enables culling", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetFrustumPlanes([6]animator.GPUFrustumPlane{})
		suite.True(a.CullingEnabled())
	})

	suite.Run("bounding radius is stored and retrieved", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetBoundingRadius(3.14)
		suite.InDelta(float32(3.14), a.BoundingRadius(), 1e-6)
	})
}

func (suite *animatorTest) TestResetIndirectArgsSkeletal() {
	suite.Run("stages write", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.ResetIndirectArgs(36, 3)
		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)
	})
}

func (suite *animatorTest) TestReleaseSkeletal() {
	suite.Run("release does not panic", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(2)
		a.SetBone(0, [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, [3]float32{}, [4]float32{0, 0, 0, 1}, [3]float32{1, 1, 1}, -1, 0)
		a.Release()
	})
}

func (suite *animatorTest) TestBlendCompletionSkeletal() {
	suite.Run("blend completes and transitions to target clip", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(1)
		channels := []uint32{0, 0, 1, 0, 0, 0, 0}
		kf := []float32{0.0}
		trans := [][3]float32{{0, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}}
		a.AddClip(2.0, 30.0, channels, kf, trans, rots, scls, 2)
		a.AddClip(3.0, 30.0, channels, kf, trans, rots, scls, 2)
		_ = a.StagedWriteData()

		a.PlayAnimation(0, 0, true)
		a.BlendToAnimation(0, 1, 0.5) // 0.5 second blend
		suite.True(a.IsBlending(0))

		// Advance past blend duration
		a.PrepareFrame(0.6, 0)
		_ = a.StagedWriteData()

		// Blend should have completed
		suite.False(a.IsBlending(0))
	})
}

func (suite *animatorTest) TestPrepareFrameLoopingSkeletal() {
	suite.Run("animation time wraps when looping", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(1)

		channels := []uint32{0, 0, 2, 0, 0, 0, 0}
		kf := []float32{0.0, 1.0}
		trans := [][3]float32{{0, 0, 0}, {1, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}, {0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}, {1, 1, 1}}
		a.AddClip(1.0, 1.0, channels, kf, trans, rots, scls, 2)
		_ = a.StagedWriteData()

		a.PlayAnimation(0, 0, true)
		// Speed is 1.0 by default, advance 2.5 seconds past a 1.0s clip
		a.PrepareFrame(2.5, 0)
		_ = a.StagedWriteData()

		// Animation should still be playing (looped)
		// We can check it didn't crash and blending did not activate
		suite.False(a.IsBlending(0))
	})
}

func (suite *animatorTest) TestFlushWithBoneAndModelDataSkeletal() {
	suite.Run("flush stages instance, bone, and model data", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(1)
		a.SetBone(0, [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, [3]float32{}, [4]float32{0, 0, 0, 1}, [3]float32{1, 1, 1}, -1, 1)
		_ = a.StagedWriteData()

		// Set transform to make model dirty
		a.SetInstanceTransform(0, [3]float32{5, 0, 0}, [3]float32{1, 1, 1})

		// PrepareFrame to make instance data dirty
		a.PlayAnimation(0, 0, true)
		channels := []uint32{0, 0, 1, 0, 0, 0, 0}
		a.AddClip(1.0, 30.0, channels, []float32{0.0}, [][3]float32{{0, 0, 0}}, [][4]float32{{0, 0, 0, 1}}, [][3]float32{{1, 1, 1}}, 2)
		_ = a.StagedWriteData()
		a.PrepareFrame(0.016, 0)

		count := a.Flush(0, 1, 2)
		// At least 1 dirty instance from PrepareFrame
		suite.True(count > 0)

		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)
	})
}

func (suite *animatorTest) TestFlushCoalesceSimple() {
	suite.Run("contiguous dirty instances produce fewer writes", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(100))
		for i := 0; i < 10; i++ {
			_, _ = a.AddInstance()
		}

		// Make a contiguous range dirty
		for i := uint32(0); i < 10; i++ {
			a.SetInstanceTransform(i, [3]float32{float32(i), 0, 0}, [3]float32{1, 1, 1})
		}

		count := a.Flush(0, 1, 2)
		suite.Equal(uint32(10), count)

		writes := a.StagedWriteData()
		// contiguous range should produce a single write
		suite.Equal(1, len(writes))
	})

	suite.Run("non-contiguous dirty instances produce separate writes", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(100))
		for i := 0; i < 10; i++ {
			_, _ = a.AddInstance()
		}

		// Make non-contiguous dirty (indices 0, 5, 9)
		a.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(5, [3]float32{5, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(9, [3]float32{9, 0, 0}, [3]float32{1, 1, 1})

		count := a.Flush(0, 1, 2)
		suite.Equal(uint32(3), count)

		writes := a.StagedWriteData()
		// 3 non-contiguous indices should produce 3 separate writes
		suite.Equal(3, len(writes))
	})

	suite.Run("reverse-order dirty instances are sorted before flush", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(100))
		for i := 0; i < 10; i++ {
			_, _ = a.AddInstance()
		}

		// Dirty instances in descending order so sortUint32 must actually swap
		a.SetInstanceTransform(9, [3]float32{9, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(5, [3]float32{5, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(0, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})

		count := a.Flush(0, 1, 2)
		suite.Equal(uint32(3), count)

		writes := a.StagedWriteData()
		suite.Equal(3, len(writes))
	})
}

type gpuTypesTest struct {
	suite.Suite
}

func TestGPUTypes(t *testing.T) {
	suite.Run(t, new(gpuTypesTest))
}

func (suite *gpuTypesTest) TestGPUInstanceData() {
	suite.Run("size is 64 bytes", func() {
		g := &animator.GPUInstanceData{}
		suite.Equal(64, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUInstanceData{}
		buf := g.Marshal()
		suite.Len(buf, 64)
	})

	suite.Run("marshal encodes identity matrix correctly", func() {
		g := &animator.GPUInstanceData{
			Model: [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
		}
		buf := g.Marshal()
		// Check diagonal elements are 1.0
		for _, idx := range []int{0, 5, 10, 15} {
			bits := binary.LittleEndian.Uint32(buf[idx*4 : (idx+1)*4])
			suite.InDelta(float32(1.0), math.Float32frombits(bits), 1e-6)
		}
		// Check off-diagonal element is 0.0
		bits := binary.LittleEndian.Uint32(buf[4:8])
		suite.InDelta(float32(0.0), math.Float32frombits(bits), 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUAnimationData() {
	suite.Run("size is 64 bytes", func() {
		g := &animator.GPUAnimationData{}
		suite.Equal(64, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUAnimationData{}
		buf := g.Marshal()
		suite.Len(buf, 64)
	})

	suite.Run("marshal encodes fields correctly", func() {
		g := &animator.GPUAnimationData{
			RotSpeed: [3]float32{0.1, 0.2, 0.3},
			Rot:      [3]float32{1.0, 2.0, 3.0},
			Pos:      [3]float32{10, 20, 30},
			Scale:    [3]float32{4, 5, 6},
		}
		buf := g.Marshal()
		// Check RotSpeed[0] at offset 0
		suite.InDelta(float32(0.1), math.Float32frombits(binary.LittleEndian.Uint32(buf[0:4])), 1e-6)
		// Check Pos[0] at offset 32
		suite.InDelta(float32(10), math.Float32frombits(binary.LittleEndian.Uint32(buf[32:36])), 1e-6)
		// Check Scale[2] at offset 56
		suite.InDelta(float32(6), math.Float32frombits(binary.LittleEndian.Uint32(buf[56:60])), 1e-6)
		// Check padding at offset 12 is zero
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

func (suite *gpuTypesTest) TestGPUFrustumPlane() {
	suite.Run("size is 16 bytes", func() {
		g := &animator.GPUFrustumPlane{}
		suite.Equal(16, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUFrustumPlane{}
		buf := g.Marshal()
		suite.Len(buf, 16)
	})

	suite.Run("marshal encodes normal and distance", func() {
		g := &animator.GPUFrustumPlane{
			Normal:   [3]float32{0, 1, 0},
			Distance: 5.5,
		}
		buf := g.Marshal()
		suite.InDelta(float32(0), math.Float32frombits(binary.LittleEndian.Uint32(buf[0:4])), 1e-6)
		suite.InDelta(float32(1), math.Float32frombits(binary.LittleEndian.Uint32(buf[4:8])), 1e-6)
		suite.InDelta(float32(0), math.Float32frombits(binary.LittleEndian.Uint32(buf[8:12])), 1e-6)
		suite.InDelta(float32(5.5), math.Float32frombits(binary.LittleEndian.Uint32(buf[12:16])), 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUGlobalData() {
	suite.Run("size is 112 bytes", func() {
		g := &animator.GPUGlobalData{}
		suite.Equal(112, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUGlobalData{}
		buf := g.Marshal()
		suite.Len(buf, 112)
	})

	suite.Run("marshal encodes fields correctly", func() {
		g := &animator.GPUGlobalData{
			InstanceCount:  42,
			DeltaTime:      0.016,
			BoundingRadius: 2.5,
		}
		buf := g.Marshal()
		suite.Equal(uint32(42), binary.LittleEndian.Uint32(buf[0:4]))
		suite.InDelta(float32(0.016), math.Float32frombits(binary.LittleEndian.Uint32(buf[4:8])), 1e-6)
		suite.InDelta(float32(2.5), math.Float32frombits(binary.LittleEndian.Uint32(buf[8:12])), 1e-6)
		// padding at offset 12
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

func (suite *gpuTypesTest) TestGPUAnimationGlobals() {
	suite.Run("size is 128 bytes", func() {
		g := &animator.GPUAnimationGlobals{}
		suite.Equal(128, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUAnimationGlobals{}
		buf := g.Marshal()
		suite.Len(buf, 128)
	})

	suite.Run("marshal encodes fields correctly", func() {
		g := &animator.GPUAnimationGlobals{
			InstanceCount:      10,
			BoneCount:          50,
			BoundingRadius:     1.5,
			ChannelDataOffset:  200,
			KeyframeDataOffset: 500,
		}
		buf := g.Marshal()
		suite.Equal(uint32(10), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(50), binary.LittleEndian.Uint32(buf[4:8]))
		suite.InDelta(float32(1.5), math.Float32frombits(binary.LittleEndian.Uint32(buf[8:12])), 1e-6)
		suite.Equal(uint32(200), binary.LittleEndian.Uint32(buf[12:16]))
		suite.Equal(uint32(500), binary.LittleEndian.Uint32(buf[16:20]))
	})
}

func (suite *gpuTypesTest) TestGPUIndirectArgs() {
	suite.Run("size is 20 bytes", func() {
		g := &animator.GPUIndirectArgs{}
		suite.Equal(20, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUIndirectArgs{}
		buf := g.Marshal()
		suite.Len(buf, 20)
	})

	suite.Run("marshal encodes all fields", func() {
		g := &animator.GPUIndirectArgs{
			IndexCount:    36,
			InstanceCount: 100,
			FirstIndex:    0,
			BaseVertex:    0,
			FirstInstance: 0,
		}
		buf := g.Marshal()
		suite.Equal(uint32(36), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(100), binary.LittleEndian.Uint32(buf[4:8]))
	})

	suite.Run("marshal encodes negative base vertex", func() {
		g := &animator.GPUIndirectArgs{
			BaseVertex: -5,
		}
		buf := g.Marshal()
		val := int32(binary.LittleEndian.Uint32(buf[12:16]))
		suite.Equal(int32(-5), val)
	})
}

func (suite *gpuTypesTest) TestGPUBoneInfo() {
	suite.Run("size is 112 bytes", func() {
		g := &animator.GPUBoneInfo{}
		suite.Equal(112, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUBoneInfo{}
		buf := g.Marshal()
		suite.Len(buf, 112)
	})

	suite.Run("marshal encodes bone data correctly", func() {
		g := &animator.GPUBoneInfo{
			InverseBindMatrix: [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
			LocalTranslation:  [3]float32{1, 2, 3},
			ParentIndex:       -1,
			LocalScale:        [3]float32{1, 1, 1},
			LocalRotation:     [4]float32{0, 0, 0, 1},
		}
		buf := g.Marshal()
		// Check InverseBindMatrix[0] = 1.0
		suite.InDelta(float32(1.0), math.Float32frombits(binary.LittleEndian.Uint32(buf[0:4])), 1e-6)
		// Check LocalTranslation[0] at offset 64
		suite.InDelta(float32(1.0), math.Float32frombits(binary.LittleEndian.Uint32(buf[64:68])), 1e-6)
		// Check ParentIndex at offset 76
		suite.Equal(int32(-1), int32(binary.LittleEndian.Uint32(buf[76:80])))
		// Check LocalRotation[3] (w) at offset 108
		suite.InDelta(float32(1.0), math.Float32frombits(binary.LittleEndian.Uint32(buf[108:112])), 1e-6)
	})
}

func (suite *gpuTypesTest) TestGPUKeyFrame() {
	suite.Run("size is 64 bytes", func() {
		g := &animator.GPUKeyFrame{}
		suite.Equal(64, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUKeyFrame{}
		buf := g.Marshal()
		suite.Len(buf, 64)
	})

	suite.Run("marshal encodes keyframe data", func() {
		g := &animator.GPUKeyFrame{
			Time:        0.5,
			Translation: [3]float32{1, 2, 3},
			Rotation:    [4]float32{0, 0, 0, 1},
			Scale:       [3]float32{1, 1, 1},
		}
		buf := g.Marshal()
		// Time at offset 0
		suite.InDelta(float32(0.5), math.Float32frombits(binary.LittleEndian.Uint32(buf[0:4])), 1e-6)
		// Translation[0] at offset 16
		suite.InDelta(float32(1.0), math.Float32frombits(binary.LittleEndian.Uint32(buf[16:20])), 1e-6)
		// Rotation[3] at offset 44
		suite.InDelta(float32(1.0), math.Float32frombits(binary.LittleEndian.Uint32(buf[44:48])), 1e-6)
		// Scale[0] at offset 48
		suite.InDelta(float32(1.0), math.Float32frombits(binary.LittleEndian.Uint32(buf[48:52])), 1e-6)
		// Padding at offset 4 is zero
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[4:8]))
	})
}

func (suite *gpuTypesTest) TestGPUChannelHeader() {
	suite.Run("size is 32 bytes", func() {
		g := &animator.GPUChannelHeader{}
		suite.Equal(32, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUChannelHeader{}
		buf := g.Marshal()
		suite.Len(buf, 32)
	})

	suite.Run("marshal encodes all channel fields", func() {
		g := &animator.GPUChannelHeader{
			BoneIndex:         3,
			PositionKeyOffset: 10,
			PositionKeyCount:  5,
			RotationKeyOffset: 15,
			RotationKeyCount:  5,
			ScaleKeyOffset:    20,
			ScaleKeyCount:     5,
		}
		buf := g.Marshal()
		suite.Equal(uint32(3), binary.LittleEndian.Uint32(buf[0:4]))
		suite.Equal(uint32(10), binary.LittleEndian.Uint32(buf[4:8]))
		suite.Equal(uint32(5), binary.LittleEndian.Uint32(buf[8:12]))
		// Padding at offset 28
		suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[28:32]))
	})
}

func (suite *gpuTypesTest) TestGPUClipHeader() {
	suite.Run("size is 16 bytes", func() {
		g := &animator.GPUClipHeader{}
		suite.Equal(16, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUClipHeader{}
		buf := g.Marshal()
		suite.Len(buf, 16)
	})

	suite.Run("marshal encodes clip header fields", func() {
		g := &animator.GPUClipHeader{
			Duration:       2.5,
			TicksPerSecond: 30.0,
			ChannelOffset:  4,
			ChannelCount:   2,
		}
		buf := g.Marshal()
		suite.InDelta(float32(2.5), math.Float32frombits(binary.LittleEndian.Uint32(buf[0:4])), 1e-6)
		suite.InDelta(float32(30.0), math.Float32frombits(binary.LittleEndian.Uint32(buf[4:8])), 1e-6)
		suite.Equal(uint32(4), binary.LittleEndian.Uint32(buf[8:12]))
		suite.Equal(uint32(2), binary.LittleEndian.Uint32(buf[12:16]))
	})
}

func (suite *gpuTypesTest) TestGPUSkeletalAnimationData() {
	suite.Run("size is 48 bytes", func() {
		g := &animator.GPUSkeletalAnimationData{}
		suite.Equal(48, g.Size())
	})

	suite.Run("marshal produces correct length", func() {
		g := &animator.GPUSkeletalAnimationData{}
		buf := g.Marshal()
		suite.Len(buf, 48)
	})

	suite.Run("marshal encodes animation state fields", func() {
		g := &animator.GPUSkeletalAnimationData{
			AnimationIndex:     2,
			AnimationTime:      1.5,
			BlendWeight:        0.7,
			SecondaryAnimIndex: 3,
			SecondaryAnimTime:  0.3,
		}
		buf := g.Marshal()
		suite.Equal(uint32(2), binary.LittleEndian.Uint32(buf[0:4]))
		suite.InDelta(float32(1.5), math.Float32frombits(binary.LittleEndian.Uint32(buf[4:8])), 1e-6)
		suite.InDelta(float32(0.7), math.Float32frombits(binary.LittleEndian.Uint32(buf[8:12])), 1e-6)
		suite.Equal(uint32(3), binary.LittleEndian.Uint32(buf[12:16]))
		suite.InDelta(float32(0.3), math.Float32frombits(binary.LittleEndian.Uint32(buf[16:20])), 1e-6)
		// Padding bytes should be zero
		for i := 20; i < 48; i += 4 {
			suite.Equal(uint32(0), binary.LittleEndian.Uint32(buf[i:i+4]))
		}
	})
}

func (suite *gpuTypesTest) TestGPUWGSLSources() {
	suite.Run("GPUInstanceDataSource is non-empty", func() {
		suite.NotEmpty(animator.GPUInstanceDataSource)
	})

	suite.Run("GPUAnimationDataSource is non-empty", func() {
		suite.NotEmpty(animator.GPUAnimationDataSource)
	})

	suite.Run("GPUFrustumPlaneSource is non-empty", func() {
		suite.NotEmpty(animator.GPUFrustumPlaneSource)
	})

	suite.Run("GPUGlobalDataSource is non-empty", func() {
		suite.NotEmpty(animator.GPUGlobalDataSource)
	})

	suite.Run("GPUAnimationGlobalsSource is non-empty", func() {
		suite.NotEmpty(animator.GPUAnimationGlobalsSource)
	})

	suite.Run("GPUIndirectArgsSource is non-empty", func() {
		suite.NotEmpty(animator.GPUIndirectArgsSource)
	})

	suite.Run("GPUBoneInfoSource is non-empty", func() {
		suite.NotEmpty(animator.GPUBoneInfoSource)
	})

	suite.Run("GPUSkeletalAnimationDataSource is non-empty", func() {
		suite.NotEmpty(animator.GPUSkeletalAnimationDataSource)
	})
}

// newNonSkinnedModel creates a mock model that is not skinned and has no skeleton.
func newNonSkinnedModel() *modelmocks.MockModel {
	m := &modelmocks.MockModel{}
	m.EXPECT().Skinned().Return(false).Maybe()
	m.EXPECT().Skeleton().Return(nil).Maybe()
	m.EXPECT().Animations().Return(nil).Maybe()
	return m
}

// newSkinnedModel creates a mock model with a skeleton and animation clips.
func newSkinnedModel() *modelmocks.MockModel {
	m := &modelmocks.MockModel{}

	skel := &model.Skeleton{
		Bones: []model.Bone{
			{
				Name:        "root",
				ParentIndex: -1,
				InverseBindMatrix: [16]float32{
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1,
				},
				LocalTransform: model.Transform{
					Translation: [3]float32{0, 0, 0},
					Rotation:    [4]float32{0, 0, 0, 1},
					Scale:       [3]float32{1, 1, 1},
				},
			},
			{
				Name:        "arm",
				ParentIndex: 0,
				InverseBindMatrix: [16]float32{
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
					0, 1, 0, 1,
				},
				LocalTransform: model.Transform{
					Translation: [3]float32{0, 1, 0},
					Rotation:    [4]float32{0, 0, 0, 1},
					Scale:       [3]float32{1, 1, 1},
				},
			},
		},
		RootBoneIndices: []int32{0},
		BoneNameToIndex: map[string]int32{"root": 0, "arm": 1},
	}

	clips := []*model.AnimationClip{
		{
			Name:           "idle",
			Duration:       2.0,
			TicksPerSecond: 30.0,
			Channels: []model.AnimationChannel{
				{
					BoneIndex: 0,
					PositionKeys: []model.VectorKeyframe{
						{Time: 0.0, Value: [3]float32{0, 0, 0}},
						{Time: 1.0, Value: [3]float32{0, 0.5, 0}},
					},
					RotationKeys: []model.QuaternionKeyframe{
						{Time: 0.0, Value: [4]float32{0, 0, 0, 1}},
					},
					ScaleKeys: []model.VectorKeyframe{
						{Time: 0.0, Value: [3]float32{1, 1, 1}},
					},
				},
			},
		},
	}

	m.EXPECT().Skinned().Return(true).Maybe()
	m.EXPECT().Skeleton().Return(skel).Maybe()
	m.EXPECT().Animations().Return(clips).Maybe()
	return m
}

func (suite *animatorTest) TestSetModelNonSkinned() {
	suite.Run("stores a non-skinned model without processing bones", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		m := newNonSkinnedModel()
		a.SetModel(m, 1, 2)
		suite.Equal(m, a.Model())
	})
}

func (suite *animatorTest) TestSetModelSkinned() {
	suite.Run("flattens skeleton and animation data into skeletal backend", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		m := newSkinnedModel()
		a.SetModel(m, 1, 2)

		suite.Equal(m, a.Model())
		// SetBoneCount + SetBone calls + AddClip should have staged writes
		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)
	})

	suite.Run("processes onto simple backend without panic for non-skinned", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		m := newNonSkinnedModel()
		a.SetModel(m, 1, 2)
		suite.Equal(m, a.Model())
	})

	suite.Run("processes skinned model onto simple backend without panic", func() {
		a := animator.NewAnimator(animator.BackendTypeSimple, animator.WithMaxInstances(10))
		m := newSkinnedModel()
		a.SetModel(m, 1, 2)
		suite.Equal(m, a.Model())
	})
}

func (suite *animatorTest) TestWithModel() {
	suite.Run("builder option sets model on skeletal backend", func() {
		m := newSkinnedModel()
		a := animator.NewAnimator(animator.BackendTypeSkeletal,
			animator.WithMaxInstances(10),
			animator.WithModel(m, 1, 2),
		)
		suite.Equal(m, a.Model())
	})

	suite.Run("builder option sets non-skinned model", func() {
		m := newNonSkinnedModel()
		a := animator.NewAnimator(animator.BackendTypeSimple,
			animator.WithModel(m, 1, 2),
		)
		suite.Equal(m, a.Model())
	})
}

func (suite *animatorTest) TestSkeletalSetInstanceDataFullDirtyTracking() {
	suite.Run("sets transform and rotation in one call with dirty tracking", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		_, _ = a.AddInstance()
		pos := [3]float32{5, 10, 15}
		scale := [3]float32{2, 2, 2}
		rotSpeed := [3]float32{0.1, 0.2, 0.3}
		rot := [3]float32{0.5, 0.6, 0.7}

		// First instance dirties the model matrix
		a.SetInstanceData(0, pos, scale, rotSpeed, rot)

		// Second instance expands the dirty range
		a.SetInstanceData(1, [3]float32{20, 30, 40}, [3]float32{3, 3, 3}, [3]float32{}, [3]float32{0, 0, 0})

		count := a.Flush(0, 1, 2)
		// model flush should have happened even though instance anim data wasn't dirty
		writes := a.StagedWriteData()
		suite.True(len(writes) > 0 || count > 0)
	})
}

func (suite *animatorTest) TestSkeletalSetInstanceRotationDirtyTracking() {
	suite.Run("rotation expands dirty range correctly", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		_, _ = a.AddInstance()
		_, _ = a.AddInstance()

		// Set transform to initialize model matrices
		a.SetInstanceTransform(0, [3]float32{1, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(2, [3]float32{3, 0, 0}, [3]float32{1, 1, 1})

		// Set rotation on index 1 to expand dirty range between 0 and 2
		a.SetInstanceRotation(1, [3]float32{}, [3]float32{0.5, 0.5, 0.5})

		count := a.Flush(0, 1, 2)
		// model data should be flushed
		writes := a.StagedWriteData()
		suite.True(len(writes) > 0 || count > 0)
	})
}

func (suite *animatorTest) TestSkeletalRemoveInstanceDirtyTracking() {
	suite.Run("swap-remove marks both instance and model data dirty", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance() // 0
		_, _ = a.AddInstance() // 1
		_, _ = a.AddInstance() // 2
		a.SetInstanceTransform(0, [3]float32{10, 0, 0}, [3]float32{1, 1, 1})
		a.SetInstanceTransform(1, [3]float32{20, 0, 0}, [3]float32{2, 2, 2})
		a.SetInstanceTransform(2, [3]float32{30, 0, 0}, [3]float32{3, 3, 3})
		a.Flush(0, 1, 2)
		_ = a.StagedWriteData()

		// Now remove index 0 (swaps with index 2)
		old, swapped := a.RemoveInstance(0)
		suite.True(swapped)
		suite.Equal(uint32(2), old)

		// The swap should have dirtied both instance data and model matrix data
		count := a.Flush(0, 1, 2)
		writes := a.StagedWriteData()
		suite.True(count > 0 || len(writes) > 0)
	})
}

func (suite *animatorTest) TestSkeletalInstanceTransformIdentity() {
	suite.Run("new instance starts with identity matrix position and unit scale", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()

		pos, scale := a.InstanceTransform(0)
		// Identity matrix: position = (0,0,0), scale = (1,1,1)
		suite.InDelta(float32(0), pos[0], 1e-5)
		suite.InDelta(float32(0), pos[1], 1e-5)
		suite.InDelta(float32(0), pos[2], 1e-5)
		suite.InDelta(float32(1), scale[0], 1e-5)
		suite.InDelta(float32(1), scale[1], 1e-5)
		suite.InDelta(float32(1), scale[2], 1e-5)
	})
}

func (suite *animatorTest) TestSkeletalInstanceRotationReturnsTrackedData() {
	suite.Run("returns stored rotation euler data", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetInstanceData(0, [3]float32{0, 0, 0}, [3]float32{1, 1, 1}, [3]float32{0.1, 0.2, 0.3}, [3]float32{0.4, 0.5, 0.6})

		_, rot := a.InstanceRotation(0)
		suite.InDelta(float32(0.4), rot[0], 1e-5)
		suite.InDelta(float32(0.5), rot[1], 1e-5)
		suite.InDelta(float32(0.6), rot[2], 1e-5)
	})

	suite.Run("returns zeros for unset instance", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		rotSpeed, rot := a.InstanceRotation(0)
		suite.Equal([3]float32{}, rotSpeed)
		suite.Equal([3]float32{}, rot)
	})
}

func (suite *animatorTest) TestSkeletalPrepareFrameBlendLooping() {
	suite.Run("blend-to animation wraps when looping", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(1)

		channels := []uint32{0, 0, 2, 0, 0, 0, 0}
		kf := []float32{0.0, 0.5}
		trans := [][3]float32{{0, 0, 0}, {1, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}, {0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}, {1, 1, 1}}
		a.AddClip(0.5, 1.0, channels, kf, trans, rots, scls, 2) // clip 0: 0.5s
		a.AddClip(0.5, 1.0, channels, kf, trans, rots, scls, 2) // clip 1: 0.5s
		_ = a.StagedWriteData()

		a.PlayAnimation(0, 0, true)
		a.BlendToAnimation(0, 1, 2.0) // long blend so it doesn't complete

		// Advance enough for both primary and secondary clips to exceed their duration
		a.PrepareFrame(0.8, 0) // 0.8 > 0.5 clip duration
		_ = a.StagedWriteData()

		suite.True(a.IsBlending(0))
	})
}

func (suite *animatorTest) TestIndirectBufferSkeletal() {
	suite.Run("returns nil when no buffer at binding", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		buf := a.IndirectBuffer(99)
		suite.Nil(buf)
	})
}

func (suite *animatorTest) TestSetBoneCountZeroSkeletal() {
	suite.Run("zero count does not panic", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetBoneCount(0)
	})
}

func (suite *animatorTest) TestInstanceRotationOutOfBoundsSkeletal() {
	suite.Run("returns zeros for out of bounds index", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		rotSpeed, rot := a.InstanceRotation(999)
		suite.Equal([3]float32{}, rotSpeed)
		suite.Equal([3]float32{}, rot)
	})
}

func (suite *animatorTest) TestInstanceTransformOutOfBoundsSkeletal() {
	suite.Run("returns zeros for out of bounds index", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		pos, scale := a.InstanceTransform(999)
		suite.Equal([3]float32{}, pos)
		suite.Equal([3]float32{}, scale)
	})
}

func (suite *animatorTest) TestSetInstanceRotationFirstDirtySkeletal() {
	suite.Run("sets model dirty when called without prior transform", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		// Call SetInstanceRotation directly without SetInstanceTransform
		// so modelDirty starts as false and the first-dirty branch executes
		a.SetInstanceRotation(0, [3]float32{0.1, 0.2, 0.3}, [3]float32{0.4, 0.5, 0.6})

		_, rot := a.InstanceRotation(0)
		suite.InDelta(float32(0.6), rot[2], 1e-5)
	})

	suite.Run("expands dirty range when called on multiple instances", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance() // 0
		_, _ = a.AddInstance() // 1
		_, _ = a.AddInstance() // 2
		// First call sets dirty range to [2,3)
		a.SetInstanceRotation(2, [3]float32{}, [3]float32{0.1, 0, 0})
		// Second call with lower index expands start
		a.SetInstanceRotation(0, [3]float32{}, [3]float32{0.2, 0, 0})
		// Third call with higher index expands end — already covered since 2 > 0+1

		_, rot0 := a.InstanceRotation(0)
		_, rot2 := a.InstanceRotation(2)
		suite.InDelta(float32(0.2), rot0[0], 1e-5)
		suite.InDelta(float32(0.1), rot2[0], 1e-5)
	})
}

func (suite *animatorTest) TestSetInstanceTransformDirtyExpansionSkeletal() {
	suite.Run("expands dirty range when setting high then low index", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance() // 0
		_, _ = a.AddInstance() // 1
		_, _ = a.AddInstance() // 2
		// Set high index first to establish dirty range at [2,3)
		a.SetInstanceTransform(2, [3]float32{30, 0, 0}, [3]float32{1, 1, 1})
		// Now set low index to trigger dirty range expansion at start
		a.SetInstanceTransform(0, [3]float32{10, 0, 0}, [3]float32{1, 1, 1})

		pos0, _ := a.InstanceTransform(0)
		pos2, _ := a.InstanceTransform(2)
		suite.InDelta(float32(10), pos0[0], 1e-5)
		suite.InDelta(float32(30), pos2[0], 1e-5)
	})
}

func (suite *animatorTest) TestSetInstanceDataOutOfBoundsSkeletal() {
	suite.Run("out of bounds index is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetInstanceData(999, [3]float32{}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{})
	})

	suite.Run("expands dirty range when setting high then low index", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance() // 0
		_, _ = a.AddInstance() // 1
		_, _ = a.AddInstance() // 2
		a.SetInstanceData(2, [3]float32{30, 0, 0}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{})
		a.SetInstanceData(0, [3]float32{10, 0, 0}, [3]float32{1, 1, 1}, [3]float32{}, [3]float32{})

		pos0, _ := a.InstanceTransform(0)
		pos2, _ := a.InstanceTransform(2)
		suite.InDelta(float32(10), pos0[0], 1e-5)
		suite.InDelta(float32(30), pos2[0], 1e-5)
	})
}

func (suite *animatorTest) TestRemoveInstanceDirtyExpansionSkeletal() {
	suite.Run("second swap-remove expands dirty start range", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance() // 0
		_, _ = a.AddInstance() // 1
		_, _ = a.AddInstance() // 2
		_, _ = a.AddInstance() // 3

		// Flush initial state so dirty flags reset
		a.Flush(0, 1, 2)
		_ = a.StagedWriteData()

		// First remove sets dirtyStart=2 (swaps 3 into 2, count=3)
		a.RemoveInstance(2)
		// Second remove with lower index expands dirtyStart (swaps 2 into 0, count=2)
		a.RemoveInstance(0)

		suite.Equal(uint32(2), a.InstanceCount())
	})
}

func (suite *animatorTest) TestPrepareFrameBlendCompletesSkeletal() {
	suite.Run("blend completes when progress reaches 1", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(1)

		channels := []uint32{0, 0, 2, 0, 0, 0, 0}
		kf := []float32{0.0, 0.5}
		trans := [][3]float32{{0, 0, 0}, {1, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}, {0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}, {1, 1, 1}}
		a.AddClip(1.0, 1.0, channels, kf, trans, rots, scls, 2) // clip 0
		a.AddClip(1.0, 1.0, channels, kf, trans, rots, scls, 2) // clip 1
		_ = a.StagedWriteData()

		a.PlayAnimation(0, 0, false)
		a.BlendToAnimation(0, 1, 0.01) // very short blend duration

		// Advance enough to complete the blend (deltaTime >> blendDuration)
		a.PrepareFrame(1.0, 0)
		_ = a.StagedWriteData()

		suite.False(a.IsBlending(0))
	})
}

func (suite *animatorTest) TestSetInstanceTransformOutOfBoundsSkeletal() {
	suite.Run("out of bounds index is no-op", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		a.SetInstanceTransform(999, [3]float32{1, 2, 3}, [3]float32{1, 1, 1})
	})
}

func (suite *animatorTest) TestPrepareFrameNonLoopingExceedsDurationSkeletal() {
	suite.Run("non-looping animation time exceeds duration without wrapping", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance()
		a.SetBoneCount(1)

		channels := []uint32{0, 0, 2, 0, 0, 0, 0}
		kf := []float32{0.0, 0.5}
		trans := [][3]float32{{0, 0, 0}, {1, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}, {0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}, {1, 1, 1}}
		a.AddClip(0.5, 1.0, channels, kf, trans, rots, scls, 2)
		_ = a.StagedWriteData()

		// Play non-looping, advance past clip duration
		a.PlayAnimation(0, 0, false)
		a.PrepareFrame(2.0, 0) // time=2.0 exceeds 0.5 duration, but no loop so no wrap

		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)
	})

	suite.Run("multiple instances expand dirty range in PrepareFrame", func() {
		a := animator.NewAnimator(animator.BackendTypeSkeletal, animator.WithMaxInstances(10))
		_, _ = a.AddInstance() // 0
		_, _ = a.AddInstance() // 1
		_, _ = a.AddInstance() // 2
		a.SetBoneCount(1)

		channels := []uint32{0, 0, 2, 0, 0, 0, 0}
		kf := []float32{0.0, 0.5}
		trans := [][3]float32{{0, 0, 0}, {1, 0, 0}}
		rots := [][4]float32{{0, 0, 0, 1}, {0, 0, 0, 1}}
		scls := [][3]float32{{1, 1, 1}, {1, 1, 1}}
		a.AddClip(0.5, 1.0, channels, kf, trans, rots, scls, 2)
		// Drain initial staged writes
		a.Flush(0, 1, 2)
		_ = a.StagedWriteData()

		// PrepareFrame with 3 instances exercises the dirty range expansion
		a.PrepareFrame(0.016, 0)

		writes := a.StagedWriteData()
		suite.True(len(writes) > 0)
	})
}
