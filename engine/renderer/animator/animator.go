package animator

import (
	"github.com/Carmen-Shannon/oxy-go/engine/model"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// animator is the implementation of the Animator interface.
type animator struct {
	backendType AnimatorBackendType
	backend     AnimatorBackend
	model       model.Model
}

// Animator defines the public interface for the animation system.
//
// The Animator manages per-instance animation state and stages GPU buffer writes each frame
// for processing by a compute shader. It delegates to an AnimatorBackend which provides
// the actual implementation for either simple instanced animation or skeletal bone animation.
//
// Methods specific to a particular backend type will no-op when called on an Animator
// using a different backend. The simple-only methods (SetInstanceTransform, SetInstanceRotation)
// no-op on skeletal backends, and the skeletal-only methods (SetBoneCount, SetBone, AddClip,
// PlayAnimation, BlendToAnimation, SetAnimationTime, SetAnimationSpeed, IsBlending,
// BlendProgress, CancelBlend) no-op on simple backends.
type Animator interface {
	// MaxInstances returns the maximum number of instances this animator can manage.
	//
	// Returns:
	//   - uint32: the maximum number of instances supported
	MaxInstances() uint32

	// ComputeBindGroupProvider returns the BindGroupProvider for the compute shader.
	// The provider holds layout info and GPU resources needed for the animation compute pass.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the compute shader BindGroupProvider
	ComputeBindGroupProvider() bind_group_provider.BindGroupProvider

	// OutputBindGroupProvider returns the BindGroupProvider for the vertex shader.
	// The provider holds the output buffer that the compute shader writes and the vertex shader reads.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the output BindGroupProvider
	OutputBindGroupProvider() bind_group_provider.BindGroupProvider

	// AddInstance registers a new instance with this animator.
	// If the current capacity is exceeded, the backend will automatically grow.
	//
	// Returns:
	//   - uint32: the index of the newly registered instance
	//   - error: an error if the instance could not be added
	AddInstance() (uint32, error)

	// TODO: experimental method for managing the Animator's instance size.
	// Grow increases the maximum instance capacity to newMax, preserving all existing data.
	// CPU-side slices are reallocated and MinBindingSizes updated; the needsRebuild flag is
	// set so the render thread recreates GPU buffers on the next frame.
	// No-op if newMax is less than or equal to the current capacity.
	//
	// Parameters:
	//   - newMax: the new maximum number of instances to support
	Grow(newMax uint32)

	// TODO: experimental method for managing the Animator's instance size.
	// RemoveInstance removes the instance at the given index using a swap-remove strategy.
	// Returns the old last index that was swapped and whether a swap occurred.
	//
	// Parameters:
	//   - index: the instance index to remove
	//
	// Returns:
	//   - uint32: the old last index that was swapped into the removed slot (only meaningful when bool is true)
	//   - bool: true if the last instance was swapped into the removed slot
	RemoveInstance(index uint32) (uint32, bool)

	// TODO: experimental method for managing the Animator's instance size.
	// NeedsRebuild reports whether GPU buffers need to be recreated after a Grow.
	//
	// Returns:
	//   - bool: true if a rebuild is pending
	NeedsRebuild() bool

	// TODO: experimental method for managing the Animator's instance size.
	// ClearNeedsRebuild resets the needsRebuild flag.
	ClearNeedsRebuild()

	// InstanceCount returns the current number of registered instances.
	//
	// Returns:
	//   - uint32: the number of active instances
	InstanceCount() uint32

	// StagedWriteData returns and clears the pending GPU buffer writes.
	// The Renderer should call this to drain staged writes and submit them via WriteBuffers.
	//
	// Returns:
	//   - []bind_group_provider.BufferWrite: the slice of pending buffer writes
	StagedWriteData() []bind_group_provider.BufferWrite

	// Flush stages the dirty instance data as GPU buffer writes.
	// Call this after modifying instance transforms, rotations, or animation state.
	//
	// Parameters:
	//   - instanceBinding: the bind group binding index for the compute shader's instance buffer, used for staging GPU writes
	//   - boneBinding: the bind group binding index for the compute shader's bone buffer, used for staging GPU writes
	//   - modelBinding: the bind group binding index for the compute shader's model buffer, used for staging GPU writes
	//
	// Returns:
	//   - uint32: the number of instances that were flushed
	Flush(instanceBinding, boneBinding, modelBinding int) uint32

	// PrepareFrame advances animation state by deltaTime and stages per-frame uniform data.
	// For skeletal backends this also advances playback time, handles looping, and resolves blends.
	// The binding parameter specifies which bind group index the per-frame data should be written to.
	//
	// Parameters:
	//   - deltaTime: elapsed time since the last frame in seconds
	//   - binding: the bind group index for per-frame uniform data in the compute shader, used for staging GPU writes
	PrepareFrame(deltaTime float32, binding int)

	// Release frees all GPU resources held by this animator and its providers.
	Release()

	// BackendType returns the type of backend this animator is using.
	//
	// Returns:
	//   - AnimatorBackendType: the backend type (BackendTypeSimple or BackendTypeSkeletal)
	BackendType() AnimatorBackendType

	// SetInstanceTransform sets the position and scale for a specific instance.
	// No-op on skeletal backends.
	//
	// Parameters:
	//   - index: the instance index to update
	//   - posXYZ: the position as [3]float32 (x, y, z)
	//   - scaleXYZ: the scale as [3]float32 (x, y, z)
	SetInstanceTransform(index uint32, posXYZ, scaleXYZ [3]float32)

	// SetInstanceRotation sets the rotation speed and current rotation for a specific instance.
	// No-op on skeletal backends.
	//
	// Parameters:
	//   - index: the instance index to update
	//   - rotSpeedXYZ: rotation speed in radians per frame around each axis as [3]float32
	//   - rotXYZ: current rotation angles around each axis as [3]float32
	SetInstanceRotation(index uint32, rotSpeedXYZ, rotXYZ [3]float32)

	// SetInstanceData sets all transform data for a specific instance in a single call,
	// combining SetInstanceTransform and SetInstanceRotation to reduce mutex overhead.
	// On skeletal backends, rotSpeedXYZ is ignored and a full model matrix is built from
	// position, scale, and rotation.
	//
	// Parameters:
	//   - index: the instance index to update
	//   - posXYZ: the position as [3]float32 (x, y, z)
	//   - scaleXYZ: the scale as [3]float32 (x, y, z)
	//   - rotSpeedXYZ: rotation speed in radians per frame around each axis as [3]float32
	//   - rotXYZ: current rotation angles around each axis as [3]float32
	SetInstanceData(index uint32, posXYZ, scaleXYZ, rotSpeedXYZ, rotXYZ [3]float32)

	// SetBoneCount allocates the bone slice for the skeleton.
	// Must be called before SetBone. No-op on simple backends.
	//
	// Parameters:
	//   - count: the number of bones in the skeleton
	SetBoneCount(count uint32)

	// SetBone sets bone data at the given index in the skeleton.
	// No-op on simple backends.
	//
	// Parameters:
	//   - index: the bone index
	//   - inverseBindMatrix: the 4x4 inverse bind matrix as 16 floats (column-major)
	//   - localTranslation: the bone's local translation as [3]float32
	//   - localRotation: the bone's local rotation quaternion as [4]float32 (x, y, z, w)
	//   - localScale: the bone's local scale as [3]float32
	//   - parentIndex: the parent bone index, or -1 for root bones
	//   - binding: the bind group index for the bone buffer uniform in the compute shader, used for staging GPU writes
	SetBone(index uint32, inverseBindMatrix [16]float32, localTranslation [3]float32, localRotation [4]float32, localScale [3]float32, parentIndex int32, binding int)

	// AddClip adds an animation clip from pre-flattened channel and keyframe data.
	// No-op on simple backends (returns 0).
	//
	// Parameters:
	//   - duration: the total clip duration in seconds
	//   - ticksPerSecond: the playback tick rate
	//   - channels: flat slice of channel data, 7 uint32 values per channel: [boneIndex, posKeyOffset, posKeyCount, rotKeyOffset, rotKeyCount, scaleKeyOffset, scaleKeyCount, ...]
	//   - keyframeTimes: time value for each keyframe
	//   - keyframeTranslations: translation per keyframe as [][3]float32
	//   - keyframeRotations: rotation per keyframe as [][4]float32
	//   - keyframeScales: scale per keyframe as [][3]float32
	//   - binding: the bind group index for clip data uniforms in the compute shader, used for staging GPU writes
	//
	// Returns:
	//   - uint32: the index of the added clip
	AddClip(duration, ticksPerSecond float32, channels []uint32, keyframeTimes []float32, keyframeTranslations [][3]float32, keyframeRotations [][4]float32, keyframeScales [][3]float32, binding int) uint32

	// PlayAnimation starts playback of an animation clip on a specific instance.
	// No-op on simple backends.
	//
	// Parameters:
	//   - instanceIndex: the instance to animate
	//   - clipIndex: the animation clip to play
	//   - loop: whether the animation should loop
	PlayAnimation(instanceIndex, clipIndex uint32, loop bool)

	// BlendToAnimation smoothly transitions an instance from its current animation to a new clip.
	// No-op on simple backends.
	//
	// Parameters:
	//   - instanceIndex: the instance to blend
	//   - targetClipIndex: the clip to blend to
	//   - blendDuration: the transition time in seconds
	BlendToAnimation(instanceIndex, targetClipIndex uint32, blendDuration float32)

	// SetAnimationTime sets the playback position for an instance.
	// No-op on simple backends.
	//
	// Parameters:
	//   - instanceIndex: the instance to update
	//   - time: the playback time in seconds
	SetAnimationTime(instanceIndex uint32, time float32)

	// SetAnimationSpeed sets the playback speed multiplier for an instance.
	// No-op on simple backends.
	//
	// Parameters:
	//   - instanceIndex: the instance to update
	//   - speed: the speed multiplier (1.0 = normal, 0.5 = half speed)
	SetAnimationSpeed(instanceIndex uint32, speed float32)

	// IsBlending returns whether an instance is currently blending between animations.
	// Returns false on simple backends.
	//
	// Parameters:
	//   - instanceIndex: the instance to check
	//
	// Returns:
	//   - bool: true if the instance is blending
	IsBlending(instanceIndex uint32) bool

	// BlendProgress returns the current blend progress for an instance.
	// Returns 0 on simple backends.
	//
	// Parameters:
	//   - instanceIndex: the instance to check
	//
	// Returns:
	//   - float32: blend progress from 0.0 (start) to 1.0 (complete)
	BlendProgress(instanceIndex uint32) float32

	// CancelBlend stops an in-progress blend and keeps the current primary animation.
	// No-op on simple backends.
	//
	// Parameters:
	//   - instanceIndex: the instance to cancel blending for
	CancelBlend(instanceIndex uint32)

	// Model retrieves the Model associated with this animator, or nil if not set.
	//
	// Returns:
	//   - model.Model: the associated model or nil
	Model() model.Model

	// SetModel assigns a Model and internalizes its skeleton and animation data into the backend.
	// For skinned models this calls SetBoneCount and SetBone for each bone, then AddClip for each
	// animation clip. For non-skinned models only the reference is stored.
	//
	// Parameters:
	//   - m: the Model to associate with this animator
	//   - boneBinding: the binding index for the bone data buffer in the compute shader
	//   - packedBinding: the binding index for the packed animation data buffer in the compute shader
	SetModel(m model.Model, boneBinding, packedBinding int)

	// SetFrustumPlanes updates the six frustum planes used for GPU frustum culling.
	// Calling this enables culling for the animator. Planes should be extracted from
	// the current view-projection matrix each frame.
	//
	// Parameters:
	//   - planes: the six frustum planes in GPU-aligned format
	SetFrustumPlanes(planes [6]GPUFrustumPlane)

	// SetBoundingRadius sets the object-space bounding sphere radius used for frustum culling.
	//
	// Parameters:
	//   - radius: the bounding sphere radius
	SetBoundingRadius(radius float32)

	// BoundingRadius returns the current bounding sphere radius used for frustum culling.
	//
	// Returns:
	//   - float32: the bounding sphere radius
	BoundingRadius() float32

	// IndirectBuffer returns the GPU buffer used for DrawIndexedIndirect arguments.
	// Returns nil when culling is not enabled or GPU resources are not initialized.
	//
	// Parameters:
	//   - binding: the bind group index for the indirect buffer
	//
	// Returns:
	//   - *wgpu.Buffer: the indirect draw arguments buffer, or nil
	IndirectBuffer(binding int) *wgpu.Buffer

	// CullingEnabled returns whether GPU frustum culling is active for this animator.
	//
	// Returns:
	//   - bool: true if frustum planes have been set and culling is active
	CullingEnabled() bool

	// ResetIndirectArgs stages a buffer write that zeros the indirect args instance count
	// before each compute dispatch, so the shader can atomically count visible instances.
	//
	// Parameters:
	//   - indexCount: the number of indices in the mesh's index buffer
	//  - binding: the bind group index for the indirect args buffer
	ResetIndirectArgs(indexCount uint32, binding int)

	// InstanceTransform returns the position and scale for a specific instance.
	// On skeletal backends the values are extracted from the instance's model matrix.
	//
	// Parameters:
	//   - index: the instance index to query
	//
	// Returns:
	//   - pos: the position as [3]float32 (x, y, z)
	//   - scale: the scale as [3]float32 (x, y, z)
	InstanceTransform(index uint32) (pos, scale [3]float32)

	// InstanceRotation returns the rotation speed and current rotation for a specific instance.
	// Returns zeros on skeletal backends where rotation is bone-driven.
	//
	// Parameters:
	//   - index: the instance index to query
	//
	// Returns:
	//   - rotSpeed: the rotation speed as [3]float32
	//   - rot: the current rotation as [3]float32
	InstanceRotation(index uint32) (rotSpeed, rot [3]float32)
}

var _ Animator = &animator{}

// NewAnimator creates a new Animator instance with the specified backend type.
// The backend is created based on the type and then configured using the provided options.
// Binding indices are configured via WithBinding options rather than fixed struct parameters,
// allowing any shader binding layout.
//
// Parameters:
//   - backendType: the type of animation backend to use (BackendTypeSimple or BackendTypeSkeletal)
//   - options: variadic list of AnimatorBuilderOption functions to configure the Animator
//
// Returns:
//   - Animator: a new instance of Animator configured with the specified backend and options
func NewAnimator(backendType AnimatorBackendType, options ...AnimatorBuilderOption) Animator {
	a := &animator{
		backendType: backendType,
	}
	switch backendType {
	case BackendTypeSkeletal:
		a.backend = newSkeletalAnimatorBackend()
	case BackendTypeSimple:
		fallthrough
	default:
		a.backend = newSimpleAnimatorBackend()
	}
	for _, opt := range options {
		opt(a)
	}
	return a
}

func (a *animator) MaxInstances() uint32 {
	return a.backend.MaxInstances()
}

func (a *animator) ComputeBindGroupProvider() bind_group_provider.BindGroupProvider {
	return a.backend.ComputeBindGroupProvider()
}

func (a *animator) OutputBindGroupProvider() bind_group_provider.BindGroupProvider {
	return a.backend.OutputBindGroupProvider()
}

func (a *animator) AddInstance() (uint32, error) {
	return a.backend.AddInstance()
}

func (a *animator) Grow(newMax uint32) {
	a.backend.Grow(newMax)
}

func (a *animator) RemoveInstance(index uint32) (uint32, bool) {
	return a.backend.RemoveInstance(index)
}

func (a *animator) NeedsRebuild() bool {
	return a.backend.NeedsRebuild()
}

func (a *animator) ClearNeedsRebuild() {
	a.backend.ClearNeedsRebuild()
}

func (a *animator) InstanceCount() uint32 {
	return a.backend.InstanceCount()
}

func (a *animator) StagedWriteData() []bind_group_provider.BufferWrite {
	return a.backend.StagedWriteData()
}

func (a *animator) SetInstanceTransform(index uint32, posXYZ, scaleXYZ [3]float32) {
	a.backend.SetInstanceTransform(index, posXYZ, scaleXYZ)
}

func (a *animator) SetInstanceRotation(index uint32, rotSpeedXYZ, rotXYZ [3]float32) {
	a.backend.SetInstanceRotation(index, rotSpeedXYZ, rotXYZ)
}

func (a *animator) SetInstanceData(index uint32, posXYZ, scaleXYZ, rotSpeedXYZ, rotXYZ [3]float32) {
	a.backend.SetInstanceData(index, posXYZ, scaleXYZ, rotSpeedXYZ, rotXYZ)
}

func (a *animator) Flush(instanceBinding, boneBinding, modelBinding int) uint32 {
	return a.backend.Flush(instanceBinding, boneBinding, modelBinding)
}

func (a *animator) PrepareFrame(deltaTime float32, binding int) {
	a.backend.PrepareFrame(deltaTime, binding)
}

func (a *animator) Release() {
	a.backend.Release()
}

func (a *animator) BackendType() AnimatorBackendType {
	return a.backendType
}

func (a *animator) SetBoneCount(count uint32) {
	a.backend.SetBoneCount(count)
}

func (a *animator) SetBone(index uint32, inverseBindMatrix [16]float32, localTranslation [3]float32, localRotation [4]float32, localScale [3]float32, parentIndex int32, binding int) {
	a.backend.SetBone(index, inverseBindMatrix, localTranslation, localRotation, localScale, parentIndex, binding)
}

func (a *animator) AddClip(duration, ticksPerSecond float32, channels []uint32, keyframeTimes []float32, keyframeTranslations [][3]float32, keyframeRotations [][4]float32, keyframeScales [][3]float32, binding int) uint32 {
	return a.backend.AddClip(duration, ticksPerSecond, channels, keyframeTimes, keyframeTranslations, keyframeRotations, keyframeScales, binding)
}

func (a *animator) PlayAnimation(instanceIndex, clipIndex uint32, loop bool) {
	a.backend.PlayAnimation(instanceIndex, clipIndex, loop)
}

func (a *animator) BlendToAnimation(instanceIndex, targetClipIndex uint32, blendDuration float32) {
	a.backend.BlendToAnimation(instanceIndex, targetClipIndex, blendDuration)
}

func (a *animator) SetAnimationTime(instanceIndex uint32, time float32) {
	a.backend.SetAnimationTime(instanceIndex, time)
}

func (a *animator) SetAnimationSpeed(instanceIndex uint32, speed float32) {
	a.backend.SetAnimationSpeed(instanceIndex, speed)
}

func (a *animator) IsBlending(instanceIndex uint32) bool {
	return a.backend.IsBlending(instanceIndex)
}

func (a *animator) BlendProgress(instanceIndex uint32) float32 {
	return a.backend.BlendProgress(instanceIndex)
}

func (a *animator) CancelBlend(instanceIndex uint32) {
	a.backend.CancelBlend(instanceIndex)
}

func (a *animator) Model() model.Model {
	return a.model
}

func (a *animator) SetModel(m model.Model, boneBinding, packedBinding int) {
	a.model = m

	if !m.Skinned() || m.Skeleton() == nil {
		return
	}

	// Flatten skeleton into backend bone data
	skel := m.Skeleton()
	a.SetBoneCount(uint32(len(skel.Bones)))
	for i, bone := range skel.Bones {
		a.SetBone(
			uint32(i),
			bone.InverseBindMatrix,
			bone.LocalTransform.Translation,
			bone.LocalTransform.Rotation,
			bone.LocalTransform.Scale,
			bone.ParentIndex,
			boneBinding,
		)
	}

	// Flatten each animation clip into the backend's flat format
	for _, clip := range m.Animations() {
		var channels []uint32
		var times []float32
		var translations [][3]float32
		var rotations [][4]float32
		var scales [][3]float32

		for _, ch := range clip.Channels {
			posOff := uint32(len(times))
			posCnt := uint32(len(ch.PositionKeys))
			for _, k := range ch.PositionKeys {
				times = append(times, k.Time)
				translations = append(translations, k.Value)
				rotations = append(rotations, [4]float32{})
				scales = append(scales, [3]float32{1, 1, 1})
			}

			rotOff := uint32(len(times))
			rotCnt := uint32(len(ch.RotationKeys))
			for _, k := range ch.RotationKeys {
				times = append(times, k.Time)
				translations = append(translations, [3]float32{})
				rotations = append(rotations, k.Value)
				scales = append(scales, [3]float32{1, 1, 1})
			}

			scaleOff := uint32(len(times))
			scaleCnt := uint32(len(ch.ScaleKeys))
			for _, k := range ch.ScaleKeys {
				times = append(times, k.Time)
				translations = append(translations, [3]float32{})
				rotations = append(rotations, [4]float32{})
				scales = append(scales, k.Value)
			}

			channels = append(channels,
				uint32(ch.BoneIndex),
				posOff, posCnt,
				rotOff, rotCnt,
				scaleOff, scaleCnt,
			)
		}

		a.AddClip(clip.Duration, clip.TicksPerSecond, channels, times, translations, rotations, scales, packedBinding)
	}
}

func (a *animator) SetFrustumPlanes(planes [6]GPUFrustumPlane) {
	a.backend.SetFrustumPlanes(planes)
}

func (a *animator) SetBoundingRadius(radius float32) {
	a.backend.SetBoundingRadius(radius)
}

func (a *animator) BoundingRadius() float32 {
	return a.backend.BoundingRadius()
}

func (a *animator) IndirectBuffer(binding int) *wgpu.Buffer {
	return a.backend.IndirectBuffer(binding)
}

func (a *animator) CullingEnabled() bool {
	return a.backend.CullingEnabled()
}

func (a *animator) ResetIndirectArgs(indexCount uint32, binding int) {
	a.backend.ResetIndirectArgs(indexCount, binding)
}

func (a *animator) InstanceTransform(index uint32) (pos, scale [3]float32) {
	return a.backend.InstanceTransform(index)
}

func (a *animator) InstanceRotation(index uint32) (rotSpeed, rot [3]float32) {
	return a.backend.InstanceRotation(index)
}
