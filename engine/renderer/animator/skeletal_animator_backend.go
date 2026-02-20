package animator

import (
	"math"
	"sync"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// skeletalInstanceState holds the CPU-side animation playback state for a single skeletal instance.
// This is not a GPU struct; it tracks playback time, speed, looping, and blend state that
// PrepareFrame uses to compute the GPU-facing skeletalAnimationData each frame.
type skeletalInstanceState struct {
	clipIndex uint32

	time, speed                 float32
	loop, blending              bool
	blendFrom, blendTo          uint32
	blendFromTime, blendToTime  float32
	blendDuration, blendElapsed float32
}

// skeletalAnimatorBackendImpl is the concrete implementation of the skeletal animator backend.
// It manages bone data, animation clips, per-instance playback state, and GPU buffer staging
// for the skeletal animation compute shader.
type skeletalAnimatorBackendImpl struct {
	mu *sync.Mutex

	computeProvider, outputProvider bind_group_provider.BindGroupProvider

	maxInstances, instanceCount, boneCount uint32
	channelDataOffset, keyframeDataOffset  uint32

	stagedWriteData []bind_group_provider.BufferWrite

	instanceData []GPUSkeletalAnimationData

	dirty, boneDirty, modelDirty, needsRebuild, cullingEnabled bool
	dirtyStart, dirtyEnd                                       uint32

	instanceStateData []skeletalInstanceState

	perFrameSlice []GPUAnimationGlobals

	bones          []GPUBoneInfo
	clipHeaders    []GPUClipHeader
	channelHeaders []GPUChannelHeader
	keyFrames      []GPUKeyFrame

	// Per-instance model matrices (flat float32, 16 per instance) for world transform
	instanceModelData              []float32
	modelDirtyStart, modelDirtyEnd uint32

	// Per-instance rotation tracking (CPU-side only, not uploaded to GPU).
	// Stored so InstanceRotation can return the values set via SetInstanceRotation/SetInstanceData.
	instanceRotSpeedData [][3]float32
	instanceRotEulerData [][3]float32

	// Reusable staging buffers to avoid per-frame heap allocations.
	// wgpu's queue.WriteBuffer copies data internally before returning,
	// so a single buffer reused every frame is safe.
	stagingInstance, stagingBones, stagingModel, stagingUniform, stagingIndirectRst []byte

	// Frustum culling state
	frustumPlanes  [6]GPUFrustumPlane
	boundingRadius float32
}

// skeletalAnimatorBackend defines the interface for the skeletal animation backend.
// It provides bone data management, animation clip storage, per-instance playback control,
// and blend transitions. Methods shared with simpleAnimatorBackend (transforms, culling,
// lifecycle, capacity management) are inherited through AnimatorBackend and not repeated here.
type skeletalAnimatorBackend interface {
	// BoneCount returns the number of bones in the skeleton. This is used to determine the size of the bone buffer and should be set before adding bone data.
	//
	// Returns:
	//   - uint32: the number of bones in the skeleton
	BoneCount() uint32

	// SetBone sets the bone data at the given index in the skeleton.
	// All parameters are primitives matching the GPU BoneInfo layout.
	//
	// Parameters:
	//   - index: the bone index in the skeleton
	//   - inverseBindMatrix: the 4x4 inverse bind matrix as 16 floats (column-major)
	//   - localTranslation: the local translation of the bone as [3]float32
	//   - localRotation: the local rotation quaternion as [4]float32 (x, y, z, w)
	//   - localScale: the local scale of the bone as [3]float32
	//   - parentIndex: the index of the parent bone, or -1 for root bones
	//  - binding: the bind group index for the bone buffer uniform in the compute shader, used for staging GPU writes
	SetBone(index uint32, inverseBindMatrix [16]float32, localTranslation [3]float32, localRotation [4]float32, localScale [3]float32, parentIndex int32, binding int)

	// SetBoneCount sets the total number of bones in the skeleton.
	// This must be called before SetBone to allocate the bone slice.
	// It stages a full bone buffer write on the next Flush.
	//
	// Parameters:
	//   - count: the number of bones in the skeleton
	SetBoneCount(count uint32)

	// AddClip adds an animation clip from pre-flattened channel and keyframe data.
	// The caller is responsible for flattening their own animation data into
	// channelCount sets of (boneIndex, posKeyOffset, posKeyCount, rotKeyOffset, rotKeyCount, scaleKeyOffset, scaleKeyCount)
	// packed into the channels slice, and flattened keyframes into the frames slice.
	//
	// Parameters:
	//   - duration: the total clip duration in seconds
	//   - ticksPerSecond: the playback tick rate
	//   - channels: flat slice of channel data, 7 uint32 values per channel: [boneIndex, posKeyOffset, posKeyCount, rotKeyOffset, rotKeyCount, scaleKeyOffset, scaleKeyCount, ...]
	//   - keyframeTimes: time value for each keyframe
	//   - keyframeTranslations: translation per keyframe as [][3]float32
	//   - keyframeRotations: rotation per keyframe as [][4]float32
	//   - keyframeScales: scale per keyframe as [][3]float32
	//  - binding: the bind group index for clip data uniforms
	//
	// Returns:
	//   - uint32: the index of the added clip
	AddClip(duration, ticksPerSecond float32, channels []uint32, keyframeTimes []float32, keyframeTranslations [][3]float32, keyframeRotations [][4]float32, keyframeScales [][3]float32, binding int) uint32

	// PlayAnimation starts an animation on a specific instance.
	//
	// Parameters:
	//   - instanceIndex: the index of the instance to animate
	//   - clipIndex: the index of the animation clip to play
	//   - loop: whether the animation should loop
	PlayAnimation(instanceIndex, clipIndex uint32, loop bool)

	// BlendToAnimation smoothly transitions an instance to a new animation clip.
	//
	// Parameters:
	//   - instanceIndex: the index of the instance
	//   - targetClipIndex: the animation clip to blend to
	//   - blendDuration: the time in seconds for the blend transition
	BlendToAnimation(instanceIndex, targetClipIndex uint32, blendDuration float32)

	// SetAnimationTime sets the playback position for an instance.
	//
	// Parameters:
	//   - instanceIndex: the index of the instance
	//   - time: the playback time in seconds
	SetAnimationTime(instanceIndex uint32, time float32)

	// SetAnimationSpeed sets the playback speed multiplier for an instance.
	//
	// Parameters:
	//   - instanceIndex: the index of the instance
	//   - speed: the speed multiplier (1.0 = normal, 0.5 = half speed)
	SetAnimationSpeed(instanceIndex uint32, speed float32)

	// IsBlending returns whether an instance is currently blending between animations.
	//
	// Parameters:
	//   - instanceIndex: the index of the instance to check
	//
	// Returns:
	//   - bool: true if the instance is currently blending
	IsBlending(instanceIndex uint32) bool

	// BlendProgress returns the current blend progress for an instance.
	//
	// Parameters:
	//   - instanceIndex: the index of the instance to check
	//
	// Returns:
	//   - float32: blend progress from 0.0 (start) to 1.0 (complete), or 0.0 if not blending
	BlendProgress(instanceIndex uint32) float32

	// CancelBlend stops an in-progress blend and keeps the current primary animation.
	//
	// Parameters:
	//   - instanceIndex: the index of the instance to cancel blending for
	CancelBlend(instanceIndex uint32)
}

var _ AnimatorBackend = &skeletalAnimatorBackendImpl{}

// newSkeletalAnimatorBackend creates and initializes a new instance of the skeletal animator backend.
// It allocates instance data, playback state, and per-frame uniform slices using the default max instance count,
// and configures the compute and output BindGroupProviders with the skeletal animation buffer layouts.
//
// Returns:
//   - AnimatorBackend: a new instance of the skeletal animator backend
func newSkeletalAnimatorBackend() AnimatorBackend {
	s := &skeletalAnimatorBackendImpl{
		mu:           &sync.Mutex{},
		maxInstances: 200, // TODO: investigate a good default based on performance tests
	}
	s.instanceData = make([]GPUSkeletalAnimationData, s.maxInstances)
	s.instanceStateData = make([]skeletalInstanceState, s.maxInstances)
	s.perFrameSlice = make([]GPUAnimationGlobals, 1)
	s.instanceRotSpeedData = make([][3]float32, s.maxInstances)
	s.instanceRotEulerData = make([][3]float32, s.maxInstances)

	// Initialize instance model matrices to identity
	s.instanceModelData = make([]float32, s.maxInstances*16)
	for i := uint32(0); i < s.maxInstances; i++ {
		common.Identity(s.instanceModelData[i*16 : (i+1)*16])
	}

	s.computeProvider = bind_group_provider.NewBindGroupProvider("skeletal_animator_compute")
	s.outputProvider = bind_group_provider.NewBindGroupProvider("skeletal_animator_output")
	s.stagedWriteData = make([]bind_group_provider.BufferWrite, 0, 8)
	s.initStagingPool()

	return s
}

// initStagingPool allocates (or reallocates) the triple-buffered staging byte slices
// sized to the current maxInstances and boneCount. Called at init time and after
// Grow/SetMaxInstances/SetBoneCount.
func (s *skeletalAnimatorBackendImpl) initStagingPool() {
	instanceBytes := int(s.maxInstances) * (&GPUSkeletalAnimationData{}).Size()
	boneBytes := int(s.boneCount) * (&GPUBoneInfo{}).Size()
	if boneBytes < 1 {
		boneBytes = 1
	}
	modelBytes := int(s.maxInstances) * 64
	uniformBytes := (&GPUAnimationGlobals{}).Size()
	s.stagingInstance = make([]byte, instanceBytes)
	s.stagingBones = make([]byte, boneBytes)
	s.stagingModel = make([]byte, modelBytes)
	s.stagingUniform = make([]byte, uniformBytes)
	s.stagingIndirectRst = make([]byte, (&GPUIndirectArgs{}).Size())
}

func (s *skeletalAnimatorBackendImpl) AddInstance() (uint32, error) {
	s.mu.Lock()
	if s.instanceCount >= s.maxInstances {
		// Auto-grow: double capacity (minimum 8). Unlock first because Grow acquires its own lock.
		newCap := s.maxInstances * 2
		if newCap < 8 {
			newCap = 8
		}
		s.mu.Unlock()
		s.Grow(newCap)
		s.mu.Lock()
	}
	idx := s.instanceCount
	s.instanceCount++
	s.mu.Unlock()
	return idx, nil
}

// TODO: experimental method for managing the Animator's instance size.
// RemoveInstance removes the instance at the given index using a swap-remove strategy.
// The last instance's animation data, playback state, and model matrix are copied into the
// removed slot and instanceCount is decremented. Returns true if a swap was performed
// (i.e. the removed index was not the last element), meaning the caller must update the
// swapped GameObject's stored instance index to `index`. Returns false if no swap occurred.
//
// Parameters:
//   - index: the instance index to remove
//
// Returns:
//   - uint32: the old last index that was swapped into the removed slot (only meaningful when bool is true)
//   - bool: true if the last instance was swapped into the removed slot
func (s *skeletalAnimatorBackendImpl) RemoveInstance(index uint32) (uint32, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.instanceCount == 0 || index >= s.instanceCount {
		return 0, false
	}

	last := s.instanceCount - 1
	swapped := index != last

	if swapped {
		// Copy last instance animation data into the removed slot
		s.instanceData[index] = s.instanceData[last]

		// Copy last instance playback state
		s.instanceStateData[index] = s.instanceStateData[last]

		// Copy last instance model matrix (16 floats)
		copy(s.instanceModelData[index*16:(index+1)*16], s.instanceModelData[last*16:(last+1)*16])

		// Copy last instance rotation tracking data
		s.instanceRotSpeedData[index] = s.instanceRotSpeedData[last]
		s.instanceRotEulerData[index] = s.instanceRotEulerData[last]

		// Mark the swapped slot dirty for instance data
		if !s.dirty {
			s.dirtyStart = index
			s.dirtyEnd = index + 1
			s.dirty = true
		} else {
			if index < s.dirtyStart {
				s.dirtyStart = index
			}
			if index+1 > s.dirtyEnd {
				s.dirtyEnd = index + 1
			}
		}

		// Mark the swapped slot dirty for model matrix
		if !s.modelDirty {
			s.modelDirtyStart = index
			s.modelDirtyEnd = index + 1
			s.modelDirty = true
		} else {
			if index < s.modelDirtyStart {
				s.modelDirtyStart = index
			}
			if index+1 > s.modelDirtyEnd {
				s.modelDirtyEnd = index + 1
			}
		}
	}

	// Zero out the now-unused last slot and decrement
	s.instanceData[last] = GPUSkeletalAnimationData{}
	s.instanceStateData[last] = skeletalInstanceState{}
	common.Identity(s.instanceModelData[last*16 : (last+1)*16])
	s.instanceRotSpeedData[last] = [3]float32{}
	s.instanceRotEulerData[last] = [3]float32{}
	s.instanceCount--

	return last, swapped
}

func (s *skeletalAnimatorBackendImpl) SetComputeBindGroupProvider(provider bind_group_provider.BindGroupProvider) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.computeProvider = provider
}

func (s *skeletalAnimatorBackendImpl) SetOutputBindGroupProvider(provider bind_group_provider.BindGroupProvider) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.outputProvider = provider
}

func (s *skeletalAnimatorBackendImpl) InstanceCount() uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.instanceCount
}

func (s *skeletalAnimatorBackendImpl) MaxInstances() uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxInstances
}

func (s *skeletalAnimatorBackendImpl) ComputeBindGroupProvider() bind_group_provider.BindGroupProvider {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.computeProvider
}

func (s *skeletalAnimatorBackendImpl) OutputBindGroupProvider() bind_group_provider.BindGroupProvider {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.outputProvider
}

func (s *skeletalAnimatorBackendImpl) StagedWriteData() []bind_group_provider.BufferWrite {
	s.mu.Lock()
	defer s.mu.Unlock()
	w := s.stagedWriteData
	s.stagedWriteData = s.stagedWriteData[:0]
	return w
}

func (s *skeletalAnimatorBackendImpl) SetMaxInstances(maxInstances uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.maxInstances = maxInstances
	s.instanceData = make([]GPUSkeletalAnimationData, s.maxInstances)
	s.instanceStateData = make([]skeletalInstanceState, s.maxInstances)
	s.instanceRotSpeedData = make([][3]float32, s.maxInstances)
	s.instanceRotEulerData = make([][3]float32, s.maxInstances)
	s.instanceCount = 0
	s.dirty = false
	s.dirtyStart = 0
	s.dirtyEnd = 0

	// Reinitialize instance model matrices to identity
	s.instanceModelData = make([]float32, s.maxInstances*16)
	for i := uint32(0); i < s.maxInstances; i++ {
		common.Identity(s.instanceModelData[i*16 : (i+1)*16])
	}
	s.modelDirty = false

	s.initStagingPool()
}

func (s *skeletalAnimatorBackendImpl) Flush(instanceBinding, boneBinding, modelBinding int) uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.needsRebuild {
		return 0
	}

	if !s.dirty && !s.boneDirty && !s.modelDirty {
		return 0
	}

	var count uint32

	if s.dirty {
		count = s.dirtyEnd - s.dirtyStart
		offset := uint64(s.dirtyStart) * uint64((&GPUSkeletalAnimationData{}).Size())

		dirty := s.instanceData[s.dirtyStart:s.dirtyEnd]
		raw := common.SliceToBytes(dirty)
		buf := s.stagingInstance[:len(raw)]
		copy(buf, raw)

		s.stagedWriteData = append(s.stagedWriteData, bind_group_provider.BufferWrite{
			Provider: s.computeProvider,
			Binding:  instanceBinding,
			Offset:   offset,
			Data:     buf,
		})

		s.dirty = false
		s.dirtyStart = 0
		s.dirtyEnd = 0
	}

	if s.boneDirty {
		raw := common.SliceToBytes(s.bones)
		buf := s.stagingBones[:len(raw)]
		copy(buf, raw)

		s.stagedWriteData = append(s.stagedWriteData, bind_group_provider.BufferWrite{
			Provider: s.computeProvider,
			Binding:  boneBinding,
			Offset:   0,
			Data:     buf,
		})

		s.boneDirty = false
	}

	if s.modelDirty {
		startBytes := uint64(s.modelDirtyStart) * 64
		dirtySlice := s.instanceModelData[s.modelDirtyStart*16 : s.modelDirtyEnd*16]
		raw := common.SliceToBytes(dirtySlice)
		buf := s.stagingModel[:len(raw)]
		copy(buf, raw)

		s.stagedWriteData = append(s.stagedWriteData, bind_group_provider.BufferWrite{
			Provider: s.computeProvider,
			Binding:  modelBinding,
			Offset:   startBytes,
			Data:     buf,
		})

		s.modelDirty = false
		s.modelDirtyStart = 0
		s.modelDirtyEnd = 0
	}

	return count
}

func (s *skeletalAnimatorBackendImpl) PrepareFrame(deltaTime float32, binding int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.needsRebuild {
		return
	}

	// wgpu copies buffer data internally, so reusing the same staging
	// buffer each frame is safe — no rotation needed.

	for i := uint32(0); i < s.instanceCount; i++ {
		state := &s.instanceStateData[i]

		state.time += deltaTime * state.speed

		if state.loop && state.clipIndex < uint32(len(s.clipHeaders)) {
			duration := s.clipHeaders[state.clipIndex].Duration
			if duration > 0 && state.time > duration {
				state.time = float32(math.Mod(float64(state.time), float64(duration)))
			}
		}

		if state.blending {
			state.blendElapsed += deltaTime
			state.blendToTime += deltaTime * state.speed

			if state.loop && state.blendTo < uint32(len(s.clipHeaders)) {
				duration := s.clipHeaders[state.blendTo].Duration
				if duration > 0 && state.blendToTime > duration {
					state.blendToTime = float32(math.Mod(float64(state.blendToTime), float64(duration)))
				}
			}

			progress := state.blendElapsed / state.blendDuration
			if progress >= 1.0 {
				state.clipIndex = state.blendTo
				state.time = state.blendToTime
				state.blending = false
				state.blendElapsed = 0
				progress = 0
			}

			s.instanceData[i] = GPUSkeletalAnimationData{
				AnimationIndex:     state.clipIndex,
				AnimationTime:      state.time,
				BlendWeight:        progress,
				SecondaryAnimIndex: state.blendTo,
				SecondaryAnimTime:  state.blendToTime,
			}
		} else {
			s.instanceData[i] = GPUSkeletalAnimationData{
				AnimationIndex: state.clipIndex,
				AnimationTime:  state.time,
			}
		}

		if !s.dirty {
			s.dirtyStart = i
			s.dirtyEnd = i + 1
			s.dirty = true
		} else {
			if i < s.dirtyStart {
				s.dirtyStart = i
			}
			if i+1 > s.dirtyEnd {
				s.dirtyEnd = i + 1
			}
		}
	}

	s.perFrameSlice[0] = GPUAnimationGlobals{
		InstanceCount:      s.instanceCount,
		BoneCount:          s.boneCount,
		BoundingRadius:     s.boundingRadius,
		ChannelDataOffset:  s.channelDataOffset,
		KeyframeDataOffset: s.keyframeDataOffset,
		Planes:             s.frustumPlanes,
	}

	raw := common.SliceToBytes(s.perFrameSlice)
	buf := s.stagingUniform[:len(raw)]
	copy(buf, raw)

	s.stagedWriteData = append(s.stagedWriteData, bind_group_provider.BufferWrite{
		Provider: s.computeProvider,
		Binding:  binding,
		Offset:   0,
		Data:     buf,
	})
}

func (s *skeletalAnimatorBackendImpl) SetBoneCount(count uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.boneCount = count
	s.bones = make([]GPUBoneInfo, count)
	s.boneDirty = true

	// Reallocate bone staging buffers to match new boneCount
	boneBytes := int(count) * (&GPUBoneInfo{}).Size()
	if boneBytes < 1 {
		boneBytes = 1
	}
	s.stagingBones = make([]byte, boneBytes)
}

func (s *skeletalAnimatorBackendImpl) SetBone(index uint32, inverseBindMatrix [16]float32, localTranslation [3]float32, localRotation [4]float32, localScale [3]float32, parentIndex int32, binding int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.boneCount {
		return
	}
	s.bones[index] = GPUBoneInfo{
		InverseBindMatrix: inverseBindMatrix,
		LocalTranslation:  localTranslation,
		ParentIndex:       parentIndex,
		LocalScale:        localScale,
		LocalRotation:     localRotation,
	}
	s.boneDirty = true

	// Stage full bone buffer write
	raw := common.SliceToBytes(s.bones)
	snapshot := make([]byte, len(raw))
	copy(snapshot, raw)
	s.stagedWriteData = append(s.stagedWriteData, bind_group_provider.BufferWrite{
		Provider: s.computeProvider,
		Binding:  binding,
		Offset:   0,
		Data:     snapshot,
	})
}

func (s *skeletalAnimatorBackendImpl) AddClip(duration, ticksPerSecond float32, channels []uint32, keyframeTimes []float32, keyframeTranslations [][3]float32, keyframeRotations [][4]float32, keyframeScales [][3]float32, binding int) uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()

	channelCount := uint32(len(channels) / 7)
	channelOffset := uint32(len(s.channelHeaders))
	keyframeOffset := uint32(len(s.keyFrames))

	// Append clip header
	clipIndex := uint32(len(s.clipHeaders))
	s.clipHeaders = append(s.clipHeaders, GPUClipHeader{
		Duration:       duration,
		TicksPerSecond: ticksPerSecond,
		ChannelOffset:  channelOffset,
		ChannelCount:   channelCount,
	})

	// Unpack channels — 7 uint32 per channel
	for i := uint32(0); i < channelCount; i++ {
		base := i * 7
		s.channelHeaders = append(s.channelHeaders, GPUChannelHeader{
			BoneIndex:         channels[base+0],
			PositionKeyOffset: channels[base+1] + keyframeOffset,
			PositionKeyCount:  channels[base+2],
			RotationKeyOffset: channels[base+3] + keyframeOffset,
			RotationKeyCount:  channels[base+4],
			ScaleKeyOffset:    channels[base+5] + keyframeOffset,
			ScaleKeyCount:     channels[base+6],
		})
	}

	// Pack keyframes
	for i := range keyframeTimes {
		s.keyFrames = append(s.keyFrames, GPUKeyFrame{
			Time:        keyframeTimes[i],
			Translation: keyframeTranslations[i],
			Rotation:    keyframeRotations[i],
			Scale:       keyframeScales[i],
		})
	}

	// Recompute packed buffer offsets (in u32 units)
	// Layout: [clips (4 u32 each)] [channels (8 u32 each)] [keyframes (16 u32 each)]
	clipU32Count := uint32(len(s.clipHeaders)) * 4
	channelU32Count := uint32(len(s.channelHeaders)) * 8
	keyframeU32Count := uint32(len(s.keyFrames)) * 16
	s.channelDataOffset = clipU32Count
	s.keyframeDataOffset = clipU32Count + channelU32Count

	totalU32s := clipU32Count + channelU32Count + keyframeU32Count

	// Build the packed u32 buffer
	packed := make([]uint32, totalU32s)

	// Pack clip headers: [duration_bits, tps_bits, channelOffset, channelCount]
	for i, ch := range s.clipHeaders {
		base := uint32(i) * 4
		packed[base+0] = math.Float32bits(ch.Duration)
		packed[base+1] = math.Float32bits(ch.TicksPerSecond)
		packed[base+2] = ch.ChannelOffset
		packed[base+3] = ch.ChannelCount
	}

	// Pack channel headers: [boneIndex, posKeyOffset, posKeyCount, rotKeyOffset, rotKeyCount, scaleKeyOffset, scaleKeyCount, pad]
	for i, ch := range s.channelHeaders {
		base := s.channelDataOffset + uint32(i)*8
		packed[base+0] = ch.BoneIndex
		packed[base+1] = ch.PositionKeyOffset
		packed[base+2] = ch.PositionKeyCount
		packed[base+3] = ch.RotationKeyOffset
		packed[base+4] = ch.RotationKeyCount
		packed[base+5] = ch.ScaleKeyOffset
		packed[base+6] = ch.ScaleKeyCount
		packed[base+7] = 0 // pad
	}

	// Pack keyframes: [time_bits, pad0, pad0, pad0, tx, ty, tz, pad1, rx, ry, rz, rw, sx, sy, sz, pad2]
	for i, kf := range s.keyFrames {
		base := s.keyframeDataOffset + uint32(i)*16
		packed[base+0] = math.Float32bits(kf.Time)
		packed[base+1] = 0 // pad0
		packed[base+2] = 0
		packed[base+3] = 0
		packed[base+4] = math.Float32bits(kf.Translation[0])
		packed[base+5] = math.Float32bits(kf.Translation[1])
		packed[base+6] = math.Float32bits(kf.Translation[2])
		packed[base+7] = 0 // pad1
		packed[base+8] = math.Float32bits(kf.Rotation[0])
		packed[base+9] = math.Float32bits(kf.Rotation[1])
		packed[base+10] = math.Float32bits(kf.Rotation[2])
		packed[base+11] = math.Float32bits(kf.Rotation[3])
		packed[base+12] = math.Float32bits(kf.Scale[0])
		packed[base+13] = math.Float32bits(kf.Scale[1])
		packed[base+14] = math.Float32bits(kf.Scale[2])
		packed[base+15] = 0 // pad2
	}

	packedRaw := common.SliceToBytes(packed)
	packedSnap := make([]byte, len(packedRaw))
	copy(packedSnap, packedRaw)
	s.stagedWriteData = append(s.stagedWriteData, bind_group_provider.BufferWrite{
		Provider: s.computeProvider,
		Binding:  binding,
		Offset:   0,
		Data:     packedSnap,
	})

	return clipIndex
}

func (s *skeletalAnimatorBackendImpl) PlayAnimation(instanceIndex, clipIndex uint32, loop bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if instanceIndex >= s.instanceCount {
		return
	}
	state := &s.instanceStateData[instanceIndex]
	state.clipIndex = clipIndex
	state.time = 0
	state.speed = 1.0
	state.loop = loop
	state.blending = false
	state.blendElapsed = 0
}

func (s *skeletalAnimatorBackendImpl) BlendToAnimation(instanceIndex, targetClipIndex uint32, blendDuration float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if instanceIndex >= s.instanceCount {
		return
	}
	state := &s.instanceStateData[instanceIndex]
	state.blending = true
	state.blendFrom = state.clipIndex
	state.blendFromTime = state.time
	state.blendTo = targetClipIndex
	state.blendToTime = 0
	state.blendDuration = blendDuration
	state.blendElapsed = 0
}

func (s *skeletalAnimatorBackendImpl) SetAnimationTime(instanceIndex uint32, time float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if instanceIndex >= s.instanceCount {
		return
	}
	s.instanceStateData[instanceIndex].time = time
}

func (s *skeletalAnimatorBackendImpl) SetAnimationSpeed(instanceIndex uint32, speed float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if instanceIndex >= s.instanceCount {
		return
	}
	s.instanceStateData[instanceIndex].speed = speed
}

func (s *skeletalAnimatorBackendImpl) IsBlending(instanceIndex uint32) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if instanceIndex >= s.instanceCount {
		return false
	}
	return s.instanceStateData[instanceIndex].blending
}

func (s *skeletalAnimatorBackendImpl) BlendProgress(instanceIndex uint32) float32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if instanceIndex >= s.instanceCount {
		return 0
	}
	state := &s.instanceStateData[instanceIndex]
	if !state.blending {
		return 0
	}
	return state.blendElapsed / state.blendDuration
}

func (s *skeletalAnimatorBackendImpl) CancelBlend(instanceIndex uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if instanceIndex >= s.instanceCount {
		return
	}
	s.instanceStateData[instanceIndex].blending = false
	s.instanceStateData[instanceIndex].blendElapsed = 0
}

func (s *skeletalAnimatorBackendImpl) BoneCount() uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.boneCount
}

func (s *skeletalAnimatorBackendImpl) Release() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.computeProvider != nil {
		s.computeProvider.Release()
	}
	if s.outputProvider != nil {
		s.outputProvider.Release()
	}
	s.instanceData = nil
	s.instanceStateData = nil
	s.perFrameSlice = nil
	s.stagedWriteData = nil
	s.bones = nil
	s.clipHeaders = nil
	s.channelHeaders = nil
	s.keyFrames = nil
	s.stagingInstance = nil
	s.stagingBones = nil
	s.stagingModel = nil
	s.stagingUniform = nil
	s.stagingIndirectRst = nil
	s.instanceRotSpeedData = nil
	s.instanceRotEulerData = nil
}

func (s *skeletalAnimatorBackendImpl) SetInstanceTransform(index uint32, posXYZ, scaleXYZ [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.maxInstances {
		return
	}

	base := index * 16
	mat := s.instanceModelData[base : base+16]
	// Rebuild model matrix from new position + scale, preserving zero rotation
	common.BuildModelMatrix(mat, posXYZ[0], posXYZ[1], posXYZ[2], 0, 0, 0, scaleXYZ[0], scaleXYZ[1], scaleXYZ[2])

	if !s.modelDirty {
		s.modelDirtyStart = index
		s.modelDirtyEnd = index + 1
		s.modelDirty = true
	} else {
		if index < s.modelDirtyStart {
			s.modelDirtyStart = index
		}
		if index+1 > s.modelDirtyEnd {
			s.modelDirtyEnd = index + 1
		}
	}
}

func (s *skeletalAnimatorBackendImpl) SetInstanceData(index uint32, posXYZ, scaleXYZ, rotSpeedXYZ [3]float32, rotXYZ [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.maxInstances {
		return
	}

	s.instanceRotSpeedData[index] = rotSpeedXYZ
	s.instanceRotEulerData[index] = rotXYZ

	base := index * 16
	mat := s.instanceModelData[base : base+16]
	common.BuildModelMatrix(mat, posXYZ[0], posXYZ[1], posXYZ[2], rotXYZ[0], rotXYZ[1], rotXYZ[2], scaleXYZ[0], scaleXYZ[1], scaleXYZ[2])

	if !s.modelDirty {
		s.modelDirtyStart = index
		s.modelDirtyEnd = index + 1
		s.modelDirty = true
	} else {
		if index < s.modelDirtyStart {
			s.modelDirtyStart = index
		}
		if index+1 > s.modelDirtyEnd {
			s.modelDirtyEnd = index + 1
		}
	}
}

func (s *skeletalAnimatorBackendImpl) InstanceTransform(index uint32) (pos, scale [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.instanceCount {
		return
	}
	base := index * 16
	mat := s.instanceModelData[base : base+16]
	pos = [3]float32{mat[12], mat[13], mat[14]}
	sx := float32(math.Sqrt(float64(mat[0]*mat[0] + mat[1]*mat[1] + mat[2]*mat[2])))
	sy := float32(math.Sqrt(float64(mat[4]*mat[4] + mat[5]*mat[5] + mat[6]*mat[6])))
	sz := float32(math.Sqrt(float64(mat[8]*mat[8] + mat[9]*mat[9] + mat[10]*mat[10])))
	scale = [3]float32{sx, sy, sz}
	return
}

// TODO: experimental method for managing the Animator's instance size.
// Grow increases the maximum instance capacity to newMax, preserving all existing instance,
// state, and model-matrix data. CPU-side slices are reallocated and copied, MinBindingSizes on
// compute and output layout entries are updated, and the needsRebuild flag is set so the render
// thread recreates GPU buffers on the next frame. No-op if newMax is less than or equal to the
// current maxInstances.
//
// Parameters:
//   - newMax: the new maximum number of instances to support
func (s *skeletalAnimatorBackendImpl) Grow(newMax uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if newMax <= s.maxInstances {
		return
	}

	// Grow instanceData (GPUSkeletalAnimationData)
	newData := make([]GPUSkeletalAnimationData, newMax)
	copy(newData, s.instanceData[:s.instanceCount])
	s.instanceData = newData

	// Grow instanceStateData (CPU-side playback state)
	newState := make([]skeletalInstanceState, newMax)
	copy(newState, s.instanceStateData[:s.instanceCount])
	s.instanceStateData = newState

	// Grow instanceModelData (16 floats per instance); initialize new slots to identity
	newModel := make([]float32, newMax*16)
	copy(newModel, s.instanceModelData[:s.instanceCount*16])
	for i := s.instanceCount; i < newMax; i++ {
		common.Identity(newModel[i*16 : (i+1)*16])
	}
	s.instanceModelData = newModel

	// Grow rotation tracking slices
	newRotSpeed := make([][3]float32, newMax)
	copy(newRotSpeed, s.instanceRotSpeedData[:s.instanceCount])
	s.instanceRotSpeedData = newRotSpeed
	newRotEuler := make([][3]float32, newMax)
	copy(newRotEuler, s.instanceRotEulerData[:s.instanceCount])
	s.instanceRotEulerData = newRotEuler

	s.maxInstances = newMax

	// Mark instance data dirty for full re-upload after rebuild
	if s.instanceCount > 0 {
		s.dirty = true
		s.dirtyStart = 0
		s.dirtyEnd = s.instanceCount
		s.modelDirty = true
		s.modelDirtyStart = 0
		s.modelDirtyEnd = s.instanceCount
	}

	// Bone data is immutable per skeleton; mark dirty so it is re-uploaded into new buffers
	if s.boneCount > 0 {
		s.boneDirty = true
	}

	// Reallocate staging pool to match new capacity
	s.initStagingPool()

	// Discard stale staged writes and signal rebuild
	s.stagedWriteData = s.stagedWriteData[:0]
	s.needsRebuild = true
}

// TODO: experimental method for managing the Animator's instance size.
// NeedsRebuild reports whether a Grow has occurred that requires GPU buffer recreation.
func (s *skeletalAnimatorBackendImpl) NeedsRebuild() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.needsRebuild
}

// TODO: experimental method for managing the Animator's instance size.
// ClearNeedsRebuild resets the needsRebuild flag. This is typically called after a
// successful RebuildGPU, but is also available for manual control.
func (s *skeletalAnimatorBackendImpl) ClearNeedsRebuild() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.needsRebuild = false
}

func (s *skeletalAnimatorBackendImpl) SetInstanceRotation(index uint32, rotSpeedXYZ, rotXYZ [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.maxInstances {
		return
	}

	// For skeletal instances, rotXYZ is Euler rotation (rotSpeed ignored).
	// Rebuild the model matrix incorporating rotation. We need to read back position and scale
	// from the existing matrix, so we just rebuild from scratch.
	// Since SetInstanceTransform is always called before SetInstanceRotation in Scene.Draw,
	// the position/scale are already set. We re-read them from the flat matrix.
	base := index * 16
	mat := s.instanceModelData[base : base+16]

	// Extract position from column 3
	px, py, pz := mat[12], mat[13], mat[14]
	// Extract scale from column magnitudes (assuming no shear from previous transform build)
	sx := float32(math.Sqrt(float64(mat[0]*mat[0] + mat[1]*mat[1] + mat[2]*mat[2])))
	sy := float32(math.Sqrt(float64(mat[4]*mat[4] + mat[5]*mat[5] + mat[6]*mat[6])))
	sz := float32(math.Sqrt(float64(mat[8]*mat[8] + mat[9]*mat[9] + mat[10]*mat[10])))

	s.instanceRotEulerData[index] = rotXYZ
	common.BuildModelMatrix(mat, px, py, pz, rotXYZ[0], rotXYZ[1], rotXYZ[2], sx, sy, sz)

	if !s.modelDirty {
		s.modelDirtyStart = index
		s.modelDirtyEnd = index + 1
		s.modelDirty = true
	} else {
		if index < s.modelDirtyStart {
			s.modelDirtyStart = index
		}
		if index+1 > s.modelDirtyEnd {
			s.modelDirtyEnd = index + 1
		}
	}
}

func (s *skeletalAnimatorBackendImpl) SetFrustumPlanes(planes [6]GPUFrustumPlane) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.frustumPlanes = planes
	s.cullingEnabled = true
}

func (s *skeletalAnimatorBackendImpl) SetBoundingRadius(radius float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.boundingRadius = radius
}

func (s *skeletalAnimatorBackendImpl) BoundingRadius() float32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.boundingRadius
}

func (s *skeletalAnimatorBackendImpl) IndirectBuffer(binding int) *wgpu.Buffer {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.computeProvider.Buffer(binding)
}

func (s *skeletalAnimatorBackendImpl) CullingEnabled() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cullingEnabled
}

func (s *skeletalAnimatorBackendImpl) ResetIndirectArgs(indexCount uint32, binding int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	args := GPUIndirectArgs{
		IndexCount:    indexCount,
		InstanceCount: 0,
		FirstIndex:    0,
		BaseVertex:    0,
		FirstInstance: 0,
	}
	raw := common.StructToBytes(&args)
	buf := s.stagingIndirectRst[:len(raw)]
	copy(buf, raw)

	s.stagedWriteData = append(s.stagedWriteData, bind_group_provider.BufferWrite{
		Provider: s.computeProvider,
		Binding:  binding,
		Offset:   0,
		Data:     buf,
	})
}

func (s *skeletalAnimatorBackendImpl) InstanceRotation(index uint32) (rotSpeed, rot [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.instanceCount {
		return
	}
	return s.instanceRotSpeedData[index], s.instanceRotEulerData[index]
}
