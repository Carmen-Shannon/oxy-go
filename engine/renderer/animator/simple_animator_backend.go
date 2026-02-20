package animator

import (
	"sync"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// simpleAnimatorBackendImpl is a concrete implementation of the simple animator backend.
type simpleAnimatorBackendImpl struct {
	mu *sync.Mutex

	// computeProvider and outputProvider are the BindGroupProviders for the compute and output shaders, respectively. The computeProvider provides access to GPU resources needed for the compute shader (e.g. storage buffers for instance data), while the outputProvider provides access to GPU resources needed for rendering (e.g. the buffer with updated transforms). These providers allow the backend to manage and update its GPU resource bindings as needed, especially when GPU buffers are recreated after capacity changes.
	computeProvider, outputProvider bind_group_provider.BindGroupProvider

	// stagedWriteData is a slice of BufferWrite structs that have been staged for writing to the GPU. Whenever instance data is updated (e.g. via SetInstanceTransform), a corresponding BufferWrite is created and added to this slice. The Flush method then processes this slice to perform the actual GPU buffer updates. This allows the backend to batch multiple updates together and minimize the number of GPU writes each frame.
	stagedWriteData []bind_group_provider.BufferWrite

	// maxInstances and instanceCount track the current capacity and number of active instances.
	maxInstances, instanceCount uint32

	// instanceData holds the CPU-side transform and rotation data for each instance. This is the source of truth for instance state, which gets staged to GPU buffers on Flush.
	instanceData []GPUAnimationData

	// Sparse dirty tracking: dirtyIndices holds instance indices that were mutated since
	// the last Flush. dirtyBitset provides O(1) dedup so the same index isn't enqueued twice.
	// This replaces the old contiguous dirty range (dirtyStart/dirtyEnd) to avoid uploading
	// large untouched spans when only a few scattered instances change.
	dirtyIndices []uint32
	dirtyBitset  []uint64 // 1 bit per instance index; word = index/64, bit = index%64

	// perFrameSlice is a reusable slice for staging per-instance culling data each frame, to avoid heap allocations.
	perFrameSlice []GPUGlobalData

	// needsRebuild is set to true when the instance capacity changes (e.g. via Grow) and GPU buffers need to be recreated to match the new capacity. This flag is checked by the render thread before rendering and triggers GPU resource reinitialization if set.
	needsRebuild bool

	// Reusable staging buffers to avoid per-frame heap allocations.
	// wgpu's queue.WriteBuffer copies data internally before returning,
	// so a single buffer reused every frame is safe.
	stagingInstance, stagingUniform, stagingIndirectRst []byte

	// Frustum culling state
	frustumPlanes  [6]GPUFrustumPlane
	boundingRadius float32
	cullingEnabled bool
}

// simpleAnimatorBackend defines the interface for the simple instanced animation backend.
// It manages per-instance transform, rotation, and scale data and stages GPU buffer writes
// for the compute shader that updates instance transforms each frame.
type simpleAnimatorBackend interface {
	// ComputeBindGroupProvider returns the BindGroupProvider for the compute shader, which provides access to the GPU resources needed for compute operations.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the BindGroupProvider for the compute shader
	ComputeBindGroupProvider() bind_group_provider.BindGroupProvider

	// SetComputeBindGroupProvider sets the BindGroupProvider for the compute shader. This allows the backend to update its reference to the compute provider, which may be necessary if GPU resources are recreated (e.g. after a Grow operation).
	//
	// Parameters:
	//   - provider: the new BindGroupProvider for the compute shader
	SetComputeBindGroupProvider(provider bind_group_provider.BindGroupProvider)

	// OutputBindGroupProvider returns the BindGroupProvider for the output shader, which provides access to the GPU resources needed for rendering the animated instances.
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the BindGroupProvider for the output shader
	OutputBindGroupProvider() bind_group_provider.BindGroupProvider

	// SetOutputBindGroupProvider sets the BindGroupProvider for the output shader. This allows the backend to update its reference to the output provider, which may be necessary if GPU resources are recreated (e.g. after a Grow operation).
	//
	// Parameters:
	//   - provider: the new BindGroupProvider for the output shader
	SetOutputBindGroupProvider(provider bind_group_provider.BindGroupProvider)

	// AddInstance registers a new instance with this animator.
	// If the current capacity is exceeded, the backend will automatically grow.
	//
	// Returns:
	//   - uint32: the index of the newly registered instance
	//   - error: an error if the instance could not be added
	AddInstance() (uint32, error)

	// InstanceCount returns the current number of instances being animated by this backend.
	//
	// Returns:
	//   - uint32: the current number of instances being animated
	InstanceCount() uint32

	// MaxInstances returns the maximum number of instances that this animator backend can handle. This is used to determine the capacity of the backend and should be set before adding instances.
	//
	// Returns:
	//   - uint32: the maximum number of instances supported by this backend
	MaxInstances() uint32

	// StagedWriteData returns the slice of BufferWrite structs that have been staged for writing to the GPU. This allows the caller to see what data is pending to be written and can be used for debugging or optimization purposes.
	//
	// Returns:
	//   - []bind_group_provider.BufferWrite: a slice of BufferWrite structs representing the staged data for GPU writes
	StagedWriteData() []bind_group_provider.BufferWrite

	// SetInstanceTransform sets the position and scale for a specific instance.
	//
	// Parameters:
	//   - index: the index of the instance to update
	//   - posXYZ: the new position of the instance as [3]float32 (x, y, z)
	//   - scaleXYZ: the new scale of the instance as [3]float32 (x, y, z)
	SetInstanceTransform(index uint32, posXYZ, scaleXYZ [3]float32)

	// SetInstanceRotation sets the rotation speed and current rotation for a specific instance.
	//
	// Parameters:
	//   - index: the index of the instance to update
	//   - rotSpeedXYZ: the rotation speed for the instance as [3]float32 (radians per second around x, y, z axes)
	//   - rotXYZ: the current rotation for the instance as [3]float32 (current angles around x, y, z axes)
	SetInstanceRotation(index uint32, rotSpeedXYZ, rotXYZ [3]float32)

	// InstanceTransform returns the position and scale for a specific instance.
	//
	// Parameters:
	//   - index: the index of the instance to query
	//
	// Returns:
	//   - pos: the position as [3]float32 (x, y, z)
	//   - scale: the scale as [3]float32 (x, y, z)
	InstanceTransform(index uint32) (pos, scale [3]float32)

	// InstanceRotation returns the rotation speed and current rotation for a specific instance.
	//
	// Parameters:
	//   - index: the index of the instance to query
	//
	// Returns:
	//   - rotSpeed: the rotation speed as [3]float32 (radians per second around x, y, z axes)
	//   - rot: the current rotation as [3]float32 (current angles around x, y, z axes)
	InstanceRotation(index uint32) (rotSpeed, rot [3]float32)

	// SetInstanceData sets all transform data for a specific instance in a single call,
	// combining SetInstanceTransform and SetInstanceRotation to reduce mutex overhead.
	//
	// Parameters:
	//   - index: the index of the instance to update
	//   - posXYZ: the new position of the instance as [3]float32 (x, y, z)
	//   - scaleXYZ: the new scale of the instance as [3]float32 (x, y, z)
	//   - rotSpeedXYZ: the rotation speed as [3]float32 (radians per second around x, y, z axes)
	//   - rotXYZ: the current rotation as [3]float32 (current angles around x, y, z axes)
	SetInstanceData(index uint32, posXYZ, scaleXYZ, rotSpeedXYZ, rotXYZ [3]float32)

	// SetMaxInstances sets the maximum number of instances that this animator backend can handle. This should be called before adding instances to ensure that the backend is configured with the correct capacity.
	//
	// Parameters:
	//   - maxInstances: the maximum number of instances to support
	SetMaxInstances(maxInstances uint32)

	// Flush updates the GPU buffers with any changes made to the instance data. This method should be called after making changes to instance transforms or rotations to ensure that the GPU has the latest data for rendering.
	//
	// Parameters:
	//   - instanceBinding: the bind group binding index for the compute shader's instance buffer, used for staging GPU writes
	//
	// Returns:
	//   - uint32: the number of instances that were updated and flushed to the GPU
	Flush(instanceBinding, _, _ int) uint32

	// PrepareFrame performs any necessary updates or calculations before rendering a frame. This method should be called once per frame, typically in the main render loop, to ensure that the animator is ready for rendering.
	//
	// Parameters:
	//   - deltaTime: the time elapsed since the last frame in seconds, which can be used for time-based animations and updates
	//   - binding: the bind group binding index for the compute shader's instance buffer, used for staging GPU writes
	PrepareFrame(deltaTime float32, binding int)

	// Release releases any resources held by the simple animator backend. This should be called when the animator is no longer needed to free up GPU resources and prevent memory leaks.
	Release()

	// SetFrustumPlanes updates the six frustum planes used for GPU frustum culling.
	// Planes should be extracted from the current view-projection matrix each frame.
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
	//   - binding: the bind group binding index for the compute shader's indirect args buffer, used for staging GPU writes
	//
	// Returns:
	//   - *wgpu.Buffer: the indirect draw arguments buffer, or nil
	IndirectBuffer(binding int) *wgpu.Buffer

	// CullingEnabled returns whether GPU frustum culling is active for this backend.
	//
	// Returns:
	//   - bool: true if frustum planes have been set and culling is active
	CullingEnabled() bool

	// ResetIndirectArgs stages a buffer write that zeros the indirect args instance count
	// before each compute dispatch, so the shader can atomically count visible instances.
	// The indexCount must be set so the GPU knows how many indices to draw per visible instance.
	//
	// Parameters:
	//   - indexCount: the number of indices in the mesh's index buffer
	//   - binding: the bind group binding index for the compute shader's indirect args buffer, used for staging GPU writes
	ResetIndirectArgs(indexCount uint32, binding int)

	// TODO: experimental method for managing the Animator's instance size.
	// Grow increases the maximum instance capacity to newMax, preserving all existing instance data.
	// CPU-side slices are reallocated and MinBindingSizes on layout entries are updated.
	// Sets the needsRebuild flag so the render thread recreates GPU buffers on the next frame.
	// No-op if newMax is less than or equal to the current maxInstances.
	//
	// Parameters:
	//   - newMax: the new maximum number of instances to support
	Grow(newMax uint32)

	// TODO: experimental method for managing the Animator's instance size.
	// RemoveInstance removes the instance at the given index using swap-remove.
	// The last active instance is moved into the removed slot to keep the array dense.
	// Returns the index that was swapped from and whether a swap occurred.
	//
	// Parameters:
	//   - index: the index of the instance to remove
	//
	// Returns:
	//   - uint32: the original index of the instance that was swapped into the removed slot
	//   - bool: true if a swap occurred (i.e. the removed index was not the last instance)
	RemoveInstance(index uint32) (uint32, bool)

	// TODO: experimental method for managing the Animator's instance size.
	// NeedsRebuild returns whether the GPU buffers need to be recreated due to a capacity change.
	//
	// Returns:
	//   - bool: true if GPU buffers are stale and need recreation
	NeedsRebuild() bool

	// TODO: experimental method for managing the Animator's instance size.
	// ClearNeedsRebuild resets the rebuild flag after GPU buffers have been successfully recreated.
	ClearNeedsRebuild()

	// --- No-op stubs for skeletalAnimatorBackend interface compliance ---

	// no-op
	SetBoneCount(count uint32)
	// no-op
	SetBone(index uint32, inverseBindMatrix [16]float32, localTranslation [3]float32, localRotation [4]float32, localScale [3]float32, parentIndex int32, binding int)
	// no-op
	AddClip(duration, ticksPerSecond float32, channels []uint32, keyframeTimes []float32, keyframeTranslations [][3]float32, keyframeRotations [][4]float32, keyframeScales [][3]float32, binding int) uint32
	// no-op
	PlayAnimation(instanceIndex, clipIndex uint32, loop bool)
	// no-op
	BlendToAnimation(instanceIndex, targetClipIndex uint32, blendDuration float32)
	// no-op
	SetAnimationTime(instanceIndex uint32, time float32)
	// no-op
	SetAnimationSpeed(instanceIndex uint32, speed float32)
	// no-op
	IsBlending(instanceIndex uint32) bool
	// no-op
	BlendProgress(instanceIndex uint32) float32
	// no-op
	CancelBlend(instanceIndex uint32)
	// no-op
	BoneCount() uint32
}

// compile-time check to ensure simpleAnimatorBackendImpl implements AnimatorBackend interface.
var _ AnimatorBackend = &simpleAnimatorBackendImpl{}

// newSimpleAnimatorBackend creates and initializes a new instance of the simple animator backend.
//
// Returns:
//   - simpleAnimatorBackend: a new instance of the simple animator backend
func newSimpleAnimatorBackend() simpleAnimatorBackend {
	s := &simpleAnimatorBackendImpl{
		mu:           &sync.Mutex{},
		maxInstances: 25000,
	}

	s.instanceData = make([]GPUAnimationData, s.maxInstances)
	s.perFrameSlice = make([]GPUGlobalData, 1)
	s.computeProvider = bind_group_provider.NewBindGroupProvider("animator_compute")
	s.outputProvider = bind_group_provider.NewBindGroupProvider("animator_output")
	s.stagedWriteData = make([]bind_group_provider.BufferWrite, 0, 2)
	s.dirtyIndices = make([]uint32, 0, s.maxInstances)
	s.dirtyBitset = make([]uint64, (s.maxInstances+63)/64)
	s.initStagingPool()
	return s
}

// initStagingPool allocates (or reallocates) the reusable staging byte slices
// sized to the current maxInstances. Called at init time and after Grow/SetMaxInstances.
func (s *simpleAnimatorBackendImpl) initStagingPool() {
	instanceBytes := int(s.maxInstances) * (&GPUAnimationData{}).Size()
	uniformBytes := (&GPUGlobalData{}).Size()
	indirectBytes := (&GPUIndirectArgs{}).Size()
	s.stagingInstance = make([]byte, instanceBytes)
	s.stagingUniform = make([]byte, uniformBytes)
	s.stagingIndirectRst = make([]byte, indirectBytes)
}

func (s *simpleAnimatorBackendImpl) ComputeBindGroupProvider() bind_group_provider.BindGroupProvider {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.computeProvider
}

func (s *simpleAnimatorBackendImpl) OutputBindGroupProvider() bind_group_provider.BindGroupProvider {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.outputProvider
}

func (s *simpleAnimatorBackendImpl) AddInstance() (uint32, error) {
	s.mu.Lock()
	if s.instanceCount >= s.maxInstances {
		// Auto-grow: double capacity (minimum 8). Unlock first because Grow acquires its own lock.
		newCap := max(s.maxInstances*2, 8)
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
// The last instance's data is copied into the removed slot and instanceCount is decremented.
// Returns true if a swap was performed (i.e. the removed index was not the last element),
// meaning the caller must update the swapped GameObject's stored instance index to `index`.
// Returns false if no swap occurred (the removed instance was already the last one, or the
// index was out of range).
//
// Parameters:
//   - index: the instance index to remove
//
// Returns:
//   - uint32: the old last index that was swapped into the removed slot (only meaningful when bool is true)
//   - bool: true if the last instance was swapped into the removed slot
func (s *simpleAnimatorBackendImpl) RemoveInstance(index uint32) (uint32, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.instanceCount == 0 || index >= s.instanceCount {
		return 0, false
	}

	last := s.instanceCount - 1
	swapped := index != last

	if swapped {
		// Copy last instance data into the removed slot
		s.instanceData[index] = s.instanceData[last]

		// Mark the swapped slot dirty so it gets re-uploaded
		s.enqueueDirty(index)
	}

	// Zero out the now-unused last slot and decrement
	s.instanceData[last] = GPUAnimationData{}
	s.instanceCount--

	return last, swapped
}

func (s *simpleAnimatorBackendImpl) SetComputeBindGroupProvider(provider bind_group_provider.BindGroupProvider) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.computeProvider = provider
}

func (s *simpleAnimatorBackendImpl) SetOutputBindGroupProvider(provider bind_group_provider.BindGroupProvider) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.outputProvider = provider
}

func (s *simpleAnimatorBackendImpl) InstanceCount() uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.instanceCount
}

func (s *simpleAnimatorBackendImpl) MaxInstances() uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxInstances
}

func (s *simpleAnimatorBackendImpl) StagedWriteData() []bind_group_provider.BufferWrite {
	s.mu.Lock()
	defer s.mu.Unlock()
	w := s.stagedWriteData
	s.stagedWriteData = s.stagedWriteData[:0]
	return w
}

func (s *simpleAnimatorBackendImpl) SetInstanceTransform(index uint32, posXYZ, scaleXYZ [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.maxInstances {
		return
	}

	s.instanceData[index].Pos = posXYZ
	s.instanceData[index].Scale = scaleXYZ
	s.enqueueDirty(index)
}

func (s *simpleAnimatorBackendImpl) SetInstanceRotation(index uint32, rotSpeedXYZ, rotXYZ [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.maxInstances {
		return
	}

	s.instanceData[index].RotSpeed = rotSpeedXYZ
	s.instanceData[index].Rot = rotXYZ
	s.enqueueDirty(index)
}

func (s *simpleAnimatorBackendImpl) SetInstanceData(index uint32, posXYZ, scaleXYZ, rotSpeedXYZ, rotXYZ [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.maxInstances {
		return
	}

	s.instanceData[index].Pos = posXYZ
	s.instanceData[index].Scale = scaleXYZ
	s.instanceData[index].RotSpeed = rotSpeedXYZ
	s.instanceData[index].Rot = rotXYZ
	s.enqueueDirty(index)
}

func (s *simpleAnimatorBackendImpl) InstanceTransform(index uint32) (pos, scale [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.instanceCount {
		return
	}
	return s.instanceData[index].Pos, s.instanceData[index].Scale
}

func (s *simpleAnimatorBackendImpl) InstanceRotation(index uint32) (rotSpeed, rot [3]float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if index >= s.instanceCount {
		return
	}
	return s.instanceData[index].RotSpeed, s.instanceData[index].Rot
}

func (s *simpleAnimatorBackendImpl) SetMaxInstances(maxInstances uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.maxInstances = maxInstances
	s.instanceData = make([]GPUAnimationData, maxInstances)
	s.instanceCount = 0
	s.dirtyIndices = s.dirtyIndices[:0]
	s.dirtyBitset = make([]uint64, (maxInstances+63)/64)
	s.initStagingPool()
}

func (s *simpleAnimatorBackendImpl) Flush(instanceBinding, _, _ int) uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.dirtyIndices) == 0 || s.needsRebuild {
		return 0
	}

	// Sort dirty indices so we can coalesce adjacent ones into contiguous buffer writes,
	// minimizing the number of GPU write commands while only uploading mutated data.
	sortUint32(s.dirtyIndices)

	instSize := uint64((&GPUAnimationData{}).Size())
	count := uint32(len(s.dirtyIndices))

	// Walk sorted indices and merge contiguous runs into single writes
	runStart := s.dirtyIndices[0]
	runEnd := runStart + 1 // exclusive

	for i := 1; i < len(s.dirtyIndices); i++ {
		idx := s.dirtyIndices[i]
		if idx == runEnd {
			runEnd++
		} else {
			s.flushRange(runStart, runEnd, instSize, instanceBinding)
			runStart = idx
			runEnd = idx + 1
		}
	}
	s.flushRange(runStart, runEnd, instSize, instanceBinding)

	// Clear dirty state: reset indices slice and zero the bitset
	s.dirtyIndices = s.dirtyIndices[:0]
	for i := range s.dirtyBitset {
		s.dirtyBitset[i] = 0
	}

	return count
}

func (s *simpleAnimatorBackendImpl) PrepareFrame(deltaTime float32, binding int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.needsRebuild {
		return
	}

	s.perFrameSlice[0] = GPUGlobalData{
		InstanceCount:  s.instanceCount,
		DeltaTime:      deltaTime,
		BoundingRadius: s.boundingRadius,
		Planes:         s.frustumPlanes,
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

// enqueueDirty adds an instance index to the dirty queue if not already present.
// Uses a bitset for O(1) dedup. Caller must hold s.mu.
func (s *simpleAnimatorBackendImpl) enqueueDirty(index uint32) {
	word := index / 64
	bit := uint64(1) << (index % 64)
	if s.dirtyBitset[word]&bit != 0 {
		return // already queued
	}
	s.dirtyBitset[word] |= bit
	s.dirtyIndices = append(s.dirtyIndices, index)
}

// flushRange stages a contiguous run of dirty instance data [start, end) as a single
// GPU buffer write. Caller must hold s.mu.
func (s *simpleAnimatorBackendImpl) flushRange(start, end uint32, instSize uint64, binding int) {
	offset := uint64(start) * instSize
	dirty := s.instanceData[start:end]
	raw := common.SliceToBytes(dirty)
	buf := s.stagingInstance[offset : offset+uint64(len(raw))]
	copy(buf, raw)

	s.stagedWriteData = append(s.stagedWriteData, bind_group_provider.BufferWrite{
		Provider: s.computeProvider,
		Binding:  binding,
		Offset:   offset,
		Data:     buf,
	})
}

// sortUint32 sorts a uint32 slice in ascending order using insertion sort.
// For the typical dirty queue sizes (0 to a few hundred), insertion sort
// outperforms sort.Slice due to zero allocation and low overhead.
func sortUint32(s []uint32) {
	for i := 1; i < len(s); i++ {
		key := s[i]
		j := i - 1
		for j >= 0 && s[j] > key {
			s[j+1] = s[j]
			j--
		}
		s[j+1] = key
	}
}

func (s *simpleAnimatorBackendImpl) Release() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.computeProvider != nil {
		s.computeProvider.Release()
	}
	if s.outputProvider != nil {
		s.outputProvider.Release()
	}
	s.instanceData = nil
	s.perFrameSlice = nil
	s.stagedWriteData = nil
	s.stagingInstance = nil
	s.stagingUniform = nil
	s.stagingIndirectRst = nil
	s.dirtyIndices = nil
	s.dirtyBitset = nil
}

// TODO: experimental method for managing the Animator's instance size.
// Grow increases the maximum instance capacity to newMax, preserving all existing instance data.
// CPU-side slices are reallocated and copied, MinBindingSizes on layout entries are updated,
// and the needsRebuild flag is set so the render thread recreates GPU buffers on the next frame.
// No-op if newMax is less than or equal to the current maxInstances.
//
// Parameters:
//   - newMax: the new maximum number of instances to support
func (s *simpleAnimatorBackendImpl) Grow(newMax uint32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if newMax <= s.maxInstances {
		return
	}

	// Allocate new slice and copy existing instance data
	newData := make([]GPUAnimationData, newMax)
	copy(newData, s.instanceData[:s.instanceCount])
	s.instanceData = newData
	s.maxInstances = newMax

	// Mark all existing instances dirty for full re-upload after rebuild.
	// Rebuild the bitset for the new capacity and enqueue every live index.
	s.dirtyBitset = make([]uint64, (newMax+63)/64)
	s.dirtyIndices = s.dirtyIndices[:0]
	for i := uint32(0); i < s.instanceCount; i++ {
		s.enqueueDirty(i)
	}

	// Reallocate staging pool to match new capacity
	s.initStagingPool()

	// Discard stale staged writes and signal rebuild
	s.stagedWriteData = s.stagedWriteData[:0]
	s.needsRebuild = true
}

// TODO: experimental method for managing the Animator's instance size.
// NeedsRebuild reports whether a Grow has occurred that requires GPU buffer recreation.
func (s *simpleAnimatorBackendImpl) NeedsRebuild() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.needsRebuild
}

// TODO: experimental method for managing the Animator's instance size.
// ClearNeedsRebuild resets the needsRebuild flag. This is typically called after a
// successful RebuildGPU, but is also available for manual control.
func (s *simpleAnimatorBackendImpl) ClearNeedsRebuild() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.needsRebuild = false
}

// --- No-op stubs for skeletalAnimatorBackend interface compliance ---

func (s *simpleAnimatorBackendImpl) SetBoneCount(count uint32) {}
func (s *simpleAnimatorBackendImpl) SetBone(index uint32, inverseBindMatrix [16]float32, localTranslation [3]float32, localRotation [4]float32, localScale [3]float32, parentIndex int32, binding int) {
}
func (s *simpleAnimatorBackendImpl) AddClip(duration, ticksPerSecond float32, channels []uint32, keyframeTimes []float32, keyframeTranslations [][3]float32, keyframeRotations [][4]float32, keyframeScales [][3]float32, binding int) uint32 {
	return 0
}
func (s *simpleAnimatorBackendImpl) PlayAnimation(instanceIndex, clipIndex uint32, loop bool) {}
func (s *simpleAnimatorBackendImpl) BlendToAnimation(instanceIndex, targetClipIndex uint32, blendDuration float32) {
}
func (s *simpleAnimatorBackendImpl) SetAnimationTime(instanceIndex uint32, time float32)   {}
func (s *simpleAnimatorBackendImpl) SetAnimationSpeed(instanceIndex uint32, speed float32) {}
func (s *simpleAnimatorBackendImpl) IsBlending(instanceIndex uint32) bool                  { return false }
func (s *simpleAnimatorBackendImpl) BlendProgress(instanceIndex uint32) float32            { return 0 }
func (s *simpleAnimatorBackendImpl) CancelBlend(instanceIndex uint32)                      {}
func (s *simpleAnimatorBackendImpl) BoneCount() uint32                                     { return 0 }

func (s *simpleAnimatorBackendImpl) SetFrustumPlanes(planes [6]GPUFrustumPlane) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.frustumPlanes = planes
	s.cullingEnabled = true
}

func (s *simpleAnimatorBackendImpl) SetBoundingRadius(radius float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.boundingRadius = radius
}

func (s *simpleAnimatorBackendImpl) BoundingRadius() float32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.boundingRadius
}

func (s *simpleAnimatorBackendImpl) IndirectBuffer(binding int) *wgpu.Buffer {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.cullingEnabled {
		return nil
	}
	return s.computeProvider.Buffer(binding)
}

func (s *simpleAnimatorBackendImpl) CullingEnabled() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cullingEnabled
}

func (s *simpleAnimatorBackendImpl) ResetIndirectArgs(indexCount uint32, binding int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.needsRebuild || !s.cullingEnabled {
		return
	}

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
