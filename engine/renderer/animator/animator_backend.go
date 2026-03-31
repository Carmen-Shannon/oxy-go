package animator

// AnimatorBackendType identifies the type of animation backend used by an Animator.
type AnimatorBackendType int

const (
	// BackendTypeSimple is the simple instanced animation backend, supporting per-instance
	// position, rotation, and scale driven by a compute shader.
	BackendTypeSimple AnimatorBackendType = iota

	// BackendTypeSkeletal is the skeletal animation backend, supporting bone-based animation
	// with blending and per-instance playback state driven by a compute shader.
	BackendTypeSkeletal
)

// AnimatorBackend is the interface all animation backends must implement.
// It covers the common per-instance transform, culling, and lifecycle operations.
// Skeletal-only capabilities (bones, clips, playback) are provided by skeletalAnimatorBackend
// and accessed via type assertion in the animator delegation layer.
type AnimatorBackend interface {
	simpleAnimatorBackend
}
