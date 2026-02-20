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

// AnimatorBackend is the union interface that all animation backends must implement.
// It embeds both simpleAnimatorBackend and skeletalAnimatorBackend, requiring concrete
// implementations to provide the full method set. Methods that do not apply to a given
// backend type are implemented as no-ops.
type AnimatorBackend interface {
	simpleAnimatorBackend
	skeletalAnimatorBackend
}
