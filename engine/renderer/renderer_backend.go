package renderer

// RendererBackendType identifies the GPU backend implementation used by the Renderer.
type RendererBackendType int

const (
	// BackendTypeWGPU selects the WebGPU-based rendering backend.
	BackendTypeWGPU RendererBackendType = iota
)

// PresentMode controls how rendered frames are presented to the display surface.
type PresentMode int

const (
	// PresentModeVSync waits for the next vertical blank before presenting, capping frame rate
	// to the monitor's refresh rate. Eliminates tearing.
	PresentModeVSync PresentMode = iota

	// PresentModeUncapped presents frames immediately without waiting for vertical blank.
	// May cause screen tearing but provides the lowest latency.
	PresentModeUncapped
)

// MSAASampleCount controls the number of samples used for multisample anti-aliasing (MSAA).
// Only specific power-of-two values are valid for GPU hardware. WebGPU guarantees support for
// 1 (off) and 4; higher values (8, 16) are adapter-dependent and may not be available.
type MSAASampleCount uint32

const (
	// MSAAOff disables multisample anti-aliasing (sample count 1).
	MSAAOff MSAASampleCount = 1

	// MSAA4x enables 4× multisample anti-aliasing. This is the default.
	MSAA4x MSAASampleCount = 4

	// MSAA8x enables 8× multisample anti-aliasing. Adapter-dependent; not all hardware supports this.
	MSAA8x MSAASampleCount = 8

	// MSAA16x enables 16× multisample anti-aliasing. Adapter-dependent; not all hardware supports this.
	MSAA16x MSAASampleCount = 16
)

// RendererBackend is the top-level backend interface for the Renderer.
// It embeds the concrete backend interface for the selected GPU API.
type RendererBackend interface {
	wgpuRendererBackend
}
