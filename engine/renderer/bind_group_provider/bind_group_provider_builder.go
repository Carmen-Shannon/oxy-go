package bind_group_provider

import "github.com/cogentcore/webgpu/wgpu"

// BindGroupProviderOption is a functional option used to configure a BindGroupProvider during construction.
type BindGroupProviderOption func(*bindGroupProvider)

// WithBindGroup sets the bind group for this provider.
//
// Parameters:
//   - bg: the bind group to set for this provider
//
// Returns:
//   - BindGroupProviderOption: a function that sets the bind group for this provider
func WithBindGroup(bg *wgpu.BindGroup) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.bindGroup = bg
	}
}

// WithBindGroupLayout sets the bind group layout for this provider.
//
// Parameters:
//   - bgl: the bind group layout to use for this provider
//
// Returns:
//   - BindGroupProviderOption: a function that sets the bind group layout for this provider
func WithBindGroupLayout(bgl *wgpu.BindGroupLayout) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.bindGroupLayout = bgl
	}
}

// WithBuffer sets a buffer for a specific binding index.
//
// Parameters:
//   - binding: the binding index for this buffer
//   - buf: the buffer to associate with this binding
//
// Returns:
//   - BindGroupProviderOption: a function that sets the buffer for the specified binding
func WithBuffer(binding int, buf *wgpu.Buffer) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.buffers[binding] = buf
	}
}

// WithBuffers sets multiple buffers for this provider using a map of binding indices to buffers.
//
// Parameters:
//   - buffers: a map of binding indices to buffers to associate with this provider
//
// Returns:
//   - BindGroupProviderOption: a function that sets multiple buffers for this provider
func WithBuffers(buffers map[int]*wgpu.Buffer) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.buffers = buffers
	}
}
