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
		p.bindGroups[p.activeSlot] = bg
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
		p.buffers[p.activeSlot][binding] = buf
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
		p.buffers[p.activeSlot] = buffers
	}
}

// WithTextureView sets a texture view for a specific binding index.
//
// Parameters:
//   - binding: the binding index for this texture view
//   - tv: the texture view to associate with this binding
//
// Returns:
//   - BindGroupProviderOption: a function that sets the texture view for the specified binding
func WithTextureView(binding int, tv *wgpu.TextureView) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.textureViews[binding] = tv
	}
}

// WithTextureViews sets multiple texture views for this provider using a map of binding indices to texture views.
//
// Parameters:
//   - textureViews: a map of binding indices to texture views to associate with this provider
//
// Returns:
//   - BindGroupProviderOption: a function that sets multiple texture views for this provider
func WithTextureViews(textureViews map[int]*wgpu.TextureView) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.textureViews = textureViews
	}
}

// WithSampler sets a sampler for a specific binding index.
//
// Parameters:
//   - binding: the binding index for this sampler
//   - s: the sampler to associate with this binding
//
// Returns:
//   - BindGroupProviderOption: a function that sets the sampler for the specified binding
func WithSampler(binding int, s *wgpu.Sampler) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.samplers[binding] = s
	}
}

// WithSamplers sets multiple samplers for this provider using a map of binding indices to samplers.
//
// Parameters:
//   - samplers: a map of binding indices to samplers to associate with this provider
//
// Returns:
//   - BindGroupProviderOption: a function that sets multiple samplers for this provider
func WithSamplers(samplers map[int]*wgpu.Sampler) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.samplers = samplers
	}
}

// WithVertexBuffer sets the vertex buffer for this provider.
//
// Parameters:
//   - buf: the vertex buffer to set for this provider
//
// Returns:
//   - BindGroupProviderOption: a function that sets the vertex buffer for this provider
func WithVertexBuffer(buf *wgpu.Buffer) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.vertexBuffer = buf
	}
}

// WithIndexBuffer sets the index buffer for this provider.
//
// Parameters:
//   - buf: the index buffer to set for this provider
//
// Returns:
//   - BindGroupProviderOption: a function that sets the index buffer for this provider
func WithIndexBuffer(buf *wgpu.Buffer) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.indexBuffer = buf
	}
}

// WithIndexCount sets the index count for draw calls on this provider.
//
// Parameters:
//   - count: the number of indices for draw calls
//
// Returns:
//   - BindGroupProviderOption: a function that sets the index count for this provider
func WithIndexCount(count int) BindGroupProviderOption {
	return func(p *bindGroupProvider) {
		p.indexCount = count
	}
}

// NewBindGroupProvider creates a new BindGroupProvider with the provided options.
//
// Parameters:
//   - options: a variadic list of options to configure the provider
//
// Returns:
//   - BindGroupProvider: a new instance of BindGroupProvider configured with the provided options
func NewBindGroupProvider(label string, options ...BindGroupProviderOption) BindGroupProvider {
	p := &bindGroupProvider{
		label: label,
		buffers: [2]map[int]*wgpu.Buffer{
			make(map[int]*wgpu.Buffer),
			make(map[int]*wgpu.Buffer),
		},
		textureViews: make(map[int]*wgpu.TextureView),
		samplers:     make(map[int]*wgpu.Sampler),
	}
	for _, opt := range options {
		opt(p)
	}
	p.Delegate = p
	return p
}
