// Package bind_group_provider defines interfaces for renderer bind group provisioning.
//
// The [BindGroupProvider] interface is the primary abstraction in this package, which
// centralizes GPU bind group resources used by renderer initialization and update
// paths, including buffers, texture views, samplers, and mesh buffers.
package bind_group_provider

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/cogentcore/webgpu/wgpu"
)

// BindGroupProvider defines the interface for components that require GPU bind group resources.
// Components (Camera, GameObject, etc.) hold a BindGroupProvider to describe their GPU binding
// requirements. The Renderer then uses this provider to initialize and update GPU resources.
//
// Usage pattern:
//  1. Component creates a BindGroupProvider with layout entries and a unique key
//  2. Component stores the provider via SetBindGroupProvider()
//  3. Scene/Renderer calls Renderer.InitBindGroup(provider) to create GPU resources
//  4. Scene/Renderer calls Renderer.WriteBindGroup(provider, data) to update uniforms
//  5. Component accesses BindGroup() for draw calls
type BindGroupProvider interface {
	common.Delegate[BindGroupProvider]

	// Release releases any GPU resources held by this provider.
	// It will clean up all buffers and bind groups, and remove them from the map or slice they belonged to.
	Release()

	// Label returns the debug label for this provider.
	// Used for debugging and profiling purposes.
	//
	// Returns:
	//   - string: the debug label
	Label() string

	// BindGroup returns the created bind group for shader binding.
	// Returns nil if GPU resources have not been initialized.
	//
	// Returns:
	//   - *wgpu.BindGroup: the bind group or nil
	BindGroup() *wgpu.BindGroup

	// BindGroupLayout returns the created bind group layout for this provider.
	// Returns nil if GPU resources have not been initialized.
	//
	// Returns:
	//   - *wgpu.BindGroupLayout: the bind group layout or nil
	BindGroupLayout() *wgpu.BindGroupLayout

	// Buffer returns the created uniform buffer for data writes.
	// Returns nil if GPU resources have not been initialized.
	//
	// Returns:
	//   - *wgpu.Buffer: the buffer or nil
	Buffer(binding int) *wgpu.Buffer

	// Buffers returns a map of all buffers associated with this provider, keyed by binding index.
	// This allows providers to manage multiple buffers if needed.
	//
	// Returns:
	//   - map[int]*wgpu.Buffer: a map of buffers keyed by binding index
	Buffers() map[int]*wgpu.Buffer

	// TextureView returns the GPU texture view for a specific binding, or nil if not set.
	//
	// Parameters:
	//   - binding: the binding index
	//
	// Returns:
	//   - *wgpu.TextureView: the texture view or nil
	TextureView(binding int) *wgpu.TextureView

	// TextureViews returns a map of all texture views associated with this provider, keyed by binding index.
	//
	// Returns:
	//   - map[int]*wgpu.TextureView: a map of texture views keyed by binding index
	TextureViews() map[int]*wgpu.TextureView

	// Sampler returns the GPU sampler for a specific binding, or nil if not set.
	//
	// Parameters:
	//   - binding: the binding index
	//
	// Returns:
	//   - *wgpu.Sampler: the sampler or nil
	Sampler(binding int) *wgpu.Sampler

	// Samplers returns a map of all samplers associated with this provider, keyed by binding index.
	//
	// Returns:
	//   - map[int]*wgpu.Sampler: a map of samplers keyed by binding index
	Samplers() map[int]*wgpu.Sampler

	// VertexBuffer returns the GPU vertex buffer, or nil if not initialized.
	//
	// Returns:
	//   - *wgpu.Buffer: the vertex buffer or nil
	VertexBuffer() *wgpu.Buffer

	// IndexBuffer returns the GPU index buffer, or nil if not initialized.
	//
	// Returns:
	//   - *wgpu.Buffer: the index buffer or nil
	IndexBuffer() *wgpu.Buffer

	// IndexCount returns the number of indices for draw calls.
	//
	// Returns:
	//   - int: the index count
	IndexCount() int

	// SetBindGroup sets the bind group after GPU initialization.
	// Called by Renderer.InitBindGroup().
	//
	// Parameters:
	//   - bg: the created bind group
	SetBindGroup(bg *wgpu.BindGroup)

	// SetBindGroupLayout sets the bind group layout after GPU initialization.
	// Called by Renderer.InitBindGroup().
	//
	// Parameters:
	//   - bgl: the created bind group layout
	SetBindGroupLayout(bgl *wgpu.BindGroupLayout)

	// SetBuffer sets the uniform buffer after GPU initialization.
	// Called by Renderer.InitBindGroup().
	//
	// Parameters:
	//   - buf: the created buffer
	SetBuffer(binding int, buf *wgpu.Buffer)

	// SetBuffers sets multiple buffers at once after GPU initialization.
	// This is a convenience method for providers that manage multiple buffers.
	//
	// Parameters:
	//   - buffers: a map of buffers keyed by binding index
	SetBuffers(buffers map[int]*wgpu.Buffer)

	// SetTextureView stores a GPU texture view for a specific binding.
	//
	// Parameters:
	//   - binding: the binding index
	//   - tv: the texture view to store
	SetTextureView(binding int, tv *wgpu.TextureView)

	// SetTextureViews stores multiple GPU texture views at once.
	//
	// Parameters:
	//   - textureViews: a map of texture views keyed by binding index
	SetTextureViews(textureViews map[int]*wgpu.TextureView)

	// SetSampler stores a GPU sampler for a specific binding.
	//
	// Parameters:
	//   - binding: the binding index
	//   - s: the sampler to store
	SetSampler(binding int, s *wgpu.Sampler)

	// SetSamplers stores multiple GPU samplers at once.
	//
	// Parameters:
	//   - samplers: a map of samplers keyed by binding index
	SetSamplers(samplers map[int]*wgpu.Sampler)

	// SetVertexBuffer stores the GPU vertex buffer after creation by InitBindGroup.
	//
	// Parameters:
	//   - buf: the created vertex buffer
	SetVertexBuffer(buf *wgpu.Buffer)

	// SetIndexBuffer stores the GPU index buffer after creation by InitBindGroup.
	//
	// Parameters:
	//   - buf: the created index buffer
	SetIndexBuffer(buf *wgpu.Buffer)

	// SetIndexCount sets the number of indices for draw calls.
	//
	// Parameters:
	//   - count: the index count
	SetIndexCount(count int)

	// SetSlot sets the active frame-in-flight slot for this provider.
	// Slot-sensitive methods (Buffer, Buffers, SetBuffer, SetBuffers, BindGroup, SetBindGroup)
	// will operate on the specified slot's resources.
	//
	// Parameters:
	//   - slot: the frame-in-flight slot index (0 or 1)
	SetSlot(slot int)
}

// Compile-time check that bindGroupProvider implements BindGroupProvider
var _ BindGroupProvider = &bindGroupProvider{}

func (p *bindGroupProvider) Label() string                                { return p.label }
func (p *bindGroupProvider) BindGroup() *wgpu.BindGroup                   { return p.bindGroups[p.activeSlot] }
func (p *bindGroupProvider) BindGroupLayout() *wgpu.BindGroupLayout       { return p.bindGroupLayout }
func (p *bindGroupProvider) Buffer(binding int) *wgpu.Buffer              { return p.buffers[p.activeSlot][binding] }
func (p *bindGroupProvider) Buffers() map[int]*wgpu.Buffer                { return p.buffers[p.activeSlot] }
func (p *bindGroupProvider) TextureViews() map[int]*wgpu.TextureView      { return p.textureViews }
func (p *bindGroupProvider) Sampler(binding int) *wgpu.Sampler            { return p.samplers[binding] }
func (p *bindGroupProvider) Samplers() map[int]*wgpu.Sampler              { return p.samplers }
func (p *bindGroupProvider) VertexBuffer() *wgpu.Buffer                   { return p.vertexBuffer }
func (p *bindGroupProvider) IndexBuffer() *wgpu.Buffer                    { return p.indexBuffer }
func (p *bindGroupProvider) IndexCount() int                              { return p.indexCount }
func (p *bindGroupProvider) SetBindGroup(bg *wgpu.BindGroup)              { p.bindGroups[p.activeSlot] = bg }
func (p *bindGroupProvider) SetBindGroupLayout(bgl *wgpu.BindGroupLayout) { p.bindGroupLayout = bgl }
func (p *bindGroupProvider) SetBuffers(buffers map[int]*wgpu.Buffer) {
	p.buffers[p.activeSlot] = buffers
}
func (p *bindGroupProvider) SetVertexBuffer(buf *wgpu.Buffer)           { p.vertexBuffer = buf }
func (p *bindGroupProvider) SetIndexBuffer(buf *wgpu.Buffer)            { p.indexBuffer = buf }
func (p *bindGroupProvider) SetIndexCount(count int)                    { p.indexCount = count }
func (p *bindGroupProvider) SetSamplers(samplers map[int]*wgpu.Sampler) { p.samplers = samplers }
func (p *bindGroupProvider) SetSlot(slot int)                           { p.activeSlot = slot }

func (p *bindGroupProvider) TextureView(binding int) *wgpu.TextureView {
	return p.textureViews[binding]
}

func (p *bindGroupProvider) SetBuffer(binding int, buf *wgpu.Buffer) {
	if p.buffers[p.activeSlot] == nil {
		p.buffers[p.activeSlot] = make(map[int]*wgpu.Buffer)
	}
	p.buffers[p.activeSlot][binding] = buf
}

func (p *bindGroupProvider) SetTextureView(binding int, tv *wgpu.TextureView) {
	if p.textureViews == nil {
		p.textureViews = make(map[int]*wgpu.TextureView)
	}
	p.textureViews[binding] = tv
}

func (p *bindGroupProvider) SetTextureViews(textureViews map[int]*wgpu.TextureView) {
	p.textureViews = textureViews
}

func (p *bindGroupProvider) SetSampler(binding int, s *wgpu.Sampler) {
	if p.samplers == nil {
		p.samplers = make(map[int]*wgpu.Sampler)
	}
	p.samplers[binding] = s
}

func (p *bindGroupProvider) Release() {
	for i, tv := range p.textureViews {
		if tv != nil {
			tv.Release()
			delete(p.textureViews, i)
		}
	}
	for i, s := range p.samplers {
		if s != nil {
			s.Release()
			delete(p.samplers, i)
		}
	}
	for slot := range p.buffers {
		for i, buf := range p.buffers[slot] {
			if buf != nil {
				buf.Release()
				delete(p.buffers[slot], i)
			}
		}
	}
	for i := range p.bindGroups {
		if p.bindGroups[i] != nil {
			p.bindGroups[i].Release()
			p.bindGroups[i] = nil
		}
	}
	if p.bindGroupLayout != nil {
		p.bindGroupLayout.Release()
		p.bindGroupLayout = nil
	}
	if p.vertexBuffer != nil {
		p.vertexBuffer.Release()
		p.vertexBuffer = nil
	}
	if p.indexBuffer != nil {
		p.indexBuffer.Release()
		p.indexBuffer = nil
	}
}
