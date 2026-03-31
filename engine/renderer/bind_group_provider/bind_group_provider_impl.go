package bind_group_provider

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/cogentcore/webgpu/wgpu"
)

// bindGroupProvider is the unexported implementation of BindGroupProvider.
type bindGroupProvider struct {
	common.DelegateImpl[BindGroupProvider]

	// label is a debug label added for convenience.
	label string

	// The following fields are GPU allocated resources and must be released when no longer needed. They are populated by the Renderer during initialization, not by user-creation.

	// activeSlot is the currently active frame-in-flight slot (0 or 1). Defaults to 0.
	activeSlot int
	// bindGroups holds one bind group per frame-in-flight slot.
	bindGroups [2]*wgpu.BindGroup
	// bindGroupLayout is the GPU bind group layout created for this provider, or nil if not initialized with the Renderer.
	// TODO: Investigate whether this even needs to remain persisted anywhere, once the layout is created via the Shader that holds the BindGroupLayoutDescriptor what do we need this for?
	bindGroupLayout *wgpu.BindGroupLayout
	// buffers holds the GPU buffers created for this provider per frame-in-flight slot, keyed by binding index.
	buffers [2]map[int]*wgpu.Buffer
	// textureViews holds the GPU texture views created for this provider, keyed by binding index.
	textureViews map[int]*wgpu.TextureView
	// samplers holds the GPU samplers created for this provider, keyed by binding index.
	samplers map[int]*wgpu.Sampler

	// The following fields are specific to vertex pulling providers. They are used to stage vertex/index data and describe vertex formats before GPU upload.

	// vertexBuffer is the GPU vertex buffer created for this provider, or nil if not initialized with the Renderer.
	vertexBuffer *wgpu.Buffer
	// indexBuffer is the GPU index buffer created for this provider, or nil if not initialized with the Renderer.
	indexBuffer *wgpu.Buffer
	// indexCount is the number of indices for draw calls, used by the Renderer to issue drawIndexed calls for this provider.
	indexCount int
}
