package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// contactShadowHandlerImpl is the implementation of the ContactShadowHandler interface.
type contactShadowHandlerImpl struct {
	enabled bool

	stepCount   int
	maxDistance float32
	thickness   float32

	pipelineKeys map[string]string
	bgps         map[string]bind_group_provider.BindGroupProvider

	// Contact shadow output (R32Float, screen resolution).
	texture     *wgpu.Texture
	textureView *wgpu.TextureView

	// Linear sampler for the lit shader to sample the contact shadow texture.
	linearSampler *wgpu.Sampler
}
