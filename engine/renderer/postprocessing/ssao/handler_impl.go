package ssao

import (
	"github.com/oliverbestmann/webgpu/wgpu"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// handlerImpl is the implementation of the Handler interface.
type handlerImpl struct {
	enabled bool

	screenWidth  int
	screenHeight int

	// Quality parameters.
	sampleCount    int
	maxSamples     int
	screenRadius   float32
	bias           float32
	power          float32
	blurRadius     int
	halfResolution bool

	pipelineKeys map[string]string
	bgps         map[string]bind_group_provider.BindGroupProvider

	activeSlot int

	// Raw SSAO output (R8Unorm, screen resolution).
	rawTextures     [2]*wgpu.Texture
	rawTextureViews [2]*wgpu.TextureView

	// Blurred SSAO output (R8Unorm, screen resolution).
	blurredTextures     [2]*wgpu.Texture
	blurredTextureViews [2]*wgpu.TextureView

	// Intermediate scratch texture for the separable bilateral blur
	// (horizontal pass writes here, vertical pass reads from here).
	scratchTextures     [2]*wgpu.Texture
	scratchTextureViews [2]*wgpu.TextureView

	// Linear sampler used for the final SSAO texture bound to the lit shader.
	linearSampler *wgpu.Sampler
}
