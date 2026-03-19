package light

import (
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/cogentcore/webgpu/wgpu"
)

// ssaoHandlerImpl is the implementation of the SSAOHandler interface.
type ssaoHandlerImpl struct {
	enabled bool

	screenWidth  int
	screenHeight int

	// Quality parameters.
	sampleCount    int
	maxSamples     int
	radius         float32
	bias           float32
	power          float32
	blurRadius     int
	halfResolution bool

	pipelineKeys map[string]string
	bgps         map[string]bind_group_provider.BindGroupProvider

	// Raw SSAO output (R8Unorm, screen resolution).
	rawTexture     *wgpu.Texture
	rawTextureView *wgpu.TextureView

	// Blurred SSAO output (R8Unorm, screen resolution).
	blurredTexture     *wgpu.Texture
	blurredTextureView *wgpu.TextureView

	// Intermediate scratch texture for the separable bilateral blur
	// (horizontal pass writes here, vertical pass reads from here).
	scratchTexture     *wgpu.Texture
	scratchTextureView *wgpu.TextureView

	// 4×4 noise texture (RGBA16Float) for kernel rotation.
	noiseTexture     *wgpu.Texture
	noiseTextureView *wgpu.TextureView

	// Linear sampler used for the final SSAO texture bound to the lit shader.
	linearSampler *wgpu.Sampler
}
