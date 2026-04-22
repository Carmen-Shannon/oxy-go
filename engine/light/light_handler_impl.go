package light

import "github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"

// lightingHandlerImpl is the implementation of the LightingHandler interface.
type lightingHandlerImpl struct {
	enabled bool

	lights       []Light
	ambientColor [3]float32
	maxGPULights int

	bgps         map[string]bind_group_provider.BindGroupProvider
	pipelineKeys map[string]string

	shadowHandler ShadowHandler

	screenWidth      int
	screenHeight     int
	tileCountX       int
	tileCountY       int
	tileSize         int
	maxLightsPerTile int

	// GI sub-handlers — owned by the lighting system and lazily initialized
	// by the scene when the first light is added.
	contactShadowHandler ContactShadowHandler
}
