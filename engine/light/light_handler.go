package light

import (
	"encoding/binary"
	"math"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// LightingHandler defines the interface for the scene's lighting subsystem.
//
// The LightingHandler manages the light list, ambient color, shadow mapping
// configuration, Forward+ tile culling state, and all associated GPU resources
// (bind group providers, pipeline keys, shadow textures). It is created via
// NewLightingHandler with builder options and attached to a scene via
// WithLighting. GPU resources are initialized lazily by the scene when the
// first light is added.
//
// Thread safety is provided by the owning scene's mutex — the handler itself
// does not perform internal locking.
type LightingHandler interface {
	// Enabled returns whether the lighting subsystem has been GPU-initialized and
	// is ready for rendering.
	//
	// Returns:
	//   - bool: true if lighting GPU resources have been initialized
	Enabled() bool

	// SetEnabled sets whether the lighting subsystem is GPU-initialized.
	//
	// Parameters:
	//   - enabled: true to mark as initialized
	SetEnabled(enabled bool)

	// Lights returns a copy of the current light list.
	//
	// Returns:
	//   - []Light: a copy of the lights registered with the handler
	Lights() []Light

	// AddLight appends a light to the handler's light list.
	//
	// Parameters:
	//   - l: the Light to add
	AddLight(l Light)

	// RemoveLight removes a light from the handler's list by reference equality.
	//
	// Parameters:
	//   - l: the Light to remove
	RemoveLight(l Light)

	// AmbientColor returns the scene's ambient light color.
	//
	// Returns:
	//   - [3]float32: the ambient RGB color
	AmbientColor() [3]float32

	// SetAmbientColor sets the scene's ambient light color.
	//
	// Parameters:
	//   - color: the ambient RGB color
	SetAmbientColor(color [3]float32)

	// Bgp retrieves the BindGroupProvider associated with the given key.
	// Returns nil if the key does not exist.
	//
	// Known keys:
	//   - "lights": light storage buffer BGP
	//   - "shadow_data": shadow depth pass BGP (light VP uniform)
	//   - "shadow_lit": lit pass shadow sampling BGP (texture + sampler + uniform)
	//   - "light_cull": compute shader cull BGP
	//   - "tile_lit": fragment shader tile data BGP
	//   - "ssao_lit": lit pass SSAO occlusion texture + sampler BGP (real or fallback white)
	//
	// Parameters:
	//   - key: the bind group provider identifier
	//
	// Returns:
	//   - bind_group_provider.BindGroupProvider: the provider, or nil if not found
	Bgp(key string) bind_group_provider.BindGroupProvider

	// Bgps returns the full map of bind group providers.
	//
	// Returns:
	//   - map[string]bind_group_provider.BindGroupProvider: all registered providers
	Bgps() map[string]bind_group_provider.BindGroupProvider

	// PipelineKey retrieves the pipeline key associated with the given name.
	// Returns an empty string if the name does not exist.
	//
	// Parameters:
	//   - name: the pipeline name
	//
	// Returns:
	//   - string: the pipeline key, or empty if not found
	PipelineKey(name string) string

	// PipelineKeys returns the full map of pipeline keys.
	//
	// Returns:
	//   - map[string]string: all registered pipeline name-to-key mappings
	PipelineKeys() map[string]string

	// SetPipelineKey stores a pipeline key under the given name.
	//
	// Parameters:
	//   - name: the pipeline name
	//   - key: the pipeline key
	SetPipelineKey(name, key string)

	// ScreenWidth returns the current screen width in pixels used for tile calculations.

	// ScreenWidth returns the current screen width in pixels used for tile calculations.
	//
	// Returns:
	//   - int: the screen width
	ScreenWidth() int

	// ScreenHeight returns the current screen height in pixels used for tile calculations.
	//
	// Returns:
	//   - int: the screen height
	ScreenHeight() int

	// TileCountX returns the number of Forward+ tile columns for the current screen width.
	//
	// Returns:
	//   - int: the tile column count
	TileCountX() int

	// TileCountY returns the number of Forward+ tile rows for the current screen height.
	//
	// Returns:
	//   - int: the tile row count
	TileCountY() int

	// TileSize returns the Forward+ tile size in pixels.
	//
	// Returns:
	//   - int: the tile width and height in pixels
	TileSize() int

	// MaxLightsPerTile returns the maximum number of light indices stored per tile.
	//
	// Returns:
	//   - int: the max light indices per tile
	MaxLightsPerTile() int

	// Resize updates the screen dimensions and recalculates tile counts for Forward+
	// light culling. This should be called when the window is resized.
	//
	// Parameters:
	//   - width: the new screen width in pixels
	//   - height: the new screen height in pixels
	Resize(width, height int)

	// ShadowHandler returns the ShadowHandler attached to this lighting
	// subsystem, managing shadow frustum configuration, PCF parameters,
	// and CSM GPU resources.
	//
	// Returns:
	//   - ShadowHandler: the shadow handler
	ShadowHandler() ShadowHandler

	// ContactShadowHandler returns the ContactShadowHandler attached to this
	// lighting subsystem, or nil if contact shadows are not configured.
	//
	// Returns:
	//   - ContactShadowHandler: the contact shadow handler, or nil
	ContactShadowHandler() ContactShadowHandler

	// MaxGPULights returns the maximum number of lights that can be marshaled
	// into the GPU storage buffer per frame.
	//
	// Returns:
	//   - int: the max GPU lights cap
	MaxGPULights() int

	// SetMaxGPULights sets the maximum number of lights that can be marshaled
	// into the GPU storage buffer per frame.
	//
	// Parameters:
	//   - max: the new max GPU lights cap
	SetMaxGPULights(max int)

	// MarshalLightBuffer marshals a slice of enabled lights into a byte buffer
	// suitable for GPU upload. The buffer layout is:
	//
	//   [GPULightHeader (16 bytes)] [GPULight × count (64 bytes each)]
	//
	// Only enabled lights are included, up to MaxGPULights. Lights beyond the
	// budget are silently dropped; callers should pre-sort by priority if
	// truncation is expected. The handler's ambient color and max GPU lights
	// cap are used internally.
	//
	// Parameters:
	//   - lights: the slice of lights to marshal (only enabled lights are included)
	//   - shadowIndices: optional map from Light to shadow entry index (may be nil)
	//
	// Returns:
	//   - []byte: the marshaled buffer ready for GPU upload
	MarshalLightBuffer(lights []Light, shadowIndices map[Light]uint32) []byte
}

var _ LightingHandler = &lightingHandlerImpl{}

func (h *lightingHandlerImpl) Enabled() bool                                          { return h.enabled }
func (h *lightingHandlerImpl) SetEnabled(enabled bool)                                { h.enabled = enabled }
func (h *lightingHandlerImpl) AmbientColor() [3]float32                               { return h.ambientColor }
func (h *lightingHandlerImpl) SetAmbientColor(color [3]float32)                       { h.ambientColor = color }
func (h *lightingHandlerImpl) Bgps() map[string]bind_group_provider.BindGroupProvider { return h.bgps }
func (h *lightingHandlerImpl) PipelineKey(name string) string                         { return h.pipelineKeys[name] }
func (h *lightingHandlerImpl) PipelineKeys() map[string]string                        { return h.pipelineKeys }
func (h *lightingHandlerImpl) SetPipelineKey(name, key string)                        { h.pipelineKeys[name] = key }
func (h *lightingHandlerImpl) ScreenWidth() int                                       { return h.screenWidth }
func (h *lightingHandlerImpl) ScreenHeight() int                                      { return h.screenHeight }
func (h *lightingHandlerImpl) TileCountX() int                                        { return h.tileCountX }
func (h *lightingHandlerImpl) TileCountY() int                                        { return h.tileCountY }
func (h *lightingHandlerImpl) TileSize() int                                          { return h.tileSize }
func (h *lightingHandlerImpl) MaxLightsPerTile() int                                  { return h.maxLightsPerTile }
func (h *lightingHandlerImpl) MaxGPULights() int                                      { return h.maxGPULights }
func (h *lightingHandlerImpl) SetMaxGPULights(max int)                                { h.maxGPULights = max }
func (h *lightingHandlerImpl) ShadowHandler() ShadowHandler                           { return h.shadowHandler }

func (h *lightingHandlerImpl) ContactShadowHandler() ContactShadowHandler {
	return h.contactShadowHandler
}
func (h *lightingHandlerImpl) Bgp(key string) bind_group_provider.BindGroupProvider {
	return h.bgps[key]
}

func (h *lightingHandlerImpl) Lights() []Light {
	out := make([]Light, len(h.lights))
	copy(out, h.lights)
	return out
}

func (h *lightingHandlerImpl) AddLight(l Light) {
	h.lights = append(h.lights, l)
}

func (h *lightingHandlerImpl) RemoveLight(l Light) {
	for i, existing := range h.lights {
		if existing == l {
			h.lights = append(h.lights[:i], h.lights[i+1:]...)
			return
		}
	}
}

func (h *lightingHandlerImpl) Resize(width, height int) {
	h.screenWidth = width
	h.screenHeight = height
	h.tileCountX = (width + h.tileSize - 1) / h.tileSize
	h.tileCountY = (height + h.tileSize - 1) / h.tileSize
}

func (h *lightingHandlerImpl) MarshalLightBuffer(lights []Light, shadowIndices map[Light]uint32) []byte {
	headerSize := (&GPULightHeader{}).Size()
	lightSize := (&GPULight{}).Size()
	maxLights := int(h.maxGPULights)

	// Pre-count enabled lights to size the buffer.
	enabledCount := 0
	for _, l := range lights {
		if l.Enabled() {
			enabledCount++
			if enabledCount >= maxLights {
				break
			}
		}
	}

	buf := make([]byte, headerSize+enabledCount*lightSize)

	// Write header.
	binary.LittleEndian.PutUint32(buf[0:4], math.Float32bits(h.ambientColor[0]))
	binary.LittleEndian.PutUint32(buf[4:8], math.Float32bits(h.ambientColor[1]))
	binary.LittleEndian.PutUint32(buf[8:12], math.Float32bits(h.ambientColor[2]))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(enabledCount))

	// Write each enabled light.
	offset := headerSize
	written := 0
	for _, l := range lights {
		if !l.Enabled() {
			continue
		}
		if written >= maxLights {
			break
		}
		gpu := ToGPULight(l)
		if shadowIndices != nil {
			if idx, ok := shadowIndices[l]; ok {
				gpu.ShadowIndex = idx
			}
		}
		copy(buf[offset:offset+lightSize], gpu.Marshal())
		offset += lightSize
		written++
	}

	return buf
}
