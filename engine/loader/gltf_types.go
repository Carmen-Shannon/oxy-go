// gltf_types.go contains glTF 2.0 spec data structures for JSON deserialization.
// These types map directly to the glTF 2.0 JSON schema and are internal to the loader package.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
package loader

// --- glTF Root Structure ---

// gltfDocument represents the root of a glTF JSON document.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-gltf
type gltfDocument struct {
	// Asset contains metadata about the glTF asset.
	Asset gltfAsset `json:"asset"`

	// Scene is the index of the default scene.
	Scene *int `json:"scene,omitempty"`

	// Scenes is an array of scenes.
	Scenes []gltfScene `json:"scenes,omitempty"`

	// Nodes is an array of nodes (transform hierarchy).
	Nodes []gltfNode `json:"nodes,omitempty"`

	// Meshes is an array of meshes.
	Meshes []gltfMesh `json:"meshes,omitempty"`

	// Accessors define how to interpret buffer data.
	Accessors []gltfAccessor `json:"accessors,omitempty"`

	// BufferViews define portions of buffers.
	BufferViews []gltfBufferView `json:"bufferViews,omitempty"`

	// Buffers are raw binary data containers.
	Buffers []gltfBuffer `json:"buffers,omitempty"`

	// Materials is an array of materials.
	Materials []gltfMaterial `json:"materials,omitempty"`

	// Textures is an array of textures.
	Textures []gltfTexture `json:"textures,omitempty"`

	// Images is an array of images.
	Images []gltfImage `json:"images,omitempty"`

	// Samplers define texture sampling parameters.
	Samplers []gltfSampler `json:"samplers,omitempty"`

	// Skins is an array of skins (skeletal animation binding).
	Skins []gltfSkin `json:"skins,omitempty"`

	// Animations is an array of animations.
	Animations []gltfAnimation `json:"animations,omitempty"`

	// ExtensionsUsed lists extensions used by this asset.
	ExtensionsUsed []string `json:"extensionsUsed,omitempty"`

	// ExtensionsRequired lists extensions required to load this asset.
	ExtensionsRequired []string `json:"extensionsRequired,omitempty"`
}

// --- Asset Metadata ---

// gltfAsset contains metadata about the glTF asset.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-asset
type gltfAsset struct {
	// Version is the glTF version (required, must be "2.0").
	Version string `json:"version"`

	// MinVersion is the minimum glTF version required.
	MinVersion string `json:"minVersion,omitempty"`

	// Generator is the tool that generated this asset.
	Generator string `json:"generator,omitempty"`

	// Copyright information.
	Copyright string `json:"copyright,omitempty"`
}

// --- Scene Graph ---

// gltfScene is a set of visual objects to render.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-scene
type gltfScene struct {
	// Name is an optional name for this scene.
	Name string `json:"name,omitempty"`

	// Nodes are the indices of root nodes in this scene.
	Nodes []int `json:"nodes,omitempty"`
}

// gltfNode is a node in the node hierarchy.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-node
type gltfNode struct {
	// Name is an optional name for this node.
	Name string `json:"name,omitempty"`

	// Children are indices of child nodes.
	Children []int `json:"children,omitempty"`

	// Mesh is the index of the mesh in this node.
	Mesh *int `json:"mesh,omitempty"`

	// Skin is the index of the skin for this node (skeletal animation).
	Skin *int `json:"skin,omitempty"`

	// Matrix is a 4x4 transformation matrix (column-major).
	Matrix *[16]float32 `json:"matrix,omitempty"`

	// Translation is the node's translation (x, y, z).
	Translation *[3]float32 `json:"translation,omitempty"`

	// Rotation is the node's rotation as a quaternion (x, y, z, w).
	Rotation *[4]float32 `json:"rotation,omitempty"`

	// Scale is the node's scale (x, y, z).
	Scale *[3]float32 `json:"scale,omitempty"`

	// Weights are morph target weights (for blend shapes).
	Weights []float32 `json:"weights,omitempty"`
}

// --- Mesh Data ---

// gltfMesh is a set of primitives to be rendered.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-mesh
type gltfMesh struct {
	// Name is an optional name for this mesh.
	Name string `json:"name,omitempty"`

	// Primitives defines the geometry to render.
	Primitives []gltfPrimitive `json:"primitives"`

	// Weights are default morph target weights.
	Weights []float32 `json:"weights,omitempty"`
}

// gltfPrimitive defines geometry for rendering.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-mesh-primitive
type gltfPrimitive struct {
	// Attributes is a map of attribute semantic to accessor index.
	// Standard attributes: POSITION, NORMAL, TANGENT, TEXCOORD_0, COLOR_0, JOINTS_0, WEIGHTS_0
	Attributes map[string]int `json:"attributes"`

	// Indices is the accessor index for the index buffer.
	Indices *int `json:"indices,omitempty"`

	// Material is the material index.
	Material *int `json:"material,omitempty"`

	// Mode is the primitive topology.
	// 0=POINTS, 1=LINES, 2=LINE_LOOP, 3=LINE_STRIP, 4=TRIANGLES (default), 5=TRIANGLE_STRIP, 6=TRIANGLE_FAN
	Mode *int `json:"mode,omitempty"`

	// Targets are morph targets for this primitive.
	Targets []map[string]int `json:"targets,omitempty"`
}

// PrimitiveMode constants
const (
	// gltfPrimitiveModePoints        = 0
	// gltfPrimitiveModeLines         = 1
	// gltfPrimitiveModeLineLoop      = 2
	// gltfPrimitiveModeLineStrip     = 3
	gltfPrimitiveModeTriangles = 4
	// gltfPrimitiveModeTriangleStrip = 5
	// gltfPrimitiveModeTriangleFan   = 6
)

// --- Buffer Data ---

// gltfAccessor defines how to interpret buffer data.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-accessor
type gltfAccessor struct {
	// Name is an optional name.
	Name string `json:"name,omitempty"`

	// BufferView is the index of the bufferView.
	BufferView *int `json:"bufferView,omitempty"`

	// ByteOffset is the offset within the bufferView.
	ByteOffset int `json:"byteOffset,omitempty"`

	// ComponentType is the data type of components.
	// 5120=BYTE, 5121=UNSIGNED_BYTE, 5122=SHORT, 5123=UNSIGNED_SHORT, 5125=UNSIGNED_INT, 5126=FLOAT
	ComponentType int `json:"componentType"`

	// Normalized indicates if integer data should be normalized.
	Normalized bool `json:"normalized,omitempty"`

	// Count is the number of elements.
	Count int `json:"count"`

	// Type is the element type (SCALAR, VEC2, VEC3, VEC4, MAT2, MAT3, MAT4).
	Type string `json:"type"`

	// Max is the maximum value of each component.
	Max []float32 `json:"max,omitempty"`

	// Min is the minimum value of each component.
	Min []float32 `json:"min,omitempty"`

	// Sparse defines sparse storage of accessor values.
	Sparse *gltfAccessorSparse `json:"sparse,omitempty"`
}

// ComponentType constants
const (
	gltfComponentTypeByte          = 5120
	gltfComponentTypeUnsignedByte  = 5121
	gltfComponentTypeShort         = 5122
	gltfComponentTypeUnsignedShort = 5123
	gltfComponentTypeUnsignedInt   = 5125
	gltfComponentTypeFloat         = 5126
)

// AccessorType constants
const (
	gltfAccessorTypeScalar = "SCALAR"
	gltfAccessorTypeVec2   = "VEC2"
	gltfAccessorTypeVec3   = "VEC3"
	gltfAccessorTypeVec4   = "VEC4"
	gltfAccessorTypeMat2   = "MAT2"
	gltfAccessorTypeMat3   = "MAT3"
	gltfAccessorTypeMat4   = "MAT4"
)

// gltfAccessorSparse defines sparse storage.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-accessor-sparse
//
// NOTE: Only Count is retained for deserialization. The parser does not support sparse
// accessors and returns an error when Sparse is non-nil. The Indices/Values sub-types
// were removed because they are never read; encoding/json silently ignores unknown fields.
type gltfAccessorSparse struct {
	// Count is the number of sparse entries.
	Count int `json:"count"`
}

// gltfBufferView represents a subset of a buffer.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-bufferview
type gltfBufferView struct {
	// Name is an optional name.
	Name string `json:"name,omitempty"`

	// Buffer is the index of the buffer.
	Buffer int `json:"buffer"`

	// ByteOffset is the offset into the buffer.
	ByteOffset int `json:"byteOffset,omitempty"`

	// ByteLength is the length of the bufferView.
	ByteLength int `json:"byteLength"`

	// ByteStride is the stride for interleaved data (optional).
	ByteStride *int `json:"byteStride,omitempty"`

	// Target is the intended GPU buffer type.
	// 34962=ARRAY_BUFFER, 34963=ELEMENT_ARRAY_BUFFER
	Target *int `json:"target,omitempty"`
}

// gltfBuffer represents binary data.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-buffer
type gltfBuffer struct {
	// Name is an optional name.
	Name string `json:"name,omitempty"`

	// URI is the URI of the buffer data (can be data: URI or external file).
	URI string `json:"uri,omitempty"`

	// ByteLength is the length of the buffer.
	ByteLength int `json:"byteLength"`

	// Data holds the loaded binary data (not part of JSON, populated during load).
	Data []byte `json:"-"`
}

// --- Materials and Textures ---

// gltfMaterial defines the material appearance of a primitive.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-material
type gltfMaterial struct {
	// Name is an optional name.
	Name string `json:"name,omitempty"`

	// PbrMetallicRoughness is the PBR metallic-roughness model.
	PbrMetallicRoughness *gltfPbrMetallicRoughness `json:"pbrMetallicRoughness,omitempty"`

	// NormalTexture is the normal map.
	NormalTexture *gltfNormalTextureInfo `json:"normalTexture,omitempty"`

	// TODO: Uncomment when PBR rendering is implemented:
	// // OcclusionTexture is the occlusion map.
	// OcclusionTexture *gltfOcclusionTextureInfo `json:"occlusionTexture,omitempty"`
	//
	// // EmissiveTexture is the emissive map.
	// EmissiveTexture *gltfTextureInfo `json:"emissiveTexture,omitempty"`
	//
	// // EmissiveFactor is the emissive color (RGB).
	// EmissiveFactor *[3]float32 `json:"emissiveFactor,omitempty"`
	//
	// // AlphaMode is the alpha rendering mode.
	// // "OPAQUE" (default), "MASK", "BLEND"
	// AlphaMode string `json:"alphaMode,omitempty"`
	//
	// // AlphaCutoff is the alpha cutoff for MASK mode.
	// AlphaCutoff *float32 `json:"alphaCutoff,omitempty"`
	//
	// // DoubleSided indicates if the material is double-sided.
	// DoubleSided bool `json:"doubleSided,omitempty"`
}

// gltfPbrMetallicRoughness is the metallic-roughness material model.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-material-pbrmetallicroughness
type gltfPbrMetallicRoughness struct {
	// BaseColorFactor is the base color (RGBA).
	BaseColorFactor *[4]float32 `json:"baseColorFactor,omitempty"`

	// BaseColorTexture is the base color texture.
	BaseColorTexture *gltfTextureInfo `json:"baseColorTexture,omitempty"`

	// MetallicFactor is the metalness (0.0 = dielectric, 1.0 = metal).
	MetallicFactor *float32 `json:"metallicFactor,omitempty"`

	// RoughnessFactor is the roughness (0.0 = smooth, 1.0 = rough).
	RoughnessFactor *float32 `json:"roughnessFactor,omitempty"`

	// MetallicRoughnessTexture contains metallic (B) and roughness (G) channels.
	MetallicRoughnessTexture *gltfTextureInfo `json:"metallicRoughnessTexture,omitempty"`
}

// gltfTextureInfo references a texture.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-textureinfo
type gltfTextureInfo struct {
	// Index is the texture index.
	Index int `json:"index"`

	// TexCoord is the UV set to use (default 0).
	TexCoord int `json:"texCoord,omitempty"`
}

// gltfNormalTextureInfo references a normal map.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-material-normaltextureinfo
type gltfNormalTextureInfo struct {
	gltfTextureInfo

	// Scale is the normal scale factor.
	Scale *float32 `json:"scale,omitempty"`
}

// TODO: Uncomment when PBR rendering is implemented:
// // gltfOcclusionTextureInfo references an occlusion map.
// // Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-material-occlusiontextureinfo
// type gltfOcclusionTextureInfo struct {
// 	gltfTextureInfo
//
// 	// Strength is the occlusion strength.
// 	Strength *float32 `json:"strength,omitempty"`
// }

// gltfTexture combines an image and a sampler.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-texture
type gltfTexture struct {
	// Name is an optional name.
	Name string `json:"name,omitempty"`

	// Sampler is the sampler index.
	Sampler *int `json:"sampler,omitempty"`

	// Source is the image index.
	Source *int `json:"source,omitempty"`
}

// gltfImage is a texture image source.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-image
type gltfImage struct {
	// Name is an optional name.
	Name string `json:"name,omitempty"`

	// URI is the image URI (can be data: URI or external file).
	URI string `json:"uri,omitempty"`

	// MimeType is the MIME type when embedded in a bufferView.
	MimeType string `json:"mimeType,omitempty"`

	// BufferView is the index of the bufferView containing the image.
	BufferView *int `json:"bufferView,omitempty"`
}

// gltfSampler defines texture sampling parameters.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-sampler
type gltfSampler struct {
	// Name is an optional name.
	Name string `json:"name,omitempty"`

	// MagFilter is the magnification filter.
	// 9728=NEAREST, 9729=LINEAR
	MagFilter *int `json:"magFilter,omitempty"`

	// MinFilter is the minification filter.
	// 9728=NEAREST, 9729=LINEAR, 9984-9987=mipmapped variants
	MinFilter *int `json:"minFilter,omitempty"`

	// WrapS is the U wrapping mode.
	// 33071=CLAMP_TO_EDGE, 33648=MIRRORED_REPEAT, 10497=REPEAT (default)
	WrapS *int `json:"wrapS,omitempty"`

	// WrapT is the V wrapping mode.
	WrapT *int `json:"wrapT,omitempty"`
}

// Sampler filter constants
const (
	gltfFilterNearest              = 9728
	gltfFilterLinear               = 9729
	gltfFilterNearestMipmapNearest = 9984
	gltfFilterLinearMipmapNearest  = 9985
	gltfFilterNearestMipmapLinear  = 9986
	gltfFilterLinearMipmapLinear   = 9987
)

// Sampler wrap constants
const (
	gltfWrapClampToEdge    = 33071
	gltfWrapMirroredRepeat = 33648
	gltfWrapRepeat         = 10497
)

// --- Skeletal Animation ---

// gltfSkin defines how a mesh is deformed by a skeleton.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-skin
type gltfSkin struct {
	// Name is an optional name.
	Name string `json:"name,omitempty"`

	// InverseBindMatrices is the accessor index for the inverse bind matrices.
	InverseBindMatrices *int `json:"inverseBindMatrices,omitempty"`

	// Skeleton is the node index of the skeleton root (optional).
	Skeleton *int `json:"skeleton,omitempty"`

	// Joints are the node indices of the skeleton joints (bones).
	Joints []int `json:"joints"`
}

// gltfAnimation defines keyframe animation.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-animation
type gltfAnimation struct {
	// Name is an optional name.
	Name string `json:"name,omitempty"`

	// Channels connect samplers to target nodes/properties.
	Channels []gltfAnimChannel `json:"channels"`

	// Samplers define the keyframe data.
	Samplers []gltfAnimSampler `json:"samplers"`
}

// gltfAnimChannel connects a sampler to a target.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-animation-channel
type gltfAnimChannel struct {
	// Sampler is the sampler index.
	Sampler int `json:"sampler"`

	// Target specifies what to animate.
	Target gltfAnimTarget `json:"target"`
}

// gltfAnimTarget specifies the animated property.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-animation-channel-target
type gltfAnimTarget struct {
	// Node is the target node index.
	Node *int `json:"node,omitempty"`

	// Path is the animated property.
	// "translation", "rotation", "scale", "weights"
	Path string `json:"path"`
}

// gltfAnimSampler defines animation keyframe data.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-animation-sampler
type gltfAnimSampler struct {
	// Input is the accessor index for keyframe times.
	Input int `json:"input"`

	// Output is the accessor index for keyframe values.
	Output int `json:"output"`

	// Interpolation mode: "LINEAR" (default), "STEP", "CUBICSPLINE".
	Interpolation string `json:"interpolation,omitempty"`
}

// Animation interpolation constants
// const (
// 	gltfAnimInterpolationLinear      = "LINEAR"
// 	gltfAnimInterpolationStep        = "STEP"
// 	gltfAnimInterpolationCubicSpline = "CUBICSPLINE"
// )

// Animation path constants
const (
	gltfAnimPathTranslation = "translation"
	gltfAnimPathRotation    = "rotation"
	gltfAnimPathScale       = "scale"
	gltfAnimPathWeights     = "weights"
)

// --- GLB Binary Format ---

// gltfGLBHeader is the header of a GLB file (12 bytes).
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#glb-file-format-specification
type gltfGLBHeader struct {
	Magic   uint32 // Must be 0x46546C67 ("glTF" in ASCII)
	Version uint32 // Must be 2
	Length  uint32 // Total file length
}

// gltfGLBChunkHeader is the header of a GLB chunk (8 bytes).
type gltfGLBChunkHeader struct {
	ChunkLength uint32
	ChunkType   uint32 // 0x4E4F534A for JSON, 0x004E4942 for BIN
}

// GLB magic number and chunk type constants
const (
	gltfGLBMagic     = 0x46546C67 // "glTF" in little-endian ASCII
	gltfGLBVersion   = 2
	gltfGLBChunkJSON = 0x4E4F534A // "JSON" in little-endian ASCII
	gltfGLBChunkBIN  = 0x004E4942 // "BIN\0" in little-endian ASCII
)
