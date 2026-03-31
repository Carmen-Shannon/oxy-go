package loader

import (
	"fmt"
	"math"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// gltfMeshExtractorImpl is the implementation of the gltfMeshExtractor interface.
type gltfMeshExtractorImpl struct {
	parser gltfParser
}

// gltfMeshExtractor defines the interface for extracting mesh data from a parsed glTF document.
// It converts raw glTF accessor data into engine-ready ImportedMesh structs.
type gltfMeshExtractor interface {
	// ExtractMesh extracts a single mesh by index.
	// Returns one ImportedMesh per primitive (glTF meshes can have multiple primitives).
	//
	// Parameters:
	//   - meshIndex: the index of the mesh to extract
	//
	// Returns:
	//   - []model.ImportedMesh: one ImportedMesh per primitive
	//   - error: error if extraction fails
	ExtractMesh(meshIndex int) ([]model.ImportedMesh, error)

	// ExtractAllMeshes extracts all meshes from the document.
	// Returns a flattened slice with one ImportedMesh per primitive across all meshes.
	//
	// Returns:
	//   - []model.ImportedMesh: all meshes (flattened, one per primitive)
	//   - error: error if extraction fails
	ExtractAllMeshes() ([]model.ImportedMesh, error)
}

var _ gltfMeshExtractor = &gltfMeshExtractorImpl{}

// newGLTFMeshExtractor creates a new mesh extractor for a parsed document.
//
// Parameters:
//   - parser: the parser containing a loaded document
//
// Returns:
//   - gltfMeshExtractor: the mesh extractor
func newGLTFMeshExtractor(parser gltfParser) gltfMeshExtractor {
	return &gltfMeshExtractorImpl{parser: parser}
}

func (e *gltfMeshExtractorImpl) ExtractMesh(meshIndex int) ([]model.ImportedMesh, error) {
	doc := e.parser.Document()
	if doc == nil {
		return nil, fmt.Errorf("no document loaded")
	}
	if meshIndex < 0 || meshIndex >= len(doc.Meshes) {
		return nil, fmt.Errorf("mesh index %d out of range", meshIndex)
	}

	mesh := &doc.Meshes[meshIndex]
	var result []model.ImportedMesh

	for primIdx := range mesh.Primitives {
		prim := &mesh.Primitives[primIdx]
		imported, err := e.extractPrimitive(prim, mesh.Name, primIdx)
		if err != nil {
			return nil, fmt.Errorf("mesh %d primitive %d: %w", meshIndex, primIdx, err)
		}
		result = append(result, *imported)
	}

	return result, nil
}

func (e *gltfMeshExtractorImpl) ExtractAllMeshes() ([]model.ImportedMesh, error) {
	doc := e.parser.Document()
	if doc == nil {
		return nil, fmt.Errorf("no document loaded")
	}

	// Fallback: no active scene → iterate raw mesh list with identity transform.
	if doc.Scene == nil || len(doc.Scenes) == 0 {
		var allMeshes []model.ImportedMesh
		for i := range doc.Meshes {
			meshes, err := e.ExtractMesh(i)
			if err != nil {
				return nil, fmt.Errorf("mesh %d: %w", i, err)
			}
			allMeshes = append(allMeshes, meshes...)
		}
		return allMeshes, nil
	}

	scene := &doc.Scenes[*doc.Scene]
	var identity [16]float32
	common.Identity(identity[:])

	var allMeshes []model.ImportedMesh
	for _, nodeIdx := range scene.Nodes {
		meshes, err := e.gltfWalkNode(doc, nodeIdx, identity)
		if err != nil {
			return nil, err
		}
		allMeshes = append(allMeshes, meshes...)
	}

	return allMeshes, nil
}

// gltfWalkNode recursively traverses the node hierarchy of the active scene.
// It accumulates world transforms and extracts mesh primitives at each mesh node.
// For nodes with a skin the world transform is not applied to vertex data, because
// the skeleton drives world positioning at runtime.
//
// Parameters:
//   - doc: the parsed glTF document
//   - nodeIdx: index into doc.Nodes for the current node
//   - parentWorld: accumulated column-major 4×4 world transform from ancestor nodes
//
// Returns:
//   - []model.ImportedMesh: all meshes extracted from this node and its descendants
//   - error: error if any extraction step fails
func (e *gltfMeshExtractorImpl) gltfWalkNode(doc *gltfDocument, nodeIdx int, parentWorld [16]float32) ([]model.ImportedMesh, error) {
	if nodeIdx < 0 || nodeIdx >= len(doc.Nodes) {
		return nil, fmt.Errorf("node index %d out of range", nodeIdx)
	}
	node := &doc.Nodes[nodeIdx]

	// Accumulate world transform: worldMat = parentWorld × nodeLocal
	local := gltfNodeLocalMatrix(node)
	var worldMat [16]float32
	common.Mul4(worldMat[:], parentWorld[:], local[:])

	var result []model.ImportedMesh

	if node.Mesh != nil {
		meshIdx := *node.Mesh
		if meshIdx < 0 || meshIdx >= len(doc.Meshes) {
			return nil, fmt.Errorf("node %d references out-of-range mesh %d", nodeIdx, meshIdx)
		}
		mesh := &doc.Meshes[meshIdx]

		for primIdx := range mesh.Primitives {
			prim := &mesh.Primitives[primIdx]
			imported, err := e.extractPrimitive(prim, mesh.Name, primIdx)
			if err != nil {
				return nil, fmt.Errorf("node %d mesh %d primitive %d: %w", nodeIdx, meshIdx, primIdx, err)
			}

			// Skinned nodes: the skeleton repositions vertices at runtime; do not
			// bake the world transform into vertex data.
			if node.Skin == nil {
				gltfApplyWorldTransform(imported, worldMat)
			}

			result = append(result, *imported)
		}
	}

	for _, childIdx := range node.Children {
		childMeshes, err := e.gltfWalkNode(doc, childIdx, worldMat)
		if err != nil {
			return nil, err
		}
		result = append(result, childMeshes...)
	}

	return result, nil
}

// extractPrimitive extracts a single primitive as an ImportedMesh.
func (e *gltfMeshExtractorImpl) extractPrimitive(prim *gltfPrimitive, meshName string, primIndex int) (*model.ImportedMesh, error) {
	// Check for triangle mode (default is TRIANGLES)
	if prim.Mode != nil && *prim.Mode != gltfPrimitiveModeTriangles {
		return nil, fmt.Errorf("unsupported primitive mode: %d (only triangles supported)", *prim.Mode)
	}

	// Extract positions (required)
	posAccessor, ok := prim.Attributes["POSITION"]
	if !ok {
		return nil, fmt.Errorf("primitive has no POSITION attribute")
	}

	positions, err := e.parser.ReadVec3Accessor(posAccessor)
	if err != nil {
		return nil, fmt.Errorf("failed to read positions: %w", err)
	}

	// Initialize vertices with positions
	vertexCount := len(positions)
	vertices := make([]model.GPUSkinnedVertex, vertexCount)
	for i, pos := range positions {
		vertices[i].Position = pos
		vertices[i].Color = [4]float32{1, 1, 1, 1}
	}

	// Extract normals (optional — generated from geometry if absent)
	hasNormals := false
	if normalAccessor, ok := prim.Attributes["NORMAL"]; ok {
		normals, err := e.parser.ReadVec3Accessor(normalAccessor)
		if err != nil {
			return nil, fmt.Errorf("failed to read normals: %w", err)
		}
		for i := range normals {
			if i < vertexCount {
				vertices[i].Normal = normals[i]
			}
		}
		hasNormals = true
	}

	// Extract texture coordinates (optional)
	if texCoordAccessor, ok := prim.Attributes["TEXCOORD_0"]; ok {
		texCoords, err := e.parser.ReadVec2Accessor(texCoordAccessor)
		if err != nil {
			return nil, fmt.Errorf("failed to read texcoords: %w", err)
		}
		for i := range texCoords {
			if i < vertexCount {
				vertices[i].TexCoord = texCoords[i]
			}
		}
	}

	// Extract vertex colors (optional)
	if colorAccessor, ok := prim.Attributes["COLOR_0"]; ok {
		colors, err := e.readColorAccessor(colorAccessor)
		if err != nil {
			return nil, fmt.Errorf("failed to read colors: %w", err)
		}
		for i := range colors {
			if i < vertexCount {
				vertices[i].Color = colors[i]
			}
		}
	}

	// Extract tangent vectors (optional, for normal mapping).
	// glTF TANGENT is VEC4: xyz = tangent direction, w = handedness (±1).
	hasTangents := false
	if tangentAccessor, ok := prim.Attributes["TANGENT"]; ok {
		tangents, err := e.parser.ReadVec4Accessor(tangentAccessor)
		if err != nil {
			return nil, fmt.Errorf("failed to read tangents: %w", err)
		}
		for i := range tangents {
			if i < vertexCount {
				vertices[i].Tangent = tangents[i]
			}
		}
		hasTangents = true
	}

	// Extract bone indices (optional, for skeletal animation)
	if jointsAccessor, ok := prim.Attributes["JOINTS_0"]; ok {
		joints, err := e.parser.ReadJointsAccessor(jointsAccessor)
		if err != nil {
			return nil, fmt.Errorf("failed to read joints: %w", err)
		}
		for i := range joints {
			if i < vertexCount {
				vertices[i].BoneIndices = joints[i]
			}
		}
	}

	// Extract bone weights (optional, for skeletal animation)
	if weightsAccessor, ok := prim.Attributes["WEIGHTS_0"]; ok {
		weights, err := e.parser.ReadVec4Accessor(weightsAccessor)
		if err != nil {
			return nil, fmt.Errorf("failed to read weights: %w", err)
		}
		for i := range weights {
			if i < vertexCount {
				vertices[i].BoneWeights = weights[i]
			}
		}
	}

	// Extract indices
	var indices []uint32
	if prim.Indices != nil {
		indices, err = e.parser.ReadIndicesAccessor(*prim.Indices)
		if err != nil {
			return nil, fmt.Errorf("failed to read indices: %w", err)
		}
	} else {
		// Generate sequential indices if none provided
		indices = make([]uint32, vertexCount)
		for i := range indices {
			indices[i] = uint32(i)
		}
	}

	// Generate smooth vertex normals from triangle geometry when the glTF
	// file omits the NORMAL attribute. Must run before tangent generation
	// because generateTangents orthonormalizes tangents against the normal.
	if !hasNormals && len(indices) >= 3 {
		generateNormals(vertices, indices)
	}

	// Generate tangent vectors when the glTF file does not provide them.
	// Uses the MikkTSpace-compatible algorithm: per-triangle UV gradients define the
	// tangent and bitangent directions, which are accumulated per-vertex and then
	// orthonormalized against the vertex normal.
	if !hasTangents && len(indices) >= 3 {
		generateTangents(vertices, indices)
	}

	// Calculate bounding box
	bmin, bmax := gltfCalculateBoundingBox(positions)

	// Determine material index
	materialIndex := 0
	if prim.Material != nil {
		materialIndex = *prim.Material
	}

	// Build mesh name
	name := meshName
	if name == "" {
		name = fmt.Sprintf("mesh_%d", primIndex)
	}
	if len(prim.Attributes) > 0 && primIndex > 0 {
		name = fmt.Sprintf("%s_prim%d", name, primIndex)
	}

	return &model.ImportedMesh{
		Name:          name,
		Vertices:      vertices,
		Indices:       indices,
		MaterialIndex: materialIndex,
		BoundingMin:   bmin,
		BoundingMax:   bmax,
	}, nil
}

// readColorAccessor reads a color accessor, handling various formats.
// glTF colors can be VEC3 or VEC4, and can be float or normalized int.
func (e *gltfMeshExtractorImpl) readColorAccessor(accessorIndex int) ([][4]float32, error) {
	doc := e.parser.Document()
	acc := &doc.Accessors[accessorIndex]

	// Handle VEC4 FLOAT (most common)
	if acc.Type == gltfAccessorTypeVec4 && acc.ComponentType == gltfComponentTypeFloat {
		return e.parser.ReadVec4Accessor(accessorIndex)
	}

	// Handle VEC3 FLOAT (RGB, no alpha)
	if acc.Type == gltfAccessorTypeVec3 && acc.ComponentType == gltfComponentTypeFloat {
		vec3s, err := e.parser.ReadVec3Accessor(accessorIndex)
		if err != nil {
			return nil, err
		}
		result := make([][4]float32, len(vec3s))
		for i, v := range vec3s {
			result[i] = [4]float32{v[0], v[1], v[2], 1.0}
		}
		return result, nil
	}

	// Handle normalized unsigned byte (0-255 -> 0.0-1.0)
	if acc.ComponentType == gltfComponentTypeUnsignedByte {
		data, err := e.parser.ReadAccessorData(accessorIndex)
		if err != nil {
			return nil, err
		}
		result := make([][4]float32, acc.Count)
		switch acc.Type {
		case gltfAccessorTypeVec4:
			for i := 0; i < acc.Count; i++ {
				offset := i * 4
				result[i] = [4]float32{
					float32(data[offset]) / 255.0,
					float32(data[offset+1]) / 255.0,
					float32(data[offset+2]) / 255.0,
					float32(data[offset+3]) / 255.0,
				}
			}
		case gltfAccessorTypeVec3:
			for i := 0; i < acc.Count; i++ {
				offset := i * 3
				result[i] = [4]float32{
					float32(data[offset]) / 255.0,
					float32(data[offset+1]) / 255.0,
					float32(data[offset+2]) / 255.0,
					1.0,
				}
			}
		}
		return result, nil
	}

	// Handle normalized unsigned short (0-65535 -> 0.0-1.0)
	if acc.ComponentType == gltfComponentTypeUnsignedShort {
		data, err := e.parser.ReadAccessorData(accessorIndex)
		if err != nil {
			return nil, err
		}
		result := make([][4]float32, acc.Count)
		switch acc.Type {
		case gltfAccessorTypeVec4:
			for i := 0; i < acc.Count; i++ {
				offset := i * 8
				result[i] = [4]float32{
					float32(uint16(data[offset])|uint16(data[offset+1])<<8) / 65535.0,
					float32(uint16(data[offset+2])|uint16(data[offset+3])<<8) / 65535.0,
					float32(uint16(data[offset+4])|uint16(data[offset+5])<<8) / 65535.0,
					float32(uint16(data[offset+6])|uint16(data[offset+7])<<8) / 65535.0,
				}
			}
		case gltfAccessorTypeVec3:
			for i := 0; i < acc.Count; i++ {
				offset := i * 6
				result[i] = [4]float32{
					float32(uint16(data[offset])|uint16(data[offset+1])<<8) / 65535.0,
					float32(uint16(data[offset+2])|uint16(data[offset+3])<<8) / 65535.0,
					float32(uint16(data[offset+4])|uint16(data[offset+5])<<8) / 65535.0,
					1.0,
				}
			}
		}
		return result, nil
	}

	return nil, fmt.Errorf("unsupported color format: type=%s, componentType=%d", acc.Type, acc.ComponentType)
}

// gltfQuatToMatrix converts a unit quaternion (xyzw, GLTF convention) to a
// column-major 4×4 rotation matrix.
//
// Parameters:
//   - q: quaternion as [x, y, z, w]
//
// Returns:
//   - [16]float32: column-major 4×4 rotation matrix
func gltfQuatToMatrix(q [4]float32) [16]float32 {
	x, y, z, w := q[0], q[1], q[2], q[3]
	var m [16]float32
	m[0] = 1 - 2*(y*y+z*z)
	m[1] = 2 * (x*y + w*z)
	m[2] = 2 * (x*z - w*y)
	m[3] = 0
	m[4] = 2 * (x*y - w*z)
	m[5] = 1 - 2*(x*x+z*z)
	m[6] = 2 * (y*z + w*x)
	m[7] = 0
	m[8] = 2 * (x*z + w*y)
	m[9] = 2 * (y*z - w*x)
	m[10] = 1 - 2*(x*x+y*y)
	m[11] = 0
	m[12], m[13], m[14], m[15] = 0, 0, 0, 1
	return m
}

// gltfNodeLocalMatrix builds the local column-major 4×4 transform for a glTF node.
// If the node specifies a Matrix it is returned directly (GLTF stores it column-major).
// Otherwise the TRS components are composed as T × R × S using default identity
// values for any missing component.
//
// Parameters:
//   - node: the glTF node to compute a local matrix for
//
// Returns:
//   - [16]float32: column-major 4×4 local transform matrix
func gltfNodeLocalMatrix(node *gltfNode) [16]float32 {
	if node.Matrix != nil {
		return *node.Matrix
	}

	translation := [3]float32{0, 0, 0}
	rotation := [4]float32{0, 0, 0, 1}
	scale := [3]float32{1, 1, 1}

	if node.Translation != nil {
		translation = *node.Translation
	}
	if node.Rotation != nil {
		rotation = *node.Rotation
	}
	if node.Scale != nil {
		scale = *node.Scale
	}

	rot := gltfQuatToMatrix(rotation)
	sx, sy, sz := scale[0], scale[1], scale[2]
	tx, ty, tz := translation[0], translation[1], translation[2]

	var m [16]float32
	m[0] = rot[0] * sx
	m[1] = rot[1] * sx
	m[2] = rot[2] * sx
	m[3] = 0
	m[4] = rot[4] * sy
	m[5] = rot[5] * sy
	m[6] = rot[6] * sy
	m[7] = 0
	m[8] = rot[8] * sz
	m[9] = rot[9] * sz
	m[10] = rot[10] * sz
	m[11] = 0
	m[12] = tx
	m[13] = ty
	m[14] = tz
	m[15] = 1
	return m
}

// gltfTransformPoint applies a column-major 4×4 matrix to a 3D point (w=1).
//
// Parameters:
//   - m: column-major 4×4 transform matrix
//   - p: the 3D point to transform
//
// Returns:
//   - [3]float32: the transformed point
func gltfTransformPoint(m [16]float32, p [3]float32) [3]float32 {
	return [3]float32{
		m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12],
		m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13],
		m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14],
	}
}

// gltfTransformNormal applies the inverse-transpose of a column-major 4×4 matrix
// to a 3D normal vector and renormalizes. If the matrix is singular the original
// normal is returned unchanged.
//
// Parameters:
//   - m: column-major 4×4 world transform matrix
//   - n: the surface normal to transform
//
// Returns:
//   - [3]float32: the transformed and renormalized normal
func gltfTransformNormal(m [16]float32, n [3]float32) [3]float32 {
	var inv [16]float32
	if !common.Invert4(inv[:], m[:]) {
		return n
	}
	// Normal transform is M^{-T}: (M^{-T} × n)_i = Σ_j inv[i*4+j] * n[j]
	nx := inv[0]*n[0] + inv[1]*n[1] + inv[2]*n[2]
	ny := inv[4]*n[0] + inv[5]*n[1] + inv[6]*n[2]
	nz := inv[8]*n[0] + inv[9]*n[1] + inv[10]*n[2]

	length := float32(math.Sqrt(float64(nx*nx + ny*ny + nz*nz)))
	if length < 1e-6 {
		return n
	}
	invLen := 1.0 / length
	return [3]float32{nx * invLen, ny * invLen, nz * invLen}
}

// gltfApplyWorldTransform transforms all vertex positions, normals, and tangents
// in an imported mesh by the given column-major world matrix, then recomputes the
// axis-aligned bounding box from the transformed positions.
//
// Parameters:
//   - imported: the mesh whose vertex data will be transformed in-place
//   - worldMat: column-major 4×4 world transform to apply
func gltfApplyWorldTransform(imported *model.ImportedMesh, worldMat [16]float32) {
	bmin := [3]float32{float32(1e38), float32(1e38), float32(1e38)}
	bmax := [3]float32{float32(-1e38), float32(-1e38), float32(-1e38)}

	for i := range imported.Vertices {
		v := &imported.Vertices[i]

		v.Position = gltfTransformPoint(worldMat, v.Position)
		v.Normal = gltfTransformNormal(worldMat, v.Normal)

		txyz := gltfTransformNormal(worldMat, [3]float32{v.Tangent[0], v.Tangent[1], v.Tangent[2]})
		v.Tangent = [4]float32{txyz[0], txyz[1], txyz[2], v.Tangent[3]}

		for j := range 3 {
			if v.Position[j] < bmin[j] {
				bmin[j] = v.Position[j]
			}
			if v.Position[j] > bmax[j] {
				bmax[j] = v.Position[j]
			}
		}
	}

	imported.BoundingMin = bmin
	imported.BoundingMax = bmax
}

// gltfCalculateBoundingBox computes the axis-aligned bounding box for positions.
func gltfCalculateBoundingBox(positions [][3]float32) ([3]float32, [3]float32) {
	if len(positions) == 0 {
		return [3]float32{}, [3]float32{}
	}

	bmin := [3]float32{
		float32(math.MaxFloat32),
		float32(math.MaxFloat32),
		float32(math.MaxFloat32),
	}
	bmax := [3]float32{
		-float32(math.MaxFloat32),
		-float32(math.MaxFloat32),
		-float32(math.MaxFloat32),
	}

	for _, pos := range positions {
		for j := 0; j < 3; j++ {
			if pos[j] < bmin[j] {
				bmin[j] = pos[j]
			}
			if pos[j] > bmax[j] {
				bmax[j] = pos[j]
			}
		}
	}

	return bmin, bmax
}

// generateNormals computes smooth vertex normals from the triangle geometry when the
// glTF file does not provide a NORMAL attribute. For each triangle, the face normal is
// computed as the cross product of its two edges, then accumulated (area-weighted) onto
// every vertex of that triangle. All vertex normals are normalized at the end to produce
// smooth shading across shared vertices.
//
// Parameters:
//   - vertices: the vertex slice to write normal data into
//   - indices: the triangle index buffer (must be a multiple of 3)
func generateNormals(vertices []model.GPUSkinnedVertex, indices []uint32) {
	n := len(vertices)
	accum := make([][3]float32, n)

	for i := 0; i+2 < len(indices); i += 3 {
		i0, i1, i2 := indices[i], indices[i+1], indices[i+2]
		if int(i0) >= n || int(i1) >= n || int(i2) >= n {
			continue
		}

		p0, p1, p2 := vertices[i0].Position, vertices[i1].Position, vertices[i2].Position

		edge1 := [3]float32{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}
		edge2 := [3]float32{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]}

		// Cross product: face normal (length proportional to triangle area)
		faceNormal := [3]float32{
			edge1[1]*edge2[2] - edge1[2]*edge2[1],
			edge1[2]*edge2[0] - edge1[0]*edge2[2],
			edge1[0]*edge2[1] - edge1[1]*edge2[0],
		}

		for _, idx := range []uint32{i0, i1, i2} {
			accum[idx][0] += faceNormal[0]
			accum[idx][1] += faceNormal[1]
			accum[idx][2] += faceNormal[2]
		}
	}

	// Normalize accumulated normals
	for i := range n {
		length := float32(math.Sqrt(float64(accum[i][0]*accum[i][0] + accum[i][1]*accum[i][1] + accum[i][2]*accum[i][2])))
		if length < 1e-6 {
			// Degenerate: default to up vector
			vertices[i].Normal = [3]float32{0, 1, 0}
			continue
		}
		invLen := 1.0 / length
		vertices[i].Normal = [3]float32{
			accum[i][0] * invLen,
			accum[i][1] * invLen,
			accum[i][2] * invLen,
		}
	}
}

// generateTangents computes per-vertex tangent vectors from triangle topology using the
// MikkTSpace-compatible UV-gradient method. For each triangle the tangent and bitangent
// are derived from the UV coordinate differences, accumulated per-vertex, and then
// orthonormalized against the vertex normal. The W component stores handedness (±1).
//
// Parameters:
//   - vertices: the vertex slice to write tangent data into
//   - indices: the triangle index buffer (must be a multiple of 3)
func generateTangents(vertices []model.GPUSkinnedVertex, indices []uint32) {
	n := len(vertices)
	tan := make([][3]float32, n)  // accumulated tangent
	btan := make([][3]float32, n) // accumulated bitangent

	for i := 0; i+2 < len(indices); i += 3 {
		i0, i1, i2 := indices[i], indices[i+1], indices[i+2]
		if int(i0) >= n || int(i1) >= n || int(i2) >= n {
			continue
		}

		p0, p1, p2 := vertices[i0].Position, vertices[i1].Position, vertices[i2].Position
		uv0, uv1, uv2 := vertices[i0].TexCoord, vertices[i1].TexCoord, vertices[i2].TexCoord

		edge1 := [3]float32{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}
		edge2 := [3]float32{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]}

		duv1 := [2]float32{uv1[0] - uv0[0], uv1[1] - uv0[1]}
		duv2 := [2]float32{uv2[0] - uv0[0], uv2[1] - uv0[1]}

		det := duv1[0]*duv2[1] - duv1[1]*duv2[0]
		if det == 0 {
			continue
		}
		invDet := 1.0 / det

		t := [3]float32{
			invDet * (duv2[1]*edge1[0] - duv1[1]*edge2[0]),
			invDet * (duv2[1]*edge1[1] - duv1[1]*edge2[1]),
			invDet * (duv2[1]*edge1[2] - duv1[1]*edge2[2]),
		}
		b := [3]float32{
			invDet * (-duv2[0]*edge1[0] + duv1[0]*edge2[0]),
			invDet * (-duv2[0]*edge1[1] + duv1[0]*edge2[1]),
			invDet * (-duv2[0]*edge1[2] + duv1[0]*edge2[2]),
		}

		for _, idx := range []uint32{i0, i1, i2} {
			tan[idx][0] += t[0]
			tan[idx][1] += t[1]
			tan[idx][2] += t[2]
			btan[idx][0] += b[0]
			btan[idx][1] += b[1]
			btan[idx][2] += b[2]
		}
	}

	// Orthonormalize each vertex tangent against its normal and compute handedness.
	for i := 0; i < n; i++ {
		normal := vertices[i].Normal
		t := tan[i]

		// Gram-Schmidt orthonormalize: T' = normalize(T - N * dot(N, T))
		nDotT := normal[0]*t[0] + normal[1]*t[1] + normal[2]*t[2]
		ortho := [3]float32{
			t[0] - normal[0]*nDotT,
			t[1] - normal[1]*nDotT,
			t[2] - normal[2]*nDotT,
		}

		length := float32(math.Sqrt(float64(ortho[0]*ortho[0] + ortho[1]*ortho[1] + ortho[2]*ortho[2])))
		if length < 1e-6 {
			// Degenerate tangent: use a default tangent perpendicular to the normal.
			vertices[i].Tangent = [4]float32{1, 0, 0, 1}
			continue
		}
		invLen := 1.0 / length
		ortho[0] *= invLen
		ortho[1] *= invLen
		ortho[2] *= invLen

		// Handedness: sign of dot(cross(N, T), B) determines if bitangent is flipped.
		cross := [3]float32{
			normal[1]*ortho[2] - normal[2]*ortho[1],
			normal[2]*ortho[0] - normal[0]*ortho[2],
			normal[0]*ortho[1] - normal[1]*ortho[0],
		}
		w := float32(1.0)
		if cross[0]*btan[i][0]+cross[1]*btan[i][1]+cross[2]*btan[i][2] < 0 {
			w = -1.0
		}

		vertices[i].Tangent = [4]float32{ortho[0], ortho[1], ortho[2], w}
	}
}
