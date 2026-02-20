package loader

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Common errors returned by the parser
var (
	errInvalidGLTFVersion = errors.New("invalid glTF version: must be 2.0")
	errInvalidGLBMagic    = errors.New("invalid GLB magic number")
	errInvalidGLBVersion  = errors.New("invalid GLB version: must be 2")
	errMissingJSONChunk   = errors.New("GLB file missing JSON chunk")
	errInvalidBufferURI   = errors.New("invalid buffer URI")
	errBufferSizeMismatch = errors.New("buffer size mismatch")
)

// gltfParserImpl is the implementation of the gltfParser interface.
type gltfParserImpl struct {
	baseDir        string
	document       *gltfDocument
	glbBinaryChunk []byte
}

// gltfParser defines the interface for loading and parsing glTF/GLB files.
// It handles file I/O, JSON deserialization, buffer loading, and typed accessor reads.
// This is internal to the loader package.
type gltfParser interface {
	// Parse loads and parses a glTF/GLB file from the given path.
	// Automatically detects .gltf (JSON) vs .glb (binary) format.
	//
	// Parameters:
	//   - path: path to the glTF or GLB file
	//
	// Returns:
	//   - error: error if parsing fails
	Parse(path string) error

	// ParseReader parses a glTF document from a reader.
	// Use this when loading from embedded resources or network streams.
	//
	// Parameters:
	//   - r: reader containing glTF JSON or GLB data
	//   - isGLB: true if the data is in GLB format
	//
	// Returns:
	//   - error: error if parsing fails
	ParseReader(r io.Reader, isGLB bool) error

	// Document returns the parsed glTF document.
	// Returns nil if Parse has not been called successfully.
	//
	// Returns:
	//   - *gltfDocument: the parsed document or nil
	Document() *gltfDocument

	// BaseDir returns the directory containing the loaded glTF file.
	// Used for resolving relative URIs to external resources.
	//
	// Returns:
	//   - string: the base directory path
	BaseDir() string

	// ReadAccessorData reads raw bytes from an accessor.
	//
	// Parameters:
	//   - accessorIndex: the index of the accessor
	//
	// Returns:
	//   - []byte: the raw data
	//   - error: error if reading fails
	ReadAccessorData(accessorIndex int) ([]byte, error)

	// ReadVec2Accessor reads an accessor as vec2 float data.
	//
	// Parameters:
	//   - accessorIndex: the index of the accessor
	//
	// Returns:
	//   - [][2]float32: the vec2 data
	//   - error: error if reading fails
	ReadVec2Accessor(accessorIndex int) ([][2]float32, error)

	// ReadVec3Accessor reads an accessor as vec3 float data.
	//
	// Parameters:
	//   - accessorIndex: the index of the accessor
	//
	// Returns:
	//   - [][3]float32: the vec3 data
	//   - error: error if reading fails
	ReadVec3Accessor(accessorIndex int) ([][3]float32, error)

	// ReadVec4Accessor reads an accessor as vec4 float data.
	//
	// Parameters:
	//   - accessorIndex: the index of the accessor
	//
	// Returns:
	//   - [][4]float32: the vec4 data
	//   - error: error if reading fails
	ReadVec4Accessor(accessorIndex int) ([][4]float32, error)

	// ReadScalarAccessor reads an accessor as scalar float data.
	//
	// Parameters:
	//   - accessorIndex: the index of the accessor
	//
	// Returns:
	//   - []float32: the scalar data
	//   - error: error if reading fails
	ReadScalarAccessor(accessorIndex int) ([]float32, error)

	// ReadMat4Accessor reads an accessor as mat4 float data.
	//
	// Parameters:
	//   - accessorIndex: the index of the accessor
	//
	// Returns:
	//   - [][16]float32: the mat4 data
	//   - error: error if reading fails
	ReadMat4Accessor(accessorIndex int) ([][16]float32, error)

	// ReadIndicesAccessor reads an accessor as index data (uint32).
	// Handles UNSIGNED_BYTE, UNSIGNED_SHORT, and UNSIGNED_INT component types.
	//
	// Parameters:
	//   - accessorIndex: the index of the accessor
	//
	// Returns:
	//   - []uint32: the index data (converted to uint32)
	//   - error: error if reading fails
	ReadIndicesAccessor(accessorIndex int) ([]uint32, error)

	// ReadJointsAccessor reads an accessor as joint indices (vec4 uint).
	// Handles UNSIGNED_BYTE and UNSIGNED_SHORT component types.
	//
	// Parameters:
	//   - accessorIndex: the index of the accessor
	//
	// Returns:
	//   - [][4]uint32: the joint indices (converted to uint32)
	//   - error: error if reading fails
	ReadJointsAccessor(accessorIndex int) ([][4]uint32, error)
}

var _ gltfParser = &gltfParserImpl{}

// newGLTFParser creates a new glTF parser instance.
//
// Returns:
//   - gltfParser: a new parser instance
func newGLTFParser() gltfParser {
	return &gltfParserImpl{}
}

func (p *gltfParserImpl) Document() *gltfDocument {
	return p.document
}

func (p *gltfParserImpl) BaseDir() string {
	return p.baseDir
}

func (p *gltfParserImpl) Parse(path string) error {
	p.baseDir = filepath.Dir(path)

	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	ext := strings.ToLower(filepath.Ext(path))
	if ext == ".glb" || (len(data) >= 4 && binary.LittleEndian.Uint32(data[:4]) == gltfGLBMagic) {
		return p.parseGLB(data)
	}

	return p.parseGLTF(data)
}

func (p *gltfParserImpl) ParseReader(r io.Reader, isGLB bool) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return fmt.Errorf("failed to read data: %w", err)
	}

	if isGLB {
		return p.parseGLB(data)
	}
	return p.parseGLTF(data)
}

// parseGLTF parses a glTF JSON file.
func (p *gltfParserImpl) parseGLTF(data []byte) error {
	var doc gltfDocument
	if err := json.Unmarshal(data, &doc); err != nil {
		return fmt.Errorf("failed to parse glTF JSON: %w", err)
	}

	if !strings.HasPrefix(doc.Asset.Version, "2.") {
		return errInvalidGLTFVersion
	}

	if err := p.loadBuffers(&doc); err != nil {
		return fmt.Errorf("failed to load buffers: %w", err)
	}

	p.document = &doc
	return nil
}

// parseGLB parses a GLB binary file.
// Reference: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#glb-file-format-specification
func (p *gltfParserImpl) parseGLB(data []byte) error {
	if len(data) < 12 {
		return errors.New("GLB file too small")
	}

	r := bytes.NewReader(data)

	var header gltfGLBHeader
	if err := binary.Read(r, binary.LittleEndian, &header); err != nil {
		return fmt.Errorf("failed to read GLB header: %w", err)
	}

	if header.Magic != gltfGLBMagic {
		return errInvalidGLBMagic
	}
	if header.Version != gltfGLBVersion {
		return errInvalidGLBVersion
	}

	var jsonData []byte
	var binData []byte

	for {
		var chunkHeader gltfGLBChunkHeader
		if err := binary.Read(r, binary.LittleEndian, &chunkHeader); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("failed to read chunk header: %w", err)
		}

		chunkData := make([]byte, chunkHeader.ChunkLength)
		if _, err := io.ReadFull(r, chunkData); err != nil {
			return fmt.Errorf("failed to read chunk data: %w", err)
		}

		switch chunkHeader.ChunkType {
		case gltfGLBChunkJSON:
			jsonData = chunkData
		case gltfGLBChunkBIN:
			binData = chunkData
		}
	}

	if jsonData == nil {
		return errMissingJSONChunk
	}

	p.glbBinaryChunk = binData

	var doc gltfDocument
	if err := json.Unmarshal(jsonData, &doc); err != nil {
		return fmt.Errorf("failed to parse glTF JSON: %w", err)
	}

	if !strings.HasPrefix(doc.Asset.Version, "2.") {
		return errInvalidGLTFVersion
	}

	if err := p.loadBuffers(&doc); err != nil {
		return fmt.Errorf("failed to load buffers: %w", err)
	}

	p.document = &doc
	return nil
}

// loadBuffers loads all buffer data (from URIs, embedded data, or GLB binary chunk).
func (p *gltfParserImpl) loadBuffers(doc *gltfDocument) error {
	for i := range doc.Buffers {
		buf := &doc.Buffers[i]

		if buf.URI == "" {
			if i == 0 && p.glbBinaryChunk != nil {
				buf.Data = p.glbBinaryChunk
				if len(buf.Data) < buf.ByteLength {
					return fmt.Errorf("buffer %d: %w", i, errBufferSizeMismatch)
				}
				continue
			}
			return fmt.Errorf("buffer %d has no URI and no GLB binary chunk", i)
		}

		data, err := p.loadBufferURI(buf.URI)
		if err != nil {
			return fmt.Errorf("buffer %d: %w", i, err)
		}
		buf.Data = data

		if len(buf.Data) < buf.ByteLength {
			return fmt.Errorf("buffer %d: %w", i, errBufferSizeMismatch)
		}
	}

	return nil
}

// loadBufferURI loads buffer data from a URI (data: URI or file path).
func (p *gltfParserImpl) loadBufferURI(uri string) ([]byte, error) {
	if strings.HasPrefix(uri, "data:") {
		return p.loadDataURI(uri)
	}

	fullPath := filepath.Join(p.baseDir, uri)
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load buffer file %q: %w", uri, err)
	}

	return data, nil
}

// loadDataURI decodes a base64 data URI.
// Format: data:[<mediatype>][;base64],<data>
func (p *gltfParserImpl) loadDataURI(uri string) ([]byte, error) {
	commaIdx := strings.Index(uri, ",")
	if commaIdx < 0 {
		return nil, errInvalidBufferURI
	}

	header := uri[5:commaIdx]
	dataStr := uri[commaIdx+1:]

	if !strings.Contains(header, "base64") {
		return nil, fmt.Errorf("unsupported data URI encoding: %s", header)
	}

	data, err := base64.StdEncoding.DecodeString(dataStr)
	if err != nil {
		return nil, fmt.Errorf("failed to decode base64: %w", err)
	}

	return data, nil
}

// --- Accessor Data Reading ---

func (p *gltfParserImpl) ReadAccessorData(accessorIndex int) ([]byte, error) {
	if p.document == nil {
		return nil, errors.New("no document loaded")
	}
	if accessorIndex < 0 || accessorIndex >= len(p.document.Accessors) {
		return nil, fmt.Errorf("accessor index %d out of range", accessorIndex)
	}

	acc := &p.document.Accessors[accessorIndex]

	if acc.Sparse != nil {
		return nil, errors.New("sparse accessors not yet supported")
	}

	if acc.BufferView == nil {
		return nil, errors.New("accessor has no bufferView")
	}

	bv := &p.document.BufferViews[*acc.BufferView]
	buf := &p.document.Buffers[bv.Buffer]

	componentSize := gltfComponentTypeSize(acc.ComponentType)
	componentCount := gltfAccessorTypeComponentCount(acc.Type)
	elementSize := componentSize * componentCount

	stride := elementSize
	if bv.ByteStride != nil && *bv.ByteStride > 0 {
		stride = *bv.ByteStride
	}

	bufferOffset := bv.ByteOffset + acc.ByteOffset

	result := make([]byte, acc.Count*elementSize)
	for i := 0; i < acc.Count; i++ {
		srcOffset := bufferOffset + i*stride
		dstOffset := i * elementSize
		copy(result[dstOffset:dstOffset+elementSize], buf.Data[srcOffset:srcOffset+elementSize])
	}

	return result, nil
}

func (p *gltfParserImpl) ReadVec2Accessor(accessorIndex int) ([][2]float32, error) {
	acc := &p.document.Accessors[accessorIndex]
	if acc.Type != gltfAccessorTypeVec2 || acc.ComponentType != gltfComponentTypeFloat {
		return nil, fmt.Errorf("accessor is not VEC2 FLOAT: type=%s, componentType=%d", acc.Type, acc.ComponentType)
	}

	data, err := p.ReadAccessorData(accessorIndex)
	if err != nil {
		return nil, err
	}

	result := make([][2]float32, acc.Count)
	r := bytes.NewReader(data)
	for i := 0; i < acc.Count; i++ {
		if err := binary.Read(r, binary.LittleEndian, &result[i]); err != nil {
			return nil, err
		}
	}

	return result, nil
}

func (p *gltfParserImpl) ReadVec3Accessor(accessorIndex int) ([][3]float32, error) {
	acc := &p.document.Accessors[accessorIndex]
	if acc.Type != gltfAccessorTypeVec3 || acc.ComponentType != gltfComponentTypeFloat {
		return nil, fmt.Errorf("accessor is not VEC3 FLOAT: type=%s, componentType=%d", acc.Type, acc.ComponentType)
	}

	data, err := p.ReadAccessorData(accessorIndex)
	if err != nil {
		return nil, err
	}

	result := make([][3]float32, acc.Count)
	r := bytes.NewReader(data)
	for i := 0; i < acc.Count; i++ {
		if err := binary.Read(r, binary.LittleEndian, &result[i]); err != nil {
			return nil, err
		}
	}

	return result, nil
}

func (p *gltfParserImpl) ReadVec4Accessor(accessorIndex int) ([][4]float32, error) {
	acc := &p.document.Accessors[accessorIndex]
	if acc.Type != gltfAccessorTypeVec4 || acc.ComponentType != gltfComponentTypeFloat {
		return nil, fmt.Errorf("accessor is not VEC4 FLOAT: type=%s, componentType=%d", acc.Type, acc.ComponentType)
	}

	data, err := p.ReadAccessorData(accessorIndex)
	if err != nil {
		return nil, err
	}

	result := make([][4]float32, acc.Count)
	r := bytes.NewReader(data)
	for i := 0; i < acc.Count; i++ {
		if err := binary.Read(r, binary.LittleEndian, &result[i]); err != nil {
			return nil, err
		}
	}

	return result, nil
}

func (p *gltfParserImpl) ReadScalarAccessor(accessorIndex int) ([]float32, error) {
	acc := &p.document.Accessors[accessorIndex]
	if acc.Type != gltfAccessorTypeScalar || acc.ComponentType != gltfComponentTypeFloat {
		return nil, fmt.Errorf("accessor is not SCALAR FLOAT: type=%s, componentType=%d", acc.Type, acc.ComponentType)
	}

	data, err := p.ReadAccessorData(accessorIndex)
	if err != nil {
		return nil, err
	}

	result := make([]float32, acc.Count)
	r := bytes.NewReader(data)
	for i := 0; i < acc.Count; i++ {
		if err := binary.Read(r, binary.LittleEndian, &result[i]); err != nil {
			return nil, err
		}
	}

	return result, nil
}

func (p *gltfParserImpl) ReadMat4Accessor(accessorIndex int) ([][16]float32, error) {
	acc := &p.document.Accessors[accessorIndex]
	if acc.Type != gltfAccessorTypeMat4 || acc.ComponentType != gltfComponentTypeFloat {
		return nil, fmt.Errorf("accessor is not MAT4 FLOAT: type=%s, componentType=%d", acc.Type, acc.ComponentType)
	}

	data, err := p.ReadAccessorData(accessorIndex)
	if err != nil {
		return nil, err
	}

	result := make([][16]float32, acc.Count)
	r := bytes.NewReader(data)
	for i := 0; i < acc.Count; i++ {
		if err := binary.Read(r, binary.LittleEndian, &result[i]); err != nil {
			return nil, err
		}
	}

	return result, nil
}

func (p *gltfParserImpl) ReadIndicesAccessor(accessorIndex int) ([]uint32, error) {
	acc := &p.document.Accessors[accessorIndex]
	if acc.Type != gltfAccessorTypeScalar {
		return nil, fmt.Errorf("index accessor is not SCALAR: type=%s", acc.Type)
	}

	data, err := p.ReadAccessorData(accessorIndex)
	if err != nil {
		return nil, err
	}

	result := make([]uint32, acc.Count)
	r := bytes.NewReader(data)

	switch acc.ComponentType {
	case gltfComponentTypeUnsignedByte:
		for i := 0; i < acc.Count; i++ {
			var v uint8
			if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
				return nil, err
			}
			result[i] = uint32(v)
		}
	case gltfComponentTypeUnsignedShort:
		for i := 0; i < acc.Count; i++ {
			var v uint16
			if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
				return nil, err
			}
			result[i] = uint32(v)
		}
	case gltfComponentTypeUnsignedInt:
		if err := binary.Read(r, binary.LittleEndian, &result); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unsupported index component type: %d", acc.ComponentType)
	}

	return result, nil
}

func (p *gltfParserImpl) ReadJointsAccessor(accessorIndex int) ([][4]uint32, error) {
	acc := &p.document.Accessors[accessorIndex]
	if acc.Type != gltfAccessorTypeVec4 {
		return nil, fmt.Errorf("joints accessor is not VEC4: type=%s", acc.Type)
	}

	data, err := p.ReadAccessorData(accessorIndex)
	if err != nil {
		return nil, err
	}

	result := make([][4]uint32, acc.Count)
	r := bytes.NewReader(data)

	switch acc.ComponentType {
	case gltfComponentTypeUnsignedByte:
		for i := 0; i < acc.Count; i++ {
			var v [4]uint8
			if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
				return nil, err
			}
			result[i] = [4]uint32{uint32(v[0]), uint32(v[1]), uint32(v[2]), uint32(v[3])}
		}
	case gltfComponentTypeUnsignedShort:
		for i := 0; i < acc.Count; i++ {
			var v [4]uint16
			if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
				return nil, err
			}
			result[i] = [4]uint32{uint32(v[0]), uint32(v[1]), uint32(v[2]), uint32(v[3])}
		}
	default:
		return nil, fmt.Errorf("unsupported joints component type: %d", acc.ComponentType)
	}

	return result, nil
}

// --- Helper Functions ---

// gltfComponentTypeSize returns the byte size of a component type.
func gltfComponentTypeSize(componentType int) int {
	switch componentType {
	case gltfComponentTypeByte, gltfComponentTypeUnsignedByte:
		return 1
	case gltfComponentTypeShort, gltfComponentTypeUnsignedShort:
		return 2
	case gltfComponentTypeUnsignedInt, gltfComponentTypeFloat:
		return 4
	default:
		return 0
	}
}

// gltfAccessorTypeComponentCount returns the number of components for an accessor type.
func gltfAccessorTypeComponentCount(accessorType string) int {
	switch accessorType {
	case gltfAccessorTypeScalar:
		return 1
	case gltfAccessorTypeVec2:
		return 2
	case gltfAccessorTypeVec3:
		return 3
	case gltfAccessorTypeVec4:
		return 4
	case gltfAccessorTypeMat2:
		return 4
	case gltfAccessorTypeMat3:
		return 9
	case gltfAccessorTypeMat4:
		return 16
	default:
		return 0
	}
}
