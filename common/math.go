package common

import (
	"math"
	"unsafe"
)

// Identity resets a 4x4 matrix (flat slice) to the identity matrix.
// The matrix is stored in column-major order.
//
// Parameters:
//   - m: destination slice (must be at least 16 elements)
func Identity(m []float32) {
	for i := range m {
		m[i] = 0
	}
	m[0], m[5], m[10], m[15] = 1, 1, 1, 1
}

// SliceToBytes converts any slice to a byte slice for GPU buffer uploads.
// Uses unsafe pointer operations to create a view into the original data.
// WARNING: The returned slice shares memory with the input - do not modify.
//
// Parameters:
//   - data: source slice of any type
//
// Returns:
//   - []byte: byte slice view of the input data, or nil if input is empty
func SliceToBytes[T any](data []T) []byte {
	if len(data) == 0 {
		return nil
	}
	var zero T
	size := unsafe.Sizeof(zero)
	totalBytes := int(size) * len(data)
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), totalBytes)
}

// StructToBytes reinterprets a pointer to a struct as a raw byte slice using unsafe.
// The returned slice has length equal to the struct's size in memory.
//
// Parameters:
//   - v: pointer to the struct to reinterpret
//
// Returns:
//   - []byte: byte slice view of the struct's memory
func StructToBytes[T any](v *T) []byte {
	size := unsafe.Sizeof(*v)
	return unsafe.Slice((*byte)(unsafe.Pointer(v)), int(size))
}

// Mul4 multiplies two 4x4 matrices and stores the result in out.
// All matrices are stored in column-major order (OpenGL/WebGPU convention).
// Result: out = a * b
//
// Parameters:
//   - out: destination slice (must be at least 16 elements)
//   - a: left-hand matrix (16 elements)
//   - b: right-hand matrix (16 elements)
func Mul4(out, a, b []float32) {
	var buf [16]float32
	for i := 0; i < 4; i++ { // column of B
		for j := 0; j < 4; j++ { // row of A
			sum := float32(0)
			for k := 0; k < 4; k++ {
				sum += a[k*4+j] * b[i*4+k]
			}
			buf[i*4+j] = sum
		}
	}
	copy(out, buf[:])
}

// Perspective creates a perspective projection matrix.
// Uses infinite far plane convention compatible with WebGPU clip space [0, 1].
//
// Parameters:
//   - out: destination slice (must be at least 16 elements)
//   - fovY: vertical field of view in radians
//   - aspect: viewport aspect ratio (width/height)
//   - near: near clipping plane distance (must be > 0)
//   - far: far clipping plane distance (must be > near)
func Perspective(out []float32, fovY, aspect, near, far float32) {
	f := 1.0 / float32(math.Tan(float64(fovY)/2.0))
	Identity(out)

	out[0] = f / aspect
	out[5] = f
	out[10] = far / (near - far)
	out[11] = -1.0
	out[14] = (near * far) / (near - far)
	out[15] = 0.0
}

// BuildModelMatrix constructs a 4x4 model matrix from position, Euler rotation, and scale.
// The rotation order is Y * X * Z (yaw-pitch-roll). All matrices are column-major.
//
// Parameters:
//   - out: destination slice (must be at least 16 elements)
//   - posX, posY, posZ: translation in world space
//   - rotX, rotY, rotZ: rotation angles in radians around each axis
//   - scaleX, scaleY, scaleZ: scale factors along each axis
func BuildModelMatrix(out []float32, posX, posY, posZ, rotX, rotY, rotZ, scaleX, scaleY, scaleZ float32) {
	cx := float32(math.Cos(float64(rotX)))
	sx := float32(math.Sin(float64(rotX)))
	cy := float32(math.Cos(float64(rotY)))
	sy := float32(math.Sin(float64(rotY)))
	cz := float32(math.Cos(float64(rotZ)))
	sz := float32(math.Sin(float64(rotZ)))

	// R = Ry * Rx * Rz, column-major
	out[0] = (cy*cz + sy*sx*sz) * scaleX
	out[1] = (cx * sz) * scaleX
	out[2] = (-sy*cz + cy*sx*sz) * scaleX
	out[3] = 0

	out[4] = (cy*-sz + sy*sx*cz) * scaleY
	out[5] = (cx * cz) * scaleY
	out[6] = (sy*sz + cy*sx*cz) * scaleY
	out[7] = 0

	out[8] = (sy * cx) * scaleZ
	out[9] = (-sx) * scaleZ
	out[10] = (cy * cx) * scaleZ
	out[11] = 0

	out[12] = posX
	out[13] = posY
	out[14] = posZ
	out[15] = 1
}

// Invert4 computes the inverse of a 4x4 column-major matrix using the Laplace
// expansion (cofactor) method. If the matrix is singular (determinant ≈ 0) the
// output is left unchanged and the function returns false.
//
// Parameters:
//   - out: destination slice (must be at least 16 elements)
//   - m: source matrix (16 elements, column-major)
//
// Returns:
//   - bool: true if the matrix was successfully inverted, false if singular
func Invert4(out, m []float32) bool {
	// 2x2 sub-determinants of the upper-left and lower-right quadrants.
	s0 := m[0]*m[5] - m[4]*m[1]
	s1 := m[0]*m[6] - m[4]*m[2]
	s2 := m[0]*m[7] - m[4]*m[3]
	s3 := m[1]*m[6] - m[5]*m[2]
	s4 := m[1]*m[7] - m[5]*m[3]
	s5 := m[2]*m[7] - m[6]*m[3]

	c5 := m[10]*m[15] - m[14]*m[11]
	c4 := m[9]*m[15] - m[13]*m[11]
	c3 := m[9]*m[14] - m[13]*m[10]
	c2 := m[8]*m[15] - m[12]*m[11]
	c1 := m[8]*m[14] - m[12]*m[10]
	c0 := m[8]*m[13] - m[12]*m[9]

	det := s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0
	if det == 0 {
		return false
	}

	invDet := 1.0 / det

	out[0] = (m[5]*c5 - m[6]*c4 + m[7]*c3) * invDet
	out[1] = (-m[1]*c5 + m[2]*c4 - m[3]*c3) * invDet
	out[2] = (m[13]*s5 - m[14]*s4 + m[15]*s3) * invDet
	out[3] = (-m[9]*s5 + m[10]*s4 - m[11]*s3) * invDet

	out[4] = (-m[4]*c5 + m[6]*c2 - m[7]*c1) * invDet
	out[5] = (m[0]*c5 - m[2]*c2 + m[3]*c1) * invDet
	out[6] = (-m[12]*s5 + m[14]*s2 - m[15]*s1) * invDet
	out[7] = (m[8]*s5 - m[10]*s2 + m[11]*s1) * invDet

	out[8] = (m[4]*c4 - m[5]*c2 + m[7]*c0) * invDet
	out[9] = (-m[0]*c4 + m[1]*c2 - m[3]*c0) * invDet
	out[10] = (m[12]*s4 - m[13]*s2 + m[15]*s0) * invDet
	out[11] = (-m[8]*s4 + m[9]*s2 - m[11]*s0) * invDet

	out[12] = (-m[4]*c3 + m[5]*c1 - m[6]*c0) * invDet
	out[13] = (m[0]*c3 - m[1]*c1 + m[2]*c0) * invDet
	out[14] = (-m[12]*s3 + m[13]*s1 - m[14]*s0) * invDet
	out[15] = (m[8]*s3 - m[9]*s1 + m[10]*s0) * invDet

	return true
}

// LookAt creates a view matrix that positions and orients the camera.
// The resulting matrix transforms world coordinates to view/camera space.
//
// Parameters:
//   - out: destination slice (must be at least 16 elements)
//   - eyeX, eyeY, eyeZ: camera position in world space
//   - centerX, centerY, centerZ: target point the camera looks at
//   - upX, upY, upZ: up vector defining camera orientation (typically 0,1,0)
func LookAt(out []float32, eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ float32) {
	z0 := eyeX - centerX
	z1 := eyeY - centerY
	z2 := eyeZ - centerZ
	val := float64(z0*z0 + z1*z1 + z2*z2)
	if val == 0 {
		val = 1
	}
	invLen := 1.0 / float32(math.Sqrt(val))
	z0 *= invLen
	z1 *= invLen
	z2 *= invLen

	x0 := upY*z2 - upZ*z1
	x1 := upZ*z0 - upX*z2
	x2 := upX*z1 - upY*z0
	val = float64(x0*x0 + x1*x1 + x2*x2)
	if val == 0 {
		val = 1
	}
	invLen = 1.0 / float32(math.Sqrt(val))
	x0 *= invLen
	x1 *= invLen
	x2 *= invLen

	y0 := z1*x2 - z2*x1
	y1 := z2*x0 - z0*x2
	y2 := z0*x1 - z1*x0

	out[0], out[4], out[8], out[12] = x0, x1, x2, -(x0*eyeX + x1*eyeY + x2*eyeZ)
	out[1], out[5], out[9], out[13] = y0, y1, y2, -(y0*eyeX + y1*eyeY + y2*eyeZ)
	out[2], out[6], out[10], out[14] = z0, z1, z2, -(z0*eyeX + z1*eyeY + z2*eyeZ)
	out[3], out[7], out[11], out[15] = 0, 0, 0, 1
}
