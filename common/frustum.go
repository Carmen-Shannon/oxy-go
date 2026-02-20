package common

import (
	"math"
)

// Plane represents a plane in 3D space using the equation: ax + by + cz + d = 0
// where (a, b, c) is the normal and d is the distance from origin.
type Plane struct {
	Normal   [3]float32
	Distance float32
}

// Frustum represents the six planes of a view frustum for culling.
// Planes are oriented so that positive half-space is inside the frustum.
type Frustum struct {
	Planes [6]Plane // Left, Right, Bottom, Top, Near, Far
}

// FrustumPlane indices for clarity
const (
	FrustumLeft   = 0
	FrustumRight  = 1
	FrustumBottom = 2
	FrustumTop    = 3
	FrustumNear   = 4
	FrustumFar    = 5
)

// ExtractFrustumFromMatrix extracts frustum planes from a view-projection matrix.
// The matrix should be the combined View * Projection matrix.
// Uses the Gribb/Hartmann method for plane extraction.
//
// Reference: https://www8.cs.umu.se/kurser/5DV051/HT12/lab/plane_extraction.pdf
//
// Parameters:
//   - viewProj: 16 float32 values representing the view-projection matrix (column-major)
//
// Returns:
//   - Frustum: the extracted frustum with normalized planes
func ExtractFrustumFromMatrix(viewProj []float32) Frustum {
	var f Frustum

	// For column-major matrix M, element M[row][col] is at index col*4 + row
	// So M[i][j] = viewProj[j*4 + i]

	// Left plane: row3 + row0
	f.Planes[FrustumLeft].Normal[0] = viewProj[3] + viewProj[0]  // m[0][3] + m[0][0]
	f.Planes[FrustumLeft].Normal[1] = viewProj[7] + viewProj[4]  // m[1][3] + m[1][0]
	f.Planes[FrustumLeft].Normal[2] = viewProj[11] + viewProj[8] // m[2][3] + m[2][0]
	f.Planes[FrustumLeft].Distance = viewProj[15] + viewProj[12] // m[3][3] + m[3][0]

	// Right plane: row3 - row0
	f.Planes[FrustumRight].Normal[0] = viewProj[3] - viewProj[0]
	f.Planes[FrustumRight].Normal[1] = viewProj[7] - viewProj[4]
	f.Planes[FrustumRight].Normal[2] = viewProj[11] - viewProj[8]
	f.Planes[FrustumRight].Distance = viewProj[15] - viewProj[12]

	// Bottom plane: row3 + row1
	f.Planes[FrustumBottom].Normal[0] = viewProj[3] + viewProj[1]
	f.Planes[FrustumBottom].Normal[1] = viewProj[7] + viewProj[5]
	f.Planes[FrustumBottom].Normal[2] = viewProj[11] + viewProj[9]
	f.Planes[FrustumBottom].Distance = viewProj[15] + viewProj[13]

	// Top plane: row3 - row1
	f.Planes[FrustumTop].Normal[0] = viewProj[3] - viewProj[1]
	f.Planes[FrustumTop].Normal[1] = viewProj[7] - viewProj[5]
	f.Planes[FrustumTop].Normal[2] = viewProj[11] - viewProj[9]
	f.Planes[FrustumTop].Distance = viewProj[15] - viewProj[13]

	// Near plane: row3 + row2
	f.Planes[FrustumNear].Normal[0] = viewProj[3] + viewProj[2]
	f.Planes[FrustumNear].Normal[1] = viewProj[7] + viewProj[6]
	f.Planes[FrustumNear].Normal[2] = viewProj[11] + viewProj[10]
	f.Planes[FrustumNear].Distance = viewProj[15] + viewProj[14]

	// Far plane: row3 - row2
	f.Planes[FrustumFar].Normal[0] = viewProj[3] - viewProj[2]
	f.Planes[FrustumFar].Normal[1] = viewProj[7] - viewProj[6]
	f.Planes[FrustumFar].Normal[2] = viewProj[11] - viewProj[10]
	f.Planes[FrustumFar].Distance = viewProj[15] - viewProj[14]

	// Normalize all planes
	for i := range f.Planes {
		f.normalizePlane(i)
	}

	return f
}

// normalizePlane normalizes a frustum plane so that the normal has unit length.
func (f *Frustum) normalizePlane(index int) {
	p := &f.Planes[index]
	length := float32(math.Sqrt(float64(
		p.Normal[0]*p.Normal[0] +
			p.Normal[1]*p.Normal[1] +
			p.Normal[2]*p.Normal[2],
	)))

	if length > 0 {
		invLen := 1.0 / length
		p.Normal[0] *= invLen
		p.Normal[1] *= invLen
		p.Normal[2] *= invLen
		p.Distance *= invLen
	}
}
