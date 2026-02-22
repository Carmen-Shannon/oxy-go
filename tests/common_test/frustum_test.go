package common_test

import (
	"math"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/stretchr/testify/suite"
)

func TestFrustum(t *testing.T) {
	suite.Run(t, new(frustumTest))
}

type frustumTest struct {
	suite.Suite
}

func (suite *frustumTest) TestFrustumPlaneConstants() {
	suite.Run("left index is 0", func() {
		suite.Equal(0, common.FrustumLeft)
	})

	suite.Run("right index is 1", func() {
		suite.Equal(1, common.FrustumRight)
	})

	suite.Run("bottom index is 2", func() {
		suite.Equal(2, common.FrustumBottom)
	})

	suite.Run("top index is 3", func() {
		suite.Equal(3, common.FrustumTop)
	})

	suite.Run("near index is 4", func() {
		suite.Equal(4, common.FrustumNear)
	})

	suite.Run("far index is 5", func() {
		suite.Equal(5, common.FrustumFar)
	})
}

func (suite *frustumTest) TestExtractFrustumFromMatrix() {
	// Standard VP: camera at (0,0,5) looking at origin, 45° FOV, 16:9 aspect
	vp := buildViewProj(0, 0, 5, float32(math.Pi/4), 16.0/9.0, 0.1, 100.0)
	f := common.ExtractFrustumFromMatrix(vp)

	suite.Run("should produce exactly 6 planes", func() {
		suite.Len(f.Planes, 6)
	})

	suite.Run("all plane normals should be unit length", func() {
		for i, p := range f.Planes {
			length := planeNormalLength(p)
			suite.InDelta(1.0, length, 1e-5, "plane index %d", i)
		}
	})

	suite.Run("near and far Z normals should point in opposite directions", func() {
		nearNz := f.Planes[common.FrustumNear].Normal[2]
		farNz := f.Planes[common.FrustumFar].Normal[2]
		suite.True(nearNz*farNz < 0)
	})

	suite.Run("left and right planes should have opposing X normal signs", func() {
		leftNx := f.Planes[common.FrustumLeft].Normal[0]
		rightNx := f.Planes[common.FrustumRight].Normal[0]
		suite.True(leftNx*rightNx < 0)
	})

	suite.Run("top and bottom planes should have opposing Y normal signs", func() {
		bottomNy := f.Planes[common.FrustumBottom].Normal[1]
		topNy := f.Planes[common.FrustumTop].Normal[1]
		suite.True(bottomNy*topNy < 0)
	})
}

func (suite *frustumTest) TestExtractFrustumFromMatrixSymmetric() {
	// Square aspect, camera centered on Z-axis ⇒ left/right and top/bottom are symmetric
	vp := buildViewProj(0, 0, 10, float32(math.Pi/4), 1.0, 0.1, 100.0)
	f := common.ExtractFrustumFromMatrix(vp)

	suite.Run("left and right X normals should be equal magnitude for unit aspect", func() {
		leftNx := float64(f.Planes[common.FrustumLeft].Normal[0])
		rightNx := float64(f.Planes[common.FrustumRight].Normal[0])
		suite.InDelta(math.Abs(leftNx), math.Abs(rightNx), 1e-5)
	})

	suite.Run("bottom and top Y normals should be equal magnitude for unit aspect", func() {
		bottomNy := float64(f.Planes[common.FrustumBottom].Normal[1])
		topNy := float64(f.Planes[common.FrustumTop].Normal[1])
		suite.InDelta(math.Abs(bottomNy), math.Abs(topNy), 1e-5)
	})

	suite.Run("left and right distances should be equal for centered camera with unit aspect", func() {
		leftD := float64(f.Planes[common.FrustumLeft].Distance)
		rightD := float64(f.Planes[common.FrustumRight].Distance)
		suite.InDelta(leftD, rightD, 1e-5)
	})

	suite.Run("bottom and top distances should be equal for centered camera with unit aspect", func() {
		bottomD := float64(f.Planes[common.FrustumBottom].Distance)
		topD := float64(f.Planes[common.FrustumTop].Distance)
		suite.InDelta(bottomD, topD, 1e-5)
	})
}

func (suite *frustumTest) TestExtractFrustumFromIdentityMatrix() {
	id := make([]float32, 16)
	common.Identity(id)
	f := common.ExtractFrustumFromMatrix(id)

	suite.Run("should still produce 6 planes from identity", func() {
		suite.Len(f.Planes, 6)
	})

	suite.Run("all plane normals should be unit length from identity", func() {
		for i, p := range f.Planes {
			length := planeNormalLength(p)
			// Identity produces valid planes; normals are normalized
			suite.InDelta(1.0, length, 1e-5, "plane index %d", i)
		}
	})

	// The identity matrix forms a clip-space cube [-1,1]^3.
	// Left plane: row3 + row0 → normal (1+1, 0, 0) = (2,0,0), d = 1+0 = 1 → normalized (1,0,0), d=0.5
	// Right plane: row3 - row0 → normal (1-1, 0, 0) = (0,0,0) — but actually:
	// For identity: m[0]=1, m[3]=0, m[4]=0, m[7]=0, m[8]=0, m[11]=0, m[12]=0, m[15]=1
	// Left: (0+1, 0+0, 0+0) = (1,0,0), d = 1+0 = 1 → normalized (1,0,0), d=1
	suite.Run("left plane normal should be along +X from identity", func() {
		suite.InDelta(1.0, float64(f.Planes[common.FrustumLeft].Normal[0]), 1e-6)
		suite.InDelta(0.0, float64(f.Planes[common.FrustumLeft].Normal[1]), 1e-6)
		suite.InDelta(0.0, float64(f.Planes[common.FrustumLeft].Normal[2]), 1e-6)
	})

	suite.Run("right plane normal should be along -X from identity", func() {
		suite.InDelta(-1.0, float64(f.Planes[common.FrustumRight].Normal[0]), 1e-6)
		suite.InDelta(0.0, float64(f.Planes[common.FrustumRight].Normal[1]), 1e-6)
		suite.InDelta(0.0, float64(f.Planes[common.FrustumRight].Normal[2]), 1e-6)
	})

	suite.Run("bottom plane normal should be along +Y from identity", func() {
		suite.InDelta(0.0, float64(f.Planes[common.FrustumBottom].Normal[0]), 1e-6)
		suite.InDelta(1.0, float64(f.Planes[common.FrustumBottom].Normal[1]), 1e-6)
		suite.InDelta(0.0, float64(f.Planes[common.FrustumBottom].Normal[2]), 1e-6)
	})

	suite.Run("top plane normal should be along -Y from identity", func() {
		suite.InDelta(0.0, float64(f.Planes[common.FrustumTop].Normal[0]), 1e-6)
		suite.InDelta(-1.0, float64(f.Planes[common.FrustumTop].Normal[1]), 1e-6)
		suite.InDelta(0.0, float64(f.Planes[common.FrustumTop].Normal[2]), 1e-6)
	})
}

func (suite *frustumTest) TestExtractFrustumFromZeroMatrix() {
	zeroVP := make([]float32, 16)
	f := common.ExtractFrustumFromMatrix(zeroVP)

	suite.Run("zero-matrix planes should have zero-length normals without panic", func() {
		for i, p := range f.Planes {
			length := planeNormalLength(p)
			suite.InDelta(0.0, length, 1e-6, "plane index %d", i)
		}
	})

	suite.Run("zero-matrix plane distances should all be zero", func() {
		for i, p := range f.Planes {
			suite.InDelta(0.0, float64(p.Distance), 1e-6, "plane index %d", i)
		}
	})
}

func (suite *frustumTest) TestExtractFrustumFromMatrixWideAspect() {
	// Wide aspect (2:1) should produce left/right planes that are more "open" (smaller X normal magnitude)
	// compared to a narrow (1:2) frustum.
	wide := buildViewProj(0, 0, 5, float32(math.Pi/4), 2.0, 0.1, 100.0)
	narrow := buildViewProj(0, 0, 5, float32(math.Pi/4), 0.5, 0.1, 100.0)
	fWide := common.ExtractFrustumFromMatrix(wide)
	fNarrow := common.ExtractFrustumFromMatrix(narrow)

	suite.Run("wider aspect should have smaller left plane X normal magnitude than narrow", func() {
		wideLeftNx := math.Abs(float64(fWide.Planes[common.FrustumLeft].Normal[0]))
		narrowLeftNx := math.Abs(float64(fNarrow.Planes[common.FrustumLeft].Normal[0]))
		suite.True(wideLeftNx < narrowLeftNx)
	})
}

func (suite *frustumTest) TestExtractFrustumFromMatrixDifferentCameraPositions() {
	f1 := common.ExtractFrustumFromMatrix(buildViewProj(0, 0, 5, float32(math.Pi/4), 1.0, 0.1, 100.0))
	f2 := common.ExtractFrustumFromMatrix(buildViewProj(0, 0, 50, float32(math.Pi/4), 1.0, 0.1, 100.0))

	suite.Run("moving camera farther should change near plane distance", func() {
		d1 := float64(f1.Planes[common.FrustumNear].Distance)
		d2 := float64(f2.Planes[common.FrustumNear].Distance)
		suite.NotEqual(d1, d2)
	})

	suite.Run("moving camera farther should change far plane distance", func() {
		d1 := float64(f1.Planes[common.FrustumFar].Distance)
		d2 := float64(f2.Planes[common.FrustumFar].Distance)
		suite.NotEqual(d1, d2)
	})
}

func (suite *frustumTest) TestPlaneStruct() {
	suite.Run("zero-value plane should have zero normal and zero distance", func() {
		var p common.Plane
		suite.Equal(float32(0), p.Normal[0])
		suite.Equal(float32(0), p.Normal[1])
		suite.Equal(float32(0), p.Normal[2])
		suite.Equal(float32(0), p.Distance)
	})

	suite.Run("plane fields should store assigned values", func() {
		p := common.Plane{
			Normal:   [3]float32{0, 1, 0},
			Distance: 5.0,
		}
		suite.Equal(float32(0), p.Normal[0])
		suite.Equal(float32(1), p.Normal[1])
		suite.Equal(float32(0), p.Normal[2])
		suite.Equal(float32(5.0), p.Distance)
	})
}

func (suite *frustumTest) TestFrustumStruct() {
	suite.Run("zero-value frustum should have 6 zero planes", func() {
		var f common.Frustum
		suite.Len(f.Planes, 6)
		for _, p := range f.Planes {
			suite.Equal(float32(0), p.Normal[0])
			suite.Equal(float32(0), p.Normal[1])
			suite.Equal(float32(0), p.Normal[2])
			suite.Equal(float32(0), p.Distance)
		}
	})
}

// buildViewProj builds a view-projection matrix for a camera positioned at eye
// looking at the origin with Y-up, using the given perspective parameters.
// Returns a 16-element column-major VP matrix (Projection * View).
func buildViewProj(eyeX, eyeY, eyeZ, fovY, aspect, near, far float32) []float32 {
	view := make([]float32, 16)
	proj := make([]float32, 16)
	vp := make([]float32, 16)
	common.LookAt(view, eyeX, eyeY, eyeZ, 0, 0, 0, 0, 1, 0)
	common.Perspective(proj, fovY, aspect, near, far)
	common.Mul4(vp, proj, view)
	return vp
}

// planeNormalLength returns the Euclidean length of a Plane's normal vector.
func planeNormalLength(p common.Plane) float64 {
	nx := float64(p.Normal[0])
	ny := float64(p.Normal[1])
	nz := float64(p.Normal[2])
	return math.Sqrt(nx*nx + ny*ny + nz*nz)
}
