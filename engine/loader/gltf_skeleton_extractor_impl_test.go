package loader

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

type mockGltfParserSkeleton struct {
	mockGltfParser
	mat4Result [][16]float32
	mat4Err    error
}

func (m *mockGltfParserSkeleton) ReadMat4Accessor(_ int) ([][16]float32, error) {
	return m.mat4Result, m.mat4Err
}

type gltfSkeletonExtractorImplTest struct {
	suite.Suite
	mock      *mockGltfParserSkeleton
	extractor gltfSkeletonExtractor
}

func TestGltfSkeletonExtractorImpl(t *testing.T) {
	suite.Run(t, new(gltfSkeletonExtractorImplTest))
}

func (suite *gltfSkeletonExtractorImplTest) SetupSubTest() {
	suite.mock = &mockGltfParserSkeleton{}
	suite.extractor = newGLTFSkeletonExtractor(suite.mock)
}

func (suite *gltfSkeletonExtractorImplTest) TestFindSkeletonForMesh() {
	suite.Run("nil document returns -1", func() {
		suite.mock.doc = nil
		result := suite.extractor.FindSkeletonForMesh(0)
		suite.Equal(-1, result)
	})

	suite.Run("no matching node returns -1", func() {
		meshIdx := 1
		suite.mock.doc = &gltfDocument{
			Nodes: []gltfNode{
				{Mesh: &meshIdx},
			},
		}
		result := suite.extractor.FindSkeletonForMesh(0)
		suite.Equal(-1, result)
	})

	suite.Run("node with mesh but no skin returns -1", func() {
		meshIdx := 0
		suite.mock.doc = &gltfDocument{
			Nodes: []gltfNode{
				{Mesh: &meshIdx},
			},
		}
		result := suite.extractor.FindSkeletonForMesh(0)
		suite.Equal(-1, result)
	})

	suite.Run("matching node with skin returns skin index", func() {
		meshIdx := 0
		skinIdx := 2
		suite.mock.doc = &gltfDocument{
			Nodes: []gltfNode{
				{Mesh: &meshIdx, Skin: &skinIdx},
			},
		}
		result := suite.extractor.FindSkeletonForMesh(0)
		suite.Equal(2, result)
	})
}

func (suite *gltfSkeletonExtractorImplTest) TestExtractSkeletonWithMapping() {
	suite.Run("nil document returns error", func() {
		suite.mock.doc = nil
		_, _, err := suite.extractor.ExtractSkeletonWithMapping(0)
		suite.Error(err)
	})
}

func (suite *gltfSkeletonExtractorImplTest) TestExtractSkeletonInternal() {
	suite.Run("nil document returns error", func() {
		suite.mock.doc = nil
		_, _, err := suite.extractor.ExtractSkeletonWithMapping(0)
		suite.Error(err)
	})

	suite.Run("negative skinIndex returns error", func() {
		suite.mock.doc = &gltfDocument{
			Skins: []gltfSkin{{Joints: []int{0}}},
		}
		_, _, err := suite.extractor.ExtractSkeletonWithMapping(-1)
		suite.Error(err)
	})

	suite.Run("skinIndex out of range returns error", func() {
		suite.mock.doc = &gltfDocument{
			Skins: []gltfSkin{{Joints: []int{0}}},
		}
		_, _, err := suite.extractor.ExtractSkeletonWithMapping(5)
		suite.Error(err)
	})

	suite.Run("ReadMat4Accessor error returns error", func() {
		ibmIdx := 0
		suite.mock.mat4Err = errors.New("mat4 read error")
		suite.mock.doc = &gltfDocument{
			Nodes: []gltfNode{{Name: "joint"}},
			Skins: []gltfSkin{{InverseBindMatrices: &ibmIdx, Joints: []int{0}}},
		}
		_, _, err := suite.extractor.ExtractSkeletonWithMapping(0)
		suite.Error(err)
	})

	suite.Run("invalid joint node index returns error", func() {
		suite.mock.doc = &gltfDocument{
			Nodes: []gltfNode{{Name: "joint"}},
			Skins: []gltfSkin{{Joints: []int{99}}},
		}
		_, _, err := suite.extractor.ExtractSkeletonWithMapping(0)
		suite.Error(err)
	})

	suite.Run("bone with empty name gets bone_N name", func() {
		suite.mock.doc = &gltfDocument{
			Nodes: []gltfNode{{Name: ""}},
			Skins: []gltfSkin{{Joints: []int{0}}},
		}
		skel, _, err := suite.extractor.ExtractSkeletonWithMapping(0)
		suite.NoError(err)
		suite.Equal("bone_0", skel.Bones[0].Name)
	})

	suite.Run("bone uses inverse bind matrix when available", func() {
		ibmIdx := 0
		ibm := [16]float32{
			2, 0, 0, 0,
			0, 2, 0, 0,
			0, 0, 2, 0,
			0, 0, 0, 1,
		}
		suite.mock.mat4Result = [][16]float32{ibm}
		suite.mock.doc = &gltfDocument{
			Nodes: []gltfNode{{Name: "joint"}},
			Skins: []gltfSkin{{InverseBindMatrices: &ibmIdx, Joints: []int{0}}},
		}
		skel, _, err := suite.extractor.ExtractSkeletonWithMapping(0)
		suite.NoError(err)
		suite.Equal(ibm, skel.Bones[0].InverseBindMatrix)
	})

	suite.Run("bone index beyond inverseBindMatrices uses identity matrix", func() {
		ibmIdx := 0
		suite.mock.mat4Result = [][16]float32{} // 0 entries, but 1 joint
		suite.mock.doc = &gltfDocument{
			Nodes: []gltfNode{{Name: "joint"}},
			Skins: []gltfSkin{{InverseBindMatrices: &ibmIdx, Joints: []int{0}}},
		}
		identity := gltfIdentityMatrix()
		skel, _, err := suite.extractor.ExtractSkeletonWithMapping(0)
		suite.NoError(err)
		suite.Equal(identity, skel.Bones[0].InverseBindMatrix)
	})

	suite.Run("parent found sets correct parent index", func() {
		suite.mock.doc = &gltfDocument{
			Nodes: []gltfNode{
				{Name: "root", Children: []int{1}},
				{Name: "child"},
			},
			Skins: []gltfSkin{{Joints: []int{0, 1}}},
		}
		skel, _, err := suite.extractor.ExtractSkeletonWithMapping(0)
		suite.NoError(err)
		suite.Require().Len(skel.Bones, 2)
		// child bone should reference root as parent
		childNewIdx := skel.BoneNameToIndex["child"]
		rootNewIdx := skel.BoneNameToIndex["root"]
		suite.Equal(rootNewIdx, skel.Bones[childNewIdx].ParentIndex)
	})

	suite.Run("success with no inverse bind matrices", func() {
		suite.mock.doc = &gltfDocument{
			Nodes: []gltfNode{{Name: "root"}},
			Skins: []gltfSkin{{Joints: []int{0}}},
		}
		skel, mapping, err := suite.extractor.ExtractSkeletonWithMapping(0)
		suite.NoError(err)
		suite.Len(skel.Bones, 1)
		suite.Len(mapping, 1)
	})
}

func (suite *gltfSkeletonExtractorImplTest) TestGltfExtractNodeTransform() {
	suite.Run("nil matrix nil translation rotation scale returns defaults", func() {
		node := &gltfNode{}
		t := gltfExtractNodeTransform(node)
		suite.Equal([3]float32{0, 0, 0}, t.Translation)
		suite.Equal([4]float32{0, 0, 0, 1}, t.Rotation)
		suite.Equal([3]float32{1, 1, 1}, t.Scale)
	})

	suite.Run("non-nil matrix decomposes matrix", func() {
		m := [16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			5, 6, 7, 1,
		}
		node := &gltfNode{Matrix: &m}
		t := gltfExtractNodeTransform(node)
		suite.InDelta(5.0, float64(t.Translation[0]), 1e-6)
		suite.InDelta(6.0, float64(t.Translation[1]), 1e-6)
		suite.InDelta(7.0, float64(t.Translation[2]), 1e-6)
	})

	suite.Run("translation set", func() {
		tr := [3]float32{1, 2, 3}
		node := &gltfNode{Translation: &tr}
		t := gltfExtractNodeTransform(node)
		suite.Equal([3]float32{1, 2, 3}, t.Translation)
	})

	suite.Run("rotation set", func() {
		rot := [4]float32{0, 0, 0, 1}
		node := &gltfNode{Rotation: &rot}
		t := gltfExtractNodeTransform(node)
		suite.Equal([4]float32{0, 0, 0, 1}, t.Rotation)
	})

	suite.Run("scale set", func() {
		sc := [3]float32{2, 2, 2}
		node := &gltfNode{Scale: &sc}
		t := gltfExtractNodeTransform(node)
		suite.Equal([3]float32{2, 2, 2}, t.Scale)
	})
}

func (suite *gltfSkeletonExtractorImplTest) TestGltfDecomposeMatrix() {
	suite.Run("identity matrix branch trace greater than 0", func() {
		m := [16]float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			5, 6, 7, 1,
		}
		result := gltfDecomposeMatrix(m)
		suite.Equal([3]float32{5, 6, 7}, result.Translation)
		suite.InDelta(1.0, float64(result.Scale[0]), 1e-6)
		suite.InDelta(1.0, float64(result.Scale[1]), 1e-6)
		suite.InDelta(1.0, float64(result.Scale[2]), 1e-6)
		suite.InDelta(0.0, float64(result.Rotation[0]), 1e-4)
		suite.InDelta(0.0, float64(result.Rotation[1]), 1e-4)
		suite.InDelta(0.0, float64(result.Rotation[2]), 1e-4)
		suite.InDelta(1.0, float64(result.Rotation[3]), 1e-4)
	})

	suite.Run("r00 dominant branch 180 degrees around X", func() {
		// col-major: col0=(1,0,0,0), col1=(0,-1,0,0), col2=(0,0,-1,0), col3=(0,0,0,1)
		m := [16]float32{
			1, 0, 0, 0,
			0, -1, 0, 0,
			0, 0, -1, 0,
			0, 0, 0, 1,
		}
		result := gltfDecomposeMatrix(m)
		// 180° around X → quaternion (1,0,0,0)
		suite.InDelta(1.0, float64(result.Rotation[0]), 1e-4)
		suite.InDelta(0.0, float64(result.Rotation[1]), 1e-4)
		suite.InDelta(0.0, float64(result.Rotation[2]), 1e-4)
		suite.InDelta(0.0, float64(result.Rotation[3]), 1e-4)
	})

	suite.Run("r11 dominant branch 180 degrees around Y", func() {
		// col-major: col0=(-1,0,0,0), col1=(0,1,0,0), col2=(0,0,-1,0), col3=(0,0,0,1)
		m := [16]float32{
			-1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1, 0,
			0, 0, 0, 1,
		}
		result := gltfDecomposeMatrix(m)
		// 180° around Y → quaternion (0,1,0,0)
		suite.InDelta(0.0, float64(result.Rotation[0]), 1e-4)
		suite.InDelta(1.0, float64(result.Rotation[1]), 1e-4)
		suite.InDelta(0.0, float64(result.Rotation[2]), 1e-4)
		suite.InDelta(0.0, float64(result.Rotation[3]), 1e-4)
	})

	suite.Run("r22 dominant branch 180 degrees around Z", func() {
		// col-major: col0=(-1,0,0,0), col1=(0,-1,0,0), col2=(0,0,1,0), col3=(0,0,0,1)
		m := [16]float32{
			-1, 0, 0, 0,
			0, -1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}
		result := gltfDecomposeMatrix(m)
		// 180° around Z → quaternion (0,0,1,0)
		suite.InDelta(0.0, float64(result.Rotation[0]), 1e-4)
		suite.InDelta(0.0, float64(result.Rotation[1]), 1e-4)
		suite.InDelta(1.0, float64(result.Rotation[2]), 1e-4)
		suite.InDelta(0.0, float64(result.Rotation[3]), 1e-4)
	})

	suite.Run("zero scale columns trigger guard and do not divide by zero", func() {
		m := [16]float32{} // all zeros
		result := gltfDecomposeMatrix(m)
		// scale is extracted before guard, so it's {0,0,0}
		suite.Equal([3]float32{0, 0, 0}, result.Scale)
		// translation column (m[12..14]) is all zero
		suite.Equal([3]float32{0, 0, 0}, result.Translation)
		// quaternion should be normalized (no panic)
		length := result.Rotation[0]*result.Rotation[0] +
			result.Rotation[1]*result.Rotation[1] +
			result.Rotation[2]*result.Rotation[2] +
			result.Rotation[3]*result.Rotation[3]
		suite.InDelta(1.0, float64(length), 1e-4)
	})
}

func (suite *gltfSkeletonExtractorImplTest) TestGltfVectorLength() {
	suite.Run("zero vector returns 0", func() {
		result := gltfVectorLength(0, 0, 0)
		suite.InDelta(0.0, float64(result), 1e-6)
	})

	suite.Run("known length 3 4 0 returns 5", func() {
		result := gltfVectorLength(3, 4, 0)
		suite.InDelta(5.0, float64(result), 1e-6)
	})
}

func (suite *gltfSkeletonExtractorImplTest) TestGltfMatrixToQuaternion() {
	suite.Run("identity matrix trace greater than 0 gives identity quaternion", func() {
		m := [9]float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
		q := gltfMatrixToQuaternion(m)
		suite.InDelta(0.0, float64(q[0]), 1e-4)
		suite.InDelta(0.0, float64(q[1]), 1e-4)
		suite.InDelta(0.0, float64(q[2]), 1e-4)
		suite.InDelta(1.0, float64(q[3]), 1e-4)
	})

	suite.Run("r00 dominant branch 180 degrees around X", func() {
		// r00=1, r11=-1, r22=-1 → trace=-1, r00>r11 && r00>r22
		m := [9]float32{1, 0, 0, 0, -1, 0, 0, 0, -1}
		q := gltfMatrixToQuaternion(m)
		suite.InDelta(1.0, float64(q[0]), 1e-4)
		suite.InDelta(0.0, float64(q[1]), 1e-4)
		suite.InDelta(0.0, float64(q[2]), 1e-4)
		suite.InDelta(0.0, float64(q[3]), 1e-4)
	})

	suite.Run("r11 dominant branch 180 degrees around Y", func() {
		// r00=-1, r11=1, r22=-1 → trace=-1, NOT r00>r11, r11>r22
		m := [9]float32{-1, 0, 0, 0, 1, 0, 0, 0, -1}
		q := gltfMatrixToQuaternion(m)
		suite.InDelta(0.0, float64(q[0]), 1e-4)
		suite.InDelta(1.0, float64(q[1]), 1e-4)
		suite.InDelta(0.0, float64(q[2]), 1e-4)
		suite.InDelta(0.0, float64(q[3]), 1e-4)
	})

	suite.Run("r22 dominant else branch 180 degrees around Z", func() {
		// r00=-1, r11=-1, r22=1 → trace=-1, NOT r00>r11, NOT r11>r22
		m := [9]float32{-1, 0, 0, 0, -1, 0, 0, 0, 1}
		q := gltfMatrixToQuaternion(m)
		suite.InDelta(0.0, float64(q[0]), 1e-4)
		suite.InDelta(0.0, float64(q[1]), 1e-4)
		suite.InDelta(1.0, float64(q[2]), 1e-4)
		suite.InDelta(0.0, float64(q[3]), 1e-4)
	})
}

func (suite *gltfSkeletonExtractorImplTest) TestGltfTopologicalSortBones() {
	suite.Run("empty bones returns early", func() {
		sorted, roots, nameToIdx, oldToNew := gltfTopologicalSortBones(
			[]model.Bone{},
			[]int32{},
			map[string]int32{},
		)
		suite.Empty(sorted)
		suite.Empty(roots)
		suite.Empty(nameToIdx)
		suite.Empty(oldToNew)
	})

	suite.Run("normal hierarchy sorts parents before children", func() {
		bones := []model.Bone{
			{Name: "root", ParentIndex: -1},
			{Name: "child", ParentIndex: 0},
		}
		rootIndices := []int32{0}
		nameToIndex := map[string]int32{"root": 0, "child": 1}
		sorted, newRoots, newNameToIdx, oldToNew := gltfTopologicalSortBones(bones, rootIndices, nameToIndex)
		suite.Len(sorted, 2)
		suite.Equal("root", sorted[0].Name)
		suite.Equal("child", sorted[1].Name)
		suite.Equal(int32(-1), sorted[0].ParentIndex)
		suite.Equal(int32(0), sorted[1].ParentIndex)
		suite.Len(newRoots, 1)
		suite.Equal(int32(0), newRoots[0])
		suite.Equal(int32(0), newNameToIdx["root"])
		suite.Equal(int32(1), newNameToIdx["child"])
		suite.Equal(int32(0), oldToNew[0])
		suite.Equal(int32(1), oldToNew[1])
	})

	suite.Run("disconnected bones are appended after reachable bones", func() {
		bones := []model.Bone{
			{Name: "root", ParentIndex: -1},
			{Name: "orphan1", ParentIndex: -1},
			{Name: "orphan2", ParentIndex: -1},
		}
		// Only bone 0 is listed as a root; bones 1 and 2 are disconnected
		rootIndices := []int32{0}
		nameToIndex := map[string]int32{"root": 0, "orphan1": 1, "orphan2": 2}
		sorted, _, _, oldToNew := gltfTopologicalSortBones(bones, rootIndices, nameToIndex)
		suite.Len(sorted, 3)
		// first sorted bone is root (old index 0)
		suite.Equal(int32(0), oldToNew[0])
		// orphans are appended at positions 1 and 2
		suite.Contains([]int32{oldToNew[1], oldToNew[2]}, int32(1))
		suite.Contains([]int32{oldToNew[1], oldToNew[2]}, int32(2))
	})
}
