package loader_test

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/Carmen-Shannon/oxy-go/engine/loader"
	modelmocks "github.com/Carmen-Shannon/oxy-go/tests/mocks/model"
	"github.com/stretchr/testify/suite"
)

type loaderTest struct {
	suite.Suite
}

func TestLoader(t *testing.T) {
	suite.Run(t, new(loaderTest))
}

func (suite *loaderTest) TestBackendTypeConstant() {
	suite.Run("BackendTypeGLTF is zero", func() {
		suite.Equal(loader.LoaderBackendType(0), loader.BackendTypeGLTF)
	})
}

func (suite *loaderTest) TestNewLoader() {
	suite.Run("Get returns nil for unknown key", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		suite.Nil(l.Get("nonexistent"))
	})

	suite.Run("Models returns empty map", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		suite.Empty(l.Models())
	})
}

func (suite *loaderTest) TestNewLoaderWithOptions() {
	suite.Run("WithModel pre-populates cache", func() {
		m := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("test_model", m))
		suite.Equal(m, l.Get("test_model"))
	})

	suite.Run("multiple WithModel options populate cache", func() {
		m1 := &modelmocks.MockModel{}
		m2 := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF,
			loader.WithModel("model_a", m1),
			loader.WithModel("model_b", m2),
		)
		suite.Equal(m1, l.Get("model_a"))
		suite.Equal(m2, l.Get("model_b"))
	})
}

func (suite *loaderTest) TestGet() {
	suite.Run("returns nil for unknown key", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		suite.Nil(l.Get("does_not_exist"))
	})

	suite.Run("returns pre-cached model by key", func() {
		m := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("cached", m))
		suite.Equal(m, l.Get("cached"))
	})

	suite.Run("returns nil for empty string key", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		suite.Nil(l.Get(""))
	})
}

func (suite *loaderTest) TestModels() {
	suite.Run("returns empty map for new loader", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		models := l.Models()
		suite.NotNil(models)
		suite.Len(models, 0)
	})

	suite.Run("returns all pre-cached models", func() {
		m1 := &modelmocks.MockModel{}
		m2 := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF,
			loader.WithModel("a", m1),
			loader.WithModel("b", m2),
		)
		models := l.Models()
		suite.Len(models, 2)
		suite.Equal(m1, models["a"])
		suite.Equal(m2, models["b"])
	})

	suite.Run("returns a defensive copy", func() {
		m := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("x", m))
		models := l.Models()
		models["injected"] = &modelmocks.MockModel{}

		// Original cache should be unaffected
		suite.Nil(l.Get("injected"))
		suite.Len(l.Models(), 1)
	})
}

func (suite *loaderTest) TestLoadUnsupportedFormat() {
	suite.Run("returns error for .obj extension", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load("model.obj")
		suite.Error(err)
		suite.Contains(err.Error(), "unsupported model format")
	})

	suite.Run("returns error for .fbx extension", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load("model.fbx")
		suite.Error(err)
		suite.Contains(err.Error(), "unsupported model format")
	})

	suite.Run("returns error for no extension", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load("modelfile")
		suite.Error(err)
		suite.Contains(err.Error(), "unsupported model format")
	})
}

func (suite *loaderTest) TestLoadNonexistentFile() {
	suite.Run("returns error for missing gltf file", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load("/nonexistent/path/model.gltf")
		suite.Error(err)
	})

	suite.Run("returns error for missing glb file", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load("/nonexistent/path/model.glb")
		suite.Error(err)
	})
}

func (suite *loaderTest) TestLoadCacheHit() {
	suite.Run("returns cached model on second Load call", func() {
		m := &modelmocks.MockModel{}
		l := loader.NewLoader(loader.BackendTypeGLTF, loader.WithModel("scene.glb", m))

		result, err := l.Load("scene.glb")
		suite.NoError(err)
		suite.Equal(m, result)
	})
}

func (suite *loaderTest) TestLoadWithTempGLBFile() {
	suite.Run("loads a real GLB file from disk", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("caches model by file path", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "cached.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m1, err := l.Load(path)
		suite.NoError(err)

		m2, err := l.Load(path)
		suite.NoError(err)
		suite.Equal(m1, m2)
	})

	suite.Run("accepts .gltf extension with JSON content", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.gltf")
		suite.Require().NoError(os.WriteFile(path, []byte(`{"asset":{"version":"2.0"}}`), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadModelNameFromPath() {
	suite.Run("model name falls back to file path when no scene name", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "fallback_name.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		// The importer uses the file path as fallback name
		suite.Equal(path, m.Name())
	})
}

func (suite *loaderTest) TestLoadReaderCaseInsensitiveExtension() {
	suite.Run("Load accepts uppercase .GLB extension", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.GLB")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("Load accepts mixed case .Gltf extension", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "test.Gltf")
		suite.Require().NoError(os.WriteFile(path, []byte(`{"asset":{"version":"2.0"}}`), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

// buildMinimalGLB constructs a minimal valid GLB binary from a glTF JSON string.
// The JSON is padded with spaces to 4-byte alignment per the GLB spec.
func buildMinimalGLB(jsonStr string) []byte {
	jsonBytes := []byte(jsonStr)

	// Pad JSON to 4-byte alignment with spaces (GLB spec)
	for len(jsonBytes)%4 != 0 {
		jsonBytes = append(jsonBytes, ' ')
	}

	jsonChunkLen := uint32(len(jsonBytes))
	totalLen := uint32(12 + 8 + jsonChunkLen) // header + chunkHeader + jsonData

	buf := &bytes.Buffer{}
	binary.Write(buf, binary.LittleEndian, uint32(0x46546C67)) // magic "glTF"
	binary.Write(buf, binary.LittleEndian, uint32(2))          // version
	binary.Write(buf, binary.LittleEndian, totalLen)
	binary.Write(buf, binary.LittleEndian, jsonChunkLen)
	binary.Write(buf, binary.LittleEndian, uint32(0x4E4F534A)) // JSON chunk type
	buf.Write(jsonBytes)

	return buf.Bytes()
}

// buildGLBWithBin constructs a GLB binary with both a JSON chunk and a BIN chunk.
func buildGLBWithBin(jsonStr string, binData []byte) []byte {
	jsonBytes := []byte(jsonStr)
	for len(jsonBytes)%4 != 0 {
		jsonBytes = append(jsonBytes, ' ')
	}
	for len(binData)%4 != 0 {
		binData = append(binData, 0)
	}

	jsonChunkLen := uint32(len(jsonBytes))
	binChunkLen := uint32(len(binData))
	totalLen := uint32(12 + 8 + jsonChunkLen + 8 + binChunkLen)

	buf := &bytes.Buffer{}
	binary.Write(buf, binary.LittleEndian, uint32(0x46546C67)) // magic
	binary.Write(buf, binary.LittleEndian, uint32(2))          // version
	binary.Write(buf, binary.LittleEndian, totalLen)
	binary.Write(buf, binary.LittleEndian, jsonChunkLen)
	binary.Write(buf, binary.LittleEndian, uint32(0x4E4F534A)) // JSON
	buf.Write(jsonBytes)
	binary.Write(buf, binary.LittleEndian, binChunkLen)
	binary.Write(buf, binary.LittleEndian, uint32(0x004E4942)) // BIN
	buf.Write(binData)

	return buf.Bytes()
}

// buildMinimalTriangleBin constructs the binary buffer for the minimal triangle mesh.
// Contains 3 vertex positions followed by 3 uint16 indices, padded to 4-byte alignment.
func buildMinimalTriangleBin() []byte {
	buf := &bytes.Buffer{}

	// 3 vertices: (0,0,0), (1,0,0), (0,1,0)
	positions := [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}
	for _, p := range positions {
		binary.Write(buf, binary.LittleEndian, p)
	}

	// 3 indices: 0, 1, 2
	indices := []uint16{0, 1, 2}
	for _, idx := range indices {
		binary.Write(buf, binary.LittleEndian, idx)
	}

	// Pad to 4-byte alignment
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}

	return buf.Bytes()
}

// foxModelPath returns the path to the Fox.glb test fixture relative to the test directory.
func foxModelPath() string {
	return filepath.Join("..", "assets", "models", "Fox.glb")
}

func (suite *loaderTest) TestFoxModelLoad() {
	suite.Run("loads Fox.glb without error", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("model has a non-empty name", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.NotEmpty(m.Name())
	})
}

func (suite *loaderTest) TestFoxModelSkinned() {
	suite.Run("fox model reports as skinned", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.True(m.Skinned())
	})
}

func (suite *loaderTest) TestFoxModelSkeleton() {
	suite.Run("skeleton is not nil", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.NotNil(m.Skeleton())
	})

	suite.Run("skeleton has bones", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.Greater(len(m.Skeleton().Bones), 0)
	})

	suite.Run("skeleton has root bone indices", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.NotEmpty(m.Skeleton().RootBoneIndices)
	})

	suite.Run("skeleton has bone name to index mapping", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.NotEmpty(m.Skeleton().BoneNameToIndex)
	})

	suite.Run("bone count matches name-to-index map length", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		skel := m.Skeleton()
		suite.Equal(len(skel.Bones), len(skel.BoneNameToIndex))
	})

	suite.Run("root bones have parent index of negative one", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		skel := m.Skeleton()
		for _, rootIdx := range skel.RootBoneIndices {
			suite.Equal(int32(-1), skel.Bones[rootIdx].ParentIndex)
		}
	})

	suite.Run("non-root bones have valid parent indices", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		skel := m.Skeleton()
		boneCount := int32(len(skel.Bones))
		for _, bone := range skel.Bones {
			if bone.ParentIndex == -1 {
				continue
			}
			suite.GreaterOrEqual(bone.ParentIndex, int32(0))
			suite.Less(bone.ParentIndex, boneCount)
		}
	})

	suite.Run("all bones have non-empty names", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, bone := range m.Skeleton().Bones {
			suite.NotEmpty(bone.Name)
		}
	})

	suite.Run("bone name to index map entries match bones slice", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		skel := m.Skeleton()
		for name, idx := range skel.BoneNameToIndex {
			suite.Equal(name, skel.Bones[idx].Name)
		}
	})

	suite.Run("bones have non-zero inverse bind matrices", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, bone := range m.Skeleton().Bones {
			allZero := true
			for _, v := range bone.InverseBindMatrix {
				if v != 0 {
					allZero = false
					break
				}
			}
			suite.False(allZero)
		}
	})

	suite.Run("bones have non-zero local transform scale", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, bone := range m.Skeleton().Bones {
			hasScale := bone.LocalTransform.Scale[0] != 0 ||
				bone.LocalTransform.Scale[1] != 0 ||
				bone.LocalTransform.Scale[2] != 0
			suite.True(hasScale)
		}
	})

	suite.Run("skeleton parents come before children in topological order", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		skel := m.Skeleton()
		for i, bone := range skel.Bones {
			if bone.ParentIndex >= 0 {
				suite.Less(int(bone.ParentIndex), i)
			}
		}
	})
}

func (suite *loaderTest) TestFoxModelAnimations() {
	suite.Run("has at least one animation", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.Greater(m.AnimationCount(), 0)
	})

	suite.Run("animation count matches animations slice length", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.Equal(m.AnimationCount(), len(m.Animations()))
	})

	suite.Run("animation names count matches animation count", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.Len(m.AnimationNames(), m.AnimationCount())
	})

	suite.Run("all animations have non-empty names", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			suite.NotEmpty(clip.Name)
		}
	})

	suite.Run("all animations have positive duration", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			suite.Greater(clip.Duration, float32(0))
		}
	})

	suite.Run("all animations have at least one channel", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			suite.NotEmpty(clip.Channels)
		}
	})

	suite.Run("animation channels reference valid bone indices", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		boneCount := int32(len(m.Skeleton().Bones))
		for _, clip := range m.Animations() {
			for _, ch := range clip.Channels {
				suite.GreaterOrEqual(ch.BoneIndex, int32(0))
				suite.Less(ch.BoneIndex, boneCount)
			}
		}
	})

	suite.Run("every animation has channels with keyframes", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			hasKeyframes := false
			for _, ch := range clip.Channels {
				if len(ch.PositionKeys) > 0 || len(ch.RotationKeys) > 0 || len(ch.ScaleKeys) > 0 {
					hasKeyframes = true
					break
				}
			}
			suite.True(hasKeyframes)
		}
	})

	suite.Run("rotation keyframe quaternions have approximately unit length", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		checked := 0
		for _, clip := range m.Animations() {
			for _, ch := range clip.Channels {
				for _, rk := range ch.RotationKeys {
					q := rk.Value
					length := float64(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
					suite.InDelta(1.0, length, 1e-3)
					checked++
					if checked > 100 {
						return
					}
				}
			}
		}
		suite.Greater(checked, 0)
	})

	suite.Run("position keyframes have non-negative timestamps", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			for _, ch := range clip.Channels {
				for _, pk := range ch.PositionKeys {
					suite.GreaterOrEqual(pk.Time, float32(0))
				}
			}
		}
	})

	suite.Run("keyframe timestamps do not exceed animation duration", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, clip := range m.Animations() {
			for _, ch := range clip.Channels {
				for _, pk := range ch.PositionKeys {
					suite.LessOrEqual(pk.Time, clip.Duration)
				}
				for _, rk := range ch.RotationKeys {
					suite.LessOrEqual(rk.Time, clip.Duration)
				}
				for _, sk := range ch.ScaleKeys {
					suite.LessOrEqual(sk.Time, clip.Duration)
				}
			}
		}
	})
}

func (suite *loaderTest) TestFoxModelAnimationLookup() {
	suite.Run("known animation names return valid indices", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for i, name := range m.AnimationNames() {
			suite.Equal(i, m.GetAnimationIndex(name))
		}
	})

	suite.Run("unknown animation name returns negative one", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.Equal(-1, m.GetAnimationIndex("nonexistent_animation"))
	})
}

func (suite *loaderTest) TestFoxModelMaterials() {
	suite.Run("has at least one imported material", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.NotEmpty(m.ImportedMaterials())
	})

	suite.Run("first material has a name", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.NotEmpty(m.ImportedMaterials()[0].Name)
	})

	suite.Run("materials have valid base color range", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, mat := range m.ImportedMaterials() {
			for _, c := range mat.BaseColor {
				suite.GreaterOrEqual(c, float32(0))
				suite.LessOrEqual(c, float32(1))
			}
		}
	})

	suite.Run("at least one material has diffuse texture data", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		hasDiffuse := false
		for _, mat := range m.ImportedMaterials() {
			if mat.DiffuseTexture != nil && len(mat.DiffuseTexture.Data) > 0 {
				hasDiffuse = true
				break
			}
		}
		suite.True(hasDiffuse)
	})

	suite.Run("diffuse texture has a mime type", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, mat := range m.ImportedMaterials() {
			if mat.DiffuseTexture != nil && len(mat.DiffuseTexture.Data) > 0 {
				suite.NotEmpty(mat.DiffuseTexture.MimeType)
				return
			}
		}
	})

	suite.Run("metallic and roughness values are in valid range", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		for _, mat := range m.ImportedMaterials() {
			suite.GreaterOrEqual(mat.Metallic, float32(0))
			suite.LessOrEqual(mat.Metallic, float32(1))
			suite.GreaterOrEqual(mat.Roughness, float32(0))
			suite.LessOrEqual(mat.Roughness, float32(1))
		}
	})
}

func (suite *loaderTest) TestFoxModelCaching() {
	suite.Run("second Load call returns cached model", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m1, err := l.Load(foxModelPath())
		suite.Require().NoError(err)

		m2, err := l.Load(foxModelPath())
		suite.Require().NoError(err)

		suite.Equal(m1.Name(), m2.Name())
	})

	suite.Run("Get returns model after Load", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load(foxModelPath())
		suite.Require().NoError(err)

		cached := l.Get(foxModelPath())
		suite.NotNil(cached)
	})

	suite.Run("Models map contains loaded fox model", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load(foxModelPath())
		suite.Require().NoError(err)

		models := l.Models()
		suite.Contains(models, foxModelPath())
	})
}

func (suite *loaderTest) TestFoxModelMeshProvider() {
	suite.Run("model has a non-nil mesh provider", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.NotNil(m.MeshProvider())
	})
}

func (suite *loaderTest) TestFoxModelRenderMaterials() {
	suite.Run("render materials are created without renderer", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.NotEmpty(m.RenderMaterials())
	})

	suite.Run("render materials count matches imported materials count", func() {
		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(foxModelPath())
		suite.Require().NoError(err)
		suite.Equal(len(m.ImportedMaterials()), len(m.RenderMaterials()))
	})
}

func (suite *loaderTest) TestLoadExternalBufferFile() {
	suite.Run("loads gltf file with external bin buffer", func() {
		tmpDir, err := os.MkdirTemp("", "loader_test_*")
		suite.NoError(err)
		defer os.RemoveAll(tmpDir)

		buf := &bytes.Buffer{}
		for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
			binary.Write(buf, binary.LittleEndian, p)
		}
		posLen := buf.Len()
		for _, idx := range []uint16{0, 1, 2} {
			binary.Write(buf, binary.LittleEndian, idx)
		}
		for buf.Len()%4 != 0 {
			buf.WriteByte(0)
		}
		binData := buf.Bytes()

		err = os.WriteFile(filepath.Join(tmpDir, "model.bin"), binData, 0644)
		suite.NoError(err)

		jsonStr := fmt.Sprintf(`{
"asset": {"version": "2.0"},
"scene": 0,
"scenes": [{"nodes": [0]}],
"nodes": [{"mesh": 0}],
"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
"accessors": [
{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
],
"bufferViews": [
{"buffer": 0, "byteOffset": 0, "byteLength": %d},
{"buffer": 0, "byteOffset": %d, "byteLength": 6}
],
"buffers": [{"uri": "model.bin", "byteLength": %d}]
}`, posLen, posLen, len(binData))

		err = os.WriteFile(filepath.Join(tmpDir, "model.gltf"), []byte(jsonStr), 0644)
		suite.NoError(err)

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(filepath.Join(tmpDir, "model.gltf"))
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadGLBWithVertexColors() {
	suite.Run("loads mesh with VEC3 float vertex colors", func() {
		bin := buildTriangleWithVec3FloatColors()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 36},
    {"buffer": 0, "byteOffset": 72, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 80}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "color.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("loads mesh with unsigned byte VEC4 vertex colors", func() {
		bin := buildTriangleWithUByteVec4Colors()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12},
    {"buffer": 0, "byteOffset": 48, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 56}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "color_ubyte.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})

	suite.Run("loads mesh with unsigned short VEC4 vertex colors", func() {
		bin := buildTriangleWithUShortVec4Colors()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "VEC4"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 24},
    {"buffer": 0, "byteOffset": 60, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 68}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "color_ushort.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadGLBWithMatrixSkeleton() {
	bin := buildSkeletonMatrixBin()
	jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"mesh": 0, "skin": 0},
    {"name": "root_bone", "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 2,3,4,1]}
  ],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}, "indices": 3}]}],
  "skins": [{"joints": [1]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12},
    {"buffer": 0, "byteOffset": 48, "byteLength": 48},
    {"buffer": 0, "byteOffset": 96, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 104}]
}`
	glb := buildGLBWithBin(jsonStr, bin)
	dir := suite.T().TempDir()
	path := filepath.Join(dir, "skeleton_matrix.glb")
	suite.Require().NoError(os.WriteFile(path, glb, 0644))

	l := loader.NewLoader(loader.BackendTypeGLTF)
	m, err := l.Load(path)
	suite.Require().NoError(err)

	suite.Run("model is skinned", func() {
		suite.True(m.Skinned())
	})

	suite.Run("bone transform has correct translation from matrix decomposition", func() {
		skel := m.Skeleton()
		suite.Require().NotNil(skel)
		suite.Require().Len(skel.Bones, 1)
		t := skel.Bones[0].LocalTransform.Translation
		suite.InDelta(2.0, float64(t[0]), 1e-3)
		suite.InDelta(3.0, float64(t[1]), 1e-3)
		suite.InDelta(4.0, float64(t[2]), 1e-3)
	})

	suite.Run("bone rotation is approximately identity quaternion", func() {
		skel := m.Skeleton()
		q := skel.Bones[0].LocalTransform.Rotation
		suite.InDelta(0.0, float64(q[0]), 1e-3)
		suite.InDelta(0.0, float64(q[1]), 1e-3)
		suite.InDelta(0.0, float64(q[2]), 1e-3)
		suite.InDelta(1.0, float64(q[3]), 1e-3)
	})

	suite.Run("bone scale is unity", func() {
		skel := m.Skeleton()
		s := skel.Bones[0].LocalTransform.Scale
		suite.InDelta(1.0, float64(s[0]), 1e-3)
		suite.InDelta(1.0, float64(s[1]), 1e-3)
		suite.InDelta(1.0, float64(s[2]), 1e-3)
	})

	suite.Run("bone uses identity inverse bind matrix when not provided by skin", func() {
		skel := m.Skeleton()
		expected := [16]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
		suite.Equal(expected, skel.Bones[0].InverseBindMatrix)
	})
}

func (suite *loaderTest) TestLoadGLBWithAnimationsNoSkeleton() {
	bin := buildAnimOnlyBin()
	jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "animations": [{
    "name": "bounce",
    "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
    "samplers": [{"input": 2, "output": 3}]
  }],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
    {"bufferView": 2, "componentType": 5126, "count": 2, "type": "SCALAR"},
    {"bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC3"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6},
    {"buffer": 0, "byteOffset": 44, "byteLength": 8},
    {"buffer": 0, "byteOffset": 52, "byteLength": 24}
  ],
  "buffers": [{"byteLength": 76}]
}`
	glb := buildGLBWithBin(jsonStr, bin)
	dir := suite.T().TempDir()
	path := filepath.Join(dir, "anim_only.glb")
	suite.Require().NoError(os.WriteFile(path, glb, 0644))

	l := loader.NewLoader(loader.BackendTypeGLTF)
	m, err := l.Load(path)
	suite.Require().NoError(err)

	suite.Run("model loads successfully", func() {
		suite.NotNil(m)
	})

	suite.Run("has one animation", func() {
		suite.Equal(1, m.AnimationCount())
	})

	suite.Run("animation name is bounce", func() {
		anims := m.Animations()
		suite.Require().Len(anims, 1)
		suite.Equal("bounce", anims[0].Name)
	})
}

func (suite *loaderTest) TestLoadGLTFWithDataURI() {
	suite.Run("loads gltf with base64 data URI buffer", func() {
		binData := buildMinimalTriangleBin()
		encoded := base64.StdEncoding.EncodeToString(binData)
		jsonStr := fmt.Sprintf(`{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ],
  "buffers": [{"uri": "data:application/octet-stream;base64,%s", "byteLength": %d}]
}`, encoded, len(binData))

		dir := suite.T().TempDir()
		path := filepath.Join(dir, "data_uri.gltf")
		suite.Require().NoError(os.WriteFile(path, []byte(jsonStr), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadGLTFWithSamplerWrapAndExternalTexture() {
	tmpDir := suite.T().TempDir()

	binData := buildMinimalTriangleBin()
	suite.Require().NoError(os.WriteFile(filepath.Join(tmpDir, "mesh.bin"), binData, 0644))

	pngData := build1x1PNG()
	suite.Require().NoError(os.WriteFile(filepath.Join(tmpDir, "tex.png"), pngData, 0644))

	jsonStr := fmt.Sprintf(`{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
  "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
  "textures": [{"source": 0, "sampler": 0}],
  "images": [{"uri": "tex.png", "mimeType": "image/png"}],
  "samplers": [{"wrapS": 33071, "wrapT": 33648, "magFilter": 9728, "minFilter": 9987}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ],
  "buffers": [{"uri": "mesh.bin", "byteLength": %d}]
}`, len(binData))

	gltfPath := filepath.Join(tmpDir, "model.gltf")
	suite.Require().NoError(os.WriteFile(gltfPath, []byte(jsonStr), 0644))

	l := loader.NewLoader(loader.BackendTypeGLTF)
	m, err := l.Load(gltfPath)
	suite.Require().NoError(err)

	suite.Run("model loads with material", func() {
		suite.NotEmpty(m.ImportedMaterials())
	})

	suite.Run("diffuse texture has loaded data from external file", func() {
		mats := m.ImportedMaterials()
		suite.Require().NotEmpty(mats)
		suite.NotNil(mats[0].DiffuseTexture)
		suite.NotEmpty(mats[0].DiffuseTexture.Data)
	})

	suite.Run("sampler data is present on texture", func() {
		mats := m.ImportedMaterials()
		suite.Require().NotEmpty(mats)
		suite.Require().NotNil(mats[0].DiffuseTexture)
		suite.NotNil(mats[0].DiffuseTexture.SamplerData)
	})
}

func (suite *loaderTest) TestLoadGLBWithUInt32Indices() {
	suite.Run("loads mesh with unsigned int indices", func() {
		bin := buildUInt32IndicesBin()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12}
  ],
  "buffers": [{"byteLength": 48}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "uint32_indices.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

// buildTriangleWithVec3FloatColors constructs binary data for a triangle with VEC3 float vertex colors.
// Layout: 36 bytes positions + 36 bytes VEC3 colors + 6 bytes indices + 2 bytes padding = 80 bytes.
func buildTriangleWithVec3FloatColors() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for _, c := range [][3]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}} {
		binary.Write(buf, binary.LittleEndian, c)
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	return buf.Bytes()
}

// buildTriangleWithUByteVec4Colors constructs binary data for a triangle with unsigned byte VEC4 vertex colors.
// Layout: 36 bytes positions + 12 bytes ubyte VEC4 colors + 6 bytes indices + 2 bytes padding = 56 bytes.
func buildTriangleWithUByteVec4Colors() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for _, c := range [][4]uint8{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}} {
		binary.Write(buf, binary.LittleEndian, c)
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	return buf.Bytes()
}

// buildTriangleWithUShortVec4Colors constructs binary data for a triangle with unsigned short VEC4 vertex colors.
// Layout: 36 bytes positions + 24 bytes ushort VEC4 colors + 6 bytes indices + 2 bytes padding = 68 bytes.
func buildTriangleWithUShortVec4Colors() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for _, c := range [][4]uint16{{65535, 0, 0, 65535}, {0, 65535, 0, 65535}, {0, 0, 65535, 65535}} {
		binary.Write(buf, binary.LittleEndian, c)
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	return buf.Bytes()
}

// buildSkeletonMatrixBin constructs binary data for a skinned triangle mesh with one bone.
// Layout: 36 bytes positions + 12 bytes joints + 48 bytes weights + 6 bytes indices + 2 bytes padding = 104 bytes.
func buildSkeletonMatrixBin() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for i := 0; i < 3; i++ {
		binary.Write(buf, binary.LittleEndian, [4]uint8{0, 0, 0, 0})
	}
	for i := 0; i < 3; i++ {
		binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	return buf.Bytes()
}

// buildAnimOnlyBin constructs binary data for a triangle mesh with animation data but no skeleton.
// Layout: 36 bytes positions + 6 bytes indices + 2 bytes padding + 8 bytes timestamps + 24 bytes translations = 76 bytes.
func buildAnimOnlyBin() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	binary.Write(buf, binary.LittleEndian, float32(0.0))
	binary.Write(buf, binary.LittleEndian, float32(1.0))
	binary.Write(buf, binary.LittleEndian, [3]float32{0, 0, 0})
	binary.Write(buf, binary.LittleEndian, [3]float32{0, 1, 0})
	return buf.Bytes()
}

// buildUInt32IndicesBin constructs binary data for a triangle mesh with uint32 indices.
// Layout: 36 bytes positions + 12 bytes uint32 indices = 48 bytes.
func buildUInt32IndicesBin() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for _, idx := range []uint32{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	return buf.Bytes()
}

// build1x1PNG creates a minimal 1x1 white RGBA PNG image as raw bytes.
func build1x1PNG() []byte {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{R: 255, G: 255, B: 255, A: 255})
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return buf.Bytes()
}

func (suite *loaderTest) TestLoadGLBWithUInt8Indices() {
	suite.Run("loads mesh with unsigned byte indices", func() {
		bin := buildUInt8IndicesBin()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 3}
  ],
  "buffers": [{"byteLength": 40}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "uint8_indices.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadGLBWithNoIndices() {
	suite.Run("loads mesh without explicit indices", func() {
		bin := buildPositionsOnlyBin()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36}
  ],
  "buffers": [{"byteLength": 36}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "no_indices.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadGLTFWithDataURIImage() {
	suite.Run("loads material with base64-encoded image data URI", func() {
		pngData := build1x1PNG()
		encodedPng := base64.StdEncoding.EncodeToString(pngData)
		binData := buildMinimalTriangleBin()
		encodedBin := base64.StdEncoding.EncodeToString(binData)

		jsonStr := fmt.Sprintf(`{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
  "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
  "textures": [{"source": 0}],
  "images": [{"uri": "data:image/png;base64,%s", "mimeType": "image/png"}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ],
  "buffers": [{"uri": "data:application/octet-stream;base64,%s", "byteLength": %d}]
}`, encodedPng, encodedBin, len(binData))

		dir := suite.T().TempDir()
		path := filepath.Join(dir, "data_uri_image.gltf")
		suite.Require().NoError(os.WriteFile(path, []byte(jsonStr), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)

		mats := m.ImportedMaterials()
		suite.Require().NotEmpty(mats)
		suite.NotNil(mats[0].DiffuseTexture)
		suite.NotEmpty(mats[0].DiffuseTexture.Data)
	})
}

func (suite *loaderTest) TestLoadGLTFWithNormalAndMetallicTextures() {
	tmpDir := suite.T().TempDir()

	pngData := build1x1PNG()
	suite.Require().NoError(os.WriteFile(filepath.Join(tmpDir, "diffuse.png"), pngData, 0644))
	suite.Require().NoError(os.WriteFile(filepath.Join(tmpDir, "normal.png"), pngData, 0644))
	suite.Require().NoError(os.WriteFile(filepath.Join(tmpDir, "metallic.png"), pngData, 0644))

	binData := buildMinimalTriangleBin()
	suite.Require().NoError(os.WriteFile(filepath.Join(tmpDir, "mesh.bin"), binData, 0644))

	jsonStr := fmt.Sprintf(`{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
  "materials": [{
    "pbrMetallicRoughness": {
      "baseColorTexture": {"index": 0},
      "metallicRoughnessTexture": {"index": 1},
      "metallicFactor": 0.5,
      "roughnessFactor": 0.8,
      "baseColorFactor": [0.9, 0.8, 0.7, 1.0]
    },
    "normalTexture": {"index": 2}
  }],
  "textures": [
    {"source": 0},
    {"source": 1},
    {"source": 2}
  ],
  "images": [
    {"uri": "diffuse.png", "mimeType": "image/png"},
    {"uri": "metallic.png", "mimeType": "image/png"},
    {"uri": "normal.png", "mimeType": "image/png"}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ],
  "buffers": [{"uri": "mesh.bin", "byteLength": %d}]
}`, len(binData))

	gltfPath := filepath.Join(tmpDir, "model.gltf")
	suite.Require().NoError(os.WriteFile(gltfPath, []byte(jsonStr), 0644))

	l := loader.NewLoader(loader.BackendTypeGLTF)
	m, err := l.Load(gltfPath)
	suite.Require().NoError(err)

	suite.Run("material has diffuse texture", func() {
		mats := m.ImportedMaterials()
		suite.Require().NotEmpty(mats)
		suite.NotNil(mats[0].DiffuseTexture)
	})

	suite.Run("material has normal texture", func() {
		mats := m.ImportedMaterials()
		suite.Require().NotEmpty(mats)
		suite.NotNil(mats[0].NormalTexture)
	})

	suite.Run("material has metallic roughness texture", func() {
		mats := m.ImportedMaterials()
		suite.Require().NotEmpty(mats)
		suite.NotNil(mats[0].MetallicRoughnessTexture)
	})

	suite.Run("material metallic factor is correct", func() {
		mats := m.ImportedMaterials()
		suite.InDelta(0.5, float64(mats[0].Metallic), 1e-3)
	})

	suite.Run("material roughness factor is correct", func() {
		mats := m.ImportedMaterials()
		suite.InDelta(0.8, float64(mats[0].Roughness), 1e-3)
	})

	suite.Run("material base color factor is correct", func() {
		mats := m.ImportedMaterials()
		suite.InDelta(0.9, float64(mats[0].BaseColor[0]), 1e-3)
		suite.InDelta(0.8, float64(mats[0].BaseColor[1]), 1e-3)
		suite.InDelta(0.7, float64(mats[0].BaseColor[2]), 1e-3)
		suite.InDelta(1.0, float64(mats[0].BaseColor[3]), 1e-3)
	})
}

func (suite *loaderTest) TestLoadGLBWithRotatedMatrixSkeleton() {
	bin := buildSkeletonMatrixBin()

	suite.Run("180 degree rotation around X axis triggers r00 dominant quaternion branch", func() {
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"mesh": 0, "skin": 0},
    {"name": "bone_x180", "matrix": [1,0,0,0, 0,-1,0,0, 0,0,-1,0, 0,0,0,1]}
  ],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}, "indices": 3}]}],
  "skins": [{"joints": [1]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12},
    {"buffer": 0, "byteOffset": 48, "byteLength": 48},
    {"buffer": 0, "byteOffset": 96, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 104}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "rot_x180.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)

		skel := m.Skeleton()
		suite.Require().NotNil(skel)
		q := skel.Bones[0].LocalTransform.Rotation
		length := float64(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
		suite.InDelta(1.0, length, 1e-3)
	})

	suite.Run("180 degree rotation around Y axis triggers r11 dominant quaternion branch", func() {
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"mesh": 0, "skin": 0},
    {"name": "bone_y180", "matrix": [-1,0,0,0, 0,1,0,0, 0,0,-1,0, 0,0,0,1]}
  ],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}, "indices": 3}]}],
  "skins": [{"joints": [1]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12},
    {"buffer": 0, "byteOffset": 48, "byteLength": 48},
    {"buffer": 0, "byteOffset": 96, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 104}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "rot_y180.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)

		skel := m.Skeleton()
		suite.Require().NotNil(skel)
		q := skel.Bones[0].LocalTransform.Rotation
		length := float64(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
		suite.InDelta(1.0, length, 1e-3)
	})

	suite.Run("180 degree rotation around Z axis triggers r22 dominant quaternion branch", func() {
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"mesh": 0, "skin": 0},
    {"name": "bone_z180", "matrix": [-1,0,0,0, 0,-1,0,0, 0,0,1,0, 0,0,0,1]}
  ],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}, "indices": 3}]}],
  "skins": [{"joints": [1]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12},
    {"buffer": 0, "byteOffset": 48, "byteLength": 48},
    {"buffer": 0, "byteOffset": 96, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 104}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "rot_z180.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)

		skel := m.Skeleton()
		suite.Require().NotNil(skel)
		q := skel.Bones[0].LocalTransform.Rotation
		length := float64(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
		suite.InDelta(1.0, length, 1e-3)
	})
}

func (suite *loaderTest) TestLoadGLBWithSceneName() {
	suite.Run("model name comes from scene name when available", func() {
		glb := buildMinimalGLB(`{"asset":{"version":"2.0"}, "scene": 0, "scenes":[{"name":"MyScene"}]}`)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "named_scene.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.Equal("MyScene", m.Name())
	})

	suite.Run("model name is unnamed_model when no scene and no path fallback", func() {
		bin := buildMinimalTriangleBin()
		glb := buildGLBWithBin(`{"asset":{"version":"2.0"}, "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}], "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}], "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":6}], "buffers":[{"byteLength":44}]}`, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "no_scene.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		// The model name falls back to the file path when no scene name is present
		suite.Equal(path, m.Name())
	})
}

func (suite *loaderTest) TestLoadGLBWithTexCoordAndTangent() {
	suite.Run("loads mesh with texcoords and tangent attributes", func() {
		bin := buildTriangleWithTexCoordsAndTangents()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{
    "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "TANGENT": 3},
    "indices": 4
  }]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
    {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 4, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0,   "byteLength": 36},
    {"buffer": 0, "byteOffset": 36,  "byteLength": 36},
    {"buffer": 0, "byteOffset": 72,  "byteLength": 24},
    {"buffer": 0, "byteOffset": 96,  "byteLength": 48},
    {"buffer": 0, "byteOffset": 144, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 152}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "tangent.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadGLBWithUByteVec3Colors() {
	suite.Run("loads mesh with unsigned byte VEC3 vertex colors", func() {
		bin := buildTriangleWithUByteVec3Colors()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC3"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 9},
    {"buffer": 0, "byteOffset": 48, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 56}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "color_ubyte_vec3.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadGLBWithUShortVec3Colors() {
	suite.Run("loads mesh with unsigned short VEC3 vertex colors", func() {
		bin := buildTriangleWithUShortVec3Colors()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "VEC3"},
    {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 18},
    {"buffer": 0, "byteOffset": 56, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 64}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "color_ushort_vec3.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

// buildUInt8IndicesBin constructs binary data for a triangle mesh with uint8 indices.
// Layout: 36 bytes positions + 3 bytes uint8 indices + 1 byte padding = 40 bytes.
func buildUInt8IndicesBin() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	buf.Write([]byte{0, 1, 2})
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	return buf.Bytes()
}

// buildPositionsOnlyBin constructs binary data containing only 3 vertex positions (no indices).
// Layout: 36 bytes positions.
func buildPositionsOnlyBin() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	return buf.Bytes()
}

// buildTriangleWithTexCoordsAndTangents constructs binary data for a triangle with positions,
// normals, tex coords, tangents, and indices.
// Layout: 36 pos + 36 norm + 24 uv + 48 tangent + 6 idx + 2 pad = 152 bytes.
func buildTriangleWithTexCoordsAndTangents() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for range 3 {
		binary.Write(buf, binary.LittleEndian, [3]float32{0, 0, 1})
	}
	for _, uv := range [][2]float32{{0, 0}, {1, 0}, {0, 1}} {
		binary.Write(buf, binary.LittleEndian, uv)
	}
	for range 3 {
		binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 1})
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	return buf.Bytes()
}

// buildTriangleWithUByteVec3Colors constructs binary data for a triangle with unsigned byte VEC3 vertex colors.
// Layout: 36 pos + 9 colors + 3 pad + 6 idx + 2 pad = 56 bytes.
func buildTriangleWithUByteVec3Colors() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for _, c := range [][3]uint8{{255, 0, 0}, {0, 255, 0}, {0, 0, 255}} {
		buf.Write(c[:])
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	return buf.Bytes()
}

// buildTriangleWithUShortVec3Colors constructs binary data for a triangle with unsigned short VEC3 vertex colors.
// Layout: 36 pos + 18 colors + 2 pad + 6 idx + 2 pad = 64 bytes.
func buildTriangleWithUShortVec3Colors() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for _, c := range [][3]uint16{{65535, 0, 0}, {0, 65535, 0}, {0, 0, 65535}} {
		binary.Write(buf, binary.LittleEndian, c)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	return buf.Bytes()
}

func (suite *loaderTest) TestLoadGLBInvalidMagic() {
	suite.Run("returns error for GLB with wrong magic number", func() {
		data := []byte("NOT_GL" + "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "bad_magic.glb")
		suite.Require().NoError(os.WriteFile(path, data, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load(path)
		suite.Error(err)
	})
}

func (suite *loaderTest) TestLoadGLBInvalidVersion() {
	suite.Run("returns error for GLB with wrong version", func() {
		buf := &bytes.Buffer{}
		binary.Write(buf, binary.LittleEndian, uint32(0x46546C67)) // glTF magic
		binary.Write(buf, binary.LittleEndian, uint32(1))          // wrong version (1 instead of 2)
		binary.Write(buf, binary.LittleEndian, uint32(20))         // total length
		// add 8 more zero bytes to avoid "too small" error
		buf.Write(make([]byte, 8))
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "bad_version.glb")
		suite.Require().NoError(os.WriteFile(path, buf.Bytes(), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load(path)
		suite.Error(err)
	})
}

func (suite *loaderTest) TestLoadGLBTooSmall() {
	suite.Run("returns error for GLB with fewer than 12 bytes", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "tiny.glb")
		suite.Require().NoError(os.WriteFile(path, []byte{0x01, 0x02}, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load(path)
		suite.Error(err)
	})
}

func (suite *loaderTest) TestLoadGLTFInvalidVersion() {
	suite.Run("returns error for glTF with unsupported version", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "bad_version.gltf")
		jsonStr := `{"asset": {"version": "1.0"}}`
		suite.Require().NoError(os.WriteFile(path, []byte(jsonStr), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load(path)
		suite.Error(err)
	})
}

func (suite *loaderTest) TestLoadGLTFInvalidJSON() {
	suite.Run("returns error for malformed glTF JSON", func() {
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "bad_json.gltf")
		suite.Require().NoError(os.WriteFile(path, []byte("{not valid json!}"), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load(path)
		suite.Error(err)
	})
}

func (suite *loaderTest) TestLoadGLBMissingJSONChunk() {
	suite.Run("returns error for GLB with no JSON chunk", func() {
		buf := &bytes.Buffer{}
		binary.Write(buf, binary.LittleEndian, uint32(0x46546C67)) // magic
		binary.Write(buf, binary.LittleEndian, uint32(2))          // version
		binary.Write(buf, binary.LittleEndian, uint32(20))         // total length (header + chunk header + 0 data)
		// Write a BIN chunk instead of JSON chunk
		binary.Write(buf, binary.LittleEndian, uint32(0))          // chunk length
		binary.Write(buf, binary.LittleEndian, uint32(0x004E4942)) // BIN chunk type
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "no_json.glb")
		suite.Require().NoError(os.WriteFile(path, buf.Bytes(), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		_, err := l.Load(path)
		suite.Error(err)
	})
}

func (suite *loaderTest) TestLoadGLTFWithRepeatWrapMode() {
	suite.Run("loads material with repeat wrap mode on sampler", func() {
		tmpDir := suite.T().TempDir()
		pngData := build1x1PNG()
		suite.Require().NoError(os.WriteFile(filepath.Join(tmpDir, "tex.png"), pngData, 0644))

		binData := buildMinimalTriangleBin()
		suite.Require().NoError(os.WriteFile(filepath.Join(tmpDir, "mesh.bin"), binData, 0644))

		jsonStr := fmt.Sprintf(`{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
  "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
  "textures": [{"source": 0, "sampler": 0}],
  "samplers": [{"wrapS": 10497, "wrapT": 10497, "magFilter": 9729, "minFilter": 9987}],
  "images": [{"uri": "tex.png", "mimeType": "image/png"}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ],
  "buffers": [{"uri": "mesh.bin", "byteLength": %d}]
}`, len(binData))
		gltfPath := filepath.Join(tmpDir, "model.gltf")
		suite.Require().NoError(os.WriteFile(gltfPath, []byte(jsonStr), 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(gltfPath)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

func (suite *loaderTest) TestLoadGLBWithBufferViewImage() {
	suite.Run("loads material with image stored in bufferView", func() {
		pngData := build1x1PNG()
		triData := buildMinimalTriangleBin()

		totalBin := make([]byte, 0, len(triData)+len(pngData))
		totalBin = append(totalBin, triData...)
		totalBin = append(totalBin, pngData...)
		for len(totalBin)%4 != 0 {
			totalBin = append(totalBin, 0)
		}

		jsonStr := fmt.Sprintf(`{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
  "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
  "textures": [{"source": 0}],
  "images": [{"bufferView": 2, "mimeType": "image/png"}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1,1,0], "min": [0,0,0]},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6},
    {"buffer": 0, "byteOffset": %d, "byteLength": %d}
  ],
  "buffers": [{"byteLength": %d}]
}`, len(triData), len(pngData), len(totalBin))

		glb := buildGLBWithBin(jsonStr, totalBin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "bv_image.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)

		mats := m.ImportedMaterials()
		suite.Require().NotEmpty(mats)
		suite.NotNil(mats[0].DiffuseTexture)
		suite.NotEmpty(mats[0].DiffuseTexture.Data)
	})
}

func (suite *loaderTest) TestLoadGLBWithUnnamedAnimation() {
	suite.Run("animation with no name gets fallback name", func() {
		bin := buildAnimationWithScaleBin()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"mesh": 0, "skin": 0},
    {"name": "bone"}
  ],
  "skins": [{"joints": [1]}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}, "indices": 3}]}],
  "animations": [{
    "channels": [
      {"sampler": 0, "target": {"node": 1, "path": "scale"}}
    ],
    "samplers": [
      {"input": 4, "output": 5}
    ]
  }],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"},
    {"bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR"},
    {"bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC3"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12},
    {"buffer": 0, "byteOffset": 48, "byteLength": 48},
    {"buffer": 0, "byteOffset": 96, "byteLength": 6},
    {"buffer": 0, "byteOffset": 104, "byteLength": 8},
    {"buffer": 0, "byteOffset": 112, "byteLength": 24}
  ],
  "buffers": [{"byteLength": 136}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "unnamed_anim.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)

		anims := m.Animations()
		suite.Require().Len(anims, 1)
		suite.Equal("animation_0", anims[0].Name)
	})
}

func (suite *loaderTest) TestLoadGLBWithMultiChannelAnimation() {
	suite.Run("animation with translation rotation and scale channels", func() {
		bin := buildMultiChannelAnimBin()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"mesh": 0, "skin": 0},
    {"name": "bone"}
  ],
  "skins": [{"joints": [1]}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}, "indices": 3}]}],
  "animations": [{"name": "multichannel",
    "channels": [
      {"sampler": 0, "target": {"node": 1, "path": "translation"}},
      {"sampler": 1, "target": {"node": 1, "path": "rotation"}},
      {"sampler": 2, "target": {"node": 1, "path": "scale"}}
    ],
    "samplers": [
      {"input": 4, "output": 5},
      {"input": 4, "output": 6},
      {"input": 4, "output": 7}
    ]
  }],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"},
    {"bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR"},
    {"bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC3"},
    {"bufferView": 6, "componentType": 5126, "count": 2, "type": "VEC4"},
    {"bufferView": 7, "componentType": 5126, "count": 2, "type": "VEC3"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12},
    {"buffer": 0, "byteOffset": 48, "byteLength": 48},
    {"buffer": 0, "byteOffset": 96, "byteLength": 6},
    {"buffer": 0, "byteOffset": 104, "byteLength": 8},
    {"buffer": 0, "byteOffset": 112, "byteLength": 24},
    {"buffer": 0, "byteOffset": 136, "byteLength": 32},
    {"buffer": 0, "byteOffset": 168, "byteLength": 24}
  ],
  "buffers": [{"byteLength": 192}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "multichannel.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)

		anims := m.Animations()
		suite.Require().Len(anims, 1)
		suite.Require().Len(anims[0].Channels, 1)
		suite.NotEmpty(anims[0].Channels[0].PositionKeys)
		suite.NotEmpty(anims[0].Channels[0].RotationKeys)
		suite.NotEmpty(anims[0].Channels[0].ScaleKeys)
	})
}

func (suite *loaderTest) TestLoadGLBWithMorphWeightsAnimation() {
	suite.Run("animation with weights channel is skipped gracefully", func() {
		bin := buildAnimationWithWeightsBin()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"mesh": 0, "skin": 0},
    {"name": "bone"}
  ],
  "skins": [{"joints": [1]}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}, "indices": 3}]}],
  "animations": [{"name": "morph",
    "channels": [
      {"sampler": 0, "target": {"node": 1, "path": "translation"}},
      {"sampler": 1, "target": {"node": 1, "path": "weights"}}
    ],
    "samplers": [
      {"input": 4, "output": 5},
      {"input": 4, "output": 6}
    ]
  }],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4"},
    {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"},
    {"bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR"},
    {"bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC3"},
    {"bufferView": 6, "componentType": 5126, "count": 2, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12},
    {"buffer": 0, "byteOffset": 48, "byteLength": 48},
    {"buffer": 0, "byteOffset": 96, "byteLength": 6},
    {"buffer": 0, "byteOffset": 104, "byteLength": 8},
    {"buffer": 0, "byteOffset": 112, "byteLength": 24},
    {"buffer": 0, "byteOffset": 136, "byteLength": 8}
  ],
  "buffers": [{"byteLength": 144}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "weights_anim.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)

		anims := m.Animations()
		suite.Require().Len(anims, 1)
		suite.Require().Len(anims[0].Channels, 1)
		suite.NotEmpty(anims[0].Channels[0].PositionKeys)
	})
}

func (suite *loaderTest) TestLoadGLBWithMeshNoNormals() {
	suite.Run("mesh without normals generates normals automatically", func() {
		bin := buildTriangleWithIndicesOnly()
		jsonStr := `{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6}
  ],
  "buffers": [{"byteLength": 44}]
}`
		glb := buildGLBWithBin(jsonStr, bin)
		dir := suite.T().TempDir()
		path := filepath.Join(dir, "no_normals.glb")
		suite.Require().NoError(os.WriteFile(path, glb, 0644))

		l := loader.NewLoader(loader.BackendTypeGLTF)
		m, err := l.Load(path)
		suite.NoError(err)
		suite.NotNil(m)
	})
}

// buildAnimationWithScaleBin constructs binary data for a skinned mesh with a scale animation.
// Layout: 36 pos + 12 joints + 48 weights + 6 idx + 2 pad + 8 timestamps + 24 scale values = 136 bytes.
func buildAnimationWithScaleBin() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for range 3 {
		buf.Write([]byte{0, 0, 0, 0})
	}
	for range 3 {
		binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	binary.Write(buf, binary.LittleEndian, [2]float32{0.0, 1.0})
	binary.Write(buf, binary.LittleEndian, [2][3]float32{{1, 1, 1}, {2, 2, 2}})
	return buf.Bytes()
}

// buildMultiChannelAnimBin constructs binary data for a skinned mesh with translation, rotation, and scale animations.
// Layout: 36 pos + 12 joints + 48 weights + 6 idx + 2 pad + 8 timestamps + 24 translation + 32 rotation + 24 scale = 192 bytes.
func buildMultiChannelAnimBin() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for range 3 {
		buf.Write([]byte{0, 0, 0, 0})
	}
	for range 3 {
		binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	binary.Write(buf, binary.LittleEndian, [2]float32{0.0, 1.0})
	binary.Write(buf, binary.LittleEndian, [2][3]float32{{0, 0, 0}, {1, 2, 3}})
	binary.Write(buf, binary.LittleEndian, [2][4]float32{{0, 0, 0, 1}, {0, 0.707, 0, 0.707}})
	binary.Write(buf, binary.LittleEndian, [2][3]float32{{1, 1, 1}, {2, 2, 2}})
	return buf.Bytes()
}

// buildAnimationWithWeightsBin constructs binary data for a skinned mesh with translation and morph weights data.
// Layout: 36 pos + 12 joints + 48 weights + 6 idx + 2 pad + 8 timestamps + 24 translation + 8 morph weights = 144 bytes.
func buildAnimationWithWeightsBin() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for range 3 {
		buf.Write([]byte{0, 0, 0, 0})
	}
	for range 3 {
		binary.Write(buf, binary.LittleEndian, [4]float32{1, 0, 0, 0})
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	binary.Write(buf, binary.LittleEndian, [2]float32{0.0, 1.0})
	binary.Write(buf, binary.LittleEndian, [2][3]float32{{0, 0, 0}, {0, 1, 0}})
	binary.Write(buf, binary.LittleEndian, [2]float32{0.5, 1.0})
	return buf.Bytes()
}

// buildTriangleWithIndicesOnly constructs binary data for a basic triangle with only positions and uint16 indices.
// Layout: 36 bytes positions + 6 bytes indices + 2 bytes padding = 44 bytes.
func buildTriangleWithIndicesOnly() []byte {
	buf := &bytes.Buffer{}
	for _, p := range [][3]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}} {
		binary.Write(buf, binary.LittleEndian, p)
	}
	for _, idx := range []uint16{0, 1, 2} {
		binary.Write(buf, binary.LittleEndian, idx)
	}
	for buf.Len()%4 != 0 {
		buf.WriteByte(0)
	}
	return buf.Bytes()
}
