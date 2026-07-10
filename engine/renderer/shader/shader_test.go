package shader_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/Carmen-Shannon/oxy-go/engine/renderer/shader"
)

// ── Suite 1: ShaderType constants ─────────────────────────────────────────────

type shaderTypeTest struct {
	suite.Suite
}

func TestShaderType(t *testing.T) {
	suite.Run(t, new(shaderTypeTest))
}

func (suite *shaderTypeTest) TestShaderTypeConstants() {
	suite.Run("ShaderTypeCompute equals zero", func() {
		suite.Equal(shader.ShaderType(0), shader.ShaderTypeCompute)
	})
	suite.Run("ShaderTypeVertex equals one", func() {
		suite.Equal(shader.ShaderType(1), shader.ShaderTypeVertex)
	})
	suite.Run("ShaderTypeFragment equals two", func() {
		suite.Equal(shader.ShaderType(2), shader.ShaderTypeFragment)
	})
}

// ── Suite 2: PreProcessor ─────────────────────────────────────────────────────

type preProcessorTest struct {
	suite.Suite
}

func TestPreProcessor(t *testing.T) {
	suite.Run(t, new(preProcessorTest))
}

func (suite *preProcessorTest) TestDeclarations() {
	suite.Run("returns nil before first Process call", func() {
		pp := shader.NewPreProcessor()
		suite.Nil(pp.Declarations())
	})
}

func (suite *preProcessorTest) TestProcess() {
	suite.Run("plain source unchanged", func() {
		pp := shader.NewPreProcessor()
		src := "fn main() {}"
		out, err := pp.Process(src)
		suite.NoError(err)
		suite.Equal(src, out)
	})

	suite.Run("include camera replaces line", func() {
		pp := shader.NewPreProcessor()
		src := "@oxy: include camera"
		out, err := pp.Process(src)
		suite.NoError(err)
		suite.NotEmpty(out)
		suite.False(strings.Contains(out, "@oxy: include camera"))
	})

	suite.Run("group annotation generates declaration", func() {
		pp := shader.NewPreProcessor()
		src := "@oxy: group 0 0 storage_uniform myVar camera"
		out, err := pp.Process(src)
		suite.NoError(err)
		suite.Contains(out, "@group(0) @binding(0) var<uniform> myVar: CameraUniform;")
		decls := pp.Declarations()
		suite.Len(decls, 1)
		suite.Equal(shader.AnnotationTypeBindingGroup, decls[0].Type)
	})

	suite.Run("group annotation with array type", func() {
		pp := shader.NewPreProcessor()
		src := "@oxy: group 0 0 storage_read myVar array<camera>"
		out, err := pp.Process(src)
		suite.NoError(err)
		suite.Contains(out, "var<storage, read> myVar: array<CameraUniform>;")
	})

	suite.Run("inject annotation generates const", func() {
		pp := shader.NewPreProcessor()
		src := "@oxy: inject MY_CONST u32 max_bones"
		out, err := pp.Process(src, map[string]string{"max_bones": "64"})
		suite.NoError(err)
		suite.Contains(out, "const MY_CONST: u32 = 64;")
	})

	suite.Run("provider annotation adds to declarations only", func() {
		pp := shader.NewPreProcessor()
		src := "@oxy: provider 0 0 camera"
		out, err := pp.Process(src)
		suite.NoError(err)
		suite.False(strings.Contains(out, "@oxy:"))
		decls := pp.Declarations()
		suite.Len(decls, 1)
		suite.Equal(shader.AnnotationTypeProvider, decls[0].Type)
		suite.Equal(shader.AnnotationArgCamera, decls[0].Args[0])
	})

	suite.Run("provider with binding role", func() {
		pp := shader.NewPreProcessor()
		src := "@oxy: provider 0 0 material diffuse_texture"
		_, err := pp.Process(src)
		suite.NoError(err)
		decls := pp.Declarations()
		suite.Len(decls[0].Args, 2)
		suite.Equal(shader.AnnotationArgDiffuseTexture, decls[0].Args[1])
	})

	suite.Run("group and provider both in declarations", func() {
		pp := shader.NewPreProcessor()
		src := "@oxy: group 0 0 storage_uniform myVar camera\n@oxy: provider 0 1 camera"
		_, err := pp.Process(src)
		suite.NoError(err)
		suite.Len(pp.Declarations(), 2)
	})

	suite.Run("declarations reset on second process call", func() {
		pp := shader.NewPreProcessor()
		_, _ = pp.Process("@oxy: provider 0 0 camera\n@oxy: provider 0 1 material")
		_, err := pp.Process("@oxy: provider 0 0 camera")
		suite.NoError(err)
		suite.Len(pp.Declarations(), 1)
	})

	suite.Run("multiple injection maps later overrides earlier", func() {
		pp := shader.NewPreProcessor()
		src := "@oxy: inject MY_CONST u32 max_bones"
		out, err := pp.Process(src,
			map[string]string{"max_bones": "32"},
			map[string]string{"max_bones": "128"},
		)
		suite.NoError(err)
		suite.Contains(out, "const MY_CONST: u32 = 128;")
	})

	suite.Run("error on malformed include", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: include unknown_xyz")
		suite.Error(err)
	})

	suite.Run("error on inject missing key", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: inject MY_CONST u32 max_bones")
		suite.Error(err)
	})

	suite.Run("error on unknown annotation type", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: badtype arg")
		suite.Error(err)
	})
}

func (suite *preProcessorTest) TestProcessAnnotationErrors() {
	suite.Run("error on empty annotation", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy:")
		suite.Error(err)
	})

	suite.Run("error on include with wrong arg count", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: include")
		suite.Error(err)
	})

	suite.Run("error on group with wrong arg count", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: group 0 0 storage_uniform myVar")
		suite.Error(err)
	})

	suite.Run("error on group with invalid group number", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: group abc 0 storage_uniform myVar camera")
		suite.Error(err)
	})

	suite.Run("error on group with invalid binding number", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: group 0 abc storage_uniform myVar camera")
		suite.Error(err)
	})

	suite.Run("error on group with invalid address space", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: group 0 0 invalid_space myVar camera")
		suite.Error(err)
	})

	suite.Run("error on group with invalid array element type", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: group 0 0 storage_read myVar array<unknown_xyz>")
		suite.Error(err)
	})

	suite.Run("error on group with invalid struct type", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: group 0 0 storage_uniform myVar unknown_xyz")
		suite.Error(err)
	})

	suite.Run("error on inject with wrong arg count", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: inject MY_CONST")
		suite.Error(err)
	})

	suite.Run("error on inject with invalid wgsl type", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: inject MY_CONST bad_type max_bones")
		suite.Error(err)
	})

	suite.Run("error on inject with unknown injection key", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: inject MY_CONST u32 unknown_key")
		suite.Error(err)
	})

	suite.Run("error on provider with wrong arg count", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: provider 0")
		suite.Error(err)
	})

	suite.Run("error on provider with invalid group number", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: provider abc 0 camera")
		suite.Error(err)
	})

	suite.Run("error on provider with invalid binding number", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: provider 0 abc camera")
		suite.Error(err)
	})

	suite.Run("error on provider with unknown identity", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: provider 0 0 unknown_identity")
		suite.Error(err)
	})

	suite.Run("error on provider with invalid binding role", func() {
		pp := shader.NewPreProcessor()
		_, err := pp.Process("@oxy: provider 0 0 material unknown_role")
		suite.Error(err)
	})
}

// ── Suite 3: NewShader ─────────────────────────────────────────────────────────

const (
	vertexWGSL = `struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    return vec4<f32>(in.position, 1.0);
}

@group(0) @binding(0) var<uniform> camera: vec4<f32>;`

	fragmentWGSL = `@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> color: vec4<f32>;`

	computeWGSL = `@compute @workgroup_size(8, 8, 1)
fn cs_main() {}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;`

	// textureSamplerFragWGSL exercises classifySampledTexture, classifyDepthTexture,
	// classifyStorageTexture, splitTypeParams, and sampler/comparison-sampler classify paths.
	textureSamplerFragWGSL = `@fragment
fn fs_tex() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@group(0) @binding(0) var myTex: texture_2d<f32>;
@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myDepthTex: texture_depth_2d;
@group(0) @binding(3) var myStorageTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var myCompSampler: sampler_comparison;`

	// structLayoutFragWGSL exercises block comment stripping, nested struct layout
	// resolution (resolveTypeLayout knownTypes branch), and fixed-size array layout
	// (resolveTypeLayout array<T,N> branch).
	structLayoutFragWGSL = `/* This shader exercises block comment stripping
   and nested struct size computation. */
struct Inner {
    x: f32,
    y: f32,
}

/* Outer references Inner as a field type */
struct Outer {
    inner: Inner,
    count: u32,
}

@fragment
fn fs_struct() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> outer: Outer;
@group(0) @binding(1) var<uniform> arr: array<f32, 16>;`

	// builtinOutVertexWGSL has an explicit output struct with a @builtin field to
	// exercise the isVertexInputStruct false-return branch.
	builtinOutVertexWGSL = `struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_builtin(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(in.position, 1.0);
    return out;
}

@group(0) @binding(0) var<uniform> camera: vec4<f32>;`

	// noWorkgroupComputeWGSL has no @workgroup_size annotation to exercise the
	// parseWorkgroupSize nil-match (default size) path.
	noWorkgroupComputeWGSL = `@compute
fn cs_nosize() {}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;`
)

const (
	// badAnnotationWGSL has a line that the pre-processor finds and rejects, triggering a panic in NewShader.
	badAnnotationWGSL = `//@oxy: include unknown_xyz`

	// unknownVertexFormatWGSL has a vertex input field typed mat4x4<f32>, which is absent from
	// wgslVertexFormatMap, causing buildVertexBufferLayout to return false and the layout to be skipped.
	unknownVertexFormatWGSL = `struct VertexInput {
    @location(0) transform: mat4x4<f32>,
}

@vertex
fn vs_mat() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> camera: vec4<f32>;`

	// readOnlyStorageFragWGSL exercises the classifyResource storage-read branch.
	readOnlyStorageFragWGSL = `@fragment
fn fs_ros() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@group(0) @binding(0) var<storage, read> myBuf: array<f32>;`

	// lineCommentVertexWGSL has inline // comments after struct fields to exercise the
	// stripLineComments comment-found branch inside parseVertexLayouts.
	lineCommentVertexWGSL = `struct VertexInput { // this struct has comments
    @location(0) position: vec3<f32>, // position attribute
    @location(1) uv: vec2<f32>, // uv attribute
}

@vertex
fn vs_comments(in: VertexInput) -> @builtin(position) vec4<f32> {
    return vec4<f32>(in.position, 1.0);
}

@group(0) @binding(0) var<uniform> camera: vec4<f32>;`

	// unknownArrayElemFragWGSL has a storage binding typed array<UnknownType> to hit the
	// resolveTypeLayout unknown-array-element early-return in parseBindGroupLayouts.
	unknownArrayElemFragWGSL = `@fragment
fn fs_uae() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@group(0) @binding(0) var<storage, read> myBuf: array<UnknownType>;`

	// invalidArrayCountFragWGSL has a struct field with a fixed-size array whose count is
	// non-numeric, exercising the strconv.ParseUint error path in resolveTypeLayout.
	invalidArrayCountFragWGSL = `struct BadCount {
    data: array<f32, abc>,
}

@fragment
fn fs_iac() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> myBuf: BadCount;`

	// computeStructLayoutCoverageFragWGSL covers two branches in computeStructLayout:
	//   PrefixRuntimeArr: resolved prefix field (count: u32) followed by an unknown-element
	//   runtime array, hitting the offset!=0 return path.
	//   BareUnknownField: a field of plain unknown type that is not an array, hitting the
	//   non-array unresolvable-field return and the computeStructSizes !progress break.
	computeStructLayoutCoverageFragWGSL = `struct PrefixRuntimeArr {
    count: u32,
    data: array<UnknownTypeXyz>,
}

struct BareUnknownField {
    x: CompletelyUnknownTypeAbc,
}

@fragment
fn fs_csl() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> myBuf: PrefixRuntimeArr;`

	// noAngleBracketTextureFragWGSL declares a texture_2d binding without angle-bracket type
	// parameters, exercising the splitTypeParams no-"<" early-return inside classifySampledTexture.
	noAngleBracketTextureFragWGSL = `@fragment
fn fs_nab() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@group(0) @binding(0) var myPlainTex: texture_2d;`

	// splitFieldWGSL has a struct with an attribute-only comma token (@location(0) alone,
	// not followed by an identifier: type pair) to exercise the fieldRegex no-match
	// else-continue in parseStructFields.
	splitFieldWGSL = `struct Split {
    @location(0),
    x: f32,
}

@fragment
fn fs_split() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@group(0) @binding(0) var<uniform> color: vec4<f32>;`
)

type newShaderTest struct {
	suite.Suite
	vsPath                          string
	fsPath                          string
	csPath                          string
	texFragPath                     string
	structFragPath                  string
	builtinVSPath                   string
	noWorkgroupCSPath               string
	vs                              shader.Shader
	fs                              shader.Shader
	cs                              shader.Shader
	texFrag                         shader.Shader
	structFrag                      shader.Shader
	builtinVS                       shader.Shader
	noWorkgroupCS                   shader.Shader
	unknownVertexFormatPath         string
	readOnlyStoragePath             string
	lineCommentVertexPath           string
	unknownArrayElemFragPath        string
	invalidArrayCountFragPath       string
	computeStructLayoutCoveragePath string
	noAngleBracketTexturePath       string
	splitFragPath                   string
	unknownVertexFormatVS           shader.Shader
	readOnlyStorageFrag             shader.Shader
	lineCommentVS                   shader.Shader
	unknownArrayElemFrag            shader.Shader
	invalidArrayCountFrag           shader.Shader
	computeStructLayoutCoverageFrag shader.Shader
	noAngleBracketTextureFrag       shader.Shader
	splitFrag                       shader.Shader
}

func TestNewShader(t *testing.T) {
	suite.Run(t, new(newShaderTest))
}

func (suite *newShaderTest) SetupSuite() {
	tmp := suite.T().TempDir()
	suite.vsPath = filepath.Join(tmp, "vertex.wgsl")
	suite.fsPath = filepath.Join(tmp, "fragment.wgsl")
	suite.csPath = filepath.Join(tmp, "compute.wgsl")
	suite.texFragPath = filepath.Join(tmp, "tex_frag.wgsl")
	suite.structFragPath = filepath.Join(tmp, "struct_frag.wgsl")
	suite.builtinVSPath = filepath.Join(tmp, "builtin_vs.wgsl")
	suite.noWorkgroupCSPath = filepath.Join(tmp, "no_workgroup_cs.wgsl")

	suite.Require().NoError(os.WriteFile(suite.vsPath, []byte(vertexWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.fsPath, []byte(fragmentWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.csPath, []byte(computeWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.texFragPath, []byte(textureSamplerFragWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.structFragPath, []byte(structLayoutFragWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.builtinVSPath, []byte(builtinOutVertexWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.noWorkgroupCSPath, []byte(noWorkgroupComputeWGSL), 0o644))

	suite.vs = shader.NewShader("vs", shader.ShaderTypeVertex, suite.vsPath)
	suite.fs = shader.NewShader("fs", shader.ShaderTypeFragment, suite.fsPath)
	suite.cs = shader.NewShader("cs", shader.ShaderTypeCompute, suite.csPath)
	suite.texFrag = shader.NewShader("tex-frag", shader.ShaderTypeFragment, suite.texFragPath)
	suite.structFrag = shader.NewShader("struct-frag", shader.ShaderTypeFragment, suite.structFragPath)
	suite.builtinVS = shader.NewShader("builtin-vs", shader.ShaderTypeVertex, suite.builtinVSPath)
	suite.noWorkgroupCS = shader.NewShader("no-wg-cs", shader.ShaderTypeCompute, suite.noWorkgroupCSPath)

	suite.unknownVertexFormatPath = filepath.Join(tmp, "unknown_vertex_format.wgsl")
	suite.readOnlyStoragePath = filepath.Join(tmp, "read_only_storage.wgsl")
	suite.lineCommentVertexPath = filepath.Join(tmp, "line_comment_vertex.wgsl")
	suite.unknownArrayElemFragPath = filepath.Join(tmp, "unknown_array_elem_frag.wgsl")
	suite.invalidArrayCountFragPath = filepath.Join(tmp, "invalid_array_count_frag.wgsl")
	suite.computeStructLayoutCoveragePath = filepath.Join(tmp, "compute_struct_layout_coverage.wgsl")
	suite.noAngleBracketTexturePath = filepath.Join(tmp, "no_angle_bracket_texture.wgsl")

	suite.Require().NoError(os.WriteFile(suite.unknownVertexFormatPath, []byte(unknownVertexFormatWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.readOnlyStoragePath, []byte(readOnlyStorageFragWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.lineCommentVertexPath, []byte(lineCommentVertexWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.unknownArrayElemFragPath, []byte(unknownArrayElemFragWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.invalidArrayCountFragPath, []byte(invalidArrayCountFragWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.computeStructLayoutCoveragePath, []byte(computeStructLayoutCoverageFragWGSL), 0o644))
	suite.Require().NoError(os.WriteFile(suite.noAngleBracketTexturePath, []byte(noAngleBracketTextureFragWGSL), 0o644))

	suite.unknownVertexFormatVS = shader.NewShader("unknown-vf", shader.ShaderTypeVertex, suite.unknownVertexFormatPath)
	suite.readOnlyStorageFrag = shader.NewShader("ros-frag", shader.ShaderTypeFragment, suite.readOnlyStoragePath)
	suite.lineCommentVS = shader.NewShader("lc-vs", shader.ShaderTypeVertex, suite.lineCommentVertexPath)
	suite.unknownArrayElemFrag = shader.NewShader("uae-frag", shader.ShaderTypeFragment, suite.unknownArrayElemFragPath)
	suite.invalidArrayCountFrag = shader.NewShader("iac-frag", shader.ShaderTypeFragment, suite.invalidArrayCountFragPath)
	suite.computeStructLayoutCoverageFrag = shader.NewShader("csl-frag", shader.ShaderTypeFragment, suite.computeStructLayoutCoveragePath)
	suite.noAngleBracketTextureFrag = shader.NewShader("nab-frag", shader.ShaderTypeFragment, suite.noAngleBracketTexturePath)

	suite.splitFragPath = filepath.Join(tmp, "split_frag.wgsl")
	suite.Require().NoError(os.WriteFile(suite.splitFragPath, []byte(splitFieldWGSL), 0o644))
	suite.splitFrag = shader.NewShader("split-frag", shader.ShaderTypeFragment, suite.splitFragPath)
}

func (suite *newShaderTest) TestNewShader() {
	suite.Run("panics on empty source path", func() {
		suite.Panics(func() { shader.NewShader("k", shader.ShaderTypeVertex, "") })
	})
	suite.Run("panics on nonexistent file", func() {
		suite.Panics(func() { shader.NewShader("k", shader.ShaderTypeVertex, "/nonexistent_path_xyz.wgsl") })
	})
}

func (suite *newShaderTest) TestVertexShader() {
	vs := suite.vs
	suite.Run("Key returns the provided key", func() {
		suite.Equal("vs", vs.Key())
	})
	suite.Run("Source is non-empty", func() {
		suite.NotEmpty(vs.Source())
	})
	suite.Run("ShaderType is vertex", func() {
		suite.Equal(shader.ShaderTypeVertex, vs.ShaderType())
	})
	suite.Run("EntryPoint is vs_main", func() {
		suite.Equal("vs_main", vs.EntryPoint())
	})
	suite.Run("Module is non-nil", func() {
		suite.NotNil(vs.Module())
	})
	suite.Run("Module label matches key", func() {
		suite.Equal("vs", vs.Module().Label)
	})
	suite.Run("Module WGSL descriptor is non-nil", func() {
		suite.NotNil(vs.Module().WGSLSource)
	})
	suite.Run("VertexLayouts is non-empty", func() {
		suite.NotEmpty(vs.VertexLayouts())
	})
	suite.Run("VertexLayout(0) is non-nil", func() {
		suite.NotNil(vs.VertexLayout(0))
	})
	suite.Run("VertexLayout for absent key returns nil", func() {
		suite.Nil(vs.VertexLayout(99))
	})
	suite.Run("BindGroupLayoutDescriptors is non-nil", func() {
		suite.NotNil(vs.BindGroupLayoutDescriptors())
	})
	suite.Run("BindGroupLayoutDescriptor(0) has entries", func() {
		suite.NotEmpty(vs.BindGroupLayoutDescriptor(0).Entries)
	})
	suite.Run("BindGroupVarNames is non-nil", func() {
		suite.NotNil(vs.BindGroupVarNames())
	})
	suite.Run("BindGroupVarName found", func() {
		suite.Equal("camera", vs.BindGroupVarName(0, 0))
	})
	suite.Run("BindGroupVarName nil group returns empty", func() {
		suite.Equal("", vs.BindGroupVarName(99, 0))
	})
	suite.Run("BindGroupFromVarName found", func() {
		binding, ok := vs.BindGroupFromVarName(0, "camera")
		suite.True(ok)
		suite.Equal(0, binding)
	})
	suite.Run("BindGroupFromVarName not found in existing group", func() {
		_, ok := vs.BindGroupFromVarName(0, "nonexistent")
		suite.False(ok)
	})
	suite.Run("BindGroupFromVarName nil group returns -1 false", func() {
		binding, ok := vs.BindGroupFromVarName(99, "camera")
		suite.Equal(-1, binding)
		suite.False(ok)
	})
	suite.Run("Declarations is accessible without panic", func() {
		_ = vs.Declarations()
	})
}

func (suite *newShaderTest) TestFragmentShader() {
	fs := suite.fs
	suite.Run("ShaderType is fragment", func() {
		suite.Equal(shader.ShaderTypeFragment, fs.ShaderType())
	})
	suite.Run("EntryPoint is fs_main", func() {
		suite.Equal("fs_main", fs.EntryPoint())
	})
}

func (suite *newShaderTest) TestComputeShader() {
	cs := suite.cs
	suite.Run("ShaderType is compute", func() {
		suite.Equal(shader.ShaderTypeCompute, cs.ShaderType())
	})
	suite.Run("EntryPoint is cs_main", func() {
		suite.Equal("cs_main", cs.EntryPoint())
	})
	suite.Run("WorkgroupSize is 8x8x1", func() {
		suite.Equal([3]uint32{8, 8, 1}, cs.WorkgroupSize())
	})
}

func (suite *newShaderTest) TestWithInjections() {
	suite.Run("WithInjections option applies without panic", func() {
		vsPath := suite.vsPath
		var s shader.Shader
		suite.NotPanics(func() {
			s = shader.NewShader("vs-inj", shader.ShaderTypeVertex, vsPath,
				shader.WithInjections(map[string]string{"key": "val"}),
			)
		})
		suite.NotEmpty(s.Source())
	})
}

func (suite *newShaderTest) TestTextureSamplerShader() {
	tf := suite.texFrag

	suite.Run("ShaderType is fragment", func() {
		suite.Equal(shader.ShaderTypeFragment, tf.ShaderType())
	})
	suite.Run("BindGroupLayoutDescriptor(0) has five entries", func() {
		suite.Len(tf.BindGroupLayoutDescriptor(0).Entries, 5)
	})
	suite.Run("sampled texture binding is present", func() {
		suite.Equal("myTex", tf.BindGroupVarName(0, 0))
	})
	suite.Run("sampler binding is present", func() {
		suite.Equal("mySampler", tf.BindGroupVarName(0, 1))
	})
	suite.Run("depth texture binding is present", func() {
		suite.Equal("myDepthTex", tf.BindGroupVarName(0, 2))
	})
	suite.Run("storage texture binding is present", func() {
		suite.Equal("myStorageTex", tf.BindGroupVarName(0, 3))
	})
	suite.Run("comparison sampler binding is present", func() {
		suite.Equal("myCompSampler", tf.BindGroupVarName(0, 4))
	})
}

func (suite *newShaderTest) TestStructLayoutShader() {
	sf := suite.structFrag

	suite.Run("ShaderType is fragment", func() {
		suite.Equal(shader.ShaderTypeFragment, sf.ShaderType())
	})
	suite.Run("BindGroupLayoutDescriptor(0) has two entries", func() {
		suite.Len(sf.BindGroupLayoutDescriptor(0).Entries, 2)
	})
	suite.Run("Outer uniform binding is present", func() {
		suite.Equal("outer", sf.BindGroupVarName(0, 0))
	})
	suite.Run("fixed-size array binding is present", func() {
		suite.Equal("arr", sf.BindGroupVarName(0, 1))
	})
	suite.Run("EntryPoint is fs_struct", func() {
		suite.Equal("fs_struct", sf.EntryPoint())
	})
}

func (suite *newShaderTest) TestBuiltinOutputVertexShader() {
	bvs := suite.builtinVS

	suite.Run("ShaderType is vertex", func() {
		suite.Equal(shader.ShaderTypeVertex, bvs.ShaderType())
	})
	suite.Run("EntryPoint is vs_builtin", func() {
		suite.Equal("vs_builtin", bvs.EntryPoint())
	})
	suite.Run("VertexLayouts contains only VertexInput not VertexOutput", func() {
		suite.Len(bvs.VertexLayouts(), 1)
	})
}

func (suite *newShaderTest) TestNoWorkgroupSizeCompute() {
	nwcs := suite.noWorkgroupCS

	suite.Run("ShaderType is compute", func() {
		suite.Equal(shader.ShaderTypeCompute, nwcs.ShaderType())
	})
	suite.Run("WorkgroupSize defaults to 1x1x1 when not specified", func() {
		suite.Equal([3]uint32{1, 1, 1}, nwcs.WorkgroupSize())
	})
	suite.Run("EntryPoint is cs_nosize", func() {
		suite.Equal("cs_nosize", nwcs.EntryPoint())
	})
}

func (suite *newShaderTest) TestCustomShaderType() {
	suite.Run("unknown ShaderType yields empty entry point and accessible source", func() {
		tmp := suite.T().TempDir()
		path := filepath.Join(tmp, "custom.wgsl")
		src := "@group(0) @binding(0) var<uniform> foo: vec4<f32>;"
		suite.Require().NoError(os.WriteFile(path, []byte(src), 0o644))
		s := shader.NewShader("custom", shader.ShaderType(99), path)
		suite.Equal("", s.EntryPoint())
		suite.NotEmpty(s.Source())
	})
	suite.Run("entry point returns empty when source has no matching annotation", func() {
		tmp := suite.T().TempDir()
		path := filepath.Join(tmp, "mismatch.wgsl")
		suite.Require().NoError(os.WriteFile(path, []byte(fragmentWGSL), 0o644))
		s := shader.NewShader("mismatch", shader.ShaderTypeVertex, path)
		suite.Equal("", s.EntryPoint())
	})
}

func (suite *newShaderTest) TestParseSourceFromPathPanic() {
	suite.Run("NewShader panics when pre-processor encounters bad annotation", func() {
		tmp := suite.T().TempDir()
		path := filepath.Join(tmp, "bad_annotation.wgsl")
		suite.Require().NoError(os.WriteFile(path, []byte(badAnnotationWGSL), 0o644))
		suite.Panics(func() {
			shader.NewShader("bad-ann", shader.ShaderTypeFragment, path)
		})
	})
}

func (suite *newShaderTest) TestUnknownVertexFormat() {
	s := suite.unknownVertexFormatVS
	suite.Run("VertexLayouts is empty when vertex field type is not in format map", func() {
		suite.Empty(s.VertexLayouts())
	})
}

func (suite *newShaderTest) TestReadOnlyStorageBinding() {
	s := suite.readOnlyStorageFrag
	suite.Run("storage read binding is present by variable name", func() {
		suite.Equal("myBuf", s.BindGroupVarName(0, 0))
	})
	suite.Run("bind group layout descriptor has entries for group 0", func() {
		suite.NotEmpty(s.BindGroupLayoutDescriptor(0).Entries)
	})
}

func (suite *newShaderTest) TestLineCommentStripping() {
	s := suite.lineCommentVS
	suite.Run("vertex layouts are non-empty after stripping inline comments", func() {
		suite.NotEmpty(s.VertexLayouts())
	})
	suite.Run("entry point is vs_comments", func() {
		suite.Equal("vs_comments", s.EntryPoint())
	})
}

func (suite *newShaderTest) TestUnknownArrayElementBinding() {
	s := suite.unknownArrayElemFrag
	suite.Run("binding with array of unknown element type is still tracked by variable name", func() {
		suite.Equal("myBuf", s.BindGroupVarName(0, 0))
	})
}

func (suite *newShaderTest) TestInvalidArrayCountBinding() {
	s := suite.invalidArrayCountFrag
	suite.Run("binding with non-numeric array count is still tracked by variable name", func() {
		suite.Equal("myBuf", s.BindGroupVarName(0, 0))
	})
}

func (suite *newShaderTest) TestComputeStructLayoutCoverage() {
	s := suite.computeStructLayoutCoverageFrag
	suite.Run("shader with mixed resolvable and unresolvable struct fields creates successfully", func() {
		suite.NotNil(s)
	})
	suite.Run("binding backed by prefix-plus-runtime-array struct is tracked by variable name", func() {
		suite.Equal("myBuf", s.BindGroupVarName(0, 0))
	})
}

func (suite *newShaderTest) TestNoAngleBracketTexture() {
	s := suite.noAngleBracketTextureFrag
	suite.Run("texture binding without angle-bracket type parameter is tracked by variable name", func() {
		suite.Equal("myPlainTex", s.BindGroupVarName(0, 0))
	})
}

func (suite *newShaderTest) TestSplitFieldShader() {
	suite.Run("struct with attribute-only token parsed without panic", func() {
		suite.NotNil(suite.splitFrag)
	})
	suite.Run("entry point is fs_split", func() {
		suite.Equal("fs_split", suite.splitFrag.EntryPoint())
	})
	suite.Run("ShaderType is fragment", func() {
		suite.Equal(shader.ShaderTypeFragment, suite.splitFrag.ShaderType())
	})
}
