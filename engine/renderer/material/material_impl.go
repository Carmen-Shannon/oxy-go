package material

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
)

// material is the implementation of the Material interface.
type material struct {
	common.DelegateImpl[Material]

	name                     string
	baseColor                [4]float32
	metallic                 float32
	roughness                float32
	alphaCutoff              float32
	diffuseTexture           *common.ImportedTexture
	normalTexture            *common.ImportedTexture
	metallicRoughnessTexture *common.ImportedTexture
	pipelineKey              string
	bindGroupProvider        bind_group_provider.BindGroupProvider
	pipelineOpts             []any
	providers                map[int]bind_group_provider.BindGroupProvider
}
