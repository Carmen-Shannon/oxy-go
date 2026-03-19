package model

import (
	"github.com/Carmen-Shannon/oxy-go/common"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/bind_group_provider"
	"github.com/Carmen-Shannon/oxy-go/engine/renderer/material"
)

// model is the implementation of the Model interface.
type model struct {
	common.DelegateImpl[Model]

	name                  string
	skinned               bool
	skeleton              *Skeleton
	animations            []*AnimationClip
	importedMaterials     []common.ImportedMaterial
	renderMaterials       []material.Material
	meshProvider          bind_group_provider.BindGroupProvider
	effectProvider        bind_group_provider.BindGroupProvider
	computePipelineKey    string
	boundingRadius        float32
	vertexData, indexData []byte
	indexCount            int
	shadowCullMode        ShadowCullMode
	castsShadows          bool
}
