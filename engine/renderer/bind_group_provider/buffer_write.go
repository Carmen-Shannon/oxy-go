package bind_group_provider

// BufferWrite describes a single GPU buffer write operation targeting a specific binding
// on a BindGroupProvider at a given byte offset.
type BufferWrite struct {
	Provider BindGroupProvider
	Binding  int
	Offset   uint64
	Data     []byte
}
