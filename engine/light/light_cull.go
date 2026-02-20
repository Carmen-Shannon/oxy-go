package light

// TileSize is the width and height in pixels of each screen-space tile used
// for Forward+ light culling. The screen is divided into a grid of tiles, each
// TileSize × TileSize pixels, and lights are assigned to tiles via a compute
// shader so the fragment shader only evaluates lights relevant to each tile.
const TileSize = 16

// MaxLightsPerTile is the maximum number of light indices stored per tile in
// the Forward+ tile light index buffer. If more lights overlap a tile, excess
// lights are silently dropped.
const MaxLightsPerTile = 256

// TileCounts computes the number of tiles in each dimension for a given screen
// resolution and the configured TileSize.
//
// Parameters:
//   - screenWidth: screen width in pixels
//   - screenHeight: screen height in pixels
//
// Returns:
//   - tileCountX: number of tile columns
//   - tileCountY: number of tile rows
func TileCounts(screenWidth, screenHeight int) (tileCountX, tileCountY uint32) {
	tileCountX = (uint32(screenWidth) + TileSize - 1) / TileSize
	tileCountY = (uint32(screenHeight) + TileSize - 1) / TileSize
	return
}
