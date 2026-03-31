package physics

import (
	"encoding/binary"
	"math"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/Carmen-Shannon/automation/tools/worker"
	"github.com/Carmen-Shannon/oxy-go/engine/model"
)

// Particle represents a single spherical collision element within a rigid body.
// Each particle stores a body-local position offset, an index identifying
// which body it belongs to (assigned during scene registration), and an
// optional bone index for kinematic skeletal bodies.
//
// SurfaceNormal holds the outward-facing surface normal for particles created
// by VoxelizeMesh with surfaceOnly=true. It points from the solid interior
// toward empty space (i.e. toward the box interior for container inner faces).
// The collision shader uses this to prevent reversed spring forces when a
// particle penetrates past the wall surface.
type Particle struct {
	LocalPosition [3]float32
	SurfaceNormal [3]float32
	BodyIndex     uint32
	BoneIndex     uint32
}

// VoxelizeMesh converts a Model's mesh geometry into a set of spherical particles via
// ray-cast voxelization, following the method described in GPU Gems 3, Ch. 29 §1.3
// (https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus).
//
// The function extracts vertex positions and triangle indices directly from the Model's
// retained byte data. Vertex stride is determined by Model.Skinned() (96 bytes for
// GPUSkinnedVertex, 64 bytes for GPUVertex); position is always at byte offset 0.
//
// The algorithm subdivides the mesh's AABB into a uniform 3D grid with cell size equal
// to the particle diameter (2 × particleRadius). For each (y, z) column of the grid a
// ray is cast along the +X axis and all triangle intersections are collected. The parity
// of crossings determines inside/outside status for each voxel in the column.
//
// When surfaceOnly is true an additional filter pass removes interior particles, keeping
// only those adjacent to at least one empty (outside) neighbor. This dramatically reduces
// particle count for large solid bodies while preserving collision surface fidelity.
//
// The column ray-casting phase is parallelized across available CPU cores using goroutines,
// as each (y, z) column writes to a disjoint region of the voxel grid.
//
// Parameters:
//   - src: the Model whose mesh data will be voxelized
//   - particleRadius: the uniform radius of each generated particle
//   - surfaceOnly: when true, only surface-adjacent particles are retained
//
// Returns:
//   - []Particle: the generated particle set in body-local space, or nil if the Model
//     has insufficient vertex/index data
func VoxelizeMesh(src model.Model, particleRadius float32, surfaceOnly bool) []Particle {
	stride := 64
	if src.Skinned() {
		stride = 96
	}

	// Decode vertex positions from raw byte data.
	// Position is always at byte offset 0 within each vertex (both GPUVertex and GPUSkinnedVertex).
	vertData := src.VertexData()
	if len(vertData) < stride {
		return nil
	}
	numVerts := len(vertData) / stride
	positions := make([][3]float32, numVerts)
	for i := range numVerts {
		off := i * stride
		positions[i] = [3]float32{
			math.Float32frombits(binary.LittleEndian.Uint32(vertData[off : off+4])),
			math.Float32frombits(binary.LittleEndian.Uint32(vertData[off+4 : off+8])),
			math.Float32frombits(binary.LittleEndian.Uint32(vertData[off+8 : off+12])),
		}
	}

	// Decode triangle indices from raw byte data (little-endian uint32s).
	idxData := src.IndexData()
	numIdx := len(idxData) / 4
	if numIdx < 3 {
		return nil
	}
	indices := make([]uint32, numIdx)
	for i := range numIdx {
		indices[i] = binary.LittleEndian.Uint32(idxData[i*4 : i*4+4])
	}

	// Compute axis-aligned bounding box from vertex positions.
	aabbMin := [3]float32{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32}
	aabbMax := [3]float32{-math.MaxFloat32, -math.MaxFloat32, -math.MaxFloat32}
	for _, p := range positions {
		for axis := range 3 {
			if p[axis] < aabbMin[axis] {
				aabbMin[axis] = p[axis]
			}
			if p[axis] > aabbMax[axis] {
				aabbMax[axis] = p[axis]
			}
		}
	}

	diameter := particleRadius * 2.0

	nx := int(math.Ceil(float64(aabbMax[0]-aabbMin[0]) / float64(diameter)))
	ny := int(math.Ceil(float64(aabbMax[1]-aabbMin[1]) / float64(diameter)))
	nz := int(math.Ceil(float64(aabbMax[2]-aabbMin[2]) / float64(diameter)))
	if nx <= 0 || ny <= 0 || nz <= 0 {
		return nil
	}

	// Build a 3D boolean grid marking which voxels are "inside" the mesh.
	// We need the grid for surfaceOnly neighbor checks. Even when surfaceOnly
	// is false, building the grid first and collecting particles second keeps
	// the code path unified and allocation predictable.
	grid := make([]bool, nx*ny*nz)
	numTriangles := len(indices) / 3

	// Parallelize the column ray-casting phase using a DynamicWorkerPool.
	// Each (iy, iz) column writes to a disjoint region of the grid, so no
	// synchronization is required on the grid itself. A WaitGroup provides
	// barrier sync (same pattern as Scene's PrepareCompute), while the pool
	// manages worker lifecycle.
	totalColumns := ny * nz
	numWorkers := max(min(runtime.NumCPU(), totalColumns), 1)
	columnsPerWorker := (totalColumns + numWorkers - 1) / numWorkers

	pool := worker.NewDynamicWorkerPool(numWorkers, totalColumns, 1*time.Second)
	var wg sync.WaitGroup
	for w := range numWorkers {
		colStart := w * columnsPerWorker
		colEnd := min(colStart+columnsPerWorker, totalColumns)

		wg.Add(1)
		taskID := w
		pool.SubmitTask(worker.Task{
			ID: taskID,
			Do: func() (any, error) {
				defer wg.Done()

				const epsilon = 1e-8
				var intersections []float32

				for col := colStart; col < colEnd; col++ {
					iy := col / nz
					iz := col % nz

					oy := aabbMin[1] + particleRadius + float32(iy)*diameter
					oz := aabbMin[2] + particleRadius + float32(iz)*diameter
					ox := aabbMin[0] - diameter

					intersections = intersections[:0]

					for ti := range numTriangles {
						i0 := indices[ti*3]
						i1 := indices[ti*3+1]
						i2 := indices[ti*3+2]
						if int(i0) >= len(positions) || int(i1) >= len(positions) || int(i2) >= len(positions) {
							continue
						}

						v0, v1, v2 := positions[i0], positions[i1], positions[i2]

						// Möller–Trumbore ray-triangle intersection specialized for +X axis
						// (direction = [1, 0, 0]). This eliminates most cross-product terms.
						e1x := v1[0] - v0[0]
						e1y := v1[1] - v0[1]
						e1z := v1[2] - v0[2]
						e2x := v2[0] - v0[0]
						e2y := v2[1] - v0[1]
						e2z := v2[2] - v0[2]

						// h = cross(rayDir, e2) where rayDir = (1,0,0) → (0, -e2z, e2y)
						hy := -e2z
						hz := e2y

						// a = dot(e1, h) = e1y*hy + e1z*hz (hx is zero)
						a := e1y*hy + e1z*hz
						if a > -epsilon && a < epsilon {
							continue
						}
						f := 1.0 / a

						// s = rayOrigin - v0
						sx := ox - v0[0]
						sy := oy - v0[1]
						sz := oz - v0[2]

						// u = f * dot(s, h) = f * (sy*hy + sz*hz)
						u := f * (sy*hy + sz*hz)
						if u < 0.0 || u > 1.0 {
							continue
						}

						// q = cross(s, e1)
						qx := sy*e1z - sz*e1y
						qy := sz*e1x - sx*e1z
						qz := sx*e1y - sy*e1x

						// v = f * dot(rayDir, q) = f * qx (since rayDir = (1,0,0))
						v := f * qx
						if v < 0.0 || u+v > 1.0 {
							continue
						}

						// t = f * dot(e2, q)
						t := f * (e2x*qx + e2y*qy + e2z*qz)
						intersections = append(intersections, ox+t)
					}

					slices.Sort(intersections)

					// Walk voxels in this column, flipping inside/outside at each crossing.
					inside := false
					intIdx := 0
					for ix := range nx {
						x := aabbMin[0] + particleRadius + float32(ix)*diameter
						for intIdx < len(intersections) && intersections[intIdx] < x {
							inside = !inside
							intIdx++
						}
						if inside {
							grid[ix+iy*nx+iz*nx*ny] = true
						}
					}
				}
				return nil, nil
			},
		})
	}
	wg.Wait()
	pool.Stop()

	// Collect particles from the grid.
	var particles []Particle
	for iz := range nz {
		for iy := range ny {
			for ix := range nx {
				if !grid[ix+iy*nx+iz*nx*ny] {
					continue
				}

				// Surface-only filter: check if any of the 6 axis-aligned neighbors
				// is either outside the grid bounds or marked as empty (outside).
				// Simultaneously compute the surface normal as the average direction
				// toward all empty neighbors. This normal points from solid interior
				// toward empty space — for a container's inner face it points into
				// the box interior, which the collision shader uses to prevent
				// reversed spring forces from multi-layer walls.
				var snX, snY, snZ float32
				if surfaceOnly {
					isSurface := false
					for _, off := range [6][3]int{
						{-1, 0, 0}, {1, 0, 0},
						{0, -1, 0}, {0, 1, 0},
						{0, 0, -1}, {0, 0, 1},
					} {
						adjX, adjY, adjZ := ix+off[0], iy+off[1], iz+off[2]
						isEmpty := adjX < 0 || adjX >= nx || adjY < 0 || adjY >= ny || adjZ < 0 || adjZ >= nz ||
							!grid[adjX+adjY*nx+adjZ*nx*ny]
						if isEmpty {
							isSurface = true
							snX += float32(off[0])
							snY += float32(off[1])
							snZ += float32(off[2])
						}
					}
					if !isSurface {
						continue
					}
					// Normalize the surface normal
					snLen := float32(math.Sqrt(float64(snX*snX + snY*snY + snZ*snZ)))
					if snLen > 0 {
						snX /= snLen
						snY /= snLen
						snZ /= snLen
					}
				}

				particles = append(particles, Particle{
					LocalPosition: [3]float32{
						aabbMin[0] + particleRadius + float32(ix)*diameter,
						aabbMin[1] + particleRadius + float32(iy)*diameter,
						aabbMin[2] + particleRadius + float32(iz)*diameter,
					},
					SurfaceNormal: [3]float32{snX, snY, snZ},
				})
			}
		}
	}

	return particles
}

// AssignBoneIndices associates each voxel particle with its nearest bone from the
// source skinned model, transforming the particle's model-space position into
// bone-local space using the bone's inverse bind matrix. After this call every
// Particle in the slice has its BoneIndex set and its LocalPosition converted to
// bone-local coordinates, ready for GPU upload and per-frame bone-driven updates.
//
// The algorithm is O(particles × vertices): for each particle, the nearest vertex
// is found by squared-distance, and the vertex's primary bone (highest weight among
// its four blend weights) is chosen. The particle's model-space position is then
// multiplied by the bone's inverse bind matrix so the GPU bone_particle_update
// shader can transform it back to world space using only the bone world matrix.
//
// Parameters:
//   - particles: the voxelized particle set (model-space positions)
//   - src: the skinned Model whose vertex and skeleton data provide bone assignments
func AssignBoneIndices(particles []Particle, src model.Model) {
	if !src.Skinned() || src.Skeleton() == nil {
		return
	}

	vertData := src.VertexData()
	const stride = 96 // GPUSkinnedVertex size
	numVerts := len(vertData) / stride
	if numVerts == 0 {
		return
	}

	// Pre-extract vertex positions, primary bone indices, and bone weights from
	// the packed vertex buffer so the inner loop avoids repeated byte decoding.
	type vertInfo struct {
		pos     [3]float32
		boneIdx uint32
	}
	verts := make([]vertInfo, numVerts)
	for v := range numVerts {
		off := v * stride
		verts[v].pos = [3]float32{
			math.Float32frombits(binary.LittleEndian.Uint32(vertData[off : off+4])),
			math.Float32frombits(binary.LittleEndian.Uint32(vertData[off+4 : off+8])),
			math.Float32frombits(binary.LittleEndian.Uint32(vertData[off+8 : off+12])),
		}

		// BoneIndices at offset 64 (4 × u32), BoneWeights at offset 80 (4 × f32).
		// Pick the bone with the highest weight.
		bestBone := uint32(0)
		bestWeight := float32(0)
		for k := range 4 {
			w := math.Float32frombits(binary.LittleEndian.Uint32(vertData[off+80+k*4 : off+84+k*4]))
			if w > bestWeight {
				bestWeight = w
				bestBone = binary.LittleEndian.Uint32(vertData[off+64+k*4 : off+68+k*4])
			}
		}
		verts[v].boneIdx = bestBone
	}

	skel := src.Skeleton()
	bones := skel.Bones

	// For each particle, find the closest vertex and inherit its primary bone.
	for i := range particles {
		px, py, pz := particles[i].LocalPosition[0], particles[i].LocalPosition[1], particles[i].LocalPosition[2]
		minDist := float32(math.MaxFloat32)
		bestBone := uint32(0)
		for _, vi := range verts {
			dx := px - vi.pos[0]
			dy := py - vi.pos[1]
			dz := pz - vi.pos[2]
			d2 := dx*dx + dy*dy + dz*dz
			if d2 < minDist {
				minDist = d2
				bestBone = vi.boneIdx
			}
		}
		particles[i].BoneIndex = bestBone

		// Transform model-space position to bone-local space via the inverse bind matrix.
		// bone_local = inverseBind * model_pos  (homogeneous, w=1)
		if int(bestBone) < len(bones) {
			m := bones[bestBone].InverseBindMatrix
			mx, my, mz := particles[i].LocalPosition[0], particles[i].LocalPosition[1], particles[i].LocalPosition[2]
			particles[i].LocalPosition = [3]float32{
				m[0]*mx + m[4]*my + m[8]*mz + m[12],
				m[1]*mx + m[5]*my + m[9]*mz + m[13],
				m[2]*mx + m[6]*my + m[10]*mz + m[14],
			}
		}
	}
}
