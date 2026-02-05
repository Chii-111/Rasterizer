#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <atomic>
#include <chrono>
#include <immintrin.h>
#include <thread>
#include <vector>

#include "RNG.h"
#include "ThreadPool.h"
#include "colour.h"
#include "light.h"
#include "matrix.h"
#include "mesh.h"
#include "renderer.h"
#include "triangle.h"
#include "zbuffer.h"
#include <cmath>

// Divide screen into 64x64 tiles
// Each thread processes assigned tiles
struct Tile {
  int x, y, w, h; // Tile position and size in pixels
};

// Generate tiles for the screen (64x64 pixel tiles)
inline std::vector<Tile> generateTiles(int screenWidth, int screenHeight,
                                       int tileSize = 64) {
  std::vector<Tile> tiles;
  for (int ty = 0; ty < screenHeight; ty += tileSize) {
    for (int tx = 0; tx < screenWidth; tx += tileSize) {
      Tile t;
      t.x = tx;
      t.y = ty;
      t.w = std::min(tileSize, screenWidth - tx);
      t.h = std::min(tileSize, screenHeight - ty);
      tiles.push_back(t);
    }
  }
  return tiles;
}

// Store results back to temporary aligned arrays
struct TransformedVertices8 {
  alignas(32) float px[8], py[8], pz[8], pw[8]; // Screen-space positions
  alignas(32) float nx[8], ny[8], nz[8];        // World-space normals
  alignas(32) float cr[8], cg[8], cb[8];        // Colors
};

// Transform 8 vertices using AVX
// m: 4x4 matrix stored as m[row][col], row-major in memory as a[16]
inline void transformVertices8_AVX(
    const float *__restrict srcPx, const float *__restrict srcPy,
    const float *__restrict srcPz, const float *__restrict srcPw,
    const float *__restrict srcNx, const float *__restrict srcNy,
    const float *__restrict srcNz, const float *__restrict srcCr,
    const float *__restrict srcCg, const float *__restrict srcCb,
    const matrix &mvp, const matrix &world, float halfWidth, float halfHeight,
    float screenHeight, TransformedVertices8 &out) {
  // Load 8 vertices' components
  __m256 vx = _mm256_loadu_ps(srcPx);
  __m256 vy = _mm256_loadu_ps(srcPy);
  __m256 vz = _mm256_loadu_ps(srcPz);
  __m256 vw = _mm256_loadu_ps(srcPw);

  // MVP matrix rows (broadcast each element)
  // Row 0: result.x = m00*x + m01*y + m02*z + m03*w
  __m256 m00 = _mm256_set1_ps(mvp(0, 0));
  __m256 m01 = _mm256_set1_ps(mvp(0, 1));
  __m256 m02 = _mm256_set1_ps(mvp(0, 2));
  __m256 m03 = _mm256_set1_ps(mvp(0, 3));
  __m256 rx = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(m00, vx), _mm256_mul_ps(m01, vy)),
      _mm256_add_ps(_mm256_mul_ps(m02, vz), _mm256_mul_ps(m03, vw)));

  // Row 1: result.y
  __m256 m10 = _mm256_set1_ps(mvp(1, 0));
  __m256 m11 = _mm256_set1_ps(mvp(1, 1));
  __m256 m12 = _mm256_set1_ps(mvp(1, 2));
  __m256 m13 = _mm256_set1_ps(mvp(1, 3));
  __m256 ry = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(m10, vx), _mm256_mul_ps(m11, vy)),
      _mm256_add_ps(_mm256_mul_ps(m12, vz), _mm256_mul_ps(m13, vw)));

  // Row 2: result.z
  __m256 m20 = _mm256_set1_ps(mvp(2, 0));
  __m256 m21 = _mm256_set1_ps(mvp(2, 1));
  __m256 m22 = _mm256_set1_ps(mvp(2, 2));
  __m256 m23 = _mm256_set1_ps(mvp(2, 3));
  __m256 rz = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(m20, vx), _mm256_mul_ps(m21, vy)),
      _mm256_add_ps(_mm256_mul_ps(m22, vz), _mm256_mul_ps(m23, vw)));

  // Row 3: result.w
  __m256 m30 = _mm256_set1_ps(mvp(3, 0));
  __m256 m31 = _mm256_set1_ps(mvp(3, 1));
  __m256 m32 = _mm256_set1_ps(mvp(3, 2));
  __m256 m33 = _mm256_set1_ps(mvp(3, 3));
  __m256 rw = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(m30, vx), _mm256_mul_ps(m31, vy)),
      _mm256_add_ps(_mm256_mul_ps(m32, vz), _mm256_mul_ps(m33, vw)));

  // Perspective divide: x/=w, y/=w, z/=w
  __m256 inv_w = _mm256_div_ps(_mm256_set1_ps(1.0f), rw);
  rx = _mm256_mul_ps(rx, inv_w);
  ry = _mm256_mul_ps(ry, inv_w);
  rz = _mm256_mul_ps(rz, inv_w);

  // Map to screen space: x = (x+1)*halfWidth, y = screenHeight -
  // (y+1)*halfHeight
  __m256 one = _mm256_set1_ps(1.0f);
  __m256 hw = _mm256_set1_ps(halfWidth);
  __m256 hh = _mm256_set1_ps(halfHeight);
  __m256 sh = _mm256_set1_ps(screenHeight);

  rx = _mm256_mul_ps(_mm256_add_ps(rx, one), hw);
  ry = _mm256_sub_ps(sh, _mm256_mul_ps(_mm256_add_ps(ry, one), hh));

  _mm256_store_ps(out.px, rx);
  _mm256_store_ps(out.py, ry);
  _mm256_store_ps(out.pz, rz);
  _mm256_store_ps(out.pw, _mm256_set1_ps(1.0f));

  // Transform normals (world matrix only, no perspective)
  __m256 nx = _mm256_loadu_ps(srcNx);
  __m256 ny = _mm256_loadu_ps(srcNy);
  __m256 nz = _mm256_loadu_ps(srcNz);

  __m256 w00 = _mm256_set1_ps(world(0, 0));
  __m256 w01 = _mm256_set1_ps(world(0, 1));
  __m256 w02 = _mm256_set1_ps(world(0, 2));
  __m256 tnx = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(w00, nx), _mm256_mul_ps(w01, ny)),
      _mm256_mul_ps(w02, nz));

  __m256 w10 = _mm256_set1_ps(world(1, 0));
  __m256 w11 = _mm256_set1_ps(world(1, 1));
  __m256 w12 = _mm256_set1_ps(world(1, 2));
  __m256 tny = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(w10, nx), _mm256_mul_ps(w11, ny)),
      _mm256_mul_ps(w12, nz));

  __m256 w20 = _mm256_set1_ps(world(2, 0));
  __m256 w21 = _mm256_set1_ps(world(2, 1));
  __m256 w22 = _mm256_set1_ps(world(2, 2));
  __m256 tnz = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(w20, nx), _mm256_mul_ps(w21, ny)),
      _mm256_mul_ps(w22, nz));

  // Normalize normals: length = sqrt(x^2 + y^2 + z^2), then divide
  __m256 len_sq = _mm256_add_ps(
      _mm256_add_ps(_mm256_mul_ps(tnx, tnx), _mm256_mul_ps(tny, tny)),
      _mm256_mul_ps(tnz, tnz));
  __m256 inv_len = _mm256_rsqrt_ps(len_sq); // Fast approximate reciprocal sqrt
  tnx = _mm256_mul_ps(tnx, inv_len);
  tny = _mm256_mul_ps(tny, inv_len);
  tnz = _mm256_mul_ps(tnz, inv_len);

  _mm256_store_ps(out.nx, tnx);
  _mm256_store_ps(out.ny, tny);
  _mm256_store_ps(out.nz, tnz);

  // Copy colors (just load and store, no transformation)
  _mm256_storeu_ps(out.cr, _mm256_loadu_ps(srcCr));
  _mm256_storeu_ps(out.cg, _mm256_loadu_ps(srcCg));
  _mm256_storeu_ps(out.cb, _mm256_loadu_ps(srcCb));
}

// Main rendering function that processes a mesh, transforms its vertices,
// applies lighting, and draws triangles on the canvas. Input Variables:
// - renderer: The Renderer object used for drawing.
// - mesh: Pointer to the Mesh object containing vertices and triangles to
// render.
// - camera: Matrix representing the camera's transformation.
// - L: Light object representing the lighting parameters.
void render(Renderer &renderer, Mesh *mesh, matrix &camera, Light &L) {
  // Combine perspective, camera, and world transformations for the mesh
  matrix p = renderer.perspective * camera * mesh->world;

  // Iterate through all triangles in the mesh
  for (triIndices &ind : mesh->triangles) {
    Vertex t[3]; // Temporary array to store transformed triangle vertices

    // Transform each vertex of the triangle
    for (unsigned int i = 0; i < 3; i++) {
      t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
      t[i].p.divideW(); // Perspective division to normalize coordinates

      // Transform normals into world space for accurate lighting
      // no need for perspective correction as no shearing or non-uniform
      // scaling
      t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal;
      t[i].normal.normalise();

      // Map normalized device coordinates to screen space
      t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f *
                  static_cast<float>(renderer.canvas.getWidth());
      t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f *
                  static_cast<float>(renderer.canvas.getHeight());
      t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

      // Copy vertex colours
      t[i].rgb = mesh->vertices[ind.v[i]].rgb;
    }

    // Clip triangles with Z-values outside [-1, 1]
    if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f ||
        fabs(t[2].p[2]) > 1.0f)
      continue;

    // Create a triangle object and render it
    triangle tri(t[0], t[1], t[2]);
    tri.draw(renderer, L, mesh->ka, mesh->kd);
  }
}

// Access vertex data from separate arrays instead of interleaved AoS
void renderSoA(Renderer &renderer, MeshSoA &mesh, matrix &camera, Light &L) {
  // Combine perspective, camera, and world transformations
  matrix p = renderer.perspective * camera * mesh.world;

  const float halfWidth = static_cast<float>(renderer.canvas.getWidth()) * 0.5f;
  const float halfHeight =
      static_cast<float>(renderer.canvas.getHeight()) * 0.5f;
  const float screenHeight = static_cast<float>(renderer.canvas.getHeight());

  // Iterate through all triangles
  for (size_t triIdx = 0; triIdx < mesh.triangleCount; triIdx++) {
    // Get vertex indices for this triangle
    unsigned int idx0 = mesh.triIdx0[triIdx];
    unsigned int idx1 = mesh.triIdx1[triIdx];
    unsigned int idx2 = mesh.triIdx2[triIdx];

    Vertex t[3]; // Temporary array to store transformed triangle vertices
    unsigned int indices[3] = {idx0, idx1, idx2};

    // Transform each vertex of the triangle
    for (unsigned int i = 0; i < 3; i++) {
      unsigned int vi = indices[i];

      // Load position from SoA layout
      vec4 pos(mesh.px[vi], mesh.py[vi], mesh.pz[vi], mesh.pw[vi]);
      t[i].p = p * pos;
      t[i].p.divideW();

      // Load and transform normal
      vec4 normal(mesh.nx[vi], mesh.ny[vi], mesh.nz[vi], 0.f);
      t[i].normal = mesh.world * normal;
      t[i].normal.normalise();

      // Map to screen space
      t[i].p[0] = (t[i].p[0] + 1.f) * halfWidth;
      t[i].p[1] = screenHeight - (t[i].p[1] + 1.f) * halfHeight;

      // Load color from SoA layout
      t[i].rgb = colour(mesh.cr[vi], mesh.cg[vi], mesh.cb[vi]);
    }

    // Clip triangles with Z-values outside [-1, 1]
    if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f ||
        fabs(t[2].p[2]) > 1.0f)
      continue;

    // Create a triangle object and render it
    triangle tri(t[0], t[1], t[2]);
    tri.draw(renderer, L, mesh.ka, mesh.kd);
  }
}

void renderSoA_SIMD(Renderer &renderer, MeshSoA &mesh, matrix &camera,
                    Light &L) {
  const size_t vertCount = mesh.vertexCount;
  if (vertCount == 0)
    return;

  // Combine perspective, camera, and world transformations
  matrix mvp = renderer.perspective * camera * mesh.world;

  const float halfWidth = static_cast<float>(renderer.canvas.getWidth()) * 0.5f;
  const float halfHeight =
      static_cast<float>(renderer.canvas.getHeight()) * 0.5f;
  const float screenHeight = static_cast<float>(renderer.canvas.getHeight());

  // Pre-allocate aligned arrays for transformed vertices
  const size_t alignedCount =
      ((vertCount + 7) / 8) * 8; // Round up to multiple of 8
  alignas(32) std::vector<float> tPx(alignedCount), tPy(alignedCount),
      tPz(alignedCount);
  alignas(32) std::vector<float> tNx(alignedCount), tNy(alignedCount),
      tNz(alignedCount);
  alignas(32) std::vector<float> tCr(alignedCount), tCg(alignedCount),
      tCb(alignedCount);

  // Pad source arrays if needed (copy to aligned temp)
  std::vector<float> srcPx = mesh.px, srcPy = mesh.py, srcPz = mesh.pz,
                     srcPw = mesh.pw;
  std::vector<float> srcNx = mesh.nx, srcNy = mesh.ny, srcNz = mesh.nz;
  std::vector<float> srcCr = mesh.cr, srcCg = mesh.cg, srcCb = mesh.cb;
  srcPx.resize(alignedCount, 0.f);
  srcPy.resize(alignedCount, 0.f);
  srcPz.resize(alignedCount, 0.f);
  srcPw.resize(alignedCount, 1.f);
  srcNx.resize(alignedCount, 0.f);
  srcNy.resize(alignedCount, 0.f);
  srcNz.resize(alignedCount, 0.f);
  srcCr.resize(alignedCount, 0.f);
  srcCg.resize(alignedCount, 0.f);
  srcCb.resize(alignedCount, 0.f);

  // Batch transform all vertices 8 at a time
  for (size_t i = 0; i < alignedCount; i += 8) {
    TransformedVertices8 tv;
    transformVertices8_AVX(&srcPx[i], &srcPy[i], &srcPz[i], &srcPw[i],
                           &srcNx[i], &srcNy[i], &srcNz[i], &srcCr[i],
                           &srcCg[i], &srcCb[i], mvp, mesh.world, halfWidth,
                           halfHeight, screenHeight, tv);
    // Copy results to output arrays
    for (int j = 0; j < 8; j++) {
      tPx[i + j] = tv.px[j];
      tPy[i + j] = tv.py[j];
      tPz[i + j] = tv.pz[j];
      tNx[i + j] = tv.nx[j];
      tNy[i + j] = tv.ny[j];
      tNz[i + j] = tv.nz[j];
      tCr[i + j] = tv.cr[j];
      tCg[i + j] = tv.cg[j];
      tCb[i + j] = tv.cb[j];
    }
  }

  // Iterate through all triangles using pre-transformed vertices
  for (size_t triIdx = 0; triIdx < mesh.triangleCount; triIdx++) {
    unsigned int idx0 = mesh.triIdx0[triIdx];
    unsigned int idx1 = mesh.triIdx1[triIdx];
    unsigned int idx2 = mesh.triIdx2[triIdx];

    // Clip triangles with Z-values outside [-1, 1]
    if (fabs(tPz[idx0]) > 1.0f || fabs(tPz[idx1]) > 1.0f ||
        fabs(tPz[idx2]) > 1.0f)
      continue;

    // Build vertices from pre-transformed data
    Vertex t[3];
    unsigned int indices[3] = {idx0, idx1, idx2};
    for (int i = 0; i < 3; i++) {
      unsigned int vi = indices[i];
      t[i].p = vec4(tPx[vi], tPy[vi], tPz[vi], 1.0f);
      t[i].normal = vec4(tNx[vi], tNy[vi], tNz[vi], 0.0f);
      t[i].rgb = colour(tCr[vi], tCg[vi], tCb[vi]);
    }

    // Create a triangle object and render it using SIMD
    triangle tri(t[0], t[1], t[2]);
    tri.drawSIMD(renderer, L, mesh.ka, mesh.kd);
  }
}

// Use alignas(64) to ensure 64-byte alignment
// Add padding to fill the entire cache line
struct alignas(64) ThreadStats {
  long trianglesProcessed = 0;
  // Padding to fill 64-byte cache line, preventing false sharing
  char padding[64 - sizeof(long)];
};

void renderSoA_MT(Renderer &renderer, std::vector<MeshSoA> &meshes,
                  matrix &camera, Light &L) {
  const int screenWidth = renderer.canvas.getWidth();
  const int screenHeight = renderer.canvas.getHeight();

  // Compute grid dimensions and total tile count
  const int tileSize = 64;
  const int numTilesX = (screenWidth + tileSize - 1) / tileSize;
  const int numTilesY = (screenHeight + tileSize - 1) / tileSize;
  const int totalTiles = numTilesX * numTilesY;

  // Pre-transform all meshes and build triangle list with bounding boxes
  struct TransformedMesh {
    std::vector<float> tPx, tPy, tPz;
    std::vector<float> tNx, tNy, tNz;
    std::vector<float> tCr, tCg, tCb;
    std::vector<unsigned int> triIdx0, triIdx1, triIdx2;
    float ka, kd;
    // Screen-space bounding box for coarse culling
    float minX, minY, maxX, maxY;
  };

  std::vector<TransformedMesh> transformedMeshes;
  const float halfWidth = static_cast<float>(screenWidth) * 0.5f;
  const float halfHeight = static_cast<float>(screenHeight) * 0.5f;

  for (auto &mesh : meshes) {
    if (mesh.vertexCount == 0)
      continue;

    TransformedMesh tm;
    tm.ka = mesh.ka;
    tm.kd = mesh.kd;
    tm.triIdx0 = mesh.triIdx0;
    tm.triIdx1 = mesh.triIdx1;
    tm.triIdx2 = mesh.triIdx2;
    tm.minX = static_cast<float>(screenWidth);
    tm.minY = static_cast<float>(screenHeight);
    tm.maxX = 0.f;
    tm.maxY = 0.f;

    matrix mvp = renderer.perspective * camera * mesh.world;
    const size_t vertCount = mesh.vertexCount;
    const size_t alignedCount = ((vertCount + 7) / 8) * 8;

    tm.tPx.resize(alignedCount);
    tm.tPy.resize(alignedCount);
    tm.tPz.resize(alignedCount);
    tm.tNx.resize(alignedCount);
    tm.tNy.resize(alignedCount);
    tm.tNz.resize(alignedCount);
    tm.tCr.resize(alignedCount);
    tm.tCg.resize(alignedCount);
    tm.tCb.resize(alignedCount);

    // Pad source arrays
    std::vector<float> srcPx = mesh.px, srcPy = mesh.py, srcPz = mesh.pz,
                       srcPw = mesh.pw;
    std::vector<float> srcNx = mesh.nx, srcNy = mesh.ny, srcNz = mesh.nz;
    std::vector<float> srcCr = mesh.cr, srcCg = mesh.cg, srcCb = mesh.cb;
    srcPx.resize(alignedCount, 0.f);
    srcPy.resize(alignedCount, 0.f);
    srcPz.resize(alignedCount, 0.f);
    srcPw.resize(alignedCount, 1.f);
    srcNx.resize(alignedCount, 0.f);
    srcNy.resize(alignedCount, 0.f);
    srcNz.resize(alignedCount, 0.f);
    srcCr.resize(alignedCount, 0.f);
    srcCg.resize(alignedCount, 0.f);
    srcCb.resize(alignedCount, 0.f);

    // Batch transform vertices and compute bounding box
    for (size_t i = 0; i < alignedCount; i += 8) {
      TransformedVertices8 tv;
      transformVertices8_AVX(&srcPx[i], &srcPy[i], &srcPz[i], &srcPw[i],
                             &srcNx[i], &srcNy[i], &srcNz[i], &srcCr[i],
                             &srcCg[i], &srcCb[i], mvp, mesh.world, halfWidth,
                             halfHeight, static_cast<float>(screenHeight), tv);
      for (int j = 0; j < 8 && (i + j) < vertCount; j++) {
        tm.tPx[i + j] = tv.px[j];
        tm.tPy[i + j] = tv.py[j];
        tm.tPz[i + j] = tv.pz[j];
        tm.tNx[i + j] = tv.nx[j];
        tm.tNy[i + j] = tv.ny[j];
        tm.tNz[i + j] = tv.nz[j];
        tm.tCr[i + j] = tv.cr[j];
        tm.tCg[i + j] = tv.cg[j];
        tm.tCb[i + j] = tv.cb[j];
        // Update bounding box
        tm.minX = std::min(tm.minX, tv.px[j]);
        tm.minY = std::min(tm.minY, tv.py[j]);
        tm.maxX = std::max(tm.maxX, tv.px[j]);
        tm.maxY = std::max(tm.maxY, tv.py[j]);
      }
    }
    transformedMeshes.push_back(std::move(tm));
  }

  // Get thread pool reference
  ThreadPool &pool = ThreadPool::getInstance();
  const int numWorkers = pool.getNumWorkers();

  // Each worker atomically fetches next tile index until exhausted
  std::atomic<int> nextTileIndex{0};

  // Alignas(64) + padding ensures each ThreadStats is on its own line
  std::vector<ThreadStats> stats(numWorkers);

  // Loop until no tiles remain, compute tile coords from linear index
  auto worker = [&](int workerId) {
    // Thread-local light copy for independent access
    Light localL = L;
    localL.omega_i.normalise();

    while (true) {
      // Atomically grab next tile index
      int myTileIdx = nextTileIndex.fetch_add(1, std::memory_order_relaxed);

      if (myTileIdx >= totalTiles)
        break;

      // Compute tile position from linear index
      int tx = (myTileIdx % numTilesX) * tileSize;
      int ty = (myTileIdx / numTilesX) * tileSize;
      int tw = std::min(tileSize, screenWidth - tx);
      int th = std::min(tileSize, screenHeight - ty);

      // Process all visible meshes for this tile
      for (size_t mi = 0; mi < transformedMeshes.size(); mi++) {
        const auto &tm = transformedMeshes[mi];

        // Skip mesh if bounding box doesn't overlap tile
        if (tm.maxX < tx || tm.minX >= (tx + tw) || tm.maxY < ty ||
            tm.minY >= (ty + th)) {
          continue;
        }

        for (size_t triIdx = 0; triIdx < tm.triIdx0.size(); triIdx++) {
          unsigned int idx0 = tm.triIdx0[triIdx];
          unsigned int idx1 = tm.triIdx1[triIdx];
          unsigned int idx2 = tm.triIdx2[triIdx];

          if (fabs(tm.tPz[idx0]) > 1.0f || fabs(tm.tPz[idx1]) > 1.0f ||
              fabs(tm.tPz[idx2]) > 1.0f)
            continue;

          Vertex t[3];
          unsigned int indices[3] = {idx0, idx1, idx2};
          for (int i = 0; i < 3; i++) {
            unsigned int vi = indices[i];
            t[i].p = vec4(tm.tPx[vi], tm.tPy[vi], tm.tPz[vi], 1.0f);
            t[i].normal = vec4(tm.tNx[vi], tm.tNy[vi], tm.tNz[vi], 0.0f);
            t[i].rgb = colour(tm.tCr[vi], tm.tCg[vi], tm.tCb[vi]);
          }

          triangle tri(t[0], t[1], t[2]);
          tri.drawSIMD_Tiled(renderer, localL, tm.ka, tm.kd, tx, ty, tw, th);

          // Update per-thread stats
          stats[workerId].trianglesProcessed++;
        }
      }
    }
  };

  // Enqueue worker tasks with their worker IDs
  for (int i = 0; i < numWorkers; i++) {
    pool.enqueue([&worker, i]() { worker(i); });
  }

  // Wait for all workers to complete
  pool.waitFinished();
}

// Test scene function to demonstrate rendering with user-controlled
// transformations No input variables
void sceneTest() {
  Renderer renderer;
  // create light source {direction, diffuse intensity, ambient intensity}
  Light L{vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f),
          colour(0.2f, 0.2f, 0.2f)};
  // camera is just a matrix
  matrix camera =
      matrix::makeIdentity(); // Initialize the camera with identity matrix

  bool running = true; // Main loop control variable

  std::vector<Mesh *> scene; // Vector to store scene objects

  // Create a sphere and a rectangle mesh
  Mesh mesh = Mesh::makeSphere(1.0f, 10, 20);
  // Mesh mesh2 = Mesh::makeRectangle(-2, -1, 2, 1);

  // add meshes to scene
  scene.push_back(&mesh);
  // scene.push_back(&mesh2);

  float x = 0.0f, y = 0.0f, z = -4.0f; // Initial translation parameters
  mesh.world = matrix::makeTranslation(x, y, z);
  // mesh2.world = matrix::makeTranslation(x, y, z) *
  // matrix::makeRotateX(0.01f);

  // Main rendering loop
  while (running) {
    renderer.canvas.checkInput(); // Handle user input
    renderer.clear();             // Clear the canvas for the next frame

    // Apply transformations to the meshes
    // mesh2.world = matrix::makeTranslation(x, y, z) *
    // matrix::makeRotateX(0.01f);
    mesh.world = matrix::makeTranslation(x, y, z);

    // Handle user inputs for transformations
    if (renderer.canvas.keyPressed(VK_ESCAPE))
      break;
    if (renderer.canvas.keyPressed('A'))
      x += -0.1f;
    if (renderer.canvas.keyPressed('D'))
      x += 0.1f;
    if (renderer.canvas.keyPressed('W'))
      y += 0.1f;
    if (renderer.canvas.keyPressed('S'))
      y += -0.1f;
    if (renderer.canvas.keyPressed('Q'))
      z += 0.1f;
    if (renderer.canvas.keyPressed('E'))
      z += -0.1f;

    // Render each object in the scene
    for (auto &m : scene)
      render(renderer, m, camera, L);

    renderer.present(); // Display the rendered frame
  }
}

// Utility function to generate a random rotation matrix
// No input variables
matrix makeRandomRotation() {
  RandomNumberGenerator &rng = RandomNumberGenerator::getInstance();
  unsigned int r = rng.getRandomInt(0, 3);

  switch (r) {
  case 0:
    return matrix::makeRotateX(rng.getRandomFloat(0.f, 2.0f * M_PI));
  case 1:
    return matrix::makeRotateY(rng.getRandomFloat(0.f, 2.0f * M_PI));
  case 2:
    return matrix::makeRotateZ(rng.getRandomFloat(0.f, 2.0f * M_PI));
  default:
    return matrix::makeIdentity();
  }
}

// Function to render a scene with multiple objects and dynamic transformations
// No input variables
void scene1() {
  Renderer renderer;
  matrix camera;
  Light L{vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f)};

  bool running = true;

  // Create Mesh objects, convert to MeshSoA, then render with renderSoA
  std::vector<MeshSoA> sceneSoA;

  // Create a scene of 40 cubes with random rotations
  for (unsigned int i = 0; i < 20; i++) {
    Mesh m1 = Mesh::makeCube(1.f);
    m1.world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
    sceneSoA.push_back(MeshSoA::fromMesh(m1));

    Mesh m2 = Mesh::makeCube(1.f);
    m2.world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) *
               makeRandomRotation();
    sceneSoA.push_back(MeshSoA::fromMesh(m2));
  }

  float zoffset = 8.0f; // Initial camera Z-offset
  float step = -0.1f;   // Step size for camera movement

  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  int cycle = 0;

  // Main rendering loop
  while (running) {

    renderer.canvas.checkInput();
    renderer.clear();

    camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

    // Rotate the first two cubes in the scene (update world matrix in SoA)
    sceneSoA[0].world = sceneSoA[0].world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
    sceneSoA[1].world = sceneSoA[1].world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

    if (renderer.canvas.keyPressed(VK_ESCAPE))
      break;

    zoffset += step;
    if (zoffset < -60.f || zoffset > 8.f) {
      step *= -1.f;
      if (++cycle % 2 == 0) {
        end = std::chrono::high_resolution_clock::now();
        std::cout << cycle / 2 << " :"
                  << std::chrono::duration<double, std::milli>(end - start).count()
                  << "ms\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }

    // Use multi-threaded tile-based rendering with thread pool
    renderSoA_MT(renderer, sceneSoA, camera, L);
    renderer.present();
  }
}

// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2() {
  Renderer renderer;
  matrix camera = matrix::makeIdentity();
  Light L{vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f)};

  std::vector<MeshSoA> sceneSoA;

  struct rRot {
    float x;
    float y;
    float z;
  }; // Structure to store random rotation parameters
  std::vector<rRot> rotations;

  RandomNumberGenerator &rng = RandomNumberGenerator::getInstance();

  // Create a grid of cubes with random rotations
  for (unsigned int y = 0; y < 6; y++) {
    for (unsigned int x = 0; x < 8; x++) {
      Mesh m = Mesh::makeCube(1.f);
      m.world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
      sceneSoA.push_back(MeshSoA::fromMesh(m));
      rRot r{rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f)};
      rotations.push_back(r);
    }
  }

  // Create a sphere and add it to the scene
  Mesh sphereMesh = Mesh::makeSphere(1.0f, 10, 20);
  float sphereOffset = -6.f;
  float sphereStep = 0.1f;
  sphereMesh.world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
  sceneSoA.push_back(MeshSoA::fromMesh(sphereMesh));
  size_t sphereIdx = sceneSoA.size() - 1; // Index of sphere in SoA array

  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  int cycle = 0;

  bool running = true;
  while (running) {

    renderer.canvas.checkInput();
    renderer.clear();

    // Rotate each cube in the grid (update world matrix in SoA)
    for (unsigned int i = 0; i < rotations.size(); i++)
      sceneSoA[i].world = sceneSoA[i].world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

    // Move the sphere back and forth
    sphereOffset += sphereStep;
    sceneSoA[sphereIdx].world =
        matrix::makeTranslation(sphereOffset, 0.f, -6.f);
    if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
      sphereStep *= -1.f;
      if (++cycle % 2 == 0) {
        end = std::chrono::high_resolution_clock::now();
        std::cout << cycle / 2 << " :"
                  << std::chrono::duration<double, std::milli>(end - start).count()
                  << "ms\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }

    if (renderer.canvas.keyPressed(VK_ESCAPE))
      break;

    // Use multi-threaded tile-based rendering with dynamic allocation
    renderSoA_MT(renderer, sceneSoA, camera, L);
    renderer.present();
  }
}
// Spheres move in a wave pattern to ensure dynamic rendering
void scene3() {
  Renderer renderer;
  matrix camera = matrix::makeIdentity();
  Light L{vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f),
          colour(0.2f, 0.2f, 0.2f)};

  std::vector<MeshSoA> sceneSoA;
  RandomNumberGenerator &rng = RandomNumberGenerator::getInstance();

  const int gridX = 10;
  const int gridY = 20;
  const float sphereRadius = 0.3f;
  const float spacing = 0.8f;

  // Store initial positions for wave animation
  struct SpherePos {
    float baseX, baseY, baseZ;
    float phase; // Random phase offset for wave animation
  };
  std::vector<SpherePos> spherePositions;

  for (int y = 0; y < gridY; y++) {
    for (int x = 0; x < gridX; x++) {
      // Create a small sphere with moderate tessellation
      Mesh sphere = Mesh::makeSphere(sphereRadius, 10, 20);

      // Position in a grid pattern, centered on screen
      float posX = -4.0f + (static_cast<float>(x) * spacing);
      float posY = 5.0f - (static_cast<float>(y) * spacing);
      float posZ = -8.0f;

      sphere.world = matrix::makeTranslation(posX, posY, posZ);
      sceneSoA.push_back(MeshSoA::fromMesh(sphere));

      // Store position data for animation
      SpherePos sp;
      sp.baseX = posX;
      sp.baseY = posY;
      sp.baseZ = posZ;
      sp.phase = rng.getRandomFloat(0.f, 2.0f * static_cast<float>(M_PI));
      spherePositions.push_back(sp);
    }
  }

  // Animation variables
  float time = 0.0f;
  const float timeStep = 0.05f;

  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  int cycle = 0;
  int frameCount = 0;
  const int cycleFrames = 200; // Report every 200 frames

  bool running = true;
  while (running) {

    renderer.canvas.checkInput();
    renderer.clear();

    // Apply sinusoidal Z-offset based on phase and global time
    time += timeStep;
    for (size_t i = 0; i < spherePositions.size(); i++) {
      const SpherePos &sp = spherePositions[i];
      float zOffset = 2.0f * std::sin(time + sp.phase);
      sceneSoA[i].world = matrix::makeTranslation(sp.baseX, sp.baseY, sp.baseZ + zOffset) * matrix::makeRotateY(time * 0.5f);
    }

    if (renderer.canvas.keyPressed(VK_ESCAPE))
      break;

    // Use multi-threaded tile-based rendering
    renderSoA_MT(renderer, sceneSoA, camera, L);
    renderer.present();

    frameCount++;
    if (frameCount % cycleFrames == 0) {
      end = std::chrono::high_resolution_clock::now();
      std::cout << ++cycle << " :"
                << std::chrono::duration<double, std::milli>(end - start).count()
                << "ms\n";
      start = std::chrono::high_resolution_clock::now();
    }
  }
}

// Entry point of the application
// No input variables
int main() {
  // Initialize global thread pool at program start
  ThreadPool::getInstance().start();

  // Uncomment the desired scene function to run
  scene1(); // 40 cubes moving corridor
  //scene2();  // 6x8 cube grid with moving sphere
  //scene3(); // Stress test: 200 spheres (~80,000 triangles)

  return 0;
}