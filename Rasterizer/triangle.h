#pragma once

#include "colour.h"
#include "light.h"
#include "mesh.h"
#include "renderer.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h> // AVX/AVX2 intrinsics for SIMD rasterization
#include <iostream>

// Simple support class for a 2D vector
class vec2D {
public:
  float x, y;

  // Default constructor initializes both components to 0
  vec2D() { x = y = 0.f; };

  // Constructor initializes components with given values
  vec2D(float _x, float _y) : x(_x), y(_y) {}

  // Constructor initializes components from a vec4
  vec2D(vec4 v) {
    x = v[0];
    y = v[1];
  }

  // Display the vector components
  void display() { std::cout << x << '\t' << y << std::endl; }

  // Overloaded subtraction operator for vector subtraction
  vec2D operator-(vec2D &v) {
    vec2D q;
    q.x = x - v.x;
    q.y = y - v.y;
    return q;
  }
};

// Class representing a triangle for rendering purposes
class triangle {
  Vertex v[3];   // Vertices of the triangle
  float area;    // Area of the triangle
  colour col[3]; // Colors for each vertex of the triangle

public:
  // Constructor initializes the triangle with three vertices
  // Input Variables:
  // - v1, v2, v3: Vertices defining the triangle
  triangle(const Vertex &v1, const Vertex &v2, const Vertex &v3) {
    v[0] = v1;
    v[1] = v2;
    v[2] = v3;

    // Calculate the 2D area of the triangle
    vec2D e1 = vec2D(v[1].p - v[0].p);
    vec2D e2 = vec2D(v[2].p - v[0].p);
    area = std::fabs(e1.x * e2.y - e1.y * e2.x);
  }

  // Helper function to compute the cross product for barycentric coordinates
  // Input Variables:
  // - v1, v2: Edges defining the vector
  // - p: Point for which coordinates are being calculated
  float getC(vec2D v1, vec2D v2, vec2D p) {
    vec2D e = v2 - v1;
    vec2D q = p - v1;
    return q.y * e.x - q.x * e.y;
  }

  // Compute barycentric coordinates for a given point
  // Input Variables:
  // - p: Point to check within the triangle
  // Output Variables:
  // - alpha, beta, gamma: Barycentric coordinates of the point
  // Returns true if the point is inside the triangle, false otherwise
  bool getCoordinates(vec2D p, float &alpha, float &beta, float &gamma) {
    alpha = getC(vec2D(v[0].p), vec2D(v[1].p), p) / area;
    beta = getC(vec2D(v[1].p), vec2D(v[2].p), p) / area;
    gamma = getC(vec2D(v[2].p), vec2D(v[0].p), p) / area;

    if (alpha < 0.f || beta < 0.f || gamma < 0.f)
      return false;
    return true;
  }

  // Template function to interpolate values using barycentric coordinates
  // Input Variables:
  // - alpha, beta, gamma: Barycentric coordinates
  // - a1, a2, a3: Values to interpolate
  // Returns the interpolated value
  template <typename T>
  T interpolate(float alpha, float beta, float gamma, T a1, T a2, T a3) {
    return (a1 * alpha) + (a2 * beta) + (a3 * gamma);
  }

  // Draw the triangle on the canvas
  // Input Variables:
  // - renderer: Renderer object for drawing
  // - L: Light object for shading calculations
  // - ka, kd: Ambient and diffuse lighting coefficients
  void draw(Renderer &renderer, Light &L, float ka, float kd) {
    vec2D minV, maxV;

    // Get the screen-space bounds of the triangle
    getBoundsWindow(renderer.canvas, minV, maxV);

    // Skip very small triangles
    if (area < 1.f)
      return;

    // Iterate over the bounding box and check each pixel
    for (int y = (int)(minV.y); y < (int)ceil(maxV.y); y++) {
      for (int x = (int)(minV.x); x < (int)ceil(maxV.x); x++) {
        float alpha, beta, gamma;

        // Check if the pixel lies inside the triangle
        if (getCoordinates(vec2D((float)x, (float)y), alpha, beta, gamma)) {
          // Interpolate color, depth, and normals
          colour c =
              interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
          c.clampColour();
          float depth =
              interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
          vec4 normal = interpolate(beta, gamma, alpha, v[0].normal,
                                    v[1].normal, v[2].normal);
          normal.normalise();

          // Perform Z-buffer test and apply shading
          if (renderer.zbuffer(x, y) > depth && depth > 0.001f) {
            // typical shader begin
            L.omega_i.normalise();
            float dot = std::max(vec4::dot(L.omega_i, normal), 0.0f);
            colour a = (c * kd) * (L.L * dot) +
                       (L.ambient * ka); // using kd instead of ka for ambient
            // typical shader end
            unsigned char r, g, b;
            a.toRGB(r, g, b);
            renderer.canvas.draw(x, y, r, g, b);
            renderer.zbuffer(x, y) = depth;
          }
        }
      }
    }
  }

  void drawSIMD(Renderer &renderer, Light &L, float ka, float kd) {
    vec2D minV, maxV;
    getBoundsWindow(renderer.canvas, minV, maxV);

    if (area < 1.f)
      return;

    // Pre-compute edge coefficients for barycentric calculation
    // Edge from v0 to v1: e01
    const float e01_x = v[1].p[0] - v[0].p[0];
    const float e01_y = v[1].p[1] - v[0].p[1];
    // Edge from v1 to v2: e12
    const float e12_x = v[2].p[0] - v[1].p[0];
    const float e12_y = v[2].p[1] - v[1].p[1];
    // Edge from v2 to v0: e20
    const float e20_x = v[0].p[0] - v[2].p[0];
    const float e20_y = v[0].p[1] - v[2].p[1];

    const float inv_area = 1.0f / area;

    // Vertex positions
    const float v0x = v[0].p[0], v0y = v[0].p[1], v0z = v[0].p[2];
    const float v1x = v[1].p[0], v1y = v[1].p[1], v1z = v[1].p[2];
    const float v2x = v[2].p[0], v2y = v[2].p[1], v2z = v[2].p[2];

    // Pre-normalize light direction
    L.omega_i.normalise();

    const int startX = (int)minV.x;
    const int endX = (int)ceil(maxV.x);
    const int startY = (int)minV.y;
    const int endY = (int)ceil(maxV.y);

    // AVX constants
    __m256 zero = _mm256_setzero_ps();
    __m256 inv_area_v = _mm256_set1_ps(inv_area);
    __m256 depth_threshold = _mm256_set1_ps(0.001f);

    // X offset for 8 consecutive pixels: [0, 1, 2, 3, 4, 5, 6, 7]
    __m256 x_offset = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);

    for (int y = startY; y < endY; y++) {
      const float py = static_cast<float>(y);

      // Process 8 pixels at a time
      for (int x = startX; x < endX; x += 8) {
        // Generate X coordinates for 8 pixels
        __m256 px =
            _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x)), x_offset);
        __m256 py_v = _mm256_set1_ps(py);

        // Compute barycentric coordinates for all 8 pixels
        // alpha = ((py - v0y) * e01_x - (px - v0x) * e01_y) / area
        __m256 qx_0 = _mm256_sub_ps(px, _mm256_set1_ps(v0x));
        __m256 qy_0 = _mm256_sub_ps(py_v, _mm256_set1_ps(v0y));
        __m256 alpha = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_mul_ps(qy_0, _mm256_set1_ps(e01_x)),
                          _mm256_mul_ps(qx_0, _mm256_set1_ps(e01_y))),
            inv_area_v);

        // beta = ((py - v1y) * e12_x - (px - v1x) * e12_y) / area
        __m256 qx_1 = _mm256_sub_ps(px, _mm256_set1_ps(v1x));
        __m256 qy_1 = _mm256_sub_ps(py_v, _mm256_set1_ps(v1y));
        __m256 beta = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_mul_ps(qy_1, _mm256_set1_ps(e12_x)),
                          _mm256_mul_ps(qx_1, _mm256_set1_ps(e12_y))),
            inv_area_v);

        // gamma = ((py - v2y) * e20_x - (px - v2x) * e20_y) / area
        __m256 qx_2 = _mm256_sub_ps(px, _mm256_set1_ps(v2x));
        __m256 qy_2 = _mm256_sub_ps(py_v, _mm256_set1_ps(v2y));
        __m256 gamma = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_mul_ps(qy_2, _mm256_set1_ps(e20_x)),
                          _mm256_mul_ps(qx_2, _mm256_set1_ps(e20_y))),
            inv_area_v);

        // Generate inside mask: all coords >= 0
        __m256 mask_alpha = _mm256_cmp_ps(alpha, zero, _CMP_GE_OQ);
        __m256 mask_beta = _mm256_cmp_ps(beta, zero, _CMP_GE_OQ);
        __m256 mask_gamma = _mm256_cmp_ps(gamma, zero, _CMP_GE_OQ);
        __m256 inside_mask =
            _mm256_and_ps(mask_alpha, _mm256_and_ps(mask_beta, mask_gamma));

        // Early exit if no pixels are inside
        int inside_bits = _mm256_movemask_ps(inside_mask);
        if (inside_bits == 0)
          continue;

        // Interpolate depth: z = beta*v0z + gamma*v1z + alpha*v2z
        __m256 depth = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(beta, _mm256_set1_ps(v0z)),
                          _mm256_mul_ps(gamma, _mm256_set1_ps(v1z))),
            _mm256_mul_ps(alpha, _mm256_set1_ps(v2z)));

        // Check depth > 0.001
        __m256 depth_valid_mask =
            _mm256_cmp_ps(depth, depth_threshold, _CMP_GT_OQ);

        // Combine masks
        __m256 valid_mask = _mm256_and_ps(inside_mask, depth_valid_mask);
        int valid_bits = _mm256_movemask_ps(valid_mask);
        if (valid_bits == 0)
          continue;

        // Extract values and process valid pixels individually
        // (Z-buffer access requires scalar operations)
        alignas(32) float alpha_arr[8], beta_arr[8], gamma_arr[8], depth_arr[8];
        _mm256_store_ps(alpha_arr, alpha);
        _mm256_store_ps(beta_arr, beta);
        _mm256_store_ps(gamma_arr, gamma);
        _mm256_store_ps(depth_arr, depth);

        for (int i = 0; i < 8 && (x + i) < endX; i++) {
          if (!(valid_bits & (1 << i)))
            continue;

          int px_i = x + i;
          float d = depth_arr[i];

          // Z-buffer test (must be scalar due to random access)
          if (renderer.zbuffer(px_i, y) > d) {
            float a = alpha_arr[i], b = beta_arr[i], g = gamma_arr[i];

            // Interpolate color
            colour c = interpolate(b, g, a, v[0].rgb, v[1].rgb, v[2].rgb);
            c.clampColour();

            // Interpolate and normalize normal
            vec4 normal =
                interpolate(b, g, a, v[0].normal, v[1].normal, v[2].normal);
            normal.normalise();

            // Shading
            float dot_val = std::max(vec4::dot(L.omega_i, normal), 0.0f);
            colour shaded = (c * kd) * (L.L * dot_val) + (L.ambient * ka);

            unsigned char r, g_c, b_c;
            shaded.toRGB(r, g_c, b_c);
            renderer.canvas.draw(px_i, y, r, g_c, b_c);
            renderer.zbuffer(px_i, y) = d;
          }
        }
      }
    }
  }

  // Clip triangle bounds to tile bounds before rasterization loop
  void drawSIMD_Tiled(Renderer &renderer, Light &L, float ka, float kd,
                      int tileX, int tileY, int tileW, int tileH) {
    vec2D minV, maxV;
    getBoundsWindow(renderer.canvas, minV, maxV);

    if (area < 1.f)
      return;

    // Clip to tile bounds
    int startX = std::max((int)minV.x, tileX);
    int endX = std::min((int)ceil(maxV.x), tileX + tileW);
    int startY = std::max((int)minV.y, tileY);
    int endY = std::min((int)ceil(maxV.y), tileY + tileH);

    // Early exit if triangle doesn't overlap tile
    if (startX >= endX || startY >= endY)
      return;

    // Pre-compute edge coefficients
    const float e01_x = v[1].p[0] - v[0].p[0];
    const float e01_y = v[1].p[1] - v[0].p[1];
    const float e12_x = v[2].p[0] - v[1].p[0];
    const float e12_y = v[2].p[1] - v[1].p[1];
    const float e20_x = v[0].p[0] - v[2].p[0];
    const float e20_y = v[0].p[1] - v[2].p[1];

    const float inv_area = 1.0f / area;
    const float v0x = v[0].p[0], v0y = v[0].p[1], v0z = v[0].p[2];
    const float v1x = v[1].p[0], v1y = v[1].p[1], v1z = v[1].p[2];
    const float v2x = v[2].p[0], v2y = v[2].p[1], v2z = v[2].p[2];

    L.omega_i.normalise();

    __m256 zero = _mm256_setzero_ps();
    __m256 inv_area_v = _mm256_set1_ps(inv_area);
    __m256 depth_threshold = _mm256_set1_ps(0.001f);
    __m256 x_offset = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);

    for (int y = startY; y < endY; y++) {
      const float py = static_cast<float>(y);

      for (int x = startX; x < endX; x += 8) {
        __m256 px =
            _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x)), x_offset);
        __m256 py_v = _mm256_set1_ps(py);

        // Barycentric coords
        __m256 qx_0 = _mm256_sub_ps(px, _mm256_set1_ps(v0x));
        __m256 qy_0 = _mm256_sub_ps(py_v, _mm256_set1_ps(v0y));
        __m256 alpha = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_mul_ps(qy_0, _mm256_set1_ps(e01_x)),
                          _mm256_mul_ps(qx_0, _mm256_set1_ps(e01_y))),
            inv_area_v);

        __m256 qx_1 = _mm256_sub_ps(px, _mm256_set1_ps(v1x));
        __m256 qy_1 = _mm256_sub_ps(py_v, _mm256_set1_ps(v1y));
        __m256 beta = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_mul_ps(qy_1, _mm256_set1_ps(e12_x)),
                          _mm256_mul_ps(qx_1, _mm256_set1_ps(e12_y))),
            inv_area_v);

        __m256 qx_2 = _mm256_sub_ps(px, _mm256_set1_ps(v2x));
        __m256 qy_2 = _mm256_sub_ps(py_v, _mm256_set1_ps(v2y));
        __m256 gamma = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_mul_ps(qy_2, _mm256_set1_ps(e20_x)),
                          _mm256_mul_ps(qx_2, _mm256_set1_ps(e20_y))),
            inv_area_v);

        __m256 mask_alpha = _mm256_cmp_ps(alpha, zero, _CMP_GE_OQ);
        __m256 mask_beta = _mm256_cmp_ps(beta, zero, _CMP_GE_OQ);
        __m256 mask_gamma = _mm256_cmp_ps(gamma, zero, _CMP_GE_OQ);
        __m256 inside_mask =
            _mm256_and_ps(mask_alpha, _mm256_and_ps(mask_beta, mask_gamma));

        int inside_bits = _mm256_movemask_ps(inside_mask);
        if (inside_bits == 0)
          continue;

        __m256 depth = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(beta, _mm256_set1_ps(v0z)),
                          _mm256_mul_ps(gamma, _mm256_set1_ps(v1z))),
            _mm256_mul_ps(alpha, _mm256_set1_ps(v2z)));

        __m256 depth_valid_mask =
            _mm256_cmp_ps(depth, depth_threshold, _CMP_GT_OQ);
        __m256 valid_mask = _mm256_and_ps(inside_mask, depth_valid_mask);
        int valid_bits = _mm256_movemask_ps(valid_mask);
        if (valid_bits == 0)
          continue;

        alignas(32) float alpha_arr[8], beta_arr[8], gamma_arr[8], depth_arr[8];
        _mm256_store_ps(alpha_arr, alpha);
        _mm256_store_ps(beta_arr, beta);
        _mm256_store_ps(gamma_arr, gamma);
        _mm256_store_ps(depth_arr, depth);

        for (int i = 0; i < 8 && (x + i) < endX; i++) {
          if (!(valid_bits & (1 << i)))
            continue;

          int px_i = x + i;
          float d = depth_arr[i];

          if (renderer.zbuffer(px_i, y) > d) {
            float a = alpha_arr[i], b = beta_arr[i], g = gamma_arr[i];
            colour c = interpolate(b, g, a, v[0].rgb, v[1].rgb, v[2].rgb);
            c.clampColour();
            vec4 normal =
                interpolate(b, g, a, v[0].normal, v[1].normal, v[2].normal);
            normal.normalise();
            float dot_val = std::max(vec4::dot(L.omega_i, normal), 0.0f);
            colour shaded = (c * kd) * (L.L * dot_val) + (L.ambient * ka);
            unsigned char r, g_c, b_c;
            shaded.toRGB(r, g_c, b_c);
            renderer.canvas.draw(px_i, y, r, g_c, b_c);
            renderer.zbuffer(px_i, y) = d;
          }
        }
      }
    }
  }

  // Compute the 2D bounds of the triangle
  // Output Variables:
  // - minV, maxV: Minimum and maximum bounds in 2D space
  void getBounds(vec2D &minV, vec2D &maxV) {
    minV = vec2D(v[0].p);
    maxV = vec2D(v[0].p);
    for (unsigned int i = 1; i < 3; i++) {
      minV.x = std::min(minV.x, v[i].p[0]);
      minV.y = std::min(minV.y, v[i].p[1]);
      maxV.x = std::max(maxV.x, v[i].p[0]);
      maxV.y = std::max(maxV.y, v[i].p[1]);
    }
  }

  // Compute the 2D bounds of the triangle, clipped to the canvas
  // Input Variables:
  // - canvas: Reference to the rendering canvas
  // Output Variables:
  // - minV, maxV: Clipped minimum and maximum bounds
  void getBoundsWindow(GamesEngineeringBase::Window &canvas, vec2D &minV,
                       vec2D &maxV) {
    getBounds(minV, maxV);
    minV.x = std::max(minV.x, static_cast<float>(0));
    minV.y = std::max(minV.y, static_cast<float>(0));
    maxV.x = std::min(maxV.x, static_cast<float>(canvas.getWidth()));
    maxV.y = std::min(maxV.y, static_cast<float>(canvas.getHeight()));
  }

  // Debugging utility to display the triangle bounds on the canvas
  // Input Variables:
  // - canvas: Reference to the rendering canvas
  void drawBounds(GamesEngineeringBase::Window &canvas) {
    vec2D minV, maxV;
    getBounds(minV, maxV);

    for (int y = (int)minV.y; y < (int)maxV.y; y++) {
      for (int x = (int)minV.x; x < (int)maxV.x; x++) {
        canvas.draw(x, y, 255, 0, 0);
      }
    }
  }

  // Debugging utility to display the coordinates of the triangle vertices
  void display() {
    for (unsigned int i = 0; i < 3; i++) {
      v[i].p.display();
    }
    std::cout << std::endl;
  }
};
