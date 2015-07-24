/*
 * fmatrix4.hpp
 *
 *  Created on: 2014-10-05
 *      Author: tombr
 */

#ifndef TBBLAS_IMGPROC_FMATRIX4_HPP_
#define TBBLAS_IMGPROC_FMATRIX4_HPP_

#include <cuda_runtime.h>

namespace tbblas {

namespace imgproc {

struct fmatrix4 {
  union {
    struct { float4 r1, r2, r3, r4; };
    float _array[16];
  };
};

// additional constructors
inline __host__ __device__ fmatrix4 make_fmatrix4(float4 r1, float4 r2, float4 r3, float4 r4) {
  fmatrix4 fm4;
  fm4.r1 = r1;
  fm4.r2 = r2;
  fm4.r3 = r3;
  fm4.r4 = r4;
  return fm4;
}

inline __host__ __device__ fmatrix4 make_fmatrix4(float m11, float m12, float m13, float m14,
    float m21, float m22, float m23, float m24,
    float m31, float m32, float m33, float m34,
    float m41, float m42, float m43, float m44) {
  fmatrix4 fm4;
  fm4.r1 = make_float4(m11, m12, m13, m14);
  fm4.r2 = make_float4(m21, m22, m23, m24);
  fm4.r3 = make_float4(m31, m32, m33, m34);
  fm4.r4 = make_float4(m41, m42, m43, m44);
  return fm4;
}

// column getter
inline __host__ __device__ float4 get_x_column(fmatrix4 fm4) {
  return make_float4(fm4.r1.x, fm4.r2.x, fm4.r3.x, fm4.r4.x);
}

inline __host__ __device__ float4 get_y_column(fmatrix4 fm4) {
  return make_float4(fm4.r1.y, fm4.r2.y, fm4.r3.y, fm4.r4.y);
}

inline __host__ __device__ float4 get_z_column(fmatrix4 fm4) {
  return make_float4(fm4.r1.z, fm4.r2.z, fm4.r3.z, fm4.r4.z);
}

inline __host__ __device__ float4 get_w_column(fmatrix4 fm4) {
  return make_float4(fm4.r1.w, fm4.r2.w, fm4.r3.w, fm4.r4.w);
}

// special getter for float4
inline __host__ __device__ float get_x(float4 f) {
  return f.x/f.w;
}

inline __host__ __device__ float get_y(float4 f) {
  return f.y/f.w;
}

inline __host__ __device__ float get_z(float4 f) {
  return f.z/f.w;
}

// special matrices
inline __host__ __device__ fmatrix4 make_fmatrix4_identity() {
  return make_fmatrix4(1, 0, 0, 0,
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1);
}

inline __host__ __device__ fmatrix4 make_fmatrix4_translation(float x = 0, float y = 0, float z = 0) {
  return make_fmatrix4(1, 0, 0, x,
                       0, 1, 0, y,
                       0, 0, 1, z,
                       0, 0, 0, 1);
}

inline __host__ __device__ fmatrix4 make_fmatrix4_translation(float3 v) {
  return make_fmatrix4_translation(v.x, v.y, v.z);
}

inline __host__ __device__ fmatrix4 make_fmatrix4_scaling(float x) {
  return make_fmatrix4(x, 0, 0, 0,
                       0, x, 0, 0,
                       0, 0, x, 0,
                       0, 0, 0, 1);
}

inline __host__ __device__ fmatrix4 make_fmatrix4_scaling(float x, float y, float z) {
  return make_fmatrix4(x, 0, 0, 0,
                       0, y, 0, 0,
                       0, 0, z, 0,
                       0, 0, 0, 1);
}

inline __host__ __device__ fmatrix4 make_fmatrix4_scaling(float3 v) {
  return make_fmatrix4_scaling(v.x, v.y, v.z);
}

inline __host__ __device__ fmatrix4 make_fmatrix4_rotationX(float angle) {
  return make_fmatrix4( 1,          0,           0, 0,
                        0, cos(angle), -sin(angle), 0,
                        0, sin(angle),  cos(angle), 0,
                        0,          0,           0, 1);
}

inline __host__ __device__ fmatrix4 make_fmatrix4_rotationY(float angle) {
  return make_fmatrix4(  cos(angle), 0, sin(angle), 0,
                                  0, 1,          0, 0,
                        -sin(angle), 0, cos(angle), 0,
                                  0, 0,          0, 1);
}

inline __host__ __device__ fmatrix4 make_fmatrix4_rotationZ(float angle) {
  return make_fmatrix4( cos(angle), -sin(angle), 0, 0,
                        sin(angle),  cos(angle), 0, 0,
                                 0,           0, 1, 0,
                                 0,           0, 0, 1);
}

// addition
inline __host__ __device__ float4 operator+(const float4& a, const float4& b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __host__ __device__ fmatrix4 operator+(fmatrix4 a, fmatrix4 b) {
  return make_fmatrix4(a.r1+b.r1, a.r2+b.r2, a.r3+b.r3, a.r4+b.r4);
}

// dot
inline __host__ __device__ float dot(const float4& a, const float4& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// multiplication
inline __host__ __device__ float4 operator*(const fmatrix4& a, const float4& b) {
  return make_float4(dot(a.r1,b), dot(a.r2,b), dot(a.r3,b), dot(a.r4,b));
}

inline __host__ __device__ fmatrix4 operator*(const fmatrix4& a, const fmatrix4& b) {
  float4 xcol = get_x_column(b);
  float4 ycol = get_y_column(b);
  float4 zcol = get_z_column(b);
  float4 wcol = get_w_column(b);
  return make_fmatrix4(dot(a.r1, xcol), dot(a.r1, ycol), dot(a.r1, zcol), dot(a.r1, wcol),
                       dot(a.r2, xcol), dot(a.r2, ycol), dot(a.r2, zcol), dot(a.r2, wcol),
                       dot(a.r3, xcol), dot(a.r3, ycol), dot(a.r3, zcol), dot(a.r3, wcol),
                       dot(a.r4, xcol), dot(a.r4, ycol), dot(a.r4, zcol), dot(a.r4, wcol));
}

}

}

#endif /* TRANSFORM_HPP_ */
