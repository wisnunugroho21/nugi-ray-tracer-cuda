#pragma once

#include "struct/hit_record.cuh"
#include "struct/scatter_record.cuh"

class Material {
  public:
    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const = 0;
    __host__ __device__ virtual Arr3 emitted(float u, float v, const Arr3 &point) const {
      return Arr3(0.0f, 0.0f, 0.0f);
    }
};
