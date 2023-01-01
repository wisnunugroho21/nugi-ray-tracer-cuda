#pragma once

#include "../struct/hit_record.cuh"
#include "../struct/scatter_record.cuh"

class Material {
  public:
    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const = 0;
};
