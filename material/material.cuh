#pragma once

#include "struct/hit_record.cuh"
#include "struct/scatter_record.cuh"

class Material {
  public:
    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const { return false; };
    __host__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const { return false; };

    __device__ virtual float scatteringPdf(const Ray &ray, const HitRecord &hit, const Ray &scatteredRay, curandState* randState) const { return 0; }
    __host__ virtual float scatteringPdf(const Ray &ray, const HitRecord &hit, const Ray &scatteredRay) const { return 0; }

    __host__ __device__ virtual Arr3 emitted(const Ray &ray, const HitRecord &hit) const {
      return Arr3(0.0f, 0.0f, 0.0f);
    }
};
