#pragma once

#include "struct/hit_record.cuh"
#include "struct/material_record.cuh"
#include "material/material.cuh"
#include "struct/bounding_record.cuh"

class Hittable {
  public:
    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat, curandState* randState) const = 0;
    __host__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat) const = 0;

    __host__ __device__ virtual float numCompare(int index) const = 0;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) = 0;

    __host__ virtual Hittable* copyToDevice();
};

__host__ Hittable* Hittable::copyToDevice() {
  Hittable *cudaHit;

  cudaMalloc((void**) &cudaHit, sizeof(*this));
  cudaMemcpy(cudaHit, this, sizeof(*this), cudaMemcpyHostToDevice);

  return cudaHit;
}
