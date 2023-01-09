#pragma once

#include "struct/hit_record.cuh"
#include "struct/material_record.cuh"
#include "material/material.cuh"
#include "struct/bounding_record.cuh"

class Hittable {
  public:
    __host__ __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat) const = 0;
    __host__ __device__ virtual float numCompare(int index) const = 0;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) = 0;

    __host__ __device__
    virtual float pdfValue(const Arr3 &origin, const Arr3 &direction) const {
      return 0.0f;
    }

    __device__
    virtual Arr3 random(const Arr3 &origin, curandState* randState) const {
      return Arr3(1.0f, 0.0f, 0.0f);
    }
};
