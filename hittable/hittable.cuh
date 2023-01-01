#pragma once

#include "../struct/hit_record.cuh"
#include "../struct/material_record.cuh"
#include "../material/material.cuh"

class Hittable {
  public:
    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord &rec, MaterialRecord &mat) const = 0;
};
