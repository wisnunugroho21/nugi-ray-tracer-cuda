#pragma once

#include "helper/helper.cuh"
#include "utility/ray.cuh"

class AABB {
  public:
    __host__ __device__ AABB() {}
    __host__ __device__ AABB(Arr3 minPoint, Arr3 maxPoint) : minPoint{minPoint}, maxPoint{maxPoint}  {}

    __host__ __device__ Arr3 minimum() const { return this->minPoint; }
    __host__ __device__ Arr3 maximum() const { return this->maxPoint; }

    __host__ __device__ bool hit(const Ray &ray, float tMin, float tMax) const;

    __host__ __device__ static AABB surrondingBox(AABB box0, AABB box1);

  private:
    Arr3 minPoint;
    Arr3 maxPoint;
};

__host__ __device__
bool AABB::hit(const Ray &ray, float tMin, float tMax) const {
  for (int a = 0; a < 3; a++) {
    auto invD = 1.0f / ray.direction().get(a);
    
    auto t0 = (this->minimum().get(a) - ray.origin().get(a)) * invD;
    auto t1 = (this->maximum().get(a) - ray.origin().get(a)) * invD;

    if (invD < 0.0f) {
      float temp = t0;
      t0 = t1;
      t1 = temp;
    }

    tMin = t0 > tMin ? t0 : tMin;
    tMax = t1 < tMax ? t1 : tMax;

    if (tMax <= tMin) {
      return false;
    }
  }
  
  return true;
}

__host__ __device__
AABB AABB::surrondingBox(AABB box0, AABB box1) {
  Arr3 minPoint(
    fminf(box0.minimum().x(), box1.minimum().x()), 
    fminf(box0.minimum().y(), box1.minimum().y()),
    fminf(box0.minimum().z(), box1.minimum().z())
  );

  Arr3 maxPoint(
    fmaxf(box0.maximum().x(), box1.maximum().x()), 
    fmaxf(box0.maximum().y(), box1.maximum().y()),
    fmaxf(box0.maximum().z(), box1.maximum().z())
  );

  return AABB(minPoint, maxPoint);
}