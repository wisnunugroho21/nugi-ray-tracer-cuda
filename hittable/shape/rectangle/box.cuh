#pragma once

#include "helper/helper.cuh"
#include "hittable/hittable_list.cuh"

#include "hittable/shape/rectangle/xy_rect.cuh"
#include "hittable/shape/rectangle/xz_rect.cuh"
#include "hittable/shape/rectangle/yz_rect.cuh"

class Box : public Hittable {
  public:
    __host__ __device__ Box() {}
    __host__ __device__ Box(Arr3 minBoxPoint, Arr3 maxBoxPoint, Material *material);

    __host__ __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;
    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

  private:
    Arr3 minBoxPoint;
    Arr3 maxBoxPoint;

    Hittable **itemSides;
    HittableList *sides;
};

__host__ __device__ 
Box::Box(Arr3 minBoxPoint, Arr3 maxBoxPoint, Material *material) {
  this->minBoxPoint = minBoxPoint;
  this->maxBoxPoint = maxBoxPoint;

  itemSides = (Hittable**) malloc(6 * sizeof(Hittable*));

  itemSides[0] = new XYRect(minBoxPoint.x(), maxBoxPoint.x(), minBoxPoint.y(), maxBoxPoint.y(), maxBoxPoint.z(), material);
  itemSides[1] = new XYRect(minBoxPoint.x(), maxBoxPoint.x(), minBoxPoint.y(), maxBoxPoint.y(), minBoxPoint.z(), material);

  itemSides[2] = new XZRect(minBoxPoint.x(), maxBoxPoint.x(), minBoxPoint.z(), maxBoxPoint.z(), maxBoxPoint.y(), material);
  itemSides[3] = new XZRect(minBoxPoint.x(), maxBoxPoint.x(), minBoxPoint.z(), maxBoxPoint.z(), minBoxPoint.y(), material);

  itemSides[4] = new YZRect(minBoxPoint.y(), maxBoxPoint.y(), minBoxPoint.z(), maxBoxPoint.z(), maxBoxPoint.x(), material);
  itemSides[5] = new YZRect(minBoxPoint.y(), maxBoxPoint.y(), minBoxPoint.z(), maxBoxPoint.z(), minBoxPoint.x(), material);

  this->sides = new HittableList(itemSides, 6);
}

__host__ __device__ 
bool Box::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const {
  return this->sides->hit(r, tMin, tMax, hit, mat);
}

__host__ __device__ 
float Box::numCompare(int index) const {
  return this->minBoxPoint.get(index);
}

__host__ __device__ 
bool Box::boundingBox(BoundingRecord *box) {
  box->boundingBox = AABB(this->minBoxPoint, this->maxBoxPoint);
  return true;
}


