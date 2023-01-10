#pragma once

#include "hittable/hittable.cuh"

class Translation : public Hittable {
  public: 
    __host__ __device__ Translation(Hittable *object, const Arr3 &offset) : object{object}, offset{offset} {}
    
    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const override;
    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

  private:
    Hittable *object;
    Arr3 offset;
};

__device__ 
bool Translation::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const {
  Ray movedRay(r.origin() - this->offset, r.direction(), r.time());

  if (!this->object->hit(movedRay, tMin, tMax, hit, mat, randState)) {
    return false;
  }

  hit->point += this->offset;
  hit->faceNormal = FaceNormal(movedRay, hit->faceNormal.normal);

  return true;
}

__host__ __device__ 
float Translation::numCompare(int index) const {
  return this->object->numCompare(index) + this->offset.get(index);
}

__host__ __device__ 
bool Translation::boundingBox(BoundingRecord *box) {
  BoundingRecord outputBox;

  if (!this->object->boundingBox(&outputBox)) {
    return false;
  }

  box->boundingBox = AABB(
    outputBox.boundingBox.minimum() + this->offset,
    outputBox.boundingBox.maximum() + this->offset
  );

  return true;
}
