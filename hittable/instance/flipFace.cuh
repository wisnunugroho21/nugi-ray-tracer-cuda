#pragma once

#include "hittable/hittable.cuh"

class FlipFace : public Hittable {
  public:
    __host__ __device__ FlipFace(Hittable *object) : object{object} {}

    __host__ __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;
    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

  private:
    Hittable *object;
};

__host__ __device__ 
bool FlipFace::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const {
  if (!this->object->hit(r, tMin, tMax, hit, mat)) {
    return false;
  }

  hit->faceNormal.frontFace = !hit->faceNormal.frontFace;
  return true;
}

__host__ __device__ 
float FlipFace::numCompare(int index) const {
  return this->object->numCompare(index);
}

__host__ __device__ 
bool FlipFace::boundingBox(BoundingRecord *box) {
  return this->object->boundingBox(box);
}
