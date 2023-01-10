#pragma once

#include "hittable/hittable.cuh"
#include "material/isotropic.cuh"
#include "texture/texture.cuh"

#include <limits>

class ConstantMedium : public Hittable {
  public:
    __host__ __device__ ConstantMedium(Hittable *boundary, float density, Material *material) : 
      boundary{boundary}, 
      negInvDensity{-1.0f / density}, 
      material{material}
      {}

    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat, curandState* randState) const override;
    __host__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat) const override;

    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

  private:
    Hittable *boundary;
    Material *material;
    float negInvDensity;
};

__device__ 
bool ConstantMedium::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const {
  // Print occasional samples when debugging. To enable, set enableDebug true.
    HitRecord hit1, hit2;
    MaterialRecord mat1, mat2;

    if (!this->boundary->hit(r, -9999.0f, 9999.0f, &hit1, &mat1, randState)) {
      return false;
    } 

    if (!this->boundary->hit(r, hit1.t + 0.0001f, 9999.0f, &hit2, &mat2, randState)) {
      return false;
    } 

    if (hit1.t < tMin) hit1.t = tMin;
    if (hit2.t > tMax) hit2.t = tMax;

    if (hit1.t >= hit2.t) {
      return false;
    }   

    if (hit1.t < 0) {
      hit1.t = 0;
    }

    const auto rayLength = r.direction().length();
    const auto distance_inside_boundary = (hit2.t - hit1.t) * rayLength;
    const auto hit_distance = this->negInvDensity * logf(randomFloat(randState));

    if (hit_distance > distance_inside_boundary) {
      return false;
    }

    hit->t = hit1.t + hit_distance / rayLength;
    hit->point = r.at(hit->t);

    hit->faceNormal.normal = Arr3(1.0f, 0.0f, 0.0f);
    hit->faceNormal.frontFace = true;

    mat->material = this->material;
    return true;
}

__host__ 
bool ConstantMedium::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const {
  // Print occasional samples when debugging. To enable, set enableDebug true.
    HitRecord hit1, hit2;
    MaterialRecord mat1, mat2;

    if (!this->boundary->hit(r, -9999.0f, 9999.0f, &hit1, &mat1)) {
      return false;
    } 

    if (!this->boundary->hit(r, hit1.t + 0.0001f, 9999.0f, &hit2, &mat2)) {
      return false;
    } 

    if (hit1.t < tMin) hit1.t = tMin;
    if (hit2.t > tMax) hit2.t = tMax;

    if (hit1.t >= hit2.t) {
      return false;
    }   

    if (hit1.t < 0) {
      hit1.t = 0;
    }

    const auto rayLength = r.direction().length();
    const auto distance_inside_boundary = (hit2.t - hit1.t) * rayLength;
    const auto hit_distance = this->negInvDensity * logf(randomFloat());

    if (hit_distance > distance_inside_boundary) {
      return false;
    }

    hit->t = hit1.t + hit_distance / rayLength;
    hit->point = r.at(hit->t);

    hit->faceNormal.normal = Arr3(1.0f, 0.0f, 0.0f);
    hit->faceNormal.frontFace = true;

    mat->material = this->material;
    return true;
}

__host__ __device__ 
float ConstantMedium::numCompare(int index) const {
  return this->boundary->numCompare(index);
}

__host__ __device__ 
bool ConstantMedium::boundingBox(BoundingRecord *box) {
  BoundingRecord outputBox;

  if (!this->boundary->boundingBox(&outputBox)) {
    return false;
  }

  box->boundingBox = outputBox.boundingBox;
  return true;
}
