#pragma once

#include "helper/helper.cuh"
#include "hittable/hittable.cuh"

class XZRect : public Hittable {
  public:
    __host__ __device__ XZRect() {}
    __host__ __device__ XZRect(float x0, float x1, float z0, float z1, float k, Material *material) : x0{x0}, x1{x1}, z0{z0}, z1{z1}, k{k}, material{material} {}

    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const override;
    __host__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;

    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

  private:
    float x0, x1, z0, z1, k;
    Material *material;
};

__device__ 
bool XZRect::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const {
  auto t = (k - r.origin().y()) / r.direction().y();
  if (t < tMin || t > tMax) {
    return false;
  }

  auto x = r.origin().x() + t * r.direction().x();
  auto z = r.origin().z() + t * r.direction().z();

  if (x < this->x0 || x > this->x1 || z < this->z0 || z > this->z1) {
    return false;
  }

  if (hit != nullptr && hit != NULL) {
    hit->t = t;
    hit->point = r.at(t);

    hit->textCoord.u = (x - this->x0) / (this->x1 - this->x0);
    hit->textCoord.v = (z - this->z0) / (this->z1 - this->z0);

    auto outwardNormal = Arr3(0.0f, 1.0f, 0.0f);
    hit->faceNormal = FaceNormal(r, outwardNormal);
  }

  if (mat != nullptr && mat != NULL) {
    mat->material = this->material;
  }
  
  return true;
}

__host__ 
bool XZRect::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const {
  auto t = (k - r.origin().y()) / r.direction().y();
  if (t < tMin || t > tMax) {
    return false;
  }

  auto x = r.origin().x() + t * r.direction().x();
  auto z = r.origin().z() + t * r.direction().z();

  if (x < this->x0 || x > this->x1 || z < this->z0 || z > this->z1) {
    return false;
  }

  if (hit != nullptr && hit != NULL) {
    hit->t = t;
    hit->point = r.at(t);

    hit->textCoord.u = (x - this->x0) / (this->x1 - this->x0);
    hit->textCoord.v = (z - this->z0) / (this->z1 - this->z0);

    auto outwardNormal = Arr3(0.0f, 1.0f, 0.0f);
    hit->faceNormal = FaceNormal(r, outwardNormal);
  }

  if (mat != nullptr && mat != NULL) {
    mat->material = this->material;
  }

  return true;
}

__host__ __device__ 
float XZRect::numCompare(int index) const {
  return Arr3(this->x0, this->k - 0.0001f, this->z0).get(index);
}

__host__ __device__ 
bool XZRect::boundingBox(BoundingRecord *box) {
  if (box != nullptr && box != NULL) {
    box->boundingBox = AABB(Arr3(this->x0, this->k - 0.0001f, this->z0), Arr3(this->x1, this->k + 0.0001f, this->z1));
  }
  
  return true;
}
