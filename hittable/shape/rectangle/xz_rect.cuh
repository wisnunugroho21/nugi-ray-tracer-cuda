#pragma once

#include "helper/helper.cuh"
#include "hittable/hittable.cuh"

#include <limits>

class XZRect : public Hittable {
  public:
    __host__ __device__ XZRect() {}
    __host__ __device__ XZRect(float x0, float x1, float z0, float z1, float k, Material *material) : x0{x0}, x1{x1}, z0{z0}, z1{z1}, k{k}, material{material} {}

    __host__ __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;
    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

    __host__ __device__ virtual float pdfValue(const Arr3 &origin, const Arr3 &direction) const override;
    __device__ virtual Arr3 random(const Arr3 &origin, curandState* randState) const override;

  private:
    float x0, x1, z0, z1, k;
    Material *material;
};

__host__ __device__ 
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
  box->boundingBox = AABB(Arr3(this->x0, this->k - 0.0001f, this->z0), Arr3(this->x1, this->k + 0.0001f, this->z1));
  return true;
}

__host__ __device__ 
float XZRect::pdfValue(const Arr3 &origin, const Arr3 &direction) const {
  HitRecord hit;
  MaterialRecord mat;
  if (!this->hit(Ray(origin, direction), 0.001, FLT_MAX, &hit, &mat)) {
    return 0;
  } 

  auto area = (this->x1 - this->x0) * (this->z1 - this->z0);
  auto distanceSquared = hit.t * hit.t * direction.lengthSquared();
  auto cosine = fabsf(Arr3::dot(direction, hit.faceNormal.normal) / direction.length());

  return distanceSquared / (cosine * area);
}

__device__
Arr3 XZRect::random(const Arr3 &origin, curandState* randState) const {
  auto random_point = Arr3(randomFloat(x0,x1, randState), k, randomFloat(z0, z1, randState));
  return random_point - origin;
}
