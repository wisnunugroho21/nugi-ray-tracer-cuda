#pragma once

#include "helper/helper.cuh"
#include "hittable/hittable.cuh"

class YZRect : public Hittable {
  public:
    __host__ __device__ YZRect() {}
    __host__ __device__ YZRect(float y0, float y1, float z0, float z1, float k, Material *material) : y0{y0}, y1{y1}, z0{z0}, z1{z1}, k{k}, material{material} {}

    __host__ __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;
    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

  private:
    float y0, y1, z0, z1, k;
    Material *material;
};

__host__ __device__ 
bool YZRect::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const {
  auto t = (k - r.origin().x()) / r.direction().x();
  if (t < tMin || t > tMax) {
    return false;
  }

  auto y = r.origin().y() + t * r.direction().y();
  auto z = r.origin().z() + t * r.direction().z();

  if (y < this->y0 || y > this->y1 || z < this->z0 || z > this->z1) {
    return false;
  }

  hit->t = t;
  hit->point = r.at(t);

  hit->textCoord.u = (y - this->y0) / (this->y1 - this->y0);
  hit->textCoord.v = (z - this->z0) / (this->z1 - this->z0);

  auto outwardNormal = Arr3(1.0f, 0.0f, 0.0f);
  hit->faceNormal = FaceNormal(r, outwardNormal);

  mat->material = this->material;
  return true;
}

__host__ __device__ 
float YZRect::numCompare(int index) const {
  return Arr3(this->k - 0.0001f, this->y0, this->z0).get(index);
}

__host__ __device__ 
bool YZRect::boundingBox(BoundingRecord *box) {
  box->boundingBox = AABB(Arr3(this->k - 0.0001f, this->y0, this->z0), Arr3(this->k + 0.0001f, this->y1, this->z1));
  return true;
}
