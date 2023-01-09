#pragma once

#include "helper/helper.cuh"
#include "hittable/hittable.cuh"

class XYRect : public Hittable {
  public:
    __host__ __device__ XYRect() {}
    __host__ __device__ XYRect(float x0, float x1, float y0, float y1, float k, Material *material) : x0{x0}, x1{x1}, y0{y0}, y1{y1}, k{k}, material{material} {}

    __host__ __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;
    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

  private:
    float x0, x1, y0, y1, k;
    Material *material;
};

__host__ __device__ 
bool XYRect::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const {
  auto t = (k - r.origin().z()) / r.direction().z();
  if (t < tMin || t > tMax) {
    return false;
  }

  auto x = r.origin().x() + t * r.direction().x();
  auto y = r.origin().y() + t * r.direction().y();

  if (x < this->x0 || x > this->x1 || y < this->y0 || y > this->y1) {
    return false;
  }

  if (hit != nullptr && hit != NULL) {
    hit->t = t;
    hit->point = r.at(t);

    hit->textCoord.u = (x - this->x0) / (this->x1 - this->x0);
    hit->textCoord.v = (y - this->y0) / (this->y1 - this->y0);

    auto outwardNormal = Arr3(0.0f, 0.0f, 1.0f);
    hit->faceNormal = FaceNormal(r, outwardNormal);
  }

  if (mat != nullptr && mat != NULL) {
    mat->material = this->material;
  }
    
  return true;
}

__host__ __device__ 
float XYRect::numCompare(int index) const {
  return Arr3(this->x0, this->y0, this->k - 0.0001f).get(index);
}

__host__ __device__ 
bool XYRect::boundingBox(BoundingRecord *box) {
  box->boundingBox = AABB(Arr3(this->x0, this->y0, this->k - 0.0001f), Arr3(this->x1, this->y1, this->k + 0.0001f));
  return true;
}
