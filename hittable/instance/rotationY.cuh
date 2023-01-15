#pragma once

#include "hittable/hittable.cuh"
#include "helper/helper.cuh"

class RotationY : public Hittable {
  public:
    __host__ __device__ RotationY(Hittable *object, float angle);

    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const override;
    __host__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;

    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

  private:
    float sinTheta, cosTheta;
    bool hasBox; AABB box;
    Hittable *object;
};

__host__ __device__
RotationY::RotationY(Hittable *object, float angle) : object{object} {
  auto radians = degreesToRadians(angle);
  this->sinTheta = sinf(radians);
  this->cosTheta = cosf(radians);

  BoundingRecord boxRec;
  this->hasBox = this->object->boundingBox(&boxRec);
  this->box = boxRec.boundingBox;

  Arr3 min( 9999.0f, 9999.0f, 9999.0f );
  Arr3 max( -9999.0f, -9999.0f, -9999.0f );

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        auto x = i * this->box.maximum().x() + (1 - i) * this->box.minimum().x();
        auto y = j * this->box.maximum().y() + (1 - j) * this->box.minimum().y();
        auto z = k * this->box.maximum().z() + (1 - k) * this->box.minimum().z();

        auto newX = this->cosTheta * x + this->sinTheta * z;
        auto newZ = -1.0f * this->sinTheta * x + this->cosTheta * z;

        Arr3 tester(newX, y, newZ);

        for (int c = 0; c < 3; c++) {
          min[c] = fminf(min[c], tester[c]);
          max[c] = fmaxf(max[c], tester[c]);
        }
      }
    }
  }

  this->box = AABB(min, max);
}

__device__ 
bool RotationY::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const {
  auto origin = r.origin();
  auto direction = r.direction();

  origin[0] = this->cosTheta * r.origin()[0] - this->sinTheta * r.origin()[2];
  origin[2] = this->sinTheta * r.origin()[0] + this->cosTheta * r.origin()[2];

  direction[0] = this->cosTheta * r.direction()[0] - this->sinTheta * r.direction()[2];
  direction[2] = this->sinTheta * r.direction()[0] + this->cosTheta * r.direction()[2];

  Ray rotatedRay(origin, direction, r.time());

  if (!this->object->hit(rotatedRay, tMin, tMax, hit, mat, randState)) {
    return false;
  }

  if (hit != nullptr && hit != NULL) {
    auto p = hit->point;
    auto normal = hit->faceNormal.normal;

    p[0] =  this->cosTheta * hit->point[0] + this->sinTheta * hit->point[2];
    p[2] = -1.0f * this->sinTheta * hit->point[0] + this->cosTheta * hit->point[2];

    normal[0] =  this->cosTheta * hit->faceNormal.normal[0] + this->sinTheta * hit->faceNormal.normal[2];
    normal[2] = -1.0f * this->sinTheta * hit->faceNormal.normal[0] + this->cosTheta * hit->faceNormal.normal[2];

    hit->point = p;
    hit->faceNormal = FaceNormal(rotatedRay, normal);
  }

  return true;
}

__host__ 
bool RotationY::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const {
  auto origin = r.origin();
  auto direction = r.direction();

  origin[0] = this->cosTheta * r.origin()[0] - this->sinTheta * r.origin()[2];
  origin[2] = this->sinTheta * r.origin()[0] + this->cosTheta * r.origin()[2];

  direction[0] = this->cosTheta * r.direction()[0] - this->sinTheta * r.direction()[2];
  direction[2] = this->sinTheta * r.direction()[0] + this->cosTheta * r.direction()[2];

  Ray rotatedRay(origin, direction, r.time());

  if (!this->object->hit(rotatedRay, tMin, tMax, hit, mat)) {
    return false;
  }

  if (hit != nullptr && hit != NULL) {
    auto p = hit->point;
    auto normal = hit->faceNormal.normal;

    p[0] =  this->cosTheta * hit->point[0] + this->sinTheta * hit->point[2];
    p[2] = -1.0f * this->sinTheta * hit->point[0] + this->cosTheta * hit->point[2];

    normal[0] =  this->cosTheta * hit->faceNormal.normal[0] + this->sinTheta * hit->faceNormal.normal[2];
    normal[2] = -1.0f * this->sinTheta * hit->faceNormal.normal[0] + this->cosTheta * hit->faceNormal.normal[2];

    hit->point = p;
    hit->faceNormal = FaceNormal(rotatedRay, normal);
  }

  return true;
}

__host__ __device__ 
float RotationY::numCompare(int index) const {
  return this->box.minimum().get(index);
}

__host__ __device__ 
bool RotationY::boundingBox(BoundingRecord *box) {
  if (box != nullptr && box != NULL) {
    box->boundingBox = this->box;
  }
  
  return this->hasBox;
}
