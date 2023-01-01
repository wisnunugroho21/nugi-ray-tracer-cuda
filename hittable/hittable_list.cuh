#pragma once

#include "hittable.cuh"

class HittableList : public Hittable {
  public:
    __device__ HittableList() {}
    __device__ HittableList(Hittable **objects, int n) : objects{objects}, n{n} {}

    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec, MaterialRecord &mat) const override;

  private:
    Hittable **objects;
    int n;
};

__device__
bool HittableList::hit(const Ray &r, float tMin, float tMax, HitRecord &rec, MaterialRecord &mat) const {
  MaterialRecord tempMat;
  HitRecord tempRec;

  bool isHitAnything = false;
  float tClosest = tMax;

  for (size_t i = 0; i < this->n; i++) {
    if (this->objects[i]->hit(r, tMin, tClosest, tempRec, tempMat)) {
      isHitAnything = true;
      tClosest = tempRec.t;

      rec.faceNormal = tempRec.faceNormal;
      rec.point = tempRec.point;
      rec.t = tempRec.t;

      mat.material = tempMat.material;
    }
  }

  return isHitAnything;
}