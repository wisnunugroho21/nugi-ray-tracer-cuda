#pragma once

#include "hittable.hpp"

class HittableList : public Hittable {
  public:
    HittableList() {}
    HittableList(Hittable **objects, int n) : objects{objects}, n{n} {}

    virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord *rec, MaterialRecord *mat) const override;

  private:
    Hittable **objects;
    int n;
};

bool HittableList::hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat) const {
  MaterialRecord *tempMat = new MaterialRecord();
  HitRecord *tempRec = new HitRecord();

  bool isHitAnything = false;
  float tClosest = tMax;

  for (size_t i = 0; i < this->n; i++) {
    if (this->objects[i]->hit(r, tMin, tClosest, tempRec, tempMat)) {
      isHitAnything = true;
      tClosest = tempRec->t;

      rec->faceNormal = tempRec->faceNormal;
      rec->point = tempRec->point;
      rec->t = tempRec->t;

      mat->material = tempMat->material;
    }
  }

  return isHitAnything;
}