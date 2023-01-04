#pragma once

#include "hittable.cuh"

class HittableList : public Hittable {
  public:
    __host__ __device__ HittableList() {}
    __host__ __device__ HittableList(Hittable **objects, int n) : objects{objects}, n{n} {}

    __host__ __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord *rec, MaterialRecord *mat) const override;
    __host__ __device__ virtual float getNumCompare(int index) const override;
    __host__ __device__ virtual bool buildBoundingBox(BoundingRecord *outputBox) override;

  private:
    Hittable **objects;
    int n;
};

__host__ __device__
bool HittableList::hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat) const {
  MaterialRecord tempMat;
  HitRecord tempRec;

  bool isHitAnything = false;
  float tClosest = tMax;

  for (size_t i = 0; i < this->n; i++) {
    if (this->objects[i]->hit(r, tMin, tClosest, &tempRec, &tempMat)) {
      isHitAnything = true;
      tClosest = tempRec.t;

      rec->faceNormal = tempRec.faceNormal;
      rec->point = tempRec.point;
      rec->t = tempRec.t;

      mat->material = tempMat.material;
    }
  }

  return isHitAnything;
}

__host__ __device__ 
bool HittableList::buildBoundingBox(BoundingRecord *outputBox) {
  if (this->objects == NULL || this->n == 0) return false;

  bool firstBox = true;
  BoundingRecord tempBox;

  for (int i = 0; i < this->n; i++) {
    if (this->objects[n]->buildBoundingBox(&tempBox)) {
      outputBox->boundingBox = firstBox ? tempBox.boundingBox : AABB::surrondingBox(outputBox->boundingBox, tempBox.boundingBox);
      firstBox = false;
    }
  }

  return true;
}

__host__ __device__ 
float HittableList::getNumCompare(int index) const {
  if (this->objects == NULL || this->n == 0) return -99;

  float minNum = 99;

  for (int i = 0; i < this->n; i++) {
    if (this->objects[i]->getNumCompare(index) < minNum) {
      minNum = this->objects[i]->getNumCompare(index);
    }
  }

  return minNum;
}