#pragma once

#include "hittable.cuh"

class HittableList : public Hittable {
  public:
    __host__ __device__ HittableList() {}
    __host__ __device__ HittableList(Hittable **objects, int n) : objects{objects}, n{n} {}

    __device__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord *rec, MaterialRecord *mat, curandState* randState) const override;
    __host__ virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord *rec, MaterialRecord *mat) const override;

    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *outputBox) override;

  private:
    Hittable **objects;
    int n;
};

__device__
bool HittableList::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const {
  MaterialRecord tempMat;
  HitRecord tempHit;

  bool isHitAnything = false;
  float tClosest = tMax;

  for (int i = 0; i < this->n; i++) {
    if (this->objects[i]->hit(r, tMin, tClosest, &tempHit, &tempMat, randState)) {
      isHitAnything = true;
      tClosest = tempHit.t;

      hit->textCoord = tempHit.textCoord;
      hit->faceNormal = tempHit.faceNormal;
      hit->point = tempHit.point;
      hit->t = tempHit.t;

      mat->material = tempMat.material;
    }
  }

  return isHitAnything;
}

__host__
bool HittableList::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const {
  MaterialRecord tempMat;
  HitRecord tempHit;

  bool isHitAnything = false;
  float tClosest = tMax;

  for (int i = 0; i < this->n; i++) {
    if (this->objects[i]->hit(r, tMin, tClosest, &tempHit, &tempMat)) {
      isHitAnything = true;
      tClosest = tempHit.t;

      hit->textCoord = tempHit.textCoord;
      hit->faceNormal = tempHit.faceNormal;
      hit->point = tempHit.point;
      hit->t = tempHit.t;

      mat->material = tempMat.material;
    }
  }

  return isHitAnything;
}

__host__ __device__ 
bool HittableList::boundingBox(BoundingRecord *outputBox) {
  if (this->objects == NULL || this->n == 0) return false;

  bool firstBox = true;
  BoundingRecord tempBox;

  for (int i = 0; i < this->n; i++) {
    if (this->objects[n]->boundingBox(&tempBox)) {
      outputBox->boundingBox = firstBox ? tempBox.boundingBox : AABB::surrondingBox(outputBox->boundingBox, tempBox.boundingBox);
      firstBox = false;
    }
  }

  return true;
}

__host__ __device__ 
float HittableList::numCompare(int index) const {
  if (this->objects == NULL || this->n == 0) return -9999.0f;

  float minNum = 9999.0f;

  for (int i = 0; i < this->n; i++) {
    if (this->objects[i]->numCompare(index) < minNum) {
      minNum = this->objects[i]->numCompare(index);
    }
  }

  return minNum;
}