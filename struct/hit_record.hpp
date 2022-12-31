#pragma once

#include "../ray.hpp"

struct FaceNormal {
  Arr3 normal;
  bool frontFace;

  FaceNormal() {}
  FaceNormal(const Ray &comingRay, const Arr3 &outwardNormal);
};

FaceNormal::FaceNormal(const Ray &comingRay, const Arr3 &outwardNormal) {
  this->frontFace = dot(comingRay.direction(), outwardNormal);
  this->normal = this->frontFace ? outwardNormal : -outwardNormal;
}

struct HitRecord {
  FaceNormal faceNormal;
  Arr3 point;
  float t;
};
