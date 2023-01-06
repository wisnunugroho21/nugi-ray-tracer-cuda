#pragma once

#include "ray.cuh"

struct FaceNormal {
	Arr3 normal;
	bool frontFace;

	__host__ __device__ FaceNormal() {}
	__host__ __device__ FaceNormal(const Ray &comingRay, const Arr3 &outwardNormal);
};

__host__ __device__
FaceNormal::FaceNormal(const Ray &comingRay, const Arr3 &outwardNormal) {
  this->frontFace = Arr3::dot(comingRay.direction(), outwardNormal) < 0;
  this->normal = this->frontFace ? outwardNormal : -outwardNormal;
}

struct TextureCoordinate {
  float u;
  float v;
};


struct HitRecord {
	FaceNormal faceNormal;
  TextureCoordinate textCoord;
	Arr3 point;
	float t;
};
