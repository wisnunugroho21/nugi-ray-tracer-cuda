#pragma once

#include "../hittable.cuh"
#include "../../material/material.cuh"

class Sphere : public Hittable {
  public:
    __device__ Sphere(Arr3 center, float radius, Material *material) : center{center}, radius{radius}, material{material} {}

    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;

  private:
    Arr3 center;
    float radius;
    Material *material;
};

__device__
bool Sphere::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const { 
	Arr3 oc = r.origin() - this->center;

	auto a = r.direction().lengthSquared();
	auto half_b = Arr3::dot(oc, r.direction());
	auto c = oc.lengthSquared() - this->radius * this->radius;

	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0.0f) return false;

	auto sqrtDiscrim = sqrtf(discriminant);

	auto root = (-half_b - sqrtf(discriminant)) / a;
	if (root < tMin || root > tMax) {
		root = (-half_b + sqrtf(discriminant)) / a;
		if (root < tMin || root > tMax) {
			return false;
		}
	}

	hit->t = root;
	hit->point = r.at(root);

	Arr3 outwardNormal = (hit->point - this->center) / this->radius;
	hit->faceNormal = FaceNormal(r, outwardNormal);

	mat->material = this->material;
	return true;
}