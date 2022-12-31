#pragma once

#include "../hittable.hpp"
#include "../../material/material.hpp"

class Sphere : public Hittable {
  public:
    Sphere(Arr3 center, float radius, Material *material) : center{center}, radius{radius}, material{material} {}

    virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;

  private:
    Arr3 center;
    float radius;
    Material *material;
};

bool Sphere::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const { 
  Arr3 oc = r.origin() - center;

	auto a = r.direction().lengthSquared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.lengthSquared() - radius * radius;

	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0.0f) return false;

	auto sqrtDiscrim = sqrt(discriminant);

	auto root = (-half_b - sqrt(discriminant)) / a;
	if (root < tMin || root > tMax) {
		root = (-half_b + sqrt(discriminant)) / a;
		if (root < tMin || root > tMax) {
			return false;
		}
	}

	hit->t = root;
	hit->point = r.at(root);

	Arr3 outwardNormal = (hit->point - center) / radius;
	hit->faceNormal = FaceNormal(r, outwardNormal);

	mat->material = this->material;
	return true;
}