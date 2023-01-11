#pragma once

#include "hittable/hittable.cuh"
#include "material/material.cuh"

class Sphere : public Hittable {
  public:
    __host__ __device__ Sphere(Arr3 center, float radius, Material *material) : center{center}, radius{radius}, material{material} {}

    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const override;
    __host__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;

    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

    __host__ __device__ static TextureCoordinate getUV(const Arr3 &point);

  private:
    Arr3 center;
    float radius;
    Material *material;
};

__device__
bool Sphere::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const { 
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

  hit->textCoord = Sphere::getUV(outwardNormal);
	mat->material = this->material;

	return true;
}

__host__
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

  hit->textCoord = Sphere::getUV(outwardNormal);
	mat->material = this->material;

	return true;
}

__host__ __device__
bool Sphere::boundingBox(BoundingRecord *box) {
  box->boundingBox = AABB(
    this->center - Arr3(this->radius, this->radius, this->radius),
    this->center + Arr3(this->radius, this->radius, this->radius)
  );

  return true;
}

__host__ __device__ 
float Sphere::numCompare(int index) const {
  return (this->center - Arr3(this->radius, this->radius, this->radius)).get(index);
}

__host__ __device__ 
TextureCoordinate Sphere::getUV(const Arr3 &point) {
  float pi = 3.1415926535897932385f;

  auto theta = acosf(-1.0f * point.y());
  auto phi = atan2f(-1.0f * point.z(), point.x()) + pi;

  TextureCoordinate textCoord;
  textCoord.u = phi / (2.0f * pi);
  textCoord.v = theta / pi;

  return textCoord;
}