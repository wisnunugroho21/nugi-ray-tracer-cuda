#pragma once

#include "hittable/hittable.cuh"
#include "material/material.cuh"

class MovingSphere : public Hittable {
  public:
    __host__ __device__ 
    MovingSphere(Arr3 center0, float time0, Arr3 center1, float time1, float radius, Material *material) 
    : center0{center0}, center1{center1}, time0{time0}, time1{time1}, radius{radius}, material{material} 
    {}

    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const override;
    __host__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;

    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

    __host__ __device__ Arr3 center(float time) const;

  private:
    Arr3 center0, center1;
    float time0, time1;
    float radius;
    Material *material;
};

__device__
bool MovingSphere::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const { 
	Arr3 oc = r.origin() - this->center(r.time());

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

	Arr3 outwardNormal = (hit->point - this->center(r.time())) / this->radius;
	hit->faceNormal = FaceNormal(r, outwardNormal);

	mat->material = this->material;
	return true;
}

__host__
bool MovingSphere::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const { 
	Arr3 oc = r.origin() - this->center(r.time());

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

	Arr3 outwardNormal = (hit->point - this->center(r.time())) / this->radius;
	hit->faceNormal = FaceNormal(r, outwardNormal);

	mat->material = this->material;
	return true;
}

__host__ __device__
bool MovingSphere::boundingBox(BoundingRecord *box) {
  AABB box0(
    this->center(this->time0) - Arr3(this->radius, this->radius, this->radius),
    this->center(this->time0) + Arr3(this->radius, this->radius, this->radius)
  );

  AABB box1(
    this->center(this->time1) - Arr3(this->radius, this->radius, this->radius),
    this->center(this->time1) + Arr3(this->radius, this->radius, this->radius)
  );

  box->boundingBox = AABB::surrondingBox(box0, box1);
  return true;
}

__host__ __device__ 
float MovingSphere::numCompare(int index) const {
  Arr3 min0 = this->center(this->time0) - Arr3(this->radius, this->radius, this->radius);
  Arr3 min1 = this->center(this->time1) - Arr3(this->radius, this->radius, this->radius);

  return (min0.get(index) < min1.get(index)) ? min0.get(index) : min1.get(index);
}

__host__ __device__ 
Arr3 MovingSphere::center(float time) const {
  return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}