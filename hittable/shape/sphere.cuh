#pragma once

#include "hittable/hittable.cuh"
#include "material/material.cuh"

#include <limits>

class Sphere : public Hittable {
  public:
    __host__ __device__ Sphere(Arr3 center, float radius, Material *material) : center{center}, radius{radius}, material{material} {}

    __host__ __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;
    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

    __host__ __device__ virtual float pdfValue(const Arr3 &origin, const Arr3 &direction) const override;
    __device__ virtual Arr3 random(const Arr3 &origin, curandState* randState) const override;

    __host__ __device__ static TextureCoordinate getUV(const Arr3 &point);

  private:
    Arr3 center;
    float radius;
    Material *material;
};

__host__ __device__
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

  if (hit != nullptr && hit != NULL) {
    hit->t = root;
    hit->point = r.at(root);

    Arr3 outwardNormal = (hit->point - this->center) / this->radius;
    hit->faceNormal = FaceNormal(r, outwardNormal);

    hit->textCoord = Sphere::getUV(outwardNormal);
  }

	if (mat != nullptr && mat != NULL) {
    mat->material = this->material;
  }

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
float Sphere::pdfValue(const Arr3 &origin, const Arr3 &direction) const {
  HitRecord hit;

  if (!this->hit(Ray(origin, direction), 0.001f, FLT_MAX, &hit, nullptr)) {
    return 0.0f;
  }

  auto cosThetaMax = sqrtf(1.0f - this->radius * this->radius / (this->center - origin).lengthSquared());
  auto solidAngle = 2.0f * 3.1415926535897932385f * (1.0f - cosThetaMax);

  return 1.0f / solidAngle;
}

__device__ 
Arr3 Sphere::random(const Arr3 &origin, curandState* randState) const {
  Arr3 direction = this->center - origin;
  auto distanceSquared = direction.lengthSquared();
  
  ONB uvw;
  uvw.buildFromW(direction);
  return uvw.local(Arr3::randomToSphere(this->radius, distanceSquared, randState));
}

__host__ __device__ 
TextureCoordinate Sphere::getUV(const Arr3 &point) {
  float pi = 3.1415926535897932385f;

  auto theta = acosf(-1.0f * point.y());
  auto phi = atan2f(-1.0f * point.z(), point.x()) + pi;

  TextureCoordinate textCoord;
  textCoord.u = phi / (2 * pi);
  textCoord.v = theta / pi;

  return textCoord;
}