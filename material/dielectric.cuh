#pragma once

#include "material.cuh"

class Dielectric : public Material {
	public:
		__host__ __device__ Dielectric(float indexRefraction) : indexRefraction{indexRefraction} {}

		__device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const override;
    __host__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const override;

		__host__ __device__ static float reflactance(float cosine, float refIdx);

	private:
		float indexRefraction;
};

__device__
bool Dielectric::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const {
	float refractionRatio = hit.faceNormal.frontFace ? (1.0f / this->indexRefraction) : this->indexRefraction;

	Arr3 unitDirection = ray.direction().unitVector();
	float cosTheta = fminf(Arr3::dot(-unitDirection, hit.faceNormal.normal), 1.0f);
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

	bool cannotRefract = refractionRatio * sinTheta > 1.0f;
	Arr3 direction;

	if (cannotRefract || Dielectric::reflactance(cosTheta, refractionRatio) > randomFloat(randState)) {
		direction = Arr3::reflect(unitDirection, hit.faceNormal.normal);
	} else {
		direction = Arr3::refract(unitDirection, hit.faceNormal.normal, refractionRatio);
	}	

	if (scattered != nullptr && scattered != NULL) {
    scattered->colorAttenuation = Arr3(1.0f, 1.0f, 1.0f);
	  scattered->newRay = Ray(hit.point, direction, ray.time());
  }

	return true;
}

__host__
bool Dielectric::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const {
	float refractionRatio = hit.faceNormal.frontFace ? (1.0f / this->indexRefraction) : this->indexRefraction;

	Arr3 unitDirection = ray.direction().unitVector();
	float cosTheta = fminf(Arr3::dot(-unitDirection, hit.faceNormal.normal), 1.0f);
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

	bool cannotRefract = refractionRatio * sinTheta > 1.0f;
	Arr3 direction;

	if (cannotRefract || Dielectric::reflactance(cosTheta, refractionRatio) > randomFloat()) {
		direction = Arr3::reflect(unitDirection, hit.faceNormal.normal);
	} else {
		direction = Arr3::refract(unitDirection, hit.faceNormal.normal, refractionRatio);
	}

  if (scattered != nullptr && scattered != NULL) {
    scattered->colorAttenuation = Arr3(1.0f, 1.0f, 1.0f);
	  scattered->newRay = Ray(hit.point, direction, ray.time());
  }

	return true;
}

__host__ __device__
float Dielectric::reflactance(float cosine, float refIdx) {
	auto r0 = (1.0f - refIdx) / (1.0f + refIdx);
	r0 = r0 * r0;

	return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}
