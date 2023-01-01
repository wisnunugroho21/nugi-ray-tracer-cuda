#pragma once

#include "material.cuh"

class Lambertian : public Material {
  public:
    __device__ Lambertian(const Arr3 &colorAlbedo) : colorAlbedo{colorAlbedo} {}
    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const override;

  private:
    Arr3 colorAlbedo;
};

__device__
bool Lambertian::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const {
	auto scatterDirection = hit.faceNormal.normal + Arr3::randomInUnitSphere(randState).unitVector();

  if (scatterDirection.nearZero())
    scatterDirection = hit.faceNormal.normal;

	scattered->newRay = Ray(hit.point, scatterDirection);
	scattered->colorAttenuation = this->colorAlbedo;

	return true;
}
