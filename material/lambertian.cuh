#pragma once

#include "material.cuh"
#include "texture/solid.cuh"

class Lambertian : public Material {
  public:
    __host__ __device__ Lambertian(const Arr3 &color) : texture{new Solid(color)} {}
    __host__ __device__ Lambertian(Texture *texture) : texture{texture} {}

    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const override;
    __host__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const override;

  private:
    Texture *texture;
};

__device__
bool Lambertian::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const {
	auto scatterDirection = hit.faceNormal.normal + Arr3::randomInUnitSphere(randState).unitVector();

  if (scatterDirection.nearZero()) {
    scatterDirection = hit.faceNormal.normal;
  }

  if (scattered != nullptr && scattered != NULL) {
    scattered->newRay = Ray(hit.point, scatterDirection, ray.time());
	  scattered->colorAttenuation = this->texture->map(hit.textCoord.u, hit.textCoord.v, hit.point);
  }

	return true;
}

__host__ 
bool Lambertian::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const {
  auto scatterDirection = hit.faceNormal.normal + Arr3::randomInUnitSphere().unitVector();

  if (scatterDirection.nearZero()) {
    scatterDirection = hit.faceNormal.normal;
  }

  if (scattered != nullptr && scattered != NULL) { 
    scattered->newRay = Ray(hit.point, scatterDirection, ray.time());
	  scattered->colorAttenuation = this->texture->map(hit.textCoord.u, hit.textCoord.v, hit.point);
  }

	return true;
}
