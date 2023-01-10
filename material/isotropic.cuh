#pragma once

#include "material.cuh"
#include "texture/solid.cuh"
#include "helper/helper.cuh"

class Isotropic :public Material {
  public:
    __host__ __device__ Isotropic(Texture *texture) : texture{texture} {}

    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const override;

  private:
    Texture *texture;
};

__device__ 
bool Isotropic::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const {
  scattered->newRay = Ray(hit.point, Arr3::randomInUnitSphere(randState), ray.time());
  scattered->colorAttenuation = this->texture->map(hit.textCoord.u, hit.textCoord.v, hit.point);

  return true;
}
