#pragma once

#include "material.cuh"
#include "texture/solid.cuh"
#include "helper/helper.cuh"

class Isotropic :public Material {
  public:
    __host__ __device__ Isotropic(Texture *texture) : texture{texture} {}

    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const override;
    __host__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const override;
    __host__ virtual Material* copyToDevice() override;

  public:
    Texture *texture;
};

__device__ 
bool Isotropic::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const {
  if (scattered != nullptr && scattered != NULL) {
    scattered->newRay = Ray(hit.point, Arr3::randomInUnitSphere(randState), ray.time());
    scattered->colorAttenuation = this->texture->map(hit.textCoord.u, hit.textCoord.v, hit.point);
  }

  return true;
}

__host__ 
bool Isotropic::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const {
  if (scattered != nullptr && scattered != NULL) {
    scattered->newRay = Ray(hit.point, Arr3::randomInUnitSphere(), ray.time());
    scattered->colorAttenuation = this->texture->map(hit.textCoord.u, hit.textCoord.v, hit.point);
  }

  return true;
}

__host__ 
Material* Isotropic::copyToDevice() {
  Texture *cudaTxt = this->texture->copyToDevice();
  this->texture = cudaTxt;

  Isotropic *cudaMat;

  cudaMalloc((void**) &cudaMat, sizeof(*this));
  cudaMemcpy(cudaMat, this, sizeof(*this), cudaMemcpyHostToDevice);

  return cudaMat;
}
