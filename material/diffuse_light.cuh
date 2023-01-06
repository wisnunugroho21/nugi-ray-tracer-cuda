#pragma once

#include "material.cuh"
#include "../texture/texture.cuh"

class DiffuseLight : public Material {
  public:
    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const override;
    __host__ __device__ virtual Arr3 emitted(float u, float v, const Arr3 &point) const override;

  private:
    Texture *emit;
};

__device__ 
bool DiffuseLight::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const {
  return false;
}

__host__ __device__ 
Arr3 DiffuseLight::emitted(float u, float v, const Arr3 &point) const {
  return this->emit->map(u, v, point);
}