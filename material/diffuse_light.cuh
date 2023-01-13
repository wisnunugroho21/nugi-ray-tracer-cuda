#pragma once

#include "material.cuh"
#include "texture/texture.cuh"
#include "texture/solid.cuh"

class DiffuseLight : public Material {
  public:
    __host__ __device__ DiffuseLight(Texture *texture) : texture{texture} {}
    __host__ __device__ DiffuseLight(Arr3 color) : texture{new Solid(color)} {}
    
    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const override;
    __host__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const override;

    __host__ __device__ virtual Arr3 emitted(float u, float v, const Arr3 &point) const override;
    __host__ virtual Material* copyToDevice();

  public:
    Texture *texture;
};

__device__ 
bool DiffuseLight::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const {
  return false;
}

__host__ 
bool DiffuseLight::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const {
  return false;
}

__host__ __device__ 
Arr3 DiffuseLight::emitted(float u, float v, const Arr3 &point) const {
  return this->texture->map(u, v, point);
}

__host__ 
Material* DiffuseLight::copyToDevice() {
  Texture *cudaTxt = this->texture->copyToDevice();
  this->texture = cudaTxt;

  DiffuseLight *cudaMat;

  cudaMalloc((void**) &cudaMat, sizeof(*this));
  cudaMemcpy(cudaMat, this, sizeof(*this), cudaMemcpyHostToDevice);

  return cudaMat;
}