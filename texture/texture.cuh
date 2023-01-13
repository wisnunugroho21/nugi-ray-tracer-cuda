#pragma once

#include "math/arr3.cuh"

class Texture {
  public:
    __host__ __device__ virtual Arr3 map(float u, float v, const Arr3 &point) const = 0;
    __host__ virtual Texture* copyToDevice();
};

__host__ 
Texture* Texture::copyToDevice() {
  Texture *cudaTxt;

  cudaMalloc((void**) &cudaTxt, sizeof(*this));
  cudaMemcpy(cudaTxt, this, sizeof(*this), cudaMemcpyHostToDevice);

  return cudaTxt;
}