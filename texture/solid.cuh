#pragma once

#include "texture.cuh"

class Solid : public Texture {
  public:
    __host__ __device__ Solid() {}
    __host__ __device__ Solid(Arr3 color) : color{color} {}
    __host__ __device__ Solid(float red, float green, float blue) : color{Arr3(red, green, blue)} {}

    __host__ __device__ virtual Arr3 map(float u, float v, const Arr3 &point) const override;
  
  private:
    Arr3 color;
};

__host__ __device__
Arr3 Solid::map(float u, float v, const Arr3 &point) const {
  return this->color;
}