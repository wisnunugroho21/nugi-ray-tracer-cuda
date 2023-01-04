#pragma once

#include "arr3.cuh"

class Texture {
  public:
    __host__ __device__ virtual Arr3 map(float u, float v, const Arr3 &point) const = 0;
};