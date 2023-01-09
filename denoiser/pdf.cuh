#pragma once

#include "math/arr3.cuh"
#include "helper/helper.cuh"

__device__
Arr3 randomCosineDirection(curandState *randState) {
  auto r1 = randomFloat(randState);
  auto r2 = randomFloat(randState);
  auto z = sqrtf(1 - r2);

  auto phi = 2 * 3.141592653589f * r1;
  auto x = cosf(phi) * sqrtf(r2);
  auto y = sinf(phi) * sqrtf(r2);

  return Arr3(x, y, z);
}

class PDF {
  public:
    __host__ __device__ virtual float value(const Arr3 &direction) const = 0;
    __device__ virtual Arr3 generate(curandState *randState) const = 0;
};