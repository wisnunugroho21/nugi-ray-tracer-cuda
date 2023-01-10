#pragma once

#include "solid.cuh"

class Checker : public Texture {
  public:
    __host__ __device__ Checker() {}
    __host__ __device__ Checker(Texture* even, Texture* odd) : even{even}, odd{odd} {}
    __host__ __device__ Checker(const Arr3 &evenColor, const Arr3 &oddColor) : even{new Solid(evenColor)}, odd{new Solid{oddColor}} {}

  __host__ __device__ virtual Arr3 map(float u, float v, const Arr3 &point) const override;

  private:
    Texture *even;
    Texture *odd;
};

__host__ __device__
Arr3 Checker::map(float u, float v, const Arr3 &point) const {
  auto sines = sinf(10.0f * point.x()) * sinf(10.0f * point.y()) * sinf(10.0f * point.z());

  if (sines < 0.0f) {
    return this->odd->map(u, v, point);
  }

  return this->even->map(u, v, point);
}
