#pragma once

#include "texture.cuh"
#include "utility/perlin.cuh"

class Noise : public Texture {
  public:
    __device__ Noise(curandState *randState) : perlin{Perlin(randState)} {}
    __device__ Noise(curandState *randState, float scale) : perlin{Perlin(randState)}, scale{scale} {}

    __host__ __device__ virtual Arr3 map(float u, float v, const Arr3 &point) const override;

  private:
    Perlin perlin;
    float scale;
};

__host__ __device__
Arr3 Noise::map(float u, float v, const Arr3 &point) const {
  return Arr3(1.0f, 1.0f, 1.0f) * 0.5 * (1.0f + sin(scale * point.z() + 10.f * this->perlin.turbulence(point)));
}