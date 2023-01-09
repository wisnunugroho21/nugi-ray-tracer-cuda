#pragma once

#include "pdf.cuh"
#include "onb.cuh"

class CosinePdf : public PDF {
  public:
    __host__ __device__ CosinePdf() {}
    __host__ __device__ CosinePdf(const Arr3 &w) { uvw.buildFromW(w); }

    __host__ __device__ virtual float value(const Arr3 &direction) const override;
    __device__ virtual Arr3 generate(curandState *randState) const override;

  private:
    ONB uvw;
};

__host__ __device__
float CosinePdf::value(const Arr3 &direction) const {
  auto cosine = Arr3::dot(direction.unitVector(), this->uvw.w());
  return (cosine <= 0) ? 0 : cosine / 3.141592653589f;
}

__device__
Arr3 CosinePdf::generate(curandState *randState) const {
  return uvw.local(randomCosineDirection(randState));
}