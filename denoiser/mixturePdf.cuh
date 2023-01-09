#pragma once

#include "pdf.cuh"
#include "onb.cuh"
#include "hittable/hittable.cuh"

#include "cosinePdf.cuh"
#include "hittablePdf.cuh"

class MixturePdf : public PDF {
  public:
    __host__ __device__
    MixturePdf(const CosinePdf &cosinePDf, const HittablePdf &hittablepdf) : cosinePDf{cosinePDf}, hittablepdf{hittablepdf} {}

    __host__ __device__ virtual float value(const Arr3 &direction) const override;
    __device__ virtual Arr3 generate(curandState *randState) const override;

  private:
    CosinePdf cosinePDf;
    HittablePdf hittablepdf;
};

__host__ __device__
float MixturePdf::value(const Arr3 &direction) const {
  return 0.5 * this->cosinePDf.value(direction) + 0.5 * this->hittablepdf.value(direction);
}

__device__
Arr3 MixturePdf::generate(curandState *randState) const {
  if (randomFloat(randState) < 0.5) {
    return this->cosinePDf.generate(randState);
  }
  
  return this->hittablepdf.generate(randState);
}