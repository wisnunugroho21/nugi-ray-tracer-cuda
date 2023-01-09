#pragma once

#include "pdf.cuh"
#include "onb.cuh"
#include "hittable/hittable.cuh"

class HittablePdf : public PDF {
  public:
    __host__ __device__ HittablePdf(Hittable **object, const Arr3 &origin) : object(object), origin(origin) {}

    __host__ __device__ virtual float value(const Arr3 &direction) const override;
    __device__ virtual Arr3 generate(curandState *randState) const override;

  private:
    Arr3 origin;
    Hittable **object;
};

__host__ __device__
float HittablePdf::value(const Arr3 &direction) const {
  return this->object[0]->pdfValue(this->origin, direction);
}

__device__
Arr3 HittablePdf::generate(curandState *randState) const {
  return this->object[0]->random(this->origin, randState);
}