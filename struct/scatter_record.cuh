#pragma once

#include "utility/ray.cuh"
#include "denoiser/cosinePdf.cuh"

struct Specular {
  bool isSpecular;
  Ray ray;
};

struct ScatterRecord {
	Arr3 colorAttenuation;
	Specular specular;
  CosinePdf pdf;

  __host__ __device__ ScatterRecord() {}
};
