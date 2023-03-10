#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "library/stb_image.h"

__host__
unsigned char* loadImageToCUDA(const char *filename, int *width, int *height) {
  int components_per_pixel = 3;

  unsigned char *dataCPU = stbi_load(filename, width, height, &components_per_pixel, components_per_pixel);

  unsigned char *dataCUDA;
  cudaMalloc((void**) &dataCUDA, (*width) * (*height) * components_per_pixel * sizeof(unsigned char));
  cudaMemcpy(dataCUDA, dataCPU, (*width) * (*height) * components_per_pixel * sizeof(unsigned char), cudaMemcpyHostToDevice);

  return dataCUDA;
}