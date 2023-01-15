#pragma once

#include "texture.cuh"

class Image : public Texture {
  public:
    const static int bytes_per_pixel = 3;

    __host__ __device__ Image(unsigned char *data, int width, int height) : data{data}, width{width}, height{height} {
      this->bytes_per_scanline = bytes_per_pixel * width;
    }

    __host__ __device__ virtual Arr3 map(float u, float v, const Arr3 &point) const override;

  private:
    unsigned char *data;
    int width, height;
    int bytes_per_scanline;
};

__host__ __device__ 
Arr3 Image::map(float u, float v, const Arr3 &point) const {
  // Clamp input texture coordinates to [0,1] x [1,0]
  u = clamp(u, 0.0f, 1.0f);
  v = 1.0 - clamp(v, 0.0f, 1.0f);  // Flip V to image coordinates

  auto i = static_cast<int>(u * this->width);
  auto j = static_cast<int>(v * this->height);

  // Clamp integer mapping, since actual coordinates should be less than 1.0
  if (i >= this->width)  i = this->width - 1.0f;
  if (j >= this->height) j = this->height - 1.0f;

  const auto color_scale = 1.0f / 255.0f;

  return Arr3(
    color_scale * this->data[j * bytes_per_scanline + i * bytes_per_pixel], 
    color_scale * this->data[j * bytes_per_scanline + i * bytes_per_pixel + 1], 
    color_scale * this->data[j * bytes_per_scanline + i * bytes_per_pixel + 2]
  );
}


