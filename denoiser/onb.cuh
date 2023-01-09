#pragma once

#include "math/arr3.cuh"

class ONB {
  public:
    __host__ __device__ ONB() {}

    __host__ __device__ Arr3 operator[] (int i) { return this->axis[i]; }

    __host__ __device__ Arr3 u() const { return this->axis[0]; }
    __host__ __device__ Arr3 v() const { return this->axis[1]; }
    __host__ __device__ Arr3 w() const { return this->axis[2]; }

    __host__ __device__ Arr3 local(float a, float b, float c) const;
    __host__ __device__ Arr3 local(const Arr3 &a) const;

    __host__ __device__ void buildFromW(const Arr3 &n);

  private:
    Arr3 axis[3];
};

__host__ __device__
void ONB::buildFromW(const Arr3 &n) {
  this->axis[2] = n.unitVector();
  Arr3 a = (fabs(w().x()) > 0.9f) ? Arr3(0.0f, 1.0f, 0.0f) : Arr3(1.0f, 0.0f, 0.0f);
  this->axis[1] = Arr3::cross(this->w(), a).unitVector();
  this->axis[0] = Arr3::cross(this->w(), this->v());
}

__host__ __device__
Arr3 ONB::local(float a, float b, float c) const {
  return a * this->u() + b * this->v() + c * this->w();
}

__host__ __device__
Arr3 ONB::local(const Arr3 &a) const {
  return a.x() * this->u() + a.y() * this->v() + a.z() * this->w();
}

