#pragma once

#include "arr3.cuh"

class Ray {
    public:
        __host__ __device__ Ray() {}
        __host__ __device__ Ray(Arr3 origin, Arr3 direction) : org{origin}, dir{direction} {}

        __host__ __device__ Arr3 origin() const { return this->org; }
        __host__ __device__ Arr3 direction() const { return this->dir; }

        __host__ __device__ Arr3 at(float t) const;

  private:
        Arr3 org;
        Arr3 dir;
};

__device__
Arr3 Ray::at(float t) const {
    return this->org + t * this->dir;
}