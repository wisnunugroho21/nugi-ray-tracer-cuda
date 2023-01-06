#pragma once

#include "math/arr3.cuh"

class Ray {
	public:
		__host__ __device__ Ray() {}
		__host__ __device__ Ray(Arr3 origin, Arr3 direction, float time = 0.0) : org{origin}, dir{direction}, tm{time} {}

		__host__ __device__ Arr3 origin() const { return this->org; }
		__host__ __device__ Arr3 direction() const { return this->dir; }
		__host__ __device__ float time() const { return this->tm; }

		__host__ __device__ Arr3 at(float t) const;

  private:
		Arr3 org;
		Arr3 dir;
		float tm;
};

__host__ __device__
Arr3 Ray::at(float t) const {
	return this->org + t * this->dir;
}