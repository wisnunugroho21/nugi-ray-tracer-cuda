#pragma once

#include <cstdlib>
#include <curand_kernel.h>

__device__ float randomFloat(curandState *randState);
__device__ float randomFloat(float min, float max, curandState *randState);

__host__ float randomFloat();
__host__ float randomFloat(float min, float max);

__host__ __device__ float clamp(float value, float min, float max);

__host__ __device__ float degreesToRadians(float degrees);
__device__ int randInt(int min, int max, curandState *randState);

__device__
float randomFloat(curandState *randState) {
	return curand_uniform(randState);
}

__device__
float randomFloat(float min, float max, curandState *randState) {
	return min + (max - min) * randomFloat(randState);
}

__host__
float randomFloat() {
	return rand() / (RAND_MAX + 1.0f);
}

__host__
float randomFloat(float min, float max) {
	return min + (max - min) * randomFloat();
}

__host__ __device__
float clamp(float value, float min, float max) {
	if (value < min) return min;
	if (value > max) return max;
	return value;
}

__host__ __device__
float degreesToRadians(float degrees) {
	return degrees * 3.141592653589f / 180.0f;
}

__device__
int randInt(int min, int max, curandState *randState) {
	return static_cast<int>(randomFloat(min, max + 1, randState));
}