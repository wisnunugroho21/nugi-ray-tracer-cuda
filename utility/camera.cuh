#pragma once

#include "utility/ray.cuh"
#include "helper/helper.cuh"

class Camera {
public:

  __host__ __device__
  Camera(Arr3 lookfrom, Arr3 lookat, Arr3 vup, 
    float vfov, float aspectRatio, float aperture, 
    float focusDist, float time0 = 0.0f, float time1 = 0.0f);
	
  __device__ Ray transform(float xScreen, float yScreen, curandState* randState);
  __host__ Ray transform(float xScreen, float yScreen);

  private:
	Arr3 origin;
	Arr3 lowerLeftCorner;
	Arr3 horizontal, vertical;
	Arr3 u, v, w;

	float lensRadius;
  float time0, time1;
};

__host__ __device__
Camera::Camera(
	Arr3 lookfrom, Arr3 lookat, Arr3 vup, 
  float vfov, float aspectRatio, float aperture, 
  float focusDist,float time0, float time1
) {
	double theta = degreesToRadians(vfov);
	double height = tanf(theta / 2.0f);
	double viewportHeight = 2.0 * height;
	double viewportWidth = aspectRatio * viewportHeight;

	this->w = (lookfrom - lookat).unitVector();
	this->u = Arr3::cross(vup, this->w).unitVector();
	this->v = Arr3::cross(this->w, this->u); 

	this->origin = lookfrom;
	this->horizontal = focusDist * viewportWidth * this->u;
	this->vertical = focusDist * viewportHeight * this->v;
	this->lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f - focusDist * this->w;

	this->lensRadius = aperture / 2.0f;
}

__device__
Ray Camera::transform(float xScreen, float yScreen, curandState* randState) {
	Arr3 radius = this->lensRadius * Arr3::randomInUnitDisk(randState);
	Arr3 offset = this->u * radius.x() + this->v * radius.y();

	return Ray(
		this->origin + offset, 
		this->lowerLeftCorner + xScreen * this->horizontal + yScreen * this->vertical - this->origin - offset,
    randomFloat(this->time0, this->time1, randState)
	);
}

__host__ 
Ray Camera::transform(float xScreen, float yScreen) {
  Arr3 radius = this->lensRadius * Arr3::randomInUnitDisk();
	Arr3 offset = this->u * radius.x() + this->v * radius.y();

	return Ray(
		this->origin + offset, 
		this->lowerLeftCorner + xScreen * this->horizontal + yScreen * this->vertical - this->origin - offset,
    randomFloat(this->time0, this->time1)
	);
}