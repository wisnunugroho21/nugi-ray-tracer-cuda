#pragma once

#include "ray.hpp"
#include "helper.hpp"

class Camera {
public:
  Camera(Arr3 lookfrom, Arr3 lookat, Arr3 vup, 
    float vfov, float aspectRatio, float aperture, 
    float focusDist);

  Ray transform(float xScreen, float yScreen);

  private:
		Arr3 origin;
		Arr3 lowerLeftCorner;
		Arr3 horizontal, vertical;
		Arr3 u, v, w;

		float lensRadius;
};

Camera::Camera(
	Arr3 lookfrom, Arr3 lookat, Arr3 vup, 
  float vfov, float aspectRatio, float aperture, 
  float focusDist
) {
	double theta = degreesToRadians(vfov);
	double height = tan(theta / 2);
	double viewportHeight = 2.0 * height;
	double viewportWidth = aspectRatio * viewportHeight;

	this->w = (lookfrom - lookat).unitVector();
	this->u = cross(vup, this->w).unitVector();
	this->v = cross(this->w, this->u); 

	this->origin = lookfrom;
	this->horizontal = focusDist * viewportWidth * this->u;
	this->vertical = focusDist * viewportHeight * this->v;
	this->lowerLeftCorner = origin - horizontal / 2 - vertical / 2 - focusDist * this->w;

	this->lensRadius = aperture / 2;
}

Ray Camera::transform(float xScreen, float yScreen) {
	Arr3 radius = this->lensRadius * Arr3::randomInUnitDisk();
	Arr3 offset = this->u * radius.x() + this->v * radius.y();

	return Ray(
		this->origin + offset, 
		this->lowerLeftCorner + xScreen * this->horizontal + yScreen * this->vertical - this->origin - offset
	);
}