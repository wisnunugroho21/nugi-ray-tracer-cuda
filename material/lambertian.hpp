#pragma once

#include "material.hpp"

class Lambertian : public Material {

  public:
    Lambertian(Arr3 colorAlbedo) : colorAlbedo{colorAlbedo} {}

    virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const override;

  private:
    Arr3 colorAlbedo;
};

bool Lambertian::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const {
	auto scatterDirection = hit.faceNormal.normal + Arr3::randomInUnitSphere().unitVector();

  if (scatterDirection.nearZero())
    scatterDirection = hit.faceNormal.normal;

	scattered->newRay = Ray(hit.point, scatterDirection);
	scattered->colorAttenuation = this->colorAlbedo;

	return true;
}
