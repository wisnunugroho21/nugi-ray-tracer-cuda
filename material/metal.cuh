#pragma once

#include "material.cuh"

class Metal : public Material {
	public:
		__host__ __device__ Metal(const Arr3 &colorAlbedo, float fuzziness) : colorAlbedo{colorAlbedo} {
			this->fuzziness = (fuzziness < 1.0f) ? fuzziness : 1.0f;
		}

    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const override;

    private:
			Arr3 colorAlbedo;
			float fuzziness;
};

__device__
bool Metal::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const {
	Arr3 reflected = Arr3::reflect(ray.direction().unitVector(), hit.faceNormal.normal);
	scattered->newRay = Ray(hit.point, reflected + this->fuzziness * Arr3::randomInUnitSphere(randState), ray.time());
	scattered->colorAttenuation = this->colorAlbedo;

	return Arr3::dot(scattered->newRay.direction(), hit.faceNormal.normal) > 0;
}