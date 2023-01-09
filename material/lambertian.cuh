#pragma once

#include "material.cuh"
#include "texture/solid.cuh"
#include "denoiser/onb.cuh"
#include "denoiser/pdf.cuh"

class Lambertian : public Material {
  public:
    __host__ __device__ Lambertian(const Arr3 &color) : texture{new Solid(color)} {}
    __host__ __device__ Lambertian(Texture *texture) : texture{texture} {}

    __device__ virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const override;
    __device__ virtual float scatteringPdf(const Ray &ray, const HitRecord &hit, const Ray &scatteredRay, curandState* randState) const override;

  private:
    Texture *texture;
};

__device__
bool Lambertian::scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered, curandState* randState) const {
  ONB uvw;
  uvw.buildFromW(hit.faceNormal.normal);
  auto scatterDirection = uvw.local(randomCosineDirection(randState));

	scattered->newRay = Ray(hit.point, scatterDirection.unitVector(), ray.time());
	scattered->colorAttenuation = this->texture->map(hit.textCoord.u, hit.textCoord.v, hit.point);
  scattered->pdf = Arr3::dot(uvw.w(), scattered->newRay.direction()) / 3.141592653589f;

	return true;
}

__device__ 
float Lambertian::scatteringPdf(const Ray &ray, const HitRecord &hit, const Ray &scatteredRay, curandState* randState) const {
  auto cosine = Arr3::dot(hit.faceNormal.normal, scatteredRay.direction().unitVector());
  return cosine < 0 ? 0 : cosine / 3.141592653589f;
}
