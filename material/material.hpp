#pragma once

#include "../struct/hit_record.hpp"
#include "../struct/scatter_record.hpp"

class Material {
  public:
    virtual bool scatter(const Ray &ray, const HitRecord &hit, ScatterRecord *scattered) const = 0;
};
