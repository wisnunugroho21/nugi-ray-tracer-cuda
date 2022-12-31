#pragma once

#include "../struct/hit_record.hpp"
#include "../struct/material_record.hpp"
#include "../material/material.hpp"

class Hittable {
  public:
    virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *rec, MaterialRecord *mat) const = 0;
};
