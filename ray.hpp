#pragma once

#include "arr3.hpp"

class Ray {
  public:
    Ray() {}
    Ray(Arr3 origin, Arr3 direction) : org{origin}, dir{direction} {}

    Arr3 origin() const { return this->org; }
    Arr3 direction() const { return this->dir; }

    Arr3 at(float t) const;

  private:
    Arr3 org;
    Arr3 dir;
};

Arr3 Ray::at(float t) const {
  return this->org + t * this->dir;
}