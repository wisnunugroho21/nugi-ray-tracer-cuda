#pragma once

#include "../ray.hpp"

struct ScatterRecord {
  Arr3 colorAttenuation;
  Ray newRay;
};
