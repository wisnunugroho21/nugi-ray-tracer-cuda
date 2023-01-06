#pragma once

#include "utility/ray.cuh"

struct ScatterRecord {
	Arr3 colorAttenuation;
	Ray newRay;
};
