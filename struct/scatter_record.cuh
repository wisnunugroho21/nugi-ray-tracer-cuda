#pragma once

#include "ray.cuh"

struct ScatterRecord {
	Arr3 colorAttenuation;
	Ray newRay;
};
