#pragma once

#include <limits>
#include <random>

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.141592653589f;

float randomFloat();
float randomFloat(float min, float max);
float clamp(float value, float min, float max);

float degreesToRadians(float degrees);
int randInt(int min, int max);

float randomFloat() {
	return rand() / (RAND_MAX + 1.0);
}

float randomFloat(float min, float max) {
	return min + (max - min) * randomFloat();
}

float clamp(float value, float min, float max) {
	if (value < min) return min;
	if (value > max) return max;
	return value;
}

float degreesToRadians(float degrees) {
	return degrees * pi / 180.0f;
}

int randInt(int min, int max) {
	return static_cast<int>(randomFloat(min, max + 1));
}