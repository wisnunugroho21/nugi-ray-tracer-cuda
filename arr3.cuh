#pragma once

#include "helper.cuh"

#include <cmath>

class Arr3
{
	public:
		__host__ __device__ Arr3() { 
			this->e[0] = 0.0f;
			this->e[1] = 0.0f;
			this->e[2] = 0.0f;
		}

		__host__ __device__ Arr3(float e1, float e2, float e3);

		__host__ __device__ float x() const { return this->e[0]; }
		__host__ __device__ float y() const { return this->e[1]; }
		__host__ __device__ float z() const { return this->e[2]; }

		__host__ __device__ float r() const { return this->e[0]; }
		__host__ __device__ float g() const { return this->e[1]; }
		__host__ __device__ float b() const { return this->e[2]; }
		
		__host__ __device__ float operator [](int i) const;
		__host__ __device__ float& operator [](int i);

		__host__ __device__ Arr3 operator -() const;
 
		__host__ __device__ Arr3& operator += (const Arr3 &a);
		__host__ __device__ Arr3& operator += (const float t);
 
		__host__ __device__ Arr3& operator -= (const Arr3 &a);
		__host__ __device__ Arr3& operator -= (const float t);

		__host__ __device__ Arr3& operator *= (const Arr3 &a);
		__host__ __device__ Arr3& operator *= (const float t);
 
		__host__ __device__ Arr3& operator /= (const Arr3 &a);
		__host__ __device__ Arr3& operator /= (const float t);
 
		__host__ __device__ float lengthSquared() const;
		__host__ __device__ float length() const;
		__host__ __device__ Arr3 unitVector() const;
		__host__ __device__ bool nearZero() const;

		__device__ static Arr3 random(curandState *randState);
		__device__ static Arr3 random(float min, float max, curandState *randState);
		__device__ static Arr3 randomUnitLength(curandState *randState);
		__device__ static Arr3 randomUnitVector(curandState *randState);
		__device__ static Arr3 randomInUnitDisk(curandState *randState);
		__device__ static Arr3 randomInUnitSphere(curandState *randState);

	public:
		float e[3];
};

__host__ __device__ Arr3 operator + (Arr3 u, Arr3 v);
__host__ __device__ Arr3 operator - (Arr3 u, Arr3 v);
__host__ __device__ Arr3 operator * (Arr3 u, Arr3 v);
__host__ __device__ Arr3 operator / (Arr3 u, Arr3 v);
 
__host__ __device__ Arr3 operator + (float t, Arr3 v);
__host__ __device__ Arr3 operator - (float t, Arr3 v);
__host__ __device__ Arr3 operator * (float t, Arr3 v);
__host__ __device__ Arr3 operator / (float t, Arr3 v);
 
__host__ __device__ Arr3 operator + (Arr3 u, float t);
__host__ __device__ Arr3 operator - (Arr3 u, float t);
__host__ __device__ Arr3 operator * (Arr3 u, float t);
__host__ __device__ Arr3 operator / (Arr3 u, float t);

__host__ __device__ float dot (const Arr3 &u, const Arr3 &v);
__host__ __device__ Arr3 cross (const Arr3 &u, const Arr3 &v);

__host__ __device__
Arr3::Arr3(float e1, float e2, float e3) {
	this->e[0] = e1;
	this->e[1] = e2;
	this->e[2] = e3;
}

__host__ __device__
float Arr3::operator [](int i) const {
	return this->e[i];
}

__host__ __device__
float& Arr3::operator [](int i) {
	return this->e[i];
}

__host__ __device__
Arr3 Arr3::operator -() const {
	return Arr3(-1 * this->e[0], -1 * this->e[1], -1 * this->e[2]);
}


__host__ __device__
Arr3& Arr3::operator += (const Arr3& a) {
	this->e[0] += a.x();
	this->e[1] += a.y();
	this->e[2] += a.z();

	return *this;
}

__host__ __device__
Arr3& Arr3::operator += (const float t) {
	this->e[0] += t;
	this->e[1] += t;
	this->e[2] += t;

	return *this;
}

__host__ __device__
Arr3& Arr3::operator -= (const Arr3& a) {
	this->e[0] -= a.x();
	this->e[1] -= a.y();
	this->e[2] -= a.z();

	return *this;
}

__host__ __device__
Arr3& Arr3::operator -= (const float t) {
	this->e[0] -= t;
	this->e[1] -= t;
	this->e[2] -= t;

	return *this;
}

__host__ __device__
Arr3& Arr3::operator *= (const Arr3& a) {
	this->e[0] *= a.x();
	this->e[1] *= a.y();
	this->e[2] *= a.z();

	return *this;
}

__host__ __device__
Arr3& Arr3::operator *= (const float t) {
	this->e[0] *= t;
	this->e[1] *= t;
	this->e[2] *= t;

	return *this;
}

__host__ __device__
Arr3& Arr3::operator /= (const Arr3& a) {
	this->e[0] /= a.x();
	this->e[1] /= a.y();
	this->e[2] /= a.z();

	return *this;
}

__host__ __device__
Arr3& Arr3::operator /= (const float t) {
	this->e[0] /= t;
	this->e[1] /= t;
	this->e[2] /= t;

	return *this;
}

__host__ __device__
float Arr3::lengthSquared() const {
	return this->e[0] * this->e[0] + this->e[1] * this->e[1] + this->e[2] * this->e[2];
}

__host__ __device__
float Arr3::length() const {
	return sqrt(this->lengthSquared());
}

__host__ __device__
Arr3 Arr3::unitVector() const {
	return Arr3(this->e[0] / this->length(), this->e[1] / this->length(), this->e[2] / this->length());
}

__host__ __device__
bool Arr3::nearZero() const {
	float s = 1e-8;
	return (fabs(this->e[0]) < s) && (fabs(this->e[1]) < s) && (fabs(this->e[2]) < s);
}


__device__
Arr3 Arr3::random(curandState *randState) {
	return Arr3(randomFloat(randState), randomFloat(randState), randomFloat(randState));
}

__device__
Arr3 Arr3::random(float min, float max, curandState *randState) {
	return Arr3(randomFloat(min, max, randState), randomFloat(min, max, randState), randomFloat(min, max, randState));
}

__device__
Arr3 Arr3::randomUnitLength(curandState *randState) {
	while (true) {
		Arr3 p = Arr3::random(-1, 1, randState);
		if (p.lengthSquared() < 1) {
			return p;
		}
	}
}

__device__
Arr3 Arr3::randomUnitVector(curandState *randState) {
	return Arr3::randomUnitLength(randState).unitVector();
}

__device__
Arr3 Arr3::randomInUnitDisk(curandState *randState) {
	while (true) {
		Arr3 p = Arr3(randomFloat(-1, 1, randState), randomFloat(-1, 1, randState), 0);
		if (p.lengthSquared() < 1) {
			return p;
		}
	}
}

__device__
Arr3 Arr3::randomInUnitSphere(curandState *randState) {
	while (true) {
		auto p = Arr3::random(-1, 1, randState);
		if (p.lengthSquared() < 1) return p;
	}
}


__host__ __device__
Arr3 operator + (Arr3 u, Arr3 v) {
	return Arr3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__
Arr3 operator - (Arr3 u, Arr3 v) {
	return Arr3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__host__ __device__
Arr3 operator * (Arr3 u, Arr3 v) {
	return Arr3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__
Arr3 operator / (Arr3 u, Arr3 v) {
	return Arr3(u.x() / v.x(), u.y() / v.y(), u.z() / v.z());
}


__host__ __device__
Arr3 operator + (float t, Arr3 v) {
	return Arr3(t + v.x(), t + v.y(), t + v.z());
}

__host__ __device__
Arr3 operator - (float t, Arr3 v) {
	return Arr3(t - v.x(), t - v.y(), t - v.z());
}

__host__ __device__
Arr3 operator * (float t, Arr3 v) {
	return Arr3(t * v.x(), t * v.y(), t * v.z());
}

__host__ __device__
Arr3 operator / (float t, Arr3 v) {
	return Arr3(t / v.x(), t / v.y(), t / v.z());
}


__host__ __device__
Arr3 operator + (Arr3 u, float t) {
	return Arr3(u.x() + t, u.y() + t, u.z() + t);
}

__host__ __device__
Arr3 operator - (Arr3 u, float t) {
	return Arr3(u.x() - t, u.y() - t, u.z() - t);
}

__host__ __device__
Arr3 operator * (Arr3 u, float t) {
	return Arr3(u.x() * t, u.y() * t, u.z() * t);
}

__host__ __device__
Arr3 operator / (Arr3 u, float t) {
	return Arr3(u.x() / t, u.y() / t, u.z() / t);
}

__host__ __device__
float dot(const Arr3 &u, const Arr3 &v) {
	return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

__host__ __device__
Arr3 cross(const Arr3 &u, const Arr3 &v) {
	return Arr3(
		u.y() * v.z() - u.z() * v.y(),
		u.z() * v.x() - u.x() * v.z(),
		u.x() * v.y() - u.y() * v.x()
	);
}