#pragma once

#include <cmath>

class Arr4 {
	public:
		__host__ __device__
    Arr4() { 
			this->e[0] = 0.0f;
			this->e[1] = 0.0f;
			this->e[2] = 0.0f;
      this->e[3] = 0.0f;
		}

		__host__ __device__ Arr4(float e1, float e2, float e3, float e4);

		__host__ __device__ float x() const { return this->e[0]; }
		__host__ __device__ float y() const { return this->e[1]; }
		__host__ __device__ float z() const { return this->e[2]; }
    __host__ __device__ float w() const { return this->e[3]; }

		__host__ __device__ float r() const { return this->e[0]; }
		__host__ __device__ float g() const { return this->e[1]; }
		__host__ __device__ float b() const { return this->e[2]; }
    __host__ __device__ float a() const { return this->e[3]; }

    __host__ __device__ float get(int index) const { return this->e[index]; }
		
		__host__ __device__ float operator [](int i) const;
		__host__ __device__ float& operator [](int i);

		__host__ __device__ Arr4 operator -() const;
  
		__host__ __device__ Arr4& operator += (const Arr4 &a);
		__host__ __device__ Arr4& operator += (const float t);
  
		__host__ __device__ Arr4& operator -= (const Arr4 &a);
		__host__ __device__ Arr4& operator -= (const float t);
 
		__host__ __device__ Arr4& operator *= (const Arr4 &a);
		__host__ __device__ Arr4& operator *= (const float t);
  
		__host__ __device__ Arr4& operator /= (const Arr4 &a);
		__host__ __device__ Arr4& operator /= (const float t);
  
		__host__ __device__ float lengthSquared() const;
		__host__ __device__ float length() const;
		__host__ __device__ Arr4 unitVector() const;
		__host__ __device__ bool nearZero() const;
 
		__host__ __device__ static float dot (const Arr4 &u, const Arr4 &v);

	private:
		float e[4];
};

__host__ __device__ Arr4 operator + (Arr4 u, Arr4 v);
__host__ __device__ Arr4 operator - (Arr4 u, Arr4 v);
__host__ __device__ Arr4 operator * (Arr4 u, Arr4 v);
__host__ __device__ Arr4 operator / (Arr4 u, Arr4 v);
  
__host__ __device__ Arr4 operator + (float t, Arr4 v);
__host__ __device__ Arr4 operator - (float t, Arr4 v);
__host__ __device__ Arr4 operator * (float t, Arr4 v);
__host__ __device__ Arr4 operator / (float t, Arr4 v);
  
__host__ __device__ Arr4 operator + (Arr4 u, float t);
__host__ __device__ Arr4 operator - (Arr4 u, float t);
__host__ __device__ Arr4 operator * (Arr4 u, float t);
__host__ __device__ Arr4 operator / (Arr4 u, float t);

__host__ __device__
Arr4::Arr4(float e1, float e2, float e3, float e4) {
	this->e[0] = e1;
	this->e[1] = e2;
	this->e[2] = e3;
  this->e[3] = e4;
}

__host__ __device__
float Arr4::operator [](int i) const {
	return this->e[i];
}

__host__ __device__
float& Arr4::operator [](int i) {
	return this->e[i];
}

__host__ __device__
Arr4 Arr4::operator -() const {
	return Arr4(-1 * this->e[0], -1 * this->e[1], -1 * this->e[2], -1 * this->e[3]);
}


__host__ __device__
Arr4& Arr4::operator += (const Arr4& a) {
	this->e[0] += a.x();
	this->e[1] += a.y();
	this->e[2] += a.z();
  this->e[3] += a.w();

	return *this;
}

__host__ __device__
Arr4& Arr4::operator += (const float t) {
	this->e[0] += t;
	this->e[1] += t;
	this->e[2] += t;
  this->e[3] += t;

	return *this;
}

__host__ __device__
Arr4& Arr4::operator -= (const Arr4& a) {
	this->e[0] -= a.x();
	this->e[1] -= a.y();
	this->e[2] -= a.z();
  this->e[3] -= a.w();

	return *this;
}

__host__ __device__
Arr4& Arr4::operator -= (const float t) {
	this->e[0] -= t;
	this->e[1] -= t;
	this->e[2] -= t;
  this->e[3] -= t;

	return *this;
}

__host__ __device__
Arr4& Arr4::operator *= (const Arr4& a) {
	this->e[0] *= a.x();
	this->e[1] *= a.y();
	this->e[2] *= a.z();
  this->e[3] *= a.w();

	return *this;
}

__host__ __device__
Arr4& Arr4::operator *= (const float t) {
	this->e[0] *= t;
	this->e[1] *= t;
	this->e[2] *= t;
  this->e[3] *= t;

	return *this;
}

__host__ __device__
Arr4& Arr4::operator /= (const Arr4& a) {
	this->e[0] /= a.x();
	this->e[1] /= a.y();
	this->e[2] /= a.z();
  this->e[3] /= a.w();

	return *this;
}

__host__ __device__
Arr4& Arr4::operator /= (const float t) {
	this->e[0] /= t;
	this->e[1] /= t;
	this->e[2] /= t;
  this->e[3] /= t;

	return *this;
}

__host__ __device__
float Arr4::lengthSquared() const {
	return this->e[0] * this->e[0] + this->e[1] * this->e[1] + this->e[2] * this->e[2] + this->e[3] * this->e[3];
}

__host__ __device__
float Arr4::length() const {
	return sqrtf(this->lengthSquared());
}

__host__ __device__
Arr4 Arr4::unitVector() const {
	return Arr4(this->e[0] / this->length(), this->e[1] / this->length(), this->e[2] / this->length(), this->e[3] / this->length());
}

__host__ __device__
bool Arr4::nearZero() const {
	float s = 1e-8;
	return (fabs(this->e[0]) < s) && (fabs(this->e[1]) < s) && (fabs(this->e[2]) < s) && (fabs(this->e[3]) < s);
}

__host__ __device__
Arr4 operator + (Arr4 u, Arr4 v) {
	return Arr4(u.x() + v.x(), u.y() + v.y(), u.z() + v.z(), u.w() + v.w());
}

__host__ __device__
Arr4 operator - (Arr4 u, Arr4 v) {
	return Arr4(u.x() - v.x(), u.y() - v.y(), u.z() - v.z(), u.w() - v.w());
}

__host__ __device__
Arr4 operator * (Arr4 u, Arr4 v) {
	return Arr4(u.x() * v.x(), u.y() * v.y(), u.z() * v.z(), u.w() * v.w());
}

__host__ __device__
Arr4 operator / (Arr4 u, Arr4 v) {
	return Arr4(u.x() / v.x(), u.y() / v.y(), u.z() / v.z(), u.w() / v.w());
}


__host__ __device__
Arr4 operator + (float t, Arr4 v) {
	return Arr4(t + v.x(), t + v.y(), t + v.z(), t + v.w());
}

__host__ __device__
Arr4 operator - (float t, Arr4 v) {
	return Arr4(t - v.x(), t - v.y(), t - v.z(), t - v.w());
}

__host__ __device__
Arr4 operator * (float t, Arr4 v) {
	return Arr4(t * v.x(), t * v.y(), t * v.z(), t * v.w());
}

__host__ __device__
Arr4 operator / (float t, Arr4 v) {
	return Arr4(t / v.x(), t / v.y(), t / v.z(), t / v.w());
}


__host__ __device__
Arr4 operator + (Arr4 u, float t) {
	return Arr4(u.x() + t, u.y() + t, u.z() + t, u.w() + t);
}

__host__ __device__
Arr4 operator - (Arr4 u, float t) {
	return Arr4(u.x() - t, u.y() - t, u.z() - t, u.w() - t);
}

__host__ __device__
Arr4 operator * (Arr4 u, float t) {
	return Arr4(u.x() * t, u.y() * t, u.z() * t, u.w() * t);
}

__host__ __device__
Arr4 operator / (Arr4 u, float t) {
	return Arr4(u.x() / t, u.y() / t, u.z() / t, u.w() / t);
}

__host__ __device__
float Arr4::dot(const Arr4 &u, const Arr4 &v) {
	return u.x() * v.x() + u.y() * v.y() + u.z() * v.z() + u.w() * v.w();
}