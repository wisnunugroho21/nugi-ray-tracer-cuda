#pragma once

#include "helper.hpp"

#include <cmath>

class Arr3
{
	public:
		Arr3() {}
		Arr3(float e1, float e2, float e3);

		float x() const { return this->e[0]; }
		float y() const { return this->e[1]; }
		float z() const { return this->e[2]; }

		float r() const { return this->e[0]; }
		float g() const { return this->e[1]; }
		float b() const { return this->e[2]; }
		
		float operator [](int i) const;
		float& operator [](int i);

		Arr3 operator -() const;

		Arr3& operator += (const Arr3 &a);
		Arr3& operator += (const float t);
 
		Arr3& operator -= (const Arr3 &a);
		Arr3& operator -= (const float t);
 
		Arr3& operator *= (const Arr3 &a);
		Arr3& operator *= (const float t);
 
		Arr3& operator /= (const Arr3 &a);
		Arr3& operator /= (const float t);

		float lengthSquared() const;
		float length() const;
		Arr3 unitVector() const;
		bool nearZero() const;

		static Arr3 random();
		static Arr3 random(float min, float max);
		static Arr3 randomUnitLength();
		static Arr3 randomUnitVector();
		static Arr3 randomInUnitDisk();
		static Arr3 randomInUnitSphere();

	private:
		float e[3];
};

Arr3 operator + (Arr3 u, Arr3 v);
Arr3 operator - (Arr3 u, Arr3 v);
Arr3 operator * (Arr3 u, Arr3 v);
Arr3 operator / (Arr3 u, Arr3 v);

Arr3 operator + (float t, Arr3 v);
Arr3 operator - (float t, Arr3 v);
Arr3 operator * (float t, Arr3 v);
Arr3 operator / (float t, Arr3 v);

Arr3 operator + (Arr3 u, float t);
Arr3 operator - (Arr3 u, float t);
Arr3 operator * (Arr3 u, float t);
Arr3 operator / (Arr3 u, float t);

float dot (const Arr3 &u, const Arr3 &v);
Arr3 cross (const Arr3 &u, const Arr3 &v);

Arr3::Arr3(float e1, float e2, float e3) {
	this->e[0] = e1;
	this->e[1] = e2;
	this->e[2] = e3;
}

float Arr3::operator [](int i) const {
	return this->e[i];
}

float& Arr3::operator [](int i) {
	return this->e[i];
}

Arr3 Arr3::operator -() const {
	return Arr3(-1 * this->e[0], -1 * this->e[1], -1 * this->e[2]);
}


Arr3& Arr3::operator += (const Arr3& a) {
	this->e[0] += a.x();
	this->e[1] += a.y();
	this->e[2] += a.z();

	return *this;
}

Arr3& Arr3::operator += (const float t) {
	this->e[0] += t;
	this->e[1] += t;
	this->e[2] += t;

	return *this;
}

Arr3& Arr3::operator -= (const Arr3& a) {
	this->e[0] -= a.x();
	this->e[1] -= a.y();
	this->e[2] -= a.z();

	return *this;
}

Arr3& Arr3::operator -= (const float t) {
	this->e[0] -= t;
	this->e[1] -= t;
	this->e[2] -= t;

	return *this;
}

Arr3& Arr3::operator *= (const Arr3& a) {
	this->e[0] *= a.x();
	this->e[1] *= a.y();
	this->e[2] *= a.z();

	return *this;
}

Arr3& Arr3::operator *= (const float t) {
	this->e[0] *= t;
	this->e[1] *= t;
	this->e[2] *= t;

	return *this;
}

Arr3& Arr3::operator /= (const Arr3& a) {
	this->e[0] /= a.x();
	this->e[1] /= a.y();
	this->e[2] /= a.z();

	return *this;
}

Arr3& Arr3::operator /= (const float t) {
	this->e[0] /= t;
	this->e[1] /= t;
	this->e[2] /= t;

	return *this;
}

float Arr3::lengthSquared() const {
	return this->e[0] * this->e[0] + this->e[1] * this->e[1] + this->e[2] * this->e[2];
}

float Arr3::length() const {
	return sqrt(this->lengthSquared());
}

Arr3 Arr3::unitVector() const {
	return Arr3(this->e[0] / this->length(), this->e[1] / this->length(), this->e[2] / this->length());
}

bool Arr3::nearZero() const {
	float s = 1e-8;
	return (fabs(this->e[0]) < s) && (fabs(this->e[1]) < s) && (fabs(this->e[2]) < s);
}


Arr3 Arr3::random() {
	return Arr3(randomFloat(), randomFloat(), randomFloat());
}

Arr3 Arr3::random(float min, float max) {
	return Arr3(randomFloat(min, max), randomFloat(min, max), randomFloat(min, max));
}

Arr3 Arr3::randomUnitLength() {
	while (true) {
		Arr3 p = Arr3::random(-1, 1);
		if (p.lengthSquared() < 1) {
			return p;
		}
	}
}

Arr3 Arr3::randomUnitVector() {
	return Arr3::randomUnitLength().unitVector();
}

Arr3 Arr3::randomInUnitDisk() {
	while (true) {
		Arr3 p = Arr3(randomFloat(-1, 1), randomFloat(-1, 1), 0);
		if (p.lengthSquared() < 1) {
			return p;
		}
	}
}

Arr3 Arr3::randomInUnitSphere() {
	while (true) {
		auto p = Arr3::random(-1,1);
		if (p.lengthSquared() < 1) return p;
	}
}


Arr3 operator + (Arr3 u, Arr3 v) {
	return Arr3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

Arr3 operator - (Arr3 u, Arr3 v) {
	return Arr3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

Arr3 operator * (Arr3 u, Arr3 v) {
	return Arr3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

Arr3 operator / (Arr3 u, Arr3 v) {
	return Arr3(u.x() / v.x(), u.y() / v.y(), u.z() / v.z());
}


Arr3 operator + (float t, Arr3 v) {
	return Arr3(t + v.x(), t + v.y(), t + v.z());
}

Arr3 operator - (float t, Arr3 v) {
	return Arr3(t - v.x(), t - v.y(), t - v.z());
}

Arr3 operator * (float t, Arr3 v) {
	return Arr3(t * v.x(), t * v.y(), t * v.z());
}

Arr3 operator / (float t, Arr3 v) {
	return Arr3(t / v.x(), t / v.y(), t / v.z());
}


Arr3 operator + (Arr3 u, float t) {
	return Arr3(u.x() + t, u.y() + t, u.z() + t);
}

Arr3 operator - (Arr3 u, float t) {
	return Arr3(u.x() - t, u.y() - t, u.z() - t);
}

Arr3 operator * (Arr3 u, float t) {
	return Arr3(u.x() * t, u.y() * t, u.z() * t);
}

Arr3 operator / (Arr3 u, float t) {
	return Arr3(u.x() / t, u.y() / t, u.z() / t);
}

float dot(const Arr3 &u, const Arr3 &v) {
	return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

Arr3 cross(const Arr3 &u, const Arr3 &v) {
	return Arr3(
		u.y() * v.z() - u.z() * v.y(),
		u.z() * v.x() - u.x() * v.z(),
		u.x() * v.y() - u.y() * v.x()
	);
}