#pragma once

#include "hittable/hittable.cuh"
#include "material/material.cuh"

constexpr float kEpsilon = 1e-8f;

class Triangle : public Hittable {
  public:
    __host__ __device__ Triangle(Arr3 firstPoint, Arr3 secondPoint, Arr3 thirdPoint, Material *material) 
    : firstPoint{firstPoint}, secondPoint{secondPoint}, thirdPoint{thirdPoint}, material{material} 
    {}

    __device__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const override;
    __host__ virtual bool hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const override;

    __host__ __device__ virtual float numCompare(int index) const override;
    __host__ __device__ virtual bool boundingBox(BoundingRecord *box) override;

  private:
    Arr3 firstPoint;
    Arr3 secondPoint;
    Arr3 thirdPoint;

    Material *material;
};

__device__ 
bool Triangle::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat, curandState* randState) const {
  #ifdef MOLLER_TRUMBORE
    Arr3 v0v1 = this->secondPoint - this->firstPoint;
    Arr3 v0v2 = this->thirdPoint - this->firstPoint;
    Arr3 pvec = Arr3::cross(r.direction().unitVector(), v0v2);
    float det = Arr3::dot(v0v1, pvec);
  #ifdef CULLING
    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < kEpsilon) return false;
  #else
    // ray and triangle are parallel if det is close to 0
    if (fabsf(det) < kEpsilon) return false;
  #endif
    float invDet = 1.0f / det;

    Arr3 tvec = r.origin() - this->firstPoint;
    float u = Arr3::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    Arr3 qvec = Arr3::cross(tvec, v0v1);
    float v = Arr3::dot(r.direction().unitVector(), qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;
    
    float t = Arr3::dot(v0v2, qvec) * invDet;

    if (hit != nullptr && hit != NULL) {
      hit->t = t;
      hit->point = r.at(t);
      
      hit->textCoord.u = u;
      hit->textCoord.v = v;

      Arr3 outwardNormal = Arr3::cross(v0v1, v0v2).unitVector();
      hit->faceNormal = FaceNormal(r, outwardNormal);
    }

    if (mat != nullptr && mat != NULL) {
      mat->material = this->material;
    }
    
    return true;
  #else
    // compute plane's normal
    Arr3 v0v1 = this->secondPoint - this->firstPoint;
    Arr3 v0v2 = this->thirdPoint - this->firstPoint;

    // no need to normalize
    Arr3 N = Arr3::cross(v0v1, v0v2); // N
    float denom = Arr3::dot(N, N);
    
    // Step 1: finding P
    
    // check if ray and plane are parallel ?
    float NdotRayDirection = Arr3::dot(N, r.direction().unitVector());

    if (fabsf(NdotRayDirection) < kEpsilon) // almost 0
        return false; // they are parallel so they don't intersect ! 

    // compute d parameter using equation 2
    float d = -1.0f * Arr3::dot(N, this->firstPoint);
    
    // compute t (equation 3)
    float t = -1.0f * (Arr3::dot(N, r.origin()) + d) / NdotRayDirection;
    
    // check if the triangle is in behind the ray
    if (t < 0) return false; // the triangle is behind
 
    // compute the intersection point using equation 1
    Arr3 P = r.at(t);
 
    // Step 2: inside-outside test
    Arr3 C; // vector perpendicular to triangle's plane

    float u, v;
 
    // edge 0
    Arr3 edge0 = this->secondPoint - this->firstPoint; 
    Arr3 vp0 = P - this->firstPoint;
    C = Arr3::cross(edge0, vp0);
    if ( Arr3::dot(N, C) < 0.0f) return false; // P is on the right side
 
    // edge 1
    Arr3 edge1 = this->thirdPoint - this->secondPoint; 
    Arr3 vp1 = P - this->secondPoint;
    C = Arr3::cross(edge1, vp1);
    if ((u = Arr3::dot(N, C)) < 0)  return false; // P is on the right side
 
    // edge 2
    Arr3 edge2 = this->firstPoint - this->thirdPoint; 
    Arr3 vp2 = P - this->thirdPoint;
    C = Arr3::cross(edge2, vp2);
    if ((v = Arr3::dot(N, C)) < 0) return false; // P is on the right side;

    u /= denom;
    v /= denom;

    if (hit != nullptr && hit != NULL) {
      hit->t = t;
      hit->point = P;
      
      hit->textCoord.u = u;
      hit->textCoord.v = v;

      Arr3 outwardNormal = N.unitVector();
      hit->faceNormal = FaceNormal(r, outwardNormal);
    }

    if (mat != nullptr && mat != NULL) {
      mat->material = this->material;
    }

    return true; // this ray hits the triangle
  #endif
}

__host__ 
bool Triangle::hit(const Ray &r, float tMin, float tMax, HitRecord *hit, MaterialRecord *mat) const {
  #ifdef MOLLER_TRUMBORE
    Arr3 v0v1 = this->secondPoint - this->firstPoint;
    Arr3 v0v2 = this->thirdPoint - this->firstPoint;
    Arr3 pvec = Arr3::cross(r.direction().unitVector(), v0v2);
    float det = Arr3::dot(pvec, v0v1);
  #ifdef CULLING
    // if the determinant is negative the triangle is backfacing
    // if the determinant is close to 0, the ray misses the triangle
    if (det < kEpsilon) return false;
  #else
    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < kEpsilon) return false;
  #endif
    float invDet = 1 / det;

    Arr3 tvec = r.origin() - this->firstPoint;
    float u = Arr3::dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    Arr3 qvec = Arr3::cross(tvec, v0v1);
    float v = Arr3::dot(r.direction().unitVector(), qvec) * invDet;
    if (v < 0 || u + v > 1) return false;
    
    float t = Arr3::dot(v0v2, qvec) * invDet;

    if (hit != nullptr && hit != NULL) {
      hit->t = t;
      hit->point = r.at(t);
      
      hit->textCoord.u = u;
      hit->textCoord.v = v;

      Arr3 outwardNormal = Arr3::cross(v0v1, v0v2).unitVector();
      hit->faceNormal = FaceNormal(r, outwardNormal);
    }

    if (mat != nullptr && mat != NULL) {
      mat->material = this->material;
    }
    
    return true;
  #else
    // compute plane's normal
    Arr3 v0v1 = this->secondPoint - this->firstPoint;
    Arr3 v0v2 = this->thirdPoint - this->firstPoint;

    // no need to normalize
    Arr3 N = Arr3::cross(v0v1, v0v2); // N
    float denom = Arr3::dot(N, N);
    
    // Step 1: finding P
    
    // check if ray and plane are parallel ?
    float NdotRayDirection = Arr3::dot(N, r.direction().unitVector());

    if (fabs(NdotRayDirection) < kEpsilon) // almost 0
        return false; // they are parallel so they don't intersect ! 

    // compute d parameter using equation 2
    float d = -1.0f * Arr3::dot(N, this->firstPoint);
    
    // compute t (equation 3)
    float t = -1.0f * (Arr3::dot(N, r.origin()) + d) / NdotRayDirection;
    
    // check if the triangle is in behind the ray
    if (t < 0) return false; // the triangle is behind
 
    // compute the intersection point using equation 1
    Arr3 P = r.at(t);
 
    // Step 2: inside-outside test
    Arr3 C; // vector perpendicular to triangle's plane

    float u, v;
 
    // edge 0
    Arr3 edge0 = this->secondPoint - this->firstPoint; 
    Arr3 vp0 = P - this->firstPoint;
    C = Arr3::cross(edge0, vp0);
    if ( Arr3::dot(N, C) < 0.0f) return false; // P is on the right side
 
    // edge 1
    Arr3 edge1 = this->thirdPoint - this->secondPoint; 
    Arr3 vp1 = P - this->secondPoint;
    C = Arr3::cross(edge1, vp1);
    if ((u = Arr3::dot(N, C)) < 0)  return false; // P is on the right side
 
    // edge 2
    Arr3 edge2 = this->firstPoint - this->thirdPoint; 
    Arr3 vp2 = P - this->thirdPoint;
    C = Arr3::cross(edge2, vp2);
    if ((v = Arr3::dot(N, C)) < 0) return false; // P is on the right side;

    u /= denom;
    v /= denom;

    if (hit != nullptr && hit != NULL) {
      hit->t = t;
      hit->point = P;
      
      hit->textCoord.u = u;
      hit->textCoord.v = v;

      Arr3 outwardNormal = N.unitVector();
      hit->faceNormal = FaceNormal(r, outwardNormal);
    }

    if (mat != nullptr && mat != NULL) {
      mat->material = this->material;
    }

    return true; // this ray hits the triangle
  #endif
}

__host__ __device__ 
float Triangle::numCompare(int index) const {
  float min = 999.0f;

  min = fminf(min, this->firstPoint[index]);
  min = fminf(min, this->secondPoint[index]);
  min = fminf(min, this->thirdPoint[index]);

  return min;
}

__host__ __device__ 
bool Triangle::boundingBox(BoundingRecord *box) {
  Arr3 min( 9999.0f, 9999.0f, 9999.0f );
  Arr3 max( -9999.0f, -9999.0f, -9999.0f );

  for (int i = 0; i < 3; i++) {
    max[i] = fmaxf(max[i], this->firstPoint[i]);
    max[i] = fmaxf(max[i], this->secondPoint[i]);
    max[i] = fmaxf(max[i], this->thirdPoint[i]);
    max[i] += 0.01f;

    min[i] = fminf(min[i], this->firstPoint[i]);
    min[i] = fminf(min[i], this->secondPoint[i]);
    min[i] = fminf(min[i], this->thirdPoint[i]);
    min[i] -= 0.01f;
  }

  box->boundingBox = AABB(min, max);
  return true;
}