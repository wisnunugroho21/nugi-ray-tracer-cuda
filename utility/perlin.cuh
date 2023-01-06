#pragma once

#include "helper/helper.cuh"
#include "math/arr3.cuh"

class Perlin {
  public:
    __device__ Perlin(curandState *randState);
    __host__ __device__~Perlin();

    __host__ __device__ float noise(const Arr3 &point) const;
    __host__ __device__ float turbulence(const Arr3 &point, int depth = 7) const;

  private:
    static const int pointCount = 256;
    int *permX, *permY, *permZ;
    Arr3 *randvec;

    __device__ static int* perlinGeneratePerm(curandState *randState);
    __device__ static void permute(int *p, int n, curandState *randState);
    __host__ __device__ static float trilinearInterp(float c[2][2][2], float u, float v, float w);
    __host__ __device__ static float perlinInterp(Arr3 c[2][2][2], float u, float v, float w);
};

__device__
Perlin::Perlin(curandState *randState) {
  this->randvec = new Arr3[Perlin::pointCount];
  for (int i = 0; i < Perlin::pointCount; i++) {
    this->randvec[i] = Arr3::random(-1, 1, randState).unitVector();
  }

  this->permX = perlinGeneratePerm(randState);
  this->permY = perlinGeneratePerm(randState);
  this->permZ = perlinGeneratePerm(randState);
}

__host__ __device__
Perlin::~Perlin() {
  delete[] this->randvec;
  delete[] this->permX;
  delete[] this->permY;
  delete[] this->permZ;
}

__host__ __device__
float Perlin::noise(const Arr3 &point) const {
  auto u = point.x() - floorf(point.x());
  auto v = point.y() - floorf(point.y());
  auto w = point.z() - floorf(point.z());
  
  auto i = static_cast<int>(floorf(point.x()));
  auto j = static_cast<int>(floorf(point.y()));
  auto k = static_cast<int>(floorf(point.z()));

  Arr3 c[2][2][2];

  for (int di = 0; di < 2; di++) {
    for (int dj = 0; dj < 2; dj++) {
      for (int dk = 0; dk < 2; dk++) {
        c[di][dj][dk] = this->randvec[
          this->permX[(i + di) & 255] ^ 
          this->permY[(j + dj) & 255] ^
          this->permZ[(k + dk) & 255]
        ];
      }
    }
  }

  return Perlin::perlinInterp(c, u, v, w);
}

__host__ __device__ 
float Perlin::turbulence(const Arr3 &point, int depth) const {
  auto accum = 0.0f;
  auto tempPoint = point;
  auto  weight = 1.0f;

  for (int i = 0; i < depth; i++) {
    accum += weight * this->noise(tempPoint);
    weight *= 0.5f;
    tempPoint *= 2.0f;
  }

  return fabsf(accum);
}

__device__
int* Perlin::perlinGeneratePerm(curandState *randState) {
  auto p = new int[Perlin::pointCount];

  for (int i = 0; i < Perlin::pointCount; i++) {
    p[i] = i;
  }

  permute(p, Perlin::pointCount, randState);

  return p;
}

__device__
void Perlin::permute(int *p, int n, curandState *randState) {
  for (int i = n - 1; i > 0; i--) {
    int target = randInt(0, i, randState);
    int temp = p[i];
    p[i] = p[target];
    p[target] = temp;
  }
}

__host__ __device__ 
float Perlin::trilinearInterp(float c[2][2][2], float u, float v, float w) {
  auto accum = 0.0f;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        accum += (i * u + (1 - i) * (1 - u)) * 
                (j * v + (1 - j) * (1 - v)) * 
                (k * w + (1 - k) * (1 - w)) *
                c[i][j][k];
      }
    }
  }

  return accum;
}

__host__ __device__ 
float Perlin::perlinInterp(Arr3 c[2][2][2], float u, float v, float w) {
  auto uu = u * u * (3 - 2 * u);
  auto vv = v * v * (3 - 2 * v);
  auto ww = w * w * (3 - 2 * w);

  auto accum = 0.0f;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        Arr3 weight(u - i, v - j, w - k);
        accum += (i * uu + (1 - i) * (1 - uu)) * 
                (j * vv + (1 - j) * (1 - vv)) * 
                (k * ww + (1 - k) * (1 - ww)) *
                Arr3::dot(c[i][j][k], weight);
      }
    }
  }

  return accum;
}


