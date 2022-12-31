#include "camera.hpp"
#include "hittable/hittable_list.hpp"
#include "hittable/shape/sphere.hpp"
#include "material/lambertian.hpp"

#include <iostream>
#include <fstream>
#include <limits>

void writeColor(std::ofstream &ofl, Arr3 &frameBuffer, int samplePerPixel = 1) {
  auto scale = 1.0f / samplePerPixel;
  
  auto r = sqrt(scale * frameBuffer.r());
  auto g = sqrt(scale * frameBuffer.g());
  auto b = sqrt(scale * frameBuffer.b());

  ofl << static_cast<int>(256 * clamp(r, 0.0f, 0.999f)) << " "
      << static_cast<int>(256 * clamp(g, 0.0f, 0.999f)) << " "
      << static_cast<int>(256 * clamp(b, 0.0f, 0.999f)) << "\n";
}

Arr3 tracing(const Ray &r, Hittable **world) {
  Arr3 curAttenuation = Arr3(1.0f, 1.0f, 1.0f);
  Ray curRay = r;

  HitRecord *hit = new HitRecord();
  ScatterRecord *scat = new ScatterRecord();
  MaterialRecord *mat = new MaterialRecord();

  for (size_t i = 0; i < 50; i++) {
    if (world[0]->hit(curRay, 0.001f, std::numeric_limits<float>::infinity(), hit, mat)) {
      if (mat->material->scatter(r, *hit, scat)) {
        curAttenuation *= scat->colorAttenuation;
        curRay = scat->newRay;
      } else {
        return Arr3(0.0f, 0.0f, 0.0f);
      }
    } else {
      auto unitDirection = r.direction().unitVector();
      auto t = 0.5f * (unitDirection.y() + 1.0f);
      auto color = (1.0f - t) * Arr3(1.0f, 1.0f, 1.0f) + t * Arr3(0.5f, 0.7f, 1.0f);

      return curAttenuation * color;
    }
  }

  return Arr3(0.0f, 0.0f, 0.0f);
}

void render(Arr3 *frameBuffer, int width, int height, int nSample, Camera **cam, Hittable **world, int i, int j) {
  if (i >= width || j >= height) return;

  int pixelIndex = i + j * width;
  Arr3 color(0.0f, 0.0f, 0.0f);

  for (size_t s = 0; s < nSample; s++) {
    auto u = float(i + randomFloat()) / (width - 1);
    auto v = float(j + randomFloat()) / (height - 1);

    Ray r = cam[0]->transform(u, v);
    color += tracing(r, world);
  }

  frameBuffer[pixelIndex] = color;
}

void createWorld(Camera **camera, Hittable **hits, Material **mats, Hittable **world) {
  mats[0] = new Lambertian(Arr3(0.7f, 0.3f, 0.3f));
  mats[1] = new Lambertian(Arr3(0.8f, 0.8f, 0.0f));
  mats[2] = new Lambertian(Arr3(0.1f, 0.2f, 0.5f));
  hits[0] = new Sphere(Arr3(0.0f, 0.0f, -1.0f), 0.5f, mats[0]);
  hits[1] = new Sphere(Arr3(0.0f, -100.5f, -1.0f), 100.0f, mats[1]);
  hits[2] = new Sphere(Arr3(1.0f, 0.0f, -1.0f), 0.5f, mats[2]);
  world[0] = new HittableList(hits, 3);
  camera[0] = new Camera(Arr3(-2.0f, 2.0f, 1.0f), Arr3(0.0f, 0.0f, -1.0f), Arr3(0.0f, 1.0f, 0.0f), 40.0f, 1.0f, 0.0f, 10.0f);
}

void freeWorld(Camera **camera, Hittable **hits, Material **mats, Hittable **world) {
  delete camera[0];
  delete world[0];
  delete hits[0];
  delete hits[1];
  delete mats[0];
  delete mats[1];
}

int main() {
  const int imageWidth = 400;
  const int imageHeight = 400;
  const int samplePerPixel = 10;

  Arr3 *frameBuffers;
  size_t fb_size = imageWidth * imageHeight * sizeof(Arr3);

  Camera **camera;
  Hittable **hits;
  Material **mats;
  Hittable **world;

  frameBuffers = (Arr3*) malloc(fb_size);
  camera = (Camera**) malloc(sizeof(Camera*));
  hits = (Hittable**) malloc(3 * sizeof(Hittable*));
  mats = (Material**) malloc(3 * sizeof(Material*));
  world = (Hittable**) malloc(sizeof(Hittable*));

  createWorld(camera, hits, mats, world);

  std::cerr << "render" << std::flush;

  for (int j = imageHeight - 1; j >= 0; j--) {
    for (int i = 0; i < imageWidth; i++) {
      std::cerr << "\rRendering: " << j << ' ' << std::flush;
      render(frameBuffers, imageWidth, imageHeight, samplePerPixel, camera, world, i, j);
    }
  }

  std::cerr << "\rwriting" << std::flush;

  std::ofstream ofl("image.ppm");
  ofl << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

  for (int j = imageHeight - 1; j >= 0; j--) {
    for (int i = 0; i < imageWidth; i++) {
      size_t pixelIndex = i + j * imageWidth;
      std::cerr << "\rWriting: " << j << ' ' << std::flush;
      writeColor(ofl, frameBuffers[pixelIndex], samplePerPixel);
    }
  }

  freeWorld(camera, hits, mats, world);
  return 0;
}