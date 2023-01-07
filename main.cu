#include "utility/camera.cuh"
#include "hittable/hittable_list.cuh"
#include "hittable/shape/sphere.cuh"
#include "hittable/shape/moving_sphere.cuh"
#include "hittable/bvh.cuh"
#include "material/lambertian.cuh"
#include "material/metal.cuh"
#include "material/dielectric.cuh"
#include "texture/checker.cuh"
#include "texture/noise.cuh"
#include "math/mat4.cuh"
#include "math/arr4.cuh"
#include "material/diffuse_light.cuh"
#include "hittable/shape/rectangle/xy_rect.cuh"
#include "hittable/shape/rectangle/xz_rect.cuh"
#include "hittable/shape/rectangle/yz_rect.cuh"
#include "hittable/shape/rectangle/box.cuh"
#include "hittable/instance/rotationY.cuh"
#include "hittable/instance/translation.cuh"

#include <iostream>
#include <fstream>
#include <limits>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr	<< "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") at " <<
      file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__host__
void writeColor(std::ofstream& ofl, Arr3& frameBuffer, int samplePerPixel = 1) {
	ofl << static_cast<int>(256 * clamp(frameBuffer.r(), 0.0f, 0.999f)) << " "
			<< static_cast<int>(256 * clamp(frameBuffer.g(), 0.0f, 0.999f)) << " "
			<< static_cast<int>(256 * clamp(frameBuffer.b(), 0.0f, 0.999f)) << "\n";
}

__device__
Arr3 tracing(const Ray &r, Hittable** world, const Arr3 &background, curandState* randState) {
	Ray curRay = r;

	HitRecord hit;
	ScatterRecord scat;
	MaterialRecord mat;

  Arr4 lastNum(0.0f, 0.0f, 0.0f, 0.0f);
  Mat4 rayTransform(
    Arr4(1.0f, 0.0f, 0.0f, 0.0f),
    Arr4(0.0f, 1.0f, 0.0f, 0.0f),
    Arr4(0.0f, 0.0f, 1.0f, 0.0f),
    Arr4(0.0f, 0.0f, 0.0f, 1.0f)
  );

	for (int i = 0; i < 50; i++) {
    if (!world[0]->hit(curRay, 0.001f, FLT_MAX, &hit, &mat)) {
      lastNum = Arr4(background.x(), background.y(), background.z(), 1.0f);
      break;
    }

    Arr3 emitted = mat.material->emitted(hit.textCoord.u, hit.textCoord.v, hit.point);

    if (!mat.material->scatter(curRay, hit, &scat, randState)) {
      lastNum = Arr4(emitted.x(), emitted.y(), emitted.z(), 1.0f);
      break;
    }

    Mat4 emitTransf(
      Arr4(1.0f, 0.0f, 0.0f, emitted.x()),
      Arr4(0.0f, 1.0f, 0.0f, emitted.y()),
      Arr4(0.0f, 0.0f, 1.0f, emitted.z()),
      Arr4(0.0f, 0.0f, 0.0f, 1.0f)
    );

    Mat4 attentTransf(
      Arr4(scat.colorAttenuation.x(), 0.0f, 0.0f, 0.0f),
      Arr4(0.0f, scat.colorAttenuation.y(), 0.0f, 0.0f),
      Arr4(0.0f, 0.0f, scat.colorAttenuation.z(), 0.0f),
      Arr4(0.0f, 0.0f, 0.0f, 1.0f)
    );

    rayTransform = emitTransf * attentTransf * rayTransform;
    curRay = scat.newRay;
	}

  Arr4 total = rayTransform * lastNum;
	return Arr3(total.x(), total.y(), total.z());
}

__global__
void render(Arr3 *frameBuffer, int width, int height, int nSample, Arr3 *background, Camera **cam, Hittable **world, curandState *randState) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if (i >= width || j >= height || k >= nSample) return;
	int pixelIndex = k + i * nSample + j * width * nSample;

	Arr3 color(0.0f, 0.0f, 0.0f);
	auto localRandState = randState[pixelIndex];

	auto u = float(i + randomFloat(&localRandState)) / (width - 1);
	auto v = float(j + randomFloat(&localRandState)) / (height - 1);

	Ray r = cam[0]->transform(u, v, &localRandState);
	color += tracing(r, world, background[0], &localRandState);

	frameBuffer[pixelIndex] = color;
}

__global__
void sampling(Arr3 *frameBuffers, Arr3 *finalImageBuffers, int width, int height, int nSample) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= width || j >= height) return;
	Arr3 color(0.0f, 0.0f, 0.0f);

	for (int k = 0; k < nSample; k++) {
		int sampleIndex = k + i * nSample + j * width * nSample;
		color += frameBuffers[sampleIndex];
	}

	color /= float(nSample);
	color[0] = sqrtf(color[0]);
	color[1] = sqrtf(color[1]);
	color[2] = sqrtf(color[2]);

	int pixelIndex = i + j * width;
	finalImageBuffers[pixelIndex] = color;
}

__global__
void freeWorld(Camera **cam, Hittable **hits, Material **mats, Hittable **world, int numObjects) {
	delete cam[0];
	delete world[0];

	for (int i = 0; i < numObjects; i++) {
		delete hits[i];
		delete mats[i];
	}
}

__global__
void initPixelRand(int max_x, int max_y, int max_z, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if (i >= max_x || j >= max_y || k >= max_z) return;
	int pixel_index = k + i * max_z + j * max_x * max_z;

	curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__
void initGlobalRand(curandState* rand_state) {
	curand_init(1983, 0, 0, &rand_state[0]);
}

__global__
void randomScenes(Camera **cam, Hittable **hits, Material **mats, Hittable **world, Texture **texts, curandState *randState, Arr3 *background) {
	auto localRandState = randState[0];
  
  texts[0] = new Checker(Arr3(0.2f, 0.3f, 0.1f), Arr3(0.9f, 0.9f, 0.9f));
	mats[0] = new Lambertian(texts[0]);
	hits[0] = new Sphere(Arr3(0.0f, -1000.0f, 0.0f), 1000.0f, mats[0]);

	int objIndex = 1;

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = randomFloat(&localRandState);
			Arr3 center(a + 0.9f * randomFloat(&localRandState), 0.2f, b + 0.9f * randomFloat(&localRandState));

			if ((center - Arr3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
        if (choose_mat < 0.3f) {
					auto albedo = Arr3::random(&localRandState) * Arr3::random(&localRandState);
					mats[objIndex] = new Lambertian(albedo);
          hits[objIndex] = new Sphere(center, 0.2f, mats[objIndex]);

				} else if (choose_mat < 0.75f) {
					auto albedo = Arr3::random(0.5f, 1.0f, &localRandState);
					auto fuzz = randomFloat(0.0f, 0.5f, &localRandState);

					mats[objIndex] = new Metal(albedo, fuzz);
          hits[objIndex] = new Sphere(center, 0.2f, mats[objIndex]);

				} else {
					mats[objIndex] = new Dielectric(1.5f);
          hits[objIndex] = new Sphere(center, 0.2f, mats[objIndex]);
				}

				objIndex++;
			}
		}
	}

	mats[objIndex] = new Dielectric(1.5f);
	hits[objIndex] = new Sphere(Arr3(0.0f, 1.0f, 0.0f), 1.0f, mats[objIndex]);

	mats[objIndex + 1] = new Lambertian(Arr3(0.4f, 0.2f, 0.1f));
	hits[objIndex + 1] = new Sphere(Arr3(-4.0f, 1.0f, 0.0f), 1.0f, mats[objIndex + 1]);

	mats[objIndex + 2] = new Metal(Arr3(0.7f, 0.6f, 0.5f), 0.1f);
	hits[objIndex + 2] = new Sphere(Arr3(4.0f, 1.0f, 0.0f), 1.0f, mats[objIndex + 2]);

	world[0] = BvhNode::build(hits, objIndex + 3, &localRandState);

	Arr3 lookfrom(13.0f, 2.0f, 3.0f);
	Arr3 lookat(0.0f, 0.0f, 0.0f);
	Arr3 vup(0.0f, 1.0f, 0.0f);
	auto dist_to_focus = 10.0f;
	auto aperture = 0.1f;
	auto aspect_ratio = 1.0f;

	cam[0] = new Camera(lookfrom, lookat, vup, 40.0f, aspect_ratio, aperture, dist_to_focus);
  background[0] = Arr3(0.7f, 0.8f, 1.0f);
}

__global__
void twoSpheres(Camera **cam, Hittable **hits, Material **mats, Hittable **world, Texture **texts, curandState *randState, Arr3 *background) {
  auto localRandState = randState[0];

  texts[0] = new Checker(Arr3(0.2f, 0.3f, 0.1f), Arr3(0.9f, 0.9f, 0.9f));
  mats[0] = new Lambertian(texts[0]);

  hits[0] = new Sphere(Arr3(0.0f, -10.0f, 0.0f), 10.0f, mats[0]);
  hits[1] = new Sphere(Arr3(0.0f, 10.0f, 0.0f), 10.0f, mats[0]);

  world[0] = BvhNode::build(hits, 2, &localRandState);

  Arr3 lookfrom(13.0f, 2.0f, 3.0f);
	Arr3 lookat(0.0f, 0.0f, 0.0f);
	Arr3 vup(0.0f, 1.0f, 0.0f);
	auto dist_to_focus = 10.0f;
	auto aperture = 0.1f;
	auto aspect_ratio = 1.0f;

	cam[0] = new Camera(lookfrom, lookat, vup, 40.0f, aspect_ratio, aperture, dist_to_focus);
  background[0] = Arr3(0.7f, 0.8f, 1.0f);
}

__global__
void simpleLights(Camera **cam, Hittable **hits, Material **mats, Hittable **world, Texture **texts, curandState *randState, Arr3 *background) {
  auto localRandState = randState[0];

  texts[0] = new Noise(&localRandState, 4.0f);

  mats[0] = new Lambertian(texts[0]);
  mats[1] = new DiffuseLight(Arr3(4.0f, 4.0f, 4.0f));

  hits[0] = new Sphere(Arr3(0.0f, -1000.0f, 0.0f), 1000.0f, mats[0]);
  hits[1] = new Sphere(Arr3(0.0f, 2.0f, 0.0f), 2.0f, mats[0]);
  hits[2] = new XYRect(3.0f, 5.0f, 1.0f, 3.0f, -2.0f, mats[1]);

  world[0] = BvhNode::build(hits, 3, &localRandState);

  Arr3 lookfrom(26.0f, 3.0f, 6.0f);
	Arr3 lookat(0.0f, 2.0f, 0.0f);
	Arr3 vup(0.0f, 1.0f, 0.0f);
	auto dist_to_focus = 20.0f;
	auto aperture = 0.0f;
	auto aspect_ratio = 1.0f;

	cam[0] = new Camera(lookfrom, lookat, vup, 40.0f, aspect_ratio, aperture, dist_to_focus);
  background[0] = Arr3(0.0f, 0.0f, 0.0f);
}

__global__
void twoPerlinSpheres(Camera **cam, Hittable **hits, Material **mats, Hittable **world, Texture **texts, curandState *randState, Arr3 *background) {
  auto localRandState = randState[0];

  texts[0] = new Noise(&localRandState, 4.0f);
  mats[0] = new Lambertian(texts[0]);

  hits[0] = new Sphere(Arr3(0.0f, -1000.0f, 0.0f), 1000.0f, mats[0]);
  hits[1] = new Sphere(Arr3(0.0f, 2.0f, 0.0f), 2.0f, mats[0]);

  world[0] = BvhNode::build(hits, 2, &localRandState);

  Arr3 lookfrom(13.0f, 2.0f, 3.0f);
	Arr3 lookat(0.0f, 0.0f, 0.0f);
	Arr3 vup(0.0f, 1.0f, 0.0f);
	auto dist_to_focus = 10.0f;
	auto aperture = 0.1f;
	auto aspect_ratio = 1.0f;

	cam[0] = new Camera(lookfrom, lookat, vup, 40.0f, aspect_ratio, aperture, dist_to_focus);
  background[0] = Arr3(0.7f, 0.8f, 1.0f);
}

__global__
void cornellBox(Camera **cam, Hittable **hits, Material **mats, Hittable **world, Texture **texts, curandState *randState, Arr3 *background) {
  auto localRandState = randState[0];

  mats[0] = new Lambertian(Arr3(0.65f, 0.05f, 0.05f));
  mats[1] = new Lambertian(Arr3(0.73f, 0.73f, 0.73f));
  mats[2] = new Lambertian(Arr3(0.12f, 0.45f, 0.15f));
  mats[3] = new DiffuseLight(Arr3(15.0f, 15.0f, 15.0f));

  hits[0] = new YZRect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, mats[2]);
  hits[1] = new YZRect(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, mats[0]);
  hits[2] = new XZRect(213.0f, 343.0f, 227.0f, 332.0f, 554.0f, mats[3]);
  hits[3] = new XZRect(0.0f, 555.0f, 0.0f, 555.0f, 0.0f, mats[1]);
  hits[4] = new XZRect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, mats[1]);
  hits[5] = new XYRect(0.0f, 555.0f, 0.0f, 555.0f, 555.0f, mats[1]);

  hits[8] = new Box(Arr3(0.0f, 0.0f, 0.0f), Arr3(165.0f, 330.0f, 165.0f), mats[1]);
  hits[9] = new RotationY(hits[8], 15.0f);
  hits[6] = new Translation(hits[9], Arr3(265.0f, 0.0f, 295.0f));

  hits[10] = new Box(Arr3(0.0f, 0.0f, 0.0f), Arr3(165.0f, 165.0f, 165.0f), mats[1]);
  hits[11] = new RotationY(hits[10], -18.0f);
  hits[7] = new Translation(hits[11], Arr3(130.0f, 0.0f, 65.0f));

  world[0] = new HittableList(hits, 8);

  Arr3 lookfrom(278.0f, 278.0f, -800.0f);
	Arr3 lookat(278.0f, 278.0f, 0.0f);
	Arr3 vup(0.0f, 1.0f, 0.0f);
	auto dist_to_focus = 10.0f;
	auto aperture = 0.1f;
	auto aspect_ratio = 1.0f;

	cam[0] = new Camera(lookfrom, lookat, vup, 40.0f, aspect_ratio, aperture, dist_to_focus);
  background[0] = Arr3(0.0f, 0.0f, 0.0f);
}

int main() {
  int scene = 5;

	const int imageWidth = 1024;
	const int imageHeight = 1024;
	const int samplePerPixel = 128;

	int tx = 8;
	int ty = 8;
	int tz = 8;

	Arr3* frameBuffers;
	Arr3* finalImageBuffers;

	curandState* globalRandState;
	curandState* pixelRandState;

	size_t fb_size = static_cast<unsigned long long>(imageWidth * imageHeight * samplePerPixel) * sizeof(Arr3);
	size_t finalImage_size = static_cast<unsigned long long>(imageWidth * imageHeight) * sizeof(Arr3);
	size_t pixel_rand_size = static_cast<unsigned long long>(imageWidth * imageHeight * samplePerPixel) * sizeof(curandState);

	Camera** camera;
	Hittable** hits;
	Material** mats;
	Hittable** world;
  Texture** texts;
  Arr3 *background;

	checkCudaErrors(cudaMallocManaged((void**)&frameBuffers, fb_size));
	checkCudaErrors(cudaMallocManaged((void**)&finalImageBuffers, finalImage_size));
	checkCudaErrors(cudaMalloc((void**)&pixelRandState, pixel_rand_size));
	checkCudaErrors(cudaMalloc((void**)&globalRandState, sizeof(curandState)));
  checkCudaErrors(cudaMalloc((void**)&background, sizeof(Arr3)));

  int numObjects = 0;

  switch (scene){
    case 1:
      numObjects = 22 * 22 + 3 + 1; break;
  
    case 2:
      numObjects = 2; break;

    case 3:
      numObjects = 2; break;

    case 4:
      numObjects = 3; break;

    case 5:
      numObjects = 12; break;
  }

	checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Camera*)));
	checkCudaErrors(cudaMalloc((void**)&hits, numObjects * sizeof(Hittable*)));
	checkCudaErrors(cudaMalloc((void**)&mats, numObjects * sizeof(Material*)));
	checkCudaErrors(cudaMalloc((void**)&world, sizeof(Hittable*)));
  checkCudaErrors(cudaMalloc((void**)&texts, 1 * sizeof(Texture*)));

	initGlobalRand<<<1, 1>>>(globalRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

  switch (scene) {
    case 1:
      randomScenes<<<1, 1>>>(camera, hits, mats, world, texts, globalRandState, background);
      break;
    
    case 2:
      twoSpheres<<<1, 1>>>(camera, hits, mats, world, texts, globalRandState, background);
      break;

    case 3:
      twoPerlinSpheres<<<1, 1>>>(camera, hits, mats, world, texts, globalRandState, background);
      break;

    case 4:
      simpleLights<<<1, 1>>>(camera, hits, mats, world, texts, globalRandState, background);
      break;

    case 5:
      cornellBox<<<1, 1>>>(camera, hits, mats, world, texts, globalRandState, background);
      break;
  }
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::cerr << "\rrendering" << std::flush;

	dim3 blocks(imageWidth / tx, imageHeight / ty, samplePerPixel / tz);
	dim3 threads(tx, ty, tz);

	initPixelRand<<<blocks, threads>>>(imageWidth, imageHeight, samplePerPixel, pixelRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render<<<blocks, threads>>>(frameBuffers, imageWidth, imageHeight, samplePerPixel, background, camera, world, pixelRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	blocks = dim3(imageWidth / tx, imageHeight / ty);
	threads = dim3(tx, ty);

	std::cerr << "\rsampling" << std::flush;

	sampling<<<blocks, threads>>>(frameBuffers, finalImageBuffers, imageWidth, imageHeight, samplePerPixel);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::cerr << "\rwriting" << std::flush;

	std::ofstream ofl("bin/image.ppm");
	ofl << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

	for (int j = imageHeight - 1; j >= 0; j--) {
		for (int i = 0; i < imageWidth; i++) {
			int pixelIndex = i + j * imageWidth;
			writeColor(ofl, finalImageBuffers[pixelIndex]);
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());
	freeWorld<<<1, 1>>>(camera, hits, mats, world, numObjects);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(camera));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(mats));
	checkCudaErrors(cudaFree(hits));
	checkCudaErrors(cudaFree(pixelRandState));
	checkCudaErrors(cudaFree(globalRandState));
  checkCudaErrors(cudaFree(finalImageBuffers));
	checkCudaErrors(cudaFree(frameBuffers));

	cudaDeviceReset();
	return 0;
}