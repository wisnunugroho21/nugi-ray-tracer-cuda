#include "camera.cuh"
#include "hittable/hittable_list.cuh"
#include "hittable/shape/sphere.cuh"
#include "hittable/shape/moving_sphere.cuh"
#include "hittable/bvh.cuh"
#include "material/lambertian.cuh"
#include "material/metal.cuh"
#include "material/dielectric.cuh"

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
Arr3 tracing(const Ray& r, Hittable** world, curandState* randState) {
	Arr3 curAttenuation = Arr3(1.0f, 1.0f, 1.0f);
	Ray curRay = r;

	HitRecord hit;
	ScatterRecord scat;
	MaterialRecord mat;

	for (size_t i = 0; i < 50; i++) {
		if (world[0]->hit(curRay, 0.001f, FLT_MAX, &hit, &mat)) {
			if (mat.material->scatter(curRay, hit, &scat, randState)) {
				curAttenuation *= scat.colorAttenuation;
				curRay = scat.newRay;
			}
			else {
				return Arr3(0.0f, 0.0f, 0.0f);
			}
		}
		else {
			auto unitDirection = r.direction().unitVector();
			auto t = 0.5f * (unitDirection.y() + 1.0f);
			auto color = (1.0f - t) * Arr3(1.0f, 1.0f, 1.0f) + t * Arr3(0.5f, 0.7f, 1.0f);

			return curAttenuation * color;
		}
	}

	return Arr3(0.0f, 0.0f, 0.0f);
}

__global__
void render(Arr3* frameBuffer, int width, int height, int nSample, Camera** cam, Hittable** world, curandState* randState) {
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
	color += tracing(r, world, &localRandState);

	frameBuffer[pixelIndex] = color;
}

__global__
void sampling(Arr3* frameBuffers, Arr3* finalImageBuffers, int width, int height, int nSample) {
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
void createWorld(Camera** cam, Hittable** hits, Material** mats, Hittable** world, curandState* randState) {
	auto localRandState = randState[0];

	mats[0] = new Lambertian(Arr3(0.5f, 0.5f, 0.5f));
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

	world[0] = BvhNode::build(hits, objIndex + 3, 0.0f, 0.0f, &localRandState);

	Arr3 lookfrom(13.0f, 2.0f, 3.0f);
	Arr3 lookat(0.0f, 0.0f, 0.0f);
	Arr3 vup(0.0f, 1.0f, 0.0f);
	auto dist_to_focus = 10.0f;
	auto aperture = 0.1f;
	auto aspect_ratio = 1.0f;

	cam[0] = new Camera(lookfrom, lookat, vup, 60.0f, aspect_ratio, aperture, dist_to_focus);
}

__global__
void freeWorld(Camera** cam, Hittable** hits, Material** mats, Hittable** world) {
	delete cam[0];
	delete world[0];

	int numObjects = 22 * 22 + 3 + 1;

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

int main() {
	const int imageWidth = 1200;
	const int imageHeight = 1200;
	const int samplePerPixel = 32;

	int tx = 8;
	int ty = 8;
	int tz = 8;

	int numObjects = 22 * 22 + 3 + 1;

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

	checkCudaErrors(cudaMallocManaged((void**)&frameBuffers, fb_size));
	checkCudaErrors(cudaMallocManaged((void**)&finalImageBuffers, finalImage_size));
	checkCudaErrors(cudaMalloc((void**)&pixelRandState, pixel_rand_size));
	checkCudaErrors(cudaMalloc((void**)&globalRandState, sizeof(curandState)));
	checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Camera*)));
	checkCudaErrors(cudaMalloc((void**)&hits, numObjects * sizeof(Hittable*)));
	checkCudaErrors(cudaMalloc((void**)&mats, numObjects * sizeof(Material*)));
	checkCudaErrors(cudaMalloc((void**)&world, sizeof(Hittable*)));

	initGlobalRand<<<1, 1>>>(globalRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	createWorld<<<1, 1>>>(camera, hits, mats, world, globalRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::cerr << "\rrendering" << std::flush;

	dim3 blocks(imageWidth / tx, imageHeight / ty, samplePerPixel / tz);
	dim3 threads(tx, ty, tz);

	initPixelRand<<<blocks, threads>>>(imageWidth, imageHeight, samplePerPixel, pixelRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render<<<blocks, threads>>>(frameBuffers, imageWidth, imageHeight, samplePerPixel, camera, world, pixelRandState);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	blocks = dim3(imageWidth / tx, imageHeight / ty);
	threads = dim3(tx, ty);

	std::cerr << "\rsampling" << std::flush;

	sampling<<<blocks, threads>>>(frameBuffers, finalImageBuffers, imageWidth, imageHeight, samplePerPixel);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::cerr << "\rwriting" << std::flush;

	std::ofstream ofl("image.ppm");
	ofl << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

	for (int j = imageHeight - 1; j >= 0; j--) {
		for (int i = 0; i < imageWidth; i++) {
			int pixelIndex = i + j * imageWidth;
			writeColor(ofl, finalImageBuffers[pixelIndex]);
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());
	freeWorld<<<1, 1>>>(camera, hits, mats, world);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(camera));
	checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(mats));
	checkCudaErrors(cudaFree(hits));
	checkCudaErrors(cudaFree(pixelRandState));
	checkCudaErrors(cudaFree(globalRandState));
	checkCudaErrors(cudaFree(frameBuffers));

	cudaDeviceReset();
	return 0;
}