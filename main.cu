#include "camera.cuh"
#include "hittable/hittable_list.cuh"
#include "hittable/shape/sphere.cuh"
#include "material/lambertian.cuh"

#include <iostream>
#include <fstream>
#include <limits>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") at " <<
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
        if (world[0]->hit(curRay, 0.001f, FLT_MAX, hit, mat)) {
            if (mat.material->scatter(curRay, hit, scat, randState)) {
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
    color[0] = sqrt(color[0]);
    color[1] = sqrt(color[1]);
    color[2] = sqrt(color[2]);

    int pixelIndex = i + j * width;
    finalImageBuffers[pixelIndex] = color;
}

__global__
void createWorld(Camera** cam, Hittable** hits, Material** mats, Hittable** world) {
    mats[0] = new Lambertian(Arr3(0.7f, 0.3f, 0.3f));
    mats[1] = new Lambertian(Arr3(0.8f, 0.8f, 0.0f));
    mats[2] = new Lambertian(Arr3(0.1f, 0.2f, 0.5f));
    hits[0] = new Sphere(Arr3(0.0f, 0.0f, -1.0f), 0.5f, mats[0]);
    hits[1] = new Sphere(Arr3(0.0f, -100.5f, -1.0f), 100.0f, mats[1]);
    hits[2] = new Sphere(Arr3(1.0f, 0.0f, -1.0f), 0.5f, mats[2]);
    world[0] = new HittableList(hits, 3);
    cam[0] = new Camera(Arr3(-2.0f, 2.0f, 1.0f), Arr3(0.0f, 0.0f, -1.0f), Arr3(0.0f, 1.0f, 0.0f), 40.0f, 1.0f, 0.0f, 10.0f);
}

__global__
void freeWorld(Camera** cam, Hittable** hits, Material** mats, Hittable** world) {
    delete cam[0];
    delete world[0];
    delete hits[0];
    delete hits[1];
    delete hits[2];
    delete mats[0];
    delete mats[1];
    delete mats[2];
}

__global__
void render_init(int max_x, int max_y, int max_z, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= max_x || j >= max_y || k >= max_z) return;
    int pixel_index = k + i * max_z + j * max_x * max_z;

    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

int main() {
    const int imageWidth = 1200;
    const int imageHeight = 1200;
    const int samplePerPixel = 32;

    int tx = 8;
    int ty = 8;
    int tz = 8;

    Arr3* frameBuffers;
    Arr3* finalImageBuffers;
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
    checkCudaErrors(cudaMalloc((void**)&camera, sizeof(Camera*)));
    checkCudaErrors(cudaMalloc((void**)&hits, 3 * sizeof(Hittable*)));
    checkCudaErrors(cudaMalloc((void**)&mats, 3 * sizeof(Material*)));
    checkCudaErrors(cudaMalloc((void**)&world, sizeof(Hittable*)));

    createWorld<<<1, 1>>> (camera, hits, mats, world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "\rrendering" << std::flush;

    dim3 blocks(imageWidth / tx, imageHeight / ty, samplePerPixel / tz);
    dim3 threads(tx, ty, tz);

    render_init<<<blocks, threads>>>(imageWidth, imageHeight, samplePerPixel, pixelRandState);
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
    checkCudaErrors(cudaFree(frameBuffers));

    cudaDeviceReset();
    return 0;
}