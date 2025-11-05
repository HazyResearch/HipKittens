#include "kittens.cuh"
#include "utils.cpp"
#include <random>
#include <chrono>

constexpr int NUM_DEVICES = 8;
constexpr size_t N = 4096;

using namespace kittens;

using global_layout = gl<bf16, 1, 1, -1, -1>;
using pgl_m = pgl_manager_amd<global_layout>;
using PGL = pgl_amd<global_layout>;


__global__ void all_reduce_tile(float* dst, PGL input, int nelem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nelem) return;

    float acc = 0.0f;
    for (int dev = 0; dev < PGL::num_devices; ++dev) {
        acc += __bfloat162float(input.ptrs[dev][idx]);  // or use ptr_at()
    }
    dst[idx] = acc;
}


int main() {

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    hipEvent_t start[NUM_DEVICES], stop[NUM_DEVICES];
    for (int dev = 0; dev < NUM_DEVICES; ++dev) {
        hipSetDevice(dev);
        hipEventCreate(&start[dev]);
        hipEventCreate(&stop[dev]);
    }

    // Setup
    int nelem = N * N;
    size_t size = nelem * sizeof(bf16);

    // Allocate and initialize host memory
    float **host_mats = new float*[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        host_mats[dev_idx] =  new float[nelem];
        for (int i = 0; i < nelem; ++i) host_mats[dev_idx][i] = dis(gen);
    }
    bf16 **host_mats_bf16 = new bf16*[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        host_mats_bf16[dev_idx] = new bf16[nelem];
        for (int i = 0; i < nelem; ++i)
            host_mats_bf16[dev_idx][i] = __float2bfloat16(host_mats[dev_idx][i]);
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    float *expected = new float[nelem];
    for (int i = 0; i < nelem; ++i) {
        expected[i] = 0.0f;
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx)
            expected[i] += host_mats[dev_idx][i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();

    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) { // Print data
        std::cout << "Device " << dev_idx << ": ";
        for (int i = 0; i < std::min(nelem, 10); ++i) {
            std::cout << host_mats[dev_idx][i] << " ";
        }
        std::cout << "... (" << nelem << " elements)" << std::endl;
    }
    std::cout << "Expected: ";
    for (int i = 0; i < std::min(nelem, 10); ++i) {
        std::cout << expected[i] << " ";
    }
    std::cout << "... (" << nelem << " elements)" << std::endl;
    // Done with setup. 


    // Start kernel.
    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;

    float** dev_output = new float*[NUM_DEVICES];
    for (int dev = 0; dev < NUM_DEVICES; ++dev) {
        hipSetDevice(dev);
        hipMallocManaged(&dev_output[dev], nelem * sizeof(float));
        hipMemset(dev_output[dev], 0, nelem * sizeof(float));
        hipMemAdvise(dev_output[dev], nelem * sizeof(float), hipMemAdviseSetAccessedBy, dev);
    }

    bf16** dev_mats = new bf16*[NUM_DEVICES];
    for (int dev = 0; dev < NUM_DEVICES; ++dev) { // Allocate and fill those device-visible bf16* UVM buffers 
        hipSetDevice(dev);
        hipMallocManaged(&dev_mats[dev], size);
        hipMemAdvise(dev_mats[dev], size, hipMemAdviseSetAccessedBy, dev);
        hipMemcpy(dev_mats[dev], host_mats_bf16[dev], size, hipMemcpyHostToDevice);
    }

    pgl_m dev_mat_pgl(device_ids, dev_mats,
        nullptr, nullptr,                    
        kittens::ducks::gl::make_arg_t<-1>(N),  
        kittens::ducks::gl::make_arg_t<-1>(N));

    dim3 block(256);
    dim3 grid((nelem + block.x - 1) / block.x);
    for (int dev = 0; dev < NUM_DEVICES; ++dev) {
        hipSetDevice(dev);
        hipEventRecord(start[dev]);
        all_reduce_tile<<<grid, block>>>(
            dev_output[dev], dev_mat_pgl.get_pgl_obj(dev), nelem);
        hipEventRecord(stop[dev]);
    }

    
    for (int dev = 0; dev < NUM_DEVICES; ++dev) {
        hipSetDevice(dev);
        hipEventSynchronize(stop[dev]);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start[dev], stop[dev]);
        std::cout << "Device " << dev << " kernel time: " << ms << " ms" << std::endl;
    }

    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    std::cout << "[CPU] All-reduce time: " << cpu_time.count() << " ms" << std::endl;

    // Convert back to float and verify
    int device_id = 1;
    float* result_host = new float[nelem];
    hipMemcpy(result_host, dev_output[device_id], nelem * sizeof(float), hipMemcpyDeviceToHost);
    float TOL = 1e-1;
    for (int i = 0; i < nelem; ++i) {
        float val = result_host[i];
        if (fabs(val - expected[i]) > TOL) {
            std::cerr << "Mismatch at index " << i << ": got " << val
                    << ", expected " << expected[i] << std::endl;
            exit(EXIT_FAILURE);
        }

        if (i < 3) { 
            std::cout << "Expected: " << expected[i] << ", Got: " << val << std::endl;
        }
    }
    std::cout << "All-reduce succeeded. " << "Device: " << device_id << std::endl;


    // Cleanup and exit
    for (int dev = 0; dev < NUM_DEVICES; ++dev) {
        hipEventDestroy(start[dev]);
        hipEventDestroy(stop[dev]);
    }

    for (int dev = 0; dev < NUM_DEVICES; ++dev) {
        hipFree(dev_mats[dev]);
        hipFree(dev_output[dev]);
    }
    delete[] dev_mats;
    delete[] dev_output;
    
    for (int dev = 0; dev < NUM_DEVICES; ++dev) {
        delete[] host_mats[dev];
        delete[] host_mats_bf16[dev];
    }
    delete[] host_mats;
    delete[] host_mats_bf16;
    delete[] expected;
    delete[] result_host;

    std::cout << "Done!" << std::endl;
    return 0;
}

