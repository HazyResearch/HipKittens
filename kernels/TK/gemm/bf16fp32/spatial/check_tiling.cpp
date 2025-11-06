#include <hip/hip_runtime.h>
#include <iostream>
#include <iostream>
#include <iomanip>
#include <algorithm> // for std::max
#include <cmath>     // for log10

#include "utils.cpp"

#define HIP_CHECK(call)                                                   \
    {                                                                     \
        hipError_t err = call;                                            \
        if (err != hipSuccess) {                                          \
            std::cerr << "HIP error: " << hipGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";  \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

void printGrid(const int* array, int x, int y) {
    // Determine the max width of any number in the array
    int maxVal = array[0];
    for (int i = 1; i < x * y; ++i)
        maxVal = std::max(maxVal, std::abs(array[i]));

    // Width for formatting (add 1 for negative sign if needed)
    int width = (maxVal > 0 ? static_cast<int>(std::log10(maxVal)) + 1 : 1);
    bool hasNegative = std::any_of(array, array + x * y, [](int n) { return n < 0; });
    if (hasNegative) ++width;

    // Print grid with right-aligned values
    for (int row = 0; row < y; ++row) {
        for (int col = 0; col < x; ++col) {
            std::cout << std::setw(width) << array[row * x + col] << ' ';
        }
        std::cout << '\n';
    }
}


constexpr int BLOCK_SIZE_M      = 128;
constexpr int BLOCK_SIZE_N      = 256;  

#define M 8192
#define K 8192
#define N 8192

// Kernel: each workgroup writes its global workgroup ID into (y,x) location
__global__ void write_wgid_to_matrix(int* matrix) {

    int wgid = blockIdx.y * gridDim.x + blockIdx.x;


    // Swizzle for better L2 within the same XCD.
    const int WGM = 16;
    const int num_pid_m = gridDim.y;
    const int num_pid_n = gridDim.x;

    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;

    // Assign the tile's row/column based on the pid_m and pid_n.
    const int row = pid_m; // blockIdx.x
    const int col = pid_n; // blockIdx.y
    int idx = pid_m * num_pid_n + pid_n;

    // int idx = swizzle_l2_tile(wgid, gridDim.y, gridDim.x, 16, 16);
    // int idx = swizzle_l2_tile_col_major(wgid, gridDim.y, gridDim.x, 16, 16);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        matrix[idx] = wgid;
    }
}

int main() {
    // TODO: Appears to be bug w/ my tiling when width=height=16?
    const int width = 64;
    const int height = 32;
    const int total_elements = width * height;

    int* d_matrix;
    int* h_matrix = new int[total_elements];

    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_matrix, total_elements * sizeof(int)));

    dim3 grid_dim(width, height);   
    dim3 block_dim(1, 1);   

    write_wgid_to_matrix<<<grid_dim, block_dim>>>(d_matrix);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy results back to host
    HIP_CHECK(hipMemcpy(h_matrix, d_matrix, total_elements * sizeof(int), hipMemcpyDeviceToHost));

    printGrid(h_matrix, width, height);

    // Cleanup
    HIP_CHECK(hipFree(d_matrix));
    delete[] h_matrix;

    return 0;
}
