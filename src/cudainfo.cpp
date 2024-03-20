#include <iostream>
#include <cuda_runtime_api.h>
#include <thread>
#include <chrono>
#include <string>
void showcudainfo(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << " (" << deviceProp.name << "):" << std::endl;
        std::cout << "  Device name: " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Total constant memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Maximum threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Maximum block dimensions: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Maximum grid dimensions: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate * 1e-3 << " MHz" << std::endl;
        std::cout << "  Memory clock rate: " << deviceProp.memoryClockRate * 1e-3 << " MHz" << std::endl;
        std::cout << "  Memory bus width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "  L2 cache size: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << std::endl;
    }
    std::string s;
    std::cin >>s;
}

int main() {

    showcudainfo();
    return 0;
}