#include <CL/cl2.hpp>
#include <iostream>
#include <vector>

int main() {
    // 获取平台列表
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // 遍历平台
    for (const auto& platform : platforms) {
        std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        // 获取设备列表
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // 遍历设备
        for (const auto& device : devices) {
            std::cout << "    Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "      Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "      Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
            std::cout << "      Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "      Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz" << std::endl;
            std::cout << "      Global Memory Size: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << " MB" << std::endl;
            std::cout << "      Local Memory Size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB" << std::endl;
        }
    }

    return 0;
}
