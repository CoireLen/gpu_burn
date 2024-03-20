cmake_minimum_required(VERSION 3.10.0)
project(CDC VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV2.4)
include_directories(gpu_burn_cu "D:/Lib/vcpkg/installed/x64-windows/include/opencv2.4" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include")
link_directories(gpu_burn  "D:/Lib/vcpkg/buildtrees/opencv2/x64-windows-rel/lib" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64")

add_library(gpu_burn_cu ./src/compare.cu)
set_target_properties( gpu_burn_cu
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(gpu_burn ./src/compare.cu)
#target_include_directories(gpu_burn PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include")
#target_link_directories(gpu_burn PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64")
#target_include_directories(gpu_burn PRIVATE "C:/Users/Coire/.conan2/p/opencc96a95d38232e/p/include")
#target_link_directories()
target_link_libraries(gpu_burn PRIVATE  cuda cudart gpu_burn_cu opencv_core2 opencv_highgui2
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
