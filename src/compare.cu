#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>

#define MAX_ITER 1000

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
// CUDA kernel to generate Julia set
__global__ void generateJuliaSet(uchar3 *image, double c_real, double c_imag, double zoom, int offsetX, int offsetY,int width,int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    double2 z, z_new;
    z.x = -2.0f + 4.0f * (x - offsetX) / width / zoom;
    z.y = -1.5f + 3.0f * (y - offsetY) / height / zoom;

    int iteration = 0;
    while (z.x * z.x + z.y * z.y <= 4.0f && iteration < MAX_ITER) {
        z_new.x = z.x * z.x - z.y * z.y + c_real;
        z_new.y = 2.0f * z.x * z.y + c_imag;
        z = z_new;
        iteration++;
    }

    // Color mapping based on iteration count
    image[index].x = 255 * (iteration % 256);
    image[index].y = 255 * ((iteration / 8) % 256);
    image[index].z = 255 * ((iteration / 64) % 256);
}

int main(int argc,char **argv) {
    // Allocate memory for image
    int WIDTH =1920;
    int HEIGHT =1080;
    if (argc==3){
        WIDTH=atoi(argv[1]);
        HEIGHT=atoi(argv[2]);
    }
    uchar3 *h_image;
    cudaMallocHost(&h_image, WIDTH * HEIGHT * sizeof(uchar3));
    uchar3 *d_image;
    cudaMalloc(&d_image, WIDTH * HEIGHT * sizeof(uchar3));

    // Create OpenCV window
    cv::namedWindow("Julia Set", cv::WINDOW_NORMAL);
    cv::resizeWindow("Julia Set", WIDTH, HEIGHT);

    // Julia set parameters
    double c_real = -0.7f;
    double c_imag = 0.27f;
    double zoom = 1.0f;
    int offsetX = WIDTH / 2;
    int offsetY = HEIGHT / 2;
    int i=0;
    long int copymin=9999999,copymax=0,rtmin=9999999,rtmax=0;
    double allcopytime=0,allrttime=0;
    auto start = std::chrono::system_clock::now();
    while (i++<1000) {
        // Launch kernel to generate Julia set
        dim3 blockSize(16, 16);
        dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

        auto rtstart = std::chrono::system_clock::now();
        generateJuliaSet<<<gridSize, blockSize>>>(d_image, c_real, c_imag, zoom, offsetX, offsetY,WIDTH,HEIGHT);
        cudaDeviceSynchronize();
        auto rtend=  std::chrono::system_clock::now();
        // Copy result back to host
        auto copystart = std::chrono::system_clock::now();
        cudaMemcpy(h_image, d_image, WIDTH * HEIGHT * sizeof(uchar3), cudaMemcpyDeviceToHost);
        auto copyend = std::chrono::system_clock::now();
        long int copytime=std::chrono::duration_cast<std::chrono::microseconds>(copyend - copystart).count();
        long int rttime=std::chrono::duration_cast<std::chrono::microseconds>(rtend - rtstart).count();
        std::cout<<"frame:"<<i  <<"    copytime:"<<copytime
        <<"ms"<<
        "   runtime:"<<rttime<<"ms"<<
        std::endl;
        //计算最大最小
        copymax=std::max(copymax,copytime);
        copymin=std::min(copymin,copytime);
        rtmax=std::max(rtmax,rttime);
        rtmin=std::min(rtmin,rttime);
        allcopytime+=copytime;
        allrttime+=rttime;
        // Display image
        cv::Mat frame(HEIGHT, WIDTH, CV_8UC3, h_image);
        cv::imshow("Julia Set", frame);

        // Check for ESC key
        if (cv::waitKey(1) == 27) {
            break;
        }
        // Update zoom factor
        zoom *= 1.01f; // Increase zoom by 1%
        offsetX = -(905.0/1920*WIDTH)*zoom;
        offsetY = -(620.0/1080*HEIGHT)*zoom;
    }
    auto end = std::chrono::system_clock::now();

    std::cout<<"ALL Time Run in"<<WIDTH<<"x"<<HEIGHT<<":"<<
    double(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())*std::chrono::microseconds::period::num / std::chrono::microseconds::period::den 
    <<"s   Copy Time(min,max):"<<copymin<<","<<copymax<<"ms   RunTime(min,max)"<<rtmin<<","<<rtmax<<"ms"<<
    "    ALL copy time:" <<allcopytime*std::chrono::microseconds::period::num / std::chrono::microseconds::period::den <<"s"
    <<"   ALL RunTime:"<<allrttime*std::chrono::microseconds::period::num / std::chrono::microseconds::period::den <<"s"<<
    std::endl;
    showcudainfo();
    // Release resources
    cudaFree(d_image);
    cudaFreeHost(h_image);
    cv::destroyAllWindows();

    return 0;
}
