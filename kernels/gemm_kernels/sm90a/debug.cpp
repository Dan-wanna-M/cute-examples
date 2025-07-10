#include <iostream>
#include <cuda_runtime.h>

// Forward declaration of the debug function
void debug_mbarrier();

int main() {
    std::cout << "Starting CUDA debug_mbarrier test..." << std::endl;
    
    // Check CUDA device availability
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return -1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Call the debug function
    try {
        for(int i=0;i<1000000;i++)
            debug_mbarrier();
        std::cout << "debug_mbarrier completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    // Check for any CUDA errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error after kernel execution: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    // Synchronize to ensure all operations complete
    cudaDeviceSynchronize();
    
    std::cout << "Program completed successfully!" << std::endl;
    return 0;
}