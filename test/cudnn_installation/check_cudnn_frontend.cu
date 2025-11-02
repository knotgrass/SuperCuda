#include <iostream>
#include <cudnn.h>
#include <cudnn_frontend.h>

// Try to include cuDNN frontend header
#ifdef __has_include
#  if __has_include(<cudnn_frontend.h>)
#    include <cudnn_frontend.h>
#    define HAS_CUDNN_FRONTEND 1
#  else
#    define HAS_CUDNN_FRONTEND 0
#  endif
#else
#  include <cudnn_frontend.h>
#  define HAS_CUDNN_FRONTEND 1
#endif

int main() {
    std::cout << "=== Checking cuDNN and cuDNN-frontend Installation ===" << std::endl;

    // Check cuDNN version
    size_t cudnn_version = cudnnGetVersion();
    std::cout << "cuDNN Version: " << cudnn_version << std::endl;

    // Check cuDNN-frontend
#if HAS_CUDNN_FRONTEND
    std::cout << "cuDNN-frontend: INSTALLED" << std::endl;
    std::cout << "cuDNN-frontend Version: " << CUDNN_FRONTEND_VERSION << std::endl;

    // Try to create a simple cuDNN handle using frontend
    try {
        cudnnHandle_t handle;
        cudnnStatus_t status = cudnnCreate(&handle);

        if (status == CUDNN_STATUS_SUCCESS) {
            std::cout << "cuDNN handle created successfully!" << std::endl;
            cudnnDestroy(handle);
        } else {
            std::cout << "Failed to create cuDNN handle: " << status << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
#else
    std::cout << "cuDNN-frontend: NOT FOUND" << std::endl;
#endif

    std::cout << "=== Check Complete ===" << std::endl;
    return 0;
}
