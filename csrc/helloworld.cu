#include <iostream>

__global__ void cuda_helloworld(){
    printf("Hello World from GPU!\n");
}

int main(int argc, char* argv[]){
    cuda_helloworld <<<1,1>>> ();
    cudaDeviceSynchronize();  // Synchronize to flush the GPU's console buffer
    return 0;
}
