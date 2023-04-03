#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>

using std::vector;

#define BLOCK_SIZE 1024

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                                        \
 cudaError_t e=cudaGetLastError();                                                \
 if(e!=cudaSuccess) {                                                             \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));  \
   exit(0);                                                                       \
 }                                                                                \
}

__global__ void powKernel(double *dev_vect, int N) {
  // Глобольный тред id
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) {
        dev_vect[tid] = dev_vect[tid] * dev_vect[tid];
    }
}


void testCpu( vector<double> &vect, int N){
    for (int i = 0; i<N; i++){
        vect[i] = vect[i] * vect[i];
    }
}

int main() {
    int N;
    std::cin>>N;

    using clock = std::chrono::system_clock;
    using sec = std::chrono::duration<double, std::milli>;

    if (N>pow(2,25)){
        fprintf(stderr,"ERROR: N is too big");
        return 0;
    }

    size_t bytes = sizeof(double) * N;

    vector<double> host_vect;
    vector<double> host_vect_out(N);
    double num;

    while (std::cin >> num) {
        // add number to vector
        host_vect.push_back(num);
    }

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double *dev_vect;
    cudaMalloc(&dev_vect, bytes);

    cudaMemcpy(dev_vect, host_vect.data(), bytes, cudaMemcpyHostToDevice);
    cudaCheckError();


    cudaEvent_t start,stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    //run kernel
    powKernel<<<1024, 1024>>>(dev_vect, N);
    cudaCheckError();

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Уничтожаем созданные эвенты
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(host_vect_out.data(), dev_vect, bytes, cudaMemcpyDeviceToHost);
    cudaCheckError();

    for (int i = N-5; i<N; i++)
        printf("%.10e ",host_vect_out[i]);

    const auto before = clock::now();

    testCpu(host_vect,N);

    const sec duration = clock::now() - before;

    std::cout << std::endl;
    std::cout << "Time elapsed on GPU: " << gpuTime << "ms" << std::endl;
    std::cout << "Time elapsed on CPU: " << duration.count() << "ms" << std::endl;

    cudaFree(dev_vect);

    return 0;
}