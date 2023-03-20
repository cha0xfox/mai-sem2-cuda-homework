#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <vector>

using std::vector;

#define BLOCK_SIZE 1024

__global__ void powKernel(double *dev_vect, int N) {
  // Глобольный тред id
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    printf("test: %i",tid);
    if (tid < N) {
        dev_vect[tid] = pow(dev_vect[tid], dev_vect[tid]);
    }
}

int main() {
    int N;
    std::cin>>N;

    size_t bytes = sizeof(double) * N;

    vector<double> host_vect(N);
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


    cudaEvent_t start,stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    //run kernel
    powKernel<<<num_blocks, BLOCK_SIZE>>>(dev_vect, N);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Уничтожаем созданные эвенты
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(host_vect_out.data(), dev_vect, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; i++)
        std::cout<<host_vect_out[i]<<" ";
    std::cout<<std::endl;

    cudaFree(dev_vect);

    return 0;
}