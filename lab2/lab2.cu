#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 1024

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                                        \
cudaError_t e=cudaGetLastError();                                                \
if(e!=cudaSuccess) {                                                             \
    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));  \
    exit(0);                                                                       \
}                                                                                \
}

__global__ void mykernel(uchar4* out_dev, int w, int h, cudaTextureObject_t textureObject) {
    // global thread ids
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h)
    {
        int gx = 0, gy = 0;

        uchar4 p00 = tex2D<uchar4>(textureObject, x, y);
        uchar4 p10 = tex2D<uchar4>(textureObject, x + 1, y);
        uchar4 p01 = tex2D<uchar4>(textureObject, x, y + 1);
        uchar4 p11 = tex2D<uchar4>(textureObject, x + 1, y + 1);

        gx = p00.x - p11.x;
        gy = p01.x - p10.x;
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        magnitude = fminf(fmaxf(magnitude, 0.0f), 255.0f);

        int offset = y * w + x;
        out_dev[offset] = make_uchar4((unsigned char)magnitude, (unsigned char)magnitude, (unsigned char)magnitude, 0);
    }

}



int main() {

    // file names
    char* infile = new char[256];
    char* outfile = new char[256];

    // read file names
    std::cin >> infile;
    std::cin >> outfile;

    // image param read
    int w,h;
    FILE* File = fopen(infile, "rb");
    fread(&w, sizeof(int), 1, File);
    fread(&h, sizeof(int), 1, File);
    if (w >= 655536 || h >= 655536 || w * h > pow(10,8)) {
        fprintf(stderr, "Dimensions too big\n");
        return 1;
    }


    printf("check 1\n");
    uchar4* imagemem = (uchar4*)malloc(sizeof(uchar4) * w * h);
    uchar4* imagemem_out = (uchar4*)malloc(sizeof(uchar4) * w * h);
    
    fread(imagemem, sizeof(uchar4), w * h, File);
    fclose(File);

    // init cuda memory
    cudaArray_t imagemem_dev;
    cudaChannelFormatDesc channel = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&imagemem_dev, &channel, w, h);
    cudaMemcpy2DToArray(imagemem_dev, 0,0, imagemem, sizeof(uchar4) * w, sizeof(uchar4) * w, h, cudaMemcpyHostToDevice);

    printf("check 2\n");
    // textures
    struct cudaResourceDesc resourceD;
    memset(&resourceD, 0, sizeof(resourceD));
    resourceD.resType = cudaResourceTypeArray;
    resourceD.res.array.array = imagemem_dev;
    // texture coords
    struct cudaTextureDesc textureD;
    memset(&textureD, 0, sizeof(textureD));
    textureD.normalizedCoords = false;
    textureD.filterMode = cudaFilterModePoint;
    textureD.addressMode[0] = cudaAddressModeClamp;
    textureD.addressMode[1] = cudaAddressModeClamp;

    printf("check 3\n");
    // texObject
    cudaTextureObject_t texObject = 0;
    cudaCreateTextureObject(&texObject, &resourceD, &textureD, NULL);

    // run kernel

    printf("check 4\n");
    uchar4 *out_dev;
    cudaMalloc(&out_dev, sizeof(uchar4) * w * h);
    dim3 blockDim(16, 16);
    dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);

    printf("check 5\n");
    mykernel <<<gridDim, blockDim>>> (out_dev, w, h, texObject);
    cudaCheckError();

    // write to host

    cudaMemcpy(imagemem_out, out_dev, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost);

    // write to file 
    printf("check 6\n");
    File = fopen(outfile, "wb");
    fwrite(&w, sizeof(int), 1, File);
    fwrite(&h, sizeof(int), 1, File);
    fwrite(imagemem_out, sizeof(uchar4), w * h, File);
    fclose(File);


    printf("check 7\n");
    // clear
    cudaDestroyTextureObject(texObject);
    cudaFree(out_dev);
    cudaFreeArray(imagemem_dev);
    free(imagemem_out);

    return 0;
}