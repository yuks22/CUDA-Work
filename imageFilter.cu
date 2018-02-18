#include <stdio.h>

// A macro for checking the error codes of cuda runtime calls
#define CUDA_ERROR_CHECK(expr) \
  {                            \
    cudaError_t err = expr;    \
    if (err != cudaSuccess)    \
    {                          \
      printf("CUDA call failed!\n%s\n", cudaGetErrorString(err)); \
      exit(1);                 \
    }                          \
  }


__global__
void swapChannel_kernel(uchar3 * device_inputImage, uchar3 * device_outputImage, int rows, int cols)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  device_outputImage[idx].y = device_inputImage[idx].x;
  device_outputImage[idx].x = device_inputImage[idx].y;
  device_outputImage[idx].z = device_inputImage[idx].z;


}

__global__
void blurImage_kernel(uchar3 * device_inputImage, uchar3 * device_outputImage, int rows, int cols)
{

  int idx = blockIdx.x * blockDim.x  + threadIdx.x;
  int tIdx;

  float sumX, sumY, sumZ;
  sumX = sumY = sumZ = 0.0;

  float spx = 0.0;

  for(int x = -4; x <= 4; x++ ){
    for(int y = -4; y <= 4; y++ ){
      if(((blockIdx.x + x) >= 0) && ((blockIdx.x + x) < 512) && ((threadIdx.x + y) >= 0) && ((threadIdx.x + y) < 512)){

        tIdx = ((blockIdx.x + x) * blockDim.x + threadIdx.x + y);

        sumX += device_inputImage[idx].x;
        sumY += device_inputImage[tIdx].y;
        sumZ += device_inputImage[tIdx].z;

        spx++;
      }
    }
  }

  device_outputImage[idx].x = sumX / spx;
  device_outputImage[idx].y = sumY / spx;
  device_outputImage[idx].z = sumZ / spx;

}

__global__
void inplaceFlip_kernel(uchar3 * device_outputImage, int rows, int cols)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ uchar3 s[511];

  int tr = 511-threadIdx.x;

  s[threadIdx.x] = device_outputImage[idx];

  __syncthreads();

  device_outputImage[idx] = s[tr];


}

__global__
void creative_kernel(uchar3 * device_inputImage, uchar3 * device_outputImage, int rows, int cols)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tIdx;
  float sumY = 0.0;
  float spx = 0.0;

  if ((blockIdx.x >= 0) && (blockIdx.x < 96) && (threadIdx.x >= 0) && (threadIdx.x < 96)){

    for(int x = -4; x <= 4; x++ ){
      for(int y = -4; y <= 4; y++ ){
        if(((blockIdx.x + x) >= 0) && ((blockIdx.x + x) < 512) && ((threadIdx.x + y) >= 0) && ((threadIdx.x + y) < 512)){

          tIdx = ((blockIdx.x + x) * blockDim.x + threadIdx.x + y);
          sumY += device_inputImage[tIdx].y;
          spx++;

        }
      }
    }

    device_outputImage[idx].x = 0.75 * sumY / spx;
    device_outputImage[idx].y = 0.75 * sumY / spx;
    device_outputImage[idx].z = 0.75 * sumY / spx;

  }

  if((blockIdx.x >= 96) && (blockIdx.x < 448) && (threadIdx.x >= 96) && (threadIdx.x < 448)){

    device_outputImage[idx].y = device_inputImage[idx].y;
    device_outputImage[idx].x = device_inputImage[idx].y;
    device_outputImage[idx].z = device_inputImage[idx].y;

  }

  else{

    tIdx = 0;
    sumY = 0.0;
    spx = 0.0;

    for(int x = -4; x <= 4; x++ ){
      for(int y = -4; y <= 4; y++ ){
        if(((blockIdx.x + x) >= 0) && ((blockIdx.x + x) < 512) && ((threadIdx.x + y) >= 0) && ((threadIdx.x + y) < 512)){

          tIdx = ((blockIdx.x + x) * blockDim.x + threadIdx.x + y);
          sumY += device_inputImage[tIdx].y;
          spx++;

        }
      }
    }

    device_outputImage[idx].x = 0.75 * sumY / spx;
    device_outputImage[idx].y = 0.75 * sumY / spx;
    device_outputImage[idx].z = 0.75 * sumY / spx;

  }

}



__host__
float filterImage(uchar3 *host_inputImage, uchar3 *host_outputImage, int rows, int cols, int filterNumber){

  int numPixels = rows * cols;

  //allocate memory on device (GPU)
  uchar3 *device_inputImage;
  uchar3 *device_outputImage;

  CUDA_ERROR_CHECK(cudaMalloc(&device_inputImage, sizeof(uchar3) * numPixels));
  CUDA_ERROR_CHECK(cudaMalloc(&device_outputImage, sizeof(uchar3) * numPixels));
  CUDA_ERROR_CHECK(cudaMemset(device_outputImage, 0,  sizeof(uchar3) * numPixels)); //make sure no memory is left laying around

  //copy input image to the device (GPU)
  CUDA_ERROR_CHECK(cudaMemcpy(device_inputImage, host_inputImage, sizeof(uchar3) * numPixels, cudaMemcpyHostToDevice));

  //start timing to measure length of kernel call
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  int gridSize = 512;
  int blockSize = 512;

  switch(filterNumber){
    case 1:
      swapChannel_kernel<<<gridSize,blockSize>>>(device_inputImage, device_outputImage, rows, cols);
      break;
    case 2:
      blurImage_kernel<<<gridSize,blockSize>>>(device_inputImage, device_outputImage, rows, cols);
      break;
    case 3:
      inplaceFlip_kernel<<<gridSize,blockSize>>>(device_inputImage, rows, cols);
      break;
    case 4:
      creative_kernel<<<gridSize,blockSize>>>(device_inputImage, device_outputImage, rows, cols);
      break;
    default:
      break;
  }

  //----------------------------------------------------------------
  // END KERNEL CALLS - Do not modify code beyond this point!
  //----------------------------------------------------------------

  //stop timing
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float timeElapsedInMs = 0;
  cudaEventElapsedTime(&timeElapsedInMs, start, stop);

  //synchronize
  cudaDeviceSynchronize(); CUDA_ERROR_CHECK(cudaGetLastError());

  //copy device output image back to host output image
  //special case for filter swap - since it is in place, we actually copy the input image back to the host output
  if (filterNumber==3){
    CUDA_ERROR_CHECK(cudaMemcpy(host_outputImage, device_inputImage, sizeof(uchar3) * numPixels, cudaMemcpyDeviceToHost));
  }else{
    CUDA_ERROR_CHECK(cudaMemcpy(host_outputImage, device_outputImage, sizeof(uchar3) * numPixels, cudaMemcpyDeviceToHost));
  }


  //free Memory
  CUDA_ERROR_CHECK(cudaFree(device_inputImage));
  CUDA_ERROR_CHECK(cudaFree(device_outputImage));

  return timeElapsedInMs;
}
