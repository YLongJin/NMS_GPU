#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "NvInfer.h"
#include "cuda_fp16.h"

//yolo layer
inline __device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void gpuYoloLayerV3(float *output, float* input, const unsigned int gridSize, const unsigned int numOutputClasses,
                               const unsigned int numBBoxes)
{
    unsigned int x_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_id = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z_id = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x_id >= gridSize) || (y_id >= gridSize) || (z_id >= numBBoxes))
    {
        return;
    }

    const int numGridCells = gridSize * gridSize;
    const int bbindex = y_id * gridSize + x_id;

	output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]);

	output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]);

	output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]);

	output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]
        = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]);

	output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]
        = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]);

    for (unsigned int i = 0; i < numOutputClasses; ++i)
    {
		output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]
            = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))]);
    }
}

cudaError_t cudaYoloLayerV3(void* output, const void* input, const unsigned int& batchSize, const unsigned int& gridSize,
                            const unsigned int& numOutputClasses, const unsigned int& numBBoxes,
                            uint64_t outputSize, cudaStream_t stream)
{
    dim3 threads_per_block(16, 16, 4);
    dim3 number_of_blocks((gridSize / threads_per_block.x) + 1,
                          (gridSize / threads_per_block.y) + 1,
                          (numBBoxes / threads_per_block.z) + 1);
    for (int batch = 0; batch < batchSize; ++batch)
    {
        gpuYoloLayerV3<<<number_of_blocks, threads_per_block, 0, stream>>>((float*)(output) + (batch * outputSize),
			(float*)(input) + (batch * outputSize), gridSize, numOutputClasses,
            numBBoxes);
    }
    return cudaGetLastError();
}

//mish layer

inline __device__ float softplus(const float &x)
{
	if (x > 20)
		return x;
	else if (x < -20)
		return __expf(x);
	else
		return __logf(__expf(x) + 1);
}

inline __device__ float mish(const float &x)
{
	return x * tanh(softplus(x));
}

__global__ void gpuMish(float*output, float* input, int channel, int height, int width)
{
	int w_id = blockIdx.x * blockDim.x + threadIdx.x;
	int h_id = blockIdx.y * blockDim.y + threadIdx.y;
	int c_id = blockIdx.z;
	int b_id = threadIdx.z;
	if (c_id >= channel || h_id >= height || w_id >= width || b_id >= 16)
	{
		return;
	}
	int index = b_id * channel* height * width + c_id * height * width + h_id * width + w_id;
	input[index] = mish(input[index]);
	//output[index] = input[index] * tanh(__logf(__expf((input[index]) + 1)));
}

cudaError_t cudaMishLayer(void *output, const void* input, const unsigned int& batchSize, const unsigned int& channel,
	const unsigned int& height, const unsigned int& width,
	uint64_t outputSize, cudaStream_t stream)
{
	
	dim3 threads_per_block(8, 8, 16);
	dim3 number_of_blocks((width / threads_per_block.x + 1),
		(height / threads_per_block.y + 1),
		channel);
	gpuMish << <number_of_blocks, threads_per_block, 0, stream >> > ((float*)(output), (float*)(input), channel, height, width);
	//dim3 threads_per_block(8, 8, 8);
	//dim3 number_of_blocks((width / threads_per_block.x + 1),
	//	(height / threads_per_block.y + 1),
	//	(channel / threads_per_block.z + 1));
	//for (int batch = 0; batch < batchSize; ++batch)
	//{
	//	gpuMish <<<number_of_blocks, threads_per_block, 0, stream >>>((float*)(output) + (batch * outputSize),
	//		(float*)(input) + (batch * outputSize), channel, height,
	//		width);
	//}
	return cudaGetLastError();
}

