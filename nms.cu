#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "nms.hpp"
#include <vector>
#include <iostream>
#define dimen 16

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;



inline __device__ int entry_index(int w, int h, int location, int entry,int classNB)
{
	int n = location / (w * h);
	int loc = location % (w * h);
	return n * w * h * (4 + classNB + 1) + entry * w * h + loc;
}


inline __device__ int get_box_index(int index,int stride)
{
	int x,y,w,h,t;

	x = index + 0 * stride;
	y = index + 1 * stride;
	w = index + 2 * stride;
	h = index + 3 * stride;
	t = index + 4 * stride;
	//end
	return x,y,w,h,t;
}

__global__ void decode( float* yoloBoxs,  int* boxLocation, int gride_size,int gride, int threshold_index, int classNB,
	int archorsNB , int box_len, float threshold ,float alpha , float beat ,const float *archor_w , const float *archor_h , const float w, const float h)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z;
	int batchIndex = blockIdx.z;

	//printf("batchIndex is %d\n", blockIdx.z);
	//printf("blockIdx.z is %d\n", z);
	//printf("%f\n %f\n %f\n", archor_h[0], archor_h[1], archor_h[2]);
	int index_threshold = x + y * gride + (threshold_index+11*z) * gride_size + batchIndex * gride_size *box_len*3;
	int index_x = x + y * gride + (0 + 11 * z) * gride_size + batchIndex * gride_size *box_len*3;
	int index_y = x + y * gride + (1 + 11 * z) * gride_size + batchIndex * gride_size *box_len*3;
	int index_w = x + y * gride + (2 + 11 * z) * gride_size + batchIndex * gride_size *box_len*3;
	int index_h = x + y * gride + (3 + 11 * z) * gride_size + batchIndex * gride_size *box_len*3;
	//int test_index = 300;

	//if (yoloBoxs[index_threshold] >= threshold)
	//{
	//	printf("\nyoloBoxs[index_threshold] is %f,\n index_x=%d index_y=%d index_w=%d index_h=%d,threshold=%d,w=%f h=%f \n, yoloBoxs[index_x]=%f yoloBoxs[index_y]=%f\n yoloBoxs[index_w]=%f yoloBoxs[index_h]=%f\n", yoloBoxs[index_threshold], index_x, index_y, index_w, index_h, yoloBoxs[index_threshold], w, h, yoloBoxs[index_x], yoloBoxs[index_y], yoloBoxs[index_w], yoloBoxs[index_h]);
	//	//printf("yoloBoxs[index_x]=%f yoloBoxs[index_y]=%f\n yoloBoxs[index_w]=%f yoloBoxs[index_h]=%f\n", yoloBoxs[index_x], yoloBoxs[index_y], yoloBoxs[index_w], yoloBoxs[index_h]);
	//	//printf("index_x=%d index_y=%d index_w=%d index_h=%d,threshold=%d,w=%f h=%f \n", index_x, index_y, index_w, index_h, yoloBoxs[index_threshold], w, h);

	//}

	//int box_index = entry_index(gride, gride, z * gride_size + x, 0, classNB);
	//int index_x, index_y, index_w, index_h, index_threshold = get_box_index(box_index, gride_size);

		if (yoloBoxs[index_threshold] >= threshold)
		{
			
			yoloBoxs[index_x] = (x + yoloBoxs[index_x] * alpha + beat) / gride;
			yoloBoxs[index_y] = (y + yoloBoxs[index_y] * alpha + beat) / gride;
			yoloBoxs[index_w] = yoloBoxs[index_w] * archor_w[z] / w;
			yoloBoxs[index_h] = yoloBoxs[index_h] * archor_h[z] / h;

			int class_index = -1;
			float class_threshold = 0;
			
			for (int j = 0; j < classNB; j++)
			{
				if (class_threshold <= yoloBoxs[x + y * gride + (5+j+11 * z) * gride_size + batchIndex * gride_size *box_len*3])
				{
					class_threshold = yoloBoxs[x + y * gride + (5+j + 11 * z) * gride_size + batchIndex * gride_size *box_len*3];
					class_index = j;
					//printf("when classNB is %d,threshold=%f\n", j, class_threshold);
				}
				//printf("is boxLocation[x + y * gride]%f\n", boxLocation[x + y * gride]);
			}
			//printf("when classNB is %d,threshold=%f\n", class_index, class_threshold);
			boxLocation[x + y * gride+ gride_size*z+ batchIndex * gride_size*3] = class_index;
			//printf("threadID is %d\n", x+y * gride + gride_size* z+ batchIndex * gride_size);
		}
		else {
			int class_index = -1;
			boxLocation[x + y * gride + gride_size * z + batchIndex * gride_size * 3] = class_index;
			
		}
	//	if (x + y * gride + gride_size * z + batchIndex * gride_size * 3 == test_index) 
	//	{
	//		
	//		printf("yoloBoxs[index_x] = (x + yoloBoxs[index_x] * alpha + beat) / gride \n");
	//		printf("x=%d, yoloBoxs[index_x]=%f, w=%f , alpha=%f, beat=%f, gride=%d \n", x, yoloBoxs[index_x], yoloBoxs[index_w],alpha, beat, gride);
	//		printf("y=%d, yoloBoxs[index_y]=%f, h=%f ,alpha=%f, beat=%f, gride=%d \n", y, yoloBoxs[index_y], yoloBoxs[index_h],alpha, beat, gride);

	//	}
	////}

}
 



inline __device__ float overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}



inline __device__ float box_intersection(float *yoloBoxs, int index_box1, int index_box2)
{
	float w = overlap(yoloBoxs[0 + index_box1], yoloBoxs[2 + index_box1], yoloBoxs[0 + index_box2], yoloBoxs[2 + index_box2]);
	float h = overlap(yoloBoxs[1 + index_box1], yoloBoxs[3 + index_box1], yoloBoxs[1 + index_box2], yoloBoxs[3 + index_box2]);
	if (w < 0 || h < 0) return 0;
	float area = w * h;
	return area;
}

inline __device__ float box_union(float *yoloBoxs, int index_box1, int index_box2)
{
	float i = box_intersection(yoloBoxs, index_box1, index_box2);
	float u = yoloBoxs[2 + index_box1] * yoloBoxs[3 + index_box1] + yoloBoxs[2 + index_box2] * yoloBoxs[3 + index_box2] - i;
	return u;
}


__global__ void IoUCal(float *yoloBoxs, float * IoU, int boxNB, int boxLen)
{


	//printf("%f", threadIdx.x);

	//int batch = blockIdx.x;

	//int blockId = blockIdx.x + blockIdx.y * gridDim.x
	//	+ gridDim.x * gridDim.y * blockIdx.z;
	//int threadId = blockId * (blockDim.x * blockDim.y)
	//	+ (threadIdx.y * blockDim.x) + threadIdx.x;

	int batch = 0;
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;

	//int quotient = blockId / 4;
	//int remainder = blockId / 4 > 0 ? blockId % 4 : blockId;

	//threadId = (quotient * 4 + threadIdx.y)*(blockDim.x * 4) + (remainder*blockDim.x + threadIdx.x);
	//threadId = (quotient * 4 + threadIdx.y)*boxNB + (remainder*blockDim.x + threadIdx.x);
	/*int index_box2 = (threadIdx.x + blockDim.x*blockIdx.x) * boxLen + blockIdx.z*blockDim.x*blockIdx.x;
	int index_box1 = (threadIdx.y + blockDim.x*blockIdx.y) * boxLen + blockIdx.z*blockDim.x*blockIdx.x;*/

	int index_box2 = threadIdx.x*boxLen;
	int index_box1 = blockIdx.x*boxLen;


	float inter_area = box_intersection(yoloBoxs, index_box1, index_box2);
	if (inter_area > 0.8 * yoloBoxs[2 + index_box1] * yoloBoxs[4 + index_box1] || yoloBoxs[2 + index_box2] * yoloBoxs[4 + index_box2])
	{
		IoU[threadId] = 1;
	
	}
	float U = box_union(yoloBoxs, index_box1, index_box2);
	if (inter_area == 0 || U == 0) {
		IoU[threadId] = 0;

	}
	else
	{
		IoU[threadId] = inter_area / U;

	}

		//threadId = blockId * (blockDim.x * blockDim.y)+ threadIdx.x+ blockDim.x*threadIdx.y;

		//printf("blockIdx.y %d\n" , blockIdx.y);
		//printf("index_box1%d\n", (threadIdx.x + blockDim.x*blockId) * boxLen);


		//float inter_area = box_intersection(yoloBoxs, index_box1, index_box2);

		//if (inter_area > 0.8 * yoloBoxs[2 + index_box1] * yoloBoxs[4 + index_box1] || yoloBoxs[2 + index_box2] * yoloBoxs[4 + index_box2])
		//{
		//	IoU[threadIdx.x + threadIdx.y * blockDim.x + batch * blockDim.x *  blockDim.y] = 1;
		//	//printf("index_box1=%f,index_box2=%f\n", index_box1, index_box2, 10000);
		//}

		//float U = box_union(yoloBoxs, index_box1, index_box2);

		//if (inter_area == 0 || U == 0) {
		//	IoU[threadIdx.x + threadIdx.y * blockDim.x + batch * blockDim.x *  blockDim.y] = 0;
		//	//printf("index_box1=%f,index_box2=%f\n", index_box1, index_box2, U);
		//}

		//else 
		//{
		//	IoU[threadIdx.x + threadIdx.y * blockDim.x + batch * blockDim.x *  blockDim.y] = inter_area / U;
		//	//printf("index_box1=%f,index_box2=%f\n", index_box1, index_box2,U);
		//}


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
		//float area_x1 = yoloBoxs[0 + index_box1] > yoloBoxs[0 + index_box2] ? yoloBoxs[0 + index_box1] : yoloBoxs[0 + index_box2];
		//float area_y1 = yoloBoxs[1 + index_box1] < yoloBoxs[1 + index_box2] ? yoloBoxs[1 + index_box1] : yoloBoxs[1 + index_box2];
		//float area_x2 = yoloBoxs[2 + index_box1] > yoloBoxs[2 + index_box2] ? yoloBoxs[2 + index_box1] : yoloBoxs[2 + index_box2];
		//float area_y2 = yoloBoxs[3 + index_box1] < yoloBoxs[3 + index_box2] ? yoloBoxs[3 + index_box1] : yoloBoxs[3 + index_box2];


		//float inter_area = (area_x2 - area_x1) * (area_y2 - area_y1);
		//float box1_area = (yoloBoxs[2 + index_box1] - yoloBoxs[0 + index_box1]) * (yoloBoxs[3 + index_box1] - yoloBoxs[1 + index_box1]);
		//float box2_area = (yoloBoxs[2 + index_box2] - yoloBoxs[0 + index_box2]) * (yoloBoxs[3 + index_box2] - yoloBoxs[1 + index_box2]);

		//IoU[threadId] = inter_area / (box1_area + box2_area - inter_area);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//printf("%d\n", threadIdx.x + threadIdx.y * blockDim.x + batch * blockDim.x *  blockDim.y);
		//printf("%d", threadIdx.x);
		//IoU[threadIdx.x + threadIdx.y * blockDim.x + batch * blockDim.x *  blockDim.y] = inter_area / (box1_area + box2_area - inter_area);
		//IoU[0] = inter_area / (box1_area + box2_area - inter_area);
		//IoU[0] = inter_area / (box1_area + box2_area - inter_area);

		//IoU[threadIdx.x + threadIdx.y * blockDim.x + blockDim.x+batch * blockDim.x *  blockDim.y] = inter_area / (box1_area + box2_area - inter_area);
		

		//if (threadId == 0)
		//{
		//	printf("b1_GPU=(%f,%f,%f,%f)\nb2_GPU=(%f,%f,%f,%f)\n,index is %d, iou is %f\n", yoloBoxs[0 + index_box1], yoloBoxs[1 + index_box1], yoloBoxs[2 + index_box1], yoloBoxs[3 + index_box1], yoloBoxs[0 + index_box2], yoloBoxs[1 + index_box2], yoloBoxs[2 + index_box2], yoloBoxs[3 + index_box2], threadIdx.x + threadIdx.y * blockDim.x + batch * blockDim.x *  blockDim.y, IoU[threadIdx.x + threadIdx.y * blockDim.x + batch * blockDim.x *  blockDim.y]);
		//	printf("x1 is %f y1 is %f x2 i %f y2 is %f\n ", area_x1, area_y1, area_x2, area_y2);
		//	printf("b2_GPU=(%f,%f,%f,%f)\n", yoloBoxs[0 + index_box2], yoloBoxs[1 + index_box2], yoloBoxs[2 + index_box2], yoloBoxs[3 + index_box2]);
		//	printf("index is %d, iou is %f\n", threadId, IoU[threadId]);
		//	printf("threadIdx.x=%d,threadIdx.y=%d,blockDim.x=%d blockId=%d,blockIdx.x=%d\n", threadIdx.x, threadIdx.y, blockDim.x, blockId, blockIdx.x);
		//	printf("index_box1 %d\n", (threadIdx.x + blockDim.x*blockIdx.x) * boxLen);
		//	printf("index_box2 %d\n", (threadIdx.y + blockDim.x*blockIdx.y) * boxLen);

		//}
	//}

}

void _set_device(int device_id) {
	int current_device;
	CUDA_CHECK(cudaGetDevice(&current_device));
	if (current_device == device_id) {
		return;
	}
	// The call to cudaSetDevice must come before any calls to Get, which
	// may perform initialization using the GPU.
	CUDA_CHECK(cudaSetDevice(device_id));
}



void IoUCalculation(float *yoloBoxs, float * IoU, int boxNB, int boxLen, int batchSize, cudaStream_t stream)
{
	//const dim3 block(boxNB, boxNB);
	//const dim3 gride(batchSize);
	//if (boxNB < 16) 
	//{
	//	const dim3 block(boxNB, boxNB);
	//	const dim3 gride(1,1,batchSize);
	//	IoUCal << <gride, block, 0, stream >> > (yoloBoxs, IoU, boxNB, boxLen);
	//}
	//else 
	

	const dim3 block(boxNB);
	const dim3 gride(boxNB, batchSize);

	IoUCal << <gride, block, 0, stream >> > (yoloBoxs, IoU, boxNB, boxLen);
		

	//const dim3 gride(4, 4, batchSize);
	
	//printf("boxNB is %d\n", boxNB);
	//printf("batchSize is %d\n", batchSize);
	//printf("boxLen is %d\n", boxLen);

	
	//cudaStreamSynchronize(stream);

}
void Decode(float* yoloBoxs, int* boxLocation, int gride, int threshold_index, int classNB, int batchSize,
	int archorsNB, int box_len, float threshold, float alpha, float beat, const float *archor_w, const float *archor_h, const float w, const float h, 
	cudaStream_t stream)
{
	//printf("%f\n",archor_w[1]);
	const dim3 block1(16, 16, 3);
	const dim3 gride1((gride-1)/16+1, (gride - 1) / 16 + 1 , batchSize);
	//printf("batchSize is %d", batchSize);
	int gride_size = gride * gride;
	decode << <gride1,block1, 0, stream >> > (yoloBoxs, boxLocation, gride_size, gride, threshold_index, classNB,
		archorsNB, box_len, threshold, alpha, beat, archor_w, archor_h, w, h);

	//cudaStreamSynchronize(stream);
}

void batchNMS(const float* input, float* ouput, const float* gride, const float* archor , int grideNB , int archorNB,  float nms_thresh, int batch ,int device_id) 
{
	_set_device(device_id);

	int bboxNB = 0;

	float* boxes_dev = NULL;
	unsigned long long* mask_dev = NULL;

	const dim3 block(64, 64);
	const dim3 grid(batch, 33);
	

	
	CUDA_CHECK(cudaFree(boxes_dev));
	CUDA_CHECK(cudaFree(mask_dev));
}

