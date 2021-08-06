#ifndef _NMS_HPP
#define _NMS_HPP

#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <vector>
#include <iostream>

void Decode(float* yoloBoxs, int* boxLocation, int gride, int threshold_index, int classNB, int batchSize,
	int archorsNB, int box_len, float threshold, float alpha, float beat, const float *archor_w, const float *archor_h, const float w, const float h, cudaStream_t stream);

void IoUCalculation(float *yoloBoxs, float * IoU, int boxNB, int boxLen, int batchSize, cudaStream_t stream);

#endif 