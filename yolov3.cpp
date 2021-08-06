
#include "yolov3.h"
#include "network_config.h"
#include "nms.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>

#include <opencv2/imgcodecs.hpp>

#define SIX_CLASS_NUM true
#if SIX_CLASS_NUM
//6 class num
const std::vector<std::string> kCLASS_NAMES
= { "person", "bicycle", "car", "motorcycle", "bus", "truck" };
const std::vector<cv::Scalar> kCLASS_COLORS
= { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)
, cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255),cv::Scalar(0, 255, 255) };

#else

const std::vector<std::string> kCLASS_NAMES
= { "person",        "bicycle",       "car",           "motorcycle",
"aeroplane",     "bus",           "train",         "truck" };
const std::vector<cv::Scalar> kCLASS_COLORS
= { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0),
cv::Scalar(255, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 255), cv::Scalar(0, 255, 255) };

#endif // SIX_CLASS_NUM


typedef struct T_BOX {
	float box[4];
	float score;
} ANCHOR_BOX;

ANCHOR_BOX det_tmp;


std::vector<BBoxInfo> getRealDetections(std::vector<ANCHOR_BOX>& dets, int class_ )
{
	std::vector<BBoxInfo> result_arr;
	int selected_num = 0;
	int dets_num = dets.size();
	int i;
	for (i = 0; i < dets_num; ++i) {

			BBoxInfo result;
			result.box.x = dets[i].box[0] - dets[i].box[2] / 2.;
			result.box.y = dets[i].box[1] - dets[i].box[3] / 2.;
			result.box.w = dets[i].box[2];
			result.box.h = dets[i].box[3];
			result.label = class_;
			result.prob = dets[i].score;
			result_arr.push_back(result);
		}
	
	return result_arr;
}

float iou(ANCHOR_BOX ab1, ANCHOR_BOX ab2,int i) {
	float x1 = std::max(ab1.box[0], ab2.box[0]);
	float x2 = std::max(ab1.box[2], ab2.box[2]);
	float y1 = std::min(ab1.box[1], ab2.box[1]);
	float y2 = std::min(ab1.box[3], ab2.box[3]);
	float over_area = (x2 - x1) * (y2 - y1);
	float union_area = (ab1.box[3] - ab1.box[1]) * (ab1.box[2] - ab1.box[0]) + (ab2.box[3] - ab2.box[1]) * (ab2.box[2] - ab2.box[0]);
	float iou;
	iou = over_area / (union_area - over_area);
	/*if (i == 432) {
		std::cout << "inter_area = " << over_area << "box1_area = " << (ab1.box[3] - ab1.box[1]) * (ab1.box[2] - ab1.box[0]) << "box2_area = " << (ab2.box[3] - ab2.box[1]) * (ab2.box[2] - ab2.box[0]) << std::endl;
		std::cout << "b1 = (" << ab1.box[0] << "," << ab1.box[1] << "," << ab1.box[2] << "," << ab1.box[3] << ")" << std::endl;
		std::cout << "b2 = (" << ab2.box[0] << "," << ab2.box[1] << "," << ab2.box[2] << "," << ab2.box[3] << ")" << std::endl;
		std::cout << "index is " << i << "the iou is " << iou << std::endl;
		printf("x1 is %f y1 is %f x2 i %f y2 is %f\n ", x1, y1, x2, y2);
	}*/
	return iou;
}


void cal_iou_cpu(std::vector<ANCHOR_BOX> dets, float* iouMat, const int nx)
{
	for (int i = 0; i < dets.size(); i++)
	{
		for (int j = 0; j < dets.size(); j++)
		{
			iouMat[i*nx + j] = iou(dets[j], dets[i], i*nx + j);
		}
	}

}


void transferBbox(std::vector<BBoxInfo>& bboxes, int ori_image_h=1080, int ori_image_w=1920)
{
	for (int i = 0; i < bboxes.size(); i++)
	{
		bboxes[i].box.x *= ori_image_w;
		bboxes[i].box.y *= ori_image_h;
		bboxes[i].box.w *= ori_image_w;
		bboxes[i].box.h *= ori_image_h;
	}
}

void do_nms(std::vector<ANCHOR_BOX>& dets, float* iou, int nx, int ny, float thresh)
{
	for (int i = 0; i < nx; ++i) {
		if (dets[i].score == 0) continue;
		for (int j = i + 1; j < ny; ++j) {
			if (iou[i * nx + j] > thresh) {
				dets[j].score = 0;
			}
		}
	}
}


void set_anchor(float *anchorw, float *anchorh,const std::vector<float> m_Anchors,int n)
{
	anchorw[0] = m_Anchors[0 + 6 * n];
	anchorw[1] = m_Anchors[2 + 6 * n];
	anchorw[2] = m_Anchors[4 + 6 * n];
	anchorh[0] = m_Anchors[1 + 6 * n];
	anchorh[1] = m_Anchors[3 + 6 * n];
	anchorh[2] = m_Anchors[5 + 6 * n];
}

bool cmpScore(ANCHOR_BOX ab1, ANCHOR_BOX ab2) {
	return ab1.score > ab2.score;
}

void fill_det(std::vector<ANCHOR_BOX> &dets, float *box, int index,int stride) 
{
	if (index == 46512)
	{
		int a = 1;
	}
	det_tmp.box[0] = box[index];
	det_tmp.box[1] = box[index+1 * stride];
	det_tmp.box[2] = box[index+2 * stride];
	det_tmp.box[3] = box[index+3 * stride];
	det_tmp.score = box[index+4 * stride];
	dets.push_back(det_tmp);
}

void collect_det(float *dets, const int nx, const int ny, int* p0,
	std::vector<ANCHOR_BOX> &dets_0, std::vector<ANCHOR_BOX> &dets_1, std::vector<ANCHOR_BOX> &dets_2,
	std::vector<ANCHOR_BOX> &dets_3, std::vector<ANCHOR_BOX> &dets_4, std::vector<ANCHOR_BOX> &dets_5)
{
	int stride = nx * ny;
	for (int z = 0; z < 3; z++) 
	{
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				switch (p0[z*nx*ny + i * nx + j])
				{
				case 0: {
					fill_det(dets_0, dets, z*11*nx*ny + i * nx + j, stride);
					//dets_0.push_back(dets[i * nx + j]);
					break;
				case 1: {
					fill_det(dets_1, dets, z*11*nx*ny + i * nx + j, stride);
					break; }
				case 2: {
					fill_det(dets_2, dets, z*11*nx*ny + i * nx + j, stride);
					break; }
				case 3: {
					fill_det(dets_3, dets, z*11*nx*ny + i * nx + j, stride);
					break; }
				case 4: {
					fill_det(dets_4, dets, z*11*nx*ny + i * nx + j, stride);
					break; }
				case 5: {
					fill_det(dets_5, dets, z*11*nx*ny + i * nx + j, stride);
					break; }
				default:
					break;
				}
				}
			}
		}
	}
}

void assgin_det(std::vector<ANCHOR_BOX> &dets_, float *dets) 
{
	for (int i = 0; i < dets_.size(); i++) 
	{
		dets[i * 5 + 0] = dets_[i].box[0];
		dets[i * 5 + 1] = dets_[i].box[1];
		dets[i * 5 + 2] = dets_[i].box[2];
		dets[i * 5 + 3] = dets_[i].box[3];
		dets[i * 5 + 4] = dets_[i].score;
	}
}


YoloV3::YoloV3(void (plog)(const char *, int), unsigned int batchSize, const std::string& gpu, const std::string& pre, bool genEng) :
    Yolo(plog, batchSize,gpu, pre, genEng),
    m_Stride1(config::yoloV3::kSTRIDE_1),
    m_Stride2(config::yoloV3::kSTRIDE_2),
    m_Stride3(config::yoloV3::kSTRIDE_3),
    m_OutputIndex1(-1),
    m_OutputIndex2(-1),
    m_OutputIndex3(-1),
    m_Mask1(config::yoloV3::kMASK_1),
    m_Mask2(config::yoloV3::kMASK_2),
    m_Mask3(config::yoloV3::kMASK_3),
    m_OutputBlobName1(config::yoloV3::kOUTPUT_BLOB_NAME_1),
    m_OutputBlobName2(config::yoloV3::kOUTPUT_BLOB_NAME_2),
    m_OutputBlobName3(config::yoloV3::kOUTPUT_BLOB_NAME_3),
	m_ScaleXY1(config::yoloV3::kSCALE_X_Y_1),
	m_ScaleXY2(config::yoloV3::kSCALE_X_Y_2),
	m_ScaleXY3(config::yoloV3::kSCALE_X_Y_3)
{
	log = plog;
	m_GridSize1 = m_InputH / m_Stride1;
	m_GridSize2 = m_InputH / m_Stride2;
	m_GridSize3 = m_InputH / m_Stride3;

    assert(m_NetworkType == "yolov3");
	if (!m_Engine)
	{
		return;
	}
    // Allocate Buffers
    m_OutputIndex1 = m_Engine->getBindingIndex(m_OutputBlobName1.c_str());
    assert(m_OutputIndex1 != -1);
    m_OutputIndex2 = m_Engine->getBindingIndex(m_OutputBlobName2.c_str());
    assert(m_OutputIndex2 != -1);
    m_OutputIndex3 = m_Engine->getBindingIndex(m_OutputBlobName3.c_str());
    assert(m_OutputIndex3 != -1);

	m_OutputSize1 = 1;
	m_OutputSize2 = 1;
	m_OutputSize3 = 1;
	nvinfer1::Dims out_dim1 = m_Engine->getBindingDimensions(m_OutputIndex1);
	nvinfer1::Dims out_dim2 = m_Engine->getBindingDimensions(m_OutputIndex2);
	nvinfer1::Dims out_dim3 = m_Engine->getBindingDimensions(m_OutputIndex3);
	for (size_t i = 0; i < out_dim1.nbDims; i++)
	{
		m_OutputSize1 *= out_dim1.d[i];
	}
	for (size_t i = 0; i < out_dim2.nbDims; i++)
	{
		m_OutputSize2 *= out_dim2.d[i];
	}
	for (size_t i = 0; i < out_dim3.nbDims; i++)
	{
		m_OutputSize3 *= out_dim3.d[i];
	}

    NV_CUDA_CHECK(
        cudaMalloc(&m_Bindings.at(m_InputIndex), m_BatchSize * m_InputSize * sizeof(float)));
    NV_CUDA_CHECK(
        cudaMalloc(&m_Bindings.at(m_OutputIndex1), m_BatchSize * m_OutputSize1 * sizeof(float)));
    NV_CUDA_CHECK(
        cudaMalloc(&m_Bindings.at(m_OutputIndex2), m_BatchSize * m_OutputSize2 * sizeof(float)));
    NV_CUDA_CHECK(
        cudaMalloc(&m_Bindings.at(m_OutputIndex3), m_BatchSize * m_OutputSize3 * sizeof(float)));
	//m_TrtOutputBuffers.at(0) = new float[m_OutputSize1 * m_BatchSize];
    //m_TrtOutputBuffers.at(1) = new float[m_OutputSize2 * m_BatchSize];
    //m_TrtOutputBuffers.at(2) = new float[m_OutputSize3 * m_BatchSize];
    cudaMallocHost(&m_TrtOutputBuffers.at(0), m_OutputSize1 * m_BatchSize * sizeof(float));
    cudaMallocHost(&m_TrtOutputBuffers.at(1), m_OutputSize2 * m_BatchSize * sizeof(float));
    cudaMallocHost(&m_TrtOutputBuffers.at(2), m_OutputSize3 * m_BatchSize * sizeof(float));
};

bool YoloV3::doInference(const unsigned char* input)
{
	//应用使用这行时是cudaMemcpyDeviceToDevice
    cudaError cudaError = cudaMemcpyAsync(m_Bindings.at(m_InputIndex), input,
                                  m_BatchSize * m_InputSize * sizeof(float), cudaMemcpyDeviceToDevice,
                                  m_CudaStream);
	if (cudaError != cudaSuccess)
	{
		log("memcpy device to device error!", 1);
		cudaStreamSynchronize(m_CudaStream);
		return false;
	}
	float time_elapsed = 0;
	cudaEvent_t start,stop;
	cudaEventCreate(&start);    //创建Event
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);    //记录当前时间
    m_Context->enqueue(m_BatchSize, m_Bindings.data(), m_CudaStream, nullptr);
	cudaEventRecord(stop, 0);    //记录当前时间
	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
	cudaEventElapsedTime(&time_elapsed, start,stop);    //计算时间差	
	cudaEventDestroy(start);    //destory the event
	cudaEventDestroy(stop);
	std::cout << "do inference time one map for a batch = " << time_elapsed << std::endl;
	cudaStreamSynchronize(m_CudaStream);

    /*cudaError = cudaMemcpyAsync(m_TrtOutputBuffers.at(0), m_Bindings.at(m_OutputIndex1),
                                  m_BatchSize * m_OutputSize1 * sizeof(float),
                                  cudaMemcpyDeviceToHost, m_CudaStream);*/



	float* inputCpu = new float[m_BatchSize * m_OutputSize1];
	float* box1 = new float[m_OutputSize1 * m_BatchSize];
	float* box2 = new float[m_OutputSize2 * m_BatchSize];
	float* box3 = new float[m_OutputSize3 * m_BatchSize];


	int* p1 = new int[m_BatchSize * 64 * 64*3];
	int* p2 = new int[m_BatchSize * 32 * 32*3];
	int* p3 = new int[m_BatchSize * 16 * 16*3];

	

	int* boxLocation1;
	cudaMalloc(&boxLocation1, m_BatchSize * 64 * 64 * 3 * sizeof(int));
	int* boxLocation2;
	cudaMalloc(&boxLocation2, m_BatchSize * 32 * 32 * 3 * sizeof(int));
	int* boxLocation3;
	cudaMalloc(&boxLocation3, m_BatchSize * 16 * 16 * 3 * sizeof(int));

	float alpha1 = 1.2;
	float beta1 = -0.1;

	float alpha2 = 1.1;
	float beta2 = -0.05;

	float alpha3 = 1.05;
	float beta3 = -0.025;

	float *anchorw_64 = new float[3];
	float *anchorh_64 = new float[3];
	float *anchorw_32 = new float[3];
	float *anchorh_32 = new float[3];
	float *anchorw_16 = new float[3];
	float *anchorh_16 = new float[3];

	float *anchorw_64_GPU;
	cudaMalloc(&anchorw_64_GPU, 3 * sizeof(float));
	float *anchorh_64_GPU;
	cudaMalloc(&anchorh_64_GPU, 3 * sizeof(float));

	float *anchorw_32_GPU;
	cudaMalloc(&anchorw_32_GPU, 3 * sizeof(float));
	float *anchorh_32_GPU;
	cudaMalloc(&anchorh_32_GPU, 3 * sizeof(float));

	float *anchorw_16_GPU;
	cudaMalloc(&anchorw_16_GPU, 3 * sizeof(float));
	float *anchorh_16_GPU;
	cudaMalloc(&anchorh_16_GPU, 3 * sizeof(float));


	set_anchor(anchorw_64, anchorh_64, m_Anchors, 0);
	set_anchor(anchorw_32, anchorh_32, m_Anchors, 1);
	set_anchor(anchorw_16, anchorh_16, m_Anchors, 2);

	cudaMemcpyAsync(anchorw_64_GPU, anchorw_64,
		3 * sizeof(float),
		cudaMemcpyHostToDevice, m_CudaStream);
	cudaMemcpyAsync(anchorh_64_GPU, anchorh_64,
		3 * sizeof(float),
		cudaMemcpyHostToDevice, m_CudaStream);

	cudaMemcpyAsync(anchorw_32_GPU, anchorw_32,
		3 * sizeof(float),
		cudaMemcpyHostToDevice, m_CudaStream);
	cudaMemcpyAsync(anchorh_32_GPU, anchorh_32,
		3 * sizeof(float),
		cudaMemcpyHostToDevice, m_CudaStream);

	cudaMemcpyAsync(anchorw_16_GPU, anchorw_16,
		3 * sizeof(float),
		cudaMemcpyHostToDevice, m_CudaStream);
	cudaMemcpyAsync(anchorh_16_GPU, anchorh_16,
		3 * sizeof(float),
		cudaMemcpyHostToDevice, m_CudaStream);


	cudaStreamSynchronize(m_CudaStream);

	int classNb = 6;
	int anchorNb = 3;
	const float GPU_wh = 512;
	int box_len = 11;
	float thresh = 0.1;

	Decode((float*)m_Bindings.at(m_OutputIndex1), boxLocation1, 64, 4, classNb, m_BatchSize,
		anchorNb, box_len, thresh, alpha1, beta1, anchorw_64_GPU, anchorh_64_GPU, GPU_wh, GPU_wh, m_CudaStream);

	Decode((float*)m_Bindings.at(m_OutputIndex2), boxLocation2, 32, 4, classNb, m_BatchSize,
		anchorNb, box_len, thresh, alpha2, beta2, anchorw_32_GPU, anchorh_32_GPU, GPU_wh, GPU_wh, m_CudaStream);

	Decode((float*)m_Bindings.at(m_OutputIndex3), boxLocation3, 16, 4, classNb, m_BatchSize,
		anchorNb, box_len, thresh, alpha3, beta3, anchorw_16_GPU, anchorh_16_GPU, GPU_wh, GPU_wh, m_CudaStream);


	cudaStreamSynchronize(m_CudaStream);
	
	cudaMemcpyAsync(box1, m_Bindings.at(m_OutputIndex1),
		m_BatchSize * m_OutputSize1 * sizeof(float),
		cudaMemcpyDeviceToHost, m_CudaStream);
	cudaStreamSynchronize(m_CudaStream);

	cudaMemcpyAsync(box2, m_Bindings.at(m_OutputIndex2),
		m_BatchSize * m_OutputSize2 * sizeof(float),
		cudaMemcpyDeviceToHost, m_CudaStream);

	cudaMemcpyAsync(box3, m_Bindings.at(m_OutputIndex3),
		m_BatchSize * m_OutputSize3 * sizeof(float),
		cudaMemcpyDeviceToHost, m_CudaStream);

	cudaStreamSynchronize(m_CudaStream);

	cudaError = cudaMemcpy(p1, boxLocation1,
		m_BatchSize * 64 * 64 * 3 * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaError = cudaMemcpy(p2, boxLocation2,
		m_BatchSize * 32 * 32 * 3 * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaError = cudaMemcpy(p3, boxLocation3,
		m_BatchSize * 16 * 16 * 3 * sizeof(int),
		cudaMemcpyDeviceToHost);
	cudaStreamSynchronize(m_CudaStream);


	const int nx64 = 64, ny64 = 64;
	const int nx32 = 32, ny32 = 32;
	const int nx16 = 16, ny16 = 16;

	std::vector <std::vector<std::vector<ANCHOR_BOX>>> batchBox;
	batchBox.resize(m_BatchSize);

	for (int k=0;k<m_BatchSize;k++)
	{
		std::vector<std::vector<ANCHOR_BOX>>dets_;
		dets_.resize(classNb);

		collect_det(box1+k*m_OutputSize1, nx64, ny64, p1+ k * 64 * 64 * 3, dets_[0], dets_[1], dets_[2], dets_[3], dets_[4], dets_[5]);
		collect_det(box2+k*m_OutputSize2, nx32, ny32, p2+ k * 32 * 32 * 3, dets_[0], dets_[1], dets_[2], dets_[3], dets_[4], dets_[5]);
		collect_det(box3+k*m_OutputSize3, nx16, ny16, p3+ k * 16 * 16 * 3, dets_[0], dets_[1], dets_[2], dets_[3], dets_[4], dets_[5]);

		float* dets_0 = new float[dets_[0].size() * 5];
		float* dets_1 = new float[dets_[1].size() * 5];
		float* dets_2 = new float[dets_[2].size() * 5];
		float* dets_3 = new float[dets_[3].size() * 5];
		float* dets_4 = new float[dets_[4].size() * 5];
		float* dets_5 = new float[dets_[5].size() * 5];


		for (int i = 0; i < 6; i++)
		{
			std::sort(dets_[i].begin(), dets_[i].end(), cmpScore);
		}

		assgin_det(dets_[0], dets_0);
		assgin_det(dets_[1], dets_1);
		assgin_det(dets_[2], dets_2);
		assgin_det(dets_[3], dets_3);
		assgin_det(dets_[4], dets_4);
		assgin_det(dets_[5], dets_5);

		float* IOU_cpu_0 = new float[dets_[0].size() * dets_[0].size()];
		float* boxData_gpu_0;
		float* IoU_0;
		cudaMalloc(&boxData_gpu_0, dets_[0].size() * 5 * sizeof(float));
		cudaMemcpyAsync(boxData_gpu_0, dets_0, dets_[0].size() * 5 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
		cudaMalloc(&IoU_0, dets_[0].size() * dets_[0].size() * sizeof(float));

		float* IOU_cpu_1 = new float[dets_[1].size() * dets_[1].size()];
		float* boxData_gpu_1;
		float* IoU_1;
		cudaMalloc(&boxData_gpu_1, dets_[1].size() * 5 * sizeof(float));
		cudaMemcpyAsync(boxData_gpu_1, dets_1, dets_[1].size() * 5 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
		cudaMalloc(&IoU_1, dets_[1].size() * dets_[1].size() * sizeof(float));

		float* IOU_cpu_2 = new float[dets_[2].size() * dets_[2].size()];
		float* boxData_gpu_2;
		float* IoU_2;
		cudaMalloc(&boxData_gpu_2, dets_[2].size() * 5 * sizeof(float));
		cudaMemcpyAsync(boxData_gpu_2, dets_2, dets_[2].size() * 5 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
		cudaMalloc(&IoU_2, dets_[2].size() * dets_[2].size() * sizeof(float));

		float* IOU_cpu_3 = new float[dets_[3].size() * dets_[3].size()];
		float* boxData_gpu_3;
		float* IoU_3;
		cudaMalloc(&boxData_gpu_3, dets_[3].size() * 5 * sizeof(float));
		cudaMemcpyAsync(boxData_gpu_3, dets_3, dets_[3].size() * 5 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
		cudaMalloc(&IoU_3, dets_[3].size() * dets_[3].size() * sizeof(float));

		float* IOU_cpu_4 = new float[dets_[4].size() * dets_[4].size()];
		float* boxData_gpu_4;
		float* IoU_4;
		cudaMalloc(&boxData_gpu_4, dets_[4].size() * 5 * sizeof(float));
		cudaMemcpyAsync(boxData_gpu_4, dets_4, dets_[4].size() * 5 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
		cudaMalloc(&IoU_4, dets_[4].size() * dets_[4].size() * sizeof(float));


		float* IOU_cpu_5 = new float[dets_[5].size() * dets_[5].size()];
		float* boxData_gpu_5;
		float* IoU_5;
		cudaMalloc(&boxData_gpu_5, dets_[5].size() * 5 * sizeof(float));
		cudaMemcpyAsync(boxData_gpu_5, dets_5, dets_[5].size() * 5 * sizeof(float), cudaMemcpyHostToDevice, m_CudaStream);
		cudaMalloc(&IoU_5, dets_[5].size() * dets_[5].size() * sizeof(float));

		cudaStreamSynchronize(m_CudaStream);

		IoUCalculation(boxData_gpu_0, IoU_0, dets_[0].size(), 5, 1, m_CudaStream);
		IoUCalculation(boxData_gpu_1, IoU_1, dets_[1].size(), 5, 1, m_CudaStream);
		IoUCalculation(boxData_gpu_2, IoU_2, dets_[2].size(), 5, 1, m_CudaStream);
		IoUCalculation(boxData_gpu_3, IoU_3, dets_[3].size(), 5, 1, m_CudaStream);
		IoUCalculation(boxData_gpu_4, IoU_4, dets_[4].size(), 5, 1, m_CudaStream);
		IoUCalculation(boxData_gpu_5, IoU_5, dets_[5].size(), 5, 1, m_CudaStream);


		cudaStreamSynchronize(m_CudaStream);
		cudaMemcpyAsync(IOU_cpu_0, IoU_0, dets_[0].size() * dets_[0].size() * sizeof(float),
			cudaMemcpyDeviceToHost, m_CudaStream);

		cudaMemcpyAsync(IOU_cpu_1, IoU_1, dets_[1].size() * dets_[1].size() * sizeof(float),
			cudaMemcpyDeviceToHost, m_CudaStream);

		cudaMemcpyAsync(IOU_cpu_2, IoU_2, dets_[2].size() * dets_[2].size() * sizeof(float),
			cudaMemcpyDeviceToHost, m_CudaStream);

		cudaMemcpyAsync(IOU_cpu_3, IoU_3, dets_[3].size() * dets_[3].size() * sizeof(float),
			cudaMemcpyDeviceToHost, m_CudaStream);

		cudaMemcpyAsync(IOU_cpu_4, IoU_4, dets_[4].size() * dets_[4].size() * sizeof(float),
			cudaMemcpyDeviceToHost, m_CudaStream);

		cudaMemcpyAsync(IOU_cpu_5, IoU_5, dets_[5].size() * dets_[5].size() * sizeof(float),
			cudaMemcpyDeviceToHost, m_CudaStream);
		cudaStreamSynchronize(m_CudaStream);


		do_nms(dets_[0], IOU_cpu_0, dets_[0].size(), dets_[0].size(), 0.3);
		do_nms(dets_[1], IOU_cpu_1, dets_[1].size(), dets_[1].size(), 0.3);
		do_nms(dets_[2], IOU_cpu_2, dets_[2].size(), dets_[2].size(), 0.3);
		do_nms(dets_[3], IOU_cpu_3, dets_[3].size(), dets_[3].size(), 0.3);
		do_nms(dets_[4], IOU_cpu_4, dets_[4].size(), dets_[4].size(), 0.3);
		do_nms(dets_[5], IOU_cpu_5, dets_[5].size(), dets_[5].size(), 0.3);

		batchBox[k] = dets_;


		cudaFree(boxData_gpu_0);
		cudaFree(boxData_gpu_1);
		cudaFree(boxData_gpu_2);
		cudaFree(boxData_gpu_3);
		cudaFree(boxData_gpu_4);
		cudaFree(boxData_gpu_5);

		cudaFree(IoU_0);
		cudaFree(IoU_1);
		cudaFree(IoU_2);
		cudaFree(IoU_3);
		cudaFree(IoU_4);
		cudaFree(IoU_5);
	}

	cudaFree(anchorw_64_GPU);
	cudaFree(anchorw_32_GPU);
	cudaFree(anchorw_16_GPU);
	cudaFree(inputCpu);
	
	

	
	//for (int k = 0; k < m_BatchSize; k++) {
	//	std::vector<std::vector<ANCHOR_BOX>>dets_ = batchBox[k];
	//	cv::Mat m_zero;
	//	m_zero = cv::imread("D:\\code\\copy\\object_detector_yolov4_DEBUG_FP16_INT8\\test\\testImages\\0000.jpg");
	//	for (int j = 0; j < 6; j++) {
	//		std::vector<BBoxInfo> out_box;
	//		out_box = getRealDetections(dets_[j], j);
	//		transferBbox(out_box);

	//		for (int i = 0; i < out_box.size(); i++)
	//		{

	//			if (out_box[i].prob > 0) {
	//				cv::Rect rect(out_box[i].box.x, out_box[i].box.y, out_box[i].box.w, out_box[i].box.h);
	//				cv::rectangle(m_zero, rect, kCLASS_COLORS[j], 1, cv::LINE_8, 0);
	//			}//lift top, right bottom
	//		}
	//	}
	//	cv::imshow("out", m_zero);
	//	cv::waitKey(0);
	//}
	return true;
}

void YoloV3::decodeDetections(std::vector<detection> &dets, const int& imageIdx, const int& imageH, const int& imageW, float thresh)
{
    std::vector<BBoxInfo> binfo;
	float* p0 = (float*)m_TrtOutputBuffers.at(0);
	float* p1 = (float*)m_TrtOutputBuffers.at(1);
	float* p2 = (float*)m_TrtOutputBuffers.at(2);

	getYoloDetections(&p0[imageIdx * m_OutputSize1], imageH, imageW, m_Mask1,
		m_GridSize1, m_Stride1, m_ScaleXY1, thresh, dets);
	getYoloDetections(&p1[imageIdx * m_OutputSize2], imageH, imageW, m_Mask2,
		m_GridSize2, m_Stride2, m_ScaleXY2, thresh, dets);
	getYoloDetections(&p2[imageIdx * m_OutputSize3], imageH, imageW, m_Mask3,
		m_GridSize3, m_Stride3, m_ScaleXY3, thresh, dets);

}

static int entry_index(int w, int h, int location, int entry)
{
	int n = location / ( w * h);
	int loc = location % ( w * h);
	return n * w * h * (4 + config::kOUTPUT_CLASSES + 1) + entry * w * h + loc;
}

box get_yolo_box(float *x, float pw, float ph, float alpha, float beta, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
	box b;

	b.x = (i + x[index + 0 * stride] * alpha + beta) / lw;
	b.y = (j + x[index + 1 * stride]) / lh;
	b.w = x[index + 2 * stride] * pw / w;
	b.h = x[index + 3 * stride] * ph / h;
	//end
	return b;
}
//得到所有detections
void YoloV3::getYoloDetections(float *output, const int &imageH, const int &imageW, const std::vector<int> mask, const unsigned int gridSize, 
	const unsigned int stride, const float scale_x_y, float thresh, std::vector<detection> &dets)
{
	int i, j, n;
	float *predictions = output;
	detection det;
	//l.n是num of boxes at one pixel
	float alpha = scale_x_y;
	float beta = -0.5*(scale_x_y - 1);
	for (i = 0; i < gridSize * gridSize; ++i) 
	{
		int row = i / gridSize;
		int col = i % gridSize;
		for (n = 0; n < m_NumBBoxes; ++n) 
		{
			const float pw = m_Anchors[mask[n] * 2];
			const float ph = m_Anchors[mask[n] * 2 + 1];
			int obj_index = entry_index(gridSize, gridSize,  n*gridSize*gridSize + i, 4);
			float objectness = predictions[obj_index];
			//std::cout << "===" << objectness << std::endl;
			if (objectness > thresh) 
			{
				int box_index = entry_index(gridSize, gridSize, n * gridSize * gridSize + i, 0);
				det.bbox = get_yolo_box(predictions, pw, ph, alpha, beta, box_index, col, row, gridSize, gridSize, m_InputW, m_InputW, gridSize*gridSize);
				det.objectness = objectness;
				det.classes = config::kOUTPUT_CLASSES;
				for (j = 0; j < config::kOUTPUT_CLASSES; ++j) {
					int class_index = entry_index(gridSize, gridSize, n * gridSize * gridSize + i, 4 + 1 + j);
					float prob = objectness*predictions[class_index];
					det.prob[j] = (prob > thresh) ? prob : 0;
				}
				//std::cout << "******" << std::endl;
				dets.push_back(det);
			}
		}
	}
}

float overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float box_intersection(box a, box b)
{
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

float box_union(box a, box b)
{
	float i = box_intersection(a, b);
	float u = a.w*a.h + b.w*b.h - i;
	return u;
}

float box_iou(box a, box b)
{
	//return box_intersection(a, b)/box_union(a, b);

	float I = box_intersection(a, b);
	if (I > 0.8 * a.w * a.h || I > 0.8 * b.w * b.h)
	{
		return 1;
	}
	float U = box_union(a, b);
	if (I == 0 || U == 0) {
		return 0;
	}
	return I / U;
}

int nms_comparator(detection &a, detection &b)
{
	float diff = 0;
	if (b.sort_class >= 0) {
		diff = a.prob[b.sort_class] - b.prob[b.sort_class]; // there is already: prob = objectness*prob
	}
	else {
		diff = a.objectness - b.objectness;
	}
	return (diff > 0);
}
void YoloV3::doNmsSort(std::vector<detection> &dets, int classes, float thresh)
{
	int i, j, k;
	k = dets.size() - 1;
	for (i = 0; i <= k; ++i) {
		if (dets[i].objectness == 0) {
			dets.erase(dets.begin() + i);
			--i;
		}
	}

	int total = dets.size();

	for (k = 0; k < classes; ++k) {
		for (i = 0; i < total; ++i) {
			dets[i].sort_class = k;
		}
		sort(dets.begin(), dets.end(), nms_comparator);

		float threshold = (k == 2) ? 0.3 : thresh;

		for (i = 0; i < total; ++i) {
			if (dets[i].prob[k] == 0) continue;
			box a = dets[i].bbox;
			for (j = i + 1; j < total; ++j) {
				box b = dets[j].bbox;
				if (box_iou(a, b) > thresh) {
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

// Creates array of detections with prob > thresh and fills best_class for them
std::vector<BBoxInfo> YoloV3::getActualDetections(std::vector<detection> &
	
	, float threshPed, float threshOther, int imageH, int imageW)
{
	std::vector<BBoxInfo> result_arr;
	int selected_num = 0;
	int dets_num = dets.size();
	int i;
	for (i = 0; i < dets_num; ++i) {
		int best_class = -1;
		float best_class_prob = threshPed;
		int j;
		for (j = 0; j < dets[i].classes; ++j) {
			if (dets[i].prob[j] > best_class_prob) {
				best_class = j;
				best_class_prob = dets[i].prob[j];
			}
		}
		if (!(best_class == 0 || best_class == 1 || best_class == 3) && best_class_prob < threshOther)
		{
			best_class = -1;
		}
		if (best_class >= 0) 
		{
			BBoxInfo result;
			result.box.x = dets[i].bbox.x - dets[i].bbox.w / 2.;
			result.box.y = dets[i].bbox.y - dets[i].bbox.h / 2.;
			result.box.w = dets[i].bbox.w;
			result.box.h = dets[i].bbox.h;
			result.label = best_class;
			result.prob = best_class_prob;
			result_arr.push_back(result);
		}
	}	
	return result_arr;
}
