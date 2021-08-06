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



bool donms(std::vector<void*> m_Bindings)//the yolo output 64*64*3/32*32*3/*16*16*3
{
	
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

