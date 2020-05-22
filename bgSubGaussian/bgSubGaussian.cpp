﻿#include <iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void calGaussianBackground(std::vector<cv::Mat> srcMats, cv::Mat & meanMat, cv::Mat &varMat)
{

	int rows = srcMats[0].rows;
	int cols = srcMats[0].cols;


	for (int h = 0; h < rows; h++)
	{
		for (int w = 0; w < cols; w++)
		{

			int sum = 0;
			float var = 0;
			//求均值
			for (int i = 0; i < srcMats.size(); i++) {
				sum += srcMats[i].at<uchar>(h, w);
			}
			meanMat.at<uchar>(h, w) = sum / srcMats.size();
			//求方差
			for (int i = 0; i < srcMats.size(); i++) {
				var += pow((srcMats[i].at<uchar>(h, w) - meanMat.at<uchar>(h, w)), 2);
			}
			varMat.at<float>(h, w) = var / srcMats.size();
		}
	}


}


void gaussianThreshold(cv::Mat srcMat, cv::Mat meanMat, cv::Mat varMat, float weight, cv::Mat & dstMat)
{
	int srcI;
	int meanI;
	int dstI;
	int rows = srcMat.rows;
	int cols = srcMat.cols;

	for (int h = 0; h < rows; h++)
	{
		for (int w = 0; w < cols; w++)
		{
			srcI = srcMat.at<uchar>(h, w);
			meanI = meanMat.at<uchar>(h, w);
			int dif = abs(srcI - meanI);
			int th = weight * varMat.at<float>(h, w);

			if (dif > th) {

				dstMat.at<uchar>(h, w) = 255;
			}
			else {
				dstMat.at<uchar>(h, w) = 0;
			}
		}
	}

}

int main()
{
	VideoCapture cap(0);

	//如果摄像头打开失败
	if (!cap.isOpened())
	{
		std::cout << "Fail to open camera" << std::endl;
		return -1;
	}

	std::vector<cv::Mat>srcMat;

	//参数设置
	int nBg = 200;//用来建立背景模型的数量
	float wVar = 1;//方差权重

	int cnt = 0;
	cv::Mat frame;
	cv::Mat meanMat;
	cv::Mat varMat;
	cv::Mat dstMat;

	while (1)
	{
		cap >> frame;
		cvtColor(frame, frame, COLOR_BGR2GRAY);

		//计算背景
		if (cnt < nBg)
		{
			srcMat.push_back(frame);

			if (cnt == 0)
			{
				std::cout << "reading frame" << std::endl;
			}

		}
		else if (cnt == nBg)
		{
			//计算模型
			meanMat.create(frame.size(), CV_8UC1);
			varMat.create(frame.size(), CV_32FC1);
			std::cout << "calculating background models" << std::endl;
			calGaussianBackground(srcMat, meanMat, varMat);
		}
		else
		{
			//背景差分
			dstMat.create(frame.size(), CV_8UC1);
			gaussianThreshold(frame, meanMat, varMat, wVar, dstMat);
			imshow("result", dstMat);
			imshow("frame", frame);
			waitKey(30);
		}
		cnt++;
	}
	return 0;
}

