/*************************************************************************
	> File Name: ./object_detection.h
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: Fri May 26 02:41:45 2017
 ************************************************************************/
#ifndef __object_detection_h__
#define __object_detection_h__

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cassert>
//using namespace std;

typedef struct _config
{
	//model
	std::string prototxt;
	std::string caffemodel;

	//
	int resized_width;
	int resized_height;
} config;

const int mean_b = 104;
const int mean_g = 117;
const int mean_r = 123;

const int resized_width = 300;
const int resized_height = 300;

class SSD
{
public:
	SSD(const std::string& configfile);

	void detect(const cv::Mat& img);
private:
	bool init(const std::string configfile);
	void wrapInputLayer(std::vector<cv::Mat>& input_channesls);
	void preprocess(const cv::Mat& img);

private:
	caffe::shared_ptr<caffe::Net<float>> net_;
	config config_;
};

#endif //__object_detection_h__
