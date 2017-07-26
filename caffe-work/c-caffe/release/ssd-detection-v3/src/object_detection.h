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
	std::string detection_out_layer_name;
	std::string label_map_file;

	//
	int resized_width;
	int resized_height;
} config;

struct LabelBBox
{
	std::string object_name;
	float score;
	size_t label;
	size_t frame_idx;
	float x1;
	float x2;
	float y1;
	float y2;

};

const int mean_b = 104;
const int mean_g = 117;
const int mean_r = 123;

const int resized_width = 300;
const int resized_height = 300;
//const std::string detection_out_layer_name = "detection_out";

class SSD
{
public:
	SSD(const std::string& configfile);

	void detect(const cv::Mat& img, std::vector<LabelBBox>& resultbbox, const int idx);
private:
	bool init(const std::string configfile);
	void wrapInputLayer(std::vector<cv::Mat>& input_channesls);
	void preprocess(const cv::Mat& img);
	void getResults(const std::vector<caffe::Blob<float>*> &top);
	void convert2labelbbox(const std::vector<caffe::Blob<float>*>& output, std::vector<LabelBBox>& labelbbox, const int idx);

	inline int getBlobIdxByName(std::string& query_name)
	{
		std::vector<std::string> const& blob_names = net_->blob_names();
		for(unsigned int i = 0; i != blob_names.size(); ++i)
		{
			if(query_name == blob_names[i])
			{
				return i;
			}
		}
		LOG(FATAL) << "Unknown blob name: " <<query_name;
		return -1;
	}

private:
	caffe::shared_ptr<caffe::Net<float>> net_;
	config config_;
	int detection_blob_idx_;
	std::map<int, std::string> label_to_display_name_;
};

#endif //__object_detection_h__
