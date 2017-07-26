/*************************************************************************
	> File Name: ./object_detection.cpp
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: Fri May 26 02:38:43 2017
 ************************************************************************/
#include "object_detection.h"
//using namespace std;
SSD::SSD(const std::string& configfile)
{
	init(configfile);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	net_.reset(new caffe::Net<float>(config_.prototxt, caffe::TEST));
	net_->CopyTrainedLayersFrom(config_.caffemodel);	
}

void SSD::detect(const cv::Mat& img)
{
	preprocess(img);
//	printf("forwarding..\n");
	net_->Forward();
}

void SSD::preprocess(const cv::Mat& img)
{
	cv::Size size = cv::Size(config_.resized_width, config_.resized_height);
	cv::Mat sample_resized;
	cv::resize(img, sample_resized, size);
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, 3, sample_resized.rows, sample_resized.cols);
	
	//Compute blob sizes of all layers
	net_->Reshape();

	cv::Mat sample_float;
	sample_resized.convertTo(sample_float, CV_32FC3);

	cv::Mat mean;
	cv::Mat sample_normalized;
	mean = cv::Mat(sample_float.size(), CV_32FC3, cv::Scalar(mean_b, mean_g, mean_r));
	cv::subtract(sample_float, mean, sample_normalized);

	std::vector<cv::Mat> input_channels;
	wrapInputLayer(input_channels);
	cv::split(sample_normalized, input_channels);
	assert(reinterpret_cast<float*>(input_channels.at(0).data) == input_layer->cpu_data());
}

void SSD::wrapInputLayer(std::vector<cv::Mat>& input_channels)
{
	input_channels.clear();
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	int w = input_layer->width();
	int h = input_layer->height();

	float* input_data = input_layer->mutable_cpu_data();
	for(int i = 0; i < input_layer->channels(); ++i)
	{
		cv::Mat channel(h, w, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += w * h;
	}
}

bool SSD::init(const std::string configfile)
{
	cv::FileStorage fs;
	try
	{
		if(!fs.open(configfile, cv::FileStorage::READ))
		{
			printf("fail to open config_file, file = %s\n", configfile.c_str());
			exit(1);
		}
		cv::FileNode model_node = fs["model"];
		config_.prototxt = std::string(model_node["prototxt"]);
		config_.caffemodel = std::string(model_node["caffemodel"]); 
		cv::FileNode input_node = fs["input"];
		config_.resized_width= resized_width;
		//int(input_node["resized_width"]);
		config_.resized_height = resized_height;
			//int(input_node["resize_height"]); 
	}
	 catch (...)
	 {
		printf("parse config_file failed.");
		fs.release();
		exit(1);
	 }

	 return true;

}
//public:
//	SSD(const std::string& configfile);
//
//	void detect(const cv::Mat& img);
//private:
//	bool init(const std::string configfile);
//	bool readConfigFile(const std::string configfile);
//	void wrapInputLayer(std::vector<cv::Mat>& input_channesls);
//	void preprocess(const cv::Mat& img);
