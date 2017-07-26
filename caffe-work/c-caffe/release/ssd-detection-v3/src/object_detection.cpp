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
	config_.label_map_file = "../models/labelmap_voc.prototxt";
	config_.detection_out_layer_name = "detection_out";
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	net_.reset(new caffe::Net<float>(config_.prototxt, caffe::TEST));
	net_->CopyTrainedLayersFrom(config_.caffemodel);
	detection_blob_idx_ = getBlobIdxByName(config_.detection_out_layer_name);
	assert(detection_blob_idx_ >= 0);
	std::cout<<"label_map_file: " <<config_.label_map_file <<std::endl;
	std::cout<<"model_file: " <<config_.caffemodel <<std::endl;
	//
	caffe::LabelMap label_map;
	caffe::ReadProtoFromTextFile(config_.label_map_file, &label_map);
	caffe::MapLabelToDisplayName(label_map, true, &label_to_display_name_);
}

void SSD::getResults(const std::vector<caffe::Blob<float>*>& top)
{
	assert(top.size() == 1);
	boost::shared_ptr<caffe::Blob<float>> detection_blob = net_->blobs()[detection_blob_idx_];
	const float* detection_data = detection_blob ->gpu_data();
	//top[0]->Reshape(detection_blob->shape());
	assert(detection_blob->num() == 1);
	assert(detection_blob->channels() == 1);
	assert(detection_blob->width() == 7);
	top[0]->ReshapeLike(*(detection_blob));
	float* top_data = top[0]->mutable_cpu_data();
	(*top[0]).CopyFrom(*(detection_blob));
	//	for(unsigned i = 0; i < detection_blob->count(); ++i)
//	{
//		top_data[i] = detection_data[i];
//	}
}

void SSD::convert2labelbbox(const std::vector<caffe::Blob<float>*>& output, std::vector<LabelBBox>& bbox_vector, const int idx)
{
	bbox_vector.clear();
	const float* label_data = output[0]->cpu_data();
	assert(output[0]->num() == 1);
	assert(output[0]->channels() == 1);
	assert(output[0]->width() == 7);
	int num_object = output[0]->height();
//	std::cout<<"Detection out shape is: \t";
//	std::cout<<output[0]->shape_string() <<std::endl;
	int count = 0;
	LabelBBox labelbbox;
	for(unsigned int j = 0; j < num_object; ++j)
	{
		labelbbox.label = label_data[count*7 + 1];
		labelbbox.score = label_data[count*7 + 2];
		labelbbox.x1 = label_data[count*7 + 3];
		labelbbox.y1 = label_data[count*7 + 4];
		labelbbox.x2 = label_data[count*7 + 5];
		labelbbox.y2 = label_data[count*7 + 6];
		labelbbox.object_name = "";
		if (label_to_display_name_.find(labelbbox.label) != label_to_display_name_.end()) 
		{
			labelbbox.object_name = label_to_display_name_.find(labelbbox.label)->second;
	    }
		else
			printf("There is no name corresponding to label %d", labelbbox.label);
		labelbbox.frame_idx = idx; 
		count++;
		bbox_vector.push_back(labelbbox);
	}
}

void SSD::detect(const cv::Mat& img, std::vector<LabelBBox>& labelbbox, const int frame_idx)
{
	preprocess(img);
//	printf("forwarding..\n");
	net_->Forward();
	caffe::Blob<float> output_blob;
	std::vector<caffe::Blob<float>*> output(1, &output_blob);
	getResults(output);
	convert2labelbbox(output, labelbbox, frame_idx);

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
		config_.label_map_file = std::string(model_node["labelfile"]); 
//		cv::FileNode input_node = fs["output"];
		config_.resized_width=  resized_width;	//input_node["resized_width"];
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
