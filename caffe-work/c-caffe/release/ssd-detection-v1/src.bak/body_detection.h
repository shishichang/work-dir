#ifndef __body_detection_h__
#define __body_detection_h__

#include <boost/foreach.hpp>
#include <caffe/caffe.hpp>
#include <caffe/util/bbox_util.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cassert>

//using namespace std;
//using namespace caffe;
//#define POST_PROCESS_BY_GPU 1

typedef struct _config
{
	int resized_width;
	int resized_height;
	
	std::string location_output_layer_name;
	std::string confidence_output_layer_name; 
	std::string priorbox_output_layer_name;
	bool visualize;

} config; 

const int mean_b = 104;
const int mean_g = 117;
const int mean_r = 123;

const std::string location_output_layer_name = "mbox_loc";				// blob name
const std::string confidence_output_layer_name = "mbox_conf_flatten";	// blob name
const std::string priorbox_output_layer_name = "mbox_priorbox";			// blob name

class SSD
{
public:
	SSD(const std::string& prototxt, const std::string& caffemodel);

	void detect(const cv::Mat& img);

private:
	/* Wrap the input layer of the network in separate cv::Mat objects
	 * (one per channel). This way we save one memcpy operation and we
	 * don't need to rely on cudaMemcpy2D. The last preprocessing
	 * operation will write the separate channels directly to the input
	 * layer. */
	void wrapInputLayer(std::vector<cv::Mat>& input_channels);

	void preprocess(const cv::Mat& img);

	void getResults(const std::vector<caffe::Blob<float>*>& top);

	inline int getBlobIdxByName(std::string& query_name)
	{
		std::vector<std::string> const& blob_names = net_->blob_names();
		for (unsigned int i = 0; i != blob_names.size(); ++i)
		{
			if (query_name == blob_names[i])
			{
				return i;
			}
		}
		
		LOG(FATAL) << "Unknown blob name: " << query_name;
		return -1;
	}

private:
	caffe::shared_ptr<caffe::Net<float> > net_;
	config config_;
};


#endif // __body_detection_h__
