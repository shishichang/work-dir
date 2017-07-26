#include "detection.h"
#include "time_helper.h"

//using namespace std;
//using namespace caffe;

SSD::SSD(const std::string& configfile)
{
	init(configfile);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	net_.reset(new caffe::Net<float>(config_.prototxt, caffe::TEST));
	net_->CopyTrainedLayersFrom(config_.caffemodel);

	assert(config_.resized_width == resized_width);
	assert(config_.resized_height == resized_height);	
	loc_blob_idx_ = getBlobIdxByName(config_.location_output_layer_name);
	conf_blob_idx_ = getBlobIdxByName(config_.confidence_output_layer_name);
	prior_blob_idx_ = getBlobIdxByName(config_.priorbox_output_layer_name);
	assert(loc_blob_idx_ >= 0);
	assert(conf_blob_idx_ >= 0);
	assert(prior_blob_idx_ >= 0);

	if (true == config_.visualize)
	{
		caffe::LabelMap label_map;
		caffe::ReadProtoFromTextFile(config_.label_map_file, &label_map);
		caffe::MapLabelToDisplayName(label_map, true, &label_to_display_name_);
	}
}

bool SSD::init(const std::string configfile)
{
	assert(readConfigFile(configfile));
	assert(config_.location_output_layer_name == location_output_layer_name);
	assert(config_.confidence_output_layer_name == confidence_output_layer_name);
	assert(config_.priorbox_output_layer_name == priorbox_output_layer_name);
	assert(config_.share_location == true);

	return true;
}

bool SSD::readConfigFile(const std::string configfile)
{
	cv::FileStorage fs;
	try
	{
		if (!fs.open(configfile, cv::FileStorage::READ))
		{
			printf("failed to load config_file, file = %s\n", configfile.c_str());
			exit(1);
		}
	
		// model
		cv::FileNode model_node = fs["model"];
		config_.prototxt = std::string(model_node["prototxt"]);
		config_.caffemodel = std::string(model_node["caffemodel"]);

		// output
		cv::FileNode output_node = fs["output"];
		config_.resized_width = output_node["resized_width"];
		config_.resized_height = output_node["resized_height"];
		config_.location_output_layer_name = std::string(output_node["location_output_layer_name"]);	
		config_.confidence_output_layer_name = std::string(output_node["confidence_output_layer_name"]);	
		config_.priorbox_output_layer_name = std::string(output_node["priorbox_output_layer_name"]);	
		config_.share_location = true;
		if (std::string(output_node["share_location"]) == std::string("false"))
		{
			config_.share_location = false;
		}
		config_.num_classes = output_node["num_classes"];
		config_.background_label_id = output_node["background_label_id"];
		config_.label_map_file = std::string(output_node["label_map_file"]);
		config_.code_type = (caffe::CodeType)(2);
		if (std::string("CENTER_SIZE") != std::string(output_node["code_type"]))
		{
			printf("not support other CodeType yet!");
			assert(0);
		}
		config_.confidence_threshold = output_node["confidence_threshold"];

		// nms
		cv:: FileNode nms_node = fs["nms"];
		config_.nms_threshold = nms_node["nms_threshold"];
		config_.top_k = nms_node["top_k"];
		config_.keep_top_k = nms_node["keep_top_k"];

		// visual
		cv:: FileNode visual_node = fs["visual"];
		config_.visualize = true;
		if (std::string(visual_node["visualize"]) == std::string("false"))
		{
			config_.visualize = false;		
		}
		config_.visualize_threshold = visual_node["visualize_threshold"];
		config_.show_image_width = visual_node["show_image_width"];
		config_.show_image_height = visual_node["show_image_height"];

		// save_result
		cv:: FileNode save_node = fs["save_result"];
		config_.need_save = false;
		if (std::string(save_node["need_save"]) == std::string("true"))
		{
			printf("not support save yet!");
			assert(0);
		}
		config_.save_format = std::string(save_node["save_format"]);
		config_.save_directory = std::string(save_node["save_directory"]);
		
	
		fs.release();
	}
	catch (cv::Exception& e)
	{
		printf("parse configure file exception, exception[%s]\n", e.what());
		fs.release();
		exit(1);
	}
	catch (...)
	{
		printf("parse configure file encounter unknown exception\n");
		fs.release();
		exit(1);    
	}

	return true;
}

void SSD::detect(const cv::Mat& img)
{
	__TIC__();
	preprocess(img);
	
	printf("sg.xu: preprocess elapsed time: ");
	__TOC__();
	// forward ...
	printf("forwarding ...\n");
	net_->Forward();

	caffe::Blob<float> output_blob;
	std::vector<caffe::Blob<float>*> output(1, &output_blob);
	getResults(output);

	if (config_.visualize)
	{
		std::vector<cv::Mat> cv_imgs(1, img);
		 __TIC__();
		std::vector<cv::Scalar> colors = caffe::GetColors(label_to_display_name_.size());
		caffe::VisualizeBBox(cv_imgs, output[0],config_. visualize_threshold, colors, label_to_display_name_);
		printf("show result elapsed time: ");
		__TOC__();
	}
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void SSD::wrapInputLayer(std::vector<cv::Mat>& input_channels)
{
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	int w = input_layer->width();
	int h = input_layer->height();

    input_channels.clear();

	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i)
	{
		cv::Mat channel(h, w, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += w * h;
	}

}

void SSD::preprocess(const cv::Mat& img)
{
	//float scale = float(config_.test_scales) / img.cols; 
	//int w = config_.test_scales;
	//int h = img.rows * scale;
	//cv::Size size = cv::Size(w, h);
	//cv::Size size = cv::Size(config_.resized_width, int(img.rows * (float(config_.resized_width) / img.cols)));
	cv::Size size = cv::Size(config_.resized_width, config_.resized_height);
	cv::Mat sample_resized;
	if (img.size() != size)
	{
		cv::resize(img, sample_resized, size);
	}
	else
	{
		sample_resized = img;
	}
	
	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, 3, sample_resized.rows, sample_resized.cols);
	
	// Forward dimension change to all layers.
	net_->Reshape();

	
	cv::Mat sample_float;
	sample_resized.convertTo(sample_float, CV_32FC3);

	cv::Mat mean;
	cv::Mat sample_normalized;
	
	mean = cv::Mat(sample_float.size(), CV_32FC3, cv::Scalar(mean_b, mean_g, mean_r));
	cv::subtract(sample_float, mean, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	 * input layer of the network because it is wrapped by the cv::Mat
	 * objects in input_channels. */
	std::vector<cv::Mat> input_channels;
	wrapInputLayer(input_channels);
	cv::split(sample_normalized, input_channels);
	assert(reinterpret_cast<float*>(input_channels.at(0).data) == input_layer->cpu_data());
}

// top blob shape: [n, c, h, w] = [1, 1, N, 7];
// which N is the number of objects.
void SSD::getResults(const std::vector<caffe::Blob<float>*>& top)
{
	assert(top.size() == 1);
	boost::shared_ptr<caffe::Blob<float> >  loc_blob = net_->blobs()[loc_blob_idx_];
	const float* loc_data = loc_blob->gpu_data();
	const int num = loc_blob->num();
	//printf("loc_data, N = %d\n", loc_blob->num());
	//printf("loc_data, C = %d\n", loc_blob->channels());
	//printf("loc_data, H = %d\n", loc_blob->height());
	//printf("loc_data, W = %d\n", loc_blob->width());
	//printf("----------------------------------------------------\n");
	
	boost::shared_ptr<caffe::Blob<float> >  conf_blob = net_->blobs()[conf_blob_idx_];
	//printf("conf_data, N = %d\n", conf_blob->num());
	//printf("conf_data, C = %d\n", conf_blob->channels());
	//printf("conf_data, H = %d\n", conf_blob->height());
	//printf("conf_data, W = %d\n", conf_blob->width());
	//printf("----------------------------------------------------\n");
	
	boost::shared_ptr<caffe::Blob<float> >  prior_blob = net_->blobs()[prior_blob_idx_];
	const float* prior_data = prior_blob->gpu_data();
	//printf("prior_data, N = %d\n", prior_blob->num());
	//printf("prior_data, C = %d\n", prior_blob->channels());
	//printf("prior_data, H = %d\n", prior_blob->height());
	//printf("prior_data, W = %d\n", prior_blob->width());
	//printf("----------------------------------------------------\n");
	
	// Decode predictions.
	caffe::Blob<float> bbox_preds;
	bbox_preds.ReshapeLike(*(loc_blob));
	float* bbox_data = bbox_preds.mutable_gpu_data();
	const int loc_count = bbox_preds.count();

	bool variance_encoded_in_target = false;
	int num_priors = prior_blob->height() / 4;
	int num_loc_classes = 1;
	caffe::DecodeBBoxesGPU(loc_count,
		loc_data, 
		prior_data, 
		config_.code_type,
	    variance_encoded_in_target, 
		num_priors, 
		config_.share_location,
	    num_loc_classes, 
		config_.background_label_id, 
		bbox_data);
	
	// Retrieve all decoded location predictions.
	const float* bbox_cpu_data = bbox_preds.cpu_data();
	std::vector<caffe::LabelBBox> all_decode_bboxes;
	caffe::GetLocPredictions(bbox_cpu_data, 
		num, 
		num_priors, 
		num_loc_classes,
	    config_.share_location, 
		&all_decode_bboxes);
		
	// Retrieve all confidences.
	const float* conf_data;
	caffe::Blob<float> conf_permute;
	conf_permute.ReshapeLike(*(conf_blob));
	float* conf_permute_data = conf_permute.mutable_gpu_data();
	
	caffe::PermuteDataGPU(conf_permute.count(), 
		conf_blob->gpu_data(),
	    config_.num_classes, 
		num_priors, 
		1, 
		conf_permute_data);
	conf_data = conf_permute.cpu_data();
	const bool class_major = true;
	std::vector<std::map<int, std::vector<float> > > all_conf_scores;
	caffe::GetConfidenceScores(conf_data, 
		num, 
		num_priors, 
		config_.num_classes,
	    class_major, 
		&all_conf_scores);
	
	int num_kept = 0;
	std::vector<std::map<int, std::vector<int> > > all_indices;
	for (int i = 0; i < num; ++i)
	{
		const caffe::LabelBBox& decode_bboxes = all_decode_bboxes[i];
		const std::map<int, std::vector<float> >& conf_scores = all_conf_scores[i];
		std::map<int, std::vector<int> > indices;
		int num_det = 0;
		for (int c = 0; c < config_.num_classes; ++c)
		{
			if (c == config_.background_label_id)
			{
				// Ignore background class.
				continue;
			}
			if (conf_scores.find(c) == conf_scores.end())
			{
		    	// Something bad happened if there are no predictions for current label.
		    	LOG(FATAL) << "Could not find confidence predictions for label " << c;
		  	}
			const std::vector<float>& scores = conf_scores.find(c)->second;
			int label = config_.share_location ? -1 : c;
			if (decode_bboxes.find(label) == decode_bboxes.end())
			{
		    	// Something bad happened if there are no predictions for current label.
		    	LOG(FATAL) << "Could not find location predictions for label " << label;
		    	continue;
			}
				
			//__TIC__();
			const std::vector<caffe::NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
			caffe::ApplyNMSFast(bboxes, 
				scores, 
				config_.confidence_threshold, 
				config_.nms_threshold, 
				config_.top_k, 
				&(indices[c]));
			//printf("sg.xu: ApplyNMSFast elapsed time: ");
			//__TOC__();
			num_det += indices[c].size();
		}
		if (config_.keep_top_k > -1 && num_det > config_.keep_top_k)
		{
			std::vector<std::pair<float, std::pair<int, int> > > score_index_pairs;
			for (std::map<int, std::vector<int> >::iterator it = indices.begin();
				it != indices.end(); ++it)
			{
		    	int label = it->first;
				const std::vector<int>& label_indices = it->second;
				if (conf_scores.find(label) == conf_scores.end())
				{
					// Something bad happened for current label.
					LOG(FATAL) << "Could not find location predictions for " << label;
					continue;
				}
				const std::vector<float>& scores = conf_scores.find(label)->second;
				for (unsigned int j = 0; j < label_indices.size(); ++j)
				{
					int idx = label_indices[j];
					CHECK_LT(idx, scores.size());
					score_index_pairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
				}
			}
			
			// Keep top k results per image.
			std::sort(score_index_pairs.begin(), score_index_pairs.end(), caffe::SortScorePairDescend<std::pair<int, int> >);
			score_index_pairs.resize(config_.keep_top_k);
			
			// Store the new indices.
			std::map<int, std::vector<int> > new_indices;
			for (unsigned int j = 0; j < score_index_pairs.size(); ++j)
			{
				int label = score_index_pairs[j].second.first;
				int idx = score_index_pairs[j].second.second;
				new_indices[label].push_back(idx);
			}
			all_indices.push_back(new_indices);
			num_kept += config_.keep_top_k;
		}
		else
		{
			all_indices.push_back(indices);
			num_kept += num_det;
		}
	}

	// top_shape = [1, 1, num_kept, 7]
	std::vector<int> top_shape(2, 1);
	top_shape.push_back(num_kept);
	top_shape.push_back(7);
	if (num_kept == 0)
	{
		LOG(INFO) << "Couldn't find any detections";
		top_shape[2] = 1;
		top[0]->Reshape(top_shape);
		caffe::caffe_set<float>(top[0]->count(), -1, top[0]->mutable_cpu_data());
		return;
	}
	top[0]->Reshape(top_shape);
	float* top_data = top[0]->mutable_cpu_data();
	int count = 0;

	for (int i = 0; i < num; ++i)
	{
		const std::map<int, std::vector<float> >& conf_scores = all_conf_scores[i];
		const caffe::LabelBBox& decode_bboxes = all_decode_bboxes[i];
		for (std::map<int, std::vector<int> >::iterator it = all_indices[i].begin(); 
			it != all_indices[i].end(); ++it)
		{
			int label = it->first;
			if (conf_scores.find(label) == conf_scores.end())
			{
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find confidence predictions for " << label;
				continue;
			}
			const std::vector<float>& scores = conf_scores.find(label)->second;
			int loc_label = config_.share_location ? -1 : label;
			if (decode_bboxes.find(loc_label) == decode_bboxes.end())
			{
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find location predictions for " << loc_label;
				continue;
			}
			const std::vector<caffe::NormalizedBBox>& bboxes = decode_bboxes.find(loc_label)->second;
			std::vector<int>& indices = it->second;
			for (unsigned int j = 0; j < indices.size(); ++j)
			{
				int idx = indices[j];
				top_data[count * 7] = i;
				top_data[count * 7 + 1] = label;
				top_data[count * 7 + 2] = scores[idx];
				
				caffe::NormalizedBBox clip_bbox;
				caffe::ClipBBox(bboxes[idx], &clip_bbox);
				top_data[count * 7 + 3] = clip_bbox.xmin();
				top_data[count * 7 + 4] = clip_bbox.ymin();
				top_data[count * 7 + 5] = clip_bbox.xmax();
				top_data[count * 7 + 6] = clip_bbox.ymax();
				
				++count;
			}
		}
	}
}
