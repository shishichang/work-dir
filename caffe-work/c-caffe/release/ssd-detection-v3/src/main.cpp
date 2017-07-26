#include "object_detection.h"
//#include "time_helper.h"
#include <fstream>
#include <string>
#include <iostream>
// test
int main()
{
	std::ifstream in("../config/config_file.txt");
	std::string s;
	getline(in, s);
	std::string configfile(s);
	SSD ssd(configfile);

	getline(in, s);
	std::string datafile(s);
    cv::VideoCapture vcap(datafile);
    if (false == vcap.isOpened())
    {
        fprintf(stderr, "video cannot open!\n");
        return -1;
    }
	//double rate =vcap.get(CV_CAP_PROP_FPS);
	//cv::namedWindow("testVideo");
    //int delay = 1000/rate;
    cv::Mat frame;
	int frame_count = 0;
	while (true)
	{
		if(!vcap.read(frame))
			break;
				
		//__TIC__();
		//The shape of output_blob is 1,1,N,7
		frame_count++;
		std::vector<LabelBBox> resultbbox;
		ssd.detect(frame, resultbbox, frame_count);
		if(resultbbox.empty())
		{
			std::cout<<"There is no object in this frame."<<std::endl;
		}
		else
		{
			std::cout<<"The number of objects in "<<resultbbox[0].frame_idx<< "th frame is: "<<resultbbox.size()<<std::endl;
			std::cout<<"Objects:"<<std::endl;
			for(unsigned i = 0; i < resultbbox.size(); ++i)
			{
				std::cout<<"Class: "<<resultbbox[i].object_name 
					<<"\tScore: " <<resultbbox[i].score 
					<<"\tx1: " <<resultbbox[i].x1
					<<"\ty1: " <<resultbbox[i].y1 
					<<"\tx2: " <<resultbbox[i].x2 
					<<"\ty2: " <<resultbbox[i].y2 <<std::endl;
			}
		}
		//printf("current frame elapsed time: ");
		//__TOC__();
		cv::waitKey(100);
	}	
}

