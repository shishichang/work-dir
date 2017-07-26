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
	while (true)
	{
		if(!vcap.read(frame))
			break;
				
		//__TIC__();
        ssd.detect(frame);
		//printf("current frame elapsed time: ");
		//__TOC__();
	}	
}

