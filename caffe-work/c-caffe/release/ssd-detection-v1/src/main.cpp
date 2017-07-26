#include "detection.h"
#include "time_helper.h"
#include <fstream>
#include <string>
#include <iostream>
// test
int main()
{
	std::ifstream in("task.txt");
	std::string s;
	getline(in, s);
	std::string configfile("../config/config_"+s+".xml");
	SSD ssd(configfile);

    cv::VideoCapture vcap("./test.mp4");
    if (false == vcap.isOpened())
    {
        fprintf(stderr, "video cannot open!\n");
        return -1;
    }
	double rate =vcap.get(CV_CAP_PROP_FPS);
	cv::namedWindow("testVideo");
    int delay = 1000/rate;
    cv::Mat frame;
	while (true)
	{
		if(!vcap.read(frame))
			break;
				
		__TIC__();
        ssd.detect(frame);
		printf("current frame  elapsed time: ");
		__TOC__();
		//cv::imshow("testVideo", frame);
       // if (cv::waitKey(1000)>0)
       // {
       //     break;
       // }
	}	
}

