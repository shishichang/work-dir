#include "body_detection.h"
#include "time_helper.h"

// test
int main()
{
	const std::string prototxt("../models/test.prototxt");
	//const std::string caffemodel("../models/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel");
	//const std::string caffemodel("../models/VGG-Person_SSD_224x224.caffemodel");
	//const std::string caffemodel("../models/VGG_Person_SSD_300x300_iter_44000.caffemodel");
	//const std::string caffemodel("../models/VGG_Person_SSD_300x300_iter_14000.caffemodel");
	const std::string caffemodel("../models/VGG_Person_SSD_300x300_iter_48000.caffemodel");
	
	SSD ssd(prototxt, caffemodel);

    cv::VideoCapture vcap(0);
    if (false == vcap.isOpened())
    {
        fprintf(stderr, "camera cannot open!\n");
        return -1;
    }
    
	// const char *window = "body-detect";
    cv::Mat frame;
    int idx = -1;
    while (true)
    {
		++idx;
		{
			__TIC__();
			vcap >> frame;
			printf("frame w = %d\th = %d\n", frame.cols, frame.rows);
			printf("sg.xu: camera capture elapsed time: ");
			__TOC__();
		}

		__TIC__();
        ssd.detect(frame);
		printf("sg.xu: total elapsed time: ");
		__TOC__();
        //detector.draw_result(frame, result);
        //cv::imshow(window, frame);
        
		int key = cv::waitKey(1);
        if (27 == key)
        {
            break;
        }
    }

    //cv::cvReleaseCapture(&vcap);
    //cv::cvDestroyWindow(window);
}

